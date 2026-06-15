#!/usr/bin/env python3
"""Unit/integration tests for the DOB-MPC port (advisor punch-list order).

#1 frames  : FLU<->NED transform correctness (round-trip, tilt sign, velocity
             consistency, wrench invariance) -- MUST pass before the observer.
#2 predictor: marinegym-matched params reproduce the plant (damping decelerates;
             surge->pitch trim 6 N ~ 23 deg).
#4 EAOB    : disturbance estimate is unbiased under a steady force when the
             acceleration is finite-differenced (no qacc double-count).
plus: the copied CasADi MPC model matches the copied NumPy fossen model.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dobmpc import frames, fossen, params as P
from dobmpc.eaob import EAOB


def _rand_R(rng):
    """A random body->world rotation matrix (via a random quaternion)."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ============================================================ #1 frames
def test_frames():
    rng = np.random.default_rng(0)
    # (a) round-trip identity: FLU -> NED -> FLU exact to 1e-10
    for _ in range(200):
        p = rng.standard_normal(3)
        R = _rand_R(rng)
        # avoid the |theta|=90 deg gimbal where ZYX Euler is singular
        if abs(frames.flu_to_ned_eta(p, R)[4]) > 1.4:
            continue
        nu = rng.standard_normal(6)
        eta_ned = frames.flu_to_ned_eta(p, R)
        nu_ned = frames.flu_to_ned_nu(nu)
        p2, R2 = frames.ned_to_flu_eta(eta_ned)
        nu2 = frames.ned_to_flu_nu(nu_ned)
        assert np.allclose(p, p2, atol=1e-10), "position round-trip"
        assert np.allclose(R, R2, atol=1e-10), "rotation round-trip"
        assert np.allclose(nu, nu2, atol=1e-10), "nu round-trip"

    # (b) static-tilt sign: a pure FLU yaw psi -> NED psi = -psi (z flips)
    psi = 0.5
    eta = frames.flu_to_ned_eta(np.zeros(3), _Rz(psi))
    assert abs(eta[5] + psi) < 1e-9, f"yaw should flip sign: {eta[5]:+.3f} vs {-psi:+.3f}"
    # a known FLU roll/pitch/yaw stays within bounds and inverts cleanly
    R = _Rz(np.radians(30)) @ np.array([[np.cos(np.radians(5)), 0, np.sin(np.radians(5))],
                                        [0, 1, 0],
                                        [-np.sin(np.radians(5)), 0, np.cos(np.radians(5))]])
    e = frames.flu_to_ned_eta(np.zeros(3), R)
    _, Rb = frames.ned_to_flu_eta(e)
    assert np.allclose(R, Rb, atol=1e-9), "tilt round-trip"

    # (c) velocity consistency: NED eta_dot = J(eta) nu, position part rotated
    #     back to FLU == FLU world velocity R_flu @ nu_lin. Catches the
    #     qvel[:3] world-vs-body trap.
    for _ in range(100):
        R = _rand_R(rng)
        if abs(frames.flu_to_ned_eta(np.zeros(3), R)[4]) > 1.4:
            continue
        nu = rng.standard_normal(6)
        v_world_flu = R @ nu[:3]                       # FLU world linear velocity
        eta_ned = frames.flu_to_ned_eta(np.zeros(3), R)
        nu_ned = frames.flu_to_ned_nu(nu)
        eta_dot_ned = fossen.jacobian_eta(eta_ned) @ nu_ned
        v_world_back = frames.S @ eta_dot_ned[:3]      # NED world vel -> FLU world
        assert np.allclose(v_world_flu, v_world_back, atol=1e-9), "velocity frame consistency"

    # (d) wrench invariance: FLU surge -> NED -> FLU is identity; NED +N -> FLU -N
    tau_flu = np.array([5.0, 0, 0, 0, 0, 0])
    assert np.allclose(frames.ned_wrench_to_flu(frames.flu_wrench_to_ned(tau_flu)),
                       tau_flu, atol=1e-12), "wrench round-trip"
    assert np.allclose(frames.ned_wrench_to_flu([0, 0, 0, 0, 0, 3.0]),
                       [0, 0, 0, 0, 0, -3.0]), "yaw moment sign"
    print("[#1 frames] OK  round-trip<1e-10, yaw sign flips, velocity consistent, wrench invariant")


# ============================================================ #2 predictor
def test_predictor():
    # (a) damping must DECELERATE a moving vehicle (sign of DL/DNL correct)
    for axis, v in [(0, 0.6), (1, 0.6), (2, 0.6)]:
        nu = np.zeros(6); nu[axis] = v
        eta = np.zeros(6)
        a = fossen.nu_dot(eta, nu, np.zeros(6), np.zeros(6))
        assert a[axis] < 0, f"axis {axis}: predicted nu_dot {a[axis]:+.3f} not decelerating (damping sign!)"
    # angular damping too
    nu = np.zeros(6); nu[5] = 0.5
    assert fossen.nu_dot(np.zeros(6), nu, np.zeros(6), np.zeros(6))[5] < 0, "yaw-rate not damped"

    # (b) equilibrium pitch: the surge->pitch coupling My=-k*Fx balanced by the
    #     model restoring (slope ZG*WEIGHT) -> 6 N ~ 23 deg (controller.py:43).
    k = P.SURGE_PITCH_COUPLING
    slope = P.ZG * P.WEIGHT                              # d(restoring moment)/d(theta)
    for Fx, want in [(6.0, 23.0)]:
        theta = np.degrees(np.arcsin(k * Fx / slope))
        assert abs(theta - want) < 3.0, f"trim pitch at {Fx} N = {theta:.1f} deg, want ~{want}"
    # (c) stability: integrating the model under sustained surge+coupling must stay
    #     FINITE and BOUNDED (no anti-damping blow-up, no tumble). The exact open-loop
    #     trim is a coupled fixed point (surge vel, buoyancy ascent, body-frame thrust
    #     rotating with pitch) -- the precise slope is the closed-form check above.
    x = np.zeros(12)
    tau = np.array([6.0, 0, 0, 0, -k * 6.0, 0])         # surge + coupled pitch moment (NED)
    for _ in range(20000):                              # 40 s at 2 ms
        x = fossen.rk4(fossen.f_state, x, 0.002, tau, np.zeros(6))
    theta_eq = np.degrees(x[4])
    assert np.all(np.isfinite(x)), "predictor blew up (anti-damping?)"
    assert abs(theta_eq) < 45.0, f"integrated trim pitch {theta_eq:.1f} deg -> tumbling"
    assert np.linalg.norm(x[6:9]) < 2.0, "predicted linear velocity unbounded"
    print(f"[#2 predictor] OK  damping decelerates; restoring slope -> {theta:.1f} deg (closed-form); "
          f"model stable & bounded (open-loop trim {theta_eq:.1f} deg)")


# ============================================================ #4 EAOB unbiased
def test_eaob_no_accel_doublecount():
    """On a clean NED Fossen plant with a STEADY body force and NO disturbance,
    the EAOB (fed finite-differenced nu as acceleration) must converge w_hat -> 0,
    not to an acceleration-proportional bias. This is the no-double-count check the
    marinegym controller guarantees by using FD instead of data.qacc."""
    dt = P.DT_CTRL
    tau = np.array([3.0, 1.0, -2.0, 0.0, 0.0, 0.5])     # steady commanded wrench
    x = np.zeros(12)                                    # eta; nu  (NED)
    eaob = EAOB(eta0=x[:6].copy())
    nu_prev = x[6:].copy()
    w_hist = []
    for kk in range(400):                               # 20 s
        # advance the true plant one control tick (no disturbance)
        for _ in range(25):
            x = fossen.rk4(fossen.f_state, x, P.DT_SIM, tau, np.zeros(6))
        nu = x[6:].copy()
        nudot = (nu - nu_prev) / dt                     # FD, exactly what the controller does
        nu_prev = nu
        meas = {"eta": x[:6].copy(), "nu": nu, "nudot": nudot}
        _, _, w_hat = eaob.update(meas, tau)
        w_hist.append(w_hat.copy())
    w_tail = np.array(w_hist[-40:])                     # last 2 s
    bias = np.abs(w_tail.mean(axis=0))
    assert np.all(bias < 0.5), f"EAOB w_hat biased under steady force (no disturbance): {bias}"
    print(f"[#4 EAOB] OK  steady-force w_hat -> 0 (max |bias| {bias.max():.3f} N, no accel double-count)")


# ===================================================== copied-model consistency
def test_casadi_matches_numpy():
    """The copied CasADi MPC model (_f_casadi) must equal the copied NumPy fossen
    model (f_state) -- confirms the copy + shared params are consistent."""
    import casadi as ca
    from dobmpc.mpc import _f_casadi
    rng = np.random.default_rng(3)
    xs = ca.SX.sym("x", 12); us = ca.SX.sym("u", 4); ws = ca.SX.sym("w", 6)
    f = ca.Function("f", [xs, us, ws], [_f_casadi(xs, us, ws)])
    maxerr = 0.0
    for _ in range(200):
        x = rng.standard_normal(12); x[4] = np.clip(x[4], -1.0, 1.0)   # keep theta sane
        u = rng.standard_normal(4); w = rng.standard_normal(6)
        tau = np.array([u[0], u[1], u[2], 0.0, 0.0, u[3]])
        num = fossen.f_state(x, tau, w)
        sym = np.array(f(x, u, w)).ravel()
        maxerr = max(maxerr, np.abs(num - sym).max())
    assert maxerr < 1e-9, f"CasADi vs NumPy model mismatch: {maxerr:.2e}"
    print(f"[copy] OK  CasADi model == NumPy fossen model (max |diff| {maxerr:.1e})")


def main():
    test_frames()
    test_predictor()
    test_eaob_no_accel_doublecount()
    test_casadi_matches_numpy()
    print("\nDOBMPC UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
