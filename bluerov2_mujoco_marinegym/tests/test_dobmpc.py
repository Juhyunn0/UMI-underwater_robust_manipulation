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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        kap = P.SURGE_PITCH_COUPLING if getattr(P, "PITCH_AWARE", False) else 0.0
        tau = np.array([u[0], u[1], u[2], 0.0, kap * u[0], u[3]])   # match _f_casadi (option-b coupling)
        num = fossen.f_state(x, tau, w)
        sym = np.array(f(x, u, w)).ravel()
        maxerr = max(maxerr, np.abs(num - sym).max())
    assert maxerr < 1e-9, f"CasADi vs NumPy model mismatch: {maxerr:.2e}"
    print(f"[copy] OK  CasADi model == NumPy fossen model (max |diff| {maxerr:.1e})")


def test_pitch_aware():
    """option (b): the MPC model includes the surge->pitch coupling with the correct
    (+NED) sign, and the pitch state bound is tightened to THETA_MAX. (Empirically this
    caps the square pitch ~67->23 deg while keeping position tracking; see CONTROL_METHODOLOGY.)"""
    import casadi as ca
    from dobmpc.mpc import _f_casadi, NMPC
    assert getattr(P, "PITCH_AWARE", False), "PITCH_AWARE expected on by default"
    xs = ca.SX.sym("x", 12); us = ca.SX.sym("u", 4); ws = ca.SX.sym("w", 6)
    f = ca.Function("f", [xs, us, ws], [_f_casadi(xs, us, ws)])
    Fx = 3.0
    qdd = float(np.array(f(np.zeros(12), [Fx, 0, 0, 0], np.zeros(6))).ravel()[10])  # nu_dot[4]=pitch acc
    pred = P.SURGE_PITCH_COUPLING * Fx / fossen.M_TOTAL[4, 4]                         # +kappa*Fx / M_theta
    assert qdd > 0 and abs(qdd - pred) < 1e-6, f"coupling sign/mag wrong: {qdd:.4f} vs {pred:.4f}"
    nmpc = NMPC(N=5)
    th_ub = nmpc._ubx[1 * 12 + 4]                    # |theta| upper bound at a horizon step
    assert abs(th_ub - P.THETA_MAX) < 1e-9 and th_ub < 1.2, f"theta bound not THETA_MAX: {th_ub}"
    print(f"[option-b] OK  surge->pitch coupling +kappa (pitch acc {qdd:.4f}==pred), "
          f"|theta|<={P.THETA_MAX} rad bound (<1.2)")


def test_xref_yaw_preview():
    """_xref_ned: (a) r_ref=0 & v_ref=0 reduces EXACTLY to the constant-pose tile
    (DP unchanged), (b) a world-FLU +yaw-rate lands in the NED yaw-rate slot xref[11]
    with the correct -r_ref sign (S flip), (c) the yaw-angle preview ramps toward the
    final heading and is CLAMPED there (never over-rotates past the corner)."""
    import types
    from dobmpc_controller import DOBMPCController, _Rz_flu

    def stub(yaw_ref, yaw_target, r_ref, psi_now=0.0, v_ref=(0, 0, 0)):
        s = types.SimpleNamespace()
        s.nmpc = types.SimpleNamespace(N=60)
        s.p_ref = np.zeros(3)
        s.v_ref = np.asarray(v_ref, float)
        s.yaw_ref = yaw_ref
        s.yaw_target = yaw_target
        s.r_ref = r_ref
        s._psi_ned_now = psi_now
        return s

    # (a) DP-equivalence: no motion, no turn -> velocity/rate rows 0, orientation constant
    x0 = DOBMPCController._xref_ned(stub(0.3, 0.3, 0.0, psi_now=-0.3))
    assert np.allclose(x0[6:12, :], 0.0), "DP: velocity/rate reference must be zero"
    assert np.allclose(x0[3:6, :], x0[3:6, :1]), "DP: orientation must be constant over horizon"

    # (b) turning: world-FLU +yaw-rate r -> NED yaw-rate slot xref[11] = -r while ramping
    r = 1.047
    x1 = DOBMPCController._xref_ned(stub(yaw_ref=0.0, yaw_target=np.pi / 2, r_ref=r, psi_now=0.0))
    assert np.isclose(x1[11, 1], -r), f"xref[11] must be -r_ref (S sign flip), got {x1[11, 1]:.4f}"

    # (c) clamp: NED target = -pi/2 (yaw flips FLU->NED); ramp reaches it and stops, rate->0
    assert np.isclose(x1[5, -1], -np.pi / 2, atol=1e-6), f"yaw not clamped at target: {x1[5, -1]:.4f}"
    assert np.isclose(x1[11, -1], 0.0), "rate FF must be 0 once clamped at the target"
    # monotone toward target, never past it (all stages within [target, start])
    assert np.all(x1[5, :] >= -np.pi / 2 - 1e-9) and np.all(x1[5, :] <= 1e-9), "yaw preview over-rotated"
    print("[yaw-preview] OK  DP-equivalent at r_ref=0; xref[11]=-r_ref (S sign); ramp clamped at target")


def test_xref_traj_preview():
    """_xref_ned with a mission trajectory sampler (tracking mode):
    (a) a constant sampler reproduces the constant-pose DP tile EXACTLY,
    (b) a corner inside the horizon BENDS the position preview at the corner stage
        (the setpoint body would extrapolate straight through),
    (c) the stage-wise body-velocity reference rotates onto the new leg + the
        yaw-rate FF lands in xref[11] with the -r sign flip,
    (d) stage-to-stage yaw unwrap stays continuous across +-pi, anchored at psi_now."""
    import types
    from dobmpc_controller import DOBMPCController

    N = 60
    dt = P.DT_CTRL

    def stub(traj, psi_now=0.0):
        s = types.SimpleNamespace()
        s.nmpc = types.SimpleNamespace(N=N)
        s._ref_traj = traj
        s._psi_ned_now = psi_now
        return s

    # (a) constant sampler == the constant-pose DP tile from the setpoint body
    p0, yaw0 = np.array([0.4, -0.3, 0.2]), 0.3

    def const_traj(ts):
        K = np.asarray(ts).size
        return (np.tile(p0[:, None], (1, K)), np.full(K, yaw0),
                np.zeros((3, K)), np.zeros(K))

    sp = types.SimpleNamespace(nmpc=types.SimpleNamespace(N=N), p_ref=p0,
                               v_ref=np.zeros(3), yaw_ref=yaw0, yaw_target=yaw0,
                               r_ref=0.0, _psi_ned_now=-0.3, _ref_traj=None)
    x_dp = DOBMPCController._xref_ned(sp)                          # setpoint body
    x_tr = DOBMPCController._xref_ned_traj(stub(const_traj, psi_now=-0.3), 5.0)
    assert np.allclose(x_dp, x_tr, atol=1e-12), "constant sampler must equal the DP tile"

    # (b)+(c) corner at t_c: +x leg then +y leg, yaw ramps at w from the vertex
    v, t_c, w = 0.15, 1.0, np.radians(60.0)

    def corner_traj(ts):
        ts = np.asarray(ts, float)
        K = ts.size
        p = np.empty((3, K)); vv = np.empty((3, K))
        yaw = np.empty(K); r = np.empty(K)
        for j, t in enumerate(ts):
            if t < t_c:
                p[:, j] = (v * t, 0.0, 0.0); vv[:, j] = (v, 0.0, 0.0)
                yaw[j] = 0.0; r[j] = 0.0
            else:
                p[:, j] = (v * t_c, v * (t - t_c), 0.0); vv[:, j] = (0.0, v, 0.0)
                ramp = min(w * (t - t_c), np.pi / 2)
                yaw[j] = ramp; r[j] = w if ramp < np.pi / 2 else 0.0
        return p, yaw, vv, r

    x = DOBMPCController._xref_ned_traj(stub(corner_traj), 0.0)
    kc = int(np.ceil(t_c / dt))                     # first stage at/past the vertex
    assert np.allclose(x[0, :kc], v * np.arange(kc) * dt), "pre-corner: preview along +x"
    assert np.allclose(x[0, kc:], v * t_c), "post-corner: x must freeze at the vertex"
    assert np.all(x[1, kc + 1:] < -1e-6), "post-corner: NED y must decrease (FLU +y leg)"
    assert np.allclose(x[7, :kc], 0.0, atol=1e-9), "pre-corner: no sway reference"
    assert np.any(np.abs(x[7, kc:]) > 0.01), "post-corner: sway reference must appear"
    assert np.allclose(x[11, :kc], 0.0) and np.isclose(x[11, kc + 1], -w), \
        "yaw-rate FF must be 0 before the corner and -r (S flip) during the ramp"

    # (d) unwrap continuity: FLU yaw 170 -> 190 deg crosses +-pi in NED (-170 -> -190)
    def wrapcross_traj(ts):
        K = np.asarray(ts).size
        yaw = np.radians(170.0) + np.radians(20.0) * np.linspace(0.0, 1.0, K)
        return (np.zeros((3, K)), yaw, np.zeros((3, K)), np.zeros(K))

    psi0 = -np.radians(170.0)                       # NED yaw = -FLU yaw
    xw = DOBMPCController._xref_ned_traj(stub(wrapcross_traj, psi_now=psi0), 0.0)
    dpsi = np.diff(xw[5, :])
    assert np.all(np.abs(dpsi) < 0.02), f"yaw preview jumped across +-pi: {np.abs(dpsi).max():.3f}"
    assert np.isclose(xw[5, 0], psi0, atol=1e-9), "yaw preview must anchor at psi_now"
    print("[traj-preview] OK  constant==DP tile; corner bends position+velocity preview; "
          "rate-FF sign; yaw unwrap continuous across +-pi")


def test_square_ref_matches_live_loop():
    """make_square_ref's load-bearing claim: the precomputed heading profile equals
    the live loop's slew_heading recursion at every physics step (the heading
    command has no vehicle feedback, so its future is knowable). One small lap
    (all 4 corners incl. the +-pi leg) at the real physics dt; p/v must equal
    square_setpoint exactly and yaw/r must match the recursion exactly on-grid.
    If someone ever adds vehicle feedback to the heading command, this fails."""
    from experiments.run_compare import square_setpoint, slew_heading, make_square_ref

    size, speed, depth, dt = 0.25, 0.15, 0.0, 0.002
    yaw_rate = np.radians(60.0)
    T = 4.0 * size / speed                                     # one lap = 6.67 s
    ref = make_square_ref(size, speed, depth, True, yaw_rate, dt, T + 4.0)

    yaw_cmd = 0.0
    ts = np.arange(0.0, T, dt)
    p, yaw, v, r = ref(ts)                                     # one vectorized call
    for i, t in enumerate(ts):
        (rx, ry), (tx, ty) = square_setpoint(t, size, speed)
        yaw_new = slew_heading(yaw_cmd, tx, ty, yaw_rate, dt)
        r_cmd = (yaw_new - yaw_cmd) / dt
        yaw_cmd = yaw_new
        assert abs(p[0, i] - rx) < 1e-12 and abs(p[1, i] - ry) < 1e-12, f"p mismatch @t={t}"
        assert abs(v[0, i] - speed * tx) < 1e-12 and abs(v[1, i] - speed * ty) < 1e-12
        assert abs(yaw[i] - yaw_cmd) < 1e-9, f"yaw sampler != live loop @t={t:.3f}"
        assert abs(r[i] - r_cmd) < 1e-9, f"r sampler != live loop @t={t:.3f}"
    # at t=T-dt the 4th-corner turn (vertex exactly at t=T) has not started yet, so
    # the CCW accumulation stands at 3 completed 90-deg turns = 3pi/2 (no wrap-back)
    assert abs(yaw_cmd - 1.5 * np.pi) < 0.05, f"expected ~3pi/2 accumulated, got {yaw_cmd:.3f}"
    print("[square-ref] OK  sampler == live slew recursion over a full lap "
          "(p/v exact, yaw/r < 1e-9)")


def main():
    test_frames()
    test_predictor()
    test_eaob_no_accel_doublecount()
    test_casadi_matches_numpy()
    test_pitch_aware()
    test_xref_yaw_preview()
    test_xref_traj_preview()
    test_square_ref_matches_live_loop()
    print("\nDOBMPC UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
