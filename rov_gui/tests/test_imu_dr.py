#!/usr/bin/env python3
"""test_imu_dr.py — the dead reckoner against its own closed-form error laws.

Pure numpy, no Qt, no hardware:  python rov_gui/tests/test_imu_dr.py

The first three tests demand EXACTNESS (1e-9), and that is the point of them.
This module is judged on how a t^2 term grows, so the integrator must not
contribute a t^2 term of its own — a rectangle rule or a leaked gravity
component would pass a "close enough" tolerance and then show up in a pool run
as sensor drift that is actually arithmetic.

The rest pin the laws quoted in the plan and in imu_dr's docstring:

    accel bias b        e = 0.5 b t^2
    gyro bias beta      e = (1/6) g beta t^3      (pure integration)
    AHRS at tau         standing tilt = beta*tau
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.control.imu_dr import (            # noqa: E402
    G, ImuCalibration, ImuDeadReckoner, euler_zyx, orthonormalize, rodrigues)
from rov_gui.control.state_assembler import G_NED, rot_zyx   # noqa: E402

HZ = 200.0
DT = 1.0 / HZ


# =============================================================================
# helpers — a synthetic IMU that is exactly consistent with a chosen motion
# =============================================================================
def samples(T, f_body, w_body=(0.0, 0.0, 0.0), t0=0.0, hz=HZ):
    """(N,7) of a sensor reading ``f_body`` / ``w_body``, constant."""
    n = int(round(T * hz)) + 1
    t = t0 + np.arange(n) / hz
    a = np.zeros((n, 7))
    a[:, 0] = t
    a[:, 1:4] = np.asarray(f_body, float)
    a[:, 4:7] = np.asarray(w_body, float)
    return a


def level_static_f():
    """What a perfect accelerometer reads, level and still: specific force."""
    return -np.eye(3).T @ G_NED          # (0, 0, -9.80665)


def dr(attitude="gyro", **kw):
    d = ImuDeadReckoner(attitude=attitude, z_source="imu", **kw)
    d.anchor(np.zeros(6), t=0.0)
    return d


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")


# =============================================================================
# 1-3: exactness of the mechanization
# =============================================================================
def test_constant_velocity_integrates_exactly():
    """Level, still sensor + a non-zero anchor velocity must reproduce
    p = v*t to machine precision for a minute. Any gravity leak lands here."""
    T, v0 = 60.0, np.array([0.05, -0.03, 0.0])
    d = ImuDeadReckoner(attitude="gyro", z_source="imu")
    d.anchor(np.zeros(6), nu_world_ned=v0, t=0.0, zero_velocity=False)
    d.integrate(samples(T, level_static_f()))
    err = float(np.linalg.norm(d.p - v0 * T))
    check("constant velocity", err < 1e-9, f"{err:.3e} m after {T} s")
    check("velocity held", float(np.linalg.norm(d.v - v0)) < 1e-12)


def test_constant_acceleration_integrates_exactly():
    T, a0 = 20.0, np.array([0.05, 0.0, 0.0])
    d = dr()
    d.integrate(samples(T, a0 - G_NED))           # f = R^T (a - g), R = I
    want = 0.5 * a0 * T * T
    err = float(np.linalg.norm(d.p - want))
    check("constant acceleration", err < 1e-9,
          f"{err:.3e} m, got {d.p}, want {want}")


def test_accel_bias_grows_exactly_as_half_b_t_squared():
    """The law the whole error budget is built on, checked against the
    integrator rather than assumed."""
    T, b = 30.0, np.array([0.01, 0.0, 0.0])
    d = dr()
    d.integrate(samples(T, level_static_f() + b))
    want = 0.5 * b * T * T                        # 4.5 m at 30 s
    err = float(np.linalg.norm(d.p - want))
    check("accel bias", err < 1e-9, f"{err:.3e} m (want |e| = {want[0]:.2f} m)")


# =============================================================================
# 4-6: attitude
# =============================================================================
def test_gyro_bias_grows_as_one_sixth_g_beta_t_cubed():
    beta = math.radians(0.01)                     # 0.01 deg/s about body y
    T = 20.0
    d = dr()
    d.integrate(samples(T, level_static_f(), (0.0, beta, 0.0)))
    want = (1.0 / 6.0) * G * beta * T ** 3
    got = float(np.linalg.norm(d.p))
    check("gyro bias t^3", abs(got - want) < 0.01 * want,
          f"got {got:.4f} m, want {want:.4f} m")
    # ...and it leans the way a nose-up pitch error should: +pitch tips the
    # measured gravity into -x, so the estimate runs backwards.
    check("gyro bias sign", d.p[0] < 0, f"p = {d.p}")


def test_ahrs_bounds_the_tilt_a_gyro_bias_would_run_away_with():
    """Two claims at once: the standing tilt is beta*tau, and levelling only
    beats pure integration once t is well past 3*tau (imu_dr._level)."""
    beta, tau, T = math.radians(0.05), 2.0, 40.0
    a = ImuDeadReckoner(attitude="ahrs", z_source="imu", ahrs_tau_s=tau)
    a.anchor(np.zeros(6), t=0.0)
    a.integrate(samples(T, level_static_f(), (0.0, beta, 0.0)))
    _phi, theta, _psi = euler_zyx(a.R)
    check("ahrs standing tilt", abs(abs(theta) - beta * tau) < 0.1 * beta * tau,
          f"tilt {math.degrees(theta):.4f} deg, want "
          f"{math.degrees(beta * tau):.4f} deg")

    g = dr()
    g.integrate(samples(T, level_static_f(), (0.0, beta, 0.0)))
    ratio = float(np.linalg.norm(g.p)) / max(1e-9, float(np.linalg.norm(a.p)))
    check("ahrs beats gyro past 3 tau", ratio > 3.0,
          f"gyro/ahrs = {ratio:.2f} at t={T}s, tau={tau}s")
    check("ahrs never touches yaw", abs(euler_zyx(a.R)[2]) < 1e-12,
          f"yaw {euler_zyx(a.R)[2]:.3e} rad")


def test_ahrs_ignores_the_accelerometer_during_a_thruster_spike():
    """A 2 m/s^2 kick is not gravity moving, and must not be allowed to pull
    the attitude. The gate is |‖f‖ - g| > accel_trust_m_s2."""
    d = ImuDeadReckoner(attitude="ahrs", z_source="imu", accel_trust_m_s2=0.15)
    d.anchor(np.zeros(6), t=0.0)
    kick = samples(0.2, level_static_f() + np.array([2.0, 0.0, 0.0]))
    d.integrate(kick)
    check("spike rejected", d.n_ungated == kick.shape[0] - 1,
          f"ungated {d.n_ungated} of {kick.shape[0] - 1}")
    check("attitude untouched", float(np.linalg.norm(d.R - np.eye(3))) < 1e-12)
    # ...and the kick still moves the position: the gate suppresses levelling,
    # not the accelerometer.
    check("spike still integrates", d.p[0] > 0.03, f"p = {d.p}")


# =============================================================================
# 7-9: calibration, anchoring, staleness
# =============================================================================
def test_static_correction_is_exact_at_its_own_attitude_and_decays_off_it():
    """The +20% scale is removed exactly where it was measured, and the
    residual off that attitude is s*g*sin(delta) — the number that makes the
    bench ellipsoid fit non-optional."""
    s = 1.20
    calib = ImuCalibration()                       # no bench fit
    d = ImuDeadReckoner(attitude="gyro", z_source="imu", calib=calib,
                        static_window_s=8.0)
    f_true = level_static_f()
    d.integrate(samples(8.0, f_true * s))          # settle: reads 20% high
    out = d.calibrate_static(np.eye(3))
    check("static ok", out["ok"], out.get("why", ""))
    check("static magnitude seen", abs(out["accel_mag_mean"] - s * G) < 1e-9)

    d.anchor(np.zeros(6), t=8.0)
    d.integrate(samples(10.0, f_true * s, t0=8.0))
    check("corrected at its own attitude", float(np.linalg.norm(d.p)) < 1e-9,
          f"p = {d.p}")

    # Now hold a 2 degree pitch: the SAME correction leaves s*g*sin(2 deg).
    delta = math.radians(2.0)
    R = rot_zyx(0.0, delta, 0.0)
    d2 = ImuDeadReckoner(attitude="gyro", z_source="imu", static_window_s=8.0)
    d2.integrate(samples(8.0, (-np.eye(3) @ G_NED) * s))
    d2.calibrate_static(np.eye(3))
    d2.anchor(np.array([0.0, 0.0, 0.0, 0.0, delta, 0.0]), t=8.0)
    d2.integrate(samples(10.0, (-R.T @ G_NED) * s, t0=8.0))
    want = 0.5 * (s - 1.0) * G * math.sin(delta) * 100.0     # 0.5*a*t^2, t=10
    got = float(np.linalg.norm(d2.p))
    check("residual off-attitude", abs(got - want) < 0.15 * want,
          f"got {got:.3f} m, want ~{want:.3f} m at 2 deg after 10 s")


def test_static_correction_follows_the_attitude_the_window_actually_held():
    """A station-keeping vehicle holds its POSITION, not its attitude. Pinning
    the window's mean specific force to the anchor's single attitude turns that
    difference into permanent bias; measuring the residual per sample does not.

    The ramp here is the one that was actually flown: pitch -0.34 -> +2.15 deg
    across an 8 s settle
    [측정: sessions/low_level_controller_data/20260818/0818_160139/mpc_161904
     _rov.jsonl, ATTITUDE over the settle window].
    """
    T, n = 8.0, int(8.0 * HZ) + 1
    t = np.arange(n) / HZ
    the = np.radians(-0.34 + (2.15 + 0.34) * t / T)
    arr = np.zeros((n, 7))
    arr[:, 0] = t
    for i in range(n):                       # a PERFECT sensor: true c = 0
        arr[i, 1:4] = -rot_zyx(0.0, float(the[i]), 0.0).T @ G_NED
    att_ref = np.column_stack([t, np.zeros(n), the])
    eta_end = np.array([0.0, 0.0, 0.0, 0.0, float(the[-1]), 0.0])
    R_end = rot_zyx(0.0, float(the[-1]), 0.0)

    frozen = ImuDeadReckoner(attitude="gyro", z_source="imu",
                             static_window_s=T)
    frozen.integrate(arr)
    out = frozen.calibrate_static(R_end)                 # no att_ref
    check("frozen accepted", out["accel_ok"], str(out))
    leaked = float(np.linalg.norm(frozen._c_static))
    check("frozen attitude leaks bias", leaked > 0.15,
          f"only {leaked:.4f} m/s^2 leaked — the ramp should leave ~0.21")
    check("frozen says so", "single sample" in out["accel_ref"],
          out["accel_ref"])

    live = ImuDeadReckoner(attitude="gyro", z_source="imu", static_window_s=T)
    live.integrate(arr)
    out2 = live.calibrate_static(R_end, att_ref=att_ref)
    check("per-sample is exact", float(np.linalg.norm(live._c_static)) < 1e-9,
          f"c_static = {live._c_static}")
    check("per-sample says so", "attitude series" in out2["accel_ref"],
          out2["accel_ref"])

    # ...and the consequence, on 10 s of held-still samples at that attitude.
    for d, name, want in ((frozen, "frozen", 0.5 * leaked * 100.0),
                          (live, "per-sample", 0.0)):
        d.anchor(eta_end, t=T)
        d.integrate(samples(10.0, -R_end.T @ G_NED, t0=T))
        got = float(np.linalg.norm(d.p))
        check(f"{name} drift", abs(got - want) < max(1e-9, 0.05 * want),
              f"{name}: {got:.4f} m, want {want:.4f} m")


def test_a_deck_measured_gyro_bias_is_rotated_into_the_body_frame():
    """The JSON's gyro bias lives in IMU axes, like the accel bias beside it,
    but _propagate subtracts it from rates calib.correct() has ALREADY rotated.
    R_frd_imu is a ~48 deg rotation on this vehicle, so an unrotated
    subtraction lands a deck-measured bias on the wrong axes."""
    R = rot_zyx(0.3, -0.4, 1.1)
    b_imu = np.array([0.004, -0.002, 0.001])
    calib = ImuCalibration(R_frd_imu=R, gyro_bias=b_imu)
    d = ImuDeadReckoner(calib, attitude="gyro", z_source="imu",
                        static_window_s=8.0)
    # A still window whose gyro reads exactly the bias, in IMU axes.
    arr = samples(8.0, level_static_f() @ R, w_body=b_imu)
    d.integrate(arr)
    out = d.calibrate_static(np.eye(3))
    check("used the file", "deck" in out["gyro_bias_source"],
          out.get("gyro_bias_source"))
    check("rotated", np.allclose(d._b_gyro, R @ b_imu, atol=1e-12),
          f"{d._b_gyro} vs {R @ b_imu}")

    # ...and the consequence: with it rotated the attitude must not move.
    d.anchor(np.zeros(6), t=8.0)
    d.integrate(samples(20.0, level_static_f() @ R, w_body=b_imu, t0=8.0))
    check("attitude held", float(np.linalg.norm(d.R - np.eye(3))) < 1e-9,
          f"drifted {math.degrees(float(np.linalg.norm(d.R - np.eye(3)))):.3f}")


def test_the_settle_window_reports_how_well_it_determined_anything():
    """Accepted is the weaker statement. Two real windows passed the SEM gate
    with 27x and 50x margin and still disagreed on the yaw-gyro bias by 68%
    [측정: 20260818/0818_160139 vs 0818_163342, dr_yaw_deg - yaw_deg], so the
    numbers behind the verdict have to reach the run meta."""
    d = ImuDeadReckoner(attitude="ahrs", z_source="imu", static_window_s=8.0)
    d.integrate(samples(8.0, level_static_f()))
    d.calibrate_static(np.eye(3))
    q = d.meta()["static_quality"]
    for k in ("n", "gyro_bias_sem", "gyro_std", "accel_mag_std",
              "accel_mag_mean"):
        check(f"meta carries {k}", q.get(k) is not None, str(q))
    check("meta carries the accel reference",
          d.meta()["accel_static_source"] != "none", str(d.meta()))


def test_calibrate_static_refuses_a_window_that_was_not_still():
    d = ImuDeadReckoner(attitude="ahrs", z_source="imu",
                        gyro_static_std_max=0.02)
    n = int(8.0 * HZ)
    arr = samples(8.0, level_static_f())
    rng = np.random.default_rng(0)
    arr[:, 4:7] += rng.normal(0.0, 0.05, size=(n + 1, 3))
    d.integrate(arr)
    out = d.calibrate_static(np.eye(3))
    check("not-still refused", not out["ok"] and "not still" in out["why"],
          str(out))
    check("note surfaced", "not still" in d.static_note, d.static_note)


def test_anchor_seeds_the_state_exactly_and_zeroes_velocity():
    d = ImuDeadReckoner(attitude="gyro", z_source="imu")
    eta = np.array([1.5, -0.4, -1.1, 0.02, -0.03, 1.2])
    d.anchor(eta, nu_world_ned=np.array([9.0, 9.0, 9.0]), t=3.0)
    check("position", np.allclose(d.p, eta[:3], atol=0, rtol=0))
    check("velocity zeroed", np.allclose(d.v, 0.0))
    check("attitude", np.allclose(euler_zyx(d.R), eta[3:6], atol=1e-12))
    d.anchor(eta, nu_world_ned=np.array([0.1, 0.2, 0.3]), t=3.0,
             zero_velocity=False)
    check("velocity kept", np.allclose(d.v, [0.1, 0.2, 0.3]))


def test_a_starved_dead_reckoner_reports_not_ok_rather_than_a_frozen_point():
    """The worst failure mode this experiment has: no samples looks like a
    perfect estimate. It must be loud."""
    d = dr()
    st = d.state(1.0)
    check("no samples not ok", not st["ok"] and "no imu samples" in st["why"],
          str(st))
    d.integrate(samples(1.0, level_static_f()))
    check("fresh is ok", d.state(1.0)["ok"])
    st = d.state(5.0)
    check("stale not ok", not st["ok"] and "stale" in st["why"], str(st))
    check("elapsed still reported", st["elapsed"] == 5.0, str(st["elapsed"]))


def test_holes_and_out_of_order_samples_are_dropped_not_integrated():
    d = dr(max_dt_s=0.1)
    arr = samples(1.0, level_static_f() + np.array([1.0, 0.0, 0.0]))
    arr[100, 0] += 5.0                       # a 5 s hole, then time goes back
    n = d.integrate(arr)
    check("rejected counted", d.n_rejected == 2, f"{d.n_rejected}")
    check("propagated the rest", n == arr.shape[0] - 3, f"{n}/{arr.shape[0]}")
    # 5 s of a 1 m/s^2 bias would be 12.5 m if the hole had been integrated.
    check("hole not integrated", float(np.linalg.norm(d.p)) < 1.0,
          f"p = {d.p}")


def test_before_the_anchor_nothing_propagates():
    d = ImuDeadReckoner(attitude="gyro", z_source="imu")
    n = d.integrate(samples(2.0, level_static_f() + np.array([1.0, 0, 0])))
    check("nothing propagated", n == 0 and np.allclose(d.p, 0.0))
    check("but the window filled", d.static_window().shape[0] > 300,
          str(d.static_window().shape))


# =============================================================================
# 10: the pressure passthrough (the operator's z decision)
# =============================================================================
def test_pressure_z_comes_from_the_barometer_not_from_the_accelerometer():
    d = ImuDeadReckoner(attitude="gyro", z_source="pressure")
    d.anchor(np.zeros(6), t=0.0)
    # A heavy vertical bias that would sink the estimate if z were integrated.
    d.integrate(samples(10.0, level_static_f() + np.array([0.0, 0.0, 0.5])))
    d.note_depth(-1.23, 100.0)
    st = d.state(10.0)
    check("ok", st["ok"], st["why"])
    check("z from pressure", abs(st["meas"]["eta"][2] + 1.23) < 1e-12,
          str(st["meas"]["eta"]))
    # ...and the IMU's own z is still reported, for the offline comparison.
    check("imu z recorded", abs(st["pz_imu"] - 0.5 * 0.5 * 100.0) < 1e-6,
          f"pz_imu = {st['pz_imu']}")
    # A heave RATE, differenced from the barometer rather than integrated.
    d.note_depth(-1.13, 101.0)
    st = d.state(10.0)
    w = float(st["meas"]["nu"][2])
    check("heave rate differenced", 0.02 < w < 0.10, f"w = {w}")


def test_the_barometer_keeps_the_estimate_alive_when_the_tag_dies():
    """The regression this design exists for: reading z out of the TAG state
    made the dead reckoner report not-ok on exactly the ticks where it was the
    only thing that still knew anything."""
    d = ImuDeadReckoner(attitude="gyro", z_source="pressure")
    d.anchor(np.zeros(6), t=0.0)
    d.integrate(samples(2.0, level_static_f() + np.array([0.05, 0.0, 0.0])))
    d.note_depth(-1.0, 100.0)
    st = d.state(2.0, meas_tag=None)        # no tag state at all
    check("alive without a tag", st["ok"], st["why"])
    check("and still moving", st["p_ned"][0] > 0.05, str(st["p_ned"]))


def test_without_any_depth_source_it_says_so_rather_than_guessing():
    d = ImuDeadReckoner(attitude="gyro", z_source="pressure")
    d.anchor(np.zeros(6), t=0.0)
    d.integrate(samples(1.0, level_static_f()))
    st = d.state(1.0)
    check("refuses", not st["ok"] and "depth" in st["why"], str(st))


# =============================================================================
# small numerics
# =============================================================================
def test_rodrigues_and_orthonormalize():
    w = np.array([0.3, -0.2, 0.7])
    R = rodrigues(w, 1.0)
    check("rodrigues is a rotation", abs(np.linalg.det(R) - 1.0) < 1e-12)
    check("rodrigues matches rot_zyx round trip",
          np.allclose(R @ R.T, np.eye(3), atol=1e-12))
    check("tiny angle is identity", np.allclose(rodrigues(w, 0.0), np.eye(3)))
    bad = np.eye(3) + 1e-3 * np.ones((3, 3))
    Rn = orthonormalize(bad)
    check("orthonormalize", abs(np.linalg.det(Rn) - 1.0) < 1e-12)


def test_euler_round_trips_through_rot_zyx():
    for e in ([0.1, -0.2, 0.3], [0.0, 0.0, -3.0], [-0.4, 0.5, 2.9]):
        got = euler_zyx(rot_zyx(*e))
        check("euler round trip", np.allclose(got, e, atol=1e-12),
              f"{e} -> {got}")


# =============================================================================
# the bench calibrations (rov_gui/tools/calib_c3_imu.py)
# =============================================================================
def _tumble(poses=40, hold=100, reads_high=(1.20, 1.17, 1.22),
            bias=(0.30, -0.15, 0.05), seed=3, hz=HZ):
    """A synthetic tumble, HELD at each attitude — the shape a real one has.

    Poses, not a cloud of random directions: the static detector keys off the
    steadiness of |a| over a window, so a fixture that teleports between
    orientations every sample is correctly rejected as "never still" and would
    make this test pass only by disabling the gate it should be exercising.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(poses, 3))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    meas = (u * G) * np.asarray(reads_high) + np.asarray(bias)
    a = np.repeat(meas, hold, axis=0)
    t = np.arange(a.shape[0]) / hz
    return a, np.zeros_like(a), t


def test_accel_ellipsoid_fit_recovers_a_known_scale_and_bias():
    from rov_gui.tools.calib_c3_imu import fit_accel

    high, bias = (1.20, 1.17, 1.22), (0.30, -0.15, 0.05)
    a, w, t = _tumble(reads_high=high, bias=bias)
    out = fit_accel(a, w, t)
    check("fit succeeded", "error" not in out, out.get("error", ""))
    # The correction is (a - b) * s, so s must UNDO the over-reading.
    got_s = np.asarray(out["scale"])
    want_s = 1.0 / np.asarray(high)
    check("scale", np.allclose(got_s, want_s, rtol=2e-3),
          f"{got_s} vs {want_s}")
    check("bias", np.allclose(out["bias"], bias, atol=2e-3), str(out["bias"]))
    check("residual improved",
          out["residual_rms_after"] < 0.02 < out["residual_rms_before"],
          f"{out['residual_rms_before']:.3f} -> {out['residual_rms_after']:.3f}")
    # And it recovers the +20%-ish magnitude the real camera shows.
    check("magnitude reported", out["mean_magnitude_before"] > G,
          str(out["mean_magnitude_before"]))


def test_accel_fit_refuses_a_tumble_that_did_not_cover_the_sphere():
    """The dangerous failure: a lazy tumble produces three CONFIDENT wrong
    scale factors, and nothing downstream would know."""
    from rov_gui.tools.calib_c3_imu import fit_accel

    rng = np.random.default_rng(5)
    # A wobble about one attitude only.
    u = np.array([0.02, 0.01, 1.0]) + rng.normal(0.0, 0.002, size=(3000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    out = fit_accel(u * G, np.zeros((3000, 3)), np.arange(3000) / HZ)
    check("refused", "error" in out, str(out)[:200])
    # ...and it must name the attitudes that are MISSING, not just complain.
    # A refusal an operator cannot act on sends them to repeat the same take.
    check("names the missing poses", "nose up" in out["error"], out["error"])
    check("only the covered axis passes",
          out["axis_coverage"]["+z"] >= 100
          and out["axis_coverage"]["-z"] < 100, str(out["axis_coverage"]))


def test_kabsch_recovers_a_known_rotation_and_a_known_clock_lag():
    from rov_gui.control.state_assembler import rot_zyx as _R
    from rov_gui.tools.calib_c3_imu import estimate_lag, fit_rotation

    R_true = _R(0.3, -0.7, 2.1)          # body <- imu
    lag_true = 0.037                      # the C3 clock runs 37 ms early
    rng = np.random.default_rng(11)
    t = np.arange(0.0, 90.0, 1.0 / 200.0)
    # Three incommensurate frequencies so all three axes carry signal.
    w_imu = np.stack([0.6 * np.sin(2 * np.pi * 0.31 * t),
                      0.5 * np.sin(2 * np.pi * 0.47 * t + 1.0),
                      0.7 * np.sin(2 * np.pi * 0.23 * t + 2.0)], axis=1)
    w_imu += rng.normal(0.0, 0.002, w_imu.shape)
    c3 = {"t": t, "g": w_imu, "a": np.tile([0.0, 0.0, -G], (t.size, 1))}
    t_rov = np.arange(0.0, 90.0, 1.0 / 100.0)
    w_body = np.stack([np.interp(t_rov, t - lag_true, (w_imu @ R_true.T)[:, k])
                       for k in range(3)], axis=1)
    rov = {"t": t_rov, "w": w_body + 0.01}        # + a constant bias

    # Sign convention: the fixture stamps the autopilot copy 37 ms EARLIER
    # than the C3's, so the offset that lines them up is -37 ms — see
    # estimate_lag's docstring, which is the only place this is written down.
    lag, corr = estimate_lag(c3["t"], c3["g"], rov["t"], rov["w"])
    check("lag", abs(lag + lag_true) < 0.002, f"{lag * 1000:.1f} ms")
    check("correlation", corr > 0.9, f"{corr:.3f}")

    out = fit_rotation(c3, rov)
    check("fit succeeded", "error" not in out, out.get("error", ""))
    R = np.asarray(out["R_frd_imu"])
    ang = math.degrees(math.acos(
        max(-1.0, min(1.0, (np.trace(R_true.T @ R) - 1.0) / 2.0))))
    check("rotation", ang < 0.5, f"{ang:.3f} deg off the true rotation")
    check("residual small", out["rms_deg"] < 3.0, str(out["rms_deg"]))


def test_kabsch_refuses_a_single_axis_wiggle():
    """One axis leaves the rotation under-determined, and Kabsch will still
    hand back a matrix. Refusing is the whole value of the check."""
    from rov_gui.tools.calib_c3_imu import fit_rotation

    t = np.arange(0.0, 60.0, 1.0 / 200.0)
    w = np.zeros((t.size, 3))
    w[:, 2] = 0.6 * np.sin(2 * np.pi * 0.3 * t)         # yaw only
    c3 = {"t": t, "g": w, "a": np.tile([0.0, 0.0, -G], (t.size, 1))}
    rov = {"t": t, "w": w.copy()}
    out = fit_rotation(c3, rov)
    check("refused", "error" in out, str(out)[:200])
    check("says why", "three-dimensional" in out["error"], out["error"])


def test_the_gravity_crosscheck_catches_an_axis_swap():
    from rov_gui.tools.calib_c3_imu import gravity_crosscheck

    n = 2000
    c3 = {"t": np.arange(n) / 200.0,
          "a": np.tile([0.0, 0.0, -G], (n, 1)),
          "g": np.zeros((n, 3))}
    ok = gravity_crosscheck(c3, np.eye(3), [1.0] * 3, [0.0] * 3)
    check("identity passes", ok["angle_from_expected_deg"] < 1e-6, str(ok))
    swap = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])   # y <-> z
    bad = gravity_crosscheck(c3, swap, [1.0] * 3, [0.0] * 3)
    check("swap caught", bad["angle_from_expected_deg"] > 80.0, str(bad))


# =============================================================================
# the offline analysis (rov_gui/tools/plot_imu_dr.py)
# =============================================================================
def _fake_run(tmp: Path, b=(0.03, 0.0, 0.0), yaw_deg=0.0, T=20.0, hz=200.0):
    """A synthetic run folder: a vehicle parked at the origin, and an IMU that
    reads gravity plus a known bias. The dead-reckoned answer is exactly
    0.5*b*t^2 in the BODY x direction, so the re-estimator has one right
    answer and no room to be plausibly wrong."""
    import json as _json

    n = int(T * hz)
    t = np.arange(n) / hz
    psi = math.radians(-yaw_deg)                   # yaw_deg is FLU
    f = -rot_zyx(0.0, 0.0, psi).T @ G_NED          # level, still, at this yaw
    stem = tmp / "mpc_000000"
    with open(f"{stem}_c3_imu.jsonl", "w") as fh:
        for k in range(n):
            fh.write(_json.dumps({
                "t": float(t[k]), "src": "c3", "msg": "IMU",
                "t_device": float(t[k]), "seq": k,
                "ax": float(f[0] + b[0]), "ay": float(f[1] + b[1]),
                "az": float(f[2] + b[2]),
                "gx": 0.0, "gy": 0.0, "gz": 0.0}) + "\n")
    cols = ("t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap,rz,ryaw_deg,t_traj,mode,"
            "engaged,traj_on,solver,solver_status,solve_ms,n_tags,pnp_rms_px,"
            "tag_age_s,imu_age_s,z_src,ambig,dr_px,dr_py,dr_pz,dr_t_s,dr_ok,"
            "roll_deg")
    with open(f"{stem}.csv", "w") as fh:
        fh.write(cols + "\n")
        for k in range(0, n, int(hz / 20)):        # a 20 Hz controller log
            fh.write(f"{t[k]:.4f},0,0,0,nan,nan,{yaw_deg},0,0,nan,nan,nan,"
                     f"pid,1,0,stub,0,nan,4,0.5,0.05,0.05,pressure,0,"
                     f"nan,nan,nan,nan,0,0\n")
    (tmp / "mpc_000000.meta.json").write_text(_json.dumps(
        {"schema_version": 3, "imu_dr": {"enabled": True, "attitude": "gyro"},
         "hardware": {"datum_tag_frame": None}}))
    return Path(f"{stem}.csv")


def test_offline_reestimation_reproduces_the_closed_form():
    """--from-jsonl is the payoff of logging every sample, so it has to be
    right: a run re-estimated offline must land where the arithmetic says."""
    import json as _json
    import tempfile

    from rov_gui.tools.plot_imu_dr import find_run, reestimate
    from rov_gui.tools.plot_runs import read_csv

    for yaw_deg in (0.0, 40.0):
        tmp = Path(tempfile.mkdtemp(prefix="reest_"))
        b = 0.03
        _fake_run(tmp, b=(b, 0.0, 0.0), yaw_deg=yaw_deg)
        csv, meta_p, jl = find_run(tmp)
        d = read_csv(csv)
        v = reestimate(jl, d, _json.loads(meta_p.read_text()), "gyro", 0.0,
                       None)
        i = int(np.nanargmax(np.where(v["dr_ok"] > 0.5, v["dr_t_s"], np.nan)))
        el = float(v["dr_t_s"][i])
        # Body +x at this heading, expressed in world FLU.
        want = 0.5 * b * el ** 2
        psi = math.radians(yaw_deg)
        got = np.array([v["dr_px"][i], v["dr_py"][i]])
        exp = want * np.array([math.cos(psi), math.sin(psi)])
        check(f"reestimated drift yaw={yaw_deg}",
              float(np.linalg.norm(got - exp)) < 0.02 * max(want, 1e-9) + 1e-3,
              f"got {got.round(4)} want {exp.round(4)} after {el:.1f} s")


def test_segments_time_to_exceed_and_growth_fit():
    from rov_gui.tools.plot_imu_dr import (fit_growth, segments,
                                           time_to_exceed)

    t = np.arange(0.0, 30.0, 0.05)
    # two 10 s windows separated by a gap the estimator was not ok for
    dr_t = np.where(t < 10.0, t, np.where(t < 12.0, np.nan, t - 12.0))
    ok = np.isfinite(dr_t)
    segs = segments(t, np.nan_to_num(dr_t), ok)
    check("two windows", len(segs) == 2, str(segs))

    b = 0.02
    err = 0.5 * b * np.nan_to_num(dr_t) ** 2
    tte = time_to_exceed(segs, np.nan_to_num(dr_t), err)
    # 0.5*0.02*t^2 = 0.10 m at t = sqrt(0.2/0.02) = 3.16 s
    check("10 cm crossing", abs(tte["10cm"]["p50_s"] - 3.16) < 0.1,
          str(tte["10cm"]))
    # 1.00 m needs t = 10.0 s exactly: the 10 s window just misses it, the
    # 18 s one reaches it. A threshold some windows never cross must be
    # REPORTED with its window count, not dropped for having no number.
    check("100 cm reached by one window of two",
          tte["100cm"]["windows_reaching_it"] == 1
          and tte["100cm"]["windows_not"] == 1
          and abs(tte["100cm"]["p50_s"] - 10.0) < 0.1, str(tte["100cm"]))
    check("500 cm never", tte["50cm"]["windows_reaching_it"] == 2
          and tte["100cm"]["p50_s"] is not None, str(tte["50cm"]))

    g = fit_growth(np.nan_to_num(dr_t), err)
    check("b recovered", abs(g["b_eff_m_s2"] - b) < 1e-9, str(g))
    check("t^2 wins", g["b_eff_r2"] > g["beta_eff_r2"], str(g))


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
