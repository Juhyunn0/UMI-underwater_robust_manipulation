#!/usr/bin/env python3
"""test_mpcc.py — the contouring controller and the C1 mission curve.

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_mpcc.py

The geometry tests are cheap and always run. The closed-loop tests build a real
acados solver (~3 s) and fly the vehicle against the same Fossen model the
controller predicts with; they SKIP rather than fail when acados is missing, so
this file is still useful on a machine without the toolchain.

What is pinned here is what the 2026-08-16 rewrite exists to guarantee:
  * a rectangle corner has a CONTINUOUS heading (the old sharp vertex is what
    floored tracking error near 2 cm no matter the tuning),
  * the reference SPEED is feasible — curvature-limited, and braking starts
    before the corner rather than at it,
  * the NMPC horizon reaches PAST the corner it is approaching, which is the
    thing the operator asked for and the thing the old corner gate forbade,
  * v_ref is never a fixed point of the optimizer (the deadlock that silently
    reports a healthy solver status and never starts the mission).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

# Set BEFORE anything can import rov_gui.qt: Qt reads the platform plugin name
# when the QApplication is constructed, and a later setdefault is too late.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bluerov2_mujoco_marinegym"))

from rov_gui.control.path_geometry import (ArcPath, PathCursor,  # noqa: E402
                                           path_from_scenario)

SQUARE = {"kind": "square", "size": 2.0, "size_y": 1.0, "speed": 0.12,
          "laps": 2, "origin_ned": [0.0, 0.0], "depth_ned": 0.5,
          "rot_deg": 0.0, "heading_follow": False}
LINE = {"kind": "line", "length": 2.0, "speed": 0.05, "laps": 2,
        "origin_ned": [0.0, 0.0], "depth_ned": 0.5, "dir_deg": 90.0,
        "ramp_s": 1.0}
CIRCLE = {"kind": "circle", "radius": 0.5, "speed": 0.05, "laps": 2,
          "origin_ned": [0.0, 0.0], "depth_ned": 0.5, "rot_deg": 0.0,
          "heading_follow": False}


# ------------------------------------------------------------------ geometry
def test_fillet_makes_the_corner_heading_continuous():
    p = path_from_scenario(SQUARE, fillet_m=0.15)
    s = np.linspace(0.0, p.lap_length, 4001)
    _x, _y, psi, k = p.sample(s)
    step = np.abs(np.diff(np.unwrap(psi)))
    # A sharp vertex is a pi/2 jump between adjacent samples; an arc of radius
    # R turns ds/R per sample. Anything above a few times that is a corner
    # that was never rounded.
    ds = s[1] - s[0]
    assert step.max() < 3.0 * ds / 0.15, f"heading jumps {step.max():.3f} rad"
    assert abs(np.unwrap(psi)[-1] - psi[0] - 2 * math.pi) < 1e-3, \
        "one lap of a rectangle must turn exactly 360 degrees"
    assert abs(np.abs(k[np.isfinite(k)]).max() - 1 / 0.15) < 1e-6


def test_fillet_shortens_the_lap_and_cannot_eat_the_side():
    sharp = 2 * (2.0 + 1.0)
    assert path_from_scenario(SQUARE, fillet_m=0.15).lap_length < sharp
    # a fillet larger than the side is clamped, not accepted
    big = path_from_scenario(SQUARE, fillet_m=5.0)
    assert big.lap_length > 0.5 * sharp


def test_speed_profile_is_curvature_limited_and_brakes_early():
    p = path_from_scenario(SQUARE, fillet_m=0.15)
    s, v = p.speed_profile(v_cmd=0.12, a_lat=0.05, a_long=0.05)
    assert abs(v.max() - 0.12) < 1e-9, "cruise must reach the commanded speed"
    k = p.sample(s)[3]
    corner = np.isfinite(k) & (np.abs(k) > 1e-6)
    # The FIRST corner only: the last one also has the end-of-mission stop
    # ramping through it, so its minimum is 0 for a reason that is not
    # curvature.
    first = int(np.argmax(corner))
    last = first + int(np.argmin(corner[first:]))
    assert abs(v[first:last].min() - math.sqrt(0.05 * 0.15)) < 2e-3, \
        f"corner speed {v[first:last].min():.4f} != sqrt(a_lat*R)"
    # Braking starts BEFORE the corner, and by the distance the accel limit
    # implies: (v_cruise^2 - v_corner^2) / (2 a_long) = 0.069 m here, so the
    # profile must already be off cruise a few samples ahead of the arc.
    brake_m = (0.12 ** 2 - 0.05 * 0.15) / (2.0 * 0.05)
    n_brake = int(brake_m / (s[1] - s[0]))
    assert n_brake >= 2
    assert v[first - 1] < 0.12 - 1e-3, "reference brakes at the corner, not before"
    assert v[first - n_brake - 2] > 0.12 - 1e-3, "brakes far too early"


def test_v_ref_is_never_a_fixed_point_of_the_optimizer():
    """v_ref == 0 anywhere the path still has distance to cover is a silent
    deadlock: theta_dot's target is zero, so theta never moves, so v_ref never
    rises. Measured 2026-08-16 — the mission simply never started."""
    for scen, kw in ((SQUARE, {"fillet_m": 0.15}), (LINE, {})):
        p = path_from_scenario(scen, **kw)
        s, v = p.speed_profile(scen["speed"], 0.05, 0.05, v_creep=0.02)
        assert v[0] > 0.0, "the mission cannot leave its start"
        stuck = (v <= 0.0) & (s < p.total_length - 1e-9)
        assert not stuck.any(), f"{stuck.sum()} deadlock samples"


def test_line_reverses_in_place_and_keeps_the_missions_geometry():
    p = path_from_scenario(LINE, turn_radius_m=0.0)
    s = np.linspace(0.0, p.lap_length, 2001)
    x, y, psi, _k = p.sample(s)
    assert np.abs(x).max() < 1e-9, "an out-and-back line must not bulge sideways"
    assert abs(y.max() - 2.0) < 1e-6 and abs(y.min()) < 1e-6
    assert abs(np.unwrap(psi)[-1] - psi[0] - 2 * math.pi) < 1e-6


def test_sample_extends_past_both_ends_instead_of_flattening():
    """The MPCC window straddles the start. Clamping s would make the sampled
    path FLAT there, dp/dtheta = 0 at exactly theta0, and the optimizer sees no
    reason to advance at all — the mission never starts."""
    p = path_from_scenario(SQUARE, fillet_m=0.15)
    s = np.array([-0.30, -0.15, 0.0])
    x, y, _psi, _k = p.sample(s)
    d = np.hypot(np.diff(x), np.diff(y))
    assert np.all(d > 0.10), f"window behind the start is flat: {d}"


def test_cursor_never_runs_backwards_or_past_the_lead():
    p = path_from_scenario(SQUARE, fillet_m=0.15)
    s, v = p.speed_profile(0.12, 0.05, 0.05)
    cur = PathCursor(p, lead_m=0.10, s_grid=s, v_grid=v)
    x0, y0, _psi, _k = p.sample(np.array([0.0]))
    for _ in range(20):                      # a hull that never moves
        target, _psi_t, _al, _cr, _v = cur.step([x0[0], y0[0], 0.5], 0.05)
    assert cur.theta <= 1e-6
    assert math.hypot(target[0] - x0[0], target[1] - y0[0]) <= 0.10 + 1e-9
    # and one that rides the path advances monotonically
    prev = cur.theta
    for t in np.arange(0.0, 1.0, 0.02):
        px, py, _p, _k2 = p.sample(np.array([t]))
        cur.step([px[0], py[0], 0.5], 0.05)
        assert cur.theta >= prev - 1e-12
        prev = cur.theta
    assert cur.theta > 0.9


def test_shipped_settings_match_the_best_recorded_mission():
    """The shipped tuning must stay equal to the best mission actually flown.

    This exists because I drifted away from it three times in one day on the
    strength of simulator sweeps, and each step made the vehicle worse. The
    reference is an ARTIFACT, not an opinion: sessions/.../0817_110145 is the
    only run on file that completed 5/5 laps with cross-track p95 6.6 cm and
    zero actuator saturation. Anything that changes these numbers should be
    changing them against a NEW run, not against a simulation — the simulator's
    plant has no ESC deadband and 8-12x too little drag, which is exactly the
    regime where a push-vs-geometry knob cannot be ranked.

    Skips (rather than fails) when the session artifact is absent, so a fresh
    clone still runs the suite.
    """
    import json

    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control import mpcc_acados as M

    ref = (ROOT / "sessions/low_level_controller_data/20260817/0817_110145"
           / "mpc_110145.meta.json")
    if not ref.exists():
        print("  SKIP (best-run artifact not on disk)")
        return
    good = json.loads(ref.read_text())
    c = good["controller"]
    w = c["weights"]
    cfg = MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))

    assert cfg.axis_gain["surge_n"] == good["hardware"]["axis_gain"]["surge_n"]
    assert cfg.axis_gain["sway_n"] == good["hardware"]["axis_gain"]["sway_n"]
    assert cfg.axis_cap == good["hardware"]["axis_cap"]
    assert cfg.path_lat_accel_m_s2 == c["lat_accel_m_s2"]
    assert cfg.path_long_accel_m_s2 == c["long_accel_m_s2"]
    # The fillet is a MISSION setting, not a controller constant: 0.15 is the
    # baseline (and what MPCC needs, since its frame needs a tangent), 0.0 is
    # the waypoint experiment. Anything else is drift.
    assert cfg.path_fillet_m in (c["fillet_m"], 0.0), (
        f"fillet {cfg.path_fillet_m} is neither the baseline "
        f"{c['fillet_m']} nor the waypoint 0.0")
    assert (M.W_CONTOUR, M.W_LAG, M.W_PROGRESS) == (
        w["contour"], w["lag"], w["progress"])


# --------------------------------------------------------------- closed loop
def _acados_or_skip():
    try:
        from rov_gui.control.mpcc_acados import AcadosMPCC
        return AcadosMPCC
    except Exception as e:                                    # noqa: BLE001
        print(f"  SKIP (no acados: {type(e).__name__}: {e})")
        return None


def _plant():
    import casadi as ca
    from dobmpc.mpc import _f_casadi

    xs, us, ws = ca.SX.sym("x", 12), ca.SX.sym("u", 6), ca.SX.sym("w", 6)
    F = ca.Function("F", [xs, us, ws], [_f_casadi(xs, us, ws)])
    z = np.zeros(6)

    def rk4(x, u, dt=0.05, n=5):
        h = dt / n
        for _ in range(n):
            k1 = np.array(F(x, u, z)).ravel()
            k2 = np.array(F(x + h / 2 * k1, u, z)).ravel()
            k3 = np.array(F(x + h / 2 * k2, u, z)).ravel()
            k4 = np.array(F(x + h * k3, u, z)).ravel()
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    return rk4


def _fly(mpcc, path, s_g, v_g, ticks=3000):
    rk4 = _plant()
    win = mpcc.window_for(float(v_g.max()))
    x = np.zeros(13)
    px, py, _p, _k = path.sample(np.array([0.0]))
    x[0], x[1], x[2] = px[0], py[0], 0.5
    mpcc.reset()
    rec, look = [], []
    for _ in range(ticks):
        par = mpcc.pack_params(np.zeros(6), x[12], path, 0.0, False,
                               s_g, v_g, win, float(x[5]))
        # the speed target is a CONSTANT ceiling, not a schedule
        u = mpcc.solve(x, par, 0.5, float(v_g.max()))
        look.append(float(mpcc.horizon_theta()[-1]) - float(x[12]))
        x[0:12] = rk4(x[0:12], u[:6])
        x[12] = min(path.total_length, x[12] + float(u[6]) * 0.05)
        rec.append((x[0], x[1], x[12]))
        if x[12] >= path.total_length - 1e-6:
            break
    return np.array(rec), np.array(look)


def test_mpcc_flies_the_whole_square_and_stays_on_the_curve():
    A = _acados_or_skip()
    if A is None:
        return
    m = A()
    path = path_from_scenario(SQUARE, fillet_m=0.15)
    s_g, v_g = path.speed_profile(0.12, 0.05, 0.05)
    rec, _look = _fly(m, path, s_g, v_g)
    assert rec[-1, 2] >= path.total_length - 1e-3, \
        f"only {rec[-1, 2]:.2f} of {path.total_length:.2f} m flown"
    cx, cy, _p, _k = path.sample(rec[:, 2])
    err = np.hypot(rec[:, 0] - cx, rec[:, 1] - cy)[40:]
    assert np.percentile(err, 95) < 0.02, \
        f"p95 contour error {np.percentile(err, 95) * 1000:.1f} mm"
    assert m.n_fail == 0


def test_the_horizon_reaches_past_the_corner_it_is_approaching():
    """The whole point of the 2026-08-16 rewrite. The corner gate it replaced
    made this impossible BY DESIGN: it hid the next leg until the hull had
    stopped inside 5 cm of the vertex."""
    A = _acados_or_skip()
    if A is None:
        return
    m = A()
    path = path_from_scenario(SQUARE, fillet_m=0.15)
    s_g, v_g = path.speed_profile(0.12, 0.05, 0.05)
    rec, look = _fly(m, path, s_g, v_g)
    # the first corner's arc, in arclength
    k = np.array([path.sample(np.array([t]))[3][0] for t in rec[:, 2]])
    arc = np.isfinite(k) & (np.abs(k) > 1e-6)
    arc_start = rec[int(np.argmax(arc)), 2]
    arc_end = arc_start + 0.15 * math.pi / 2.0
    appr = (rec[:, 2] > arc_start - 0.5) & (rec[:, 2] < arc_start)
    reach = np.median(rec[appr, 2] + look[appr])
    assert reach > arc_end, (
        f"horizon reaches {reach:.2f} m; the corner ends at {arc_end:.2f} m")


def test_mpcc_solves_inside_the_control_period():
    A = _acados_or_skip()
    if A is None:
        return
    m = A()
    path = path_from_scenario(SQUARE, fillet_m=0.15)
    s_g, v_g = path.speed_profile(0.12, 0.05, 0.05)
    win = m.window_for(0.12)
    x = np.zeros(13)
    x[2] = 0.5
    ms = []
    for i in range(60):
        par = m.pack_params(np.zeros(6), 0.01 * i, path, 0.0, False,
                            s_g, v_g, win, 0.0)
        m.solve(x, par, 0.5, float(v_g.max()))
        ms.append(m.solve_ms())
    p99 = float(np.percentile(ms, 99))
    assert p99 < 25.0, f"solve p99 {p99:.1f} ms exceeds the engage probe gate"


def test_the_mpcc_curve_and_the_legacy_placement_agree_off_the_corners():
    """The fillet is the ONLY intended difference. If the two constructions
    disagreed anywhere else, a PID run (legacy sampler) and an MPCC run would
    be flying different missions -- the exact failure this repo has already
    paid for three times (reference.py's placement header)."""
    from rov_gui.control.reference import place_square_ned

    fn, scen = place_square_ned(
        {"size": 2.0, "size_y": 1.0, "speed": 0.12, "laps": 1,
         "heading_follow": False, "rot_deg": 0.0},
        (0.0, 0.0), 0.0, 0.5, dt=0.05)
    arc = path_from_scenario(scen, fillet_m=0.15)
    # sample the legacy time-parameterised path, keep only points well away
    # from the four vertices, and check each lies on the arc path
    ts = np.linspace(0.0, scen["T_run_s"], 600)
    p_flu, _yaw, _v, _r = fn(ts)
    legacy = np.stack([p_flu[0], -p_flu[1]])          # FLU -> NED mirror
    s = np.linspace(0.0, arc.lap_length, 3000)
    ax, ay, _p, _k = arc.sample(s)
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    worst = 0.0
    for x, y in legacy.T:
        if np.hypot(verts[:, 0] - x, verts[:, 1] - y).min() < 0.30:
            continue                                   # corner pocket: differs
        worst = max(worst, float(np.hypot(ax - x, ay - y).min()))
    assert worst < 2e-3, f"the two constructions differ by {worst * 1000:.1f} mm"


def test_the_commanded_speed_actually_reaches_the_reference():
    """The operator's speed box must be IN the loop.

    Measured on hardware 2026-08-17 before this was fixed: 0.2 m/s commanded,
    1.09 m of path covered in 42 s = 0.026 m/s
    (sessions/low_level_controller_data/20260817/0817_103431/mpc_103624.csv).
    The cursor advanced its setpoint by PROJECTION only, so the reference was a
    carrot pinned lead_m ahead of the hull and the speed number never entered
    the loop at all."""
    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 5, "origin_ned": [0.0, 0.0], "depth_ned": 0.0,
            "rot_deg": 0.0}
    p = path_from_scenario(scen, fillet_m=0.15)
    s_g, v_g = p.speed_profile(0.20, 0.05, 0.05)
    cur = PathCursor(p, lead_m=0.10, s_grid=s_g, v_grid=v_g)
    # a hull that keeps up (stays one lead behind the setpoint)
    hull = np.array(p.sample(np.array([0.0]))[:2]).ravel()
    T, dt = 40.0, 0.05
    for _ in range(int(T / dt)):
        _tgt, _psi, _al, _cr, _v = cur.step([hull[0], hull[1], 0.0], dt)
        back = max(0.0, cur.target_theta - 0.05)
        hx, hy, _ps, _k = p.sample(np.array([back]))
        hull = np.array([hx[0], hy[0]])
    mean = cur.target_theta / T
    # the feasible profile's own mean is the ceiling (corners are speed-limited
    # by sqrt(a_lat*R)); anything near it means the command is in the loop
    ideal = p.total_length / float(np.sum(
        np.diff(s_g) / np.maximum(0.5 * (v_g[:-1] + v_g[1:]), 1e-3)))
    assert mean > 0.6 * ideal, (
        f"reference covered {mean:.4f} m/s; the profile allows {ideal:.4f}")
    # 4x the measured pre-fix figure. The ceiling here IS `ideal` (0.125 m/s
    # for this 1 m square: its corners are limited to sqrt(a_lat*R) = 0.087),
    # so the assertion cannot be tightened past it without changing the path.
    assert mean > 4 * 0.026, "still the pre-2026-08-17 carrot behaviour"


def test_a_stalled_hull_still_leashes_the_reference():
    """The speed-driven advance must not undo path following: a vehicle that
    stops must not have its reference walk away (that is trajectory tracking,
    and it is what the governed clock was introduced to stop)."""
    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 5, "origin_ned": [0.0, 0.0], "depth_ned": 0.0,
            "rot_deg": 0.0}
    p = path_from_scenario(scen, fillet_m=0.15)
    s_g, v_g = p.speed_profile(0.20, 0.05, 0.05)
    cur = PathCursor(p, lead_m=0.10, s_grid=s_g, v_grid=v_g)
    start = np.array(p.sample(np.array([0.0]))[:2]).ravel()
    for _ in range(600):                      # 30 s of a hull that never moves
        cur.step([start[0], start[1], 0.0], 0.05)
    assert cur.target_theta <= 0.10 + 1e-9, \
        f"reference ran {cur.target_theta:.3f} m from a stationary hull"


def test_a_brisk_speed_profile_does_not_freeze_progress():
    """Progress must not stall, whatever the geometry asks for.

    History: while the speed target was a FUNCTION of theta, v_ref'(theta)
    entered theta's Gauss-Newton Hessian and the RTI's single iteration could
    not cross it — a_long 0.05 gave 0.118 m/s and a_long 0.15 gave EXACTLY
    ZERO on the same path (2026-08-17). The target is a constant ceiling now,
    so no profile shape can rebuild that barrier."""
    A = _acados_or_skip()
    if A is None:
        return
    m = A()
    scen = {"kind": "square", "size": 1.0, "size_y": 1.0, "speed": 0.20,
            "laps": 3, "origin_ned": [0.0, 0.0], "depth_ned": 0.5,
            "rot_deg": 0.0}
    path = path_from_scenario(scen, fillet_m=0.06)
    s_g, v_g = path.speed_profile(0.20, 0.30, 0.15)     # the brisk case
    rec, _look = _fly(m, path, s_g, v_g, ticks=1200)
    mean = rec[-1, 2] / (len(rec) * 0.05)
    assert mean > 0.12, f"progress {mean:.4f} m/s — the theta barrier is back"


def test_the_small_fillet_stays_close_to_the_rectangle():
    """A 6 cm fillet must read as a rectangle: the operator compares MPCC with
    PID by eye, and a visibly rounded square is a different mission."""
    p = path_from_scenario({"kind": "square", "size": 1.0, "size_y": 1.0,
                            "speed": 0.2, "laps": 1, "origin_ned": [0.0, 0.0],
                            "depth_ned": 0.0, "rot_deg": 0.0}, fillet_m=0.06)
    s = np.linspace(0.0, p.lap_length, 4000)
    x, y, _psi, _k = p.sample(s)
    # every point is within the fillet's corner-cut of the sharp rectangle
    inside = np.maximum.reduce([-x, x - 1.0, -y, y - 1.0])
    assert inside.max() < 1e-9, "the curve leaves the rectangle"
    d = np.hypot(x - 1.0, y - 0.0).min()        # closest approach to a vertex
    assert d < 0.03, f"corner cut {d * 1000:.0f} mm — too round to compare"
    # a 90-degree fillet of radius R replaces 2R of straight with R*pi/2 of
    # arc, so the lap is exactly 4*(2R - R*pi/2) shorter than the rectangle
    lost = 4.0 * (2 * 0.06 - 0.06 * math.pi / 2.0)
    assert abs(p.lap_length - (4.0 - lost)) < 1e-6, \
        f"lap {p.lap_length:.4f}, expected {4.0 - lost:.4f}"
    assert lost < 0.11, "the rectangle lost more than 3 % of its perimeter"


def test_the_fillet_must_exceed_the_cross_track_error():
    """A fillet smaller than the tracking error is a degenerate projection.

    The arc's centre of curvature is R from the path; once |e_cross| > R on the
    concave side the nearest point sweeps ACROSS the arc, so the reference
    darts forward and the vehicle chases it. Measured on hardware 2026-08-17
    with R = 0.06 (.../0817_135600/mpc_140502.csv): |e_cross| exceeded R on
    17 % of arc ticks, the reference outran the hull by >3x on 12 % of them,
    and arc error was 3.44 cm against 1.67 on the straights.

    This pins the SHIPPED geometry against the error the vehicle actually
    achieves, so shrinking one without checking the other cannot pass quietly.
    """
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))
    # closed-loop p95 cross-track against the measured actuator, ~1.4 cm; the
    # hardware runs come out about 2.3x worse than this simulation, so budget
    # for that before comparing.
    expected_p95_m = 0.014 * 2.3
    assert cfg.path_fillet_m > 2.0 * expected_p95_m, (
        f"fillet {cfg.path_fillet_m:.3f} m is not comfortably above the "
        f"{expected_p95_m * 100:.1f} cm cross-track error to expect")


# -------------------------------------------------------------------- circle
def test_circle_is_one_arc_that_starts_at_its_tag_and_never_needs_a_fillet():
    """The circle is the one mission with nothing to repair.

    Every other shape reaches path_geometry with an unrealizable feature in
    it — a 90-degree vertex, a 180-degree reversal — and the fillet and the
    pivot exist to make those flyable. A circle arrives already C1, so it is
    a single ``_Arc`` of 2*pi and the fillet knob must not change it at all.
    """
    p = path_from_scenario(CIRCLE, fillet_m=0.15)
    assert len(p.prims) == 1, [type(q).__name__ for q in p.prims]
    assert abs(p.lap_length - 2 * math.pi * 0.5) < 1e-12
    assert abs(p.total_length - 2 * p.lap_length) < 1e-12
    # a fillet cannot shorten what has no corner
    assert abs(path_from_scenario(CIRCLE, fillet_m=0.0).lap_length
               - p.lap_length) < 1e-12

    s = np.linspace(0.0, p.lap_length, 721)
    x, y, psi, k = p.sample(s)
    # THE requirement: the entered tag is the START and the circle's min x,
    # with the centre one radius up-plot (+x) from it — not the centre.
    assert abs(x[0]) < 1e-12 and abs(y[0]) < 1e-12
    assert abs(x.min()) < 1e-12 and abs(x.max() - 1.0) < 1e-12
    assert np.abs(np.hypot(x - 0.5, y) - 0.5).max() < 1e-12
    # constant curvature, and the heading is the true tangent everywhere
    assert np.ptp(k) < 1e-12 and abs(k[0] - 1.0 / 0.5) < 1e-12
    tangent = np.arctan2(np.gradient(y, s), np.gradient(x, s))
    err = np.abs(np.arctan2(np.sin(tangent - psi), np.cos(tangent - psi)))
    assert err[2:-2].max() < 1e-6, err[2:-2].max()
    # one lap turns the heading exactly once, and sample() hands it over
    # UNWRAPPED so no consumer sees a 2*pi drop at the lap boundary
    assert abs((psi[-1] - psi[0]) - 2 * math.pi) < 1e-9
    x2, y2, psi2, _k2 = p.sample(np.linspace(0.0, p.total_length, 1441))
    assert abs((psi2[-1] - psi2[0]) - 4 * math.pi) < 1e-9
    assert np.all(np.diff(psi2) > 0.0), "the heading wrapped at a lap boundary"


def test_circle_rot_deg_swings_the_centre_and_leaves_the_tag_alone():
    """rot_deg is a MAP bearing from the tag TO the centre. The tag is the one
    thing the operator physically pointed at, so it must not move when the
    circle is turned about it."""
    for deg in (0.0, 37.0, 90.0, 180.0, -120.0):
        scen = dict(CIRCLE, rot_deg=deg, origin_ned=[1.0, -2.0])
        p = path_from_scenario(scen)
        x, y, _psi, _k = p.sample(np.linspace(0.0, p.lap_length, 361))
        assert abs(x[0] - 1.0) < 1e-9 and abs(y[0] + 2.0) < 1e-9, deg
        cx = 1.0 + 0.5 * math.cos(math.radians(deg))
        cy = -2.0 + 0.5 * math.sin(math.radians(deg))
        assert np.abs(np.hypot(x - cx, y - cy) - 0.5).max() < 1e-9, deg


def test_circle_speed_is_curvature_limited_over_the_whole_lap():
    """A polygon's straights run at the operator's speed and only its corners
    are slower. A circle is curvature-limited EVERYWHERE, so a small radius
    overrides the speed box for the entire run — which is why the worker says
    so in the log rather than quietly flying something else."""
    p = path_from_scenario(CIRCLE)            # R = 0.5
    a_lat = 0.05
    s, v = p.speed_profile(0.20, a_lat, 0.05, v_creep=0.02)
    cap = math.sqrt(a_lat * 0.5)              # 0.158 m/s
    assert abs(v.max() - cap) < 1e-6, v.max()
    # ...and it is the cap everywhere except the ramps off and onto rest
    assert np.median(v) > 0.99 * cap
    # no pivot and no corner, so nothing is pinned to zero mid-run
    assert v[1:-1].min() > 0.0
    # a bigger radius lets the operator's number through untouched
    wide = path_from_scenario(dict(CIRCLE, radius=2.0))
    _s2, v2 = wide.speed_profile(0.20, a_lat, 0.05, v_creep=0.02)
    assert abs(v2.max() - 0.20) < 1e-6, v2.max()


def test_circle_cursor_runs_every_lap_and_completes():
    """The PID/MPC spatial follower has to traverse a closed constant-curvature
    loop the same way it traverses a rectangle — including counting laps, which
    on a circle come from arclength alone."""
    p = path_from_scenario(CIRCLE)
    s, v = p.speed_profile(0.05, 0.05, 0.05, v_creep=0.02)
    cur = PathCursor(p, 0.10, s, v)
    pos = np.array([0.0, 0.0])
    laps_seen, t, dt = set(), 0.0, 0.05
    for _ in range(40000):
        target, _psi, _along, _cross, _v = cur.step(pos, dt)
        pos = pos + (target - pos) * 0.35     # a crude but honest follower
        laps_seen.add(cur.lap())
        t += dt
        if cur.complete:
            break
    assert cur.complete, cur.theta
    assert laps_seen == {0, 1, 2}, sorted(laps_seen)
    # the vehicle stayed ON the circle, not inside it
    assert abs(math.hypot(pos[0] - 0.5, pos[1]) - 0.5) < 0.05


def test_circle_lap_boundary_never_commands_a_full_spin():
    """A circle is the only mission whose heading turns a full 2*pi per lap,
    so it is the only one that can hand a controller a 2*pi STEP.

    ``PathCursor.plan`` samples the path one stage at a time, and each sample
    comes back on its own lap's branch — so a plan that straddles the boundary
    genuinely contains a 6.28 rad jump in ``yaw_ned``. Two things downstream
    absorb it, and both are pinned here because the failure they prevent is a
    360-degree spin commanded mid-lap: ``plan`` unwraps before differencing
    (so the yaw RATE it reports is the circle's own), and HwDobMpc walks the
    reference onto the branch nearest the vehicle (mpc_bridge._xref_ned_plan).
    """
    scen = dict(CIRCLE, heading_follow=True)
    p = path_from_scenario(scen)
    s, v = p.speed_profile(0.05, 0.05, 0.05, v_creep=0.02)
    cur = PathCursor(p, 0.10, s, v)
    cur.theta = cur._theta_cmd = p.lap_length - 0.05     # just short of a lap
    plan = cur.plan(61, 0.05, 0.5, 0.0, heading_follow=True)

    # the raw jump is REAL — if this stops being true the test below is vacuous
    assert np.abs(np.diff(plan.yaw_ned)).max() > 6.0

    # ...but the yaw RATE handed to the solver is the circle's, not a spin
    assert np.abs(plan.r_ned - 0.05 / 0.5).max() < 1e-9, plan.r_ned

    # ...and so is the yaw reference, once put on the vehicle's own branch the
    # way mpc_bridge._xref_ned_plan does it
    def wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    psi, walked = 0.0, []
    for y in plan.yaw_ned:
        psi += wrap(float(y) - psi)
        walked.append(psi)
    step = np.abs(np.diff(np.asarray(walked))).max()
    assert step < 2.0 * (0.05 / 0.5) * 0.05, step


def test_cursor_cannot_jump_most_of_a_short_lap_in_one_tick():
    """PathCursor's projection is monotone, so an argmin that picks the wrong
    place is PERMANENT progress. A circle is where that becomes reachable: the
    projection is radial, so a hull displaced toward the CENTRE is nearly
    equidistant from the whole search window and the argmin is decided by the
    direction of the error rather than by position along the curve.

    With the old fixed 0.60 m window a 0.15 m circle (lap 0.94 m) jumped 64 %
    of a lap in a single tick. The window is now capped at a quarter lap, so
    the worst case is bounded by construction — and the setpoint, leashed to
    the projection, cannot be dragged with it.
    """
    for R in (0.15, 0.30, 0.50):
        scen = dict(CIRCLE, radius=R)
        p = path_from_scenario(scen)
        s, v = p.speed_profile(0.05, 0.05, 0.05, v_creep=0.02)
        worst = 0.0
        for frac in np.linspace(0.05, 1.0, 21):          # how far in
            for ang in np.linspace(-math.pi, math.pi, 37):
                cur = PathCursor(p, 0.10, s, v)
                hull = (np.array([R, 0.0])
                        + frac * R * np.array([math.cos(ang), math.sin(ang)]))
                cur.step(hull, 0.05)
                worst = max(worst, cur.theta)
        assert worst <= 0.25 * p.lap_length + 1e-9, (R, worst, p.lap_length)
        # ...and the window is still wide enough to track the vehicle: the
        # hull covers 25 mm per tick at the panel's 0.50 m/s ceiling
        assert PathCursor(p, 0.10, s, v).search_m >= 0.10


def test_the_window_spline_reproduces_a_circle_at_every_offered_radius():
    """THE ARTIFACT for the resolution claim in mpcc_acados' N_GRID comment.

    That comment justifies 41 samples by the smallest fillet, which a circle
    does not have — on a circle the path IS the curved feature and the window
    can span more than a whole lap at a small radius. Rather than argue about
    it, fit the same B-spline the solver's contouring frame uses and measure
    the error against the true arc, over the panel's whole radius range.
    """
    try:
        from rov_gui.control import mpcc_acados as M
        mp = M.AcadosMPCC(build=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"    (skipped: {type(e).__name__}: {e})")
        return
    worst = 0.0
    for R in (0.15, 0.20, 0.30, 0.50, 1.00):
        for v_cmd in (0.05, 0.12, 0.50):
            p = path_from_scenario(dict(CIRCLE, radius=R, speed=v_cmd))
            # the window HwMpcc._install actually installs: sized on the
            # feasible speed, which on a circle is the curvature cap
            _s, v_grid = p.speed_profile(v_cmd, 0.05, 0.05, v_creep=0.02)
            w = mp.window_for(float(v_grid.max()))
            u = np.linspace(0.0, 1.0, M.N_GRID)
            px, py, _psi, _k = p.sample(-mp.back_m + u * w)
            cx, cy = mp._A_inv @ px, mp._A_inv @ py
            uu = np.linspace(0.0, 1.0, 501)
            fx = np.array([float(mp.spline(t, cx)) for t in uu])
            fy = np.array([float(mp.spline(t, cy)) for t in uu])
            tx, ty, _p2, _k2 = p.sample(-mp.back_m + uu * w)
            worst = max(worst, float(np.hypot(fx - tx, fy - ty).max()))
    # 0.5 mm, against the 1.5 mm the same comment quotes for a 6 cm fillet:
    # the circle is the EASIER curve for this fit, not the harder one.
    assert worst < 5e-4, f"{worst * 1000:.3f} mm"


def test_circle_rejects_a_radius_that_is_not_a_circle():
    for bad in (0.0, -0.5):
        try:
            path_from_scenario(dict(CIRCLE, radius=bad))
        except ValueError:
            continue
        raise AssertionError(f"radius {bad} was accepted")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
