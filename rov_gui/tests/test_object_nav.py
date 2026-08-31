#!/usr/bin/env python3
"""test_object_nav.py — placing the tracked object in the pool, and following it.

Two halves, the same shape ``test_station_bridge.py`` uses. The PURE tests
pin the frame algebra and the anchor's gates with no Qt, no torch, no camera
and no acados. The WORKER tests drive the real :class:`MpcWorker` with the
stub controller from ``test_control.py`` and pin the behaviour that actually
matters at the pool: arming a follow must not move the vehicle, a lost object
must not disengage one, and the leash must hold on every tick.

The first test is the load-bearing one. Everything this feature reports about
where an object is rests on the camera extrinsic cancelling out of the
composition, which it only does when the object pose and the tag fix come
from ONE camera frame — so the test composes with a DELIBERATELY WRONG
extrinsic and requires the answer to be unchanged.

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_object_nav.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import object_nav as ON          # noqa: E402
from rov_gui.control.geometry import NavConfig        # noqa: E402
from rov_gui.control.state_assembler import rot_zyx   # noqa: E402


# =============================================================================
# helpers
# =============================================================================
def _T_cam_obj(R_cam_obj, t_cam_obj) -> tuple:
    """16 row-major floats, the shape PoseTrack carries."""
    M = np.eye(4)
    M[0:3, 0:3] = np.asarray(R_cam_obj, float)
    M[0:3, 3] = np.asarray(t_cam_obj, float)
    return tuple(float(v) for v in M.ravel())


def _Rz(yaw: float) -> np.ndarray:
    return rot_zyx(0.0, 0.0, float(yaw))


def _synth(p_veh, yaw_veh, p_obj, yaw_obj, R_bc, t_bc,
           R_use=None, t_use=None):
    """A synthetic camera frame: the truth in, the two estimators' reports out.

    ``R_bc``/``t_bc`` are the TRUE extrinsic, which places the lens; the
    camera then sees the object exactly (that part is physics and has no
    opinions). ``R_use``/``t_use`` are the extrinsic the LOCALIZER BELIEVES —
    what ``hw_nav.yaml`` says — and TagNav divides by that to turn its PnP
    camera pose into the body pose it reports. Defaults to the true one.

    Splitting the two is the whole point of the first test: whatever the
    localizer believes, ``compose_map_pose`` multiplies the SAME belief back
    in, so it cancels and the object lands in the right place regardless.
    """
    R_use = R_bc if R_use is None else R_use
    t_use = t_bc if t_use is None else t_use
    R_nb_true = _Rz(yaw_veh)
    p_nb_true = np.asarray(p_veh, float)
    # where the lens really is (physics)
    R_mc = R_nb_true @ np.asarray(R_bc, float)
    p_mc = p_nb_true + R_nb_true @ np.asarray(t_bc, float)
    # what the camera really sees (physics)
    R_mo = _Rz(yaw_obj)
    R_co = R_mc.T @ R_mo
    t_co = R_mc.T @ (np.asarray(p_obj, float) - p_mc)
    # what TagNav REPORTS: the PnP camera pose with the BELIEVED extrinsic
    # divided out (tagnav.py:336-350). With a wrong belief this is not the
    # vehicle's true pose — a known consequence, and not the one under test.
    R_nb_rep = R_mc @ np.asarray(R_use, float).T
    p_nb_rep = p_mc - R_nb_rep @ np.asarray(t_use, float)
    return p_nb_rep, R_nb_rep, _T_cam_obj(R_co, t_co)


def _extrinsic(tilt_deg=43.3, t_flu=None):
    cfg = NavConfig()
    cfg.cam_tilt_deg = float(tilt_deg)
    if t_flu is not None:
        cfg.cam_t_flu = tuple(float(v) for v in t_flu)
    return cfg.R_t_frd_cam("main")


# =============================================================================
# 1-12: pure
# =============================================================================
def test_the_extrinsic_cancels_when_the_pose_and_the_fix_share_a_frame():
    """THE PROPERTY THE WHOLE FEATURE RESTS ON.

    ``TagNav._solution`` DIVIDES the camera extrinsic out to report a body
    pose; ``compose_map_pose`` MULTIPLIES it back in. Within one camera frame
    those are exact inverses, so the object's map position does not depend on
    the extrinsic AT ALL — which is what makes the feature usable while the
    C3 re-mount's translation is still unmeasured (KNOWN_ISSUES 2026-08-17).

    Proved by being wrong on purpose: the world is synthesized with the real
    extrinsic and recovered with one whose tilt is off by 23 degrees and whose
    lens has been moved 10 cm, and the answer must not move by a nanometre.
    """
    R_true, t_true = _extrinsic(43.3)
    R_bad, t_bad = _extrinsic(20.0, t_flu=(0.33949, 0.00547, -0.15537))
    assert not np.allclose(R_true, R_bad) and not np.allclose(t_true, t_bad)
    for p_veh, yaw_veh, p_obj, yaw_obj in (
            ((-2.0, 0.3, -0.8), 0.0, (-1.6, 0.35, -0.55), 0.4),
            ((0.7, -1.1, -1.2), 1.9, (0.4, -0.8, -0.9), -2.2),
            ((3.1, 2.2, -0.6), -0.8, (3.4, 1.9, -0.45), 3.0)):
        # the RIGHT extrinsic, believed and used: the object is recovered
        p_ned, R_nb, T = _synth(p_veh, yaw_veh, p_obj, yaw_obj,
                                R_true, t_true)
        p_ok, R_ok = ON.compose_map_pose(T, p_ned, R_nb, R_true, t_true)
        assert np.allclose(p_ok, p_obj, atol=1e-9), (p_ok, p_obj)
        assert abs(ON.wrap_pi(ON.yaw_of(R_ok, 0)[0] - yaw_obj)) < 1e-9
        # ...and now a WRONG belief, applied consistently: the localizer
        # divides by it and this module multiplies by it. The reported body
        # pose is wrong (a known consequence of a bad tilt) —
        p_ned_b, R_nb_b, T_b = _synth(p_veh, yaw_veh, p_obj, yaw_obj,
                                      R_true, t_true,
                                      R_use=R_bad, t_use=t_bad)
        assert not np.allclose(p_ned_b, p_ned, atol=1e-6), (
            "the fixture did not actually perturb the reported body pose")
        # — but the OBJECT still lands in exactly the right place.
        p_bad, R_bad_obj = ON.compose_map_pose(T_b, p_ned_b, R_nb_b,
                                               R_bad, t_bad)
        assert np.allclose(p_bad, p_obj, atol=1e-9), (
            f"the extrinsic did NOT cancel: "
            f"{np.linalg.norm(np.asarray(p_bad) - np.asarray(p_obj))} m")
        assert np.allclose(R_bad_obj, R_ok, atol=1e-9)


def test_pairing_across_frames_costs_the_lever_arm():
    """The negative control, and the number behind ``pair_tol_s``.

    Pair an object pose with a fix from a DIFFERENT frame and the cancellation
    breaks: the residual is the camera lever arm swept through the angle
    between the two frames, |t_frd_cam| * 2*sin(theta/2). This is what
    ``pair_exact`` on the panel and in the CSV is warning about.
    """
    R_bc, t_bc = _extrinsic(43.3)
    arm = float(np.linalg.norm(t_bc))
    assert arm > 0.2, arm            # the measured C3 lever arm, ~0.2855 m
    p_veh, p_obj = (0.0, 0.0, -1.0), (0.5, 0.1, -0.7)
    _p, _R, T = _synth(p_veh, 0.0, p_obj, 0.0, R_bc, t_bc)
    for th in (math.radians(1.0), math.radians(5.0), math.radians(15.0)):
        # the SAME object pose, paired with a fix from a frame yawed by th
        p_ned2, R_nb2, _T2 = _synth(p_veh, th, p_obj, 0.0, R_bc, t_bc)
        p_bad, _ = ON.compose_map_pose(T, p_ned2, R_nb2, R_bc, t_bc)
        err = float(np.linalg.norm(np.asarray(p_bad) - np.asarray(p_obj)))
        # the vehicle did not move, only its heading, so the whole residual is
        # the lever arm plus the object's own offset swung through th
        assert err > 0.9 * arm * 2.0 * math.sin(th / 2.0), (th, err)
    # ...and 5 deg is centimetres, which is the scale that matters here
    p_ned5, R_nb5, _ = _synth(p_veh, math.radians(5.0), p_obj, 0.0,
                              R_bc, t_bc)
    p5, _ = ON.compose_map_pose(T, p_ned5, R_nb5, R_bc, t_bc)
    assert 0.01 < float(np.linalg.norm(np.asarray(p5)
                                       - np.asarray(p_obj))) < 0.20


def test_frame_round_trip():
    """Capture an offset, and the goal it implies is where the vehicle IS.
    Then rotate the object 90 degrees: the vehicle must ORBIT to the same
    face, and its target heading must swing by the same 90 degrees."""
    p_veh = np.array([1.0, 0.0, -0.8])
    yaw_veh = 0.3
    p_obj = np.array([1.5, 0.0, -0.6])
    yaw_obj = 0.0
    off, dyaw = ON.offset_in_object_frame(p_veh, yaw_veh, p_obj, yaw_obj)
    goal, gyaw, v = ON.follow_goal(p_obj, yaw_obj, off, dyaw,
                                   np.zeros(3), 0.0)
    assert np.allclose(goal, p_veh, atol=1e-12)
    assert abs(ON.wrap_pi(gyaw - yaw_veh)) < 1e-12
    assert np.allclose(v, np.zeros(3))
    # now the object turns 90 deg on the spot
    goal2, gyaw2, _ = ON.follow_goal(p_obj, yaw_obj + math.pi / 2, off, dyaw,
                                     np.zeros(3), 0.0)
    assert abs(ON.wrap_pi(gyaw2 - (yaw_veh + math.pi / 2))) < 1e-12
    # the offset was (-0.5, 0, -0.2) in the object frame; rotated 90 deg it
    # points along the map's +y, so the vehicle orbits to the object's side
    assert np.allclose(goal2, p_obj + np.array([0.0, -0.5, -0.2]), atol=1e-12)
    # ...and the radius is preserved, which is what "orbit" means
    assert abs(np.linalg.norm(goal2 - p_obj)
               - np.linalg.norm(goal - p_obj)) < 1e-12
    # a ROTATING object gives the goal a velocity even when it is not moving
    _g, _y, v_orb = ON.follow_goal(p_obj, yaw_obj, off, dyaw,
                                   np.zeros(3), 0.5)
    assert np.linalg.norm(v_orb) > 0.2, v_orb


def test_object_roll_and_pitch_are_deliberately_ignored():
    """A rocking object must not translate the vehicle. MANUAL_CONTROL has no
    K/M axis, so a roll the follow tried to answer would be dropped by the
    allocation anyway — better to never ask."""
    R_flat = _Rz(0.7)
    R_tipped = rot_zyx(0.5, -0.4, 0.7)          # same yaw, 30 deg of tilt
    y_flat, _ = ON.yaw_of(R_flat, 0)
    y_tipped, h = ON.yaw_of(R_tipped, 0)
    assert abs(ON.wrap_pi(y_flat - y_tipped)) < 1e-9, (y_flat, y_tipped)
    assert h > 0.25


def test_the_yaw_axis_is_pinned_at_first_lock_and_never_switches():
    """A mid-run re-pick is a 90 degree step in the heading reference."""
    a = ON.ObjectAnchor({"yaw_axis": "auto", "max_jump_m": 10.0})
    t = 100.0
    # first lock: +y is the most horizontal axis, so axis 1 is chosen
    R0 = np.column_stack([np.array([0.0, 0.0, 1.0]),
                          np.array([1.0, 0.0, 0.0]),
                          np.array([0.0, 1.0, 0.0])])
    a.update([0.0, 0.0, -1.0], R0, t, t, 0.5, "tracking")
    assert a.axis == 1, a.axis
    # ...and now +x becomes the most horizontal one. The axis must NOT move.
    R1 = np.eye(3)
    a.update([0.0, 0.0, -1.0], R1, t + 0.1, t + 0.1, 0.5, "tracking")
    assert a.axis == 1, "the heading axis switched mid-run"
    # an explicit axis is pinned from construction, before any observation
    for name, k in (("x", 0), ("y", 1), ("z", 2)):
        assert ON.ObjectAnchor({"yaw_axis": name}).axis == k
    assert ON.ObjectAnchor({"yaw_axis": "none"}).axis is None


def test_yaw_axis_none_stays_none_after_the_first_lock():
    """The regression that flew: "none" and "auto" both leave ``axis`` None at
    construction, so a pinned-check written as ``axis is not None`` let the
    first lock pick an axis for "none" as well. Everything object-frame then
    came alive — the hold offset rotated with the object AND the heading
    reference tracked its yaw — while the meta still said "none".

    Asserting at construction (the old test) cannot see any of it: the axis is
    picked by the first UPDATE, and only an object whose yaw is not zero shows
    the difference. 2026-08-23 flew this way (KNOWN_ISSUES 2026-08-24).
    """
    a = ON.ObjectAnchor({"yaw_axis": "none", "max_jump_m": 10.0})
    t = 100.0
    # an object rotated 40 deg about map +z: a picked axis WOULD report a yaw
    R = rot_zyx(0.0, 0.0, math.radians(40.0))
    a.update([1.0, 0.0, -1.0], R, t, t, 0.5, "tracking")
    assert a.axis is None, "yaw_axis 'none' picked an axis at the first lock"
    assert a.yaw is None and not a.yaw_ok, "'none' must not report a heading"
    assert a.meta()["yaw_axis_used"] == "none"
    # a second, differently-rotated observation must not change that
    a.update([1.0, 0.0, -1.0], rot_zyx(0.0, 0.0, math.radians(-70.0)),
             t + 0.1, t + 0.1, 0.5, "tracking")
    assert a.axis is None and a.yaw is None
    # ...and the follow geometry it feeds stays map-frame: rotating the object
    # moves the goal not at all.
    off, dyaw = ON.offset_in_object_frame([1.0, 1.0, -1.0], 0.3,
                                          [1.0, 0.0, -1.0], None)
    g0, y0, v0 = ON.follow_goal([1.0, 0.0, -1.0], None, off, dyaw,
                                np.zeros(3), 2.0)
    assert np.allclose(g0, [1.0, 1.0, -1.0])
    assert np.allclose(v0, np.zeros(3)), "no orbit term without an axis"
    assert abs(y0 - 0.3) < 1e-12, "heading must not follow the object's yaw"


def test_auto_still_picks_at_the_first_lock():
    """The other half of the same fix: "auto" must remain unpinned until the
    first observation, or it would never choose an axis at all."""
    a = ON.ObjectAnchor({"yaw_axis": "auto", "max_jump_m": 10.0})
    assert a.axis is None, "auto must not have an axis before it has seen one"
    t = 5.0
    a.update([0.0, 0.0, -1.0], np.eye(3), t, t, 0.5, "tracking")
    assert a.axis == 0 and a.yaw_ok


def test_yaw_is_undefined_when_the_chosen_axis_is_vertical():
    """``yaw_from_R`` would happily return a number here; projecting a chosen
    axis degrades honestly instead."""
    a = ON.ObjectAnchor({"yaw_axis": "x", "yaw_min_horiz": 0.25,
                         "max_jump_m": 10.0})
    t = 50.0
    a.update([0.0, 0.0, -1.0], np.eye(3), t, t, 0.5, "tracking")
    assert a.yaw_ok and a.yaw is not None
    good = float(a.yaw)
    # +x now points almost straight down (map +z): horizontal length ~0.09
    R_vert = np.column_stack([np.array([0.06, 0.06, 0.99]),
                              np.array([0.0, 1.0, 0.0]),
                              np.array([-1.0, 0.0, 0.0])])
    _yaw, h = ON.yaw_of(R_vert, 0)
    assert h < 0.25
    a.update([0.0, 0.0, -1.0], R_vert, t + 0.1, t + 0.1, 0.5, "tracking")
    assert not a.yaw_ok, "a vertical axis must not report a heading"
    assert abs(float(a.yaw) - good) < 1e-9, "the last good heading is kept"
    # ...and yaw_axis: none never claims one at all
    assert ON.yaw_of(np.eye(3), None) == (None, 0.0)


def test_one_jump_is_rejected_and_repeated_jumps_reseed():
    a = ON.ObjectAnchor({"yaw_axis": "none", "max_jump_m": 0.35,
                         "reseed_after_n": 5, "pos_lp_alpha": 1.0})
    t = 10.0
    a.update([0.0, 0.0, -1.0], np.eye(3), t, t, 0.5, "tracking")
    p0 = a.p.copy()
    # one outlier 3 m away: rejected, the estimate does not move
    r = a.update([3.0, 0.0, -1.0], np.eye(3), t + 0.1, t + 0.1, 0.5,
                 "tracking")
    assert not r["accepted"] and a.n_reject == 1
    assert np.allclose(a.p, p0)
    # a good one in between clears the streak
    a.update([0.01, 0.0, -1.0], np.eye(3), t + 0.2, t + 0.2, 0.5, "tracking")
    assert a.n_reject == 0
    # ...but five in a row mean the object really did move
    for i in range(5):
        r = a.update([3.0, 0.0, -1.0], np.eye(3), t + 1.0 + 0.1 * i,
                     t + 1.0 + 0.1 * i, 0.5, "tracking")
    assert r["accepted"] and r["reseeded"], r
    assert np.allclose(a.p, [3.0, 0.0, -1.0])
    assert np.allclose(a.v, np.zeros(3)), "a reseed must not inherit velocity"
    assert a.n_reseed == 1


def test_velocity_uses_t_capture_not_wall_time():
    """The pose is published at 10 Hz from a 30 fps camera through a GPU, so a
    host-clock difference measures the pipeline, not the object."""
    a = ON.ObjectAnchor({"yaw_axis": "none", "pos_lp_alpha": 1.0,
                         "vel_lp_alpha": 1.0, "max_jump_m": 10.0})
    a.update([0.0, 0.0, -1.0], np.eye(3), 100.0, 900.0, 0.5, "tracking")
    # 0.5 s of CAPTURE time, but 5 s of wall clock: 0.2 m in 0.5 s = 0.4 m/s
    a.update([0.1, 0.0, -1.0], np.eye(3), 100.5, 905.0, 0.5, "tracking")
    assert abs(float(a.v[0]) - 0.2) < 1e-9, a.v
    assert float(np.linalg.norm(a.v[1:])) < 1e-12


def test_predict_clamps_the_extrapolation():
    """Past ``max_extrap_s`` the point STOPS and only the age keeps rising —
    an estimate that kept gliding would look most confident exactly when it
    knows least."""
    a = ON.ObjectAnchor({"yaw_axis": "none", "pos_lp_alpha": 1.0,
                         "vel_lp_alpha": 1.0, "max_jump_m": 10.0,
                         "max_extrap_s": 0.3, "stale_s": 0.5})
    a.update([0.0, 0.0, -1.0], np.eye(3), 10.0, 10.0, 0.5, "tracking")
    a.update([0.1, 0.0, -1.0], np.eye(3), 10.1, 10.1, 0.5, "tracking")
    assert abs(float(a.v[0]) - 1.0) < 1e-9
    at_clamp = a.predict(10.4)
    assert abs(float(at_clamp["p"][0]) - (0.1 + 0.3)) < 1e-9
    far = a.predict(20.0)
    assert np.allclose(far["p"], at_clamp["p"]), "the point must STOP"
    assert far["extrapolated_s"] == 0.3
    assert far["age_s"] > 9.0 and not far["ok"]
    assert far["state"] == ON.LOST


def test_pick_fix_prefers_the_exact_match():
    """Float equality is the RIGHT test here: _tap_pose computes t_capture
    once per colour frame and puts the identical float in both mailboxes, so
    equality answers exactly the question being asked."""
    hist = [(100.00, np.zeros(3), np.eye(3)),
            (100.05, np.ones(3), np.eye(3)),
            (100.10, 2 * np.ones(3), np.eye(3))]
    e, dt, exact = ON.pick_fix(hist, 100.05, 0.08)
    assert exact and dt == 0.0 and np.allclose(e[1], np.ones(3))
    # no exact match: the nearest inside the tolerance, flagged as loose
    e, dt, exact = ON.pick_fix(hist, 100.06, 0.08)
    assert not exact and abs(dt - 0.01) < 1e-9 and np.allclose(e[1],
                                                               np.ones(3))
    # ...and nothing at all past it
    e, dt, exact = ON.pick_fix(hist, 101.0, 0.08)
    assert e is None and not exact and dt > 0.08
    assert ON.pick_fix([], 100.0, 0.08)[0] is None


def test_a_typo_in_the_object_nav_block_raises():
    for bad in ({"yaw_axes": "auto"}, {"pair_tol": 0.1},
                {"max_excursion": 1.0}, {"enabled": True}):
        try:
            ON.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")
    for bad in ({"yaw_axis": "w"}, {"pair_tol_s": 0.0},
                {"min_distance_m": 2.0, "max_distance_m": 1.0},
                {"pos_lp_alpha": 0.0}, {"reseed_after_n": 0},
                {"hold_s": -1.0}, {"nav_source_required": "third"}):
        try:
            ON.resolve(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_the_shipped_config_resolves():
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig.load(str(ROOT / "config" / "hw_mpc.yaml"))
    r = ON.resolve(cfg.object_nav)
    assert r["nav_source_required"] == "main"
    assert 0.0 < r["pair_tol_s"] <= 0.2, r
    assert r["min_distance_m"] < r["max_distance_m"] <= 3.0
    assert r["max_excursion_m"] == 1.5      # the operator's decision
    assert r["yaw_axis"] in ("auto", "x", "y", "z", "none")


def test_an_unknown_square_shape_raises():
    """`_arm_path`'s else branch is "square", so a typo used to fly a
    RECTANGLE in silence — a mission of a completely different size and
    duration from the one that was asked for."""
    import tempfile as _tf

    from rov_gui.control.geometry import MpcConfig, SHAPES

    assert "follow" in SHAPES
    with _tf.TemporaryDirectory() as td:
        good = Path(td) / "good.yaml"
        good.write_text("mode: dobmpc\nsquare:\n  shape: follow\n")
        assert MpcConfig.load(str(good)).square["shape"] == "follow"
        bad = Path(td) / "bad.yaml"
        bad.write_text("mode: dobmpc\nsquare:\n  shape: squrae\n")
        try:
            MpcConfig.load(str(bad))
        except ValueError as e:
            assert "squrae" in str(e), e
        else:
            raise AssertionError("a misspelled shape was accepted")


# =============================================================================
# 13-25: the worker
# =============================================================================
def _harness():
    import rov_gui.tests.test_control as TC
    return TC


class _Clock:
    """A controllable ``now()`` for the worker tests.

    The follow's rate limits, its freshness ladder and the anchor's ages are
    all in SECONDS OF REAL TIME, and a tight test loop covers milliseconds of
    it — so without this the setpoint walk moves 3 mm, the heading swings 4
    degrees and nothing ever goes stale. The tests would then pass by
    measuring nothing. ``main()`` restores the real clock after every test.
    """

    def __init__(self, t0: float):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float = 0.05) -> float:
        self.t += float(dt)
        return self.t


def _follow_worker(tmp, yaw_axis="auto", **over):
    """A worker with the object anchor armed and a controllable clock."""
    import rov_gui.control.workers as W

    TC = _harness()
    clk = _Clock(TC.now())
    W.now = clk                      # restored by main()
    w, bus, pilots, logs = TC._worker(tmp)
    w.cfg.engage["settle_s"] = 0.0
    w.opts.pose = True
    cfg = {"yaw_axis": yaw_axis, "pos_lp_alpha": 1.0, "vel_lp_alpha": 1.0}
    cfg.update(over)
    w.cfg.object_nav = cfg
    w._setup_object_nav()
    assert w._obj is not None and w._obj_src_ok, w._obj_note
    return TC, w, bus, pilots, logs, clk


def _see(w, TC, t, p_obj, yaw_obj=0.0, p_veh=None, yaw_veh=0.0,
         state="tracking", pair_offset=0.0):
    """Feed ONE camera frame: a tag fix and the object pose that came with it.

    Everything shares the one capture stamp, exactly as
    ``C3VideoWorker._tap_pose`` arranges on hardware. ``pair_offset`` breaks
    that deliberately — the negative control for the cancellation.
    """
    from rov_gui.state import PoseTrack

    p_veh = [-2.0, 0.0, 0.8] if p_veh is None else list(p_veh)
    R_bc, t_bc = w._obj_ext
    p_ned, R_nb, T = _synth(p_veh, yaw_veh, p_obj, yaw_obj, R_bc, t_bc)
    w.on_nav_fix(TC._fix(p_ned, yaw_veh, t))
    w.on_vehicle_imu(TC._imu(yaw=yaw_veh, depth=p_veh[2], t=t))
    w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                conn=TC.Conn.ONLINE, stamp=t))
    if state:
        w.on_pose(PoseTrack(state=state, T_cam_obj=T,
                            t_capture=t + pair_offset, stamp=t))
    return t


def _tag_only(w, TC, t, p_veh=None, yaw_veh=0.0):
    """A camera frame the TAG solved but the object tracker said nothing
    about — the object goes quiet while localization keeps working."""
    return _see(w, TC, t, (0.0, 0.0, 0.0), p_veh=p_veh, yaw_veh=yaw_veh,
                state="")


def _spin(w, clk, n, dt=0.05, feed=None):
    """``n`` control ticks of ``dt`` each, optionally feeding a frame first."""
    for i in range(n):
        t = clk.advance(dt)
        if feed is not None:
            feed(i, t)
        w.tick()


def _engage(w, TC, clk):
    w.on_enable(True)
    w.set_engaged(True)
    assert w.engaged, w.reason
    w._warmup_left = 0


def test_the_worker_composes_the_object_into_the_map_frame():
    """End to end through the real slots: a pose plus the fix from its own
    frame becomes an ObjectFix at the object's true map position."""
    tmp = tempfile.mkdtemp(prefix="objnav_")
    TC, w, bus, _pilots, _logs, clk = _follow_worker(tmp)
    fixes = []
    bus.object_fix.connect(fixes.append)
    p_obj = (-1.6, 0.2, 0.55)
    _see(w, TC, clk.advance(), p_obj, yaw_obj=0.6)
    TC._app().processEvents()
    assert fixes, "no ObjectFix was published"
    o = fixes[-1]
    assert o.ok and o.state == "live", (o.state, o.note)
    assert o.pair_exact and o.pair_dt_ms == 0.0
    assert np.allclose(o.p_map, p_obj, atol=1e-6), (o.p_map, p_obj)
    assert abs(ON.wrap_pi(o.yaw_map - 0.6)) < 1e-6
    # ...and a pose paired with a fix from a DIFFERENT frame is flagged
    _see(w, TC, clk.advance(), p_obj, yaw_obj=0.6, pair_offset=0.01)
    TC._app().processEvents()
    assert not fixes[-1].pair_exact and fixes[-1].pair_dt_ms > 0.0
    w.teardown()


def test_follow_refuses_without_an_object_lock():
    """Each cause independently, and each with its own sentence — "follow
    refused" with no reason is a pool session spent guessing."""
    tmp = tempfile.mkdtemp(prefix="objref_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp)
    _tag_only(w, TC, clk.advance())
    _engage(w, TC, clk)
    w.set_scenario({"shape": "follow", "speed": 0.1})
    # 1) no lock at all
    w.set_traj(True)
    assert w.follow is None and "no object lock" in w.reason, w.reason
    # 2) a pipeline that is REGISTERING is not a lock, however fresh it is
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55), state="registering")
    w.set_traj(True)
    assert w.follow is None and "no object lock" in w.reason, w.reason
    # 3) an object outside the working range stops being a lock — but only
    #    once it has AGED out. One bad frame must not drop a lock, so the
    #    range gate is fed for longer than max_extrap_s.
    R_bc, t_bc = w._obj_ext
    # 5 cm down the optical axis, computed from the real extrinsic rather
    # than guessed — well inside min_distance_m whatever the mount is.
    too_close = (np.array([-2.0, 0.0, 0.8]) + np.asarray(t_bc, float)
                 + np.asarray(R_bc, float) @ np.array([0.0, 0.0, 0.05]))
    for _ in range(6):
        _see(w, TC, clk.advance(0.15), tuple(too_close))
    w.set_traj(True)
    assert w.follow is None and "no object lock" in w.reason, w.reason
    # 4) a lock: it arms
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    w.set_traj(True)
    assert w.follow is not None, w.reason
    w.teardown()

    # 5) no --pose at all
    tmp2 = tempfile.mkdtemp(prefix="objref2_")
    w2, _b, _p, _l = TC._worker(tmp2)
    w2.cfg.engage["settle_s"] = 0.0
    t = clk.advance()          # the fake clock is what this worker reads too
    w2.on_nav_fix(TC._fix([-2.0, 0.0, 0.8], 0.0, t))
    w2.on_vehicle_imu(TC._imu(depth=0.8, t=t))
    w2.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                 conn=TC.Conn.ONLINE, stamp=t))
    _engage(w2, TC, clk)
    assert w2._obj is None
    w2.set_scenario({"shape": "follow", "speed": 0.1})
    w2.set_traj(True)
    assert w2.follow is None and "--pose" in w2.reason, w2.reason
    w2.teardown()


def test_follow_refuses_under_mpcc():
    """HwMpcc opens set_target_ned with `del v_ned, r_ned` and rebuilds its
    whole path every call — a 20 Hz setpoint with no feedforward is precisely
    the unflyable case."""
    from rov_gui.control.mpcc_bridge import HwMpcc
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.pid import HwPid

    assert HwMpcc.follow_ok is False
    assert HwDobMpc.follow_ok is True and HwPid.follow_ok is True

    tmp = tempfile.mkdtemp(prefix="objmpcc_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp)
    _engage(w, TC, clk) if _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55)) \
        else None
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    w.set_scenario({"shape": "follow", "speed": 0.1})
    w.ctrl.follow_ok = False            # stand in for the mpcc bridge
    w.set_traj(True)
    assert w.follow is None and "cannot follow" in w.reason, w.reason
    # fail-CLOSED: a controller that never considered the question refuses too
    del w.ctrl.follow_ok
    w.ctrl.__class__.follow_ok = False
    try:
        w.set_traj(True)
        assert w.follow is None, "a controller with no follow_ok was followed"
    finally:
        w.ctrl.__class__.follow_ok = True
    w.teardown()


def test_follow_refuses_when_nav_source_is_second():
    """The object rides the C3. If the tag fix comes from the ROV RGB the
    extrinsic cannot cancel, and the ROV RGB's whole extrinsic is [예측]."""
    tmp = tempfile.mkdtemp(prefix="objsrc_")
    TC, w, bus, _pilots, _logs, clk = _follow_worker(tmp)
    w.nav_cfg.nav_source = "second"
    w._setup_object_nav()
    # The anchor stays BUILT on purpose, so the panel is told why rather than
    # falling silent — but nothing is composed and follow refuses.
    assert w._obj is not None and not w._obj_src_ok
    assert "nav_source" in w._obj_note, w._obj_note
    fixes = []
    bus.object_fix.connect(fixes.append)
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    TC._app().processEvents()
    assert fixes and not fixes[-1].ok and fixes[-1].p_map is None
    assert "nav_source" in fixes[-1].note
    _engage(w, TC, clk)
    w.set_scenario({"shape": "follow", "speed": 0.1})
    w.set_traj(True)
    assert w.follow is None and "nav_source" in w.reason, w.reason
    w.teardown()


def test_arming_follow_does_not_move_the_vehicle():
    """THE PROPERTY of capturing the offset from the CURRENT pose: the target
    the instant a follow arms is the vehicle's own position. START never
    causes a lunge. That is a property, not an omission."""
    tmp = tempfile.mkdtemp(prefix="objarm_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp)
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.2, 0.55)))
    eta = np.asarray(w._eta, float).copy()
    w.set_scenario({"shape": "follow", "speed": 0.1})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    target = np.asarray(w.ctrl._target[0], float)
    assert float(np.linalg.norm(target - eta[:3])) < 1e-3, (target, eta[:3])
    assert abs(ON.wrap_pi(float(w.ctrl._target[1]) - float(eta[5]))) < 1e-6
    assert np.allclose(w.ctrl._target[2], np.zeros(3)), "no feedforward at arm"
    # ...and a second later it is still there (the object has not moved)
    _spin(w, clk, 20, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.2, 0.55)))
    assert float(np.linalg.norm(np.asarray(w.ctrl._target[0], float)
                                - eta[:3])) < 1e-2
    w.teardown()


def test_a_moving_object_moves_the_target_and_the_feedforward_is_not_zero():
    """The direct defence of the measured leash limit: with no feedforward the
    setpoint settles where kp*lead balances drag — 0.084 m/s — and a follow
    could not keep up with anything faster (memory:
    approach-speed-leash-limited)."""
    tmp = tempfile.mkdtemp(prefix="objmove_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.2, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.3})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    p0 = np.asarray(w.ctrl._target[0], float).copy()
    # the object now translates at 0.2 m/s along map +y for 2 s
    t_go = clk.t
    _spin(w, clk, 40, feed=lambda i, t: _see(
        w, TC, t, (-1.6, 0.2 + 0.2 * (t - t_go), 0.55)))
    p1 = np.asarray(w.ctrl._target[0], float)
    assert abs(float(p1[1] - p0[1])) > 0.10, (p0, p1)
    v = np.asarray(w.ctrl._target[2], float)
    assert float(np.linalg.norm(v)) > 0.05, (
        f"the follow issued no feedforward ({v}) — it cannot outrun the leash")
    # the feedforward must point the way the object went, not just be nonzero
    assert float(v[1]) > 0.05, v
    w.teardown()


def test_a_rotating_object_makes_the_vehicle_orbit():
    """The orbit case, and the one a wrong yaw_axis shows up in fastest."""
    tmp = tempfile.mkdtemp(prefix="objorbit_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="x")
    p_obj = (-1.6, 0.0, 0.55)
    _see(w, TC, clk.advance(), p_obj, yaw_obj=0.0)
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, p_obj, yaw_obj=0.0))
    w.set_scenario({"shape": "follow", "speed": 0.5})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    f = w.follow
    r0 = float(np.linalg.norm(np.asarray(f["offset_obj"], float)[:2]))
    assert r0 > 0.1, r0
    yaw0 = float(f["sp_yaw"])
    # the object turns 90 degrees over 3 s; the goal must swing round it
    t_go = clk.t
    _spin(w, clk, 80, feed=lambda i, t: _see(
        w, TC, t, p_obj,
        yaw_obj=(math.pi / 2) * min(1.0, (t - t_go) / 3.0)))
    sp = np.asarray(w.follow["sp"], float)
    r1 = float(np.linalg.norm(sp[:2] - np.asarray(p_obj)[:2]))
    assert abs(r1 - r0) < 0.3 * r0, f"the orbit radius changed: {r0} -> {r1}"
    assert abs(ON.wrap_pi(float(w.follow["sp_yaw"]) - yaw0)
               - math.pi / 2) < math.radians(20), (
        "the target heading did not follow the object round: "
        f"{math.degrees(ON.wrap_pi(float(w.follow['sp_yaw']) - yaw0)):.0f} deg")
    w.teardown()


def test_the_leash_clamps_a_jumped_estimate():
    """Gating the STEP is not enough — one long tick jumps the leash, which is
    `_tick_approach`'s lesson. The clamp is on the RESULT, so it holds on
    EVERY tick including a 1 s one."""
    tmp = tempfile.mkdtemp(prefix="objleash_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=99.0, max_distance_m=99.0,
        max_excursion_m=99.0)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 5.0})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    lead = float(w.cfg.engage["approach_lead_m"])
    # the object teleports 3 m, and two of the ticks last a full second
    for dt in (0.05, 1.0, 0.05, 0.05, 1.0, 0.05):
        t = clk.advance(dt)
        _see(w, TC, t, (1.4, 0.0, 0.55))
        w.tick()
        sp = np.asarray(w.follow["sp"], float)
        hull = w._datum_to_map_p(w._eta[:3])
        assert float(np.linalg.norm(sp - hull)) <= lead + 1e-6, (
            f"a dt={dt} tick put the setpoint "
            f"{np.linalg.norm(sp - hull):.3f} m ahead of a {lead} m leash")
    w.teardown()


def test_the_feedforward_is_capped_at_what_the_vehicle_can_swim():
    """The leash bounds the SETPOINT; nothing bounded the FEEDFORWARD, and the
    feedforward is the half that reaches the far end of the horizon
    (``_xref_ned``: pos_world = p_ref + v_ref*k*dt). 2026-08-23 a mis-registered
    object drove it to 3.15 m/s while the logged stage-0 setpoint sat frozen
    4 cm from the datum — invisible in the CSV, and the reference acados
    finally failed on (KNOWN_ISSUES 2026-08-24)."""
    tmp = tempfile.mkdtemp(prefix="objff_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=99.0, max_distance_m=99.0,
        max_excursion_m=99.0)
    cap = float(w.cfg.engage["follow_ff_max_m_s"])
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 5.0})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    # the object "moves" 0.4 m per 50 ms frame = 8 m/s of apparent velocity
    t_go = clk.t
    for i in range(8):
        t = clk.advance(0.05)
        _see(w, TC, t, (-1.6 + 0.4 * (i + 1), 0.0, 0.55))
        w.tick()
        v = np.asarray(w.ctrl._target[2], float)
        assert float(np.linalg.norm(v)) <= cap + 1e-6, (
            f"feedforward {np.linalg.norm(v):.2f} m/s exceeded the "
            f"{cap} m/s cap — the horizon is being relocated")
    assert int(w.follow["ff_clipped_n"]) > 0, "the clip never engaged"
    assert t_go < clk.t
    w.teardown()


def test_a_reseed_ends_the_follow_instead_of_chasing_the_new_anchor():
    """object_nav's jump gate REJECTS one outlier but reseeds after
    ``reseed_after_n`` of them — 0.5 s at 10 Hz. That is right for a display
    marker and wrong under a live follow: a genuine drag arrives as accepted
    small steps, so a reseed here is a re-registration, and on 2026-08-23 the
    follow consumed a 1.32 m snap onto a phantom 0.44 m from the lens.

    Ending the follow keeps depth and heading (it is a demotion to STATION,
    not a disengage) and makes the operator re-arm on a pose they can see."""
    tmp = tempfile.mkdtemp(prefix="objreseed_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=0.35, reseed_after_n=3,
        max_distance_m=99.0, max_excursion_m=99.0)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.1})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    sp0 = np.asarray(w.follow["sp"], float).copy()
    # three consecutive 1.3 m outliers: reject, reject, then RESEED
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-0.3, 0.0, 0.55)))
    assert w._obj.n_reseed == 1, "the anchor did not reseed"
    assert w.follow is None, "the follow chased a re-seeded anchor"
    assert w.station is not None, "a reseed must DEMOTE, not disengage"
    assert w.engaged, "a reseed must never disengage"
    assert "re-seeded" in str(w.station.get("from_follow", "")), w.station
    # the hold is the last good setpoint, not the phantom
    held = np.asarray([w.station["origin_ned"][0], w.station["origin_ned"][1],
                       w.station["depth_ned"]], float)
    assert float(np.linalg.norm(w._map_to_datum_p(sp0) - held)) < 1e-6
    w.teardown()


def test_the_excursion_clamp_freezes_rather_than_disengaging():
    """The operator's 1.5 m limit is a REFERENCE clamp — not a geofence, not
    a refusal, and never a disengage."""
    tmp = tempfile.mkdtemp(prefix="objexc_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=99.0, max_distance_m=99.0,
        max_excursion_m=0.15)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 5.0})
    w.set_traj(True)
    arm = np.asarray(w.follow["arm_p_map"], float)
    t_go = clk.t
    # the object walks steadily away at 0.15 m/s for 3 s — 45 cm, three times
    # the clamp
    for _ in range(60):
        t = clk.advance()
        _see(w, TC, t, (-1.6, 0.15 * (t - t_go), 0.55))
        w.tick()
        assert w.engaged, w.reason
    assert w.follow is not None, "the clamp disengaged or demoted"
    assert w.follow["state"] == "leashed", w.follow["state"]
    sp = np.asarray(w.follow["sp"], float)
    assert float(np.linalg.norm(sp - arm)) <= 0.15 + 1e-6, sp
    w.teardown()


def test_a_registering_pose_is_not_a_lock():
    """PoseWorker publishes at 10 Hz whatever the pipeline is doing, so a
    30 s `registering` arrives perfectly fresh. Age alone can never catch it."""
    tmp = tempfile.mkdtemp(prefix="objreg_")
    TC, w, bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    fixes = []
    bus.object_fix.connect(fixes.append)
    t0 = clk.advance()
    _see(w, TC, t0, (-1.6, 0.0, 0.55))
    assert w._obj.state(t0) == ON.LIVE
    # ...and now it drops back to registering, at 10 Hz, with a fresh stamp
    for _ in range(9):
        _see(w, TC, clk.advance(0.1), (-1.6, 0.0, 0.55), state="registering")
    TC._app().processEvents()
    assert w._obj.state(clk.t) != ON.LIVE, (
        "a registering pipeline read as a lock")
    o = fixes[-1]
    assert not o.ok and o.pose_state == "registering", (o.ok, o.pose_state)
    # the age is TINY — this could only ever have been caught by the state
    assert o.age_s is not None and o.age_s < 1.5, o.age_s
    w.teardown()


def test_a_lost_object_freezes_then_drops_to_station_and_never_disengages():
    """Disengaging drops DEPTH hold on a negatively buoyant vehicle, and a
    sinking vehicle's view changes — which makes re-acquiring the object LESS
    likely. Same argument station_bridge.py makes at length."""
    tmp = tempfile.mkdtemp(prefix="objlost_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", hold_s=0.5, max_extrap_s=0.2)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    sp_at_freeze = np.asarray(w.follow["sp"], float).copy()
    # the tracker goes quiet, but the TAG keeps coming — so no bridge and no
    # interlock fire; only the object is gone.
    saw_stale = False
    for _ in range(8):
        t = clk.advance()
        _tag_only(w, TC, t)
        w.tick()
        if w.follow is not None and w.follow["state"] == "stale":
            saw_stale = True
            assert np.allclose(np.asarray(w.follow["sp"], float),
                               sp_at_freeze, atol=1e-9), "the freeze moved"
            assert np.allclose(w.ctrl._target[2], np.zeros(3)), \
                "a frozen follow must issue NO feedforward"
    assert saw_stale, "the follow never entered the freeze"
    assert w.follow is not None, "it dropped to station before hold_s"
    for _ in range(30):
        t = clk.advance()
        _tag_only(w, TC, t)
        w.tick()
    assert w.engaged, f"a lost object disengaged: {w.reason}"
    assert w.follow is None and w.station is not None, w.reason
    assert w.station["from_follow"] == "object lost", w.station
    assert np.allclose(w.station["origin_ned"],
                       w._map_to_datum_p(sp_at_freeze)[:2], atol=1e-9)
    w.teardown()


def test_a_tag_dropout_during_follow_demotes_to_station_and_the_bridge_carries_it():
    """Losing the tag loses the OBJECT too — T_map_obj is composed FROM a
    NavFix. So the follow is demoted FIRST and the existing station ladder
    picks it up; station_bridge.py itself is untouched."""
    from rov_gui.control import station_bridge as SB

    tmp = tempfile.mkdtemp(prefix="objdrop_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    assert w._bridge_anchor is not None
    # THE LOCALIZER STOPS. Nothing else does: the autopilot keeps talking, so
    # only the tag-fix fault is in play and it is a bridgeable one.
    for _ in range(40):
        t = clk.advance()
        w.on_vehicle_imu(TC._imu(depth=0.8, t=t))
        w.on_telemetry(TC.Telemetry(armed=True, mode="MANUAL",
                                    conn=TC.Conn.ONLINE, stamp=t))
        w.tick()
        if not w.engaged:
            break
    assert w.engaged, f"a tag dropout during a follow disengaged: {w.reason}"
    assert w.follow is None, "the follow kept walking with no fix"
    assert w.station is not None and w.station["from_follow"] == "tag fix lost"
    assert w._bridge.active and w._bridge.tier in (SB.TIER_IMU, SB.TIER_COAST)
    w.teardown()


def test_disengage_and_stop_both_clear_the_follow():
    """`self.follow` must die everywhere `self.station` does — all four —
    or a follow survives a disengage and keeps walking a setpoint for a loop
    that is no longer running."""
    tmp = tempfile.mkdtemp(prefix="objclr_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")

    def _arm():
        _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
        if not w.engaged:
            _engage(w, TC, clk)
        _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
        w.set_scenario({"shape": "follow", "speed": 0.2})
        w.set_traj(True)
        assert w.follow is not None, w.reason

    # 1) STOP TRAJ
    _arm()
    w.set_traj(False)
    assert w.follow is None and w.station is None
    # 2) arming another mission over the top
    _arm()
    w.set_scenario({"shape": "station", "origin_tag": None,
                    "origin": "current", "yaw_map_deg": 0.0})
    w.set_traj(True)
    assert w.follow is None and w.station is not None
    w.set_traj(False)
    # 3) disengage
    _arm()
    w.disengage("test")
    assert w.follow is None and not w.engaged
    # 4) ...and a fresh engagement does not inherit it
    _arm()
    w.follow["state"] = "sentinel"
    w.disengage("test 2")
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    w.set_engaged(True)
    assert w.engaged and w.follow is None and w.station is None
    w.teardown()


def test_stop_traj_ends_a_follow_and_holds_where_it_is():
    """STOP is keyed on ``traj_on`` everywhere else, and a follow — like a
    station — never sets it. Without the panel's extra term the only way to
    end one would be DISENGAGE, which drops depth hold."""
    tmp = tempfile.mkdtemp(prefix="objstop_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None and not w.traj_on
    # the panel's own gate, which is what makes the button pressable
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryWindow

    TC._app()
    panel = TrajectoryWindow()
    st = MpcStatus(engaged=True, traj_on=False, scenario=w.follow)
    panel.add_status(st)
    assert panel.btn_stop.isEnabled(), "STOP was dead during a follow"
    # ...and the object is MOVING when STOP is pressed, so a feedforward is
    # live. This is the half the test used to miss: `set_traj(False)` re-issued
    # a stationary hold only `if self.traj_on`, which a follow never sets, so
    # clearing self.follow left the controller holding the last follow target
    # WITH its v_ned — and `_xref_ned` rebuilds the whole horizon as
    # p_ref + v_ref*k*dt every tick. STOP said "stopped" while the vehicle kept
    # being commanded along a velocity ray, with nothing left to walk or clamp
    # it (KNOWN_ISSUES 2026-08-24).
    t_go = clk.t
    _spin(w, clk, 10, feed=lambda i, t: _see(
        w, TC, t, (-1.6, 0.0 + 0.15 * (t - t_go), 0.55)))
    assert float(np.linalg.norm(np.asarray(w.ctrl._target[2], float))) > 1e-3, \
        "the object was not moving, so this test cannot see the bug"
    eta = np.asarray(w._eta, float).copy()
    w.set_traj(False)
    assert w.follow is None and w.engaged
    assert float(np.linalg.norm(np.asarray(w.ctrl._target[0], float)
                                - eta[:3])) < 1e-9, "STOP moved the vehicle"
    assert np.allclose(np.asarray(w.ctrl._target[2], float), np.zeros(3)), (
        f"STOP left a velocity feedforward installed "
        f"({w.ctrl._target[2]}) — the reference keeps running")
    w.teardown()


def test_a_leashed_follow_stops_pushing_outward():
    """The excursion clamp stops the GOAL at the sphere, but the feedforward is
    a second channel into the same reference: a clamped follow could sit at the
    limit with v_ff pinned outward on every tick, leaning the horizon ~0.9 m
    past a boundary the operator set. Tangential motion must survive."""
    tmp = tempfile.mkdtemp(prefix="objleash2_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=99.0, max_distance_m=99.0,
        max_excursion_m=0.10)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.5})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    arm = np.asarray(w.follow["arm_p_map"], float)
    # the object walks steadily away; the goal pins at 0.10 m from the arm point
    t_go = clk.t
    leashed_seen = False
    for i in range(30):
        t = clk.advance(0.05)
        _see(w, TC, t, (-1.6 + 0.10 * (t - t_go), 0.0, 0.55))
        w.tick()
        if w.follow is None or w.follow["state"] != "leashed":
            continue
        leashed_seen = True
        goal_dir = np.asarray(w.follow["sp"], float) - arm
        n = float(np.linalg.norm(goal_dir))
        if n < 1e-9:
            continue
        v_ff = np.asarray(w.ctrl._target[2], float)
        # _target[2] is DATUM frame; the outward direction is MAP. Both share
        # the datum's yaw rotation, so compare in the datum frame.
        out = w._map_to_datum_v(goal_dir / n)
        assert float(v_ff @ out) <= 1e-6, (
            f"a leashed follow still fed {float(v_ff @ out):.3f} m/s outward")
    assert leashed_seen, "the excursion clamp never engaged — test is blind"
    w.teardown()


def test_an_implausible_object_velocity_ends_the_follow():
    """The cap alone is not an answer. A phantom that slides SMOOTHLY stays
    under the jump gate, never reseeds, and would otherwise buy a permanent
    pull at the cap. The 2026-08-23 estimates demanded 3.15 and 2.97 m/s —
    ten times what this vehicle can swim."""
    tmp = tempfile.mkdtemp(prefix="objfast_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(
        tmp, yaw_axis="none", max_jump_m=99.0, max_distance_m=99.0,
        max_excursion_m=99.0)
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    # 0.15 m per 50 ms frame = 3 m/s of apparent object velocity
    t_go = clk.t
    _spin(w, clk, 40, feed=lambda i, t: _see(
        w, TC, t, (-1.6 + 3.0 * (t - t_go), 0.0, 0.55)))
    assert w.follow is None, "a 3 m/s object estimate kept its follow"
    assert w.station is not None and w.engaged, "it must DEMOTE, not disengage"
    assert "implausible" in str(w.station.get("from_follow", "")), w.station
    w.teardown()


def test_an_over_budget_tick_demotes_a_follow_and_is_measured_here():
    """Every controller used to report its own solve time and none reported the
    truth — both acados bridges prefer `time_tot`, which times only the QP (the
    6.5 s tick of 2026-08-23 logged 1.15 ms), and HwPid returns a constant.
    The bracket lives in MpcWorker.tick now, so it cannot be gamed."""
    import time as _time

    tmp = tempfile.mkdtemp(prefix="objslow_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    _see(w, TC, clk.advance(), (-1.6, 0.0, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 3, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    w.cfg.engage["tick_overrun_ms"] = 20.0
    real_step = w.ctrl.step

    def slow_step(*a, **k):
        _time.sleep(0.05)                     # 50 ms > the 20 ms budget
        return real_step(*a, **k)

    w.ctrl.step = slow_step
    _spin(w, clk, 1, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    assert w._tick_ms > 20.0, w._tick_ms
    # A follow DEMOTES on the first overrun: the reference is the likeliest
    # thing that broke, and disengaging drops depth hold on a -5.7 N vehicle.
    assert w.follow is None and w.station is not None and w.engaged, w.reason
    assert "budget" in str(w.station.get("from_follow", "")), w.station
    # ...and with no follow left, two in a row disengage.
    _spin(w, clk, 2, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.0, 0.55)))
    assert not w.engaged, "a sustained overrun must still disengage"
    w.teardown()


def test_the_engage_block_rejects_typos_and_disabled_limits():
    """The one safety block that used to merge blind. `follow_ff_max_m_s: 0` is
    exactly what an operator would type for "no feedforward", and it used to
    mean UNCAPPED — the 2026-08-23 failure verbatim."""
    import tempfile as _tf

    from rov_gui.control.geometry import MpcConfig

    with _tf.TemporaryDirectory() as td:
        def _load(body):
            p = Path(td) / "c.yaml"
            p.write_text("mode: dobmpc\n" + body)
            return MpcConfig.load(str(p))

        assert _load("engage:\n  follow_ff_max_m_s: 0.25\n"
                     ).engage["follow_ff_max_m_s"] == 0.25
        for body, needle in (
                ("engage:\n  follow_ff_max:  0.25\n", "follow_ff_max"),
                ("engage:\n  follow_ff_max_m_s: 0\n", "follow_ff_max_m_s"),
                ("engage:\n  follow_ff_max_m_s: -1\n", "follow_ff_max_m_s"),
                ("engage:\n  tick_overrun_ms: 0\n", "tick_overrun_ms"),
                ("engage:\n  max_solver_fails: 0\n", "max_solver_fails")):
            try:
                _load(body)
            except ValueError as e:
                assert needle in str(e), (body, e)
            else:
                raise AssertionError(f"accepted {body!r}")


def test_the_csv_has_the_object_columns_and_the_meta_says_schema_7():
    from rov_gui.control.workers import CSV_HEADER

    cols = CSV_HEADER.strip().split(",")
    for name in ("obj_px", "obj_py", "obj_pz", "obj_yaw_deg", "obj_age_s",
                 "obj_pair_dt_ms", "obj_pair_exact", "obj_state",
                 "follow_state", "follow_err_m"):
        assert name in cols, name
    # appended at the END, so every by-name reader of an older run still
    # works (the 2026-08-30 replay columns landed after tick_ms, same rule)
    assert cols[-14:] == ["obj_px", "obj_py", "obj_pz", "obj_yaw_deg",
                          "obj_age_s", "obj_pair_dt_ms", "obj_pair_exact",
                          "obj_state", "follow_state", "follow_err_m",
                          "tick_ms", "plan_id", "ref_src", "grip_cmd"]

    tmp = tempfile.mkdtemp(prefix="objcsv_")
    TC, w, _bus, _pilots, _logs, clk = _follow_worker(tmp, yaw_axis="none")
    _see(w, TC, clk.advance(), (-1.6, 0.2, 0.55))
    _engage(w, TC, clk)
    _spin(w, clk, 4, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.2, 0.55)))
    w.set_scenario({"shape": "follow", "speed": 0.2})
    w.set_traj(True)
    assert w.follow is not None, w.reason
    _spin(w, clk, 4, feed=lambda i, t: _see(w, TC, t, (-1.6, 0.2, 0.55)))
    path = w._csv_path
    m = w._run_meta("test")
    w.disengage("done")
    assert m["schema_version"] == 8, m["schema_version"]   # 8: + plan_stream
    on = m["object_nav"]
    assert on["enabled"] and on["source_ok"]
    assert on["pair_exact_ratio"] == 1.0, on
    assert on["follow"]["kind"] == "follow" and on["follow"]["hold_m"] > 0.0
    rows = Path(path).read_text().strip().split("\n")
    head = rows[0].split(",")
    last = rows[-1].split(",")
    assert len(last) == len(head)
    ix = {n: i for i, n in enumerate(head)}
    assert last[ix["obj_pair_exact"]] == "1", last[ix["obj_pair_exact"]]
    assert last[ix["obj_state"]] == "live"
    assert last[ix["follow_state"]] in ("following", "leashed")
    # obj_p* are world FLU in the DATUM frame, exactly like px/py/pz, so the
    # relative vector is a subtraction and nothing else.
    rel = np.array([float(last[ix["obj_px"]]) - float(last[ix["px"]]),
                    float(last[ix["obj_py"]]) - float(last[ix["py"]]),
                    float(last[ix["obj_pz"]]) - float(last[ix["pz"]])])
    assert 0.1 < float(np.linalg.norm(rel)) < 1.5, rel
    w.teardown()


def main() -> int:
    import rov_gui.control.workers as W

    real_now = W.now
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
        finally:
            # _follow_worker installs a fake clock; never let one leak into
            # the next test, however the current one ended.
            W.now = real_now
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
