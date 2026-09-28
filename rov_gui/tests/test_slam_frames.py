#!/usr/bin/env python3
"""
test_slam_frames.py — the ORB-SLAM3 <-> NED frame algebra and the health
machine, offline.

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_slam_frames.py

Pure numpy: no Qt, no cv2, no ORB-SLAM3 build. What it pins down
(control/slam_frames.py):

* the wire's 'TWC ' payload is 72 bytes, checked against a real struct format
  and against the arithmetic 4+8+4+1+3+4+48 — a wrong length does not throw,
  it desynchronises the stream forever;
* ``P_NED_SLAM`` is a ROTATION (det +1), not a mirror, and is exactly
  p_ned = (z, x, y)_slam;
* the forward chain is ``TagNav._solution`` (tagnav.py:336-350) with the SLAM
  world in the map's place — checked against a literal transcription of that
  method, so the two nav front-ends cannot drift apart;
* FORWARD-then-BACKWARD is the identity to 1e-12 on synthetic poses, both for
  ``Twc <-> (p_ned, R_ned_body)`` and for datum-NED knots through the engage
  isometry — this is what makes a drawn knot land with zero registration
  error rather than "close enough";
* a global rigid transform of the SLAM world (the yaw+translation family the
  engage datum can absorb) leaves datum-frame knots UNCHANGED, and a
  world-tilt does not — the datum is horizontal by construction;
* ``rpy_from_R`` inverts ``state_assembler.rot_zyx`` on random rotations, and
  the datum's yaw crossing survives the +-pi seam;
* the session counter increments exactly once per OK -> LOST -> OK cycle and
  not at all during a steady LOST run;
* the jump guard fires on a teleport, is silent on fast-but-plausible
  handheld motion, and is suppressed across a gap (the plotter's rule,
  "…/UMI_Underwater/slam/plot_trajectory.py":121-125);
* degenerate poses — non-orthonormal R, NaN, an all-zero Twc arriving WITH
  state == OK — are rejected with a reason and classified LOST, which is the
  case that arms a session rather than quietly flying a garbage pose.
"""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.control.geometry import NavConfig
from rov_gui.control.slam_frames import (
    DEFAULT_GAP_S, DEFAULT_MAX_SPEED_MPS, KNOT_FLAG_ANCHOR, POSE_MAGIC,
    POSE_RECORD_LEN, POSE_RECORD_STRUCT, POSE_VERSION, P_NED_SLAM, TAG_TWC,
    TRACK_STATE_OK, EngageDatum, PoseRecord, SlamNav, TrackMonitor,
    TrackStatus, check_twc, rpy_from_R, wrap_pi)
from rov_gui.control.state_assembler import rot_zyx

R_BC, T_BC = NavConfig().R_t_frd_cam("main")     # the real C3 extrinsic
NAV = SlamNav(R_BC, T_BC)
TOL = 1e-12


# --------------------------------------------------------------- fixtures
def _rand_R(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-ish random rotation via QR (test_policy_frames' helper)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, [0, 1]] = q[:, [1, 0]]
    return q


def _twc(R: np.ndarray, t) -> np.ndarray:
    out = np.zeros((3, 4))
    out[:3, :3] = R
    out[:3, 3] = np.asarray(t, float).reshape(3)
    return out


def _rec(Twc, t=0.0, seq=0, state=TRACK_STATE_OK, map_changed=False,
         n_tracked=1200) -> PoseRecord:
    return PoseRecord(seq=seq, t=t, state=state, map_changed=map_changed,
                      n_tracked=n_tracked, Twc=np.asarray(Twc, float))


def _tagnav_solution(R_cm, t_cm, R_bc, t_bc, R_nm):
    """``TagNav._solution`` (tagnav.py:336-350) + its precomputed inverse
    extrinsic (tagnav.py:279-283), transcribed here so the parity check is
    against that method's algebra and not against slam_frames' own."""
    R_cb = np.asarray(R_bc, float).T
    t_cb = -np.asarray(R_bc, float).T @ np.asarray(t_bc, float)
    R_mc = np.asarray(R_cm, float).T
    t_mc = -np.asarray(R_cm, float).T @ np.asarray(t_cm, float)
    R_map_body = R_mc @ R_cb
    t_map_body = R_mc @ t_cb + t_mc
    return np.asarray(R_nm, float) @ t_map_body, np.asarray(R_nm, float) @ R_map_body


# =============================================================================
# the wire
# =============================================================================
def test_wire_record_len():
    """72 = 4 (seq) + 8 (t) + 4 (state) + 1 (map_changed) + 3 (rsv) +
    4 (n_tracked) + 48 (12 f32). Both sides state this arithmetic; here it is
    also checked against a real struct format, so a silently-realigned or
    re-ordered layout fails instead of drifting."""
    assert POSE_RECORD_LEN == 4 + 8 + 4 + 1 + 3 + 4 + 48 == 72
    assert struct.calcsize(POSE_RECORD_STRUCT) == POSE_RECORD_LEN
    assert POSE_RECORD_STRUCT.startswith("<"), "native alignment would pad"
    assert POSE_MAGIC == b"POSE" and len(POSE_MAGIC) == 4 and POSE_VERSION == 1
    assert TAG_TWC == b"TWC " and len(TAG_TWC) == 4, "tags are 4 bytes"
    assert KNOT_FLAG_ANCHOR == 1


def test_P_is_a_rotation():
    """A cyclic permutation: det +1 (a rotation, not a mirror) and
    p_ned = (z, x, y)_slam — forward becomes north, right becomes east, down
    stays down."""
    assert abs(np.linalg.det(P_NED_SLAM) - 1.0) < TOL
    assert np.abs(P_NED_SLAM.T @ P_NED_SLAM - np.eye(3)).max() < TOL
    p = np.array([1.0, 2.0, 3.0])                      # (right, down, fwd)
    assert np.allclose(P_NED_SLAM @ p, [3.0, 1.0, 2.0])


# =============================================================================
# forward
# =============================================================================
def test_forward_is_tagnav_solution():
    rng = np.random.default_rng(7)
    for _ in range(50):
        R_wc, t_wc = _rand_R(rng), rng.standard_normal(3) * 2.0
        p, R = NAV.ned_from_twc(_twc(R_wc, t_wc))
        # camera_T_map that the PnP path would have produced for this Twc.
        p_ref, R_ref = _tagnav_solution(R_wc.T, -R_wc.T @ t_wc,
                                        R_BC, T_BC, P_NED_SLAM)
        assert np.abs(p - p_ref).max() < TOL, np.abs(p - p_ref).max()
        assert np.abs(R - R_ref).max() < TOL


def test_forward_matches_the_brief_algebra():
    """The chain as the design states it, term by term:
    R_slam_body = R_wc R_bc.T, t_slam_body = t_wc - R_wc R_bc.T t_bc."""
    rng = np.random.default_rng(11)
    R_wc, t_wc = _rand_R(rng), rng.standard_normal(3)
    p, R = NAV.ned_from_twc(_twc(R_wc, t_wc))
    R_slam_body = R_wc @ R_BC.T
    t_slam_body = t_wc - R_wc @ R_BC.T @ T_BC
    assert np.abs(p - P_NED_SLAM @ t_slam_body).max() < TOL
    assert np.abs(R - P_NED_SLAM @ R_slam_body).max() < TOL


def test_fix_shape_and_euler():
    rng = np.random.default_rng(3)
    rec = _rec(_twc(_rand_R(rng), rng.standard_normal(3)), t=12.5, seq=9,
               n_tracked=884)
    fix = NAV.fix_from_pose(rec, session=2)
    assert fix.t_capture == 12.5 and fix.n_tracked == 884 and fix.session == 2
    assert np.abs(rot_zyx(fix.roll, fix.pitch, fix.yaw)
                  - fix.R_ned_body).max() < 1e-12
    eta = fix.eta6()
    assert np.allclose(eta[:3], fix.p_ned) and eta[5] == fix.yaw


def test_navfix_kwargs_are_real_navfix_fields():
    """The dict must be constructible into ``state.NavFix`` — the field list
    is state.py:416-458 and a typo here would only surface on hardware."""
    from rov_gui.state import NavFix

    rng = np.random.default_rng(5)
    fix = NAV.fix_from_pose(_rec(_twc(_rand_R(rng), rng.standard_normal(3)),
                                 t=1.0, n_tracked=731))
    kw = fix.as_navfix_kwargs()
    names = {f.name for f in dataclass_fields(NavFix)}
    assert set(kw) <= names, set(kw) - names
    nf = NavFix(**kw)
    assert nf.ok and nf.n_tags == 0
    # NOT 0.0: window.py:1016 would print a fabricated "0.0 px" (CLAUDE.md).
    assert nf.reproj_rms_px is None
    assert "731 pts" in nf.note
    assert np.allclose(np.asarray(nf.R_ned_body).reshape(3, 3), fix.R_ned_body)


# =============================================================================
# backward — the exact inverses
# =============================================================================
def test_twc_roundtrip():
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(200):
        Twc = _twc(_rand_R(rng), rng.standard_normal(3) * 5.0)
        p, R = NAV.ned_from_twc(Twc)
        back = NAV.twc_from_ned(p, R)
        worst = max(worst, float(np.abs(back - Twc).max()))
    assert worst < 1e-12, worst


def test_knot_roundtrip_through_the_datum():
    """A datum-NED knot -> SLAM -> back is the identity, for (3,) and (3, K)
    and with no datum at all."""
    rng = np.random.default_rng(17)
    datum = EngageDatum(p0=[1.5, -0.4, 0.9], yaw0=2.7)
    knots = rng.standard_normal((3, 16)) * 0.8
    p_slam = NAV.slam_from_datum_p(knots, datum)
    back = datum.datum_from_map_p(NAV.ned_from_slam_p(p_slam))
    assert np.abs(back - knots).max() < TOL, np.abs(back - knots).max()
    one = NAV.slam_from_datum_p(knots[:, 3], datum)
    assert np.abs(one - p_slam[:, 3]).max() < TOL
    # no datum == identity isometry (workers.py:3874-3880's None branch)
    assert np.abs(NAV.slam_from_datum_p(knots)
                  - NAV.slam_from_ned_p(knots)).max() < TOL


def test_datum_eta_roundtrip_and_the_worker_formula():
    """``EngageDatum`` is MpcWorker's isometry: forward ``Rz @ (p - p0)``
    (workers.py:1678-1691), inverse ``p0 + Rz.T @ p`` (workers.py:3874-3880),
    z INCLUDED (the 2026-08-23 plot bug)."""
    rng = np.random.default_rng(19)
    eta_map = np.array([2.0, -1.0, 0.75, 0.05, -0.03, 1.9])
    datum = EngageDatum.from_eta(eta_map)
    d = datum.datum_from_eta(eta_map)
    assert np.abs(d[:3]).max() < TOL and abs(d[5]) < TOL, d
    assert d[3] == eta_map[3] and d[4] == eta_map[4]   # horizontal isometry
    Rz = rot_zyx(0.0, 0.0, -eta_map[5])
    for _ in range(50):
        q = rng.standard_normal(3) * 3.0
        assert np.abs(datum.datum_from_map_p(q) - Rz @ (q - eta_map[:3])).max() < TOL
        assert np.abs(datum.map_from_datum_p(q)
                      - (eta_map[:3] + Rz.T @ q)).max() < TOL
        assert np.abs(datum.map_from_datum_p(datum.datum_from_map_p(q))
                      - q).max() < TOL
    e2 = rng.standard_normal(6)
    e2[5] = wrap_pi(float(e2[5]))
    assert np.abs(datum.map_from_eta(datum.datum_from_eta(e2)) - e2).max() < 1e-12


def test_slam_anchor_is_the_inverse_of_the_fix():
    """The 'KNOT' anchor is the BODY frame in the SLAM world, and running the
    forward step on it returns the eta it came from."""
    datum = EngageDatum(p0=[0.3, 1.2, -0.6], yaw0=-2.9)
    eta_d = np.array([0.4, -0.2, 0.15, 0.02, -0.06, 0.8])
    A = NAV.slam_anchor(eta_d, datum)
    assert A.shape == (3, 4)
    p_map = P_NED_SLAM @ A[:3, 3]
    R_map = P_NED_SLAM @ A[:3, :3]
    eta_map = datum.map_from_eta(eta_d)
    assert np.abs(p_map - eta_map[:3]).max() < TOL
    assert np.abs(R_map - rot_zyx(eta_map[3], eta_map[4], eta_map[5])).max() < TOL
    # and the anchor's translation is the knot path's own image of eta[:3]
    assert np.abs(A[:3, 3] - NAV.slam_from_datum_p(eta_d[:3], datum)).max() < TOL


# =============================================================================
# invariance
# =============================================================================
def _datum_track(nav, Twcs):
    """Fixes for a list of Twc, datum taken at the first one -> datum-frame
    positions (3, N) and the datum object."""
    fixes = [nav.fix_from_pose(_rec(T, t=float(i))) for i, T in enumerate(Twcs)]
    datum = EngageDatum.from_eta(fixes[0].eta6())
    P = np.stack([datum.datum_from_map_p(f.p_ned) for f in fixes], axis=1)
    return P, datum


def test_global_rigid_transform_leaves_datum_knots_unchanged():
    """Re-anchoring the SLAM world (the first keyframe landing elsewhere) must
    not move anything the station navigates in.

    The engage datum is a HORIZONTAL isometry (workers.py:1485-1491: p0 and
    yaw0 only), so the family it can absorb is a SLAM-world transform whose
    NED image is a yaw rotation plus a translation, i.e.
    ``R_g = P.T Rz(a) P``. That is exactly the freedom a stereo re-init has if
    the camera was level; a world TILT is a different gravity direction, which
    the rig genuinely cannot absorb — asserted at the end so this test cannot
    be read as "invariant to everything".
    """
    rng = np.random.default_rng(23)
    Twcs = [_twc(_rand_R(rng), rng.standard_normal(3)) for _ in range(8)]
    base, _ = _datum_track(NAV, Twcs)

    for a in (0.0, 0.7, -2.4, math.pi):
        R_g = P_NED_SLAM.T @ rot_zyx(0.0, 0.0, a) @ P_NED_SLAM
        t_g = rng.standard_normal(3) * 4.0
        moved = [_twc(R_g @ T[:3, :3], R_g @ T[:3, 3] + t_g) for T in Twcs]
        got, datum_g = _datum_track(NAV, moved)
        assert np.abs(got - base).max() < 1e-11, (a, np.abs(got - base).max())
        # and the knot return path moves WITH the world by exactly T_g
        k = rng.standard_normal((3, 5))
        _, datum_0 = _datum_track(NAV, Twcs)
        s0 = NAV.slam_from_datum_p(k, datum_0)
        sg = NAV.slam_from_datum_p(k, datum_g)
        assert np.abs(sg - (R_g @ s0 + t_g[:, None])).max() < 1e-11

    # counter-check: a world tilt is NOT absorbed (it changes roll/pitch, and
    # the datum has no roll/pitch to cancel it with).
    R_tilt = P_NED_SLAM.T @ rot_zyx(0.0, 0.25, 0.0) @ P_NED_SLAM
    tilted = [_twc(R_tilt @ T[:3, :3], R_tilt @ T[:3, 3]) for T in Twcs]
    got_t, _ = _datum_track(NAV, tilted)
    assert np.abs(got_t - base).max() > 1e-3


# =============================================================================
# euler / wrapping
# =============================================================================
def test_rpy_agrees_with_rot_zyx():
    rng = np.random.default_rng(29)
    worst = 0.0
    for _ in range(500):
        roll = rng.uniform(-math.pi, math.pi)
        pitch = rng.uniform(-1.4, 1.4)              # away from the singularity
        yaw = rng.uniform(-math.pi, math.pi)
        r, p, y = rpy_from_R(rot_zyx(roll, pitch, yaw))
        worst = max(worst, abs(wrap_pi(r - roll)), abs(p - pitch),
                    abs(wrap_pi(y - yaw)))
    assert worst < 1e-9, worst
    # and on a random rotation the extraction reconstructs the matrix exactly
    for _ in range(200):
        R = _rand_R(rng)
        assert np.abs(rot_zyx(*rpy_from_R(R)) - R).max() < 1e-9


def test_angle_wrapping_at_the_seam():
    """What the atan2 wrap actually guarantees: the result is the SAME angle
    and its magnitude is at most pi. The open/closed end of "(-pi, pi]" is a
    statement about the maths, not about binary floating point — ``sin(pi)``
    is 1.2e-16, not 0, so the sign exactly ON the seam follows that residue
    and ``wrap_pi(-pi)`` comes back as -pi. Asserting +pi there would pin a
    coincidence of the libm, and the consumers (a yaw error, a heading) only
    ever care about the two properties below."""
    for a in (-8.0, -math.pi - 1e-9, -math.pi, -3.0, -1e-12, 0.0, 3.0,
              math.pi, math.pi + 1e-9, 3 * math.pi, 8.0, 100.0):
        w = wrap_pi(a)
        assert abs(w) <= math.pi + 1e-12, (a, w)
        assert abs(wrap_pi(w - a)) < 1e-9, (a, w)        # same angle mod 2pi
    for a in (math.pi, -math.pi, 3 * math.pi):
        assert abs(abs(wrap_pi(a)) - math.pi) < 1e-12    # lands ON the seam
    # just past +pi flips to just past -pi, which is the property that matters
    assert abs(wrap_pi(math.pi + 0.01) - (-math.pi + 0.01)) < 1e-12
    assert abs(wrap_pi(-math.pi - 0.01) - (math.pi - 0.01)) < 1e-12
    # the datum crossing must take the SHORT way across the seam
    d = EngageDatum(p0=[0, 0, 0], yaw0=math.pi - 0.05)
    assert abs(d.datum_from_map_yaw(-math.pi + 0.05) - 0.1) < 1e-12
    for yaw in (-math.pi + 1e-9, -3.0, 0.0, 3.0, math.pi):
        assert abs(wrap_pi(d.map_from_datum_yaw(d.datum_from_map_yaw(yaw))
                           - yaw)) < 1e-12


# =============================================================================
# health
# =============================================================================
def _feed(mon, specs):
    """specs = [(state, dt, step_m)] -> the events, walking a straight line."""
    out, t, p = [], 0.0, np.zeros(3)
    for state, dt, step in specs:
        t += dt
        p = p + np.array([step, 0.0, 0.0])
        out.append(mon.update(_rec(_twc(np.eye(3), p), t=t, state=state)))
    return out


def test_session_counter_one_per_cycle():
    mon = TrackMonitor()
    ev = _feed(mon, [(2, 0.033, 0.01)] * 3
                    + [(4, 0.033, 0.0), (0, 0.033, 0.0), (0, 0.033, 0.0)]
                    + [(2, 0.033, 0.01)] * 3
                    + [(3, 0.033, 0.0)]                    # RECENTLY_LOST
                    + [(2, 0.033, 0.01)] * 2)
    assert [e.status for e in ev[:3]] == [TrackStatus.OK] * 3
    assert [e.session for e in ev] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert [i for i, e in enumerate(ev) if e.session_started] == [6, 10]
    assert [e.status for e in ev[3:6]] == [TrackStatus.LOST] * 3
    assert mon.n_lost == 4 and mon.n_ok == 8
    assert mon.summary()["sessions"] == 3
    # a session start is an END OF RUN, and it clears the drawn overlay
    assert ev[6].clear_overlay and not ev[7].clear_overlay


def test_session_counter_quiet_during_a_steady_loss():
    mon = TrackMonitor()
    ev = _feed(mon, [(2, 0.033, 0.01)] + [(4, 0.033, 0.0)] * 50)
    assert all(e.session == 0 for e in ev)
    assert not any(e.session_started for e in ev)
    assert mon.session == 0
    # ...and the very first records being LOST does not start a session either
    mon2 = TrackMonitor()
    ev2 = _feed(mon2, [(0, 0.033, 0.0)] * 5 + [(2, 0.033, 0.0)] * 2)
    assert mon2.session == 0 and not any(e.session_started for e in ev2)


def test_map_changed_is_ok_but_stale():
    """A LoopClosing big change deforms the map: the pose is fine, the drawn
    overlay is not, and it is NOT a new session."""
    mon = TrackMonitor()
    a = mon.update(_rec(_twc(np.eye(3), [0, 0, 0]), t=0.0))
    b = mon.update(_rec(_twc(np.eye(3), [0.01, 0, 0]), t=0.033, map_changed=True))
    assert a.status is TrackStatus.OK and b.status is TrackStatus.MAP_CHANGED
    assert b.ok and b.clear_overlay and not b.session_started
    assert b.session == 0 and mon.n_map_changed == 1


def test_jump_guard():
    """Fires on a teleport, silent on fast-but-plausible handheld motion,
    suppressed across a gap — the plotter's two tests."""
    mon = TrackMonitor()
    dt = 1.0 / 30.0
    fast = 1.8 * dt                        # 1.8 m/s, under the 3.0 default
    ev = _feed(mon, [(2, dt, 0.0)] + [(2, dt, fast)] * 10)
    assert not any(e.jump for e in ev), [e.speed_mps for e in ev]
    assert abs(ev[-1].speed_mps - 1.8) < 1e-9

    mon = TrackMonitor()
    ev = _feed(mon, [(2, dt, 0.0), (2, dt, fast), (2, dt, 5.0), (2, dt, fast)])
    assert [e.jump for e in ev] == [False, False, True, False]
    assert ev[2].speed_mps > 100.0 and "jump" in ev[2].note
    assert mon.n_jumps == 1
    assert ev[2].ok, "a jump flags the record, it does not disengage"

    # a GAP implies no speed at all (plot_trajectory.py:123-124)
    mon = TrackMonitor()
    ev = _feed(mon, [(2, dt, 0.0), (2, DEFAULT_GAP_S + 0.3, 4.0)])
    assert ev[1].speed_mps > DEFAULT_MAX_SPEED_MPS and not ev[1].jump

    # no speed is measured across a loss, or across a session boundary
    mon = TrackMonitor()
    ev = _feed(mon, [(2, dt, 0.0), (2, dt, 0.01), (4, dt, 0.0), (2, dt, 9.0)])
    assert ev[3].session_started and ev[3].speed_mps is None and not ev[3].jump

    # a non-advancing stamp is reported, not divided by
    mon = TrackMonitor()
    ev = _feed(mon, [(2, dt, 0.0), (2, 0.0, 0.01)])
    assert ev[1].speed_mps is None and "did not advance" in ev[1].note


def test_degenerate_poses():
    """The three shapes a bad 48-byte block takes, and the dangerous one:
    garbage arriving WITH state == OK."""
    good = _twc(np.eye(3), [1.0, 2.0, 3.0])
    assert check_twc(good) == ""

    bad_R = good.copy()
    bad_R[:3, :3] = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert "orthonormal" in check_twc(bad_R)

    nan = good.copy()
    nan[1, 3] = float("nan")
    assert "non-finite" in check_twc(nan)
    inf = good.copy()
    inf[0, 0] = float("inf")
    assert "non-finite" in check_twc(inf)

    zeros = np.zeros((3, 4))
    assert "orthonormal" in check_twc(zeros)      # an uninitialised buffer

    mirror = good.copy()
    mirror[:3, :3] = np.diag([1.0, 1.0, -1.0])
    assert "reflection" in check_twc(mirror)

    for bad in (bad_R, nan, zeros, mirror):
        try:
            NAV.ned_from_twc(bad)
            raise AssertionError("ned_from_twc accepted a bad pose")
        except ValueError:
            pass
        try:
            NAV.fix_from_pose(_rec(bad))
            raise AssertionError("fix_from_pose accepted a bad pose")
        except ValueError:
            pass

    # the whole point: state says OK, the pose is garbage -> LOST, with the
    # reason, and the recovery afterwards starts a NEW session.
    mon = TrackMonitor()
    mon.update(_rec(good, t=0.0))
    e = mon.update(_rec(zeros, t=0.033, state=TRACK_STATE_OK))
    assert e.status is TrackStatus.LOST and "orthonormal" in e.note
    e2 = mon.update(_rec(good, t=0.066))
    assert e2.session_started and e2.session == 1

    # f32 round-off must NOT be called non-orthonormal (the wire is f32)
    rng = np.random.default_rng(31)
    for _ in range(50):
        R = _rand_R(rng)
        assert check_twc(_twc(R, rng.standard_normal(3)).astype(np.float32)) == ""

    # a wrong SHAPE is a programmer error, not a stream error
    try:
        check_twc(np.zeros((2, 4)))
        raise AssertionError("check_twc accepted a (2, 4)")
    except ValueError:
        pass


# =============================================================================
# runner (same shape as test_policy_frames.py — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
