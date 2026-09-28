#!/usr/bin/env python3
"""test_plan_stream.py — the streamed reference-plan seam (filter + stitcher
+ replay helpers), pure numpy, no Qt, no acados, no pytest dependency
(the `robust` env has none; the plain-assert functions still collect under
pytest where it exists, e.g. rovgui-pose).

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_plan_stream.py

What is pinned here:
  * every PlanFilter gate fires with a numeric margin and a named reason —
    the workspace box is the ONLY position-based protection since the
    geofence was removed, and the speed cap is what stands between a bad
    plan and the 2026-08 unbounded-feedforward runaway;
  * a minor speed violation is repaired by time dilation, not rejected —
    same path points, longer clock;
  * the stitched reference is C0 at every hand-over and its velocity is the
    TRUE derivative of its position (cross term w_dot*(p_new - p_old),
    verified against a finite difference, not against the formula);
  * yaw blends the short way across +/-pi;
  * an adversarial 1 Hz plan oscillation cannot push the reference speed
    past the analytic bound v_max + jump_max_m / blend_s (parameters chosen
    inside the cosine-ramp validity region: peak w_dot is pi/(2*blend_s), so
    the bound needs max overlap deviation <= (2/pi) * jump_max_m);
  * a synthetic recorded session round-trips through load_replay_track
    (body-frame anchor, NaN dropping, still trim, dilation), and the chopped
    1 Hz feed M0(b) reproduces the single-plan reference M0(a) through the
    stitcher;
  * (2026-09-02, live policy) `require_obs_t` rejects a stampless plan, a
    stamp AHEAD of now is rejected as a clock-domain error (an unconverted
    monotonic obs_t, v2 A1), the optional yaw-jump gate fires with a margin
    (v2 A11), `margins["need"]` is always written, `reject_external` counts
    toward escalation like any other reject, and `preview_install` reports
    the blend's own cross-term peak without installing anything;
  * (2026-09-26, pos_rpy_width) the attitude channel `PlanMsg.rp`: schema /
    rp_mag HARD reject, rp_jump via `cur_sample_att`, rp_rate joining `need`
    so the one dilation repairs it, `sample_att` C0 + cross term (finite
    difference), `sample()` still a 5-tuple, a plan without rp holding the
    previous attitude / None when never carried, preview `rp_rate_peak`
    only for an rp-carrying candidate, `load_replay_track(with_rp=True)`,
    and time_dilate / anchor_track / chop_track carrying rp — while a 4-DoF
    message yields the IDENTICAL margins dict, and every 4-DoF path's bytes
    match golden sha256 pins recorded on the pre-variant code.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control.plan_stream import (  # noqa: E402
    PLAN_STREAM_SCHEMA, FilterLimits, PlanFilter, PlanMsg, PlanStitcher,
    ReplayTrack, anchor_track, chop_track, load_replay_track, time_dilate)


# --------------------------------------------------------------------- utils
def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


@contextlib.contextmanager
def _raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None:
            assert match in str(e), f"{match!r} not in {e!r}"
    else:
        raise AssertionError(f"{exc.__name__} was not raised")


def _line_plan(pid, t0, dt, p_start, v, K, yaw=0.0, **kw):
    """Constant-velocity plan: the workhorse fixture."""
    tt = np.arange(K) * dt
    p = (np.asarray(p_start, float)[:, None]
         + np.outer(np.asarray(v, float), tt))
    return PlanMsg(plan_id=pid, t0=t0, dt=dt, p_ned=p,
                   yaw=np.full(K, float(yaw)), **kw)


ORIGIN = (np.zeros(3), 0.0)


def _knot_speed(msg: PlanMsg) -> float:
    v = np.diff(np.asarray(msg.p_ned, float), axis=1) / msg.dt
    return float(np.linalg.norm(v, axis=0).max())


# ------------------------------------------------------------ filter: gates
def test_schema_gate_rejects_nonfinite():
    filt = PlanFilter(FilterLimits())
    msg = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9)
    bad = np.asarray(msg.p_ned).copy()
    bad[1, 4] = np.nan
    msg = PlanMsg(1, 0.0, 0.25, bad, np.asarray(msg.yaw))
    v = filt.evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and v.plan is None
    assert any("schema" in r for r in v.reasons)
    assert PLAN_STREAM_SCHEMA == 1


def test_anchor_gate_fires_with_margin():
    filt = PlanFilter(FilterLimits())
    msg = _line_plan(1, 0.0, 0.25, [1.0, 0, 0], [0, 0, 0], 9)
    v = filt.evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and v.plan is None
    assert any("anchor" in r for r in v.reasons)
    assert _close(v.margins["anchor_m"], 0.30 - 1.0)


def test_stale_obs_rejects():
    filt = PlanFilter(FilterLimits())
    msg = _line_plan(1, 10.0, 0.25, [0, 0, 0], [0, 0, 0], 9, obs_t=8.0)
    v = filt.evaluate(msg, r_now=ORIGIN, now=10.0)
    assert v.status == "reject"
    assert any("stale" in r for r in v.reasons)
    assert _close(v.margins["obs_age"], 0.7 - 2.0)


def test_overlap_jump_gate_catches_divergent_plan():
    """A plan that passes the first-knot anchor but diverges from the
    current reference mid-window must be caught by the overlap gate."""
    st = PlanStitcher(blend_s=0.4)
    st.install(_line_plan(1, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17), now=0.0)
    filt = PlanFilter(FilterLimits())
    # Starts exactly on the reference (anchor margin is FULL) then veers off
    # at 0.15 m/s while the active reference stays put: 0.6 m apart by t=4.
    div = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0, 0.15, 0], 17)
    v = filt.evaluate(div, r_now=ORIGIN, cur_sample=st.sample, now=0.0)
    assert v.status == "reject"
    assert any("jump" in r for r in v.reasons)
    assert _close(v.margins["anchor_m"], 0.30)             # passed
    assert _close(v.margins["jump_m"], 0.20 - 0.60)


def test_workspace_box_rejects_any_knot_outside():
    lim = FilterLimits(box_ned_min=(-0.5, -0.5, -0.5),
                       box_ned_max=(0.5, 0.5, 0.5))
    filt = PlanFilter(lim)
    # Slow (0.1 m/s, passes kinematics) but dives to z = 0.8 — outside.
    msg = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0, 0, 0.1], 33)
    v = filt.evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject"
    assert any("workspace box" in r for r in v.reasons)
    assert _close(v.margins["box_m"], -0.3)


def test_major_speed_violation_rejects():
    filt = PlanFilter(FilterLimits())
    msg = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.40, 0, 0], 9)  # 2.67x cap
    v = filt.evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and v.plan is None
    assert any("speed" in r for r in v.reasons)
    assert _close(v.margins["speed"], 0.15 - 0.40)


# ------------------------------------------------------------- filter: clip
def test_minor_speed_violation_dilates_not_rejects():
    filt = PlanFilter(FilterLimits())
    msg = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.18, 0, 0], 9)   # 1.2x cap
    v = filt.evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "clip" and v.plan is not None
    assert _close(v.margins["dilation"], 0.18 / 0.15)
    # Geometry preserved bit-for-bit, only the clock stretched.
    assert np.array_equal(np.asarray(v.plan.p_ned), np.asarray(msg.p_ned))
    assert _close(v.plan.dt, 0.25 * 1.2)
    assert v.plan.t_end > msg.t_end
    assert _knot_speed(v.plan) <= 0.15 + 1e-9
    assert v.consec_rejects == 0 and not v.escalate


# -------------------------------------------------- filter: reject counting
def test_clip_ratio_max_is_a_limit_not_a_constant():
    """The dilation band is per-limits (policy.clip_ratio_max), not the
    module constant: a plan needing 2.5x is REJECTED under the replay band
    (1.5) and CLIPPED under a 3.0 band with the SAME geometry and a 2.5x
    slower clock — the caps still bound the vehicle either way."""
    K, dt = 6, 0.2
    msg = _line_plan(1, 0.0, dt, (0, 0, 0), (0.25, 0, 0), K)      # 0.25 m/s
    lim = FilterLimits(v_max=0.10, a_max=10.0, r_max=10.0)        # need 2.5
    assert FilterLimits().clip_ratio_max == 1.5
    v = PlanFilter(lim).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and "clip band" in v.reasons[0], v
    lim3 = FilterLimits(v_max=0.10, a_max=10.0, r_max=10.0, clip_ratio_max=3.0)
    v3 = PlanFilter(lim3).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v3.status == "clip", v3
    assert _close(v3.margins["dilation"], 2.5, 1e-9)
    assert _close(v3.plan.dt, dt * 2.5, 1e-12)
    assert np.allclose(v3.plan.p_ned, msg.p_ned)                 # geometry kept
    assert _knot_speed(v3.plan) <= 0.10 + 1e-9


def test_consecutive_rejects_escalate_and_reset():
    filt = PlanFilter(FilterLimits())            # reject_escalate = 3
    far = _line_plan(1, 0.0, 0.25, [5.0, 0, 0], [0, 0, 0], 9)
    good = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9)
    for want, esc in ((1, False), (2, False), (3, True)):
        v = filt.evaluate(far, r_now=ORIGIN, now=0.0)
        assert (v.consec_rejects, v.escalate) == (want, esc)
    filt.reset()
    v = filt.evaluate(far, r_now=ORIGIN, now=0.0)
    assert (v.consec_rejects, v.escalate) == (1, False)
    v = filt.evaluate(good, r_now=ORIGIN, now=0.0)
    assert v.status == "accept" and v.consec_rejects == 0
    v = filt.evaluate(far, r_now=ORIGIN, now=0.0)
    assert v.consec_rejects == 1                 # accept cleared the streak


# ----------------------------------------------------------------- stitcher
def test_stitcher_c0_continuity_and_cross_term():
    st = PlanStitcher(blend_s=0.4)
    A = _line_plan(1, 0.0, 0.5, [0, 0, 0], [0.10, 0, 0], 21)     # 10 s
    st.install(A, now=0.0)
    p_old, _, v_old, _, _ = (x.copy() for x in st.sample(np.array([5.0])))
    B = _line_plan(2, 5.0, 0.5, [0.5, 0.12, 0], [0.10, 0, 0], 21, yaw=0.3)
    st.install(B, now=5.0)
    assert st.active_plan_id() == 2
    # C0 at blend start: blended ref equals the OLD reference...
    p0, _, v0, _, _ = st.sample(np.array([5.0]))
    assert np.allclose(p0, p_old, atol=1e-12)
    assert np.allclose(v0, v_old, atol=1e-12)    # C1 too: w_dot(0) = 0
    # ...and the NEW plan at blend end.
    pe = st.sample(np.array([5.4]))[0]
    p_b = np.array([0.5 + 0.10 * 0.4, 0.12, 0.0])[:, None]
    assert np.allclose(pe, p_b, atol=1e-12)
    assert st.source_at(5.2) == "blend"
    assert st.source_at(5.6) == "plan"
    # v must be the actual derivative of p, cross term included. The naive
    # (1-w)v_old + w*v_new misses a 0.47 m/s y-term at mid-blend, so a 1e-4
    # agreement with the finite difference proves the cross term is there.
    h = 1e-3
    ts = np.arange(5.0 + 2 * h, 5.4 - 2 * h, h)
    _, _, v, _, _ = st.sample(ts)
    p_plus = st.sample(ts + h)[0]
    p_minus = st.sample(ts - h)[0]
    v_fd = (p_plus - p_minus) / (2 * h)
    assert np.max(np.abs(v - v_fd)) < 1e-4
    assert np.max(np.abs(v[1])) > 0.4            # the cross term itself


def test_stitcher_endpoint_hold_and_source_transitions():
    st = PlanStitcher(blend_s=0.4)
    assert not st.has_plan()
    assert st.source_at(0.0) == "none"
    with _raises(RuntimeError):
        st.end_time()
    with _raises(RuntimeError):
        st.sample(np.array([0.0]))
    A = _line_plan(7, 0.0, 0.5, [0, 0, 0], [0.10, 0, 0], 21, yaw=0.2)
    st.install(A, now=0.0)
    assert st.has_plan() and st.active_plan_id() == 7
    assert _close(st.end_time(), 10.0)
    assert st.source_at(9.9) == "plan"
    assert st.source_at(10.5) == "hold"
    p, yaw, v, r, g = st.sample(np.array([11.0, 25.0]))
    assert np.allclose(p, np.array([[1.0], [0.0], [0.0]]), atol=1e-12)
    assert np.all(v == 0.0) and np.all(r == 0.0)
    assert np.allclose(yaw, 0.2)
    st.clear()
    assert not st.has_plan() and st.source_at(0.0) == "none"


def test_stitcher_plan_without_g_holds_previous_g():
    st = PlanStitcher(blend_s=0.4)
    A = _line_plan(1, 0.0, 0.5, [0, 0, 0], [0, 0, 0], 9,
                   g=np.linspace(0.0, 1.0, 9))
    st.install(A, now=0.0)
    B = _line_plan(2, 2.0, 0.5, [0, 0, 0], [0, 0, 0], 9)     # no g
    st.install(B, now=2.0)
    g = st.sample(np.array([3.0]))[4]            # past the blend
    assert _close(g[0], 1.0)                     # held A's final width


def test_yaw_blends_the_short_way_across_pi():
    st = PlanStitcher(blend_s=0.4)
    A = _line_plan(1, 0.0, 0.5, [0, 0, 0], [0, 0, 0], 9, yaw=+3.1)
    st.install(A, now=0.0)
    B = _line_plan(2, 1.0, 0.5, [0, 0, 0], [0, 0, 0], 9, yaw=-3.1)
    st.install(B, now=1.0)
    ts = np.arange(1.0, 1.4001, 1e-3)
    _, yaw, _, r, _ = st.sample(ts)
    short_way = 2 * math.pi - 6.2                # 0.083 rad through +/-pi
    assert _close(abs(yaw[-1] - yaw[0]), short_way, tol=1e-6)
    # A long-way blend would need |r| up to 2*pi * pi/(2*0.4) ~ 24 rad/s;
    # the short way peaks at 0.083 * pi/(2*0.4) ~ 0.33.
    assert np.max(np.abs(r)) < 1.0
    assert np.max(np.abs(np.diff(yaw))) < 0.01   # no 2*pi step anywhere


def test_oscillating_plans_cannot_exceed_analytic_speed_bound():
    """Adversarial source: two mirrored plans alternating at 1 Hz, each one
    individually ACCEPTED by the filter. The blended reference speed must
    stay below v_max + jump_max_m / blend_s. (The mirror offset 0.05 m keeps
    the 0.1 m overlap deviation under (2/pi)*jump_max_m = 0.127 m, which is
    the region where the cosine ramp's peak w_dot = pi/(2*blend_s) keeps the
    cross term under jump_max_m / blend_s.)"""
    lim = FilterLimits()
    filt = PlanFilter(lim)
    st = PlanStitcher(blend_s=0.4)
    vx = 0.10
    v_seen = 0.0
    for k in range(7):
        now = float(k)
        y = 0.05 if k % 2 == 0 else -0.05
        msg = _line_plan(k, now, 0.25, [vx * now, y, 0.0], [vx, 0, 0], 17)
        if st.has_plan():
            pr, yr = st.sample(np.array([now]))[:2]
            verdict = filt.evaluate(msg, r_now=(pr[:, 0], float(yr[0])),
                                    cur_sample=st.sample, now=now)
        else:
            verdict = filt.evaluate(msg, r_now=(np.array([0.0, y, 0.0]), 0.0),
                                    now=now)
        assert verdict.status == "accept", verdict.reasons
        st.install(verdict.plan, now=now)
        ts = np.arange(now, now + 1.0, 0.002)
        v = st.sample(ts)[2]
        v_seen = max(v_seen, float(np.linalg.norm(v, axis=0).max()))
    bound = lim.v_max + lim.jump_max_m / st.blend_s
    assert v_seen <= bound + 1e-9, f"{v_seen:.3f} > bound {bound:.3f}"
    assert v_seen > lim.v_max                    # the hand-overs do add speed


# ---------------------------------------------------------- replay: loading
def _quat_zyx_wxyz(roll, pitch, yaw):
    """(T,4) w-first quaternion of rot_z(yaw) rot_y(pitch) rot_x(roll)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr], axis=1)


def _write_session(sdir: Path, T=240, hz=20.0, nan_rows=(50, 51, 120),
                   schema="umi_handheld_poses/1", rp_wobble=None):
    """Synthetic 12 s handheld session: still 0-2 s, smoothstep motion
    2-9 s, still 9-12 s. Pure-yaw quaternions so yaw extraction is exact;
    with ``rp_wobble=(roll_amp, pitch_amp)`` [rad] the quaternions carry a
    roll/pitch wobble on top (returns rp (2, T) as a 5th value then)."""
    t = np.arange(T) / hz
    s = np.clip((t - 2.0) / 7.0, 0.0, 1.0)
    f = 3 * s ** 2 - 2 * s ** 3                  # C1 ramp 0 -> 1
    base = np.array([3.0, -1.0, 0.7])
    p = np.vstack([base[0] + 0.6 * f,
                   base[1] + 0.3 * np.sin(math.pi * f),
                   base[2] + 0.1 * f])
    yaw = 2.0 + 0.4 * np.sin(math.pi * f)
    if rp_wobble is None:
        q = np.zeros((T, 4))
        q[:, 0] = np.cos(yaw / 2)
        q[:, 3] = np.sin(yaw / 2)
        rp = None
    else:
        roll = 0.05 + rp_wobble[0] * np.sin(1.1 * t)
        pitch = -0.03 + rp_wobble[1] * np.sin(0.7 * t + 0.4)
        q = _quat_zyx_wxyz(roll, pitch, yaw)
        rp = np.vstack([roll, pitch])
    g = np.clip(0.5 + 0.5 * np.sin(0.8 * t), 0.0, 1.0)
    rows = np.hstack([(1.7e9 + t)[:, None], p.T, q]).astype(np.float64)
    rows[list(nan_rows), :] = np.nan
    sdir.mkdir(parents=True, exist_ok=True)
    np.save(sdir / "poses.npy", rows)
    with (sdir / "poses.json").open("w") as fh:
        json.dump({"schema": schema, "source": "synthetic-test"}, fh)
    with (sdir / "frames.csv").open("w") as fh:
        fh.write("idx,t_frame\n")
        for i in range(T):
            fh.write(f"{i},{1.7e9 + t[i]:.6f}\n")
    np.save(sdir / "gripper_width.npy",
            g.astype(np.float32).reshape(-1, 1))
    if rp is not None:
        return t, p, yaw, g, rp
    return t, p, yaw, g


def test_replay_load_body_frame_and_nan_drop():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "sess"
        t, p_gt, yaw_gt, g_gt = _write_session(sdir)
        tr = load_replay_track(str(sdir), trim_still=False)
        assert tr.meta["n_raw"] == 240 and tr.meta["n_used"] == 237
        assert tr.meta["poses_json"]["source"] == "synthetic-test"
        # First retained pose maps exactly to origin / zero yaw.
        assert np.allclose(tr.p[:, 0], 0.0, atol=1e-12)
        assert tr.yaw[0] == 0.0
        assert tr.t[0] == 0.0
        # Geometry matches the ground-truth relative track (smoothing bias
        # for this signal is < 1 mm; 1 cm tolerance is generous).
        keep = np.ones(240, bool)
        keep[[50, 51, 120]] = False
        c, s = math.cos(-yaw_gt[0]), math.sin(-yaw_gt[0])
        Rm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p_exp = Rm @ (p_gt[:, keep] - p_gt[:, :1])
        assert np.max(np.abs(tr.p - p_exp)) < 0.01
        assert np.max(np.abs(tr.yaw - (yaw_gt[keep] - yaw_gt[0]))) < 0.01
        assert tr.g is not None and tr.g.shape == tr.t.shape
        assert np.max(np.abs(tr.g - g_gt[keep])) < 1e-6   # g is not smoothed


def test_replay_trim_still_head_and_tail():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "sess"
        _write_session(sdir)
        full = load_replay_track(str(sdir), trim_still=False)
        tr = load_replay_track(str(sdir), trim_still=True)
        assert tr.meta["duration_s"] < full.meta["duration_s"] - 3.0
        assert 5.5 < tr.meta["duration_s"] < 8.0  # ~7 s of actual motion
        # Re-anchored to the first RETAINED pose: the invariants hold.
        assert tr.t[0] == 0.0
        assert np.allclose(tr.p[:, 0], 0.0, atol=1e-12)
        assert tr.yaw[0] == 0.0


def test_replay_rejects_wrong_schema_and_too_few_fixes():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "bad_schema"
        _write_session(sdir, schema="bogus/9")
        with _raises(ValueError, match="schema"):
            load_replay_track(str(sdir))
        sdir2 = Path(td) / "few_fixes"
        _write_session(sdir2, nan_rows=tuple(range(5, 240)))  # 5 fixes left
        with _raises(ValueError, match="valid pose rows"):
            load_replay_track(str(sdir2))


def test_replay_time_dilation_slows_clock_not_geometry():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "sess"
        _write_session(sdir)
        tr = load_replay_track(str(sdir))
        slow, alpha = time_dilate(tr, v_max=0.05)
        assert alpha > 1.5                        # demo peaks near 0.24 m/s
        assert np.allclose(slow.t, tr.t * alpha)
        assert np.array_equal(slow.p, tr.p)
        assert np.array_equal(slow.yaw, tr.yaw)
        assert slow.meta["time_dilation_alpha"] == alpha
        same, a1 = time_dilate(tr, v_max=10.0)
        assert a1 == 1.0 and np.array_equal(same.t, tr.t)


# ---------------------------------------------------- replay: plan emission
def _synthetic_track(dur=10.0, dt=0.05):
    tt = np.arange(0.0, dur + dt / 2, dt)
    p = np.vstack([0.8 * np.sin(0.5 * tt),
                   0.4 * (1 - np.cos(0.4 * tt)),
                   0.1 * np.sin(0.3 * tt)])
    p = p - p[:, :1]
    yaw = 0.3 * np.sin(0.4 * tt)
    g = np.clip(0.5 + 0.4 * np.sin(0.5 * tt), 0.0, 1.0)
    return ReplayTrack(t=tt, p=p, yaw=yaw - yaw[0], g=g, meta={})


def test_anchor_track_places_body_frame_at_pose():
    tt = np.linspace(0.0, 4.0, 81)
    track = ReplayTrack(t=tt, p=np.vstack([0.1 * tt, 0 * tt, 0 * tt]),
                        yaw=np.zeros_like(tt), g=None, meta={})
    p0 = np.array([1.0, 2.0, 0.5])
    msg = anchor_track(track, p0, math.pi / 2, t0=50.0, dt=0.25, plan_id=3)
    assert msg.plan_id == 3 and msg.t0 == 50.0 and msg.dt == 0.25
    assert msg.t_end >= 54.0                     # grid covers the whole demo
    assert np.allclose(np.asarray(msg.p_ned)[:, 0], p0, atol=1e-12)
    assert _close(msg.yaw[0], math.pi / 2)
    # Body +x under yaw0 = pi/2 is world +y (NED): 1 s in => +0.1 m east.
    assert np.allclose(np.asarray(msg.p_ned)[:, 4],
                       p0 + np.array([0.0, 0.1, 0.0]), atol=1e-9)


def test_chopped_stream_reproduces_single_plan_reference():
    """M0(b) ~ M0(a): the 1 Hz chopped feed, installed sequentially, samples
    identically to the one-shot anchored plan — the knot grids align, so old
    and new agree over every overlap and even the blend windows are exact."""
    track = _synthetic_track()
    p0, yaw0, t0 = np.array([2.0, -1.0, 0.5]), 0.7, 100.0
    single = anchor_track(track, p0, yaw0, t0=t0, dt=0.25)
    msgs = chop_track(track, p0, yaw0, t0=t0, horizon_s=4.0,
                      period_s=1.0, dt=0.25)
    assert len(msgs) == 11
    assert [m.plan_id for m in msgs] == list(range(11))
    assert _close(msgs[3].t0, 103.0)

    st_a = PlanStitcher(blend_s=0.4)
    st_a.install(single, now=t0)
    st_b = PlanStitcher(blend_s=0.4)
    for m in msgs:
        st_b.install(m, now=m.t0)
        ts = np.arange(m.t0, m.t0 + 1.0, 0.02)
        pa, ya, va, ra, ga = st_a.sample(ts)
        pb, yb, vb, rb, gb = st_b.sample(ts)
        assert np.allclose(pb, pa, atol=1e-6)
        assert np.allclose(yb, ya, atol=1e-6)
        assert np.allclose(vb, va, atol=1e-6)
        assert np.allclose(gb, ga, atol=1e-6)
    # Terminal hold: both streams park on the demo's endpoint.
    ts = np.arange(110.5, 112.0, 0.02)
    assert np.allclose(st_b.sample(ts)[0], st_a.sample(ts)[0], atol=1e-6)
    assert np.all(st_b.sample(ts)[2] == 0.0)


# ------------------------------------------------- live-policy additions
def test_require_obs_t_rejects_a_stampless_plan_only_when_set():
    msg = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9)  # obs_t None
    v = PlanFilter(FilterLimits()).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "accept"                    # replay: no stamp is fine
    v = PlanFilter(FilterLimits(require_obs_t=True)).evaluate(
        msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and v.plan is None
    assert any("obs_t" in r for r in v.reasons)
    # ...and a stamped plan passes the same limits.
    good = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9, obs_t=-0.1)
    v = PlanFilter(FilterLimits(require_obs_t=True)).evaluate(
        good, r_now=ORIGIN, now=0.0)
    assert v.status == "accept" and v.margins["obs_age"] > 0.0


def test_future_obs_t_is_a_clock_domain_reject():
    """A monotonic-domain stamp handed to the filter unconverted lies far
    AHEAD of the mission clock — the age gate cannot catch it (age is
    negative), so the domain gate must (v2 A1)."""
    from rov_gui.control.plan_stream import OBS_T_FUTURE_TOL_S

    filt = PlanFilter(FilterLimits(require_obs_t=True))
    mono = 12345.678                       # "now()" on the monotonic clock
    msg = _line_plan(1, 3.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9, obs_t=mono)
    v = filt.evaluate(msg, r_now=ORIGIN, now=3.0)
    assert v.status == "reject"
    assert any("clock domain" in r for r in v.reasons), v.reasons
    assert v.margins["obs_age"] > 0.0      # the age gate alone would PASS it
    # Inside the tolerance (stamp/tick jitter) is fine.
    ok = _line_plan(2, 3.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9,
                    obs_t=3.0 + 0.5 * OBS_T_FUTURE_TOL_S)
    assert filt.evaluate(ok, r_now=ORIGIN, now=3.0).status == "accept"


def test_yaw_jump_gate_fires_with_margin_only_when_set():
    st = PlanStitcher(blend_s=0.4)
    st.install(_line_plan(1, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17, yaw=0.0),
               now=0.0)
    # Same position, heading 20 deg away for the whole window.
    turned = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17,
                        yaw=math.radians(20.0))
    v = PlanFilter(FilterLimits()).evaluate(
        turned, r_now=ORIGIN, cur_sample=st.sample, now=0.0)
    assert v.status == "accept" and "yaw_jump" not in v.margins
    lim = FilterLimits(yaw_jump_max_rad=math.radians(8.0))
    v = PlanFilter(lim).evaluate(turned, r_now=ORIGIN, cur_sample=st.sample,
                                 now=0.0)
    assert v.status == "reject"
    assert any("yaw jump" in r for r in v.reasons)
    assert _close(v.margins["yaw_jump"], math.radians(8.0 - 20.0), 1e-9)
    # ...and the wrap: +179 deg vs -179 deg is a 2 deg jump, not 358.
    st2 = PlanStitcher(blend_s=0.4)
    st2.install(_line_plan(1, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17,
                           yaw=math.radians(179.0)), now=0.0)
    near = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17,
                      yaw=math.radians(-179.0))
    v = PlanFilter(lim).evaluate(near, r_now=ORIGIN, cur_sample=st2.sample,
                                 now=0.0)
    assert v.status == "accept", v.reasons
    assert _close(v.margins["yaw_jump"], math.radians(8.0 - 2.0), 1e-6)


def test_need_margin_is_always_written():
    filt = PlanFilter(FilterLimits())
    slow = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9)
    v = filt.evaluate(slow, r_now=ORIGIN, now=0.0)
    assert v.status == "accept" and v.margins["need"] <= 1.0
    fast = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0.18, 0, 0], 9)
    v = filt.evaluate(fast, r_now=ORIGIN, now=0.0)
    assert v.status == "clip"
    assert _close(v.margins["need"], v.margins["dilation"])


def test_reject_external_counts_toward_escalation():
    filt = PlanFilter(FilterLimits())              # reject_escalate = 3
    v1 = filt.reject_external({"blend_v": -0.1}, ["blend: too fast"])
    assert v1.status == "reject" and v1.consec_rejects == 1
    far = _line_plan(1, 0.0, 0.25, [5.0, 0, 0], [0, 0, 0], 9)
    v2 = filt.evaluate(far, r_now=ORIGIN, now=0.0)
    assert v2.consec_rejects == 2 and not v2.escalate
    v3 = filt.reject_external({}, ["blend: too fast"])
    assert v3.consec_rejects == 3 and v3.escalate
    assert v3.margins == {} and v3.reasons == ["blend: too fast"]
    good = _line_plan(2, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9)
    assert filt.evaluate(good, r_now=ORIGIN, now=0.0).consec_rejects == 0


def test_preview_install_reports_blend_peaks_without_installing():
    st = PlanStitcher(blend_s=0.4)
    A = _line_plan(1, 0.0, 0.5, [0, 0, 0], [0.05, 0, 0], 21, yaw=0.0)
    st.install(A, now=0.0)
    # A new plan offset 0.12 m sideways at the same along-track speed: its
    # own knots move at 0.05 m/s, but the cosine blend has to cover 0.12 m
    # in 0.4 s — the cross term peaks at pi/(2*0.4) * 0.12 = 0.47 m/s.
    B = _line_plan(2, 2.0, 0.5, [0.1, 0.12, 0], [0.05, 0, 0], 21, yaw=0.3)
    before = st.sample(np.array([2.0, 2.2, 2.4]))[0].copy()
    peaks = st.preview_install(B, now=2.0)
    assert set(peaks) >= {"v_peak", "a_peak", "r_peak", "n_samples"}
    assert 0.40 < peaks["v_peak"] < 0.50, peaks
    assert peaks["a_peak"] > 1.0                 # the cosine ramp's accel
    assert peaks["r_peak"] > 0.5                 # 0.3 rad through the blend
    assert peaks["n_samples"] == 9               # 0.4 s / 0.05 s + 1
    # Nothing was installed: the reference is unchanged and B is not active.
    assert np.array_equal(st.sample(np.array([2.0, 2.2, 2.4]))[0], before)
    assert st.active_plan_id() == 1
    # A plan that continues the current reference exactly previews at its
    # own knot speed (no cross term).
    C = _line_plan(3, 2.0, 0.5, [0.1, 0.0, 0], [0.05, 0, 0], 21, yaw=0.0)
    calm = st.preview_install(C, now=2.0)
    assert _close(calm["v_peak"], 0.05, 1e-9) and calm["a_peak"] < 1e-9
    # With NO plan installed the preview is the new plan's own kinematics.
    empty = PlanStitcher(blend_s=0.4)
    solo = empty.preview_install(B, now=2.0)
    assert _close(solo["v_peak"], 0.05, 1e-9)
    assert not empty.has_plan()



# ------------------------------------------- attitude channel (2026-09-26)
def _rp_plan(pid, t0, dt, K, roll=0.0, pitch=0.0, rp_rate=(0.0, 0.0), **kw):
    """A stationary plan carrying roll/pitch knots: constant + a ramp."""
    tt = np.arange(K) * dt
    rp = np.vstack([roll + rp_rate[0] * tt, pitch + rp_rate[1] * tt])
    return PlanMsg(plan_id=pid, t0=t0, dt=dt, p_ned=np.zeros((3, K)),
                   yaw=np.zeros(K), rp=rp, **kw)


RP_LIM = dict(rp_reject_rad=math.radians(30.0), rp_rate_max=0.35,
              rp_jump_max_rad=math.radians(5.0))


def test_rp_schema_and_mag_reject_and_margins_identical_without_rp():
    """rp must be a finite (2, K); max |rp| > rp_reject_rad is a HARD reject
    (even with follower_owns_dynamics); and a message WITHOUT rp evaluated
    under attitude limits yields the IDENTICAL margins dict, status and
    reasons as under limits without them -- rp_mag / rp_rate / rp_jump
    never appear on a 4-DoF message."""
    base = _line_plan(1, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9, obs_t=-0.1)
    assert base.rp is None
    lim_att = FilterLimits(**RP_LIM)
    lim_plain = FilterLimits()
    st = PlanStitcher(blend_s=0.4)
    st.install(_line_plan(0, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 9), now=0.0)
    for msg in (base, _line_plan(2, 0.0, 0.25, [0, 0, 0], [0.18, 0, 0], 9, obs_t=-0.1)):
        va = PlanFilter(lim_att).evaluate(msg, r_now=ORIGIN, cur_sample=st.sample,
                                          cur_sample_att=st.sample_att, now=0.0)
        vp = PlanFilter(lim_plain).evaluate(msg, r_now=ORIGIN, cur_sample=st.sample,
                                            now=0.0)
        assert va.status == vp.status and va.reasons == vp.reasons
        assert va.margins == vp.margins, (va.margins, vp.margins)
        assert not ({"rp_mag", "rp_rate", "rp_jump"} & set(va.margins))
        assert (va.plan is None) == (vp.plan is None)
        if va.plan is not None:
            assert va.plan.rp is None and va.plan.dt == vp.plan.dt
    # schema: wrong shape / non-finite
    for bad_rp in (np.zeros((3, 9)), np.zeros((2, 8)), np.full((2, 9), np.nan)):
        bad = PlanMsg(3, 0.0, 0.25, np.zeros((3, 9)), np.zeros(9), rp=bad_rp)
        v = PlanFilter(lim_att).evaluate(bad, r_now=ORIGIN, now=0.0)
        assert v.status == "reject" and any("schema: rp" in r for r in v.reasons), v.reasons
    # rp_mag: HARD, with a margin, even when the follower owns dynamics
    big = _rp_plan(4, 0.0, 0.25, 9, roll=math.radians(35.0))
    for fod in (False, True):
        v = PlanFilter(FilterLimits(follower_owns_dynamics=fod, **RP_LIM)).evaluate(
            big, r_now=ORIGIN, now=0.0)
        assert v.status == "reject" and v.plan is None
        assert any("attitude" in r and "reject limit" in r for r in v.reasons), v.reasons
        assert _close(v.margins["rp_mag"], math.radians(30.0 - 35.0), 1e-12)
    # ...and inside the limit it passes with the margin written
    ok = _rp_plan(5, 0.0, 0.25, 9, roll=math.radians(10.0), pitch=math.radians(-20.0))
    v = PlanFilter(lim_att).evaluate(ok, r_now=ORIGIN, now=0.0)
    assert v.status == "accept" and _close(v.margins["rp_mag"], math.radians(10.0), 1e-12)
    assert v.plan.rp is ok.rp or np.array_equal(v.plan.rp, ok.rp)
    assert "rp_rate" in v.margins and _close(v.margins["rp_rate"], 0.35)
    # no rp limits set: an rp-carrying plan gets no rp margins at all
    v = PlanFilter(lim_plain).evaluate(big, r_now=ORIGIN, now=0.0)
    assert v.status == "accept" and not ({"rp_mag", "rp_rate", "rp_jump"} & set(v.margins))


def test_rp_jump_gate_via_cur_sample_att():
    """rp_jump = max |wrap(rp_new - rp_cur)| over the overlap against the
    CURRENT attitude reference: absent when the limit is unset, when no
    cur_sample_att is given, or when the stitcher never carried attitude;
    a 10 deg roll step vs a 5 deg limit is a _soft failure (recorded and
    waved through under follower_owns_dynamics, rejected otherwise)."""
    st = PlanStitcher(blend_s=0.4)
    st.install(_rp_plan(1, 0.0, 0.25, 17, roll=math.radians(2.0)), now=0.0)
    turned = _rp_plan(2, 0.0, 0.25, 17, roll=math.radians(12.0))
    # limit unset -> no margin
    v = PlanFilter(FilterLimits()).evaluate(turned, r_now=ORIGIN, cur_sample=st.sample,
                                            cur_sample_att=st.sample_att, now=0.0)
    assert v.status == "accept" and "rp_jump" not in v.margins
    # limit set, no cur_sample_att -> no margin
    lim = FilterLimits(**RP_LIM)
    v = PlanFilter(lim).evaluate(turned, r_now=ORIGIN, cur_sample=st.sample, now=0.0)
    assert v.status == "accept" and "rp_jump" not in v.margins
    # limit set + cur_sample_att -> the gate fires
    v = PlanFilter(lim).evaluate(turned, r_now=ORIGIN, cur_sample=st.sample,
                                 cur_sample_att=st.sample_att, now=0.0)
    assert v.status == "reject" and any("attitude jump" in r for r in v.reasons), v.reasons
    assert _close(v.margins["rp_jump"], math.radians(5.0 - 10.0), 1e-9)
    v = PlanFilter(FilterLimits(follower_owns_dynamics=True, **RP_LIM)).evaluate(
        turned, r_now=ORIGIN, cur_sample=st.sample, cur_sample_att=st.sample_att, now=0.0)
    assert v.status == "accept" and any("not enforced" in r and "attitude jump" in r
                                        for r in v.reasons)
    # a stitcher that never carried attitude: sample_att is None -> no gate
    st4 = PlanStitcher(blend_s=0.4)
    st4.install(_line_plan(1, 0.0, 0.25, [0, 0, 0], [0, 0, 0], 17), now=0.0)
    assert st4.sample_att(np.array([0.5])) is None
    v = PlanFilter(lim).evaluate(turned, r_now=ORIGIN, cur_sample=st4.sample,
                                 cur_sample_att=st4.sample_att, now=0.0)
    assert v.status == "accept" and "rp_jump" not in v.margins
    # within the limit: margin positive
    near = _rp_plan(3, 0.0, 0.25, 17, roll=math.radians(4.0))
    v = PlanFilter(lim).evaluate(near, r_now=ORIGIN, cur_sample=st.sample,
                                 cur_sample_att=st.sample_att, now=0.0)
    assert v.status == "accept" and _close(v.margins["rp_jump"], math.radians(3.0), 1e-9)


def test_rp_rate_joins_need_and_is_repaired_by_the_same_dilation():
    """A 0.70 rad/s pitch ramp (-20 -> +20 deg over the 1 s chunk, inside
    the 30 deg reject) under rp_rate_max 0.35 needs 2.0x: rejected under
    the 1.5x replay band, dilated (same rp knots, slower clock) under a 3x
    band -- margins['rp_rate'] after dilation is exactly 0, need ==
    dilation, and the position/yaw margins are untouched."""
    K, dt = 6, 0.2
    msg = _rp_plan(1, 0.0, dt, K, pitch=-0.35, rp_rate=(0.0, 0.70))
    lim = FilterLimits(**RP_LIM)
    v = PlanFilter(lim).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v.status == "reject" and "clip band" in v.reasons[0], v
    assert _close(v.margins["need"], 2.0, 1e-9) and _close(v.margins["rp_rate"], 0.35 - 0.70, 1e-9)
    assert v.margins["rp_mag"] > 0.0                             # the magnitude gate passed
    lim3 = FilterLimits(clip_ratio_max=3.0, **RP_LIM)
    v3 = PlanFilter(lim3).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v3.status == "clip", v3
    assert _close(v3.margins["dilation"], 2.0, 1e-9) and _close(v3.margins["need"], 2.0, 1e-9)
    assert _close(v3.margins["rp_rate"], 0.0, 1e-12)
    assert _close(v3.plan.dt, dt * 2.0, 1e-12)
    assert np.array_equal(v3.plan.rp, msg.rp)                  # knots kept, clock stretched
    assert _close(v3.margins["speed"], lim.v_max) and _close(v3.margins["yaw_rate"], lim.r_max)
    # the rate is the max over BOTH axes
    both = _rp_plan(2, 0.0, dt, K, rp_rate=(-0.42, 0.10))
    vb = PlanFilter(lim3).evaluate(both, r_now=ORIGIN, now=0.0)
    assert vb.status == "clip" and _close(vb.margins["dilation"], 1.2, 1e-9)
    # rp_rate_max None: no rp_rate margin, need unchanged (1.0 for a still plan)
    v0 = PlanFilter(FilterLimits(rp_reject_rad=1.0)).evaluate(msg, r_now=ORIGIN, now=0.0)
    assert v0.status == "accept" and "rp_rate" not in v0.margins and v0.margins["need"] <= 1.0


def test_stitcher_sample_att_c0_and_cross_term_sample_arity_5():
    """sample_att blends attitude with the SAME cosine ramp and its rate is
    the true derivative (cross term), verified against a finite difference;
    sample() keeps its 5-tuple."""
    st = PlanStitcher(blend_s=0.4)
    A = _rp_plan(1, 0.0, 0.5, 21, roll=0.10, pitch=-0.05, rp_rate=(0.02, 0.0))
    st.install(A, now=0.0)
    assert len(st.sample(np.array([1.0]))) == 5
    rp_old, rr_old = (x.copy() for x in st.sample_att(np.array([5.0])))
    assert _close(rp_old[0, 0], 0.10 + 0.02 * 5.0) and _close(rr_old[0, 0], 0.02)
    B = _rp_plan(2, 5.0, 0.5, 21, roll=0.30, pitch=0.10, rp_rate=(0.0, -0.01))
    st.install(B, now=5.0)
    rp0, rr0 = st.sample_att(np.array([5.0]))
    assert np.allclose(rp0, rp_old, atol=1e-12) and np.allclose(rr0, rr_old, atol=1e-12)
    rpe, _ = st.sample_att(np.array([5.4]))
    assert np.allclose(rpe[:, 0], [0.30, 0.10 - 0.01 * 0.4], atol=1e-12)
    h = 1e-3
    ts = np.arange(5.0 + 2 * h, 5.4 - 2 * h, h)
    rp, rr = st.sample_att(ts)
    rp_plus = st.sample_att(ts + h)[0]
    rp_minus = st.sample_att(ts - h)[0]
    rr_fd = (rp_plus - rp_minus) / (2 * h)
    assert np.max(np.abs(rr - rr_fd)) < 1e-4
    assert np.max(np.abs(rr[1])) > 0.3            # the cross term itself (0.15 rad in 0.4 s)
    # endpoint hold: attitude = last knot, rate 0 (to the sin(pi) residue
    # of the finished blend's w_dot, as on the position path)
    rph, rrh = st.sample_att(np.array([30.0]))
    assert np.allclose(rph[:, 0], [0.30, 0.10 - 0.01 * 10.0], atol=1e-12)
    assert np.allclose(rrh, 0.0, atol=1e-12)
    assert len(st.sample(np.array([5.2]))) == 5


def test_stitcher_plan_without_rp_holds_previous_rp_or_none():
    """The g pattern for attitude: a stream that never carried rp samples
    None; once a plan carried it, a later plan WITHOUT rp holds the last
    flown attitude (never a step to level inside the stream); clear()
    forgets it; no plan -> RuntimeError like sample()."""
    st = PlanStitcher(blend_s=0.4)
    with _raises(RuntimeError):
        st.sample_att(np.array([0.0]))
    st.install(_line_plan(1, 0.0, 0.5, [0, 0, 0], [0, 0, 0], 9), now=0.0)
    assert st.sample_att(np.array([1.0, 2.0])) is None
    st.install(_rp_plan(2, 2.0, 0.5, 9, roll=0.10, pitch=-0.05, rp_rate=(0.01, 0.0)), now=2.0)
    # blend from a side WITHOUT rp: the old side holds the new side's rp,
    # so the hand-over starts AT the new plan's attitude (no level step)
    rp_s, rr_s = st.sample_att(np.array([2.0, 2.2]))
    assert np.allclose(rp_s[:, 0], [0.10, -0.05], atol=1e-12)
    assert np.allclose(rr_s[:, 0], [0.01, 0.0], atol=1e-12)
    st.install(_line_plan(3, 7.0, 0.5, [0, 0, 0], [0, 0, 0], 9), now=7.0)   # no rp
    rp_h, rr_h = st.sample_att(np.array([8.0, 20.0]))
    assert np.allclose(rp_h, [[0.10 + 0.01 * 4.0], [-0.05]], atol=1e-12)   # A's last knot, held
    assert np.all(rr_h == 0.0)
    assert st.sample_att(np.array([7.0]))[0][0, 0] == rp_h[0, 0]           # C0 (no blend needed)
    st.clear()
    assert st._last_rp is None
    st.install(_line_plan(4, 0.0, 0.5, [0, 0, 0], [0, 0, 0], 9), now=0.0)
    assert st.sample_att(np.array([1.0])) is None


def test_preview_install_rp_rate_peak_only_for_rp_candidate():
    st = PlanStitcher(blend_s=0.4)
    st.install(_rp_plan(1, 0.0, 0.5, 21, roll=0.0), now=0.0)
    B4 = _line_plan(2, 2.0, 0.5, [0, 0, 0], [0, 0, 0], 21)               # 4-DoF candidate
    pk = st.preview_install(B4, now=2.0)
    assert set(pk) == {"v_peak", "a_peak", "r_peak", "n_samples"}
    B6 = _rp_plan(3, 2.0, 0.5, 21, roll=0.12)                            # 0.12 rad in 0.4 s
    pk6 = st.preview_install(B6, now=2.0)
    assert set(pk6) == {"v_peak", "a_peak", "r_peak", "n_samples", "rp_rate_peak"}
    # cosine ramp peak: pi/(2*0.4) * 0.12 = 0.47 rad/s (the finite difference
    # on a 0.05 s grid lands a little under)
    assert 0.40 < pk6["rp_rate_peak"] < 0.48, pk6
    assert st.active_plan_id() == 1                                       # nothing installed
    # a still candidate that continues the attitude previews at 0
    C6 = _rp_plan(4, 2.0, 0.5, 21, roll=0.0)
    assert st.preview_install(C6, now=2.0)["rp_rate_peak"] < 1e-12
    # with no plan installed the preview is the candidate's own knot rate
    empty = PlanStitcher(blend_s=0.4)
    solo = empty.preview_install(_rp_plan(5, 2.0, 0.5, 21, rp_rate=(0.0, 0.2)), now=2.0)
    assert _close(solo["rp_rate_peak"], 0.2, 1e-9)


def test_replay_load_with_rp_and_default_unchanged():
    """A session with a roll/pitch wobble: the default load ignores it
    (rp None, t/p/yaw/g array_equal to a wobble-free load of the same
    positions); with_rp=True carries (2, T) roll/pitch relative to the
    first retained pose (rp[:, 0] == 0) and leaves t/p/yaw/g IDENTICAL."""
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "wobble"
        t, p_gt, yaw_gt, g_gt, rp_gt = _write_session(sdir, rp_wobble=(0.08, 0.05))
        plain = load_replay_track(str(sdir))
        assert plain.rp is None and plain.meta["with_rp"] is False
        tr = load_replay_track(str(sdir), with_rp=True)
        assert tr.meta["with_rp"] is True
        for f in ("t", "p", "yaw", "g"):
            assert np.array_equal(getattr(tr, f), getattr(plain, f)), f
        assert tr.rp is not None and tr.rp.shape == (2, tr.t.size)
        assert np.allclose(tr.rp[:, 0], 0.0, atol=1e-12)
        # the relative wobble is recovered: the 0.3 s centred moving average
        # (edge window shrinking to one-sided at row 0, which the relative
        # subtraction then carries into every row) biases it by ~12 mrad
        # [유도: this fixture]; 20 mrad is the tolerance, the wobble is 80
        keep = np.ones(240, bool)
        keep[[50, 51, 120]] = False
        full = load_replay_track(str(sdir), trim_still=False, with_rp=True)
        want = rp_gt[:, keep] - rp_gt[:, keep][:, :1]
        assert np.max(np.abs(full.rp - want)) < 0.02, np.max(np.abs(full.rp - want))
        assert np.corrcoef(full.rp[0], want[0])[0, 1] > 0.99
        assert np.max(np.abs(full.rp)) > 0.05                    # the wobble is there
        # the wobble-free session loads identically to before (no rp keys used)
        sdir0 = Path(td) / "plain"
        _write_session(sdir0)
        t0 = load_replay_track(str(sdir0))
        assert t0.rp is None
        t0r = load_replay_track(str(sdir0), with_rp=True)
        assert np.allclose(t0r.rp, 0.0, atol=1e-12)              # pure-yaw session: level
        for f in ("t", "p", "yaw", "g"):
            assert np.array_equal(getattr(t0r, f), getattr(t0, f)), f


def test_time_dilate_anchor_track_chop_track_carry_rp():
    track = _synthetic_track()
    assert track.rp is None
    tt = track.t
    rp = np.vstack([0.05 * np.sin(0.3 * tt), -0.04 * np.cos(0.5 * tt) + 0.04])
    track_rp = ReplayTrack(t=tt, p=track.p, yaw=track.yaw, g=track.g, meta={}, rp=rp)
    # time_dilate copies rp (and leaves a None alone)
    slow, alpha = time_dilate(track_rp, v_max=0.05)
    assert alpha > 1.0 and np.array_equal(slow.rp, rp) and slow.rp is not rp
    assert time_dilate(track, v_max=0.05)[0].rp is None
    # anchor_track: rp forwarded, rp0 offset, None on a track without rp
    p0, yaw0, t0 = np.array([2.0, -1.0, 0.5]), 0.7, 100.0
    m0 = anchor_track(track, p0, yaw0, t0=t0, dt=0.25)
    assert m0.rp is None
    m1 = anchor_track(track_rp, p0, yaw0, t0=t0, dt=0.25)
    assert m1.rp is not None and m1.rp.shape == (2, m1.n_knots)
    assert np.allclose(m1.rp[:, 0], rp[:, 0], atol=1e-12)
    assert np.array_equal(m1.p_ned, m0.p_ned) and np.array_equal(m1.yaw, m0.yaw)
    m2 = anchor_track(track_rp, p0, yaw0, t0=t0, dt=0.25, rp0=(0.1, -0.05))
    assert np.allclose(m2.rp - m1.rp, [[0.1], [-0.05]], atol=1e-12)
    assert np.array_equal(m2.p_ned, m1.p_ned)
    # rp0 on a track WITHOUT rp changes nothing (there is no attitude to place)
    assert anchor_track(track, p0, yaw0, t0=t0, dt=0.25, rp0=(0.1, -0.05)).rp is None
    # chop_track: every window forwards its slice, rp0 applied, None otherwise
    ms0 = chop_track(track, p0, yaw0, t0=t0)
    ms1 = chop_track(track_rp, p0, yaw0, t0=t0, rp0=(0.1, -0.05))
    assert len(ms0) == len(ms1) == 11
    for a, b in zip(ms0, ms1):
        assert a.rp is None and b.rp.shape == (2, b.n_knots)
        assert np.array_equal(a.p_ned, b.p_ned) and np.array_equal(a.yaw, b.yaw)
        t_rel = b.t0 - t0 + np.arange(b.n_knots) * b.dt
        want = np.vstack([np.interp(t_rel, tt, rp[i]) for i in range(2)]) + np.array([[0.1], [-0.05]])
        assert np.allclose(b.rp, want, atol=1e-12)
    # ...and the chopped rp stream reproduces the single-plan attitude
    st_a = PlanStitcher(blend_s=0.4)
    st_a.install(anchor_track(track_rp, p0, yaw0, t0=t0, dt=0.25), now=t0)
    st_b = PlanStitcher(blend_s=0.4)
    for m in chop_track(track_rp, p0, yaw0, t0=t0):
        st_b.install(m, now=m.t0)
        ts = np.arange(m.t0, m.t0 + 1.0, 0.02)
        ra, _ = st_a.sample_att(ts)
        rb, _ = st_b.sample_att(ts)
        assert np.allclose(rb, ra, atol=1e-6)


# ---------------------------------------- 4-DoF byte-identity golden pins
def _sha(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


#: sha256[:16] of the exact output bytes of each 4-DoF path, computed with
#: the SAME fixtures on the code as it stood before the pos_rpy_width change
#: (scratchpad frames/golden_before.json, conda env `robust`, numpy < 2).
#: A mismatch means the variant changed a 4-DoF number. A different BLAS may
#: round a matmul differently -- then re-pin on the pre-variant commit, never
#: on the current one.
_GOLDEN_4DOF = {
    "filter": "d223988aaa6a8ab1",
    "filter_keys": ["accel", "anchor_m", "box_m", "dilation", "jump_m", "need",
                    "obs_age", "speed", "yaw_jump", "yaw_rate"],
    "stitch": "52c7aeccf37783f2",
    "preview": "7155baac3bf26c8e",
    "replay": "11344bacfaf7d488",
}


def test_four_dof_paths_byte_identical_to_pre_variant():
    """The filter's margins/verdict/plan, the stitcher's blended samples,
    preview_install and the replay helpers produce the SAME BYTES as before
    the attitude channel existed (golden sha256 pins recorded on the
    pre-variant code with these exact fixtures)."""
    st = PlanStitcher(blend_s=0.4)
    st.install(_line_plan(1, 0.0, 0.25, [0, 0, 0], [0.05, 0, 0], 17), now=0.0)
    lim = FilterLimits(yaw_jump_max_rad=math.radians(8), box_ned_min=(-2, -2, -2),
                       box_ned_max=(2, 2, 2), follower_owns_dynamics=True)
    msg = _line_plan(2, 0.1, 0.25, [0.005, 0.01, 0], [0.18, 0.02, 0], 17, yaw=0.05,
                     obs_t=0.0, g=np.linspace(0, 1, 17))
    v = PlanFilter(lim).evaluate(msg, r_now=(np.array([0.005, 0, 0]), 0.0),
                                 cur_sample=st.sample, now=0.1)
    assert v.status == "clip" and sorted(v.margins) == _GOLDEN_4DOF["filter_keys"]
    got = _sha([v.margins[k] for k in sorted(v.margins)], v.plan.p_ned, v.plan.yaw, [v.plan.dt])
    assert got == _GOLDEN_4DOF["filter"], (got, "4-DoF filter bytes changed")
    assert v.plan.rp is None
    # the same verdict with attitude limits SET and cur_sample_att given
    va = PlanFilter(FilterLimits(yaw_jump_max_rad=math.radians(8), box_ned_min=(-2, -2, -2),
                                 box_ned_max=(2, 2, 2), follower_owns_dynamics=True,
                                 **RP_LIM)).evaluate(
        msg, r_now=(np.array([0.005, 0, 0]), 0.0), cur_sample=st.sample,
        cur_sample_att=st.sample_att, now=0.1)
    assert va.margins == v.margins and va.reasons == v.reasons and va.status == v.status
    st.install(_line_plan(2, 2.0, 0.5, [0.1, 0.12, 0], [0.05, 0, 0], 21, yaw=0.3), now=2.0)
    ts = np.arange(1.9, 4.0, 0.01)
    s = st.sample(ts)
    assert len(s) == 5
    assert _sha(*s) == _GOLDEN_4DOF["stitch"], "4-DoF stitcher bytes changed"
    assert st.sample_att(ts) is None
    pv = st.preview_install(_line_plan(3, 3.0, 0.5, [0.15, 0.0, 0], [0.05, 0, 0], 21, yaw=0.0),
                            now=3.0)
    assert sorted(pv) == ["a_peak", "n_samples", "r_peak", "v_peak"]
    assert _sha([pv[k] for k in sorted(pv)]) == _GOLDEN_4DOF["preview"]
    tt = np.arange(0.0, 10.0 + 0.025, 0.05)
    p = np.vstack([0.8 * np.sin(0.5 * tt), 0.4 * (1 - np.cos(0.4 * tt)), 0.1 * np.sin(0.3 * tt)])
    p = p - p[:, :1]
    yaw = 0.3 * np.sin(0.4 * tt)
    g = np.clip(0.5 + 0.4 * np.sin(0.5 * tt), 0.0, 1.0)
    tr = ReplayTrack(t=tt, p=p, yaw=yaw - yaw[0], g=g, meta={})
    sl, al = time_dilate(tr, 0.05)
    am = anchor_track(tr, np.array([2.0, -1.0, 0.5]), 0.7, t0=100.0, dt=0.25)
    ch = chop_track(tr, np.array([2.0, -1.0, 0.5]), 0.7, t0=100.0)
    got = _sha(sl.t, sl.p, sl.yaw, sl.g, [al], am.p_ned, am.yaw, am.g,
               *[x for m in ch for x in (m.p_ned, m.yaw, m.g)])
    assert got == _GOLDEN_4DOF["replay"], (got, "4-DoF replay helper bytes changed")
    assert sl.rp is None and am.rp is None and all(m.rp is None for m in ch)


# ------------------------------------------------------------------- runner
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
