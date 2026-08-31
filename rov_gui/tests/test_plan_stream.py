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
    stitcher.
"""

from __future__ import annotations

import contextlib
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
def _write_session(sdir: Path, T=240, hz=20.0, nan_rows=(50, 51, 120),
                   schema="umi_handheld_poses/1"):
    """Synthetic 12 s handheld session: still 0-2 s, smoothstep motion
    2-9 s, still 9-12 s. Pure-yaw quaternions so yaw extraction is exact."""
    t = np.arange(T) / hz
    s = np.clip((t - 2.0) / 7.0, 0.0, 1.0)
    f = 3 * s ** 2 - 2 * s ** 3                  # C1 ramp 0 -> 1
    base = np.array([3.0, -1.0, 0.7])
    p = np.vstack([base[0] + 0.6 * f,
                   base[1] + 0.3 * np.sin(math.pi * f),
                   base[2] + 0.1 * f])
    yaw = 2.0 + 0.4 * np.sin(math.pi * f)
    q = np.zeros((T, 4))
    q[:, 0] = np.cos(yaw / 2)
    q[:, 3] = np.sin(yaw / 2)
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
