#!/usr/bin/env python3
"""
test_policy_worker.py — the diffusion-policy worker (backends/policy.py),
offline: StubPolicySession, identity depth grid, a synthetic PolicyState feed.

    QT_QPA_PLATFORM=offscreen \\
      ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_policy_worker.py

No GPU, no torch, no camera. The worker is constructed and ``setup()`` /
``tick()`` are called directly (no thread), the same way test_replay drives
MpcWorker. What is pinned (spec DP_LIVE_POLICY_SPEC_V2):

* A18 depth pairing: the partner nearest ``t_d - obs_dt`` within ±0.6·obs_dt;
  else the nearest OLDER frame within 3.0·obs_dt (recorded honestly as a
  longer baseline — and the PROPRIO rows are taken at that same spacing, so
  image and proprio baselines agree); else SKIP — except the first inference
  after START, which may duplicate the newest frame (``pair_dup``);
* the trigger: inference fires on depth-frame ARRIVAL once ``period_s`` has
  elapsed, and a skip does not burn the period (verify 2026-09-02);
* A6 / A8 / A9 skips: degenerate proprio rows (both on one fix), an epoch
  change clearing the history, a newest row that is not a fresh fix, and a
  depth stamp ahead of the newest fix by > 0.5·obs_dt (``skip_fix_lag``);
* the emitted ``PolicyPlan`` carries every record field (rows, fix stamps,
  pair spacing, coverage, validity, epoch, obs_t on the monotonic clock,
  the checkpoint's obs stride);
* A13 status cadence: ≥ 1 Hz while idle, the SENSORS row "DP policy", and a
  PolicyState older than 1 s counts as inactive; the status carries the
  GRID state (``grid_ok`` / ``grid_why``) the controller's refusal reads;
* the mailbox drops a frame whose ``src`` is not the declared grid kind, and
  the FoundationStereo tap RETURNS when the mailbox refuses another grid;
* ``rect_left_grid_from_rig`` refuses a rig without R1/P1 by name;
* ``meta()`` keys the run record relies on;
* the CLI table: --policy moves the FoundationStereo defaults to the
  training store's (iters 16, scale 1.0, alpha 0.5) unless typed, and
  ``check_policy`` refuses before it warns, and names every warning;
* the two backends wire the policy hops the same way (an AST scan of their
  ``__init__`` bodies, like test_offline's LoopWorker-slot rule);
* a ``--source demo`` run meta never says "real vehicle" (A15).
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.bus import DataBus, PolicyMailbox
from rov_gui.perception.dp_policy import StubPolicySession
from rov_gui.perception.policy_obs import (GRID_IDENTITY, GRID_RECT_LEFT,
                                           IdentityGrid)
from rov_gui.state import Conn, PolicyState, now
from rov_gui.tests.test_control import _app, _test_opts

OBS_DT = 2.0 / 30.0


# --------------------------------------------------------------- fixtures
class _Opts:
    """The CLI namespace the worker reads, minus argparse."""
    source = "demo"
    mpc = True
    policy = True
    policy_ckpt = "stub"
    policy_repo = None
    fstereo_iters = 16
    fstereo_scale = 1.0
    fstereo_size = None
    fstereo_alpha = 0.5
    fstereo_repo = None
    fstereo_ckpt = None
    fstereo_align_size = "400x250"
    policy_allow_fs_mismatch = False
    depth_scale = 0.64
    nav_geometry = None
    mpc_mode = None
    rov_model = None


def _worker(tmp, factory=None, *, run_dir_fn=None):
    """A PolicyWorker set up on THIS thread with the stub session (or the
    ``factory`` given) and an identity grid declared on its mailbox; plus
    the bus and collectors. ``run_dir_fn`` (the controller's ``_run_dir``
    in the backends) also turns ``--record-depth`` on, so the worker builds
    a real DepthRecorder filing into whatever folder the callable names."""
    _app()
    from rov_gui.backends.policy import PolicyWorker

    to = _test_opts(tmp)
    opts = _Opts()
    opts.nav_config = to.nav_config
    opts.mpc_config = to.mpc_config
    if run_dir_fn is not None:
        opts.record_depth = True
    bus = DataBus()
    mb = PolicyMailbox()
    if factory is None:
        # `ckpt=` is what PolicyWorker.set_ckpt passes (the panel pick,
        # 2026-09-11); setup() calls factory(pc, opts) without it.
        factory = lambda pc, o, ckpt=None: StubPolicySession(     # noqa: E731
            ckpt or "stub", "", dataset_fps=float(pc["dataset_fps"]))
    w = PolicyWorker(bus, mb, opts, session_factory=factory)
    if run_dir_fn is not None:
        w.run_dir_fn = run_dir_fn
    plans, statuses, sensors, logs = [], [], [], []
    bus.policy_plan.connect(plans.append)
    bus.policy_status.connect(statuses.append)
    bus.sensor_stat.connect(sensors.append)
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    w.setup()
    assert w.session.ready, "the stub must be ready synchronously"
    assert mb.wanted(), "setup must want depth frames"
    assert mb.set_grid(IdentityGrid(size=(640, 400)), GRID_IDENTITY)
    return w, bus, mb, plans, statuses, sensors, logs


def _depth(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.integers(400, 2500, size=(400, 640)).astype(np.uint16)
    d[rng.random((400, 640)) < 0.02] = 0
    return d


def _state(t_fix, eta, *, epoch=1, active=True, fresh=True, halted=False,
           eta_start=(0.0,) * 6, width=0.069, stamp=None):
    return PolicyState(t=now(), t_fix=t_fix, eta=tuple(float(v) for v in eta),
                       fix_fresh=fresh, epoch=epoch, active=active,
                       halted=halted, t0_traj=now() - 1.0, eta_start=eta_start,
                       grip_width_m=width, engaged=True,
                       stamp=(now() if stamp is None else stamp))


def _feed_history(w, t0, n=6, dt=0.1, speed=0.05, **kw):
    """n fix rows ending at t0, the vehicle moving +x at `speed`."""
    for i in range(n):
        t = t0 - (n - 1 - i) * dt
        w.on_policy_state(_state(t, (speed * (t - t0), 0, 0.8, 0, 0, 0.1), **kw))


def _put(mb, t, seed=0, src=GRID_IDENTITY):
    return mb.put(_depth(seed), t, src)


def _frame(w, mb, t, seed=0):
    """put + ingest: the mailbox is LATEST-WINS (one item), so two puts
    before a tick would collapse into one frame — the worker's ring is what
    keeps history, and it only grows when the worker takes."""
    assert _put(mb, t, seed)
    w._ingest_depth()


def _infer(w, mb, plans, *, epoch=1, engaged=True, seed=10):
    """One full inference on fresh frames in ``epoch``: feeds the history,
    two paired frames, clears the period, ticks, and asserts a plan came
    out. Returns the plan."""
    n = len(plans)
    t = now()
    w._ring.clear()
    _feed_history(w, t, epoch=epoch)
    if not engaged:
        import dataclasses
        w.on_policy_state(dataclasses.replace(
            _state(t, (0.0,) * 6, epoch=epoch), engaged=False))
    _frame(w, mb, t - OBS_DT - 0.005, seed=seed)
    _frame(w, mb, t - 0.005, seed=seed + 1)
    w._last_infer = 0.0
    w.tick()
    assert len(plans) == n + 1, (w.counters, w._note)
    return plans[-1]


def _drain_rec(rec, tries: int = 100):
    """Wait for the recorder's writer thread to catch up (test_depth_record's
    helper), so a folder assertion is not a race against it."""
    for _ in range(tries):
        c = rec.counters
        if c["written"] + c["write_errors_writer"] >= c["offered"] \
                - c["dropped_full"] - c["dropped_budget"] - c["dropped_no_writer"]:
            return
        time.sleep(0.02)


# ---------------------------------------------------------------- pairing
def test_pairing_rules_pick_near_then_fallback_then_skip_or_dup():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, *_ = _worker(tmp)
        # ring entries are what _ingest_depth builds
        def ring(*ts):
            w._ring.clear()
            for i, t in enumerate(ts):
                w._ring.append({"t": float(t), "depth": None, "obs": None,
                                "stats": None})
        # 1) partner within +-0.6 obs_dt of t_d - obs_dt
        ring(10.0 - 0.20, 10.0 - 0.09, 10.0 - 0.03, 10.0)
        e, how = w._pick_pair(OBS_DT, allow_dup=False)
        assert how == "near" and abs(e["t"] - (10.0 - 0.09)) < 1e-9, (how, e)
        # 2) nothing near, an OLDER frame within 3.0 obs_dt -> fallback (the
        #    newest of those; the too-close frame at -0.02 is NOT a partner)
        ring(10.0 - 0.30, 10.0 - 0.15, 10.0 - 0.02, 10.0)
        e, how = w._pick_pair(OBS_DT, allow_dup=False)
        assert how == "fallback" and abs(e["t"] - (10.0 - 0.15)) < 1e-9, (how, e)
        assert 10.0 - e["t"] <= 3.0 * OBS_DT
        # 2b) the measured 7.26 Hz frame train (138 ms spacing): fallback,
        #     never skip — 2.5·obs_dt (167 ms) left 29 ms for jitter
        from rov_gui.backends.policy import PAIR_FALLBACK
        assert PAIR_FALLBACK == 3.0
        ring(10.0 - 2 * 0.1377, 10.0 - 0.1377, 10.0)
        e, how = w._pick_pair(OBS_DT, allow_dup=False)
        assert how == "fallback" and abs(10.0 - e["t"] - 0.1377) < 1e-9
        ring(10.0 - 0.19, 10.0)                      # one slow solve: still a pair
        assert w._pick_pair(OBS_DT, allow_dup=False)[1] == "fallback"
        # 3) only frames beyond 3.0 obs_dt -> skip, unless the first inference
        ring(10.0 - 0.30, 10.0)
        e, how = w._pick_pair(OBS_DT, allow_dup=False)
        assert e is None and how == "skip"
        e, how = w._pick_pair(OBS_DT, allow_dup=True)
        assert how == "dup" and e is w._ring[-1]
        # 4) a single frame: skip, or dup on the first inference
        ring(10.0)
        assert w._pick_pair(OBS_DT, allow_dup=False) == (None, "skip")
        assert w._pick_pair(OBS_DT, allow_dup=True)[1] == "dup"


def test_first_inference_may_duplicate_and_later_ones_may_not():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t)
        assert _put(mb, t - 0.01, seed=1)
        w.tick()
        assert len(plans) == 1, statuses[-1] if statuses else "no status"
        assert plans[0].pair_dup is True and plans[0].obs_pair_dt_s == 0.0
        assert w.counters["pair_dup"] == 1
        # a duplicated pair has no image baseline: the rows keep obs_dt
        assert abs(plans[0].obs_rows_t[1] - plans[0].obs_rows_t[0] - OBS_DT) < 1e-9
        # next period: the next frame arrives with the only older frame
        # beyond 3.0 obs_dt -> skip_pair, no plan, and no dup any more
        w._last_infer = 0.0
        _frame(w, mb, t - 0.01 + 0.25, seed=5)
        w.tick()
        assert len(plans) == 1
        assert w.counters["skip_pair"] == 1
        # a partner arrives obs_dt later -> a real pair
        w._ring.clear()
        t2 = now()
        _feed_history(w, t2)
        _frame(w, mb, t2 - OBS_DT - 0.005, seed=2)
        _frame(w, mb, t2 - 0.005, seed=3)
        w._last_infer = 0.0
        w.tick()
        assert len(plans) == 2, w.counters
        p = plans[-1]
        assert p.pair_dup is False
        assert abs(p.obs_pair_dt_s - OBS_DT) < 0.6 * OBS_DT
        assert w.counters["pair_near"] == 1


def test_the_fallback_pair_is_recorded_with_its_real_spacing():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t)
        w._first_pending = False           # not the first inference
        _frame(w, mb, t - 0.15, seed=1)    # 2.25 obs_dt back: fallback
        _frame(w, mb, t - 0.005, seed=2)
        w.tick()
        assert len(plans) == 1, w.counters
        assert plans[0].pair_dup is False
        assert abs(plans[0].obs_pair_dt_s - 0.145) < 1e-6
        assert w.counters["pair_fallback"] == 1
        # ...and the PROPRIO rows are the same 0.145 s apart, ending on the
        # depth stamp: image and proprio baselines agree (verify 2026-09-02)
        r0, r1 = plans[0].obs_rows_t
        assert abs(r1 - (t - 0.005)) < 1e-9 and abs(r1 - r0 - 0.145) < 1e-9, plans[0].obs_rows_t
        # beyond 3.0 obs_dt (0.200 s): skip
        w._ring.clear()
        w._last_infer = 0.0
        t = now()
        _feed_history(w, t)
        _frame(w, mb, t - 0.25, seed=3)
        _frame(w, mb, t - 0.005, seed=4)
        w.tick()
        assert len(plans) == 1 and w.counters["skip_pair"] == 1


# ------------------------------------------------------------------ skips
def test_degenerate_rows_epoch_change_and_stale_fix_each_skip():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        t = now()
        # FIX LAG: two fixes far in the past, the depth 1.5 s newer than the
        # newest fix -> the 'now' row would be clamped to a fix 1.5 s stale
        # while the image is current: skip_fix_lag (not degenerate — the
        # rows themselves interpolate fine).
        w.on_policy_state(_state(t - 2.0, (0, 0, 0.8, 0, 0, 0)))
        w.on_policy_state(_state(t - 1.5, (0.1, 0, 0.8, 0, 0, 0)))
        _frame(w, mb, t - OBS_DT - 0.005, seed=1)
        _frame(w, mb, t - 0.005, seed=2)
        w.tick()
        assert not plans and w.counters["skip_fix_lag"] == 1, w.counters
        assert w._last_infer == 0.0, "a skip must not burn the period"
        # DEGENERATE: every fix NEWER than the depth stamp -> both rows clamp
        # to the first fix (no lag: the depth is not ahead of the fixes).
        w.hist.clear()
        w.on_policy_state(_state(t + 1.0, (0, 0, 0.8, 0, 0, 0)))
        w.on_policy_state(_state(t + 1.5, (0.1, 0, 0.8, 0, 0, 0)))
        _frame(w, mb, t - 0.004, seed=6)
        w.tick()
        assert not plans and w.counters["skip_degenerate"] == 1, w.counters
        # EPOCH CHANGE clears the history -> skip_history until >= obs_dt
        # of NEW-epoch rows exist
        w._last_infer = 0.0
        w.on_policy_state(_state(t - 0.05, (0.2, 0, 0.8, 0, 0, 0), epoch=2))
        assert len(w.hist) == 1 and w.counters["epoch_changes"] == 1
        _frame(w, mb, t + 0.0, seed=3)         # the attempt needs an arrival
        w.tick()
        assert not plans and w.counters["skip_history"] == 1
        # ...and the first inference of the new epoch may duplicate again
        assert w._first_pending is True
        # STALE FIX: the newest row is not a fresh tag solution -> skip_fresh.
        # Rows must ADVANCE past the first new-epoch row (t - 0.05): the
        # history dedups a non-advancing stamp (A6), which is also why a
        # stale row cannot be smuggled in behind a fresh one.
        w._last_infer = 0.0
        for i, dt in enumerate((0.02, -0.01, -0.03)):
            w.on_policy_state(_state(t - dt, (0.2 + 0.01 * i, 0, 0.8, 0, 0, 0),
                                     epoch=2, fresh=(dt != -0.03)))
        assert len(w.hist) == 4 and w.hist.span() >= OBS_DT
        _frame(w, mb, t + 0.02, seed=4)
        w.tick()
        assert not plans and w.counters["skip_fresh"] == 1, w.counters
        # a fresh row on top + a frame -> it infers (the frame at t+0.0 is
        # the near partner of t+0.07)
        w._last_infer = 0.0
        w.on_policy_state(_state(t + 0.05, (0.24, 0, 0.8, 0, 0, 0), epoch=2))
        _frame(w, mb, t + 0.07, seed=5)
        w.tick()
        assert len(plans) == 1, w.counters
        assert plans[0].epoch == 2
        assert plans[0].obs_rows_t[1] == t + 0.05, "the 'now' row stops at the newest fix"


def test_inference_fires_on_frame_arrival_and_a_skip_does_not_burn_the_period():
    """verify 2026-09-02: a period-phase trigger left the newest frame a mean
    half frame interval old at inference (69 ms at 7.26 Hz) on an intake
    budget with no margin; and a skip stamped the period, so one jittered
    frame cost a whole second between plans."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t)
        # a frame put + taken -> that arrival fires the first inference
        assert _put(mb, t - 0.01, seed=1)
        assert w._frame_pending is False
        w.tick()
        assert len(plans) == 1 and w._last_infer > 0.0
        # the period elapses with NO new frame: nothing fires, nothing is
        # counted as a skip — the worker is armed, waiting for an arrival
        w._last_infer = 0.0
        n_skip = sum(v for k, v in w.counters.items() if k.startswith("skip_"))
        for _ in range(5):
            w.tick()
        assert len(plans) == 1
        assert sum(v for k, v in w.counters.items() if k.startswith("skip_")) == n_skip
        assert w._last_infer == 0.0
        # a frame with no partner within 3.0 obs_dt -> skip_pair, the
        # period is NOT stamped...
        _frame(w, mb, t - 0.01 + 0.25, seed=2)
        assert w._frame_pending is True
        w.tick()
        assert len(plans) == 1 and w.counters["skip_pair"] == 1
        assert w._last_infer == 0.0 and w._frame_pending is False
        # ...so the very next frame retries inside the same period and pairs
        # with the one before it (fallback, 0.1 s)
        t2 = now()
        w._ring.clear()
        _feed_history(w, t2)
        _frame(w, mb, t2 - 0.105, seed=3)
        _frame(w, mb, t2 - 0.005, seed=4)
        w.tick()
        assert len(plans) == 2, w.counters
        assert w._last_infer > 0.0
        assert abs(plans[-1].obs_pair_dt_s - 0.1) < 1e-6


def test_proprio_rows_follow_the_depth_pair_spacing():
    """verify 2026-09-02: a fallback pair 138 ms apart with proprio rows
    66.7 ms apart showed the network 2x the image motion of the proprio
    motion — a combination training never produced. Rows now span the real
    pair spacing, ending on the depth stamp."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        speed = 0.05
        _feed_history(w, t, n=10, dt=0.05, speed=speed)
        w._first_pending = False
        t_new = t - 0.004
        pair = 0.1377                                   # the 7.26 Hz spacing
        _frame(w, mb, t_new - pair, seed=1)
        _frame(w, mb, t_new, seed=2)
        w.tick()
        assert len(plans) == 1, w.counters
        p = plans[0]
        assert w.counters["pair_fallback"] == 1
        assert abs(p.obs_pair_dt_s - pair) < 1e-9
        assert abs(p.obs_rows_t[1] - t_new) < 1e-9
        assert abs((p.obs_rows_t[1] - p.obs_rows_t[0]) - pair) < 1e-9
        # the displacement the network sees is speed x the PAIR spacing
        disp = float(np.linalg.norm(p.lowdim["robot0_eef_pos"][0]))
        assert abs(disp - speed * pair) < 1e-6, (disp, speed * pair)
        assert p.obs_dt_s == OBS_DT                     # the stride is still recorded
        # the fix stamps are the ones bracketing THOSE rows
        assert min(p.obs_fix_t) <= p.obs_rows_t[0] and max(p.obs_fix_t) >= p.obs_rows_t[1] - 1e-9


def test_a_depth_stamp_ahead_of_the_newest_fix_is_skipped_or_clamped_honestly():
    """verify 2026-09-02: a 'now' row clamped to the newest fix while the
    'prev' row sat obs_dt before the DEPTH stamp scaled the motion cue by a
    factor in (0, 1] and the record still claimed (t_d - obs_dt, t_d)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t, n=8, dt=0.05, speed=0.05)    # newest fix at t
        w._first_pending = False
        # depth 0.75 obs_dt ahead of the newest fix -> skip_fix_lag
        t_d = t + 0.75 * OBS_DT
        _frame(w, mb, t_d - OBS_DT, seed=1)
        _frame(w, mb, t_d, seed=2)
        w.tick()
        assert not plans and w.counters["skip_fix_lag"] == 1, w.counters
        # depth 0.3 obs_dt ahead (inside the tolerance) -> a plan whose rows
        # are (t - pair, t): clamped to the fix, spacing intact, RECORDED
        w._ring.clear()
        t_d = t + 0.3 * OBS_DT
        _frame(w, mb, t_d - OBS_DT, seed=3)
        _frame(w, mb, t_d, seed=4)
        w.tick()
        assert len(plans) == 1, w.counters
        p = plans[0]
        assert abs(p.obs_rows_t[1] - t) < 1e-9, "the 'now' row stops at the newest fix"
        assert abs((p.obs_rows_t[1] - p.obs_rows_t[0]) - OBS_DT) < 1e-9
        disp = float(np.linalg.norm(p.lowdim["robot0_eef_pos"][0]))
        assert abs(disp - 0.05 * OBS_DT) < 1e-6, "no under-scaled motion cue"
        assert p.obs_t == t_d, "obs_t is still the depth stamp"


def test_no_inference_while_inactive_halted_or_the_state_is_stale():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        _frame(w, mb, t - OBS_DT - 0.005, seed=1)
        _frame(w, mb, t - 0.005, seed=2)
        _feed_history(w, t, active=False)
        w.tick()
        assert not plans
        _feed_history(w, t, halted=True)
        w.tick()
        assert not plans
        # active but the state is older than 1 s -> inactive (A13)
        _feed_history(w, t, stamp=now() - 1.5)
        w.tick()
        assert not plans
        # a fresh active state -> infers
        _feed_history(w, t)
        w.tick()
        assert len(plans) == 1


# --------------------------------------------------------------- the plan
def test_the_plan_carries_every_record_field():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t, width=0.05)
        t_new = t - 0.005
        _frame(w, mb, t_new - OBS_DT, seed=1)
        _frame(w, mb, t_new, seed=2)
        w.tick()
        assert len(plans) == 1
        p = plans[0]
        assert p.plan_id == 1 and p.epoch == 1
        assert p.obs_t == t_new, "obs_t is the newest depth frame's t_capture"
        assert p.t_emit >= p.obs_t and p.stamp == p.t_emit
        assert p.action.shape == (16, 5) and p.action.dtype == np.float32
        assert p.action_repr == "pos_yaw_width"   # the checkpoint contract rides along
        assert set(p.lowdim) == {"robot0_eef_pos", "robot0_eef_rot_axis_angle",
                                 "robot0_gripper_width",
                                 "robot0_eef_rot_axis_angle_wrt_start"}
        assert p.lowdim["robot0_eef_pos"].shape == (2, 3)
        assert np.allclose(p.lowdim["robot0_eef_pos"][1], 0.0), "last row is self-relative"
        assert p.lowdim["robot0_eef_pos"][0, 0] != 0.0, "the older row shows the motion"
        assert np.allclose(p.lowdim["robot0_gripper_width"], 0.05)
        assert p.obs_rows_t == (t_new - OBS_DT, t_new)
        assert 2 <= len(p.obs_fix_t) <= 4
        assert p.depth_src == GRID_IDENTITY
        assert p.depth_coverage == 1.0
        assert 0.9 < p.depth_valid <= 1.0
        assert p.pair_dup is False
        assert p.ckpt_sha1 == ""              # the stub has no checkpoint
        assert p.infer_ms >= 0.0
        assert p.obs_dt_s == OBS_DT           # the checkpoint's stride rides along
        # the status published with the plan says so
        st = statuses[-1]
        assert st.ready and st.n_plans == 1 and st.depth_src == GRID_IDENTITY
        assert st.conn is Conn.ONLINE
        assert st.grid_ok is True and st.grid_why == ""
        assert st.obs_dt_s == OBS_DT
        assert st.action_repr == "pos_yaw_width"
        assert st.mount_ok is True and st.mount_why == ""   # stub: no yaw-axis keys


def test_mount_check_reads_the_checkpoint_yaw_axis():
    """2026-09-07: a pos_yaw_width checkpoint carries the rotation its yaw
    label was defined on (contract yaw_axis_R_frd_cam); the worker compares
    it with hw_nav's R_frd_cam('main') at setup (stub) and again when the
    session turns ready (tick), and the status the controller's arm check
    reads says so. Matching -> ok; a different rotation -> mount_ok False
    with a reason naming R_frd_cam and ONE error line; malformed -> refused
    too; and the checkpoint's action_repr rides on plan and status as the
    checkpoint states it, not as a constant."""
    from rov_gui.control.geometry import NavConfig

    class _MountStub(StubPolicySession):
        def __init__(self, R, *a, **kw):
            super().__init__(*a, **kw)
            self._contract["yaw_axis_R_frd_cam"] = R
            self._contract["yaw_axis_cam_tilt_deg"] = 43.3

    with tempfile.TemporaryDirectory() as tmp:
        R_bc = NavConfig.load(_test_opts(tmp).nav_config).R_t_frd_cam("main")[0]
        R_bc = np.asarray(R_bc, float).reshape(3, 3)
        # (a) the training mount == the vehicle's: ok (both the flat 9 and
        #     the 3x3 spelling the loader may hand over)
        for R in (R_bc.reshape(-1).tolist(), R_bc.tolist()):
            w, bus, mb, plans, statuses, sensors, logs = _worker(
                tmp, factory=lambda pc, o, R=R, ckpt=None: _MountStub(
                    R, "stub", "", dataset_fps=float(pc["dataset_fps"])))
            assert np.allclose(w._R_bc, R_bc)
            assert w._mount_ok is True and w._mount_why == ""
            w.tick()
            assert w._mount_ok is True
            assert statuses[-1].mount_ok is True and statuses[-1].mount_why == ""
            assert statuses[-1].action_repr == "pos_yaw_width"
            assert not any("MOUNT MISMATCH" in m for _l, m in logs)
        # (b) a re-measured mount 1e-4 away: refused, with the reason, once
        R_bad = R_bc.copy()
        R_bad[0, 1] += 1e-4
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _MountStub(
                R_bad.tolist(), "stub", "", dataset_fps=float(pc["dataset_fps"])))
        assert w._mount_ok is False
        assert "R_frd_cam" in w._mount_why and "43.3" in w._mount_why
        assert "1.00e-04" in w._mount_why, w._mount_why
        for _ in range(3):
            w.tick()
        assert sum("MOUNT MISMATCH" in m and lvl == "error"
                   for lvl, m in logs) == 1, logs
        st = statuses[-1]
        assert st.mount_ok is False and "R_frd_cam" in st.mount_why
        assert st.ready and st.action_repr == "pos_yaw_width"  # the arm check, not this, refuses
        # (c) a malformed rotation is a refusal, not a crash
        w, *_ , statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _MountStub(
                [1.0, 2.0, 3.0], "stub", "", dataset_fps=float(pc["dataset_fps"])))
        assert w._mount_ok is False and "malformed" in w._mount_why
        # (d) the checkpoint's OWN action_repr rides, whatever it is: a
        #     legacy pose10d stub reports pose10d and emits (16, 10)
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: StubPolicySession(
                "stub", "", dataset_fps=float(pc["dataset_fps"]),
                action_repr="pose10d"))
        assert w._mount_ok is True                    # no yaw-axis keys
        t = now()
        _feed_history(w, t)
        t_new = t - 0.005
        _frame(w, mb, t_new - OBS_DT, seed=1)
        _frame(w, mb, t_new, seed=2)
        w.tick()
        assert len(plans) == 1
        assert plans[0].action.shape == (16, 10)
        assert plans[0].action_repr == "pose10d"
        assert statuses[-1].action_repr == "pose10d"


# ----------------------------------------------------------------- status
def test_status_is_published_at_least_once_a_second_when_idle():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, _logs = _worker(tmp)
        n0 = len(statuses)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.3:
            w.tick()
            time.sleep(0.02)
        assert len(statuses) - n0 >= 2, len(statuses) - n0
        assert all(st.ready and not st.error for st in statuses)
        assert all(now() - st.stamp < 3.0 for st in statuses)
        rows = [s for s in sensors if s.name == "DP policy"]
        assert rows, "the SENSORS row must be published"
        assert rows[-1].conn in (Conn.DEGRADED, Conn.ONLINE)   # no depth yet
        assert not plans


def test_the_mailbox_drops_a_frame_on_another_grid():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, *_ = _worker(tmp)
        assert _put(mb, now(), src=GRID_RECT_LEFT) is False
        assert mb.counters()["dropped_src"] == 1
        assert mb.set_grid(IdentityGrid(size=(640, 360)), GRID_IDENTITY) is False
        assert mb.counters()["grid_refused"] == 1
        assert _put(mb, now(), src=GRID_IDENTITY) is True
        w.tick()
        assert w.counters["depth_frames"] == 1 and w.builder.grid_kind == GRID_IDENTITY


def test_a_refused_grid_is_visible_in_the_status_the_controller_reads():
    """verify 2026-09-02: a grid the builder REFUSED (coverage below
    policy.min_obs_coverage) left the status ready=True / error='' /
    depth_src set, so the controller armed a mission that could never
    receive a plan. grid_ok / grid_why now say so."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        w.builder.min_coverage = 1.01         # nothing can cover that
        _frame(w, mb, now() - 0.005, seed=1)
        w._publish_status(force=True)
        st = statuses[-1]
        assert st.ready and not st.error and st.depth_src == GRID_IDENTITY
        assert st.grid_ok is False
        assert "obs coverage" in st.grid_why and "<" in st.grid_why, st.grid_why
        assert st.conn is Conn.DEGRADED
        assert any("REFUSED" in m for _l, m in logs)
        row = [s for s in sensors if s.name == "DP policy"][-1]
        assert "grid REFUSED" in row.detail
        # and the worker itself never infers on it
        _feed_history(w, now())
        _frame(w, mb, now() - 0.004, seed=2)
        w.tick()
        assert not plans and w.counters["skip_grid"] == 1


def test_sensor_row_notes_are_short_and_canonical():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, _logs = _worker(tmp)
        _frame(w, mb, now() - 0.005, seed=1)
        w.tick()                              # no active mission -> idle
        w._publish_status(force=True)         # the idle publish is 1 Hz
        assert statuses[-1].note == "idle"
        row = [s for s in sensors if s.name == "DP policy"][-1]
        assert row.detail.endswith(" idle"), row.detail
        t = now()
        _feed_history(w, t)
        _frame(w, mb, t + 0.75 * OBS_DT, seed=2)   # ahead of the fixes
        w.tick()
        w._publish_status(force=True)
        assert statuses[-1].note == "skip_fix_lag"
        row = [s for s in sensors if s.name == "DP policy"][-1]
        assert row.detail.endswith(" skip_fix_lag"), row.detail
        w._note = "skip_stale_depth"
        assert w._sensor_detail(statuses[-1]).endswith(" skip_stale_depth")


def test_rect_left_grid_from_rig_refuses_a_rig_without_rectification():
    from types import SimpleNamespace

    from rov_gui.backends.policy import rect_left_grid_from_rig
    from rov_gui.perception.policy_obs import GridError

    bare = SimpleNamespace(R1=None, P1=None, mono_size=(640, 400), alpha=0.5,
                           provenance={})
    try:
        rect_left_grid_from_rig(bare)
    except GridError as e:
        assert "R1/P1" in str(e)
    else:
        raise AssertionError("a rig without R1/P1 must be refused by name")
    ok = SimpleNamespace(R1=np.eye(3), P1=np.array([[500.0, 0, 320, 0],
                                                    [0, 500.0, 200, 0],
                                                    [0, 0, 1, 0]]),
                         mono_size=(640, 400), alpha=0.5, provenance={"x": 1})
    g = rect_left_grid_from_rig(ok)
    assert g.fingerprint and g.mono_size == (640, 400)


def test_the_fstereo_tap_returns_when_the_mailbox_refuses_another_grid():
    """verify 2026-09-02: the tap logged 'drops these frames' and then put
    the frame anyway — the mailbox gates on src, still rect_left, so the
    frame was warped through the FIRST grid's maps."""
    from types import SimpleNamespace

    from rov_gui.backends.hardware import FStereoWorker

    _app()
    bus = DataBus()
    logs = []
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    fsw = FStereoWorker(bus, None, _Opts(), {})
    mb = PolicyMailbox()
    mb.set_wanted(True)
    fsw.policy_mb = mb
    P1 = np.array([[500.0, 0, 320, 0], [0, 500.0, 200, 0], [0, 0, 1, 0]])
    rig_a = SimpleNamespace(R1=np.eye(3), P1=P1, mono_size=(640, 400),
                            alpha=0.5, provenance={})
    c, s_ = np.cos(np.radians(0.5)), np.sin(np.radians(0.5))
    rig_b = SimpleNamespace(R1=np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]),
                            P1=P1, mono_size=(640, 400), alpha=0.5, provenance={})
    native = np.full((400, 640), 1000, np.uint16)
    fsw._tap_policy({"depth_native": native}, {"rig": rig_a, "t_capture": 1.0})
    assert mb.counters()["put"] == 1 and fsw._policy_tap["put"] == 1
    # a DIFFERENT rig: refused by the mailbox, and NOT put
    fsw._tap_policy({"depth_native": native}, {"rig": rig_b, "t_capture": 1.1})
    fsw._tap_policy({"depth_native": native}, {"rig": rig_b, "t_capture": 1.2})
    assert mb.counters()["grid_refused"] == 2
    assert mb.counters()["put"] == 1, "a refused grid's frame must not be put"
    assert fsw._policy_tap["dropped_grid_refused"] == 2
    assert sum(1 for _l, m in logs if "CHANGED mid-run" in m) == 1
    # the first rig again: accepted, put
    fsw._tap_policy({"depth_native": native}, {"rig": rig_a, "t_capture": 1.3})
    assert mb.counters()["put"] == 2
    # a rig without R1/P1: GridError caught once, logged once, frames skipped
    bare = SimpleNamespace(R1=None, P1=None, mono_size=(640, 400), alpha=0.5,
                           provenance={})
    fsw._tap_policy({"depth_native": native}, {"rig": bare, "t_capture": 1.4})
    fsw._tap_policy({"depth_native": native}, {"rig": bare, "t_capture": 1.5})
    assert mb.counters()["put"] == 2
    assert fsw._policy_tap["dropped_no_grid"] == 2
    assert sum(1 for _l, m in logs if "no policy depth grid" in m) == 1
    m = fsw.meta()
    assert m["enabled"] is False and "policy_tap" not in m   # no session built


def test_meta_has_the_record_keys():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        t = now()
        _feed_history(w, t)
        _frame(w, mb, t - OBS_DT - 0.005, seed=1)
        _frame(w, mb, t - 0.005, seed=2)
        w.tick()
        m = w.meta()
        for k in ("enabled", "synthetic", "session", "obs", "depth_src",
                  "obs_dt_s", "period_s", "tcp", "fstereo", "counters",
                  "pairing", "infer_ms", "policy_block_source", "mailbox",
                  "depth_scale_applied"):
            assert k in m, k
        assert m["synthetic"] is True            # stub + demo source
        assert m["session"]["stub"] is True
        assert m["obs"]["grid_kind"] == GRID_IDENTITY
        assert m["depth_scale_applied"] is None  # not the device path
        assert m["tcp"]["handheld_tcp_offset_cam_m"] == [0.0355, 0.1293, 0.3186]
        assert m["tcp"]["t_body_tcp_norm_m"] > 0.1
        assert m["counters"]["plans"] == 1 and m["pairing"]["n"] == 1
        assert m["pairing"]["fallback_max"] == 3.0 and "skip_fix_lag" in m["pairing"]
        assert "skip_fix_lag" in m["counters"]
        assert m["infer_ms"]["p50"] is not None
        import json
        json.dumps(m, default=str)               # it must serialise
        from rov_gui.backends.policy import HANDHELD_TCP_OFFSET_CAM_M
        from rov_gui.control.geometry import \
            HANDHELD_TCP_OFFSET_CAM_M as GEOM
        assert tuple(HANDHELD_TCP_OFFSET_CAM_M) == tuple(GEOM)


# ------------------------------------------------------- --record-depth
def test_the_depth_recorder_follows_the_controllers_run_folder():
    """Review 2026-09-11 (safety + record lenses). The recorder resolved the
    run folder ONCE (writer thread, first frame) and `_start_recorder` armed
    once per worker life, so a later engagement's frames were appended to
    the FIRST engagement's policy_obs/ — and with LOW level None selectable
    at runtime the two engagements can be in different TREES (water vs
    policy_observe), the pooling guard. Now, once per epoch, the worker asks
    the controller's `_run_dir` again: same folder -> nothing (a fresh
    recorder would re-open index.csv "w" and clobber; runstore joins within
    90 s and a re-START is the same pinned folder); a different folder ->
    the old recorder is closed (meta.json written) and a new one files
    there, armed with the CURRENT checkpoint; the run record lists the
    hand-over under depth_record.previous."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        run_a = Path(tmp) / "20260911" / "0911_100000"
        run_b = Path(tmp) / "20260911" / "0911_100200_observe"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        cur = [run_a]
        calls = []

        def run_dir():                    # the controller's _run_dir
            calls.append(cur[0])
            return cur[0]

        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, run_dir_fn=run_dir)
        rec0 = w.depth_rec
        assert rec0 is not None and not w._rec_started
        # engagement 1, epoch 1: the recorder arms and files under run_a
        _infer(w, mb, plans, epoch=1, seed=10)
        _drain_rec(rec0)
        assert rec0.dir == run_a / "policy_obs" and rec0.counters["written"] == 2
        assert w.depth_rec is rec0 and calls == [run_a], \
            "the writer thread resolves the folder once; the epoch-1 check " \
            "ran before the recorder had a folder and asked nothing"
        # a re-START in the SAME engagement (epoch 2, same pinned folder):
        # the check runs ONCE (one more call), nothing rotates, seq goes on
        _infer(w, mb, plans, epoch=2, seed=20)
        _drain_rec(rec0)
        assert w.depth_rec is rec0 and len(calls) == 2
        assert rec0.counters["written"] == 4 and rec0.next_seq == 4
        assert not any("ROTATED" in m for _, m in logs)
        # a tick in the same epoch does not ask again (per epoch, not per frame)
        _infer(w, mb, plans, epoch=2, seed=30)
        assert len(calls) == 2
        # (drained: close() drops whatever is still queued — frame_record's
        # bounded shutdown — and in the field the previous engagement ended
        # seconds before the next START, so its queue is long empty)
        _drain_rec(rec0)
        # engagement 2 under LOW None: the controller pins the OTHER tree
        cur[0] = run_b
        w.pc["ckpt"] = "/picked/later.ckpt"          # what set_ckpt does
        _infer(w, mb, plans, epoch=3, seed=40)
        rec1 = w.depth_rec
        assert rec1 is not rec0
        _drain_rec(rec1)
        # the epoch-3 check (producer) + the new writer's own first-frame
        # resolution: two more calls, no other
        assert calls == [run_a, run_a, run_b, run_b], calls
        assert rec1.dir == run_b / "policy_obs"
        assert rec1.counters["written"] == 2 and rec1.next_seq == 2, \
            "the new folder's index starts at seq 0"
        # the old recording is CLOSED and complete: its meta.json is on disk,
        # its index has exactly the first engagement's 6 frames, untouched
        meta_a = json.loads((run_a / "policy_obs" / "meta.json").read_text())
        assert meta_a["counters"]["written"] == 6 and meta_a["counts_complete"]
        assert meta_a["extra"]["ckpt"] == "stub"
        rows_a = (run_a / "policy_obs" / "index.csv").read_text().splitlines()
        assert len(rows_a) == 7
        assert not (run_b / "policy_obs" / "meta.json").exists(), \
            "the new recorder is still writing"
        # ...and the new one was armed with the CURRENT checkpoint
        assert rec1._meta_extra["extra"]["ckpt"] == "/picked/later.ckpt"
        assert any(lvl == "info" and "ROTATED" in m and str(run_b) in m
                   and "6 frames stay in" in m for lvl, m in logs), logs[-4:]
        # the run record says the depth is split, and where the rest is
        d = w.meta()["depth_record"]
        assert d["dir"] == str(run_b / "policy_obs") and d["rotations"] == 1
        assert d["previous"][0]["dir"] == str(run_a / "policy_obs")
        assert d["previous"][0]["counters"]["written"] == 6
        json.dumps(w.meta(), default=str)
        # frames submitted after the swap keep going to run_b
        _infer(w, mb, plans, epoch=3, seed=50)
        _drain_rec(rec1)
        assert rec1.counters["written"] == 4 and rec0.counters["written"] == 6
        # a run_dir_fn that RAISES at the check keeps the recorder (rule 3)
        def boom():
            raise OSError("disk went away")
        w.run_dir_fn = boom
        _infer(w, mb, plans, epoch=4, seed=60)
        assert w.depth_rec is rec1
        assert any("could not re-check the run folder" in m for _, m in logs)
        _drain_rec(rec1)                  # see above: close() drops the queue
        w.teardown()
        meta_b = json.loads((run_b / "policy_obs" / "meta.json").read_text())
        assert meta_b["counters"]["written"] == 6
        assert meta_b["extra"]["ckpt"] == "/picked/later.ckpt"


def test_a_panel_checkpoint_swap_is_noted_in_the_depth_recording():
    """Review 2026-09-11 (record lens): with the recorder armed in one run
    folder, a panel pick between engagements (allowed: disengaged) used to
    leave policy_obs/meta.json naming the FIRST checkpoint for every later
    frame. set_ckpt now notes the swap at the seq boundary, and the READY
    tick refreshes obs_dt_s / ckpt_sha1 for the new network."""
    import json

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "water" / "0911_110000"
        run.mkdir(parents=True)
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, run_dir_fn=lambda: run)
        rec = w.depth_rec
        _infer(w, mb, plans, epoch=1, seed=10)
        _drain_rec(rec)
        assert rec.counters["written"] == 2
        assert rec._meta_extra["extra"]["ckpt"] == "stub"
        assert "ckpt_sha1" in rec._meta_extra["extra"]
        # STOP / DISENG, then the pick
        import dataclasses
        w.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        f = Path(tmp) / "picked.ckpt"
        f.write_bytes(b"x")
        n_log = len(logs)
        w.set_ckpt(str(f))
        assert w.counters["ckpt_swaps"] == 1 and w.depth_rec is rec, \
            "same folder: the recorder is NOT re-armed (index.csv would be clobbered)"
        ex = rec._meta_extra["extra"]
        assert ex["ckpt"] == str(f)
        assert ex["ckpt_swaps"][0]["next_seq"] == 2
        assert ex["ckpt_swaps"][0]["from"] == "stub" and ex["ckpt_swaps"][0]["to"] == str(f)
        assert ex["ckpt_swaps"][0]["swap_no"] == 1
        assert any("frames from seq 2 on belong to picked.ckpt" in m
                   for _, m in logs[n_log:]), logs[n_log:]
        # the stub is READY synchronously: the next tick refreshes the contract
        w.tick()
        assert w._said_ready
        assert ex["obs_dt_s"] == w._obs_dt() and ex["ckpt_sha1"] == w._ckpt_sha1
        # engage + START again inside the join window: same folder, seq goes on
        _infer(w, mb, plans, epoch=2, seed=20)
        _drain_rec(rec)
        assert rec.counters["written"] == 4 and w.depth_rec is rec
        w.teardown()
        meta = json.loads((run / "policy_obs" / "meta.json").read_text())
        assert meta["extra"]["ckpt"] == str(f)
        assert [s["next_seq"] for s in meta["extra"]["ckpt_swaps"]] == [2]
        rows = (run / "policy_obs" / "index.csv").read_text().splitlines()
        assert [r.split(",")[0] for r in rows[1:]] == ["0", "1", "2", "3"]
        # no recorder, or one never armed: the note is a no-op, not an error
        w2, *_ = _worker(tmp)
        assert w2.depth_rec is None
        w2.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        w2.set_ckpt(str(f))
        assert w2.counters["ckpt_swaps"] == 1


# -------------------------------------------------------------------- CLI
def test_policy_moves_the_fstereo_defaults_unless_typed():
    from rov_gui.__main__ import (POLICY_DEPENDENT_DEFAULTS, build_parser,
                                  check_policy)

    table = {name: (off, on) for name, off, on in POLICY_DEPENDENT_DEFAULTS}
    assert set(table) == {"fstereo_iters", "fstereo_scale", "fstereo_alpha"}
    o = build_parser().parse_args([])
    for name, (off, _on) in table.items():
        assert getattr(o, name) == off, (name, getattr(o, name))
    o = build_parser().parse_args(["--policy"])
    for name, (_off, on) in table.items():
        assert getattr(o, name) == on, (name, getattr(o, name))
    o = build_parser().parse_args(["--policy", "--fstereo-scale", "0.5",
                                   "--fstereo-iters", "12"])
    assert o.fstereo_scale == 0.5 and o.fstereo_iters == 12 and o.fstereo_alpha == 0.5
    # the parser never hands out the sentinel
    ns, _rest = build_parser().parse_known_args(["--bogus"])
    for name in table:
        assert getattr(ns, name) is not None
    assert check_policy(build_parser().parse_args([])) is None
    lvl, msg = check_policy(build_parser().parse_args(["--policy"]))
    assert lvl == "warn" and "--mpc" in msg
    lvl, msg = check_policy(build_parser().parse_args(
        ["--policy", "--mpc", "--source", "hw"]))
    assert lvl == "refuse" and "--fstereo" in msg
    # the REFUSAL is evaluated before the no-mpc warning (it used to hide
    # behind it, verify 2026-09-02)
    lvl, msg = check_policy(build_parser().parse_args(["--policy", "--source", "hw"]))
    assert lvl == "refuse" and "--fstereo" in msg, (lvl, msg)
    # device depth: allowed, but a WARNING that names the bench experiment
    lvl, msg = check_policy(build_parser().parse_args(
        ["--policy", "--mpc", "--source", "hw", "--policy-allow-device-depth"]))
    assert lvl == "warn" and "bench experiment" in msg and "meta" in msg, msg
    # The --policy-ckpt flag is GONE (2026-09-11, the panel picker); the
    # stub is set on the namespace the way tests/tools do it, so check_policy
    # does not fall back to hw_mpc.yaml's real checkpoint (whose FS-parity
    # warnings would change what these assert).
    def _stub(argv):
        a = build_parser().parse_args(argv)
        a.policy_ckpt = "stub"
        return a
    # the stub has no training store to compare against: nothing to say
    assert check_policy(_stub(
        ["--policy", "--mpc", "--source", "hw", "--fstereo"])) is None
    # a typed alpha below 0.5 under --policy: the builder will refuse the grid
    lvl, msg = check_policy(_stub(
        ["--policy", "--mpc", "--source", "hw", "--fstereo",
         "--fstereo-alpha", "0"]))
    assert lvl == "warn" and "--fstereo-alpha 0" in msg and "0.5" in msg, msg
    # every applicable warning is named, joined into one line
    lvl, msg = check_policy(_stub(
        ["--policy", "--source", "hw", "--fstereo", "--fstereo-alpha", "0.2"]))
    assert lvl == "warn" and "--mpc" in msg and "--fstereo-alpha 0.2" in msg, msg
    # The note states the speed/parity trade the --policy defaults make, with
    # an artifact behind every number (CLAUDE.md) and BOTH sides named — the
    # rate it buys and the instrument parity it gives up, plus the way back.
    # It replaced a '~9 Hz [기억]' guess, which is what the last assert pins.
    from rov_gui.__main__ import FS_RATE_NOTE
    import inspect
    import rov_gui.__main__ as M
    assert "74.9 ms" in FS_RATE_NOTE and "sweep.txt" in FS_RATE_NOTE, FS_RATE_NOTE
    assert "140.2 ms" in FS_RATE_NOTE, "the parity cost is not stated"
    assert "2.07x" in FS_RATE_NOTE and "obs_pair_dt_s" in FS_RATE_NOTE
    assert "--fstereo-scale 1.0" in FS_RATE_NOTE, "no way back to parity"
    assert "~9 Hz" not in inspect.getsource(M)


def test_the_parser_rejects_policy_ckpt_and_observe_preselects_low_none():
    """2026-09-11 (operator request): the checkpoint is picked in the panel,
    so ``--policy-ckpt`` is no longer a flag; ``--policy-observe`` stays as
    the launch ALIAS of LOW level = None (``--mpc-mode none``), refuses to
    disagree with a typed controller, and keeps its own refusals."""
    import contextlib
    import io
    from rov_gui.__main__ import build_parser, check_policy, resolve_defaults

    p = build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            p.parse_args(["--policy", "--policy-ckpt", "stub"])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("--policy-ckpt was accepted by the parser")
    assert not hasattr(p.parse_args(["--policy"]), "policy_ckpt")
    # none is a LOW level choice; the alias preselects it
    assert p.parse_args(["--mpc-mode", "none"]).mpc_mode == "none"
    a = resolve_defaults(p.parse_args(["--policy", "--mpc", "--policy-observe"]))
    assert a.mpc_mode == "none"
    assert resolve_defaults(a).mpc_mode == "none"          # idempotent
    assert check_policy(a) is None, check_policy(a)
    # the alias and an explicit none agree
    a = resolve_defaults(p.parse_args(
        ["--policy", "--mpc", "--policy-observe", "--mpc-mode", "none"]))
    assert a.mpc_mode == "none" and check_policy(a) is None
    # a typed controller contradicts the alias: refused, both named
    a = resolve_defaults(p.parse_args(
        ["--policy", "--mpc", "--policy-observe", "--mpc-mode", "dobmpc"]))
    assert a.mpc_mode == "dobmpc", "resolve_defaults must not overwrite a typed mode"
    lvl, msg = check_policy(a)
    assert lvl == "refuse" and "disagree" in msg and "dobmpc" in msg \
        and "--policy-observe" in msg, (lvl, msg)
    # a typed controller without the alias is untouched
    a = resolve_defaults(p.parse_args(["--mpc", "--mpc-mode", "pid"]))
    assert a.mpc_mode == "pid" and check_policy(a) is None
    # --mpc-mode none alone is a harmless idle station: no refusal
    assert check_policy(resolve_defaults(p.parse_args(["--mpc", "--mpc-mode", "none"]))) is None
    # the alias keeps its own refusals (needs --policy, needs --mpc)
    lvl, msg = check_policy(resolve_defaults(p.parse_args(["--policy-observe"])))
    assert lvl == "refuse" and "--policy" in msg, (lvl, msg)
    lvl, msg = check_policy(resolve_defaults(p.parse_args(["--policy", "--policy-observe"])))
    assert lvl == "refuse" and "--mpc" in msg, (lvl, msg)


def test_fallback_policy_block_max_run_s_is_the_operator_decision():
    """[결정: operator 2026-09-11 — 120 -> 500 s] in BOTH definitions the
    worker can fly on (geometry.default_policy_block is the source of truth,
    the backend block the fallback); the YAMLs are pinned by
    test_policy_ckpt_paths.py."""
    from rov_gui.backends.policy import _FALLBACK_POLICY_BLOCK
    from rov_gui.control.geometry import default_policy_block

    assert _FALLBACK_POLICY_BLOCK["max_run_s"] == 500.0
    assert default_policy_block()["max_run_s"] == _FALLBACK_POLICY_BLOCK["max_run_s"], (
        default_policy_block()["max_run_s"], _FALLBACK_POLICY_BLOCK["max_run_s"])


def test_policy_status_carries_the_checkpoint_after_setup():
    """PolicyStatus.ckpt is the session the worker HOLDS (the panel picker's
    confirmed name, 2026-09-11), populated by setup's forced publish; the
    sha1 head is "" for the stub; no refusal note; the meta says the launch
    checkpoint flew."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        assert statuses, "setup must publish a status"
        st = statuses[-1]
        assert st.ckpt == "stub" and st.ckpt_sha1 == "" and st.ckpt_note == ""
        assert st.ready and not st.loading
        m = w.meta()
        assert m["ckpt_swaps"] == 0 and m["ckpt_source"] == "launch"
        assert w.counters["ckpt_swaps"] == 0


def test_set_ckpt_swaps_when_idle_and_is_a_no_op_on_the_same_path():
    """The panel pick (bus.cmd_policy_ckpt -> set_ckpt, 2026-09-11): when
    nothing is armed the old session is CLOSED and a NEW one is built through
    the factory with ``ckpt=path``, pc["ckpt"] and the status follow it, the
    swap is counted; the same path (also through a symlink) is a no-op that
    republishes the status; and the stub can be picked back."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        first = w.session
        f = Path(tmp) / "picked.ckpt"
        f.write_bytes(b"not a torch pickle; the stub factory never opens it")
        w.set_ckpt(str(f))
        assert w.session is not first, "no new session was built"
        assert not first.ready, "the old session must be closed"
        assert w.pc["ckpt"] == str(f) and w.session.ckpt == str(f)
        st = statuses[-1]
        assert st.ckpt == str(f) and st.ckpt_note == "" and st.ready, st
        assert w.counters["ckpt_swaps"] == 1
        assert any(lvl == "info" and "picked.ckpt" in msg
                   and "chosen in the panel" in msg for lvl, msg in logs), logs[-3:]
        m = w.meta()
        assert m["ckpt_swaps"] == 1 and m["ckpt_source"] == "panel"
        assert m["session"]["ckpt"] == str(f)
        # the worker keeps working on the new session: a plan flows
        t = now()
        _feed_history(w, t)
        _frame(w, mb, t - OBS_DT - 0.005, seed=1)
        _frame(w, mb, t - 0.005, seed=2)
        w.tick()
        assert len(plans) == 1
        # same path -> no swap, status republished, no note
        second = w.session
        n_log, n_st = len(logs), len(statuses)
        w.set_ckpt(str(f))
        assert w.session is second and w.counters["ckpt_swaps"] == 1
        assert len(statuses) > n_st and statuses[-1].ckpt_note == ""
        assert any("already the loaded checkpoint" in msg for _, msg in logs[n_log:])
        # ...also through a symlink (selected.ckpt -> the file)
        link = Path(tmp) / "selected.ckpt"
        link.symlink_to(f)
        w.set_ckpt(str(link))
        assert w.session is second and w.counters["ckpt_swaps"] == 1
        # a stale refusal note does not outlive a same-path pick
        w._ckpt_note = "stale"
        w.set_ckpt(str(f))
        assert statuses[-1].ckpt_note == ""
        # the mission is over: a disengaged, fresh state. (_feed_history fed
        # engaged=True states, under which a swap is refused "DISENG first"
        # — the refusals test — while the same-path no-ops above still ran,
        # because step 3 precedes step 4.)
        import dataclasses
        w.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        # and back to the stub (a second swap)
        w.set_ckpt("stub")
        assert w.session is not second and w.session.ckpt == "stub"
        assert not second.ready and w.counters["ckpt_swaps"] == 2
        assert statuses[-1].ckpt == "stub" and w.pc["ckpt"] == "stub"
        assert w.meta()["ckpt_source"] == "panel"


def test_set_ckpt_refusals_name_their_reason_in_the_status():
    """Every refusal is logged at warn AND reaches PolicyStatus.ckpt_note (the
    panel shows it in red), in the §10.2 order: a stale PolicyState (silent
    controller) before an engaged one (DISENG first), a session still
    loading, a path that is not a file, a torn-down worker; an empty pick is
    ignored silently; a disengaged fresh state allows the swap."""
    import dataclasses
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        f = Path(tmp) / "a.ckpt"
        f.write_bytes(b"x")
        g = Path(tmp) / "b.ckpt"
        g.write_bytes(b"y")

        def refused(path, needle):
            held, swaps = w.session, w.counters["ckpt_swaps"]
            n_st, n_log = len(statuses), len(logs)
            w.set_ckpt(path)
            assert w.session is held and w.counters["ckpt_swaps"] == swaps, needle
            assert len(statuses) > n_st, f"{needle}: no status was published"
            note = statuses[-1].ckpt_note
            assert "REFUSED" in note and needle in note, (needle, note)
            assert any(lvl == "warn" and needle in msg for lvl, msg in logs[n_log:]), \
                (needle, logs[n_log:])

        # not a file (no controller state at all: that alone would allow it)
        refused(str(Path(tmp) / "missing.ckpt"), "not a file")
        # an ENGAGED state, active mission -> DISENG first
        w.on_policy_state(_state(now(), (0.0,) * 6, active=True))
        assert w._active()
        refused(str(f), "DISENG")
        # engaged but not active (a bare HOLD): still refused
        w.on_policy_state(_state(now(), (0.0,) * 6, active=False))
        assert not w._active()
        refused(str(f), "DISENG")
        # a STALE state is "controller silent", checked before engaged
        w.on_policy_state(_state(now(), (0.0,) * 6, active=True,
                                 stamp=now() - 5.0))
        refused(str(f), "silent")
        # a disengaged, fresh state allows the swap (nothing can be armed)
        w.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        held = w.session
        w.set_ckpt(str(f))
        assert w.session is not held and w.counters["ckpt_swaps"] == 1
        assert statuses[-1].ckpt == str(f) and statuses[-1].ckpt_note == ""
        # a session still loading -> wait for READY or ERROR
        w.session.loading = True
        refused(str(g), "still loading")
        assert "a.ckpt" in statuses[-1].ckpt_note
        w.session.loading = False
        # an empty pick is ignored silently: no status, no swap, no note
        n_st = len(statuses)
        held = w.session
        w.set_ckpt("   ")
        assert len(statuses) == n_st and w.session is held
        # a torn-down worker refuses (and keeps its session for meta())
        w.teardown()
        assert w.session is held and w._torn_down
        refused(str(g), "not running")
        assert w.meta()["session"]["ckpt"] == str(f)


def test_set_ckpt_a_factory_that_raises_leaves_an_honest_error():
    """The old session is already closed when the factory runs; a factory
    that raises must not raise out of the slot — the status says ERROR, the
    note says why, and the NEXT pick starts from `session is None`."""
    from pathlib import Path
    from rov_gui.state import Conn

    with tempfile.TemporaryDirectory() as tmp:
        calls = []

        def factory(pc, o, ckpt=None):
            calls.append(ckpt)
            if ckpt and ckpt.endswith("bad.ckpt"):
                raise ValueError("weights must be one of ...")
            return StubPolicySession(ckpt or "stub", "",
                                     dataset_fps=float(pc["dataset_fps"]))

        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp, factory=factory)
        assert calls == [None], "setup calls the factory without ckpt"
        bad = Path(tmp) / "bad.ckpt"
        bad.write_bytes(b"x")
        first = w.session
        w.set_ckpt(str(bad))
        assert w.session is None and not first.ready
        st = statuses[-1]
        assert st.error and "ValueError" in st.error and st.conn == Conn.FAULT
        assert st.ckpt == str(bad) and "bad.ckpt" in st.ckpt_note
        assert w.counters["ckpt_swaps"] == 0
        w.tick()                                   # a tick with no session is harmless
        # the next pick recovers
        w.set_ckpt("stub")
        assert w.session is not None and w.session.ready
        assert statuses[-1].error == "" and statuses[-1].ckpt_note == ""
        assert w.counters["ckpt_swaps"] == 1


def test_fs_settings_mismatch_names_each_knob():
    from rov_gui.backends.policy import fs_settings_mismatch, training_depth_source

    from rov_gui.backends.policy import effective_fstereo_ckpt
    from rov_gui.perception.fstereo import DEFAULT_CKPT, DEFAULT_REPO

    class O:
        fstereo_iters = 8
        fstereo_scale = 0.5
        fstereo_size = None
        fstereo_ckpt = None
        fstereo_repo = None
    # the EFFECTIVE checkpoint: the session default when none is typed
    os.environ.pop("FOUNDATION_STEREO_REPO", None)
    default_ck = effective_fstereo_ckpt(O())
    assert default_ck == Path(DEFAULT_REPO) / DEFAULT_CKPT
    src = {"status": "ok", "iters": 16, "scale": 1.0, "checkpoint": str(default_ck)}
    bad = fs_settings_mismatch(O(), src)
    assert len(bad) == 2 and any("iters" in b for b in bad) \
        and any("scale" in b for b in bad), bad
    O.fstereo_iters, O.fstereo_scale = 16, 1.0
    assert fs_settings_mismatch(O(), src) == []
    # a training store built from ANOTHER checkpoint is a mismatch even with
    # --fstereo-ckpt left at its default (verify 2026-09-02)
    bad = fs_settings_mismatch(O(), dict(src, checkpoint="/x/other.pth"))
    assert len(bad) == 1 and bad[0].startswith("default fstereo ckpt"), bad
    O.fstereo_ckpt = "/x/other.pth"
    assert fs_settings_mismatch(O(), dict(src, checkpoint="/x/other.pth")) == []
    assert fs_settings_mismatch(O(), src)[0].startswith("--fstereo-ckpt"), \
        fs_settings_mismatch(O(), src)
    O.fstereo_ckpt = None
    O.fstereo_size = "224x224"
    assert any("size" in b for b in fs_settings_mismatch(O(), src))
    assert fs_settings_mismatch(O(), {"status": "no_hydra_config"}) == []
    assert training_depth_source("stub")["status"] == "stub"
    assert training_depth_source("/nonexistent/x.ckpt")["status"] == "no_hydra_config"


# ------------------------------------------------------------ the wiring
def _init_source(cls) -> str:
    import inspect
    return inspect.getsource(cls.__init__)


def test_both_backends_wire_the_policy_hops_the_same_way():
    """An AST scan of the two backends' ``__init__`` (the test_offline
    LoopWorker-slot pattern): both build the worker only under --policy,
    both hand its mailbox to their depth producer, both go through ONE
    wiring helper — so the hops cannot drift apart — and that helper makes
    every connection the spec names."""
    import inspect

    from rov_gui.backends import demo, hardware

    import textwrap

    for cls in (demo.DemoBackend, hardware.HardwareBackend):
        src = inspect.getsource(cls.__init__)
        tree = ast.parse(textwrap.dedent(src))
        calls = {ast.unparse(n.func) for n in ast.walk(tree)
                 if isinstance(n, ast.Call)}
        assert any(c.endswith("PolicyWorker") for c in calls), cls.__name__
        assert any(c.endswith("PolicyMailbox") for c in calls), cls.__name__
        assert any(c.endswith("wire_policy") for c in calls), cls.__name__
        assert 'getattr(opts, "policy", False)' in src or \
            "getattr(opts, 'policy', False)" in src, cls.__name__
        assert ".policy_mb = pmb" in src, cls.__name__
    src = inspect.getsource(hardware.wire_policy)
    for hop in ("bus.policy_state.connect(policy.on_policy_state)",
                "bus.policy_plan.connect(mpc.on_policy_plan)",
                "bus.policy_status.connect(mpc.on_policy_status)",
                "bus.cmd_gripper_drive.connect(mpc.on_gripper_drive)",
                "mpc.policy_present = True",
                "mpc.policy_meta_fn = policy.meta",
                "policy.failed.connect("):
        assert hop in src, hop
    # the hardware backend feeds the mailbox from exactly one producer
    hw = inspect.getsource(hardware.HardwareBackend.__init__)
    assert "self.fstereo.policy_mb = pmb" in hw
    assert "self.video.policy_mb = pmb" in hw
    assert "policy_allow_device_depth" in hw


def test_wiring_runs_against_the_demo_backend_and_routes_a_failure():
    """The helper on real objects: MpcWorker gets the flags and the hook,
    and a worker failure lands in the controller's status as an error."""
    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        class Opts(_Opts):
            fps = 15.0
            ui_fps = 30.0
            thrusters = 8
            rec_dir = tmp
            rec_fps = 12.0
            fullscreen = False
            joystick = "none"
            nav_config = _test_opts(tmp).nav_config
            mpc_config = _test_opts(tmp).mpc_config
        win = MainWindow(Opts())
        be = make_backend("demo", win.bus, win.mailboxes, Opts())
        assert be.policy is not None and be.mpc is not None
        assert be.mpc.policy_present is True
        assert be.mpc.policy_meta_fn == be.policy.meta
        assert be.video.policy_mb is be.policy.mailbox
        got = []
        win.bus.policy_status.connect(got.append)
        be.policy.failed.emit("dp-policy: boom")
        app.processEvents()
        assert got and got[-1].error == "dp-policy: boom" and got[-1].conn is Conn.FAULT
        win.close()


def test_a_demo_run_meta_never_claims_a_real_vehicle():
    """v2 A15: the meta `source` derives from the backend."""
    from rov_gui.tests.test_control import _worker as _mpc_worker

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _mpc_worker(tmp)       # opts.source == "demo"
        m = w._run_meta("test")
        assert "real vehicle" not in m["source"], m["source"]
        assert "SYNTHETIC" in m["source"]
        assert m["policy"]["synthetic"] is True


# ------------------------------------------------------------------ runner
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# THE 6-DoF VARIANT (2026-09-26): the 7-dim stub contract on plan + status,
# and the pos_rpy_width mount rule (yaw-axis keys REQUIRED, convention pinned)
# =============================================================================
def test_seven_dim_stub_contract_rides_on_plan_and_status():
    """A pos_rpy_width stub (stub7 / stub_rp spellings) emits (16, 7) and
    the checkpoint contract rides on the plan AND the status exactly as the
    5-dim one does; a stub whose mount keys match hw_nav is mount_ok, so
    the controller's arm check (test_policy) can pin it."""
    from rov_gui.control.geometry import NavConfig
    from rov_gui.state import ACTION_REPR_POS_RPY_WIDTH

    class _Mount7(StubPolicySession):
        def __init__(self, R, *a, **kw):
            super().__init__(*a, action_repr=ACTION_REPR_POS_RPY_WIDTH, **kw)
            self._contract["yaw_axis_R_frd_cam"] = R
            self._contract["yaw_axis_cam_tilt_deg"] = 43.3

    with tempfile.TemporaryDirectory() as tmp:
        R_bc = NavConfig.load(_test_opts(tmp).nav_config).R_t_frd_cam("main")[0]
        R_bc = np.asarray(R_bc, float).reshape(3, 3)
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _Mount7(
                R_bc.tolist(), "stub7", "", dataset_fps=float(pc["dataset_fps"]),
                pitch_ramp_deg=5.0))
        assert w._mount_ok is True, w._mount_why
        c = w.session.contract
        assert c["action_repr"] == "pos_rpy_width" and c["action_dim"] == 7
        assert c["rpy_convention"]["columns"][3:6] == ["dyaw", "droll", "dpitch"]
        t = now()
        _feed_history(w, t)
        t_new = t - 0.005
        _frame(w, mb, t_new - OBS_DT, seed=1)
        _frame(w, mb, t_new, seed=2)
        w.tick()
        assert len(plans) == 1
        p = plans[0]
        assert p.action.shape == (16, 7) and p.action.dtype == np.float32
        assert p.action_repr == "pos_rpy_width"
        assert abs(float(p.action[-1, 5]) - np.radians(5.0)) < 1e-6
        assert float(np.abs(p.action[:, 4]).max()) == 0.0
        assert statuses[-1].action_repr == "pos_rpy_width"
        assert statuses[-1].mount_ok is True and statuses[-1].mount_why == ""


def test_pos_rpy_width_without_yaw_axis_keys_or_convention_is_refused():
    """I11: for pos_rpy_width the yaw-axis keys are REQUIRED (no legacy
    permissive path — roll, pitch and yaw are all labelled through R_bt)
    and rpy_convention must be the station's; the status says so and the
    arm check in the controller refuses on mount_ok False."""
    from rov_gui.control.geometry import NavConfig
    from rov_gui.state import ACTION_REPR_POS_RPY_WIDTH

    class _Bare7(StubPolicySession):
        def __init__(self, *a, conv="keep", R="none", **kw):
            super().__init__(*a, action_repr=ACTION_REPR_POS_RPY_WIDTH, **kw)
            if R == "none":
                self._contract["yaw_axis_R_frd_cam"] = None
                self._contract["yaw_axis_cam_tilt_deg"] = None
            else:
                self._contract["yaw_axis_R_frd_cam"] = R
                self._contract["yaw_axis_cam_tilt_deg"] = 43.3
            if conv != "keep":
                self._contract["rpy_convention"] = conv

    with tempfile.TemporaryDirectory() as tmp:
        R_bc = NavConfig.load(_test_opts(tmp).nav_config).R_t_frd_cam("main")[0]
        R_bc = np.asarray(R_bc, float).reshape(3, 3).tolist()
        # (a) no yaw-axis keys: refused, naming them
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _Bare7(
                "stub7", "", dataset_fps=float(pc["dataset_fps"])))
        assert w._mount_ok is False and "yaw_axis" in w._mount_why, w._mount_why
        w.tick()
        assert statuses[-1].mount_ok is False and "yaw_axis" in statuses[-1].mount_why
        assert sum("MOUNT MISMATCH" in m and lvl == "error"
                   for lvl, m in logs) == 1, logs
        # (b) the keys match but the convention is another: refused
        bad = {"order": "xyz", "frame": "body_frd_via_R_bt",
               "columns": ["dx", "dy", "dz", "dyaw", "droll", "dpitch", "width"]}
        w, *_ , statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _Bare7(
                "stub7", "", dataset_fps=float(pc["dataset_fps"]),
                conv=bad, R=R_bc))
        assert w._mount_ok is False and "rpy_convention" in w._mount_why
        # (c) keys + the station convention: ok
        w, *_ , statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _Bare7(
                "stub7", "", dataset_fps=float(pc["dataset_fps"]), R=R_bc))
        assert w._mount_ok is True, w._mount_why
        # (d) a 5-dim stub without any keys keeps the pre-variant path: ok
        w, *_ = _worker(tmp)
        assert w._mount_ok is True

