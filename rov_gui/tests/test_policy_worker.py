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
* a ``--source demo`` run meta never says "real vehicle" (A15);
* ``--policy-fs-schedule`` (2026-10-01, perception/fs_gate.py): `free`
  builds no gate and the plan / status / meta still say so; `yield` wraps the
  forward in hold -> wait_idle -> release (released on every way out, no
  hold on a pre-forward skip, the same frames as `free`, never a heartbeat
  or a burst); `only` asks for a burst of two a lead time before the period
  ends and infers on the frames that ARRIVE after the request — never before
  the period has elapsed — and every way out of a burst or of the mission
  asks again or opens the gate; while NO forward could run (no fresh fix, no
  proprio history, no grid, no start pose) it asks for nothing, lets
  FoundationStereo run free and counts the skip as a `free` run does; a
  burst is charged per frame DELIVERED to the policy mailbox, so a failed
  inference or a frame the tap dropped does not spend it;
  ``FStereoWorker.tick`` asks the gate before it takes the pair, always ends
  a frame it began and reports each delivery; the CLI flag; ONE gate
  injected into both hardware workers.
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


def _worker(tmp, factory=None, *, run_dir_fn=None, grid=True):
    """A PolicyWorker set up on THIS thread with the stub session (or the
    ``factory`` given) and an identity grid declared on its mailbox; plus
    the bus and collectors. ``run_dir_fn`` (the controller's ``_run_dir``
    in the backends) also turns ``--record-depth`` on, so the worker builds
    a real DepthRecorder filing into whatever folder the callable names.
    ``grid=False`` declares NO grid: the mailbox is left for a producer to
    declare its own (the FoundationStereo tap's rect_left, ``_wired``)."""
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
    if grid:
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


def test_fs_ckpt_identity_survives_a_moved_checkout():
    """2026-09-30: the checkout moved into the repo (external/FoundationStereo)
    while every training store still names ~/Desktop/FoundationStereo/...
    (deleted) or a byte-identical copy elsewhere. Same weights must not read
    as a different checkpoint; a different model still must."""
    from rov_gui.backends.policy import fs_settings_mismatch, same_fstereo_ckpt
    from rov_gui.perception import fstereo

    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "old" / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
        new = Path(tmp) / "new" / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
        vits = Path(tmp) / "new" / "pretrained_models" / "11-33-40" / "model_best_bp2.pth"
        for p, data in ((old, b"vitl"), (new, b"vitl"), (vits, b"vits")):
            p.parent.mkdir(parents=True)
            p.write_bytes(data)
        assert same_fstereo_ckpt(old, new)              # same bytes, two paths
        assert not same_fstereo_ckpt(new, vits)         # another model
        new.write_bytes(b"retrained")                   # same id, other bytes:
        assert not same_fstereo_ckpt(old, new)          #   content decides
        new.write_bytes(b"vitl")
        old.unlink()                                    # the old path deleted:
        assert not same_fstereo_ckpt(old, new)          #   these bytes are not upstream's
        # an unstat-able store path (ENAMETOOLONG) counts as gone — never raises
        unstat = Path(tmp) / ("x" * 300) / "23-51-11" / "model_best_bp2.pth"
        assert not same_fstereo_ckpt(unstat, new)

        pinned = dict(fstereo.UPSTREAM_CKPT_SHA1)
        try:                                            # ... pretend they are
            fstereo.UPSTREAM_CKPT_SHA1["23-51-11/model_best_bp2.pth"] = \
                fstereo._sha1_head(new)
            assert same_fstereo_ckpt(old, new)          # the survivor is upstream's file
            assert same_fstereo_ckpt(unstat, new)
            assert not same_fstereo_ckpt(old, vits)     # ViT-S is not ViT-L
            del fstereo.UPSTREAM_CKPT_SHA1["23-51-11/model_best_bp2.pth"]
            assert not same_fstereo_ckpt(old, new)      # an unknown id is unverifiable
            fstereo.UPSTREAM_CKPT_SHA1["23-51-11/model_best_bp2.pth"] = \
                fstereo._sha1_head(new)

            class O:
                fstereo_iters = 16
                fstereo_scale = 1.0
                fstereo_size = None
                fstereo_ckpt = None
                fstereo_repo = str(Path(tmp) / "new")
            src = {"status": "ok", "iters": 16, "scale": 1.0, "checkpoint": str(old)}
            assert fs_settings_mismatch(O(), src) == []
            bad = fs_settings_mismatch(O(), dict(src, checkpoint=str(vits)))
            assert len(bad) == 1 and bad[0].startswith("default fstereo ckpt"), bad
        finally:
            fstereo.UPSTREAM_CKPT_SHA1.clear()
            fstereo.UPSTREAM_CKPT_SHA1.update(pinned)


def test_hydra_config_follows_the_published_symlink():
    """data/checkpoints/*.ckpt are symlinks into the run folder (2026-09-14
    layout); the .hydra beside the TARGET is the training config. Not
    following the link silently skipped the FS parity check on every launch
    from 2026-09-14 to 2026-09-30 (status no_hydra_config)."""
    from rov_gui.backends.policy import hydra_config_for

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "20260907" / "0907_113747_train_x"
        (run / ".hydra").mkdir(parents=True)
        (run / ".hydra" / "config.yaml").write_text("task: {}\n")
        (run / "checkpoints").mkdir()
        target = run / "checkpoints" / "epoch=0195.ckpt"
        target.write_bytes(b"ckpt")
        pub = Path(tmp) / "checkpoints"
        pub.mkdir()
        link = pub / "20260907_113747_x.ckpt"
        link.symlink_to(Path("..") / "20260907" / run.name / "checkpoints" / target.name)
        assert hydra_config_for(target) == run / ".hydra" / "config.yaml"
        assert hydra_config_for(link) is not None
        assert hydra_config_for(link).resolve() == (run / ".hydra" / "config.yaml").resolve()
        assert hydra_config_for(pub / "dangling.ckpt") is None


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
                "bus.jaw_drive_seen.connect(mpc.on_gripper_drive)",
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


# =============================================================================
# --policy-fs-schedule {free,yield,only} (2026-10-01): how FoundationStereo
# and the policy share the one GPU. perception/fs_gate.py is the ONE object
# the two workers share; its own state machine is pinned in test_fs_gate.py,
# and what follows pins what each WORKER does with it:
#
# * free (the default): no gate exists, nothing is called, and the plan /
#   status / meta still SAY free (the record boundary);
# * yield: hold -> observation build -> wait_idle -> forward -> release, the
#   release on every way out, no hold on a pre-forward skip, the same
#   frames / rows / skips as free, and no heartbeat or burst during a mission;
# * only: a burst of two is asked for a lead time before the period ends, the
#   attempt fires on the two frames that ARRIVE after the request and never
#   before the period has elapsed, the lead follows the measured burst time
#   (down at once, up by EMA), and every way out without a forward (skip,
#   timeout, leaving the mission) asks again or opens the gate —
#   FoundationStereo is never left waiting. While NO forward could run (no
#   usable grid, no proprio history, no fresh fix, no start pose) nothing is
#   asked for, FoundationStereo runs free and the skip is counted by the free
#   trigger; a burst already pending is resolved first;
# * a burst is charged per frame DELIVERED to the policy mailbox
#   (FsGate.fs_delivered), so a failed inference or a frame the tap dropped
#   does not spend it;
# * FStereoWorker.tick asks the gate BEFORE it takes the pair, always reports
#   the end of a frame it began (ran=False when no map came out), reports
#   each delivery, and discards between bursts;
# * the CLI flag and the one place the gate is built.
#
# There is no fake clock (policy.py / hardware.py import `now` by name), so
# stamps are built relative to now() and the few sleeps are short and real.
# =============================================================================
class _RecGate:
    """A recording stand-in with FsGate's policy-side surface. Every call is
    appended to ``log`` — a list a test may share with a session stub, so the
    gate calls and the forward land in ONE ordered list."""

    def __init__(self, mode="yield", log=None, wait=(1.5, False)):
        self.mode = mode
        self.log = [] if log is None else log
        self.wait = wait                  # what wait_idle() answers
        self.grant = True                 # what request_burst() answers
        self.burst_timeout_s = 0.6

    def hold(self):
        self.log.append("hold")

    def wait_idle(self, timeout=None):
        self.log.append("wait_idle")
        return self.wait

    def release(self):
        self.log.append("release")

    def set_active(self, active):
        self.log.append("active" if active else "inactive")

    def request_burst(self, n=2):
        self.log.append(f"burst{int(n)}")
        return self.grant

    def snapshot(self):
        return {"mode": self.mode, "recording_fake": True}

    def protocol(self):
        """The hold protocol's calls and the forward, in the order made."""
        return [c for c in self.log
                if c in ("hold", "wait_idle", "predict", "release")]


def _logging_session_factory(log, ctl=None):
    """A session factory whose stub appends "predict" to ``log`` when the
    forward runs. ``ctl`` (a dict the test keeps) may carry ``boom`` — text
    that makes the forward raise — and ``probe``, a callable run INSIDE the
    forward (to look at the gate from where the GPU would be busy)."""
    ctl = {} if ctl is None else ctl

    class _Stub(StubPolicySession):
        def predict(self, obs, **kw):
            log.append("predict")
            if ctl.get("probe") is not None:
                ctl["probe"]()
            if ctl.get("boom"):
                raise RuntimeError(ctl["boom"])
            return super().predict(obs, **kw)

    return lambda pc, o, ckpt=None: _Stub(                       # noqa: E731
        ckpt or "stub", "", dataset_fps=float(pc["dataset_fps"]))


def _stage(w, mb, *, seed=10, epoch=1):
    """Everything one inference needs, short of the tick (what ``_infer``
    does before it ticks): a fresh proprio history, two depth frames obs_dt
    apart, the period cleared. Returns the history's end stamp."""
    t = now()
    w._ring.clear()
    _feed_history(w, t, epoch=epoch)
    _frame(w, mb, t - OBS_DT - 0.005, seed=seed)
    _frame(w, mb, t - 0.005, seed=seed + 1)
    w._last_infer = 0.0
    return t


def _fix_now(w, t0, speed=0.05, **kw):
    """One more FRESH tag fix stamped now(), continuing the motion of
    ``_feed_history(w, t0)`` — what the controller's 20 Hz PolicyState keeps
    doing while a test sleeps. It keeps the mission active (a state older
    than 1 s counts as inactive) and the newest fix within FIX_LAG_TOL of a
    depth frame stamped now."""
    t = now()
    w.on_policy_state(_state(t, (speed * (t - t0), 0, 0.8, 0, 0, 0.1), **kw))
    return t


def _fs_frame(gate):
    """What FStereoWorker.tick does to the gate around ONE computed frame
    that reaches the policy mailbox, for the tests that have no stereo
    worker: begin (must be allowed), end (a map came out), DELIVERED (the
    tap put it — what a burst under `only` is charged for since round 2).
    The caller then puts the frame into the policy mailbox itself; the
    order of that put and ``fs_delivered`` is immaterial to the gate."""
    from rov_gui.perception.fs_gate import FS_GO

    verdict = gate.fs_begin()
    assert verdict == FS_GO, verdict
    gate.fs_end(ran=True)
    gate.fs_delivered()


def _adopt_grid(w, mb):
    """Under `only` a worker whose builder has no usable grid is BLOCKED
    (``_only_blocked``): it asks for nothing and FoundationStereo runs free,
    which is how the first frame reaches it in a real run — the builder
    adopts the mailbox's grid only when a frame is TAKEN. One frame, taken
    outside any mission, does that here; the ring and the arrival flag are
    then cleared so the frame takes part in nothing else."""
    _frame(w, mb, now() - 1.0, seed=99)
    assert w.builder.usable, w.builder.why
    w._ring.clear()
    w._frame_pending = False


class _FakeFsSession:
    """FStereoSession's surface as FStereoWorker.tick / _publish_state /
    meta read it, with no torch behind it. ``seen`` is the tag (the pixel
    value ``_fs_worker``'s put() paints) of every pair infer() was handed,
    ``spans`` the (start, end) of every call on the monotonic clock."""

    error = ""
    loading = False
    ready = True
    graph = False
    graph_requested = False
    load_seconds = 0.0

    def __init__(self):
        self.seen = []
        self.spans = []
        self.boom = None                  # text -> infer raises RuntimeError
        self.sleep_s = 0.0                # the "GPU time" of one frame
        self.on_infer = None              # called INSIDE infer (frame in flight)
        self.closed = False

    def infer(self, left, right, rig, out_size=None):
        t0 = now()
        self.seen.append(int(left[0, 0]))
        try:
            if self.on_infer is not None:
                self.on_infer()
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.boom:
                raise RuntimeError(self.boom)
            native = np.full((400, 640), 1000, np.uint16)
            return {"depth_mm": native.copy(), "depth_native": native,
                    "rect_left": left, "valid_native": 97.0, "valid_out": 90.0,
                    "filled_out": 0.0, "solve_ms": 1.0}
        finally:
            self.spans.append((t0, now()))

    def describe(self):
        return {"fake": True}

    def close(self):
        self.closed = True


def _fs_worker(f=500.0):
    """An FStereoWorker with the fake session: enabled, NOT set up (setup()
    would import torch and load weights), no display mailbox (the panel
    publish returns early), a policy mailbox to tap into. Returns
    ``(worker, stereo mailbox, session, policy mailbox, put)`` where
    ``put(tag)`` lands a mono pair painted ``tag`` and returns its capture
    stamp. ``f`` is the rig's rectified focal length in pixels: at the
    default 500 the rect_left grid covers only 72 % of the policy's obs
    crop and the BUILDER refuses it (fine for the tap alone); f=300 covers
    it fully, for the tests that wire the real tap into a real
    PolicyWorker (``_worker(..., grid=False)``)."""
    from types import SimpleNamespace

    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.bus import StereoMailbox

    _app()
    smb = StereoMailbox()
    fsw = FStereoWorker(DataBus(), smb, _Opts(), {})
    fsw.session = _FakeFsSession()
    fsw.enabled = True
    smb.set_wanted(True)
    pmb = PolicyMailbox()
    pmb.set_wanted(True)
    fsw.policy_mb = pmb
    f = float(f)
    rig = SimpleNamespace(R1=np.eye(3),
                          P1=np.array([[f, 0, 320, 0], [0, f, 200, 0],
                                       [0, 0, 1, 0]]),
                          mono_size=(640, 400), alpha=0.5, provenance={})

    def put(tag):
        t = now()
        img = np.full((400, 640), int(tag), np.uint8)
        smb.put(img, img.copy(), rig, t, frame_seq=int(tag))
        return t

    return fsw, smb, fsw.session, pmb, put


# ------------------------------------------------------------------- free
def test_free_schedule_has_no_gate_and_the_plan_status_and_meta_say_free():
    """--policy-fs-schedule free, the default: the worker holds NO gate
    (``fs_gate is None`` — every call site is behind that test, so nothing is
    called), its only-mode bookkeeping never moves, and the record still
    says which schedule flew: the plan (``fs_schedule`` "free", the wait
    fields None — never 0.0, which would read as "waited, and it took no
    time"), the status, and the always-written meta block. The plan's new
    stamps split its age: captured <= arrived here <= the attempt began <=
    emitted. ``pairing.trigger`` is the string it always was."""
    import json
    import math

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        assert w.fs_gate is None and w._burst is None
        sched0 = dict(w._sched_n)
        assert sched0 == {"bursts": 0, "burst_timeouts": 0, "burst_refused": 0}
        w._fs_idle()                               # a no-op without a gate
        _stage(w, mb)
        t_before = now()
        w.tick()
        t_after = now()
        assert len(plans) == 1, (w.counters, w._note)
        p = plans[0]
        assert p.fs_schedule == "free"
        assert p.fs_wait_ms is None and p.fs_wait_timeout is None
        assert p.fs_input == "", "no FoundationStereo feeds this worker"
        assert isinstance(p.depth_arrive_t, float) and math.isfinite(p.depth_arrive_t)
        assert p.depth_arrive_t >= p.obs_t
        assert p.depth_arrive_t == w._ring[-1]["t_arrive"]
        assert isinstance(p.trigger_t, float)
        assert t_before <= p.trigger_t <= t_after
        assert p.depth_arrive_t <= p.trigger_t <= p.t_emit
        assert statuses[-1].fs_schedule == "free"
        assert w.fs_gate is None and w._burst is None and w._sched_n == sched0
        m = w.meta()
        fs = m["fs_schedule"]
        assert fs["requested"] == "free" and fs["effective"] == "free"
        assert fs["gate"] is None and fs["why"] == ""
        assert fs["only_lead_s"] is None and fs["worker"] == sched0
        assert m["pairing"]["trigger"] == "depth-frame arrival after period_s"
        json.dumps(m, default=str)                 # it must serialise
        # TYPED but not in effect (the demo source, device depth: no stereo
        # worker to schedule): the meta says what was asked, what flew, why.
        for typed in ("yield", "only"):
            w.opts.policy_fs_schedule = typed
            m = w.meta()
            fs = m["fs_schedule"]
            assert fs["requested"] == typed and fs["effective"] == "free", fs
            assert fs["gate"] is None and "nothing to schedule" in fs["why"], fs
            assert m["pairing"]["trigger"] == "depth-frame arrival after period_s"
            json.dumps(m, default=str)
        w._publish_status(force=True)
        assert statuses[-1].fs_schedule == "free", "the status says what is IN EFFECT"


def test_the_plan_names_the_fstereo_input_only_when_fstereo_feeds_the_policy():
    """``PolicyPlan.fs_input`` ("scale S itN" / "size WxH itN"; an explicit
    size governs; the GRU iterations ride along since round 2 — iters is
    the other half of the network input) lets a plans.jsonl that two
    launches appended to be split by depth setting. "" when this run's
    depth is not FoundationStereo — a demo or device-depth plan must not
    claim a setting nothing used. An unreadable iters drops only the
    suffix; an unreadable scale drops the tag; neither raises."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        assert w.fs_gate is None and w.fstereo_meta_fn is None
        assert w._fs_input_tag() == ""
        w.fstereo_meta_fn = lambda: {}             # what HardwareBackend injects
        # _Opts: fstereo_scale 1.0, fstereo_iters 16
        assert w._fs_input_tag() == "scale 1 it16"
        w.opts.fstereo_scale = 0.75
        w.opts.fstereo_iters = 8                   # the --policy defaults
        assert w._fs_input_tag() == "scale 0.75 it8"
        w.opts.fstereo_size = "224X224"
        assert w._fs_input_tag() == "size 224x224 it8"
        w.opts.fstereo_iters = None                # not stated: no suffix
        assert w._fs_input_tag() == "size 224x224"
        w.opts.fstereo_iters = "many"              # unreadable: no suffix
        assert w._fs_input_tag() == "size 224x224"
        w.opts.fstereo_iters = 16
        w.opts.fstereo_size = None
        w.opts.fstereo_scale = None
        assert w._fs_input_tag() == ""
        w.opts.fstereo_scale = "fast"              # never raises
        assert w._fs_input_tag() == ""
        # a gate alone says the same (it exists only beside a stereo worker)
        w.fstereo_meta_fn = None
        w.opts.fstereo_scale = 0.5
        w.fs_gate = _RecGate("yield")
        assert w._fs_input_tag() == "scale 0.5 it16"
        _stage(w, mb)
        w.tick()
        assert len(plans) == 1 and plans[0].fs_input == "scale 0.5 it16"


# ------------------------------------------------------------------ yield
def test_yield_calls_hold_then_wait_idle_then_predict_then_release():
    """One inference under `yield`: the hold is taken BEFORE the observation
    build (CPU work the stereo frame in flight finishes under), the bounded
    wait comes just before the forward, the release after it. What the wait
    answered rides on the plan. No heartbeat is sent (round 2: set_active is
    `only`'s and a no-op under yield, so an active tick makes no call
    outside the hold protocol) and no burst is ever asked for (that is
    `only`)."""
    with tempfile.TemporaryDirectory() as tmp:
        log = []
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=_logging_session_factory(log))
        gate = _RecGate("yield", log, wait=(12.5, False))
        w.fs_gate = gate
        real_obs_for = w._obs_for

        def obs_for(entry):
            log.append("obs")
            return real_obs_for(entry)

        w._obs_for = obs_for
        _stage(w, mb, seed=10)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note)
        assert log == ["hold", "obs", "obs", "wait_idle", "predict",
                       "release"], log
        p = plans[0]
        assert p.fs_schedule == "yield"
        assert isinstance(p.fs_wait_ms, float) and p.fs_wait_ms == 12.5
        assert p.fs_wait_timeout is False
        assert statuses[-1].fs_schedule == "yield"
        # a wait that TIMED OUT is recorded as such, and is never a failure
        del log[:]
        gate.wait = (120.0, True)
        _stage(w, mb, seed=20)
        w.tick()
        assert len(plans) == 2 and w.counters["infer_errors"] == 0
        assert gate.protocol() == ["hold", "wait_idle", "predict", "release"], log
        assert plans[1].fs_wait_timeout is True and plans[1].fs_wait_ms == 120.0
        # ticks that do not attempt (the period is still running, or it
        # elapsed with no new frame) touch the gate not at all
        del log[:]
        w.tick()
        w._last_infer = 0.0
        w.tick()
        assert log == [], log
        assert len(plans) == 2


def test_yield_releases_on_every_exit_and_never_holds_on_a_pre_forward_skip():
    """The hold is taken once the pre-forward skips are passed and released
    in a ``finally``: when the forward raises, when the observation build
    raises, and when an UNGUARDED line between them raises (that one
    propagates out of tick() — TimerWorker._tick swallows it, so nobody
    else would release). A pre-forward skip (no pair, fix lag, stale depth)
    takes no hold at all. Run twice: against a recording gate (the exact
    calls) and against a real FsGate (it is not left held, and
    FoundationStereo may start a frame at once)."""
    import rov_gui.control.policy_frames as PF
    from rov_gui.perception.fs_gate import FS_GO, FsGate

    with tempfile.TemporaryDirectory() as tmp:
        for real in (False, True):
            log, ctl = [], {}
            w, bus, mb, plans, statuses, sensors, logs = _worker(
                tmp, factory=_logging_session_factory(log, ctl))
            gate = FsGate("yield") if real else _RecGate("yield", log)
            w.fs_gate = gate

            def after(what, want, holds):
                if real:
                    snap = gate.snapshot()
                    assert snap["held"] is False, what
                    assert snap["counters"]["holds"] == holds, (what, snap)
                    assert gate.fs_begin() == FS_GO, what
                    gate.fs_end(ran=False)
                else:
                    assert gate.protocol() == want, (what, log)
                    assert log.count("hold") == log.count("release"), (what, log)
                del log[:]

            # (a) the forward raises
            ctl["boom"] = "CUDA error: device-side assert"
            _stage(w, mb, seed=10)
            w.tick()
            ctl["boom"] = None
            assert not plans and w.counters["infer_errors"] == 1
            assert w._last_infer > 0.0, "the network was asked: the period is consumed"
            after("the forward raised",
                  ["hold", "wait_idle", "predict", "release"], 1)

            # (b) the observation build raises (before the wait)
            def bad_obs(entry):
                raise ValueError("warp failed")

            w._obs_for = bad_obs
            try:
                _stage(w, mb, seed=20)
                w.tick()
            finally:
                del w._obs_for                     # back to the class method
            assert not plans and w.counters["infer_errors"] == 2
            after("the observation build raised", ["hold", "release"], 2)

            # (c) an unguarded line raises: out of tick(), and still released
            real_lowdim = PF.lowdim_obs

            def bad_lowdim(*a, **kw):
                raise FloatingPointError("an unguarded line")

            PF.lowdim_obs = bad_lowdim
            try:
                _stage(w, mb, seed=30)
                try:
                    w.tick()
                except FloatingPointError:
                    pass
                else:
                    raise AssertionError("the unguarded exception must "
                                         "propagate out of tick()")
            finally:
                PF.lowdim_obs = real_lowdim
            assert not plans and w.counters["infer_errors"] == 2
            after("an unguarded line raised out of tick()",
                  ["hold", "release"], 3)

            # (d) pre-forward skips: no hold, hence nothing to release
            t = now()
            w._ring.clear()
            _feed_history(w, t)
            w._first_pending = False               # no duplicate allowed
            _frame(w, mb, t - 0.005, seed=40)      # ONE frame: no partner
            w._last_infer = 0.0
            w.tick()
            assert w.counters["skip_pair"] == 1 and not plans, w.counters
            after("skip_pair", [], 3)
            t = now()
            w._ring.clear()
            _feed_history(w, t)
            w._first_pending = False
            t_d = t + 0.75 * OBS_DT                # ahead of the newest fix
            _frame(w, mb, t_d - OBS_DT, seed=41)
            _frame(w, mb, t_d, seed=42)
            w._last_infer = 0.0
            w.tick()
            assert w.counters["skip_fix_lag"] == 1 and not plans, w.counters
            after("skip_fix_lag", [], 3)
            t = now()
            w._ring.clear()
            _feed_history(w, t)
            _frame(w, mb, t - 0.90, seed=43)       # older than obs_max_age_s
            _frame(w, mb, t - 0.83, seed=44)
            w._last_infer = 0.0
            w.tick()
            assert w.counters["skip_stale_depth"] == 1 and not plans, w.counters
            after("skip_stale_depth", [], 3)

            # ...and the worker is healthy afterwards: a plan, one more hold
            _stage(w, mb, seed=50)
            w.tick()
            assert len(plans) == 1, (w.counters, w._note)
            after("a plan", ["hold", "wait_idle", "predict", "release"], 4)


def test_yield_with_a_real_gate_waits_bounded_for_the_frame_in_flight_and_leaves_no_hold():
    """`yield` against the real FsGate. With nothing in flight the forward
    goes at once (``fs_wait_timeout`` False, a wait of ~0); a stereo frame
    in flight that ENDS during the wait is waited for; one that does not end
    within ``wait_s`` costs exactly that bound and the forward runs anyway
    (``fs_wait_timeout`` True — today's behaviour, never a failure). During
    the forward FoundationStereo is told "hold"; after the tick it is not."""
    import json
    import threading

    from rov_gui.perception.fs_gate import FS_GO, FS_HOLD, FsGate

    with tempfile.TemporaryDirectory() as tmp:
        log, ctl, seen = [], {}, []
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=_logging_session_factory(log, ctl))
        gate = FsGate("yield", wait_s=0.03)
        w.fs_gate = gate
        w.opts.policy_fs_schedule = "yield"

        # (a) nothing in flight
        ctl["probe"] = lambda: seen.append(gate.fs_begin())
        _stage(w, mb, seed=10)
        w.tick()
        ctl["probe"] = None
        assert len(plans) == 1, (w.counters, w._note)
        p = plans[0]
        assert seen == [FS_HOLD], \
            "FoundationStereo must not START a frame during the forward"
        assert p.fs_schedule == "yield" and p.fs_wait_timeout is False
        assert isinstance(p.fs_wait_ms, float) and 0.0 <= p.fs_wait_ms < 20.0, \
            p.fs_wait_ms
        assert gate.snapshot()["held"] is False

        # (b) a frame in flight that does not end: the wait is the bound
        assert gate.fs_begin() == FS_GO            # the stereo tick's mark
        _stage(w, mb, seed=20)
        t0 = now()
        w.tick()
        took = now() - t0
        assert len(plans) == 2, (w.counters, w._note)
        p = plans[1]
        assert p.fs_wait_timeout is True
        assert 29.0 <= p.fs_wait_ms < 1000.0, p.fs_wait_ms
        assert took >= 0.029, took
        assert p.infer_ms < 29.0, "the wait is its own number, not in infer_ms"
        assert w.counters["infer_errors"] == 0 and w.counters["plans"] == 2
        snap = gate.snapshot()
        assert snap["held"] is False
        assert snap["counters"]["holds"] == 2 and snap["counters"]["waits"] == 2
        assert snap["counters"]["wait_timeouts"] == 1
        # after the tick FoundationStereo is free to start again
        assert gate.fs_begin() == FS_GO

        # (c) the frame in flight (the fs_begin just above) ENDS during the wait
        gate.wait_s = 2.0
        _stage(w, mb, seed=30)
        timer = threading.Timer(0.1, gate.fs_end)
        timer.start()
        try:
            w.tick()
        finally:
            timer.join(5.0)
        assert len(plans) == 3, (w.counters, w._note)
        p = plans[2]
        assert p.fs_wait_timeout is False
        assert 10.0 < p.fs_wait_ms < 2000.0, p.fs_wait_ms
        snap = gate.snapshot()
        assert snap["held"] is False and snap["in_flight"] is False
        assert snap["counters"]["wait_timeouts"] == 1

        # the record
        assert statuses[-1].fs_schedule == "yield"
        m = w.meta()
        fs = m["fs_schedule"]
        assert fs["requested"] == "yield" and fs["effective"] == "yield"
        assert fs["why"] == "" and fs["only_lead_s"] is None
        assert fs["gate"]["mode"] == "yield" and fs["gate"]["counters"]["holds"] == 3
        assert fs["gate"]["wait_ms"]["n"] == 3 and fs["gate"]["hold_ms"]["n"] == 3
        assert m["pairing"]["trigger"] == "depth-frame arrival after period_s", \
            "yield keeps the free trigger"
        json.dumps(m, default=str)


def test_yield_picks_the_same_frames_rows_and_skips_as_free():
    """`yield` changes WHEN the forward runs, never WHAT it runs on: fed the
    same states and frames, a worker under a (real) yield gate makes the
    same decisions as one without — duplicate on the first frame, skip_pair,
    no attempt without an arrival, a fallback pair, a near pair — and emits
    plans with the same frames, rows, fix stamps and action."""
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        wf, _bf, mbf, plans_f, st_f, *_ = _worker(tmp)
        wy, _by, mby, plans_y, st_y, *_ = _worker(tmp)
        wy.fs_gate = FsGate("yield")
        both = ((wf, mbf), (wy, mby))
        t = now()
        for w, mb in both:                 # 1) first inference, ONE frame: dup
            _feed_history(w, t)
            _frame(w, mb, t - 0.01, seed=1)
            w.tick()
        for w, mb in both:                 # 2) no partner within 3 obs_dt: skip
            w._last_infer = 0.0
            _frame(w, mb, t - 0.01 + 0.25, seed=2)
            w.tick()
        for w, mb in both:                 # 3) period over, no arrival: nothing
            w._last_infer = 0.0
            w.tick()
        t2 = now()
        for w, mb in both:                 # 4) a fallback pair 0.145 s apart
            w._ring.clear()
            _feed_history(w, t2)
            _frame(w, mb, t2 - 0.15, seed=3)
            _frame(w, mb, t2 - 0.005, seed=4)
            w._last_infer = 0.0
            w.tick()
        t3 = now()
        for w, mb in both:                 # 5) a near pair
            w._ring.clear()
            _feed_history(w, t3)
            _frame(w, mb, t3 - OBS_DT - 0.005, seed=5)
            _frame(w, mb, t3 - 0.005, seed=6)
            w._last_infer = 0.0
            w.tick()
        assert len(plans_f) == 3, wf.counters
        assert len(plans_y) == 3, wy.counters
        assert wf.counters == wy.counters, (wf.counters, wy.counters)
        assert wf.counters["pair_dup"] == 1 and wf.counters["skip_pair"] == 1
        assert wf.counters["pair_fallback"] == 1 and wf.counters["pair_near"] == 1
        for pf, py in zip(plans_f, plans_y):
            for k in ("plan_id", "epoch", "obs_t", "obs_rows_t", "obs_fix_t",
                      "obs_pair_dt_s", "pair_dup", "depth_src",
                      "depth_coverage", "depth_valid", "obs_dt_s",
                      "action_repr", "ckpt_sha1"):
                assert getattr(pf, k) == getattr(py, k), (k, getattr(pf, k),
                                                          getattr(py, k))
            assert np.array_equal(pf.action, py.action)
            assert set(pf.lowdim) == set(py.lowdim)
            for k in pf.lowdim:
                assert np.array_equal(pf.lowdim[k], py.lowdim[k]), k
            assert pf.fs_schedule == "free" and py.fs_schedule == "yield"
            assert pf.fs_wait_ms is None and isinstance(py.fs_wait_ms, float)
            assert py.fs_wait_timeout is False
        assert st_f[-1].n_plans == st_y[-1].n_plans == 3
        assert st_f[-1].n_skip == st_y[-1].n_skip == 1


# ------------------------------------------------------------------- only
def test_only_requests_a_burst_and_infers_on_the_two_frames_that_arrive_after_it():
    """`only`, one period. FoundationStereo is idle between bursts, so
    nothing arrives unless the worker asks: the first tick of a due period
    REQUESTS two frames and emits nothing, even though a perfectly good
    pair already sits in the ring with the `free` trigger's arrival flag set
    (under `free` that same ring fires at once — the control below). Frames
    that arrive AFTER the request count; one is not a pair; with two there
    is exactly one plan, on those two frames. Afterwards the burst is
    forgotten, the gate's allowance is spent by the two DELIVERIES
    (FoundationStereo drains), the lead time is updated from the burst time
    it measured — exactly the round-2 rule (down at once, up by the 0.8/0.2
    EMA, inside its bounds) — and the next burst is asked for BEFORE the
    period ends."""
    import json

    from rov_gui.backends.policy import (ONLY_LEAD_MAX_S, ONLY_LEAD_MIN_S,
                                         ONLY_LEAD_S)
    from rov_gui.perception.fs_gate import FS_DRAIN, FsGate

    with tempfile.TemporaryDirectory() as tmp:
        # the control: this ring fires on the first tick under `free`
        wf, _bf, mbf, plans_f, *_ = _worker(tmp)
        t = now()
        _feed_history(wf, t)
        _frame(wf, mbf, t - 0.30, seed=1)
        _frame(wf, mbf, t - 0.22, seed=2)
        wf.tick()
        assert len(plans_f) == 1, wf.counters

        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        w.opts.policy_fs_schedule = "only"
        keys0 = set(w.counters)
        assert w._only_lead_s == ONLY_LEAD_S
        t0 = now()
        _feed_history(w, t0)
        _frame(w, mb, t0 - 0.30, seed=1)
        _frame(w, mb, t0 - 0.22, seed=2)
        assert w._frame_pending is True, "the `free` arrival flag is set"
        w._last_infer = 0.0
        w.tick()
        assert not plans, "frames from before the request must not fire an attempt"
        assert w._burst is not None and w._frame_pending is False
        t_req = float(w._burst["t"])
        snap = gate.snapshot()
        assert snap["mission_active"] is True
        assert snap["burst_allowance"] == 2 and snap["counters"]["bursts"] == 1
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 0}
        for _ in range(3):                         # nothing arrives: nothing fires
            w.tick()
        assert not plans and w._burst is not None
        assert float(w._burst["t"]) == t_req, "one request per burst"
        assert gate.snapshot()["counters"]["bursts"] == 1

        # FoundationStereo computes the two frames (both pairs captured
        # after the request), and each reaches the worker's mailbox.
        time.sleep(0.1)
        _fix_now(w, t0)
        cB = now() - 0.005
        cA = cB - 0.07
        assert cA > t_req
        _fs_frame(gate)
        _frame(w, mb, cA, seed=3)
        w.tick()
        assert not plans and w._burst is not None, "one frame is not a pair"
        assert gate.snapshot()["burst_allowance"] == 1, "one delivery, one charge"
        b = w._burst
        _fs_frame(gate)
        _frame(w, mb, cB, seed=4)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note)
        p = plans[0]
        assert b["t_done"] == p.trigger_t, \
            "the burst completed on the tick that made the attempt"
        assert p.fs_schedule == "only"
        assert p.obs_t == cB and p.pair_dup is False
        assert abs(p.obs_pair_dt_s - (cB - cA)) < 1e-9, \
            "the pair is the two frames of the burst"
        assert p.depth_arrive_t >= t_req and p.trigger_t >= p.depth_arrive_t
        assert isinstance(p.fs_wait_ms, float) and p.fs_wait_timeout is False
        assert w._burst is None and w._last_infer == p.trigger_t
        assert statuses[-1].fs_schedule == "only"
        assert gate.fs_begin() == FS_DRAIN, \
            "the burst is spent: FoundationStereo discards until asked again"
        snap = gate.snapshot()
        assert snap["held"] is False and snap["counters"]["burst_frames"] == 2
        assert snap["counters"]["delivered"] == 2 and snap["counters"]["frames"] == 2
        assert snap["burst_allowance"] == 0

        # the lead is updated from the measured burst time, by the round-2
        # rule: a burst shorter than the lead sets it at once, a longer one
        # moves it by the 0.8/0.2 EMA, inside [MIN, MAX]. (The burst here
        # took the 0.1 s sleep plus a few ticks: the "at once" branch.)
        lead = w._only_lead_s
        burst_s = p.trigger_t - t_req
        want = (burst_s if burst_s < ONLY_LEAD_S
                else 0.8 * ONLY_LEAD_S + 0.2 * burst_s)
        want = min(ONLY_LEAD_MAX_S, max(ONLY_LEAD_MIN_S, want))
        assert abs(lead - want) < 1e-12, (lead, want, burst_s)
        assert ONLY_LEAD_MIN_S <= lead <= ONLY_LEAD_MAX_S, lead

        # the record
        m = w.meta()
        fs = m["fs_schedule"]
        assert fs["requested"] == "only" and fs["effective"] == "only"
        assert fs["why"] == "" and fs["worker"]["bursts"] == 1
        assert fs["only_lead_s"] == round(lead, 3)
        assert fs["gate"]["mode"] == "only"
        trig = m["pairing"]["trigger"]
        assert "requested from FoundationStereo" in trig and "only" in trig, trig
        json.dumps(m, default=str)
        assert set(w.counters) == keys0, "the schedule added a counter key"

        # the next burst: not yet, then a lead time BEFORE the period ends
        period = float(w.pc["period_s"])
        w._last_infer = now() - 0.05
        w.tick()
        assert w._burst is None and w._sched_n["bursts"] == 1, "too early to ask"
        w._last_infer = now() - (period - lead) - 0.02
        assert now() - w._last_infer < period, "the period has NOT elapsed"
        w.tick()
        assert w._burst is not None
        assert w._sched_n == {"bursts": 2, "burst_timeouts": 0, "burst_refused": 0}
        assert len(plans) == 1


def test_only_the_lead_is_capped_below_a_short_period():
    """`only` with a ``period_s`` shorter than the lead time: the lead is
    capped below the period, so the next burst is NOT asked for the moment
    an attempt ended (the plans would then come faster than the config
    says) — it still waits for the first part of the period. (The worker
    is NOT blocked here — a usable grid, fresh fixes, a start pose — so the
    no-burst tick below is the lead cap, not the blocked path: the gate
    holds the mission heartbeat.)"""
    from rov_gui.backends.policy import ONLY_LEAD_MIN_S, ONLY_LEAD_S
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        period = 0.15
        assert period < ONLY_LEAD_S == w._only_lead_s
        w.pc["period_s"] = period
        _adopt_grid(w, mb)
        _feed_history(w, now())
        assert not w._only_blocked()
        w._last_infer = now() - 0.02               # an attempt just ended
        w.tick()
        assert w._burst is None and w._sched_n["bursts"] == 0, \
            "asked again the moment an attempt ended"
        snap = gate.snapshot()
        assert snap["counters"]["bursts"] == 0
        assert snap["mission_active"] is True, "the burst path ran (heartbeat)"
        # ...and still ahead of the period's end, by what is left of the lead
        # (capped at period - ONLY_LEAD_MIN_S, i.e. asked ONLY_LEAD_MIN_S in)
        w._last_infer = now() - ONLY_LEAD_MIN_S - 0.02
        assert now() - w._last_infer < period, "the period has NOT elapsed"
        w.tick()
        assert w._burst is not None and w._sched_n["bursts"] == 1


def test_only_a_skipped_timed_out_or_refused_burst_is_asked_for_again_without_new_counters():
    """`only`: every way out of a burst WITHOUT a forward leaves the period
    unstamped and the burst forgotten, so the next tick asks again —
    FoundationStereo is never left waiting for a request that will not come,
    and the worker never waits for frames nobody was asked for:
    (a) the attempt is refused by a status gate (the newest fix is not a
        fresh tag solution) — and since that refusal is one no depth frame
        can cure, the worker then asks for NOTHING (round 2: blocked, the
        gate opens, FoundationStereo runs free) until a fresh fix comes,
        and asks at once when it does;
    (b) the two frames are too far apart to pair (skip_pair — and no hold);
    (c) the burst does not complete within the gate's burst_timeout_s: it is
        dropped, asked for again, and the frame that DID arrive for the
        dropped burst does not count towards the new one;
    (d) the gate refuses the request (its previous allowance is still
        pending): counted, nothing armed;
    (e) the attempt raises out of tick() from an unguarded line: released,
        unstamped, asked again.
    (f) is the contrast: a FORWARD that raises did ask the network, so it
        consumes the period exactly as under `free`.
    All of it is counted in ``_sched_n`` (meta fs_schedule.worker): the
    worker's ``counters`` gain no key and no ``skip_*`` moves for a burst
    that merely timed out or was refused (PolicyStatus.n_skip sums them).
    Every worker here first takes one frame outside the mission
    (``_adopt_grid``): without a usable grid it would be blocked."""
    from rov_gui.perception.fs_gate import FS_GO, FsGate

    def skips(w):
        return {k: v for k, v in w.counters.items() if k.startswith("skip_")}

    with tempfile.TemporaryDirectory() as tmp:
        # (a) a _skip_reason skip after a COMPLETED burst
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        keys0 = set(w.counters)
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # burst 1
        assert w._burst is not None
        w.on_policy_state(_state(now(), (0.0, 0, 0.8, 0, 0, 0.1), fresh=False))
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=1)
        _fs_frame(gate)
        _frame(w, mb, now() - 0.005, seed=2)
        w.tick()                                   # complete -> skip_fresh
        assert not plans and w.counters["skip_fresh"] == 1, w.counters
        assert w._burst is None and w._last_infer == 0.0
        assert gate.snapshot()["counters"]["holds"] == 0
        w.tick()                                   # blocked: asks for nothing
        assert w._burst is None and w._sched_n["bursts"] == 1
        snap = gate.snapshot()
        assert snap["mission_active"] is False and snap["counters"]["bursts"] == 1
        assert gate.fs_begin() == FS_GO, "blocked: FoundationStereo runs free"
        gate.fs_end(ran=False)
        assert w.counters["skip_fresh"] == 1, "no arrival, no skip counted"
        _fix_now(w, t0)                            # a fresh fix
        w.tick()                                   # asked again, at once
        assert w._burst is not None
        assert w._sched_n == {"bursts": 2, "burst_timeouts": 0, "burst_refused": 0}
        assert gate.snapshot()["counters"]["bursts"] == 2
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=3)
        w.tick()
        assert not plans
        _fs_frame(gate)
        _fix_now(w, t0)                            # a fresh fix: this one flies
        _frame(w, mb, now() - 0.005, seed=4)
        w.tick()
        assert len(plans) == 1 and plans[0].fs_schedule == "only", w.counters
        assert set(w.counters) == keys0

        # (b) skip_pair: the burst's two frames are 0.25 s apart
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        _feed_history(w, now())
        w._first_pending = False                   # not the first inference
        w._last_infer = 0.0
        w.tick()                                   # burst 1
        t = now()
        _fs_frame(gate)
        _frame(w, mb, t - 0.25, seed=1)
        _fs_frame(gate)
        _frame(w, mb, t - 0.005, seed=2)
        w.tick()
        assert not plans and w.counters["skip_pair"] == 1, w.counters
        assert gate.snapshot()["counters"]["holds"] == 0, \
            "a pre-forward skip must not hold FoundationStereo"
        assert w._burst is None and w._last_infer == 0.0
        w.tick()
        assert w._burst is not None and w._sched_n["bursts"] == 2
        assert set(w.counters) == keys0

        # (c) the burst does not complete in time
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only", burst_timeout_s=0.25)
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        skips0 = skips(w)
        w._last_infer = 0.0
        w.tick()                                   # burst 1
        _fs_frame(gate)
        _frame(w, mb, now() - 0.005, seed=1)       # only ONE frame comes
        w.tick()
        assert w._burst is not None and not plans
        time.sleep(0.3)
        _fix_now(w, t0)
        w.tick()                                   # timed out: dropped
        assert w._burst is None and not plans
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 1, "burst_refused": 0}
        w.tick()                                   # asked again
        assert w._burst is not None
        assert w._sched_n == {"bursts": 2, "burst_timeouts": 1, "burst_refused": 0}
        c = gate.snapshot()["counters"]
        assert c["bursts"] == 2 and c["burst_void"] == 1, c
        assert skips(w) == skips0, "a burst timeout is not a skip_* counter"
        w._publish_status(force=True)
        assert statuses[-1].n_skip == 0
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=2)
        w.tick()
        assert not plans and w._burst is not None, \
            "the frame of the DROPPED burst must not count towards the new one"
        _fs_frame(gate)
        _fix_now(w, t0)
        _frame(w, mb, now() - 0.005, seed=3)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note)
        assert set(w.counters) == keys0

        # (d) the gate refuses: its allowance from the last request is still
        #     pending (nothing here told it a frame was DELIVERED — the only
        #     thing that spends it)
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        _feed_history(w, now())
        w._first_pending = False
        skips0 = skips(w)
        w._last_infer = 0.0
        w.tick()                                   # burst 1, granted
        t = now()
        _frame(w, mb, t - 0.25, seed=1)            # two arrivals the gate
        _frame(w, mb, t - 0.005, seed=2)           # never counted
        w.tick()                                   # complete -> skip_pair
        assert w._burst is None and w.counters["skip_pair"] == 1
        w.tick()                                   # asks; refused
        assert w._burst is None and not plans
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 1}
        assert gate.snapshot()["counters"]["bursts"] == 1
        assert set(w.counters) == keys0
        assert {k: v for k, v in skips(w).items() if k != "skip_pair"} \
            == {k: v for k, v in skips0.items() if k != "skip_pair"}
        m = w.meta()["fs_schedule"]
        assert m["worker"] == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 1}

        # (e) the attempt RAISES out of tick() (an unguarded line): the hold
        #     is released, nothing is stamped, and the next tick asks again
        import rov_gui.control.policy_frames as PF

        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # burst 1
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=1)
        _fs_frame(gate)
        _fix_now(w, t0)
        _frame(w, mb, now() - 0.005, seed=2)
        real_lowdim = PF.lowdim_obs

        def bad_lowdim(*a, **kw):
            raise FloatingPointError("an unguarded line")

        PF.lowdim_obs = bad_lowdim
        try:
            try:
                w.tick()
            except FloatingPointError:
                pass
            else:
                raise AssertionError("the exception must propagate out of tick()")
        finally:
            PF.lowdim_obs = real_lowdim
        snap = gate.snapshot()
        assert snap["held"] is False and snap["counters"]["holds"] == 1, snap
        assert w._burst is None and w._last_infer == 0.0 and not plans
        w.tick()
        assert w._burst is not None and w._sched_n["bursts"] == 2

        # (f) the FORWARD raises: the network was asked, so the period is
        #     consumed exactly as under `free` — no burst until the lead point
        ctl = {"boom": "CUDA error"}
        w, bus, mb, plans, statuses, *_ = _worker(
            tmp, factory=_logging_session_factory([], ctl))
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # burst 1
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=1)
        _fs_frame(gate)
        _fix_now(w, t0)
        _frame(w, mb, now() - 0.005, seed=2)
        w.tick()
        assert not plans and w.counters["infer_errors"] == 1
        assert w._burst is None and w._last_infer > 0.0
        assert gate.snapshot()["held"] is False
        w.tick()
        assert w._burst is None and w._sched_n["bursts"] == 1, \
            "a forward error consumes the period: no burst before the lead point"


def test_only_every_way_out_of_the_mission_opens_the_gate_and_forgets_the_burst():
    """`only` computes nothing unless asked, so a worker that stops asking
    must say so on EVERY path that leaves the mission — the gate's own
    heartbeat timeout is the backstop, not the mechanism: a PolicyState that
    is inactive, halted, disengaged or stale; a session that failed, is
    still loading, or is gone; a panel checkpoint swap; teardown. Each one
    leaves ``mission_active`` False, the allowance at 0, no burst pending,
    no hold — and FoundationStereo free to start a frame. While the mission
    runs, every tick heartbeats."""
    import dataclasses

    from rov_gui.perception.fs_gate import FS_GO, FsGate

    def arm(w, gate):
        """An active mission with a burst asked for and not yet arrived."""
        w._first_pending = True
        _feed_history(w, now())
        w._last_infer = 0.0
        w.tick()
        snap = gate.snapshot()
        assert w._burst is not None and snap["mission_active"] is True, snap
        assert snap["burst_allowance"] == 2

    def opened(w, gate, what):
        snap = gate.snapshot()
        assert w._burst is None, what
        assert snap["mission_active"] is False, (what, snap)
        assert snap["burst_allowance"] == 0 and snap["held"] is False, (what, snap)
        assert gate.fs_begin() == FS_GO, what      # runs free
        gate.fs_end(ran=False)

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)                         # else blocked: never armed
        # before any mission: free
        w.tick()
        opened(w, gate, "no PolicyState yet")

        states = {
            "inactive": lambda: _state(now(), (0.0,) * 6, active=False),
            "halted": lambda: _state(now(), (0.0,) * 6, halted=True),
            "disengaged": lambda: dataclasses.replace(
                _state(now(), (0.0,) * 6), engaged=False),
            "stale": lambda: _state(now(), (0.0,) * 6, stamp=now() - 1.5),
        }
        for what, make in states.items():
            arm(w, gate)
            gate.hold()                            # a hold that leaked, too
            w.on_policy_state(make())
            assert gate.snapshot()["mission_active"] is True, \
                "only the TICK tells the gate"
            w.tick()
            opened(w, gate, what)
            assert not plans

        # the session: failed / loading / gone
        arm(w, gate)
        w.session.error = "CUDA out of memory"
        w.tick()
        opened(w, gate, "session error")
        w.session.error = ""
        arm(w, gate)
        w.session.ready, w.session.loading = False, True
        w.tick()
        opened(w, gate, "session loading")
        w.session.ready, w.session.loading = True, False
        arm(w, gate)
        held = w.session
        w.session = None
        w.tick()
        opened(w, gate, "no session")
        w.session = held

        # a panel checkpoint swap (allowed: disengaged), with NO tick between
        # the DISENG state and the pick — the swap itself must open the gate
        arm(w, gate)
        w.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        assert gate.snapshot()["mission_active"] is True
        f = Path(tmp) / "picked.ckpt"
        f.write_bytes(b"x")
        w.set_ckpt(str(f))
        assert w.counters["ckpt_swaps"] == 1
        opened(w, gate, "set_ckpt")

        # teardown: this worker stops BEFORE the stereo worker
        arm(w, gate)
        w.teardown()
        opened(w, gate, "teardown")

    # the heartbeat: every tick of an active mission, in every branch of the
    # burst logic (waiting for the lead, asking, waiting for frames, the
    # attempt — which is still wrapped in the hold protocol under `only`)
    with tempfile.TemporaryDirectory() as tmp:
        log = []
        w, bus, mb, plans, *_ = _worker(tmp, factory=_logging_session_factory(log))
        gate = _RecGate("only", log)
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = now()                      # the period just began
        for _ in range(3):
            w.tick()
        assert log == ["active"] * 3, log
        w._last_infer = 0.0
        for _ in range(3):
            w.tick()
        assert log == ["active"] * 3 + ["active", "burst2", "active",
                                        "active"], log
        assert w._burst is not None and not plans
        del log[:]
        _frame(w, mb, now() - 0.075, seed=1)
        _fix_now(w, t0)
        _frame(w, mb, now() - 0.005, seed=2)
        w.tick()
        assert log == ["active", "hold", "wait_idle", "predict", "release"], log
        assert len(plans) == 1 and plans[0].fs_schedule == "only"
        del log[:]
        w.on_policy_state(_state(now(), (0.0,) * 6, active=False))
        w.tick()
        assert log == ["inactive", "release"], log


# ---------------------------------------------------------- FStereoWorker
def test_fstereo_tick_without_a_gate_takes_infers_and_taps():
    """FStereoWorker.tick with ``fs_gate`` None — the default, and every run
    without --policy: one pair in, one inference, the frame counter and the
    policy tap advance; an inference that raises is counted and the worker
    keeps running (the next pair is processed and the count resets); an
    empty mailbox computes nothing. The state chip carries no schedule tag
    and the meta says ``schedule: None``."""
    fsw, smb, s, pmb, put = _fs_worker()
    states = []
    fsw.bus.fstereo_state.connect(states.append)
    assert fsw.fs_gate is None
    fsw.tick()                                     # nothing waiting
    assert s.seen == [] and fsw._frames == 0
    put(1)
    fsw.tick()
    assert s.seen == [1] and fsw._frames == 1
    assert fsw._policy_tap["put"] == 1 and pmb.counters()["put"] == 1
    assert smb.counters() == {"put": 1, "taken": 1, "conflated": 0}
    a1 = fsw._last_arrival
    assert a1 > 0.0 and fsw._hz == 0.0, "one arrival is not a rate"
    s.boom = "bad frame"
    put(2)
    fsw.tick()
    assert s.seen == [1, 2] and fsw._infer_fails == 1
    assert fsw._frames == 1 and fsw._policy_tap["put"] == 1
    assert fsw.enabled is True and smb.wanted() is True, "one bad frame is not fatal"
    assert fsw._last_arrival == a1, "a failed frame is not an arrival"
    s.boom = None
    put(3)
    fsw.tick()
    assert s.seen == [1, 2, 3] and fsw._frames == 2 and fsw._infer_fails == 0
    assert fsw._policy_tap["put"] == 2
    # the rate without a gate is what it always was: the EMA of the ARRIVAL
    # interval (its first sample is 1/dt itself), not the gated window count
    a2 = fsw._last_arrival
    assert a2 > a1 and abs(fsw._hz - 1.0 / (a2 - a1)) <= 1e-9 * fsw._hz, \
        (fsw._hz, 1.0 / (a2 - a1))
    put(4)
    fsw.tick()
    a3 = fsw._last_arrival
    want = 0.8 * (1.0 / (a2 - a1)) + 0.2 * (1.0 / (a3 - a2))
    assert abs(fsw._hz - want) <= 1e-9 * want, (fsw._hz, want)
    for _ in range(3):                             # empty again
        fsw.tick()
    assert s.seen == [1, 2, 3, 4] and fsw._frames == 3
    assert not fsw._done_marks, "the gated-rate window is not fed without a gate"
    m = fsw.meta()
    assert m["schedule"] is None and m["frames"] == 3 and m["enabled"] is True
    assert m["measured_hz"] == round(fsw._hz, 2)
    assert states and all(st.schedule == "" for st in states)
    assert "[" not in states[-1].chip, states[-1].chip


def test_fstereo_tick_under_yield_leaves_the_pair_while_held_and_always_ends_its_frame():
    """FStereoWorker.tick under a `yield` gate. It asks BEFORE it takes
    (take() is destructive): while the policy holds, the pair stays in the
    mailbox — newest wins, so the frame started after the release is the
    freshest pair — and nothing is inferred; the tick returns at once (it
    must keep returning to the event loop). A frame it began is marked in
    flight DURING the inference and always reported ended: after a normal
    frame, when the inference raises, when there was no pair to start —
    and only a frame that produced a map counts in the gate's ``frames``
    (round 2: ``fs_end(ran=False)`` for a raise as for an empty tick), only
    one the tap put in the policy mailbox in ``delivered`` (under `yield`
    that charges no burst). A hold whose release never came is a lease:
    FoundationStereo resumes."""
    from rov_gui.perception.fs_gate import FsGate

    fsw, smb, s, pmb, put = _fs_worker()
    states = []
    fsw.bus.fstereo_state.connect(states.append)
    gate = FsGate("yield")
    fsw.fs_gate = gate
    during = []
    s.on_infer = lambda: during.append(gate.snapshot()["in_flight"])

    put(1)                                         # not held: as always
    fsw.tick()
    assert s.seen == [1] and fsw._frames == 1 and during == [True]
    snap = gate.snapshot()
    assert snap["in_flight"] is False and snap["counters"]["frames"] == 1
    assert snap["counters"]["delivered"] == 1 and pmb.counters()["put"] == 1
    assert snap["counters"]["burst_frames"] == 0, "yield has no bursts"

    gate.hold()                                    # the policy is inferring
    put(2)
    t0 = now()
    for _ in range(20):
        fsw.tick()
    assert now() - t0 < 0.4, "the stereo tick must never block on the gate"
    assert s.seen == [1], "no frame may START while the policy holds"
    assert smb.counters()["taken"] == 1, "the waiting pair must stay in the mailbox"
    assert gate.snapshot()["counters"]["held_polls"] == 20
    put(3)                                         # a newer pair replaces it
    fsw.tick()
    assert s.seen == [1] and smb.counters()["conflated"] == 1
    gate.release()
    fsw.tick()
    assert s.seen == [1, 3], "after the release: the FRESHEST pair"
    assert fsw._frames == 2 and fsw._policy_tap["put"] == 2

    s.boom = "bad frame"                           # the inference raises
    put(4)
    fsw.tick()
    s.boom = None
    assert s.seen == [1, 3, 4] and fsw._infer_fails == 1
    snap = gate.snapshot()
    assert snap["in_flight"] is False, "fs_end must be reported when infer raises"
    assert snap["counters"]["inflight_reset"] == 0
    assert snap["counters"]["frames"] == 2, "a raised inference is not a frame"
    assert snap["counters"]["delivered"] == 2, "...and nothing was delivered"
    fsw.tick()                                     # no pair: begun, nothing started
    snap = gate.snapshot()
    assert snap["in_flight"] is False and snap["counters"]["inflight_reset"] == 0
    assert snap["counters"]["frames"] == 2, "an empty tick is not a frame"
    assert snap["counters"]["delivered"] == 2 == pmb.counters()["put"]
    assert during == [True, True, True]

    gate.lease_s = 0.2
    gate.hold()                                    # ...and never released
    put(5)
    fsw.tick()
    assert s.seen == [1, 3, 4]
    time.sleep(0.25)                               # the lease runs out
    fsw.tick()
    assert s.seen == [1, 3, 4, 5], "a hold is a lease: it cannot stop FS for good"
    assert gate.snapshot()["counters"]["lease_expired"] == 1
    gate.release()

    # the record: the chip and the meta name the schedule; the rate is the
    # count of completions over the window (three frames here, the window
    # not yet full), not the EMA of 1/dt the ungated path keeps
    assert states[-1].schedule == "yield"
    fsw._publish_state()
    assert states[-1].schedule == "yield" and states[-1].chip.endswith("[yield]"), \
        states[-1].chip
    m = fsw.meta()
    assert m["schedule"]["mode"] == "yield" and m["frames"] == fsw._frames == 3
    marks = list(fsw._done_marks)
    assert len(marks) == 3, "one mark per COMPLETED frame (1, 3, 5)"
    if marks[-1] - marks[0] < fsw.GATED_HZ_WINDOW_S:
        assert abs(fsw._hz - 2.0 / (marks[-1] - marks[0])) < 1e-9, fsw._hz
    assert m["measured_hz"] == round(fsw._hz, 2)

    # a broken configuration (INFER_FAIL_LIMIT failures in a row) stops the
    # stereo worker for good — it must not leave a frame "in flight" behind
    # for the policy to wait on, and it asks the gate nothing afterwards
    s.boom = "broken configuration"
    for k in range(fsw.INFER_FAIL_LIMIT):
        put(10 + k)
        fsw.tick()
    assert fsw.enabled is False and fsw._infer_fails == fsw.INFER_FAIL_LIMIT
    snap = gate.snapshot()
    assert snap["in_flight"] is False and snap["counters"]["inflight_reset"] == 0
    assert snap["counters"]["frames"] == 3, "failed inferences are not frames"
    n_seen, n_polls = len(s.seen), snap["counters"]["held_polls"]
    gate.hold()
    put(99)
    fsw.tick()
    gate.release()
    assert len(s.seen) == n_seen
    assert gate.snapshot()["counters"]["held_polls"] == n_polls, \
        "a stopped stereo worker does not poll the gate"


def test_fstereo_tick_under_only_discards_between_bursts_and_computes_exactly_the_burst():
    """FStereoWorker.tick under an `only` gate. Outside a mission it runs
    free. With a mission active and no burst asked for it DISCARDS the
    waiting pair (so that a burst begins on a pair that arrived after it was
    requested) and computes nothing; after ``request_burst(2)`` exactly two
    pairs are computed and the third is discarded; a tick with no pair
    waiting does not spend the allowance. And it can never be stopped for
    good: when the heartbeat stops, or nobody asks for ``starve_s``, it runs
    free again."""
    from rov_gui.perception.fs_gate import FsGate

    fsw, smb, s, pmb, put = _fs_worker()
    states = []
    fsw.bus.fstereo_state.connect(states.append)
    gate = FsGate("only")
    fsw.fs_gate = gate

    put(1)                                         # no mission: free
    fsw.tick()
    assert s.seen == [1]

    gate.set_active(True)                          # a mission, nothing asked
    put(2)
    fsw.tick()
    assert s.seen == [1], "between bursts nothing is computed"
    c = gate.snapshot()["counters"]
    assert c["drained"] == 1 and smb.counters()["taken"] == 2, \
        "the waiting pair is discarded, not left for the burst"
    fsw.tick()                                     # nothing waiting
    assert gate.snapshot()["counters"]["drained"] == 1, "nothing to discard"

    assert gate.request_burst(2) is True           # the policy asks for two
    fsw.tick()                                     # no pair yet
    assert gate.snapshot()["burst_allowance"] == 2, \
        "a tick with no pair must not spend the allowance"
    for tag in (3, 4, 5):
        put(tag)
        fsw.tick()
    assert s.seen == [1, 3, 4], "exactly the two frames asked for"
    snap = gate.snapshot()
    assert snap["counters"]["drained"] == 2 and snap["counters"]["burst_frames"] == 2
    assert snap["burst_allowance"] == 0 and snap["in_flight"] is False
    assert fsw._frames == 3 and fsw._policy_tap["put"] == 3
    # the burst is charged per DELIVERY: frame 1 (no mission) was delivered
    # without a charge, 3 and 4 each spent one
    assert snap["counters"]["delivered"] == 3 and snap["counters"]["frames"] == 3
    assert states[-1].schedule == "only"
    fsw._publish_state()
    assert states[-1].chip.endswith("[only]"), states[-1].chip
    assert fsw.meta()["schedule"]["mode"] == "only"

    gate.set_active(False)                         # the mission is over: free
    put(6)
    fsw.tick()
    assert s.seen == [1, 3, 4, 6]

    # the heartbeat stops (the policy worker died mid-mission): free again
    fsw, smb, s, pmb, put = _fs_worker()
    gate = FsGate("only", heartbeat_s=0.2)
    fsw.fs_gate = gate
    gate.set_active(True)
    put(1)
    fsw.tick()
    assert s.seen == []
    time.sleep(0.25)
    put(2)
    fsw.tick()
    assert s.seen == [2], "no heartbeat for heartbeat_s: FoundationStereo runs free"

    # the heartbeat keeps coming but nobody asks for a frame: free again
    fsw, smb, s, pmb, put = _fs_worker()
    gate = FsGate("only", starve_s=0.2)
    fsw.fs_gate = gate
    gate.set_active(True)
    put(1)
    fsw.tick()
    assert s.seen == []
    time.sleep(0.25)
    gate.set_active(True)
    put(2)
    fsw.tick()
    assert s.seen == [2], "active but unasked for starve_s: it runs free"
    assert gate.snapshot()["counters"]["starve_free"] == 1


def test_gated_hz_reads_the_completion_rate_not_the_pair_spacing():
    """Under a schedule the arrivals are not evenly spaced (`only`: two
    frames, then a gap), and the EMA of 1/dt would read the pair's spacing
    or the gap as the rate — on the synthetic pattern below, 1/0.08 =
    12.5 Hz and 1/0.42 = 2.4 Hz [유도: this test's own timestamps].
    ``_gated_hz`` counts completions over the last GATED_HZ_WINDOW_S, the
    half-open window (t - w, t] since round 2 (closed at both ends it held
    one frame too many: a steady 15 fps read 15.5). So once the window is
    full the rate is EXACT for these streams: bursts of two every 0.5 s
    read 4.0, an even 10 Hz stream 10.0, the 15 fps camera 15.0; a
    completion exactly w old is out; and a stream too fast for the 64-mark
    buffer to span the window still reads its own rate. (All stamps here
    are synthetic, built as base + i / rate so that no float drift moves a
    stamp across the window edge.)"""
    fsw, *_ = _fs_worker()
    assert fsw.GATED_HZ_WINDOW_S == 2.0
    got = []
    for k in range(12):                            # 6 s of bursts
        for off in (0.0, 0.08):
            t = 100.0 + 0.5 * k + off
            got.append((t - 100.0, fsw._gated_hz(t)))
    full = [hz for t, hz in got if t >= 2.0]
    assert full == [4.0] * 16, full
    fsw._done_marks.clear()
    got = [fsw._gated_hz(200.0 + 0.1 * i) for i in range(60)]
    assert got[20:] == [10.0] * 40, got[20:]
    fsw._done_marks.clear()
    got = [fsw._gated_hz(300.0 + i / 15.0) for i in range(90)]
    assert got[30:] == [15.0] * 60, got[30:]
    # the window's edge: (500, 502] holds 501 and 502 — two frames in 2 s
    fsw._done_marks.clear()
    for t in (500.0, 501.0):
        fsw._gated_hz(t)
    assert fsw._gated_hz(502.0) == 1.0, list(fsw._done_marks)
    fsw._done_marks.clear()
    got = [fsw._gated_hz(600.0 + 0.02 * i) for i in range(300)]
    assert all(abs(hz - 50.0) < 1e-6 for hz in got[100:]), got[100:]
    fsw._done_marks.clear()
    assert fsw._gated_hz(400.0) == 0.0, "one completion is not a rate"


# ---------------------------------------------- the two workers, one gate
def test_only_one_gate_two_workers_on_one_thread_compute_exactly_what_the_policy_asks():
    """The REAL PolicyWorker and the REAL FStereoWorker around one FsGate
    (`only`), ticked alternately on this thread, joined by the REAL tap
    (FStereoWorker._tap_policy declares the rect_left grid on the policy's
    mailbox and puts each map — and reports each delivery to the gate,
    which is what a burst is charged for since round 2): outside a mission
    the stereo worker computes every pair, and the first one gives the
    policy its grid; the policy's first active tick asks for two; the
    stereo worker computes exactly the next two pairs; the policy infers on
    those two; every pair after that is discarded until the mission ends,
    and then it runs free again."""
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp, grid=False)
        fsw, smb, s, _pmb, put = _fs_worker(f=300.0)
        fsw.policy_mb = mb                         # the real tap, into w's mailbox
        gate = FsGate("only")
        w.fs_gate = gate
        fsw.fs_gate = gate

        def stereo(tag):
            """A pair lands and the stereo worker ticks once. Its capture
            stamp when it was COMPUTED (the tap then put its depth in the
            policy mailbox), None when it was not."""
            n = len(s.seen)
            t_cap = put(tag)
            fsw.tick()
            return t_cap if len(s.seen) > n else None

        w.tick()                                   # no mission
        assert stereo(1) is not None, "outside a mission: free"
        w.tick()                                   # taken: the grid is adopted
        assert w.builder.grid_kind == GRID_RECT_LEFT and w.builder.usable, \
            w.builder.why
        time.sleep(0.12)                           # frame 1 is no partner below

        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # the policy asks for two
        assert w._burst is not None and not plans
        cA = stereo(2)
        assert cA is not None
        w.tick()
        assert not plans
        time.sleep(0.06)
        cB = stereo(3)
        assert cB is not None
        _fix_now(w, t0)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note)
        p = plans[0]
        assert p.fs_schedule == "only" and p.obs_t == cB
        assert abs(p.obs_pair_dt_s - (cB - cA)) < 1e-9
        assert p.depth_src == GRID_RECT_LEFT and p.fs_input == "scale 1 it16"

        for tag in (4, 5, 6):                      # between bursts
            assert stereo(tag) is None
            w.tick()
        assert s.seen == [1, 2, 3] and len(plans) == 1
        c = gate.snapshot()["counters"]
        assert c["drained"] == 3 and c["burst_frames"] == 2 and c["bursts"] == 1, c
        assert c["frames"] == 3 and c["delivered"] == 3, c
        assert fsw._policy_tap == {"put": 3, "dropped_grid_refused": 0,
                                   "dropped_no_grid": 0}
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 0}

        w.on_policy_state(_state(now(), (0.0,) * 6, active=False))
        w.tick()                                   # the mission is over
        assert stereo(7) is not None and s.seen[-1] == 7, "free again"


def test_yield_on_two_threads_the_forward_never_overlaps_a_stereo_frame():
    """The point of `yield`, end to end: the REAL FStereoWorker.tick on its
    own thread running back to back (a pair always waiting; the fake session
    sleeps 30 ms a frame), the REAL PolicyWorker.tick on this one (the stub
    sleeps 20 ms a forward), one FsGate. No stereo inference may overlap a
    policy forward — the frame in flight is waited for, and none starts
    until the release. (A forward whose wait timed out is allowed to
    overlap; with these fake durations against the gate's 120 ms bound none
    should.) The stereo worker keeps computing in between."""
    import threading

    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        forwards = []

        class _Slow(StubPolicySession):
            def predict(self, obs, **kw):
                t0 = now()
                time.sleep(0.02)
                out = super().predict(obs, **kw)
                forwards.append((t0, now()))
                return out

        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=lambda pc, o, ckpt=None: _Slow(
                ckpt or "stub", "", dataset_fps=float(pc["dataset_fps"])))
        fsw, smb, s, _pmb, put = _fs_worker()
        fsw.policy_mb = None
        s.sleep_s = 0.03
        gate = FsGate("yield")
        w.fs_gate = gate
        fsw.fs_gate = gate
        stop = threading.Event()
        errs = []

        def stereo_thread():
            k = 0
            try:
                while not stop.is_set():
                    k += 1
                    put(k % 200)                   # a pair is always waiting
                    fsw.tick()
                    time.sleep(0.001)
            except Exception as e:                               # noqa: BLE001
                errs.append(e)

        th = threading.Thread(target=stereo_thread, daemon=True)
        th.start()
        try:
            for i in range(6):
                _stage(w, mb, seed=10 * (i + 1))
                w.tick()
                time.sleep(0.015)
        finally:
            stop.set()
            th.join(5.0)
        assert not th.is_alive() and not errs, errs
        assert len(plans) == 6 and len(forwards) == 6, (w.counters, w._note)
        assert len(s.spans) >= 6, "the stereo worker must keep computing"
        checked = 0
        for p, (a, b) in zip(plans, forwards):
            assert p.fs_schedule == "yield" and isinstance(p.fs_wait_ms, float)
            if p.fs_wait_timeout:
                continue
            checked += 1
            clash = [(fa, fb) for fa, fb in s.spans if fa < b and a < fb]
            assert not clash, (f"plan {p.plan_id}: forward {a:.4f}..{b:.4f} "
                               f"overlaps stereo frame(s) {clash}")
        assert checked >= 1
        snap = gate.snapshot()
        assert snap["held"] is False and snap["counters"]["holds"] == 6
        assert snap["counters"]["frames"] == len(s.spans)
        # ...and the test did exercise both halves: the stereo tick was turned
        # away during a forward, and a forward waited for a frame in flight
        assert snap["counters"]["held_polls"] >= 1, snap["counters"]
        assert any(p.fs_wait_ms > 0.5 for p in plans), [p.fs_wait_ms for p in plans]


def test_only_on_two_threads_plans_keep_coming_and_fstereo_computes_only_the_bursts():
    """`only`, end to end on two threads: a fake 15 fps camera and the REAL
    FStereoWorker.tick (the fake session sleeps 30 ms a frame) feeding the
    REAL tap on one, the REAL PolicyWorker.tick and a controller feeding
    fresh fixes on this one, one FsGate. Plans keep coming, never closer
    together than period_s (a lower bound in every schedule since round 2);
    each is built on two CONSECUTIVELY computed frames; while the mission
    runs every frame FoundationStereo computes was asked for and delivered
    (two per burst, each charged on delivery) and the pairs in between are
    discarded; when the mission ends it runs free again without being
    asked. (The first pair is computed outside the mission on this thread,
    so the policy has its grid before the mission starts — without one it
    would be blocked, which the blocked-path tests pin.)"""
    import threading

    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp, grid=False)
        fsw, smb, s, _pmb, put = _fs_worker(f=300.0)
        fsw.policy_mb = mb                         # the real tap, into w's mailbox
        s.sleep_s = 0.03
        gate = FsGate("only")
        w.fs_gate = gate
        fsw.fs_gate = gate
        period = float(w.pc["period_s"])
        caps = {}                                  # tag -> capture stamp
        computed = []                              # capture stamps, in order
        stop = threading.Event()
        errs = []

        # outside the mission: one pair computed free, its depth tapped and
        # taken — the policy adopts the rect_left grid
        caps[1] = put(1)
        fsw.tick()
        assert s.seen == [1]
        computed.append(caps[1])
        w.tick()
        assert w.builder.usable, w.builder.why

        def stereo_thread():
            tag, due = 1, now()
            try:
                while not stop.is_set():
                    if now() >= due:               # the camera, 15 fps
                        tag += 1
                        caps[tag] = put(tag)
                        due += 1.0 / 15.0
                    n = len(s.seen)
                    fsw.tick()
                    if len(s.seen) > n:            # computed (and tapped)
                        computed.append(caps[s.seen[-1]])
                    time.sleep(0.003)
            except Exception as e:                               # noqa: BLE001
                errs.append(e)

        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # active + the first request
        assert w._burst is not None
        snap0 = gate.snapshot()                    # the mission's baseline
        th = threading.Thread(target=stereo_thread, daemon=True)
        th.start()
        try:
            deadline = now() + 8.0
            while len(plans) < 3 and now() < deadline:
                _fix_now(w, t0)                    # the controller, ~100 Hz
                w.tick()
                time.sleep(0.01)
            snap = gate.snapshot()
            n_mission = len(computed)
            # the mission ends: nobody asks any more
            deadline = now() + 4.0
            while len(computed) < n_mission + 4 and now() < deadline:
                w.on_policy_state(_state(now(), (0.0,) * 6, active=False))
                w.tick()
                time.sleep(0.01)
            n_free = len(computed) - n_mission
        finally:
            stop.set()
            th.join(5.0)
        assert not th.is_alive() and not errs, errs
        assert len(plans) >= 3, (w.counters, w._sched_n, snap)
        c, c0 = snap["counters"], snap0["counters"]
        assert c0["frames"] == c0["delivered"] == 1 and c0["burst_frames"] == 0, c0
        d = {k: c[k] - c0[k] for k in ("frames", "delivered", "burst_frames")}
        assert d["frames"] == d["burst_frames"] == d["delivered"], \
            f"a frame was computed or delivered that nobody asked for: {c0} -> {c}"
        assert c["burst_frames"] <= 2 * c["bursts"], c
        assert c["drained"] >= 1, "the pairs between bursts must be discarded"
        for p in plans:
            assert p.fs_schedule == "only" and p.depth_src == GRID_RECT_LEFT
            if p.pair_dup:
                continue
            i = computed.index(p.obs_t)
            assert i >= 1 and abs((p.obs_t - p.obs_pair_dt_s) - computed[i - 1]) < 1e-9, \
                "the pair must be two consecutively computed frames"
        gaps = [b.trigger_t - a.trigger_t for a, b in zip(plans, plans[1:])]
        assert all(period <= g <= 1.5 for g in gaps), (period, gaps)
        assert n_free >= 4, "after the mission FoundationStereo must run free"
        assert w.counters["infer_errors"] == 0


# -------------------------------------------------------------------- CLI
def test_cli_fs_schedule_defaults_to_free_needs_policy_and_fstereo_and_warns_what_it_does():
    """``--policy-fs-schedule``: default free (nothing said); yield / only
    without --policy, or on --source hw without --fstereo, are REFUSED and
    the refusal names what is missing (the gate is built only where a
    stereo worker feeds a policy worker — the run would silently be
    `free`); with both on the hardware source a WARNING names the schedule
    and the record boundary; on another source — with or without --fstereo,
    which builds no stereo worker there either (round 2: no longer a
    refusal) — it WARNS that it does nothing; `only` with --pose adds the
    FoundationPose warning; an invalid choice exits 2. It is not one of the
    --policy dependent defaults."""
    import contextlib
    import io

    from rov_gui.__main__ import (POLICY_DEPENDENT_DEFAULTS, build_parser,
                                  check_policy)
    from rov_gui.perception.fs_gate import SCHEDULES

    p = build_parser()
    a = p.parse_args([])
    assert a.policy_fs_schedule == "free" and check_policy(a) is None
    assert SCHEDULES == ("free", "yield", "only")
    for s in SCHEDULES:
        assert p.parse_args(["--policy-fs-schedule", s]).policy_fs_schedule == s
    assert p.parse_args(["--policy"]).policy_fs_schedule == "free"
    assert "policy_fs_schedule" not in {n for n, _off, _on in POLICY_DEPENDENT_DEFAULTS}

    def _stub(argv):
        # the stub checkpoint, set the way tests/tools do it, so check_policy
        # does not read hw_mpc.yaml's real one (its FS-parity warning would
        # change what these assert)
        ns = p.parse_args(argv)
        ns.policy_ckpt = "stub"
        return ns

    hw = ["--source", "hw", "--mpc", "--policy", "--fstereo"]
    assert check_policy(_stub(hw)) is None
    assert check_policy(_stub(hw + ["--policy-fs-schedule", "free"])) is None
    for sched in ("yield", "only"):
        flag = ["--policy-fs-schedule", sched]
        for argv, missing in (
                (flag, "--policy"),                          # demo source
                (["--fstereo"] + flag, "--policy"),
                (["--source", "hw", "--fstereo"] + flag, "--policy"),
                (["--source", "hw"] + flag, "--policy and --fstereo"),
                (["--source", "hw", "--mpc", "--policy",
                  "--policy-allow-device-depth"] + flag, "--fstereo")):
            lvl, msg = check_policy(_stub(argv))
            assert lvl == "refuse", (argv, lvl, msg)
            assert msg.startswith(f"--policy-fs-schedule {sched} needs "
                                  f"{missing}: "), (argv, msg)
            assert f"Add {missing}, or drop --policy-fs-schedule." in msg, msg
        # --policy on a non-hw source without --fstereo: nothing to refuse
        # (--fstereo would build no stereo worker there either) — it warns
        lvl, msg = check_policy(_stub(["--policy"] + flag))
        assert lvl == "warn", (lvl, msg)
        assert f"--policy-fs-schedule {sched} does nothing with --source demo" \
            in msg, msg
        lvl, msg = check_policy(_stub(hw + flag))
        assert lvl == "warn" and "schedule" in msg and f"`{sched}`" in msg, msg
        assert "UNVERIFIED" in msg and "fs_schedule" in msg, msg
        assert "does nothing" not in msg and "with --pose" not in msg, msg
        for src in ("demo", "ros2"):
            lvl, msg = check_policy(_stub(
                ["--source", src, "--mpc", "--policy", "--fstereo"] + flag))
            assert lvl == "warn", (src, lvl, msg)
            assert (f"--policy-fs-schedule {sched} does nothing with "
                    f"--source {src}") in msg and "`free`" in msg, msg
    lvl, msg = check_policy(_stub(hw + ["--pose", "--policy-fs-schedule", "only"]))
    assert lvl == "warn" and "--policy-fs-schedule only with --pose" in msg, msg
    assert "FoundationPose" in msg and "`yield`" in msg, msg
    lvl, msg = check_policy(_stub(hw + ["--pose", "--policy-fs-schedule", "yield"]))
    assert lvl == "warn" and "with --pose" not in msg, msg
    # every applicable warning is named, joined into one line
    lvl, msg = check_policy(_stub(["--source", "hw", "--policy", "--fstereo",
                                   "--policy-fs-schedule", "yield"]))
    assert lvl == "warn" and "--mpc" in msg and "`yield`" in msg, msg
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            p.parse_args(["--policy-fs-schedule", "sometimes"])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("an invalid --policy-fs-schedule was accepted")


# ------------------------------------------------------------- the wiring
def test_one_gate_is_injected_into_both_hardware_workers_and_the_demo_builds_none():
    """HardwareBackend.__init__ builds ONE FsGate and hands the same object
    to the stereo worker and the policy worker — only when the schedule is
    not `free`, and only where a FoundationStereo worker feeds the policy
    (an AST scan, like the wiring test above: the hardware backend is not
    constructed offline). The demo backend has no stereo worker and builds
    no gate, whatever was typed; its meta says requested vs effective."""
    import inspect
    import textwrap

    from rov_gui.backends import demo, hardware, make_backend
    from rov_gui.perception.fs_gate import FsGate
    from rov_gui.window import MainWindow

    assert hardware.FsGate is FsGate
    hw = inspect.getsource(hardware.HardwareBackend.__init__)
    # the pins that were already there
    assert "self.fstereo.policy_mb = pmb" in hw
    assert "self.video.policy_mb = pmb" in hw
    assert "policy_allow_device_depth" in hw
    # the gate
    assert "self.fs_gate = None" in hw
    assert "self.fstereo.fs_gate = self.fs_gate" in hw
    assert "self.policy.fs_gate = self.fs_gate" in hw
    tree = ast.parse(textwrap.dedent(hw))
    builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and ast.unparse(n.func).endswith("FsGate")]
    assert len(builds) == 1, "exactly ONE gate is built"
    ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If)
           and any(c is builds[0] for b in n.body for c in ast.walk(b))]
    tests = [ast.unparse(n.test) for n in ifs]
    assert any("!= 'free'" in t for t in tests), tests
    assert any("self.fstereo is not None" in t for t in tests), tests
    assert any("'policy'" in t for t in tests), tests
    inner = min(ifs, key=lambda n: sum(1 for _ in ast.walk(n)))
    assert "!= 'free'" in ast.unparse(inner.test), ast.unparse(inner.test)
    assigned = {ast.unparse(tgt): ast.unparse(st.value) for st in inner.body
                if isinstance(st, ast.Assign) for tgt in st.targets}
    assert set(assigned) == {"self.fs_gate", "self.fstereo.fs_gate",
                             "self.policy.fs_gate"}, assigned
    assert assigned["self.fs_gate"].startswith("FsGate("), assigned
    assert assigned["self.fstereo.fs_gate"] == "self.fs_gate"
    assert assigned["self.policy.fs_gate"] == "self.fs_gate"
    # both workers start without one, and say so in their own __init__
    assert "self.fs_gate = None" in inspect.getsource(hardware.FStereoWorker.__init__)
    from rov_gui.backends.policy import PolicyWorker
    assert "self.fs_gate = None" in inspect.getsource(PolicyWorker.__init__)

    # the demo backend: no stereo worker, no gate
    assert "FsGate" not in inspect.getsource(demo)
    assert "fs_gate" not in inspect.getsource(demo.DemoBackend.__init__)
    _app()
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
            policy_fs_schedule = "only"            # typed; the CLI only warns
        win = MainWindow(Opts())
        be = make_backend("demo", win.bus, win.mailboxes, Opts())
        assert be.policy is not None and be.policy.fs_gate is None
        assert getattr(be, "fs_gate", None) is None
        fs = be.policy._fs_schedule_meta()
        assert fs["requested"] == "only" and fs["effective"] == "free", fs
        assert fs["gate"] is None and "nothing to schedule" in fs["why"], fs
        win.close()


# =============================================================================
# --policy-fs-schedule, round 2 (2026-10-01): what spends a burst (a frame
# DELIVERED to the policy mailbox, FsGate.fs_delivered), the blocked path
# (_only_blocked: no burst, FoundationStereo free, the free trigger counts),
# period_s as a lower bound under `only`, the asymmetric lead, the burst
# timeout's warning and note, set_ckpt opening the gate before close(), and
# the record (meta / stereo recording / CLI).
# =============================================================================
def _wired(tmp, gate, **worker_kw):
    """The REAL PolicyWorker and the REAL FStereoWorker joined by the REAL
    tap — the stereo worker declares the rect_left grid on the policy's
    mailbox and puts each map into it (the f=300 rig is one the policy's
    builder accepts) and reports each delivery to the gate — both holding
    ``gate``. One pair is computed outside any mission and taken, so the
    policy has adopted its grid; the ring is then cleared. Returns
    ``(w, mb, plans, statuses, logs, fsw, smb, s, put)``."""
    w, bus, mb, plans, statuses, sensors, logs = _worker(tmp, grid=False,
                                                         **worker_kw)
    fsw, smb, s, _pmb, put = _fs_worker(f=300.0)
    fsw.policy_mb = mb
    w.fs_gate = gate
    fsw.fs_gate = gate
    w.tick()                                       # no mission: the gate is open
    put(1)
    fsw.tick()
    assert s.seen == [1] and fsw._policy_tap["put"] == 1
    w.tick()                                       # taken: the grid is adopted
    assert w.builder.grid_kind == GRID_RECT_LEFT and w.builder.usable, \
        w.builder.why
    w._ring.clear()
    w._frame_pending = False
    return w, mb, plans, statuses, logs, fsw, smb, s, put


def test_only_a_failed_fstereo_inference_inside_a_burst_does_not_spend_it():
    """Round 2: a frame whose inference RAISED is ``fs_end(ran=False)`` —
    not a frame, not a delivery — so under `only` it does not use up the
    burst. Single-threaded and all real: FStereoWorker.tick with a session
    whose infer raises once, the tap into the policy mailbox, the
    PolicyWorker, one FsGate("only"). The burst completes on the next two
    pairs, a plan comes out without a burst timeout, the gate's
    ``burst_frames`` equals the frames delivered during the burst, and its
    ``frames`` does not count the failed one. (Had the failure spent one of
    the two, FoundationStereo would have computed one more pair and then
    drained: one frame for the policy, a burst timeout — round 1.)"""
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        gate = FsGate("only")
        w, mb, plans, statuses, logs, fsw, smb, s, put = _wired(tmp, gate)
        c0 = gate.snapshot()["counters"]
        assert c0["frames"] == 1 and c0["delivered"] == 1 and c0["burst_frames"] == 0
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # the request
        assert w._burst is not None and gate.snapshot()["burst_allowance"] == 2
        put0 = fsw._policy_tap["put"]

        s.boom = "CUDA error: an illegal memory access was encountered"
        put(2)
        fsw.tick()                                 # allowed, and it raises
        s.boom = None
        assert s.seen == [1, 2] and fsw._infer_fails == 1
        snap = gate.snapshot()
        assert snap["burst_allowance"] == 2, \
            "a failed inference must not spend the burst"
        assert snap["in_flight"] is False
        assert snap["counters"]["frames"] == 1 and snap["counters"]["delivered"] == 1
        assert snap["counters"]["burst_frames"] == 0
        assert fsw._policy_tap["put"] == put0
        w.tick()
        assert w._burst is not None and not plans

        cA = put(3)                                # the next pair: 1 of 2
        fsw.tick()
        assert s.seen == [1, 2, 3] and gate.snapshot()["burst_allowance"] == 1
        w.tick()
        assert not plans
        time.sleep(0.06)
        cB = put(4)                                # 2 of 2
        fsw.tick()
        assert s.seen == [1, 2, 3, 4] and gate.snapshot()["burst_allowance"] == 0
        _fix_now(w, t0)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note, w._sched_n)
        p = plans[0]
        assert p.obs_t == cB and abs(p.obs_pair_dt_s - (cB - cA)) < 1e-9
        assert p.fs_schedule == "only" and p.depth_src == GRID_RECT_LEFT
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 0}
        c = gate.snapshot()["counters"]
        delivered = fsw._policy_tap["put"] - put0
        assert delivered == 2 and c["burst_frames"] == delivered, (c, delivered)
        assert c["delivered"] - c0["delivered"] == delivered
        assert c["frames"] == 3, "1, 3 and 4 made a map; 2 raised"
        assert c["burst_void"] == 0
        put(5)                                     # spent: discarded
        fsw.tick()
        assert s.seen == [1, 2, 3, 4] and gate.snapshot()["counters"]["drained"] == 1


def test_only_a_frame_the_tap_drops_does_not_spend_the_burst():
    """Round 2: FStereoWorker.tick reports a delivery only when the tap's
    put count rose. Three ways the tap drops a frame it computed — the
    policy mailbox does not want frames, the rig has no rectification to
    build a grid from, the rig is a DIFFERENT geometry the mailbox refuses —
    each is a frame (a map came out) but not a delivery: the allowance
    stays at 2, and the next two good pairs complete the burst."""
    from types import SimpleNamespace

    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        gate = FsGate("only")
        w, mb, plans, statuses, logs, fsw, smb, s, put = _wired(tmp, gate)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()                                   # the request
        assert w._burst is not None and gate.snapshot()["burst_allowance"] == 2
        c0 = gate.snapshot()["counters"]
        tap0 = dict(fsw._policy_tap)

        def stereo_on(rig, tag):
            img = np.full((400, 640), int(tag), np.uint8)
            smb.put(img, img.copy(), rig, now(), frame_seq=int(tag))
            fsw.tick()

        mb.set_wanted(False)                       # (1) not wanted
        put(2)
        fsw.tick()
        mb.set_wanted(True)
        bare = SimpleNamespace(R1=None, P1=None, mono_size=(640, 400),
                               alpha=0.5, provenance={})
        stereo_on(bare, 3)                         # (2) no grid from this rig
        c_, s_ = np.cos(np.radians(0.5)), np.sin(np.radians(0.5))
        other = SimpleNamespace(
            R1=np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]),
            P1=np.array([[300.0, 0, 320, 0], [0, 300.0, 200, 0], [0, 0, 1, 0]]),
            mono_size=(640, 400), alpha=0.5, provenance={})
        stereo_on(other, 4)                        # (3) a grid the mailbox refuses
        assert s.seen == [1, 2, 3, 4], "all three were computed"
        snap = gate.snapshot()
        c = snap["counters"]
        assert snap["burst_allowance"] == 2, \
            "a frame the tap dropped must not spend the burst"
        assert c["frames"] - c0["frames"] == 3, "each of them produced a map"
        assert c["delivered"] == c0["delivered"], "...and none was delivered"
        assert c["burst_frames"] == c0["burst_frames"] == 0
        assert fsw._policy_tap["put"] == tap0["put"]
        assert fsw._policy_tap["dropped_no_grid"] == tap0["dropped_no_grid"] + 1
        assert fsw._policy_tap["dropped_grid_refused"] == \
            tap0["dropped_grid_refused"] + 1
        w.tick()
        assert w._burst is not None and not plans and not w._ring

        cA = put(5)                                # good pairs again
        fsw.tick()
        w.tick()
        time.sleep(0.06)
        cB = put(6)
        fsw.tick()
        _fix_now(w, t0)
        w.tick()
        assert len(plans) == 1, (w.counters, w._note, w._sched_n)
        p = plans[0]
        assert p.obs_t == cB and abs(p.obs_pair_dt_s - (cB - cA)) < 1e-9
        c = gate.snapshot()["counters"]
        assert c["burst_frames"] == 2 and c["delivered"] - c0["delivered"] == 2, c
        assert gate.snapshot()["burst_allowance"] == 0
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 0}


def test_only_blocked_asks_for_nothing_and_counts_the_skip_as_free_does():
    """Round 2: under `only`, while NO forward could run whatever frames
    arrived — no grid adopted yet (no frame taken), the newest proprio row
    is not a fresh fix, the history is EMPTY right after an epoch change,
    there is no start pose, the grid was refused — tick() requests no burst
    and tells the gate the mission is not active, so FoundationStereo runs
    free (fs_begin says go); the frames that then arrive go through the
    ordinary `free` trigger, which counts the skip on each arrival exactly
    as a worker without a gate does (fed the same states and frames, the
    two workers' counters and notes are identical); no plan; and no
    heartbeat is ever sent while blocked. When the newest row is a fresh fix
    again (with history and a start pose) the next tick asks for a burst
    and a plan follows."""
    from rov_gui.perception.fs_gate import FS_GO, FsGate

    with tempfile.TemporaryDirectory() as tmp:
        wf, _bf, mbf, plans_f, st_f, *_ = _worker(tmp)
        wo, _bo, mbo, plans_o, st_o, *_ = _worker(tmp)
        gate = FsGate("only")
        wo.fs_gate = gate
        both = ((wf, mbf), (wo, mbo))
        seed = [0]

        def gate_free(what):
            snap = gate.snapshot()
            assert snap["mission_active"] is False, (what, snap)
            assert snap["counters"]["bursts"] == 0, (what, snap)
            assert snap["counters"]["activations"] == 0, \
                f"{what}: a heartbeat was sent while blocked"
            assert gate.fs_begin() == FS_GO, f"{what}: FoundationStereo must run free"
            gate.fs_end(ran=False)

        # (0) a fresh, active mission — but no frame taken yet, so no grid
        t00 = now()
        for w, _mb in both:
            _feed_history(w, t00 - 1.0)
            for _ in range(3):
                w.tick()
        assert wo.builder.grid_kind == "" and not wo.builder.usable
        assert wo._active() and wo._only_blocked()
        assert wo._burst is None and wo._sched_n["bursts"] == 0
        assert wf.counters == wo.counters and not plans_f and not plans_o
        gate_free("no grid yet")

        def arrive():
            """One frame for both workers, stamped now; each ticks once."""
            seed[0] += 1
            t = now() - 0.005
            for w, mb in both:
                _frame(w, mb, t, seed=seed[0])
                w.tick()

        def quiet_ticks(n=3):
            for w, _mb in both:
                for _ in range(n):
                    w.tick()

        def same(what, skip, n):
            assert wf.counters == wo.counters, (what, wf.counters, wo.counters)
            assert wo.counters[skip] == n, (what, skip, wo.counters)
            assert wf._note == wo._note == skip, (what, wf._note, wo._note)
            assert not plans_f and not plans_o, what
            assert wo._only_blocked(), what
            assert wo._burst is None, what
            assert wo._sched_n == {"bursts": 0, "burst_timeouts": 0,
                                   "burst_refused": 0}, (what, wo._sched_n)
            gate_free(what)

        # (1) the newest row is not a fresh fix (the first frame adopts the grid)
        t0 = now()
        for w, _mb in both:
            _feed_history(w, t0, fresh=False)
        arrive()
        same("not fresh", "skip_fresh", 1)
        quiet_ticks()
        same("not fresh, no arrival", "skip_fresh", 1)
        arrive()
        arrive()
        same("not fresh, three arrivals", "skip_fresh", 3)
        # (2) an epoch change, and no fix yet in the new epoch: EMPTY history
        for w, _mb in both:
            w.on_policy_state(_state(None, (0.0,) * 6, epoch=2))
            assert w.hist is not None and len(w.hist) == 0
        arrive()
        same("empty history after an epoch change", "skip_history", 1)
        # (3) history again, fresh, but no start pose
        t1 = now()
        for w, _mb in both:
            _feed_history(w, t1, epoch=2, eta_start=None)
        arrive()
        same("no start pose", "skip_no_start", 1)
        wf._publish_status(force=True)             # (the idle publish is 1 Hz)
        wo._publish_status(force=True)
        assert st_f[-1].n_skip == st_o[-1].n_skip == 5
        assert st_o[-1].fs_schedule == "only" and st_f[-1].fs_schedule == "free"

        # (4) a fresh fix, history and a start pose: the next tick ASKS
        t2 = now()
        _feed_history(wo, t2, epoch=2)
        assert not wo._only_blocked()
        wo.tick()
        assert wo._burst is not None and wo._sched_n["bursts"] == 1
        snap = gate.snapshot()
        assert snap["mission_active"] is True and snap["burst_allowance"] == 2
        _fs_frame(gate)
        _frame(wo, mbo, now() - 0.075, seed=50)
        wo.tick()
        assert not plans_o
        _fs_frame(gate)
        _fix_now(wo, t2, epoch=2)
        _frame(wo, mbo, now() - 0.005, seed=51)
        wo.tick()
        assert len(plans_o) == 1, (wo.counters, wo._note)
        assert plans_o[0].fs_schedule == "only" and plans_o[0].epoch == 2

        # (5) a grid the builder REFUSED: no frame can ever cure it
        wf, _bf, mbf, plans_f, st_f, *_ = _worker(tmp)
        wo, _bo, mbo, plans_o, st_o, *_ = _worker(tmp)
        gate = FsGate("only")
        wo.fs_gate = gate
        both = ((wf, mbf), (wo, mbo))
        t3 = now()
        for w, _mb in both:
            w.builder.min_coverage = 1.01          # nothing can cover that
            _feed_history(w, t3)
        arrive()
        arrive()
        assert wo.builder.grid_kind == GRID_IDENTITY and not wo.builder.usable
        same("a refused grid", "skip_grid", 2)


def test_only_a_pending_burst_is_resolved_before_the_worker_goes_free():
    """Round 2: tick() takes the burst path while a burst is PENDING even if
    the worker has just become blocked — the request is not abandoned
    mid-way (FoundationStereo may be computing it). (a) Its two frames
    arrive: the burst completes through _tick_only (``t_done`` stamped, the
    lead updated, the skip counted there — no plan, the period unstamped),
    and only THEN does the worker take the free path: the gate is told the
    mission is not active, no new burst is asked for, and the next arrival
    is counted by the free trigger. (b) Its frames never come: it times out
    through _tick_only (counted, said), then the same."""
    from rov_gui.backends.policy import ONLY_LEAD_MAX_S, ONLY_LEAD_MIN_S
    from rov_gui.perception.fs_gate import FS_GO, FsGate

    def gone_free(w, gate, what):
        snap = gate.snapshot()
        assert w._burst is None, what
        assert snap["mission_active"] is False, (what, snap)
        assert gate.fs_begin() == FS_GO, what
        gate.fs_end(ran=False)

    with tempfile.TemporaryDirectory() as tmp:
        # (a) the frames arrive
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()
        b = w._burst
        assert b is not None and gate.snapshot()["burst_allowance"] == 2
        lead0 = w._only_lead_s
        w.on_policy_state(_state(now(), (0.03, 0, 0.8, 0, 0, 0.1), fresh=False))
        assert w._only_blocked()
        w.tick()
        assert w._burst is b, "a pending burst must not be abandoned"
        assert gate.snapshot()["mission_active"] is True
        _fs_frame(gate)
        _frame(w, mb, now() - 0.075, seed=1)
        w.tick()
        assert w._burst is b and "t_done" not in b
        _fs_frame(gate)
        _frame(w, mb, now() - 0.005, seed=2)
        n_fresh = w.counters["skip_fresh"]
        w.tick()                                   # resolved through _tick_only
        assert w._burst is None and "t_done" in b
        assert w.counters["skip_fresh"] == n_fresh + 1 and w._note == "skip_fresh"
        assert not plans and w._last_infer == 0.0
        dur = b["t_done"] - b["t"]
        want = dur if dur < lead0 else 0.8 * lead0 + 0.2 * dur
        want = min(ONLY_LEAD_MAX_S, max(ONLY_LEAD_MIN_S, want))
        assert abs(w._only_lead_s - want) < 1e-12, (w._only_lead_s, want)
        snap = gate.snapshot()
        assert snap["burst_allowance"] == 0 and snap["counters"]["burst_frames"] == 2
        assert snap["counters"]["holds"] == 0
        w.tick()                                   # ...and only now: free
        gone_free(w, gate, "after the burst")
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 0, "burst_refused": 0}
        assert w.counters["skip_fresh"] == n_fresh + 1, "no arrival, no count"
        _frame(w, mb, now() - 0.005, seed=3)       # FoundationStereo, running free
        w.tick()
        assert w.counters["skip_fresh"] == n_fresh + 2, \
            "the free trigger counts the arrival"
        gone_free(w, gate, "an arrival while blocked")
        assert w._sched_n["bursts"] == 1 and not plans

        # (b) the frames never come
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        gate = FsGate("only", burst_timeout_s=0.15)
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0
        w.tick()
        b = w._burst
        assert b is not None
        w.on_policy_state(_state(now(), (0.03, 0, 0.8, 0, 0, 0.1), fresh=False))
        deadline = now() + 2.0
        while w._burst is b and now() < deadline:
            w.tick()
            assert gate.snapshot()["mission_active"] is True
            time.sleep(0.01)
        assert w._burst is None and now() - float(b["t"]) > 0.15
        assert "t_done" not in b
        assert w._sched_n == {"bursts": 1, "burst_timeouts": 1, "burst_refused": 0}
        assert w._note == "burst_timeout"
        assert sum(1 for lvl, m in logs
                   if lvl == "warn" and "delivered 0 of 2" in m) == 1, logs
        w.tick()
        gone_free(w, gate, "after the timeout")
        assert w._sched_n["bursts"] == 1 and not plans


def test_only_period_s_is_a_lower_bound_even_when_the_burst_comes_in_early():
    """Round 2: under `only` the attempt needs the burst's frames AND the
    period — period_s is the lower bound on the interval between attempts
    in every schedule. Driven by hand: an attempt was just stamped, and a
    burst asked for 0.12 s ago (a lead longer than this burst took)
    completes at once. Nothing fires while t - _last_infer < period; the
    first tick after it fires exactly ONE plan, on the burst's two frames;
    ``t_done`` is stamped once — on the completing tick — and the lead is
    updated then and never again (not on the waiting ticks, not at the
    attempt)."""
    from rov_gui.backends.policy import ONLY_LEAD_S
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _adopt_grid(w, mb)
        period = 0.3
        w.pc["period_s"] = period
        t0 = now()
        _feed_history(w, t0)
        t_last = now()
        w._last_infer = t_last                     # an attempt just ran
        b = {"t": now() - 0.12}
        w._burst = b
        cA = now() - 0.075
        _frame(w, mb, cA, seed=1)
        _fix_now(w, t0)
        cB = now() - 0.005
        _frame(w, mb, cB, seed=2)
        w.tick()                                   # the burst is in
        assert "t_done" in b and w._burst is b and not plans
        t_done, lead1 = b["t_done"], w._only_lead_s
        assert t_done - t_last < period, "the period must still be running"
        assert lead1 == t_done - b["t"] < ONLY_LEAD_S, \
            "a burst shorter than the lead sets it at once"
        waits = 0
        deadline = t_last + period + 1.0
        while not plans and now() < deadline:
            _fix_now(w, t0)
            w.tick()
            if not plans:
                waits += 1
                assert w._burst is b and w._last_infer == t_last
            assert b["t_done"] == t_done and w._only_lead_s == lead1, \
                "the lead is updated once per burst"
            time.sleep(0.01)
        assert len(plans) == 1 and waits >= 5, (len(plans), waits)
        p = plans[0]
        assert p.trigger_t - t_last >= period, p.trigger_t - t_last
        assert p.trigger_t - t_last < period + 0.2, "it fires once the period is over"
        assert p.obs_t == cB and abs(p.obs_pair_dt_s - (cB - cA)) < 1e-9
        assert w._burst is None and w._last_infer == p.trigger_t
        for _ in range(5):
            _fix_now(w, t0)
            w.tick()
        assert len(plans) == 1 and w._only_lead_s == lead1


def test_only_the_lead_moves_down_at_once_and_up_by_ema_inside_its_bounds():
    """Round 2: when a burst completes, the lead (how long before the
    period ends the next one is asked for) is updated from the burst's
    duration — DOWN AT ONCE when the burst was shorter than the lead (a
    lead longer than the burst makes the frames wait for the period: an
    older observation), UP by the 0.8/0.2 EMA otherwise (too short a lead
    only makes the plan a little late) — clipped to [ONLY_LEAD_MIN_S,
    ONLY_LEAD_MAX_S]. Driven by hand: a burst stamped ``dur`` ago, its two
    frames, one tick with the period still running (only the lead moves)."""
    from rov_gui.backends.policy import ONLY_LEAD_MAX_S, ONLY_LEAD_MIN_S
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, *_ = _worker(tmp)
        w.fs_gate = FsGate("only")
        _adopt_grid(w, mb)
        _feed_history(w, now())

        def complete(lead0, dur):
            w._only_lead_s = lead0
            w._ring.clear()
            b = {"t": now() - dur}
            w._burst = b
            _frame(w, mb, now() - 0.075, seed=1)
            _frame(w, mb, now() - 0.005, seed=2)
            w._last_infer = now()                  # the period is running
            w.tick()
            assert "t_done" in b and w._burst is b and not plans
            w._burst = None
            return b["t_done"] - b["t"], w._only_lead_s

        d, lead = complete(0.3, 0.15)              # shorter: at once
        assert d < 0.3 and lead == d, (d, lead)
        d, lead = complete(0.2, 0.02)              # ...clipped at the floor
        assert d < ONLY_LEAD_MIN_S and lead == ONLY_LEAD_MIN_S, (d, lead)
        d, lead = complete(0.15, 0.3)              # longer: the EMA
        want = 0.8 * 0.15 + 0.2 * d
        assert d >= 0.15 and abs(lead - want) < 1e-12 and lead < d, (d, lead, want)
        d, lead = complete(0.38, 1.5)              # ...clipped at the ceiling
        assert 0.8 * 0.38 + 0.2 * d > ONLY_LEAD_MAX_S, d
        assert lead == ONLY_LEAD_MAX_S, (d, lead)


def test_only_a_burst_timeout_warns_once_per_5_s_notes_it_and_asks_again():
    """Round 2: a burst whose frames do not arrive within the gate's
    burst_timeout_s is dropped, counted in ``_sched_n["burst_timeouts"]``,
    published with the status note "burst_timeout", logged as ONE warning
    (rate-limited: a second timeout within 5 s is counted and noted but not
    said again; after 5 s it is said again, with the running count) — and
    the next tick asks again (the gate voids the undelivered allowance)."""
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp)
        gate = FsGate("only", burst_timeout_s=0.1)
        w.fs_gate = gate
        _adopt_grid(w, mb)
        t0 = now()
        _feed_history(w, t0)
        w._last_infer = 0.0

        def warns_from(n):
            return [m for lvl, m in logs[n:] if lvl == "warn"]

        def time_out(n_expected):
            """Wait out the pending burst; the tick that drops it."""
            assert w._burst is not None
            time.sleep(0.12)
            _fix_now(w, t0)
            w._last_status_pub = 0.0               # the 1 Hz limiter is due
            n_st, n_log = len(statuses), len(logs)
            w.tick()
            assert w._burst is None and not plans
            assert w._sched_n["burst_timeouts"] == n_expected, w._sched_n
            assert len(statuses) == n_st + 1, "one status, published at once"
            assert statuses[-1].note == "burst_timeout", statuses[-1].note
            return warns_from(n_log)

        w.tick()                                   # burst 1
        said = time_out(1)
        assert len(said) == 1, said
        assert "delivered 0 of 2 frames within 0.1 s" in said[0], said[0]
        assert "--policy-fs-schedule only" in said[0] and "(1 so far)" in said[0]
        w.tick()                                   # asked again
        assert w._burst is not None
        assert w._sched_n == {"bursts": 2, "burst_timeouts": 1, "burst_refused": 0}
        c = gate.snapshot()["counters"]
        assert c["bursts"] == 2 and c["burst_void"] == 1, c
        _fs_frame(gate)                            # one frame of two this time
        _frame(w, mb, now() - 0.005, seed=1)
        said = time_out(2)                         # within 5 s: not said again
        assert said == [], said
        w.tick()
        w._burst_warn_t -= 5.0                     # ...5 s later
        said = time_out(3)
        assert len(said) == 1 and "(3 so far)" in said[0], said
        assert "delivered 0 of 2" in said[0], "the frame of the dropped burst is not this one's"
        w.tick()
        assert w._burst is not None and w._sched_n["bursts"] == 4
        assert not any(v for k, v in w.counters.items() if k.startswith("skip_")), \
            "a burst timeout is not a skip"
        assert w._sched_n["burst_refused"] == 0


def test_set_ckpt_opens_the_gate_before_the_old_session_is_closed():
    """Round 2: a panel checkpoint swap runs inside ONE slot — the parity
    read and the old session's close() (which frees torch and can take
    seconds) with no tick in between — so set_ckpt leaves the schedule gate
    open right after its refusal checks, BEFORE either. With a gate whose
    mission is active, a burst pending and a hold leaked, a swap to another
    stub checkpoint: when the parity read ran and when old.close() ran the
    gate was already inactive, its allowance 0 and not held; afterwards
    ``_burst`` is None and FoundationStereo may start a frame. A REFUSED
    pick (still engaged: DISENG first) leaves the gate alone."""
    import dataclasses

    import rov_gui.backends.policy as PM
    from rov_gui.perception.fs_gate import FS_GO, FsGate

    seen = []                                      # (where, what, gate snapshot)
    box = {}

    class _Closing(StubPolicySession):
        def close(self):
            g = box.get("gate")
            if g is not None:
                seen.append(("close", self.ckpt, g.snapshot()))
            super().close()

    def factory(pc, o, ckpt=None):
        return _Closing(ckpt or "stub", "", dataset_fps=float(pc["dataset_fps"]))

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, plans, statuses, sensors, logs = _worker(tmp, factory=factory)
        gate = FsGate("only")
        w.fs_gate = gate
        box["gate"] = gate
        _adopt_grid(w, mb)
        _feed_history(w, now())
        w._last_infer = 0.0
        w.tick()
        assert w._burst is not None
        snap = gate.snapshot()
        assert snap["mission_active"] is True and snap["burst_allowance"] == 2
        gate.hold()                                # a hold that leaked, too
        # a REFUSED pick (the mission's state is engaged) touches nothing
        w.set_ckpt("stub7")
        assert w.counters["ckpt_swaps"] == 0 and "DISENG" in statuses[-1].ckpt_note
        snap = gate.snapshot()
        assert w._burst is not None and snap["mission_active"] is True
        assert snap["burst_allowance"] == 2 and snap["held"] is True and not seen
        # DISENG — no tick in between — then the pick
        w.on_policy_state(dataclasses.replace(
            _state(now(), (0.0,) * 6, active=False), engaged=False))
        w.opts.fstereo = True                      # so the parity read runs
        real_tds = PM.training_depth_source

        def tds(path):
            seen.append(("parity", str(path), gate.snapshot()))
            return real_tds(path)

        PM.training_depth_source = tds
        try:
            w.set_ckpt("stub7")
        finally:
            PM.training_depth_source = real_tds
        assert w.counters["ckpt_swaps"] == 1 and w.session.ckpt == "stub7"
        assert [(k, p) for k, p, _s in seen] == [("parity", "stub7"),
                                                 ("close", "stub")], seen
        for where, _p, snap in seen:
            assert snap["mission_active"] is False, (where, snap)
            assert snap["burst_allowance"] == 0, (where, snap)
            assert snap["held"] is False, (where, snap)
        assert w._burst is None
        assert gate.snapshot()["mission_active"] is False
        assert gate.fs_begin() == FS_GO
        gate.fs_end(ran=False)


def test_yield_never_heartbeats_or_asks_for_a_burst_during_a_mission():
    """Round 2: under `yield` the worker never calls ``set_active`` or
    ``request_burst`` while a mission runs — the heartbeat and the burst are
    `only`'s (FsGate.set_active is a no-op under yield, request_burst
    answers False). Every kind of active tick, against the recording gate:
    the period still running, the period over with no new frame, a
    pre-forward skip, a plan, a forward that raises, the stretch where
    `only` would ask for its burst — the only calls are the hold
    protocol's, balanced. Leaving the mission releases (a leaked hold is
    dropped) and never heartbeats or asks for a burst either."""
    with tempfile.TemporaryDirectory() as tmp:
        log, ctl = [], {}
        w, bus, mb, plans, statuses, sensors, logs = _worker(
            tmp, factory=_logging_session_factory(log, ctl))
        w.fs_gate = _RecGate("yield", log)
        mission = []

        def ticks(n=1):
            for _ in range(n):
                n0 = len(log)
                w.tick()
                mission.extend(log[n0:])

        t0 = now()
        _feed_history(w, t0)
        assert w._active()
        w._last_infer = now()                      # the period is running
        ticks(3)
        w._last_infer = 0.0                        # over; no frame arrived
        ticks(3)
        w._first_pending = False                   # one frame, no partner
        _frame(w, mb, now() - 0.005, seed=1)
        ticks()
        assert w.counters["skip_pair"] == 1 and not plans, w.counters
        _stage(w, mb, seed=10)                     # a plan
        ticks()
        assert len(plans) == 1, (w.counters, w._note)
        ctl["boom"] = "CUDA error"                 # a forward that raises
        _stage(w, mb, seed=20)
        ticks()
        ctl["boom"] = None
        assert w.counters["infer_errors"] == 1
        w._last_infer = now() - 0.4                # where `only` would ask
        ticks(3)
        assert mission == ["hold", "wait_idle", "predict", "release"] * 2, mission
        w.on_policy_state(_state(now(), (0.0,) * 6, active=False))
        w.tick()                                   # leaving the mission
        assert "active" not in log, log
        assert not any(c.startswith("burst") for c in log), log
        assert log[-1] == "release", log
        assert w._burst is None and w._sched_n == {"bursts": 0, "burst_timeouts": 0,
                                                   "burst_refused": 0}


def test_meta_trigger_strings_and_the_fs_schedule_why_in_every_combination():
    """Round 2, the record. ``pairing.trigger`` is the string every record
    since 2026-09-02 carries, word for word, under free AND yield (only has
    its own); ``pairing.trigger_detail`` — what the code really does — is
    in every mode, the same under free and yield. ``fs_schedule.why`` is ""
    when what was typed is what flies; the "no FoundationStereo worker"
    sentence when the options asked for a schedule and no gate exists (demo
    source, device depth); the "gate was injected" sentence when a gate
    exists that the options did not ask for. ``only_lead_s`` only under
    only; a gate whose snapshot raises does not take the block down."""
    import json

    from rov_gui.perception.fs_gate import FsGate

    legacy = "depth-frame arrival after period_s"
    only_trigger = ("2 frames requested from FoundationStereo ahead of "
                    "period_s (--policy-fs-schedule only)")
    no_worker = ("no FoundationStereo worker feeds this policy (demo source "
                 "or device depth): there is nothing to schedule, so the run "
                 "is `free`")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, mb, *_ = _worker(tmp)
        assert "policy_fs_schedule" not in vars(type(w.opts))
        details = {}
        for gate_mode, typed, why in (
                (None, None, ""),
                (None, "free", ""),
                (None, "yield", no_worker),
                (None, "only", no_worker),
                ("yield", "yield", ""),
                ("only", "only", ""),
                ("yield", None, "a gate was injected with mode yield; the "
                                "options said free"),
                ("only", "yield", "a gate was injected with mode only; the "
                                  "options said yield"),
                ("yield", "only", "a gate was injected with mode yield; the "
                                  "options said only")):
            what = (gate_mode, typed)
            w.fs_gate = FsGate(gate_mode) if gate_mode else None
            if typed is None:
                vars(w.opts).pop("policy_fs_schedule", None)
                assert not hasattr(w.opts, "policy_fs_schedule")
            else:
                w.opts.policy_fs_schedule = typed
            m = w.meta()
            json.dumps(m, default=str)
            fs, pairing = m["fs_schedule"], m["pairing"]
            effective = gate_mode or "free"
            assert fs["requested"] == (typed or "free"), (what, fs)
            assert fs["effective"] == effective, (what, fs)
            assert fs["why"] == why, (what, fs["why"])
            assert fs["only_lead_s"] == (round(w._only_lead_s, 3)
                                         if effective == "only" else None), (what, fs)
            assert fs["worker"] == w._sched_n
            if gate_mode is None:
                assert fs["gate"] is None, (what, fs)
            else:
                assert fs["gate"]["mode"] == gate_mode, (what, fs)
            assert pairing["trigger"] == (only_trigger if effective == "only"
                                          else legacy), (what, pairing["trigger"])
            d = pairing["trigger_detail"]
            assert isinstance(d, str) and "period_s" in d, (what, d)
            details.setdefault(effective, set()).add(d)
        assert set(details) == {"free", "yield", "only"}
        assert all(len(v) == 1 for v in details.values()), details
        assert details["free"] == details["yield"] != details["only"], details
        assert "free trigger while no forward could run" in next(iter(details["only"]))

        class _Broken(_RecGate):
            def snapshot(self):
                raise RuntimeError("lock poisoned")

        w.fs_gate = _Broken("only")
        w.opts.policy_fs_schedule = "only"
        fs = w.meta()["fs_schedule"]
        assert fs["gate"] == {"error": "RuntimeError: lock poisoned"}, fs
        assert fs["effective"] == "only" and fs["why"] == ""


def test_fstereo_meta_without_a_session_and_the_stereo_recording_name_the_schedule():
    """Round 2, the stereo worker's record. ``FStereoWorker.meta()`` when no
    session was built is exactly {enabled, why, schedule}: ``schedule`` None
    without a gate, the gate's own snapshot with one — and still no
    ``policy_tap`` (that block is for a worker that ran). The stereo
    recorder (--record-stereo) is armed with ``fstereo.policy_fs_schedule``:
    "free" without a gate, the gate's mode with one — under `only` the pairs
    between bursts never reach the recorder (a frame_seq gap there is the
    schedule, not the camera). The session's own describe() dict is copied,
    never written into."""
    import json

    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.perception.fs_gate import FsGate

    _app()
    fsw = FStereoWorker(DataBus(), None, _Opts(), {})
    m = fsw.meta()
    assert m == {"enabled": False, "why": "not built", "schedule": None}, m
    fsw._fault = "ImportError: no module named torch"
    gate = FsGate("only")
    fsw.fs_gate = gate
    m = fsw.meta()
    assert set(m) == {"enabled", "why", "schedule"} and "policy_tap" not in m, m
    assert m["enabled"] is False and m["why"] == "ImportError: no module named torch"
    assert m["schedule"]["mode"] == "only" and set(m["schedule"]) == set(gate.snapshot())
    json.dumps(m)

    class _Rec:
        def __init__(self):
            self.starts, self.seqs, self.why = [], [], ""

        def start(self, *, rig_describe=None, fstereo=None, extra=None):
            self.starts.append(dict(fstereo or {}))
            return True

        def submit(self, t, left, right, frame_seq):
            self.seqs.append(int(frame_seq))

        def describe(self):
            return {"fake": True}

        def close(self):
            pass

    for mode in (None, "yield", "only"):
        fsw, smb, s, pmb, put = _fs_worker()
        shared = {"fake": True}
        s.describe = lambda shared=shared: shared  # the SAME dict every call
        rec = _Rec()
        fsw.stereo_rec = rec
        gate = FsGate(mode) if mode else None
        fsw.fs_gate = gate
        put(1)
        fsw.tick()
        assert rec.starts == [{"fake": True, "policy_fs_schedule": mode or "free"}], \
            (mode, rec.starts)
        assert shared == {"fake": True}, "the session's describe() was written into"
        assert rec.seqs == [1]
    # `only`, a mission: only the burst's pairs are recorded
    gate.set_active(True)
    put(2)
    fsw.tick()                                     # between bursts: discarded
    assert gate.request_burst(2)
    for tag in (3, 4, 5):
        put(tag)
        fsw.tick()
    assert rec.seqs == [1, 3, 4] and len(rec.starts) == 1, rec.seqs
    assert s.seen == [1, 3, 4]


def test_only_the_free_path_never_runs_a_forward_even_if_the_two_checks_drift_apart():
    """Round 2, the guard. Under `only` the free trigger is entered only
    while ``_only_blocked()``, and it cannot fire a forward there because
    ``_skip_reason()`` refuses on the same conditions. If the two ever drift
    apart (blocked, yet no skip reason) the free path still must not run a
    forward beside a free-running FoundationStereo and call it `only`: no
    plan, no hold, the period unstamped, the arrival consumed — and the
    next tick, the two agreeing again, takes the burst path. The control: a
    worker without a gate given the same frames and the same forced "no
    skip" DOES fire."""
    from rov_gui.perception.fs_gate import FsGate

    with tempfile.TemporaryDirectory() as tmp:
        wf, _bf, mbf, plans_f, *_ = _worker(tmp)
        wf._skip_reason = lambda: ""
        _stage(wf, mbf, seed=10)
        wf.tick()
        assert len(plans_f) == 1, (wf.counters, wf._note)

        w, bus, mb, plans, statuses, *_ = _worker(tmp)
        gate = FsGate("only")
        w.fs_gate = gate
        _stage(w, mb, seed=10)
        assert w._frame_pending and not w._only_blocked()
        w._only_blocked = lambda: True
        w._skip_reason = lambda: ""
        skips0 = {k: v for k, v in w.counters.items() if k.startswith("skip_")}
        w.tick()
        assert not plans and w._last_infer == 0.0, (w.counters, w._note)
        assert w._frame_pending is False, "the free trigger consumed the arrival"
        assert w._burst is None
        assert w._sched_n == {"bursts": 0, "burst_timeouts": 0, "burst_refused": 0}
        snap = gate.snapshot()
        assert snap["mission_active"] is False and snap["counters"]["holds"] == 0
        assert snap["counters"]["waits"] == 0
        assert {k: v for k, v in w.counters.items() if k.startswith("skip_")} == skips0
        assert w.counters["plans"] == 0 and w.counters["infer_errors"] == 0
        del w._only_blocked, w._skip_reason        # the class methods again
        w.tick()
        assert w._burst is not None and gate.snapshot()["mission_active"] is True


def test_cli_fs_schedule_refusal_names_what_is_missing_and_warns_off_hw():
    """Round 2, the launch check. The refusal names exactly what is
    missing: --policy alone when it is --policy that is missing (any
    source); --fstereo alone on --source hw with --policy and device depth
    allowed (no stereo worker to schedule). On --source demo, --policy
    --mpc without --fstereo is a WARNING that the flag does nothing there,
    not a refusal (--fstereo would build no stereo worker on demo either).
    On hw with both: the yield warning says not to pool plan age or
    infer_ms, the only warning that the pair spacing changes too. The
    default launch says nothing."""
    from rov_gui.__main__ import build_parser, check_policy

    p = build_parser()

    def _stub(argv):
        ns = p.parse_args(argv)
        ns.policy_ckpt = "stub"
        return ns

    assert check_policy(p.parse_args([])) is None
    assert check_policy(_stub([])) is None
    for sched in ("yield", "only"):
        flag = ["--policy-fs-schedule", sched]
        for argv in (flag, ["--source", "hw", "--fstereo"] + flag):
            lvl, msg = check_policy(_stub(argv))
            assert lvl == "refuse", (argv, lvl, msg)
            assert f"--policy-fs-schedule {sched} needs --policy: " in msg, msg
            assert "--fstereo" not in msg, msg
        lvl, msg = check_policy(_stub(["--source", "hw", "--mpc", "--policy",
                                       "--policy-allow-device-depth"] + flag))
        assert lvl == "refuse", (lvl, msg)
        assert f"--policy-fs-schedule {sched} needs --fstereo: " in msg, msg
        assert "Add --fstereo, or drop --policy-fs-schedule." in msg, msg
        assert "needs --policy" not in msg and "Add --policy" not in msg, msg
        lvl, msg = check_policy(_stub(["--source", "demo", "--mpc", "--policy"]
                                      + flag))
        assert lvl == "warn", (lvl, msg)
        assert f"--policy-fs-schedule {sched} does nothing with --source demo" \
            in msg, msg
        assert "`free`" in msg and "requested vs effective" in msg, msg
        lvl, msg = check_policy(_stub(["--source", "hw", "--mpc", "--policy",
                                       "--fstereo"] + flag))
        assert lvl == "warn", (lvl, msg)
        if sched == "yield":
            assert "plan age or infer_ms" in msg, msg
            assert "pair spacing" not in msg, msg
        else:
            assert "pair spacing" in msg and "plan age and infer_ms" in msg, msg


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

