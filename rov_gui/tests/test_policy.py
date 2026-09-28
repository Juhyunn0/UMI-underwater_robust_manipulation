#!/usr/bin/env python3
"""
test_policy.py — the LIVE POLICY mission (shape: policy), offline.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_policy.py

The diffusion policy's action chunks, hand-made here as ``PolicyPlan``
objects and fed through ``MpcWorker.on_policy_plan`` — StubCtrl, no acados,
no camera, no torch, no PolicyWorker (rov_gui/tests/test_policy_worker.py and
demo_e2e.py cover the worker). What it pins down, by spec decision
(DP_LIVE_POLICY_SPEC_V2 §A):

* the policy config block validates like replay (unknown keys raise, limits
  must be finite and positive, enums checked) and its FilterLimits are ALL
  explicit from the block (A5, A21);
* each refusal names its own cause: contouring controller, no --policy,
  stale/absent/error/not-ready status, no depth grid, box None (A13, A21);
* arming NEVER moves the vehicle: the first plan anchors where it is (A4);
* A1: a monotonic-domain obs_t handed to the filter is REJECTED; an
  installed plan moves the reference within one period;
* A4: the leashed anchor lands within leash_m of x_meas when the reference
  is 0.3 m away; reference/measured modes do what they say;
* A7: a late plan is skipped without a filter strike; three geometric
  rejects LATCH (halted, PolicyState.active False) and a later good plan
  does not un-latch until STOP/START;
* A8: an epoch-mismatched plan is dropped; STOP-then-START with a plan in
  flight installs nothing;
* A9: a bridge tier != none skips intake, and a long one ends the run;
* A10: ``w_stage`` reaches the controller with the right hold fraction, and
  ``HwDobMpc._mask_W`` scales only the position/velocity block;
* A11: the post-stitch blend gate rejects with a strike and records peaks;
* A12: EVERY mission-clearing site drops the jaw AND releases the estimator;
* A21: the divergence guard uses the policy's own div_max_m;
* A22: max_run_s ends with the "max_run_s reached" wording, never "complete";
* A23: a NaN action is rejected with a reason, not a disengage;
* the ACTION contract (2026-09-07, schema 12): the flown action is the
  5-dim [dx, dy, dz, dyaw, width] (state.POLICY_ACTION_REPR); a status
  reporting another action_repr (or an unset one, or a mount mismatch)
  refuses to ARM, a plan carrying another one is rejected at intake without
  a strike (reject_action_repr) even though a (16, 10) pose10d plan is
  still decodable, |dyaw| > pi is a reject_compose, and the stub's yaw ramp
  reaches the composed plan's raw yaw with the right sign;
* the record: schema 13, strategy plan_stream_policy, the always-written
  policy block (+ action_repr at the top level, in status and in run), CSV
  grip_w_est/grip_g/hold_frac, plans.jsonl fields incl. action_raw (5 wide)
  and action_repr, and a demo meta that never says "real vehicle" (A15);
* the JAW SAMPLE INSTANT (2026-09-07, schema 13): `gripper_lookahead_s`
  0.0 hands the hysteresis the stitcher's g AT now, bit for bit the old
  value, and with the 0907_145206 pattern (a fresh plan every 0.5 s whose
  width ramps open -> closed over 1 s) never closes — the deadlock; 0.8 s
  sees the chunk's far end, fires POLICY gripper CLOSE, auto-neutrals
  after hold_max_s, and `grip_g` records what was seen; the knob is
  validated (finite, 0 <= v <= 3.0, coerced) for both policy and replay.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.control.geometry import (MpcConfig, POLICY_ACTION_REPR,
                                      POLICY_DOWN_SAMPLE_STEPS,
                                      default_policy_block,
                                      validate_policy_block)
from rov_gui.control.plan_stream import FilterLimits, PlanMsg
from rov_gui.state import (Conn, POLICY_GRID_WHY_IDLE, PolicyPlan, PolicyStatus,
                           now)
from rov_gui.tests.test_control import (StubCtrl, _feed_good_state, _fix,
                                        _imu, _worker)

OBS_DT = POLICY_DOWN_SAMPLE_STEPS / 30.0
ROT6D_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# --------------------------------------------------------------- fixtures
def _straight_action(speed=0.05, K=16, width=0.069, lateral=0.0):
    """The stub policy's chunk in the FLOWN contract (pos_yaw_width, (K, 5)):
    knot k at [lateral, 0, speed*k*obs_dt] in the TCP frame (TCP +z = camera
    forward), dyaw 0, constant width."""
    a = np.zeros((K, 5), np.float32)
    a[:, 0] = lateral
    a[:, 2] = speed * np.arange(K) * OBS_DT
    a[:, 3] = 0.0
    a[:, 4] = width
    return a


def _straight_action10(speed=0.05, K=16, width=0.069, lateral=0.0):
    """The same chunk in the LEGACY pose10d contract ((K, 10): identity
    rot6d, width in column 9) -- what the 2026-09-01 checkpoint emitted and
    what the intake must now refuse."""
    a = np.zeros((K, 10), np.float32)
    a[:, 0] = lateral
    a[:, 2] = speed * np.arange(K) * OBS_DT
    a[:, 3:9] = ROT6D_IDENTITY
    a[:, 9] = width
    return a


def _mk_plan(w, action=None, plan_id=None, obs_t=None, epoch=None,
             infer_ms=30.0, action_repr="", ckpt_sha1=""):
    """A hand-made PolicyPlan on the MONOTONIC clock (A1), in the worker's
    current epoch unless told otherwise. ``action_repr`` "" (the default)
    leaves the intake to infer the contract from the width, the way an
    older producer would; the real worker fills it from the checkpoint.
    ``ckpt_sha1`` "" mirrors the REAL stub session (ckpt_sha1_head ""): the
    2026-09-11 checkpoint pin is permissive only when the armed status AND
    the plan both carry no sha, which is exactly the stub's situation."""
    if plan_id is None:
        plan_id = int(w._policy_last_seen) + 1
    t = now()
    return PolicyPlan(
        plan_id=int(plan_id),
        epoch=(int(w._policy_epoch) if epoch is None else int(epoch)),
        obs_t=(t if obs_t is None else float(obs_t)),
        t_emit=t, infer_ms=float(infer_ms),
        action=(_straight_action() if action is None else action),
        lowdim={}, obs_rows_t=(t - OBS_DT, t), obs_fix_t=(t - 0.1, t),
        obs_pair_dt_s=OBS_DT, pair_dup=False, depth_src="identity",
        depth_coverage=1.0, depth_valid=0.99, ckpt_sha1=str(ckpt_sha1),
        action_repr=str(action_repr))


def _feed_status(w, ready=True, depth_src="identity", stamp=None, **kw):
    """A PolicyStatus shaped like the worker's REAL one (verify 2026-09-02):
    a grid that was offered is `grid_ok` unless the test says otherwise, and
    `obs_dt_s` is the checkpoint-derived stride (the shipped contract)."""
    w.on_policy_status(PolicyStatus(
        ready=ready, loading=kw.pop("loading", False),
        error=kw.pop("error", ""), depth_src=depth_src,
        note=kw.pop("note", ""),
        grid_ok=kw.pop("grid_ok", bool(depth_src)),
        grid_why=kw.pop("grid_why", ""),
        obs_dt_s=kw.pop("obs_dt_s", OBS_DT),
        # the checkpoint contract's action (2026-09-07): the flown one
        # unless the test says otherwise, and a mount that agrees
        action_repr=kw.pop("action_repr", POLICY_ACTION_REPR),
        mount_ok=kw.pop("mount_ok", True),
        mount_why=kw.pop("mount_why", ""),
        # the checkpoint the worker HOLDS (2026-09-11 panel picker); "" is
        # what the stub reports, and what the arm pin then treats as "no sha"
        ckpt=kw.pop("ckpt", ""), ckpt_sha1=kw.pop("ckpt_sha1", ""),
        hz=kw.pop("hz", 0.0), infer_ms=kw.pop("infer_ms", 0.0),
        conn=Conn.ONLINE,
        stamp=(now() if stamp is None else float(stamp))))
    assert not kw, f"unknown status fields {sorted(kw)}"


def _feed(w):
    _feed_good_state(w)
    _feed_status(w)


def _observe_worker(tmp, present=True, gripper=False, **policy_over):
    """A worker in POLICY OBSERVE: `opts.policy_observe` True, engaged and
    warmed up with shape=policy selected. Deliberately built through the same
    `_policy_worker` path, so anything the control-mode fixture exercises the
    observe fixture exercises too — the only difference is the flag.

    The VEHICLE is left DISARMED and COMMAND ENABLE off on purpose: observe
    relaxes both gates (there is no wrench to stop) and a fixture that armed
    anyway would not notice if that relaxation regressed.
    """
    from rov_gui.bus import DataBus
    from rov_gui.control.workers import MpcWorker
    from rov_gui.tests.test_control import _app, _test_opts

    _app()
    bus = DataBus()
    base = _test_opts(tmp)

    class Opts(base):
        policy_observe = True

    w = MpcWorker(bus, Opts(), controller_factory=StubCtrl)
    w.setup()
    # A SUBDIRECTORY of the tmpdir. (Until 2026-09-14 `_log_base` derived
    # the observe tree as a SIBLING of log_dir, so `str(tmp)` would have
    # written real folders next to the tmpdir — review 2026-09-03; the kind
    # is a leaf suffix now, but the redirect stays a subdirectory so the
    # test never depends on that.)
    w.cfg.log_dir = str(Path(tmp) / "runs")
    w.cfg.engage["warmup_s"] = 0.1
    w.cfg.engage["settle_s"] = 0.0
    pilots, grips, logs = [], [], []
    bus.cmd_pilot.connect(pilots.append)
    bus.cmd_gripper_drive.connect(grips.append)
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    w.cfg.policy["gripper"] = bool(gripper)
    w.cfg.policy.update(policy_over)
    validate_policy_block(w.cfg.policy)
    w.policy_present = bool(present)
    w.set_scenario({"shape": "policy"})
    _feed(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    assert not w.commanding, "observe must never report commanding"
    for _ in range(4):
        _feed(w)
        w.tick()
    return w, bus, pilots, grips, logs


def _policy_worker(tmp, factory=None, present=True, mode=None, **policy_over):
    """A worker engaged and warmed up with shape=policy selected and a
    'present' policy worker reporting ready. ``mode`` selects the follower
    the way the trajectory panel's combo does (bus.cmd_mpc_mode ->
    set_mode) — applied BEFORE engage, because set_mode refuses while
    engaged (the combo is disabled then too)."""
    if factory is None:
        w, bus, pilots, logs = _worker(tmp)
    else:
        from rov_gui.bus import DataBus
        from rov_gui.control.workers import MpcWorker
        from rov_gui.tests.test_control import _app, _test_opts
        _app()
        bus = DataBus()
        w = MpcWorker(bus, _test_opts(tmp)(), controller_factory=factory)
        w.setup()
        w.cfg.log_dir = str(tmp)
        w.cfg.engage["warmup_s"] = 0.1
        w.cfg.engage["settle_s"] = 0.0
        pilots, logs = [], []
        bus.cmd_pilot.connect(pilots.append)
        bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    w.cfg.policy.update(policy_over)
    validate_policy_block(w.cfg.policy)
    w.policy_present = bool(present)
    w.set_scenario({"shape": "policy"})
    if mode is not None:
        w.set_mode(mode)
        assert w.cfg.mode == mode, (w.cfg.mode, mode)
    w.on_enable(True)
    _feed(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):                     # warmup_s 0.1 @ 20 Hz
        _feed(w)
        w.tick()
    return w, bus, pilots, logs


def _arm(w):
    w.set_traj(True)
    assert w.traj_on, w.reason
    assert w.replay is not None and w.replay["kind"] == "policy"


def _ticks(w, n=1, sleep_s=0.0):
    for _ in range(n):
        _feed(w)
        w.tick()
        if sleep_s:
            time.sleep(sleep_s)


def _run_until(w, cond, timeout_s=6.0, sleep_s=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        _ticks(w)
        if cond():
            return True
        time.sleep(sleep_s)
    return False


def _plans_jsonl(w):
    run_dir = w._csv_path.parent
    p = run_dir / "plans.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


class _FakeBridge:
    """Just enough of station_bridge.StationBridge for the worker's reads:
    a tier, an elapsed time, and the reset/meta the clear sites call."""

    enabled = False
    active = False
    cfg = {}

    def __init__(self, tier):
        self.tier = tier
        self.elapsed = 0.0

    def reset(self):
        pass

    def meta(self):
        return {"enabled": False, "why": "fake"}

    def release_horizontal(self, u):
        return u


# ------------------------------------------------------------------ config
def test_policy_config_defaults_validate_and_bad_values_raise():
    d = validate_policy_block(default_policy_block())
    assert d["anchor"] == "leash" and d["hold_tail"] == "mask"
    assert d["workspace_box_ned"] == [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]]
    assert d["gripper_width_closed_m"] < d["gripper_width_open_m"]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mpc.yaml"
        for text, needle in [
            ("mode: pid\npolicy:\n  vmax: 0.2\n", "vmax"),
            ("mode: pid\npolicy:\n  v_max_m_s: 0\n", "v_max_m_s"),
            ("mode: pid\npolicy:\n  anchor_max_m: .nan\n", "anchor_max_m"),
            ("mode: pid\npolicy:\n  anchor: leashed\n", "anchor"),
            ("mode: pid\npolicy:\n  hold_tail: taper\n", "hold_tail"),
            ("mode: pid\npolicy:\n  hold_tail_extrap_max_m: 0\n",
             "hold_tail_extrap_max_m"),
            ("mode: pid\npolicy:\n  along_scale: 0\n", "along_scale"),
            ("mode: pid\npolicy:\n  along_scale: abc\n", "along_scale"),
            ("mode: pid\npolicy:\n  eval_transforms: crop\n",
             "eval_transforms"),
            ("mode: pid\npolicy:\n  gripper_close_below: 0.8\n"
             "  gripper_open_above: 0.4\n", "close_below"),
            ("mode: pid\npolicy:\n  gripper_width_open_m: 0.03\n", "closed"),
            ("mode: pid\npolicy:\n  workspace_box_ned: [[1, 1, 1], [0, 0, 0]]\n",
             "workspace_box_ned"),
            ("mode: pid\npolicy:\n  tcp_body_flu_m: [1, 2]\n",
             "tcp_body_flu_m"),
            ("mode: pid\npolicy:\n  z_near_m: 4.0\n", "z_near_m"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: 3.0\n", "yaw_ref_filter"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 3.0}\n",
             "yaw_ref_filter keys"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 0, tau_s: 2.0, "
             "guard_deg: 15}\n", "yaw_ref_filter.rate_deg_s"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 3, tau_s: abc, "
             "guard_deg: 15}\n", "yaw_ref_filter.tau_s"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 3, tau_s: 2, "
             "guard_deg: 200}\n", "guard_deg"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 3, tau_s: 2, "
             "guard_deg: 0.5}\n", "guard_deg"),
            ("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 40, tau_s: 2, "
             "guard_deg: 15}\n", "r_max_rad_s"),
        ]:
            p.write_text(text)
            try:
                MpcConfig.load(p)
                raise AssertionError(f"did not raise: {text!r}")
            except ValueError as e:
                assert needle in str(e), (needle, str(e))
        # A good block loads; box None is ALLOWED at load (refused at START).
        p.write_text("mode: pid\npolicy:\n  workspace_box_ned: null\n"
                     "  v_max_m_s: 0.05\n  anchor: measured\n"
                     "  hold_tail: extrapolate\n  hold_tail_extrap_max_m: 0.2\n"
                     "  along_scale: 1.0\n")
        cfg = MpcConfig.load(p)
        assert cfg.policy["workspace_box_ned"] is None
        assert cfg.policy["v_max_m_s"] == 0.05
        assert cfg.policy["anchor"] == "measured"
        assert cfg.policy["hold_tail"] == "extrapolate"
        assert cfg.policy["hold_tail_extrap_max_m"] == 0.2
        assert cfg.policy["along_scale"] == 1.0
        assert default_policy_block()["along_scale"] is None
        # the yaw reference filter ships ON (2026-09-14) and null turns it off
        assert cfg.policy["yaw_ref_filter"] == {"rate_deg_s": 3.0, "tau_s": 2.0,
                                                "guard_deg": 15.0}
        p.write_text("mode: pid\npolicy:\n  yaw_ref_filter: null\n")
        assert MpcConfig.load(p).policy["yaw_ref_filter"] is None
        p.write_text("mode: pid\npolicy:\n  yaw_ref_filter: {rate_deg_s: 5, "
                     "tau_s: 1, guard_deg: 10}\n")
        assert MpcConfig.load(p).policy["yaw_ref_filter"] == {
            "rate_deg_s": 5.0, "tau_s": 1.0, "guard_deg": 10.0}
        # shape: policy is a known shape.
        p.write_text("mode: dobmpc\nsquare:\n  shape: policy\n")
        assert MpcConfig.load(p).square["shape"] == "policy"


def test_policy_filter_limits_equal_config():
    """A5/A21: nothing in the filter is a default — every limit is the
    policy block's own number, including require_obs_t and the yaw gate."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(
            tmp, v_max_m_s=0.07, a_max_m_s2=0.19, r_max_rad_s=0.4,
            anchor_max_m=0.14, jump_max_m=0.05, obs_max_age_s=0.35,
            yaw_jump_max_deg=7.0,
            workspace_box_ned=[[-2.0, -1.0, -0.5], [2.0, 1.0, 0.6]])
        _arm(w)
        pc = w.cfg.policy
        want = FilterLimits(
            v_max=0.07, a_max=0.19, r_max=0.4, anchor_max_m=0.14,
            jump_max_m=0.05, obs_max_age_s=0.35,
            box_ned_min=(-2.0, -1.0, -0.5), box_ned_max=(2.0, 1.0, 0.6),
            reject_escalate=3, require_obs_t=True,
            yaw_jump_max_rad=math.radians(7.0),
            clip_ratio_max=float(pc["clip_ratio_max"]))   # 3.0 default (A5)
        assert w._plan_filter.limits == want, w._plan_filter.limits
        assert w._plan_stitcher.blend_s == pc["blend_s"]
        assert w.ctrl.scenario["kind"] == "policy"
        assert w.ctrl.scenario["T_run_s"] == pc["max_run_s"]
        w.teardown()


# ---------------------------------------------------------------- refusals
def test_policy_refusals_each_name_their_cause():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, present=False)
        # (a) no worker
        w.set_traj(True)
        assert not w.traj_on and "--policy" in w.reason, w.reason
        w.policy_present = True
        # (b) no status ever
        w._policy_status = None
        w.set_traj(True)
        assert not w.traj_on and "not reported" in w.reason, w.reason
        # (c) stale status
        _feed_status(w, stamp=now() - 3.0)
        w.set_traj(True)
        assert not w.traj_on and "silent for" in w.reason, w.reason
        # (d) error
        _feed_status(w, ready=False, error="CUDA out of memory")
        w.set_traj(True)
        assert not w.traj_on and "CUDA out of memory" in w.reason, w.reason
        # (e) loading / not ready, with the builder's note carried
        _feed_status(w, ready=False, loading=True)
        w.set_traj(True)
        assert not w.traj_on and "loading" in w.reason, w.reason
        _feed_status(w, ready=False, note="obs coverage 91.0% < 98.5%")
        w.set_traj(True)
        assert not w.traj_on and "obs coverage 91.0%" in w.reason, w.reason
        # (f) ready but no depth grid yet
        _feed_status(w, ready=True, depth_src="")
        w.set_traj(True)
        assert not w.traj_on and "no depth grid" in w.reason, w.reason
        # (f2) a grid the BUILDER refused (verify 2026-09-02): the worker's
        # real status then reads ready=True, error="", depth_src set — only
        # grid_ok/grid_why say so, and the refusal must name the builder's
        # reason verbatim.
        why = ("obs coverage 97.92% < 98.50% on grid rect_left: pass "
               "--fstereo-alpha 0.5")
        _feed_status(w, ready=True, depth_src="rect_left", grid_ok=False,
                     grid_why=why)
        w.set_traj(True)
        assert not w.traj_on and why in w.reason, w.reason
        assert "REFUSED" in w.reason and "rect_left" in w.reason, w.reason
        # grid_ok False with a depth_src but no reason still refuses
        _feed_status(w, ready=True, depth_src="rect_left", grid_ok=False)
        w.set_traj(True)
        assert not w.traj_on and "REFUSED" in w.reason, w.reason
        # grid_ok False + grid_why but depth_src "" (the builder cleared its
        # kind) refuses with the reason too
        _feed_status(w, ready=True, depth_src="", grid_ok=False, grid_why=why)
        w.set_traj(True)
        assert not w.traj_on and why in w.reason, w.reason
        # (f4) the REAL worker's status before the first depth frame carries
        # the builder's idle sentinel (DepthObsBuilder.why == "no grid set",
        # POLICY_GRID_WHY_IDLE) with depth_src "" — that is "no grid yet",
        # not a refused grid (integration 2026-09-02).
        _feed_status(w, ready=True, depth_src="", grid_ok=False,
                     grid_why=POLICY_GRID_WHY_IDLE)
        w.set_traj(True)
        assert not w.traj_on and "no depth grid" in w.reason, w.reason
        assert "REFUSED" not in w.reason, w.reason
        # (f3) the worker's checkpoint-derived obs_dt disagrees with the
        # config-derived one: refuse, naming BOTH numbers
        _feed_status(w, obs_dt_s=0.1)
        w.set_traj(True)
        assert not w.traj_on and "obs_dt mismatch" in w.reason, w.reason
        assert "100.000 ms" in w.reason and "66.667 ms" in w.reason, w.reason
        assert "dataset_fps" in w.reason
        # an unfilled obs_dt_s (0.0, an older producer) is not a mismatch
        _feed_status(w, obs_dt_s=0.0)
        w.set_traj(True)
        assert w.traj_on, w.reason
        w.set_traj(False)
        # (g) box None
        _feed_status(w)
        w.cfg.policy["workspace_box_ned"] = None
        w.set_traj(True)
        assert not w.traj_on and "workspace_box_ned" in w.reason, w.reason
        # ...and with everything in place it arms.
        w.cfg.policy["workspace_box_ned"] = [[-1, -1, -1], [1, 1, 1]]
        w.set_traj(True)
        assert w.traj_on, w.reason
        w.teardown()

    # (h) a contouring controller refuses the shape outright
    class MpccishStub(StubCtrl):
        progress_m = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=MpccishStub)
        w.set_traj(True)
        assert not w.traj_on and "contours its own path" in w.reason, w.reason
        w.teardown()


# ------------------------------------------------------------- the mission
def test_policy_arms_at_current_pose_and_first_plan_moves_the_reference():
    """Arming moves nothing (A4: zero-offset leash); the first plan anchors
    at the vehicle and, once installed, moves the sampled reference within
    one period (A1's second half)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        eta_at_arm = np.asarray(w._eta, float).copy()
        _arm(w)
        assert w.phase == "policy"
        assert w.ctrl._path_plan is None            # nothing until a plan
        p_ref, yaw_ref, _v = w.ctrl.ref_ned_at(0.0)
        assert np.linalg.norm(p_ref - eta_at_arm[:3]) < 1e-9
        _ticks(w, 2)
        assert w.traj_on and w.ctrl._path_plan is None
        assert w.replay["installed"] == 0
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        rp = w.replay
        assert rp["received"] == 1 and rp["n_plans"] == 1
        assert rp["installed"] == 1 and rp["rejected"] == 0, rp
        plan = w.ctrl._path_plan
        assert plan is not None
        # Stage 0 sits where the vehicle is (plus < 1 cm of the 0.05 m/s
        # motion that elapsed between obs_t and this tick).
        assert np.linalg.norm(np.asarray(plan.p_ned[:, 0]) - eta_at_arm[:3]) < 0.01
        # A straight-ahead chunk through the LEVEL C3 of the test config
        # (tilt 0, camera +z = body +x) walks the reference along datum +x.
        time.sleep(0.5)
        _ticks(w)
        p_ref, _yaw, _v = w.ctrl.ref_ned_at(0.0)
        assert p_ref[0] - eta_at_arm[0] > 0.015, p_ref
        assert abs(p_ref[1] - eta_at_arm[1]) < 1e-3
        assert w.reason == "policy running"
        w.teardown()


def test_policy_monotonic_domain_obs_t_is_rejected_by_the_filter():
    """A1: only the intake converts clocks. A PlanMsg that reaches the filter
    still carrying a monotonic obs_t is in the FUTURE of the mission clock by
    the whole mission origin and must be rejected — the age gate alone would
    have passed it."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        t_rel = now() - w._t0_traj
        p0 = np.asarray(w._eta[:3], float)
        p = np.repeat(p0[:, None], 6, axis=1)
        msg = PlanMsg(plan_id=99, t0=t_rel, dt=0.2, p_ned=p, yaw=np.zeros(6),
                      obs_t=now())                     # NOT converted
        v = w._plan_filter.evaluate(msg, r_now=(p0, 0.0), now=t_rel)
        assert v.status == "reject", v
        assert any("clock domain" in r for r in v.reasons), v.reasons
        w._plan_filter.reset()
        # The same plan through the INTAKE path (a PolicyPlan on the
        # monotonic clock) is converted and accepted.
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1, w.replay
        line = _plans_jsonl(w)[-1]
        assert line["kind"] == "policy" and line["status"] == "accept"
        assert 0.0 <= line["age_at_intake_s"] < 0.4
        assert abs(line["obs_t_rel"] - (line["t_rel_at_intake"]
                                        - line["age_at_intake_s"])) < 1e-6
        assert line["margins"]["obs_age"] > 0.0
        w.teardown()


# ------------------------------------------------------------------ anchor
def test_policy_leash_anchor_lands_within_leash_of_x_meas():
    """A4: with the reference 0.3 m away the anchor moves toward it by at
    most leash_m; `reference` takes it outright; `measured` stays put."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, anchor_leash_m=0.05)
        _arm(w)
        _ticks(w, 3)
        x_meas = np.asarray(w._eta, float).copy()
        t_rel = now() - w._t0_traj
        # Put a reference 0.3 m ahead straight into the stitcher (bypassing
        # the filter — this is about the anchor rule, not the gates).
        far = x_meas[:3] + np.array([0.3, 0.0, 0.0])
        w._plan_stitcher.install(PlanMsg(
            plan_id=7, t0=t_rel - 0.5, dt=0.2,
            p_ned=np.repeat(far[:, None], 8, axis=1),
            yaw=np.full(8, x_meas[5] + math.radians(30.0)), obs_t=t_rel - 0.5),
            now=t_rel - 0.5)
        anchor, info = w._policy_anchor(now(), t_rel)
        d = np.linalg.norm(anchor[:3] - x_meas[:3])
        assert abs(d - 0.05) < 1e-9, (d, info)
        assert info["clipped_pos"] and info["clipped_yaw"]
        assert abs(info["offset_m"] - 0.3) < 1e-9
        assert abs(abs(anchor[5] - x_meas[5]) - math.radians(10.0)) < 1e-9
        assert info["anchor_pose_meas"][0] == x_meas[0]
        assert info["x_meas_src"] == "history"
        # roll/pitch are the measured ones (they become rp_level)
        assert np.allclose(anchor[3:5], x_meas[3:5])
        w.cfg.policy["anchor"] = "reference"
        anchor, info = w._policy_anchor(now(), t_rel)
        assert np.allclose(anchor[:3], far) and info["mode"] == "reference"
        w.cfg.policy["anchor"] = "measured"
        anchor, info = w._policy_anchor(now(), t_rel)
        assert np.allclose(anchor[:3], x_meas[:3])
        assert info["offset_applied_m"] == 0.0
        w.teardown()


# ------------------------------------------------- late plans / escalation
def test_policy_late_plan_is_skipped_without_a_strike():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, obs_max_age_s=0.4)
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, obs_t=now() - 1.0))
        _ticks(w)
        rp = w.replay
        assert rp["late"] == 1 and rp["rejected"] == 0 and rp["installed"] == 0
        assert w._plan_filter._consec_rejects == 0
        assert not rp["halted"] and w.traj_on
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "late" and line["age_at_intake_s"] > 0.9
        assert any("plan late" in m for _l, m in logs), logs[-3:]
        # a fresh one right after is installed as usual
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        w.teardown()


def test_policy_escalation_latch_survives_a_good_plan_until_start():
    """A7: three geometric rejects latch the mission; the reference holds,
    the GPU idles (PolicyState.active False), a later GOOD plan is dropped
    (skip_halted), and only STOP/START (a new epoch) clears it."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        states = []
        bus.policy_state.connect(states.append)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        # 15 m/s chunks: rejected by the kinematics gate every time
        bad = _straight_action(speed=15.0)
        for _ in range(3):
            w.on_policy_plan(_mk_plan(w, action=bad))
            _ticks(w)
        rp = w.replay
        assert rp["rejected"] == 3 and rp["halted"] == "escalated", rp
        assert rp["end_reason"] == "escalated"
        assert w.traj_on and w.engaged                # still armed, holding
        assert w.ctrl._path_plan is not None          # the reference holds
        assert "halted" in w.reason
        assert states[-1].active is False and states[-1].halted is True
        # a GOOD plan now: dropped, not installed
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1 and w.replay["skip_halted"] == 1
        assert w.replay["halted"] == "escalated"
        assert any("escalated" in m for _l, m in logs)
        # STOP: the record keeps the reason; START: a fresh mission
        w.set_traj(False)
        assert w.replay is None and w._replay_last["end_reason"] == "escalated"
        assert w._replay_last["halted"] == "escalated"
        _arm(w)
        assert not w.replay["halted"] and w.replay["installed"] == 0
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        assert states[-1].active is True
        w.teardown()


# ------------------------------------------------------------------- epoch
def test_policy_epoch_mismatch_and_replayed_ids_are_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        e = w._policy_epoch
        w.on_policy_plan(_mk_plan(w, epoch=e - 1))
        _ticks(w)
        assert w.replay["drop_epoch"] == 1 and w.replay["received"] == 0
        w.on_policy_plan(_mk_plan(w, plan_id=5))
        w.on_policy_plan(_mk_plan(w, plan_id=5))           # re-delivery
        w.on_policy_plan(_mk_plan(w, plan_id=3))           # older id
        _ticks(w)
        assert w.replay["received"] == 1 and w.replay["drop_old"] == 2
        assert w.replay["installed"] == 1
        # a plan arriving with NO policy mission armed is counted, not kept
        w.set_traj(False)
        w.on_policy_plan(_mk_plan(w, plan_id=50))
        assert w._policy_inbox is None
        assert w._policy_counts["drop_inactive"] == 1
        w.teardown()


def test_policy_stop_then_start_with_a_plan_in_flight_installs_nothing():
    """A8: the plan was built in the epoch that STOP ended; START opens a
    new one and the inbox is cleared, so the stale chunk can never fly."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        in_flight = _mk_plan(w)                    # epoch E, obs_t now
        w.set_traj(False)                          # STOP
        _arm(w)                                    # START, epoch E+1
        w.on_policy_plan(in_flight)
        _ticks(w, 2)
        assert w.replay["installed"] == 0 and w.replay["received"] == 0
        assert w.replay["drop_epoch"] == 1
        assert w.ctrl._path_plan is None
        # ...and one built in the NEW epoch flies.
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        w.teardown()


# ------------------------------------------------------------------ bridge
def test_policy_bridge_tier_skips_intake_and_a_long_one_ends_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, stale_s=0.3)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        w._bridge = _FakeBridge("imu")
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        rp = w.replay
        assert rp["skip_bridge"] == 1 and rp["installed"] == 1, rp
        assert _plans_jsonl(w)[-1]["status"] == "skip_bridge"
        assert w.traj_on
        # tier back to none: intake resumes
        w._bridge = _FakeBridge("none")
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 2
        # a bridge longer than stale_s ends the mission, holding here
        w._bridge = _FakeBridge("coast")
        assert _run_until(w, lambda: not w.traj_on, timeout_s=3.0)
        assert w._replay_last["end_reason"] == "bridge"
        assert "bridge" in w.reason and w.engaged
        w._bridge = None
        w.teardown()


# --------------------------------------------------------------- hold tail
def test_policy_w_stage_reaches_the_ctrl_with_the_hold_fraction():
    """A10: a 1 s plan inside a 61-stage / 3 s horizon — the stages past
    the plan's end are masked (0) and hold_frac says how many."""

    class Stub61(StubCtrl):
        @property
        def path_plan_steps(self):
            return 61

        @property
        def path_plan_dt(self):
            return 0.05

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=Stub61)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        plan = w.ctrl._path_plan
        assert plan is not None and np.asarray(plan.p_ned).shape == (3, 61)
        ws = np.asarray(plan.w_stage, float)
        assert ws.shape == (61,)
        assert set(np.unique(ws).tolist()) <= {0.0, 1.0}      # hard mask
        assert ws[0] == 1.0 and ws[-1] == 0.0
        assert np.all(np.diff(ws) <= 0.0)                     # 1...1 0...0
        hold_frac = w.replay["hold_frac"]
        assert abs(hold_frac - float(np.mean(ws == 0.0))) < 1e-12
        # 1.0 s of plan (minus the intake age) in a 3.0 s horizon
        assert 0.55 < hold_frac < 0.75, hold_frac
        assert w.ctrl.w_stage_seen and w.ctrl.w_stage_seen[-1] is not None
        # the plan tail itself still holds the endpoint with v = 0
        v = np.asarray(plan.v_ned, float)
        assert np.linalg.norm(v[:, -1]) < 1e-9
        # `track` = today's behaviour: no mask at all
        w.cfg.policy["hold_tail"] = "track"
        _ticks(w)
        assert w.ctrl._path_plan.w_stage is None
        assert w.ctrl.w_stage_seen[-1] is None
        # a cosine taper puts intermediate weights in
        w.cfg.policy["hold_tail"] = "mask"
        w.cfg.policy["hold_tail_taper_s"] = 0.5
        _ticks(w)
        ws = np.asarray(w.ctrl._path_plan.w_stage, float)
        assert np.any((ws > 0.0) & (ws < 1.0)) and ws[0] == 1.0 and ws[-1] == 0.0
        # CSV: hold_frac populated on a policy tick. BY NAME — the 2026-09-08
        # per-thruster block was appended after `observe`, so tail offsets
        # ([-1], [-2]...) no longer land on these columns.
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        assert ",grip_w_est,grip_g,hold_frac,observe," in rows[0]
        col = {n: i for i, n in enumerate(rows[0].split(","))}
        last = rows[-1].split(",")
        assert last[col["observe"]] == "0"                    # not an observe run
        assert 0.4 < float(last[col["hold_frac"]]) < 0.8, last[col["hold_frac"]]
        assert last[col["grip_g"]] == "nan"                   # grip_g: jaw off
        assert abs(float(last[col["grip_w_est"]]) - 0.069) < 1e-6


def test_policy_hold_tail_mask_only_while_a_plan_is_live_and_not_halted():
    """BLOCKER (verify 2026-09-02): once the newest plan has EXPIRED, or the
    escalation latch is set, the ctrl must receive w_stage None — the
    stitcher's endpoint hold at FULL weight — never a mask that zeroes
    x/y/z + heave on the whole horizon. hold_frac 1.0 stays as a record."""

    class Stub61(StubCtrl):
        @property
        def path_plan_steps(self):
            return 61

        @property
        def path_plan_dt(self):
            return 0.05

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=Stub61,
                                              stale_s=10.0)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        assert w.ctrl.w_stage_seen[-1] is not None            # live: masked
        t_end = float(w._plan_stitcher.end_time())
        assert now() - w._t0_traj < t_end
        # --- (a) the plan EXPIRES: fast-forward the mission clock past t_end
        w._t0_traj -= (t_end + 0.5)
        _ticks(w)
        assert now() - w._t0_traj > t_end
        assert w.replay is not None and w.traj_on              # still armed
        assert w.ctrl._path_plan is not None                   # holding
        assert w.ctrl.w_stage_seen[-1] is None, "expired plan must NOT mask"
        assert w.ctrl._path_plan.w_stage is None
        assert w.replay["hold_frac"] == 1.0                    # the record
        v = np.asarray(w.ctrl._path_plan.v_ned, float)
        assert np.linalg.norm(v) < 1e-9                        # endpoint hold
        # the taper branch is gated the same way
        w.cfg.policy["hold_tail_taper_s"] = 0.5
        _ticks(w)
        assert w.ctrl.w_stage_seen[-1] is None
        w.cfg.policy["hold_tail_taper_s"] = 0.0
        # CSV: the last rows carry hold_frac 1.000 as a RECORD only
        w.set_traj(False)
        # --- (b) the LATCH: three rejects with a plan still live
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.ctrl.w_stage_seen[-1] is not None
        bad = _straight_action(speed=15.0)
        for _ in range(3):
            w.on_policy_plan(_mk_plan(w, action=bad))
            _ticks(w)
        rp = w.replay
        assert rp["halted"] == "escalated", rp
        t_end = float(w._plan_stitcher.end_time())
        assert now() - w._t0_traj < t_end, "the plan must still be live here"
        assert w.traj_on and w.ctrl._path_plan is not None
        assert w.ctrl.w_stage_seen[-1] is None, "halted mission must NOT mask"
        assert rp["hold_frac"] < 1.0                           # record only
        _ticks(w, 2)
        assert all(x is None for x in w.ctrl.w_stage_seen[-2:])
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        _hf = rows[0].split(",").index("hold_frac")       # by NAME (schema 14)
        vals = [r.split(",")[_hf] for r in rows[1:]]
        assert "1.000" in vals, vals[-10:]


def test_policy_halted_latch_is_re_read_inside_the_pending_loop():
    """MAJOR (verify 2026-09-02): the 3rd reject sets the latch MID-LOOP;
    a plan still due behind it in `pending` must not be evaluated, let
    alone installed, and nothing installs after the latch."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        rp = w.replay
        assert rp["installed"] == 1
        t_rel = now() - w._t0_traj
        p0 = np.asarray(w._eta[:3], float)
        K = 6

        def _msg(pid, far):
            p = np.repeat(p0[:, None], K, axis=1).astype(float)
            if far:
                p[0] += np.arange(K) * 3.0            # 15 m/s: kinematics reject
            return PlanMsg(plan_id=pid, t0=t_rel - 0.01, dt=0.2, p_ned=p,
                           yaw=np.full(K, float(w._eta[5])),
                           obs_t=t_rel - 0.01)

        # three bad plans and a GOOD one behind them, all due THIS tick
        rp["pending"].extend([_msg(101, True), _msg(102, True),
                              _msg(103, True), _msg(104, False)])
        rp["extras"].update({101: {}, 102: {}, 103: {}, 104: {}})
        released_before = rp["released"]
        _ticks(w)
        assert rp["halted"] == "escalated", rp
        assert rp["rejected"] == 3 and rp["installed"] == 1, rp
        assert rp["released"] == released_before + 3           # 104 never ran
        assert rp["pending"] == [] and rp["extras"] == {}
        ids = [ln.get("plan_id") for ln in _plans_jsonl(w)]
        assert 104 not in ids, ids
        # a due plan pushed AFTER the latch is cleared at the loop head
        rp["pending"].append(_msg(105, False))
        rp["extras"][105] = {}
        _ticks(w)
        assert rp["released"] == released_before + 3
        assert rp["pending"] == [] and rp["extras"] == {}
        assert rp["installed"] == 1 and w.traj_on
        w.teardown()


def test_policy_future_obs_t_is_rejected_at_intake_not_parked_in_pending():
    """The FUTURE half of the clock-domain check at intake (verify
    2026-09-02): a stamp ahead of the mission clock used to be composed
    with a future t0 and sit in `pending` until the clock reached it."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, obs_t=now() + 5.0))
        _ticks(w)
        rp = w.replay
        assert w.engaged and w.traj_on
        assert rp["reject_clock"] == 1 and rp["rejected"] == 1, rp
        assert rp["pending"] == [] and rp["installed"] == 0
        assert w._plan_filter._consec_rejects == 0            # no strike
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject"
        assert any("clock domain" in r and "FUTURE" in r for r in line["reasons"]), line
        assert line["age_at_intake_s"] < -4.0
        # within the tolerance it is fine (its t0 is 20 ms ahead, so the
        # stitcher installs it on the tick the clock reaches it)
        w.on_policy_plan(_mk_plan(w, obs_t=now() + 0.02))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0)
        assert w.replay["reject_clock"] == 1
        w.teardown()


def test_policy_obs_dt_mismatch_is_rejected_at_intake_without_a_strike():
    """MAJOR (verify 2026-09-02): a plan whose checkpoint-derived obs_dt_s
    disagrees with the config-derived stride is rejected with both numbers
    (composing it would rescale its speed); the plans.jsonl line records the
    plan's obs_dt_s beside the config's."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        bad = _mk_plan(w)
        bad.obs_dt_s = 0.1
        w.on_policy_plan(bad)
        _ticks(w)
        rp = w.replay
        assert w.engaged and w.traj_on
        assert rp["reject_obs_dt"] == 1 and rp["rejected"] == 1, rp
        assert rp["installed"] == 0 and w._plan_filter._consec_rejects == 0
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject"
        assert any("obs_dt mismatch" in r and "100.000 ms" in r
                   and "66.667 ms" in r for r in line["reasons"]), line
        assert abs(line["obs_dt_s"] - 0.1) < 1e-12
        assert abs(line["obs_dt_cfg_s"] - OBS_DT) < 1e-12
        # the matching stride installs, and so does an unfilled one (0.0)
        good = _mk_plan(w)
        good.obs_dt_s = OBS_DT
        w.on_policy_plan(good)
        _ticks(w)
        assert w.replay["installed"] == 1
        line = [ln for ln in _plans_jsonl(w) if ln.get("status") == "accept"][-1]
        assert abs(line["obs_dt_s"] - OBS_DT) < 1e-12
        for k in ("anchor_pose_meas", "anchor_pose_used", "anchor_offset_m",
                  "anchor_mode"):
            assert k in line, k
        unset = _mk_plan(w)
        unset.obs_dt_s = 0.0
        w.on_policy_plan(unset)
        _ticks(w)
        assert w.replay["installed"] == 2
        assert _plans_jsonl(w)[-1]["obs_dt_s"] is None
        w.teardown()


def test_legacy_pose10d_plan_is_rejected_at_intake():
    """2026-09-07: a (16, 10) pose10d plan -- what the 2026-09-01 checkpoint
    emits -- is still DECODABLE by compose_plan, which is exactly why the
    intake must refuse it by contract, not by failure: reject_action_repr,
    no strike, still engaged, whether the producer named the representation
    or left it for the width to imply."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        for repr_ in ("", "pose10d"):
            w.on_policy_plan(_mk_plan(w, action=_straight_action10(),
                                      action_repr=repr_))
            _ticks(w)
        rp = w.replay
        assert w.engaged and w.traj_on
        assert rp["reject_action_repr"] == 2 and rp["rejected"] == 2, rp
        assert rp["installed"] == 0 and rp["reject_compose"] == 0
        assert w._plan_filter._consec_rejects == 0            # no strike
        assert rp["action_repr"] == ""                        # nothing composed
        rej = [ln for ln in _plans_jsonl(w) if ln["status"] == "reject"]
        assert len(rej) == 2
        for ln in rej:
            assert any("action_repr mismatch" in r and "pose10d" in r
                       and "pos_yaw_width" in r for r in ln["reasons"]), ln
            assert ln["action_repr"] == "pose10d"             # inferred / declared
            assert np.asarray(ln["action_raw"]).shape == (16, 10) \
                if "action_raw" in ln else True
        assert any("rejected" in m and "action_repr" in m for _l, m in logs)
        # the flown contract installs (inferred from the width AND declared)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action_repr=POLICY_ACTION_REPR))
        _ticks(w)
        assert w.replay["installed"] == 2 and w.replay["reject_action_repr"] == 2
        assert w.replay["action_repr"] == "pos_yaw_width"
        w.teardown()


def test_action_repr_mismatch_is_rejected():
    """A (16, 5) plan whose producer DECLARES another representation is
    rejected at intake (the declaration wins over the width -- a producer
    that mislabels its output is not trusted); the reverse (a (16, 10) array
    declared pos_yaw_width) passes the gate and falls to compose_plan's
    contract check, a reject_compose naming both."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action(),
                                  action_repr="pose10d"))
        _ticks(w)
        rp = w.replay
        assert rp["reject_action_repr"] == 1 and rp["rejected"] == 1, rp
        assert rp["installed"] == 0 and w.engaged and w.traj_on
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject" and line["action_repr"] == "pose10d"
        assert any("action_repr mismatch" in r and "(16, 5)" in r
                   for r in line["reasons"]), line
        w.on_policy_plan(_mk_plan(w, action=_straight_action10(),
                                  action_repr=POLICY_ACTION_REPR))
        _ticks(w)
        rp = w.replay
        assert rp["reject_action_repr"] == 1 and rp["reject_compose"] == 1, rp
        assert rp["rejected"] == 2 and rp["installed"] == 0
        assert w._plan_filter._consec_rejects == 0
        line = _plans_jsonl(w)[-1]
        assert any("action_repr" in r and "pose10d" in r
                   for r in line["reasons"]), line
        w.teardown()


def test_run_meta_carries_action_repr():
    """The record boundary (schema 12): the flown representation at the
    policy block's top level, the checkpoint's in status, the last composed
    plan's (and the reject counter) in run."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        meta = w._run_meta()
        assert meta["schema_version"] == 16     # 16: + 6-DoF attitude axes (2026-09-26); 15: + mode / LOW None holder (2026-09-11)
        pm = meta["policy"]
        assert pm["action_repr"] == POLICY_ACTION_REPR == "pos_yaw_width"
        assert pm["status"]["action_repr"] == "pos_yaw_width"
        assert pm["status"]["mount_ok"] is True and pm["status"]["mount_why"] == ""
        assert pm["run"]["action_repr"] == ""             # nothing composed yet
        assert pm["run"]["reject_action_repr"] == 0
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action10()))
        _ticks(w)
        run = w._run_meta()["policy"]["run"]
        assert run["action_repr"] == "pos_yaw_width"
        assert run["reject_action_repr"] == 1 and run["installed"] == 1
        json.dumps(w._run_meta())
        w.teardown()
        assert w._run_meta()["policy"]["run"]["action_repr"] == "pos_yaw_width"


def test_policy_arm_refuses_action_repr_and_mount_mismatch():
    """The ARM side of the contract: a status whose action_repr is not the
    flown one (or unset -- the worker never read a contract), or whose mount
    check failed, refuses with a sentence naming the cause; the matching
    status arms."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _feed_status(w, action_repr="pose10d")
        w.set_traj(True)
        assert not w.traj_on, w.reason
        assert "action_repr mismatch" in w.reason and "pose10d" in w.reason
        assert "pos_yaw_width" in w.reason and "fix policy.ckpt" in w.reason
        _feed_status(w, action_repr="")
        w.set_traj(True)
        assert not w.traj_on and "(unset)" in w.reason, w.reason
        _feed_status(w, mount_ok=False,
                     mount_why="checkpoint yaw axis R_frd_cam != hw_nav")
        w.set_traj(True)
        assert not w.traj_on and "mount mismatch" in w.reason, w.reason
        assert "R_frd_cam" in w.reason
        _feed_status(w)
        w.set_traj(True)
        assert w.traj_on, w.reason
        w.teardown()


def test_stub_dyaw_rate_reaches_raw_yaw():
    """The stub's yaw ramp (dyaw_rate_rad_s) composes into the installed
    plan's raw yaw with the sign and magnitude of action_raw[-1][3]: the
    5-dim decode is yaw_k = anchor_yaw + dyaw_k, so the raw yaw span equals
    the last knot's dyaw up to the 0.2 s window mean of resample_knots."""
    from rov_gui.perception.dp_policy import StubPolicySession

    s = StubPolicySession("stub", dyaw_rate_rad_s=0.1)
    s.load()
    c = s.contract
    obs = {c["image_key"]: np.zeros((c["obs_horizon"],) + tuple(c["image_shape"]),
                                    np.float32)}
    for k, shp in c["lowdim_shapes"].items():
        obs[k] = np.zeros((c["obs_horizon"],) + tuple(shp), np.float32)
    act, _ = s.predict(obs)
    assert act.shape == (16, 5) and c["action_repr"] == POLICY_ACTION_REPR
    assert abs(act[-1, 3] - 0.1 * 15 * OBS_DT) < 1e-6
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        # This pins the COMPOSITION (stub -> compose_plan); the 2026-09-14
        # yaw reference filter would replace the knots' yaw afterwards, so
        # it is off here and its own test covers the filtered case.
        w.cfg.policy["yaw_ref_filter"] = None
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=act, action_repr=c["action_repr"]))
        _ticks(w)
        assert w.replay["installed"] == 1, w.replay
        line = [ln for ln in _plans_jsonl(w) if ln["status"] in ("accept", "clip")][-1]
        raw = np.asarray(line["action_raw"])
        assert raw.shape == (16, 5) and line["action_repr"] == "pos_yaw_width"
        d_last = float(raw[-1, 3])
        yaw = np.asarray(line["raw"]["yaw"], float)
        span = float(yaw[-1] - yaw[0])
        assert d_last > 0.0 and span > 0.0, (d_last, span)
        assert 0.5 * d_last < span <= d_last + 1e-9, (d_last, span)
        assert np.all(np.diff(yaw) > 0.0), yaw                # a ramp, unwrapped
        assert line["dropped_rp_deg"] == 0.0
        w.teardown()


def test_policy_missing_obs_t_is_a_reject_line_not_a_disengage():
    """verify 2026-09-02: float(None) used to raise inside the guarded tick,
    which DISENGAGES (A23 says a bad plan is a reject with a reason)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        for i, bad_t in enumerate((None, float("nan"), "abc", float("inf"))):
            plan = _mk_plan(w)
            plan.obs_t = bad_t
            w.on_policy_plan(plan)
            _ticks(w)
            rp = w.replay
            assert w.engaged and w.traj_on and rp is not None, (bad_t, w.reason)
            assert rp["reject_clock"] == i + 1 and rp["rejected"] == i + 1, rp
            assert w._plan_filter._consec_rejects == 0
            line = _plans_jsonl(w)[-1]
            assert line["status"] == "reject"
            assert any("obs_t" in r for r in line["reasons"]), line
        assert not any("replay tick raised" in m for _l, m in logs), logs[-3:]
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        w.teardown()


def test_policy_clear_emits_neutral_only_when_the_mission_holds_a_drive():
    """verify 2026-09-02: a mission that never drove the jaw must not emit
    a 0.0 on its way out (a spurious command over a pilot's jaw) — but the
    open-loop estimator is released regardless."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, gripper=True)
        drives: list[float] = []
        bus.cmd_gripper_drive.connect(drives.append)
        _arm(w)
        _ticks(w, 2)
        assert w.replay["grip_drive"] == 0.0 and not drives
        # a PILOT drive (not the mission's) is latched in the estimator
        w.on_gripper_drive(+1.0)
        assert w._grip_est.level == +1.0
        n_before = w._grip_est.n_drives
        w.set_traj(False)
        assert drives == [], drives                            # no spurious 0.0
        assert w._grip_est.level == 0.0                        # still released
        assert w._grip_est.n_drives == n_before + 1
        # ...and a mission that DID drive still neutrals on the way out
        _arm(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action(width=0.042)))
        assert _run_until(w, lambda: -1.0 in drives, timeout_s=3.0), drives
        w.set_traj(False)
        assert drives[-1] == 0.0, drives
        w.teardown()


def test_mask_W_scales_only_the_position_velocity_block():
    from rov_gui.control.mpc_bridge import HwDobMpc

    ny = 18
    W = np.diag(np.arange(1.0, ny + 1.0))
    # an off-diagonal (x, y) term like the tuned rotation writes
    W[0, 1] = W[1, 0] = 0.5
    out = HwDobMpc._mask_W(W, 0.25)
    assert np.allclose(out, out.T)                        # symmetric
    idx = list(HwDobMpc.W_STAGE_ROWS)
    assert idx == [0, 1, 2, 6, 7, 8]
    assert np.allclose(np.diag(out)[idx], np.diag(W)[idx] * 0.25)
    keep = [i for i in range(ny) if i not in idx]
    assert np.allclose(np.diag(out)[keep], np.diag(W)[keep])   # yaw, rates, R
    assert out[0, 1] == 0.125
    assert np.array_equal(HwDobMpc._mask_W(W, 1.0), W)
    z = HwDobMpc._mask_W(W, 0.0)
    assert np.all(z[np.ix_(idx, idx)] == 0.0) and z[5, 5] == 6.0
    assert not np.shares_memory(out, W)


# -------------------------------------------------------------- blend gate
def test_policy_blend_gate_rejects_with_a_strike_and_records_peaks():
    """A11: a plan that passes every knot gate but whose BLEND would demand
    more than blend_v_max is rejected post-stitch — and it is a geometric
    strike. Isolated by opening the jump gate so only the blend can fail."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(
            tmp, jump_max_m=0.5, blend_v_max_m_s=0.16, blend_s=0.4,
            anchor_leash_m=0.0)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        # 0.06 m sideways (camera +x = body +y) for the whole chunk: the
        # cosine blend must cover it in 0.4 s -> pi/(2*0.4)*0.06 = 0.24 m/s
        # > 0.16, while the knot speed is still 0.05 m/s.
        w.on_policy_plan(_mk_plan(w, action=_straight_action(lateral=0.06)))
        _ticks(w)
        rp = w.replay
        assert rp["rejected"] == 1 and rp["reject_blend"] == 1, rp
        assert rp["installed"] == 1
        assert w._plan_filter._consec_rejects == 1
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject"
        assert any("blend" in r for r in line["reasons"]), line["reasons"]
        assert line["blend"]["v_peak"] > 0.16
        assert line["margins"]["blend_v"] < 0.0
        assert line["margins"]["speed"] > 0.0          # knots were fine
        # an accepted plan carries the peaks too (positive margins)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "accept" and line["margins"]["blend_v"] > 0.0
        assert w._plan_filter._consec_rejects == 0
        w.teardown()


# ---------------------------------------------------------------- gripper
def test_policy_every_clear_site_drops_the_jaw_and_releases_the_estimator():
    """A12: the 4-site rule, per site, plus ENABLE off — and the open-loop
    estimator stops integrating with the drive."""
    gestures = [
        ("stop_traj", lambda w: w.set_traj(False)),
        ("disengage", lambda w: w.disengage("test")),
        ("estop", lambda w: w.estop()),
        ("enable_off", lambda w: w.on_enable(False)),
        ("re_arm_station", lambda w: (w.set_scenario({"shape": "station"}),
                                      w.set_traj(True))),
        ("teardown", lambda w: w.teardown()),
    ]
    for name, gesture in gestures:
        with tempfile.TemporaryDirectory() as tmp:
            w, bus, pilots, logs = _policy_worker(tmp, gripper=True)
            drives: list[float] = []
            bus.cmd_gripper_drive.connect(drives.append)
            _arm(w)
            assert w.replay["grip_state"] == "open"    # est init = OPEN
            # width 0.042 -> g = 0 < close_below: a CLOSE edge
            w.on_policy_plan(_mk_plan(w, action=_straight_action(width=0.042)))
            assert _run_until(w, lambda: -1.0 in drives, timeout_s=3.0), \
                f"[{name}] no CLOSE drive was emitted (got {drives})"
            assert w._grip_est.level == -1.0, name
            w_before = w._grip_est.width(now())
            time.sleep(0.1)
            assert w._grip_est.width(now()) < w_before - 1e-4, name  # closing
            gesture(w)
            assert drives[-1] == 0.0, (name, drives)
            assert w._grip_est.level == 0.0, name
            assert w.replay is None or name == "re_arm_station", name
            if name == "stop_traj":
                meta = w._run_meta()
                assert meta["policy"]["run"]["grip_events"] >= 1
                assert meta["policy"]["run"]["end_reason"] == "stop"
            if name != "teardown":
                w.teardown()


def test_policy_gripper_off_by_default_never_drives():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        drives: list[float] = []
        bus.cmd_gripper_drive.connect(drives.append)
        _arm(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action(width=0.042)))
        _ticks(w, 5)
        assert w.replay["installed"] == 1 and not drives
        w.teardown()


def _ramp_action(w_open=0.069, w_closed=0.042, close_at_s=1.0, K=16):
    """A STATIONARY chunk (no motion, dyaw 0) whose width column ramps
    linearly from open at knot 0 to closed at ``close_at_s`` and stays
    closed: g(tau) = 1 - tau / close_at_s on [0, close_at_s], 0 after. The
    0907_145206 shape — "close, but at my far end" on every chunk."""
    a = np.zeros((K, 5), np.float32)
    frac = np.clip(np.arange(K) * OBS_DT / float(close_at_s), 0.0, 1.0)
    a[:, 4] = w_open + (w_closed - w_open) * frac
    return a


def _stream_ramp_plans(w, seconds, period_s=0.5, **ramp_kw):
    """Feed a fresh ramp plan every ``period_s`` of WALL time for
    ``seconds`` while ticking at ~20 Hz: a plan that always asks to close at
    its far end and is always replaced before it gets there."""
    t0 = now()
    t_next = t0
    while now() - t0 < seconds:
        if now() >= t_next:
            w.on_policy_plan(_mk_plan(w, action=_ramp_action(**ramp_kw)))
            t_next += float(period_s)
        _ticks(w)
        time.sleep(0.02)


def _spy_jaw_samples(w):
    """Wrap `_tick_replay_gripper` to record (g handed to the hysteresis,
    the stitcher's g AT now) per tick, then call the real one."""
    seen: list[tuple] = []
    orig = w._tick_replay_gripper

    def spy(g, t_rel):
        g_now = float(w._plan_stitcher.sample(np.array([float(t_rel)]))[4][0])
        seen.append((g, g_now))
        return orig(g, t_rel)
    w._tick_replay_gripper = spy
    return seen


def test_gripper_lookahead_zero_matches_now_sample():
    """lookahead 0.0 == the pre-schema-13 jaw sample, bit for bit: the value
    the hysteresis is handed is the stitcher's g AT now (horizon knot 0) and
    the CSV `grip_g` column records exactly that — and with the 0907_145206
    pattern (every 0.5 s a fresh plan whose width ramps open -> closed over
    1.0 s) NO close edge ever fires: the deadlock, reproduced."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, gripper=True,
                                              gripper_lookahead_s=0.0)
        drives, events = [], []
        bus.cmd_gripper_drive.connect(drives.append)
        bus.mpc_event.connect(events.append)
        seen = _spy_jaw_samples(w)
        _arm(w)
        assert w.replay["grip_lookahead_s"] == 0.0
        assert w.replay["grip_state"] == "open"
        _stream_ramp_plans(w, seconds=2.0, period_s=0.5, close_at_s=1.0)
        rp = w.replay
        assert rp is not None and rp["installed"] >= 3, rp
        assert len(seen) >= 20, len(seen)
        assert all(g is not None for g, _ in seen)
        assert all(g == g_now for g, g_now in seen), \
            [(g, g_now) for g, g_now in seen if g != g_now][:3]
        # every plan said "close" at its far end (width -> closed by 1 s)...
        assert _ramp_action(close_at_s=1.0)[-1, 4] <= 0.042 + 1e-6
        # ... and the sample at now never got there before the next plan
        assert min(g for g, _ in seen) > 0.30, min(seen)
        assert -1.0 not in drives and rp["grip_events"] == 0, drives
        assert rp["grip_state"] == "open"
        assert not any("gripper CLOSE" in e for e in events), events
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        cols = rows[0].split(",")
        _i = cols.index("grip_w_est")
        assert cols[_i:_i + 4] == ["grip_w_est", "grip_g", "hold_frac", "observe"]
        ig = cols.index("grip_g")
        vals = [r.split(",")[ig] for r in rows[1:]]
        assert vals[0] == "nan"                    # warm-up: no mission yet
        got = [v for v in vals if v != "nan"]
        assert got == [f"{g:.4f}" for g, _ in seen], (got[:4], seen[:4])


def test_gripper_lookahead_closes_on_intent():
    """Same plans, lookahead 0.8 s: the hysteresis sees the chunk's far end
    (g(tau + 0.8) < close_below while the sample at now still does not), a
    POLICY gripper CLOSE edge fires (cmd_gripper_drive -1), the CSV grip_cmd
    reads -1 and then 0 after gripper_hold_max_s (auto-neutral, state stays
    closed so the next fresh plan re-edges nothing), and grip_g / the meta
    record the look-ahead that was flown."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(
            tmp, gripper=True, gripper_lookahead_s=0.8, gripper_hold_max_s=0.5)
        drives, events = [], []
        bus.cmd_gripper_drive.connect(drives.append)
        bus.mpc_event.connect(events.append)
        seen = _spy_jaw_samples(w)
        _arm(w)
        assert w.replay["grip_lookahead_s"] == 0.8
        # the started line (bus.log + events.log) says what was flown
        assert any("gripper ON, lookahead 0.8 s" in m for _l, m in logs), \
            [m for _l, m in logs if "POLICY" in m][-2:]
        _stream_ramp_plans(w, seconds=2.0, period_s=0.5, close_at_s=1.0)
        rp = w.replay
        assert rp is not None and rp["installed"] >= 3, rp
        assert drives and drives[0] == -1.0, drives
        assert any("POLICY gripper CLOSE" in e for e in events), events
        assert rp["grip_events"] == 1 and rp["grip_state"] == "close", rp
        assert rp["grip_drive"] == 0.0, "hold_max_s auto-neutral never fired"
        assert drives == [-1.0, 0.0], drives
        # the SAME plans: the sample at now never crossed, the look-ahead did
        assert min(g_now for _, g_now in seen) > 0.30
        assert min(g for g, _ in seen) < 0.30
        meta = w._run_meta()
        assert meta["schema_version"] == 16     # 16: + 6-DoF attitude axes (2026-09-26); 15: + mode / LOW None holder (2026-09-11)
        assert meta["policy"]["config"]["gripper_lookahead_s"] == 0.8
        assert meta["policy"]["estimator"]["lookahead_s"] == 0.8
        assert meta["policy"]["run"]["grip_lookahead_s"] == 0.8
        assert meta["trajectory"]["gripper_lookahead_s"] == 0.8
        w.teardown()
        ev = (w._csv_path.parent / "events.log").read_text()
        assert "gripper ON, lookahead 0.8 s" in ev, ev[-400:]
        rows = w._csv_path.read_text().splitlines()
        cols = rows[0].split(",")
        ic, ig = cols.index("grip_cmd"), cols.index("grip_g")
        data = [r.split(",") for r in rows[1:]]
        cmds = [r[ic] for r in data if r[ic] != "nan"]
        assert "-1" in cmds and cmds[-1] == "0", cmds
        i_close = cmds.index("-1")
        i_neutral = next(i for i in range(i_close, len(cmds)) if cmds[i] == "0")
        assert i_neutral - i_close >= 5, (i_close, i_neutral)   # ~0.5 s @ 20 Hz
        assert all(c == "-1" for c in cmds[i_close:i_neutral])
        assert all(c == "0" for c in cmds[i_neutral:])
        got = [r[ig] for r in data if r[ig] != "nan"]
        assert got == [f"{g:.4f}" for g, _ in seen], (got[:4], seen[:4])
        # the row that first reads grip_cmd -1 saw a grip_g below close_below
        rows_cmd = [r for r in data if r[ic] != "nan"]
        assert rows_cmd[i_close][ig] != "nan"
        assert float(rows_cmd[i_close][ig]) < 0.30, rows_cmd[i_close][ig]


def test_gripper_lookahead_validated():
    """The knob is coerced and range-checked at load for BOTH stream kinds:
    finite, 0 <= v <= 3.0 (the NMPC horizon); negatives, > 3, NaN/inf and
    non-numerics raise naming the key; the default is 0.0."""
    assert default_policy_block()["gripper_lookahead_s"] == 0.0
    for bad in (-0.1, -1, 3.01, 10.0, float("nan"), float("inf"), "abc",
                None, [0.5]):
        pc = default_policy_block()
        pc["gripper_lookahead_s"] = bad
        try:
            validate_policy_block(pc)
            raise AssertionError(f"gripper_lookahead_s {bad!r} did not raise")
        except ValueError as e:
            assert "gripper_lookahead_s" in str(e), (bad, e)
    for good, want in ((0, 0.0), (0.0, 0.0), ("0.6", 0.6), (3, 3.0),
                       (np.float32(0.8), 0.8)):
        pc = default_policy_block()
        pc["gripper_lookahead_s"] = good
        validate_policy_block(pc)
        v = pc["gripper_lookahead_s"]
        assert type(v) is float and abs(v - want) < 1e-6, (good, v)
    # ...and through MpcConfig.load, for the replay block and the policy block
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mpc.yaml"
        for block in ("replay", "policy"):
            p.write_text(f"mode: pid\n{block}:\n  gripper_lookahead_s: -1\n")
            try:
                MpcConfig.load(p)
                raise AssertionError(f"{block}.gripper_lookahead_s -1 loaded")
            except ValueError as e:
                assert "gripper_lookahead_s" in str(e), e
            p.write_text(f"mode: pid\n{block}:\n  gripper_lookahead_s: 4\n")
            try:
                MpcConfig.load(p)
                raise AssertionError(f"{block}.gripper_lookahead_s 4 loaded")
            except ValueError as e:
                assert "gripper_lookahead_s" in str(e), e
            p.write_text(f"mode: pid\n{block}:\n  gripper_lookahead_s: 0.6\n")
            cfg = MpcConfig.load(p)
            assert getattr(cfg, block)["gripper_lookahead_s"] == 0.6
        cfg = MpcConfig()
        assert cfg.replay["gripper_lookahead_s"] == 0.0
        assert cfg.policy["gripper_lookahead_s"] == 0.0


# ---------------------------------------------------------- status / ends
def test_policy_status_going_silent_mid_run_ends_with_worker_silent():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        _feed_good_state(w)
        _feed_status(w, stamp=now() - 2.5)
        w.tick()
        assert not w.traj_on and w.engaged
        assert w._replay_last["end_reason"] == "worker_silent"
        assert "worker lost" in w.reason, w.reason
        w.teardown()


def test_policy_divergence_guard_uses_its_own_limit():
    """A21: the policy's div_max_m (0.25) — a 0.35 m excursion that a
    REPLAY (2 x 0.30 m) would tolerate stops the policy."""
    from rov_gui.state import Telemetry

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, div_max_m=0.25)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 3)
        assert w.traj_on and w.ctrl._path_plan is not None

        def _feed_far():
            t = now()
            w.on_nav_fix(_fix([-2.0, 0.35, 0.8], 0.0, t))
            w.on_vehicle_imu(_imu(depth=0.8, t=t))
            w.on_telemetry(Telemetry(armed=True, mode="MANUAL",
                                     conn=Conn.ONLINE))
            _feed_status(w)
        t0 = time.monotonic()
        while w.traj_on and time.monotonic() - t0 < 4.0:
            _feed_far()
            w.tick()
            time.sleep(0.02)
        assert not w.traj_on, "divergence guard never fired"
        assert "diverged" in w.reason, w.reason
        assert w._replay_last["end_reason"] == "diverged"
        assert w._replay_last["halted"] == "diverged"
        w.teardown()


def test_policy_max_run_ends_with_the_a22_wording_never_complete():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, max_run_s=0.4)
        events: list[str] = []
        bus.mpc_event.connect(events.append)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert _run_until(w, lambda: not w.traj_on, timeout_s=3.0)
        assert "max_run_s reached" in w.reason, w.reason
        assert "complete" not in w.reason
        assert w._replay_last["end_reason"] == "max_run"
        assert "POLICY max_run_s reached (1 plans installed)" in events, events
        assert not any("complete" in e.lower() for e in events), events
        assert w.engaged                                   # holding, not off
        w.teardown()


def test_policy_nan_action_is_rejected_not_a_disengage():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        bad = _straight_action()
        bad[3, 1] = np.nan
        w.on_policy_plan(_mk_plan(w, action=bad))
        _ticks(w)
        rp = w.replay
        assert w.engaged and w.traj_on
        assert rp["reject_compose"] == 1 and rp["rejected"] == 1
        assert w._plan_filter._consec_rejects == 0        # not a strike
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject"
        assert any("non-finite" in r for r in line["reasons"]), line
        # a |dyaw| > pi knot likewise (the 5-dim analogue of the degenerate
        # rot6d the pose10d path rejected)
        bad = _straight_action()
        bad[7, 3] = 4.0
        w.on_policy_plan(_mk_plan(w, action=bad))
        _ticks(w)
        assert w.replay["reject_compose"] == 2 and w.engaged
        assert any("dyaw" in r for r in _plans_jsonl(w)[-1]["reasons"]), \
            _plans_jsonl(w)[-1]
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1
        w.teardown()


# ------------------------------------------------------------ PolicyState
def test_policy_state_is_emitted_every_tick_only_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, present=False)
        states = []
        bus.policy_state.connect(states.append)
        _ticks(w, 3)
        assert not states
        w.policy_present = True
        _ticks(w, 3)
        assert len(states) == 3
        s = states[-1]
        assert s.engaged and not s.active and not s.halted
        assert s.epoch == w._policy_epoch
        assert s.t_fix == w.fix.t_capture and s.fix_fresh
        assert s.eta_start is None and s.t0_traj is None
        assert abs(s.grip_width_m - 0.069) < 1e-9
        assert np.allclose(s.eta, np.asarray(w._eta, float))
        e_before = w._policy_epoch
        _arm(w)
        assert w._policy_epoch == e_before + 1
        _ticks(w)
        s = states[-1]
        assert s.active and s.eta_start is not None and s.t0_traj == w._t0_traj
        assert s.epoch == w._policy_epoch
        # the worker's own fix-keyed history filled from the same feed
        assert len(w._policy_eta_hist) >= 1
        w.set_traj(False)
        _ticks(w)
        assert not states[-1].active
        w.teardown()


# ------------------------------------------------------------- the record
def test_policy_meta_csv_and_plans_jsonl_carry_the_record():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        w.policy_meta_fn = lambda: {"stub": True, "hz": 2.0,
                                    "session": {"stub": True}}
        _arm(w)
        w.on_policy_plan(_mk_plan(w, infer_ms=42.0))
        _ticks(w, 2)
        meta = w._run_meta()
        assert meta["schema_version"] == 16     # 16: + 6-DoF attitude axes (2026-09-26); 15: + mode / LOW None holder (2026-09-11)
        assert meta["trajectory"]["kind"] == "policy"
        assert meta["reference_clock"]["strategy"] == "plan_stream_policy"
        assert meta["plan_stream"]["enabled"] is True
        assert meta["plan_stream"]["run"]["kind"] == "policy"
        pm = meta["policy"]
        assert pm["enabled"] is True and pm["synthetic"] is True   # demo opts
        assert pm["worker"]["stub"] is True
        assert pm["config"]["anchor"] == "leash"
        assert set(pm["config_provenance"]) >= {
            "v_max_m_s", "anchor_leash_m", "gripper_width_open_m",
            "tcp_body_flu_m", "knot_dt_s"}
        # "[결정" joined the set on 2026-09-02 for `follower_owns_dynamics`.
        # The other tags all describe where a NUMBER came from; that key is
        # not a number but an architectural choice (who owns dynamics — the
        # filter or the NMPC), and squeezing an operator decision into
        # "[예측]" would misfile it as a guess at a quantity. The plan doc
        # already calls these 결정 (§1 확정 결정 로그).
        assert all(any(tag in v for tag in ("[측정", "[스펙", "[유도", "[예측",
                                            "[가정", "[결정", "None", "[예측]"))
                   for v in pm["config_provenance"].values())
        # verify 2026-09-02: the shared provenance sentences, verbatim
        prov = pm["config_provenance"]
        assert prov["gripper_width_open_m"] == prov["gripper_width_closed_m"]
        assert ("open_level_m p10–p90 0.055–0.070 (median 0.061), "
                "closed_level_m p10–p90 0.034–0.043 (median 0.040)"
                in prov["gripper_width_open_m"])
        assert prov["gripper_width_open_m"].endswith("[예측]]")
        assert "gate_20260902_clip3.log" in prov["clip_ratio_max"]
        assert prov["clip_ratio_max"].startswith("[예측: 3.0 chosen from the gate")
        assert "training_validity_20260902.txt" in prov["min_obs_coverage"]
        assert prov["min_obs_coverage"].startswith("[유도: 0.985 sits at the 0.06 %")
        blob = json.dumps(meta, ensure_ascii=False)
        assert "0.9878" not in blob and "0.062–0.069" not in blob, \
            "un-artifacted numbers must be gone from the record"
        tcp = pm["tcp"]
        assert np.asarray(tcp["T_body_tcp"]).shape == (4, 4)
        assert tcp["handheld_tcp_offset_cam_m"] == [0.0355, 0.1293, 0.3186]
        assert len(tcp["tcp_offset_cam_m_used"]) == 3
        assert tcp["tcp_offset_cam_m_override"] is None
        assert 0.3 < tcp["t_body_tcp_norm_m"] < 0.6      # the sim jaw centre
        assert pm["estimator"]["width_now_m"] == 0.069
        assert pm["anchor_mode"] == "leash" and pm["hold_tail"] == "mask"
        assert abs(pm["obs_dt_s"] - OBS_DT) < 1e-12
        run = pm["run"]
        assert run["installed"] == 1 and run["end_reason"] == ""
        assert run["infer_ms_p50"] == 42.0 and run["live"] is True
        assert run["epoch"] == w._policy_epoch
        assert run["reject_clock"] == 0 and run["reject_obs_dt"] == 0
        assert pm["status"]["grid_ok"] is True and pm["status"]["grid_why"] == ""
        assert abs(pm["status"]["obs_dt_s"] - OBS_DT) < 1e-12
        json.dumps(meta)                                   # serialisable
        # verify 2026-09-02: numpy scalars from the worker (hz/infer_ms of a
        # np.percentile) and from a plan must not lose the meta.json
        w.on_policy_plan(_mk_plan(w, infer_ms=np.float32(40.0)))
        _ticks(w)
        _feed_status(w, hz=np.float32(2.5), infer_ms=np.float64(31.0))
        meta = w._run_meta()
        assert meta["policy"]["status"]["hz"] == 2.5
        assert isinstance(meta["policy"]["status"]["hz"], float)
        json.dumps(meta)
        w._write_meta()
        written = json.loads(w._csv_path.with_suffix(".meta.json").read_text())
        assert written["policy"]["status"]["infer_ms"] == 31.0
        assert written["policy"]["run"]["installed"] == 2
        assert not any("mpc meta:" in m for _l, m in logs), logs[-3:]
        # plans.jsonl: the A1/A2/A4/A5/A11 fields and the raw action
        line = _plans_jsonl(w)[-1]
        for k in ("kind", "plan_id", "epoch", "obs_t_rel", "t_rel_at_intake",
                  "age_at_intake_s", "infer_ms", "depth_src", "depth_coverage",
                  "anchor_mode", "anchor_offset_ned", "anchor_offset_m",
                  "anchor_pose_meas", "anchor_pose_used", "dropped_rp_deg",
                  "jitter_rms_mm", "n_raw", "n_knots", "blend", "obs_rows_t",
                  "obs_fix_t", "obs_pair_dt_s", "pair_dup", "ckpt_sha1",
                  "action_raw", "action_repr", "raw", "margins"):
            assert k in line, k
        assert line["action_repr"] == "pos_yaw_width"
        assert np.asarray(line["action_raw"]).shape == (16, 5)
        assert line["n_raw"] == 16 and line["n_knots"] == 6 and line["dt"] == 0.2
        assert "need" in line["margins"]
        assert line["dropped_rp_deg"] == 0.0      # by construction on the 5-dim path
        # CSV tail
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        # BY NAME: schema 14 appended the per-thruster block after `observe`
        assert (",tick_ms,plan_id,ref_src,grip_cmd,grip_w_est,grip_g,"
                "hold_frac,observe," in rows[0])
        col = {n: i for i, n in enumerate(rows[0].split(","))}
        data = [r.split(",") for r in rows[1:]]
        assert any(r[col["hold_frac"]] != "nan" for r in data), \
            "hold_frac never populated"
        assert all(r[col["grip_g"]] == "nan" for r in data), \
            "grip_g must be nan: jaw off"
        assert all(r[col["grip_w_est"]] != "nan" for r in data), \
            "grip_w_est missing"
        # after the run the policy block is STILL written (last run)
        meta = w._run_meta()
        assert meta["policy"]["run"]["end_reason"] == "disengaged"
        assert meta["policy"]["run"]["live"] is False


def test_policy_demo_meta_never_says_real_vehicle():
    """A15: the source string comes from the backend's --source."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        assert w.opts.source == "demo"
        meta = w._run_meta()
        assert "real vehicle" not in json.dumps(meta)
        assert "SYNTHETIC" in meta["source"]
        assert meta["policy"]["synthetic"] is True
        w.opts.source = "hw"
        meta = w._run_meta()
        assert meta["source"] == "hardware rov_gui.control (real vehicle)"
        assert meta["policy"]["synthetic"] is False
        w.opts.source = "demo"
        w.teardown()


def test_policy_without_a_mission_leaves_other_shapes_untouched():
    """The policy plumbing must not leak into a plain station hold: no
    strategy relabel, no policy counters, hold_frac nan."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        w.set_scenario({"shape": "station"})
        w.set_traj(True)
        assert w.station is not None
        _ticks(w, 2)
        meta = w._run_meta()
        assert meta["reference_clock"]["strategy"] != "plan_stream_policy"
        assert meta["policy"]["run"] is None
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        assert rows[-1].split(",")[-2] == "nan"       # hold_frac
        assert (rows[-1].split(",")[rows[0].split(",").index("observe")]
                == "0")                                   # observe


# =============================================================================
# POLICY OBSERVE (--policy-observe, 2026-09-03)
# =============================================================================
# The DP runs, its plans are composed and drawn, and NOTHING leaves the
# station: the pilot flies by hand and checks the reference from wherever they
# like. These tests pin the three things that make that safe and honest —
# nothing is actuated, nothing stops the run for a follower that is not there,
# and no record from it can be mistaken for a closed-loop one.
def test_observe_never_emits_a_single_command():
    """THE guarantee. 200 ticks of a live, plan-installing policy mission and
    not one cmd_pilot / cmd_gripper_drive frame — including across the
    disengage, which in a control run emits an explicit neutral."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, gripper=True,
                                                      stale_s=100.0)
        _arm(w)
        for i in range(200):
            if i % 10 == 0:
                w.on_policy_plan(_mk_plan(w))
            _ticks(w)
        assert w.replay["installed"] >= 1, "no plan ever installed"
        assert pilots == [], f"{len(pilots)} pilot frames left an OBSERVE run"
        assert grips == [], f"{len(grips)} jaw drives left an OBSERVE run"
        w.disengage("test")
        assert pilots == [], "disengage emitted a neutral into the pilot's own "\
                             "command stream (the window never yielded)"
        assert grips == []


def test_observe_never_steps_the_controller():
    """`ctrl.step()` is not called at all — so no wrench exists to leak, the
    EAOB is never told a phantom `note_applied`, and `_axes_prev` (the slew
    memory a LATER real engagement ramps from) stays empty."""
    class CountingCtrl(StubCtrl):
        steps = 0
        applied_calls = 0

        def step(self, *a, **k):
            type(self).steps += 1
            return super().step(*a, **k)

        def note_applied(self, tau):
            type(self).applied_calls += 1
            return super().note_applied(tau)

    CountingCtrl.steps = CountingCtrl.applied_calls = 0
    with tempfile.TemporaryDirectory() as tmp:
        from rov_gui.bus import DataBus
        from rov_gui.control.workers import MpcWorker
        from rov_gui.tests.test_control import _app, _test_opts
        _app()
        bus = DataBus()
        base = _test_opts(tmp)

        class Opts(base):
            policy_observe = True

        w = MpcWorker(bus, Opts(), controller_factory=CountingCtrl)
        w.setup()
        w.cfg.log_dir = str(tmp)
        w.cfg.engage["warmup_s"] = 0.1
        w.cfg.engage["settle_s"] = 0.0
        w.policy_present = True
        w.set_scenario({"shape": "policy"})
        _feed(w)
        w.set_engaged(True)
        _ticks(w, 4)                          # warm-up, at 20 Hz
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 30)
        assert CountingCtrl.steps == 0, "the controller stepped in OBSERVE"
        assert CountingCtrl.applied_calls == 0
        assert w._axes_prev is None, "slew memory filled with unsent commands"
        # ...and the PLAN pipeline did run: this is a monitor, not an off switch
        assert w.ctrl._path_plan is not None, "no reference was composed"


def test_observe_status_says_engaged_but_not_commanding():
    """The two bits the window reads. `engaged` True (the machinery is live,
    which is what keeps the DP worker inferring); `commanding` False, which is
    what leaves the joystick with the pilot; `observe` True for the panel."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp)
        seen = []
        bus.mpc_status.connect(seen.append)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 5)
        s = seen[-1]
        assert s.engaged is True
        assert s.commanding is False, "the window would steal the joystick"
        assert s.observe is True
        assert s.u_cmd == (), "a wrench was published for a muted loop"
        assert s.axes == ()
        # The reference IS published — drawing it is the point of the mode —
        # while the TRACKING errors are not (there is no tracker).
        assert s.ref_flu is not None
        assert s.err_xy is None and s.err_cross is None and s.err_along is None
        assert s.ref_speed_m_s is None


def test_observe_disarms_the_five_stops_that_assume_a_follower():
    """Divergence, the escalation latch, bridge-too-long, the workspace box
    and the tag-loss disengage. Each one ends a control run for a good reason
    and each one would end THIS run for doing exactly what it is for."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(
            tmp, stale_s=100.0, div_max_m=0.05, anchor_max_m=0.02,
            jump_max_m=0.02, reject_escalate=2,
            workspace_box_ned=[[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]])
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 2)
        # (1) DIVERGENCE: fly the vehicle far past div_max_m for well over the
        # 0.5 s the guard debounces over.
        for _ in range(30):
            t = now()
            w.on_nav_fix(_fix([-2.0, 3.0, 0.8], 0.0, t))
            w.on_vehicle_imu(_imu(depth=0.8, t=t))
            _feed_status(w)
            w.tick()
        assert w.traj_on, "the divergence guard ended an OBSERVE run"
        assert w.replay is not None and not w.replay.get("halted")
        assert float(w.replay["div_max_seen_m"]) > 0.05, "div not recorded"
        # (2) + (4) ESCALATION LATCH and the BOX: the plans composed out there
        # are far outside the 5 cm box and far from the reference, so they
        # keep failing gates. Neither may latch the mission.
        for _ in range(12):
            w.on_policy_plan(_mk_plan(w))
            _ticks(w, 3)
        assert not w.replay.get("halted"), "the escalation latch fired"
        assert w.traj_on
        # the box was MEASURED, not enforced — the sentence is in the record
        recs = _plans_jsonl(w)
        box_lines = [r for r in recs
                     for x in r.get("reasons", []) if "workspace box" in x]
        assert box_lines, "no plan ever left the 5 cm box (test is vacuous)"
        assert any("not enforced" in x for r in box_lines
                   for x in r["reasons"] if "workspace box" in x)
        # (5) TAG LOSS: no fix at all for far longer than tag_stale_hold_s.
        for _ in range(40):
            _feed_status(w)
            w.tick()
        assert w.engaged, "losing the tag disengaged an OBSERVE run"
        assert w.traj_on


def test_observe_forces_the_jaw_off_even_when_the_config_says_on():
    """`policy.gripper: true` must not reach the jaw. The mode, not the YAML,
    decides — and the record says the mode is why."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, gripper=True,
                                                      stale_s=100.0)
        _arm(w)
        assert w.replay["gripper_on"] is False
        assert w.ctrl.scenario["gripper"] is False
        # a chunk whose width channel is hard CLOSED
        for _ in range(6):
            w.on_policy_plan(_mk_plan(w, action=_straight_action(width=0.0)))
            _ticks(w, 4)
        assert grips == [], "the jaw was driven in an OBSERVE run"
        assert float(w.replay["grip_drive"]) == 0.0


def test_observe_uses_its_own_run_clock_and_end_wording():
    """`policy.observe_max_run_s`, not `max_run_s` — and when it does end,
    the end_reason is prefixed so it cannot pass for a clean control run."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(
            tmp, max_run_s=0.5, observe_max_run_s=600.0, stale_s=100.0)
        _arm(w)
        assert float(w.replay["duration_s"]) == 600.0
        w._t0_traj -= 5.0                     # past max_run_s, not observe's
        _ticks(w, 3)
        assert w.traj_on, "the control-mode clock ended an OBSERVE run"
        w._t0_traj -= 700.0                   # now past observe's own
        _ticks(w, 3)
        assert not w.traj_on
        assert w._replay_last["end_reason"] == "observe_max_run"


def test_observe_sentinels_survive_the_no_fix_tick():
    """REGRESSION (review 2026-09-03): tick() reaches `_write_row` through
    three routes and only ONE of them is the observe branch. A tick with no
    tag fix falls to the bottom carrying the function-top defaults
    `u = zeros(6)` / `info = {}` — those rows came out as exactly 0.000 N with
    solver_status 0, which is the reading the CSV header, the schema-11 note
    and the launch banner all promise cannot appear in an observe file."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, stale_s=100.0)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 3)
        # ...now starve the localizer: status only, no fix, no IMU.
        for _ in range(20):
            _feed_status(w)
            w.tick()
        assert w.engaged and w.traj_on, "the no-fix path ended the run"
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        cols = rows[0].split(",")
        assert pilots == [] and grips == []
        # EVERY row, not just the last: the starved ticks are in the middle.
        for r in rows[1:]:
            d = dict(zip(cols, r.split(",")))
            if d["engaged"] != "1":
                continue
            assert d["observe"] == "1"
            assert d["solver_status"] == "-1", d["solver_status"]
            for k in ("uX", "uZ", "uN", "w0", "w5", "ax_surge", "nis",
                      "tick_ms", "solve_ms"):
                assert d[k] == "nan", f"{k}={d[k]!r} on an observe row"


def test_observe_tree_is_exactly_the_documented_path_with_the_shipped_config():
    """`_run_tree` names the run's KIND (the leaf suffix) on top of `log_dir`
    (rov_gui/runstore.py, 2026-09-14 — the kinds were sibling trees before).
    That derivation must still produce the exact folders the README,
    command.md and the CSV header name: data/<day>/<leaf>_observe and
    data/<day>/<leaf>_landdry, never a bare water leaf.

    2026-09-11: `observe` is a read-only property (LOW level None), so the
    flag is flipped the panel's way — set_mode — after a disengage (set_mode
    refuses while engaged), and the invariant observe == (cfg.mode == "none")
    is checked after each switch. The shipped log_dir is installed ONLY for
    the `_run_tree()` reads: set_mode("none") holds its events.log line
    PENDING for the next run folder (review 2026-09-11 — no folder per combo
    click), so nothing is written here; the redirect is still restored so a
    later `_run_dir()` lands in the tmp tree, never in the repo."""
    from rov_gui import runstore
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp)
        own = w.cfg.log_dir
        shipped = "data"
        w.cfg.log_dir = shipped
        assert w._run_tree() == runstore.Tree("data", "observe")
        assert runstore.run_dir(w._run_tree(), create=False).name.endswith("_observe")
        assert w.observe == (w.cfg.mode == "none") == True
        w.cfg.log_dir = own
        w.disengage("test")
        w.set_mode("dobmpc")
        assert w.observe == (w.cfg.mode == "none") == False
        w.land_dry_run = True
        w.cfg.log_dir = shipped
        assert w._run_tree() == runstore.Tree("data", "landdry")
        w.land_dry_run = False
        assert w._run_tree() == runstore.Tree("data", "")
        assert runstore.leaf_kind(
            runstore.run_dir(w._run_tree(), create=False).name) == ""
        w.cfg.log_dir = own
        w.set_mode("none")
        assert w.observe == (w.cfg.mode == "none") == True
        w.cfg.log_dir = shipped
        assert w._run_tree() == runstore.Tree("data", "observe")
        w.cfg.log_dir = own
        assert tmp in str(w._run_dir()), "the suite wrote outside its tmpdir"
        assert w._run_dir().name.endswith("_observe")
        w.teardown()


def test_observe_rec_button_cannot_drag_the_record_into_the_water_tree():
    """BLOCKER (review 2026-09-03): `set_sensor_log` pins the run folder to
    the DEPTH RECORDER's stem, which lives in --rec-dir. Pressing REC on the
    video feed before engaging therefore took the whole controller record —
    CSV, meta, plans.jsonl, policy_plan.csv, events.log — out of
    data/*/*_observe/ and into the water tree, silently."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp)
        water = Path(tmp) / "water_tree" / "20260903" / "0903_120000"
        water.mkdir(parents=True)
        w.set_sensor_log(True, str(water / "rec"))
        assert w._csv_path is not None
        # The controller record went to an OBSERVE folder, not the stem's.
        assert w._csv_path.parent.name.endswith("_observe"), w._csv_path
        assert str(water) not in str(w._csv_path)
        assert w._run_dir().name.endswith("_observe")
        assert tmp in str(w._csv_path), "the suite wrote outside its tmpdir"
        w.set_sensor_log(False, "")


def test_observe_end_reason_is_prefixed_on_every_path():
    """Not just `_policy_end`. A STOP or a DISENGAGE ends the mission through
    `_clear_replay`, which writes its own end_reason — and "disengaged" reads
    identically in a control run, so a reader could not tell the two apart
    from that field alone."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, stale_s=100.0)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 3)
        w.set_traj(False)                                # the STOP path
        assert w._replay_last["end_reason"] == "observe_stop"
        _arm(w)
        _ticks(w, 2)
        w.disengage("released")                          # the DISENGAGE path
        assert w._replay_last["end_reason"] == "observe_disengaged"


def test_observe_record_cannot_be_mistaken_for_a_control_run():
    """Five independent layers, any one of which identifies the run: its own
    tree, the CSV column, the muted columns, the meta `source`, and the
    per-plan `follower` field."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, stale_s=100.0)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 6)
        meta = w._run_meta()
        # (1) the KIND — the only layer that cannot be pooled by accident,
        # because runstore JOINS folders of one kind inside 90 s (see
        # _run_tree). The SHIPPED root is pinned by
        # test_observe_tree_is_exactly_the_documented_path_with_the_shipped_config;
        # here log_dir is a tmpdir, so assert the property, not the string.
        assert w._run_tree().kind == "observe", w._run_tree()
        assert w._run_dir().name.endswith("_observe"), w._run_dir()
        # (2) + (3) the CSV
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        cols = rows[0].split(",")
        assert "observe" in cols
        last = dict(zip(cols, rows[-1].split(",")))
        assert last["observe"] == "1"
        assert last["engaged"] == "1", "the machinery really was running"
        assert last["solver_status"] == "-1", "a solver status a solver "\
                                              "cannot produce"
        for k in ("uX", "uY", "uZ", "uK", "uM", "uN", "w0", "w5",
                  "ax_surge", "ax_yaw", "solve_ms", "tick_ms",
                  "e_along", "e_cross", "ref_speed_m_s"):
            assert last[k] == "nan", f"{k} = {last[k]!r}, expected nan"
        # ...and the two that must NOT be nan: what the pilot did and what the
        # network asked for. That pair is the run's whole output.
        assert last["px"] != "nan" and last["rx"] != "nan"
        # (4) the meta
        assert meta["policy_observe"] is True
        assert "POLICY OBSERVE" in meta["source"]
        assert "MUTED" in meta["source"]
        # ...and OBSERVE must not overwrite the PLANT claim. This fixture is
        # `source: demo`, so the sentence has to keep saying SYNTHETIC: the
        # observe note describes the LOOP, not the vehicle, and an earlier
        # draft of this branch made every demo observe run announce "real
        # vehicle" (caught in the e2e smoke, 2026-09-03).
        assert "SYNTHETIC" in meta["source"]
        assert "real vehicle" not in meta["source"]
        assert meta["run"]["commanding"] is False
        assert meta["run"]["observe"] is True
        pm = meta["policy"]
        assert pm["observe"] is True
        assert pm["synthetic"] is True, "an observe run must read do-not-cite"
        g = pm["gates_enforced"]
        assert g["divergence"] is False and g["escalation_latch"] is False
        assert g["bridge_stop"] is False and g["workspace_box"] is False
        assert g["tag_loss_disengage"] is False
        assert g["max_run_key"] == "observe_max_run_s"
        assert pm["run"]["observe"] is True
        # (5) plans.jsonl + policy_plan.csv
        recs = _plans_jsonl(w)
        assert recs and all(r["follower"] == "observe" for r in recs)
        pcsv = (w._csv_path.parent / "policy_plan.csv").read_text().splitlines()
        assert pcsv[0].split(",")[2] == "follower"
        assert all(r.split(",")[2] == "observe" for r in pcsv[1:])


def test_observe_off_leaves_a_control_run_byte_identical():
    """The mode must be invisible when it is off: a normal policy run still
    commands, still reports `commanding`, still writes the water tree."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, stale_s=100.0)
        assert w.observe is False and w.commanding is True
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 6)
        assert pilots, "a control run stopped commanding"
        assert w._run_tree().kind != "observe"
        seen = []
        bus.mpc_status.connect(seen.append)
        _ticks(w)
        assert seen[-1].commanding is True and seen[-1].observe is False
        assert seen[-1].err_xy is not None, "tracking error went missing"


# =============================================================================
# ---------------------------------------------------------------------------
# the low-level controller under a policy mission is WHATEVER THE MODE COMBO
# SAYS (2026-09-07 check). The trajectory panel's mode box -> bus.cmd_mpc_mode
# -> MpcWorker.set_mode swaps `self.ctrl`; the policy stream then builds its
# NedPlan from `self.ctrl.path_plan_steps` and hands it to
# `self.ctrl.set_path_plan_ned` — so pid (1 stage), mpc/dobmpc and the *_tuned
# pair (N+1 stages, psi_path rotated) all follow the same plan. Only the
# contouring pair (mpcc/dobmpcc) is refused: it ignores streamed plans.
# Pool evidence: data/20260907/* — 23 policy
# missions flew on `pid`, one (0907_152730/mpc_153134) on `mpc_tuned`, while
# hw_mpc.yaml's file default is dobmpc (CSV `mode` column, every row).
# ---------------------------------------------------------------------------
def test_policy_follows_on_the_selected_pid():
    """Mode combo = pid before START -> the REAL PID is the follower: it gets
    a one-stage plan (its path_plan_steps) and its reference walks with it."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, mode="pid")
        assert w.ctrl is w._pid and w.cfg.mode == "pid"
        assert w.ctrl.solver_kind == "pid" and w.ctrl.path_plan_steps == 1
        assert any("mode = pid" in m for _l, m in logs), logs[-3:]
        eta_at_arm = np.asarray(w._eta, float).copy()
        _arm(w)
        assert w.phase == "policy" and w.replay["kind"] == "policy"
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        rp = w.replay
        assert rp["installed"] == 1 and rp["rejected"] == 0, rp
        plan = w.ctrl._path_plan
        assert plan is not None
        assert np.asarray(plan.p_ned).shape == (3, 1), np.asarray(plan.p_ned).shape
        assert np.asarray(plan.yaw_ned).size == 1
        time.sleep(0.5)
        _ticks(w)
        p_ref, _yaw, _v = w.ctrl.ref_ned_at(0.0)
        assert p_ref[0] - eta_at_arm[0] > 0.015, p_ref        # the PID follows
        assert abs(p_ref[1] - eta_at_arm[1]) < 1e-3
        assert w.reason == "policy running"
        w.teardown()


def test_policy_follows_on_the_selected_mpc_tuned():
    """Mode combo = mpc_tuned before START -> the tracking-NMPC object is the
    follower with its mode flag flipped (the *_tuned pair is the SAME solver
    with the path-frame cost, so no rebuild), and it receives the plan with
    psi_path (what the tuned cost rotates around)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, mode="mpc_tuned")
        assert w.ctrl is w._mpc_ctrl and w.cfg.mode == "mpc_tuned"
        assert getattr(w.ctrl, "mode", None) == "mpc_tuned"
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w)
        assert w.replay["installed"] == 1, w.replay
        plan = w.ctrl._path_plan
        assert plan is not None
        K = int(w.ctrl.path_plan_steps)
        assert np.asarray(plan.p_ned).shape == (3, K)
        assert getattr(plan, "psi_path", None) is not None
        assert np.asarray(plan.psi_path).shape == (K,)
        w.teardown()


def test_policy_mode_change_is_refused_while_engaged():
    """The combo is disabled while engaged and set_mode refuses: the follower
    cannot be swapped under a running (or armed) policy mission. Pick the mode
    BEFORE START."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, mode="pid")
        _arm(w)
        ctrl_before, mode_before = w.ctrl, w.cfg.mode
        n_logs = len(logs)
        w.set_mode("mpc_tuned")
        assert w.ctrl is ctrl_before and w.cfg.mode == mode_before
        assert any("mode change refused while engaged" in m
                   for _l, m in logs[n_logs:]), logs[n_logs:]
        assert w.traj_on and w.replay["kind"] == "policy"
        w.teardown()


class _ScalingStub(StubCtrl):
    """StubCtrl + the HwDobMpc cost-scale hook, recording every call. Like
    the real bridge, reset() (called on every re-engage) forgets the scale."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._plan_q_scale = 1.0
        self.scale_calls: list = []

    def reset(self):
        super().reset()
        self._plan_q_scale = 1.0

    def set_plan_cost_scale(self, s):
        self._plan_q_scale = float(s)
        self.scale_calls.append(float(s))

    @property
    def plan_cost_scale(self):
        return self._plan_q_scale


def test_policy_q_scale_is_applied_at_arm_and_restored_at_stop():
    """policy.q_scale reaches an MPC-family follower at policy START, is
    recorded in the trajectory block, and is put back to 1.0 by STOP TRAJ —
    a scaled position weight must never outlive the mission."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_ScalingStub, q_scale=16.0)
        assert w.ctrl.plan_cost_scale == 1.0 and w.ctrl.scale_calls == []
        _arm(w)
        assert w.ctrl.plan_cost_scale == 16.0, w.ctrl.scale_calls
        assert w.ctrl.scenario["q_scale"] == 16.0
        assert any("plan position-weight scale 1 -> 16" in m for _l, m in logs), logs[-4:]
        w.set_traj(False)
        assert w.ctrl.plan_cost_scale == 1.0, w.ctrl.scale_calls
        assert w.ctrl.scale_calls[-1] == 1.0
        w.teardown()


def test_policy_q_scale_is_cleared_before_a_mode_switch():
    """The outgoing follower forgets the scale when the mode combo swaps it
    (set_mode is refused while engaged, so disengage first)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_ScalingStub, q_scale=8.0)
        _arm(w)
        assert w.ctrl.plan_cost_scale == 8.0
        w.set_traj(False)
        assert w.ctrl.plan_cost_scale == 1.0
        w.ctrl.set_plan_cost_scale(8.0)          # pretend something left it behind
        w.set_engaged(False)
        old = w.ctrl
        w.set_mode("pid")
        assert w.ctrl is w._pid and old.plan_cost_scale == 1.0, old.scale_calls
        w.teardown()


def test_policy_q_scale_never_survives_a_disengage_into_the_next_mission():
    """Safety review 2026-09-07 (CRITICAL): a policy mission that ends through
    E-STOP / ENABLE off / disengage / teardown does not pass set_traj(False),
    so the 16x weight used to stay on the follower and the next square under
    the same mode flew with it. Every one of those paths must clear it, and a
    fresh mission of any kind must start at 1.0."""
    gestures = [
        ("disengage", lambda w: w.disengage("test")),
        ("estop", lambda w: w.estop()),
        ("enable_off", lambda w: w.on_enable(False)),
        ("teardown", lambda w: w.teardown()),
    ]
    for name, gesture in gestures:
        with tempfile.TemporaryDirectory() as tmp:
            w, bus, pilots, logs = _policy_worker(tmp, factory=_ScalingStub, q_scale=16.0)
            _arm(w)
            assert w.ctrl.plan_cost_scale == 16.0, name
            gesture(w)
            assert w.ctrl.plan_cost_scale == 1.0, (name, w.ctrl.scale_calls)
            if name != "teardown":
                w.teardown()
    # the real re-engage path: policy 16x -> E-STOP -> enable -> engage ->
    # START square -> the geometric mission runs at the baseline weight
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_ScalingStub, q_scale=16.0)
        _arm(w)
        w.ctrl._plan_q_scale = 16.0          # pretend a path skipped the worker reset
        w.estop()
        w.on_enable(True)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        for _ in range(4):
            _feed(w); w.tick()
        w.set_scenario({"shape": "square"})
        w.set_traj(True)
        assert w.traj_on, w.reason
        assert w.ctrl.plan_cost_scale == 1.0, w.ctrl.scale_calls
        assert w.ctrl.scale_calls[-1] == 1.0
        w.teardown()


def test_policy_halt_reason_names_the_scale_still_in_force():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_ScalingStub, q_scale=8.0)
        _arm(w)
        w._policy_halt("test", "POLICY halted (test)", "ctrl: halted (test)")
        assert "x8 still in force" in w.reason, w.reason
        w.set_traj(False)
        assert w.ctrl.plan_cost_scale == 1.0
        w.teardown()


def test_policy_q_scale_is_ignored_by_a_follower_without_the_hook():
    """PID (and any follower without set_plan_cost_scale) is left alone: no
    AttributeError, no log line, the mission arms normally."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, mode="pid", q_scale=16.0)
        assert not hasattr(w.ctrl, "set_plan_cost_scale")
        _arm(w)
        assert w.traj_on and w.ctrl.scenario["q_scale"] == 16.0
        assert not any("position-weight scale" in m for _l, m in logs)
        w.teardown()


def test_policy_q_scale_is_validated():
    for bad in (0.0, -2.0, 101.0, float("nan"), "big", None):
        blk = default_policy_block(); blk["q_scale"] = bad
        try:
            validate_policy_block(blk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"q_scale {bad!r} accepted")
    blk = default_policy_block(); blk["q_scale"] = "16"
    assert validate_policy_block(blk)["q_scale"] == 16.0
    assert default_policy_block()["q_scale"] == 1.0


def test_policy_refuses_a_contouring_follower():
    """mpcc/dobmpcc contour their own path and ignore streamed plans
    (set_path_plan_ned is a no-op there), so a policy mission must not arm on
    them. Simulated by giving the follower the contouring-only attribute the
    refusal keys on (`progress_m`), since the stub fixture builds no MPCC."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        w.ctrl.progress_m = 0.0                     # what HwMpcc exposes
        try:
            why = w._policy_refusal()
            assert "contours its own path" in why, why
            w.set_traj(True)
            assert not w.traj_on
            assert ("contours its own path" in (w.reason or "")
                    or any("contours its own path" in m for _l, m in logs)), \
                (w.reason, logs[-3:])
        finally:
            del w.ctrl.progress_m
        w.teardown()


# runner (same shape as test_replay.py — works with or without pytest)
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
            import traceback
            traceback.print_exc()
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0





# ------------------------------------------------ hold_tail: extrapolate (2026-09-08)
def test_policy_hold_tail_extrapolate_continues_the_reference_past_the_plan():
    """hold_tail: extrapolate — horizon stages past the newest plan's last
    knot carry that plan's last-segment velocity and a position that keeps
    walking along it, capped at hold_tail_extrap_max_m and clipped to the
    workspace box; w_stage stays None (no mask); gated live-only exactly like
    the mask, so an expired stream holds its endpoint. Why: under `track`
    85-90 % of the 3 s horizon is "reach the endpoint and stop" and the NMPC's
    optimum is a ~3 N push inside the ESC deadband (0908_165517)."""

    class Stub61(StubCtrl):
        @property
        def path_plan_steps(self):
            return 61

        @property
        def path_plan_dt(self):
            return 0.05

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=Stub61, stale_s=10.0,
                                              hold_tail="extrapolate")
        _arm(w)
        w.on_policy_plan(_mk_plan(w))            # straight, 0.05 m/s forward
        _ticks(w)
        assert w.replay["installed"] == 1
        plan = w.ctrl._path_plan
        assert plan.w_stage is None                  # no mask under extrapolate
        st = w._plan_stitcher
        t_end = float(st.end_time())
        v_end = np.asarray(st.end_velocity(), float)
        assert np.linalg.norm(v_end) > 0.03, v_end

        def tail_mask():
            ts = (now() - w._t0_traj) + np.arange(61) * 0.05
            return ts, ts > t_end + 1e-9

        ts, tail = tail_mask()
        assert tail.sum() > 20, tail.sum()          # ~2 s of the 3 s horizon
        v = np.asarray(plan.v_ned, float)
        p = np.asarray(plan.p_ned, float)
        # every tail stage carries the end velocity (cap 0.3 m >> 2 s x 0.05)
        assert np.allclose(v[:, tail], v_end[:, None], atol=1e-9)
        # ...and the position keeps walking along it
        idx = np.flatnonzero(tail)
        d = p[:, idx[-1]] - p[:, idx[0]]
        assert abs(np.linalg.norm(d)
                   - np.linalg.norm(v_end) * (ts[idx[-1]] - ts[idx[0]])) < 1e-6
        assert w.replay["extrap_ticks"] >= 1
        # THE CAP: 2 cm -> the far tail holds again 2 cm past the endpoint
        w.cfg.policy["hold_tail_extrap_max_m"] = 0.02
        _ticks(w)
        plan = w.ctrl._path_plan
        v = np.asarray(plan.v_ned, float)
        p = np.asarray(plan.p_ned, float)
        ts, tail = tail_mask()
        idx = np.flatnonzero(tail)
        assert np.linalg.norm(v[:, idx[-1]]) < 1e-9
        assert np.linalg.norm(p[:, idx[-1]] - p[:, idx[0]]) <= 0.02 + 1e-6
        # THE BOX: clip the walk on the axis the plan moves along
        w.cfg.policy["hold_tail_extrap_max_m"] = 0.3
        ax = int(np.argmax(np.abs(v_end)))
        lim = float(p[ax, idx[0]] + np.sign(v_end[ax]) * 0.01)
        box = [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]]
        box[1 if v_end[ax] > 0 else 0][ax] = lim
        w.cfg.policy["workspace_box_ned"] = box
        _ticks(w)
        p = np.asarray(w.ctrl._path_plan.p_ned, float)
        ts, tail = tail_mask()
        if v_end[ax] > 0:
            assert np.all(p[ax, tail] <= lim + 1e-9) and abs(p[ax, tail].max() - lim) < 1e-9
        else:
            assert np.all(p[ax, tail] >= lim - 1e-9) and abs(p[ax, tail].min() - lim) < 1e-9
        w.cfg.policy["workspace_box_ned"] = [[-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]]
        # LIVE-ONLY: once the plan has expired the tail holds the endpoint
        w._t0_traj -= (t_end + 0.5)
        _ticks(w)
        v = np.asarray(w.ctrl._path_plan.v_ned, float)
        assert np.linalg.norm(v) < 1e-9
        assert w.ctrl._path_plan.w_stage is None
        w.teardown()


class _AlongStub(StubCtrl):
    """StubCtrl + the HwDobMpc along-scale hook, recording every call. Like
    the real bridge, reset() (called on every re-engage) forgets it."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._plan_along_scale = None
        self.along_calls: list = []

    def reset(self):
        super().reset()
        self._plan_along_scale = None

    def set_plan_along_scale(self, s):
        self._plan_along_scale = None if s is None else float(s)
        self.along_calls.append(self._plan_along_scale)

    @property
    def plan_along_scale(self):
        return self._plan_along_scale


def test_policy_along_scale_is_applied_at_arm_and_restored_at_stop():
    """policy.along_scale reaches a path-frame follower at policy START and
    is put back (None = the controller's own split) by STOP TRAJ and by
    disengage — a policy-only along-track weight must never outlive the
    mission (it would turn mpc_tuned's geometric path missions into
    isotropic ones)."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_AlongStub, along_scale=1.0)
        assert w.ctrl.plan_along_scale is None and w.ctrl.along_calls == []
        _arm(w)
        assert w.ctrl.plan_along_scale == 1.0, w.ctrl.along_calls
        assert any("along-track weight scale own -> 1" in m for _l, m in logs), logs[-4:]
        w.set_traj(False)
        assert w.ctrl.plan_along_scale is None, w.ctrl.along_calls
        assert w.ctrl.along_calls[-1] is None
        _arm(w)
        assert w.ctrl.plan_along_scale == 1.0
        w.set_engaged(False)                       # disengage path, not STOP
        assert w.ctrl.plan_along_scale is None
        w.teardown()


def test_policy_along_scale_none_and_a_follower_without_the_hook_are_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, factory=_AlongStub, along_scale=None)
        _arm(w)
        # every call the lifecycle made (stop-before-arm, arm) said "own split"
        assert w.ctrl.plan_along_scale is None
        assert w.ctrl.along_calls and all(c is None for c in w.ctrl.along_calls), w.ctrl.along_calls
        assert not any("along-track weight scale" in m for _l, m in logs)
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, mode="pid", along_scale=1.0)
        assert not hasattr(w.ctrl, "set_plan_along_scale")
        _arm(w)
        assert w.traj_on
        assert not any("along-track weight scale" in m for _l, m in logs)
        w.teardown()




# =============================================================================
# LOW level None = TELEOP (2026-09-11, operator request: HIGH/LOW split).
# `observe` is the runtime LOW level "none"; --policy-observe is its alias.
# =============================================================================
def _none_worker(tmp, factory=None, shape="policy"):
    """A worker built the CONTROL way (no --policy-observe anywhere) whose
    LOW level is switched to None the panel's way (set_mode) and then
    engaged with the vehicle DISARMED and COMMAND ENABLE off — the observe
    gates are what let it engage, and a fixture that armed anyway would not
    notice if that relaxation regressed. log_dir is a SUBDIRECTORY of tmp so
    every kind of run folder stays inside the tmpdir."""
    from rov_gui.bus import DataBus
    from rov_gui.control.workers import MpcWorker
    from rov_gui.tests.test_control import _app, _test_opts

    _app()
    bus = DataBus()
    w = MpcWorker(bus, _test_opts(tmp)(),
                  controller_factory=(factory or StubCtrl))
    w.setup()
    assert not w.observe and w.cfg.mode == "dobmpc"
    w.cfg.log_dir = str(Path(tmp) / "runs")
    w.cfg.engage["warmup_s"] = 0.1
    w.cfg.engage["settle_s"] = 0.0
    pilots, grips, logs, events, statuses = [], [], [], [], []
    bus.cmd_pilot.connect(pilots.append)
    bus.cmd_gripper_drive.connect(grips.append)
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    bus.mpc_event.connect(events.append)
    bus.mpc_status.connect(statuses.append)
    validate_policy_block(w.cfg.policy)
    w.policy_present = True
    w.set_scenario({"shape": shape})
    w.set_mode("none")
    assert w.observe == (w.cfg.mode == "none") == True
    return w, bus, dict(pilots=pilots, grips=grips, logs=logs, events=events,
                        statuses=statuses)


def test_low_none_is_observe_holds_on_the_pid_and_engages_without_a_vehicle():
    """set_mode("none") -> observe True, commanding False, cfg.mode "none",
    the PID as the reference HOLDER, MpcStatus.mode/solver "none", the WARN
    disarm banner + the mission event; engage succeeds with the vehicle
    DISARMED and COMMAND ENABLE off (the observe gates); ticks emit no
    cmd_pilot; the run lands in the policy_observe tree."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp)
        assert w.ctrl is w._pid and not w.commanding
        assert w._mpc_ctrl.mode == "dobmpc", "set_mode('none') touched _mpc_ctrl.mode"
        banner = [m for lvl, m in s["logs"] if lvl == "warn" and "LOW level = None" in m]
        assert len(banner) == 1, banner
        for word in ("TELEOP", "divergence guard", "escalation latch",
                     "bridge-too-long stop", "workspace box", "tag-loss disengage",
                     "Jaw forced off", "observe_max_run_s", "_observe/"):
            assert word in banner[0], word
        assert "LOW None (teleop) selected" in s["events"], s["events"]
        assert w._run_tree().kind == "observe", w._run_tree()
        _feed(w)
        w.set_engaged(True)                     # DISARMED, ENABLE off
        assert w.engaged and not w.commanding, w.reason
        for _ in range(8):
            _feed(w)
            w.tick()
        st = s["statuses"][-1]
        assert st.mode == "none" and st.observe and st.engaged and not st.commanding
        assert st.solver == "none", st.solver
        assert st.solver_status == -1 and st.solve_ms is None
        assert s["pilots"] == [] and s["grips"] == []
        assert w._csv_path.parent.name.endswith("_observe"), w._csv_path
        w.teardown()


def test_low_none_refuses_every_shape_but_policy_and_replay():
    """START under LOW None: square / station / follow / line / circle are
    refused with the LOW None wording (they need a follower); shape policy
    arms (the observe fixture path) with observe recorded on the mission."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _ticks(w, 4)
        for shape in ("square", "station", "follow", "line", "circle"):
            w.set_scenario({"shape": shape})
            w.set_traj(True)
            assert not w.traj_on, shape
            assert "LOW None" in w.reason and shape in w.reason, w.reason
            assert w.replay is None and w.station is None and w.follow is None
        w.set_scenario({"shape": "policy"})
        _arm(w)
        assert w.replay["observe"] is True and w.ctrl.scenario["observe"] is True
        assert w.ctrl is w._pid
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 3)
        assert w.replay["installed"] == 1, w.replay
        assert s["pilots"] == []
        w.teardown()


def test_low_none_mode_change_refused_while_engaged_then_a_controller_commands_again():
    """set_mode refuses while engaged (reason set, mode untouched); after
    the disengage set_mode("dobmpc") -> observe False, and a real engage
    (ENABLE on, armed) is commanding and emits cmd_pilot."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        w.set_mode("dobmpc")
        assert w.cfg.mode == "none" and w.observe and w.ctrl is w._pid
        assert "refused while engaged" in w.reason, w.reason
        w.disengage("test")
        w.set_mode("dobmpc")
        assert w.observe == (w.cfg.mode == "none") == False
        assert w.ctrl is w._mpc_ctrl and w.cfg.mode == "dobmpc"
        assert any("commands again" in m for _l, m in s["logs"])
        assert w._run_tree().kind != "observe"
        w.on_enable(True)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged and w.commanding, w.reason
        _ticks(w, 6)
        app = _app_of(bus)
        deadline = time.monotonic() + 2.0
        while not s["pilots"] and time.monotonic() < deadline:
            app.processEvents()
        assert s["pilots"] and s["pilots"][-1].source == "mpc"
        w.teardown()


def _app_of(_bus):
    from rov_gui.tests.test_control import _app
    return _app()


def test_policy_observe_opts_alias_still_yields_mode_none():
    """Backward compat: a fixture (or --policy-observe) that sets
    `Opts.policy_observe = True` builds a worker whose cfg.mode is "none",
    and the invariant observe == (cfg.mode == "none") holds after setup —
    even though the config file said dobmpc."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp)
        assert w.cfg.mode == "none"
        assert w.observe == (w.cfg.mode == "none") == True
        assert w.ctrl is w._pid
        w.teardown()
    # ...and `--mpc-mode none` alone (no alias) is the same worker
    with tempfile.TemporaryDirectory() as tmp:
        from rov_gui.bus import DataBus
        from rov_gui.control.workers import MpcWorker
        from rov_gui.tests.test_control import _app, _test_opts
        _app()
        base = _test_opts(tmp)

        class Opts(base):
            mpc_mode = "none"

        w = MpcWorker(DataBus(), Opts(), controller_factory=StubCtrl)
        w.setup()
        assert w.observe == (w.cfg.mode == "none") == True
        assert w.ctrl is w._pid
        w.teardown()


def test_observe_is_a_read_only_property():
    """No setter (2026-09-11): setup() and set_mode() are the only writers
    of the one bit every actuation guard and the run tree hang on."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        try:
            w.observe = True
        except AttributeError:
            pass
        else:
            raise AssertionError("MpcWorker.observe accepted an assignment")
        assert not w.observe
        w.teardown()


def test_max_run_s_is_500_everywhere_it_is_defined():
    """[결정: operator 2026-09-11 — 120 -> 500 s]: default_policy_block, the
    policy worker's fallback block and BOTH shipped YAMLs agree, and the
    provenance tag is a 결정, not a [예측]. observe_max_run_s is unchanged."""
    import yaml
    from rov_gui.backends.policy import _FALLBACK_POLICY_BLOCK
    from rov_gui.control.geometry import POLICY_PROVENANCE
    root = Path(__file__).resolve().parents[2]
    blk = default_policy_block()
    assert blk["max_run_s"] == 500.0
    assert _FALLBACK_POLICY_BLOCK["max_run_s"] == 500.0
    for name in ("hw_mpc.yaml", "land_dp.yaml"):
        raw = yaml.safe_load((root / "config" / name).read_text())
        assert raw["policy"]["max_run_s"] == 500.0, (name, raw["policy"]["max_run_s"])
    assert POLICY_PROVENANCE["max_run_s"].startswith("[결정: operator 2026-09-11")
    assert "500" in POLICY_PROVENANCE["max_run_s"]
    assert blk["observe_max_run_s"] == 1800.0
    # the shipped file's mode vocabulary names none
    hw = (root / "config" / "hw_mpc.yaml").read_text()
    assert "none | mpc | dobmpc" in hw


def test_arm_policy_pins_the_status_checkpoint_and_rejects_another():
    """Checkpoint truth = the PolicyStatus (the panel picker), not the
    config: scen carries the status's ckpt + ckpt_sha1; a plan whose sha
    differs is rejected without a strike (reject_ckpt, one plans.jsonl
    line); one with the armed sha is installed; the policy meta says which
    file inferred vs which the config seeded."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _feed_good_state(w)
        _feed_status(w, ckpt="/gui/picked.ckpt", ckpt_sha1="abc")
        _arm(w)
        scen = w.ctrl.scenario
        assert scen["ckpt"] == "/gui/picked.ckpt" and scen["ckpt_sha1"] == "abc"
        assert w.replay["ckpt_sha1"] == "abc"
        assert any("picked.ckpt" in m and "sha1 abc" in m for _l, m in logs), \
            "the arm line does not name the loaded checkpoint"
        # a plan from ANOTHER network
        w.on_policy_plan(_mk_plan(w, ckpt_sha1="zzz"))
        _feed_status(w, ckpt="/gui/picked.ckpt", ckpt_sha1="abc")
        w.tick()
        rp = w.replay
        assert rp["reject_ckpt"] == 1 and rp["rejected"] == 1, rp
        assert rp["installed"] == 0 and not rp["halted"]
        assert w._plan_filter._consec_rejects == 0, "a ckpt reject took a strike"
        recs = [r for r in _plans_jsonl(w) if r.get("status") == "reject"]
        assert recs and "ckpt mismatch" in " ".join(recs[-1].get("reasons") or [str(recs[-1])])
        # the armed network's plan flies
        w.on_policy_plan(_mk_plan(w, ckpt_sha1="abc"))
        _feed_status(w, ckpt="/gui/picked.ckpt", ckpt_sha1="abc")
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.replay
        assert w.replay["reject_ckpt"] == 1
        # ckpt_loaded is the LATEST status (what the worker holds now) —
        # `_run_until` fed the fixture's default (ckpt ""), so say it again.
        _feed_status(w, ckpt="/gui/picked.ckpt", ckpt_sha1="abc")
        pm = w._policy_meta()
        assert pm["ckpt_loaded"] == "/gui/picked.ckpt"
        assert pm["ckpt_config"] == w.cfg.policy["ckpt"]
        # The note must stay true across a REC-open swap (review
        # 2026-09-11): what inferred is the ARM pin, ckpt_loaded is the
        # status at write time.
        assert pm["config_note"] == (
            "trajectory.ckpt / ckpt_sha1 (pinned at ARM) is what inferred; "
            "ckpt_loaded is what the worker held when this meta was written; "
            "config.ckpt is the launch seed")
        assert pm["ckpt_changes_while_csv_open"] == []
        assert pm["run"]["reject_ckpt"] == 1 and pm["run"]["ckpt_sha1"] == "abc"
        w.teardown()


def test_arm_policy_falls_back_to_the_config_ckpt_when_the_status_has_none():
    """A status with ckpt "" (the fixture's default) leaves scen["ckpt"] on
    the config's seed and the sha empty — the permissive (stub) case."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        assert w.ctrl.scenario["ckpt"] == str(w.cfg.policy["ckpt"])
        assert w.ctrl.scenario["ckpt_sha1"] == ""
        pm = w._policy_meta()
        assert pm["ckpt_loaded"] == "" and pm["ckpt_source"] is None
        w.teardown()


def test_policy_worker_reloading_under_an_armed_mission_ends_with_its_own_reason():
    """A PolicyStatus with loading=True / ready=False mid-mission (a panel
    checkpoint swap that got past the worker's own refusal) is a FAULT:
    the mission ends with end_reason worker_reloading (A22 wording), not
    worker_silent, and holds HERE."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        events = []
        bus.mpc_event.connect(events.append)
        _arm(w)
        _ticks(w, 2)
        _feed_good_state(w)
        _feed_status(w, ready=False, loading=True, note="loading")
        w.tick()
        assert w.replay is None and w.engaged, w.reason
        assert w._replay_last["end_reason"] == "worker_reloading", w._replay_last
        assert "reloading" in w.reason, w.reason
        assert any("reloading" in e for e in events), events
        w.teardown()


def test_observe_engages_even_when_the_holder_reports_not_realtime():
    """The real-time probe is a COMMAND-PATH gate (moved inside the
    `not observe` branch, 2026-09-11): a holder reporting realtime_ok False
    still engages under LOW None, and the log says the probe was skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp)
        w.ctrl.realtime_ok = False
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        assert any("probe SKIPPED" in m for _l, m in s["logs"]), s["logs"][-5:]
        w.disengage("test")
        # ...and a controller with the same verdict is refused (unchanged)
        w.set_mode("dobmpc")
        w._mpc_ctrl.realtime_ok = False
        w.on_enable(True)
        _feed(w)
        w.set_engaged(True)
        assert not w.engaged and "not real-time" in w.reason, w.reason
        w.teardown()


def test_mode_change_is_refused_while_a_rec_csv_is_open():
    """A REC-opened CSV pins the run folder under the OLD tree; set_mode
    refuses (logged, reason set) until the recording stops, and
    MpcStatus.csv_open tells the panel to disable the LOW combo."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.cfg.log_dir = str(Path(tmp) / "runs")
        statuses = []
        bus.mpc_status.connect(statuses.append)
        stem = Path(tmp) / "runs" / "20260911" / "0911_120000" / "rec"
        stem.parent.mkdir(parents=True)
        w.set_sensor_log(True, str(stem))
        assert w._csv is not None
        w.set_mode("none")
        assert w.cfg.mode == "dobmpc" and not w.observe
        assert "CSV is open" in w.reason, w.reason
        assert any("CSV is open" in m for lvl, m in logs if lvl == "warn")
        _feed_good_state(w)
        w.tick()
        assert statuses and statuses[-1].csv_open is True
        w.set_sensor_log(False, "")
        assert w._csv is None
        w.set_mode("none")
        assert w.observe == (w.cfg.mode == "none") == True
        _feed_good_state(w)
        w.tick()
        assert statuses[-1].csv_open is False and statuses[-1].mode == "none"
        w.teardown()


def test_set_mode_before_ready_is_a_logged_refusal():
    """The silent `not _ready` return is gone (review 2026-09-11): the panel
    re-syncs its combo from MpcStatus.mode, so a dropped request must say
    why or it reads as a combo that snapped back for no reason."""
    from rov_gui.bus import DataBus
    from rov_gui.control.workers import MpcWorker
    from rov_gui.tests.test_control import _app, _test_opts
    with tempfile.TemporaryDirectory() as tmp:
        _app()
        bus = DataBus()
        logs = []
        bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
        w = MpcWorker(bus, _test_opts(tmp)(), controller_factory=StubCtrl)
        w.set_mode("none")                       # no setup() yet
        assert w.cfg is None and not w.observe
        assert "still building" in w.reason, w.reason
        assert any("still building" in m for lvl, m in logs if lvl == "warn")


def test_none_never_joins_the_bridge_modes_and_never_touches_its_mode():
    """"none" lives in MpcWorker.MODES (the panel vocabulary) and NOT in
    HwDobMpc.MODES — that setter raises for non-members, so a worker that
    forwarded "none" to the bridge would crash at the combo. set_mode("none")
    leaves _mpc_ctrl.mode exactly where it was."""
    from types import SimpleNamespace
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.workers import MpcWorker
    assert "none" in MpcWorker.MODES and MpcWorker.MODES[0] == "none"
    assert "none" not in HwDobMpc.MODES
    fake = SimpleNamespace(MODES=HwDobMpc.MODES, solver_kind="acados")
    try:
        HwDobMpc.mode.fset(fake, "none")
    except ValueError:
        pass
    else:
        raise AssertionError("HwDobMpc's mode setter accepted 'none'")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.cfg.log_dir = str(Path(tmp) / "runs")
        w.set_mode("mpc_tuned")
        assert w._mpc_ctrl.mode == "mpc_tuned"
        w.set_mode("none")
        assert w.ctrl is w._pid and w._mpc_ctrl.mode == "mpc_tuned"
        assert w._pid.mode == "pid"
        w.set_mode("pid")
        assert w.ctrl is w._pid and not w.observe and w.cfg.mode == "pid"
        w.teardown()


def test_run_meta_under_none_marks_the_holder_and_the_solver_none():
    """RECORD BOUNDARY (schema 15): a bare HOLD under LOW None writes
    controller {type none, role, holder: the PID's meta}, top-level mode
    "none", policy_observe true; the CSV solver cell reads "none" on every
    engaged row. plot_runs.py filters on controller.type, so such a run drops
    out of every pool by construction."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp, shape="station")
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _ticks(w, 6)
        m = w._run_meta()
        assert m["schema_version"] == 16     # 16: + 6-DoF attitude axes (2026-09-26)
        assert m["mode"] == "none" and m["policy_observe"] is True
        c = m["controller"]
        assert c["type"] == "none" == w.cfg.mode
        assert "never stepped" in c["role"] and "None" in c["role"]
        assert c["holder"]["type"] == "pid" and c["holder"]["solver"] == "pid"
        assert "POLICY OBSERVE" in m["source"]
        w.teardown()
        rows = w._csv_path.read_text().splitlines()
        cols = rows[0].split(",")
        seen = 0
        for r in rows[1:]:
            d = dict(zip(cols, r.split(",")))
            if d["engaged"] != "1":
                continue
            seen += 1
            assert d["solver"] == "none" and d["mode"] == "none", d
            assert d["solver_status"] == "-1" and d["observe"] == "1"
        assert seen >= 1
        meta = json.loads(w._csv_path.with_suffix(".meta.json").read_text())
        assert meta["controller"]["type"] == "none" and meta["mode"] == "none"
        # ...and a controller run keeps the plain shape (no holder key)
        w.set_mode("pid")
        m2 = w._run_meta()
        assert m2["mode"] == "pid" and m2["controller"]["type"] == "pid"
        assert "holder" not in m2["controller"]


def test_replay_under_none_lands_in_the_observe_tree():
    """RULE: LOW None puts EVERY engagement under data/*/*_observe/
    regardless of shape. A replay armed under None writes its record there
    with policy_observe true and trajectory.kind == "replay", emits no
    command and no jaw drive, and the holder is the PID."""
    from rov_gui.tests.test_replay import _fake_session
    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=2.0)
        w, bus, s = _none_worker(tmp, shape="replay")
        w.cfg.replay["session"] = str(sess)
        w.cfg.replay["v_max_m_s"] = 0.20
        w.cfg.replay["gripper"] = True          # forced off by None anyway
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _ticks(w, 4)
        w.set_traj(True)
        assert w.traj_on, w.reason
        assert w.replay is not None and w.replay["kind"] == "replay"
        assert w.replay["gripper_on"] is False, "the jaw drove under None"
        assert w.ctrl is w._pid
        assert _run_until(w, lambda: w.replay is not None and w.replay["installed"] >= 1,
                          timeout_s=3.0), w.reason
        assert w._csv_path.parent.name.endswith("_observe"), w._csv_path
        assert tmp in str(w._csv_path)
        w.teardown()
        meta = json.loads(w._csv_path.with_suffix(".meta.json").read_text())
        assert meta["policy_observe"] is True and meta["mode"] == "none"
        assert meta["trajectory"]["kind"] == "replay"
        assert meta["controller"]["type"] == "none"
        assert s["pilots"] == [] and s["grips"] == []


def test_arm_policy_under_none_never_scales_the_follower_cost():
    """Nothing is stepped under LOW None, so `_arm_policy` must not put
    policy.q_scale on ANY follower: the holder is the PID (no hook) and the
    idle MPC keeps its baseline — a `plan_q_scale_flown` claim for a run
    where nothing flew would be a record lie."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp, factory=_ScalingStub)
        w.cfg.policy["q_scale"] = 4.0
        validate_policy_block(w.cfg.policy)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _ticks(w, 4)
        _arm(w)
        assert w.ctrl is w._pid and not hasattr(w.ctrl, "set_plan_cost_scale")
        assert w._mpc_ctrl.scale_calls in ([], [1.0]), w._mpc_ctrl.scale_calls
        assert w._mpc_ctrl.plan_cost_scale == 1.0
        assert w.ctrl.scenario["q_scale"] == 4.0     # still RECORDED as asked
        w.teardown()


# ------------------------------------------------ LOW switch review fixes (2026-09-11)
def test_low_none_banner_names_the_tree_the_resolver_picks():
    """The DISARM banner used to hard-code the observe folder, but
    `_run_tree` files a --land-dry-run station under the `_landdry` kind
    BEFORE it consults observe — so the one line that exists to say where
    the record went named the wrong kind (review 2026-09-11). The sentence
    is now built from the resolver."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.cfg.log_dir = str(Path(tmp) / "runs")
        w.set_mode("none")
        b = [m for lvl, m in logs if lvl == "warn" and "LOW level = None" in m]
        assert b and "data/*/*_observe/ (observe=1)" in b[-1], b
        assert "landdry" not in b[-1]
        w.set_mode("dobmpc")
        w.land_dry_run = True
        w.set_mode("none")
        b = [m for lvl, m in logs if lvl == "warn" and "LOW level = None" in m]
        assert len(b) == 2 and "data/*/*_landdry/ (observe=1)" in b[-1], b
        assert "observe/" not in b[-1]
        assert w._run_tree().kind == "landdry"
        w.land_dry_run = False
        w.teardown()


def test_low_switch_drops_the_previous_runs_counters():
    """`_policy_meta` derives observe / synthetic / gates_enforced from the
    CURRENT LOW level but `run` from `_replay_last`; before this fix a REC
    NAV press (dump_run_meta is not gated on engagement) after LOW None ->
    DOBMPC wrote a controller.json with `policy.observe: false` beside
    `policy.run.observe: true` / `end_reason: observe_stop` (review
    2026-09-11). set_mode now clears `_replay_last`; the observe run's
    counters were already in its own meta.json when its CSV closed."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, s = _none_worker(tmp)
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _ticks(w, 4)
        _arm(w)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 3)
        w.set_traj(False)
        assert w._replay_last["end_reason"] == "observe_stop"
        csv_path = w._csv_path
        w.disengage("test")
        # the observe run's meta.json has its counters
        m_obs = json.loads(csv_path.with_suffix(".meta.json").read_text())
        assert m_obs["policy"]["observe"] is True
        assert m_obs["policy"]["run"]["end_reason"] == "observe_stop"
        assert m_obs["policy"]["run"]["observe"] is True
        # still there between engagements on the SAME level (REC NAV after
        # a run under the level it was flown on stays honest)
        assert w._replay_last is not None
        w.set_mode("dobmpc")
        assert w._replay_last is None
        pm = w._policy_meta()
        assert pm["observe"] is False and pm["run"] is None, pm["run"]
        assert pm["gates_enforced"]["divergence"] is True
        assert w._plan_stream_meta()["enabled"] is False
        out = Path(tmp) / "nav_rec"
        w.dump_run_meta(str(out))
        m = json.loads((out / "controller.json").read_text())
        assert m["mode"] == "dobmpc" and m["policy_observe"] is False
        assert m["policy"]["observe"] is False and m["policy"]["run"] is None
        assert m["controller"]["type"] == "dobmpc"
        w.teardown()


def test_low_combo_click_opens_no_run_folder_until_a_run_happens():
    """A LOW combo click between engagements used to `_event`, which opened
    (or joined) a one-line data/<date>/<hhmmss>_observe/events.log
    on the spot — the 2026-08-14 seven-folders pattern (review 2026-09-11).
    The screen half still fires at once; the FILE half is held pending and
    heads the NEXT events.log that gets a real line, once."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        runs = Path(tmp) / "runs"
        w.cfg.log_dir = str(runs)
        events = []
        bus.mpc_event.connect(events.append)
        w.set_mode("none")
        w.set_mode("dobmpc")
        w.set_mode("none")
        assert events.count("LOW None (teleop) selected") == 2, events
        assert not runs.exists(), sorted(str(q) for q in runs.rglob("*"))
        assert len(w._pending_events) == 3, w._pending_events
        assert "LOW dobmpc selected" in w._pending_events[1]
        # the next run's events.log opens with them, in order, before ENGAGED
        w.policy_present = True
        w.set_scenario({"shape": "policy"})
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        assert w._pending_events == [], w._pending_events
        ev = (w._csv_path.parent / "events.log").read_text()
        assert w._csv_path.parent.name.endswith("_observe"), w._csv_path
        i0 = ev.index("LOW None (teleop) selected")
        i1 = ev.index("LOW dobmpc selected")
        i2 = ev.index("LOW None (teleop) selected", i1)
        assert i0 < i1 < i2 < ev.index("ENGAGED"), ev
        w.disengage("test")
        # ...and not again in the next folder (one-shot, unlike the banner)
        first = w._csv_path
        w.set_mode("dobmpc")
        w.set_mode("none")
        assert len(w._pending_events) == 2
        _feed(w)
        w.set_engaged(True)
        assert w.engaged, w.reason
        ev2 = (w._csv_path.parent / "events.log").read_text()
        # first batch: 2 None lines; second batch (dobmpc -> none): 1 more
        if w._csv_path.parent == first.parent:      # runstore joined (< 90 s)
            assert ev2.count("LOW None (teleop) selected") == 3, ev2
            assert ev2.count("LOW dobmpc selected") == 2, ev2
        else:
            assert ev2.count("LOW None (teleop) selected") == 1, ev2
            assert ev2.count("LOW dobmpc selected") == 1, ev2
        w.teardown()


def test_low_switch_under_slam_says_the_slam_tree_does_not_follow():
    """slam.py resolves the worker's `_run_tree` ONCE, in _start_child on
    the first stereo pair, so a LOW switch after that does not move the
    ORB-SLAM3 outputs; §10.1 dropped the set_mode caveat on the assumption
    it would (review 2026-09-11). Under --slam every successful switch
    says so at WARN; without --slam nothing is said."""
    from rov_gui.bus import DataBus
    from rov_gui.control.workers import MpcWorker
    from rov_gui.tests.test_control import _app, _test_opts
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.cfg.log_dir = str(Path(tmp) / "runs")
        w.set_mode("none")
        w.set_mode("dobmpc")
        assert not any("SLAM outputs stay" in m for _l, m in logs), logs
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        _app()
        bus = DataBus()
        logs = []
        bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))

        class Opts(_test_opts(tmp)):
            slam = True

        w = MpcWorker(bus, Opts(), controller_factory=StubCtrl)
        w.setup()
        w.cfg.log_dir = str(Path(tmp) / "runs")
        w.set_mode("none")
        cav = [m for lvl, m in logs if lvl == "warn" and "SLAM outputs stay" in m]
        assert len(cav) == 1 and "first stereo pair" in cav[0], cav
        w.set_mode("pid")
        cav = [m for lvl, m in logs if lvl == "warn" and "SLAM outputs stay" in m]
        assert len(cav) == 2, cav
        w.teardown()


def test_ckpt_change_under_an_open_rec_csv_is_announced_and_recorded():
    """REC on the video feed opens a controller CSV that disengage does not
    close; the panel's picker is gated on `engaged` only, so between two
    missions the worker can come to HOLD a different network while the CSV
    is still open. The meta then said `ckpt_loaded = B` under a note that
    called it "what inferred" (review 2026-09-11). Now: the arm pin
    (trajectory.ckpt / ckpt_sha1) is what inferred, the note says so, the
    change is a mission event + WARN, and
    `policy.ckpt_changes_while_csv_open` carries the boundary."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.cfg.log_dir = str(Path(tmp) / "runs")
        events = []
        bus.mpc_event.connect(events.append)
        stem = Path(tmp) / "runs" / "20260911" / "0911_120000" / "rec"
        stem.parent.mkdir(parents=True)
        w.set_sensor_log(True, str(stem))            # REC before engage
        assert w._csv is not None and not w._csv_auto
        w.policy_present = True
        w.set_scenario({"shape": "policy"})
        w.on_enable(True)

        def feed_a():
            _feed_good_state(w)
            _feed_status(w, ckpt="/gui/A.ckpt", ckpt_sha1="aaa")
        feed_a()
        w.set_engaged(True)
        assert w.engaged, w.reason
        for _ in range(4):
            feed_a()
            w.tick()
        _arm(w)
        assert w.ctrl.scenario["ckpt"] == "/gui/A.ckpt"
        # the same checkpoint again is NOT a change
        feed_a()
        assert w._ckpt_changes == []
        w.set_traj(False)
        w.disengage("operator")
        assert w._csv is not None, "REC CSV must survive disengage"
        rows_before = w._rows
        _feed_good_state(w)
        _feed_status(w, ckpt="/gui/B.ckpt", ckpt_sha1="bbb")   # the panel pick
        assert len(w._ckpt_changes) == 1, w._ckpt_changes
        ch = w._ckpt_changes[0]
        assert ch["from"] == "/gui/A.ckpt" and ch["to"] == "/gui/B.ckpt"
        assert ch["to_sha1"] == "bbb" and ch["rows_before"] == rows_before
        assert any("A.ckpt -> B.ckpt" in e and "CSV is open" in e for e in events), events
        assert any(lvl == "warn" and "spans two networks" in m for lvl, m in logs)
        _feed_status(w, ckpt="/gui/B.ckpt", ckpt_sha1="bbb")   # repeated: no new entry
        assert len(w._ckpt_changes) == 1
        w.set_sensor_log(False, "")                  # REC off -> closing meta
        meta = json.loads((stem.parent / "rec_mpc.meta.json").read_text())
        assert meta["trajectory"]["ckpt"] == "/gui/A.ckpt"
        assert meta["trajectory"]["ckpt_sha1"] == "aaa"
        assert meta["policy"]["ckpt_loaded"] == "/gui/B.ckpt"
        assert meta["policy"]["config_note"].startswith(
            "trajectory.ckpt / ckpt_sha1 (pinned at ARM) is what inferred")
        chs = meta["policy"]["ckpt_changes_while_csv_open"]
        assert len(chs) == 1 and chs[0]["from"] == "/gui/A.ckpt" \
            and chs[0]["to"] == "/gui/B.ckpt", chs
        ev = (stem.parent / "events.log").read_text()
        assert "A.ckpt -> B.ckpt" in ev, ev
        # per-CSV: the list is gone with the CSV, and a change with NO CSV
        # open records nothing (there is no record to mislead)
        assert w._ckpt_changes == []
        _feed_status(w, ckpt="/gui/C.ckpt", ckpt_sha1="ccc")
        assert w._ckpt_changes == [] and w._policy_meta()["ckpt_changes_while_csv_open"] == []
        w.teardown()


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TEMPORARY z hold (policy.z_hold_above_floor_m, 2026-09-12) — delete with it
# =============================================================================
def test_policy_z_hold_pins_every_knot_and_says_so(monkeypatch):
    """With ``z_hold_above_floor_m`` set, every composed plan knot's z is the
    same datum-NED value (map z = -height, shifted by the datum's p0[2]) and
    the plan record carries ``z_hold_applied`` — the record boundary that
    keeps such a run from being pooled with one where the policy flew its
    own dz. Without the hold the stub's straight chunk keeps whatever small
    z the composition gives it, which is what the first assertion pins.

    The fix is fed at map z -0.205 (0.205 m above the tag floor, what the
    0908 observe runs flew), so the pinned knot lands 5 mm from the vehicle
    and inside the workspace box — the box is datum-relative and would
    rightly reject a hold a metre away from where the vehicle is."""
    from rov_gui.tests import test_control as TC
    from rov_gui.state import Telemetry, Conn

    def feed_at_pool_height(w):
        t = now()
        w.on_nav_fix(TC._fix([-2.0, 0.0, -0.205], 0.0, t))
        w.on_vehicle_imu(TC._imu(depth=0.8, t=t))
        w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE))
    monkeypatch.setitem(globals(), "_feed_good_state", feed_at_pool_height)
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        assert abs(float(w._datum["p0"][2]) + 0.205) < 1e-9, w._datum
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.reason
        rec = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        z = np.asarray(rec["raw"]["p_ned"], float)[2]
        # the stub's straight chunk carries a small z component (0.28 mm over
        # the 6 knots, measured here) — what matters is that it is NOT pinned
        assert np.ptp(z) > 1e-6, z
        assert "z_hold_applied" not in rec
        w.set_traj(False)
        # --- now with the hold: 0.20 m above the floor -> map z -0.20 -> datum +0.005
        w.cfg.policy["z_hold_above_floor_m"] = 0.20
        validate_policy_block(w.cfg.policy)
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.reason
        rec = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        z = np.asarray(rec["raw"]["p_ned"], float)[2]
        assert np.allclose(z, -0.20 + 0.205, atol=1e-9), z
        za = rec["z_hold_applied"]
        assert za["above_floor_m"] == 0.20 and za["z_map"] == -0.20
        assert abs(za["z_ned_datum"] - 0.005) < 1e-9, za
        assert any("Z HOLD" in m for _l, m in logs)
        w.teardown()


# ---------------------------------------------------------------------------
# YAW REFERENCE FILTER (policy.yaw_ref_filter, 2026-09-14)
# ---------------------------------------------------------------------------
def _turning_action(dyaw_end_deg: float, K: int = 16):
    """A stub chunk asking for a uniform turn over the chunk."""
    a = _straight_action(K=K)
    a[:, 3] = np.linspace(0.0, math.radians(dyaw_end_deg), K)
    return a


def test_policy_yaw_ref_filter_replaces_the_knot_yaw_and_says_so(monkeypatch):
    """With the filter on (the shipped default), a chunk asking for +10 deg
    over its span does NOT put a 10 deg/s ramp into the knots: the first
    plan's knots turn at omega_policy * (1 - exp(-period/tau)) — 2.2 deg/s
    at 0.5 s / 2 s — the record carries `yaw_ref_filter` with both numbers,
    the meta counts it, and the arm log announces it. The second plan
    continues the FLOWN reference yaw (the stitcher's, not the fix's) and
    the turn-rate state has moved on from the first plan's. With the
    filter null the old behaviour is back: knot yaw = anchor + dyaw_k."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        assert w.cfg.policy["yaw_ref_filter"] == {
            "rate_deg_s": 3.0, "tau_s": 2.0, "guard_deg": 15.0}
        _arm(w)
        assert any("YAW REF FILTER" in m for _l, m in logs)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_turning_action(10.0)))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.reason
        rec = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        yf = rec["yaw_ref_filter"]
        alpha = 1.0 - math.exp(-0.5 / 2.0)
        assert abs(yf["dyaw_raw_deg"] - 10.0) < 1e-6, yf
        assert abs(yf["omega_policy_deg_s"] - 10.0) < 1e-6, yf
        assert abs(yf["omega_ref_deg_s"] - alpha * 10.0) < 1e-6, yf
        assert yf["yaw_ref0_deg"] is None                    # first plan: nothing flown yet
        assert not yf["guard_clamped_start"] and yf["guard_clamped_knots"] == 0
        yaw = np.asarray(rec["raw"]["yaw"], float)
        assert np.allclose(np.degrees(np.diff(yaw)) / rec["dt"], alpha * 10.0, atol=1e-6)
        assert w.replay["yaw_filt"]["n"] == 1
        assert abs(w.replay["yaw_filt"]["omega"] - math.radians(alpha * 10.0)) < 1e-6
        # second plan: starts at the stitched reference's yaw at its t0 and
        # advances the state from the first plan's omega
        time.sleep(0.05)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_turning_action(10.0)))
        assert _run_until(w, lambda: w.replay["installed"] == 2, timeout_s=2.0), w.reason
        rec2 = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        yf2 = rec2["yaw_ref_filter"]
        assert yf2["yaw_ref0_deg"] is not None
        assert abs(yf2["yaw0_deg"] - yf2["yaw_ref0_deg"]) < 1e-9     # continues the flown ref
        assert yf2["omega_ref_deg_s"] > yf["omega_ref_deg_s"]          # converging toward 10
        # composed AGAINST the reference yaw (safety review HIGH): the
        # anchor the plan used carries that yaw, and body knot 0 sits
        # exactly on the anchor position — the TCP lever did not push the
        # yaw tracking error into the sway reference
        used = np.asarray(rec2["anchor_pose_used"], float)
        assert abs(math.degrees(used[5]) - yf2["yaw_ref0_deg"]) < 1e-9, (used, yf2)
        assert abs(math.degrees(used[5]) - yf2["yaw_anchor_deg"]) < 1e-9
        p0 = np.asarray(rec2["raw"]["p_ned"], float)[:, 0]
        assert np.allclose(p0, used[:3], atol=1e-9), (p0, used[:3])
        # the body shift is ONLY the filtered-vs-requested ramp on the TCP
        # lever (the TCP stays where the policy asked): bounded by
        # |t_bt| * |dyaw_raw - dyaw_filt|, and nonzero here because the
        # request (10 deg) is far above what the filter lets through
        lever = float(np.linalg.norm(w._policy_T_bt[:3, 3]))
        d_raw_filt = math.radians(abs(yf2["dyaw_raw_deg"] - yf2["dyaw_filt_deg"]))
        assert 0.0 < yf2["body_shift_max_m"] <= lever * d_raw_filt + 1e-9, (yf2, lever)
        meta = w._policy_meta()
        assert meta["run"]["yaw_ref_filter"]["plans"] == 2
        assert meta["run"]["yaw_ref_filter"]["nonfinite"] == 0
        assert meta["config"]["yaw_ref_filter"]["tau_s"] == 2.0
        w.set_traj(False)
        # --- filter OFF: the raw ramp reaches the knots again
        w.cfg.policy["yaw_ref_filter"] = None
        validate_policy_block(w.cfg.policy)
        n_announced = sum("YAW REF FILTER" in m for _l, m in logs)
        _arm(w)
        assert sum("YAW REF FILTER" in m for _l, m in logs) == n_announced
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_turning_action(10.0)))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.reason
        rec3 = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        assert "yaw_ref_filter" not in rec3
        yaw3 = np.asarray(rec3["raw"]["yaw"], float)
        assert abs(math.degrees(yaw3[-1] - yaw3[0]) - 10.0) < 1e-6, yaw3
        assert w._policy_meta()["run"]["yaw_ref_filter"] is None
        w.teardown()


def test_policy_yaw_ref_filter_is_skipped_under_observe():
    """OBSERVE flies nothing and the operator wants the network's raw
    heading: the filter stays out of the knots, the record carries no
    `yaw_ref_filter`, the meta says why, and the arm log says so."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(tmp, stale_s=100.0)
        assert w.cfg.policy["yaw_ref_filter"] is not None
        _arm(w)
        assert any("SKIPPED under POLICY OBSERVE" in m for _l, m in logs)
        assert not any("YAW REF FILTER" in m for _l, m in logs)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_turning_action(10.0)))
        assert _run_until(w, lambda: w.replay["installed"] == 1, timeout_s=2.0), w.reason
        rec = [r for r in _plans_jsonl(w) if "raw" in r][-1]
        assert "yaw_ref_filter" not in rec
        yaw = np.asarray(rec["raw"]["yaw"], float)
        assert abs(math.degrees(yaw[-1] - yaw[0]) - 10.0) < 1e-6, yaw
        assert w.replay["yaw_filt"]["n"] == 0
        assert w._policy_meta()["run"]["yaw_ref_filter"] == {"skipped": "observe"}
        w.teardown()


# =============================================================================
# THE 6-DoF VARIANT (2026-09-26): pinned action_repr, the attitude_track
# truth table, roll/pitch through the seam, div_rp, and the schema-16 records
# =============================================================================
from rov_gui.state import (ACTION_REPR_POS_RPY_WIDTH, POLICY_ACTION_REPRS_FLYABLE,
                           PolicyPlanViz)
from rov_gui.tests.test_control import (_AttStub, _alloc_has_attitude,
                                        _att_worker, _feed_att)


def _straight_action7(speed=0.05, K=16, width=0.069, lateral=0.0,
                      roll_end_deg=0.0, pitch_end_deg=0.0):
    """The stub policy's chunk in the 6-DoF contract (pos_rpy_width, (K, 7)):
    columns 0:4 and 6 exactly the 5-dim chunk; droll / dpitch (cols 4:6)
    ramp linearly from 0 at knot 0 to the given end values."""
    a = np.zeros((K, 7), np.float32)
    a[:, 0] = lateral
    a[:, 2] = speed * np.arange(K) * OBS_DT
    a[:, 3] = 0.0
    a[:, 4] = np.linspace(0.0, math.radians(roll_end_deg), K)
    a[:, 5] = np.linspace(0.0, math.radians(pitch_end_deg), K)
    a[:, 6] = width
    return a


def _feed7(w, **kw):
    """Attitude-axes telemetry + a 7-dim READY status (roll/pitch kwargs
    go to the IMU, the rest to Telemetry)."""
    _feed_att(w, **kw)
    _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)


def _policy_att_worker(tmp, attitude_track=True, **policy_over):
    """A worker with engage.attitude_axes LIVE (every gate satisfied by
    `_feed_att`'s telemetry, probes not required), LOW mpc, the policy
    block pinned to pos_rpy_width, engaged and warmed up with shape=policy."""
    w, bus, pilots, logs, statuses = _att_worker(tmp, mode="mpc")
    w.cfg.policy.update({"action_repr": ACTION_REPR_POS_RPY_WIDTH,
                         "attitude_track": bool(attitude_track)})
    w.cfg.policy.update(policy_over)
    validate_policy_block(w.cfg.policy)
    w.policy_present = True
    w.set_scenario({"shape": "policy"})
    w.on_enable(True)
    _feed7(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    assert w._attitude_axes is True, w.reason
    for _ in range(4):
        _feed7(w)
        w.tick()
    return w, bus, pilots, logs, statuses


def _ticks7(w, n=1, **kw):
    for _ in range(n):
        _feed7(w, **kw)
        w.tick()


def test_policy_pinned_action_repr_gates_arm_and_intake_both_ways():
    """policy.action_repr (default pos_yaw_width) is what this mission
    flies: a (16, 7) plan is rejected under the default pin, and under a
    pos_rpy_width pin a (16, 5) plan is; the arm check compares the
    checkpoint contract against the SAME pin. A 7-dim network with
    attitude_track false flies dropped-and-logged: the plan installs,
    rp_tracked is false, no rp rides the message, the follower's NedPlan
    carries no attitude, and dropped_rp_deg says what was levelled away."""
    assert set(POLICY_ACTION_REPRS_FLYABLE) == {"pos_yaw_width", "pos_rpy_width"}
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks(w)
        rp = w.replay
        assert rp["reject_action_repr"] == 1 and rp["installed"] == 0, rp
        line = _plans_jsonl(w)[-1]
        assert line["status"] == "reject" and line["rp_tracked"] is False
        assert any("action_repr mismatch" in r and "pos_rpy_width" in r
                   and "pos_yaw_width" in r for r in line["reasons"]), line
        assert w._run_meta()["policy"]["run"]["action_repr_pinned"] == "pos_yaw_width"
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(
            tmp, action_repr=ACTION_REPR_POS_RPY_WIDTH)
        # the ARM side: the status still reports the 5-dim contract
        w.set_traj(True)
        assert not w.traj_on and "action_repr mismatch" in w.reason, w.reason
        assert "pos_rpy_width" in w.reason and "pos_yaw_width" in w.reason
        _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)
        _arm(w)
        assert w.replay["action_repr_pinned"] == "pos_rpy_width"
        assert w.replay["attitude_track"] is False
        assert w.ctrl.scenario["action_repr"] == "pos_rpy_width"
        assert w.ctrl.scenario["attitude_track"] is False
        assert any("DROPPED-AND-LOGGED" in m for _l, m in logs), logs[-3:]
        _ticks(w)
        viz = []
        bus.policy_plan_viz.connect(viz.append)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=6.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks(w)
        rp = w.replay
        assert rp["installed"] == 1 and rp["rejected"] == 0, rp
        assert rp["action_repr"] == "pos_rpy_width"
        line = [ln for ln in _plans_jsonl(w) if ln["status"] in ("accept", "clip")][-1]
        assert line["rp_tracked"] is False and "rp" not in line["raw"]
        assert np.asarray(line["action_raw"]).shape == (16, 7)
        assert line["dropped_rp_deg"] > 1.0, line["dropped_rp_deg"]   # meaningful again
        assert w.ctrl._path_plan is not None and w.ctrl._path_plan.rp_ned is None
        assert viz and viz[-1].rp is None
        # ...and the other way round: a 5-dim plan is not this mission's
        w.on_policy_plan(_mk_plan(w, action=_straight_action()))
        _ticks(w)
        assert w.replay["reject_action_repr"] == 1 and w.replay["installed"] == 1
        m = w._run_meta()
        assert m["policy"]["action_repr"] == "pos_rpy_width"
        assert m["policy"]["action_repr_default"] == "pos_yaw_width"
        assert m["trajectory"]["action_repr"] == "pos_rpy_width"
        assert m["trajectory"]["attitude_track"] is False
        assert m["policy"]["run"]["attitude_track"] is False
        json.dumps(m)
        w.teardown()
        pcsv = (w._csv_path.parent / "policy_plan.csv").read_text().splitlines()
        assert pcsv[0].endswith(",reason,roll_deg,pitch_deg")
        assert all(r.endswith(",nan,nan") for r in pcsv[1:]), pcsv[1:3]


def test_policy_attitude_track_truth_table_refuses_each_cause():
    """attitude_track needs: the pos_rpy_width pin, the axes LIVE on this
    engagement (so never under observe), and a trusted attitude measurement
    (rp_residual <= 10 deg). Each failure refuses ARM with its own
    sentence; the full table arms with the rp filter gates set."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp, attitude_track=True)
        w.set_traj(True)
        assert not w.traj_on and "attitude_track needs action_repr" in w.reason, w.reason
        assert "nothing to track" in w.reason
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(
            tmp, action_repr=ACTION_REPR_POS_RPY_WIDTH, attitude_track=True)
        _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)
        w.set_traj(True)
        assert not w.traj_on and "needs engage.attitude_axes" in w.reason, w.reason
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, grips, logs = _observe_worker(
            tmp, action_repr=ACTION_REPR_POS_RPY_WIDTH, attitude_track=True)
        _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)
        w.set_traj(True)
        assert not w.traj_on and "observe" in w.reason, w.reason
        w.teardown()
    if not _alloc_has_attitude():
        import pytest
        pytest.skip("allocation without the attitude path (output group)")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _policy_att_worker(tmp)
        # the IMU says 15 deg of roll while the tag says level: residual
        _ticks7(w, 2, roll=math.radians(15.0))
        w.set_traj(True)
        assert not w.traj_on and "residual" in w.reason, w.reason
        assert "15.0 deg" in w.reason
        _ticks7(w, 2)
        _arm(w)
        rp = w.replay
        assert rp["attitude_track"] is True and rp["action_repr_pinned"] == "pos_rpy_width"
        lim = w._plan_filter.limits
        assert abs(lim.rp_reject_rad - math.radians(30.0)) < 1e-12
        assert lim.rp_rate_max == 0.35
        assert abs(lim.rp_jump_max_rad - math.radians(5.0)) < 1e-12
        assert w.ctrl.scenario["attitude_track"] is True
        assert any("roll/pitch TRACKED" in m for _l, m in logs), logs[-3:]
        w.teardown()


def test_policy_attitude_track_flies_rp_through_the_seam():
    """The tracked path end to end: a (16, 7) plan with a 5 deg pitch ramp
    composes with rp, passes the filter, installs, and the stitcher's
    sample_att reaches the follower as NedPlan.rp_ned; the records carry
    rp_tracked true, raw.rp, rp_raw, the clip counters, the viz tuple, the
    policy_plan.csv angles and the CSV rp_track bit; a second plan is
    anchored within the attitude leash of the measured attitude; the T1
    clip bounds a 40 deg request at rp_max_deg."""
    if not _alloc_has_attitude():
        import pytest
        pytest.skip("allocation without the attitude path (output group)")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _policy_att_worker(tmp, yaw_ref_filter=None)
        _arm(w)
        _ticks7(w)
        viz = []
        bus.policy_plan_viz.connect(viz.append)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=5.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        rp = w.replay
        assert rp["installed"] == 1 and rp["rejected"] == 0, rp
        line = [ln for ln in _plans_jsonl(w) if ln["status"] in ("accept", "clip")][-1]
        assert line["rp_tracked"] is True
        raw_rp = np.asarray(line["raw"]["rp"], float)
        assert raw_rp.shape == (2, line["n_knots"])
        assert np.asarray(line["rp_raw"]).shape == (2, 16)
        assert line["rp_clipped_n"] == 0 and line["rp_clipped_deg"] == 0.0
        assert 2.0 < math.degrees(raw_rp[1, -1]) <= 5.0 + 1e-6, raw_rp[1]
        assert "rp_mag" in line["margins"] and "rp_rate" in line["margins"]
        assert viz and viz[-1].rp is not None and len(viz[-1].rp[1]) == line["n_knots"]
        plan = w.ctrl._path_plan
        assert plan is not None and plan.rp_ned is not None
        assert plan.rp_ned.shape == (2, w.ctrl.path_plan_steps)
        assert 0.0 <= float(plan.rp_ned[1, 0]) <= math.radians(5.0) + 1e-9
        att = w._replay_ref_att_now(now() - w._t0_traj)
        assert att is not None and att.shape == (2,)
        w._csv.flush()
        rows = w._csv_path.read_text().splitlines()
        col = dict(zip(rows[0].split(","), rows[-1].split(",")))
        assert col["rp_track"] == "1" and col["ax_pitch"] != "nan"
        # the second plan: anchored within anchor_leash_rp_deg (3 deg) of
        # the MEASURED attitude (level here) toward the flown reference
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=5.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        assert w.replay["installed"] == 2, w.replay
        line = [ln for ln in _plans_jsonl(w) if ln["status"] in ("accept", "clip")][-1]
        assert "rp_meas" in line["anchor"] and "rp_used" in line["anchor"]
        assert abs(line["anchor"]["rp_meas"][1]) < 1e-9
        used = float(line["anchor"]["rp_used"][1])
        assert 0.0 <= used <= math.radians(3.0) + 1e-9, used
        # the attitude anchor is nested ONLY under `anchor` (record audit
        # 2026-09-26, Q6): rp_ref beside rp_meas / rp_used, and no top-level
        # anchor_rp_* duplicates nor a top-level `rp` (that is raw.rp)
        assert "rp_ref" in line["anchor"], sorted(line["anchor"])
        assert line["anchor"]["rp_ref"] is not None and len(line["anchor"]["rp_ref"]) == 2
        assert 0.0 <= float(line["anchor"]["rp_ref"][1]) <= math.radians(5.0) + 1e-9
        for ln in _plans_jsonl(w):
            assert not any(k.startswith("anchor_rp_") for k in ln), sorted(ln)
            assert "rp" not in ln, sorted(ln)
            assert "rp_raw" in ln or ln["status"] not in ("accept", "clip")
        # T1: a 25 deg request (under the 30 deg T2 reject, which compose
        # judges on the RAW attitude since 2026-09-26) is clipped to
        # rp_max_deg (20), and the clip is on the record. (The plan then
        # fails the POSITION overlap gate — the 0.53 m TCP lever swings the
        # body path — which is not the attitude reject: T2 stays at 0.)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=25.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        assert w.replay["reject_rp_mag"] == 0 and w.replay["reject_compose"] == 0, w.replay
        line = [ln for ln in _plans_jsonl(w) if "rp_raw" in ln][-1]
        assert line["rp_clipped_n"] > 0 and line["rp_clipped_deg"] > 4.0, line
        assert math.degrees(np.abs(np.asarray(line["raw"]["rp"])).max()) <= 20.0 + 1e-6
        assert not any("rp_reject" in r for r in line["reasons"]), line["reasons"]
        m = w._run_meta()
        assert m["trajectory"]["attitude_track"] is True
        assert m["policy"]["run"]["attitude_track"] is True
        assert m["run"]["attitude_axes"]["enabled"] is True
        json.dumps(m)
        w.teardown()
        pcsv = (w._csv_path.parent / "policy_plan.csv").read_text().splitlines()
        assert pcsv[0].endswith(",reason,roll_deg,pitch_deg")
        vals = [float(r.split(",")[-1]) for r in pcsv[1:]]
        assert any(math.isfinite(v) and v > 0.0 for v in vals), vals[:8]


def test_policy_attitude_track_rejects_a_plan_beyond_rp_reject_at_compose():
    """T2 (D3) at COMPOSE (2026-09-26): a tracked 7-dim plan whose RAW
    attitude exceeds policy.rp_reject_deg (30) is rejected with the reason
    (reject_compose + reject_rp_mag both count it, the plan never installs,
    the worker keeps flying) — T1 clipping to rp_max_deg no longer hides the
    excess from the filter."""
    if not _alloc_has_attitude():
        import pytest
        pytest.skip("allocation without the attitude path (output group)")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _policy_att_worker(tmp, yaw_ref_filter=None)
        _arm(w)
        _ticks7(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=40.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        rp = w.replay
        assert rp is not None and w.engaged and w.traj_on, (w.reason, rp)
        assert rp["installed"] == 0 and rp["rejected"] == 1, rp
        assert rp["reject_compose"] == 1 and rp["reject_rp_mag"] == 1, rp
        line = [ln for ln in _plans_jsonl(w) if ln["status"] == "reject"][-1]
        assert any("rp_reject" in r for r in line["reasons"]), line["reasons"]
        assert any("40.0 deg" in r for r in line["reasons"]), line["reasons"]
        assert any("rejected at composition" in m and "rp_reject" in m
                   for _l, m in logs), logs[-3:]
        # a plan inside the reject bound still flies
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(pitch_end_deg=5.0),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        assert w.replay["installed"] == 1 and w.replay["rejected"] == 1, w.replay
        m = w._run_meta()
        assert m["policy"]["run"]["reject_rp_mag"] == 1
        assert m["policy"]["run"]["reject_compose"] == 1
        w.teardown()


def test_policy_div_rp_drops_to_level_hold_with_km_flowing():
    """D9 (i): the tracked attitude reference more than div_max_rp_deg from
    the measured attitude for 0.5 s drops the POLICY to a level DP hold —
    the mission ends with end_reason div_rp, the worker stays engaged, and
    the attitude axes keep flowing (active levelling)."""
    if not _alloc_has_attitude():
        import pytest
        pytest.skip("allocation without the attitude path (output group)")
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _policy_att_worker(tmp, yaw_ref_filter=None)
        _arm(w)
        _ticks7(w)
        w.on_policy_plan(_mk_plan(w, action=_straight_action7(),
                                  action_repr=ACTION_REPR_POS_RPY_WIDTH))
        _ticks7(w, 2)
        assert w.replay["installed"] == 1
        # the hull pitches 25 deg (below the 35 deg ceiling) against a
        # level reference: > 15 deg for 10 ticks
        n_before = len([p for p in pilots if p.source == "mpc"])
        _ticks7(w, 12, pitch=math.radians(25.0))
        assert w.engaged, w.reason
        assert not w.traj_on and "attitude diverged" in w.reason, w.reason
        last = w._replay_last
        assert last["end_reason"] == "div_rp" and last["div_rp_max_seen_deg"] > 15.0
        assert w._attitude_axes is True
        assert len([p for p in pilots if p.source == "mpc"]) > n_before
        assert any("LEVEL DP hold" in m for _l, m in logs), logs[-3:]
        w.teardown()


def test_policy_records_on_the_4dof_path_carry_the_schema16_keys():
    """A plain pos_yaw_width policy run (the variant OFF): every
    plans.jsonl line says rp_tracked false, policy_plan.csv ends
    ",reason,roll_deg,pitch_deg" with nan cells, the viz carries rp None,
    the meta pins pos_yaw_width and attitude_track false, and
    run.attitude_axes says enabled false."""
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _policy_worker(tmp)
        _arm(w)
        _ticks(w)
        viz = []
        bus.policy_plan_viz.connect(viz.append)
        w.on_policy_plan(_mk_plan(w))
        _ticks(w, 2)
        assert w.replay["installed"] == 1
        lines = _plans_jsonl(w)
        assert lines and all(ln["rp_tracked"] is False for ln in lines)
        assert all("rp" not in (ln.get("raw") or {}) for ln in lines)
        assert viz and viz[-1].rp is None
        m = w._run_meta()
        assert m["policy"]["action_repr"] == "pos_yaw_width"
        assert m["policy"]["attitude"]["attitude_track"] is False
        assert m["policy"]["attitude"]["rp_max_deg"] == 20.0
        assert m["policy"]["run"]["action_repr_pinned"] == "pos_yaw_width"
        assert m["policy"]["run"]["reject_rp_mag"] == 0
        assert m["policy"]["run"]["div_rp_max_seen_deg"] == 0.0
        assert m["trajectory"]["action_repr"] == "pos_yaw_width"
        assert m["trajectory"]["attitude_track"] is False
        assert m["run"]["attitude_axes"]["enabled"] is False
        assert m["plan_stream"]["config"]["track_attitude"] is False
        w.teardown()
        pcsv = (w._csv_path.parent / "policy_plan.csv").read_text().splitlines()
        assert pcsv[0] == ("plan_id,status,follower,t_rel,k,t_knot,x_ned,y_ned,"
                           "z_ned,yaw_deg,reason,roll_deg,pitch_deg")
        assert all(r.endswith(",nan,nan") for r in pcsv[1:])


def test_stub_rp_session_pitch_ramp_reaches_the_tracked_plan():
    """The `stub_rp` session (dp_policy, 7-dim with a 5 deg pitch ramp)
    emits (16, 7) in the pos_rpy_width contract; pushed through a tracked
    mission its pitch reaches raw.rp with the ramp's sign and magnitude."""
    if not _alloc_has_attitude():
        import pytest
        pytest.skip("allocation without the attitude path (output group)")
    from rov_gui.perception.dp_policy import StubPolicySession

    s = StubPolicySession("stub_rp", "", action_repr=ACTION_REPR_POS_RPY_WIDTH,
                          pitch_ramp_deg=5.0)
    s.load()
    c = s.contract
    assert c["action_repr"] == "pos_rpy_width"
    obs = {c["image_key"]: np.zeros((c["obs_horizon"],) + tuple(c["image_shape"]),
                                    np.float32)}
    for k, shp in c["lowdim_shapes"].items():
        obs[k] = np.zeros((c["obs_horizon"],) + tuple(shp), np.float32)
    act, _ = s.predict(obs)
    assert act.shape == (16, 7)
    assert abs(act[-1, 5] - math.radians(5.0)) < 1e-6 and act[-1, 4] == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs, st = _policy_att_worker(tmp, yaw_ref_filter=None)
        _arm(w)
        _ticks7(w)
        w.on_policy_plan(_mk_plan(w, action=act, action_repr=c["action_repr"]))
        _ticks7(w, 2)
        assert w.replay["installed"] == 1, w.replay
        line = [ln for ln in _plans_jsonl(w) if ln["status"] in ("accept", "clip")][-1]
        raw_rp = np.asarray(line["raw"]["rp"], float)
        assert line["rp_tracked"] is True
        assert 2.0 < math.degrees(raw_rp[1, -1]) <= 5.0 + 1e-6, raw_rp[1]
        assert np.all(np.diff(raw_rp[1]) >= -1e-9)
        assert abs(raw_rp[0]).max() < 1e-9
        w.teardown()
