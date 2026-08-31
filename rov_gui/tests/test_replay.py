#!/usr/bin/env python3
"""
test_replay.py — the REPLAY mission (shape: replay), offline.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_replay.py

A recorded handheld demo, streamed through control/plan_stream.py into the
same MpcWorker every other mission uses — StubCtrl, no acados, no camera.
What it pins down:

* the replay config block validates like imu_dr (unknown keys raise);
* each refusal names its own cause (no session / a contouring controller);
* arming NEVER moves the vehicle first: the track anchors at the pose the
  vehicle has when START is pressed;
* the mission flies, completes by its own duration, and leaves the full
  record behind (plans.jsonl, CSV plan columns, schema-8 meta);
* the jaw drive replays the demo's gripper-width channel and is NEUTRALED by
  every mission-clearing path (the latched-level rule);
* streamed mode (stream_period_s > 0) feeds multiple plans through the
  filter — the seam a live diffusion policy will use.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.control.geometry import MpcConfig
from rov_gui.tests.test_control import (StubCtrl, _feed_good_state, _worker)


# --------------------------------------------------------------- fixtures
def _fake_session(tmp, duration_s=2.0, speed=0.08, hz=10.0,
                  grip=True) -> Path:
    """A synthetic extract_pose output: a straight +x drive at constant
    speed, gripper open (1.0) then CLOSED (0.0) at 60 % of the take."""
    d = Path(tmp) / "demo0001"
    d.mkdir(exist_ok=True)
    n = int(round(duration_s * hz)) + 1
    t = 1000.0 + np.arange(n) / hz
    poses = np.zeros((n, 8))
    poses[:, 0] = t
    poses[:, 1] = speed * (t - t[0])       # x forward, body frame == map here
    poses[:, 4] = 1.0                      # identity quaternion (w first)
    np.save(d / "poses.npy", poses)
    (d / "poses.json").write_text(json.dumps(
        {"schema": "umi_handheld_poses/1", "pose_of": "body_frd",
         "frame": "map_ned"}))
    with open(d / "frames.csv", "w", encoding="utf-8") as f:
        f.write("idx,t_unix\n")
        for i, ti in enumerate(t):
            f.write(f"{i},{ti:.6f}\n")
    if grip:
        g = np.ones((n, 1), np.float32)
        g[int(n * 0.6):] = 0.0
        np.save(d / "gripper_width.npy", g)
    return d


def _armed_replay_worker(tmp, session: Path, **replay_over):
    """A worker engaged and warmed up with shape=replay selected."""
    w, bus, pilots, logs = _worker(tmp)
    w.cfg.replay["session"] = str(session)
    w.cfg.replay["v_max_m_s"] = 0.20       # above the demo's 0.08 -> alpha 1
    w.cfg.replay.update(replay_over)
    w.set_scenario({"shape": "replay"})
    w.on_enable(True)
    _feed_good_state(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):                     # warmup_s 0.1 @ 20 Hz
        _feed_good_state(w)
        w.tick()
    return w, bus, pilots, logs


def _run_until(w, cond, timeout_s=8.0, sleep_s=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        _feed_good_state(w)
        w.tick()
        if cond():
            return True
        time.sleep(sleep_s)
    return False


# ------------------------------------------------------------------ config
def test_replay_config_rejects_unknown_keys_and_bad_values():
    import math  # noqa: F401  (parity with geometry's validation imports)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mpc.yaml"
        p.write_text("mode: pid\nreplay:\n  vmax: 0.2\n")
        try:
            MpcConfig.load(p)
            raise AssertionError("unknown replay key did not raise")
        except ValueError as e:
            assert "vmax" in str(e)
        p.write_text("mode: pid\nreplay:\n  v_max_m_s: 0\n")
        try:
            MpcConfig.load(p)
            raise AssertionError("v_max_m_s 0 did not raise")
        except ValueError as e:
            assert "v_max_m_s" in str(e)
        p.write_text("mode: pid\nreplay:\n  gripper_close_below: 0.8\n"
                     "  gripper_open_above: 0.4\n")
        try:
            MpcConfig.load(p)
            raise AssertionError("inverted gripper thresholds did not raise")
        except ValueError as e:
            assert "close_below" in str(e)
        # ...and a good block loads, with the CLI override winning over it.
        p.write_text("mode: pid\nreplay:\n  session: /nowhere\n"
                     "  v_max_m_s: 0.1\n")
        cfg = MpcConfig.load(p)
        assert cfg.replay["session"] == "/nowhere"
        assert cfg.replay["v_max_m_s"] == 0.1


# ---------------------------------------------------------------- refusals
def test_replay_refusals_each_name_their_cause():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, logs = _worker(tmp)
        w.set_scenario({"shape": "replay"})
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        for _ in range(4):
            _feed_good_state(w)
            w.tick()
        # (a) no session configured
        w.set_traj(True)
        assert not w.traj_on and "replay.session" in w.reason, w.reason
        # (b) a session dir with no poses.npy
        w.cfg.replay["session"] = tmp
        w.set_traj(True)
        assert not w.traj_on and "poses.npy" in w.reason, w.reason
        w.teardown()

    # (c) a contouring controller refuses the shape outright
    class MpccishStub(StubCtrl):
        progress_m = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        from rov_gui.bus import DataBus
        from rov_gui.control.workers import MpcWorker
        from rov_gui.tests.test_control import _app, _test_opts
        _app()
        bus = DataBus()
        w = MpcWorker(bus, _test_opts(tmp)(), controller_factory=MpccishStub)
        w.setup()
        w.cfg.log_dir = str(tmp)
        w.cfg.engage["warmup_s"] = 0.1
        w.cfg.engage["settle_s"] = 0.0
        w.cfg.replay["session"] = str(_fake_session(tmp))
        w.set_scenario({"shape": "replay"})
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        for _ in range(4):
            _feed_good_state(w)
            w.tick()
        w.set_traj(True)
        assert not w.traj_on and "contours its own path" in w.reason, w.reason
        w.teardown()


# ------------------------------------------------------------- the mission
def test_replay_arms_at_current_pose_and_completes():
    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=False)
        w, bus, pilots, logs = _armed_replay_worker(tmp, sess)
        eta_at_arm = np.asarray(w._eta[:3], float).copy()
        w.set_traj(True)
        assert w.traj_on, w.reason
        assert w.replay is not None and w.ctrl.scenario["kind"] == "replay"
        # dilation stayed 1: the demo is slower than the cap
        assert abs(w.replay["time_dilation"] - 1.0) < 1e-6
        _feed_good_state(w)
        w.tick()
        # the FIRST installed reference is where the vehicle already is —
        # arming a replay must not command a move (the follow rule)
        plan = w.ctrl._path_plan
        assert plan is not None, "no plan installed on the first tick"
        assert np.linalg.norm(np.asarray(plan.p_ned[:, 0]) - eta_at_arm) < 0.06
        # ...and the mission completes by its own (dilated) duration
        assert _run_until(w, lambda: not w.traj_on, timeout_s=8.0), \
            "replay never completed"
        assert "complete" in w.reason, w.reason
        # the record: plans.jsonl beside the CSV, one line per offered plan
        run_dir = w._csv_path.parent
        lines = [json.loads(x) for x in
                 (run_dir / "plans.jsonl").read_text().splitlines()]
        assert len(lines) == 1 and lines[0]["status"] in ("accept", "clip")
        assert lines[0]["margins"], "filter margins missing from plans.jsonl"
        # meta: schema 8, the always-written plan_stream block, the boundary
        meta = w._run_meta()
        assert meta["schema_version"] == 8
        assert meta["plan_stream"]["enabled"] is True
        assert meta["plan_stream"]["run"]["installed"] >= 1
        assert meta["trajectory"]["kind"] == "replay"
        assert meta["reference_clock"]["strategy"] == "plan_stream_replay"
        # CSV: the three schema-8 columns are present and populated
        w.teardown()
        head = (w._csv_path or run_dir / "x").read_text().splitlines()
        assert head[0].endswith("plan_id,ref_src,grip_cmd")
        assert any(",plan," in row or ",hold," in row for row in head[1:]), \
            "no row recorded a plan/hold reference source"


def test_replay_gripper_replays_and_every_clear_site_neutrals():
    """EVERY mission-clearing gesture must drop the latched jaw drive — the
    4-site rule, pinned per site rather than once (safety review 2026-08-30:
    this is exactly the invariant a later refactor breaks in one site)."""
    gestures = [
        ("stop_traj", lambda w: w.set_traj(False)),
        ("disengage", lambda w: w.disengage("test")),
        ("estop", lambda w: w.estop()),
        ("re_arm_station", lambda w: (w.set_scenario({"shape": "station"}),
                                      w.set_traj(True))),
        ("teardown", lambda w: w.teardown()),
    ]
    for name, gesture in gestures:
        with tempfile.TemporaryDirectory() as tmp:
            sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=True)
            w, bus, pilots, logs = _armed_replay_worker(tmp, sess,
                                                        gripper=True)
            drives: list[float] = []
            bus.cmd_gripper_drive.connect(drives.append)
            w.set_traj(True)
            assert w.traj_on, (name, w.reason)
            # the demo closes its jaw at 60 % of 2 s -> a CLOSE drive appears
            assert _run_until(w, lambda: -1.0 in drives, timeout_s=6.0), \
                f"[{name}] no CLOSE drive was emitted (got {drives})"
            gesture(w)
            assert drives[-1] == 0.0, (name, drives)
            assert w.replay is None, name
            if name == "stop_traj":
                # ...and the ended run still reaches the meta via _replay_last
                meta = w._run_meta()
                assert meta["plan_stream"]["enabled"] is True
                assert meta["plan_stream"]["run"]["grip_events"] >= 1
            if name != "teardown":
                w.teardown()


def test_replay_gripper_does_not_edge_on_the_demos_first_sample():
    """A demo that STARTS open must not drive an already-open jaw against its
    stop: the initial grip state is seeded from the first sample, so only a
    real threshold CROSSING emits (safety review 2026-08-30)."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=True)
        w, bus, pilots, logs = _armed_replay_worker(tmp, sess, gripper=True)
        drives: list[float] = []
        bus.cmd_gripper_drive.connect(drives.append)
        w.set_traj(True)
        assert w.replay["grip_state"] == "open"      # width 1.0 at frame one
        for _ in range(6):
            _feed_good_state(w)
            w.tick()
        assert +1.0 not in drives, \
            f"the open jaw was driven OPEN at t=0: {drives}"
        w.teardown()


def test_replay_one_shot_reject_ends_honestly_not_as_complete():
    """A rejected one-shot plan has no escalation ladder — the mission must
    END with the reject reason, never sit still for the demo's duration and
    then read 'complete' (safety review 2026-08-30)."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=False)
        # a workspace box the demo path leaves almost immediately
        w, bus, pilots, logs = _armed_replay_worker(
            tmp, sess,
            workspace_box_ned=[[-0.05, -0.05, -1.0], [0.05, 0.05, 1.0]])
        w.set_traj(True)
        assert w.traj_on, w.reason
        _feed_good_state(w)
        w.tick()                       # releases + rejects the one-shot plan
        assert not w.traj_on
        assert "rejected" in w.reason, w.reason
        assert "complete" not in w.reason
        rp = w._replay_last
        assert rp is not None and rp["installed"] == 0 and rp["rejected"] == 1
        run_dir = w._csv_path.parent
        line = json.loads(
            (run_dir / "plans.jsonl").read_text().splitlines()[0])
        assert line["status"] == "reject" and line["reasons"]
        w.teardown()


def test_replay_divergence_guard_stops_a_runaway_reference():
    """If the vehicle cannot stay with the reference (snagged tether), the
    plan must not keep marching away — the guard stops the mission instead of
    letting the NMPC wind up toward a distant point (safety review
    2026-08-30). Forced here by feeding a pose far from the anchor."""
    from rov_gui.tests.test_control import _fix, _imu
    from rov_gui.state import Conn, Telemetry
    from rov_gui.state import now as _now

    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=4.0, speed=0.08, grip=False)
        w, bus, pilots, logs = _armed_replay_worker(tmp, sess)
        w.set_traj(True)
        assert w.traj_on, w.reason
        for _ in range(3):             # let the plan install and track first
            _feed_good_state(w)
            w.tick()
        assert w.traj_on and w.ctrl._path_plan is not None
        # THEN the vehicle "snags": its measured pose jumps ~0.9 m off the
        # reference (> 2 x anchor_max_m = 0.6 m) and stays there
        def _feed_far():
            t = _now()
            w.on_nav_fix(_fix([-2.9, 0.0, 0.8], 0.0, t))
            w.on_vehicle_imu(_imu(depth=0.8, t=t))
            w.on_telemetry(Telemetry(armed=True, mode="MANUAL",
                                     conn=Conn.ONLINE))
        t0 = time.monotonic()
        while w.traj_on and time.monotonic() - t0 < 4.0:
            _feed_far()
            w.tick()
            time.sleep(0.02)
        assert not w.traj_on, "divergence guard never fired"
        assert "diverged" in w.reason, w.reason
        w.teardown()


def test_replay_streamed_mode_feeds_multiple_plans():
    with tempfile.TemporaryDirectory() as tmp:
        sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=False)
        w, bus, pilots, logs = _armed_replay_worker(
            tmp, sess, stream_period_s=0.5, horizon_s=1.0)
        w.set_traj(True)
        assert w.traj_on, w.reason
        assert w.replay["n_plans"] >= 3
        assert _run_until(w, lambda: not w.traj_on, timeout_s=10.0), \
            "streamed replay never completed"
        rp = w._replay_last
        assert rp is not None and rp["installed"] >= 3, rp
        assert rp["rejected"] == 0, \
            f"a clean chopped demo should pass its own filter: {rp}"
        run_dir = w._csv_path.parent if w._csv_path else Path(w.cfg.log_dir)
        n_lines = len((run_dir / "plans.jsonl").read_text().splitlines())
        assert n_lines == rp["released"] >= 3
        w.teardown()


def test_replay_serves_the_full_nmpc_horizon():
    """The real consumer is a 61-stage NMPC, not the 1-stage stub: the plan
    must arrive at the size the controller DECLARES, and the stages past the
    demo's end must hold the endpoint with zero velocity (never a ray)."""

    class Stub61(StubCtrl):
        @property
        def path_plan_steps(self):
            return 61

        @property
        def path_plan_dt(self):
            return 0.05

    with tempfile.TemporaryDirectory() as tmp:
        from rov_gui.bus import DataBus
        from rov_gui.control.workers import MpcWorker
        from rov_gui.tests.test_control import _app, _test_opts
        _app()
        bus = DataBus()
        w = MpcWorker(bus, _test_opts(tmp)(), controller_factory=Stub61)
        w.setup()
        w.cfg.log_dir = str(tmp)
        w.cfg.engage["warmup_s"] = 0.1
        w.cfg.engage["settle_s"] = 0.0
        sess = _fake_session(tmp, duration_s=2.0, speed=0.08, grip=False)
        w.cfg.replay["session"] = str(sess)
        w.cfg.replay["v_max_m_s"] = 0.20
        w.set_scenario({"shape": "replay"})
        w.on_enable(True)
        _feed_good_state(w)
        w.set_engaged(True)
        for _ in range(4):
            _feed_good_state(w)
            w.tick()
        w.set_traj(True)
        assert w.traj_on, w.reason
        _feed_good_state(w)
        w.tick()
        plan = w.ctrl._path_plan
        assert plan is not None and np.asarray(plan.p_ned).shape == (3, 61)
        assert np.asarray(plan.yaw_ned).size == 61
        assert np.asarray(plan.psi_path).size == 61
        # 61 x 0.05 s = 3.05 s of preview against a 2.0 s demo: the tail is
        # past the end from the very first tick — endpoint hold, v = 0
        v = np.asarray(plan.v_ned, float)
        assert np.linalg.norm(v[:, -1]) < 1e-9, v[:, -1]
        p = np.asarray(plan.p_ned, float)
        assert np.allclose(p[:, -1], p[:, -2]), "the tail is extrapolating"
        w.teardown()


# =============================================================================
# runner (same shape as test_control.py — works with or without pytest)
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
