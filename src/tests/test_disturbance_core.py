#!/usr/bin/env python3
"""
test_disturbance_core.py — the Qt-free half of the disturbance rig panel, no
hardware, no display.

    ~/miniforge3/envs/robust/bin/python src/tests/test_disturbance_core.py
    ~/miniforge3/envs/whisker/bin/python -m pytest src/tests/test_disturbance_core.py -v

Skips (exit 0, nothing run) when whisker_vision is not at $WHISKER_VISION_DIR /
~/Desktop/whisker_vision: ./c3 test runs every src/tests file on any machine.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import disturbance_core as core  # noqa: E402

try:
    WHISKER_ROOT = core.add_whisker_path()
    SKIP_REASON = None
except Exception as exc:  # noqa: BLE001
    WHISKER_ROOT = None
    SKIP_REASON = str(exc)

try:
    import pytest
    if SKIP_REASON:
        pytest.skip(SKIP_REASON, allow_module_level=True)
except ImportError:  # plain-script run
    pytest = None


def _args(*argv):
    cfg = core.load_rig_config()
    ap = core.build_arg_parser(cfg)
    return ap.parse_args(list(argv)), cfg


def test_sys_path_order_src_repo_then_whisker_last():
    assert core.sys_path_order_ok(), sys.path[:6]
    wv = str(WHISKER_ROOT.resolve())
    assert sys.path.index(wv) > sys.path.index(str(core.REPO_ROOT))
    # calling again is idempotent
    core.add_whisker_path()
    assert sys.path.count(wv) == 1


def test_missing_whisker_dir_raises_with_the_fix():
    with tempfile.TemporaryDirectory() as td:
        try:
            core.add_whisker_path(td)
        except core.WhiskerMissing as exc:
            assert core.WHISKER_DIR_ENV in str(exc)
        else:
            raise AssertionError("no WhiskerMissing")


def test_profile_validation_blocks_whisker_systemexit():
    old = os.environ.get("WHISKER_PROFILE")
    try:
        os.environ["WHISKER_PROFILE"] = "bogus"
        try:
            core.add_whisker_path()
        except RuntimeError as exc:
            assert "WHISKER_PROFILE" in str(exc)
        else:
            raise AssertionError("bogus profile accepted")
        os.environ["WHISKER_PROFILE"] = "daytime"
        core.add_whisker_path()           # valid value passes through untouched
        assert os.environ["WHISKER_PROFILE"] == "daytime"
    finally:
        if old is None:
            os.environ.pop("WHISKER_PROFILE", None)
        else:
            os.environ["WHISKER_PROFILE"] = old


def test_rig_config_yaml_and_fallbacks():
    cfg = core.load_rig_config()
    assert cfg.source.endswith("disturbance_rig.yaml"), cfg.source
    assert cfg.jog_piece_mm == 300.0
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rig.yaml"
        # an old yaml with fixed limits still loads; the limits are ignored
        p.write_text("gantry:\n  ip: 10.0.0.9\n  client_soft_limits_mm:\n"
                     "    X: [5, 3]\n    Y: [0, 123]\n  jog_piece_mm: 50\n"
                     "thruster:\n  max_pct: 40\n")
        c = core.load_rig_config(p)
        assert c.gantry_ip == "10.0.0.9"
        assert c.jog_piece_mm == 50.0
        assert not hasattr(c, "client_soft_limits_mm")
        assert c.thruster_max_pct == 40
        assert c.speed_mm_s == 50.0 and c.pattern_speed_mm_s == 10.0   # defaults
        p.write_text("pattern:\n  speed_mm_s: 999\n  circle_radius_mm: 1\n")
        c = core.load_rig_config(p)
        assert c.pattern_speed_mm_s == 999.0
        o = core.options_from_args(_args("--sim")[0], c)
        assert o.pattern_speed_mm_s == core.PATTERN_SPEED_RANGE[1]     # clamped to the UI
        assert o.circle_radius_mm == core.CIRCLE_RADIUS_RANGE[0]
        missing = core.load_rig_config(Path(td) / "nope.yaml")
        assert missing.source.startswith("defaults")
        assert missing.gantry_ip == "192.168.0.30"


def test_settings_caps_and_whisker_ports_off():
    args, cfg = _args("--sim", "--thruster-max", "250", "--ramp-step", "0",
                      "--ramp-interval", "-1", "--motor-vel", "500",
                      "--motor-vel-max", "60", "--speed", "999")
    s = core.settings_from_args(args, cfg)
    assert s.simulator is True
    assert s.thruster_max_pct == 100
    assert s.thruster_ramp_step_pct == 1
    assert s.thruster_ramp_interval_s == 0.0
    assert s.motor_vel_deg_s == 60.0            # capped by --motor-vel-max
    assert s.speed_mm_s == core.RIG_MAX_SPEED_MM_S == 100.0   # not whisker's 50
    assert s.pose_udp_port == 0 and s.estimate_udp_port == 0 and s.monitor_port == 0
    assert s.client_soft_limits_mm == {}         # taught per axis, never from config
    o = core.options_from_args(args, cfg)
    assert o.motor_interlock and o.autolog
    assert o.limits_file == o.data_root / core.LIMITS_FILE_NAME
    summary = core.settings_summary(s, o)
    for key in ("thruster_ramp_step_pct", "client_soft_limits_mm", "limits_file",
                "jog_stop_mode", "motor_interlock", "autolog", "rig_config"):
        assert key in summary, key
    assert summary["jog_stop_mode"] == 1


def test_options_flags():
    args, cfg = _args("--sim", "--no-motor-interlock", "--limits-file", "/tmp/x.json",
                      "--no-autolog", "--rig-note", "a", "--rig-note", "b")
    o = core.options_from_args(args, cfg)
    assert not o.motor_interlock and not o.autolog
    assert o.rig_notes == ["a", "b"]
    assert o.limits_file == Path("/tmp/x.json")


def test_thruster_link_constructs_without_pyserial():
    """The real factory must stay lazy: constructing the link imports nothing."""
    args, cfg = _args()
    s = core.settings_from_args(args, cfg)
    assert not s.simulator
    link = core.make_thruster_link(s)
    from motion_control import ThrusterLink
    assert isinstance(link, ThrusterLink)
    assert not link.is_alive()
    sim_args, _ = _args("--sim")
    sim_link = core.make_thruster_link(core.settings_from_args(sim_args, cfg))
    assert isinstance(sim_link, ThrusterLink)


def test_log_folder_files_and_meta():
    args, cfg = _args("--sim")
    s = core.settings_from_args(args, cfg)
    o = core.options_from_args(args, cfg)
    with tempfile.TemporaryDirectory() as td:
        log = core.DisturbLog(td, core.base_meta(s, o, WHISKER_ROOT,
                                                 preflight={"verdict": "ready"}))
        assert log.dir.name.endswith("_disturb"), log.dir
        assert log.dir.parent.parent == Path(td)
        log.sample({"host_epoch_s": time.time(), "t_monotonic_s": 1.0,
                    "x_mm": 1.23456, "motor_enabled": True, "pwm_us": 1500,
                    "latched": False})
        log.event("connect", "gantry")
        # streamed: visible BEFORE close
        lines = (log.dir / "samples.csv").read_text().splitlines()
        assert lines[0].split(",") == list(core.SAMPLE_COLUMNS)
        assert lines[1].split(",")[2] == "1.235"
        assert (log.dir / "events.csv").read_text().count("\n") == 2
        meta0 = json.loads((log.dir / "meta.json").read_text())
        assert meta0["stop_epoch_s"] is None and "t0_monotonic_s" in meta0
        assert meta0["whisker_vision"]["git"]["sha"]
        log.close(thruster_rows=[(time.time(), 10, 1540), (time.time(), 0, None)],
                  extra={"gantry": {"homed": ["X"]}})
        assert log.closed
        log.close()                                   # idempotent
        log.sample({"x_mm": 9.9})                     # ignored after close
        thr = (log.dir / "thruster.csv").read_text().splitlines()
        assert thr[0] == "host_epoch_s,throttle_pct,pwm_us"
        assert thr[-1].endswith(",0,")
        meta = json.loads((log.dir / "meta.json").read_text())
        assert meta["stop_epoch_s"] and meta["n_samples"] == 1 and meta["n_events"] == 1
        assert meta["gantry"]["homed"] == ["X"]
        assert meta["notes"]["thrust_is_commanded_not_measured"] is True
        assert sorted(p.name for p in log.dir.iterdir()) == [
            "events.csv", "meta.json", "samples.csv", "thruster.csv"]


def test_events_have_a_monotonic_column_and_meta_has_jet_calibration():
    with tempfile.TemporaryDirectory() as td:
        args, cfg = _args("--sim", "--companion", "data/20261022/1022_120000")
        s = core.settings_from_args(args, cfg)
        o = core.options_from_args(args, cfg)
        log = core.DisturbLog(td, core.base_meta(s, o, WHISKER_ROOT))
        log.event("x", "y")
        head, row = (log.dir / "events.csv").read_text().splitlines()
        assert head.split(",") == ["host_epoch_s", "t_monotonic_s", "kind", "detail"]
        assert float(row.split(",")[1]) > 0
        assert "jet_calibration" in log.meta             # None or whisker's aim file
        assert log.meta["settings"]["companion_run"] == "data/20261022/1022_120000"
        assert log.meta["settings"]["limits_file"].endswith(core.LIMITS_FILE_NAME)
        log.close()


def test_logger_never_raises_when_the_disk_refuses():
    """ENOSPC on flush must not propagate into a Qt slot: the logger goes quiet
    and records why (meta write_errors / log_failed)."""
    if not Path("/dev/full").exists():
        return
    with tempfile.TemporaryDirectory() as td:
        log = core.DisturbLog(td, {})
        for name in ("_samples_fh", "_events_fh"):
            getattr(log, name).close()
            fh = open("/dev/full", "w")
            setattr(log, name, fh)
        import csv
        log._samples = csv.writer(log._samples_fh)
        log._events = csv.writer(log._events_fh)
        log.sample({"x_mm": 1.0})          # raises nothing
        log.event("a", "b")
        assert log.failed and "samples.csv" in log.failed
        log.close()                         # thruster.csv + meta still written
        meta = json.loads((log.dir / "meta.json").read_text())
        assert meta["log_failed"].startswith("samples.csv")
        assert meta["write_errors"]


def test_provenance_is_precomputable():
    prov = core.provenance(WHISKER_ROOT)
    assert prov["repo_git"]["sha"] and prov["whisker_git"]["sha"] and prov["probed_epoch_s"]
    args, cfg = _args("--sim")
    meta = core.base_meta(core.settings_from_args(args, cfg), core.options_from_args(args, cfg),
                          WHISKER_ROOT, prov=prov)
    assert meta["panel"]["repo_git"] == prov["repo_git"]


def test_taught_limits_round_trip_and_cleaning():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sub" / core.LIMITS_FILE_NAME
        assert core.load_taught_limits(path) == {"axes": {}, "saved_epoch_s": None,
                                                 "last_pos_mm": None, "error": None}
        taught = {"X": {"lo": -900.0, "lo_epoch_s": 1.0, "hi": -100.0, "hi_epoch_s": 2.0},
                  "Y": {"hi": 40.0, "hi_epoch_s": 3.0},
                  "Z": {"lo": 5.0, "hi": 1.0},                    # inverted: dropped whole
                  "Q": {"lo": 0.0}}                               # not an axis
        assert core.save_taught_limits(path, taught, (-500.0, 12.5, 3.0)) is None
        got = core.load_taught_limits(path)
        assert got["error"] is None and got["saved_epoch_s"] > 0
        assert got["last_pos_mm"] == (-500.0, 12.5, 3.0)
        assert got["axes"] == {"X": taught["X"], "Y": taught["Y"]}
        assert "power" in json.loads(path.read_text())["frame"]
        assert core.save_taught_limits(path, taught) is None           # no position known
        assert core.load_taught_limits(path)["last_pos_mm"] is None
        path.write_text("{not json")
        bad = core.load_taught_limits(path)
        assert bad["axes"] == {} and bad["error"]
        assert core.clean_taught_limits({"X": {"lo": float("nan"), "hi": True}}) == {}
    assert core.save_taught_limits("/proc/no/such/dir/x.json", {}) is not None   # no raise


def test_rig_gantry_link():
    args, cfg = _args("--sim")
    o = core.options_from_args(args, cfg)
    # RigGantryLink relies on GantryLink internals; pin them
    from motion_control import GantryLink
    from motion_simulator import SimGantryController
    from motion.fmc4030 import Axis
    probe = GantryLink(SimGantryController, core.settings_from_args(args, cfg))
    for name in ("_state_lock", "_sdk_lock", "_soft_min_mm", "_soft_max_mm", "_generation",
                 "_pending", "_controller", "_stop_done", "_require_commandable", "_queue",
                 "_execute_relative", "_run_stop_sequence", "_execute", "_command_current",
                 "_finish_command", "_poll_positions", "_shutdown", "_settings"):
        assert hasattr(probe, name), name
    assert hasattr(probe._settings, "acc_mm_s2")
    assert o.jog_piece_mm == 300.0
    link = core.make_gantry_link(core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0) and link.is_ready
        link.set_axis_limits(Axis.X, -100.0, 0.0)
        lo, hi = link.soft_limits_mm()
        assert (lo[0], hi[0]) == (-100.0, 0.0)
        link.set_axis_limits(Axis.X, None, None)
        assert link.soft_limits_mm()[0][0] is None
        # a jog longer than the link's 10 mm cap, bounded by max_mm instead
        link.request_jog_relative(Axis.X, 82.5, 50.0, 300.0)
        import time as _t
        deadline = _t.monotonic() + 5.0
        while (link.busy or link.positions_mm()[0] < 82.0) and _t.monotonic() < deadline:
            _t.sleep(0.05)
        assert abs(link.positions_mm()[0] - 82.5) < 0.5
        # Send asks 90 mm/s: capped at SEND_MAX_SPEED_MM_S (50) until a Send has run
        # on this controller (Send needs both limits)
        link.set_axis_limits(Axis.X, -100.0, 400.0)
        link.request_move({Axis.X: 0.0}, 90.0)
        deadline = _t.monotonic() + 5.0
        while link.busy and _t.monotonic() < deadline:
            _t.sleep(0.05)
        last = [e for e in link._controller.call_log if e[0] == "jog"][-1]
        assert last[4] is False and abs(last[3] - 50.0 / 8.25) < 1e-6, last
        link.set_axis_limits(Axis.X, None, None)
        link.request_jog_relative(Axis.X, 82.5, 50.0, 300.0)
        deadline = _t.monotonic() + 5.0
        while (link.busy or link.positions_mm()[0] < 82.0) and _t.monotonic() < deadline:
            _t.sleep(0.05)
        try:
            link.request_jog_relative(Axis.X, 400.0, 50.0, 300.0)
        except ValueError:
            pass
        else:
            raise AssertionError("jog above max_mm accepted")
        for bad in ((1.0, None), (5.0, 1.0), (float("nan"), 1.0)):
            try:
                link.set_axis_limits(Axis.X, *bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {bad}")
    finally:
        link.stop()


class _MovingSim:
    """Built lazily (motion_simulator needs the whisker path)."""

    @staticmethod
    def make(only_mode_1: bool):
        from motion_simulator import SimGantryController

        class _Hw(SimGantryController):
            """A MODEL, half observed and half assumed: on 2026-10-26 stop_run +
            mode 2 were accepted and the relative jog ran on (observed,
            data/20261026/1026_131512_disturb/); mode 1 stopped 18/18 releases
            later that day (data/20261026/1026_140909_disturb/events.csv); inside STOP ALL still unobserved."""
            def stop_run(self):
                self.call_log.append(("stop_run",))
                if not only_mode_1:
                    super().stop_run()

            def stop_axis(self, axis, mode=2):
                if mode == 1 or not only_mode_1:
                    super().stop_axis(axis, mode)
                else:
                    self.call_log.append(("stop_axis", axis, mode))
        return _Hw


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_jog_stop_is_mode_1_and_does_not_latch():
    from motion.fmc4030 import Axis
    args, cfg = _args("--sim")
    hw = _MovingSim.make(only_mode_1=True)
    link = core.rig_gantry_link(hw, core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0)
        assert _wait(lambda: link.positions_mm() is not None and link.telemetry_fresh())
        link.request_jog_relative(Axis.Y, 200.0, 40.0, 300.0)
        assert _wait(lambda: link.positions_mm()[1] > 5.0)
        assert link.jog_stop(Axis.Y)
        assert _wait(lambda: not link.jog_stop_running and not link.busy)
        sim = link._controller
        stops = [e for e in sim.call_log if e[0] == "stop_axis"]
        assert stops[0] == ("stop_axis", Axis.Y, 1)           # the jog axis first, mode 1
        assert {e[1] for e in stops} == {Axis.X, Axis.Y, Axis.Z}
        assert ("stop_run",) not in sim.call_log               # not the STOP sequence
        time.sleep(0.3)                                        # the cache lags one poll
        y = link.positions_mm()[1]
        time.sleep(0.4)
        assert abs(link.positions_mm()[1] - y) < 0.05          # really stopped
        assert link.jog_stop_error is None
        link.request_jog_relative(Axis.Y, -2.0, 40.0, 300.0)   # NOT latched: next jog goes
        assert _wait(lambda: not link.busy and abs(link.positions_mm()[1] - (y - 2.0)) < 0.2)
    finally:
        link.stop()


def test_jog_stop_thread_start_failure_is_reported():
    import threading
    from motion.fmc4030 import Axis
    args, cfg = _args("--sim")
    link = core.make_gantry_link(core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0)
        real = threading.Thread.start

        def boom(self):
            if self.name == "gantry-jog-stop":
                raise RuntimeError("can't start new thread")
            return real(self)
        threading.Thread.start = boom
        try:
            assert link.jog_stop(Axis.X) is False
        finally:
            threading.Thread.start = real
        assert not link.jog_stop_running and "thread start failed" in link.jog_stop_error
    finally:
        link.stop()


def test_jog_stop_cancels_a_queued_jog():
    """Press and release before the worker dequeued the jog: nothing moves."""
    from motion.fmc4030 import Axis
    args, cfg = _args("--sim")
    link = core.make_gantry_link(core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0)
        assert _wait(lambda: link.positions_mm() is not None and link.telemetry_fresh())
        with link._sdk_lock:                       # hold the worker away from the SDK
            link.request_jog_relative(Axis.X, 100.0, 40.0, 300.0)
            link.jog_stop(Axis.X)
        assert _wait(lambda: not link.jog_stop_running and not link.busy)
        time.sleep(0.3)
        assert not any(e[0] == "jog" for e in link._controller.call_log)
    finally:
        link.stop()


def test_stop_sequence_leads_with_mode_1():
    """STOP ALL against the 10-26 MODEL (mode 1 assumed effective): mode 1 on
    every axis first, then mode 2; no stop_run (unseparated 10-26 suspect)."""
    from motion.fmc4030 import Axis
    args, cfg = _args("--sim")
    link = core.rig_gantry_link(_MovingSim.make(only_mode_1=True),
                                core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0)
        assert _wait(lambda: link.positions_mm() is not None and link.telemetry_fresh())
        link.request_jog_relative(Axis.X, 200.0, 40.0, 300.0)
        assert _wait(lambda: link.positions_mm()[0] > 5.0)
        link.stop_all(wait=False)
        assert _wait(lambda: link._stop_done.is_set())
        calls = [e for e in link._controller.call_log if e[0] in ("stop_axis", "stop_run")]
        assert calls[:3] == [("stop_axis", a, 1) for a in (Axis.X, Axis.Y, Axis.Z)]
        assert calls[3:6] == [("stop_axis", a, 2) for a in (Axis.X, Axis.Y, Axis.Z)]
        assert ("stop_run",) not in calls
        time.sleep(0.3)                             # the cache lags one poll
        x = link.positions_mm()[0]
        time.sleep(0.4)
        assert abs(link.positions_mm()[0] - x) < 0.05
        try:
            link.request_jog_relative(Axis.X, 1.0, 40.0, 300.0)
        except RuntimeError as exc:
            assert "latched" in str(exc)            # STOP ALL still latches
        else:
            raise AssertionError("STOP did not latch")
    finally:
        link.stop()


def test_pattern_geometry():
    import math
    pts = core.circle_points((100.0, 50.0), 30.0, 8.0)
    assert len(pts) == 24                                  # ceil(2 pi 30 / 8)
    assert all(abs(math.hypot(y - 100.0, z - 50.0) - 30.0) < 1e-9 for y, z in pts)
    assert abs(pts[-1][0] - 130.0) < 1e-9 and abs(pts[-1][1] - 50.0) < 1e-9   # back at the start
    assert len(core.circle_points((0.0, 0.0), 5.0, 8.0)) == 12                 # at least 12
    assert core.line_points(10.0, 100.0) == [(60.0,), (-40.0,)]
    env = core.path_envelope((100.0, 50.0), pts)
    assert all(abs(a - b) < 1e-9 for a, b in zip(env[0], (70.0, 130.0)))
    assert all(abs(a - b) < 1e-9 for a, b in zip(env[1], (20.0, 80.0)))
    for bad in ((float("nan"), 0.0), ):
        try:
            core.circle_points(bad, 10.0)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a NaN centre")
    try:
        core.line_points(0.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a zero range")


def test_request_path_refuses_targets_outside_its_box():
    from motion.fmc4030 import Axis
    args, cfg = _args("--sim")
    link = core.make_gantry_link(core.settings_from_args(args, cfg))
    link.start()
    try:
        assert link.wait_until_ready(5.0)
        assert _wait(lambda: link.positions_mm() is not None and link.telemetry_fresh())
        y, z = link.positions_mm()[1], link.positions_mm()[2]       # sim: 0, 0
        pts = [(y + 10.0, z), (y, z + 10.0)]
        for env in ([(0.0, 5.0), (0.0, 20.0)],                 # a vertex outside the box
                    [(5.0, 20.0), (0.0, 20.0)],                # the start outside the box
                    [(0.0, float("inf")), (0.0, 20.0)]):       # not a box
            try:
                link.request_path([Axis.Y, Axis.Z], pts, 10.0, env)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {pts} in {env}")
        try:
            link.request_path([Axis.Y], [(y + 10.0,)], 10.0, [(0.0, 20.0)])
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a one-vertex path")
        link.set_axis_limits(Axis.Y, -1.0, 5.0)                     # installed limits win too
        try:
            link.request_path([Axis.Y], [(y + 10.0,), (y,)], 10.0, [(0.0, 20.0)])
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a vertex outside the soft limits")
        link.set_axis_limits(Axis.Y, None, None)
        assert not link.busy                                        # nothing was queued
        link.request_path([Axis.Y], [(y + 4.0,), (y,)], 30.0, [(y, y + 4.0)])
        assert _wait(lambda: (link.path_progress or (0,))[0] >= 2, 5)
        assert link._path_axes == (Axis.Y,)               # STOP ALL would stop Y first
        sim = link._controller
        n0 = len(sim.call_log)
        link._run_stop_sequence()                         # the order, not the latch
        order = [e[1] for e in sim.call_log[n0:] if e[0] == "stop_axis" and e[2] == 1]
        assert order[0] == Axis.Y, order
        # a stop that lands mid-segment leaves Y short of its vertex: the worker
        # must call that an error, not a reached vertex (a stop ON a vertex is
        # legitimately reached, so repeat until one lands mid-segment)
        for _ in range(40):
            if _wait(lambda: not link.busy, 0.3):
                break
            link._run_stop_sequence()
        assert not link.busy
        assert link.last_command_error and "from its vertex" in link.last_command_error
        link.reset_stop()
        assert link.path_progress is not None
        assert all(0.0 - 1e-6 <= p <= 4.0 + 1e-6 for p in (link.positions_mm()[1],))
    finally:
        link.stop()


def test_two_logs_in_the_same_second_get_different_folders():
    with tempfile.TemporaryDirectory() as td:
        now = time.time()
        a = core.DisturbLog(td, {}, when=now)
        b = core.DisturbLog(td, {}, when=now)
        assert a.dir != b.dir
        a.close()
        b.close()


def test_parse_ss_estab():
    text = ("State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
            "ESTAB 0 0 192.168.0.35:41234 192.168.0.30:8088\n"
            "LISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n")
    assert core.parse_ss_estab(text, "192.168.0.30", 8088).startswith("ESTAB")
    assert core.parse_ss_estab(text, "192.168.0.30", 8089) is None
    assert core.parse_ss_estab("ESTAB 0 0 1.2.3.4:5 192.168.0.30:80880", "192.168.0.30", 8088)  # substring: documented limitation, keep loud


def test_git_head_never_raises():
    assert core.git_head("/nonexistent/path/xyz") == {"sha": None, "branch": None,
                                                       "dirty": None}
    g = core.git_head(core.REPO_ROOT)
    assert g["sha"] and len(g["sha"]) == 40


def test_argv_prescans():
    assert core.rig_config_path_from_argv(["--sim", "--rig-config", "/x.yaml"]) == "/x.yaml"
    assert core.rig_config_path_from_argv(["--sim"]) is None
    assert core.whisker_dir_from_argv(["--whisker-dir", "/w"]) == "/w"


def main() -> int:
    if SKIP_REASON:
        print(f"  skip  whisker_vision unavailable: {SKIP_REASON}")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
