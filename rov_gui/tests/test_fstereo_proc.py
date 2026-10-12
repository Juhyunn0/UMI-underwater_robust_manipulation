"""--fstereo-process: FoundationStereo in a child process (2026-10-12).

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
        -m pytest rov_gui/tests/test_fstereo_proc.py -q -p no:cacheprovider
    # or: QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
    #         rov_gui/tests/test_fstereo_proc.py

Every test that needs a child starts a REAL one (``python -m
rov_gui.perception.fstereo_proc``) with the stub session — no torch, no GPU,
no camera — so the process lifecycle, the wire and the failure paths are the
shipped code; only the network is replaced.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rov_gui.perception.fstereo import FStereoError, FStereoSession  # noqa: E402
from rov_gui.perception.fstereo_proc import RemoteFStereoSession       # noqa: E402

KEYS = {"depth_mm", "depth_native", "valid_native", "valid_out", "filled_out",
        "solve_ms", "rig_k"}


def _session(stub=None, **kw):
    s = RemoteFStereoSession(iters=8, scale=0.5, stub=stub or {}, **kw)
    logs = []
    s.start_async(on_log=lambda lvl, msg: logs.append((lvl, msg)))
    t0 = time.monotonic()
    while s.loading and time.monotonic() - t0 < 20.0:
        time.sleep(0.01)
    return s, logs


def _pair(tag=7):
    """Two DIFFERENT images (left = tag, right = tag + 1): the stub answers
    left*10 + right, so a swapped or duplicated image changes the number."""
    return (np.full((400, 640), int(tag), np.uint8),
            np.full((400, 640), int(tag) + 1, np.uint8))


def _alive(pid: int) -> bool:
    """True while the pid exists and is not a zombie."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------ surface
def test_the_proxy_has_the_surface_the_worker_reads():
    """Everything FStereoWorker touches on its session exists on the proxy
    (hardware.py setup / _publish_state / tick / meta / teardown)."""
    s = RemoteFStereoSession(iters=8, scale=0.5, stub={})
    for name in ("ready", "loading", "error", "load_seconds", "graph",
                 "graph_requested", "iters", "scale", "infer_size",
                 "start_async", "infer", "describe", "close", "solve_ms"):
        assert hasattr(s, name), name
    # before any load: the in-process describe() keys, honest placeholders
    assert set(s.describe()) == set(FStereoSession(scale=0.5).describe())
    assert s.ready is False and s.error == "" and s.load_seconds == 0.0
    s.close()                                  # never started: a no-op


def test_bad_settings_raise_synchronously_like_the_in_process_session():
    for kw in ({"scale": 0.0}, {"scale": 1.5}, {"size": (32, 32)}):
        try:
            RemoteFStereoSession(stub={}, **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} should raise ValueError")


# ------------------------------------------------------------------ the wire
def test_round_trip_through_a_real_child():
    s, logs = _session({"load_s": 0.05})
    try:
        assert s.ready and not s.error, s.error
        assert s.pid and _alive(s.pid)
        rig = SimpleNamespace(k=1.5)
        for i in range(3):
            left, right = _pair(7)
            out = s.infer(left, right, rig, out_size=(640, 360))
            assert set(out) == KEYS, sorted(out)       # rect_left is not sent
            assert out["depth_native"].dtype == np.uint16
            assert int(out["depth_native"][0, 0]) == 78   # 7*10 + 8, in the child
            assert out["depth_mm"].shape == (360, 640)
            assert int(out["depth_mm"][0, 0]) == i + 1    # child's frame count
            assert out["rig_k"] == 1.5
        m = s.process_meta()
        assert m["mode"] == "child" and m["alive"] and m["pid"] == s.pid
        assert m["rig_sends"] == 1, m              # one rig object, sent once
        # frame 1 (the graph capture) is recorded on its own, not in rpc_ms
        assert m["frames_ok"] == 3 and m["rpc_ms"]["n"] == 2, m
        assert m["first_frame_ms"] is not None and "STATION SESSION" in m["stats_scope"]
        d = s.describe()
        d["mutated"] = True
        assert "mutated" not in s.describe()       # a fresh dict every call
        assert any("own process" in msg.lower() for _, msg in logs), logs
        assert any("stub ready" in msg for _, msg in logs), logs  # relayed
    finally:
        s.close()


def test_a_new_rig_object_is_sent_again_and_used():
    s, _ = _session()
    try:
        a, b = SimpleNamespace(k=1.0), SimpleNamespace(k=2.0)
        assert s.infer(*_pair(), a)["rig_k"] == 1.0
        assert s.infer(*_pair(), a)["rig_k"] == 1.0
        assert s.infer(*_pair(), b)["rig_k"] == 2.0
        assert s.process_meta()["rig_sends"] == 2
    finally:
        s.close()


# ------------------------------------------------------------------ failures
def test_a_bad_frame_raises_but_does_not_fault():
    """The in-process contract: one frame's exception is counted by the
    worker (INFER_FAIL_LIMIT), it is not a dead feature."""
    s, _ = _session({"raise_on": 2})
    try:
        rig = SimpleNamespace(k=1.0)
        s.infer(*_pair(), rig)
        try:
            s.infer(*_pair(), rig)
            raise AssertionError("frame 2 should raise")
        except FStereoError as e:
            assert "stub frame 2 failed" in str(e)
        assert s.error == "" and s.ready
        assert int(s.infer(*_pair(), rig)["depth_mm"][0, 0]) == 3
    finally:
        s.close()


def test_a_dead_child_faults_at_once_not_after_the_timeout():
    s, logs = _session({"crash_on": 2}, infer_timeout_s=5.0)
    try:
        rig = SimpleNamespace(k=1.0)
        s.infer(*_pair(), rig)
        t0 = time.monotonic()
        try:
            s.infer(*_pair(), rig)
            raise AssertionError("frame 2 should raise")
        except FStereoError:
            pass
        assert time.monotonic() - t0 < 2.0, "woken by EOF, not by the timeout"
        assert "exited" in s.error and not s.ready, s.error
        t0 = time.monotonic()
        try:
            s.infer(*_pair(), rig)
            raise AssertionError("a faulted session must refuse")
        except FStereoError:
            pass
        assert time.monotonic() - t0 < 0.1
        assert s.process_meta()["exit_code"] == 7
        assert any(lvl == "error" for lvl, _ in logs), logs
    finally:
        s.close()


def test_a_hung_child_is_stopped_after_the_timeout():
    s, _ = _session({"hang_on": 2}, infer_timeout_s=0.5)
    try:
        rig = SimpleNamespace(k=1.0)
        s.infer(*_pair(), rig)
        pid = s.pid
        t0 = time.monotonic()
        try:
            s.infer(*_pair(), rig)
            raise AssertionError("frame 2 should time out")
        except FStereoError:
            pass
        dt = time.monotonic() - t0
        assert 0.4 < dt < 2.0, dt
        assert "did not answer" in s.error, s.error
        t0 = time.monotonic()
        while _alive(pid) and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        assert not _alive(pid), "a hung child holds the GPU: it must be killed"
    finally:
        s.close()


def test_a_load_error_is_a_fault():
    s, _ = _session({"load_error": "checkpoint vanished"})
    try:
        assert s.error == "checkpoint vanished" and not s.ready \
            and not s.loading, (s.error, s.ready, s.loading)
    finally:
        s.close()


def test_a_load_that_never_ends_is_stopped():
    s = RemoteFStereoSession(iters=8, scale=0.5, stub={"load_s": 30.0},
                             load_timeout_s=0.4)
    s.start_async()
    pid = s.pid
    t0 = time.monotonic()
    while s.loading and time.monotonic() - t0 < 5.0:
        time.sleep(0.02)
    try:
        assert time.monotonic() - t0 < 2.0
        assert "did not finish loading" in s.error, s.error
        t0 = time.monotonic()
        while _alive(pid) and time.monotonic() - t0 < 2.0:
            time.sleep(0.02)
        assert not _alive(pid)
    finally:
        s.close()


# ------------------------------------------------------------------ shutdown
def test_close_is_bounded_even_when_the_child_ignores_it():
    """Inside the worker's 4000 ms stop budget (backends/base.py)."""
    s, _ = _session({"ignore_close": True})
    pid = s.pid
    t0 = time.monotonic()
    s.close()
    dt = time.monotonic() - t0
    assert dt < 3.5, dt
    assert not _alive(pid)
    assert s.process_meta()["exit_code"] is not None


def test_close_during_load_is_bounded():
    s = RemoteFStereoSession(iters=8, scale=0.5, stub={"load_s": 30.0})
    s.start_async()
    pid = s.pid
    t0 = time.monotonic()
    s.close()
    assert time.monotonic() - t0 < 3.5
    assert not _alive(pid)


def _station_dies(stub_literal: str, wait_loaded: bool) -> None:
    """SIGKILL a 'station' (a helper process holding the session) and require
    the child to go too, instead of living on holding the GPU."""
    wait = ("while s.loading:\n    time.sleep(0.01)\nassert s.ready"
            if wait_loaded else
            "time.sleep(1.0)\nassert s.loading, 'must die MID-load'")
    code = (f"import sys, time\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            f"from rov_gui.perception.fstereo_proc import RemoteFStereoSession\n"
            f"s = RemoteFStereoSession(iters=8, scale=0.5, stub={stub_literal})\n"
            f"s.start_async()\n"
            f"{wait}\n"
            f"print(s.pid, flush=True)\n"
            f"time.sleep(60)\n")
    station = subprocess.Popen([sys.executable, "-c", code],
                               stdout=subprocess.PIPE, text=True)
    try:
        line = station.stdout.readline().strip()
        assert line, f"helper station failed (rc {station.poll()})"
        child = int(line)
        assert _alive(child)
        station.send_signal(signal.SIGKILL)
        station.wait(5)
        t0 = time.monotonic()
        while _alive(child) and time.monotonic() - t0 < 5.0:
            time.sleep(0.05)
        assert not _alive(child), "orphaned FoundationStereo child"
    finally:
        if station.poll() is None:
            station.kill()


def test_the_child_dies_with_the_station():
    """Loaded: the child sees EOF on its socket."""
    _station_dies("{}", wait_loaded=True)


def test_the_child_dies_with_the_station_mid_load():
    """Still loading (4.7-10.5 s for the real network), so it reads no socket
    and sees no EOF: the parent-pid watchdog must end it."""
    _station_dies("{'load_s': 60.0}", wait_loaded=False)


def test_a_child_that_stops_reading_cannot_wedge_the_worker():
    """SIGSTOPped child (a debugger, a frozen cgroup): the send has ONE
    deadline per message, so infer() fails closed in about send_timeout_s
    instead of blocking the worker thread."""
    s, _ = _session(infer_timeout_s=0.5, send_timeout_s=0.5)
    try:
        rig = SimpleNamespace(k=1.0)
        s.infer(*_pair(), rig)
        os.kill(s.pid, signal.SIGSTOP)
        big = np.zeros((1200, 1920), np.uint8)      # ~4.6 MB pair: > any buffer
        t0 = time.monotonic()
        try:
            s.infer(big, big.copy(), rig)
            raise AssertionError("a stopped child must not answer")
        except FStereoError:
            pass
        dt = time.monotonic() - t0
        assert dt < 2.0, f"worker wedged for {dt:.2f} s"
        assert s.error and not s.ready, s.error
    finally:
        try:
            os.kill(s.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        s.close()


# ------------------------------------------------------------------ worker
def _worker_with_remote(gate=None, panel=False):
    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.bus import DataBus, FrameMailbox, PolicyMailbox, StereoMailbox
    from rov_gui.tests.test_control import _app

    _app()
    opts = SimpleNamespace(source="hw", fstereo=True, fstereo_process=True,
                           fstereo_iters=8, fstereo_scale=0.5,
                           fstereo_alpha=0.5, fstereo_align_size="400x250")
    smb = StereoMailbox()
    # panel=True: a depth panel mailbox, without which _status() stays quiet
    fsw = FStereoWorker(DataBus(), smb, opts,
                        {"depth": FrameMailbox("depth")} if panel else {})
    s, _ = _session()
    fsw.session = s
    fsw.enabled = True
    smb.set_wanted(True)
    pmb = PolicyMailbox()
    pmb.set_wanted(True)
    fsw.policy_mb = pmb
    if gate is not None:
        fsw.fs_gate = gate
    f = 300.0
    rig = SimpleNamespace(R1=np.eye(3),
                          P1=np.array([[f, 0, 320, 0], [0, f, 200, 0],
                                       [0, 0, 1, 0]]),
                          mono_size=(640, 400), alpha=0.5, provenance={}, k=1.0)
    return fsw, smb, s, pmb, rig


def test_the_worker_taps_the_policy_with_the_station_capture_stamp():
    """The stamp never crosses the process boundary: the policy gets the
    child's depth under the CAMERA's t_capture, and the gate sees the round
    trip as the frame in flight."""
    from rov_gui.perception.fs_gate import FsGate
    from rov_gui.state import now

    gate = FsGate("yield")
    fsw, smb, s, pmb, rig = _worker_with_remote(gate)
    try:
        t_cap = now() - 0.123
        left, right = _pair(5)
        smb.put(left, right, rig, t_cap, frame_seq=1)
        fsw.tick()
        item = pmb.take()
        assert item is not None, pmb.counters()
        assert item["t_capture"] == t_cap
        assert int(item["depth"][0, 0]) == 56 and item["depth"].shape == (400, 640)
        assert fsw._frames == 1 and fsw._infer_fails == 0
        snap = gate.snapshot()
        assert snap["in_flight"] is False and snap["counters"]["delivered"] == 1, snap
        m = fsw.meta()
        assert m["process"]["mode"] == "child" and m["process"]["pid"] == s.pid
        assert m["process"]["frames_ok"] == 1
    finally:
        fsw.teardown()
    assert fsw.session is None


def test_a_dead_child_disarms_the_worker():
    """Fail closed: the next tick after the child dies is a FAULT, not a
    silent switch to anything else."""
    fsw, smb, s, pmb, rig = _worker_with_remote()
    try:
        os.kill(s.pid, signal.SIGKILL)
        t0 = time.monotonic()
        while not s.error and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
        assert "exited" in s.error, s.error
        fsw.tick()
        assert fsw.enabled is False
        assert smb.wanted() is False
    finally:
        fsw.teardown()


def _setup_worker(flag: bool):
    """Run the REAL FStereoWorker.setup() with the network neutralised at
    CLASS level (so even an inverted branch or a moved import cannot load
    FoundationStereo onto the GPU): the child is forced onto the stub, the
    in-station session's start_async is a no-op."""
    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.bus import DataBus, StereoMailbox
    from rov_gui.perception import fstereo as fs
    from rov_gui.perception import fstereo_proc as fp
    from rov_gui.tests.test_control import _app

    _app()
    orig_init = fp.RemoteFStereoSession.__init__
    orig_start = fs.FStereoSession.start_async

    def stub_init(self, *a, **kw):
        kw["stub"] = {}
        orig_init(self, *a, **kw)

    fp.RemoteFStereoSession.__init__ = stub_init
    fs.FStereoSession.start_async = lambda self, on_log=None: None
    states = []
    try:
        bus = DataBus()
        bus.fstereo_state.connect(states.append)
        opts = SimpleNamespace(source="hw", fstereo=True, fstereo_process=flag,
                               fstereo_iters=8, fstereo_scale=0.5,
                               fstereo_align_size="400x250")
        fsw = FStereoWorker(bus, StereoMailbox(), opts, {})
        fsw.setup()
        s = fsw.session
        t0 = time.monotonic()
        while flag and s.loading and time.monotonic() - t0 < 10.0:
            time.sleep(0.01)
        fsw._publish_state()
        return fsw, s, states
    finally:
        fp.RemoteFStereoSession.__init__ = orig_init
        fs.FStereoSession.start_async = orig_start


def test_setup_builds_the_child_session_when_the_flag_is_on():
    from rov_gui.perception.fstereo_proc import RemoteFStereoSession

    fsw, s, states = _setup_worker(True)
    try:
        assert isinstance(s, RemoteFStereoSession) and s.ready, s.error
        m = fsw.meta()["process"]
        assert m["mode"] == "child" and m["pid"] == s.pid and m["alive"], m
        assert states and states[-1].process == "proc", states[-1:]
    finally:
        fsw.teardown()


def test_setup_builds_the_in_station_session_when_the_flag_is_off():
    from rov_gui.perception.fstereo import FStereoSession

    fsw, s, states = _setup_worker(False)
    try:
        assert type(s) is FStereoSession
        assert fsw.meta()["process"] == {"mode": "in-station"}
        assert states and states[-1].process == "", states[-1:]
    finally:
        fsw.session = None              # nothing was loaded: no close needed


def test_a_dead_child_faults_without_flooding_the_gui():
    """The FAULT status is throttled (0.5 s) like the starvation branch: a
    child that dies mid-run must not queue a video_stat every 5 ms tick."""
    fsw, smb, s, pmb, rig = _worker_with_remote(panel=True)
    stats = []
    fsw.bus.video_stat.connect(stats.append)
    try:
        os.kill(s.pid, signal.SIGKILL)
        t0 = time.monotonic()
        while not s.error and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
        for _ in range(100):
            fsw.tick()
        assert fsw.enabled is False
        assert 1 <= len(stats) <= 2, len(stats)
    finally:
        fsw.teardown()


def test_meta_says_in_station_when_the_flag_is_off():
    from rov_gui.backends.hardware import FStereoWorker
    from rov_gui.bus import DataBus
    from rov_gui.tests.test_control import _app

    _app()
    fsw = FStereoWorker(DataBus(), None, SimpleNamespace(), {})
    assert fsw.meta()["process"] == {"mode": "in-station"}
    fsw.opts = SimpleNamespace(fstereo_process=True)
    assert fsw.meta()["process"] == {"mode": "child"}


# ------------------------------------------------------------------ records
def test_the_chip_names_the_layout():
    from rov_gui.state import FStereoState

    st = FStereoState(live=True, enabled=True, process="proc")
    assert st.chip.endswith("[proc]"), st.chip
    st = FStereoState(live=True, enabled=True, process="proc", schedule="yield")
    assert st.chip.endswith("[yield proc]"), st.chip
    st = FStereoState(live=True, enabled=True, schedule="only")
    assert st.chip.endswith("[only]"), st.chip
    assert "[" not in FStereoState(live=True, enabled=True).chip


def test_every_policy_plan_line_carries_fs_process():
    from rov_gui.control.workers import _plan_timing

    out = _plan_timing(SimpleNamespace(fs_process="child", fs_schedule="free",
                                       obs_t=1.0))
    assert out["fs_process"] == "child" and out["fs_schedule"] == "free"
    assert _plan_timing(SimpleNamespace())["fs_process"] is None


def test_the_policy_tags_the_layout_only_when_it_flew_fstereo():
    from rov_gui.backends.policy import PolicyWorker

    w = PolicyWorker.__new__(PolicyWorker)
    w.fs_gate, w.fstereo_meta_fn, w._obs_mode = None, (lambda: {}), "depth"
    w.opts = SimpleNamespace(fstereo_iters=8, fstereo_scale=0.5,
                             fstereo_size=None, fstereo_process=True)
    assert w._fs_process_tag() == "child"
    w.opts.fstereo_process = False
    assert w._fs_process_tag() == "in-station"
    w.fstereo_meta_fn = None                    # no FoundationStereo this run
    assert w._fs_process_tag() == ""
    w.fstereo_meta_fn, w._obs_mode = (lambda: {}), "rgb"
    assert w._fs_process_tag() == ""            # RGB ckpt reads no FS depth


# ------------------------------------------------------------------ CLI
def test_cli_flag_and_launch_check():
    from rov_gui.__main__ import (build_parser, check_fstereo_process,
                                  resolve_defaults)

    a = resolve_defaults(build_parser().parse_args(["--source", "hw"]))
    assert a.fstereo_process is False and check_fstereo_process(a) is None
    a = resolve_defaults(build_parser().parse_args(
        ["--source", "hw", "--fstereo-process"]))
    level, text = check_fstereo_process(a)
    assert level == "refuse" and "needs --fstereo" in text
    a = resolve_defaults(build_parser().parse_args(
        ["--source", "demo", "--fstereo", "--fstereo-process"]))
    level, text = check_fstereo_process(a)
    assert level == "warn" and "does nothing" in text
    a = resolve_defaults(build_parser().parse_args(
        ["--source", "hw", "--fstereo", "--fstereo-process"]))
    level, text = check_fstereo_process(a)
    assert level == "warn" and "RECORD BOUNDARY" in text


def test_main_refuses_the_flag_without_fstereo():
    """Through main(), with the hardware path FENCED: a regression in the
    check must fail here, not walk into the real preflight and station."""
    import contextlib
    import faulthandler
    import io

    import rov_gui.__main__ as M
    import rov_gui.window as W

    def hardware(*_a, **_k):
        raise AssertionError("main() went past the launch checks into the "
                             "hardware path")

    saved = (M._preflight, W.build_and_run, faulthandler.enable)
    M._preflight, W.build_and_run = hardware, hardware
    faulthandler.enable = lambda *a, **k: None
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            rc = M.main(["--source", "hw", "--fstereo-process"])
    finally:
        M._preflight, W.build_and_run, faulthandler.enable = saved
    assert rc == 2, rc
    assert "REFUSED" in err.getvalue(), err.getvalue()
    assert "--fstereo-process needs --fstereo" in err.getvalue(), err.getvalue()


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
