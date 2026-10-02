#!/usr/bin/env python3
"""
test_poolcam.py — the pool-corner cameras (``--pool-cams``), with no camera.

    ~/miniforge3/envs/rovgui-pose/bin/python -m pytest rov_gui/tests/test_poolcam.py -v
    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_poolcam.py   # no pytest

The camera process runs for real (``pool_cam.py --serve --fake N``: synthetic
test patterns, real ffmpeg), so what is checked is the actual contract:

* REC UI records every camera not left out, into the run folder, all from the
  UI recording's own t0 — the synchronisation the feature exists for;
* AUTO REC OFF leaves a camera out and session.json says so;
* the tab exists only with the flag, takes no keyboard focus, and the window
  still fits one small screen with it;
* nothing on the station side of the cameras can reach the control path.
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pool_cam import protocol
from rov_gui import theme
from rov_gui.qt import Qt, QTimer, QtWidgets


class Opts:
    """The CLI namespace, minus argparse (as test_offline.Opts)."""
    source = "demo"
    fps = 15.0
    ui_fps = 60.0
    thrusters = 8
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"

    def __init__(self, **kw):
        self.rec_dir = tempfile.mkdtemp(prefix="rov_gui_pool_")
        for k, v in kw.items():
            setattr(self, k, v)


_APP = None


def _app():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _pump(app, ms: int) -> None:
    QTimer.singleShot(ms, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()


def _config(**kw) -> str:
    """A pool_cams.yaml in a temp dir; x264 so the test needs no GPU."""
    body = {"encoder": "x264", "fps": 30, **kw}
    path = Path(tempfile.mkdtemp(prefix="pool_cfg_")) / "pool_cams.yaml"
    lines = []
    for k, v in body.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            lines += [f"  {kk}: {vv}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _wait(cond, timeout: float, app=None) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        if app is not None:
            _pump(app, 50)
        else:
            time.sleep(0.05)
    return cond()


# =============================================================================
# the wire
# =============================================================================
def test_protocol_roundtrip_and_truncation():
    msgs = [({"t": "status", "cams": [{"label": "cam0"}]}, b""),
            ({"t": "frame", "cam": "cam0", "w": 2, "h": 1}, bytes(range(6)))]
    stream = io.BytesIO(b"".join(protocol.pack(m, p) for m, p in msgs))
    for m, p in msgs:
        assert protocol.read_message(stream) == (m, p)
    assert protocol.read_message(stream) is None, "end of stream must be None"

    whole = protocol.pack({"t": "log"}, b"abc")
    assert protocol.read_message(io.BytesIO(whole[:-1])) is None, \
        "a writer that died mid-message is an end, not a message"
    bad = protocol.HEADER.pack(protocol.MAX_JSON + 1, 0)
    try:
        protocol.read_message(io.BytesIO(bad))
        raise AssertionError("an impossible header must raise")
    except ValueError:
        pass

    line = protocol.command("arm", cam="cam1", on=False)
    assert line.endswith(b"\n")
    assert protocol.parse_command(line) == {"cmd": "arm", "cam": "cam1", "on": False}
    assert protocol.parse_command(b"not json\n") is None
    assert protocol.parse_command(b'{"no": "cmd"}\n') is None


def test_child_argv_from_config_and_flags():
    from rov_gui.poolcam import DEMO_CAMERAS, child_argv, load_config

    cfg = _config(devices={"NE": "/dev/v4l/by-path/a", "SW": "/dev/v4l/by-path/b"},
                  size="1280x720")
    hw = child_argv(Opts(source="hw", pool_cam_config=cfg))
    assert hw[1].endswith("pool_cam/pool_cam.py") and "--serve" in hw
    assert hw[hw.index("--size") + 1] == "1280x720"
    assert hw[hw.index("--power-line") + 1] == "60", \
        "50 Hz anti-flicker under 60 Hz lights rolls bands through the picture"
    off = child_argv(Opts(source="hw", pool_cam_config=_config(power_line="off")))
    assert off[off.index("--power-line") + 1] == "off", "YAML `off` is False"
    devs = [hw[i + 1] for i, a in enumerate(hw) if a == "--device"]
    assert devs == ["NE=/dev/v4l/by-path/a", "SW=/dev/v4l/by-path/b"], devs

    flag = child_argv(Opts(source="hw", pool_cam_config=cfg, pool_cam=["X=3"]))
    devs = [flag[i + 1] for i, a in enumerate(flag) if a == "--device"]
    assert devs == ["X=3"], "--pool-cam must replace the file's list, not join it"

    demo = child_argv(Opts(source="demo", pool_cam_config=cfg))
    assert demo[demo.index("--fake") + 1] == str(DEMO_CAMERAS), \
        "--source demo opens no hardware: it must get synthetic cameras"
    assert "--device" not in demo

    assert load_config("/nonexistent/pool_cams.yaml")["devices"] == {}
    try:
        load_config(_config(devises={}))
        raise AssertionError("a misspelt key must be refused, not ignored")
    except ValueError:
        pass


def test_shipped_config_parses():
    from rov_gui.poolcam import load_config

    cfg = load_config(REPO / "config" / "pool_cams.yaml")
    assert cfg["fps"] == 30 and cfg["encoder"] == "auto" and cfg["quality"] is None
    assert cfg["slots"] == 4, "four pool corners: the tab must have four places"
    assert cfg["models"] is None      # = pool_cam.py POOL_MODELS
    assert cfg["power_line"] == 60


def test_fill_slots_puts_new_cameras_in_empty_slots_in_order():
    from pool_cam.pool_cam import POOL_MODELS, fill_slots, is_pool

    slots = [("cam0", "/dev/v4l/by-path/port4"), ("cam1", None), ("NE", None),
             ("cam3", None)]
    # Already-slotted camera is not new; two new ones take the first two
    # empty slots in order; a slot is never handed out twice.
    got = fill_slots(slots, ["/dev/v4l/by-path/port4", "/dev/v4l/by-path/port2",
                             "/dev/v4l/by-path/port7"])
    assert got == [(1, "/dev/v4l/by-path/port2"), (2, "/dev/v4l/by-path/port7")], got
    full = [("a", "/x/1"), ("b", "/x/2")]
    assert fill_slots(full, ["/x/3"]) == [], "no empty slot, no assignment"
    assert fill_slots([("a", None)], ["/x/1", "/x/1"]) == [(0, "/x/1")]
    # both models plugged in on 2026-09-30 count; another project's camera not
    assert is_pool("USB RGB Camera: USB RGB Camera")
    assert is_pool("SPCA2650 AV Camera: ")
    assert not is_pool("OAK-D-W"), POOL_MODELS


def test_controller_of_reads_the_pci_address_from_bus_info():
    from pool_cam.pool_cam import controller_of

    assert controller_of("usb-0000:13:00.0-3") == "0000:13:00.0"
    assert controller_of("usb-0000:79:00.4-1.2") == "0000:79:00.4"   # behind a hub
    assert controller_of("?") == "" and controller_of(None) == ""


def test_pc_port_of_sees_through_hubs():
    from pool_cam.pool_cam import pc_port_of

    assert pc_port_of("usb-0000:13:00.0-10") == ("0000:13:00.0", "10")
    assert pc_port_of("usb-0000:13:00.0-1.3") == ("0000:13:00.0", "1")    # hub on port 1
    assert pc_port_of("usb-0000:13:00.0-1.3.2") == ("0000:13:00.0", "1")  # two hubs
    assert pc_port_of("?") == ("", "")


def test_poolcam_is_isolated_from_control():
    """The cameras are CCTV. Their station-side code may not import anything
    that could put a frame or a number into the control path — not the bus,
    not the state types the workers exchange, not a backend, controller or
    perception module."""
    banned = ("rov_gui.bus", "rov_gui.state", "rov_gui.backends", "rov_gui.control",
              "rov_gui.perception", "rov_gui.window")
    files = {
        "rov_gui.poolcam": REPO / "rov_gui" / "poolcam.py",
        "rov_gui.widgets.poolcam": REPO / "rov_gui" / "widgets" / "poolcam.py",
        "pool_cam.pool_cam": REPO / "pool_cam" / "pool_cam.py",
        "pool_cam.protocol": REPO / "pool_cam" / "protocol.py",
    }
    for mod, path in files.items():
        pkg = mod.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1]
                                    + ([base] if base else []))
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            for name in names:
                assert not name.startswith(banned), f"{path.name} imports {name}"
    # ...and the window hands the client a log function, not the bus.
    src = (REPO / "rov_gui" / "window.py").read_text()
    assert "client_from_opts(self.opts, self.bus.log.emit)" in src


# =============================================================================
# the camera process, for real
# =============================================================================
def test_client_records_from_one_t0_and_leaves_out_a_disarmed_camera():
    _app()
    from rov_gui.poolcam import PoolCamClient, POOL_CAM_PY

    logs = []
    client = PoolCamClient([sys.executable, str(POOL_CAM_PY), "--serve", "--fake", "3",
                            "--size", "320x240", "--fps", "30", "--encoder", "x264"],
                           log=lambda lvl, msg: logs.append((lvl, msg)))
    out = Path(tempfile.mkdtemp(prefix="pool_rec_")) / "pool_x"
    try:
        assert client.start()
        assert _wait(lambda: client.snapshot()[0] is not None, 15), logs
        hello = client.snapshot()[0]
        assert [c["label"] for c in hello["cams"]] == ["cam0", "cam1", "cam2"]

        client.set_preview({"cam0": (160, 120)})
        assert _wait(lambda: client.take_frame("cam0", -1) is not None, 5), \
            "no preview arrived"
        seq, img = client.take_frame("cam0", -1)
        assert (img.width(), img.height()) == (160, 120)
        assert client.take_frame("cam1", -1) is None, \
            "a camera nobody asked to see must cost no preview"

        client.arm("cam2", False)
        _wait(lambda: all(c["connected"] for c in
                          (client.snapshot()[1] or {}).get("cams", [{}])), 5)
        t0 = time.monotonic()
        assert client.start_recording(t0, out, ui_video="ui_x.mp4", ui_started=t0)
        time.sleep(1.5)
        client.stop_recording()
    finally:
        client.stop()
    assert not client.alive

    session = json.loads((out / "session.json").read_text())
    assert session["recording"] == ["cam0", "cam1"] and session["left_out"] == ["cam2"]
    assert abs(session["t0_monotonic"] - t0) < 1e-5
    metas = {p.stem.split("_")[0]: json.loads(p.read_text())
             for p in out.glob("cam*.json")}
    assert set(metas) == {"cam0", "cam1"}, f"left-out camera recorded: {sorted(metas)}"
    for name, meta in metas.items():
        assert meta["complete"], (name, meta)
        assert meta["frames"] >= 30, (name, meta["frames"])
        # THE sync property: every file opens on the station's REC moment.
        assert abs(meta["t0_monotonic"] - t0) < 1e-5, (name, meta["t0_monotonic"], t0)
        assert (out / meta["video"]).stat().st_size > 0


def test_a_wedged_camera_process_never_blocks_the_caller():
    """The client is driven from the GUI thread, which also runs the 20 Hz
    command heartbeat. A child that stops reading its stdin fills the pipe
    (64 KiB) in a few hundred commands; past that a direct write would block
    the station. Commands must queue instead, and stop() must give up on time."""
    from rov_gui.poolcam import PoolCamClient

    logs = []
    client = PoolCamClient([sys.executable, "-c", "import time; time.sleep(60)"],
                           log=lambda lvl, msg: logs.append((lvl, msg)))
    assert client.start()
    try:
        t = time.monotonic()
        for i in range(3000):                         # ~600 KB of commands
            client.set_preview({f"cam{i}": (640 + i % 7, 360), "pad" * 20: (64, 64)})
        assert time.monotonic() - t < 1.0, "sending commands blocked the caller"
        t = time.monotonic()
        client.stop(timeout=1.0)
        assert time.monotonic() - t < 3.0, "stop() did not give up on time"
        assert any("still finishing" in m for _l, m in logs), logs
    finally:
        client._proc.kill()
        client._proc.wait()


def _fake_argv(n=2):
    from rov_gui.poolcam import POOL_CAM_PY
    return [sys.executable, str(POOL_CAM_PY), "--serve", "--fake", str(n),
            "--size", "320x240", "--fps", "30", "--encoder", "x264"]


def test_empty_slots_show_up_and_are_left_out_of_rec():
    """2 cameras, 4 slots: the tab gets four places, the two empty ones say
    so, and REC records the two that exist — session.json names the rest."""
    _app()
    from rov_gui.poolcam import PoolCamClient

    client = PoolCamClient(_fake_argv(2) + ["--slots", "4"], log=lambda *a: None)
    out = Path(tempfile.mkdtemp(prefix="pool_slots_")) / "pool_s"
    try:
        client.start()
        assert _wait(lambda: client.snapshot()[1] is not None, 15)
        hello, status = client.snapshot()[:2]
        assert [c["label"] for c in hello["cams"]] == ["cam0", "cam1", "cam2", "cam3"]
        empty = [c["label"] for c in status["cams"] if c["empty"]]
        assert empty == ["cam2", "cam3"], status["cams"]
        _wait(lambda: sum(c["connected"] for c in client.snapshot()[1]["cams"]) == 2, 5)
        client.start_recording(time.monotonic(), out)
        time.sleep(1.2)
        client.stop_recording()
    finally:
        client.stop()
    session = json.loads((out / "session.json").read_text())
    assert session["recording"] == ["cam0", "cam1"], session
    assert session["empty_slots"] == ["cam2", "cam3"], session
    assert sorted(p.stem.split("_")[0] for p in out.glob("cam*.json")) == ["cam0", "cam1"]


def test_a_station_killed_mid_recording_still_leaves_finished_files():
    """SIGKILL is the station dying with no cleanup at all. The child's stdin
    closes with it, and that alone must make it finish every file."""
    import signal
    import subprocess

    out = Path(tempfile.mkdtemp(prefix="pool_kill_")) / "pool_k"
    station = subprocess.Popen([sys.executable, "-c", f"""
import sys, time
sys.path.insert(0, {str(REPO)!r})
from rov_gui.poolcam import PoolCamClient
c = PoolCamClient({_fake_argv()!r}, log=lambda *a: None)
c.start()
while c.snapshot()[0] is None: time.sleep(0.05)
time.sleep(0.5)
c.start_recording(time.monotonic(), {str(out)!r})
time.sleep(1.5)
print(c._proc.pid, flush=True)
time.sleep(60)
"""], stdout=subprocess.PIPE, text=True)
    try:
        child = int(station.stdout.readline())
    finally:
        station.send_signal(signal.SIGKILL)
        station.wait()
    assert _wait(lambda: not Path(f"/proc/{child}").exists()
                 or Path(f"/proc/{child}/stat").read_text().split()[2] == "Z", 20), \
        "the camera process outlived its killed station"
    metas = [json.loads(p.read_text()) for p in out.glob("cam*.json")]
    assert len(metas) == 2, sorted(out.iterdir())
    assert all(m["complete"] and m["frames"] >= 30 for m in metas), metas


def test_restart_during_rec_resumes_recording_in_the_same_folder():
    import signal

    _app()
    from rov_gui.poolcam import PoolCamClient

    logs = []
    client = PoolCamClient(_fake_argv(), log=lambda lvl, msg: logs.append((lvl, msg)))
    out = Path(tempfile.mkdtemp(prefix="pool_restart_")) / "pool_r"
    try:
        client.start()
        assert _wait(lambda: client.snapshot()[0] is not None, 15)
        time.sleep(0.5)
        client.start_recording(time.monotonic(), out)
        time.sleep(1.0)
        client._proc.send_signal(signal.SIGKILL)             # a crash
        assert _wait(lambda: not client.alive, 5)
        assert _wait(lambda: any("NOT recording" in m for _l, m in logs), 5), logs
        assert client.recording, "REC UI is still on; the wish must survive the crash"
        client.start()                                       # RESTART
        assert _wait(lambda: len(list(out.glob("cam*.json"))) >= 4, 15), \
            sorted(p.name for p in out.iterdir())
        time.sleep(1.0)
        client.stop_recording()
    finally:
        client.stop()
    assert any("resumed" in m for _l, m in logs), logs


def test_a_garbled_stream_is_detected_not_mistaken_for_a_live_camera():
    """A child whose stdout is out of step is unreadable from then on. The
    reader must end it — alive-but-unheard would look like working cameras."""
    from rov_gui.poolcam import PoolCamClient

    logs = []
    client = PoolCamClient([sys.executable, "-c",
                            "import sys, time; sys.stdout.buffer.write(b'\\xff' * 16);"
                            " sys.stdout.flush(); time.sleep(60)"],
                           log=lambda lvl, msg: logs.append((lvl, msg)))
    client.start()
    assert _wait(lambda: not client.alive, 10), "the garbled child was left running"
    assert any("lost the camera process's stream" in m for _l, m in logs), logs


# =============================================================================
# the window
# =============================================================================
def test_window_without_the_flag_has_no_tab_and_spawns_nothing():
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    assert win.page_tabs is None and win.pages is None and win._pool_client is None
    win.shutdown()


def test_window_rec_ui_records_the_pool_cameras_into_the_run_folder():
    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    opts = Opts(pool_cams=True, pool_cam_config=_config())
    win = MainWindow(opts)
    try:
        tabs = win.page_tabs
        assert tabs is not None and tabs.count() == 2
        assert tabs.focusPolicy() == Qt.FocusPolicy.NoFocus, \
            "a tab bar that takes focus stops the pilot's keys"

        backend = make_backend("demo", win.bus, win.mailboxes, opts)
        win.resize(1366, 768)
        win.show()
        win.attach(backend)
        tabs.setCurrentIndex(1)
        assert win.pages.currentIndex() == 1
        assert _wait(lambda: len(win.poolcams.tiles) == 4, 20, app), \
            "the demo's four synthetic cameras never showed up"
        assert _wait(lambda: all(t.canvas.seq > 0 for t in win.poolcams.tiles.values()),
                     10, app), "tiles got no pictures while the tab was on screen"

        hint = win.minimumSizeHint()
        assert hint.width() <= 1366 and hint.height() <= 768, \
            f"--pool-cams broke the one-screen minimum: {hint.width()}x{hint.height()}"
        grabby = [w for w in [tabs, *win.poolcams.findChildren(QtWidgets.QWidget)]
                  if w.focusPolicy() != Qt.FocusPolicy.NoFocus]
        assert not grabby, f"widgets that can take the pilot's keyboard: {grabby}"

        win.poolcams.tiles["cam3"].arm_btn.click()           # AUTO REC OFF
        _pump(app, 600)
        win._toggle_record()                                 # REC UI on
        ui_path = win.recorder.stats.path
        assert ui_path is not None and win._pool_client.recording
        _pump(app, 2000)
        win._toggle_record()                                 # REC UI off
        assert not win._pool_client.recording
        win.estop()                     # an alarm brings DISARM & co. back
        assert tabs.currentIndex() == 0 and win.pages.currentIndex() == 0, \
            "E-STOP left the pilot looking at the pool cameras"
        _pump(app, 300)
    finally:
        win.shutdown()
    assert not win._pool_client.alive, "the camera process outlived the station"

    pool_dirs = list(ui_path.parent.glob("pool_*"))
    assert len(pool_dirs) == 1, f"expected one pool_* folder in {ui_path.parent}"
    session = json.loads((pool_dirs[0] / "session.json").read_text())
    assert session["ui_video"] == ui_path.name
    assert session["left_out"] == ["cam3"]
    ui_meta = json.loads(ui_path.with_suffix(".json").read_text())
    metas = {p.stem.split("_")[0]: json.loads(p.read_text())
             for p in pool_dirs[0].glob("cam*.json")}
    assert set(metas) == {"cam0", "cam1", "cam2"}, sorted(metas)
    for name, meta in metas.items():
        assert meta["complete"], (name, meta)
        assert abs(meta["t0_monotonic"] - ui_meta["started_monotonic"]) < 1e-5, \
            f"{name} did not start on the UI recording's t0"


def test_cctv_window_records_without_any_vehicle():
    """`./c3 cctv`: the pool cameras with no ROV and no C3. One REC press =
    one `_poolcam` run folder with the files, session.json and log.txt."""
    from rov_gui import runstore
    from rov_gui.cctv import CctvWindow, build_parser

    app = _app()
    theme.apply(app)
    root = tempfile.mkdtemp(prefix="cctv_")
    opts = build_parser().parse_args(["--demo", "--rec-dir", root,
                                      "--config", _config()])
    opts.source = "demo"
    win = CctvWindow(opts)
    try:
        assert win.rec_btn.focusPolicy() == Qt.FocusPolicy.NoFocus
        win.resize(1200, 800)
        win.show()
        win.start()
        assert _wait(lambda: len(win.panel.tiles) == 4 and all(
            t.canvas.seq > 0 for t in win.panel.tiles.values()), 20, app), \
            "the synthetic cameras never showed up"
        win.toggle_record()
        assert win.client.recording and win._rec_dir is not None
        _pump(app, 1500)
        win.toggle_record()
        assert not win.client.recording
    finally:
        win.shutdown()
    d = win._rec_dir
    assert runstore.leaf_kind(d.name) == "poolcam", d
    assert (d / "log.txt").read_text().count("recording to") == 1
    metas = [json.loads(p.read_text()) for p in d.glob("cam*.json")]
    assert len(metas) == 4 and all(m["complete"] for m in metas), metas
    assert json.loads((d / "session.json").read_text())["ui_video"] is None


def main() -> int:
    """Run without pytest."""
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
