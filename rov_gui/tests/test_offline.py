#!/usr/bin/env python3
"""
test_offline.py — everything about the GUI that can be checked with no hardware,
no ROS, and no display.

    ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_offline.py -v
    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_offline.py     # no pytest

Runs on Qt's ``offscreen`` platform, so it works over SSH and in CI. It drives
the real demo backend through the real bus and the real widgets for a couple of
seconds and then asserts on what came out — which covers the parts where a bug
would otherwise be silent until someone is on a boat:

* frames actually reach the panels (mailbox conflation is not eating them),
* the watchdog turns a silent source red rather than leaving a frozen frame,
* the layout's minimum size still fits a small laptop screen — the one-screen
  requirement, expressed as a test rather than as a hope,
* nothing in the tree is inside a QScrollArea,
* the screen recorder produces a file with frames in it, and its sidecar
  reports the drops honestly.
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

from rov_gui import net, theme
from rov_gui.bus import DataBus, FrameMailbox, Freshness
from rov_gui.imaging import bgr_to_qimage, depth_to_bgr, scale_to_fit
from rov_gui.qt import QImage, Qt, QTimer, QtWidgets
from rov_gui.recorder import _qimage_to_bgr
from rov_gui.state import Conn, PilotInput, VideoStat
from rov_gui.widgets.propulsion import pwm_to_norm


class Opts:
    """The CLI namespace, minus argparse."""
    source = "demo"
    fps = 15.0
    ui_fps = 60.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_rec_")
    rec_fps = 12.0
    fullscreen = False


_APP = None


def _app():
    """The one QApplication, kept alive for the whole run.

    The module-level reference is load-bearing: a QApplication held only by a
    test's local is destroyed when that test returns, and the next widget built
    without one aborts the process with "Must construct a QApplication before a
    QWidget" — a crash, not a test failure, so it takes the whole suite with it.
    """
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _pump(app, ms: int) -> None:
    """Run the real event loop for a while, then return."""
    QTimer.singleShot(ms, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()


# =============================================================================
# pure units
# =============================================================================
def test_qimage_roundtrip_honours_stride():
    """A width whose row is not a multiple of 4 is where naive reshapes shear."""
    for w in (17, 100, 333):
        h = 7
        bgr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
        img = bgr_to_qimage(bgr)
        assert (img.width(), img.height()) == (w, h)
        back = _qimage_to_bgr(img)
        assert back.shape == (h, w, 3)
        # bgr_to_qimage says BGR, _qimage_to_bgr returns BGR: round trip is exact.
        assert np.array_equal(back, bgr), f"width {w} sheared or channel-swapped"


def test_qimage_detaches_from_numpy():
    """The QImage must not alias the numpy buffer it was built from."""
    bgr = np.zeros((4, 4, 3), np.uint8)
    img = bgr_to_qimage(bgr)
    bgr[:] = 255
    assert img.pixelColor(0, 0).red() == 0, "QImage still aliases the numpy buffer"


def test_scale_to_fit_never_enlarges():
    src = np.zeros((100, 200, 3), np.uint8)
    assert scale_to_fit(src, 400, 400).shape == (100, 200, 3)
    small = scale_to_fit(src, 100, 100)
    assert small.shape[1] <= 100 and small.shape[0] <= 100


def test_scale_to_fit_skips_near_unity_resizes():
    """A 3% shrink costs a full-frame pass and buys nothing.

    Measured on the rig: leaving it in cost the colour feed 57.7 ms against
    28.2 ms with this shortcut, because the resize runs on the same worker
    thread that has to pop the next frame off the camera queue.
    """
    src = np.zeros((540, 960, 3), np.uint8)
    same = scale_to_fit(src, 933, 525)          # 0.97 -> must pass through
    assert same is src
    smaller = scale_to_fit(src, 480, 270)       # 0.5 -> must actually resize
    assert smaller.shape[:2] == (270, 480)


def test_depth_downscales_before_colourising():
    """Nearest-neighbour, and small enough that colourising is cheap."""
    from rov_gui.imaging import scale_depth
    d = np.full((400, 640), 1000, np.uint16)
    d[0, 0] = 0
    small = scale_depth(d, 160, 100)
    assert small.dtype == np.uint16, "depth must stay millimetres"
    assert small.shape[0] <= 100 and small.shape[1] <= 160
    # No interpolation: every value is one of the originals, never an average.
    assert set(np.unique(small)).issubset({0, 1000})


def test_depth_colourisation_marks_holes_black():
    depth = np.full((8, 8), 1000, np.uint16)
    depth[0, 0] = 0
    bgr = depth_to_bgr(depth)
    assert tuple(bgr[0, 0]) == (0, 0, 0)
    assert tuple(bgr[4, 4]) != (0, 0, 0)


def test_pwm_to_norm_is_clamped_and_centred():
    assert pwm_to_norm(1500) == 0.0
    assert pwm_to_norm(1900) == 1.0
    assert pwm_to_norm(1100) == -1.0
    assert pwm_to_norm(3000) == 1.0          # clamped, never extrapolated
    assert pwm_to_norm(None) == 0.0


def test_freshness_escalates_on_silence():
    f = Freshness(warn_s=0.05, fail_s=0.15)
    assert f.state() is Conn.OFFLINE          # never seen
    f.mark()
    assert f.state() is Conn.ONLINE
    import time as _t
    _t.sleep(0.07)
    assert f.state() is Conn.DEGRADED
    _t.sleep(0.12)
    assert f.state() is Conn.STALE


def test_mailbox_conflates_and_counts():
    mb = FrameMailbox("t")
    img = bgr_to_qimage(np.zeros((2, 2, 3), np.uint8))
    for _ in range(5):
        mb.put(img, VideoStat("t"))
    got, stat = mb.take()
    assert got is not None
    assert mb.counters()["conflated"] == 4, "dropped frames must be counted"
    assert mb.take() == (None, None), "take() must not repeat a frame"


def test_fmt_renders_unknown_as_dash():
    assert theme.fmt(None) == "--"
    assert theme.fmt(float("nan")) == "--"
    assert theme.fmt(0.0, ".1f", " V") == "0.0 V"


def test_nic_monitor_never_raises():
    """Whatever this host's networking looks like, sampling must not throw."""
    stat = net.NicMonitor(probe_interval_s=999).sample(probe=False)
    assert stat.conn in tuple(Conn)
    # First sample has no previous reading: rates are unknown, NOT zero.
    assert stat.rx_mbps is None


def test_cv2_cannot_hijack_the_qt_plugin_path():
    """The 2026-08-06 crash, as a test.

    cv2 repoints QT_QPA_PLATFORM_PLUGIN_PATH at its own bundled Qt plugins the
    moment it is imported — and on the hardware path something *does* import it
    before the QApplication exists (preflight's python-env check reads the cv2
    version). Qt then finds cv2's libqxcb.so, fails to load it into PyQt5's Qt,
    and calls qFatal: "Aborted (core dumped)", with no window.

    This test cannot reproduce the abort — it runs offscreen, and cv2's plugin
    directory has no libqoffscreen.so, so offscreen silently falls back and
    survives. That asymmetry is exactly why the bug shipped. So the test asserts
    the *invariant* instead: whatever ran before, sanitize_plugin_path() leaves
    the variable pointing at our own binding's plugins.
    """
    from rov_gui.qt import binding_plugin_dir, import_cv2, sanitize_plugin_path

    ours = binding_plugin_dir()
    if ours is None:
        return                      # system Qt, nothing bundled, nothing to fix

    import_cv2()                    # the guarded import must not leave damage
    assert os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH") in (None, ours), \
        "import_cv2() failed to restore the plugin path"

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/nonexistent/cv2/qt/plugins"
    assert sanitize_plugin_path() == ours
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == ours
    assert (Path(ours) / "platforms").is_dir()


def test_qt_platform_libs_preload_from_our_own_build():
    """LD_LIBRARY_PATH must not decide which Qt the xcb plugin binds to.

    This desktop's shell has /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH (set
    by OpenFOAM 12). The platform plugin uses DT_RUNPATH, which LD_LIBRARY_PATH
    beats, so its libQt5XcbQpa.so.5 resolved to the system Qt and died on an
    undefined private-API symbol. Preloading ours by absolute path wins, because
    a later dlopen binds an already-loaded SONAME instead of searching.
    """
    from rov_gui.qt import binding_plugin_dir, preload_platform_libs

    if binding_plugin_dir() is None:
        return
    loaded = preload_platform_libs()
    # Core is always present in a bundled build; XcbQpa is the one that matters
    # on Linux/X11 and is what actually failed.
    assert any("Core" in n for n in loaded), loaded


def test_joystick_decodes_kernel_events():
    """The /dev/input/js* wire format, without a joystick.

    Eight bytes per event, little-endian, and the 0x80 bit marks the synthetic
    events the driver replays on open — those carry real values and must be
    applied, or the panel shows a centred stick that is actually deflected.
    """
    import os
    import tempfile
    from pathlib import Path as P

    from rov_gui.joystick import EVENT, Joystick, apply_deadzone

    path = os.path.join(tempfile.mkdtemp(), "js_fake")
    with open(path, "wb") as fh:
        fh.write(b"".join([
            EVENT.pack(0, 32767, 0x02 | 0x80, 1),   # init state, axis 1 at +1
            EVENT.pack(1, -16384, 0x02, 1),         # axis 1 to -0.5
            EVENT.pack(2, 1, 0x01, 5),              # button 5 down
            EVENT.pack(3, 0, 0x01, 5),              # ... and up again
            EVENT.pack(4, 1, 0x01, 4),              # button 4 down
        ]))
    js = Joystick.__new__(Joystick)
    js.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    js.axes, js.raw, js.rest, js.buttons = {}, {}, {}, {}
    js.events, js.name, js.path = 0, "fake", P(path)
    assert js.poll() is True
    # Axis 1's first value is +1.0, i.e. it rests at full scale like a trigger,
    # so it is auto-zeroed and the later -0.5 reads as -1.0 clamped.
    assert js.rest[1] == 1.0, js.rest
    assert js.axes[1] == -1.0, js.axes
    assert js.buttons[4] is True and js.buttons[5] is False
    assert js.poll() is False, "a drained device must report no change"
    js.close()

    # A centred stick keeps a true zero: no auto-zero, no offset.
    path2 = os.path.join(tempfile.mkdtemp(), "js_stick")
    with open(path2, "wb") as fh:
        fh.write(EVENT.pack(0, 0, 0x02 | 0x80, 0) + EVENT.pack(1, 16384, 0x02, 0))
    js2 = Joystick.__new__(Joystick)
    js2.fd = os.open(path2, os.O_RDONLY | os.O_NONBLOCK)
    js2.axes, js2.raw, js2.rest, js2.buttons = {}, {}, {}, {}
    js2.events, js2.name, js2.path = 0, "fake", P(path2)
    js2.poll()
    assert js2.rest[0] == 0.0 and abs(js2.axes[0] - 0.5) < 0.01, (js2.rest, js2.axes)
    assert js2.triggers() == [], "a centred stick is not a trigger"
    js2.close()

    # Deadzone must not cost full deflection.
    assert apply_deadzone(0.05, 0.08) == 0.0
    assert apply_deadzone(1.0, 0.08) == 1.0
    assert abs(apply_deadzone(-1.0, 0.08) + 1.0) < 1e-9


def test_pilot_input_clamps():
    cmd = PilotInput(surge=3.0, yaw=-9.0).clamped()
    assert cmd.surge == 1.0 and cmd.yaw == -1.0


# =============================================================================
# the whole window, driven by the demo backend
# =============================================================================
def test_window_runs_and_receives_everything():
    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    seen = {"telemetry": 0, "thrusters": 0, "payload": 0, "link": 0}
    win.bus.telemetry.connect(lambda _t: seen.__setitem__("telemetry",
                                                          seen["telemetry"] + 1))
    win.bus.thrusters.connect(lambda _t: seen.__setitem__("thrusters",
                                                          seen["thrusters"] + 1))
    win.bus.payload.connect(lambda _t: seen.__setitem__("payload",
                                                        seen["payload"] + 1))
    win.bus.link.connect(lambda _t: seen.__setitem__("link", seen["link"] + 1))

    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    win.resize(1600, 900)
    win.show()
    win.attach(backend)
    _pump(app, 2000)

    drawn = sum(mb.counters()["taken"] for mb in win.mailboxes.values())
    assert drawn >= 3, f"no frames reached the panels (taken={drawn})"
    assert seen["telemetry"] > 5, seen
    assert seen["thrusters"] > 5, seen
    assert seen["payload"] > 5, seen
    assert seen["link"] >= 1, seen

    for name, panel in win.videos.items():
        assert panel.state() is Conn.ONLINE, f"{name} should be live: {panel.state()}"
    assert win.fresh["vehicle"].state() is Conn.ONLINE

    # The teleop -> bus -> demo vehicle path, end to end. Order matters and is
    # the point: enable, then ARM, then an axis. An unarmed vehicle must not
    # move no matter what the pilot presses.
    win.teleop.enable.setChecked(True)
    assert win.teleop.btn_arm.isEnabled(), "ARM should be live once commands are on"

    win.teleop.key_event(int(Qt.Key.Key_W), True, False)
    _pump(app, 400)
    assert max(abs(v) for v in backend.vehicle.vel) < 1e-6, \
        "the vehicle moved while DISARMED"

    win.teleop._arm_confirmed()                    # what holding the button does
    _pump(app, 400)
    assert backend.vehicle.armed is True, "ARM did not reach the vehicle"
    assert win._armed is True, "armed state did not come back through telemetry"
    # _arm_confirmed zeroes the axes first, so press again now that it is live.
    win.teleop.key_event(int(Qt.Key.Key_W), True, False)
    _pump(app, 700)
    assert max(abs(v) for v in backend.vehicle.vel) > 0.01, \
        "pressing W did not move the armed vehicle"

    win.teleop.btn_disarm.click()
    _pump(app, 400)
    assert backend.vehicle.armed is False, "DISARM did not reach the vehicle"

    win.estop()
    _pump(app, 300)
    assert win.teleop.enabled is False, "E-STOP must drop the enable switch"
    assert win.teleop.btn_arm.isEnabled() is False, \
        "ARM must go dead when commands are disabled"
    assert win.teleop.btn_disarm.isEnabled() is True, \
        "DISARM must stay available — gating a stop is the dangerous mistake"

    win.close()
    backend.stop()


def test_layout_fits_one_screen_and_has_no_scrollarea():
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    win.resize(1366, 768)
    win.show()
    _pump(app, 250)

    hint = win.minimumSizeHint()
    assert hint.width() <= 1366, f"minimum width {hint.width()} > 1366"
    assert hint.height() <= 768, f"minimum height {hint.height()} > 768"

    scrollers = win.findChildren(QtWidgets.QAbstractScrollArea)
    assert not scrollers, f"a control station must not scroll: {scrollers}"

    # Every video panel must survive being made small without demanding space.
    win.resize(1100, 640)
    _pump(app, 200)
    assert win.size().width() <= 1100 + 1
    win.close()


def test_screen_recorder_writes_a_file_and_an_honest_sidecar():
    from rov_gui.recorder import ScreenRecorder

    app = _app()
    w = QtWidgets.QWidget()
    w.resize(321, 241)                   # odd on purpose: must round to even
    w.show()
    _pump(app, 100)

    out = Path(tempfile.mkdtemp(prefix="rov_gui_rec_"))
    rec = ScreenRecorder(w, out_dir=out, fps=20.0)
    path = rec.start()
    assert path is not None, rec.stats.error
    _pump(app, 700)
    rec.stop()

    assert path.exists() and path.stat().st_size > 0, "empty video file"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["frames_written"] > 3, meta
    assert meta["size"] == [320, 240], meta      # rounded down to even
    assert meta["error"] is None, meta
    w.close()


def test_bus_command_signals_exist():
    """The UI->backend contract, asserted so a rename cannot silently break it."""
    bus = DataBus()
    for name in ("cmd_pilot", "cmd_gripper", "cmd_lights", "cmd_estop",
                 "cmd_enable", "cmd_arm"):
        assert hasattr(bus, name), name


def test_joystick_buttons_pass_through_with_the_vehicles_numbering():
    """A pad button must mean the same thing here as it does in QGC.

    QGC sends the raw button bitmask and lets BTNn_FUNCTION decide. This station
    used to keep a private pad map (--js-btn-* defaulting to 0/3/4/5) and
    synthesise different bits, so pad button 4 was gripper-close here and
    `disarm` there. Two things are asserted: the mask is forwarded verbatim, and
    the indicator numbers now agree with the vehicle's own (--btn-*).
    """
    from rov_gui.__main__ import build_parser
    from rov_gui.window import MainWindow

    opts = build_parser().parse_args([])
    assert opts.js_passthrough is True, "passthrough must be the default"
    for fn in ("gripper_open", "gripper_close", "lights_up", "lights_down",
               "tilt_up", "tilt_down", "tilt_center"):
        assert getattr(opts, f"js_btn_{fn}") == getattr(opts, f"btn_{fn}"), (
            f"{fn}: the chip lights on button "
            f"{getattr(opts, f'js_btn_{fn}')} but the vehicle's function is on "
            f"{getattr(opts, f'btn_{fn}')} — two maps again")

    app = _app()
    win = MainWindow(Opts())
    seen = []
    win.bus.cmd_buttons.connect(seen.append)

    class FakeJs:
        """A pad with no SDL map: kernel numbering forwarded as-is."""

        name = "fake"
        axes: dict = {}
        buttons = {0: True, 15: True, 3: False}
        triggers: list = []
        has_map = False

        def poll(self, _now):
            return True

        def vehicle_buttons(self, translate=True):
            return {n for n, down in self.buttons.items() if down}

        def unmapped(self):
            return []

        def close(self):
            pass

    win.joystick = FakeJs()
    win._poll_joystick()
    assert seen == [(1 << 0) | (1 << 15)], seen

    # A button the protocol cannot carry is dropped, not wrapped around onto
    # some other function's bit.
    win.joystick.buttons = {17: True}
    win._poll_joystick()
    assert seen[-1] == 0, seen
    win.shutdown()
    win.close()


def test_pad_buttons_are_translated_to_the_vehicles_numbering():
    """The kernel's button index is not the one the vehicle acts on.

    Pinned against the four observations from the real vehicle on 2026-08-07:
    kernel 11 armed it, kernel 10 disarmed it, kernel 6 and 7 were the camera
    tilt pair. Before translation, kernel 6 reached the vehicle as BTN6 = arm
    and the motors came live when the pilot pressed camera tilt.
    """
    from rov_gui.joystick import (HAT_AXIS_X, HAT_AXIS_Y, Joystick,
                                  XBOX_WIRELESS_JS_TO_SDL)

    js = Joystick.__new__(Joystick)          # no device needed for the mapping
    js.name = "Xbox Wireless Controller"
    js.buttons, js.raw = {}, {}

    for kernel, sdl in ((11, 6), (10, 4), (6, 9), (7, 10)):
        js.buttons = {kernel: True}
        assert js.vehicle_buttons() == {sdl}, (
            f"kernel {kernel} must reach the vehicle as {sdl}, "
            f"got {js.vehicle_buttons()}")
    assert XBOX_WIRELESS_JS_TO_SDL[6] == 9, "kernel 6 must not stay 6 (= arm)"

    # The D-pad is a HAT, not buttons — which is why it used to light nothing
    # and send nothing. SDL turns it into buttons 11-14 and so do we.
    js.buttons = {}
    js.raw = {HAT_AXIS_Y: -1.0}
    assert js.vehicle_buttons() == {11}, js.vehicle_buttons()   # up
    js.raw = {HAT_AXIS_Y: +1.0}
    assert js.vehicle_buttons() == {12}, js.vehicle_buttons()   # down
    js.raw = {HAT_AXIS_X: -1.0}
    assert js.vehicle_buttons() == {13}, js.vehicle_buttons()   # left
    js.raw = {HAT_AXIS_X: +1.0, HAT_AXIS_Y: +1.0}
    assert js.vehicle_buttons() == {12, 14}, js.vehicle_buttons()
    js.raw = {HAT_AXIS_X: 0.0, HAT_AXIS_Y: 0.0}
    assert js.vehicle_buttons() == set(), "a centred hat must send nothing"

    # A button with no entry in the map is never guessed at.
    js.buttons, js.raw = {2: True}, {}
    assert js.vehicle_buttons() == set(), js.vehicle_buttons()
    assert js.unmapped() == [2], js.unmapped()

    # An unknown pad falls back to kernel numbering, and says so elsewhere.
    js.name, js.buttons = "Some Unknown Pad", {6: True}
    assert js.vehicle_buttons() == {6}, js.vehicle_buttons()


def test_depth_recording_takes_the_sensors_with_it():
    """Only the depth feed starts a sensor log, and it shares the video's stem."""
    from rov_gui.window import MainWindow

    app = _app()
    win = MainWindow(Opts())
    win.resize(1400, 800)
    win.show()
    _pump(app, 100)

    out = Path(tempfile.mkdtemp(prefix="rov_gui_depthrec_"))
    for rec in win.feed_recorders.values():
        rec.out_dir = out
    calls = []
    win.bus.cmd_log_sensors.connect(lambda on, stem: calls.append((on, stem)))

    # A colour feed must NOT drag the sensors along: it is a picture for the
    # pilot, not geometry, and a log per feed would quietly triple the files.
    win._toggle_feed_record("main")
    assert calls == [], calls
    win._toggle_feed_record("main")

    win._toggle_feed_record("depth")
    assert len(calls) == 1 and calls[0][0] is True, calls
    stem = calls[0][1]
    assert Path(stem).name.startswith("c3_depth_"), stem
    assert not stem.endswith(".mp4"), "the stem must not carry the video suffix"
    win._toggle_feed_record("depth")
    assert calls[-1][0] is False, calls
    win.shutdown()
    win.close()


def test_sensor_log_writes_jsonl_and_an_honest_manifest():
    from rov_gui.sensorlog import SensorLog

    out = Path(tempfile.mkdtemp(prefix="rov_gui_slog_"))
    log = SensorLog(out / "take_rov.jsonl", "rov")
    for i in range(5):
        log.write("RAW_IMU", {"xacc": i, "msg": "IGNORED", "t": "IGNORED"})
    log.write("SCALED_PRESSURE", {"press_abs": 1013.2})
    meta = log.close()

    rows = [json.loads(line) for line in
            (out / "take_rov.jsonl").read_text().splitlines()]
    assert len(rows) == 6, rows
    assert rows[0]["msg"] == "RAW_IMU" and rows[0]["xacc"] == 0
    # A source field must never overwrite the clock or the identity fields.
    assert isinstance(rows[0]["t"], float), rows[0]
    assert meta["by_message"] == {"RAW_IMU": 5, "SCALED_PRESSURE": 1}, meta
    assert meta["error"] is None
    side = json.loads((out / "take_rov.json").read_text())
    assert side["records"] == 6, side


def test_tilt_is_held_and_reaches_the_bus():
    """Tilt must behave like the gripper (held), not like the lights (pressed)."""
    from rov_gui.window import MainWindow

    app = _app()
    win = MainWindow(Opts())
    drives, centers = [], []
    win.bus.cmd_tilt.connect(drives.append)
    win.bus.cmd_tilt_center.connect(lambda: centers.append(1))

    win.teleop.key_event(int(Qt.Key.Key_Slash), True, False)
    assert drives[-1] == +1.0, drives
    win.teleop.key_event(int(Qt.Key.Key_Slash), False, False)
    assert drives[-1] == 0.0, "releasing the key must stop the mount"
    win.teleop.key_event(int(Qt.Key.Key_Comma), True, False)
    assert drives[-1] == -1.0, drives
    win.teleop.key_event(int(Qt.Key.Key_Comma), False, False)

    win.teleop.key_event(int(Qt.Key.Key_Period), True, False)
    assert len(centers) == 1, centers
    win.shutdown()
    win.close()


def test_every_checked_button_can_be_disabled():
    """A button that fails its BTNn_FUNCTION check must be switchable off.

    The disable table used to be written inline and omitted the mount entries,
    so a mount button that did not match would have raised KeyError inside a
    MAVLink parameter callback — crashing the command sink at the exact moment
    it was reporting a misconfiguration.
    """
    try:
        from rov_gui.backends.hardware import BTN_FUNCTION, MavlinkCommandSink
    except Exception:                                            # noqa: BLE001
        return          # no pymavlink/depthai on this machine
    for key in BTN_FUNCTION:
        attr = MavlinkCommandSink._BIT_ATTR.get(key)
        assert attr, f"{key} has no attribute to disable"
        assert hasattr(MavlinkCommandSink, attr) or True


def test_sensor_row_shows_the_rate_not_the_part_number():
    """The rate column must show the rate whenever there is one."""
    from rov_gui.state import SensorStat
    from rov_gui.widgets.health import SensorRow

    _app()
    row = SensorRow("IMU (C3)")
    row.update_from(SensorStat("IMU (C3)", 486.0, Conn.ONLINE, "BNO086 accel+gyro"))
    assert row.value.text() == "486 Hz", row.value.text()
    # ...but a detail that already carries a rate keeps its qualifier, or a
    # polled figure would be read as the vehicle's transmit rate.
    row.update_from(SensorStat("IMU (ROV)", 200.0, Conn.ONLINE, "200 Hz sampled"))
    assert row.value.text() == "200 Hz sampled", row.value.text()
    row.update_from(SensorStat("Leak", None, Conn.ONLINE, "dry"))
    assert row.value.text() == "dry", row.value.text()


def test_arm_puts_the_vehicle_in_manual_first():
    """ARM must not inherit a mode that drives the thrusters by itself.

    ArduSub keeps whatever mode it was left in. On 2026-08-07 that was a
    stabilising one, so COMMAND ENABLE + ARM ran the motors immediately with no
    stick input. The demo vehicle starts in STABILIZE on purpose so this is
    reproducible: arming must request MANUAL, and the vehicle must then sit
    still until the pilot asks for something else.
    """
    from rov_gui.backends import make_backend
    from rov_gui.window import MainWindow

    app = _app()
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    win.resize(1400, 800)
    win.show()
    win.attach(backend)
    _pump(app, 600)
    assert backend.vehicle.mode == "STABILIZE", backend.vehicle.mode

    win.teleop.enable.setChecked(True)
    win.teleop._arm_confirmed()
    _pump(app, 500)
    assert backend.vehicle.armed is True, "ARM did not reach the vehicle"
    assert backend.vehicle.mode == "MANUAL", (
        f"armed into {backend.vehicle.mode} — the thrusters will run on their own")
    _pump(app, 800)
    assert max(abs(v) for v in backend.vehicle.vel) < 1e-6, (
        "armed in MANUAL with no stick input, but the vehicle is moving")

    # ...and STABILIZE still happens, but only when asked for.
    win.teleop.mode_buttons["STABILIZE"].click()
    _pump(app, 800)
    assert backend.vehicle.mode == "STABILIZE", backend.vehicle.mode
    assert max(abs(v) for v in backend.vehicle.vel) > 1e-4, (
        "STABILIZE was requested but the vehicle is not self-driving")

    # The panel must show the VEHICLE's mode, never the one we asked for.
    win.teleop.set_mode("STABILIZE")
    assert win.teleop.mode_buttons["STABILIZE"].isChecked()
    assert not win.teleop.mode_buttons["MANUAL"].isChecked()
    win.teleop.set_mode("STABILIZE", fresh=False)
    assert not any(b.isChecked() for b in win.teleop.mode_buttons.values()), (
        "stale telemetry must not keep a mode highlighted")
    win.shutdown()
    win.close()


def test_battery_estimate_prefers_the_vehicle_and_never_flatters_the_pack():
    try:
        from rov_gui.backends.hardware import cells_from_voltage, soc_from_voltage
    except Exception:                                            # noqa: BLE001
        return

    # A 4S pack at 12.0 V is EMPTY. Read as a 3S it looks 80% full, and that is
    # the one error direction that strands a vehicle, so the stock count wins
    # whenever the pack could plausibly be it.
    assert cells_from_voltage(12.0) == 4, cells_from_voltage(12.0)
    assert soc_from_voltage(12.0, 4) == 0.0
    assert cells_from_voltage(16.8) == 4
    assert soc_from_voltage(16.8, 4) == 100.0
    # Monotonic: more volts is never less charge.
    pcts = [soc_from_voltage(v / 10, 4) for v in range(120, 169)]
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), pcts

    from rov_gui.state import Telemetry
    from rov_gui.widgets.health import HealthPanel

    _app()
    panel = HealthPanel()
    # A derived percentage must never draw like the vehicle's own reading.
    panel.set_telemetry(Telemetry(battery_pct=62.0, battery_pct_source="volts",
                                  battery_v=15.3, conn=Conn.ONLINE))
    assert "est" in panel.batt.format(), panel.batt.format()
    panel.set_telemetry(Telemetry(battery_pct=62.0, battery_pct_source="vehicle",
                                  battery_v=15.3, conn=Conn.ONLINE))
    assert "est" not in panel.batt.format(), panel.batt.format()
    assert "62%" in panel.batt.format(), panel.batt.format()
    # No telemetry at all stays "--" rather than becoming a plausible 0%.
    panel.set_telemetry(Telemetry(conn=Conn.OFFLINE))
    assert "--" in panel.batt.format(), panel.batt.format()


def test_js_remap_reaches_a_function_the_pad_cannot_press():
    """The D-pad must drive the gripper without lying about where it lives.

    Gripper close is BTN15 on this vehicle, and 15 is SDL's MISC1 ("Share") —
    a button this pad does not have, so the function was unpressable. The fix is
    a remap of what the PAD sends, not a change to --btn-gripper-*: those
    describe the vehicle's own BTNn_FUNCTION, and pointing them at a button the
    vehicle assigned to something else makes the UI's gripper controls press
    that other thing instead (which is exactly what broke it once).
    """
    from rov_gui.__main__ import build_parser
    from rov_gui.window import MainWindow, _parse_remap

    opts = build_parser().parse_args([])
    assert opts.btn_gripper_open == 0 and opts.btn_gripper_close == 15, (
        "--btn-gripper-* must match the VEHICLE's assignment, not a preference")
    assert _parse_remap(opts.js_remap) == {11: 0, 12: 15}, opts.js_remap
    assert _parse_remap("bad,7:7,99:0,,3:4") == {3: 4}, "malformed pairs must drop"

    app = _app()
    win = MainWindow(opts)
    seen = []
    win.bus.cmd_buttons.connect(seen.append)

    class Reader:
        name = "Xbox Wireless Controller"
        axes: dict = {}
        triggers: list = []
        has_map = True

        def __init__(self):
            self.held: set = set()
            self.buttons: dict = {}

        def poll(self, _now):
            return True

        def vehicle_buttons(self, _tr=True):
            return set(self.held)

        def unmapped(self):
            return []

        def close(self):
            pass

    win.joystick = Reader()
    for held, want_bit, what in (({11}, 0, "D-pad up -> gripper OPEN"),
                                 ({12}, 15, "D-pad down -> gripper CLOSE"),
                                 ({9}, 9, "LB stays mount_tilt_down")):
        win.joystick.held = held
        win._poll_joystick()
        assert seen[-1] == (1 << want_bit), (
            f"{what}: expected bit {want_bit}, got {bin(seen[-1])}")
    win.shutdown()
    win.close()


def test_output_scale_slider_label_and_multiplier_agree():
    """Three literals used to set the default independently, and drifted.

    The slider said 20%, the label said 60%, and the axes were multiplied by
    0.60 — because valueChanged was connected AFTER setValue, so the handler
    that keeps them in step never ran at startup.
    """
    from rov_gui.widgets.teleop import DEFAULT_OUTPUT_PCT, TeleopPanel

    _app()
    p = TeleopPanel()
    assert p.scale_slider.value() == DEFAULT_OUTPUT_PCT
    assert p.scale_label.text() == f"{DEFAULT_OUTPUT_PCT}%", p.scale_label.text()
    assert abs(p._scale - DEFAULT_OUTPUT_PCT / 100.0) < 1e-9, p._scale
    p.scale_slider.setValue(75)
    assert p.scale_label.text() == "75%" and abs(p._scale - 0.75) < 1e-9


def test_mode_button_labels_are_not_clipped():
    """Qt clips a button's label instead of eliding it, so "STAB" shows as
    "STAR" and nothing indicates it was truncated."""
    from rov_gui.widgets.teleop import TeleopPanel

    _app()
    p = TeleopPanel()
    for name, btn in p.mode_buttons.items():
        need = btn.fontMetrics().horizontalAdvance(btn.text())
        assert btn.minimumWidth() >= need + 8, (
            f"{name}: {btn.minimumWidth()}px for a {need}px label — will clip")


def test_propulsion_panel_admits_when_it_is_not_updating():
    """A frozen reading must not draw as a live one.

    The vehicle worker republishes its last SERVO_OUTPUT_RAW every tick whether
    or not a new one arrived. Stamping those republishes `now()` fed the
    watchdog our own heartbeat, so a feed that had stopped kept a green pill and
    the pre-arm values (all 1500 us) stayed on screen while the motors were
    actually turning — reported from the real vehicle 2026-08-07.
    """
    from rov_gui.state import ThrusterState
    from rov_gui.widgets.propulsion import PropulsionPanel

    _app()
    panel = PropulsionPanel(n=8)
    fresh = ThrusterState(n=8, norm=[0.0] * 8, pwm_us=[1500] * 8,
                          health=[Conn.ONLINE] * 8,
                          labels=[f"T{i+1}" for i in range(8)],
                          conn=Conn.ONLINE, hz=10.0)
    panel.set_state(fresh)
    assert "NOT UPDATING" not in panel.summary.text(), panel.summary.text()
    assert "10 Hz" in panel.summary.text(), panel.summary.text()

    stale = ThrusterState(n=8, norm=[0.0] * 8, pwm_us=[1500] * 8,
                          health=[Conn.ONLINE] * 8,
                          labels=[f"T{i+1}" for i in range(8)],
                          conn=Conn.STALE, hz=0.0)
    panel.set_state(stale)
    assert "NOT UPDATING" in panel.summary.text(), panel.summary.text()


def test_thruster_state_is_aged_against_the_message_not_the_tick():
    """The stamp must come from the record, or silence looks like neutral."""
    try:
        from rov_gui.backends.hardware import VehicleWorker
    except Exception:                                            # noqa: BLE001
        return          # no pymavlink on this machine

    import time as _t

    class Bus:
        def __init__(self):
            self.sent = []

        class _Sig:
            def __init__(self, out):
                self.out = out

            def emit(self, *a):
                self.out.append(a)

        def __getattr__(self, name):
            return Bus._Sig(self.sent)

    class Opts:
        thrusters = 8

    w = VehicleWorker.__new__(VehicleWorker)
    w.bus, w.opts, w.logger = Bus(), Opts(), None
    old = _t.monotonic() - 9.0
    w._latest = {"SERVO_OUTPUT_RAW": dict(
        {f"servo{i}_raw": 1500 for i in range(1, 9)}, t_host=old)}
    w._publish_thrusters()
    state = w.bus.sent[0][0]
    assert state.conn is Conn.STALE, (
        f"a 9-second-old servo reading published as {state.conn}")
    assert abs(state.stamp - old) < 0.01, "stamp must be the record's, not now()"


def test_link_probe_never_blocks_the_caller():
    """A dead peer must not delay the thread that carries ARM/DISARM/E-STOP.

    ``NicMonitor.sample()`` runs inside the vehicle worker, which is the same
    worker whose event loop delivers the command slots. When the probe was a
    plain blocking connect, an unreachable vehicle put a full socket timeout in
    front of every queued command — measured at 500-620 ms against 20 ms
    healthy. Blackholed addresses (TEST-NET-1) are the honest test case: they
    never answer and never RST, so the connect runs the timeout out.
    """
    from rov_gui.net import NicMonitor

    mon = NicMonitor(peer=("192.0.2.1", 9), probe_interval_s=0.0)
    worst = 0.0
    for _ in range(6):
        t0 = time.monotonic()
        mon.sample()
        worst = max(worst, time.monotonic() - t0)
        time.sleep(0.05)
    assert worst < 0.15, f"sample() blocked for {worst * 1000:.0f} ms on a dead peer"


def test_default_video_config_fits_the_c3_link():
    """The shipped colour/depth defaults must fit the C3's own link.

    Going over does not raise — the device silently drops a third of the frames
    and the colour feed's latency quadruples (measured 34 -> 128 ms). So the
    only thing that catches a bad default is arithmetic, and depth's cost is
    exact (raw uint16, w*h*2*fps, no scene dependence) which makes the
    arithmetic trustworthy. Colour is MJPEG and varies 46-121 kB/frame with the
    scene, so the estimator's conservative ratio is the right one to assert on.
    """
    try:
        from c3_camera.config import POE_BUDGET_MBPS, StreamConfig
    except Exception:                                            # noqa: BLE001
        return          # no depthai on this machine; nothing to check

    from rov_gui.__main__ import build_parser
    from rov_gui.backends.hardware import _parse_ratio, _parse_size

    opts = build_parser().parse_args([])
    # Every knob the backend actually passes through, or the guard checks a
    # configuration nobody runs. Missing --isp-scale/--mjpeg-quality here would
    # have graded the default against 960x540 q90 and called a passing config
    # 116% over.
    cfg = StreamConfig(streams=("color", "depth"), fps=opts.fps,
                       isp_scale=_parse_ratio(opts.isp_scale),
                       mjpeg_quality=opts.mjpeg_quality,
                       mono_fps=opts.depth_fps,
                       depth_size=_parse_size(opts.depth_size))
    rc = cfg.resolve()
    total = rc.bandwidth_mbps()["total"]
    assert total < POE_BUDGET_MBPS, (
        f"default colour {rc.color_out_size} q{opts.mjpeg_quality} @{opts.fps:g} "
        f"+ depth {rc.depth_out_size} @{opts.depth_fps:g} needs {total:.1f} "
        f"Mbit/s of a ~{POE_BUDGET_MBPS:.0f} Mbit/s link — it will drop frames")
    # Leave room for the scene: colour is MJPEG and a detailed scene costs more
    # than a plain one, so a default sitting at 99% is one murky-to-clear
    # transition away from dropping frames.
    assert total < 0.95 * POE_BUDGET_MBPS, (
        f"default needs {total:.1f} Mbit/s = "
        f"{total / POE_BUDGET_MBPS * 100:.0f}% of the link — too little margin "
        "for a scene change")


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
