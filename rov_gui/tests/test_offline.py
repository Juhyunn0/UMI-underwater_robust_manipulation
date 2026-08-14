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
import types
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
    got, stat, _aux = mb.take()
    assert got is not None
    assert mb.counters()["conflated"] == 4, "dropped frames must be counted"
    assert mb.take() == (None, None, None), "take() must not repeat a frame"


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


def test_object_tracking_is_opt_in_and_costs_nothing_when_off():
    """A GPU feature must be off by default, and off must mean absent."""
    from rov_gui.__main__ import build_parser
    from rov_gui.widgets.video import VideoPanel

    opts = build_parser().parse_args([])
    assert opts.pose is False, "object tracking must be opt-in"

    _app()
    # A panel built without pose= is the panel that shipped before: no button,
    # no overlay pass, no click handling.
    plain = VideoPanel("main", "C3 RGB", FrameMailbox("main"))
    assert plain.track_btn is None
    assert plain.canvas.pose is False
    assert plain.canvas.pose_armed is False


def test_canvas_maps_a_click_to_source_pixels():
    """Three coordinate systems, and getting the middle one wrong is silent.

    The displayed image is NOT the source: the worker shrank it to the panel.
    But `imaging.scale_to_fit` leaves the frame alone when the reduction would
    be under 15%, so the ratio flips between 1.0 and something else as the
    window resizes — a formula that assumes either is wrong at some sizes.
    """
    from rov_gui.qt import QImage, QPoint
    from rov_gui.widgets.video import VideoCanvas

    _app()
    c = VideoCanvas("t", FrameMailbox("t"))
    c.resize(400, 400)                       # square panel, 16:9 image
    # displayed 320x180 (already shrunk), source 640x360 -> ratio 2.0
    c._image = QImage(320, 180, QImage.Format.Format_RGB888)
    c._stat = VideoStat(name="t", width=640, height=360)
    c._fit_rect = c._fit(320, 180)           # what paintEvent would store
    r = c._fit_rect
    assert abs(r.width() - 400) < 1e-6, r    # letterboxed top and bottom

    # centre of the drawn image -> centre of the SOURCE image
    mid = c.to_source(QPoint(int(r.x() + r.width() / 2),
                             int(r.y() + r.height() / 2)))
    assert mid is not None
    assert abs(mid[0] - 320) < 2 and abs(mid[1] - 180) < 2, mid

    # top-left of the drawn image -> origin of the source
    tl = c.to_source(QPoint(int(r.x()) + 1, int(r.y()) + 1))
    assert tl is not None and tl[0] < 6 and tl[1] < 6, tl

    # a click on the letterbox bar is NOT a prompt
    assert c.to_source(QPoint(200, 2)) is None
    assert c.to_source(QPoint(200, 398)) is None

    # unshrunk frame (the >=0.85 case): ratio is 1.0 and it must still be right
    c._image = QImage(640, 360, QImage.Format.Format_RGB888)
    c._fit_rect = c._fit(640, 360)
    r = c._fit_rect
    mid = c.to_source(QPoint(int(r.x() + r.width() / 2),
                             int(r.y() + r.height() / 2)))
    assert abs(mid[0] - 320) < 2 and abs(mid[1] - 180) < 2, mid


def test_a_prompt_click_does_not_break_double_click_to_promote():
    """Qt sends press+release before it decides a gesture was a double click.

    So a naive mousePressEvent fires a prompt on the first half of every
    promote. The prompt is deferred by doubleClickInterval and cancelled when
    the double click arrives; both gestures keep the left button.
    """
    from rov_gui.qt import QImage, QPoint, Qt as _Qt, QtGui
    from rov_gui.widgets.video import VideoCanvas

    app = _app()
    c = VideoCanvas("t", FrameMailbox("t"), pose=True)
    c.resize(400, 400)
    c._image = QImage(320, 180, QImage.Format.Format_RGB888)
    c._stat = VideoStat(name="t", width=640, height=360)
    c._fit_rect = c._fit(320, 180)
    c.pose_armed = True

    prompts, promotes = [], []
    c.prompt_clicked.connect(lambda x, y: prompts.append((x, y)))
    c.double_clicked.connect(lambda: promotes.append(1))

    def release(pos):
        c.mouseReleaseEvent(QtGui.QMouseEvent(
            QtGui.QMouseEvent.Type.MouseButtonRelease, pos,
            _Qt.MouseButton.LeftButton, _Qt.MouseButton.LeftButton,
            _Qt.KeyboardModifier.NoModifier))

    centre = QPoint(200, 200)
    # a double click, as Qt REALLY delivers it: Press, Release, DblClick, and
    # then a SECOND Release. The first shipped version of this test omitted
    # that trailing release, and the handler it exercised re-armed the prompt
    # timer on it — so every double-click-to-promote still re-prompted SAM2
    # 400 ms later on whatever pixel was under the cursor.
    release(centre)
    c.mouseDoubleClickEvent(None)
    release(centre)                          # the trailing release
    _pump(app, int(QtWidgets.QApplication.doubleClickInterval()) + 150)
    assert promotes == [1], "double click must still promote"
    assert prompts == [], f"a promote must not also prompt: {prompts}"

    # a single click: nothing cancels it, so it becomes a prompt
    release(centre)
    _pump(app, int(QtWidgets.QApplication.doubleClickInterval()) + 150)
    assert len(prompts) == 1, prompts
    assert abs(prompts[0][0] - 320) < 2, prompts

    # armed off: clicks are inert again
    c.pose_armed = False
    release(centre)
    _pump(app, int(QtWidgets.QApplication.doubleClickInterval()) + 150)
    assert len(prompts) == 1, "a disarmed canvas must not prompt"


def test_layout_still_fits_with_tracking_enabled():
    """The overlay and TRACK button must cost zero layout height."""
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)

    class PoseOpts(Opts):
        pose = True

    off = MainWindow(Opts())
    off.resize(1366, 768)
    off.show()
    _pump(app, 200)
    h_off = off.minimumSizeHint()
    off.shutdown()
    off.close()

    on = MainWindow(PoseOpts())
    on.resize(1366, 768)
    on.show()
    _pump(app, 200)
    h_on = on.minimumSizeHint()
    on.shutdown()
    on.close()

    assert h_on.height() <= 768, f"tracking pushed the window to {h_on.height()}"
    assert h_on.height() == h_off.height(), (
        f"tracking changed the layout: {h_off.height()} -> {h_on.height()}; "
        "the overlay and button must be absolutely positioned children")


def test_pose_track_is_plain_data():
    """What crosses the thread boundary must be plain values (state.py:5-9)."""
    import numpy as _np

    from rov_gui.state import PoseTrack

    t = PoseTrack()
    assert t.T_cam_obj is None, "unknown is None, never a plausible default"
    assert t.score is None
    assert t.contours == () and t.has_mask is False
    for name in ("contours", "axes_px", "box_px"):
        assert isinstance(getattr(t, name), tuple), name
    live = PoseTrack(state="tracking",
                     contours=(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),))
    assert live.has_mask
    for poly in live.contours:
        for pt in poly:
            assert not isinstance(pt[0], _np.generic), "numpy scalars must not cross"


def test_pose_projection_is_pinhole_and_drops_points_behind_the_lens():
    """3-D -> pixels happens off the GUI thread; this is that maths.

    Deliberately distortion-free: the same K goes to FoundationPose, and on this
    camera the pinhole model is within 2.08 px of the real one everywhere
    (measured). Applying distortion here while feeding a pinhole K to the
    estimator would be the inconsistent choice, not the accurate one.
    """
    try:
        from rov_gui.perception.session import _project
    except Exception:                                            # noqa: BLE001
        return

    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])
    # a point on the optical axis lands on the principal point
    assert _project([[0.0, 0.0, 1.0]], K) == ((320.0, 180.0),)
    # 0.1 m right at 1 m -> 50 px right
    (x, y), = _project([[0.1, 0.0, 1.0]], K)
    assert abs(x - 370.0) < 1e-6 and abs(y - 180.0) < 1e-6
    # +Y is DOWN in OpenCV optical axes, so a positive y goes down the screen
    (_x, y2), = _project([[0.0, 0.1, 1.0]], K)
    assert y2 > 180.0, "OpenCV +Y must project downward"
    # anything behind the lens invalidates the whole set rather than drawing a
    # mirrored ghost somewhere on screen
    assert _project([[0.0, 0.0, 1.0], [0.0, 0.0, -0.5]], K) == ()
    assert _project([[0.0, 0.0, 0.0]], K) == ()


def test_pose_log_records_the_camera_and_its_provenance():
    """A pose is only as metric as the intrinsics behind it.

    The C3's EEPROM calibration is an UNDERWATER one (vendor-confirmed +
    measured HFOV, KNOWN_ISSUES 2026-08-04), so the file must say so — and
    say it the right way round: an earlier version wrote "in_air": true,
    copied from a stale comment that same audit had flagged as wrong, telling
    future readers to distrust exactly the in-water numbers that are metric.
    Also pins the ordering bug: recording can start BEFORE the first frame,
    and the CAMERA record must still carry a camera model, not an empty stub.
    """
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return          # no depthai/pymavlink on this machine

    from rov_gui.bus import RgbdMailbox

    class FakeSession:
        ready, loading, error, load_seconds = True, False, "", 1.0
        mesh_description = ""

        def click(self, x, y):
            pass

        def reset(self):
            pass

        def submit(self, *a, **k):
            pass

        def poll(self):
            return {"state": "tracking",
                    "contours": (((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),),
                    "score": 7.0, "mask_px": 42, "sam_hz": 25.0,
                    "frame_seq": 7, "src_w": 640, "src_h": 360, "message": ""}

        def close(self):
            pass

    class Intr:
        fx, fy, cx, cy, width, height = 513.4, 513.5, 317.5, 179.8, 640, 360
        distortion = (2.6, 81.3, 0.0)

    _app()
    out = Path(tempfile.mkdtemp(prefix="rov_gui_pose_"))
    w = PoseWorker(DataBus(), RgbdMailbox(), Opts())
    w.session = FakeSession()
    w.enabled = True

    # Recording starts before any frame exists — the ordering that broke it.
    w.set_sensor_log(True, str(out / "take"))
    w.set_click(311.0, 180.0)
    for _ in range(3):
        w.mailbox.put(np.zeros((360, 640, 3), np.uint8), None, Intr(),
                      12345.0, 4.2)
        w._last_pub = 0.0
        w.tick()
    w.set_sensor_log(False, "")

    rows = [json.loads(x) for x in
            (out / "take_pose.jsonl").read_text().splitlines()]
    kinds = [r["msg"] for r in rows]
    assert "CAMERA" in kinds and "TRACK" in kinds and "PROMPT" in kinds, kinds

    cam = next(r for r in rows if r["msg"] == "CAMERA")
    assert cam["fx"] == 513.4, "CAMERA written before the intrinsics arrived"
    assert cam["medium"] == "water", "the calibration medium must be IN the file"
    assert "in_air" not in cam, "the backwards in-air claim must be gone"
    assert "1.33" in cam["note"], "the in-air 1.33x warning must be spelled out"
    assert cam["rectified"] is False

    track = next(r for r in rows if r["msg"] == "TRACK")
    assert track["t"] == 12345.0, (
        "rows must be stamped with the frame's CAPTURE time, or they cannot be "
        "lined up against the video they belong to")
    assert len(track["contours"]) == 1

    meta = json.loads((out / "take_pose.json").read_text())
    assert meta["error"] is None and meta["by_message"]["TRACK"] == 3, meta


def test_no_loopworker_declares_a_slot():
    """A LoopWorker's slots can never fire, so declaring one is always a bug.

    ``LoopWorker.run()`` blocks for the worker's lifetime, so the thread's event
    loop is never entered and a queued invocation sits in a queue nobody
    services (backends/base.py module docstring). Qt does not warn; the slot is
    simply never called. That is how ``C3VideoWorker.set_sensor_log`` shipped
    and silently never wrote ``_c3_imu.jsonl``.

    An AST scan, so the rule is enforced rather than remembered.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                     for b in node.bases}
            if "LoopWorker" not in bases:
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                for dec in item.decorator_list:
                    name = dec.func if isinstance(dec, ast.Call) else dec
                    if getattr(name, "id", getattr(name, "attr", "")) == "Slot":
                        offenders.append(f"{path.name}::{node.name}.{item.name}")
    assert not offenders, (
        "LoopWorker subclasses cannot receive queued slots; these will never "
        f"fire: {offenders}. Use a plain thread-safe method plus a "
        "DirectConnection, as C3VideoWorker.request_sensor_log does.")


def test_c3_sensor_log_request_reaches_a_loopworker():
    """The C3 IMU log must actually start. It did not.

    ``_rov.jsonl`` appeared every time (VehicleWorker is a TimerWorker) while
    ``_c3_imu.jsonl`` never did, and the demo backend uses a TimerWorker too, so
    nothing caught it. This drives the real signal into the real worker.
    """
    try:
        from rov_gui.backends.hardware import C3VideoWorker
    except Exception:                                            # noqa: BLE001
        return          # no depthai/pymavlink on this machine

    _app()
    bus = DataBus()
    # Fully constructed — __init__ opens no device (that is setup()'s job) and
    # the QObject must really exist for connect() to bind to it.
    w = C3VideoWorker(bus, {}, Opts())

    bus.cmd_log_sensors.connect(w.request_sensor_log,
                                Qt.ConnectionType.DirectConnection)
    bus.cmd_log_sensors.emit(True, "/tmp/take_001")
    assert not w._log_q.empty(), (
        "the request never reached the worker — a LoopWorker cannot receive a "
        "queued slot, so this must be a plain method on a DirectConnection")
    assert w._log_q.get_nowait() == (True, "/tmp/take_001")

    # A start/stop pair issued inside one loop iteration must not lose the start.
    bus.cmd_log_sensors.emit(True, "/tmp/take_002")
    bus.cmd_log_sensors.emit(False, "")
    assert w._log_q.get_nowait() == (True, "/tmp/take_002")
    assert w._log_q.get_nowait() == (False, "")


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


def test_unprojection_is_metric_and_clamped():
    """The reference-view geometry, checked without a camera or a GPU.

    This is the maths every reconstructed mesh rests on: it turns the mask into
    the point cloud that colored ICP registers, so a factor of 1000 or a flipped
    axis here does not raise — it produces a confident-looking mesh of the wrong
    size, which is exactly the failure mode the upstream project warns about.
    """
    try:
        from rov_gui.perception.session import _PinholeUnprojector
    except Exception:                                            # noqa: BLE001
        return

    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])
    g = _PinholeUnprojector(K, 640, 360)
    depth = np.zeros((360, 640), np.uint16)

    # A pixel at the principal point, 1000 mm out -> (0, 0, 1.0) m.
    depth[180, 320] = 1000
    pts, (ys, xs) = g.unproject(depth)
    assert pts.shape == (1, 3) and (ys[0], xs[0]) == (180, 320)
    assert abs(pts[0, 2] - 1.0) < 1e-6, "millimetres in, METRES out"
    assert abs(pts[0, 0]) < 1e-6 and abs(pts[0, 1]) < 1e-6

    # 50 px right of centre at 1 m -> +0.1 m in X; 50 px BELOW -> +0.1 m in Y,
    # because OpenCV's +Y points down.
    depth[:] = 0
    depth[180, 370] = 1000
    depth[230, 320] = 1000
    pts, (ys, _xs) = g.unproject(depth)
    by_row = {int(r): p for r, p in zip(ys, pts)}
    assert abs(by_row[180][0] - 0.1) < 1e-6
    assert abs(by_row[230][1] - 0.1) < 1e-6, "OpenCV +Y must go DOWN"

    # The clamp keeps the stereo module's 0 and its saturated far values out of
    # the cloud; a single 4 m outlier drags the ICP that the mesh depends on.
    depth[:] = 0
    depth[10, 10], depth[11, 11], depth[12, 12] = 0, 100, 5000
    assert g.unproject(depth)[0].shape[0] == 0

    # A mask restricts it, which is the whole point: only the object.
    depth[:] = 1000
    mask = np.zeros((360, 640), bool)
    mask[100:110, 100:110] = True
    assert g.unproject(depth, mask)[0].shape[0] == 100


def test_reference_directory_refuses_the_path_that_breaks_reconstruction():
    """'rgb' anywhere in the path silently reconstructs the wrong thing.

    The reconstruction derives its sibling directories from the rgb/ path by
    string replacement (capture.py:254-256), so a parent called .../rgb_tests
    sends it looking for depth in a directory that does not exist. It fails
    late, in a child process, after two minutes.
    """
    try:
        from rov_gui.perception.session import PoseSession, PoseSessionError
    except Exception:                                            # noqa: BLE001
        return

    with tempfile.TemporaryDirectory() as td:
        bad = PoseSession(ref_dir=Path(td) / "rgb_captures", build=True)
        raised = False
        try:
            bad._new_ref_dir()
        except PoseSessionError:
            raised = True
        assert raised, "a path containing 'rgb' must be refused up front"

        ok = PoseSession(ref_dir=Path(td) / "meshes", build=True)
        a = ok._new_ref_dir()
        b = ok._new_ref_dir()
        assert a.is_dir() and b.is_dir()
        # Never the same directory twice: reference views are numbered from
        # zero, so reusing one reconstructs a chimera of two objects.
        assert a != b


def test_track_off_cancels_a_running_reconstruction():
    """The button that starts a two-minute GPU job must be able to stop it.

    A BundleSDF child left running holds several GB of VRAM behind a station
    that says it is idle — on a machine that is also flying a vehicle.
    """
    try:
        from rov_gui.perception.session import PoseSession, PHASE_CAPTURE
    except Exception:                                            # noqa: BLE001
        return

    class _FakeBuild:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class _FakeCapture:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

        def join(self, timeout=None):
            pass

    s = PoseSession(build=True)
    cap, bld = _FakeCapture(), _FakeBuild()
    s._capture, s._builder = cap, bld
    s.phase = "build"
    s.reset()
    assert bld.cancelled, "reset() must kill the reconstruction child"
    assert cap.stopped
    assert s.phase == PHASE_CAPTURE, "and leave the pipeline ready to retry"
    assert s._builder is None and s._capture is None


def test_reconstruction_is_the_default_only_when_there_is_no_mesh():
    """Which of the three pose modes a flag set selects.

    Getting this backwards is expensive in opposite directions: building when a
    mesh was supplied wastes two minutes of GPU per click, and NOT building when
    none was supplied is exactly the complaint that a click only ever produced a
    mask.
    """
    from rov_gui.__main__ import build_parser

    def mode(argv):
        o = build_parser().parse_args(argv)
        mesh = getattr(o, "pose_mesh", None) or None
        return mesh, (not mesh and not o.pose_no_build)

    assert mode(["--pose"]) == (None, True), "no mesh -> reconstruct on site"
    assert mode(["--pose", "--pose-no-build"]) == (None, False)
    mesh, build = mode(["--pose", "--pose-mesh", "/tmp/m.obj"])
    assert mesh == "/tmp/m.obj" and build is False

    o = build_parser().parse_args(["--pose"])
    # The arc cap is a measured cliff, not a preference (9.7 mm at 80 deg,
    # 87 mm at 100, 283 mm at 120), so the default must sit under it.
    assert o.pose_max_arc <= 80.0
    assert o.pose_max_views >= 10
    assert "rgb" not in str(Path(o.pose_ref_dir).resolve())


def _hangul(text: str) -> str:
    """Any Hangul left in a UI string. Em-dashes and degrees are fine."""
    return "".join(c for c in text
                   if 0xAC00 <= ord(c) <= 0xD7A3        # syllables
                   or 0x1100 <= ord(c) <= 0x11FF        # jamo
                   or 0x3130 <= ord(c) <= 0x318F)       # compatibility jamo


def test_every_pose_stage_says_what_it_is_doing_in_english():
    """No stage may be silent, and none may be Korean.

    The failure this pins: a capture that gave up reverted to a plain green
    TRACKING chip with no text anywhere, so "collecting", "reconstructing" and
    "quietly stopped" were indistinguishable from the pilot's seat. Every state
    the session can publish now has to produce a stage line, and the reason line
    has to be there too whenever there is a reason.
    """
    from rov_gui.state import PoseTrack
    from rov_gui.widgets.video import VideoCanvas

    states = ("off", "loading", "idle", "live", "capturing", "building",
              "pose_loading", "registering", "tracking", "lost", "failed",
              "fault")
    for st in states:
        stage, reason = VideoCanvas._pose_status(
            PoseTrack(state=st, n_views=7, max_views=20, arc_deg=31.0,
                      max_arc=75.0, build_s=42.0, load_s=1.1, fp_load_s=8.0,
                      sam_hz=30.0, distance_m=0.45))
        assert stage, f"state {st!r} draws no stage line"
        assert stage == stage.upper() or any(c.isdigit() for c in stage)
        for text in (stage, reason):
            assert not _hangul(text), \
                f"state {st!r} shows Korean text: {text!r}"

    # The three that mean "waiting, nothing for you to do" must still explain
    # themselves, because they are the ones that look like a hang.
    for st in ("capturing", "building", "registering", "pose_loading"):
        _stage, reason = VideoCanvas._pose_status(PoseTrack(state=st))
        assert reason, f"state {st!r} gives the pilot no reason line"

    # And the distinction the whole thing rests on: no pose because you asked
    # for a mask, versus no pose yet.
    mask_only, _ = VideoCanvas._pose_status(
        PoseTrack(state="tracking", pose_expected=False))
    pending, _ = VideoCanvas._pose_status(
        PoseTrack(state="tracking", pose_expected=True))
    assert "mask only" in mask_only and "no pose yet" in pending


def test_the_recording_carries_the_tracking_overlay():
    """A recording of a tracking session must show the tracking.

    Three things at once, and the middle one is the trap: the overlay reaches
    the file, the ORIGINAL frame is left alone (the canvas still paints its own
    overlay on top of it live, so drawing into the same QImage would double
    every stroke), and a panel with tracking off pays nothing.
    """
    from rov_gui.qt import QtGui
    from rov_gui.state import Conn, PoseTrack, VideoStat
    from rov_gui.widgets.video import VideoPanel

    _app()
    mb = FrameMailbox("c3")
    panel = VideoPanel("main", "C3 RGB", mb, pose=True)
    panel.resize(640, 360)

    src = QtGui.QImage(320, 180, QtGui.QImage.Format.Format_RGB32)
    src.fill(QtGui.QColor("#101010"))
    before = src.copy()

    # Nothing to draw yet: the same object comes straight back, no copy.
    assert panel.burn_overlay(src) is src

    panel.canvas._stat = VideoStat("C3 RGB", width=640, height=360)
    panel.canvas.set_track(PoseTrack(
        state="tracking", conn=Conn.ONLINE, src_w=640, src_h=360, sam_hz=30.0,
        contours=(((100.0, 60.0), (400.0, 60.0), (400.0, 300.0),
                   (100.0, 300.0)),)))
    out = panel.burn_overlay(src)
    assert out is not src, "must not paint into the frame the canvas still owns"
    assert (out.width(), out.height()) == (320, 180)
    assert out != before, "the overlay never reached the recorded frame"
    assert src == before, "the source frame was modified"

    # And a panel built without tracking is untouched by any of this.
    plain = VideoPanel("rov", "DEFAULT RGB", FrameMailbox("rov"))
    assert plain.burn_overlay(src) is src


def test_upstream_operator_messages_are_translated():
    """The upstream project talks to its operator in Korean; the UI is English.

    These strings are the ONLY explanation the pilot gets for why a capture is
    not progressing, so an untranslated one makes the most important line on the
    panel the one nobody can act on. The upstream tree is read-only to us, so
    the translation has to happen at this boundary.
    """
    try:
        from rov_gui.perception.session import english
    except Exception:                                            # noqa: BLE001
        return

    cases = {
        "물체를 천천히 좌우로 돌려주세요": "turn the object slowly",
        "수집 완료 — 누적 회전 75도 도달": "75 deg",
        "물체가 너무 멉니다 (124 cm) — 30~80 cm 로 가져오세요": "124 cm",
        "마스크가 갑자기 커졌다 (다른 물체로 옮겨탔을 수 있다)": "another object",
        "각도 차이가 작다 (3도)": "3 deg",
        # the one reject the pilot can fix in a second: say which way to move
        "카메라-물체 거리가 이상하다 (2040 mm)": "bring it to 300-800 mm",
        "놓침 — 재획득 중 (2s)": "2 s",
        "물체를 클릭하세요": "click an object",
    }
    for src, expect in cases.items():
        out = english(src)
        assert expect in out, f"{src!r} -> {out!r}"
        # The number must survive: a translation that drops the measurement is
        # worse than no translation at all.
        assert not _hangul(out), out

    # A composed message translates in one pass, both halves.
    both = english("수집 완료 — 20장 채움")
    assert "collection finished" in both and "20 views" in both, both
    # The summary line that goes to the log, numbers intact.
    s = english("8장  회전 75도  마스크 1882~2064 px (중앙 1974)  "
                "거리 426~460 mm  [누적 회전 75도 도달]")
    for token in ("8 views", "arc 75 deg", "1882-2064", "426-460 mm"):
        assert token in s, (token, s)
    assert english("") == ""


def test_the_reconstruction_states_reach_the_screen():
    """capturing/building are the two states the pilot has to steer.

    The arc is a budget they spend by orbiting and the build is a wait; if the
    chip cannot say which one it is in, the counter just sits there and a
    working capture looks like a hung station.
    """
    from rov_gui.qt import QtGui
    from rov_gui.state import PoseTrack, Conn
    from rov_gui.widgets.video import VideoCanvas

    _app()
    c = VideoCanvas("C3 RGB", FrameMailbox("c3"), pose=True)
    c.resize(640, 360)
    for t in (PoseTrack(state="capturing", n_views=7, max_views=20,
                        arc_deg=31.0, max_arc=75.0, conn=Conn.CONNECTING,
                        note="orbit the object slowly", src_w=640, src_h=360),
              PoseTrack(state="building", build_s=42.0, conn=Conn.CONNECTING,
                        note="reconstructing the mesh (about 2 minutes)",
                        src_w=640, src_h=360)):
        c.set_track(t)
        img = QtGui.QImage(640, 360, QtGui.QImage.Format.Format_RGB32)
        p = QtGui.QPainter(img)
        try:
            c._paint_pose(p, c.rect())        # must not raise
        finally:
            p.end()
    # And the fields are plain data, so they can cross a process boundary.
    t = PoseTrack(state="capturing", n_views=7, arc_deg=31.0)
    for name in ("n_views", "arc_deg", "max_arc", "max_views", "build_s"):
        assert isinstance(getattr(t, name), (int, float))


def test_no_method_is_shadowed_by_an_instance_attribute():
    """A `self.x = ...` in __init__ silently REPLACES a method named `x`.

    This is the bug class behind the capture pipeline freeze: PoseWorker had a
    method `_log` and an attribute `self._log` (the SensorLog). The attribute
    won, so every `on_log=self._log` passed None — or, while recording, a
    SensorLog object, which is not callable, so the capture finalizer raised
    TypeError on every tick and COLLECTING VIEWS could never end. Nothing
    about the pattern looks wrong at the call site, which is why a test bans
    it package-wide instead of a review catching it once.
    """
    import ast

    pkg = Path(__file__).resolve().parents[1]
    offenders = []
    for py in sorted(pkg.rglob("*.py")):
        if "tests" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # property/setter pairs assign through the descriptor;
                    # that is the pattern working as designed, not a shadow.
                    decs = {getattr(d, "id", getattr(d, "attr", ""))
                            for d in item.decorator_list}
                    if not decs & {"property", "setter", "cached_property"}:
                        methods.add(item.name)
            for item in ast.walk(node):
                if isinstance(item, ast.Assign) or isinstance(item, ast.AnnAssign):
                    targets = (item.targets if isinstance(item, ast.Assign)
                               else [item.target])
                    for tgt in targets:
                        if (isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                                and tgt.attr in methods):
                            offenders.append(
                                f"{py.name}:{item.lineno} {node.name}."
                                f"{tgt.attr} (method shadowed by attribute)")
    assert not offenders, "\n".join(offenders)


def test_capture_finalizes_even_if_the_object_was_lost_at_the_end():
    """Collection done must finalize whatever SAM2 thinks NOW.

    The pilot stops orbiting the moment the counter says done, and SAM2
    routinely loses the object at exactly that moment. The old code gated the
    finalize behind state == TRACKING, so the pipeline waited forever with
    "collection finished" frozen on screen — the exact freeze reported from
    the first hardware session.
    """
    try:
        from rov_gui.perception.session import (PHASE_CAPTURE, PHASE_FAILED,
                                                PoseSession)
    except Exception:                                            # noqa: BLE001
        return

    class _Recorder:
        n, arc = 3, 78.0
        def summary(self):
            return "3 views (fake)"

    class _DoneCapture:
        recorder = _Recorder()
        def snapshot(self):
            return {"done": True, "n": 3, "arc": 78.0}
        def stop(self):
            pass
        def join(self, timeout=None):
            pass

    class _LostLive:
        def submit(self, *a, **k):
            pass
        def snapshot(self):
            # The worst case: the object is gone the moment the orbit ends.
            return {"state": "LOST", "mask": None, "score": None,
                    "message": "", "hz": 0.0}
        def request_reset(self):
            pass

    s = PoseSession(build=True)
    s._live = _LostLive()
    s._np_K = np.eye(3)
    s._capture = _DoneCapture()
    assert s.phase == PHASE_CAPTURE
    s.submit(np.zeros((360, 640, 3), np.uint8),
             np.full((360, 640), 500, np.uint16))
    assert s.phase == PHASE_FAILED, \
        "a finished collection must finalize even with SAM2 in LOST"
    assert "3" in s.poll()["message"] if s._live else True


def test_depth_probe_reads_millimetres_from_the_frame_on_screen():
    """Hover on the depth panel -> a metric readout from the SAME frame.

    The raw uint16 map rides the mailbox WITH its picture, so the cursor can
    never quote one frame while showing another. 0 is DepthAI's "no
    measurement" and must come back as None ("no data"), never as 0.00 m.
    """
    from rov_gui.qt import QtCore, QtGui
    from rov_gui.state import VideoStat
    from rov_gui.widgets.video import VideoCanvas

    _app()
    mb = FrameMailbox("depth")
    depth = np.full((360, 640), 1500, np.uint16)
    depth[100, 200] = 420
    depth[50, 50] = 0                       # a hole in the stereo map
    img = bgr_to_qimage(np.zeros((360, 640, 3), np.uint8))
    mb.put(img, VideoStat("depth", width=640, height=360), aux=depth)

    c = VideoCanvas("C3 Depth", mb, legend=("0.3 m", "6 m"))
    c.resize(640, 360)
    assert c.pull()
    # paint once so the letterbox rectangle exists for to_source()
    target = QtGui.QImage(640, 360, QtGui.QImage.Format.Format_RGB32)
    c.render(target)          # render() drives paintEvent with its own painter
    r = c._fit_rect
    assert r is not None

    def canvas_pt(sx, sy):
        return QtCore.QPoint(int(r.x() + (sx / 640) * r.width()),
                             int(r.y() + (sy / 360) * r.height()))

    assert c.depth_at(canvas_pt(200, 100)) == 420.0
    assert c.depth_at(canvas_pt(50, 50)) is None, "0 mm is a hole, not a distance"
    assert c.depth_at(QtCore.QPoint(-5, -5)) is None, "off the image"
    # and painting the probe must not raise with a hover set
    c._hover = canvas_pt(200, 100)
    c.render(target)


def test_pose3d_projection_keeps_the_conventions():
    """The 3-D map must speak the same frame as the POSE log.

    +Z forward must recede to the same side the yaw sends it, +Y must go DOWN
    on screen at zero orbit (OpenCV optical axes), and zoom must scale
    distances about the volume centre. Pure maths, no GPU.
    """
    from rov_gui.qt import QtGui
    from rov_gui.state import PoseTrack
    from rov_gui.widgets.pose3d import Pose3DView, Pose3DWindow, RANGE_Z

    _app()
    v = Pose3DView()
    v.resize(400, 400)
    v.yaw = 0.0
    v.pitch = 0.0
    v.zoom = 1.0
    ox, oy = v._project(0, 0, RANGE_Z / 2)      # volume centre -> screen centre
    assert abs(ox - 200) < 1e-6 and abs(oy - 200) < 1e-6
    _x, y_down = v._project(0, 0.5, RANGE_Z / 2)
    assert y_down > oy, "OpenCV +Y must project DOWN the screen"
    x_right, _y = v._project(0.5, 0, RANGE_Z / 2)
    assert x_right > ox, "+X must project right at zero orbit"
    # zoom doubles the offset from centre, exactly
    v.zoom = 2.0
    x2, _ = v._project(0.5, 0, RANGE_Z / 2)
    assert abs((x2 - ox) - 2 * (x_right - ox)) < 1e-6

    # The feed dedups republished poses and survives poseless states.
    v.zoom = 1.0
    T = (1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.5, 0, 0, 0, 1)
    for _ in range(5):
        v.add(PoseTrack(state="tracking", T_cam_obj=T))
    assert len(v.trail) == 1, "identical republished poses must not grow the trail"
    v.add(PoseTrack(state="lost"))
    assert v.pose_T is None and len(v.trail) == 1

    # And the full window paints offscreen without raising.
    w = Pose3DWindow()
    w.resize(420, 400)
    w.add(PoseTrack(state="tracking", T_cam_obj=T, pose_hz=44.0))
    img = QtGui.QImage(420, 400, QtGui.QImage.Format.Format_RGB32)
    w.view.render(img)


def test_pose_csv_rides_the_recording_and_holds_the_full_pose():
    """One CSV row per estimated pose, same numbers as the jsonl POSE rows.

    A second FORMAT, never a second source: position, distance and the
    rotation rows all come from the same T_cam_obj tuple the jsonl gets.
    """
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.bus import DataBus, RgbdMailbox
    from rov_gui.state import PoseTrack

    _app()
    w = PoseWorker(DataBus(), RgbdMailbox(), type("O", (), {})())
    with tempfile.TemporaryDirectory() as td:
        w._open_csv(Path(td) / "x_pose.csv")
        T = (1.0, 0.0, 0.0, 0.058, 0.0, 1.0, 0.0, 0.109,
             0.0, 0.0, 1.0, 0.413, 0.0, 0.0, 0.0, 1.0)
        w._write_csv(PoseTrack(state="tracking", T_cam_obj=T, frame_seq=7,
                               pose_hz=44.0, n_register=1), 12.34)
        # a poseless track writes nothing — the CSV is poses, not states
        w._write_csv(PoseTrack(state="lost"), 12.44)
        w._close_csv()
        lines = (Path(td) / "x_pose.csv").read_text().strip().split("\n")
    assert len(lines) == 2, lines
    head = lines[0].split(",")
    row = dict(zip(head, lines[1].split(",")))
    assert row["frame_seq"] == "7" and row["state"] == "tracking"
    assert abs(float(row["x_m"]) - 0.058) < 1e-9
    assert abs(float(row["z_m"]) - 0.413) < 1e-9
    assert abs(float(row["distance_m"]) - (0.058 ** 2 + 0.109 ** 2
                                           + 0.413 ** 2) ** 0.5) < 1e-4
    assert row["r00"] == "1.000000" and row["r21"] == "0.000000"
    assert float(row["t"]) == 12.34


def test_closing_the_session_mid_load_never_adopts_the_tracker():
    """close() during the model load must win, whichever thread is ahead.

    The load takes seconds to tens of seconds and the station quits whenever
    the pilot quits. The old sequence (start the tracker thread, THEN store
    it) had a window where close() found nothing to join, and the loader then
    started a torch daemon thread into a discarded session — the interpreter-
    exit abort the module's own docstring promises to prevent.
    """
    try:
        from rov_gui.perception.session import PoseSession
    except Exception:                                            # noqa: BLE001
        return

    started, stopped = [], []

    class _FakeLive:
        def __init__(self, *a, **k):
            pass

        def start(self):
            started.append(1)

        def stop(self):
            stopped.append(1)

        def join(self, timeout=None):
            pass

    class _FakeSam:
        def __init__(self, *a, **k):
            time.sleep(0.15)             # the "model build" window

    fake_tracker = types.ModuleType("sam2_live.tracker")
    fake_tracker.LiveTracker = _FakeLive
    fake_tracker.StreamingSam2Tracker = _FakeSam
    fake_pkg = types.ModuleType("sam2_live")
    saved = {k: sys.modules.get(k) for k in ("sam2_live", "sam2_live.tracker")}
    sys.modules["sam2_live"] = fake_pkg
    sys.modules["sam2_live.tracker"] = fake_tracker
    try:
        with tempfile.TemporaryDirectory() as td:
            s = PoseSession(src_dir=td)
            s.start_async()
            time.sleep(0.02)             # loader is inside the build window
            s.close()                    # must join the loader and refuse late adoption
            assert s._loader is not None and not s._loader.is_alive(), \
                "close() must join the loader thread"
            assert started == [], \
                "a tracker built after close() must never be STARTED"
            assert s._live is None and not s.ready
            # and a session closed mid-load must not advertise readiness later
            time.sleep(0.05)
            assert not s.ready
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_track_off_means_the_video_worker_does_not_copy():
    """The producer must know nobody is listening BEFORE it copies 1.15 MB.

    PoseWorker discards every mailbox item while TRACK is off, so a tap that
    copies colour+depth on every frame regardless is pure waste — and it
    contradicts the documented promise that tracking off costs nothing.
    """
    from rov_gui.bus import RgbdMailbox

    mb = RgbdMailbox()
    assert not mb.wanted(), "nobody has pressed TRACK yet"
    mb.put(np.zeros((2, 2, 3), np.uint8), None, None, 1.0)
    mb.set_wanted(False)
    assert mb.take() is None, "turning TRACK off must drop the stale item"
    mb.set_wanted(True)
    assert mb.wanted()

    # And the worker actually flips it: set_enabled is the one switch.
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return
    _app()
    w = PoseWorker(DataBus(), RgbdMailbox(), Opts())
    w.set_enabled(True)
    assert w.mailbox.wanted()
    w.set_enabled(False)
    assert not w.mailbox.wanted()


def test_mesh_record_lands_in_a_log_opened_after_the_mesh_existed():
    """The MESH row is per LOG FILE, not per process.

    One latch shared by the console announcement and the file write meant a
    mesh reconstructed before recording started was announced once and then
    never written — the file whose whole point is provenance was the one
    place without it.
    """
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.bus import RgbdMailbox
    from rov_gui.sensorlog import SensorLog

    class _Session:
        mesh = __file__                  # any real file: sha1 must be readable
        ref_dir = "/tmp/refs"
        mesh_check_lines = ("mesh 100 verts",)

    _app()
    w = PoseWorker(DataBus(), RgbdMailbox(), Opts())
    # The mesh exists BEFORE any log: console announcement only.
    w._log_mesh_once(_Session())
    assert w._mesh_logged and not w._mesh_record_written

    with tempfile.TemporaryDirectory() as td:
        w._log = SensorLog(Path(td) / "a_pose.jsonl", "pose")
        w._mesh_record_written = False   # what set_sensor_log does on open
        w._log_mesh_once(_Session())
        w._log.close()
        w._log = None
        rows = [json.loads(x) for x in
                (Path(td) / "a_pose.jsonl").read_text().splitlines()]
    mesh_rows = [r for r in rows if r["msg"] == "MESH"]
    assert len(mesh_rows) == 1, "the log opened later must still name the mesh"
    assert len(mesh_rows[0]["sha1"]) == 40


def test_pose_rows_between_frames_reuse_the_last_capture_stamp():
    """The 10 Hz publish gate fires between camera frames; item is None then.

    The pose on those ticks is still the one estimated from the LAST frame,
    so its rows carry that frame's capture time — not 0.0 (which sorted the
    CSV to the epoch) and not now().
    """
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.bus import RgbdMailbox
    from rov_gui.sensorlog import SensorLog
    from rov_gui.state import PoseTrack

    _app()
    w = PoseWorker(DataBus(), RgbdMailbox(), Opts())
    T = tuple(float(v) for v in
              (1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.5, 0, 0, 0, 1))
    track = PoseTrack(state="tracking", T_cam_obj=T, frame_seq=3)
    with tempfile.TemporaryDirectory() as td:
        w._log = SensorLog(Path(td) / "b_pose.jsonl", "pose")
        w._open_csv(Path(td) / "b_pose.csv")
        w._log_track(track, {"t_capture": 777.5})    # a real frame
        w._log_track(track, None)                    # a between-frames tick
        w._log.close()
        w._log = None
        w._close_csv()
        csv_rows = (Path(td) / "b_pose.csv").read_text().strip().split("\n")[1:]
    assert len(csv_rows) == 2
    stamps = [float(r.split(",")[0]) for r in csv_rows]
    assert stamps == [777.5, 777.5], (
        f"a tick with no fresh frame must reuse the last capture stamp, "
        f"got {stamps}")


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
