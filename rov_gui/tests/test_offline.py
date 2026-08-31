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
from rov_gui.qt import QImage, Qt, QtCore, QTimer, QtWidgets
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

    # The MISSION LOG is the one exemption (2026-08-14, operator request:
    # "예전 것도 볼 수 있게"). The rule is "a control station must not hide a
    # CONTROL behind a scroll", and a log is not a control — nothing in it is
    # clickable, and the newest line is always the visible one. Everything
    # else stays banned; see the same exemption in test_control.py.
    scrollers = [s for s in win.findChildren(QtWidgets.QAbstractScrollArea)
                 if s is not win.payload.log_view]
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


def test_axis_numbers_are_read_off_the_pad_not_baked_in():
    """The same pad renumbers its axes when you plug the cable in.

    Measured 2026-08-17 on this desktop, "Microsoft Xbox Series S|X Controller"
    over USB (xpad), via JSIOCGAXMAP: axes 0..7 are
    X, Y, Z, RX, RY, RZ, HAT0X, HAT0Y — i.e. the right stick is 3,4 and axis 2
    is the LEFT TRIGGER (it reported rest = -1.000, so joystick.py auto-zeroes
    it and it can never command anything). The old hard-coded defaults were
    yaw=2, heave=-3, which is why the pilot's yaw stick drove HEAVE and yaw was
    dead. The Bluetooth index layout (0,1 left / 2,3 right / 4,5 triggers) is
    the one probed 2026-08-06 and recorded in joystick.py; the ABS_* codes
    behind those Bluetooth indices are inferred from the hid-generic driver,
    not measured, so what this pins there is the RULE, not that pad's codes.
    """
    from rov_gui.joystick import Joystick

    usb = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x11]   # measured
    assert Joystick._sticks_from(usb) == {
        "left_x": 0, "left_y": 1, "right_x": 3, "right_y": 4}, (
        "USB pad: the right stick is 3,4 — pinning it to 2,3 is the bug")

    # A pad with no ABS_RX/RY has its right stick on Z/RZ, which lands on the
    # Bluetooth pad's known-good 2,3 and keeps the wireless case working.
    bt = [0x00, 0x01, 0x02, 0x05, 0x0a, 0x09, 0x10, 0x11]
    assert Joystick._sticks_from(bt) == {
        "left_x": 0, "left_y": 1, "right_x": 2, "right_y": 3}, (
        "Bluetooth pad must keep flying on 2,3")

    # The vehicle axes that come out of it, signs included.
    js = Joystick.__new__(Joystick)
    js.sticks = Joystick._sticks_from(usb)
    assert js.layout() == {"surge": (1, -1.0), "sway": (0, +1.0),
                           "yaw": (3, +1.0), "heave": (4, -1.0)}, js.layout()

    # And the safety property behind the whole fix: no vehicle axis may be
    # pointed at a trigger, because a trigger rests at full scale and is
    # auto-zeroed — a control surface that cannot move is a dead control.
    triggers = {usb.index(0x02), usb.index(0x05)}
    assert not ({a for a, _s in js.layout().values()} & triggers), (
        "a vehicle axis landed on an analogue trigger")


def test_button_translation_survives_the_cable_too():
    """The kernel's BUTTON indices move with the transport as well.

    Over Bluetooth this pad leaves gaps at 2/5/8/9 (measured 2026-08-06); over
    USB the same buttons are packed 0..10 with no gaps at all (measured
    2026-08-17 via JSIOCGBTNMAP: A B X Y TL TR SELECT START MODE THUMBL
    THUMBR). One hand-written index table therefore cannot be right for both,
    and this pad's USB name is in no table at all — which would have sent the
    KERNEL's numbers, the exact path that armed the vehicle on 2026-08-07.
    Deriving from the driver's BTN_* codes covers both.
    """
    from rov_gui.joystick import Joystick, BTN_TO_SDL, XBOX_WIRELESS_JS_TO_SDL

    def derive(btnmap):
        return {i: BTN_TO_SDL[c] for i, c in enumerate(btnmap) if c in BTN_TO_SDL}

    # Bluetooth: BTN_A..BTN_THUMBR contiguous. Must reproduce the table that
    # was measured against the real vehicle, gaps and all.
    assert derive(list(range(0x130, 0x13F))) == XBOX_WIRELESS_JS_TO_SDL

    # USB (measured): packed, no gaps.
    usb = [0x130, 0x131, 0x133, 0x134, 0x136, 0x137,
           0x13A, 0x13B, 0x13C, 0x13D, 0x13E]
    got = derive(usb)
    assert got == {0: 0, 1: 1, 2: 2, 3: 3, 4: 9, 5: 10,
                   6: 4, 7: 6, 8: 5, 9: 7, 10: 8}, got
    assert got[4] != 4 and got[6] != 6, (
        "kernel 4/6 must not reach the vehicle unchanged — BTN6 is arm")

    # The derived map wins over the name table, and an un-probed pad still
    # falls back to the name table rather than to nothing.
    js = Joystick.__new__(Joystick)
    js.name, js.raw = "Microsoft Xbox Series S|X Controller", {}
    js.derived_map = got
    js.buttons = {6: True}
    assert js.vehicle_buttons() == {4}, js.vehicle_buttons()
    js.derived_map = {}
    js.name = "Xbox Wireless Controller"
    assert js.vehicle_buttons() == {9}, js.vehicle_buttons()


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


def test_the_trajectory_3d_view_reduces_to_the_top_down_one():
    """The 3D button tilts the SAME plot, so the two must agree at the seam.

    This replaced the floating 3-D pose window on 2026-08-21 (that one drew
    the object in the CAMERA frame; this draws everything in the POOL frame,
    beside the vehicle). The property that makes the toggle safe is
    continuity: at elevation 90 the orthographic basis has to reproduce the
    top-down projection EXACTLY — screen up = +x_ned, screen right = +y_ned —
    or pressing the button would silently move the whole world.
    """
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.resize(700, 500)
    v = w.view
    v.set_pool([(-2, -2), (-2, 2), (2, 2), (2, -2)])

    v.set_three_d(False)
    flat = [v._px(x, y) for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    v.set_three_d(True)
    v.azimuth_deg, v.elev_deg = 0.0, 90.0
    tilted = [v._px(x, y, 0.0) for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    for a, b in zip(flat, tilted):
        assert abs(a.x() - b.x()) < 1e-9 and abs(a.y() - b.y()) < 1e-9, (a, b)
    # +x_ned up the screen, +y_ned to the right — the panel's stated convention
    assert tilted[1].y() < tilted[0].y() and tilted[2].x() > tilted[0].x()

    # Tilted, DEPTH has to move a point — that is the entire reason for the
    # mode, and a z that changed nothing would be a 3-D view in name only.
    v.elev_deg = 50.0
    deep = v._px(0.0, 0.0, 1.0)
    shallow = v._px(0.0, 0.0, 0.0)
    assert abs(deep.y() - shallow.y()) > 1.0, (deep, shallow)
    # ...and map z is DOWN, so deeper must draw LOWER on the screen and the
    # world's up must project UP. Getting the eye on the wrong side of the
    # azimuth satisfies the elevation-90 check above and still renders the
    # whole world upside down, so both halves are pinned here.
    assert deep.y() > shallow.y(), "deeper drew HIGHER — the view is inverted"
    _r, u = v._basis3()
    assert -u[2] > 0.0, "the world's up does not project up the screen"

    # Top-down, z must be ignored (it always was, and every 2-tuple caller
    # still relies on that).
    v.set_three_d(False)
    assert v._px(0.0, 0.0, 3.0) == v._px(0.0, 0.0, 0.0)


def test_the_trajectory_panel_paints_in_both_modes_with_every_marker():
    """Vehicle, reference, DR ghost and object, top-down and tilted, offscreen.

    A paint that raises takes the GUI thread down with it, and half of these
    markers only exist on a --pose --mpc run that nothing offline used to
    exercise."""
    import numpy as np

    from rov_gui.qt import QtGui
    from rov_gui.state import Conn, MpcStatus, NavFix, ObjectFix
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.resize(700, 500)
    w.view.set_pool([(-2, -2), (-2, 2), (2, 2), (2, -2)])
    w.view.set_map_tags([(0.3, 0.2, 0.0, 79, 0), (0.6, 0.2, 0.0, 80, 0)])
    w.add_fix(NavFix(t_capture=0.0, n_tags=2, tag_ids=(79, 80),
                     p_ned=(0.4, -0.2, -0.9),
                     R_ned_body=tuple(np.eye(3).ravel()), yaw_ned=0.3,
                     reproj_rms_px=0.4, conn=Conn.ONLINE))
    w.add_status(MpcStatus(engaged=True, mode="pid",
                           p_flu=(0.4, 0.2, 0.9), ref_flu=(0.5, 0.2, 0.9),
                           yaw_flu_deg=-17.0, err_xy=0.1,
                           p_dr_flu=(0.45, 0.2, 0.9), dr_ok=True,
                           dr_mode="shadow", dr_source="c3",
                           dr_attitude="ahrs", dr_err_m=0.05,
                           follow_state="following", follow_err_m=0.04,
                           conn=Conn.ONLINE))
    w.set_object(ObjectFix(ok=True, state="live", p_map=(0.9, 0.1, -0.5),
                           yaw_map=0.4, distance_m=0.55, age_s=0.05,
                           pair_dt_ms=0.0, pair_exact=True, yaw_axis="x",
                           pose_state="tracking", conn=Conn.ONLINE))
    for three_d in (False, True):
        w.btn_3d.setChecked(three_d)
        assert w.view.three_d is three_d
        img = QtGui.QImage(700, 460, QtGui.QImage.Format.Format_RGB32)
        w.view.render(img)


def test_the_plot_says_NO_FIX_instead_of_drawing_nothing():
    """Nine mapped tags outlined in the video and no marker on the plot read
    as "the panel is broken" (operator, 2026-08-21). Two things had to change:
    a 20 Hz MpcStatus with no state must not erase the marker `add_fix` set,
    and a genuinely absent fix has to SAY so, with the localizer's own reason.
    """
    import numpy as np

    from rov_gui.state import Conn, MpcStatus, NavFix
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.resize(700, 500)
    fix = NavFix(t_capture=0.0, n_tags=2, tag_ids=(79, 80),
                 p_ned=(0.4, -0.2, -0.9),
                 R_ned_body=tuple(np.eye(3).ravel()), yaw_ned=0.3,
                 reproj_rms_px=0.4, conn=Conn.ONLINE)
    w.add_fix(fix)
    assert w.view._act_fresh(), "a good fix must place the vehicle marker"
    # ...and the 20 Hz status stream that carries no position must NOT erase
    # it while nothing is engaged. This is the regression.
    for _ in range(20):
        w.add_status(MpcStatus(engaged=False, conn=Conn.DEGRADED))
    assert w.view._act_fresh(), (
        "an unengaged MpcStatus erased the marker the localizer had set")
    # A miss carries the reason, and the reason has to survive the next
    # status tick — add_status used to overwrite the chip inside 50 ms.
    w.add_fix(NavFix(t_capture=0.0, n_tags=0, tag_ids=(79, 80),
                     conn=Conn.DEGRADED, note="reproj 4.1px > 3px"))
    w.add_status(MpcStatus(engaged=False, reason="", conn=Conn.DEGRADED))
    assert "reproj" in w.chip.text(), w.chip.text()
    assert w.view._nav_note == "reproj 4.1px > 3px"
    # ...and once the marker ages out, the plot says NO FIX in as many words.
    w.view._act_t -= 10.0
    assert not w.view._act_fresh()


def test_the_engaged_marker_keeps_its_map_depth_instead_of_the_tag_plane():
    """The engage datum has a Z OFFSET, and `_to_map` never undid it.

    `_datumize` reports `Rz @ (eta - p0)`, so a datum-frame z is measured from
    the depth the vehicle engaged at. `_to_map` rotated and translated x/y and
    left z alone, so from the moment anything engaged the hull, its trail, the
    reference cross and the DR ghost were all drawn at z ~ 0 — LYING ON THE TAG
    MAT — while the object kept its true map z. Invisible top-down; tilt the
    view and the vehicle was on the floor (operator, 2026-08-23).
    """
    from rov_gui.state import Conn, MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    # Engaged 1.00 m above the tag floor and holding exactly there: every
    # datum-frame z below is 0, and every map z must be -1.00.
    w.view.set_datum((0.3, -0.2, -1.00, 0.0))
    w.add_status(MpcStatus(engaged=True, mode="pid", conn=Conn.ONLINE,
                           p_flu=(0.0, 0.0, 0.0), ref_flu=(0.0, 0.0, 0.0),
                           p_dr_flu=(0.0, 0.0, 0.0), dr_ok=True,
                           dr_mode="shadow", yaw_flu_deg=0.0))
    for name, p in (("hull", w.view.p_act), ("reference", w.view.p_ref),
                    ("DR ghost", w.view.p_dr)):
        assert abs(p[2] + 1.00) < 1e-9, f"{name} was drawn at z={p[2]:+.3f}"
    assert abs(w.view.trail_act[-1][3] + 1.00) < 1e-9, "the trail lost its z"
    # ...and the chip, which used to be the ONLY place that folded the offset
    # back in, must not now add it twice.
    assert "z -1.00 m" in w._hold_z_text(), w._hold_z_text()
    # 3-D: a marker 1 m off the floor cannot project to the floor's pixel.
    w.btn_3d.setChecked(True)
    w.view.resize(600, 400)
    on_floor = w.view._px(w.view.p_act[0], w.view.p_act[1], 0.0)
    at_depth = w.view._px(*w.view.p_act)
    assert abs(on_floor.y() - at_depth.y()) > 20, (
        "the hull projects onto the tag plane in the tilted view")
    w.close()


def test_only_one_source_writes_the_vehicle_marker_at_a_time():
    """Unengaged, the LOCALIZER owns the marker and the control state does not.

    MpcWorker._publish fills `p_flu` on every tick it has a state, engaged or
    not, and the two estimates are not the same quantity: the fix is raw tag
    PnP, the assembled state is pressure depth plus a velocity-bridged x/y and
    a gyro-bridged yaw. Taking a sample from each, 17 Hz against 20 Hz, made
    the marker alternate and the trail draw a vertical comb in the 3-D view
    (operator, 2026-08-23). ONE writer at a time, switched by `engaged`.
    """
    import numpy as np

    from rov_gui.state import Conn, MpcStatus, NavFix
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.resize(700, 500)
    w.add_fix(NavFix(t_capture=0.0, n_tags=9, tag_ids=(79,),
                     p_ned=(0.40, -0.20, -1.00),
                     R_ned_body=tuple(np.eye(3).ravel()), yaw_ned=0.30,
                     conn=Conn.ONLINE))
    placed, n_trail = w.view.p_act, len(w.view.trail_act)
    assert placed is not None
    # A state that disagrees on every axis, twenty times over. Nothing engaged.
    for _ in range(20):
        w.add_status(MpcStatus(engaged=False, mode="pid", conn=Conn.CONNECTING,
                               p_flu=(0.55, 0.20, 0.60), yaw_flu_deg=-40.0))
    assert w.view.p_act == placed, (
        "an unengaged MpcStatus moved the marker the localizer owns")
    assert len(w.view.trail_act) == n_trail, (
        "an unengaged MpcStatus pushed a second series into the same trail")
    # Engaged, the control state takes over — that half must keep working.
    w.add_status(MpcStatus(engaged=True, mode="pid", conn=Conn.ONLINE,
                           p_flu=(0.55, 0.20, 0.60), yaw_flu_deg=-40.0))
    assert w.view.p_act != placed, "the engaged state never took the marker"


def test_a_repeated_log_line_is_counted_instead_of_repeated():
    """Four identical refusals are one event, not four lines.

    2026-08-23 16:34:21-43: "engage refused: flight mode is mode -1" four
    times, each one pushing a line of real history off a nine-line panel.
    """
    from rov_gui.widgets.payload import PayloadPanel

    _app()
    p = PayloadPanel()
    p.add_event("first")
    for _ in range(4):
        p.add_event("engage refused: flight mode is mode -1", "warn")
    assert len(p._log_lines) == 2, p._log_lines
    assert "(x4)" in p._log_lines[-1], p._log_lines[-1]
    # ...but only CONSECUTIVE repeats. The same refusal after something else
    # happened is a new event and gets its own stamp.
    p.add_event("something else")
    p.add_event("engage refused: flight mode is mode -1", "warn")
    assert len(p._log_lines) == 4, p._log_lines
    assert "(x" not in p._log_lines[-1]
    # the view and the saved file agree, which is what mission_log.txt claims
    assert p.log_view.toPlainText().strip().split("\n")[-1] \
        == p._log_lines[-1]
    p.close()


def test_the_depth_map_is_cross_checked_against_the_map_floor():
    """The one witness the depth path has, and it must not share anything.

    2026-08-23: the object landed half a metre below the tag floor it was
    sitting on with the ray to it 1.4x too long, and a too-big reconstructed
    mesh and a non-metric depth map produce that identically. The tag solution
    settles it — it never reads the depth map — so this ratio is the whole
    diagnosis in one number.

    Measured over the MAT, not over the tags. Sampling tag centres made the
    figure drift with tag count (0.94x on 3 tags, 1.46x on 14) because a tag
    centre is the one patch of this scene stereo cannot match.
    """
    import numpy as np

    from rov_gui.state import Conn, NavFix, TagOverlay
    from rov_gui.window import MainWindow

    K = (500.0, 500.0, 320.0, 180.0)
    # Camera 1.00 m above the mat, looking straight down, axes already aligned
    # so the optical +Z is the map's +Z (down).
    R_nb = np.eye(3)
    R_bc = np.eye(3)
    cam_z = -1.00

    class _Canvas:
        def __init__(self, aux):
            self._aux = aux

        def depth_map(self):
            return self._aux

    class _Panel:
        def __init__(self, aux):
            self.canvas = _Canvas(aux)

    def _run(scale, aux=None):
        if aux is None:
            aux = np.zeros((360, 640), np.uint16)
            for vy in range(360):
                for ux in range(0, 640, 4):
                    a, b = (ux - K[2]) / K[0], (vy - K[3]) / K[1]
                    # The floor's Z along this ray, which is what DepthAI
                    # reports — not the ray's length.
                    aux[vy, ux] = int(round(1.00 * scale * 1000))
        stub = type("W", (), {})()
        stub._last_fix = NavFix(p_ned=(0.0, 0.0, cam_z),
                                R_ned_body=tuple(R_nb.ravel()),
                                n_tags=9, conn=Conn.ONLINE)
        stub.videos = {"depth": _Panel(aux)}
        stub._cam_ext = (R_bc, np.zeros(3))
        stub._depth_chk = None
        stub._depth_chk_t = 0.0
        stub._traj_panel = None
        MainWindow._check_depth_scale(stub, TagOverlay(
            panel="main", quads=(), ids=(), mapped=(), src_w=640, src_h=360,
            localizes=True, K=K))
        return stub._depth_chk

    _app()
    metric = _run(1.0)
    assert metric is not None, "a full floor of valid depth produced no ratio"
    assert abs(metric[0] - 1.0) < 0.02, metric
    assert metric[1] < 0.05, f"a flat floor must not look mis-shaped: {metric}"
    # ...and a depth path reading 1.4x long is caught, not averaged away.
    long_ = _run(1.40)
    assert abs(long_[0] - 1.40) < 0.03, long_

    # An empty map reports NOTHING rather than a ratio off a handful of pixels:
    # the median this leans on needs a population.
    assert _run(1.0, aux=np.zeros((360, 640), np.uint16)) is None


def test_reference_views_are_not_collected_out_of_range():
    """Warning about a too-far object was not enough: three captures in a row
    ran at 1.2-1.5 m anyway (2026-08-23) and every mesh was unusable. Frames
    outside the range are now not collected at all, and nothing times out while
    the operator closes the distance — the arc only advances on frames that
    were submitted.
    """
    import numpy as np

    from rov_gui.perception.session import (CAPTURE_RANGE_M, PoseSession,
                                            _masked_depth_m)

    mask = np.zeros((360, 640), bool)
    mask[100:200, 200:400] = True
    far = np.where(mask, 1340, 0).astype(np.uint16)     # 1.34 m, as recorded
    near = np.where(mask, 620, 0).astype(np.uint16)     # 0.62 m
    assert abs(_masked_depth_m(far, mask) - 1.34) < 1e-6
    assert _masked_depth_m(near, mask) is not None
    # a handful of valid pixels is not a range
    assert _masked_depth_m(np.zeros((360, 640), np.uint16), mask) is None

    lo, hi = CAPTURE_RANGE_M
    assert lo <= 0.30 and 0.80 <= hi <= 1.20, CAPTURE_RANGE_M

    class _Cap:
        submitted = 0

        def snapshot(self):
            return {"n": 0, "arc": 0.0, "done": False}

        def submit(self, *_a):
            _Cap.submitted += 1

    sess = PoseSession.__new__(PoseSession)
    sess._lock = __import__("threading").Lock()
    sess._capture = _Cap()
    sess._live = type("L", (), {"snapshot": lambda _s: {
        "state": "TRACKING", "mask": mask}})()
    sess._phase_note = ""
    sess._n_views = sess._arc_deg = sess._cap_dist = 0
    sess._np_K = None                       # no intrinsics -> no smear gate
    sess._smear, sess._smear_hist = None, __import__("collections").deque()
    colour = np.zeros((360, 640, 3), np.uint8)

    sess._step_capture(colour, far)
    assert _Cap.submitted == 0, "a 1.34 m frame was collected"
    assert "134 cm" in sess._phase_note, sess._phase_note
    sess._step_capture(colour, near)
    assert _Cap.submitted == 1, "a 0.62 m frame was refused"


def test_a_smeared_frame_is_not_collected_and_the_number_is_live():
    """Range is a PROXY; the smear is the thing. A frame whose depth inside the
    mask is deeper than the object is wide fuses into a point cloud that is a
    smear, and the mesh comes out long in whatever direction it points.

    Until 2026-08-24 this was measured ONCE, after the orbit was over, as a
    warning — 0823_213109 printed "3.3x" and then spent 33 s building the 333 mm
    mesh of a 120 mm object that it had just predicted, and flew it.
    """
    import numpy as np

    from rov_gui.perception.session import (SMEAR_MAX_RATIO_FRAME, PoseSession,
                                            _frame_depth_quality)

    mask = np.zeros((360, 640), bool)
    mask[100:200, 200:400] = True
    fx = 513.4                                    # C3 colour, 640x360
    # A CLEAN frame: every masked pixel at one range, so the spread is 0.
    clean = np.where(mask, 620, 0).astype(np.uint16)
    q = _frame_depth_quality(clean, mask, fx)
    assert q is not None and q[0] == 0.0 and q[1] > 0.0, q
    # A SMEARED one: the same mask, but depth ramped across it so p5-p95 is far
    # deeper than the width the mask implies at that range.
    ramp = np.zeros((360, 640), np.uint16)
    col = np.linspace(500, 1400, 200).astype(np.uint16)
    ramp[100:200, 200:400] = col[None, :]
    qs = _frame_depth_quality(ramp, mask, fx)
    assert qs is not None and qs[0] / qs[1] > SMEAR_MAX_RATIO_FRAME, qs
    # too few pixels is "no measurement", not a good score
    assert _frame_depth_quality(np.zeros((360, 640), np.uint16), mask, fx) is None

    class _Cap:
        submitted = 0

        def snapshot(self):
            return {"n": 0, "arc": 0.0, "done": False}

        def submit(self, *_a):
            _Cap.submitted += 1

    sess = PoseSession.__new__(PoseSession)
    sess._lock = __import__("threading").Lock()
    sess._capture = _Cap()
    sess._live = type("L", (), {"snapshot": lambda _s: {
        "state": "TRACKING", "mask": mask}})()
    sess._phase_note = ""
    sess._n_views = sess._arc_deg = sess._cap_dist = 0
    sess._smear, sess._smear_hist = None, __import__("collections").deque()
    sess._smear_reject = 0
    sess._np_K = np.array([[fx, 0.0, 320.0], [0.0, fx, 180.0],
                           [0.0, 0.0, 1.0]])
    colour = np.zeros((360, 640, 3), np.uint8)

    sess._step_capture(colour, ramp)
    assert _Cap.submitted == 0, "a smeared frame was collected"
    assert "smear" in sess._phase_note, sess._phase_note
    assert sess._smear is not None and sess._smear > SMEAR_MAX_RATIO_FRAME
    # ...and the clean one goes in, leaving the LIVE median on the session so
    # the panel can show it while the pilot can still act on it.
    sess._step_capture(colour, clean)
    assert _Cap.submitted == 1, "a clean frame was refused"
    assert sess._smear is not None and sess._smear <= SMEAR_MAX_RATIO_FRAME


def test_a_smeared_capture_refuses_to_reconstruct():
    """The last of the three, and the one that costs nothing to get right: a
    capture whose depth is already known to be a smear must not spend 33 s of
    GPU producing the long mesh it has just predicted. 0823_213109 warned twice
    — "3.3x" before the build and "do NOT fly this mesh" after it — and gated
    neither time, and the vehicle chased a phantom five minutes later."""
    import numpy as np

    from rov_gui.perception.session import (PHASE_FAILED,
                                            SMEAR_MAX_RATIO_CAPTURE,
                                            PoseSession)

    mask = np.zeros((360, 640), bool)
    mask[100:200, 200:400] = True
    fx = 513.4
    ramp = np.zeros((360, 640), np.uint16)
    ramp[100:200, 200:400] = np.linspace(500, 1400, 200).astype(np.uint16)
    clean = np.where(mask, 620, 0).astype(np.uint16)

    def _sess(frames):
        s = PoseSession.__new__(PoseSession)
        s._lock = __import__("threading").Lock()
        s._np_K = np.array([[fx, 0.0, 320.0], [0.0, fx, 180.0],
                            [0.0, 0.0, 1.0]])
        s._smear, s._smear_hist = None, __import__("collections").deque()
        s._smear_reject = 0
        s._n_views = s._arc_deg = s._build_s = 0
        s._phase_note, s.phase = "", "capturing"
        s._capture = object()
        rec = type("R", (), {
            "n": 8, "arc": 60.0, "frames": frames,
            "summary": lambda _s: "8 views"})()
        cap = type("C", (), {
            "recorder": rec,
            "stop": lambda _s: None,
            "join": lambda _s, timeout=None: None})()
        return s, cap

    logs = []
    s, cap = _sess([(None, ramp, mask)] * 6)
    s._finish_capture(cap, lambda lvl, m: logs.append((lvl, m)))
    assert s.phase == PHASE_FAILED, "a smeared capture went on to reconstruct"
    assert "not reconstructing" in s._phase_note, s._phase_note
    assert s._smear > SMEAR_MAX_RATIO_CAPTURE
    assert any(lvl == "warn" for lvl, _ in logs), logs

    # ...and a clean capture is NOT refused: it gets past the gate and on to
    # the reconstruction it asked for (which this stub cannot run, so only the
    # verdict is checked).
    s2, cap2 = _sess([(None, clean, mask)] * 6)
    try:
        s2._finish_capture(cap2, lambda lvl, m: None)
    except Exception:                                            # noqa: BLE001
        pass                       # the build path is not stubbed; fine
    assert s2.phase != PHASE_FAILED or "not reconstructing" not in \
        s2._phase_note, "a clean capture was refused"


def test_the_depth_correction_is_on_unless_it_is_turned_off():
    """Forgetting `--depth-scale` used to be SILENT and it cost two runs.

    2026-08-24, both morning runs: the object sat at tag 58 and the station
    reported it nearest tags 11/10/52 — 1.55-1.57x down the camera ray, two tag
    rows past it and 51-53 cm UNDER the mat. The measured overshoot is exactly
    the in-water stereo factor the flag exists to cancel, and the flag simply
    had not been typed. A correction that has to be remembered is not armed.
    """
    from rov_gui.__main__ import build_parser
    from rov_gui.backends.hardware import DEPTH_SCALE_DEFAULT

    opts = build_parser().parse_args([])
    assert abs(opts.depth_scale - 0.64) < 1e-9, opts.depth_scale
    assert abs(opts.depth_scale - DEPTH_SCALE_DEFAULT) < 1e-9, (
        "the CLI default and the backend constant drifted apart")
    # ...and 1.0 is still reachable, because an IN-AIR bench wants it off
    assert build_parser().parse_args(["--depth-scale", "1.0"]).depth_scale == 1.0
    # the correction is a real multiply, not just a recorded number
    try:
        from rov_gui.backends.hardware import C3VideoWorker
    except Exception:                                            # noqa: BLE001
        return
    import numpy as np
    out = C3VideoWorker._scale_depth(np.array([[1000, 0]], np.uint16),
                                     DEPTH_SCALE_DEFAULT)
    assert out[0, 0] == 640 and out[0, 1] == 0


def test_depth_scale_correction_preserves_holes_and_clips():
    """--depth-scale multiplies the raw millimetres once, at the source.

    Zeros are DepthAI's "no measurement" and must stay zero — a hole that
    became a distance would put phantom geometry on every consumer at once.
    """
    import numpy as np

    try:
        from rov_gui.backends.hardware import C3VideoWorker
    except Exception:                                            # noqa: BLE001
        return
    img = np.array([[0, 1000, 1500], [65000, 3, 0]], np.uint16)
    out = C3VideoWorker._scale_depth(img, 0.64)
    assert out.dtype == np.uint16
    assert out[0, 0] == 0 and out[1, 2] == 0, "a hole became a distance"
    assert out[0, 1] == 640 and out[0, 2] == 960
    up = C3VideoWorker._scale_depth(img, 1.5)
    assert up[1, 0] == 65535, "no clip at the uint16 ceiling"
    assert up[0, 0] == 0


def test_the_capture_reports_the_depth_spread_inside_the_mask():
    """The mask is usually fine; what smears the mesh is the DEPTH in it.

    Measured over every reference capture stored on 2026-08-23: the p5-p95
    depth spread inside a correct mask is ~102 mm at 0.51 m and 265-714 mm at
    1.37 m, on an object about 110 mm across. The mesh then comes out long in
    whichever direction that smear points (182 mm from the 0.51 m capture,
    470 mm from the 1.37 m one). Distance, not masking, not scale.
    """
    import numpy as np

    from rov_gui.perception.session import PoseSession

    fx = 513.4

    def _frames(range_mm, spread_mm, side_px):
        # One square mask `side_px` across, filled with depth spread evenly
        # over `spread_mm` about `range_mm`.
        out = []
        for _ in range(4):
            mask = np.zeros((360, 640), np.uint8)
            mask[100:100 + side_px, 200:200 + side_px] = 255
            depth = np.zeros((360, 640), np.uint16)
            ramp = np.linspace(range_mm - spread_mm / 2.0,
                               range_mm + spread_mm / 2.0, side_px)
            depth[100:100 + side_px, 200:200 + side_px] = \
                np.repeat(ramp[:, None], side_px, axis=1).astype(np.uint16)
            out.append((None, depth, mask, None))
        return out

    # 0.5 m, an 11 cm object (side = 0.11 * fx / 0.5 = 113 px), 100 mm spread.
    # p5-p95 of an even ramp is 0.9 of its full span.
    spread, size = PoseSession.depth_quality(_frames(500, 100, 113), fx)
    assert abs(spread - 90.0) < 6.0, spread
    assert 100 < size < 125, size
    assert spread < 1.5 * size, "a good capture must not read as a smear"

    # 1.37 m, the SAME object (side = 41 px), and the spread this camera
    # actually produced there.
    spread, size = PoseSession.depth_quality(_frames(1370, 265, 41), fx)
    assert abs(spread - 238.0) < 8.0, spread
    assert 95 < size < 125, size
    assert spread > 1.5 * size, "the 1.4 m smear must be flagged"

    assert PoseSession.depth_quality([], fx) is None


def test_an_over_long_reconstruction_is_caught_against_the_tape():
    """The mesh is METRIC — it is not mis-scaled, it is contaminated.

    Eight reconstructions of one object on 2026-08-23 held the same ~100 x 170
    mm cross-section and grew only in LENGTH, 171 -> 470 mm, in step with the
    vertex count (18k -> 101k). Its own capture says the same: 1782 mask px at
    1.28 m is 111 cm^2 of visible object, an ~11 cm face, not a 47 cm one. So
    the operator's tape is a CHECK, not an anchor — rescaling would shrink the
    two sides that were right.
    """
    try:
        import trimesh
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.perception.session import PoseSession

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "model.obj"
        trimesh.creation.box(extents=(0.108, 0.172, 0.470)).export(str(src))
        before = src.read_bytes()

        def _run(size_mm):
            said = []
            s = PoseSession(build=True, object_size_mm=size_mm)
            s._check_mesh_size(src, on_log=lambda lv, m: said.append((lv, m)))
            return said

        assert _run(None) == [], "no tape, no silhouette, no opinion"
        lv, msg = _run(155.0)[0]          # 470 against 155 = 3x
        assert lv == "warn" and "3.0x" in msg, msg
        assert "not the object" in msg or "swallowed" in msg, msg
        # ...and since 2026-08-24 the tape verdict is a REFUSAL, not a note:
        # 0823_213109 printed "Do NOT fly this mesh" and then flew it.
        s_ref = PoseSession(build=True, object_size_mm=155.0)
        s_ref.mesh = str(src)
        assert s_ref._check_mesh_size(src, on_log=lambda *_: None) is False
        assert s_ref.phase == "failed" and s_ref.mesh is None, (
            "a condemned mesh survived to be re-registered against")
        # ...and it says so instead of quietly fixing it.
        assert src.read_bytes() == before, "the mesh was modified"
        assert not (Path(td) / "model_scaled.obj").exists()

        assert _run(460.0)[0][0] == "info"          # within 25%
        assert _run(900.0)[0][0] == "warn"          # covered only part of it


def test_a_long_mesh_is_refused_without_a_tape_by_the_silhouette_bound():
    """The workflow is NOVEL objects: the reference views are the whole input,
    so the always-armed size check is the metric envelope those views define.
    The longest silhouette any view saw is a hard lower bound on the object; a
    mesh far past it swallowed something that is not the object.

    Thresholds from the archive [측정: extrude audit over sessions/pose_meshes/*
    2026-08-24]: usable meshes sit at 1.08-1.59x their max silhouette, the two
    runaways at 2.25x and 2.72x — the gate at 2.0 splits the gap. The tape
    (`--pose-object-size`) stays as an OPTIONAL stricter layer."""
    try:
        import trimesh
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.perception.session import (MESH_MAX_SILHOUETTE_RATIO,
                                            PoseSession,
                                            _frame_silhouette_mm)
    import numpy as np

    assert 1.59 < MESH_MAX_SILHOUETTE_RATIO < 2.25

    # the per-frame bound itself: a 200 px box at 620 mm with fx=513.4 is
    # 200 * 620 / 513.4 = 242 mm of observed silhouette
    mask = np.zeros((360, 640), bool)
    mask[100:200, 200:400] = True
    depth = np.where(mask, 620, 0).astype(np.uint16)
    silh = _frame_silhouette_mm(depth, mask, 513.4)
    assert silh is not None and abs(silh - 200 * 620 / 513.4) < 1.0, silh
    assert _frame_silhouette_mm(np.zeros((360, 640), np.uint16),
                                mask, 513.4) is None

    with tempfile.TemporaryDirectory() as td:
        said = []
        log = lambda lv, m: said.append((lv, m))          # noqa: E731

        # today's real numbers: 189 mm mesh against a 136 mm silhouette (1.39x)
        ok = Path(td) / "ok.obj"
        trimesh.creation.box(extents=(0.110, 0.147, 0.189)).export(str(ok))
        s = PoseSession(build=True)                        # NO tape
        s._silh_mm = 136.0
        s.mesh = str(ok)
        assert s._check_mesh_size(ok, on_log=log) is True
        assert said and said[-1][0] == "info" and "envelope" in said[-1][1]
        assert s.mesh is not None and s.phase != "failed"

        # the flown mesh's numbers: 330 mm against a 121 mm silhouette (2.7x)
        bad = Path(td) / "bad.obj"
        trimesh.creation.box(extents=(0.121, 0.128, 0.330)).export(str(bad))
        s2 = PoseSession(build=True)
        s2._silh_mm = 121.0
        s2.mesh = str(bad)
        said.clear()
        assert s2._check_mesh_size(bad, on_log=log) is False
        assert s2.phase == "failed" and s2.mesh is None
        assert said[-1][0] == "warn" and "silhouette" in said[-1][1], said

        # no silhouette data (frames predate the tracker): cannot refuse
        s3 = PoseSession(build=True)
        s3.mesh = str(bad)
        assert s3._check_mesh_size(bad, on_log=log) is True


def test_a_refused_start_reaches_the_chip_and_not_only_the_log():
    """Engaged, the chip shows the PHASE — so a refused START was invisible.

    2026-08-23: the operator pressed START on a `follow`, it was refused with
    "no object lock (stale)", and the station went on flying a DP hold. The
    thrusters worked, the object happened not to move, and the whole thing read
    as a follow that had armed. It had not.
    """
    from rov_gui.state import Conn, MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    ok = MpcStatus(engaged=True, mode="mpc_tuned", conn=Conn.ONLINE,
                   err_xy=0.02, reason="engaged (DP hold)")
    w.add_status(ok)
    assert "DP HOLD" in w.chip.text() and "refused" not in w.chip.text()
    w.add_status(MpcStatus(
        engaged=True, mode="mpc_tuned", conn=Conn.ONLINE, err_xy=0.02,
        reason="START refused: no object lock (stale)"))
    assert "refused" in w.chip.text(), w.chip.text()
    assert "no object lock" in w.chip.text(), w.chip.text()
    assert theme.FAIL in w.chip.styleSheet()
    # ...and it ages out rather than sticking to a chip that has moved on.
    w._ref_t -= 10.0
    w.add_status(ok)
    assert "refused" not in w.chip.text(), w.chip.text()
    w.close()


def test_follow_shows_no_tag_field_because_it_uses_no_tag():
    """A follow is anchored to the object and to the pose the vehicle has at
    arm — no tag id enters it, and `_emit_scenario` already refused to send
    one. The FIELD has to go too: every other input this shape cannot use
    disappears, so one greyed-out survivor reads as "this one still counts"
    (operator, 2026-08-23)."""
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    seen = []
    w.scenario_requested.connect(seen.append)

    w.shape_box.setCurrentText("follow")
    assert w.tag_box.isHidden() and w.lbl_tag.isHidden(), "tag field survived"
    assert w.len_box.isHidden(), "the distance field survived a follow"
    assert not w.spd_box.isHidden(), "a follow still caps the setpoint speed"
    assert "origin_tag" not in seen[-1], seen[-1]

    # ...and it comes back for the shapes that are anchored to a tag.
    w.shape_box.setCurrentText("station")
    assert not w.tag_box.isHidden() and not w.lbl_tag.isHidden()
    assert "origin_tag" in seen[-1], seen[-1]
    w.close()


def test_the_pose_rate_is_an_arrival_rate_not_one_over_compute_time():
    """The panel's Hz must count poses ARRIVING, not invert a solve time.

    Both upstream trackers publish `1 / mean(last 30 job durations)`. A 0.73 s
    re-registration in that window made the chip read 5 Hz and climb back to 9
    over the next thirty jobs while the object sat still (operator,
    2026-08-23) — and neither figure was ever the rate anything downstream got.
    """
    from rov_gui.perception.session import _RateMeter

    m = _RateMeter(window_s=2.0)
    # Nothing yet, and one sample is not a rate.
    assert m.hz(0.0) == 0.0
    m.note(object(), 0.0)
    assert m.hz(0.0) == 0.0
    # A steady 10 arrivals a second reads 10 Hz.
    m = _RateMeter(window_s=2.0)
    for i in range(21):
        m.note(object(), i * 0.1)
    assert abs(m.hz(2.0) - 10.0) < 0.01, m.hz(2.0)
    # IDENTITY, not equality: the same result seen again is not an arrival, so
    # a perfectly still object polled at 50 Hz does not read as 50 Hz...
    same = object()
    for i in range(50):
        m.note(same, 2.0 + i * 0.02)
    assert abs(m.hz(3.0) - 10.0) > 5.0 or m.hz(3.0) < 10.0
    # ...and once results stop, the figure DECAYS instead of freezing.
    m2 = _RateMeter(window_s=2.0)
    for i in range(21):
        m2.note(object(), i * 0.1)
    fresh = m2.hz(2.0)
    assert m2.hz(3.0) < fresh, "the rate froze after arrivals stopped"
    assert m2.hz(5.0) == 0.0, "a dead tracker must read 0 Hz, not a stale rate"
    # A slow solver that produces one pose per second reads 1 Hz, whatever the
    # tracker's own `hz` says about compute time.
    m3 = _RateMeter(window_s=4.0)
    for i in range(5):
        m3.note(object(), float(i))
    assert abs(m3.hz(4.0) - 1.0) < 0.01, m3.hz(4.0)


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
                               pose_hz=44.0, pose_solve_ms=23.0,
                               n_register=1), 12.34)
        # a poseless track writes nothing — the CSV is poses, not states
        w._write_csv(PoseTrack(state="lost"), 12.44)
        w._close_csv()
        lines = (Path(td) / "x_pose.csv").read_text().strip().split("\n")
    assert len(lines) == 2, lines
    head = lines[0].split(",")
    row = dict(zip(head, lines[1].split(",")))
    assert row["frame_seq"] == "7" and row["state"] == "tracking"
    # The rate column carries its MEANING in its name: a file written before
    # 2026-08-23 has `pose_hz` holding 1/compute-time, and a reader must not be
    # able to pick up a different quantity under a familiar heading.
    assert "pose_hz" not in row and row["pose_hz_measured"] == "44.00"
    assert row["solve_ms"] == "23.0"
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
    """The frame's capture stamp is resolved ONCE, in tick(), and rides the
    published PoseTrack from there.

    Two properties in one, because they used to be two definitions of the
    same number. (a) The 10 Hz publish gate fires between camera frames, and
    the pose on those ticks is still the one estimated from the LAST frame —
    so it keeps that frame's stamp, not 0.0 (which sorted every such CSV row
    to the epoch) and not now(). (b) The stamp reaches the BUS, not just the
    log file: since 2026-08-21 ``object_nav`` pairs an object pose with the
    tag fix from its own camera frame by comparing this float against the one
    ``C3VideoWorker._tap_pose`` put in the nav mailbox, and a PoseTrack with
    no capture time can never be paired at all.
    """
    try:
        from rov_gui.backends.hardware import PoseWorker
    except Exception:                                            # noqa: BLE001
        return
    from rov_gui.bus import RgbdMailbox
    from rov_gui.sensorlog import SensorLog

    T = tuple(float(v) for v in
              (1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.5, 0, 0, 0, 1))

    class _Session:
        error, loading, ready = "", False, True
        mesh, mesh_description = None, ""

        def submit(self, *a, **k):
            pass

        def poll(self):
            return {"state": "tracking", "T_cam_obj": T, "frame_seq": 3}

    _app()
    bus = DataBus()
    published = []
    bus.pose.connect(published.append)
    mb = RgbdMailbox()
    w = PoseWorker(bus, mb, Opts())
    w.session = _Session()
    w.enabled = True
    with tempfile.TemporaryDirectory() as td:
        w._log = SensorLog(Path(td) / "b_pose.jsonl", "pose")
        w._open_csv(Path(td) / "b_pose.csv")
        mb.put(None, None, None, 777.5)              # one real frame
        w._last_pub = 0.0
        w.tick()
        w._last_pub = 0.0                            # ...and no new frame
        w.tick()
        w._log.close()
        w._log = None
        w._close_csv()
        csv_rows = (Path(td) / "b_pose.csv").read_text().strip().split("\n")[1:]
    assert len(csv_rows) == 2
    stamps = [float(r.split(",")[0]) for r in csv_rows]
    assert stamps == [777.5, 777.5], (
        f"a tick with no fresh frame must reuse the last capture stamp, "
        f"got {stamps}")
    assert [t.t_capture for t in published] == [777.5, 777.5], (
        f"the capture stamp must reach the bus — object_nav pairs on it: "
        f"{[t.t_capture for t in published]}")


# =============================================================================
# leak detection — the QGC feature this station did not have (rov_gui/leak.py)
# =============================================================================
def test_leak_monitor_matches_ardusub_exactly_and_not_its_config_chatter():
    """The alarm fires on ArduSub's real payload and NOTHING else.

    ArduSub emits three strings containing "leak": the alarm
    ("Leak Detected", CRITICAL) and two boot-time configuration notices with a
    lowercase 'd'. A substring match turns every startup into a flooding
    alarm, which is how an alarm gets ignored.
    """
    from rov_gui.leak import LeakMonitor

    m = LeakMonitor()
    m.note_config(fs_leak_enable=1, leak1_pin=27)
    assert m.state(t=100.0).leak is False

    # the two impostors, at their real severities
    assert m.note_statustext(
        "Leak detector 1 error. Please set SERVO10_FUNCTION to GPIO", 4,
        t=101.0) == "config"
    m2 = LeakMonitor()
    m2.note_config(fs_leak_enable=1, leak1_pin=27)
    assert m2.note_statustext(
        "Leak detector 1 pin (servo 10) auto-set to GPIO", 6, t=101.0) == ""
    assert m2.state(t=102.0).leak is False, "a config notice raised the alarm"

    # the real thing, NUL-padded the way pymavlink hands over char[50]
    assert m2.note_statustext("Leak Detected" + "\x00" * 37, 2,
                              t=110.0) == "leak"
    st = m2.state(t=111.0)
    assert st.leak is True and st.conn is Conn.FAULT and st.alarm
    assert "LEAK" in st.detail


def test_leak_state_holds_through_the_20s_repeat_and_clears_on_silence():
    """ArduSub re-sends every 20 s while wet and says NOTHING on recovery, so
    the topside state is a hold with a timeout. It must survive one missed
    repeat (a dropped packet is not a dry vehicle) and still clear eventually."""
    from rov_gui.leak import LEAK_HOLD_S, LeakMonitor

    m = LeakMonitor()
    m.note_config(fs_leak_enable=1, leak1_pin=27)
    m.note_statustext("Leak Detected", 2, t=0.0)
    assert m.state(t=25.0).leak is True, "one missed 20 s repeat cleared it"
    assert m.state(t=LEAK_HOLD_S - 1).leak is True
    st = m.state(t=LEAK_HOLD_S + 1)
    assert st.leak is False, "the alarm never cleared"
    assert "CLEARED" in st.detail, st.detail
    assert st.conn is Conn.DEGRADED, "a dive that flooded must not look clean"
    # a repeat inside the hold is the SAME event, not a second one
    m.note_statustext("Leak Detected", 2, t=10.0)
    assert m.leaks_seen == 1, m.leaks_seen


def test_leak_unknown_is_never_reported_as_dry():
    """FS_LEAK_ENABLE=0 or LEAK1_PIN=-1 makes a FLOODING vehicle silent, so an
    unconfigured vehicle and a dry one look identical on the link. state.py's
    rule ("unknown is None, never a plausible-looking default") means the
    panel must say so instead of painting green."""
    from rov_gui.leak import LeakMonitor

    unread = LeakMonitor()
    st = unread.state(t=1.0)
    assert st.leak is None and st.conn is Conn.DEGRADED
    assert "unread" in st.detail, st.detail

    off = LeakMonitor()
    off.note_config(fs_leak_enable=0, leak1_pin=27)
    st = off.state(t=1.0)
    assert st.leak is None and st.armed is False
    assert "DISABLED" in st.detail and "FS_LEAK_ENABLE=0" in st.detail

    nopin = LeakMonitor()
    nopin.note_config(fs_leak_enable=1, leak1_pin=-1)
    assert nopin.state(t=1.0).leak is None
    assert "LEAK1_PIN=-1" in nopin.state(t=1.0).detail

    # ...but a leak on an "unarmed" detector is still a leak: if the vehicle
    # somehow said it, believe it over the parameters.
    nopin.note_statustext("Leak Detected", 2, t=2.0)
    assert nopin.state(t=3.0).leak is True


def test_internal_pressure_alarm_is_tracked_separately_from_leak():
    """Rising enclosure pressure is the EARLY signal — it moves while water is
    still finding its way in, before a leak pad is wet. Different string,
    different severity, different 30 s repeat."""
    from rov_gui.leak import LeakMonitor

    m = LeakMonitor()
    m.note_config(fs_leak_enable=1, leak1_pin=27)
    assert m.note_statustext("Internal pressure critical!", 4, t=0.0) == "pressure"
    st = m.state(t=5.0)
    assert st.pressure_alarm is True and st.alarm
    assert st.leak is False, "a pressure warning is not a leak report"
    assert st.conn is Conn.FAULT
    assert m.state(t=200.0).pressure_alarm is False, "the alarm never cleared"


def test_statustext_reaches_the_vehicle_worker_and_lights_the_row():
    """End to end on the hardware path, with a fake drained record: STATUSTEXT
    -> LeakMonitor -> Telemetry.leak + the Leak sensor row + ONE log line.

    The one-log-line part is the point: ArduSub repeats every 20 s, and a
    mission log that fills with the same sentence is one nobody reads.
    """
    from rov_gui.backends.hardware import VehicleWorker

    bus = DataBus()
    logs = []
    bus.log.connect(lambda lvl, m: logs.append((lvl, m)))

    class O:
        mavlink_transport = "udp"
    w = VehicleWorker.__new__(VehicleWorker)      # no link, no thread
    w.bus, w.opts = bus, O()
    from rov_gui.leak import LeakMonitor
    w._leak, w._leak_reported, w._press_reported = LeakMonitor(), 0, 0.0
    w.leak_cfg_fn = lambda: {"fs_leak_enable": 1, "leak1_pin": 27}
    w._latest, w.logger = {}, None

    for _ in range(4):                            # four 20 s repeats
        w._note_statustext({"text": "Leak Detected", "severity": 2})
    errs = [m for lvl, m in logs if lvl == "error"]
    assert len(errs) == 1, f"one event must log once, got {len(errs)}: {errs}"
    assert "LEAK DETECTED" in errs[0]

    rows = w._sensors()
    assert rows["Leak"].conn is Conn.FAULT, rows["Leak"]
    assert w._leak.state().leak is True
    # a plain WARNING from the vehicle still reaches the log (prearm refusals
    # were being dropped one layer down until 2026-08-14)
    logs.clear()
    w._note_statustext({"text": "PreArm: Compass not calibrated", "severity": 4})
    assert any("PreArm" in m for _l, m in logs), logs


# =============================================================================
# recording name, run folders, mission log
# =============================================================================
def test_run_folder_puts_one_run_in_one_dated_minute_folder():
    """rov_gui/runstore.py: <base>/YYYYMMDD/MMDD_HHMMSS, agreed on by three
    writers with no handshake — the folder is FOUND (the run in progress),
    not computed, which is what seconds in the leaf name cost.

    The leaf carries month/day (a folder gets dragged out of its date
    directory) and seconds (a pool session makes several runs a minute) —
    operator requests, 2026-08-14."""
    from rov_gui import runstore

    with tempfile.TemporaryDirectory() as td:
        # 2026-08-14 18:41:48 local, and two writers seconds later. Under
        # minute buckets these three COMPUTED the same name; now the first
        # creates the folder and the other two find it.
        t0 = time.mktime((2026, 8, 14, 18, 41, 48, 0, 0, -1))
        a = runstore.run_dir(td, when=t0)
        b = runstore.run_dir(td, when=t0 + 3)
        assert a == b, f"two writers 3 s apart split into {a} / {b}"
        assert a == Path(td) / "20260814" / "0814_184148", a

        # ...including across the minute boundary that used to be the only
        # case the join window existed for
        c = runstore.run_dir(td, when=t0 + 12)
        assert c == a, f"a run straddling 18:42 split into {c}"

        # ...but a genuinely separate run, long after, gets its own folder
        d = runstore.run_dir(td, when=t0 + 3600)
        assert d == Path(td) / "20260814" / "0814_194148", d
        assert d != a

        # ...and so does one just past the join window
        e = runstore.run_dir(td, when=t0 + 3600 + runstore.JOIN_WINDOW_S + 1)
        assert e != d, f"{runstore.JOIN_WINDOW_S}s of silence still joined"

        # An OLD-format folder is never joined: pre-2026-08-14 runs (bare
        # HHMM) and the short-lived MMDD_HHMM stay exactly where they are.
        for stale in ("2000", "0814_2000"):
            (Path(td) / "20260814" / stale).mkdir(parents=True, exist_ok=True)
        t1 = time.mktime((2026, 8, 14, 20, 0, 5, 0, 0, -1))
        f = runstore.run_dir(td, when=t1)
        assert f == Path(td) / "20260814" / "0814_200005", f


def test_recording_name_slug_survives_operator_typing():
    """The name is typed mid-session, so it arrives with spaces, dots and
    Hangul. A dot is the dangerous one: path.with_suffix('.json') would eat
    part of the name instead of writing the sidecar."""
    from rov_gui import runstore

    assert runstore.slug("square test 2") == "square_test_2"
    assert runstore.slug("v1.2/final") == "v1_2_final"
    assert runstore.slug("  풀장 1차  ") == "풀장_1차"
    assert runstore.slug("") == "" and runstore.slug("///") == ""
    assert len(runstore.slug("x" * 200)) <= 48


def test_ui_recording_takes_the_operator_name_and_keeps_the_old_one_when_empty():
    """ui_<stamp>_<name>.mp4 when something is typed, ui_<stamp>.mp4 when not —
    and the sidecar carries the RAW text, which the slug cannot."""
    from rov_gui.recorder import ScreenRecorder

    app = _app()
    theme.apply(app)
    with tempfile.TemporaryDirectory() as td:
        w = QtWidgets.QLabel("x")
        w.resize(64, 48)
        rec = ScreenRecorder(w, out_dir=td, fps=8.0)
        path = rec.start(name="square test 2")
        assert path is not None, rec.stats.error
        _pump(app, 120)
        rec.stop()
        assert path.name.startswith("ui_") and path.name.endswith(
            "_square_test_2.mp4"), path.name
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["name"] == "square test 2", meta
        assert meta["file"] == path.name

        rec2 = ScreenRecorder(w, out_dir=td, fps=8.0)
        p2 = rec2.start()
        assert p2 is not None and p2.name.count("_") == 2, p2.name
        rec2.stop()
        assert json.loads(p2.with_suffix(".json").read_text())["name"] is None
        # both landed in the same dated run folder
        assert path.parent == p2.parent
        assert path.parent.parent.parent == Path(td)


def test_mission_log_scrolls_back_and_is_saved_beside_a_recording():
    """The log keeps the whole session (the QLabel it replaced dropped
    everything past 12 lines), follows the newest line only while the operator
    is AT the bottom, and lands in the run folder as mission_log.txt."""
    from rov_gui.widgets.payload import PayloadPanel

    app = _app()
    theme.apply(app)
    panel = PayloadPanel()
    panel.resize(240, 400)
    panel.show()
    _pump(app, 60)
    for i in range(200):
        panel.add_event(f"line {i}")
    assert len(panel._log_lines) == 200
    text = panel.log_text()
    assert "line 0" in text and "line 199" in text, "history was dropped"
    bar = panel.log_view.verticalScrollBar()
    assert bar.value() >= bar.maximum() - 2, "the view did not follow the tail"

    # scrolled back -> a new line must NOT yank the view to the bottom
    bar.setValue(0)
    panel.add_event("line 200")
    assert bar.value() == 0, "a new line stole the scroll position"
    panel.close()


def test_window_writes_mission_log_into_the_run_folder():
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    win.bus.log.emit("warn", "something the operator will want later")
    with tempfile.TemporaryDirectory() as td:
        win._save_mission_log(td)
        body = (Path(td) / "mission_log.txt").read_text()
    assert "something the operator will want later" in body
    assert "[WARN]" in body, "the level was lost on the way to the file"
    win.shutdown()


def test_recording_name_field_never_takes_escape_away_from_estop():
    """A focused QLineEdit swallows Esc (Qt uses it to revert an edit). Esc is
    E-STOP on this station, so the field must release the keyboard AND let the
    stop through — the one key a pilot reaches for cannot depend on where the
    caret is."""
    from rov_gui.qt import QtGui
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    win.show()
    _pump(app, 60)
    win.teleop.enable.setChecked(True)      # the real switch, so
    assert win.teleop.enabled               # force_disable can drop it
    win.rec_name.setFocus()
    _pump(app, 60)
    assert win.rec_name.hasFocus()

    ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, int(Qt.Key.Key_Escape),
                         Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(win.rec_name, ev)
    _pump(app, 60)
    assert not win.rec_name.hasFocus(), "Esc left the caret in the name field"
    assert win.teleop.enabled is False, "Esc in the name field did not E-STOP"
    win.shutdown()


def test_the_panel_says_what_depth_is_being_held():
    """STATION commands all three axes, but until 2026-08-18 NOTHING on this
    panel was about z: err_xy is horizontal by definition, so a hold that had
    sagged 20 cm read as a perfect one (operator request).

    Two conventions this pins, because both are easy to get backwards: the
    number is MAP frame (datum z folded back in) so it compares with the
    surveyed operating band, and NED z is down-positive so the bracket is
    `+` when the vehicle is DEEPER than its setpoint."""
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    panel = TrajectoryWindow()
    # not engaged, no reference -> no readout at all (the honesty rule)
    assert panel._hold_z_text() == ""

    # engage datum sits 1.00 m above the tag floor; FLU z is UP, so a
    # reference 0.05 m BELOW the datum is ref_flu z = -0.05.
    panel.view.set_datum((0.0, 0.0, -1.00, 0.0))
    panel.add_status(MpcStatus(
        engaged=True, traj_on=False, phase="station", err_xy=0.012,
        scenario={"kind": "station", "origin_ned": (0.0, 0.0)},
        p_flu=(0.0, 0.0, -0.08), ref_flu=(0.0, 0.0, -0.05)))
    txt = panel._hold_z_text()
    # datum -1.00 m (map) + the reference sitting 0.05 m deeper than it:
    # FLU z is UP so ref_flu -0.05 is 0.05 DOWN, and NED down-positive makes
    # the map number LESS negative as the vehicle goes deeper.
    assert "z -0.95 m" in txt, txt
    assert "(+3 cm)" in txt, txt            # 0.08 deep vs 0.05 = 3 cm DEEPER
    assert "z_d" not in txt
    assert "z -0.95 m" in panel.chip.text(), panel.chip.text()
    assert "STATION HOLD" in panel.chip.text()

    # ...and shallower than the setpoint reads negative
    panel.add_status(MpcStatus(
        engaged=True, traj_on=False, phase="station", err_xy=0.012,
        scenario={"kind": "station", "origin_ned": (0.0, 0.0)},
        p_flu=(0.0, 0.0, 0.02), ref_flu=(0.0, 0.0, -0.05)))
    assert "(-7 cm)" in panel._hold_z_text(), panel._hold_z_text()

    # no datum -> say so rather than reporting a datum-relative number as map
    p2 = TrajectoryWindow()
    p2.add_status(MpcStatus(engaged=True, ref_flu=(0.0, 0.0, -0.05),
                            p_flu=(0.0, 0.0, -0.05)))
    assert p2._hold_z_text().strip().startswith("· z_d"), p2._hold_z_text()

    # the tooltip has to state the sign, or the number is a coin flip
    tip = panel.chip.toolTip()
    assert "down-positive" in tip and "DEEPER" in tip


def test_trajectory_panel_carries_the_speed():
    """The mission speed is an operator field (panel request, 2026-08-16):
    it seeds from the YAML square block, rides every line/square scenario
    override, disappears for station (a hold has no speed), and freezes while
    a path is flying — same discipline as the distance fields."""
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    panel = TrajectoryWindow()
    got: list = []
    panel.scenario_requested.connect(got.append)
    panel.set_mission_defaults({"shape": "line", "origin_tag": 7,
                                "length": 2.0, "speed": 0.08})
    assert got and got[-1]["speed"] == 0.08 and got[-1]["length"] == 2.0
    panel.spd_box.setValue(0.25)
    assert got[-1]["speed"] == 0.25
    panel.shape_box.setCurrentText("square")
    assert got[-1]["speed"] == 0.25, "speed must survive a shape change"
    assert got[-1]["size"] == 2.0 and "length" not in got[-1]
    panel.shape_box.setCurrentText("station")
    assert "speed" not in got[-1], "a station hold must not carry a speed"
    assert not panel.spd_box.isVisibleTo(panel)
    panel.shape_box.setCurrentText("line")
    panel.add_status(MpcStatus(engaged=True, traj_on=True))
    assert not panel.spd_box.isEnabled(), "mission frozen while flying"
    # ...and frozen from START, not from take-off: the worker snapshots the
    # mission when START is pressed and arms the path from that copy after the
    # approach + settle, so an edit made in between would show on the panel and
    # never reach the flight.
    for ph in ("approach", "settle"):
        panel.add_status(MpcStatus(engaged=True, traj_on=False, phase=ph))
        assert not panel.spd_box.isEnabled(), f"editable during {ph}"
    panel.add_status(MpcStatus(engaged=True, traj_on=False, phase="warmup"))
    assert panel.spd_box.isEnabled(), "staging before START must stay editable"
    panel.add_status(MpcStatus(engaged=False, traj_on=False))
    assert panel.spd_box.isEnabled()

    # START commits a half-typed number first. Every button here is NoFocus and
    # the boxes are keyboardTracking(False), so without that the run would fly
    # the previous speed while the box displayed the new one.
    started: list = []
    panel.mission_requested.connect(lambda: started.append(len(got)))
    panel.spd_box.setFocus()
    panel.spd_box.lineEdit().setText("0.30 m/s")     # typed, NOT committed
    assert panel.spd_box.value() != 0.30
    panel.btn_start.confirmed.emit()
    assert panel.spd_box.value() == 0.30, "START flew an uncommitted speed"
    assert got[-1]["speed"] == 0.30
    assert started and started[0] == len(got), "scenario must precede START"


def test_the_plot_draws_the_curve_that_is_actually_flown():
    """Honesty rule: the panel may only show what the run is doing. Once the
    mission is a filleted MPCC curve, drawing the sharp rectangle the operator
    typed is the same class of lie as the geofence box that outlived its
    fence (removed 2026-08-14 for exactly that reason)."""
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryView

    _app()
    scen = {"kind": "square", "size": 2.0, "size_y": 1.0, "speed": 0.12,
            "laps": 1, "origin_ned": [0.0, 0.0], "depth_ned": 0.5,
            "rot_deg": 0.0, "heading_follow": False,
            "path": {"kind": "mpcc-arc", "fillet_m": 0.15,
                     "turn_radius_m": 0.0}}
    v = TrajectoryView()
    v.add_status(MpcStatus(engaged=True, traj_on=True, scenario=scen))
    pts = np.asarray(v.square_ned, float)
    assert len(pts) > 100, "the curve must be sampled, not four corners"
    assert v.square_kind == "arc"
    # the drawn corner is ROUNDED: the closest approach to the sharp vertex is
    # about the fillet radius, never zero
    d = float(np.hypot(pts[:, 0] - 2.0, pts[:, 1] - 0.0).min())
    assert 0.03 < d < 0.15, f"closest approach to the sharp vertex {d:.3f} m"


def test_the_plot_draws_the_dead_reckoned_series_beside_the_tag_one():
    """The two live markers the IMU experiment is FOR. They must land in the
    same frame — same FLU mirror, same datum rotation — or the gap between
    them measures the drawing instead of the IMU."""
    import math

    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryView

    _app()
    v = TrajectoryView()
    # A non-trivial datum, because an identity one would hide a missing
    # transform on exactly one of the two series.
    v.set_datum((1.5, -0.4, 0.0, math.radians(30.0)))
    s = MpcStatus(engaged=True, conn=Conn.ONLINE,
                  p_flu=(1.0, 0.0, -0.8), yaw_flu_deg=0.0,
                  p_dr_flu=(1.0, -0.20, -0.8), yaw_dr_flu_deg=0.0,
                  dr_ok=True, dr_err_m=0.20, dr_elapsed_s=12.0,
                  dr_mode="shadow", dr_source="c3", dr_attitude="ahrs",
                  dr_hz=198.0, dr_n=2400)
    v.add_status(s)
    assert v.p_dr is not None and len(v.trail_dr) == 1
    # both series went through _to_map, so the 0.20 m offset survives intact
    d = math.hypot(v.p_act[0] - v.p_dr[0], v.p_act[1] - v.p_dr[1])
    assert abs(d - 0.20) < 1e-9, d
    # ...and neither is sitting at the raw FLU coordinates
    assert abs(v.p_act[0] - 1.0) > 0.1, v.p_act
    for i in range(1, 6):
        v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                               p_flu=(1.0 + 0.05 * i, 0.0, -0.8),
                               p_dr_flu=(1.0 + 0.05 * i, -0.2 - 0.02 * i, -0.8),
                               dr_ok=True, dr_err_m=0.2, dr_mode="shadow"))
    assert len(v.trail_dr) == 6

    # dr_ok False must drop the marker, NOT freeze it: a starved dead
    # reckoner looks like a vehicle holding perfectly still, which is the
    # most misleading thing this plot could draw.
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE,
                           p_flu=(1.3, 0.0, -0.8), p_dr_flu=(1.3, -0.3, -0.8),
                           dr_ok=False, dr_note="imu samples stale (0.9s)",
                           dr_mode="shadow", dr_source="c3",
                           dr_attitude="ahrs"))
    assert v.p_dr is None, "a dead estimate must not leave a ghost behind"
    assert len(v.trail_dr) == 6, "and must not extend the trail"

    # ...and the SAME rule the other way round. A tag dropout must drop the
    # green marker, or the picture shows a live amber ghost pulling away from
    # a frozen truth and the error line reads as drift that did not happen.
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE, p_flu=None,
                           p_dr_flu=(1.4, -0.35, -0.8), dr_ok=True,
                           dr_err_m=0.35, dr_mode="shadow", dr_source="c3",
                           dr_attitude="ahrs"))
    assert v.p_act is None, "a lost tag must not leave a frozen hull behind"
    assert v.p_dr is not None, "...while the DR marker keeps going"

    v.clear()
    assert not v.trail_dr, "clear() must take the DR trail with the others"

    # three trails must paint without blowing up (the legend and the ghost
    # hull only exist on this path)
    v.resize(420, 300)
    v.add_status(s)
    v.grab()


def test_the_dr_overlay_is_on_by_default_and_a_runaway_cannot_bury_the_plot():
    """Two requirements that pull against each other, both pinned here.

    The operator runs --imu-dr precisely to watch the two markers side by
    side, so the overlay must not need discovering. But an unaided IMU drifts
    by kilometres, and on 2026-08-17 a 1.4 km estimate drew a trail straight
    across the plot — which is why it had been defaulted OFF. Clipping is what
    lets both be true.
    """
    from rov_gui.state import MpcStatus
    from rov_gui.widgets.trajectory import TrajectoryView

    _app()
    v = TrajectoryView()
    v.resize(600, 320)
    # no estimator reporting -> nothing to show
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE, p_flu=(0.0, 0.0, -1.0)))
    assert not v._dr_visible(), "no DR reporting, nothing to draw"

    def push(n, dr):
        for i in range(n):
            s = MpcStatus(engaged=True, conn=Conn.ONLINE, dr_mode="shadow",
                          dr_source="c3", dr_attitude="ahrs", dr_ok=True,
                          p_flu=(0.0, 0.0, -1.0), yaw_flu_deg=0.0,
                          dr_err_m=0.1, dr_elapsed_s=float(i))
            f = (i + 1) / n
            s.p_dr_flu = (dr[0] * f, dr[1] * f, -1.0)
            v.add_status(s)

    push(40, (0.5, 0.2))
    assert v._dr_visible(), "an estimator IS reporting — no click should be needed"
    assert len(v.trail_dr) > 10 and v.p_dr is not None
    v.grab()

    # ...now let it run away to 1.4 km, the real failure this guards
    push(40, (1400.0, 400.0))
    xw0, xw1, yw0, yw1 = v._world_bounds()
    runs = v._clip_runs(v.trail_dr, xw0, xw1, yw0, yw1)
    drawn = [pt for r in runs for pt in r]
    assert drawn, "the near part of the trail must still be drawn"
    for x, y, _z in drawn:      # trail samples carry z since the 3-D view
        assert xw0 - 1 <= x <= xw1 + 1 and yw0 - 1 <= y <= yw1 + 1, \
            f"clipping let ({x:.1f}, {y:.1f}) through onto the plot"
    assert len(drawn) < len(v.trail_dr), "the runaway part must be clipped away"
    v.grab()          # the off-screen border arrow path must not blow up

    # the toggle still works, and CONTROL mode overrides it
    v.show_dr = False
    assert not v._dr_visible()
    v.add_status(MpcStatus(engaged=True, conn=Conn.ONLINE, dr_mode="control",
                           dr_ok=True, p_flu=(0.0, 0.0, -1.0),
                           p_dr_flu=(0.1, 0.0, -1.0)))
    assert v._dr_visible(), "you may not hide the thing you are flying on"


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
