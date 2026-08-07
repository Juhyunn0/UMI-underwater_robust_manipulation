#!/usr/bin/env python3
"""
window.py — the control station itself: one QGridLayout, one screen, no scrolling.

The layout
----------
    row 0   header  (spans all four columns, fixed height)
    row 1   MAIN VIDEO       | STEREO L | SYSTEM
    row 2   (2 cols, 2 rows) | DEPTH    | HEALTH (2 rows)
    row 3   TELEOP | PAYLOAD | PROPULS. | SENSORS
            status bar  (spans all four columns, fixed height)

    column stretch  3 : 3 : 3 : 3      row stretch  0 : 3 : 3 : 0

Four columns rather than three because of a height budget, not a width one.
Every column stacks a tall panel over a short one, and with three columns the
tallest pair (system health over propulsion) set a minimum window height of
1026 px — taller than a 1366x768 laptop, i.e. the one-screen promise was
already broken at construction. Splitting the sensor list out into its own
fourth column rebalances both rows and brings the minimum under 768. The test
``test_layout_fits_one_screen`` is what keeps it there.

Three rules keep it on one screen, and all three are about *size policy*, not
about the layout:

1. **Video widgets have no content-derived size.** ``QSizePolicy.Ignored`` +
   custom paint. (See widgets/video.py — this is the one that actually breaks
   dashboards.)
2. **Data panels are Preferred vertically, and end their body layout with a
   stretch.** They ask for their content height and give back anything they do
   not need, instead of fighting the video rows for space.
3. **Nothing is in a QScrollArea.** If a panel does not fit, it must compress or
   elide — a scroll bar on a control station is a control the pilot cannot see.

Keyboard
--------
The window is the only key handler. Every button and slider in the package sets
``NoFocus`` so it cannot steal the keyboard, and auto-repeat is filtered at this
boundary (see widgets/teleop.py).

    W A S D / arrows   surge, sway (arrows: surge, yaw)
    (a gamepad on /dev/input/js* adds analog axes to the same command; see
     joystick.py and the --js-* flags)
    Q E                yaw
    R F / PgUp PgDn    heave
    space              ALL STOP
    G H                gripper close / open (hold)
    [ ]                lights down / up
    Ctrl+R             start/stop UI recording
    F11                fullscreen
    Esc                ALL STOP + disable commands

The command heartbeat
---------------------
The window re-emits the current pilot input at 20 Hz for as long as commands are
enabled, even when nothing changed. That is not redundant with the sink's
deadman — it is what *arms* it. The sink treats input older than 500 ms as
"neutral", so a wedged GUI thread (the thing a deadman is for) stops producing
fresh stamps and the vehicle is commanded to stop. If the GUI only emitted on
change, holding a key would look identical to a frozen GUI.
"""

from __future__ import annotations

import time

from . import theme
from .bus import DataBus, FrameMailbox, Freshness
from .imaging import legend_labels
from .joystick import JoystickReader, apply_deadzone
from .qt import (QShortcut, QTimer, Qt, QtGui, QtWidgets, preload_platform_libs,
                 run_app, sanitize_plugin_path)
from .recorder import ScreenRecorder, StreamRecorder
from .state import Conn, LinkStat, PayloadState, Telemetry, ThrusterState, now
from .widgets.health import HealthPanel, SensorPanel
from .widgets.indicators import ElidedLabel, StatusPill
from .widgets.payload import PayloadPanel
from .widgets.propulsion import PropulsionPanel
from .widgets.teleop import TeleopPanel
from .widgets.video import VideoPanel

# The middle panel is whichever second camera the run was started with, so the
# key is generic ("second") and only the title changes. Backends feed mailboxes
# by key and never need to know what the pilot called it.
SECOND_TITLES = {"rov": "Default RGB", "stereo": "C3 Stereo L",
                 "none": "Second view (off)"}

# One name per feed, used for the panel title, the record button and the file
# name — so what you clicked and what landed on disk are obviously the same feed.
FEED_NAMES = {"main": "C3 RGB", "second": "Default RGB", "depth": "C3 Depth"}


def _parse_remap(text) -> dict[int, int]:
    """'11:0,12:15' -> {11: 0, 12: 15}. Silently ignores malformed pairs.

    Both sides are VEHICLE button numbers. Malformed entries are dropped rather
    than raising, because a typo here must not stop the station from starting —
    but a dropped entry means a button quietly does nothing, so the window logs
    what it kept.
    """
    out: dict[int, int] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            src, dst = (int(v) for v in part.split(":"))
        except ValueError:
            continue
        if 0 <= src <= 15 and 0 <= dst <= 15 and src != dst:
            out[src] = dst
    return out


def _file_stem(name: str) -> str:
    return name.lower().replace(" ", "_")


def panel_specs(opts) -> tuple:
    """(key, title, legend) for the three video panels."""
    second = getattr(opts, "panel2", "rov")
    return (
        ("main", "C3 RGB", None),
        ("second", SECOND_TITLES.get(second, second), None),
        ("depth", "C3 Depth", legend_labels()),
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.setWindowTitle("BlueROV2 — Control Station")
        self.bus = DataBus()

        self.panels = panel_specs(opts)
        self.mailboxes = {k: FrameMailbox(k) for k, _t, _l in self.panels}
        self.fresh = {
            "vehicle": Freshness(warn_s=1.5, fail_s=4.0),
            "thrusters": Freshness(warn_s=1.5, fail_s=4.0),
            "payload": Freshness(warn_s=2.0, fail_s=6.0),
            "link": Freshness(warn_s=3.0, fail_s=8.0),
        }

        self._build()
        self._wire()

        self.payload.set_light_steps(int(getattr(opts, "lights_steps", 8) or 8))
        arm_mode = str(getattr(opts, "arm_mode", "MANUAL") or "").upper()
        self.teleop.arm_mode = "" if arm_mode in ("", "KEEP", "NONE") else arm_mode

        rec_dir = getattr(opts, "rec_dir", "sessions/ui_recordings")
        self.recorder = ScreenRecorder(self, out_dir=rec_dir,
                                       fps=float(getattr(opts, "rec_fps", 12)))
        self.recorder.state_changed.connect(self._rec_state)
        self.feed_recorders = {k: StreamRecorder(_file_stem(FEED_NAMES.get(k, k)),
                                                 out_dir=rec_dir)
                               for k, _t, _l in self.panels}

        self.backend = None
        self._counted_mbps: dict[str, float] = {}
        self._armed: bool | None = None
        self._mode: str = ""
        self._motor_mean = 0.0
        self._shut_down = False
        self._ui_frames = 0
        self._ui_t0 = time.monotonic()
        self._ui_hz = 0.0

        # One timer drives every repaint. Not one per panel: N timers means N
        # wake-ups, N event-loop round trips, and repaints that interleave into
        # a visibly torn dashboard. One tick = one coherent frame of the UI.
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(int(1000 / max(5.0, float(getattr(opts, "ui_fps", 30)))))
        self.ui_timer.timeout.connect(self._tick)

        # The joystick, polled on the GUI thread. Reading it is a few 8-byte
        # non-blocking reads (see joystick.py), so it needs no worker; 50 Hz is
        # well above the 20 Hz the commands go out at, so no stick movement is
        # missed between transmissions.
        self.joystick = JoystickReader(getattr(opts, "joystick", "auto")
                                       if getattr(opts, "joystick", "auto") != "none"
                                       else None)
        self._js_prev: set[int] = set()
        self._js_seen = False
        self._js_mask = 0
        self._js_over_warned = False
        self._js_unmapped_warned = False
        self._js_remap = _parse_remap(getattr(opts, "js_remap", ""))
        self._extra_sensors: dict = {}
        self._tel_sensors: dict = {}
        self._tel_conn = Conn.OFFLINE
        self.js_timer = QTimer(self)
        self.js_timer.setInterval(20)
        self.js_timer.timeout.connect(self._poll_joystick)

        # The command heartbeat. See the module docstring.
        self.cmd_timer = QTimer(self)
        self.cmd_timer.setInterval(50)
        self.cmd_timer.timeout.connect(self._pump_commands)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(1100, 640)

    # ==================================================================== ui
    def _build(self) -> None:
        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        grid = QtWidgets.QGridLayout(root)
        grid.setContentsMargins(8, 8, 8, 6)
        grid.setSpacing(8)

        grid.addWidget(self._build_header(), 0, 0, 1, 4)

        self.videos: dict[str, VideoPanel] = {}
        for key, title, legend in self.panels:
            # Per-panel expected rate: the depth feed runs far slower than the
            # colour one by design, and one shared watchdog threshold would
            # either flag healthy depth as stale or let dead colour look live.
            expect = float(getattr(self.opts, "fps", 30.0) or 30.0)
            if key == "depth":
                expect = float(getattr(self.opts, "depth_fps", 20.0) or 20.0)
            self.videos[key] = VideoPanel(key, title, self.mailboxes[key],
                                          expected_fps=expect, legend=legend)
            self.videos[key].focus_requested.connect(self._toggle_focus)
            self.videos[key].record_toggled.connect(self._toggle_feed_record)

        self.health = HealthPanel()
        self.sensors = SensorPanel()
        self.teleop = TeleopPanel()
        self.payload = PayloadPanel()
        self.propulsion = PropulsionPanel(n=int(getattr(self.opts, "thrusters", 8)))

        self.grid = grid
        self._place_videos(main="main")
        grid.addWidget(self.health, 1, 3, 2, 1)
        grid.addWidget(self.teleop, 3, 0)
        grid.addWidget(self.payload, 3, 1)
        grid.addWidget(self.propulsion, 3, 2)
        grid.addWidget(self.sensors, 3, 3)

        # Rule 2 from the module docstring, applied where it matters: the row-3
        # panels must not compete with the video rows for vertical space.
        for w in (self.teleop, self.payload, self.propulsion, self.sensors):
            w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                            QtWidgets.QSizePolicy.Policy.Preferred)

        for col in range(4):
            grid.setColumnStretch(col, 3)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 3)
        grid.setRowStretch(3, 0)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        # Two channels in the status bar, deliberately: transient log messages
        # go through showMessage() on the left, and the always-on counters live
        # in a permanent widget on the right. Writing both through showMessage
        # makes the ticker erase every log line 30 times a second.
        self.stats_label = QtWidgets.QLabel("")
        self.status.addPermanentWidget(self.stats_label)
        self.status.showMessage("starting")

    def _place_videos(self, main: str) -> None:
        """(Re)assign the three feeds to the big slot and the two small ones."""
        others = [k for k, _t, _l in self.panels if k != main]
        for panel in self.videos.values():
            self.grid.removeWidget(panel)
        self.grid.addWidget(self.videos[main], 1, 0, 2, 2)
        self.grid.addWidget(self.videos[others[0]], 1, 2)
        self.grid.addWidget(self.videos[others[1]], 2, 2)
        for panel in self.videos.values():
            panel.show()
        self._main_video = main

    def _toggle_focus(self, name: str) -> None:
        """Double-click a feed to promote it to the big slot."""
        if name != getattr(self, "_main_video", "main"):
            self._place_videos(main=name)
            self.bus.log.emit("info", f"main view: {name}")

    def _build_header(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("Panel")
        bar.setFixedHeight(46)
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(10)

        title = QtWidgets.QLabel("BlueROV2  CONTROL STATION")
        title.setStyleSheet(f"font-size:14px; font-weight:700; color:{theme.TEXT};"
                            "letter-spacing:1px;")
        lay.addWidget(title)

        # Elided and Ignored-width: the backend description is the one piece of
        # header text that may be sacrificed, so it yields space to the state
        # pills rather than pushing them off the bar.
        self.source_label = ElidedLabel("", px=11, colour=theme.TEXT_FAINT,
                                        mono=False)
        lay.addWidget(self.source_label, 1)

        self.banner = QtWidgets.QLabel("SIMULATED DATA")
        self.banner.setObjectName("Banner")
        self.banner.setVisible(False)
        self.banner.setToolTip(
            "Every value on screen is synthetic. Nothing here is a measurement.")
        lay.addWidget(self.banner)

        self.pills = {
            "video": StatusPill("VIDEO", Conn.OFFLINE),
            "vehicle": StatusPill("VEHICLE", Conn.OFFLINE),
            "link": StatusPill("TETHER", Conn.OFFLINE),
            "cmd": StatusPill("CMD", Conn.OFFLINE, show_state_text=False),
        }
        for pill in self.pills.values():
            lay.addWidget(pill)

        self.clock = QtWidgets.QLabel("--:--:--")
        self.clock.setObjectName("Value")
        lay.addWidget(self.clock)

        # Only the window recording lives here; each feed's own toggle sits on
        # the feed (see widgets/video.py), where its name fits.
        self.rec_btn = QtWidgets.QPushButton("REC UI")
        self.rec_btn.setObjectName("Rec")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rec_btn.setToolTip("record this whole window to mp4  (Ctrl+R)")
        self.rec_btn.clicked.connect(self._toggle_record)
        lay.addWidget(self.rec_btn)

        self.estop_btn = QtWidgets.QPushButton("E-STOP")
        self.estop_btn.setObjectName("Danger")
        self.estop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.estop_btn.setToolTip("zero all axes, disable transmission  (Esc)")
        self.estop_btn.clicked.connect(self.estop)
        lay.addWidget(self.estop_btn)
        return bar

    # ================================================================= wiring
    def _wire(self) -> None:
        b = self.bus
        b.telemetry.connect(self._on_telemetry)
        b.thrusters.connect(self._on_thrusters)
        b.payload.connect(self._on_payload)
        b.link.connect(self._on_link)
        b.video_stat.connect(self._on_video_stat)
        b.log.connect(self._on_log)

        # UI -> bus. The panels never talk to a backend directly; they emit and
        # the bus carries it across the thread boundary.
        self.teleop.pilot_changed.connect(b.cmd_pilot)
        self.teleop.enable_changed.connect(b.cmd_enable)
        self.teleop.arm_requested.connect(b.cmd_arm)
        self.teleop.mode_requested.connect(b.cmd_mode)
        self.teleop.gripper_drive.connect(self.payload.drive_gripper)
        self.teleop.lights_nudge.connect(self.payload.nudge_lights)
        self.teleop.tilt_drive.connect(self.payload.drive_tilt)
        self.teleop.tilt_center.connect(self.payload.tilt_center)
        self.payload.gripper_cmd.connect(b.cmd_gripper)
        self.payload.lights_cmd.connect(b.cmd_lights)
        self.payload.gripper_drive.connect(b.cmd_gripper_drive)
        self.payload.tilt_drive.connect(b.cmd_tilt)
        self.payload.tilt_center.connect(b.cmd_tilt_center)
        b.sensor_stat.connect(self._on_sensor_stat)

        # Explicit .connect rather than the `activated=` constructor keyword:
        # that keyword form is a PyQt convenience and is not portable to PySide.
        for keys, slot in (("Ctrl+R", self._toggle_record),
                           ("F11", self._toggle_fullscreen),
                           ("Ctrl+Q", self.close)):
            sc = QShortcut(QtGui.QKeySequence(keys), self)
            sc.activated.connect(slot)

    def attach(self, backend) -> None:
        self.backend = backend
        self.source_label.setText(backend.describe())
        self.banner.setVisible(bool(getattr(backend, "simulated", False)))
        backend.start()
        self.ui_timer.start()
        self.cmd_timer.start()
        if getattr(self.opts, "joystick", "auto") != "none":
            self.js_timer.start()
        self.bus.log.emit("info", f"backend {backend.name} started")

    # ================================================================== slots
    def _on_telemetry(self, t: Telemetry) -> None:
        self.fresh["vehicle"].mark(t.stamp, t.conn)
        self.health.set_telemetry(t)
        # Rows from other workers (the C3's IMU) are merged in here rather than
        # routed through the vehicle worker, which owns none of them. The
        # vehicle's own conn state must not colour them: a dead tether says
        # nothing about a camera on the same desk.
        self._tel_sensors, self._tel_conn = dict(t.sensors), t.conn
        self._refresh_sensors()
        # Say what arming actually means in the mode the vehicle is in. In
        # MANUAL a disarmed-to-armed transition changes nothing visible; in
        # STABILIZE, DEPTH HOLD or POSHOLD the autopilot starts driving the
        # thrusters on its own to hold attitude/depth, with no pilot input at
        # all. Out of the water that reads as "the motors just started running
        # by themselves", which is exactly how it was reported on 2026-08-07.
        if t.armed and self._armed is not True:
            mode = (t.mode or "").upper()
            if mode and not mode.startswith("MANUAL"):
                self.bus.log.emit(
                    "warn", f"ARMED in {t.mode} — the autopilot will run the "
                            f"thrusters by itself to hold attitude/depth, with "
                            f"no stick input. Switch to MANUAL if you did not "
                            f"want that.")
            else:
                self.bus.log.emit("warn", f"ARMED ({t.mode or 'mode unknown'})")
        self._armed = t.armed
        self._mode = t.mode or ""

    def _on_sensor_stat(self, s) -> None:
        # Repaint here as well as on telemetry. The C3 can be perfectly alive
        # while the vehicle is unplugged, and a camera IMU that only appears
        # once the ROV answers is a camera IMU nobody can check before a dive.
        self._extra_sensors[s.name] = s
        self._refresh_sensors()

    def _refresh_sensors(self) -> None:
        merged = dict(self._tel_sensors)
        merged.update(self._extra_sensors)
        self.sensors.set_sensors(merged, self._tel_conn)

    def _on_thrusters(self, s: ThrusterState) -> None:
        self.fresh["thrusters"].mark(s.stamp, s.conn)
        self.propulsion.set_state(s)
        # Kept so the teleop panel can say when the vehicle is moving thrusters
        # the pilot did not command — see TeleopPanel.tick.
        live = [abs(v) for v in s.norm[:s.n]] if not s.conn.bad else []
        self._motor_mean = (sum(live) / len(live)) if live else 0.0

    def _on_payload(self, s: PayloadState) -> None:
        self.fresh["payload"].mark(s.stamp)
        self.payload.set_state(s)

    def _on_link(self, s: LinkStat) -> None:
        self.fresh["link"].mark(s.stamp, s.conn)
        self.health.set_link(s)
        # The ROV camera cannot count its own bytes: it arrives as RTP inside
        # FFmpeg, which never reports how many it read. So its figure is DERIVED
        # — the whole tether (NIC counters) minus every stream that does report
        # itself — and the overlay prefixes it with "~" so it is never read as a
        # measurement.
        #
        # Checked against an A/B on 2026-08-06: with only this receiver running,
        # the tether carried 34.3 Mbit/s, and the derivation reads about 37.5.
        # The ~9% excess is the C3's XLink/TCP framing plus MAVLink, which the
        # per-stream payload figures do not include. Good enough to answer "what
        # is eating the link"; not a number to quote.
        #
        # Worth knowing: that 34 Mbit/s is on the tether whether or not this
        # station listens. BlueOS pushes the stream to 192.168.2.1:5600
        # unconditionally, so closing the panel frees CPU, not bandwidth.
        if s.rx_mbps is None:
            return
        counted = sum(v for k, v in self._counted_mbps.items() if k != "second")
        rest = s.rx_mbps - counted
        panel = self.videos.get("second")
        if panel is not None:
            panel.set_mbps_estimate(rest if rest > 0.2 else None)

    def _on_video_stat(self, stat) -> None:
        panel = self.videos.get(stat.name)
        if panel is not None:
            panel.set_status(stat)
        if stat.mbps is not None and not stat.mbps_estimated:
            self._counted_mbps[stat.name] = stat.mbps

    def _on_log(self, level: str, message: str) -> None:
        self.status.showMessage(f"[{level}] {message}", 8000 if level == "info" else 0)
        print(f"[{level}] {message}", flush=True)

    # ================================================================== ticks
    def _tick(self) -> None:
        """One coherent UI frame: pull frames, age everything, repaint."""
        for key, panel in self.videos.items():
            image = panel.tick()
            if image is not None:
                self.feed_recorders[key].feed(image)

        video_state = _worst(p.state() for p in self.videos.values())
        self.pills["video"].set_state(video_state)
        self.pills["vehicle"].set_state(self.fresh["vehicle"].state())
        self.pills["link"].set_state(self.fresh["link"].state())

        sink_conn, sink_note = self._sink_status()
        self.pills["cmd"].set_state(sink_conn,
                                    "ENABLED" if self.teleop.enabled else "SAFE")
        sink = getattr(self.backend, "sink", None) if self.backend else None
        # The vehicle's own JS_LIGHTS_STEPS wins over the CLI guess once read.
        steps = getattr(sink, "lights_steps", None)
        if steps and steps != self.payload._light_steps:
            self.payload.set_light_steps(steps)
        self.teleop.tick(tx_hz=getattr(sink, "_rate_hz", None),
                         sink_conn=sink_conn, sink_note=sink_note,
                         motor_mean=self._motor_mean, mode=self._mode)
        # Arm state comes from the vehicle's heartbeat and is shown as unknown
        # the moment telemetry goes stale — never as the last value we saw.
        self.teleop.set_armed(self._armed,
                              fresh=not self.fresh["vehicle"].state().bad)
        self.teleop.set_mode(self._mode,
                             fresh=not self.fresh["vehicle"].state().bad)
        self.health.set_conn(self.fresh["vehicle"].state())
        self.propulsion.pill.set_state(self.fresh["thrusters"].state())

        self.clock.setText(time.strftime("%H:%M:%S"))
        self._ui_frames += 1
        dt = time.monotonic() - self._ui_t0
        if dt >= 1.0:
            self._ui_hz = self._ui_frames / dt
            self._ui_frames, self._ui_t0 = 0, time.monotonic()
        if self.recorder.stats.recording:
            self.rec_btn.setText(self.recorder.stats.label())

        self._update_status()

    def _sink_status(self) -> tuple[Conn, str]:
        sink = getattr(self.backend, "sink", None) if self.backend else None
        if sink is not None and hasattr(sink, "status"):
            return sink.status()
        if self.teleop.enabled:
            return Conn.ONLINE, ""
        return Conn.OFFLINE, ""

    def _update_status(self) -> None:
        conflated = sum(mb.counters()["conflated"] for mb in self.mailboxes.values())
        drawn = sum(mb.counters()["taken"] for mb in self.mailboxes.values())
        rec = ""
        active = [(k, r.stats) for k, r in self.feed_recorders.items()
                  if r.stats.recording]
        if self.recorder.stats.recording:
            rec = (f" | REC ui {self.recorder.stats.frames}f"
                   f" -{self.recorder.stats.dropped}")
        for k, st in active:
            rec += f" | REC {FEED_NAMES.get(k, k)} {st.frames}f -{st.dropped}"
        self.stats_label.setText(
            f"ui {self._ui_hz:4.1f} Hz | frames drawn {drawn} "
            f"conflated {conflated}{rec}")

    # ---------------------------------------------------------------- joystick
    def _js_axis(self, name: str) -> float:
        """One mapped axis, with sign, deadzone and scale applied."""
        spec = int(getattr(self.opts, f"js_axis_{name}", 0) or 0)
        number, sign = abs(spec), (-1.0 if spec < 0 else 1.0)
        if getattr(self.opts, f"js_invert_{name}", False):
            sign = -sign
        raw = self.joystick.axes.get(number)
        if raw is None:
            return 0.0
        dz = float(getattr(self.opts, "js_deadzone", 0.08) or 0.0)
        scale = float(getattr(self.opts, "js_scale", 1.0) or 1.0)
        return apply_deadzone(raw, dz) * sign * scale

    def _poll_joystick(self) -> None:
        import time as _t

        if not self.joystick.poll(_t.monotonic()):
            if self.joystick.name is None and self._js_seen:
                self._js_seen = False
                self.teleop.set_joystick({}, set(), None)
                self.bus.log.emit("warn", "joystick disconnected")
            return
        if not self._js_seen:
            self._js_seen = True
            trig = self.joystick.triggers
            note = (f" — axes {trig} rest at full scale (triggers), auto-zeroed"
                    if trig else "")
            self.bus.log.emit("info", f"joystick: {self.joystick.name}{note}")
            if self._js_remap:
                pairs = ", ".join(f"{s}->{d}" for s, d in
                                  sorted(self._js_remap.items()))
                self.bus.log.emit("info", f"joystick remap (vehicle numbers): "
                                          f"{pairs}")
            if bool(getattr(self.opts, "js_translate", True)):
                if self.joystick.has_map:
                    self.bus.log.emit(
                        "info", "joystick buttons translated to the vehicle's "
                                "(SDL/QGC) numbering — the JOY line shows "
                                "kernel>vehicle for every button you press")
                else:
                    # Untranslated is how the tilt button armed the vehicle.
                    self.bus.log.emit(
                        "warn", f"no SDL button map for {self.joystick.name!r} — "
                                "pad buttons are being sent with the KERNEL's "
                                "numbering, which is probably not what the "
                                "vehicle's BTNn_FUNCTION expects. Check every "
                                "button against QGC before arming, or pass "
                                "--no-js-passthrough.")

        # TRANSLATED to the vehicle's numbering first. The kernel's button index
        # is not the one the vehicle's BTNn_FUNCTION parameters were configured
        # against — QGC reads pads through SDL, which reorders them — and
        # forwarding the kernel's index armed the vehicle when the pilot pressed
        # camera tilt. See joystick.py for the table and the incident.
        translate = bool(getattr(self.opts, "js_translate", True))
        pressed = self.joystick.vehicle_buttons(translate)
        raw_held = {n for n, down in self.joystick.buttons.items() if down}
        # Then the operator's own rewrites, so that from here on there is
        # exactly ONE set of numbers — the vehicle's function numbers — and the
        # bitmask, the chip lookup and the JOY line all agree. A pad button that
        # is remapped is sent as its TARGET only; sending both would fire two
        # functions from one press.
        if self._js_remap:
            pressed = {self._js_remap.get(n, n) for n in pressed}

        passthrough = bool(getattr(self.opts, "js_passthrough", True))
        mask = 0
        over = [n for n in pressed if n > 15]
        for n in pressed:
            if n <= 15:                 # MANUAL_CONTROL.buttons is a uint16
                mask |= 1 << n
        if over and not self._js_over_warned:
            self._js_over_warned = True
            self.bus.log.emit("warn", f"joystick buttons {sorted(over)} are past "
                                      "15 and cannot fit MANUAL_CONTROL — ignored")
        unmapped = self.joystick.unmapped()
        if unmapped and not self._js_unmapped_warned:
            self._js_unmapped_warned = True
            self.bus.log.emit(
                "warn", f"pad buttons {unmapped} have no entry in this pad's "
                        "SDL map — NOT sent (a guess here presses an unknown "
                        "function on the vehicle)")
        if passthrough and mask != self._js_mask:
            self._js_mask = mask
            self.bus.cmd_buttons.emit(mask)

        # The on-screen chips still light up, so the pilot sees which function
        # they just triggered. These numbers now DEFAULT to the vehicle's own
        # (--btn-*), so the chip that lights and the function that fires are the
        # same button rather than two different ones.
        for opt, chip in (("js_btn_gripper_close", "G"), ("js_btn_gripper_open", "H"),
                          ("js_btn_lights_down", "["), ("js_btn_lights_up", "]"),
                          ("js_btn_tilt_down", ","), ("js_btn_tilt_center", "."),
                          ("js_btn_tilt_up", "/")):
            btn = int(getattr(self.opts, opt, -1) or -1)
            if btn < 0:
                continue
            # Compared in the VEHICLE's numbering, which is what --js-btn-*
            # holds — the same numbers as --btn-* and the QGC table.
            was, now_down = btn in self._js_prev, btn in pressed
            if was != now_down:
                # With passthrough on this is indicator ONLY: the vehicle already
                # got this button in the mask above, and also firing our own
                # command would press the function twice — two notches per
                # light press, two gripper bits per squeeze.
                self.teleop._set_action(chip, now_down, notify=not passthrough)
        self._js_prev = pressed

        self.teleop.set_joystick(
            {n: self._js_axis(n) for n in ("surge", "sway", "heave", "yaw")},
            pressed, self.joystick.name, raw=raw_held)

    def _pump_commands(self) -> None:
        """Re-stamp and re-send the pilot's intent while commands are enabled."""
        if not self.teleop.enabled:
            return
        self.bus.cmd_pilot.emit(self.teleop.current())

    # ============================================================== controls
    def estop(self) -> None:
        self.teleop.all_stop()
        self.teleop.force_disable()
        self.bus.cmd_estop.emit()
        self.bus.log.emit("error", "E-STOP pressed — axes zeroed, commands disabled")

    def _toggle_record(self) -> None:
        if self.recorder.stats.recording:
            path = self.recorder.stop()
            self.bus.log.emit("info", f"recording saved: {path}")
        else:
            path = self.recorder.start()
            if path is None:
                self.bus.log.emit("error",
                                  f"recording failed: {self.recorder.stats.error}")
            else:
                self.bus.log.emit("info", f"recording to {path}")
        self.rec_btn.setChecked(self.recorder.stats.recording)

    def _toggle_feed_record(self, key: str) -> None:
        rec = self.feed_recorders[key]
        panel = self.videos[key]
        name = FEED_NAMES.get(key, key)
        if rec.stats.recording:
            path = rec.stop()
            self.bus.log.emit("info", f"{name} recording saved: {path}")
            if key == "depth":
                self.bus.cmd_log_sensors.emit(False, "")
        else:
            stat = panel.stat()
            # Open at the rate the feed is running now; the sidecar records what
            # actually happened, so a wrong guess is correctable, not silent.
            path = rec.start(fps=(stat.fps if stat and stat.fps > 1 else 15.0))
            if path is None:
                self.bus.log.emit("error", f"{name} recording failed: {rec.stats.error}")
            else:
                self.bus.log.emit("info", f"recording {name} to {path}")
                if key == "depth":
                    # A depth video alone is not a dataset — what makes it one is
                    # knowing how the camera was moving while each frame was
                    # captured. So the sensors ride along, sharing the video's
                    # stem so the files are obviously one recording. Depth only:
                    # the colour feeds are a picture for the pilot, not geometry.
                    self.bus.cmd_log_sensors.emit(
                        True, str(path.with_suffix("")))
        panel.set_recording(rec.stats.recording, name)

    def _rec_state(self, on: bool) -> None:
        self.rec_btn.setChecked(on)
        if not on:
            self.rec_btn.setText("REC UI")

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # =============================================================== keyboard
    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            self.estop()
            return
        if self.teleop.key_event(ev.key(), True, ev.isAutoRepeat()):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev) -> None:
        if self.teleop.key_event(ev.key(), False, ev.isAutoRepeat()):
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev) -> None:
        # Losing focus loses the key-release events, which would latch an axis
        # at full deflection with no way for the pilot to know. Treat it as a
        # release of everything — the same reflex a physical deadman has.
        self.teleop.all_stop()
        super().focusOutEvent(ev)

    # ================================================================ closing
    def shutdown(self) -> None:
        """Stop everything, exactly once.

        Hooked to BOTH ``closeEvent`` and ``QApplication.aboutToQuit`` because
        the window is not the only way out: ``app.quit()``, a Ctrl+Q from a
        shortcut, or the session manager all end the process without a close
        event. Skip it and the worker threads are still holding a camera and
        running timers when the interpreter starts tearing objects down — which
        surfaces as "QObject::killTimer: Timers cannot be stopped from another
        thread" at exit, and as a DepthAI device that stays claimed.
        """
        if self._shut_down:
            return
        self._shut_down = True
        self.ui_timer.stop()
        self.cmd_timer.stop()
        self.js_timer.stop()
        self.joystick.close()
        if self.recorder.stats.recording:
            self.recorder.stop()
        for rec in self.feed_recorders.values():
            if rec.stats.recording:
                rec.stop()
        if self.backend is not None:
            self.bus.cmd_estop.emit()      # leave the vehicle neutral
            self.backend.stop()

    def closeEvent(self, ev) -> None:
        self.shutdown()
        super().closeEvent(ev)


def _worst(states) -> Conn:
    """The most serious state in a group — a header pill must not average."""
    order = [Conn.ONLINE, Conn.CONNECTING, Conn.DEGRADED, Conn.STALE,
             Conn.OFFLINE, Conn.FAULT]
    worst = Conn.OFFLINE
    rank = -1
    for s in states:
        r = order.index(s) if s in order else 0
        if r > rank:
            worst, rank = s, r
    return worst


def build_and_run(opts) -> int:
    """Construct the application, the window and the backend, then run."""
    from .backends import make_backend

    # LAST thing before the QApplication exists, because Qt reads the platform
    # plugin path exactly once, here, and anything that ran earlier may have
    # repointed it — preflight importing cv2 is the case that actually crashed.
    # See rov_gui.qt.sanitize_plugin_path.
    plugins = sanitize_plugin_path()
    preload_platform_libs()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if plugins:
        print(f"rov_gui: Qt plugins {plugins}", flush=True)
    theme.apply(app)
    win = MainWindow(opts)
    backend = make_backend(opts.source, win.bus, win.mailboxes, opts)
    win.attach(backend)
    app.aboutToQuit.connect(win.shutdown)
    if getattr(opts, "fullscreen", False):
        win.showFullScreen()
    else:
        win.showMaximized()
    win.activateWindow()
    win.setFocus()
    return run_app(app)
