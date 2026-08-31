#!/usr/bin/env python3
"""
payload.py — gripper and lights: command in, feedback out, link state beside both.

The distinction this panel exists to preserve
---------------------------------------------
A stock BlueROV2 gripper is an open-loop servo: you send it "open" and nothing
comes back. So there are two different numbers, and the panel never lets them
be confused:

    CMD   what we asked for            always known, drawn in the accent colour
    FB    what the vehicle reports     often None -> the bar reads "no feedback"

Echoing the command back as feedback is the tempting shortcut and it is exactly
what hides a jammed jaw, a stalled servo, or a disconnected payload cable: the
display keeps agreeing with the pilot. When feedback is genuinely absent the
panel says so, and the gripper's *current draw* becomes the substitute signal —
a jaw against a hard stop pulls current with no motion.

Lights are the same shape (level command, optional feedback) with one extra
consideration: they are the largest discretionary power draw on the vehicle, so
the level is shown next to the battery panel's draw figure rather than buried.
"""

from __future__ import annotations

import html as _html
import time

from .. import theme
from ..qt import Qt, QtGui, QtWidgets, QTimer, Signal
from ..state import Conn, PayloadState
from .indicators import (BiBar, ElidedLabel, HoldButton, MeterBar, Panel,
                         StatusPill)

# How fast a held OPEN/CLOSE button drives the command, in fraction per second.
# Slow enough that a short tap is a small movement; the pilot modulates by how
# long they hold, which is how the physical joystick trigger behaves.
GRIPPER_RATE = 0.9

# How much mission log to keep, in lines. A pool session emits on the order of
# hundreds (engage / refusal / phase / disengage), so this holds a whole day
# of them; the cap exists only so a station left running for a week cannot
# grow the widget's document without bound.
LOG_MAX_LINES = 5000

# Every hold/press button in this panel, one height. 22 rather than 26 since
# 2026-08-14: three rows of them were spending 12 px on nothing the finger
# notices, and the mission log below wanted the space.
BTN_H = 22


def _rule() -> QtWidgets.QFrame:
    """A 1 px separator between device sections."""
    line = QtWidgets.QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{theme.BORDER};")
    return line


class PayloadPanel(Panel):
    """Gripper + lights, with manual controls and per-device link state."""

    gripper_cmd = Signal(float)      # 0 closed .. 1 open  (position-capable backends)
    lights_cmd = Signal(float)       # 0 .. 1               (position-capable backends)
    gripper_drive = Signal(float)    # -1 close / 0 idle / +1 open  (momentary)
    tilt_drive = Signal(float)       # -1 down / 0 idle / +1 up     (momentary)
    tilt_center = Signal()           # one press of mount_center

    def __init__(self):
        super().__init__("payload", right=None, spacing=3)
        b = self.body
        # Set before the widgets, because the lights slider is built from it.
        self._light_steps = 8            # ArduSub JS_LIGHTS_STEPS; see --lights-steps
        self._held_s = 0.0

        # ------------------------------------------------------------ gripper
        # THREE ROWS PER DEVICE, not five (operator request 2026-08-14: "공간
        # 좀 줄이고 로그 더 크게"). The status NOTE — "no position feedback",
        # "draw 0.42 A", "held 1.3 s" — moved up onto the caption row, where it
        # elides into whatever width is going instead of owning a line of its
        # own. Nothing was deleted; the ~100 px this frees is the mission log's
        # (see log_view.setMinimumHeight below).
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        cap = QtWidgets.QLabel("GRIPPER")
        cap.setObjectName("Caption")
        self.grip_note = ElidedLabel("", px=10, colour=theme.TEXT_FAINT)
        self.grip_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addWidget(self.grip_note, 1)
        head.addWidget(self.grip_pill)
        b.addLayout(head)

        # A momentary jaw has no position to command, so there is no slider and
        # no "cmd" bar to disagree with reality. What there IS: which way it is
        # being driven right now, and for how long.
        self.grip_drive_bar = BiBar("drive", height=14, label_px=34, value_px=64)
        self.grip_drive_bar.set_value(0.0, "idle", Conn.OFFLINE)
        b.addWidget(self.grip_drive_bar)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.btn_close = HoldButton("◀ CLOSE", "hold to close  (key: G)")
        self.btn_open = HoldButton("OPEN ▶", "hold to open  (key: H)")
        for w in (self.btn_close, self.btn_open):
            w.setFixedHeight(BTN_H)
            row.addWidget(w)
        b.addLayout(row)

        # ------------------------------------------------------------- lights
        b.addWidget(_rule())

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        cap = QtWidgets.QLabel("LIGHTS")
        cap.setObjectName("Caption")
        self.light_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addStretch(1)
        head.addWidget(self.light_pill)
        b.addLayout(head)

        self.light_bar = MeterBar("level", "", ".0%", label_px=34, value_px=44,
                                  height=14)
        b.addWidget(self.light_bar)
        # One slider position per vehicle notch. A continuous 0-100 slider
        # promises a level the vehicle cannot be put in: ArduSub's lights move
        # in JS_LIGHTS_STEPS discrete steps (8 on this vehicle, read from the
        # parameter), so 63% is not a thing and pretending otherwise is what
        # made the displayed level and the actual light disagree.
        self.light_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.light_slider.setRange(0, self._light_steps)
        self.light_slider.setPageStep(1)
        self.light_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.light_slider.setFixedHeight(BTN_H)
        self.light_slider.valueChanged.connect(self._light_slider_moved)

        # The slider and its four presets share ONE row. They are the same
        # control — "put the light here" — and stacking them cost a full row
        # for four buttons that only ever need to be 30 px wide. Sized through
        # width/height, never a per-widget stylesheet: a widget-level sheet
        # REPLACES the application sheet for that widget and drops it back to
        # Qt's pale default (widgets/trajectory.py learned this the hard way).
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(3)
        row.addWidget(self.light_slider, 1)
        # OFF and MAX are the resync points: they drive past the end stop, so
        # whatever the sink believed about the level stops mattering.
        for label, frac in (("OFF", 0.0), ("¼", 0.25), ("½", 0.5), ("MAX", 1.0)):
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName("Compact")     # theme.py: padding only
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setFixedHeight(BTN_H)
            btn.setMinimumWidth(30)
            btn.setMaximumWidth(40)
            btn.setToolTip(f"lights {label}")
            btn.clicked.connect(
                lambda _=False, f=frac: self.light_slider.setValue(
                    int(round(f * self._light_steps))))
            row.addWidget(btn)
        b.addLayout(row)

        # --------------------------------------------------------- camera tilt
        b.addWidget(_rule())

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(6)
        cap = QtWidgets.QLabel("CAMERA TILT")
        cap.setObjectName("Caption")
        self.tilt_note = ElidedLabel("", px=10, colour=theme.TEXT_FAINT)
        self.tilt_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addWidget(self.tilt_note, 1)
        head.addWidget(self.tilt_pill)
        b.addLayout(head)

        # Same shape as the gripper and for the same reason: the mount is driven
        # by HELD buttons and reports an angle only if the vehicle sends
        # MOUNT_STATUS. So the bar shows the drive direction, and the angle —
        # when there is one — is printed beside it as a separate fact.
        self.tilt_bar = BiBar("tilt", height=14, label_px=34, value_px=64)
        self.tilt_bar.set_value(0.0, "level", Conn.OFFLINE)
        b.addWidget(self.tilt_bar)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.btn_tilt_down = HoldButton("▼ DOWN", "hold to tilt down  (key: ,)")
        self.btn_tilt_center = QtWidgets.QPushButton("LEVEL")
        self.btn_tilt_center.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_tilt_center.setToolTip("centre the mount  (key: .)")
        self.btn_tilt_up = HoldButton("UP ▲", "hold to tilt up  (key: /)")
        for w in (self.btn_tilt_down, self.btn_tilt_center, self.btn_tilt_up):
            w.setFixedHeight(BTN_H)
            row.addWidget(w)
        b.addLayout(row)

        # ----------------------------------------------------- mission log
        # Lives in the slack under CAMERA TILT (operator request 2026-08-14).
        # WALL-CLOCK stamps, not the monotonic clock everything else ages on:
        # this log exists to be lined up against a video file or a note
        # afterwards, and only the wall clock can do that.
        b.addWidget(_rule())
        sep = QtWidgets.QLabel("MISSION LOG")
        sep.setObjectName("Caption")
        b.addWidget(sep)
        # SCROLLABLE since 2026-08-14 (operator request: "예전 것도 볼 수 있게").
        # Before this it was a QLabel showing the last few lines and silently
        # dropping everything older — so a refusal from two minutes ago, the
        # thing you most want to read after a run goes wrong, was simply gone.
        #
        # This does not contradict window.py's "a control station must not hide
        # a control behind a scroll": a log is not a control. Nothing here is
        # clickable and nothing here is reachable ONLY by scrolling — the
        # newest line, the one that matters live, is always the visible one
        # (see add_event's stick-to-bottom rule).
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        # NoFocus is load-bearing: the main window is the sole key handler, and
        # a focusable text widget in the middle of the dashboard would eat
        # W/A/S/D the moment anything gave it focus. The wheel still scrolls it.
        self.log_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.log_view.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_view.setMaximumBlockCount(LOG_MAX_LINES)
        # Font and inset ONLY. Everything structural — the well background, the
        # border, and above all the SCROLL BARS — comes from theme.py, because
        # a scroll bar is a child widget and only the application sheet reaches
        # it. Left unstyled there, Qt painted a pale grey trough with a white
        # handle down the side of this log, which is what the operator was
        # looking at on 2026-08-14.
        self.log_view.setStyleSheet(
            f"QPlainTextEdit {{ font-family: {theme.MONO}; font-size: 10px; "
            f"color: {theme.TEXT_DIM}; background: {theme.BG}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; "
            f"padding: 2px 3px; }}")
        self.log_view.setToolTip("mission log — scroll back for older lines\n"
                                 "saved to the run folder as mission_log.txt "
                                 "when a recording stops")
        # ~9 lines minimum, and that number is a LAYOUT budget, not a taste.
        # It was 28 px (two lines) until 2026-08-14, when the three device
        # sections above were compacted from five rows each to three: the log
        # is spending exactly what they gave back, so the window's minimum
        # height is unchanged and the layout tests still hold
        # (test_layout_fits_one_screen..., test_mpc_panel_lives_in_the_grid...).
        # Grow this further only by taking the pixels from somewhere else in
        # this panel. At the operator's real window size the stretch below
        # gives it whatever is going on top.
        self.log_view.setMinimumHeight(112)
        b.addWidget(self.log_view, 1)
        # The full session history, kept independently of the widget: the view
        # caps at LOG_MAX_LINES blocks for memory, and this is what gets
        # written to mission_log.txt.
        self._log_lines: list[str] = []
        # ...and the state add_event needs to collapse a repeated line rather
        # than append it: the last message VERBATIM (the stamp is not part of
        # the comparison), how many times it has arrived, and the stamp of the
        # FIRST of them — a collapsed line is stamped when the thing started,
        # not when it last repeated.
        self._log_last: str | None = None
        self._log_repeat = 0
        self._log_stamp = ""

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(200)

        # ------------------------------------------- hold-to-drive machinery
        # A held button ramps the command on a timer instead of jumping to the
        # end stop, and the timer lives in the GUI thread: it only mutates a
        # float and emits a signal, so it cannot block anything.
        self._drive = 0.0
        self._cmd = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drive_step)
        self.btn_open.held.connect(lambda on: self._set_drive(+1.0 if on else 0.0))
        self.btn_close.held.connect(lambda on: self._set_drive(-1.0 if on else 0.0))
        self._tilt = 0.0
        self.btn_tilt_up.held.connect(lambda on: self.drive_tilt(+1.0 if on else 0.0))
        self.btn_tilt_down.held.connect(lambda on: self.drive_tilt(-1.0 if on else 0.0))
        self.btn_tilt_center.clicked.connect(self.tilt_center.emit)

    # ------------------------------------------------------------- commands
    def _set_drive(self, direction: float) -> None:
        self._drive = direction
        if direction:
            self._held_s = 0.0
        self.grip_drive_bar.set_value(
            direction,
            "closing" if direction < 0 else ("opening" if direction > 0 else "idle"),
            Conn.ONLINE if direction else Conn.OFFLINE)
        # The momentary truth goes out immediately and independently of the
        # simulated position below: on the real vehicle the jaw moves while the
        # button is down and there is no position to command.
        self.gripper_drive.emit(direction)
        if direction and not self._timer.isActive():
            self._timer.start()
        elif not direction:
            self._timer.stop()

    def _drive_step(self) -> None:
        """Integrate the hold into a position, for backends that HAVE one.

        The demo vehicle does; the real one does not. On the real one this value
        is never sent — see MavlinkCommandSink, which uses the drive direction.
        """
        step = self._drive * GRIPPER_RATE * (self._timer.interval() / 1000.0)
        self._cmd = max(0.0, min(1.0, self._cmd + step))
        self._held_s += self._timer.interval() / 1000.0
        self.gripper_cmd.emit(self._cmd)

    def drive_gripper(self, direction: float) -> None:
        """Keyboard entry point: −1 close, +1 open, 0 stop."""
        self._set_drive(direction)

    def drive_tilt(self, direction: float) -> None:
        """Keyboard and button entry point: −1 down, +1 up, 0 stop."""
        self._tilt = float(direction)
        self.tilt_bar.set_value(
            self._tilt,
            "down" if self._tilt < 0 else ("up" if self._tilt > 0 else "hold"),
            Conn.ONLINE if self._tilt else Conn.OFFLINE)
        self.tilt_drive.emit(self._tilt)

    def _light_slider_moved(self, notch: int) -> None:
        """Emit the ABSOLUTE level. Turning it into presses is the sink's job.

        Deliberately not "emit the delta": a delta model drifts the moment one
        press is lost, and there is no feedback to correct it with. The sink
        owns a model of where the light is and resyncs it against the end stops.
        """
        self.lights_cmd.emit(notch / max(1, self._light_steps))

    def set_light_steps(self, steps: int) -> None:
        self._light_steps = max(1, int(steps))
        self.light_slider.blockSignals(True)
        self.light_slider.setRange(0, self._light_steps)
        self.light_slider.blockSignals(False)

    def nudge_lights(self, delta: float) -> None:
        """One notch per key press, which is exactly one vehicle step."""
        step = 1 if delta > 0 else -1
        self.light_slider.setValue(
            max(0, min(self._light_steps, self.light_slider.value() + step)))

    # --------------------------------------------------------------- display
    def set_state(self, s: PayloadState) -> None:
        bits = []
        if self._drive:
            bits.append(f"held {self._held_s:.1f} s")
        if s.gripper_note:
            bits.append(s.gripper_note)
        elif s.gripper_fb is None:
            bits.append("no position feedback (open loop)")
        if s.gripper_current_a is not None:
            bits.append(f"draw {s.gripper_current_a:.2f} A")
        self.grip_note.setText("   ".join(bits) if bits else "")
        jammed = (s.gripper_current_a or 0.0) > 1.5
        self.grip_note.set_colour(theme.WARN if jammed else theme.TEXT_FAINT)

        self.grip_pill.set_state(s.gripper_conn)
        self.light_pill.set_state(s.lights_conn)
        # Measured level first (a servo output the vehicle reports), commanded
        # second. The bar's label says which one it is showing.
        if s.lights_fb is not None:
            self.light_bar._label = "level"
            self.light_bar.set_value(s.lights_fb, conn=s.lights_conn)
        else:
            self.light_bar._label = "cmd"
            self.light_bar.set_value(s.lights_cmd, conn=s.lights_conn)
        if s.lights_note:
            self.light_pill.set_state(s.lights_conn, s.lights_note)

        # --------------------------------------------------------------- tilt
        self.tilt_pill.set_state(s.tilt_conn)
        if not self._tilt:
            # Not being driven: show where the mount says it is, or that it is
            # idle. Never redraw the command as if it were the angle.
            self.tilt_bar.set_value(
                0.0 if s.tilt_deg is None else max(-1.0, min(1.0, s.tilt_deg / 45.0)),
                "hold" if s.tilt_deg is None else f"{s.tilt_deg:+.0f}°",
                s.tilt_conn)
        bits = []
        if s.tilt_deg is None:
            bits.append("no angle reported (open loop)")
        if s.tilt_note:
            bits.append(s.tilt_note)
        self.tilt_note.setText("   ".join(bits) if bits else "")

    # ------------------------------------------------------------ mission log
    #
    # Colour per level, and MISSION lines set apart. Until 2026-08-23 every
    # line was the same dim monospace grey, so "ENGAGED (mpc)" and "pose:
    # tracking" and a CUDA warning were typographically identical and the
    # operator had to read the whole panel to find the one line that said the
    # run had started. The level now carries the weight instead:
    #
    #   mission   the run's own lifecycle (ENGAGED / START / DISENGAGED /
    #             FOLLOW / STATION) — bold, in the accent colour, with a rule
    #             above it so it can be found by scanning rather than reading
    #   error     red      warn   amber      info   the old dim grey
    #   debug     NOT SHOWN — printed to stdout only (window.py routes it)
    _LEVEL_COLOR = {"mission": theme.ACCENT, "error": theme.FAIL,
                    "warn": theme.WARN, "info": theme.TEXT_DIM}

    def add_event(self, text: str, level: str = "info") -> None:
        """One short mission line, wall-clock stamped, appended to the scroll.

        Sticks to the bottom ONLY when it was already at the bottom. That rule
        is the whole point of a scrollable log on a live station: an operator
        who has scrolled up to read why a START was refused must not have the
        view yanked away from them by the next 20 Hz status line — but the
        moment they scroll back down, the log resumes following the present.

        A line identical to the one before it REWRITES that line with a count
        instead of appending. Four consecutive "engage refused: flight mode is
        mode -1" lines (2026-08-23 16:34:21-43, mission_log.txt) said exactly
        one thing four times and pushed everything else off the panel; they now
        read "... (x4)" and cost one line. Only *consecutive* repeats collapse,
        so a refusal that comes back after something else happened is still a
        new line — that is a different event and the timestamps must show it.
        """
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        repeat = bool(self._log_lines) and text == self._log_last
        bar = self.log_view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        if repeat:
            self._log_repeat += 1
            # The count goes at the FRONT, not the end: the panel does not
            # wrap (NoWrap + a horizontal scroll bar), so a count appended to
            # an already-long line sits off the right edge — invisible, which
            # is the one thing it must not be.
            line = f"{self._log_stamp} (x{self._log_repeat})  {text}"
            self._log_lines[-1] = line
            cur = self.log_view.textCursor()
            cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
            cur.select(QtGui.QTextCursor.SelectionType.BlockUnderCursor)
            cur.removeSelectedText()
        else:
            self._log_repeat = 1
            self._log_last = text
            self._log_stamp = time.strftime('%H:%M:%S')
            self._log_lines.append(line)
            del self._log_lines[:-LOG_MAX_LINES]
        col = self._LEVEL_COLOR.get(level, theme.TEXT_DIM)
        body = _html.escape(line).replace(" ", "&nbsp;")
        if level == "mission" and not repeat:
            self.log_view.appendHtml(
                f'<span style="color:{theme.TEXT_FAINT};">'
                f'{"&#9472;" * 6}</span>')
            body = f"<b>{body}</b>"
        self.log_view.appendHtml(f'<span style="color:{col};">{body}</span>')
        if at_bottom:
            bar.setValue(bar.maximum())

    def log_text(self) -> str:
        """The whole session's log, oldest first — what gets written to
        mission_log.txt beside a recording (window.py _save_mission_log)."""
        return "\n".join(self._log_lines)
