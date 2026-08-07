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

from .. import theme
from ..qt import Qt, QtWidgets, QTimer, Signal
from ..state import Conn, PayloadState
from .indicators import (BiBar, ElidedLabel, HoldButton, MeterBar, Panel,
                         StatusPill)

# How fast a held OPEN/CLOSE button drives the command, in fraction per second.
# Slow enough that a short tap is a small movement; the pilot modulates by how
# long they hold, which is how the physical joystick trigger behaves.
GRIPPER_RATE = 0.9


class PayloadPanel(Panel):
    """Gripper + lights, with manual controls and per-device link state."""

    gripper_cmd = Signal(float)      # 0 closed .. 1 open  (position-capable backends)
    lights_cmd = Signal(float)       # 0 .. 1               (position-capable backends)
    gripper_drive = Signal(float)    # -1 close / 0 idle / +1 open  (momentary)
    tilt_drive = Signal(float)       # -1 down / 0 idle / +1 up     (momentary)
    tilt_center = Signal()           # one press of mount_center

    def __init__(self):
        super().__init__("payload", right=None, spacing=5)
        b = self.body
        # Set before the widgets, because the lights slider is built from it.
        self._light_steps = 8            # ArduSub JS_LIGHTS_STEPS; see --lights-steps
        self._held_s = 0.0

        # ------------------------------------------------------------ gripper
        head = QtWidgets.QHBoxLayout()
        cap = QtWidgets.QLabel("GRIPPER")
        cap.setObjectName("Caption")
        self.grip_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addStretch(1)
        head.addWidget(self.grip_pill)
        b.addLayout(head)

        # A momentary jaw has no position to command, so there is no slider and
        # no "cmd" bar to disagree with reality. What there IS: which way it is
        # being driven right now, and for how long.
        self.grip_drive_bar = BiBar("drive", height=16, label_px=34, value_px=64)
        self.grip_drive_bar.set_value(0.0, "idle", Conn.OFFLINE)
        b.addWidget(self.grip_drive_bar)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.btn_close = HoldButton("◀ CLOSE", "hold to close  (key: G)")
        self.btn_open = HoldButton("OPEN ▶", "hold to open  (key: H)")
        for w in (self.btn_close, self.btn_open):
            w.setFixedHeight(26)
            row.addWidget(w)
        b.addLayout(row)

        self.grip_note = ElidedLabel("--", px=10, colour=theme.TEXT_FAINT)
        b.addWidget(self.grip_note)

        # ------------------------------------------------------------- lights
        line = QtWidgets.QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{theme.BORDER};")
        b.addWidget(line)

        head = QtWidgets.QHBoxLayout()
        cap = QtWidgets.QLabel("LIGHTS")
        cap.setObjectName("Caption")
        self.light_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addStretch(1)
        head.addWidget(self.light_pill)
        b.addLayout(head)

        self.light_bar = MeterBar("level", "", ".0%", label_px=34, value_px=44)
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
        self.light_slider.valueChanged.connect(self._light_slider_moved)
        b.addWidget(self.light_slider)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        # OFF and MAX are the resync points: they drive past the end stop, so
        # whatever the sink believed about the level stops mattering.
        for label, frac in (("OFF", 0.0), ("¼", 0.25), ("½", 0.5), ("MAX", 1.0)):
            btn = QtWidgets.QPushButton(label)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(
                lambda _=False, f=frac: self.light_slider.setValue(
                    int(round(f * self._light_steps))))
            row.addWidget(btn)
        b.addLayout(row)

        # --------------------------------------------------------- camera tilt
        line = QtWidgets.QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{theme.BORDER};")
        b.addWidget(line)

        head = QtWidgets.QHBoxLayout()
        cap = QtWidgets.QLabel("CAMERA TILT")
        cap.setObjectName("Caption")
        self.tilt_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(cap)
        head.addStretch(1)
        head.addWidget(self.tilt_pill)
        b.addLayout(head)

        # Same shape as the gripper and for the same reason: the mount is driven
        # by HELD buttons and reports an angle only if the vehicle sends
        # MOUNT_STATUS. So the bar shows the drive direction, and the angle —
        # when there is one — is printed beside it as a separate fact.
        self.tilt_bar = BiBar("tilt", height=16, label_px=34, value_px=64)
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
            w.setFixedHeight(26)
            row.addWidget(w)
        b.addLayout(row)

        self.tilt_note = ElidedLabel("--", px=10, colour=theme.TEXT_FAINT)
        b.addWidget(self.tilt_note)
        b.addStretch(1)

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
