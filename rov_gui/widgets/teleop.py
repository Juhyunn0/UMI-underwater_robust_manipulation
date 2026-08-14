#!/usr/bin/env python3
"""
teleop.py — virtual controller, input visualiser, and the command deadman.

Three jobs in one panel:

1. **Produce commands.** On-screen hold buttons and the keyboard both feed one
   small state object; the panel emits a :class:`~rov_gui.state.PilotInput` when
   it changes. Nothing here transmits anything — the window routes the input to
   whatever command sink is installed, and the default sink transmits nothing at
   all (see ``rov_gui/backends/vehicle.py``).

2. **Show what the vehicle is being told.** Not what the pilot thinks they
   pressed: the four axis bars are the composed, scaled, clamped values that
   actually leave the panel. If a key is stuck, or two inputs are cancelling, or
   the output scale is at 30%, that is visible here and nowhere else.

3. **Show what is being held right now.** The key chips light on
   ``keyPressEvent`` and go out on ``keyReleaseEvent``. This is the panel that
   catches the classic X11 failure: a key-release event lost to a window-manager
   focus change leaves an axis latched at full deflection, and the pilot's only
   clue is that the vehicle will not stop. The chip stays lit, the deadman
   counter keeps counting, and both say the same thing.

Auto-repeat
-----------
Holding a key produces a *stream* of press/release pairs, not one press. Every
handler here drops ``event.isAutoRepeat()`` events; without that the "held"
state flickers at the OS key-repeat rate and the visualiser strobes.

Focus
-----
Every button and slider in this package sets ``Qt.FocusPolicy.NoFocus``. If a
widget takes focus, the arrow keys pan the slider instead of the ROV and the
space bar re-clicks the last button — the pilot silently loses the keyboard to a
widget. The window keeps focus and is the only key handler.
"""

from __future__ import annotations

from dataclasses import replace

from .. import theme
from ..qt import Qt, QtWidgets, Signal
from ..state import Conn, PilotInput, now
from .indicators import (BiBar, ElidedLabel, HoldButton, HoldToConfirmButton,
                         KeyChip, Panel, StatusPill)

# key -> (axis, sign, chip label). Arrows mirror WASD so either hand works, and
# both appear on the same chip so the visualiser does not double up.
KEYMAP: dict[int, tuple[str, float, str]] = {
    Qt.Key.Key_W: ("surge", +1.0, "W"),
    Qt.Key.Key_S: ("surge", -1.0, "S"),
    Qt.Key.Key_A: ("sway", -1.0, "A"),
    Qt.Key.Key_D: ("sway", +1.0, "D"),
    Qt.Key.Key_R: ("heave", +1.0, "R"),
    Qt.Key.Key_F: ("heave", -1.0, "F"),
    Qt.Key.Key_Q: ("yaw", -1.0, "Q"),
    Qt.Key.Key_E: ("yaw", +1.0, "E"),
    Qt.Key.Key_Up: ("surge", +1.0, "W"),
    Qt.Key.Key_Down: ("surge", -1.0, "S"),
    Qt.Key.Key_Left: ("yaw", -1.0, "Q"),
    Qt.Key.Key_Right: ("yaw", +1.0, "E"),
    Qt.Key.Key_PageUp: ("heave", +1.0, "R"),
    Qt.Key.Key_PageDown: ("heave", -1.0, "F"),
}

# Non-axis keys the visualiser also shows. Handled by the window.
ACTION_KEYS = {
    Qt.Key.Key_G: "G",       # gripper close
    Qt.Key.Key_H: "H",       # gripper open
    Qt.Key.Key_BracketLeft: "[",   # lights down
    Qt.Key.Key_BracketRight: "]",  # lights up
    Qt.Key.Key_Comma: ",",         # camera tilt down
    Qt.Key.Key_Period: ".",        # camera tilt level
    Qt.Key.Key_Slash: "/",         # camera tilt up
    Qt.Key.Key_Space: "SPC",       # all stop
}

# Two rows, not one per axis pair: the visualiser competes with the video feeds
# for vertical space, and every row it takes is a row the pilot's picture loses.
# Output multiplier the panel starts at. Low on purpose: this scales every axis,
# and the first thing a pilot does after connecting is nudge a stick to see
# whether anything moves. That nudge should not be a third of full thrust in a
# confined tank or next to a diver.
DEFAULT_OUTPUT_PCT = 20

CHIP_ROWS = (
    [("Q", "yaw port"), ("W", "forward"), ("E", "yaw stbd"),
     ("A", "port"), ("S", "aft"), ("D", "starboard")],
    [("R", "up"), ("F", "down"), ("SPC", "all stop"),
     ("G", "gripper close"), ("H", "gripper open"),
     ("[", "lights down"), ("]", "lights up"),
     (",", "tilt down"), (".", "tilt level"), ("/", "tilt up")],
)


class TeleopPanel(Panel):
    """Virtual controller + input visualiser + deadman state."""

    pilot_changed = Signal(object)     # state.PilotInput
    estop = Signal()
    enable_changed = Signal(bool)
    arm_requested = Signal(bool)       # True = arm, False = disarm
    gripper_drive = Signal(float)      # −1 close / 0 stop / +1 open
    tilt_drive = Signal(float)         # −1 down / 0 stop / +1 up
    tilt_center = Signal()
    mode_requested = Signal(str)       # ArduSub mode name
    lights_nudge = Signal(float)

    def __init__(self):
        self.pill = StatusPill("COMMANDS", Conn.OFFLINE, show_state_text=False)
        super().__init__("teleoperation", right=self.pill, spacing=4)
        b = self.body

        self._held: dict[str, float] = {}     # chip label -> sign, axis keys only
        self._axis_of: dict[str, str] = {}    # chip label -> axis
        self._actions: set[str] = set()
        self._scale = 1.0
        self._last_input_t: float | None = None
        self._enabled = False
        self._tx_hz = 0.0
        self._joy_axes: dict[str, float] = {}   # composed axis -> value
        self._ext_cmd = None    # MPC's command, shown on the bars while engaged
        self._joy_buttons: set[int] = set()
        self._joy_raw: set[int] = set()
        self._joy_name: str | None = None
        self._armed: bool | None = None
        self._mode: str = ""
        self._autopilot_driving = False
        # Mode the ARM button puts the vehicle in first. "" disables that, for
        # anyone who wants ARM to inherit whatever mode the vehicle is in.
        self.arm_mode: str = "MANUAL"

        # ------------------------------------------------------- virtual pad
        pad = QtWidgets.QGridLayout()
        pad.setSpacing(3)
        self._pad_buttons: dict[str, HoldButton] = {}

        def add(row: int, col: int, chip: str, text: str, tip: str) -> None:
            btn = HoldButton(text, tip)
            btn.setMinimumWidth(54)
            btn.setFixedHeight(22)
            btn.held.connect(lambda on, c=chip: self._set_chip(c, on, "buttons"))
            pad.addWidget(btn, row, col)
            self._pad_buttons[chip] = btn

        add(0, 0, "Q", "↺ YAW", "yaw to port  (Q / ←)")
        add(0, 1, "W", "▲ FWD", "forward  (W / ↑)")
        add(0, 2, "E", "YAW ↻", "yaw to starboard  (E / →)")
        add(1, 0, "A", "◀ PORT", "translate to port  (A)")
        stop = QtWidgets.QPushButton("ALL STOP")
        stop.setFixedHeight(22)
        stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        stop.setToolTip("zero every axis  (space)")
        stop.clicked.connect(self.all_stop)
        pad.addWidget(stop, 1, 1)
        add(1, 2, "D", "STBD ▶", "translate to starboard  (D)")
        add(2, 0, "R", "⬆ UP", "ascend  (R / PgUp)")
        add(2, 1, "S", "▼ AFT", "reverse  (S / ↓)")
        add(2, 2, "F", "⬇ DOWN", "descend  (F / PgDn)")
        b.addLayout(pad)

        # ---------------------------------------------------------- axis bars
        self.bars = {
            "surge": BiBar("SRG", height=14, label_px=30, value_px=46),
            "sway": BiBar("SWY", height=14, label_px=30, value_px=46),
            "heave": BiBar("HVE", height=14, label_px=30, value_px=46),
            "yaw": BiBar("YAW", height=14, label_px=30, value_px=46),
        }
        for bar in self.bars.values():
            bar.set_value(0.0, "+0.00", Conn.ONLINE)
            bar.setToolTip(
                "The COMMAND this station is sending — composed, scaled and "
                "clamped.\nNOT the vehicle's motion: in STABILIZE or DEPTH "
                "HOLD the autopilot\nadds thrust of its own that never passes "
                "through these bars, so they\nread zero while the motors work. "
                "PROPULSION shows what the motors do.")
            b.addWidget(bar)

        # ------------------------------------------------------ output scale
        row = QtWidgets.QHBoxLayout()
        cap = QtWidgets.QLabel("OUTPUT")
        cap.setObjectName("Caption")
        self.scale_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(10, 100)
        self.scale_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scale_label = QtWidgets.QLabel("--")
        self.scale_label.setObjectName("Value")
        self.scale_slider.valueChanged.connect(self._scale_changed)
        row.addWidget(cap)
        row.addWidget(self.scale_slider, 1)
        row.addWidget(self.scale_label)
        b.addLayout(row)
        # ONE source of truth, applied through the handler. The slider position,
        # the label and the multiplier used to be three separate literals set in
        # three places, and the connect happened AFTER setValue — so changing
        # the default moved the slider to 20% while the axes kept being scaled
        # by 0.60. A pilot reading "20%" was getting three times that.
        self.scale_slider.setValue(DEFAULT_OUTPUT_PCT)
        self._scale_changed(self.scale_slider.value())

        # ------------------------------------------------------- visualiser
        cap = QtWidgets.QLabel("INPUT (held)")
        cap.setObjectName("Caption")
        b.addWidget(cap)
        self.chips: dict[str, KeyChip] = {}
        for spec in CHIP_ROWS:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(3)
            for label, hint in spec:
                chip = KeyChip(label, hint, w=26, h=22)
                self.chips[label] = chip
                row.addWidget(chip)
            row.addStretch(1)
            b.addLayout(row)

        # ------------------------------------------- enable + arm + deadman
        row = QtWidgets.QHBoxLayout()
        self.enable = QtWidgets.QCheckBox("COMMAND ENABLE")
        self.enable.setToolTip(
            "Off = this station transmits nothing. Turn it on only when no other\n"
            "command source (QGroundControl, a joystick) is commanding the vehicle:\n"
            "ArduSub acts on whichever MANUAL_CONTROL arrived last, and two sources\n"
            "make the vehicle jitter between them.")
        self.enable.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.enable.toggled.connect(self._enable_toggled)
        row.addWidget(self.enable)
        row.addStretch(1)
        self.armed_label = ElidedLabel("MOTORS --", px=10, colour=theme.TEXT_FAINT)
        self.armed_label.setToolTip(
            "What the vehicle's HEARTBEAT says, not what this station asked for.")
        self.armed_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                       QtWidgets.QSizePolicy.Policy.Fixed)
        self.armed_label.setMinimumWidth(96)
        row.addWidget(self.armed_label)
        b.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        # ARM is hold-to-confirm; DISARM is a single click. Deliberately
        # asymmetric: making a stop harder than a start is backwards, and the
        # only failure mode of an accidental disarm is that the vehicle stops.
        self.btn_arm = HoldToConfirmButton(
            "ARM", hold_ms=1200,
            tooltip="hold to arm the vehicle's motors.\n"
                    "Axes are zeroed first, so it cannot arm into a held key.\n"
                    "Requires COMMAND ENABLE.")
        self.btn_arm.setFixedHeight(22)
        self.btn_arm.confirmed.connect(self._arm_confirmed)
        self.btn_disarm = QtWidgets.QPushButton("DISARM")
        self.btn_disarm.setObjectName("Danger")
        self.btn_disarm.setFixedHeight(22)
        self.btn_disarm.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_disarm.setToolTip(
            "disarm the motors. Sent even with COMMAND ENABLE off —\n"
            "a stop must never be behind a switch.")
        self.btn_disarm.clicked.connect(lambda: self.arm_requested.emit(False))
        row.addWidget(self.btn_arm)
        row.addWidget(self.btn_disarm)
        b.addLayout(row)
        self.btn_arm.setEnabled(False)
        self.btn_arm.setText("ARM — enable first")

        # ---------------------------------------------------------- mode
        # The mode decides whether an armed vehicle sits still. In MANUAL it
        # does; in STABILIZE / DEPTH HOLD the autopilot drives the thrusters on
        # its own to hold attitude or depth. Arming into a stabilising mode is
        # what "I pressed ARM and the motors started" was, so the mode is a
        # visible switch rather than something inherited from whatever the
        # vehicle was left in.
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        for label, mode in (("MANUAL", "MANUAL"), ("STAB", "STABILIZE"),
                            ("DEPTH", "ALT_HOLD")):
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(18)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Qt CLIPS a push button's label rather than eliding it, so a
            # too-narrow button shows "STAB" as "STAR" and nobody can tell it is
            # truncated. Two things prevent that: compact padding, and a minimum
            # width measured from the font — with slack, because the checked
            # state turns the label bold AFTER this width is computed.
            btn.setStyleSheet(
                "QPushButton{font-size:10px; padding:1px 4px; border-radius:3px;}")
            fm = btn.fontMetrics()
            btn.setMinimumWidth(fm.horizontalAdvance(label) + 18)
            btn.setToolTip({
                "MANUAL": "no self-levelling: an armed vehicle with no stick "
                          "input sits still",
                "STABILIZE": "the autopilot holds attitude — it WILL run the "
                             "thrusters with no stick input",
                "ALT_HOLD": "the autopilot holds depth — it WILL run the "
                            "thrusters with no stick input",
            }[mode])
            btn.clicked.connect(lambda _=False, m=mode: self.mode_requested.emit(m))
            self.mode_buttons[mode] = btn
            row.addWidget(btn)
        b.addLayout(row)

        # Raw joystick state, shown whether or not commands are enabled. This
        # is what makes a gamepad mappable: press a button, read its number.
        self.joy_line = ElidedLabel("no joystick", px=10, colour=theme.TEXT_FAINT)
        b.addWidget(self.joy_line)

        self.deadman = ElidedLabel("idle", px=10, colour=theme.TEXT_FAINT)
        b.addWidget(self.deadman)
        b.addStretch(1)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(210)

        for chip, (axis, sign, _label) in ((v[2], v) for v in KEYMAP.values()):
            self._axis_of[chip] = axis

    # ------------------------------------------------------------- input in
    def key_event(self, key: int, pressed: bool, auto_repeat: bool) -> bool:
        """Route a key from the window. Returns True if it was ours."""
        if auto_repeat:
            return key in KEYMAP or key in ACTION_KEYS
        if key in KEYMAP:
            _axis, _sign, chip = KEYMAP[key]
            self._set_chip(chip, pressed, "keyboard")
            return True
        if key in ACTION_KEYS:
            chip = ACTION_KEYS[key]
            self._set_action(chip, pressed)
            return True
        return False

    def _set_chip(self, chip: str, on: bool, source: str) -> None:
        """A motion input went down or up. Chip label is the identity."""
        axis = self._axis_of.get(chip)
        if axis is None:
            return
        sign = next(s for (a, s, c) in KEYMAP.values() if c == chip and a == axis)
        if on:
            self._held[chip] = sign
            self._last_input_t = now()
        else:
            self._held.pop(chip, None)
        chip_w = self.chips.get(chip)
        if chip_w is not None:
            chip_w.set_active(on)
        btn = self._pad_buttons.get(chip)
        if btn is not None and btn.isDown() != on:
            btn.setDown(on)
        self._emit(source)

    def _set_action(self, chip: str, on: bool, notify: bool = True) -> None:
        w = self.chips.get(chip)
        if w is not None:
            w.set_active(on)
        if on:
            self._actions.add(chip)
            self._last_input_t = now()
        else:
            self._actions.discard(chip)
        if not notify:
            # Light the chip, command nothing. Used when the gamepad's own
            # button already went to the vehicle in the passthrough mask — the
            # pilot still needs to see WHICH function fired, but firing it a
            # second time from here would double every press.
            return
        if chip in ("G", "H"):
            self.gripper_drive.emit(0.0 if not on else (-1.0 if chip == "G" else +1.0))
        elif chip in (",", "/"):
            # Held, like the gripper — ArduSub drives the mount for as long as
            # the button is down — so the key-up matters and must be forwarded.
            self.tilt_drive.emit(0.0 if not on else (-1.0 if chip == "," else +1.0))
        elif chip == "." and on:
            self.tilt_center.emit()
        elif chip in ("[", "]") and on:
            self.lights_nudge.emit(-0.1 if chip == "[" else +0.1)
        elif chip == "SPC" and on:
            self.all_stop()

    def all_stop(self) -> None:
        """Zero every axis and clear every held input, including latched ones."""
        for chip in list(self._held):
            self._set_chip(chip, False, "stop")
        self._held.clear()
        for chip in self.chips.values():
            chip.set_active(False)
        for btn in self._pad_buttons.values():
            btn.setDown(False)
        self._emit("stop")

    def _scale_changed(self, v: int) -> None:
        self._scale = v / 100.0
        self.scale_label.setText(f"{v}%")
        self._emit("scale")

    def _enable_toggled(self, on: bool) -> None:
        self._enabled = on
        if not on:
            self.all_stop()
        # A disabled button cannot show a tooltip (it gets no mouse events), so
        # the reason has to be in the label or the pilot is left guessing why
        # nothing happens when they click.
        self.btn_arm.setEnabled(on)
        self.btn_arm.setText("ARM" if on else "ARM — enable first")
        self.enable_changed.emit(on)

    def _arm_confirmed(self) -> None:
        # Zero every axis BEFORE arming. Arming while an input is held means the
        # vehicle moves the instant the motors go live, which is the one way an
        # arm button hurts someone standing next to the boat.
        self.all_stop()
        # And put the vehicle in a mode where arming alone does NOT move it.
        # ArduSub keeps whatever mode it was left in, so without this the ARM
        # button inherits a stabilising mode and the thrusters start the moment
        # the motors go live — with no stick input and nothing on screen to
        # explain it. Emitted before the arm request so the mode change is in
        # flight first; the panel still shows the mode the VEHICLE reports.
        if self.arm_mode:
            self.mode_requested.emit(self.arm_mode)
        self.arm_requested.emit(True)

    def set_mode(self, mode: str, fresh: bool = True) -> None:
        """Show the vehicle's OWN mode (from HEARTBEAT), never the one we asked
        for.

        Same rule as :meth:`set_armed`, and for the same reason: ArduSub refuses
        some mode changes (several are rejected while armed), and a panel that
        highlighted our request would show MANUAL while the vehicle stayed in
        STABILIZE — the exact state in which pressing ARM runs the motors.
        """
        self._mode = (mode or "").upper() if fresh else ""
        for name, btn in self.mode_buttons.items():
            on = fresh and self._mode.startswith(name[:6])
            btn.setChecked(bool(on))
            # A mode that drives the thrusters by itself is drawn as a warning,
            # not as a neutral selection.
            btn.setProperty("level", "warn" if (on and name != "MANUAL") else "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def set_armed(self, armed: bool | None, fresh: bool = True) -> None:
        """Show the vehicle's OWN arm state (from HEARTBEAT), never ours.

        ``fresh`` False means telemetry has gone stale, and then the honest
        answer is "unknown" — not the last value we happened to see. Arming
        blind is exactly when a stale indicator is most dangerous.
        """
        self._armed = None if not fresh else armed
        if not fresh or armed is None:
            self.armed_label.setText("MOTORS  ?")
            self.armed_label.set_colour(theme.WARN)
        elif armed:
            self.armed_label.setText("MOTORS ARMED")
            self.armed_label.set_colour(theme.FAIL)
        else:
            self.armed_label.setText("MOTORS safe")
            self.armed_label.set_colour(theme.TEXT_DIM)
        # No set_alarm() here: tick() owns the panel's alarm border, and two
        # writers would take turns clearing each other's condition.

    # ------------------------------------------------------------ input out
    def set_joystick(self, axes: dict, buttons: set, name: str | None,
                     raw: set | None = None) -> None:
        """Analog axes and pressed buttons, already mapped by the window.

        Stored and DISPLAYED unconditionally — a disarmed vehicle, or a station
        with COMMAND ENABLE off, still shows every stick movement. That is what
        makes it safe to check a mapping before anything can move.

        ``buttons`` is in the VEHICLE's numbering and ``raw`` in the kernel's.
        Both are shown, because they differ (SDL reorders them) and the pilot
        needs to be able to check which function a button will actually fire
        BEFORE the motors can turn. Displaying only one of them is how a tilt
        button armed the vehicle without anyone being able to see why.
        """
        self._joy_axes = dict(axes)
        self._joy_buttons = set(buttons)
        self._joy_raw = set(raw) if raw is not None else set(buttons)
        self._joy_name = name
        self._emit("joystick")

    def current(self) -> PilotInput:
        axes = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
        for chip, sign in self._held.items():
            axis = self._axis_of.get(chip)
            if axis:
                axes[axis] += sign
        # Keyboard and stick add, then clamp: a held key while the stick is
        # deflected should not cancel out, and either alone must still reach
        # full deflection.
        for axis, value in self._joy_axes.items():
            if axis in axes:
                axes[axis] += value
        cmd = PilotInput(
            surge=axes["surge"] * self._scale,
            sway=axes["sway"] * self._scale,
            heave=axes["heave"] * self._scale,
            yaw=axes["yaw"] * self._scale,
            active=frozenset(set(self._held) | self._actions),
            source="teleop",
        ).clamped()
        return cmd

    def _emit(self, source: str) -> None:
        cmd = replace(self.current(), source=source, stamp=now())
        # While an external source (the MPC) owns the bars, a key edge must
        # not repaint them with the pilot's (swallowed) command — the frame
        # still goes out so the window's gate can treat it as a takeover.
        if self._ext_cmd is None:
            for axis, bar in self.bars.items():
                v = getattr(cmd, axis)
                bar.set_value(v, f"{v:+.2f}", Conn.ONLINE)
        self.pilot_changed.emit(cmd)

    def show_command(self, cmd) -> None:
        """Display an EXTERNAL command source's axes on the SRG/SWY/HVE/YAW
        bars — the MPC while it is engaged. DISPLAY ONLY: ``current()`` and
        every emitted frame stay the pilot's own input, so the takeover gate
        keeps working. ``None`` hands the bars back to the pilot.

        This is what makes the bars honest during an engagement: they show
        the command actually leaving the station, whoever computed it (the
        panel's standing rule — the bars are OUR command, not the vehicle's
        motion; PROPULSION shows the vehicle's answer)."""
        self._ext_cmd = cmd
        if cmd is None:
            own = self.current()
            for axis, bar in self.bars.items():
                v = getattr(own, axis)
                bar.set_value(v, f"{v:+.2f}", Conn.ONLINE)
            return
        for axis, bar in self.bars.items():
            v = getattr(cmd, axis)
            bar.set_value(v, f"{v:+.2f}", Conn.ONLINE)

    # ------------------------------------------------------------ live state
    def tick(self, tx_hz: float | None = None, sink_conn: Conn = Conn.OFFLINE,
             sink_note: str = "", motor_mean: float = 0.0,
             mode: str = "") -> None:
        """Called from the window's UI timer: refresh the deadman readout."""
        if tx_hz is not None:
            self._tx_hz = tx_hz
        # The four axis bars are the pilot's COMMAND — the composed, scaled,
        # clamped values this station sends. In a stabilising mode the autopilot
        # generates thrust of its own that never passes through them, so the
        # bars sit at zero while the motors work. That is correct, and it is
        # also confusing unless something says so, which is what this does.
        self._autopilot_driving = bool(
            self._armed and motor_mean > 0.02
            and not self.current().any_axis
            and (mode or "").upper() not in ("MANUAL", "ACRO", ""))
        held = len(self._held)
        # ONE rate, and it is the one that matters: how fast MANUAL_CONTROL is
        # actually leaving the host, measured by the sink. The panel's own
        # emit rate would look identical whether or not anything transmits.
        rate = f"TX {self._tx_hz:.0f} Hz" if self._tx_hz else "TX --"
        if not self._enabled:
            text, colour = ("COMMANDS DISABLED — station is receive-only",
                            theme.TEXT_FAINT)
        elif self._ext_cmd is not None:
            # The MPC owns the channel; the bars are ITS axes. Saying so here
            # is what keeps "bars moving, sticks untouched" from reading as a
            # haunted station.
            text, colour = ("MPC commanding — bars show ITS axes; move any "
                            "stick to take over", theme.ACCENT)
        elif self.current().any_axis and self._armed is not True:
            # Commanding something, but the vehicle will not act on it. Without
            # this the pilot sees full bars, a healthy TX rate, and a vehicle
            # that does nothing — with no clue which of the two switches is the
            # one that is off.
            why = ("MOTORS DISARMED" if self._armed is False
                   else "ARM STATE UNKNOWN")
            text = f"{rate}   {why} — nothing will move"
            colour = theme.WARN
        elif held or self.current().any_axis:
            age = now() - (self._last_input_t or now())
            text = f"{rate}   {held} input(s) held {age:.1f} s"
            colour = theme.OK
        elif self._autopilot_driving:
            # Bars at zero, motors turning. Saying "neutral" here would be true
            # of this station and false of the vehicle.
            text = (f"{rate}   AUTOPILOT DRIVING ({mode}) — bars show YOUR "
                    f"command only")
            colour = theme.WARN
        else:
            text = f"{rate}   neutral"
            colour = theme.TEXT_DIM
        if sink_note and not sink_note.startswith("tx "):
            text += f"   {sink_note}"

        if self._joy_name is None:
            self.joy_line.setText("no joystick")
            self.joy_line.set_colour(theme.TEXT_FAINT)
        else:
            live = "  ".join(f"{k[:3]} {v:+.2f}" for k, v in
                             sorted(self._joy_axes.items()) if abs(v) > 0.005)
            # "kernel>vehicle" when they differ, so the number the vehicle acts
            # on is always the one on the right of the arrow.
            if self._joy_raw and self._joy_raw != self._joy_buttons:
                btns = (f"{','.join(str(b) for b in sorted(self._joy_raw))}"
                        f">{','.join(str(b) for b in sorted(self._joy_buttons))}")
            else:
                btns = " ".join(str(b) for b in sorted(self._joy_buttons))
            self.joy_line.setText(
                f"JOY {self._joy_name[:18]}  {live or 'centred'}"
                + (f"   btn {btns}" if btns else ""))
            self.joy_line.set_colour(theme.OK if (live or btns) else theme.TEXT_DIM)
        self.deadman.setText(text)
        self.deadman.set_colour(colour)
        self.pill.set_state(sink_conn, "ENABLED" if self._enabled else "SAFE")
        self.set_alarm(self._enabled and sink_conn is Conn.FAULT)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def force_disable(self) -> None:
        """E-stop path: drop the enable switch from code, not from the mouse."""
        self.enable.setChecked(False)
