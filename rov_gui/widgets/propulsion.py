#!/usr/bin/env python3
"""
propulsion.py — per-thruster output and health.

Layout is one fixed row per motor: state dot, signed bar, numeric readout. Eight
rows for the BlueROV2 Heavy (six for the stock vectored-6), created once at
construction from ``n``, never added or removed at runtime — a panel that grows
a row when a motor appears would resize the grid.

Why the bar is signed
---------------------
See :class:`~rov_gui.widgets.indicators.BiBar`. A vectored ROV holding station
against a current runs half its thrusters astern continuously; a 0..100% bar
draws that as "everything is working hard" and cannot show which way. The centre
tick is the pilot's zero reference and stays visible through the fill.

Why PWM is shown next to the normalised value
---------------------------------------------
``norm`` is what the controller asked for; ``pwm_us`` is what the autopilot
actually put on the wire (SERVO_OUTPUT_RAW). They are two different measurements
and showing both is what makes a stuck output, a reversed motor, or a failed ESC
visible instead of inferred. When the vehicle does not report PWM the field
reads ``--`` rather than being back-computed from ``norm`` — a back-computed
number would agree with the command by construction and could never disagree,
which is the whole point of showing it.
"""

from __future__ import annotations

from .. import theme
from ..qt import QtWidgets
from ..state import Conn, ThrusterState
from .indicators import BiBar, Panel, StatusDot, StatusPill

# ArduSub's servo output range for a T200/BasicESC. 1500 is neutral.
PWM_MIN, PWM_MID, PWM_MAX = 1100, 1500, 1900


class ThrusterRow(QtWidgets.QWidget):
    def __init__(self, label: str):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.dot = StatusDot(size=8)
        self.bar = BiBar(label, height=16, label_px=26, value_px=62)
        lay.addWidget(self.dot)
        lay.addWidget(self.bar, 1)
        self.setFixedHeight(18)

    def update_from(self, norm: float, pwm: int | None, health: Conn) -> None:
        self.dot.set_state(health)
        text = f"{pwm} us" if pwm is not None else f"{norm:+.2f}"
        self.bar.set_value(norm, text, health)


class PropulsionPanel(Panel):
    """Eight bars, a saturation warning, and the allocation the numbers mean."""

    def __init__(self, n: int = 8, labels: list[str] | None = None):
        self.pill = StatusPill("", Conn.OFFLINE)
        super().__init__("propulsion", right=self.pill, spacing=4)
        self.n = n
        labels = labels or [f"T{i + 1}" for i in range(n)]
        self.rows: list[ThrusterRow] = []
        for i in range(n):
            row = ThrusterRow(labels[i])
            self.rows.append(row)
            self.body.addWidget(row)

        self.summary = QtWidgets.QLabel("--")
        self.summary.setObjectName("Value")
        self.summary.setStyleSheet(f"font-size:10px; color:{theme.TEXT_FAINT};")
        self.body.addWidget(self.summary)
        self.body.addStretch(1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(196)

    def set_state(self, s: ThrusterState) -> None:
        sat = 0
        for i, row in enumerate(self.rows):
            norm = s.norm[i] if i < len(s.norm) else 0.0
            pwm = s.pwm_us[i] if i < len(s.pwm_us) else None
            health = s.health[i] if i < len(s.health) else Conn.OFFLINE
            row.update_from(norm, pwm, health)
            if abs(norm) >= 0.98:
                sat += 1
        live = [abs(v) for v in s.norm[:self.n]]
        mean = sum(live) / len(live) if live else 0.0
        bits = [f"mean |u| {mean:.2f}"]
        # The update rate, always. "Every motor at neutral" and "nobody has told
        # us what the motors are doing" draw identically, and on 2026-08-07 they
        # were confused for each other while the motors were turning. A rate on
        # screen is the difference, and a stale one says so in words.
        if s.conn.bad or s.conn is Conn.DEGRADED:
            bits.append("NOT UPDATING — reading is old" if s.conn.bad
                        else "update rate low")
        if s.hz is not None:
            bits.append(f"{s.hz:.0f} Hz")
        if sat:
            bits.append(f"{sat} SATURATED")
        offline = sum(1 for h in s.health[:self.n] if h.bad)
        if offline:
            bits.append(f"{offline} motor(s) offline")
        self.summary.setText("   ".join(bits))
        bad = sat or offline or s.conn.bad or s.conn is Conn.DEGRADED
        self.summary.setStyleSheet(
            f"font-size:10px; color:{theme.WARN if bad else theme.TEXT_FAINT};")
        self.pill.set_state(s.conn)
        self.set_alarm(offline > 0 or s.conn.bad)


def pwm_to_norm(pwm: int | None) -> float:
    """SERVO_OUTPUT_RAW microseconds -> −1..+1. Clamped, never extrapolated."""
    if pwm is None:
        return 0.0
    v = (float(pwm) - PWM_MID) / float(PWM_MAX - PWM_MID)
    return max(-1.0, min(1.0, v))
