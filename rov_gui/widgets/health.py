#!/usr/bin/env python3
"""
health.py — power, vehicle state, tether/fiber link, and the sensor list.

Two panels, not one, and the split is a layout decision with a reason. Power +
vehicle + tether are read together (a stuttering picture is explained by the
link; a sluggish vehicle by the battery), so they share a panel in the video
column. The sensor liveness list is read *rarely* — at power-up and when
something has already gone wrong — so it lives in the bottom row where its
height competes with the controls rather than with the video feeds.

That is the whole trick to fitting a control station on one screen: rank panels
by how much vertical space they can afford to take from the picture, not by how
related they are.

Fixed row count
---------------
The sensor list is capped and each row is a fixed height. A panel that grows a
row every time a new MAVLink message type appears would push the window past
the screen edge — the exact failure the layout exists to prevent. Sensors past
the cap are counted in a "+N more" line instead of being drawn.
"""

from __future__ import annotations

import math

from .. import theme
from ..qt import Qt, QtWidgets
from ..state import Conn, LinkStat, SensorStat, Telemetry
from .indicators import (ElidedLabel, MeterBar, Panel, Sparkline, StatusDot,
                         StatusPill, Tile)

MAX_SENSOR_ROWS = 8

# The order sensors are shown in when present. Anything not listed lands after
# these, in arrival order, until the cap is hit.
# Two IMUs, listed as two rows on purpose. The autopilot's IMU and the C3's
# BNO086 are different devices on different clocks with different rates, and the
# single "IMU" row that used to be here made it impossible to tell which one had
# gone quiet — or which one a downstream pipeline was actually fusing.
SENSOR_ORDER = ("IMU (ROV)", "IMU (C3)", "Barometer", "Compass", "Leak", "EKF",
                "Heartbeat", "GPS")


def _caption(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text.upper())
    lab.setObjectName("Caption")
    lab.setFixedHeight(12)
    return lab


def _hline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{theme.BORDER};")
    return line


class SensorRow(QtWidgets.QWidget):
    """name … dot … rate. One line, fixed height, never relayouts."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.dot = StatusDot(size=8)
        self.label = QtWidgets.QLabel(name)
        self.label.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        self.value = QtWidgets.QLabel("--")
        self.value.setObjectName("Value")
        self.value.setStyleSheet(f"font-size:11px; color:{theme.TEXT_FAINT};")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight
                                | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.dot)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.value)
        self.setFixedHeight(15)

    def update_from(self, s: SensorStat) -> None:
        self.dot.set_state(s.conn)
        # The RATE is what this column is for, so it wins over the detail text —
        # a row that said "BNO086" where "486 Hz" belonged answered the wrong
        # question. Two exceptions: a detail that already carries a rate keeps
        # its qualifier ("200 Hz sampled" must not become a bare "200 Hz", which
        # would claim the vehicle's transmit rate when it is our polling rate),
        # and a sensor with no rate at all falls back to whatever it can say.
        if s.detail and "Hz" in s.detail:
            text = s.detail
        elif s.hz is not None:
            text = theme.fmt(s.hz, ".0f", " Hz")
        else:
            text = s.detail or "--"
        self.value.setText(text)
        self.setToolTip(f"{s.name}: {s.detail}" if s.detail else s.name)
        colour = theme.TEXT_DIM if not s.conn.bad else theme.TEXT_FAINT
        self.value.setStyleSheet(f"font-size:11px; color:{colour};")


class HealthPanel(Panel):
    """Battery, depth/heading/attitude, and the tether."""

    def __init__(self):
        self.pill = StatusPill("VEHICLE", Conn.OFFLINE)
        super().__init__("system health", right=self.pill, spacing=4)
        b = self.body

        # ---------------------------------------------------------- power
        b.addWidget(_caption("power"))
        self.batt = QtWidgets.QProgressBar()
        self.batt.setRange(0, 100)
        self.batt.setValue(0)
        self.batt.setFormat("battery  %p%")
        self.batt.setProperty("level", "idle")
        b.addWidget(self.batt)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.t_volt = Tile("pack", " V", ".1f", value_px=15, width=52)
        self.t_amp = Tile("draw", " A", ".1f", value_px=15, width=52)
        # "left" rather than "used" whenever the capacity is known: how much is
        # left is the question a pilot actually asks, and consumed-mAh alone
        # cannot answer it without doing arithmetic in your head mid-dive.
        self.t_mah = Tile("used", " mAh", ".0f", value_px=15, width=68)
        for t in (self.t_volt, self.t_amp, self.t_mah):
            row.addWidget(t)
        b.addLayout(row)

        # -------------------------------------------------------- vehicle
        b.addWidget(_hline())
        b.addWidget(_caption("vehicle"))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.t_depth = Tile("depth", " m", ".2f", value_px=17, width=68)
        self.t_hdg = Tile("heading", "°", ".0f", value_px=17, width=58)
        row.addWidget(self.t_depth)
        row.addWidget(self.t_hdg)
        b.addLayout(row)
        # Attitude, temperature, arm state and mode on ONE line: four readings
        # the pilot scans rather than reads, and four separate QLabels would
        # cost four rows of the video feed's height.
        self.vehicle_line = ElidedLabel("roll --  pitch --  --", px=11)
        b.addWidget(self.vehicle_line)

        # ------------------------------------------------- tether / fiber
        b.addWidget(_hline())
        head = QtWidgets.QHBoxLayout()
        head.addWidget(_caption("tether — data rate"))
        head.addStretch(1)
        self.link_pill = StatusPill("", Conn.OFFLINE)
        head.addWidget(self.link_pill)
        b.addLayout(head)

        # "rx"/"tx" are what the kernel calls them; nobody flying an ROV thinks
        # in those terms. This is simply which way the data is going.
        self.rx = MeterBar("from ROV", " Mb/s", ".1f", label_px=52, value_px=62,
                           height=15)
        self.tx = MeterBar("to ROV", " Mb/s", ".1f", label_px=52, value_px=62,
                           height=15)
        b.addWidget(self.rx)
        b.addWidget(self.tx)
        self.spark = Sparkline(n=150, vmax=None, color=theme.ACCENT, height=20)
        b.addWidget(self.spark)
        self.link_detail = ElidedLabel("--", px=10, colour=theme.TEXT_FAINT)
        b.addWidget(self.link_detail)
        b.addStretch(1)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(212)

        # Link scale: the bars and sparkline are scaled against the negotiated
        # link speed once it is known; until then against 100 Mbit/s, so the
        # first samples are not drawn full-scale.
        self._link_scale = 100.0

    # -------------------------------------------------------------- updates
    def set_telemetry(self, t: Telemetry) -> None:
        pct = t.battery_pct
        self.batt.setValue(int(pct) if pct is not None else 0)
        # Say where the number came from. Only the vehicle's own
        # battery_remaining is a reading; the other two are ours, and a derived
        # figure that looks like a measured one is exactly the failure this repo
        # audited itself for (CLAUDE.md, docs/MEASUREMENT_AUDIT.md).
        tag = {"mah": "  est mAh", "volts": "  est V"}.get(t.battery_pct_source, "")
        self.batt.setFormat(f"battery  {pct:.0f}%{tag}" if pct is not None
                            else "battery  --")
        self.batt.setToolTip({
            "vehicle": "the autopilot's own BATTERY_STATUS.battery_remaining",
            "mah": "DERIVED: (capacity - consumed mAh) / capacity.\n"
                   "Assumes the pack started full and the counter was reset "
                   "with it.\nSet --battery-capacity-mah to match the pack.",
            "volts": "DERIVED from pack voltage against a nominal Li-ion cell "
                     "curve.\nReads LOW under thrust — the pack sags when the "
                     "motors pull.\nFor a real figure, set BATT_CAPACITY and a "
                     "current monitor on the vehicle;\nthen the autopilot "
                     "reports it and this panel uses that instead.",
        }.get(t.battery_pct_source, "no battery telemetry"))
        level = "idle"
        if pct is not None and not t.conn.bad:
            level = "crit" if pct < 15 else ("warn" if pct < 30 else "ok")
        if self.batt.property("level") != level:
            self.batt.setProperty("level", level)
            self.batt.style().unpolish(self.batt)
            self.batt.style().polish(self.batt)

        self.t_volt.set_value(t.battery_v)
        self.t_amp.set_value(t.current_a)
        # Remaining when we can compute it, consumed otherwise. The caption
        # changes with the number so the two can never be read as each other.
        if t.battery_left_mah is not None:
            self.t_mah._caption = "left"
            self.t_mah.set_value(t.battery_left_mah,
                                 theme.WARN if t.battery_left_mah < 2000 else None)
        else:
            self.t_mah._caption = "used"
            self.t_mah.set_value(t.consumed_mah)
        self.t_depth.set_value(t.depth_m)
        self.t_hdg.set_value(t.heading_deg)

        deg = math.degrees
        armed = "--" if t.armed is None else ("ARMED" if t.armed else "disarmed")
        self.vehicle_line.setText(
            f"roll {theme.fmt(deg(t.roll) if t.roll is not None else None, '+.0f', '°')}"
            f"  pitch {theme.fmt(deg(t.pitch) if t.pitch is not None else None, '+.0f', '°')}"
            f"  {theme.fmt(t.water_temp_c, '.1f', '°C')}"
            f"  {armed} {t.mode or ''}")
        self.vehicle_line.set_colour(theme.FAIL if t.armed else theme.TEXT_DIM)
        self.set_alarm(bool(t.leak) or (pct is not None and pct < 10))

    def set_conn(self, conn: Conn, detail: str = "") -> None:
        self.pill.set_state(conn, detail)

    def set_link(self, s: LinkStat) -> None:
        if s.speed_mbps:
            self._link_scale = float(s.speed_mbps)
        scale = max(1.0, self._link_scale)
        self.rx.set_value(s.rx_mbps, frac=(s.rx_mbps or 0.0) / scale, conn=s.conn)
        self.tx.set_value(s.tx_mbps, frac=(s.tx_mbps or 0.0) / scale, conn=s.conn)
        self.spark.push((s.rx_mbps or 0.0) + (s.tx_mbps or 0.0))
        self.link_pill.set_state(s.conn, s.iface or "no iface")

        errs = (s.rx_err_per_s or 0.0) + (s.tx_err_per_s or 0.0)
        bits = [f"link {s.speed_mbps} Mbit/s" if s.speed_mbps else "link --",
                f"errors {errs:.1f}/s",
                f"round-trip {theme.fmt(s.rtt_ms, '.1f', ' ms')}"]
        if s.note:
            bits.append(s.note)
        self.link_detail.setText("  ".join(bits))


class SensorPanel(Panel):
    """Liveness of every auxiliary sensor, one fixed-height row each.

    Rate, not value: this panel answers "is the barometer talking" and at what
    rate, which is what you check before a dive and after a fault. The values
    themselves live where they are used (depth in the health panel, attitude in
    the health panel), because a number shown twice is a number that will
    eventually disagree with itself.
    """

    def __init__(self):
        self.pill = StatusPill("", Conn.OFFLINE)
        super().__init__("sensors", right=self.pill, spacing=2)
        self._rows: dict[str, SensorRow] = {}
        self._rows_box = QtWidgets.QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        self.body.addLayout(self._rows_box)
        self.overflow = QtWidgets.QLabel("")
        self.overflow.setObjectName("Caption")
        self.overflow.setVisible(False)      # hidden, so it costs no height
        self.body.addWidget(self.overflow)
        self.body.addStretch(1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(150)

    def set_sensors(self, sensors: dict[str, SensorStat], conn: Conn) -> None:
        self.pill.set_state(conn)
        if not sensors:
            return
        ordered = [n for n in SENSOR_ORDER if n in sensors]
        ordered += [n for n in sensors if n not in SENSOR_ORDER]
        shown = ordered[:MAX_SENSOR_ROWS]
        for name in shown:
            row = self._rows.get(name)
            if row is None:
                row = SensorRow(name)
                self._rows[name] = row
                self._rows_box.addWidget(row)
            row.update_from(sensors[name])
        hidden = len(ordered) - len(shown)
        self.overflow.setVisible(hidden > 0)
        if hidden > 0:
            self.overflow.setText(f"+{hidden} more not shown")
        self.set_alarm(any(s.conn is Conn.FAULT for s in sensors.values()))
