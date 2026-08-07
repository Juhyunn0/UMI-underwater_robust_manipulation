#!/usr/bin/env python3
"""
state.py — the plain data that crosses the thread boundary.

Everything a backend hands to the UI is one of these frozen-ish dataclasses:
no Qt types, no numpy views into a buffer someone else is about to overwrite, no
device handles. That is what makes the hand-off safe. A worker thread builds a
snapshot, emits it, and never touches it again; the GUI thread owns it from
then on.

Two conventions that the panels rely on:

* ``stamp`` is ``time.monotonic()`` at the moment the *data* was true (not when
  it was emitted). The freshness watchdog ages every panel off this, which is
  how a silently wedged worker still turns the panel red.
* Unknown is ``None``, never 0.0 and never a plausible-looking default. A panel
  renders ``None`` as "--". A dashboard that shows 0.0 V for "I have no idea" is
  worse than one that shows nothing, and this repo has already been bitten once
  by a placeholder number being read back as a measurement
  (docs/MEASUREMENT_AUDIT.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Conn(Enum):
    """Connection/health state, in the order the UI escalates through them."""

    OFFLINE = "offline"        # never seen, or explicitly gone
    CONNECTING = "connecting"  # opening, handshaking, waiting for first data
    ONLINE = "online"          # fresh data, within spec
    DEGRADED = "degraded"      # data arriving but wrong: drops, low rate, errors
    STALE = "stale"            # was online, nothing recently — watchdog verdict
    FAULT = "fault"            # the source reported a failure

    @property
    def ok(self) -> bool:
        return self is Conn.ONLINE

    @property
    def bad(self) -> bool:
        return self in (Conn.OFFLINE, Conn.STALE, Conn.FAULT)


def now() -> float:
    """The one clock everything in this package ages against."""
    return time.monotonic()


# =============================================================================
# video
# =============================================================================
@dataclass
class VideoStat:
    """What the overlay on a video panel shows. One per stream."""

    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    latency_ms: float | None = None
    drop_rate: float = 0.0          # 0..1, device+host drops
    mbps: float | None = None       # link cost of THIS stream
    # False = counted on the wire. True = DERIVED (tether total minus the
    # streams we can count), which the overlay prefixes with "~" so it is never
    # read as a measurement. See CLAUDE.md on measurement provenance.
    mbps_estimated: bool = False
    conflated: int = 0              # frames the UI skipped to stay current
    encoding: str = ""
    conn: Conn = Conn.OFFLINE
    note: str = ""
    stamp: float = field(default_factory=now)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "--"


# =============================================================================
# vehicle telemetry
# =============================================================================
@dataclass
class SensorStat:
    """One auxiliary sensor's liveness. ``hz`` is measured, not requested."""

    name: str
    hz: float | None = None
    conn: Conn = Conn.OFFLINE
    detail: str = ""


@dataclass
class Telemetry:
    """The vehicle's own state, in SI, already converted out of MAVLink units."""

    # power
    battery_v: float | None = None
    battery_pct: float | None = None     # 0..100
    # WHERE the percentage came from, because the three sources are not equally
    # trustworthy and a bare number hides that:
    #   "vehicle"  the autopilot's own BATTERY_STATUS.battery_remaining
    #   "mah"      DERIVED: (capacity - consumed) / capacity
    #   "volts"    DERIVED: pack voltage against a nominal cell curve, which
    #              reads LOW under thrust because the pack sags
    # The panel labels the derived ones so they are never quoted as the
    # vehicle's own figure (CLAUDE.md, docs/MEASUREMENT_AUDIT.md).
    battery_pct_source: str = ""
    battery_left_mah: float | None = None
    current_a: float | None = None
    consumed_mah: float | None = None
    # attitude (radians) and depth (metres, positive down)
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    depth_m: float | None = None
    heading_deg: float | None = None
    water_temp_c: float | None = None
    internal_temp_c: float | None = None
    # flight controller
    armed: bool | None = None
    mode: str = ""
    leak: bool | None = None
    # auxiliary sensors, keyed by name
    sensors: dict[str, SensorStat] = field(default_factory=dict)
    conn: Conn = Conn.OFFLINE
    stamp: float = field(default_factory=now)


# =============================================================================
# propulsion
# =============================================================================
@dataclass
class ThrusterState:
    """Per-motor output and health.

    ``norm`` is signed −1..+1 (reverse..forward) because a unipolar bar cannot
    show a thruster that is pushing backwards, and on a vectored ROV that is
    half of normal operation. ``pwm_us`` is the raw servo output when the
    vehicle reports it, so the two can be cross-checked rather than assumed
    consistent.
    """

    n: int = 8
    norm: list[float] = field(default_factory=list)
    pwm_us: list[int | None] = field(default_factory=list)
    health: list[Conn] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    conn: Conn = Conn.OFFLINE
    # Measured arrival rate of the message these numbers came from. Shown on the
    # panel because "the motors are at neutral" and "nobody has told us what the
    # motors are doing" look identical otherwise — and on 2026-08-07 they were
    # confused for each other while the motors were actually turning.
    hz: float | None = None
    stamp: float = field(default_factory=now)

    @classmethod
    def blank(cls, n: int = 8, labels: list[str] | None = None) -> "ThrusterState":
        return cls(n=n,
                   norm=[0.0] * n,
                   pwm_us=[None] * n,
                   health=[Conn.OFFLINE] * n,
                   labels=labels or [f"T{i + 1}" for i in range(n)])


# =============================================================================
# payload
# =============================================================================
@dataclass
class PayloadState:
    """Gripper and lights: commanded value, measured value, and link state.

    ``gripper_fb`` is separate from ``gripper_cmd`` on purpose. The stock
    BlueROV2 gripper is an open-loop servo with no position feedback, so on the
    real vehicle ``gripper_fb`` is None and the panel says "cmd only" instead of
    drawing a feedback bar that is really just the command echoed back. Showing
    a command as if it were feedback is how a jammed jaw goes unnoticed.
    """

    gripper_cmd: float = 0.0             # 0 closed .. 1 open
    gripper_fb: float | None = None
    gripper_current_a: float | None = None
    gripper_conn: Conn = Conn.OFFLINE
    lights_cmd: float = 0.0              # 0 .. 1
    lights_fb: float | None = None
    lights_conn: Conn = Conn.OFFLINE
    # Camera mount tilt. ``tilt_deg`` is the vehicle's OWN report (MOUNT_STATUS
    # pointing_a, or the tilt servo's PWM converted to degrees) and stays None
    # when the vehicle does not report one — the same rule as gripper_fb: a
    # commanded direction is not a measured angle, and a mount that has hit its
    # stop looks identical to one still moving if you draw the command.
    tilt_deg: float | None = None
    tilt_drive: float = 0.0              # -1 down / 0 idle / +1 up, as commanded
    tilt_conn: Conn = Conn.OFFLINE
    tilt_note: str = ""
    # Free text from the backend about HOW the device is controlled on this
    # vehicle ("hold", "stepped, no feedback", "no button bit mapped"). The
    # panel prints it instead of implying a position that does not exist.
    gripper_note: str = ""
    lights_note: str = ""
    stamp: float = field(default_factory=now)


# =============================================================================
# tether / fiber link
# =============================================================================
@dataclass
class LinkStat:
    """Topside end of the tether, as the operating system sees it.

    On a fiber tether the topside media converter presents itself to the desktop
    as an ordinary Ethernet interface, so ``carrier``/``speed_mbps`` *are* the
    converter's link status — if the fiber drops or a connector is dirty, the
    copper side goes down or renegotiates, and the error counters climb before
    it does. That is the honest signal available without polling the converter
    over SNMP; ``rtt_ms`` (a TCP connect probe to the ROV) covers the rest of
    the path down the tether.
    """

    iface: str = ""
    up: bool | None = None
    speed_mbps: int | None = None        # negotiated link speed, not throughput
    rx_mbps: float | None = None
    tx_mbps: float | None = None
    rx_err_per_s: float | None = None
    tx_err_per_s: float | None = None
    rx_drop_per_s: float | None = None
    rtt_ms: float | None = None
    peer: str = ""
    conn: Conn = Conn.OFFLINE
    note: str = ""
    stamp: float = field(default_factory=now)


# =============================================================================
# pilot input
# =============================================================================
@dataclass
class PilotInput:
    """One teleop command, in body axes, normalised −1..+1.

    Deliberately NOT MAVLink's ±1000 MANUAL_CONTROL units: the conversion (and
    the z-axis convention trap that ``c3_camera/control.py`` warns about) belongs
    in the command sink, next to the code that has to get it right, not spread
    across every widget that can nudge an axis.
    """

    surge: float = 0.0     # +forward
    sway: float = 0.0      # +starboard
    heave: float = 0.0     # +up
    yaw: float = 0.0       # +clockwise seen from above
    active: frozenset[str] = frozenset()   # which inputs are held right now
    source: str = ""                       # "keyboard" | "buttons" | "gamepad"
    stamp: float = field(default_factory=now)

    @property
    def any_axis(self) -> bool:
        return any(abs(v) > 1e-6 for v in (self.surge, self.sway, self.heave, self.yaw))

    def clamped(self) -> "PilotInput":
        def c(v: float) -> float:
            return max(-1.0, min(1.0, float(v)))
        return PilotInput(c(self.surge), c(self.sway), c(self.heave), c(self.yaw),
                          self.active, self.source, self.stamp)
