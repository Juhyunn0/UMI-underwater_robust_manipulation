#!/usr/bin/env python3
"""
control.py — the seam where command sources plug in. Deliberately inert today.

Nothing in this module transmits. The recorder is a passive observer: the pilot
flies with QGroundControl and we log what they did. This file exists so that
adding a command source later is a matter of implementing one interface, not
restructuring the recorder.

What stays constant across every variant
----------------------------------------
**ArduSub is always the flight controller.** It owns stabilisation, the depth and
attitude loops, arming, and every failsafe. What changes between the cases below is
only *who plays the role QGroundControl plays today* — i.e. who is the MAVLink
command source. We are never replacing the autopilot, only the operator input to it.

    today          QGroundControl -> MANUAL_CONTROL -> ArduSub -> thrusters
    next           joystick       -> MANUAL_CONTROL -> ArduSub -> thrusters
    after that     vision policy  -> MANUAL_CONTROL -> ArduSub -> thrusters
                   (or SET_ATTITUDE_TARGET / setpoints, still through ArduSub)

So a controller only ever has to produce a `ControlCommand`; `MavlinkCommandSink`
turns that into MAVLink. The recorder does not need to know which source produced
it, and the dataset records the resulting MANUAL_CONTROL either way — which means a
vision-controlled dive logs identically to a human-piloted one.

Safety notes for whoever implements the sink
--------------------------------------------
* Two command sources at once is the hazard. If a joystick or a policy sends
  MANUAL_CONTROL while QGC also does, ArduSub acts on whichever arrived last and the
  vehicle jitters between them. Close QGC's control (or stop sending from it) before
  enabling a second source — do not rely on "I'll be careful".
* A command source SHOULD send heartbeats, because ArduSub's GCS failsafe is what
  stops the vehicle if the commanding link dies. A logger must NOT, so that it is
  never counted as the commanding GCS. That asymmetry is the reason the two roles
  are separate classes here.
* Send at a steady rate (ArduSub expects roughly 10-20 Hz MANUAL_CONTROL) and send
  neutral on loss of input rather than simply stopping.
* Implement a deadman: no fresh input within N ms -> neutral, not last-value-held.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ControlCommand:
    """One pilot/policy command, in MAVLink MANUAL_CONTROL's own conventions.

    x, y, z, r are the MANUAL_CONTROL axes. On ArduSub:
        x  forward / back   -1000..1000
        y  lateral          -1000..1000
        z  throttle/vertical (check the ArduSub docs for your version before
           assuming: this axis has used both 0..1000 and -1000..1000 conventions
           in different vehicle types, and getting it wrong means the vehicle
           dives when you meant neutral)
        r  yaw              -1000..1000
        buttons  bitmask; ArduSub maps buttons to functions like gripper open/close
    """

    x: int = 0
    y: int = 0
    z: int = 0
    r: int = 0
    buttons: int = 0

    def clamped(self) -> "ControlCommand":
        def c(v: int) -> int:
            return max(-1000, min(1000, int(v)))
        return ControlCommand(c(self.x), c(self.y), c(self.z), c(self.r),
                              int(self.buttons) & 0xFFFF)


class ControlSource(Protocol):
    """Anything that can produce commands: a joystick, a policy, a replay."""

    def read(self) -> ControlCommand | None:
        """Latest command, or None if there is nothing new this tick."""
        ...

    def close(self) -> None:
        ...


class NullControlSource:
    """The current implementation: produces nothing, ever.

    Used so the recorder's main loop can already carry a control source without
    any behavioural difference.
    """

    def read(self) -> ControlCommand | None:
        return None

    def close(self) -> None:
        return None


class JoystickControlSource:
    """Sketch of the pygame joystick path. Not wired up; raises if constructed.

    Left unimplemented on purpose rather than half-implemented: a partially
    working control path on a real vehicle is worse than an obviously absent one.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Joystick control is not implemented yet. To add it:\n"
            "  1. pip install pygame\n"
            "  2. pygame.joystick.init(); read axes in read() and map them to\n"
            "     ControlCommand (watch the z-axis convention).\n"
            "  3. Instantiate MavlinkCommandSink and pump it at 10-20 Hz.\n"
            "  4. Make sure QGroundControl is NOT also commanding at the time.\n"
            "  5. Add a deadman: stale input -> neutral."
        )


class MavlinkCommandSink:
    """Sketch of the command sink. Not wired up; raises if constructed.

    Kept as a stub so the shape is fixed and reviewable, while making it
    impossible to accidentally command the vehicle from this build.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Command sending is intentionally not implemented in the recorder.\n"
            "When you add it, the call is:\n"
            "    master.mav.manual_control_send(target_system, x, y, z, r, buttons)\n"
            "and the sink must also:\n"
            "  - send HEARTBEAT at ~1 Hz (so ArduSub's GCS failsafe protects you),\n"
            "  - send MANUAL_CONTROL at a steady 10-20 Hz,\n"
            "  - send neutral on input loss (deadman), and\n"
            "  - be the ONLY active command source (stop QGC commanding first).\n"
            "ArduSub remains the flight controller throughout; this only replaces\n"
            "the role QGroundControl plays as the command source."
        )
