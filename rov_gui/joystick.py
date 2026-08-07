#!/usr/bin/env python3
"""
joystick.py — a gamepad, read straight from the kernel. No dependencies.

Linux exposes joysticks twice: the modern evdev interface (``/dev/input/event*``,
needs the ``evdev`` package or SDL) and the legacy joystick interface
(``/dev/input/js*``), which is eight bytes per event and needs nothing at all.
This uses the legacy one on purpose — a control station that cannot fly because
a python package is missing on the boat is a bad trade for an API that has been
stable since 1997.

    struct js_event {
        __u32 time;     /* ms since boot            */
        __s16 value;    /* -32767..32767, or 0/1    */
        __u8  type;     /* 0x01 button, 0x02 axis   */
        __u8  number;   /* which one                */
    };                  /* 8 bytes, little-endian   */

``type`` is OR'd with 0x80 on the synthetic events the driver replays when the
device is opened, so the initial state arrives without the pilot touching
anything. Those carry real values and are applied like any other — masking the
flag off and using the value is what makes the panel show the sticks' true
position immediately rather than a false centre.

Reading is non-blocking and drains everything pending, so this can be polled
from the GUI thread's timer: it is a handful of 8-byte reads per tick, not a
device transaction. No worker thread, no lock, no way for a wedged joystick to
stall the UI — a disconnected device just returns nothing and the panel says so.
"""

from __future__ import annotations

import array
import fcntl
import os
import struct
from pathlib import Path

EVENT = struct.Struct("<IhBB")
EVENT_BUTTON = 0x01
EVENT_AXIS = 0x02
EVENT_INIT = 0x80

AXIS_FULL_SCALE = 32767.0

# =============================================================================
# Button numbering: the kernel's is NOT the vehicle's
# =============================================================================
# This is the single most dangerous thing in this file, so it is written out.
#
# The kernel reports buttons in the order the HID descriptor lists them.
# QGroundControl reads joysticks through SDL2, which REORDERS them into
# SDL_GameControllerButton — a fixed logical layout — and sends THAT index in
# MANUAL_CONTROL.buttons. The vehicle's BTNn_FUNCTION parameters were therefore
# configured against SDL's numbering, not the kernel's.
#
# Forwarding the kernel's index straight through is what this station did at
# first, and on 2026-08-07 it did exactly what you would predict: the pilot
# pressed the left bumper (kernel 6) expecting camera tilt, the vehicle read
# BTN6_FUNCTION = arm, and the motors came live. Four independent observations
# on the real vehicle confirmed the offset, and all four fall out of the table
# below: kernel 11 -> arm, 10 -> disarm, 6 and 7 -> the mount_tilt pair.
#
# SDL_GameControllerButton, which is the index the vehicle sees:
#   0 A   1 B   2 X   3 Y   4 BACK   5 GUIDE   6 START   7 LEFTSTICK
#   8 RIGHTSTICK   9 LEFTSHOULDER   10 RIGHTSHOULDER
#   11 DPAD_UP   12 DPAD_DOWN   13 DPAD_LEFT   14 DPAD_RIGHT
SDL_BUTTON_NAMES = (
    "A", "B", "X", "Y", "BACK", "GUIDE", "START", "LEFTSTICK", "RIGHTSTICK",
    "LEFTSHOULDER", "RIGHTSHOULDER",
    "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
)

# kernel js index -> SDL index, for the "Xbox Wireless Controller" over
# Bluetooth on this desktop. The kernel side was READ OFF THE DEVICE
# (buttons 0-14 present, gaps at 2/5/8/9, axes 0-7); the pairing is SDL's
# published mapping for this pad.
XBOX_WIRELESS_JS_TO_SDL = {
    0: 0,    # A
    1: 1,    # B
    3: 2,    # X
    4: 3,    # Y
    6: 9,    # LB -> LEFTSHOULDER
    7: 10,   # RB -> RIGHTSHOULDER
    10: 4,   # View  -> BACK
    11: 6,   # Menu  -> START
    12: 5,   # Xbox  -> GUIDE
    13: 7,   # left stick click
    14: 8,   # right stick click
}

BUTTON_MAPS = {
    "Xbox Wireless Controller": XBOX_WIRELESS_JS_TO_SDL,
    "Xbox Series X Controller": XBOX_WIRELESS_JS_TO_SDL,
    "Microsoft X-Box One S pad": XBOX_WIRELESS_JS_TO_SDL,
}

# The D-pad is NOT a button on this pad — the kernel reports it as a HAT, i.e.
# two axes that rest at 0 and snap to +/-1. That is why pressing it lit nothing
# in the input visualiser and sent nothing to the vehicle: the button loop never
# saw it. SDL turns the same hat into four buttons, so that is what we do.
HAT_AXIS_X, HAT_AXIS_Y = 6, 7
HAT_TO_SDL = {"up": 11, "down": 12, "left": 13, "right": 14}
HAT_THRESHOLD = 0.5

# JSIOCGNAME(len): _IOC(_IOC_READ, 'j', 0x13, len)
def _jsiocgname(length: int) -> int:
    return (2 << 30) | (length << 16) | (ord("j") << 8) | 0x13


def devices() -> list[Path]:
    """Every joystick node present, in order."""
    return sorted(Path("/dev/input").glob("js[0-9]*"))


class Joystick:
    """One device. Poll it; read :attr:`axes` and :attr:`buttons`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fd = os.open(str(self.path), os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, float] = {}      # corrected: rest is always 0
        self.raw: dict[int, float] = {}
        self.rest: dict[int, float] = {}
        self.buttons: dict[int, bool] = {}
        self.name = self._read_name()
        self.events = 0

    def _read_name(self) -> str:
        buf = array.array("B", [0] * 128)
        try:
            fcntl.ioctl(self.fd, _jsiocgname(len(buf)), buf)
        except OSError:
            return self.path.name
        return buf.tobytes().split(b"\x00")[0].decode("utf-8", "replace") or self.path.name

    def poll(self) -> bool:
        """Drain pending events. True if anything changed, False if idle.

        Returns False and closes the device on unplug, which the caller sees as
        ``alive`` going away — a joystick that vanishes mid-dive must not look
        like a joystick holding all axes at their last value.
        """
        changed = False
        while self.fd is not None:
            try:
                data = os.read(self.fd, EVENT.size)
            except BlockingIOError:
                break
            except OSError:
                self.close()
                return False
            if len(data) < EVENT.size:
                break
            _t, value, kind, number = EVENT.unpack(data)
            kind &= ~EVENT_INIT          # synthetic initial state counts too
            if kind == EVENT_AXIS:
                v = max(-1.0, min(1.0, value / AXIS_FULL_SCALE))
                self.raw[number] = v
                if number not in self.rest:
                    # An analogue trigger rests at FULL SCALE, not at centre:
                    # measured on an Xbox Wireless Controller, axes 4 and 5 read
                    # -1.000 untouched. Mapped to a vehicle axis that becomes a
                    # standing command the pilot never gave — the GUI opened with
                    # heave at +0.60 and held it. So the first value an axis
                    # reports becomes its zero, but only when it is nowhere near
                    # centre; a stick that really is centred keeps a true zero.
                    self.rest[number] = v if abs(v) > 0.5 else 0.0
                self.axes[number] = max(-1.0, min(1.0, v - self.rest[number]))
            elif kind == EVENT_BUTTON:
                self.buttons[number] = bool(value)
            else:
                continue
            self.events += 1
            changed = True
        return changed

    @property
    def alive(self) -> bool:
        return self.fd is not None

    def triggers(self) -> list[int]:
        """Axes that were auto-zeroed because they rest at full scale."""
        return sorted(n for n, r in self.rest.items() if r != 0.0)

    # ------------------------------------------------------- vehicle numbering
    @property
    def button_map(self) -> dict[int, int]:
        """kernel index -> SDL index for this pad, or {} if we do not know it."""
        return BUTTON_MAPS.get(self.name, {})

    def hat_buttons(self) -> set[int]:
        """The D-pad hat, as the four SDL buttons SDL would report."""
        out: set[int] = set()
        x = self.raw.get(HAT_AXIS_X, 0.0)
        y = self.raw.get(HAT_AXIS_Y, 0.0)
        if x <= -HAT_THRESHOLD:
            out.add(HAT_TO_SDL["left"])
        elif x >= HAT_THRESHOLD:
            out.add(HAT_TO_SDL["right"])
        # The kernel's hat Y is positive DOWN, matching screen coordinates.
        if y <= -HAT_THRESHOLD:
            out.add(HAT_TO_SDL["up"])
        elif y >= HAT_THRESHOLD:
            out.add(HAT_TO_SDL["down"])
        return out

    def vehicle_buttons(self, translate: bool = True) -> set[int]:
        """Buttons in the numbering the VEHICLE uses (SDL), plus the D-pad.

        ``translate=False`` forwards kernel indices unchanged. That is the old
        behaviour and it is wrong on every pad SDL reorders — it is kept only so
        a pad we have no map for can still be driven by someone who has worked
        out their own BTNn_FUNCTION assignment.
        """
        held = {n for n, down in self.buttons.items() if down}
        if not translate:
            return held | self.hat_buttons()
        mapping = self.button_map
        if not mapping:
            return held | self.hat_buttons()
        return {mapping[n] for n in held if n in mapping} | self.hat_buttons()

    def unmapped(self) -> list[int]:
        """Held buttons this pad's map has no SDL index for — never sent."""
        mapping = self.button_map
        if not mapping:
            return []
        return sorted(n for n, down in self.buttons.items()
                      if down and n not in mapping)

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


class JoystickReader:
    """Finds a device, keeps it, and re-finds it after an unplug.

    Deliberately silent about not having one: a station with no gamepad is a
    perfectly normal station, so this reports ``None`` for the name and the
    panel shows the keyboard row alone.
    """

    RESCAN_S = 2.0

    def __init__(self, path: str | None = None):
        self.wanted = None if path in (None, "", "auto") else Path(path)
        self.js: Joystick | None = None
        self._next_scan = 0.0

    def poll(self, now: float) -> bool:
        if self.js is not None and not self.js.alive:
            self.js = None
        if self.js is None:
            if now < self._next_scan:
                return False
            self._next_scan = now + self.RESCAN_S
            for path in ([self.wanted] if self.wanted else devices()):
                if path is None or not Path(path).exists():
                    continue
                try:
                    self.js = Joystick(path)
                    break
                except OSError:
                    continue
            if self.js is None:
                return False
        return self.js.poll()

    @property
    def name(self) -> str | None:
        return self.js.name if self.js is not None else None

    @property
    def axes(self) -> dict[int, float]:
        return self.js.axes if self.js is not None else {}

    @property
    def triggers(self) -> list[int]:
        return self.js.triggers() if self.js is not None else []

    @property
    def buttons(self) -> dict[int, bool]:
        return self.js.buttons if self.js is not None else {}

    @property
    def has_map(self) -> bool:
        return bool(self.js is not None and self.js.button_map)

    def vehicle_buttons(self, translate: bool = True) -> set[int]:
        return self.js.vehicle_buttons(translate) if self.js is not None else set()

    def unmapped(self) -> list[int]:
        return self.js.unmapped() if self.js is not None else []

    def close(self) -> None:
        if self.js is not None:
            self.js.close()
            self.js = None


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero the centre, then rescale so the usable range still reaches 1.0.

    Without the rescale a deadzone quietly costs full deflection — the stick
    hits its mechanical stop at 0.92 of the command, and a pilot fighting a
    current never gets the last 8% they can see on the bar.
    """
    if deadzone <= 0.0:
        return value
    if abs(value) <= deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return scaled if value > 0 else -scaled
