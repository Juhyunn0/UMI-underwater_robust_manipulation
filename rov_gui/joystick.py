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

Neither the axis numbers nor the button numbers mean anything on their own, and
they are wrong in DIFFERENT ways: buttons are reordered by SDL (what the vehicle
expects), and axes are renumbered by the kernel driver (USB vs Bluetooth are not
the same pad as far as ``js0`` is concerned). Both are therefore asked for, not
assumed — see the two numbered sections below.
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

# =============================================================================
# Axis numbering: it changes with the CABLE
# =============================================================================
# The buttons above are reordered by SDL. The AXES are worse: the same physical
# pad enumerates its axes differently depending on how it is connected, because
# a different kernel driver is speaking.
#
#   over Bluetooth (hid-generic)  0,1 left stick   2,3 right stick  4,5 triggers
#   over USB       (xpad)         0,1 left stick   3,4 right stick  2,5 triggers
#
# So a mapping pinned to axis NUMBERS is only correct for one transport. On
# 2026-08-17 the pilot plugged the pad in and the right stick's horizontal —
# yaw — started commanding HEAVE while yaw did nothing at all: heave was pinned
# to axis 3 (which USB calls right-stick-X) and yaw to axis 2 (which USB calls
# the LEFT TRIGGER, auto-zeroed at rest, hence dead).
#
# The kernel will simply tell us, so we ask instead of guessing. JSIOCGAXMAP
# returns the ABS_* code behind every js axis index, and the codes are stable
# across drivers even when the indices are not:
#
#   xpad (USB)      ABS_X ABS_Y ABS_Z  ABS_RX ABS_RY ABS_RZ  ABS_HAT0X/Y
#                   Lx    Ly    LT     Rx     Ry     RT
#   hid-generic(BT) ABS_X ABS_Y ABS_Z  ABS_RZ ...            ABS_HAT0X/Y
#                   Lx    Ly    Rx     Ry
#
# The disambiguation is one rule: if the pad reports BOTH ABS_RX and ABS_RY
# they are the right stick and Z/RZ are triggers; otherwise Z/RZ are the right
# stick. Either way the right stick is found by name, not by number.
ABS_X, ABS_Y, ABS_Z = 0x00, 0x01, 0x02
ABS_RX, ABS_RY, ABS_RZ = 0x03, 0x04, 0x05
ABS_HAT0X, ABS_HAT0Y = 0x10, 0x11
ABS_CNT = 0x40

# vehicle axis -> (stick, sign). The signs are the flying convention this panel
# has always used: stick up is +surge / +heave (both sticks read negative up),
# stick right is +sway / +yaw.
AXIS_ROLES = {
    "surge": ("left_y", -1.0),
    "sway":  ("left_x", +1.0),
    "yaw":   ("right_x", +1.0),
    "heave": ("right_y", -1.0),
}

# What the numbers were before this was detected: the Bluetooth pad probed on
# 2026-08-06. Used only when the ioctl fails, so an unreadable pad behaves
# exactly as it did before rather than not at all.
DEFAULT_STICKS = {"left_x": 0, "left_y": 1, "right_x": 2, "right_y": 3}

# BTN_* code -> SDL_GameControllerButton. This is the same table as
# XBOX_WIRELESS_JS_TO_SDL above, expressed in the kernel's stable BTN_* codes
# instead of the driver's shifting indices — feeding the Bluetooth pad's own
# btnmap through it reproduces that hand-measured table exactly, gaps included
# (pinned by test_offline.py), and it also covers the USB pad, whose kernel
# indices are packed 0..10 with no gaps at all.
BTN_TO_SDL = {
    0x130: 0,    # BTN_A
    0x131: 1,    # BTN_B
    0x133: 2,    # BTN_X
    0x134: 3,    # BTN_Y
    0x13a: 4,    # BTN_SELECT  -> BACK  (View)
    0x13c: 5,    # BTN_MODE    -> GUIDE (Xbox)
    0x13b: 6,    # BTN_START   -> START (Menu)
    0x13d: 7,    # BTN_THUMBL
    0x13e: 8,    # BTN_THUMBR
    0x136: 9,    # BTN_TL      -> LEFTSHOULDER
    0x137: 10,   # BTN_TR      -> RIGHTSHOULDER
    0x220: 11,   # BTN_DPAD_UP
    0x221: 12,   # BTN_DPAD_DOWN
    0x222: 13,   # BTN_DPAD_LEFT
    0x223: 14,   # BTN_DPAD_RIGHT
}
KEY_MAX, BTN_MISC = 0x2FF, 0x100


# JSIOCGNAME(len): _IOC(_IOC_READ, 'j', 0x13, len)
def _jsiocgname(length: int) -> int:
    return (2 << 30) | (length << 16) | (ord("j") << 8) | 0x13


def _jsioc_read(nr: int, size: int) -> int:
    """_IOC(_IOC_READ, 'j', nr, size) — JSIOCGAXES/AXMAP/BTNMAP."""
    return (2 << 30) | (size << 16) | (ord("j") << 8) | nr


def default_axis_spec(name: str) -> tuple[int, float]:
    """(js index, sign) for a vehicle axis with no device to ask."""
    stick, sign = AXIS_ROLES[name]
    return DEFAULT_STICKS[stick], sign


def devices() -> list[Path]:
    """Every joystick node present, in order."""
    return sorted(Path("/dev/input").glob("js[0-9]*"))


class Joystick:
    """One device. Poll it; read :attr:`axes` and :attr:`buttons`."""

    # Class-level fallbacks, so an instance whose probe failed — or one built
    # without a device at all, as the tests do — behaves like the pad these
    # numbers were measured on instead of raising.
    sticks: dict[str, int] = dict(DEFAULT_STICKS)
    hat_x, hat_y = HAT_AXIS_X, HAT_AXIS_Y
    derived_map: dict[int, int] = {}
    probed = False

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fd = os.open(str(self.path), os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, float] = {}      # corrected: rest is always 0
        self.raw: dict[int, float] = {}
        self.rest: dict[int, float] = {}
        self.buttons: dict[int, bool] = {}
        self.name = self._read_name()
        self.events = 0
        self._probe()

    def _read_name(self) -> str:
        buf = array.array("B", [0] * 128)
        try:
            fcntl.ioctl(self.fd, _jsiocgname(len(buf)), buf)
        except OSError:
            return self.path.name
        return buf.tobytes().split(b"\x00")[0].decode("utf-8", "replace") or self.path.name

    # ------------------------------------------------------- what IS each axis
    def _probe(self) -> None:
        """Ask the driver what its axes and buttons actually ARE.

        Everything here is per-instance and falls back to the class defaults, so
        a driver that refuses the ioctls (or a pad behind an adapter) still
        flies on the old hard-coded numbers rather than not at all.
        """
        axmap = self._axis_codes()
        if axmap:
            self.probed = True
            self.sticks = self._sticks_from(axmap)
            where = {code: i for i, code in enumerate(axmap)}
            self.hat_x = where.get(ABS_HAT0X, HAT_AXIS_X)
            self.hat_y = where.get(ABS_HAT0Y, HAT_AXIS_Y)
        btnmap = self._button_codes()
        if btnmap:
            self.derived_map = {i: BTN_TO_SDL[code]
                                for i, code in enumerate(btnmap)
                                if code in BTN_TO_SDL}

    def _axis_codes(self) -> list[int]:
        """ABS_* code behind every js axis index, in index order."""
        try:
            n = array.array("B", [0])
            fcntl.ioctl(self.fd, _jsioc_read(0x11, 1), n)      # JSIOCGAXES
            buf = array.array("B", [0] * ABS_CNT)
            fcntl.ioctl(self.fd, _jsioc_read(0x32, ABS_CNT), buf)  # JSIOCGAXMAP
        except OSError:
            return []
        return list(buf[:min(n[0], ABS_CNT)])

    def _button_codes(self) -> list[int]:
        """BTN_* code behind every js button index, in index order."""
        size = KEY_MAX - BTN_MISC + 1
        try:
            n = array.array("B", [0])
            fcntl.ioctl(self.fd, _jsioc_read(0x12, 1), n)      # JSIOCGBUTTONS
            buf = array.array("H", [0] * size)
            fcntl.ioctl(self.fd, _jsioc_read(0x34, size * 2), buf)  # JSIOCGBTNMAP
        except OSError:
            return []
        return list(buf[:min(n[0], size)])

    @staticmethod
    def _sticks_from(axmap: list[int]) -> dict[str, int]:
        """Which js index is which physical stick half.

        The one rule that separates the two Xbox layouts: a pad that reports
        ABS_RX *and* ABS_RY has its right stick there and its triggers on
        Z/RZ (xpad, USB); a pad that does not has its right stick on Z/RZ
        (hid-generic, Bluetooth). Anything unrecognised keeps its default,
        which is what the numbers used to be hard-coded to.
        """
        where = {code: i for i, code in enumerate(axmap)}
        sticks = dict(DEFAULT_STICKS)
        if ABS_X in where:
            sticks["left_x"] = where[ABS_X]
        if ABS_Y in where:
            sticks["left_y"] = where[ABS_Y]
        if ABS_RX in where and ABS_RY in where:
            sticks["right_x"], sticks["right_y"] = where[ABS_RX], where[ABS_RY]
        elif ABS_Z in where and ABS_RZ in where:
            sticks["right_x"], sticks["right_y"] = where[ABS_Z], where[ABS_RZ]
        return sticks

    def axis_spec(self, name: str) -> tuple[int, float]:
        """(js index, sign) for a VEHICLE axis — surge/sway/heave/yaw."""
        stick, sign = AXIS_ROLES[name]
        return self.sticks.get(stick, DEFAULT_STICKS[stick]), sign

    def layout(self) -> dict[str, tuple[int, float]]:
        """Every vehicle axis at once, for logging and for the window."""
        return {name: self.axis_spec(name) for name in AXIS_ROLES}

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
        """kernel index -> SDL index for this pad, or {} if we do not know it.

        Derived from the driver's own BTN_* map when it gave us one — that is
        the only version that is right on BOTH transports, since the USB pad
        packs its buttons 0..10 where the Bluetooth pad leaves gaps at 2/5/8/9.
        The name table is the fallback for a driver that refuses the ioctl.
        """
        return self.derived_map or BUTTON_MAPS.get(self.name, {})

    def hat_buttons(self) -> set[int]:
        """The D-pad hat, as the four SDL buttons SDL would report."""
        out: set[int] = set()
        x = self.raw.get(self.hat_x, 0.0)
        y = self.raw.get(self.hat_y, 0.0)
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

    @property
    def probed(self) -> bool:
        """True when the driver told us what its axes are, rather than us guessing."""
        return bool(self.js is not None and self.js.probed)

    def axis_spec(self, name: str) -> tuple[int, float]:
        """(js index, sign) for a vehicle axis, detected from the device."""
        if self.js is None:
            stick, sign = AXIS_ROLES[name]
            return DEFAULT_STICKS[stick], sign
        return self.js.axis_spec(name)

    def layout(self) -> dict[str, tuple[int, float]]:
        return {name: self.axis_spec(name) for name in AXIS_ROLES}

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
