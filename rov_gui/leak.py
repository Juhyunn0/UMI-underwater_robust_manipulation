#!/usr/bin/env python3
"""
leak.py — water-in-the-enclosure alerting, the way ArduSub actually reports it.

This is the feature QGroundControl has and this station did not. QGC does not
have a leak *widget*: what it has is the vehicle-messages stream, and ArduSub
shouting into it. So the work is not drawing an indicator — it is receiving a
message the station was throwing away, and being honest about the several ways
the alert can silently not exist.

How ArduSub reports a leak
--------------------------
Over exactly ONE outbound message: ``STATUSTEXT``, payload ``"Leak Detected"``,
severity ``MAV_SEVERITY_CRITICAL`` (2), from ``Sub::failsafe_leak_check()``.
There is no leak field, no dedicated message, and no SYS_STATUS bit — the
``MAV_SYS_STATUS_SENSOR_LEAK`` bit that exists in the MAVLink spec flows the
OTHER way (a companion computer telling the autopilot about a remote leak
sensor), so polling SYS_STATUS for leak health finds nothing, forever.

Three consequences shape everything below:

* **It repeats, it does not latch on the wire.** ArduSub re-sends every 20 s
  for as long as the detector is wet. So "still leaking" is a heartbeat, and
  the topside state is a hold with a timeout — :data:`LEAK_HOLD_S`.
* **There is no "leak cleared" message.** The vehicle clears its own failsafe
  after a 3 s detector cooldown (``LEAKDETECTOR_COOLDOWN_MS``) and says nothing
  about it: no ``send_text``, only a dataflash record. So clearance is inferred
  from silence, and the hold is generous — two missed repeats.

  One qualifier, because "completely silent" would be wrong: ``failsafe.leak``
  feeds ``Sub::any_failsafe_triggered()``, which drives ``system_status`` in
  every 1 Hz HEARTBEAT (``MAV_STATE_CRITICAL`` while any failsafe holds). Both
  the trigger and the recovery are therefore visible there. It is not used
  here because that field is AGGREGATED over ten failsafes — battery, EKF,
  crash, GCS loss, internal pressure and temperature among them — so it can
  say "something is wrong", never "the leak stopped". Using it would let a
  battery failsafe hold the leak banner up.
* **Silence is ambiguous.** ``FS_LEAK_ENABLE=0`` suppresses the warning itself,
  not just the surface action, and ``LEAK1_PIN=-1`` (the ArduSub default) means
  no detector backend is ever constructed. Either one makes a *flooding*
  vehicle completely silent. So an unconfigured vehicle and a dry vehicle look
  identical on the link, and this module refuses to call that "dry": without
  a parameter readback the state is ``None``, per state.py's rule that unknown
  is never a plausible-looking default.

Matching the text exactly, and why
----------------------------------
``text.strip() == "Leak Detected"`` with ``severity == 2``, not
``"leak" in text.lower()``. ArduSub emits two OTHER strings containing "leak"
— ``"Leak detector %u error. Please set SERVO%u_FUNCTION to GPIO"`` (WARNING)
and ``"Leak detector %u pin (servo %u) auto-set to GPIO"`` (INFO) — both from
``Sub::update_leak_pins()`` at boot. A substring match turns a configuration
notice into a flooding alarm on every startup, and an alarm that cries wolf at
boot is one the pilot learns to dismiss.

Internal pressure, the second signal
------------------------------------
A separate ArduSub failsafe watches the enclosure barometer and emits
``"Internal pressure critical!"`` (WARNING) every 30 s above ``FS_PRESS_MAX``
(default 105000 Pa = 1050 hPa). It is tracked here too because rising internal
pressure is the EARLY indicator: it moves while water is still finding its way
in, before a leak pad is wet. The absolute reading is handled by the backend
(``SCALED_PRESSURE.press_abs`` is the enclosure baro; ``SCALED_PRESSURE2`` is
the external Bar30) — this module owns only the alarm.

Everything here is pure logic on (text, severity, time), so it is testable
without a vehicle, a link, or Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import Conn, now

# ---------------------------------------------------------------- the wire
LEAK_TEXT = "Leak Detected"
LEAK_SEVERITY = 2                  # MAV_SEVERITY_CRITICAL
PRESSURE_TEXT = "Internal pressure critical!"
PRESSURE_SEVERITY = 4              # MAV_SEVERITY_WARNING

# ArduSub re-sends the leak warning every 20 s and the pressure warning every
# 30 s while the condition holds. The hold is two missed repeats plus slack:
# long enough that a dropped packet on a noisy tether cannot clear a flooding
# alarm, short enough that the alert does not outlive the recovery by minutes.
LEAK_REPEAT_S = 20.0
LEAK_HOLD_S = 50.0
PRESSURE_REPEAT_S = 30.0
PRESSURE_HOLD_S = 75.0

# The boot-time configuration notices (see the module docstring). Recognised so
# a misconfigured detector can be REPORTED rather than mistaken for a leak.
_CFG_ERROR_PREFIX = "Leak detector"


@dataclass
class LeakState:
    """What the station can honestly say about water in the enclosure."""

    leak: bool | None = None       # True wet, False dry, None NOT KNOWN
    pressure_alarm: bool = False   # internal pressure over FS_PRESS_MAX
    since_s: float | None = None   # how long ago the last alert arrived
    detail: str = ""               # one line for the sensor row
    conn: Conn = Conn.OFFLINE      # how to paint the row
    armed: bool | None = None      # is the DETECTOR configured to report?

    @property
    def alarm(self) -> bool:
        """Does this need the pilot's eyes right now?"""
        return bool(self.leak) or self.pressure_alarm


class LeakMonitor:
    """Fold ArduSub's STATUSTEXT stream into a leak state.

    Feed it every STATUSTEXT (:meth:`note_statustext`) and, when they are
    known, the two parameters that decide whether the vehicle would report a
    leak at all (:meth:`note_config`). Ask :meth:`state` whenever you paint.
    """

    def __init__(self) -> None:
        self._leak_t: float | None = None
        self._press_t: float | None = None
        self._cfg_error = ""            # detector misconfiguration, as reported
        self._fs_leak_enable: int | None = None
        self._leak1_pin: int | None = None
        self.leaks_seen = 0             # how many alerts this session

    # ------------------------------------------------------------- inputs
    def note_statustext(self, text, severity, t: float | None = None) -> str:
        """Classify one STATUSTEXT. Returns "leak" | "pressure" | "config" | "".

        The return value is for the caller's log line — the state itself is
        read back through :meth:`state`.
        """
        t = now() if t is None else float(t)
        # MAVLink pads char[50] with NULs, and pymavlink hands the padding
        # straight through; strip it before comparing or nothing ever matches.
        s = str(text or "").replace("\x00", "").strip()
        try:
            sev = int(severity)
        except (TypeError, ValueError):
            return ""
        if s == LEAK_TEXT and sev <= LEAK_SEVERITY:
            if self._leak_t is None or (t - self._leak_t) > LEAK_HOLD_S:
                self.leaks_seen += 1        # a NEW event, not a repeat
            self._leak_t = t
            return "leak"
        if s == PRESSURE_TEXT and sev <= PRESSURE_SEVERITY:
            self._press_t = t
            return "pressure"
        # "Leak detector 1 error. Please set SERVO..." — lowercase 'd', so it
        # can never be confused with the alarm above. It means the alarm does
        # not work, which is its own thing worth saying out loud.
        if s.startswith(_CFG_ERROR_PREFIX) and "error" in s:
            self._cfg_error = s
            return "config"
        return ""

    def note_config(self, fs_leak_enable=None, leak1_pin=None) -> None:
        """The two parameters that decide whether silence means anything.

        ``FS_LEAK_ENABLE``: 0 disabled / 1 warn only / 2 surface. 0 suppresses
        the STATUSTEXT as well as the action. ``LEAK1_PIN``: -1 disabled (the
        ArduSub default), 27 = the Navigator's built-in pad.
        """
        if fs_leak_enable is not None:
            self._fs_leak_enable = int(fs_leak_enable)
        if leak1_pin is not None:
            self._leak1_pin = int(leak1_pin)

    # -------------------------------------------------------------- output
    def detector_armed(self) -> bool | None:
        """Would this vehicle tell us about a leak? None = we have not asked."""
        if self._fs_leak_enable is None and self._leak1_pin is None:
            return None
        if self._fs_leak_enable == 0 or (self._leak1_pin is not None
                                         and self._leak1_pin < 0):
            return False
        if self._fs_leak_enable is None or self._leak1_pin is None:
            return None                 # half the answer is not an answer
        return True

    def state(self, t: float | None = None) -> LeakState:
        t = now() if t is None else float(t)
        armed = self.detector_armed()
        press = (self._press_t is not None
                 and (t - self._press_t) <= PRESSURE_HOLD_S)

        if self._leak_t is not None and (t - self._leak_t) <= LEAK_HOLD_S:
            age = t - self._leak_t
            return LeakState(
                leak=True, pressure_alarm=press, since_s=age, armed=armed,
                detail=f"LEAK DETECTED — last report {age:.0f}s ago",
                conn=Conn.FAULT)

        if press:
            age = t - self._press_t
            return LeakState(
                leak=(False if armed else None), pressure_alarm=True,
                since_s=age, armed=armed,
                detail=f"INTERNAL PRESSURE CRITICAL — {age:.0f}s ago",
                conn=Conn.FAULT)

        if self._cfg_error:
            # The vehicle told us its own detector is wired wrong. That is a
            # louder fact than "no leak reported", because it means the alarm
            # this panel is showing cannot fire.
            return LeakState(leak=None, armed=False, conn=Conn.FAULT,
                             detail=self._cfg_error[:48])

        if self._leak_t is not None:
            # Had one, has stopped repeating. Say so rather than resetting to a
            # clean "dry": the operator should know this dive had water in it.
            cleared = t - self._leak_t
            if armed:
                return LeakState(leak=False, armed=armed, since_s=cleared,
                                 conn=Conn.DEGRADED,
                                 detail=f"leak CLEARED {cleared / 60:.0f} min "
                                        f"ago ({self.leaks_seen} this session)")
            return LeakState(leak=None, armed=armed, since_s=cleared,
                             conn=Conn.DEGRADED,
                             detail=f"leak reported {cleared / 60:.0f} min ago; "
                                    f"detector state unknown")

        if armed is True:
            return LeakState(leak=False, armed=True, conn=Conn.ONLINE,
                             detail="dry")
        if armed is False:
            why = ("FS_LEAK_ENABLE=0" if self._fs_leak_enable == 0
                   else f"LEAK1_PIN={self._leak1_pin}")
            return LeakState(leak=None, armed=False, conn=Conn.DEGRADED,
                             detail=f"detector DISABLED ({why}) — a flood "
                                    f"would be silent")
        return LeakState(leak=None, armed=None, conn=Conn.DEGRADED,
                         detail="no leak reported; detector config unread")
