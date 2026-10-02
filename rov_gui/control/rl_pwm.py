#!/usr/bin/env python3
"""rl_pwm.py — the per-thruster PWM transport of LOW mode ``rl_pwm`` (2026-09-30), the parts that need no vehicle.

WHAT THIS IS. RL_controller trains two kinds of policy (rov_rl/obs_spec.py): ``axes`` — 4 MANUAL_CONTROL axes mixed by
ArduSub, LOW mode ``rl`` — and ``pwm`` — 8 per-thruster throttles with the mixer bypassed. Until 2026-09-30 the station
refused pwm policies (HwRl still does, by name); this module is what LOW mode ``rl_pwm`` adds, transcribed from
/home/bdml/Desktop/RL_controller/deploy/{pwm.py, ardusub_bridge.py} so the station and that repo's own runner speak the
same wire.

THE TRANSPORT IS ``pwm_lua_override`` AND NOTHING ELSE. Stock ArduSub has no in-flight per-motor command. SERVOn_FUNCTION
= RCIN passthrough was rejected (RL_controller design review 2026-09-27): passthrough outputs ignore ARM/DISARM and every
failsafe, and on a receiver-less ROV a dead topside latches the last pulse forever. Instead SERVO1..8 stay Motor1..8 and
a vehicle-side Lua script (RL_controller/deploy/ardusub/rl_pwm_override.lua, a copy sits beside the shipped policy)
forces each output with a 100 ms timeout only while RC_CHANNELS_OVERRIDE frames (channels 9..16 = motors 1..8) are fresh
AND the vehicle is armed AND in MANUAL. When the frames stop, AP_Motors resumes — and its input is the NEUTRAL
MANUAL_CONTROL this station keeps sending, so the fallback is 1500 us. DISARM, E-STOP and every failsafe still stop the
thrusters. After a topside crash the last policy pulse is held ~120 ms (script STALE_MS 100 + its 20 ms loop), then
the script forces 1500 us for one more 100 ms before AP_Motors takes the outputs back [스펙: rl_pwm_override.lua].

UNTESTED ON THE VEHICLE [예측]. The Lua script has never run on the Navigator and no pulse from this path has reached a
thruster (2026-09-30). The gate below is what stands between that and the water: it is read back from the vehicle, not
assumed, and it REFUSES until every item holds —

  * SERVO1..8_FUNCTION = 33..40 (Motor1..8; not RCIN passthrough), and no SERVO1..16 output on the RCIN passthrough of a
    motor channel (a light / camera / gripper on RCIN9..16 would follow a thruster);
  * MOT_1..8_DIRECTION == the exported ``motor_direction`` — raw pulses skip the mixer's direction flip, so one wrong
    parameter reverses one thruster silently;
  * SCR_ENABLE 1 and SCR_USER1 1 (scripting on, THIS script armed);
  * RC_OPTIONS bit 1 clear, RC_OVERRIDE_TIME > 0, RC9..16_OPTION 0 (overrides accepted, the channels do nothing else);
  * SYSID_MYGCS == this station's source system (the script drops frames from anyone else);
  * the script's NAMED_VALUE_FLOAT "RLPWM" heartbeat within 2 s; and, once pulses flow, RLPWM = 1 within 1 s;
  * a MAVLink 2 link (RC_CHANNELS_OVERRIDE has no channels 9..18 on v1);
  * all of the above said by the vehicle within the last 30 s (re-asked every 10 s while RL_PWM is selected), and motors
    1..8 on RC channels 9..16 in exactly that order.

Before any water run: propellers off on the bench, then a per-thruster sign probe (RL_controller/README.md §10.3).

This module is numpy only (no Qt, no pymavlink) so the gate and the conversions can be tested without either.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

PWM_NEUTRAL_US = 1500
PWM_SPAN_US = 400                 # +-400 us = +-1 throttle (1100..1900)
PWM_MIN_US, PWM_MAX_US = 1100, 1900
N_MOTORS = 8
#: RC input channels that carry motors 1..8 to the Lua script. Sub-4.5's joystick uses 1..8 and the firmware drops 17/18.
PWM_CHANNELS_DEFAULT = (9, 10, 11, 12, 13, 14, 15, 16)
RC_OVERRIDE_IGNORE = 65535        # "leave this channel as it is" on every channel
RC_OVERRIDE_N = 18                # chan1..chan18 of the MAVLink 2 message
MOTOR_FUNCTION_BASE = 32          # SERVOn_FUNCTION "Motor k" = 32 + k (33..40) [스펙 ArduPilot SRV_Channel k_motor1..8]
RCIN_FUNCTION_BASE = 50           # SERVOn_FUNCTION "RCINk passthrough" = 50 + k (51..66) [스펙 k_rcin1..16]
SCRIPT_HEARTBEAT_NAME = "RLPWM"
SCRIPT_HEARTBEAT_STALE_S = 2.0
SCRIPT_ENGAGE_GRACE_S = 1.0
PARAM_BATCH = 8                   # Sub-4.5 queues 20 pending PARAM_REQUEST_READ; a 30-request burst loses ~10 silently
PARAM_RETRY_S = 0.5
#: A gate parameter counts only if the vehicle said it within this long. "Read back, not assumed" has to stay true for
#: a GUI that runs for hours: a reboot, a parameter file loaded in QGC or a flipped MOT_n_DIRECTION must not be judged
#: from a value cached at start-up (safety review 2026-09-30). The sink re-asks every PARAM_REFRESH_S while anyone
#: wants the transport, so a healthy vehicle never ages out.
PARAM_FRESH_S = 30.0
PARAM_REFRESH_S = 10.0
RCIN_MAPPED_FUNCTION_BASE = 139   # SERVOn_FUNCTION "RCINkScaled" = 139 + k (140..155) [미확인: ArduPilot k_rcin1_mapped]
PWM_GATE_PARAMS = ("SCR_ENABLE", "SCR_USER1", "RC_OPTIONS", "RC_OVERRIDE_TIME", "SYSID_MYGCS")
#: |throttle| ceiling on the wire the FIRST time this path meets water (RL_controller run_policy.py --pwm-cap default).
PWM_CAP_FIRST_WATER = 0.25
FLU_TO_FRD = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])


# ----------------------------------------------------------------------------- conversions
def throttles_to_pwm(thr, cap: float = 1.0) -> tuple:
    """throttle (8,) in [-1, 1] -> 8 integer pulse widths [us], |throttle| clipped to ``cap`` first."""
    c = float(max(0.0, min(1.0, cap)))
    t = np.clip(np.asarray(thr, dtype=float), -c, c)
    if not np.all(np.isfinite(t)):
        raise ValueError("throttles_to_pwm: non-finite throttle")
    return tuple(int(v) for v in np.round(PWM_NEUTRAL_US + PWM_SPAN_US * t))


def pwm_to_action(pwm_us, throttle_max: float = 1.0) -> np.ndarray:
    """The 8 pulses that actually left [us] -> the policy's action units (throttle / pwm_throttle_max): what the network
    is fed back as its previous action, exactly as the APPLIED action was during training."""
    tm = max(1e-6, float(throttle_max))
    return np.clip((np.asarray(pwm_us, dtype=float) - PWM_NEUTRAL_US) / PWM_SPAN_US / tm, -1.0, 1.0)


def _fixed_channels(channels) -> tuple:
    """The channel list, which must be 9..16 IN THAT ORDER. The vehicle's script maps chan(8+m) -> motor m and nothing
    else; a reordered list here would pass every other check and silently swap thrusters under a closed-loop policy
    (safety review 2026-09-30). It is a parameter only so a caller can state what it believes."""
    ch = tuple(int(c) for c in channels)
    if ch != PWM_CHANNELS_DEFAULT:
        raise ValueError(f"rl_pwm: motors 1..8 ride RC channels {list(PWM_CHANNELS_DEFAULT)} in that order and nothing "
                         f"else (rl_pwm_override.lua is hard-wired to it); got {list(ch)}")
    return ch


def pack_rc_override(pwm_us, channels: Iterable[int] = PWM_CHANNELS_DEFAULT) -> list:
    """8 pulses -> the 18 channel values of RC_CHANNELS_OVERRIDE: motors on ``channels``, every other channel 65535.

    Pulses are always explicit 1100..1900: on channels 9..16 the values 0 and 65535 mean "leave" and 65534 "release"
    (Sub-4.5), so none of those may ever stand for a thruster command. Raises rather than clamps a malformed frame."""
    ch = _fixed_channels(channels)
    pw = [int(p) for p in pwm_us]
    if len(pw) != N_MOTORS:
        raise ValueError(f"pack_rc_override: need {N_MOTORS} pulses, got {len(pw)}")
    out = [RC_OVERRIDE_IGNORE] * RC_OVERRIDE_N
    for c, p in zip(ch, pw):
        out[c - 1] = int(min(PWM_MAX_US, max(PWM_MIN_US, p)))
    return out


NEUTRAL_PULSES = (PWM_NEUTRAL_US,) * N_MOTORS


# ----------------------------------------------------------------------------- the exported model
class T200CurveNp:
    """Steady-state throttle -> thrust [N] (RL_controller/deploy/pwm.py, the numpy twin of rov_rl.thrusters.T200Curve)."""

    def __init__(self, cfg: dict):
        self.deadband = float(cfg["deadband"])
        self.rpm_max = float(cfg["rpm_max"])
        self.rpm_pos = [float(x) for x in cfg["rpm_pos"]]
        self.rpm_neg = [float(x) for x in cfg["rpm_neg"]]
        self.kgf_pos = [float(x) for x in cfg["kgf_pos"]]
        self.kgf_neg = [float(x) for x in cfg["kgf_neg"]]

    def thrust(self, thr) -> np.ndarray:
        thr = np.clip(np.asarray(thr, dtype=float), -1.0, 1.0)
        rpm = np.where(thr > self.deadband, self.rpm_pos[0] * thr + self.rpm_pos[1],
                       np.where(thr < -self.deadband, self.rpm_neg[0] * thr + self.rpm_neg[1], 0.0))
        rpm = np.clip(rpm, -self.rpm_max, self.rpm_max)
        a, b, c = self.kgf_pos
        an, bn, cn = self.kgf_neg
        kgf = np.where(rpm > 0, a * rpm * rpm + b * rpm + c, np.where(rpm < 0, an * rpm * rpm + bn * rpm + cn, 0.0))
        return 9.81 * kgf


class PwmModel:
    """``obs_spec.json['output']['pwm_model']`` -> throttles and the NOMINAL wrench they stand for (logging only)."""

    def __init__(self, pwm_model: dict):
        self.n = int(pwm_model.get("num_thrusters", N_MOTORS))
        if self.n != N_MOTORS:
            raise ValueError(f"pwm_model: {self.n} thrusters, this transport carries {N_MOTORS}")
        self.throttle_max = float(pwm_model.get("pwm_throttle_max", 1.0))
        self.cap_trained = float(pwm_model.get("pwm_cap_trained", 1.0))
        self.B_flu = np.asarray(pwm_model["B_flu"], dtype=float)
        if self.B_flu.shape != (6, self.n):
            raise ValueError(f"pwm_model: B_flu is {self.B_flu.shape}, expected (6, {self.n})")
        self.curve = T200CurveNp(pwm_model["t200"])
        self.voltage_scale = float(pwm_model.get("voltage_scale", 1.0))
        self.motor_order = list(pwm_model.get("motor_order", list(range(1, self.n + 1))))
        if self.motor_order != list(range(1, self.n + 1)):
            raise ValueError(f"pwm_model: motor_order {self.motor_order} is not 1..{self.n} — the wire carries motor k on "
                             f"its k-th channel and nothing here reorders it")
        md = pwm_model.get("motor_direction")
        if md is None or len(md) != self.n:
            raise ValueError("pwm_model: no motor_direction (8 values) — the vehicle's MOT_n_DIRECTION cannot be checked")
        self.motor_direction = [int(v) for v in md]
        self.transport = str(pwm_model.get("transport", "pwm_lua_override"))
        if self.transport != "pwm_lua_override":
            raise ValueError(f"pwm_model: transport {self.transport!r} is not 'pwm_lua_override' (the only one implemented)")

    def throttles(self, a) -> np.ndarray:
        thr = np.clip(np.asarray(a, dtype=float), -1.0, 1.0) * self.throttle_max
        return np.clip(thr, -self.cap_trained, self.cap_trained)

    def wrench_ned_from_throttles(self, thr) -> np.ndarray:
        """Nominal steady-state body wrench [X, Y, Z, K, M, N] (NED/FRD, N / N*m) for wire throttles: no delay, no lag,
        unit thruster gains — a number for the log, not a measurement of what the vehicle felt."""
        thr = np.asarray(thr, dtype=float).reshape(-1, self.n)
        return ((self.curve.thrust(thr) * self.voltage_scale) @ self.B_flu.T * FLU_TO_FRD)[0]


# ----------------------------------------------------------------------------- the vehicle gate
class PwmGate:
    """Everything the vehicle must say before a pulse leaves, and while pulses flow (module docstring).

    Fed by the command sink: ``note_param`` for every PARAM_VALUE of the vehicle, ``note_heartbeat`` for every "RLPWM"
    NAMED_VALUE_FLOAT, ``note_pulses`` when a policy frame goes out, ``note_stopped`` when they stop. ``check`` is the
    verdict; ``next_requests`` is what to ask the vehicle for next. No I/O, no clock of its own (``now`` is passed in).
    A parameter older than PARAM_FRESH_S is as good as unread."""

    def __init__(self, motor_direction, sysid: int, channels: Iterable[int] = PWM_CHANNELS_DEFAULT):
        self.motor_direction = [int(v) for v in motor_direction]
        if len(self.motor_direction) != N_MOTORS:
            raise ValueError("PwmGate: motor_direction needs 8 values")
        self.sysid = int(sysid)
        self.channels = _fixed_channels(channels)
        self._names = ([f"SERVO{n}_FUNCTION" for n in range(1, 17)] + [f"MOT_{n}_DIRECTION" for n in range(1, 9)]
                       + list(PWM_GATE_PARAMS) + [f"RC{c}_OPTION" for c in self.channels])
        self._name_set = frozenset(self._names)
        self.servo_functions: dict = {}
        self.params: dict = {}
        self._t_read: dict = {}          # when the vehicle last said each parameter
        self._t_req: dict = {}
        self.t_hb = None                 # last "RLPWM" heartbeat (monotonic), None = never
        self.script_engaged = None       # its last value: True = the script is forcing the outputs
        #: Seconds of policy pulses sent while the script did NOT say it was forcing the outputs. ACCUMULATED across
        #: pauses on purpose: a flow that is interrupted more often than once a second (a coast at the edge of the
        #: tags, a stalled tick) would otherwise restart a per-flow timer every time and never be judged — pulses
        #: nobody is acting on, live again the moment the script's condition is met (safety re-review 2026-09-30).
        #: Reset by a heartbeat that says 1, and by the end of the engagement.
        self.unconfirmed_s = 0.0
        self._t_last_pulse = None        # the previous policy frame (None = the flow is paused)

    # ---- what to read
    def param_names(self) -> list:
        return list(self._names)

    def missing(self, now: float | None = None) -> list:
        """Names the vehicle has not said — or, with ``now``, not said within PARAM_FRESH_S."""
        if now is None:
            return [n for n in self._names if n not in self._t_read]
        return [n for n in self._names if now - self._t_read.get(n, -1e9) > PARAM_FRESH_S]

    def next_requests(self, now: float, refresh_s: float = PARAM_REFRESH_S) -> list:
        """Up to PARAM_BATCH names to ask for now: never read, or read more than ``refresh_s`` ago — each at most every
        PARAM_RETRY_S. The sink calls this only while somebody wants the transport, so a session that never selects
        RL_PWM puts no parameter traffic on the command link at all."""
        out = []
        for name in self._names:
            if now - self._t_read.get(name, -1e9) <= float(refresh_s):
                continue
            if now - self._t_req.get(name, -1e9) < PARAM_RETRY_S:
                continue
            self._t_req[name] = now
            out.append(name)
            if len(out) >= PARAM_BATCH:
                break
        return out

    # ---- what the vehicle said
    def note_param(self, name: str, value, now: float) -> None:
        if name not in self._name_set:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v):
            return                       # a NaN/inf PARAM_VALUE is no answer; the name stays unread and the gate shut
        if name.startswith("SERVO"):
            self.servo_functions[int(name[5:-9])] = int(round(v))
        else:
            self.params[name] = v
        self._t_read[name] = float(now)

    #: Two policy frames further apart than this are not one flow: the gap is not counted as "pulses were flowing".
    FLOW_GAP_S = 0.25

    def note_heartbeat(self, value, now: float) -> None:
        self.t_hb = float(now)
        try:
            self.script_engaged = bool(float(value) >= 0.5)
        except (TypeError, ValueError):
            self.script_engaged = None
        if self.script_engaged is True:
            self.unconfirmed_s = 0.0

    def note_pulses(self, now: float) -> None:
        """A policy frame left at ``now``."""
        now = float(now)
        if self._t_last_pulse is not None and self.script_engaged is not True:
            dt = now - self._t_last_pulse
            if 0.0 < dt <= self.FLOW_GAP_S:
                self.unconfirmed_s += dt
        self._t_last_pulse = now

    def note_paused(self) -> None:
        """The flow stopped but the engagement did not (a coast, a stale follower, a latched fault): the time
        already spent unconfirmed is KEPT."""
        self._t_last_pulse = None

    def note_stopped(self) -> None:
        """The engagement ended (the follower's stop, E-STOP, COMMAND ENABLE off): the next one starts from zero."""
        self._t_last_pulse = None
        self.unconfirmed_s = 0.0

    # ---- the verdict
    def check(self, now: float, link_v2: bool = True) -> tuple:
        """(ok, why). ``why`` names the FIRST thing that does not hold, in the words an operator can act on."""
        if not link_v2:
            return False, "the command link is MAVLink 1 (RC_CHANNELS_OVERRIDE has no channels 9..18)"
        miss = self.missing(now)
        if miss:
            head = ", ".join(miss[:4]) + (f" (+{len(miss) - 4} more)" if len(miss) > 4 else "")
            return False, (f"vehicle parameters not read in the last {PARAM_FRESH_S:.0f} s: {head} "
                           f"(SCR_* never answering = no scripting in this firmware)")
        bad = {n: self.servo_functions.get(n) for n in range(1, 9)
               if self.servo_functions.get(n) != MOTOR_FUNCTION_BASE + n}
        if bad:
            return False, (f"SERVO1..8_FUNCTION must be Motor1..8 (33..40); the vehicle has {bad} — RCIN passthrough is "
                           f"NOT this transport (it bypasses ARM/DISARM and every failsafe)")
        carrier = {n: f for n, f in self.servo_functions.items()
                   if f - RCIN_FUNCTION_BASE in self.channels or f - RCIN_MAPPED_FUNCTION_BASE in self.channels}
        if carrier:
            return False, (f"SERVO{sorted(carrier)}_FUNCTION are RCIN passthrough of a motor channel {carrier}: that "
                           f"output (light / camera / gripper?) would follow a thruster pulse — move it off RC 9..16")
        mot = [int(round(self.params[f"MOT_{n}_DIRECTION"])) for n in range(1, 9)]
        if mot != self.motor_direction:
            return False, (f"MOT_1..8_DIRECTION {mot} != the policy's trained motor_direction {self.motor_direction}: "
                           f"a thruster would spin reversed")
        if int(round(self.params["SCR_ENABLE"])) != 1:
            return False, "SCR_ENABLE != 1 (the Lua override script cannot run)"
        if int(round(self.params["SCR_USER1"])) != 1:
            return False, "SCR_USER1 != 1 (rl_pwm_override.lua is not armed)"
        if int(round(self.params["RC_OPTIONS"])) & 2:
            return False, "RC_OPTIONS bit 1 is set: the vehicle ignores MAVLink RC overrides"
        if not self.params["RC_OVERRIDE_TIME"] > 0.0:
            return False, "RC_OVERRIDE_TIME is 0: RC overrides are disabled"
        opts = {c: int(round(self.params[f"RC{c}_OPTION"])) for c in self.channels
                if int(round(self.params[f"RC{c}_OPTION"])) != 0}
        if opts:
            return False, f"RC channel options are set on the motor channels {opts}: a pulse would toggle an aux function"
        if int(round(self.params["SYSID_MYGCS"])) != self.sysid:
            return False, (f"SYSID_MYGCS {int(round(self.params['SYSID_MYGCS']))} != this station's sysid {self.sysid}: "
                           f"the overrides would be dropped")
        if self.t_hb is None:
            return False, "no RLPWM heartbeat — rl_pwm_override.lua is not running on the vehicle"
        if now - self.t_hb > SCRIPT_HEARTBEAT_STALE_S:
            return False, f"RLPWM heartbeat stale ({now - self.t_hb:.1f} s) — the Lua script stopped"
        if self.unconfirmed_s > SCRIPT_ENGAGE_GRACE_S and self.script_engaged is not True:
            return False, (f"rl_pwm_override.lua is not forcing the outputs (RLPWM = 0) after "
                           f"{self.unconfirmed_s:.1f} s of pulses — armed? MANUAL? SCR_USER1? SYSID_THISMAV?")
        return True, "ok"

    def state(self, now: float, link_v2: bool = True) -> dict:
        ok, why = self.check(now, link_v2)
        return {"configured": True, "ok": bool(ok), "why": why,
                "hb_age_s": (None if self.t_hb is None else float(now - self.t_hb)),
                "script_engaged": self.script_engaged, "n_missing": len(self.missing(now)),
                "channels": list(self.channels), "motor_direction": list(self.motor_direction)}


__all__ = ["PwmModel", "PwmGate", "T200CurveNp", "throttles_to_pwm", "pwm_to_action", "pack_rc_override",
           "NEUTRAL_PULSES", "PWM_NEUTRAL_US", "PWM_SPAN_US", "PWM_MIN_US", "PWM_MAX_US", "PWM_CHANNELS_DEFAULT",
           "PWM_CAP_FIRST_WATER", "SCRIPT_HEARTBEAT_NAME", "N_MOTORS"]
