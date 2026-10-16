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

2026-10-10 (RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md §1 cause 4, §3 step 5):

  * the |throttle| cap is a THRUST-SPACE SCALE (``scale_throttles``), not a per-thruster clip. A clip at 0.25 of a
    cap-1.0 policy pinned thrusters 3/4/6 at the cap 61-94 % of the time on 2026-10-01 and collapsed the allocation
    (cos(wrench, uncapped) 0.73 [measured REPORT scratch/design-systems/alloc_numbers.txt]); one scalar on the thrust
    vector keeps the direction (cos 1.000) and the null-space content in proportion. The clip survives only as a
    REFUSE-IF-EXCEEDED assertion (``throttles_to_pwm(strict=True)``, the sink's ``pulses_within_cap``);
  * ``T200CurveNp.inverse`` — the closed-form inverse of the quadratic kgf(rpm) and linear rpm(throttle) fits;
  * the per-thruster sign-probe artefact gate (``ThrusterProbeCheck`` inside ``PwmGate``): no pwm pulse leaves until
    a recent, non-dry-run ``thruster_sign_probe.json`` written for THIS export's sigma and THIS vehicle's
    MOT_1..8_DIRECTION says every motor passed (the probe tool is rov_gui/tools/thruster_probe.py);
  * the predicted pack current (``predicted_pack_current_a``) and the null-space monitor (``null_space_us``) of the
    SENT pulses, for the CSV.

2026-10-10 (b) (REPORT.md §3 step 5, second half — interlocks, observation guards, replay):

  * ``PwmInterlocks`` — the per-thruster runtime interlocks (``engage.per_thruster``), ONE implementation for the
    worker (workers._runtime_fault, every tick, the attitude-ceiling debounce pattern) and the offline replay tool
    (rov_gui/tools/rl_pwm_interlock_replay.py): tilt, attitude rate, pack current / voltage, the null-space monitor
    on the sent pulses and the cap-pinned collapse signature. Each trip is a disengage (neutral pulses) with the
    measured value in the mission log;
  * ``PwmObsGuard`` — what the pwm follower does to the state BEFORE the network sees it (``rl_pwm.obs_guard``,
    applied inside HwRlPwm.step; the shared state_assembler is untouched): a body-velocity slew limit, a fix-jump
    reject on a thin tag fix, and the tag gate that replaces the policy's pulses with the HOLD pulses
    (``hold_throttles``: the measured net weight through the export's minimum-norm vertical allocation + T200
    inverse) until the tags are back. The CSV column ``pwm_guard`` (schema 20) says which guard acted;
  * the predicted current's thrust convention is the physical one of RL_controller (rewards.thruster_current_proxy,
    rov_env /batt/I_pred_a): F = T200(thr) * batt_v / PACK_V_CURVE, the fit's 1.0 being the ~20 V datasheet curve.

2026-10-15 (RL_controller/logs/eval/pool_trips_1013_rootcause; operator-approved station fix 1b): the 10-13 pool trips
(5 of 7 engagements on RATE/TILT) were the policy flying on an ATTITUDE 0.43 s old at the median, behind a vehicle-side
MAVLink queue the station's parse-time imu_age could not see. ``PwmInterlocks`` gains AGE (first in the order): the
attitude's VEHICLE-clock age (state_assembler.att_age_v, c3_camera/mavlink_log.VehicleClockMap) over 0.15 s for 0.5 s
or 0.25 s for 0.2 s trips, and an unknown age counts as over. The engage refusal that rides the same keys (clock map
>= 30 s old, no 'ArduSub V' banner in 30 s, no unknown tick in the last 10 s, rolling 10-s p99 <= 0.12 s, the age at
START <= age_s 0.15 s) is workers._att_age_refusal.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from datetime import datetime
from typing import Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
#: How the cap is applied (``rl_pwm.pwm_cap_mode``, 2026-10-10): "scale" = one scalar on the thrust vector
#: (scale_throttles, the default), "clip" = the per-thruster clip of 2026-09-30..10-09, kept for an A/B only.
PWM_CAP_MODES = ("scale", "clip")
PWM_CAP_MODE_DEFAULT = "scale"
#: |throttle| above the cap tolerated by the strict conversion / the sink: floating-point slack of the inverse
#: curve (max|thr'| == cap within 1e-6 is what scale_throttles guarantees), 4e-4 us on the wire.
CAP_TOL = 1e-6
#: The sign-probe artefact (rov_gui/tools/thruster_probe.py writes it; rov_gui/control/thruster_probe_artifact.py
#: reads it): newest data/<day>/<run>_probe/thruster_sign_probe.json unless rl_pwm.thruster_probe_path names one.
THRUSTER_PROBE_FILENAME = "thruster_sign_probe.json"
THRUSTER_PROBE_MAX_AGE_DAYS = 30.0
THRUSTER_PROBE_SCHEMA = 1
#: The artefact is re-stat'ed and re-read this often while the gate is asked (a probe written while the station
#: runs is picked up without a restart; a deleted one is noticed within this long).
THRUSTER_PROBE_REFRESH_S = PARAM_REFRESH_S
#: Predicted pack current (CSV i_pred_a, 2026-10-10): I = I_PACK_IDLE_A + sum_i c |F_i|^p per branch, the fit of
#: the T200 public datasheet's 15 V column [유도 REPORT scratch/design-systems/alloc_numbers.txt: "fit: I = 0.0584 *
#: |F|^1.526 (A, N); check 26 N -> 8.4 A"; reverse branch 0.0537 / 1.673, same fit family; 2.1 A = the pack's idle
#: draw]. F_i is the PHYSICAL thrust of the SENT pulse: the export's T200 curve times batt_v / PACK_V_CURVE — the
#: convention of RL_controller rov_rl/rewards.py thruster_current_proxy and rov_env.py /batt/I_pred_a (2026-10-10 (b);
#: until then the curve was scaled by batt_v / 14.8, i.e. taken as the 14.8 V curve, ~1.4x too much thrust). The
#: fit's scale 1.0 is the ~20 V datasheet curve (64.13 N at full throttle), 14.8 V -> 0.72 = the export's
#: voltage_scale [스펙 RL_controller rov_rl/thrusters.py docstring]; without a voltage reading that 0.72 is the
#: fallback (read i_pred_a next to batt_v). The att7pwm_s0 pool pattern predicts ~35-42 A over 14.8-16 V against a
#: measured 26-44 A mean under-sampled at 1 Hz [측정 REPORT.md §1 cause 3; test_rl_pwm_scale.py].
I_PACK_IDLE_A = 2.1
I_FIT_FWD = (0.0584, 1.526)
I_FIT_REV = (0.0537, 1.673)
#: The pack voltage at which the T200 fit's scale is 1.0 [스펙 rov_rl/thrusters.py: "~20 V"; 14.8 V nominal ~ 0.72
#: -> 14.8 / 0.72 = 20.5 V]. PACK_V_NOMINAL (14.8 V, the 4S nominal) is kept for readers of the earlier convention.
PACK_V_CURVE = 20.5
PACK_V_NOMINAL = 14.8


# ----------------------------------------------------------------------------- conversions
def throttles_to_pwm(thr, cap: float = 1.0, strict: bool = False) -> tuple:
    """throttle (8,) in [-1, 1] -> 8 integer pulse widths [us], |throttle| clipped to ``cap`` first.

    ``strict`` (2026-10-10, the pwm followers): a throttle beyond ``cap`` + CAP_TOL RAISES instead of being clipped —
    the cap is applied upstream as a thrust-space scale (scale_throttles), and nothing on the way to the wire may
    reshape a thrust command silently; the caller answers the exception with neutral + n_fail."""
    c = float(max(0.0, min(1.0, cap)))
    t = np.asarray(thr, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("throttles_to_pwm: non-finite throttle")
    if strict and np.any(np.abs(t) > c + CAP_TOL):
        raise ValueError(f"throttles_to_pwm: |throttle| {float(np.max(np.abs(t))):.4f} exceeds the cap {c:.4f} — "
                         f"refused, not clipped (the cap is a thrust-space scale upstream)")
    t = np.clip(t, -c, c)
    return tuple(int(v) for v in np.round(PWM_NEUTRAL_US + PWM_SPAN_US * t))


def cap_span_us(cap: float) -> int:
    """The pulse half-span [us] of a |throttle| cap: 1500 +- this is the wire window (0.25 -> 100 us)."""
    return int(round(PWM_SPAN_US * float(max(0.0, min(1.0, cap)))))


def pulses_within_cap(pwm_us, cap: float) -> bool:
    """Are all eight pulses inside 1500 +- cap_span_us(cap)? The sink's REFUSE-IF-EXCEEDED assertion (2026-10-10):
    a frame outside the window is refused and neutralised, never clamped (clamping = the per-thruster clip that
    collapsed the allocation on 2026-10-01)."""
    span = cap_span_us(cap)
    try:
        pw = [int(p) for p in pwm_us]
    except (TypeError, ValueError):
        return False
    return len(pw) == N_MOTORS and all(PWM_NEUTRAL_US - span <= p <= PWM_NEUTRAL_US + span for p in pw)


def null_space_us(pwm_us) -> tuple:
    """The two wrench-free directions of the 8-thruster allocation, read off the pulses [us] (2026-10-10):

        null_v_us = mean(pwm5..8) - 1500          = <d, n_v> / |n_v|^2,  n_v = [0,0,0,0,1,1,1,1]
        null_h_us = ((pwm1 + pwm2) - (pwm3 + pwm4)) / 4 = <d, n_h> / |n_h|^2,  n_h = [1,1,-1,-1,0,0,0,0]

    d = pulse - 1500. The null space of B (6x8, rank 6) is exactly span{n_v, n_h} [유도 REPORT.md §0: all four
    verticals at the same +PWM cancel (FR up, FL down, RR down, RL up); front horizontals forward + rear backward
    cancel] — a standing value here is thrust that produces NO wrench (att7pwm_s0 in the pool: null_v +250..+290 us
    = 25-43 N cancelling itself, 26-44 A [measured REPORT.md §1 cause 3]). Logged (CSV null_v_us / null_h_us) and,
    since 2026-10-10 (b), judged by the NULL interlock (PwmInterlocks, engage.per_thruster.null_us)."""
    p = np.asarray(pwm_us, dtype=float).reshape(-1)
    if p.shape != (N_MOTORS,):
        raise ValueError(f"null_space_us: need {N_MOTORS} pulses, got {p.shape}")
    null_v = float(np.mean(p[4:8]) - PWM_NEUTRAL_US)
    null_h = float(((p[0] + p[1]) - (p[2] + p[3])) / 4.0)
    return null_v, null_h


def thrust_scale_at_voltage(batt_v, model) -> float:
    """The thrust multiplier for the predicted-current column: logged pack voltage / PACK_V_CURVE (the physical
    convention of RL_controller's current proxy, 14.8 V -> 0.72) when a finite reading exists, else the export's
    voltage_scale (0.72) — see I_PACK_IDLE_A's note."""
    try:
        v = float(batt_v)
    except (TypeError, ValueError):
        return float(model.voltage_scale)
    if not (math.isfinite(v) and v > 0.0):
        return float(model.voltage_scale)
    return v / PACK_V_CURVE


def predicted_pack_current_a(pwm_us, model, batt_v=None) -> float:
    """I_PACK_IDLE_A + sum over the eight thrusters of the per-branch |F|^p fit (I_FIT_FWD / I_FIT_REV), F from the
    SENT pulses through the export's T200 curve at thrust_scale_at_voltage (CSV i_pred_a, 2026-10-10; physical
    convention batt_v / PACK_V_CURVE since 2026-10-10 (b)). A number for the log and the pack interlock's twin, not a
    measurement: the att7pwm_s0 pool pattern [-115,-125,+125,+120,+260,+220,+200,+270] us predicts ~35-42 A over
    14.8-16 V against a measured 26-44 A mean under-sampled at 1 Hz [measured REPORT.md §1 cause 3;
    test_rl_pwm_scale.py]."""
    thr = pwm_to_action(pwm_us, 1.0)
    F = model.curve.thrust(thr) * thrust_scale_at_voltage(batt_v, model)
    c_f, p_f = I_FIT_FWD
    c_r, p_r = I_FIT_REV
    i = np.where(F > 0.0, c_f * np.abs(F) ** p_f, np.where(F < 0.0, c_r * np.abs(F) ** p_r, 0.0))
    return float(I_PACK_IDLE_A + np.sum(i))


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

    #: The forward map is 0 AT the deadband (strict >), so the smallest throttle that produces thrust is the
    #: deadband plus this: what ``inverse`` returns for a thrust at the deadband edge (4e-7 us on the wire).
    EDGE_EPS = 1e-9

    def thrust(self, thr) -> np.ndarray:
        thr = np.clip(np.asarray(thr, dtype=float), -1.0, 1.0)
        rpm = np.where(thr > self.deadband, self.rpm_pos[0] * thr + self.rpm_pos[1],
                       np.where(thr < -self.deadband, self.rpm_neg[0] * thr + self.rpm_neg[1], 0.0))
        rpm = np.clip(rpm, -self.rpm_max, self.rpm_max)
        a, b, c = self.kgf_pos
        an, bn, cn = self.kgf_neg
        kgf = np.where(rpm > 0, a * rpm * rpm + b * rpm + c, np.where(rpm < 0, an * rpm * rpm + bn * rpm + cn, 0.0))
        return 9.81 * kgf

    def edge_thrust(self) -> tuple:
        """(F at the positive deadband edge, F at the negative edge) [N] — the smallest |thrust| each branch can
        produce; the forward map jumps from 0 to these (+1.44 / -1.10 N for the shipped fit [유도 obs_spec t200])."""
        a, b, c = self.kgf_pos
        an, bn, cn = self.kgf_neg
        rp = min(self.rpm_max, self.rpm_pos[0] * self.deadband + self.rpm_pos[1])
        rn = max(-self.rpm_max, -self.rpm_neg[0] * self.deadband + self.rpm_neg[1])
        return float(9.81 * (a * rp * rp + b * rp + c)), float(9.81 * (an * rn * rn + bn * rn + cn))

    def max_thrust(self) -> tuple:
        """(F(+1), F(-1)) [N]: the rpm_max-saturated ends of the two branches (+64.13 / -51.55 N shipped)."""
        return float(self.thrust(1.0)), float(self.thrust(-1.0))

    def inverse(self, F, below_edge: str = "edge") -> np.ndarray:
        """Closed-form inverse of ``thrust`` (2026-10-10): thrust [N] -> throttle. Per branch, the quadratic
        kgf(rpm) = a rpm^2 + b rpm + c is solved for the root on the monotone side of its vertex (the vertex lies
        INSIDE the deadband on both branches of the shipped fit, so the curve is monotone wherever it is nonzero),
        then the linear rpm(throttle) fit is inverted. Exact on the attainable range [F_edge, F(rpm_max)] of each
        branch (round trip within 1e-6 N, test_rl_pwm_scale.py); a |thrust| beyond the rpm_max value saturates at
        the throttle that reaches rpm_max; 0 -> 0; 0 < |thrust| < edge (unattainable) maps to the deadband edge
        with its sign (``"edge"``: the thruster stays ON at its minimum — never switched off by a scale, the ESC's
        110-130 ms from-rest start-up [spec BR forum] is not re-paid) or to 0 (``"zero"``)."""
        if below_edge not in ("edge", "zero"):
            raise ValueError(f"inverse: below_edge must be 'edge' or 'zero', got {below_edge!r}")
        F = np.asarray(F, dtype=float)
        if not np.all(np.isfinite(F)):
            raise ValueError("inverse: non-finite thrust")
        kgf = F / 9.81
        edge_p, edge_n = self.edge_thrust()
        a, b, c = self.kgf_pos
        an, bn, cn = self.kgf_neg
        thr_min = self.deadband + self.EDGE_EPS
        # positive branch: the larger root (a > 0, increasing past the vertex)
        kp = np.maximum(kgf, edge_p / 9.81)
        rpm_p = (-b + np.sqrt(np.maximum(b * b - 4.0 * a * (c - kp), 0.0))) / (2.0 * a)
        rpm_p = np.minimum(rpm_p, self.rpm_max)
        thr_p = np.clip((rpm_p - self.rpm_pos[1]) / self.rpm_pos[0], thr_min, 1.0)
        # negative branch: the more negative root (an < 0: (-bn + sqrt) / (2 an) is the one past the vertex)
        kn = np.minimum(kgf, edge_n / 9.81)
        rpm_n = (-bn + np.sqrt(np.maximum(bn * bn - 4.0 * an * (cn - kn), 0.0))) / (2.0 * an)
        rpm_n = np.maximum(rpm_n, -self.rpm_max)
        thr_n = np.clip((rpm_n - self.rpm_neg[1]) / self.rpm_neg[0], -1.0, -thr_min)
        pos, neg = F > 0.0, F < 0.0
        out = np.where(pos, thr_p, np.where(neg, thr_n, 0.0))
        if below_edge == "zero":
            out = np.where((pos & (F < edge_p)) | (neg & (F > edge_n)), 0.0, out)
        return out


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
        #: The allocation's thruster signs (RL_controller mixer, [1,1,-1,-1,1,-1,-1,1] on this vehicle): what the
        #: sign-probe artefact is written for and checked against (2026-10-10). None in an export older than that
        #: key — such an export cannot be probe-gated (HwRlPwm refuses it unless the gate is opted out).
        sg = pwm_model.get("sigma")
        self.sigma = None if sg is None else [float(v) for v in sg]
        if self.sigma is not None and len(self.sigma) != self.n:
            raise ValueError(f"pwm_model: sigma has {len(self.sigma)} values, expected {self.n}")
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


def scale_throttles(thr, cap: float, model: "PwmModel", return_scale: bool = False):
    """The |throttle| cap as a THRUST-SPACE SCALE (2026-10-10; REPORT.md §1 cause 4, §3 step 5):

        F   = T200(thr) * voltage_scale                       per thruster [N]
        s   = min(1, T200(+cap)*V / max F[F>0], |T200(-cap)*V| / max |F[F<0]|)   (each branch only if non-empty)
        F'  = s F
        thr' = T200^-1(F')                                   (T200CurveNp.inverse, below-edge -> edge)

    (voltage_scale cancels in s and in the inverse; it is left out of the arithmetic.) One scalar on the thrust
    vector keeps the allocation DIRECTION (B F' = s B F: cos 1 exactly, where no thruster is pushed under its
    deadband edge by the scale) and puts the largest |throttle| of the binding branch exactly at the cap; a
    per-thruster clip does neither (2026-10-01 at 0.25: thrusters 3/4/6 pinned 61-94 % of the time, cos(wrench,
    uncapped) 0.73 [measured REPORT]). Sub-deadband throttles (F = 0) go to 0 when a scale is applied; thrusters
    the scale pushes below the deadband edge stay on at the edge (T200CurveNp.inverse). When nothing exceeds the cap
    in thrust the throttles pass unchanged, except in the rpm_max-saturated zone (|thr| > cap with T200(thr) ==
    T200(cap)), where the cap value is the identical command. The result never exceeds the cap by more than CAP_TOL;
    the strict conversion / the sink refuse the frame if it ever did."""
    t = np.clip(np.asarray(thr, dtype=float).reshape(-1), -1.0, 1.0)
    if t.shape != (model.n,):
        raise ValueError(f"scale_throttles: need {model.n} throttles, got {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("scale_throttles: non-finite throttle")
    c = float(cap)
    if not (math.isfinite(c) and 0.0 < c <= 1.0):
        raise ValueError(f"scale_throttles: cap must be in (0, 1], got {cap!r}")
    curve = model.curve
    F = curve.thrust(t)
    fp_cap = float(curve.thrust(c))
    fn_cap = float(curve.thrust(-c))
    s = 1.0
    pos, neg = F > 0.0, F < 0.0
    if np.any(pos):
        s = min(s, fp_cap / float(np.max(F[pos])))
    if np.any(neg):
        s = min(s, fn_cap / float(np.min(F[neg])))          # both negative -> a positive ratio
    if s >= 1.0:
        s = 1.0
        out = np.where(np.abs(t) > c, np.sign(t) * c, t)     # only the rpm-saturated zone can be here
    else:
        out = curve.inverse(s * F, below_edge="edge")
        out = np.clip(out, -c, c)                             # numerical guard only (<= 1e-12 by construction)
    return (out, float(s)) if return_scale else out


# ----------------------------------------------------------------------------- the sign-probe artefact gate
class ThrusterProbeCheck:
    """The per-thruster sign-probe artefact gate (2026-10-10, REPORT.md §3 step 4/5): the one bench fact nothing
    else in this file can stand in for — MOT_n_DIRECTION matching the export says the PARAMETER agrees, not that
    thruster k pushes the way the export's sigma[k] says. The probe tool (rov_gui/tools/thruster_probe.py) drives
    each motor alone and writes ``thruster_sign_probe.json``; this gate REFUSES until a recent, non-dry-run artefact
    written for THIS export's sigma and THIS vehicle's MOT_1..8_DIRECTION says every motor passed. Fail-closed on
    everything: no reader module, no file, unreadable, wrong schema, dry_run, too old, written in the future,
    all_pass false, other vehicle directions, other sigma, or the reader's own verdict.

    Schema 1 of the artefact: {"schema": 1, "written": ISO time, "dry_run": bool, "vehicle": {"mot_direction": [8]},
    "export": {"sigma": [8], "B_flu": [[6x8]], "obs_spec_sha1": str}, "motors": [{"motor": i, "verdict":
    "PASS|WEAK|FAIL", ...}], "all_pass": bool}. The reader's ``load_and_check(path, expected_sigma,
    expected_mot_direction) -> (ok, reason)`` has the last word (it may check more: B_flu, the per-motor verdicts).

    No I/O on construction. ``check`` caches its verdict for THRUSTER_PROBE_REFRESH_S per expected direction set
    (the gate is asked every frame), so a probe written or removed while the station runs is noticed within that."""

    def __init__(self, sigma, path=None, max_age_days: float = THRUSTER_PROBE_MAX_AGE_DAYS, data_root=None,
                 clock=time.time):
        if sigma is None or len(list(sigma)) != N_MOTORS:
            raise ValueError("thruster probe: the export carries no 8-value sigma — it cannot be checked against a "
                             "probe (re-export with RL_controller >= 2026-10-10, or opt out: "
                             "rl_pwm.require_thruster_probe: false, recorded)")
        self.sigma = [float(v) for v in sigma]
        self.path = None if path in (None, "") else str(path)
        a = max_age_days
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not (math.isfinite(float(a)) and float(a) > 0.0):
            raise ValueError(f"rl_pwm.thruster_probe_max_age_days must be a finite number > 0, got {a!r}")
        self.max_age_days = float(a)
        self.data_root = os.path.join(REPO_ROOT, "data") if data_root is None else str(data_root)
        self._clock = clock
        self._cache = None               # (mot key, t_checked, ok, why)
        self.last: dict = {"ok": None, "why": "not checked yet", "path": None, "written": None, "age_days": None}

    # ---- which file
    def resolve_path(self):
        """The configured path (REPO_ROOT-relative if not absolute), else the newest ``data/*/*_probe/
        thruster_sign_probe.json`` by modification time; None when there is none."""
        if self.path:
            return self.path if os.path.isabs(self.path) else os.path.join(REPO_ROOT, self.path)
        cands = glob.glob(os.path.join(self.data_root, "*", "*_probe", THRUSTER_PROBE_FILENAME))
        if not cands:
            return None
        return max(cands, key=lambda f: (os.path.getmtime(f), f))

    # ---- the verdict
    def check(self, expected_mot_direction, now: float | None = None) -> tuple:
        """(ok, why) for the vehicle's current MOT_1..8_DIRECTION; cached THRUSTER_PROBE_REFRESH_S."""
        now = time.monotonic() if now is None else float(now)
        key = tuple(int(round(float(v))) for v in expected_mot_direction)
        if len(key) != N_MOTORS:
            return False, "expected_mot_direction needs 8 values"
        if self._cache is not None and self._cache[0] == key and 0.0 <= now - self._cache[1] < THRUSTER_PROBE_REFRESH_S:
            return self._cache[2], self._cache[3]
        ok, why = self._evaluate(key)
        self._cache = (key, now, ok, why)
        self.last.update(ok=bool(ok), why=why)
        return ok, why

    def forget(self) -> None:
        """Drop the cached verdict (the next check re-reads the file)."""
        self._cache = None

    def _evaluate(self, mot: tuple) -> tuple:
        self.last.update(path=None, written=None, age_days=None)
        try:
            from .thruster_probe_artifact import load_and_check
        except ImportError as e:
            return False, (f"the artefact reader rov_gui/control/thruster_probe_artifact.py is not importable "
                           f"({type(e).__name__}: {e}) — no probe can be judged, no pwm pulse leaves")
        path = self.resolve_path()
        if path is None:
            return False, (f"no artefact {os.path.join(self.data_root, '*', '*_probe', THRUSTER_PROBE_FILENAME)} — "
                           f"run `python -m rov_gui.tools.thruster_probe` (propellers ON, vehicle tethered, "
                           f"README §10.3) before any pwm flight")
        self.last["path"] = path
        name = os.path.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT) else path
        if not os.path.isfile(path):
            return False, f"artefact {name} does not exist"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            return False, f"artefact {name} is unreadable ({type(e).__name__}: {e})"
        if not isinstance(d, dict):
            return False, f"artefact {name} is not a JSON object"
        try:
            schema = int(d.get("schema", -1))
        except (TypeError, ValueError):
            schema = -1
        if schema != THRUSTER_PROBE_SCHEMA:
            return False, f"artefact {name}: schema {d.get('schema')!r} != {THRUSTER_PROBE_SCHEMA}"
        if bool(d.get("dry_run", True)):
            return False, (f"artefact {name} is a DRY-RUN (no thruster was driven: dry_run true) — not a probe; "
                           f"run the tool for real")
        written = d.get("written")
        try:
            ts = datetime.fromisoformat(str(written)).timestamp()
        except (TypeError, ValueError):
            return False, f"artefact {name}: 'written' {written!r} is not an ISO time"
        age_days = (float(self._clock()) - ts) / 86400.0
        self.last.update(written=str(written), age_days=round(age_days, 3))
        if age_days > self.max_age_days:
            return False, (f"artefact {name} is {age_days:.1f} days old (written {written}), older than "
                           f"rl_pwm.thruster_probe_max_age_days {self.max_age_days:g} — probe again")
        if age_days < -1.0:
            return False, f"artefact {name} claims to be written {-age_days:.1f} days in the FUTURE ({written}): clock?"
        if not bool(d.get("all_pass", False)):
            verdicts = {}
            for m in (d.get("motors") or []):
                if isinstance(m, dict):
                    verdicts[m.get("motor")] = m.get("verdict")
            return False, f"artefact {name}: all_pass is false (verdicts {verdicts}) — a thruster did not pass its probe"
        vm = list((d.get("vehicle") or {}).get("mot_direction") or [])
        try:
            vm_i = [int(round(float(v))) for v in vm]
        except (TypeError, ValueError):
            vm_i = []
        if vm_i != list(mot):
            return False, (f"artefact {name} was probed with MOT_1..8_DIRECTION {vm}, the vehicle now says "
                           f"{list(mot)} — probe again on this configuration")
        es = list((d.get("export") or {}).get("sigma") or [])
        try:
            es_f = [float(v) for v in es]
        except (TypeError, ValueError):
            es_f = []
        if len(es_f) != N_MOTORS or any(abs(x - y) > 1e-9 for x, y in zip(es_f, self.sigma)):
            return False, (f"artefact {name} was probed for sigma {es}, this export's sigma is {self.sigma} — "
                           f"probe again with this export")
        try:
            ok, why = load_and_check(path, list(self.sigma), list(mot))
        except Exception as e:                                   # noqa: BLE001
            return False, f"artefact {name}: load_and_check failed ({type(e).__name__}: {e})"
        if not ok:
            return False, f"artefact {name}: {why}"
        return True, "ok"

    def state(self) -> dict:
        return {"required": True, "configured_path": self.path, "data_root": self.data_root,
                "max_age_days": self.max_age_days, **dict(self.last)}


def parse_rl_pwm_block(raw: dict) -> dict:
    """The transport-level keys of a ``rl_pwm:`` block, validated, with their defaults (2026-10-10). ONE reader for
    the config loader, the follower (HwRlPwm) and the command sink, so the three cannot disagree about a default:
    pwm_cap (PWM_CAP_FIRST_WATER), pwm_cap_mode ("scale"), allow_caps_below_trained (false), require_thruster_probe
    (true), thruster_probe_path (null = newest), thruster_probe_max_age_days (30)."""
    raw = dict(raw or {})
    out = {}
    cap = raw.get("pwm_cap", PWM_CAP_FIRST_WATER)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) \
            or not (math.isfinite(float(cap)) and 0.0 < float(cap) <= 1.0):
        raise ValueError(f"rl_pwm.pwm_cap must be a number in (0, 1], got {cap!r}")
    out["pwm_cap"] = float(cap)
    mode = raw.get("pwm_cap_mode", PWM_CAP_MODE_DEFAULT)
    if not isinstance(mode, str) or mode not in PWM_CAP_MODES:
        raise ValueError(f"rl_pwm.pwm_cap_mode must be one of {list(PWM_CAP_MODES)}, got {mode!r}")
    out["pwm_cap_mode"] = mode
    for k, dflt in (("allow_caps_below_trained", False), ("require_thruster_probe", True)):
        v = raw.get(k, dflt)
        if not isinstance(v, bool):
            raise ValueError(f"rl_pwm.{k} must be true/false, got {v!r}")
        out[k] = v
    pp = raw.get("thruster_probe_path")
    if pp is not None and (not isinstance(pp, str) or not pp.strip()):
        raise ValueError(f"rl_pwm.thruster_probe_path must be null or a non-empty path, got {pp!r}")
    out["thruster_probe_path"] = None if pp is None else pp.strip()
    age = raw.get("thruster_probe_max_age_days", THRUSTER_PROBE_MAX_AGE_DAYS)
    if isinstance(age, bool) or not isinstance(age, (int, float)) \
            or not (math.isfinite(float(age)) and float(age) > 0.0):
        raise ValueError(f"rl_pwm.thruster_probe_max_age_days must be a finite number > 0, got {age!r}")
    out["thruster_probe_max_age_days"] = float(age)
    return out


#: Every key a ``rl_pwm:`` block may carry (geometry.MpcConfig.load refuses any other — raw thruster pulses: a
#: misspelt key must not fall back silently). NO pwm_channels (the order is fixed, _fixed_channels).
RL_PWM_KEYS = ("policy_dir", "pwm_cap", "pwm_cap_mode", "allow_caps_below_trained", "require_thruster_probe",
               "thruster_probe_path", "thruster_probe_max_age_days", "leash_m", "yaw_rate_ff", "yaw_rate_max",
               "obs_guard")


# ----------------------------------------------------------------------------- the per-thruster interlocks (2026-10-10 b)
#: ``engage.per_thruster`` defaults — the runtime interlocks of the pwm LOWs (rl_pwm, rl_pwm_7d) ONLY, evaluated every
#: tick by workers._runtime_fault with the attitude ceiling's debounce (consecutive ticks over the limit, counted in
#: ticks) and replayed offline by rov_gui/tools/rl_pwm_interlock_replay.py through the SAME class. Every number
#: [예측] unless tagged (REPORT.md §3 step 5: "interlocks (tilt 20-25 deg/0.25 s, rate 60 deg/s, 50 A / 13.5 V, null
#: monitor 100-150 us)"):
#:   abort_deg / abort_s   |roll| or |pitch| > 20 deg for 0.25 s (first water; engage.attitude_axes.abort_deg 35 deg
#:                         stays the OUTER ceiling, judged first);
#:   rate_dps / rate_s     |p| or |q| (the vehicle's gyro, ATTITUDE rollspeed/pitchspeed) > 40 deg/s for 0.15 s
#:                         [측정 REPORT.md §1 cause 2: the pool cascade started at 33-65 deg/s pitch rate; the
#:                         4-axis policy flew at <= 13 deg/s]. 60 deg/s was the first cut; the 2026-10-10 replay
#:                         of the 14 recorded engagements missed 2 of them at 60 and caught 14/14 at 40 with 0
#:                         false trips over 91 mixer-mode files [측정 data/20261010/1010_interlock_replay_offline/];
#:   current_a / current_s batt_a > 50 A for 1.0 s (BATTERY_STATUS is ~1 Hz: the latest sample, held)
#:                         [스펙 BR 18 Ah pack: 60 A continuous];
#:   volt_min              batt_v < 13.5 V at ANY tick (no debounce: a sagging pack under a co-contracting policy
#:                         reached 14.0 V on 10-03 [측정 REPORT.md §1 cause 3]);
#:   null_us / null_s      |null_v_us| or |null_h_us| of the SENT pulses > 80 us for 0.5 s [측정 REPORT.md: mixer
#:                         modes 0 us, pwm10_s1 squeeze +82..+107 us, att7pwm +250..+290 us; a pwm11 policy that
#:                         passed the sim gate has |c| <= 0.1 = 40 us]. 120 us / 1.0 s was the first cut: in the
#:                         replay it never caught pwm10_s1's squeeze (95-100 us peaks) and its 1.0 s hold let two
#:                         1.3-1.7 s engagements through; 80 us / 0.5 s catches all 14 at +0.43..+0.60 s, 0 of 91
#:                         mixer files [측정 same folder];
#:   pinned_n / pinned_s   >= 3 thrusters at the cap for 1.0 s (the 10-01 collapse signature
#:                         [측정 KNOWN_ISSUES: thrusters 3/4/6 at the cap 61-94 % of the time]).
#: 2026-10-15 — the attitude AGE (operator-approved station fix 1b, RL_controller/logs/eval/pool_trips_1013_rootcause;
#: "RC" below). On 10-13 the ATTITUDE the policy flew on (roll/pitch + p/q) was 0.43 s old at the median, 0-0.9 s
#: sawtooth, behind a vehicle-side MAVLink queue; RATE and TILT then acted on the same stale data:
#:   age_s / age_hold_s    the VEHICLE-clock age of that attitude (state_assembler.att_age_v) > 0.15 s for 0.5 s, OR
#:   age_hard_s / age_hard_hold_s  > 0.25 s for 0.2 s -> trip. An UNKNOWN age (no clock map, a foreign clock) counts
#:                         as over on both: the hard rule trips it in 0.2 s (fail-safe). Replayed causally — this rule,
#:                         this estimator, the station's present 20 Hz republish chain — over every recording: 21
#:                         healthy ones (1.82 h, tick age p99 0.058-0.078 s, max 0.063-0.173 s) 0 trips; the five
#:                         congested 10-13 ones caught 28 of 29 queue onsets 0.40 s (p95 <= 0.47 s) after the true age
#:                         crossed 0.10 s, 0 stretches > 0.25 s for >= 1 s missed; the 23.6-s ATTITUDE outage of
#:                         0809_202049 trips too [측정 RC/fix-design/out/fixdesign_summary.json "replay",
#:                         age_guard_replay.txt]. 0.15 s = pwm11att6_s2's last stable SUSTAINED age on the nominal
#:                         command path (0/7 programs trip at 0.15 s, 2/7 at 0.20 s) [sim RC/synth/isaac/
#:                         trip1014_att6s2_track_latency/metrics.json]. Thresholds [예측];
#:   age_engage_p99_s / age_engage_window_s
#:                         ENGAGE refusal (workers._att_age_refusal): the p99 of that age over the last 10 s of control
#:                         ticks (+ the engage instant) > 0.12 s. The replay refused 100 % of the checks on the five
#:                         congested recordings and 0 % on the 21 healthy ones [측정 same files] [예측]. Since the
#:                         verify-review / verify-replay fixes (2026-10-15, RL_controller/logs/eval/
#:                         station_fix_20261015/fixer): ONE unknown tick in the window refuses by itself (no percentile
#:                         decides it), and age_s is also an ENGAGE limit — the age at START > 0.15 s refuses (one
#:                         sample cannot move a 200-tick p99). Re-replayed: 4 of 120 982 healthy post-warm-up START
#:                         checks refused, all single-tick stalls of 0.16-0.19 s; congested still 100 % [측정
#:                         fixer/out/refusal_rules_replay.txt];
#:   age_map_min_s         ENGAGE refusal: the clock map is younger than 30 s. A map started inside a standing queue
#:                         under-reads the age until the queue first drains (by up to 0.77 s for the first 6.8 s of
#:                         1013_211913 [측정 RC/fix-design/scripts/check_online_err.py]); 30 s is margin [예측];
#:   banner_hold_s         ENGAGE refusal: an 'ArduSub V' banner (another client pulled the parameter list) less than
#:                         30 s ago — 111 of 112 banners on 10-13 21:17 started a queue [측정 RC/synth/out/
#:                         synth_checks.json C4_queue]; a reboot also says it, so a fresh boot waits 30 s [예측].
#: RATE 40 deg/s / 0.15 s and TILT 20 deg / 0.25 s are unchanged; the trip is the same neutral-pulse disengage.
PER_THRUSTER_DEFAULTS = {
    "abort_deg": 20.0, "abort_s": 0.25,
    "rate_dps": 40.0, "rate_s": 0.15,            # 2026-10-10 리플레이: 60 은 14회 중 2회 미검출, 40 은 14/14 (+0.43..0.60 s), 믹서 91 파일 오탐 0
    "current_a": 50.0, "current_s": 1.0,
    "volt_min": 13.5,
    "null_us": 80.0, "null_s": 0.5,              # 80 µs 는 pwm10_s1 의 수평 squeeze(+82..+107)도 잡음; pwm11 게이트 통과치 |c| <= 0.1 = 40 µs
    "pinned_n": 3, "pinned_s": 1.0,
    # 2026-10-15 자세 나이(차량 시계) — 위 주석 참고, 전부 [예측] (근거: RC fix-design 재생)
    "age_s": 0.15, "age_hold_s": 0.5,
    "age_hard_s": 0.25, "age_hard_hold_s": 0.2,
    "age_engage_p99_s": 0.12, "age_engage_window_s": 10.0,
    "age_map_min_s": 30.0, "banner_hold_s": 30.0,
}
PER_THRUSTER_KEYS = tuple(PER_THRUSTER_DEFAULTS)
#: The age keys (2026-10-15) — the AGE runtime interlock and the engage refusal.
AGE_KEYS = ("age_s", "age_hold_s", "age_hard_s", "age_hard_hold_s", "age_engage_p99_s", "age_engage_window_s",
            "age_map_min_s", "banner_hold_s")


def _finite_positive(name: str, v, allow_zero: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f) or f < 0.0 or (f == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return f


def parse_per_thruster_block(raw) -> dict:
    """``engage.per_thruster`` validated with its defaults (None / {} = all defaults). Unknown keys RAISE — these are
    interlocks, a misspelt key must not read as armed."""
    raw = dict(raw or {})
    unknown = set(raw) - set(PER_THRUSTER_KEYS)
    if unknown:
        raise ValueError(f"engage.per_thruster: unknown key(s) {sorted(unknown)} (allowed: {list(PER_THRUSTER_KEYS)})")
    out = dict(PER_THRUSTER_DEFAULTS)
    out.update(raw)
    for k in ("abort_deg", "abort_s", "rate_dps", "rate_s", "current_a", "current_s", "volt_min", "null_us",
              "null_s", "pinned_s") + AGE_KEYS:
        out[k] = _finite_positive(f"engage.per_thruster.{k}", out[k])
    if out["abort_deg"] >= 89.0:
        raise ValueError(f"engage.per_thruster.abort_deg must be < 89 deg, got {out['abort_deg']!r}")
    if out["age_hard_s"] < out["age_s"]:
        raise ValueError(f"engage.per_thruster.age_hard_s ({out['age_hard_s']!r}) must be >= age_s "
                         f"({out['age_s']!r}): the hard rule is the faster trip for the WORSE age")
    n = out["pinned_n"]
    if isinstance(n, bool) or not isinstance(n, (int, float)) or int(n) != n or not (1 <= int(n) <= N_MOTORS):
        raise ValueError(f"engage.per_thruster.pinned_n must be an integer 1..{N_MOTORS}, got {n!r}")
    out["pinned_n"] = int(n)
    return out


class PwmInterlocks:
    """The per-thruster runtime interlocks, debounced in TICKS like the attitude ceiling (workers._runtime_fault: a
    counter of consecutive ticks over the limit, reset by one tick under it, trips at round(hold_s * hz) ticks; a
    limit with hold 0 trips on the first tick). ``step`` is called once per control tick with what that tick knows
    — the measured attitude and gyro, the latest pack sample, the pulses that LAST LEFT (the worker judges the
    previous tick's frame; the replay judges the vehicle's SERVO_OUTPUT_RAW) — and returns "" or the FIRST trip's
    sentence, with the measured numbers, in the fixed order age, tilt, rate, volt, current, null, pinned. No I/O, no
    clock: the worker and rov_gui/tools/rl_pwm_interlock_replay.py run this one implementation.

    AGE (2026-10-15) comes first because every item after it reads the attitude it judges: a stale roll/pitch/p/q makes
    TILT and RATE late by the same age. It is the one item where a MISSING input counts as over (``att_age_s`` None or
    non-finite = the station cannot say how old the attitude is) — fail-safe, so a caller that never passes the age
    trips in age_hard_hold_s. ``judge_age=False`` is for the offline replay of a recording that never logged the age
    (CSV schema < 21): the item is then skipped and the meta says so; the worker always judges it."""

    NAMES = ("age", "tilt", "rate", "volt", "current", "null", "pinned")
    #: a pulse within this many us of the cap window's edge counts as "at the cap" (scale mode puts the binding
    #: thruster EXACTLY at it; the 2026-10-01 clip pinned several)
    PINNED_TOL_US = 2

    def __init__(self, cfg=None, hz: float = 20.0, cap: float = 1.0, judge_age: bool = True):
        self.cfg = parse_per_thruster_block(cfg)
        self.hz = float(hz)
        if not (math.isfinite(self.hz) and self.hz > 0.0):
            raise ValueError(f"PwmInterlocks: hz must be finite and > 0, got {hz!r}")
        self.cap = float(cap)
        self.judge_age = bool(judge_age)
        self.reset()

    def _need(self, key: str) -> int:
        return max(1, int(round(float(self.cfg[key]) * self.hz)))

    def reset(self) -> None:
        self.ticks = {n: 0 for n in self.NAMES}       # "age" = the soft rule (age_s / age_hold_s)
        self.age_hard_ticks = 0                        # the hard rule (age_hard_s / age_hard_hold_s)
        self.n_steps = 0
        self.n_trips = 0
        self.last_trip = None
        self.peak = {"tilt_deg": 0.0, "rate_dps": 0.0, "batt_a_max": None, "batt_v_min": None,
                     "null_us_max": 0.0, "pinned_n_max": 0, "att_age_s_max": None, "att_age_unknown_ticks": 0}

    @staticmethod
    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def _count(self, name: str, over: bool) -> None:
        self.ticks[name] = self.ticks[name] + 1 if over else 0

    def step(self, roll_rad, pitch_rad, p_rad_s, q_rad_s, batt_a=None, batt_v=None, pwm_us=None,
             cap=None, att_age_s=None) -> str:
        """One control tick -> "" or the trip sentence. Missing / non-finite inputs are "not over" for their item
        (a pack that never reported cannot trip the pack items; no pulses yet = nothing to judge) — EXCEPT the
        attitude age, where unknown is over (see the class docstring)."""
        c = self.cfg
        self.n_steps += 1
        cap = self.cap if cap is None else float(cap)
        span = cap_span_us(cap)
        trips = []
        # (0) attitude age on the vehicle clock — unknown counts as over
        age = None
        if self.judge_age:
            age = self._num(att_age_s)
            if age is None:
                self.peak["att_age_unknown_ticks"] += 1
            else:
                prev = self.peak["att_age_s_max"]
                self.peak["att_age_s_max"] = age if prev is None else max(prev, age)
            self._count("age", age is None or age > c["age_s"])
            self.age_hard_ticks = self.age_hard_ticks + 1 if (age is None or age > c["age_hard_s"]) else 0
            hard = self.age_hard_ticks >= self._need("age_hard_hold_s")
            if hard or self.ticks["age"] >= self._need("age_hold_s"):
                lim, hold = ((c["age_hard_s"], c["age_hard_hold_s"]) if hard else (c["age_s"], c["age_hold_s"]))
                what = ("UNKNOWN (no vehicle-clock map of ATTITUDE time_boot_ms — fail-safe)" if age is None
                        else f"{age:.2f} s > {lim:g} s")
                trips.append(("age", f"per-thruster interlock AGE: attitude age {what} for {hold:g} s — the roll/"
                                     f"pitch/p/q the policy acts on are stale (vehicle-side MAVLink queue?); neutral "
                                     f"pulses"))
        # (1) tilt — the inner ceiling; the 35 deg engage.attitude_axes.abort_deg stays outside it
        r, p = self._num(roll_rad), self._num(pitch_rad)
        tilt = None if r is None or p is None else math.degrees(max(abs(r), abs(p)))
        self._count("tilt", tilt is not None and tilt > c["abort_deg"])
        if tilt is not None:
            self.peak["tilt_deg"] = max(self.peak["tilt_deg"], tilt)
        if self.ticks["tilt"] >= self._need("abort_s"):
            trips.append(("tilt", f"per-thruster interlock TILT: |roll/pitch| {tilt:.1f} deg > {c['abort_deg']:g} deg "
                                  f"for {c['abort_s']:g} s (roll {math.degrees(r):+.1f}, pitch {math.degrees(p):+.1f} "
                                  f"deg) — neutral pulses, passive righting"))
        # (2) attitude rate — the gyro, the earliest sign of the pool cascade
        pq, qq = self._num(p_rad_s), self._num(q_rad_s)
        rate = None if pq is None or qq is None else math.degrees(max(abs(pq), abs(qq)))
        self._count("rate", rate is not None and rate > c["rate_dps"])
        if rate is not None:
            self.peak["rate_dps"] = max(self.peak["rate_dps"], rate)
        if self.ticks["rate"] >= self._need("rate_s"):
            trips.append(("rate", f"per-thruster interlock RATE: |p/q| {rate:.1f} deg/s > {c['rate_dps']:g} deg/s for "
                                  f"{c['rate_s']:g} s (p {math.degrees(pq):+.1f}, q {math.degrees(qq):+.1f} deg/s) — "
                                  f"neutral pulses"))
        # (3) pack voltage — any tick
        v = self._num(batt_v)
        if v is not None:
            self.peak["batt_v_min"] = v if self.peak["batt_v_min"] is None else min(self.peak["batt_v_min"], v)
        self._count("volt", v is not None and v < c["volt_min"])
        if self.ticks["volt"] >= 1:
            trips.append(("volt", f"per-thruster interlock PACK: batt_v {v:.2f} V < {c['volt_min']:g} V — neutral "
                                  f"pulses"))
        # (4) pack current — the ~1 Hz BATTERY_STATUS sample, held
        a = self._num(batt_a)
        if a is not None:
            self.peak["batt_a_max"] = a if self.peak["batt_a_max"] is None else max(self.peak["batt_a_max"], a)
        self._count("current", a is not None and a > c["current_a"])
        if self.ticks["current"] >= self._need("current_s"):
            trips.append(("current", f"per-thruster interlock PACK: batt_a {a:.1f} A > {c['current_a']:g} A for "
                                     f"{c['current_s']:g} s — neutral pulses"))
        # (5) null-space monitor and (6) the cap-pinned signature, on the pulses that left
        nv = nh = n_pin = None
        if pwm_us is not None:
            try:
                nv, nh = null_space_us(pwm_us)
                pw = np.asarray(pwm_us, dtype=float).reshape(-1)
                n_pin = int(np.sum(np.abs(pw - PWM_NEUTRAL_US) >= max(0, span - self.PINNED_TOL_US)))
            except (TypeError, ValueError):
                nv = nh = n_pin = None
        null_mag = None if nv is None else max(abs(nv), abs(nh))
        self._count("null", null_mag is not None and null_mag > c["null_us"])
        if null_mag is not None:
            self.peak["null_us_max"] = max(self.peak["null_us_max"], null_mag)
        if self.ticks["null"] >= self._need("null_s"):
            trips.append(("null", f"per-thruster interlock NULL: |null_v/null_h| {null_mag:.0f} us > {c['null_us']:g} "
                                  f"us for {c['null_s']:g} s (null_v {nv:+.0f}, null_h {nh:+.0f} us: wrench-free "
                                  f"thrust cancelling itself) — neutral pulses"))
        self._count("pinned", n_pin is not None and span > 0 and n_pin >= c["pinned_n"])
        if n_pin is not None:
            self.peak["pinned_n_max"] = max(self.peak["pinned_n_max"], n_pin)
        if self.ticks["pinned"] >= self._need("pinned_s"):
            trips.append(("pinned", f"per-thruster interlock PINNED: {n_pin} thrusters at the cap (1500 +- {span} us, "
                                    f">= {c['pinned_n']}) for {c['pinned_s']:g} s — allocation collapse, neutral "
                                    f"pulses"))
        if not trips:
            return ""
        name, why = trips[0]
        self.n_trips += 1
        self.last_trip = {"name": name, "why": why, "tick": int(self.n_steps),
                          "values": {"att_age_s": age, "tilt_deg": tilt, "rate_dps": rate, "batt_a": a, "batt_v": v,
                                     "null_v_us": nv, "null_h_us": nh, "pinned_n": n_pin}}
        return why

    def describe(self) -> str:
        c = self.cfg
        age = (f"age {c['age_s']:g} s/{c['age_hold_s']:g} s | {c['age_hard_s']:g} s/{c['age_hard_hold_s']:g} s "
               f"(vehicle clock, unknown = over), " if self.judge_age else "age NOT judged, ")
        return (f"{age}tilt {c['abort_deg']:g} deg/{c['abort_s']:g} s, rate {c['rate_dps']:g} deg/s/{c['rate_s']:g} s, "
                f"pack {c['current_a']:g} A/{c['current_s']:g} s + {c['volt_min']:g} V, null {c['null_us']:g} us/"
                f"{c['null_s']:g} s, pinned >= {c['pinned_n']}/{c['pinned_s']:g} s at cap {self.cap:.2f}")

    def meta(self) -> dict:
        return {"armed": True, "thresholds": dict(self.cfg), "hz": self.hz, "cap": self.cap,
                "age_judged": bool(self.judge_age),
                "hold_ticks": {"age": self._need("age_hold_s"), "age_hard": self._need("age_hard_hold_s"),
                               "tilt": self._need("abort_s"), "rate": self._need("rate_s"), "volt": 1,
                               "current": self._need("current_s"), "null": self._need("null_s"),
                               "pinned": self._need("pinned_s")},
                "n_steps": int(self.n_steps), "n_trips": int(self.n_trips), "last_trip": self.last_trip,
                "peak": dict(self.peak),
                "note": ("pwm LOWs only (rl_pwm, rl_pwm_7d); every number [예측] unless the yaml says otherwise; "
                         "null/pinned judge the pulses that LAST LEFT (one tick behind); age is the ATTITUDE's "
                         "vehicle-clock age (state_assembler.att_age_v), unknown = over; evaluated in "
                         "workers._runtime_fault after the attitude ceiling (35 deg stays the outer limit); the "
                         "age_engage_* / age_map_min_s / banner_hold_s keys are the ENGAGE refusal "
                         "(workers._att_age_refusal), not runtime items")}


# ----------------------------------------------------------------------------- the observation guards (2026-10-10 b)
#: ``rl_pwm.obs_guard`` defaults — what the pwm follower does to the state BEFORE the network sees it, inside
#: HwRlPwm.step (the shared state_assembler is untouched, every other LOW sees the same state as before):
#:   vel_slew_mps2   per-axis slew limit on the body velocity the policy sees, 2.0 m/s^2 [유도: the policy's
#:                   full-authority acceleration is <= 4.5 m/s^2; a single-tag velocity spike is 10-100x that —
#:                   REPORT.md §1 cause 2: "+-100..170 N heave flips" from one spike];
#:   jump_m          a fix that moves more than this between two ticks WHILE n_tags < min_tags is rejected (the
#:                   previous fix is held) [예측: 0.15 m per 50 ms tick = 3 m/s, ten times the vehicle];
#:   min_tags / age_s / hold_after_s / release_after_s
#:                   the tag gate: n_tags < 2 or fix age > 0.3 s for longer than 0.2 s -> the follower outputs the
#:                   (min_tags 3 was the first cut: the replay showed it would have HELD 5-99 % of the ticks of
#:                   the mixer-mode runs too — floor tags are often 1-2 in view [측정 replay report]);
#:                   HOLD pulses instead of the policy's, until the tags are back for >= 0.2 s [측정 REPORT.md §1
#:                   cause 2: n_tags <= 2 at the same tick as the |dpitch| > 5 deg onset or later, never before];
#:   hold_wrench_z_n the hold: Z = -4.0 N NED (4 N UP) through the export's minimum-norm vertical allocation and the
#:                   T200 inverse, horizontals neutral [측정 REPORT.md §1 cause 5: the October net weight with the
#:                   peg 2.5-5.4 N, 0.8-1.6 N without].
OBS_GUARD_DEFAULTS = {"vel_slew_mps2": 2.0, "jump_m": 0.15, "min_tags": 2, "age_s": 0.3, "hold_after_s": 0.2,
                      "release_after_s": 0.2, "hold_wrench_z_n": -4.0}
OBS_GUARD_KEYS = tuple(OBS_GUARD_DEFAULTS)
#: CSV ``pwm_guard`` codes (schema 20): the strongest guard that acted this tick.
GUARD_NONE, GUARD_SLEW, GUARD_JUMP, GUARD_HOLD = 0, 1, 2, 3


def parse_obs_guard_block(raw) -> dict:
    """``rl_pwm.obs_guard`` validated with its defaults (None / {} = all defaults). Unknown keys RAISE."""
    raw = dict(raw or {})
    unknown = set(raw) - set(OBS_GUARD_KEYS)
    if unknown:
        raise ValueError(f"rl_pwm.obs_guard: unknown key(s) {sorted(unknown)} (allowed: {list(OBS_GUARD_KEYS)})")
    out = dict(OBS_GUARD_DEFAULTS)
    out.update(raw)
    for k in ("vel_slew_mps2", "jump_m", "age_s"):
        out[k] = _finite_positive(f"rl_pwm.obs_guard.{k}", out[k])
    for k in ("hold_after_s", "release_after_s"):
        out[k] = _finite_positive(f"rl_pwm.obs_guard.{k}", out[k], allow_zero=True)
    n = out["min_tags"]
    if isinstance(n, bool) or not isinstance(n, (int, float)) or int(n) != n or int(n) < 1:
        raise ValueError(f"rl_pwm.obs_guard.min_tags must be an integer >= 1, got {n!r}")
    out["min_tags"] = int(n)
    z = out["hold_wrench_z_n"]
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)):
        raise ValueError(f"rl_pwm.obs_guard.hold_wrench_z_n must be a finite number (N, NED: negative = up), got {z!r}")
    out["hold_wrench_z_n"] = float(z)
    return out


def hold_throttles(model: "PwmModel", wrench_z_ned_n: float) -> np.ndarray:
    """The HOLD command (2026-10-10 b): a pure heave Z [N, NED] through the export's minimum-norm allocation on the
    FOUR VERTICAL thrusters (pinv of B_flu[:, 4:8] — the horizontals stay neutral by construction, never used to
    trim a residual moment) and the closed-form T200 inverse at the export's voltage_scale (the physical thrust the
    export's wrench estimate assumes: wrench_ned_from_throttles). A vertical thruster whose share is below its
    deadband edge stays ON at the edge (T200CurveNp.inverse below_edge "edge"; -4 N over four thrusters is ~1 N
    each, under the +1.44 / -1.10 N edges, so the shipped hold IS the edge pattern: ~3.7 N up [유도]). 0 N -> all
    zero. Signs: NED Z -> FLU z through FLU_TO_FRD[2] = -1 (Z -4 N NED = 4 N up)."""
    z = float(wrench_z_ned_n)
    if not math.isfinite(z):
        raise ValueError("hold_throttles: non-finite wrench")
    thr = np.zeros(model.n)
    if z == 0.0:
        return thr
    tau_flu = np.zeros(6)
    tau_flu[2] = z * FLU_TO_FRD[2]
    Bv = np.asarray(model.B_flu, dtype=float)[:, 4:8]
    F_v = np.linalg.pinv(Bv) @ tau_flu                       # physical thrust per vertical thruster [N]
    F_v = np.where(np.abs(F_v) < 1e-9, 0.0, F_v)
    vs = float(model.voltage_scale) if float(model.voltage_scale) > 0.0 else 1.0
    thr[4:8] = model.curve.inverse(F_v / vs, below_edge="edge")
    return thr


class PwmObsGuard:
    """The three policy-side observation guards of the pwm LOWs (``rl_pwm.obs_guard``), applied by HwRlPwm.step to the
    (eta, nu) it is handed — never to the worker's state, the CSV, the plot or any other follower:

      1. velocity slew: nu[0:3] may move at most vel_slew_mps2 * dt per axis per tick from what the policy last saw
         (a true step is followed at the slew rate);
      2. fix-jump reject: while the fix is thin (n_tags < min_tags) a position jump > jump_m between two ticks is
         rejected and the previous position is held (the velocity the assembler derived from it is caught by 1.);
      3. tag gate -> HOLD: n_tags < min_tags or fix age > age_s for >= hold_after_s -> ``holding`` (the follower
         sends the hold pulses) until both are good again for >= release_after_s.

    Fix quality (n_tags, age_s) comes from the worker (HwRlPwm.note_fix, every tick before step). ``n_tags None`` =
    never told (a harness stepping the follower directly): the tag gate and the jump reject are INACTIVE — recorded in
    meta as fix_reports vs steps so a run that flew without them is visible. Non-finite inputs pass through untouched
    (the follower's own non-finite check answers them with neutral + n_fail)."""

    def __init__(self, cfg=None, dt: float = 0.05):
        self.cfg = parse_obs_guard_block(cfg)
        self.dt = float(dt)
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError(f"PwmObsGuard: dt must be finite and > 0, got {dt!r}")
        self.reset()

    def _need(self, key: str) -> int:
        return max(1, int(round(float(self.cfg[key]) / self.dt)))

    def reset(self) -> None:
        self._v_prev = None
        self._p_prev = None
        self._bad_ticks = 0
        self._good_ticks = 0
        self.holding = False
        self.n_steps = 0
        self.fix_reports = 0
        self.counts = {"slew": 0, "jump": 0, "hold_ticks": 0, "holds": 0}
        self.last = {"code": GUARD_NONE, "why": ""}

    def apply(self, eta, nu, n_tags=None, age_s=None):
        """(eta_guarded, nu_guarded, code) for this tick; code = the strongest guard that acted (GUARD_*)."""
        c = self.cfg
        eta = np.array(eta, dtype=float).reshape(-1).copy()
        nu = np.array(nu, dtype=float).reshape(-1).copy()
        self.n_steps += 1
        if eta.shape[0] < 3 or nu.shape[0] < 3 or not (np.all(np.isfinite(eta[:3])) and np.all(np.isfinite(nu[:3]))):
            self.last = {"code": GUARD_NONE, "why": "non-finite state: untouched"}
            return eta, nu, GUARD_NONE
        code, why = GUARD_NONE, ""
        told = n_tags is not None
        if told:
            self.fix_reports += 1
        n = int(n_tags) if told else None
        age = self._num(age_s)
        thin = told and n < c["min_tags"]
        # 2. fix-jump reject (thin fix only)
        if thin and self._p_prev is not None:
            jump = float(np.linalg.norm(eta[:3] - self._p_prev))
            if jump > c["jump_m"]:
                eta[:3] = self._p_prev
                code, why = GUARD_JUMP, f"fix jump {jump:.2f} m > {c['jump_m']:g} m with {n} tags: previous fix held"
                self.counts["jump"] += 1
        if code != GUARD_JUMP:
            self._p_prev = eta[:3].copy()
        # 1. velocity slew
        if self._v_prev is not None:
            lim = c["vel_slew_mps2"] * self.dt
            d = nu[:3] - self._v_prev
            if np.any(np.abs(d) > lim + 1e-12):
                nu[:3] = self._v_prev + np.clip(d, -lim, lim)
                self.counts["slew"] += 1
                if code == GUARD_NONE:
                    code, why = GUARD_SLEW, (f"velocity step {float(np.max(np.abs(d))):.2f} m/s in one tick slewed to "
                                             f"{lim:.3f} m/s ({c['vel_slew_mps2']:g} m/s^2)")
        self._v_prev = nu[:3].copy()
        # 3. tag gate
        if told:
            good = (n >= c["min_tags"]) and (age is not None and age <= c["age_s"])
            if good:
                self._bad_ticks = 0
                self._good_ticks += 1
            else:
                self._bad_ticks += 1
                self._good_ticks = 0
            if not self.holding and self._bad_ticks >= self._need("hold_after_s"):
                self.holding = True
                self.counts["holds"] += 1
            elif self.holding and self._good_ticks >= self._need("release_after_s"):
                self.holding = False
            if self.holding:
                self.counts["hold_ticks"] += 1
                code = GUARD_HOLD
                why = (f"tag gate: {n} tags (min {c['min_tags']}), fix age "
                       f"{'?' if age is None else f'{age:.2f}'} s (max {c['age_s']:g}) for >= {c['hold_after_s']:g} s: "
                       f"HOLD pulses (Z {c['hold_wrench_z_n']:+.1f} N NED) until tags return for {c['release_after_s']:g} s")
        self.last = {"code": int(code), "why": why}
        return eta, nu, int(code)

    @staticmethod
    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def meta(self) -> dict:
        return {"config": dict(self.cfg), "dt_s": self.dt,
                "hold_after_ticks": self._need("hold_after_s"), "release_after_ticks": self._need("release_after_s"),
                "n_steps": int(self.n_steps), "fix_reports": int(self.fix_reports), "holding": bool(self.holding),
                "counts": dict(self.counts),
                "codes": {"0": "none", "1": "slew", "2": "jump", "3": "hold"},
                "note": ("applied INSIDE HwRlPwm.step to the state the network sees; the worker's state, CSV px/py/pz "
                         "and nu_* are untouched. fix_reports < n_steps = ticks on which no fix quality was reported "
                         "(the tag gate and the jump reject were inactive on them)")}


# ----------------------------------------------------------------------------- the vehicle gate
class PwmGate:
    """Everything the vehicle must say before a pulse leaves, and while pulses flow (module docstring).

    Fed by the command sink: ``note_param`` for every PARAM_VALUE of the vehicle, ``note_heartbeat`` for every "RLPWM"
    NAMED_VALUE_FLOAT, ``note_pulses`` when a policy frame goes out, ``note_stopped`` when they stop. ``check`` is the
    verdict; ``next_requests`` is what to ask the vehicle for next. No I/O, no clock of its own (``now`` is passed in).
    A parameter older than PARAM_FRESH_S is as good as unread."""

    def __init__(self, motor_direction, sysid: int, channels: Iterable[int] = PWM_CHANNELS_DEFAULT,
                 thruster_probe: "ThrusterProbeCheck | None" = None):
        self.motor_direction = [int(v) for v in motor_direction]
        if len(self.motor_direction) != N_MOTORS:
            raise ValueError("PwmGate: motor_direction needs 8 values")
        self.sysid = int(sysid)
        self.channels = _fixed_channels(channels)
        #: The sign-probe artefact gate (2026-10-10), judged right after the MOT_n_DIRECTION item with the
        #: directions the vehicle SAID. None = not required: the recorded opt-out rl_pwm.require_thruster_probe:
        #: false, or a unit test of the parameter items alone.
        self.thruster_probe = thruster_probe
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
        if self.thruster_probe is not None:
            ok_p, why_p = self.thruster_probe.check(mot, now)
            if not ok_p:
                return False, f"thruster probe: {why_p}"
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
                "channels": list(self.channels), "motor_direction": list(self.motor_direction),
                "thruster_probe": ({"required": False} if self.thruster_probe is None
                                   else self.thruster_probe.state())}


__all__ = ["PwmModel", "PwmGate", "T200CurveNp", "ThrusterProbeCheck", "throttles_to_pwm", "pwm_to_action",
           "pack_rc_override", "scale_throttles", "pulses_within_cap", "cap_span_us", "null_space_us",
           "predicted_pack_current_a", "thrust_scale_at_voltage", "parse_rl_pwm_block", "RL_PWM_KEYS",
           "PwmInterlocks", "PwmObsGuard", "parse_per_thruster_block", "parse_obs_guard_block", "hold_throttles",
           "PER_THRUSTER_DEFAULTS", "PER_THRUSTER_KEYS", "AGE_KEYS", "OBS_GUARD_DEFAULTS", "OBS_GUARD_KEYS",
           "PACK_V_CURVE",
           "GUARD_NONE", "GUARD_SLEW", "GUARD_JUMP", "GUARD_HOLD",
           "NEUTRAL_PULSES", "PWM_NEUTRAL_US", "PWM_SPAN_US", "PWM_MIN_US", "PWM_MAX_US", "PWM_CHANNELS_DEFAULT",
           "PWM_CAP_FIRST_WATER", "PWM_CAP_MODES", "PWM_CAP_MODE_DEFAULT", "CAP_TOL", "SCRIPT_HEARTBEAT_NAME",
           "N_MOTORS", "THRUSTER_PROBE_FILENAME", "THRUSTER_PROBE_MAX_AGE_DAYS", "I_PACK_IDLE_A", "I_FIT_FWD",
           "I_FIT_REV", "PACK_V_NOMINAL"]
