#!/usr/bin/env python3
"""thruster_probe.py — the pool "probe block" that has to run before ANY rl_pwm flight (2026-10-10 fix plan step 4).

    P=~/miniforge3/envs/rovgui-pose/bin/python
    $P -m rov_gui.tools.thruster_probe --dry-run                  # synthetic vehicle, sequence A, no hardware, fast clock
    $P -m rov_gui.tools.thruster_probe --dry-run --seq A,B,C,D,E --free-decay 30
    $P -m rov_gui.tools.thruster_probe --seq A                    # the vehicle: ARMED in MANUAL, props ON, mid-pool
    $P -m rov_gui.tools.thruster_probe --seq A,B,E --mav udpin:0.0.0.0:14550 --note "jaw empty, 16.1 V"
    $P -m rov_gui.tools.thruster_probe --seq C --free-decay 30    # free-decay recorder only (neutral pulses)

WHY. The 2026-10-10 re-analysis of the pwm pool runs (RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/
REPORT.md §0-3) found the training world's trim SIGN wrong and its hydrostatic stiffness 3-5x too stiff, the policy
co-contracting the allocation null space n_v = [0,0,0,0,1,1,1,1], an unmodelled ESC start-up delay, and an unrecorded
net weight. Three verifiers refuted a thruster-sign reversal from the closed-loop logs, but the per-thruster sign probe
of RL_controller/README.md §10.3 item 4 — the only test that can SEE one reversed vertical — has never been done. This
tool is that probe block: it drives the eight thrusters one at a time over the SAME transport the policy flies on
(RC_CHANNELS_OVERRIDE channels 9..16 -> rl_pwm_override.lua on the Navigator, rov_gui/control/rl_pwm.py), judges each
from the vehicle's own ATTITUDE body rates, and measures the four numbers the fix plan re-derives its ranges from.

SEQUENCES (``--seq``, letters in the order they are to run; default A):
  A  per-thruster sign probe — motor m = 1..8: +100 us 1.0 s, neutral 3 s (+ until the gyro is quiet, <= 10 s more),
     -100 us 1.0 s, the same neutral. Verdict from the
     mean body rate on the motor's DOMINANT axis (computed from the export's B_flu and sigma at run time, never
     hardcoded: verticals -> roll, horizontals -> yaw) over 0.15..0.55 s after the FIRST non-neutral SERVO_OUTPUT_RAW
     report of the pulse, minus the 0.5 s pre-command baseline (yaw decays slowly: the previous pulse's rotation would
     otherwise be read as the answer). PASS = sign agreement for both pulses with |mean| >= 5 deg/s, WEAK = agreement below it,
     FAIL = anything else. Writes thruster_sign_probe.json (control/thruster_probe_artifact.py: the engage gate reads it).
  B  n_v leak sweep — a common-mode c_v in {0, +100, +200, +300} us on all four verticals (the null-space direction
     n_v: T5/T8 push up, T6/T7 push down, wrench-free by construction), 5 s each, neutral 3 s between, at the current
     altitude and again after the operator moves the vehicle to a second one. Optionally under the mixer's
     depth-hold pattern (T5 +h, T6 -h, T7 -h, T8 +h us: --nv-hold-us 41 = the 2026-10 holds [측정 REPORT.md §3], or
     auto = calibrated to today's ballast from two short depth-rate reads; default 0 = ballast the vehicle near
     neutral first, E tells how far it is). Depth rate, pitch, roll, current per step; slopes against c_v. Nominal
     prediction: zero slope; a leak >= 5 N per 100 us or >= 1 N*m per 100 us is the finding. +300 us (~25 A [예측])
     last, and only above 15.0 V.
  C  open-loop stiffness — +1 / -1 N*m pitch, then +1 / -1 N*m roll, 4 s each with 4 s neutral around, through the
     minimum-norm vertical allocation F = B_v^+ [0, K, M] and the closed-form T200 inverse. Expected steady angle ~13 deg
     at k 4.5 N*m/rad vs 2.8 deg at 20.3 [유도]. ``--free-decay N``: neutral pulses only for N s while the operator
     tilts the vehicle ~15 deg with a pole and releases it; the damped period and log decrement give k = I (2 pi/T)^2
     with I_pitch 0.87-1.25, I_roll 0.49-1.05 kg m^2 [유도 RL_controller configs/rov_params.yaml + added inertia].
  D  ESC from-rest latency — one motor: 1500->1600 (from rest) vs 1540->1640 (from the deadband edge) vs 1600->1700
     (spinning), 3 repeats, 1.5 s on / 3 s at the pre-level; delay = pulse change -> gyro onset (|rate - baseline| > 5
     deg/s on the motor's dominant axis), rise = 10-90 % of the peak rate. >= 100 ms longer from rest = the BLHeli_S
     start-up penalty is real [스펙 BR forum / BLHeli_S manual]. Default motor T1 (yaw: the same Basic ESC, no heave,
     no restoring moment, so the rate is a real steady state); --esc-motor 5 is the vertical/roll variant, which lifts
     the vehicle out of the depth band within three repeats and tilts it ~26 deg at the 1700 level at k 6.7 [유도] —
     the operator re-positions at the prompt between levels.
  E  rest trim — 60 s of neutral pulses mid-water with the jaw EMPTY: mean/std of roll/pitch (NED, +pitch = nose UP),
     depth drift (net weight sign), battery current (hotel load).

SAFETY. The vehicle must be ARMED in MANUAL with the propellers ON, mid-pool (COM >= 0.6 m above the floor), tether
slack, pack >= 15.5 V, the Lua override installed (rov_gui/tools/install_rl_pwm_lua.py) and every PwmGate item of
control/rl_pwm.py read back from the vehicle and passing, the script's "RLPWM" heartbeat fresh and saying 1 within
1.5 s of the first neutral frames. Nobody in the water. Checked EVERY tick (20 Hz) and aborting with a burst of five
neutral frames (1500 us on all eight) + a neutral MANUAL_CONTROL: |roll| or |pitch| > 25 deg, depth outside 0.2..0.9 m,
battery current > 30 A, SERVO_OUTPUT_RAW older than 0.5 s, ATTITUDE older than 0.5 s, autopilot HEARTBEAT lost (3 s),
DISARMED, mode != MANUAL, the gate failing, any exception, Ctrl-C. Neutral is also sent between every step and at
exit. The |pulse| this tool ever sends is <= 300 us (sequence B) and <= 200 us elsewhere; it never writes a parameter.

ARTEFACTS. One folder data/<YYYYMMDD>/<MMDD_HHMMSS>_probe/ (rov_gui/runstore.py; dry runs: _probedry):
thruster_sign_probe.json, nv_leak.json, stiffness.json (+ free_decay inside it), esc_latency.json, rest_trim.json,
probe_run.json (what ran, what aborted), and probe_ticks.csv — every 20 Hz tick: t, seq, step, the 8 pulses sent, the 8
SERVO_OUTPUT_RAW values, roll/pitch/yaw [deg NED], p/q/r [deg/s], depth [m], battery V/A.

DRY RUN. ``--dry-run`` replaces the MAVLink link with SyntheticVehicle: the export's own T200 curve and B_flu, a
second-order attitude response with the nominal k 6.7 N*m/rad [유도 rov_params cob_arm_effective_m 0.05], a 0.25 s report
lag on every message, a 0.12 s ESC start-up delay from rest, the Lua script's freshness/armed/MANUAL semantics and the
gate parameters. ``--swap 5,8`` / ``--reverse 3`` miswire the synthetic vehicle so the probe can be seen to catch it.
Nothing in a dry run is a measurement: its artefacts say ``dry_run: true`` and the engage gate refuses them.

Every threshold here is tagged in the code; the ones marked [예측] are guesses until the first real artefact exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import select
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.backends.hardware import firmware_version_str  # noqa: E402
from rov_gui.control import thruster_probe_artifact as ART  # noqa: E402
from rov_gui.control.rl_pwm import (N_MOTORS, NEUTRAL_PULSES, PWM_MAX_US, PWM_MIN_US,  # noqa: E402
                                    PWM_NEUTRAL_US, PWM_SPAN_US, SCRIPT_HEARTBEAT_NAME, PwmGate, PwmModel,
                                    pack_rc_override)
from rov_gui.runstore import run_dir  # noqa: E402

DEFAULT_POLICY_DIR = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
STATION_SYSID = 255                      # rov_gui --cmd-sysid default = SYSID_MYGCS
TICK_S = 0.05                            # the policy's own 20 Hz; the Lua script calls a 100 ms gap stale
MANUAL_MODE = 19                         # ArduSub custom_mode MANUAL [스펙 pymavlink mode_mapping_sub]
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID = 6, 8
MAV_CMD_SET_MESSAGE_INTERVAL, MAV_CMD_REQUEST_MESSAGE = 511, 512
MSG_ID = {"SYS_STATUS": 1, "RAW_IMU": 27, "ATTITUDE": 30, "SERVO_OUTPUT_RAW": 36, "VFR_HUD": 74,
          "SCALED_PRESSURE2": 137, "BATTERY_STATUS": 147, "AUTOPILOT_VERSION": 148}
#: Message rates asked for with MAV_CMD_SET_MESSAGE_INTERVAL (ArduSub's defaults are 2-4 Hz: useless for a 0.4 s window).
RATES_HZ = {"ATTITUDE": 25.0, "SERVO_OUTPUT_RAW": 20.0, "SCALED_PRESSURE2": 10.0, "SYS_STATUS": 5.0,
            "BATTERY_STATUS": 5.0, "RAW_IMU": 25.0, "VFR_HUD": 5.0}
SEA_LEVEL_HPA, WATER_RHO, G = 1013.25, 997.0, 9.80665   # backends/hardware.py _depth_m (fresh pool water)
FLU_TO_FRD = np.array(ART.FLU_TO_FRD)
DEG = 180.0 / math.pi

#: Abort limits (every tick). [예측] unless noted — the first real run calibrates them.
LIMITS = {"tilt_abort_deg": 25.0, "depth_min_m": 0.2, "depth_max_m": 0.9, "batt_a_max": 30.0,
          "servo_age_max_s": 0.5, "att_age_max_s": 0.5, "hb_age_max_s": 3.0, "min_batt_v": 15.5,
          "min_batt_v_300": 15.0}
NEUTRAL_BURST_FRAMES = 5
ENGAGE_WAIT_S = 1.5                      # neutral frames before the first live pulse; RLPWM must say 1 by then
#: The settle gate after a step: the nominal neutral time, then more neutral until every gyro axis has stayed below
#: QUIET_DPS for QUIET_WINDOW_S, at most SETTLE_MAX_S longer. The free attitude period is 2.6-3.5 s [측정 REPORT.md §1]
#: and yaw decays with tau ~2 s, so 3 s alone can leave the previous pulse's motion under the next window [예측].
QUIET_DPS = 2.0
QUIET_WINDOW_S = 0.5
SETTLE_MAX_S = 10.0
#: Sequence B: the mixer's depth-hold pattern [측정 mixer-mode holds: REPORT.md §3, data/20261003 mpc_tuned holds:
#: up = T5/T8 > 1500, T6/T7 < 1500] and the common-mode steps along n_v.
NV_HOLD_PATTERN_SIGN = (0, 0, 0, 0, 1, -1, -1, 1)
NV_HOLD_US_SPEC = 41                     # the level the mixer held depth with (peg in the jaw) [측정 REPORT.md §3]
NV_HOLD_MAX_US = 80                      # the auto-calibrated hold level is clamped here (0.2 throttle) [예측]
NV_HOLD_MIN_US = 30                      # below the model deadband (0.075 x 400 us) a hold level does nothing -> off
NV_LEVELS_US = (0, 100, 200, 300)
NV_HOLD_S = 5.0
#: Leak thresholds and the plant constants that turn a slope into a force / moment:
#: heave linear damping 15 N*s/m [예측 RL_controller configs/rov_params.yaml hydro.linear_damping_nominal[2]],
#: pitch/roll stiffness 6.7 N*m/rad [유도 rov_params buoyancy.cob_arm_effective_m 0.05 x 134.6 N].
D_HEAVE_NSM = 15.0
K_NOMINAL_NM_PER_RAD = 6.7
NV_LEAK_N_PER_100US = 5.0
NV_LEAK_NM_PER_100US = 1.0
#: Sequence C: the moments and the inertia bounds [유도 rov_params inertia_diag + payloads + added-inertia placeholder 0.12].
STIFF_TORQUE_NM = 1.0
STIFF_HOLD_S = 4.0
I_PITCH_RANGE = (0.87, 1.25)
I_ROLL_RANGE = (0.49, 1.05)
I_YAW_NOMINAL = 0.73
#: Sequence D.
ESC_LEVELS_DEFAULT = ((1500, 1600), (1540, 1640), (1600, 1700))
ESC_MOTOR_DEFAULT = 1                    # a horizontal: same Basic ESC as the verticals, no heave, no restoring moment
ESC_REPEATS, ESC_ON_S, ESC_OFF_S = 3, 1.5, 3.0
ESC_ONSET_DPS = 5.0
ESC_PENALTY_S = 0.100                    # >= this much longer from rest = the BLHeli_S start-up penalty [스펙 BR forum]
#: Sequence E: a depth drift slower than this is "neutral within resolution" [예측 Bar30 resolution over 60 s].
REST_S = 60.0
REST_DRIFT_MIN_MPS = 0.002


class ProbeAbort(RuntimeError):
    """An abort condition held, the vehicle refused, or the operator pressed Ctrl-C: neutral has been sent."""


class FakeClock:
    """A clock that only moves when somebody sleeps — the dry run and the tests run a 5-minute probe in a second."""

    def __init__(self, t0: float = 1000.0):
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, float(s))


# ============================================================================= the export
class ExportModel:
    """The shipped pwm export, read once: B_flu, sigma, the T200 curve and its closed-form inverse, and the axis/sign
    each thruster must answer with (control/thruster_probe_artifact.expected_axis_sign, shared with the gate)."""

    def __init__(self, policy_dir):
        self.policy_dir = str(policy_dir)
        p = Path(policy_dir) / "obs_spec.json"
        raw = p.read_bytes()
        self.obs_spec_sha1 = hashlib.sha1(raw).hexdigest()
        self.spec = json.loads(raw.decode("utf-8"))
        out = self.spec.get("output") or {}
        if str(out.get("kind")) != "pwm" or not isinstance(out.get("pwm_model"), dict):
            raise ValueError(f"{p}: not a pwm export (output.kind != 'pwm' / no output.pwm_model)")
        pm = out["pwm_model"]
        self.model = PwmModel(pm)
        self.B_flu = np.asarray(self.model.B_flu, dtype=float)
        sig = pm.get("sigma")
        if sig is None or len(sig) != N_MOTORS:
            raise ValueError(f"{p}: output.pwm_model.sigma missing or not {N_MOTORS} values")
        self.sigma = [float(v) for v in sig]
        self.motor_direction = list(self.model.motor_direction)
        self.voltage_scale = float(self.model.voltage_scale)
        self.curve = self.model.curve
        self.expected = {m: ART.expected_axis_sign(self.B_flu.tolist(), m - 1) for m in range(1, N_MOTORS + 1)}

    # ---- forward chain
    def thrust_n(self, throttle: float) -> float:
        """Steady-state thrust [N] on the wire throttle, at the export's voltage_scale."""
        return float(self.curve.thrust(np.array([float(throttle)]))[0]) * self.voltage_scale

    def thrust_for_pulse(self, pulse_us: int) -> float:
        return self.thrust_n((int(pulse_us) - PWM_NEUTRAL_US) / PWM_SPAN_US)

    def wrench_frd(self, pulses) -> np.ndarray:
        """Nominal body wrench [X, Y, Z, K, M, N] (NED/FRD, N / N*m) of eight pulses — logging and predictions only."""
        thr = (np.asarray(pulses, dtype=float) - PWM_NEUTRAL_US) / PWM_SPAN_US
        return self.model.wrench_ned_from_throttles(thr)

    # ---- inverse chain
    def throttle_for_thrust(self, f_n: float) -> float:
        """Closed-form inverse of the export's fits: thrust [N] at voltage_scale -> wire throttle. kgf = a rpm^2 + b rpm + c
        is solved per sign branch for the rpm of that branch, then rpm = p thr + q is inverted. A thrust the deadband
        cannot reach (|thr| <= deadband) returns 0.0 — the caller must see that, not a pulse that does nothing. A thrust
        beyond rpm_max is clipped to |thr| = 1."""
        c = self.curve
        f = float(f_n) / self.voltage_scale
        if not math.isfinite(f):
            raise ValueError("throttle_for_thrust: non-finite thrust")
        if abs(f) < 1e-12:
            return 0.0
        kgf = f / 9.81
        if f > 0.0:
            a, b, cc = c.kgf_pos
            disc = b * b - 4.0 * a * (cc - kgf)
            if disc < 0.0:
                raise ValueError(f"throttle_for_thrust: {f_n:.2f} N outside the positive fit")
            rpm = (-b + math.sqrt(disc)) / (2.0 * a)
            if rpm > c.rpm_max:
                return 1.0
            thr = (rpm - c.rpm_pos[1]) / c.rpm_pos[0]
            return float(min(1.0, thr)) if thr > c.deadband else 0.0
        an, bn, cn = c.kgf_neg
        disc = bn * bn - 4.0 * an * (cn - kgf)
        if disc < 0.0:
            raise ValueError(f"throttle_for_thrust: {f_n:.2f} N outside the negative fit")
        rpm = (-bn + math.sqrt(disc)) / (2.0 * an)          # 2 an < 0: this is the negative root
        if rpm < -c.rpm_max:
            return -1.0
        thr = (rpm - c.rpm_neg[1]) / c.rpm_neg[0]
        return float(max(-1.0, thr)) if thr < -c.deadband else 0.0

    def pulse_for_thrust(self, f_n: float) -> int:
        return int(round(PWM_NEUTRAL_US + PWM_SPAN_US * self.throttle_for_thrust(f_n)))

    def vertical_pulses_for_moment(self, k_frd_nm: float, m_frd_nm: float) -> tuple:
        """(pulses x8, thrusts x4 [N], throttles x4): the minimum-norm allocation of a pure [K, M] body moment (NED/FRD)
        over the four verticals with zero heave, F = B_v^+ [0, K_flu, M_flu] (rows Z, K, M of B_flu, columns 5..8)."""
        w_flu = np.array([0.0, 0.0, 0.0, float(k_frd_nm), float(m_frd_nm), 0.0]) * FLU_TO_FRD
        Bv = self.B_flu[np.ix_([2, 3, 4], [4, 5, 6, 7])]
        F = np.linalg.pinv(Bv) @ w_flu[[2, 3, 4]]
        thr = [self.throttle_for_thrust(float(f)) for f in F]
        pulses = [PWM_NEUTRAL_US] * 4 + [int(round(PWM_NEUTRAL_US + PWM_SPAN_US * t)) for t in thr]
        return tuple(pulses), tuple(float(f) for f in F), tuple(float(t) for t in thr)

    def predicted_tilt_deg(self, pulses, k_nm_per_rad: float = K_NOMINAL_NM_PER_RAD) -> tuple:
        """(roll, pitch) steady angles [deg] of a pulse set at stiffness k — the static balance K/k, M/k [유도]."""
        w = self.wrench_frd(pulses)
        return float(w[3] / k_nm_per_rad * DEG), float(w[4] / k_nm_per_rad * DEG)


# ============================================================================= telemetry
class Telemetry:
    """What the vehicle said, by message, and the full-rate logs the analyses read. Only (target_sys, component 1)
    counts: BlueOS routes every companion's traffic to the same endpoint (backends/hardware.py _msg_from_vehicle)."""

    def __init__(self, target_sys: int, gate: PwmGate, gyro_source: str = "attitude"):
        self.target_sys = int(target_sys)
        self.gate = gate
        self.gyro_source = gyro_source
        self.t_hb = None
        self.armed = None
        self.mode_raw = None
        self.att = None                   # (t, roll, pitch, yaw, p, q, r) [rad, rad/s], NED/FRD
        self.att_log: list = []           # every ATTITUDE (or RAW_IMU gyro + last angles), full rate
        self.servo = None                 # (t, [8])
        self.servo_log: list = []
        self.t_depth = None
        self.depth_m = None
        self.depth_src = ""
        self.depth_log: list = []         # (t, depth)
        self.t_batt = None
        self.batt_v = None
        self.batt_a = None
        self.batt_log: list = []          # (t, v, a)
        self.fw = ""
        self.texts: list = []
        self.n_foreign = 0
        self.counts: dict = {}

    def _from_vehicle(self, msg) -> bool:
        try:
            return int(msg.get_srcSystem()) == self.target_sys and int(msg.get_srcComponent()) == 1
        except AttributeError:
            return True

    def note(self, msg, now: float) -> None:
        mt = msg.get_type()
        if not self._from_vehicle(msg):
            self.n_foreign += 1
            return
        self.counts[mt] = self.counts.get(mt, 0) + 1
        if mt == "HEARTBEAT":
            if int(getattr(msg, "autopilot", 0)) == MAV_AUTOPILOT_INVALID:
                return
            self.t_hb = now
            self.armed = bool(int(getattr(msg, "base_mode", 0) or 0) & MAV_MODE_FLAG_SAFETY_ARMED)
            raw = getattr(msg, "custom_mode", None)
            self.mode_raw = None if raw is None else int(raw)
        elif mt == "ATTITUDE":
            rec = (now, float(msg.roll), float(msg.pitch), float(msg.yaw),
                   float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed))
            self.att = rec
            if self.gyro_source == "attitude":
                self.att_log.append(rec)
        elif mt == "RAW_IMU" and self.gyro_source == "raw_imu":
            if self.att is not None:
                rec = (now, self.att[1], self.att[2], self.att[3],
                       float(msg.xgyro) * 1e-3, float(msg.ygyro) * 1e-3, float(msg.zgyro) * 1e-3)
                self.att_log.append(rec)
        elif mt == "SERVO_OUTPUT_RAW":
            vals = [int(getattr(msg, f"servo{i}_raw", 0) or 0) for i in range(1, N_MOTORS + 1)]
            self.servo = (now, vals)
            self.servo_log.append((now, vals))
        elif mt == "SCALED_PRESSURE2":
            hpa = getattr(msg, "press_abs", None)
            if hpa is not None:
                d = max(0.0, (float(hpa) - SEA_LEVEL_HPA) * 100.0 / (WATER_RHO * G))
                self._depth(now, d, "SCALED_PRESSURE2")
        elif mt == "VFR_HUD":
            if self.depth_src != "SCALED_PRESSURE2":          # the Bar30 derivation wins when it exists
                alt = getattr(msg, "alt", None)
                if alt is not None:
                    self._depth(now, max(0.0, -float(alt)), "VFR_HUD")
        elif mt == "SYS_STATUS":
            v = getattr(msg, "voltage_battery", None)
            a = getattr(msg, "current_battery", None)
            self._batt(now, None if v in (None, 0, 65535) else float(v) / 1000.0,
                       None if a in (None, -1) else float(a) / 100.0)
        elif mt == "BATTERY_STATUS":
            vs = getattr(msg, "voltages", None)
            v = None
            if isinstance(vs, (list, tuple)) and vs and vs[0] not in (0, 65535):
                v = float(vs[0]) / 1000.0
            a = getattr(msg, "current_battery", None)
            self._batt(now, v, None if a in (None, -1) else float(a) / 100.0)
        elif mt == "PARAM_VALUE":
            pid = getattr(msg, "param_id", "")
            if isinstance(pid, bytes):
                pid = pid.decode("ascii", "ignore")
            self.gate.note_param(str(pid).strip("\x00"), getattr(msg, "param_value", None), now)
        elif mt == "NAMED_VALUE_FLOAT":
            name = getattr(msg, "name", "")
            if isinstance(name, bytes):
                name = name.decode("ascii", "ignore")
            if str(name).strip("\x00") == SCRIPT_HEARTBEAT_NAME:
                self.gate.note_heartbeat(getattr(msg, "value", None), now)
        elif mt == "AUTOPILOT_VERSION":
            self.fw = firmware_version_str(getattr(msg, "flight_sw_version", None)) or self.fw
        elif mt == "STATUSTEXT":
            self.texts.append((now, str(getattr(msg, "text", ""))))

    def _depth(self, now, d, src) -> None:
        self.t_depth, self.depth_m, self.depth_src = now, float(d), src
        self.depth_log.append((now, float(d)))

    def _batt(self, now, v, a) -> None:
        if v is not None:
            self.batt_v = v
        if a is not None:
            self.batt_a = a
        self.t_batt = now
        self.batt_log.append((now, self.batt_v, self.batt_a))

    # ---- views for the analyses (degrees, deg/s)
    def att_between(self, t0: float, t1: float) -> list:
        """[(t, roll_deg, pitch_deg, yaw_deg, p_dps, q_dps, r_dps)] with t0 <= t <= t1."""
        return [(e[0], e[1] * DEG, e[2] * DEG, e[3] * DEG, e[4] * DEG, e[5] * DEG, e[6] * DEG)
                for e in self.att_log if t0 <= e[0] <= t1]

    def depth_between(self, t0: float, t1: float) -> list:
        return [e for e in self.depth_log if t0 <= e[0] <= t1]

    def batt_between(self, t0: float, t1: float) -> list:
        return [e for e in self.batt_log if t0 <= e[0] <= t1]

    def servo_between(self, t0: float, t1: float) -> list:
        return [e for e in self.servo_log if t0 <= e[0] <= t1]

    def first_servo(self, motor: int, t0: float, t1: float, level_us: int, tol_us: int) -> tuple:
        """(t, value) of the first SERVO_OUTPUT_RAW in [t0, t1] whose servo<motor> is within tol of level, or None."""
        for t, vals in self.servo_log:
            if t < t0:
                continue
            if t > t1:
                break
            if abs(vals[motor - 1] - int(level_us)) <= int(tol_us):
                return t, vals[motor - 1]
        return None


# ============================================================================= the link
class Link:
    """The pymavlink master (or the synthetic vehicle) behind the few calls this tool makes. ONE rc_channels_override
    call site; the frame comes from the station's own packer (control/rl_pwm.pack_rc_override: motors on 9..16 in that
    order, everything else 65535, pulses explicit 1100..1900)."""

    def __init__(self, master, target_sys: int = 1, target_comp: int = 1):
        self.master = master
        self.target_sys, self.target_comp = int(target_sys), int(target_comp)
        self.n_frames = 0
        self.last_pulses = None
        self.frames: list = []            # (t, pulses) — kept for the record and the tests

    def v2(self) -> bool:
        try:
            return bool(self.master.mavlink20())
        except Exception:                                        # noqa: BLE001
            return False

    def drain(self, tel: Telemetry, now: float, budget: int = 400) -> int:
        n = 0
        for _ in range(budget):
            try:
                msg = self.master.recv_msg()
            except Exception:                                    # noqa: BLE001
                break
            if msg is None:
                break
            tel.note(msg, now)
            n += 1
        return n

    def send_pulses(self, pulses, now: float) -> list:
        ch = pack_rc_override(pulses)
        self.master.mav.rc_channels_override_send(self.target_sys, self.target_comp, *ch)
        self.n_frames += 1
        self.last_pulses = tuple(int(p) for p in pulses)
        self.frames.append((now, self.last_pulses))
        return ch

    def send_neutral_manual(self) -> None:
        self.master.mav.manual_control_send(self.target_sys, 0, 0, 500, 0, 0)

    def heartbeat(self) -> None:
        self.master.mav.heartbeat_send(MAV_TYPE_GCS, MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def request_param(self, name: str) -> None:
        self.master.mav.param_request_read_send(self.target_sys, self.target_comp, name.encode("ascii"), -1)

    def request_version(self) -> None:
        self.master.mav.command_long_send(self.target_sys, self.target_comp, MAV_CMD_REQUEST_MESSAGE, 0,
                                          MSG_ID["AUTOPILOT_VERSION"], 0, 0, 0, 0, 0, 0)

    def set_rates(self, rates: dict = RATES_HZ) -> None:
        for name, hz in rates.items():
            mid = MSG_ID.get(name)
            if mid is None or hz <= 0:
                continue
            self.master.mav.command_long_send(self.target_sys, self.target_comp, MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                              mid, int(1_000_000 / hz), 0, 0, 0, 0, 0)

    def close(self) -> None:
        try:
            self.master.close()
        except Exception:                                        # noqa: BLE001
            pass


# ============================================================================= the synthetic vehicle (--dry-run)
class _Msg:
    def __init__(self, mtype: str, sysid: int = 1, comp: int = 1, **fields):
        self._t, self._s, self._c = mtype, sysid, comp
        self.__dict__.update(fields)

    def get_type(self):
        return self._t

    def get_srcSystem(self):
        return self._s

    def get_srcComponent(self):
        return self._c


class _SynthMav:
    """``master.mav`` of the synthetic vehicle: the send_* calls become events on the vehicle."""

    def __init__(self, veh):
        self.v = veh

    def rc_channels_override_send(self, target_sys, target_comp, *chans):
        if len(chans) != 18:
            raise TypeError("rc_channels_override_send() needs 18 channels (MAVLink 2)")
        self.v.on_override(int(target_sys), [int(c) for c in chans])

    def manual_control_send(self, *args):
        self.v.n_manual += 1

    def heartbeat_send(self, *args):
        self.v.n_gcs_hb += 1

    def param_request_read_send(self, target_sys, target_comp, name, index):
        self.v.on_param_request(name.decode("ascii") if isinstance(name, bytes) else str(name))

    def command_long_send(self, target_sys, target_comp, cmd, conf, p1, p2, *rest):
        self.v.on_command(int(cmd), float(p1), float(p2))


class SyntheticVehicle:
    """A BlueROV2 Heavy that answers like the real one on the wire, for ``--dry-run`` and the tests.

    Physics (NED/FRD, small angles): per thruster the export's T200 curve x voltage_scale through a first-order lag
    tau_thrust and a from-rest ESC start-up delay; w = B_flu F -> FRD; I_x p' = K - c p - k roll (k = 6.7 N*m/rad
    nominal [유도], zeta 0.35 [예측]), the same in pitch, yaw without stiffness; heave m w' = Z + W_net - d w - d2 w|w|
    (W_net 0.02 N by default: a ballasted vehicle, so a 5-minute run stays inside the depth band). Battery:
    hotel + 11 A x throttle^2 per thruster [예측], V = V0 - R I. Reports: every message is delivered ``report_lag_s``
    after the state it describes, at the stream rates MAV_CMD_SET_MESSAGE_INTERVAL asks for (defaults 4 Hz, as the
    firmware's). The Lua script's semantics are reproduced: outputs are forced to the frame's pulses only while a frame
    from SYSID_MYGCS is < 100 ms old AND armed AND MANUAL, else 1500; "RLPWM" says 1/0 at 2 Hz.

    Miswiring for the tests: ``swap=(5, 8)`` routes the pulse of motor 5 to the physical thruster 8 and vice versa;
    ``reverse=(3,)`` spins physical thruster 3 the wrong way; ``thrust_gain=0.5`` is the deadband-region gain the pool
    logs showed (a WEAK, not a FAIL)."""

    DT = 0.01

    def __init__(self, export: ExportModel, clock, *, sysid: int = 1, k_nm_per_rad: float = K_NOMINAL_NM_PER_RAD,
                 inertia=(0.75, 1.05, I_YAW_NOMINAL), zeta: float = 0.35, yaw_damping: float = 0.3,
                 report_lag_s: float = 0.25, esc_start_delay_s: float = 0.12, thrust_tau_s: float = 0.08,
                 swap=None, reverse=(), mot_direction=None, params=None, net_weight_n: float = 0.02,
                 depth0_m: float = 0.5, batt_v0: float = 16.2, batt_r_ohm: float = 0.025, hotel_a: float = 0.6,
                 disarm_at=None, mode_at=None, tilt_impulse=None, heave_mass_kg: float = 26.0,
                 heave_damping: float = D_HEAVE_NSM, heave_drag_quadratic: float = 2000.0,
                 mygcs: int = STATION_SYSID, script_enabled: bool = True, script_running: bool = True,
                 thrust_gain: float = 1.0):
        self.ex, self.clock = export, clock
        self.sysid, self.mygcs = int(sysid), int(mygcs)
        self.k, self.I, self.zeta, self.c_yaw = float(k_nm_per_rad), tuple(float(i) for i in inertia), zeta, yaw_damping
        self.lag, self.esc_delay, self.tau = float(report_lag_s), float(esc_start_delay_s), float(thrust_tau_s)
        self.phys = list(range(N_MOTORS))
        if swap:
            a, b = int(swap[0]) - 1, int(swap[1]) - 1
            self.phys[a], self.phys[b] = b, a
        self.reverse = {int(r) - 1 for r in reverse}
        self.net_weight_n, self.m_z, self.d_z = float(net_weight_n), float(heave_mass_kg), float(heave_damping)
        # N s^2/m^2: the pool runs sank 4.5-6 cm/s under a 10-18 N deficit [측정 REPORT.md §1 item 3] -> ~4000 near the
        # floor; 2000 mid-water [예측]. The open-water Z|w|w ~190 [스펙 MarineGym] would sink a 1 N heavy vehicle at 7 cm/s.
        self.d_z2 = float(heave_drag_quadratic)
        self.v0, self.r_ohm, self.hotel_a = float(batt_v0), float(batt_r_ohm), float(hotel_a)
        self.disarm_at, self.mode_at, self.tilt_impulse = disarm_at, mode_at, tilt_impulse
        self.script_enabled = bool(script_enabled)
        #: every thruster's thrust x this: 0.5 = the small-signal gain near the deadband the pool logs showed (half the
        #: datasheet's, 12 vs 19-24 N/unit at 0.1 [측정 REPORT.md §1 item 5]) or a sagging pack
        self.thrust_gain = float(thrust_gain)
        md = list(export.motor_direction) if mot_direction is None else [int(v) for v in mot_direction]
        self.params = {f"SERVO{n}_FUNCTION": float(32 + n) for n in range(1, 9)}
        self.params.update({f"SERVO{n}_FUNCTION": 0.0 for n in range(9, 17)})
        self.params.update({f"MOT_{n}_DIRECTION": float(md[n - 1]) for n in range(1, 9)})
        self.params.update({f"RC{c}_OPTION": 0.0 for c in range(9, 17)})
        self.params.update({"SCR_ENABLE": 1.0, "SCR_USER1": 1.0, "RC_OPTIONS": 0.0, "RC_OVERRIDE_TIME": 3.0,
                            "SYSID_MYGCS": float(mygcs), "SYSID_THISMAV": float(sysid), "SCR_HEAP_SIZE": 100000.0})
        self.params.update(params or {})
        # state
        self.t = float(clock())
        self.t0 = self.t
        self.armed, self.mode = True, MANUAL_MODE
        self.roll = self.pitch = self.yaw = 0.0
        self.p = self.q = self.r = 0.0
        self.depth, self.w = float(depth0_m), 0.0
        self.thrust = [0.0] * N_MOTORS            # physical, N
        self.thr_cmd = [0.0] * N_MOTORS           # physical, wire throttle (after reverse)
        self.t_spin = [None] * N_MOTORS           # when the physical thruster left zero (ESC start-up)
        self.frame_pulses = list(NEUTRAL_PULSES)
        self.t_frame = -1e9
        self.applied = list(NEUTRAL_PULSES)       # what SERVO_OUTPUT_RAW reports (logical order)
        self.engaged = False
        self.batt_v, self.batt_a = self.v0, self.hotel_a
        self.outbox: deque = deque()
        self.rates = {"HEARTBEAT": 1.0, "ATTITUDE": 4.0, "SERVO_OUTPUT_RAW": 4.0, "SCALED_PRESSURE2": 4.0,
                      "SYS_STATUS": 2.0, "BATTERY_STATUS": 2.0, "VFR_HUD": 4.0, "RAW_IMU": 4.0,
                      "RLPWM": (2.0 if script_running else 0.0)}   # 0 = rl_pwm_override.lua is not loaded at all
        self.next_t = {k: self.t for k in self.rates}
        self.n_manual = self.n_gcs_hb = 0
        self.frames_rx: list = []                 # (t, the 8 pulses) of every accepted frame
        self.raw_frames: list = []                # the 18 channel values as received
        self.foreign_frames = 0
        self._impulsed = False
        self.mav = _SynthMav(self)
        self.target_system = self.sysid

    # ---- master surface
    def mavlink20(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def recv_msg(self):
        self._advance(float(self.clock()))
        if self.outbox and self.outbox[0][0] <= float(self.clock()):
            return self.outbox.popleft()[1]
        return None

    # ---- events from the topside
    def on_override(self, target_sys: int, chans: list) -> None:
        self._advance(float(self.clock()))
        if target_sys != self.sysid:
            self.foreign_frames += 1
            return
        self.raw_frames.append(list(chans))
        any_ = False
        for m in range(N_MOTORS):
            v = chans[8 + m]
            if v not in (0, 65535, 65534):
                self.frame_pulses[m] = min(PWM_MAX_US, max(PWM_MIN_US, v))
                any_ = True
        if any_:
            self.t_frame = self.t
            self.frames_rx.append((self.t, tuple(self.frame_pulses)))

    def on_param_request(self, name: str) -> None:
        if name in self.params:
            self._emit("PARAM_VALUE", param_id=name, param_value=self.params[name], param_type=9,
                       param_count=len(self.params), param_index=-1)

    def on_command(self, cmd: int, p1: float, p2: float) -> None:
        if cmd == MAV_CMD_REQUEST_MESSAGE and int(p1) == MSG_ID["AUTOPILOT_VERSION"]:
            self._emit("AUTOPILOT_VERSION", flight_sw_version=(4 << 24) | (5 << 16) | (1 << 8) | 255)
        elif cmd == MAV_CMD_SET_MESSAGE_INTERVAL:
            for name, mid in MSG_ID.items():
                if mid == int(p1) and name in self.rates and p2 > 0:
                    self.rates[name] = 1e6 / float(p2)

    # ---- physics
    def _emit(self, mtype: str, **fields) -> None:
        self.outbox.append((self.t + self.lag, _Msg(mtype, self.sysid, 1, **fields)))

    def _advance(self, now: float) -> None:
        while self.t + self.DT <= now + 1e-9:
            self._substep(self.DT)

    def _substep(self, dt: float) -> None:
        t = self.t + dt
        if self.disarm_at is not None and t >= self.disarm_at:
            self.armed = False
        if self.mode_at is not None and t >= self.mode_at[0]:
            self.mode = int(self.mode_at[1])
        if self.tilt_impulse is not None and not self._impulsed and t >= self.tilt_impulse[0]:
            self._impulsed = True
            self.roll += float(self.tilt_impulse[1]) / DEG
            self.pitch += float(self.tilt_impulse[2]) / DEG
        fresh = (t - self.t_frame) < 0.1
        ok = self.script_enabled and self.armed and self.mode == MANUAL_MODE and fresh
        self.engaged = ok
        self.applied = list(self.frame_pulses) if ok else list(NEUTRAL_PULSES)
        # pulses (logical) -> physical thrusters
        for m in range(N_MOTORS):
            ph = self.phys[m]
            thr = (self.applied[m] - PWM_NEUTRAL_US) / PWM_SPAN_US
            if ph in self.reverse:
                thr = -thr
            self.thr_cmd[ph] = thr
        F = [0.0] * N_MOTORS
        for ph in range(N_MOTORS):
            thr = self.thr_cmd[ph]
            if abs(thr) <= self.ex.curve.deadband:
                self.t_spin[ph] = None
                target = 0.0
            else:
                if self.t_spin[ph] is None:
                    self.t_spin[ph] = t
                target = (self.ex.thrust_n(thr) * self.thrust_gain) if (t - self.t_spin[ph]) >= self.esc_delay else 0.0
            self.thrust[ph] += (target - self.thrust[ph]) * min(1.0, dt / self.tau)
            F[ph] = self.thrust[ph]
        w = (self.ex.B_flu @ np.asarray(F)) * FLU_TO_FRD
        K, M, N, Z = float(w[3]), float(w[4]), float(w[5]), float(w[2])
        cx = 2.0 * self.zeta * math.sqrt(self.k * self.I[0])
        cy = 2.0 * self.zeta * math.sqrt(self.k * self.I[1])
        self.p += dt * (K - cx * self.p - self.k * self.roll) / self.I[0]
        self.q += dt * (M - cy * self.q - self.k * self.pitch) / self.I[1]
        self.r += dt * (N - self.c_yaw * self.r) / self.I[2]
        self.roll += dt * self.p
        self.pitch += dt * self.q
        self.yaw += dt * self.r
        self.w += dt * (Z + self.net_weight_n - self.d_z * self.w - self.d_z2 * self.w * abs(self.w)) / self.m_z
        self.depth = max(0.0, self.depth + dt * self.w)
        amps = self.hotel_a + sum(11.0 * th * th for th in self.thr_cmd)
        self.batt_a = amps
        self.batt_v = self.v0 - self.r_ohm * amps
        self.t = t
        # reports
        for name, hz in self.rates.items():
            if hz > 0.0 and t + 1e-9 >= self.next_t[name]:
                self.next_t[name] = t + 1.0 / hz
                self._report(name)

    def _report(self, name: str) -> None:
        if name == "HEARTBEAT":
            self._emit("HEARTBEAT", type=12, autopilot=3, base_mode=(128 if self.armed else 0) | 64,
                       custom_mode=self.mode, system_status=4)
        elif name == "ATTITUDE":
            self._emit("ATTITUDE", time_boot_ms=int((self.t - self.t0) * 1e3), roll=self.roll, pitch=self.pitch,
                       yaw=self.yaw, rollspeed=self.p, pitchspeed=self.q, yawspeed=self.r)
        elif name == "RAW_IMU":
            self._emit("RAW_IMU", xgyro=int(self.p * 1e3), ygyro=int(self.q * 1e3), zgyro=int(self.r * 1e3),
                       xacc=0, yacc=0, zacc=-1000)
        elif name == "SERVO_OUTPUT_RAW":
            f = {f"servo{i + 1}_raw": int(self.applied[i]) for i in range(N_MOTORS)}
            f.update({f"servo{i}_raw": 0 for i in range(9, 17)})
            self._emit("SERVO_OUTPUT_RAW", **f)
        elif name == "SCALED_PRESSURE2":
            self._emit("SCALED_PRESSURE2", press_abs=SEA_LEVEL_HPA + self.depth * WATER_RHO * G / 100.0,
                       temperature=2100)
        elif name == "VFR_HUD":
            self._emit("VFR_HUD", alt=-self.depth, heading=int(self.yaw * DEG) % 360)
        elif name == "SYS_STATUS":
            self._emit("SYS_STATUS", voltage_battery=int(self.batt_v * 1000), current_battery=int(self.batt_a * 100),
                       battery_remaining=-1)
        elif name == "BATTERY_STATUS":
            self._emit("BATTERY_STATUS", voltages=[int(self.batt_v * 1000)] + [65535] * 9,
                       current_battery=int(self.batt_a * 100), current_consumed=-1, battery_remaining=-1)
        elif name == "RLPWM":
            self._emit("NAMED_VALUE_FLOAT", name="RLPWM", value=(1.0 if self.engaged else 0.0))


# ============================================================================= analysis helpers (pure)
def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return (sum(xs) / len(xs)) if xs else None


def _std(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _linfit(xs, ys) -> tuple:
    """(slope, intercept, n) least squares; (None, None, n) with fewer than 2 distinct x."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 2:
        return None, None, n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None, None, n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    return slope, my - slope * mx, n


def _axis_index(axis: str) -> int:
    """Column of an att_between row for a rate axis p/q/r (4..6) or an angle roll/pitch/yaw (1..3)."""
    return {"roll": 1, "pitch": 2, "yaw": 3, "p": 4, "q": 5, "r": 6}[axis]


def decay_fit(ts, ys, settle_frac: float = 0.2, min_amp_deg: float = 0.5) -> dict:
    """Damped-period fit of a free decay: rest = the mean of the final ``settle_frac`` of the record; extrema of the
    (lightly smoothed) deviation after its largest excursion; T_d = 2 x mean half-period; log decrement over same-sign
    extrema -> zeta; omega_n = omega_d / sqrt(1 - zeta^2). Returns what it could and says why when it could not."""
    ts = [float(t) for t in ts]
    ys = [float(y) for y in ys]
    n = len(ys)
    out = {"n_samples": n, "period_s": None, "zeta": None, "omega_n_rad_s": None, "n_extrema": 0,
           "amplitude0_deg": None, "rest_deg": None, "note": ""}
    if n < 10:
        out["note"] = "too few samples"
        return out
    k = max(1, int(n * settle_frac))
    rest = sum(ys[-k:]) / k
    out["rest_deg"] = rest
    y = [v - rest for v in ys]
    # 5-point moving average to keep gyro/ATTITUDE noise from making extrema
    sm = [sum(y[max(0, i - 2):i + 3]) / len(y[max(0, i - 2):i + 3]) for i in range(n)]
    i0 = max(range(n), key=lambda i: abs(sm[i]))
    out["amplitude0_deg"] = sm[i0]
    if abs(sm[i0]) < 2.0 * min_amp_deg:
        out["note"] = f"no excursion above {2.0 * min_amp_deg:.1f} deg — was the vehicle tilted and released?"
        return out
    ext = [(ts[i0], sm[i0])]
    for i in range(i0 + 1, n - 1):
        if (sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]) or (sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]):
            if abs(sm[i]) < min_amp_deg or (ts[i] - ext[-1][0]) < 0.3:
                continue
            if (sm[i] > 0) == (ext[-1][1] > 0):
                if abs(sm[i]) > abs(ext[-1][1]):          # same sign again: keep the larger one
                    ext[-1] = (ts[i], sm[i])
                continue
            ext.append((ts[i], sm[i]))
    out["n_extrema"] = len(ext)
    out["extrema"] = [(round(t, 3), round(v, 3)) for t, v in ext]
    if len(ext) < 3:
        out["note"] = "fewer than 3 alternating extrema — no period"
        return out
    half = [ext[i + 1][0] - ext[i][0] for i in range(len(ext) - 1)]
    T = 2.0 * sum(half) / len(half)
    decs = [math.log(abs(ext[i][1]) / abs(ext[i + 2][1])) for i in range(len(ext) - 2) if abs(ext[i + 2][1]) > 1e-6]
    delta = (sum(decs) / len(decs)) if decs else 0.0
    zeta = delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)
    wd = 2.0 * math.pi / T
    wn = wd / math.sqrt(max(1e-6, 1.0 - zeta ** 2))
    out.update(period_s=T, zeta=zeta, omega_n_rad_s=wn, log_decrement=delta)
    return out


def k_from_period(fit: dict, inertia_range) -> dict:
    wn = fit.get("omega_n_rad_s")
    if wn is None:
        return {"k_lo_nm_per_rad": None, "k_hi_nm_per_rad": None}
    return {"k_lo_nm_per_rad": inertia_range[0] * wn * wn, "k_hi_nm_per_rad": inertia_range[1] * wn * wn}


# ============================================================================= the runner
class StepRec:
    __slots__ = ("seq", "name", "pulses", "t_start", "t_end", "meta", "n_ticks")

    def __init__(self, seq, name, pulses, t_start, meta):
        self.seq, self.name, self.pulses, self.t_start, self.meta = seq, name, tuple(pulses), t_start, dict(meta)
        self.t_end = t_start
        self.n_ticks = 0

    def as_dict(self) -> dict:
        return {"seq": self.seq, "name": self.name, "pulses": list(self.pulses), "t_start": self.t_start,
                "t_end": self.t_end, "n_ticks": self.n_ticks, **self.meta}


class ProbeRunner:
    """Executes steps (a pulse set for a duration) at 20 Hz with the abort checks, logs every tick, keeps the gate fed,
    and sends neutral between steps and at exit. The sequences are functions over this."""

    CSV_COLUMNS = (["t", "seq", "step"] + [f"pwm{i}" for i in range(1, 9)] + [f"servo{i}" for i in range(1, 9)]
                   + ["roll_deg", "pitch_deg", "yaw_deg", "p_dps", "q_dps", "r_dps", "depth_m", "batt_v", "batt_a"])

    def __init__(self, link: Link, export: ExportModel, out_dir, *, dry_run: bool, cmd_sysid: int = STATION_SYSID,
                 clock=time.monotonic, sleep=time.sleep, log=print, limits: dict | None = None,
                 operator=None, gyro_source: str = "attitude", notes: str = ""):
        self.link, self.ex, self.out = link, export, Path(out_dir)
        self.dry_run, self.cmd_sysid = bool(dry_run), int(cmd_sysid)
        self.clock, self.sleep, self.log = clock, sleep, log
        self.limits = dict(LIMITS, **(limits or {}))
        self.operator = operator                      # callable(prompt, timeout_s) -> bool, or None = stdin
        self.notes = notes
        self.gate = PwmGate(export.motor_direction, cmd_sysid)
        self.tel = Telemetry(link.target_sys, self.gate, gyro_source)
        self.steps: list = []
        self.current: tuple = ("", "")
        self.abort_reason = None
        self.t_run0 = None
        self.n_ticks = 0
        self.t_last_hb = -1e9
        self.engaged_once = False
        self.results: dict = {}
        self.out.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self.out / "probe_ticks.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_fh)
        self._csv.writerow(self.CSV_COLUMNS)

    # ---- shared artefact header
    def header(self) -> dict:
        mot = [int(round(self.gate.params[f"MOT_{n}_DIRECTION"])) if f"MOT_{n}_DIRECTION" in self.gate.params else None
               for n in range(1, 9)]
        return {"schema": ART.SCHEMA, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool": ART.TOOL,
                "dry_run": self.dry_run,
                "vehicle": {"sysid": int(self.link.target_sys), "fw": self.tel.fw, "mot_direction": mot},
                "export": {"policy_dir": self.ex.policy_dir, "sigma": list(self.ex.sigma),
                           "B_flu": self.ex.B_flu.tolist(), "obs_spec_sha1": self.ex.obs_spec_sha1}}

    def write_json(self, name: str, doc: dict) -> Path:
        p = self.out / name
        p.write_text(json.dumps(doc, indent=1, default=_json_default), encoding="utf-8")
        self.log(f"wrote {p}")
        return p

    # ---- the tick
    def _check(self, now: float) -> None:
        L = self.limits
        t = self.tel
        if t.t_hb is None or now - t.t_hb > L["hb_age_max_s"]:
            raise ProbeAbort("autopilot HEARTBEAT lost")
        if not t.armed:
            raise ProbeAbort("DISARMED")
        if t.mode_raw != MANUAL_MODE:
            raise ProbeAbort(f"flight mode is not MANUAL (custom_mode {t.mode_raw})")
        if t.att is None or now - t.att[0] > L["att_age_max_s"]:
            raise ProbeAbort("ATTITUDE older than %.1f s" % L["att_age_max_s"])
        roll, pitch = t.att[1] * DEG, t.att[2] * DEG
        if abs(roll) > L["tilt_abort_deg"] or abs(pitch) > L["tilt_abort_deg"]:
            raise ProbeAbort(f"tilt roll {roll:+.1f} / pitch {pitch:+.1f} deg beyond {L['tilt_abort_deg']:.0f}")
        if t.depth_m is None or t.t_depth is None or now - t.t_depth > 2.0:
            raise ProbeAbort("no depth (SCALED_PRESSURE2 / VFR_HUD) in the last 2 s")
        if not (L["depth_min_m"] <= t.depth_m <= L["depth_max_m"]):
            raise ProbeAbort(f"depth {t.depth_m:.2f} m outside {L['depth_min_m']:.2f}..{L['depth_max_m']:.2f}")
        if t.batt_a is not None and t.batt_a > L["batt_a_max"]:
            raise ProbeAbort(f"battery current {t.batt_a:.1f} A > {L['batt_a_max']:.0f}")
        if t.servo is None or now - t.servo[0] > L["servo_age_max_s"]:
            raise ProbeAbort("SERVO_OUTPUT_RAW older than %.1f s" % L["servo_age_max_s"])
        ok, why = self.gate.check(now, self.link.v2())
        if not ok:
            raise ProbeAbort(f"rl_pwm gate: {why}")

    def _tick(self, pulses, now: float) -> None:
        self.link.drain(self.tel, now)
        self._check(now)
        if now - self.t_last_hb >= 1.0:
            self.link.heartbeat()
            self.t_last_hb = now
        for name in self.gate.next_requests(now):
            self.link.request_param(name)
        self.link.send_pulses(pulses, now)
        self.link.send_neutral_manual()
        self.gate.note_pulses(now)
        self.n_ticks += 1
        self._log_tick(now, pulses)

    def _log_tick(self, now, pulses) -> None:
        t = self.tel
        att = t.att
        srv = t.servo[1] if t.servo is not None else [None] * 8
        row = ([round(now - (self.t_run0 or now), 3), self.current[0], self.current[1]] + [int(p) for p in pulses]
               + list(srv)
               + ([None] * 6 if att is None else [round(att[1] * DEG, 3), round(att[2] * DEG, 3),
                                                 round(att[3] * DEG, 3), round(att[4] * DEG, 3),
                                                 round(att[5] * DEG, 3), round(att[6] * DEG, 3)])
               + [None if t.depth_m is None else round(t.depth_m, 4),
                  None if t.batt_v is None else round(t.batt_v, 3), None if t.batt_a is None else round(t.batt_a, 3)])
        self._csv.writerow(row)

    def step(self, seq: str, name: str, pulses, duration_s: float, **meta) -> StepRec:
        """Send ``pulses`` at 20 Hz for ``duration_s`` (abort checks every tick). Returns the record with the time of
        the first frame and of the last."""
        pw = tuple(int(p) for p in pulses)
        if len(pw) != N_MOTORS or any(not (PWM_MIN_US <= p <= PWM_MAX_US) for p in pw):
            raise ValueError(f"step {seq}/{name}: bad pulses {pw}")
        self.current = (seq, name)
        t0 = self.clock()
        rec = StepRec(seq, name, pw, t0, meta)
        self.steps.append(rec)
        t_end = t0 + float(duration_s)
        t_next = t0
        while True:
            now = self.clock()
            self._tick(pw, now)
            rec.n_ticks += 1
            rec.t_end = now
            if now >= t_end:
                break
            t_next += TICK_S
            self.sleep(max(0.0, t_next - self.clock()))
        self._csv_fh.flush()
        return rec

    def neutral(self, seq: str, seconds: float, name: str = "neutral") -> StepRec:
        return self.step(seq, name, NEUTRAL_PULSES, seconds)

    def quiet(self, now: float, dps: float = QUIET_DPS, window_s: float = QUIET_WINDOW_S) -> bool:
        """Every gyro axis below ``dps`` on every sample of the last ``window_s`` (and at least 3 samples)."""
        win = self.tel.att_between(now - window_s, now)
        return len(win) >= 3 and all(max(abs(e[4]), abs(e[5]), abs(e[6])) < dps for e in win)

    def settle(self, seq: str, seconds: float, name: str = "neutral", max_extra_s: float = SETTLE_MAX_S) -> dict:
        """Neutral for ``seconds``, then neutral in 0.25 s chunks until :meth:`quiet`, at most ``max_extra_s`` more.
        Returns {"settle_s", "extra_s", "quiet"}; the steps carry the same in their meta."""
        rec = self.neutral(seq, seconds, name)
        extra = 0.0
        q = self.quiet(self.clock())
        while not q and extra < max_extra_s:
            self.step(seq, name + "_settle", NEUTRAL_PULSES, 0.25)
            extra += 0.25
            q = self.quiet(self.clock())
        rec.meta.update(settle_extra_s=extra, quiet=q)
        return {"settle_s": seconds + extra, "extra_s": extra, "quiet": q}

    def wait_operator(self, prompt: str, timeout_s: float, seq: str = "") -> bool:
        """Neutral frames (checks on) until the operator presses ENTER, or ``timeout_s``. True if the operator did."""
        self.log(f"\n>>> {prompt}\n>>> press ENTER to continue (timeout {timeout_s:.0f} s) ...")
        ready = self.operator or _stdin_ready
        self.current = (seq, "wait_operator")
        t0 = self.clock()
        t_next = t0
        while self.clock() - t0 < timeout_s:
            now = self.clock()
            self._tick(NEUTRAL_PULSES, now)
            if ready(prompt, timeout_s):
                return True
            t_next += TICK_S
            self.sleep(max(0.0, t_next - self.clock()))
        self.log(">>> no operator input — continuing without")
        return False

    # ---- preflight / engage / exit
    def preflight(self, timeout_s: float = 20.0) -> dict:
        """Everything the vehicle must say before the first neutral frame. Raises ProbeAbort with the first thing
        missing; nothing is sent on RC 9..16 here."""
        self.t_run0 = self.clock()
        self.log(f"policy {self.ex.policy_dir} (obs_spec sha1 {self.ex.obs_spec_sha1[:12]}), sigma {self.ex.sigma}, "
                 f"expected MOT_1..8_DIRECTION {self.ex.motor_direction}")
        for m in range(1, 9):
            ax, sg = self.ex.expected[m]
            self.log(f"  T{m}: +{ART.PULSE_US} us ({self.ex.thrust_for_pulse(1500 + ART.PULSE_US):+.2f} N) -> {ax} "
                     f"{sg:+d}; -{ART.PULSE_US} us ({self.ex.thrust_for_pulse(1500 - ART.PULSE_US):+.2f} N) -> {ax} "
                     f"{-sg:+d}")
        if not self.link.v2():
            raise ProbeAbort("the link is MAVLink 1 (RC_CHANNELS_OVERRIDE has no channels 9..18) — set MAVLINK20=1")
        t_end = self.clock() + timeout_s
        asked_rates = asked_ver = False
        while self.clock() < t_end:
            now = self.clock()
            self.link.drain(self.tel, now)
            if self.tel.t_hb is not None and not asked_rates:
                self.link.set_rates()
                self.link.request_version()
                asked_rates = asked_ver = True
            if now - self.t_last_hb >= 1.0:
                self.link.heartbeat()
                self.t_last_hb = now
            for name in self.gate.next_requests(now):
                self.link.request_param(name)
            have = (self.tel.t_hb is not None and self.tel.att is not None and self.tel.servo is not None
                    and self.tel.depth_m is not None and self.tel.batt_v is not None
                    and not self.gate.missing(now) and self.gate.t_hb is not None and (self.tel.fw or not asked_ver
                                                                                        or now > t_end - 5.0))
            if have:
                break
            self.sleep(TICK_S)
        now = self.clock()
        t = self.tel
        missing = []
        if t.t_hb is None:
            missing.append("autopilot HEARTBEAT")
        if t.att is None:
            missing.append("ATTITUDE")
        if t.servo is None:
            missing.append("SERVO_OUTPUT_RAW")
        if t.depth_m is None:
            missing.append("depth (SCALED_PRESSURE2 / VFR_HUD)")
        if t.batt_v is None:
            missing.append("battery voltage (SYS_STATUS / BATTERY_STATUS)")
        miss = self.gate.missing(now)
        if miss:
            missing.append(f"gate parameters {miss[:4]}{' ...' if len(miss) > 4 else ''}")
        if self.gate.t_hb is None:
            missing.append("the script's RLPWM heartbeat (rl_pwm_override.lua not running?)")
        if missing:
            raise ProbeAbort("vehicle did not say within %.0f s: %s" % (timeout_s, "; ".join(missing)))
        if t.batt_v < self.limits["min_batt_v"]:
            raise ProbeAbort(f"pack {t.batt_v:.2f} V < {self.limits['min_batt_v']:.1f} V")
        self._check(now)                               # armed, MANUAL, tilt, depth band, current, ages, gate
        roll, pitch = t.att[1] * DEG, t.att[2] * DEG
        if max(abs(roll), abs(pitch)) > 15.0:
            self.log(f"WARNING: resting tilt roll {roll:+.1f} / pitch {pitch:+.1f} deg — a hold pattern will add to it")
        mot = [int(round(self.gate.params[f"MOT_{n}_DIRECTION"])) for n in range(1, 9)]
        info = {"sysid": self.link.target_sys, "fw": t.fw, "mot_direction": mot, "armed": t.armed,
                "mode_raw": t.mode_raw, "depth_m": t.depth_m, "depth_src": t.depth_src, "batt_v": t.batt_v,
                "batt_a": t.batt_a, "roll_deg": roll, "pitch_deg": pitch, "msg_counts": dict(t.counts),
                "foreign_msgs_ignored": t.n_foreign}
        self.log(f"vehicle sysid {info['sysid']} fw {t.fw or '?'}: ARMED {t.armed}, mode {t.mode_raw} (MANUAL), depth "
                 f"{t.depth_m:.2f} m ({t.depth_src}), pack {t.batt_v:.2f} V {t.batt_a if t.batt_a is not None else '?'} A, "
                 f"roll {roll:+.1f} pitch {pitch:+.1f} deg, MOT_1..8_DIRECTION {mot}, gate ok")
        return info

    def engage(self) -> None:
        """Neutral frames for ENGAGE_WAIT_S; the script must say RLPWM = 1 (forcing 1500 on all eight) by the end."""
        self.log(f"engaging: neutral frames for {ENGAGE_WAIT_S:.1f} s, waiting for RLPWM = 1 ...")
        self.neutral("engage", ENGAGE_WAIT_S, "engage")
        if self.gate.script_engaged is not True:
            raise ProbeAbort("rl_pwm_override.lua did not take over (RLPWM stayed 0) — armed? MANUAL? SCR_USER1? "
                             "SYSID_MYGCS == %d? SYSID_THISMAV == %d?" % (self.cmd_sysid, self.link.target_sys))
        self.engaged_once = True
        self.log("engaged: the script is forcing the outputs")

    def neutral_burst(self) -> None:
        """NEUTRAL_BURST_FRAMES neutral frames 50 ms apart + a neutral MANUAL_CONTROL — on every exit once a frame has
        left. Never raises: each send is its own try (a dead link must not keep the next frame from going out)."""
        if self.link.n_frames == 0:
            return
        for _ in range(NEUTRAL_BURST_FRAMES):
            try:
                self.link.send_pulses(NEUTRAL_PULSES, self.clock())
                self.link.send_neutral_manual()
            except Exception:                                    # noqa: BLE001
                pass
            try:
                self.sleep(TICK_S)
            except Exception:                                    # noqa: BLE001
                pass
        self.gate.note_stopped()

    def finish(self, exit_code: int, argv_record: dict | None = None) -> Path:
        self._csv_fh.flush()
        self._csv_fh.close()
        doc = dict(self.header())
        doc.update({"exit_code": int(exit_code), "abort_reason": self.abort_reason,
                    "t_run_s": (None if self.t_run0 is None else self.clock() - self.t_run0),
                    "n_ticks": self.n_ticks, "n_frames_sent": self.link.n_frames,
                    "steps": [s.as_dict() for s in self.steps], "results": self.results, "limits": self.limits,
                    "operator_notes": self.notes, "args": argv_record or {},
                    "msg_counts": dict(self.tel.counts), "foreign_msgs_ignored": self.tel.n_foreign,
                    "vehicle_texts": self.tel.texts[-20:]})
        return self.write_json("probe_run.json", doc)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _stdin_ready(prompt: str, timeout_s: float) -> bool:
    """True when a line is waiting on stdin (ENTER pressed). Non-blocking; False when stdin is not readable."""
    try:
        if sys.stdin is None or sys.stdin.closed:
            return False
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            sys.stdin.readline()
            return True
    except (OSError, ValueError):
        return False
    return False


# ============================================================================= sequence A — per-thruster sign probe
#: The pre-pulse baseline window: yaw has no restoring moment and decays with tau = I/c ~ 2 s (I_yaw 0.73 kg m^2, c ~0.3
#: N*m*s/rad at 10 deg/s from N_|r|r 1.55 [스펙 Wu 2018 BlueROV2]), so the rotation left by the previous pulse is still
#: there 3 s later and would read as the next pulse's answer (the dry run showed exactly that). The verdict is the rate
#: CHANGE: window mean minus the mean of this many seconds before the command.
BASELINE_S = 0.5


def _judge_pulse(tel: Telemetry, st: StepRec, motor: int, axis: str, pulse_us: int) -> dict:
    """The gyro answer to one pulse: t0 = the first SERVO_OUTPUT_RAW within the step (+1 s) showing the pulse, the
    mean of the dominant-axis rate over WINDOW_S after it MINUS the pre-command baseline (BASELINE_S), plus every axis
    (raw and baseline) for the record."""
    level = PWM_NEUTRAL_US + int(pulse_us)
    first = tel.first_servo(motor, st.t_start, st.t_end + 1.0, level, abs(int(pulse_us)) // 2)
    base_win = tel.att_between(st.t_start - BASELINE_S, st.t_start)
    base = {k: (_mean(e[_axis_index(k)] for e in base_win) or 0.0) for k in ("p", "q", "r")}
    out = {"t_cmd": st.t_start, "t_servo": None, "servo_latency_s": None, "servo_value_us": None,
           "window_s": list(ART.WINDOW_S), "baseline_s": BASELINE_S, "n_baseline": len(base_win),
           "baseline_p_dps": base["p"], "baseline_q_dps": base["q"], "baseline_r_dps": base["r"],
           "n": 0, "mean": None, "mean_raw": None, "mean_p_dps": None, "mean_q_dps": None, "mean_r_dps": None}
    if first is None:
        out["note"] = "SERVO_OUTPUT_RAW never showed the pulse"
        return out
    t0, val = first
    out.update(t_servo=t0, servo_latency_s=t0 - st.t_start, servo_value_us=val)
    win = tel.att_between(t0 + ART.WINDOW_S[0], t0 + ART.WINDOW_S[1])
    out["n"] = len(win)
    if win:
        raw = {k: _mean(e[_axis_index(k)] for e in win) for k in ("p", "q", "r")}
        out["mean_raw"] = raw[axis]
        out["mean"] = raw[axis] - base[axis]
        out["mean_p_dps"] = raw["p"] - base["p"]
        out["mean_q_dps"] = raw["q"] - base["q"]
        out["mean_r_dps"] = raw["r"] - base["r"]
    return out


def run_seq_a(pr: ProbeRunner) -> dict:
    ex = pr.ex
    motors: list = []
    pr.log(f"\n== A: per-thruster sign probe, +-{ART.PULSE_US} us, {ART.ON_S:.1f} s on / {ART.OFF_S:.1f} s off ==")
    try:
        for m in range(1, N_MOTORS + 1):
            axis, sign = ex.expected[m]
            res = {}
            for sgn, label in ((+1, "pos"), (-1, "neg")):
                pulses = list(NEUTRAL_PULSES)
                pulses[m - 1] = PWM_NEUTRAL_US + sgn * ART.PULSE_US
                quiet_before = pr.quiet(pr.clock())
                st = pr.step("A", f"T{m}{'+' if sgn > 0 else '-'}", pulses, ART.ON_S, motor=m, sign=sgn)
                settled = pr.settle("A", ART.OFF_S)
                res[label] = _judge_pulse(pr.tel, st, m, axis, sgn * ART.PULSE_US)
                res[label].update(quiet_before=quiet_before, settle_after=settled)
            rec = ART.motor_record(m, axis, sign, res["pos"]["mean"], res["neg"]["mean"], res["pos"]["n"],
                                   res["neg"]["n"], detail={"pos": res["pos"], "neg": res["neg"]})
            motors.append(rec)
            pr.log(f"  T{m}: expect {axis} {sign:+d}; +pulse {_fmt(rec['mean_rate_pos_dps'])} deg/s (n {rec['n_samples_pos']}), "
                   f"-pulse {_fmt(rec['mean_rate_neg_dps'])} deg/s (n {rec['n_samples_neg']}) -> {rec['verdict']}")
    except ProbeAbort as e:
        # the artefact is still written: the un-probed motors FAIL (n 0), all_pass is false, the reason is in the notes
        for m in range(len(motors) + 1, N_MOTORS + 1):
            axis, sign = ex.expected[m]
            motors.append(ART.motor_record(m, axis, sign, None, None, 0, 0, detail={"note": "not probed (aborted)"}))
        doc = _build_sign_doc(pr, motors, f"ABORTED during A: {e}")
        pr.write_json(ART.FILENAME, doc)
        raise
    doc = _build_sign_doc(pr, motors, pr.notes)
    pr.write_json(ART.FILENAME, doc)
    pr.log(f"A: {ART.summary(doc)}\nA: all_pass = {doc['all_pass']}")
    pr.results["A"] = {"all_pass": doc["all_pass"], "verdicts": {f"T{m['motor']}": m["verdict"] for m in motors}}
    return doc


def _build_sign_doc(pr: ProbeRunner, motors: list, notes: str) -> dict:
    h = pr.header()
    return ART.build(dry_run=pr.dry_run, sysid=h["vehicle"]["sysid"], fw=h["vehicle"]["fw"],
                     mot_direction=[(-9 if v is None else v) for v in h["vehicle"]["mot_direction"]],
                     policy_dir=pr.ex.policy_dir, sigma=pr.ex.sigma, B_flu=pr.ex.B_flu.tolist(),
                     obs_spec_sha1=pr.ex.obs_spec_sha1, motors=motors, operator_notes=notes, written=h["written"])


def _fmt(v, nd=1) -> str:
    return "n/a" if v is None else f"{float(v):+.{nd}f}"


# ============================================================================= sequence B — n_v leak sweep
def _hold_summary(tel: Telemetry, st: StepRec, cv: int) -> dict:
    """Depth rate (linear fit after the first second), attitude over the last 3 s, current and the reported servos."""
    d = tel.depth_between(st.t_start + 1.0, st.t_end)
    slope, _, n_d = _linfit([e[0] for e in d], [e[1] for e in d])
    att = tel.att_between(max(st.t_start + 1.0, st.t_end - 3.0), st.t_end)
    bat = tel.batt_between(st.t_start + 0.5, st.t_end)
    srv = tel.servo_between(st.t_start + 0.5, st.t_end)
    return {"cv_us": int(cv), "pulses": list(st.pulses), "t_start": st.t_start, "t_end": st.t_end,
            "depth_start_m": (d[0][1] if d else None), "depth_end_m": (d[-1][1] if d else None),
            "depth_rate_mps": slope, "n_depth": n_d,
            "pitch_mean_deg": _mean(e[2] for e in att), "roll_mean_deg": _mean(e[1] for e in att),
            "pitch_std_deg": _std(e[2] for e in att), "roll_std_deg": _std(e[1] for e in att), "n_att": len(att),
            "batt_a_mean": _mean(e[2] for e in bat), "batt_a_max": (max((e[2] for e in bat if e[2] is not None),
                                                                        default=None)),
            "batt_v_mean": _mean(e[1] for e in bat), "batt_v_min": (min((e[1] for e in bat if e[1] is not None),
                                                                        default=None)),
            "servo_mean_us": [_mean(v[i] for _, v in srv) for i in range(N_MOTORS)]}


def _nv_regress(steps: list) -> dict:
    xs = [s["cv_us"] / 100.0 for s in steps]
    out = {"n_steps": len(steps)}
    for key, name in (("depth_rate_mps", "depth_rate_mps_per_100us"), ("pitch_mean_deg", "pitch_deg_per_100us"),
                      ("roll_mean_deg", "roll_deg_per_100us"), ("batt_a_mean", "batt_a_per_100us")):
        slope, icpt, n = _linfit(xs, [s[key] for s in steps])
        out[name] = slope
        out[name + "_intercept"] = icpt
    dr = out["depth_rate_mps_per_100us"]
    out["leak_n_per_100us_down_positive"] = None if dr is None else dr * D_HEAVE_NSM
    out["leak_pitch_nm_per_100us"] = (None if out["pitch_deg_per_100us"] is None
                                      else out["pitch_deg_per_100us"] / DEG * K_NOMINAL_NM_PER_RAD)
    out["leak_roll_nm_per_100us"] = (None if out["roll_deg_per_100us"] is None
                                     else out["roll_deg_per_100us"] / DEG * K_NOMINAL_NM_PER_RAD)
    flags = []
    if out["leak_n_per_100us_down_positive"] is not None and abs(out["leak_n_per_100us_down_positive"]) >= NV_LEAK_N_PER_100US:
        flags.append(f"heave leak {out['leak_n_per_100us_down_positive']:+.1f} N per 100 us (down positive)")
    for ax in ("pitch", "roll"):
        v = out[f"leak_{ax}_nm_per_100us"]
        if v is not None and abs(v) >= NV_LEAK_NM_PER_100US:
            flags.append(f"{ax} leak {v:+.2f} N*m per 100 us")
    out["findings"] = flags
    out["conversion_note"] = (f"force = depth-rate slope x {D_HEAVE_NSM} N*s/m [예측 rov_params linear_damping heave]; "
                              f"moment = angle slope x {K_NOMINAL_NM_PER_RAD} N*m/rad [유도 rov_params cob_arm 0.05 m]")
    return out


def _depth_rate_of(pr: ProbeRunner, st: StepRec, skip_s: float = 1.0):
    d = pr.tel.depth_between(st.t_start + skip_s, st.t_end)
    slope, _, n = _linfit([e[0] for e in d], [e[1] for e in d])
    return slope, n


def calibrate_hold_us(pr: ProbeRunner, probe_us: int = NV_HOLD_US_SPEC, seconds: float = 2.5,
                      max_us: int = NV_HOLD_MAX_US) -> dict:
    """The hold level that makes the depth-hold pattern balance TODAY's net weight: the depth rate at neutral (v0,
    sinking positive) and at the pattern scaled to ``probe_us`` (v1), interpolated linearly to zero rate and clamped
    to 0..max_us. The ballast is not the vehicle's spec and the T200 gain near the deadband is not the datasheet's
    [측정 REPORT.md §1 item 5], so the level is measured, not assumed; both rates go into the artefact."""
    st0 = pr.neutral("B", seconds, "hold_cal_neutral")
    v0, n0 = _depth_rate_of(pr, st0)
    pat, _, _ = _hold_pulses(pr.ex, probe_us, 0)
    st1 = pr.step("B", f"hold_cal_{probe_us}us", pat, seconds, hold_us=probe_us)
    pr.neutral("B", 2.0)
    v1, n1 = _depth_rate_of(pr, st1, skip_s=1.0)
    out = {"probe_us": probe_us, "rate_neutral_mps": v0, "rate_pattern_mps": v1, "n_neutral": n0, "n_pattern": n1}
    if v0 is None or v1 is None:
        out.update(hold_us=0, note="no depth rate measured — pattern off")
    elif abs(v0 - v1) < 0.005:
        out.update(hold_us=0, note="the pattern changed the depth rate by < 5 mm/s — pattern off (neutral vehicle?)")
    else:
        h = probe_us * v0 / (v0 - v1)
        hc = int(round(min(max_us, max(0.0, h))))
        if hc < NV_HOLD_MIN_US:
            out.update(hold_us=0, note=f"interpolated {h:+.1f} us is below the {NV_HOLD_MIN_US} us deadband: the vehicle "
                                       f"is lighter than the pattern's minimum thrust — pattern off")
        else:
            out.update(hold_us=hc, note=f"interpolated {h:+.1f} us, clamped to {NV_HOLD_MIN_US}..{max_us}")
    pr.log(f"  hold calibration: neutral {_fmt(None if v0 is None else v0 * 100, 2)} cm/s, pattern {probe_us} us "
           f"{_fmt(None if v1 is None else v1 * 100, 2)} cm/s -> hold {out['hold_us']} us ({out['note']})")
    return out


def _hold_pulses(ex: ExportModel, hold_us: int, cv_us: int) -> tuple:
    """(pulses x8, thrust requested x4 [N], thrust achievable x4 [N]) for the depth-hold pattern at ``hold_us`` plus the
    common mode ``cv_us`` on the four verticals.

    With no hold the common mode is the spec's: 1500 + c_v on all four, equal pulses = equal thrust along each
    thruster's positive-PWM direction = the null-space direction n_v, wrench-free. ON TOP OF A HOLD it must be applied
    in THRUST space: the T200 curve is convex, so T5/T8 at +h + c_v gain far more thrust than T6/T7 at -h + c_v lose
    (the dry run showed 13.7 N net up and 3.5 deg of pitch at h 53 / c_v 100 [유도 dry run 2026-10-10]). So each
    thruster gets its hold thrust plus the thrust a lone thruster makes at 1500 + c_v, mapped back through the T200
    inverse; the deadband and the 1 us rounding leave a small per-thruster error, returned as achievable vs
    requested so the artefact carries it."""
    sgn = NV_HOLD_PATTERN_SIGN
    if int(hold_us) == 0:
        pulses = [PWM_NEUTRAL_US + int(cv_us)] * 4
        req = [ex.thrust_for_pulse(p) for p in pulses]
        return (PWM_NEUTRAL_US,) * 4 + tuple(pulses), tuple(req), tuple(req)
    c_f = ex.thrust_for_pulse(PWM_NEUTRAL_US + int(cv_us))
    req = [ex.thrust_for_pulse(PWM_NEUTRAL_US + sgn[i] * int(hold_us)) + c_f for i in range(4, 8)]
    pulses = [ex.pulse_for_thrust(f) for f in req]
    ach = [ex.thrust_for_pulse(p) for p in pulses]
    return (PWM_NEUTRAL_US,) * 4 + tuple(pulses), tuple(req), tuple(ach)


def run_seq_b(pr: ProbeRunner, levels=NV_LEVELS_US, hold_s: float = NV_HOLD_S, passes: int = 2,
              operator_timeout_s: float = 180.0, hold_us="0") -> dict:
    """``hold_us``: 0 = the pure common mode (default: c_v alone is wrench-free by construction and the leak is the
    depth-rate CHANGE between levels, so a ballasted vehicle needs no hold), an integer = the depth-hold pattern at
    that level (41 = the 2026-10 mixer holds), "auto" = the level calibrated to today's ballast. Under a hold the
    3 s rest between steps is the hold-only pattern (c_v 0) — a heavy vehicle would otherwise leave the depth band
    in the gaps; with no hold it is neutral."""
    pr.log(f"\n== B: n_v leak sweep, depth-hold pattern (sign {NV_HOLD_PATTERN_SIGN[4:]}, level {hold_us} us) + c_v in "
           f"{list(levels)} us on T5..T8, {hold_s:.0f} s each, {passes} altitude pass(es) ==")
    doc = dict(pr.header())
    doc.update({"sequence": "B", "hold_pattern_sign": list(NV_HOLD_PATTERN_SIGN), "hold_us_requested": str(hold_us),
                "hold_us_spec": NV_HOLD_US_SPEC, "levels_us": list(levels), "hold_s": hold_s, "passes": []})
    for ip in range(passes):
        if ip > 0:
            moved = pr.wait_operator("B: move the vehicle to the SECOND altitude (e.g. 0.3-0.4 m above the floor; stay "
                                     "inside the depth band), let it settle", operator_timeout_s, seq="B")
            if not moved:
                doc["passes"].append({"pass": ip + 1, "skipped": "no operator input for the second altitude"})
                break
            pr.neutral("B", 3.0)
        if str(hold_us).lower() == "auto":
            cal = calibrate_hold_us(pr)
            h = int(cal["hold_us"])
        else:
            h = int(hold_us)
            cal = {"hold_us": h, "note": "forced by --nv-hold-us"}
        steps = []
        for cv in levels:
            if cv >= 300 and (pr.tel.batt_v is None or pr.tel.batt_v <= pr.limits["min_batt_v_300"]):
                steps.append({"cv_us": int(cv), "skipped": f"pack {pr.tel.batt_v} V <= {pr.limits['min_batt_v_300']} V"})
                pr.log(f"  pass {ip + 1} c_v {cv:+d}: SKIPPED (pack {pr.tel.batt_v} V)")
                continue
            pulses, req, ach = _hold_pulses(pr.ex, h, cv)
            w = pr.ex.wrench_frd(pulses)
            st = pr.step("B", f"pass{ip + 1}_cv{cv:+d}", pulses, hold_s, cv=int(cv), altitude_pass=ip + 1,
                         hold_us=h, nominal_wrench_frd=[round(float(v), 3) for v in w])
            if h > 0:
                pr.step("B", "rest_hold", _hold_pulses(pr.ex, h, 0)[0], 3.0, hold_us=h)
            else:
                pr.settle("B", 3.0)
            s = _hold_summary(pr.tel, st, cv)
            s["nominal_wrench_frd"] = [round(float(v), 3) for v in w]
            s["thrust_requested_n"] = [round(f, 3) for f in req]
            s["thrust_achievable_n"] = [round(f, 3) for f in ach]
            steps.append(s)
            pr.log(f"  pass {ip + 1} c_v {cv:+d}: depth {_fmt(s['depth_start_m'], 2)}->{_fmt(s['depth_end_m'], 2)} m "
                   f"({_fmt(None if s['depth_rate_mps'] is None else s['depth_rate_mps'] * 100, 2)} cm/s), pitch "
                   f"{_fmt(s['pitch_mean_deg'])} roll {_fmt(s['roll_mean_deg'])} deg, {_fmt(s['batt_a_mean'])} A "
                   f"(max {_fmt(s['batt_a_max'])}), {_fmt(s['batt_v_min'], 2)} V min")
        valid = [s for s in steps if "skipped" not in s]
        reg = _nv_regress(valid) if len(valid) >= 2 else {"n_steps": len(valid), "findings": []}
        doc["passes"].append({"pass": ip + 1, "depth_at_start_m": (valid[0]["depth_start_m"] if valid else None),
                              "hold_us": h, "hold_calibration": cal, "steps": steps, "regression": reg})
        pr.log(f"  pass {ip + 1} regression: {json.dumps({k: v for k, v in reg.items() if 'per_100us' in k and 'intercept' not in k}, default=_json_default)}"
               f"\n  findings: {reg.get('findings') or 'none (slopes below thresholds)'}")
    pooled = [s for p in doc["passes"] for s in p.get("steps", []) if "skipped" not in s]
    doc["pooled_regression"] = _nv_regress(pooled) if len(pooled) >= 2 else {"n_steps": len(pooled), "findings": []}
    doc["thresholds"] = {"leak_n_per_100us": NV_LEAK_N_PER_100US, "leak_nm_per_100us": NV_LEAK_NM_PER_100US,
                         "d_heave_nsm": D_HEAVE_NSM, "k_nominal_nm_per_rad": K_NOMINAL_NM_PER_RAD}
    doc["nominal_prediction"] = ("zero slope: the common mode on the four verticals is the allocation null space n_v "
                                 "(wrench-free) — applied in pulse space when hold = 0, in thrust space on top of a hold "
                                 "(the T200 curve is convex: equal pulse offsets are NOT equal thrust offsets)")
    pr.write_json("nv_leak.json", doc)
    pr.results["B"] = {"findings": doc["pooled_regression"].get("findings", []),
                       "passes": len([p for p in doc["passes"] if "skipped" not in p])}
    return doc


# ============================================================================= sequence C — stiffness + free decay
def run_seq_c(pr: ProbeRunner, torque_nm: float = STIFF_TORQUE_NM, hold_s: float = STIFF_HOLD_S) -> dict:
    ex = pr.ex
    pr.log(f"\n== C: open-loop stiffness, +-{torque_nm:.1f} N*m pitch then roll, {hold_s:.0f} s each ==")
    doc = dict(pr.header())
    doc.update({"sequence": "C", "torque_nm": torque_nm, "hold_s": hold_s, "steps": [],
                "expected_angle_deg_note": f"{torque_nm / 4.5 * DEG:.1f} deg at k 4.5, {torque_nm / 20.3 * DEG:.1f} deg "
                                           f"at k 20.3 N*m/rad [유도]"})
    base = pr.neutral("C", hold_s, "baseline")
    for name, kd, md in (("pitch+", 0.0, +1.0), ("pitch-", 0.0, -1.0), ("roll+", +1.0, 0.0), ("roll-", -1.0, 0.0)):
        K, M = kd * torque_nm, md * torque_nm
        pulses, F, thr = ex.vertical_pulses_for_moment(K, M)
        dead = [i + 5 for i, t in enumerate(thr) if t == 0.0]
        if dead:
            pr.log(f"  {name}: thrusters {dead} would need less than the deadband lets through — step skipped; raise "
                   f"--stiffness-nm")
            doc["steps"].append({"name": name, "K_nm": K, "M_nm": M, "skipped": f"below deadband on {dead}"})
            continue
        axis = "pitch" if M else "roll"
        col = _axis_index(axis)
        pre = pr.tel.att_between(base.t_end - 1.5, base.t_end)
        st = pr.step("C", name, pulses, hold_s, K_nm=K, M_nm=M, thrusts_n=[round(f, 3) for f in F])
        pr.settle("C", hold_s)
        base = pr.steps[-1]
        resp = pr.tel.att_between(st.t_end - 1.5, st.t_end)
        allr = pr.tel.att_between(st.t_start, st.t_end)
        a0, a1 = _mean(e[col] for e in pre), _mean(e[col] for e in resp)
        delta = None if a0 is None or a1 is None else a1 - a0
        torque = K if axis == "roll" else M
        k_est = None if delta is None or abs(delta) < 0.3 else abs(torque) / (abs(delta) / DEG)
        peak = None
        if allr and a0 is not None:
            peak = max(allr, key=lambda e: abs(e[col] - a0))
        sign_ok = None if delta is None or abs(delta) < 0.3 else (delta > 0) == (torque > 0)
        rec = {"name": name, "K_nm": K, "M_nm": M, "pulses": list(pulses), "thrusts_n": [round(f, 3) for f in F],
               "throttles": [round(t, 4) for t in thr], "axis": axis, "angle_before_deg": a0, "angle_steady_deg": a1,
               "delta_deg": delta, "k_est_nm_per_rad": k_est, "sign_as_expected": sign_ok,
               "peak_deg": (None if peak is None else peak[col] - a0),
               "t_peak_s": (None if peak is None else peak[0] - st.t_start), "n_before": len(pre), "n_steady": len(resp)}
        doc["steps"].append(rec)
        pr.log(f"  {name}: pulses {list(pulses[4:])} ({[round(f, 2) for f in F]} N); {axis} {_fmt(a0)} -> {_fmt(a1)} deg "
               f"(delta {_fmt(delta)}, peak {_fmt(rec['peak_deg'])} at {_fmt(rec['t_peak_s'], 2)} s) -> k "
               f"{_fmt(k_est)} N*m/rad, sign {'ok' if sign_ok else sign_ok}")
    ks = [s["k_est_nm_per_rad"] for s in doc["steps"] if s.get("k_est_nm_per_rad")]
    doc["k_pitch_nm_per_rad"] = _mean(s["k_est_nm_per_rad"] for s in doc["steps"] if s.get("axis") == "pitch"
                                      and s.get("k_est_nm_per_rad"))
    doc["k_roll_nm_per_rad"] = _mean(s["k_est_nm_per_rad"] for s in doc["steps"] if s.get("axis") == "roll"
                                     and s.get("k_est_nm_per_rad"))
    doc["k_mean_nm_per_rad"] = _mean(ks)
    doc["note"] = ("k = |torque| / |steady angle change| with the nominal T200 x voltage_scale thrust (the real thrust "
                   "near the deadband is ~half the datasheet's [측정 REPORT.md §1 item 5]: a k from here is an UPPER "
                   "bound until the thrust is measured); the roll pulses are within ~10 us of the model deadband")
    pr.write_json("stiffness.json", doc)
    pr.results["C"] = {"k_pitch": doc["k_pitch_nm_per_rad"], "k_roll": doc["k_roll_nm_per_rad"]}
    return doc


def run_free_decay(pr: ProbeRunner, seconds: float = 30.0, before=None) -> dict:
    """``before`` (optional) runs right before the recording starts — the dry run schedules its synthetic tilt there."""
    pr.log(f"\n== C/free-decay: NEUTRAL pulses for {seconds:.0f} s — tilt the vehicle ~15 deg (pole, never hands near "
           f"the props), release, keep clear ==")
    if before is not None:
        before()
    st = pr.step("C", "free_decay", NEUTRAL_PULSES, seconds)
    att = pr.tel.att_between(st.t_start, st.t_end)
    doc = dict(pr.header())
    doc.update({"sequence": "C/free_decay", "seconds": seconds, "n_att": len(att), "axes": {}})
    for axis, irange in (("pitch", I_PITCH_RANGE), ("roll", I_ROLL_RANGE)):
        col = _axis_index(axis)
        fit = decay_fit([e[0] for e in att], [e[col] for e in att])
        fit.update(k_from_period(fit, irange))
        fit["inertia_range_kgm2"] = list(irange)
        doc["axes"][axis] = fit
        pr.log(f"  {axis}: {fit['n_extrema']} extrema, T_d {_fmt(fit['period_s'], 2)} s, zeta {_fmt(fit['zeta'], 3)}, "
               f"k {_fmt(fit['k_lo_nm_per_rad'])}..{_fmt(fit['k_hi_nm_per_rad'])} N*m/rad (I {irange}) {fit['note']}")
    doc["note"] = ("k = I (2 pi / T)^2 corrected for damping; I ranges [유도 RL_controller configs/rov_params.yaml "
                   "inertia_diag + payloads + the 0.12 kg m^2 added-inertia placeholder]")
    pr.write_json("free_decay.json", doc)
    pr.results["free_decay"] = {ax: f.get("period_s") for ax, f in doc["axes"].items()}
    return doc


# ============================================================================= sequence D — ESC from-rest latency
def _esc_step(tel: Telemetry, st_pre: StepRec, st_on: StepRec, motor: int, axis: str, sign: int, on_us: int,
              on_s: float) -> dict:
    col = _axis_index(axis)
    base = _mean(e[col] for e in tel.att_between(st_on.t_start - 0.3, st_on.t_start)) or 0.0
    win = tel.att_between(st_on.t_start, st_on.t_start + on_s + 0.5)
    first = tel.first_servo(motor, st_on.t_start, st_on.t_start + on_s, on_us, 10)
    out = {"t_cmd": st_on.t_start, "t_servo": (None if first is None else first[0]), "baseline_dps": base,
           "onset_s_from_cmd": None, "onset_s_from_servo": None, "peak_dps": None, "rise_10_90_s": None, "n": len(win)}
    if not win:
        return out
    rel = [(e[0], (e[col] - base) * sign) for e in win]          # positive = the expected direction
    peak_t, peak = max(rel, key=lambda x: x[1])
    out["peak_dps"] = peak
    onset = next((t for t, v in rel if v >= ESC_ONSET_DPS), None)
    if onset is not None:
        out["onset_s_from_cmd"] = onset - st_on.t_start
        if first is not None:
            out["onset_s_from_servo"] = onset - first[0]
    if peak > 0:
        t10 = next((t for t, v in rel if v >= 0.1 * peak), None)
        t90 = next((t for t, v in rel if v >= 0.9 * peak), None)
        if t10 is not None and t90 is not None:
            out["rise_10_90_s"] = t90 - t10
    return out


def run_seq_d(pr: ProbeRunner, motor: int = ESC_MOTOR_DEFAULT, levels=ESC_LEVELS_DEFAULT, repeats: int = ESC_REPEATS,
              on_s: float = ESC_ON_S, off_s: float = ESC_OFF_S, operator_timeout_s: float = 60.0) -> dict:
    """``motor`` 1..4 (default 1): the pulses yaw and translate the vehicle, nothing restores yaw, so the rate is a
    real steady state and no depth/tilt abort is in play. ``motor`` 5..8: the same ESC, but a lone vertical lifts
    the vehicle ~0.3 m per repeat at the 1600/1700 levels and tilts it toward the 25 deg abort [유도 — the dry run
    aborts on depth at the third repeat]; the operator re-positions at the prompt between levels."""
    ex = pr.ex
    axis_rate, sign = ex.expected[motor]
    pr.log(f"\n== D: ESC from-rest latency on T{motor} ({axis_rate} {sign:+d}), levels {list(levels)}, {repeats} repeats, "
           f"{on_s:.1f} s on / {off_s:.1f} s at the pre-level ==")
    doc = dict(pr.header())
    doc.update({"sequence": "D", "motor": motor, "axis": axis_rate, "expected_sign": sign, "on_s": on_s, "off_s": off_s,
                "onset_threshold_dps": ESC_ONSET_DPS, "cases": []})
    for il, (pre, on) in enumerate(levels):
        if il > 0:
            pr.wait_operator(f"D: re-position the vehicle mid-pool (tether slack) before the {pre}->{on} level",
                             operator_timeout_s, seq="D")
            pr.neutral("D", 2.0)
        pulses_on = list(NEUTRAL_PULSES)
        pulses_on[motor - 1] = int(on)
        pulses_pre = list(NEUTRAL_PULSES)
        pulses_pre[motor - 1] = int(pre)
        r_deg, p_deg = ex.predicted_tilt_deg(pulses_on)
        pred = max(abs(r_deg), abs(p_deg))
        warn = pred > 0.8 * pr.limits["tilt_abort_deg"]
        pr.log(f"  {pre}->{on}: predicted steady tilt {pred:.1f} deg at k {K_NOMINAL_NM_PER_RAD} [유도]"
               + (" — may trip the tilt abort; consider --esc-motor 1 (yaw) or --esc-on-s 0.8" if warn else ""))
        reps = []
        for rr in range(repeats):
            st_pre = pr.step("D", f"{pre}->{on}_r{rr + 1}_pre", pulses_pre, off_s, motor=motor, level=int(pre))
            st_on = pr.step("D", f"{pre}->{on}_r{rr + 1}_on", pulses_on, on_s, motor=motor, level=int(on))
            rec = _esc_step(pr.tel, st_pre, st_on, motor, axis_rate, sign, int(on), on_s)
            reps.append(rec)
            pr.log(f"    r{rr + 1}: onset {_fmt(rec['onset_s_from_cmd'], 3)} s from cmd / "
                   f"{_fmt(rec['onset_s_from_servo'], 3)} s from the servo report, peak {_fmt(rec['peak_dps'])} deg/s, "
                   f"rise {_fmt(rec['rise_10_90_s'], 3)} s")
        pr.neutral("D", off_s)
        case = {"pre_us": int(pre), "on_us": int(on), "predicted_tilt_deg": pred, "repeats": reps,
                "delay_cmd_mean_s": _mean(r["onset_s_from_cmd"] for r in reps),
                "delay_servo_mean_s": _mean(r["onset_s_from_servo"] for r in reps),
                "rise_mean_s": _mean(r["rise_10_90_s"] for r in reps),
                "peak_mean_dps": _mean(r["peak_dps"] for r in reps)}
        doc["cases"].append(case)
    rest = next((c for c in doc["cases"] if c["pre_us"] == PWM_NEUTRAL_US), None)
    spin = [c for c in doc["cases"] if c["pre_us"] != PWM_NEUTRAL_US]
    pen = None
    if rest is not None and spin:
        ref = _mean(c["delay_servo_mean_s"] for c in spin)
        if rest["delay_servo_mean_s"] is not None and ref is not None:
            pen = rest["delay_servo_mean_s"] - ref
    doc["from_rest_penalty_s"] = pen
    doc["penalty_threshold_s"] = ESC_PENALTY_S
    doc["verdict"] = (None if pen is None else ("BLHeli_S start-up penalty REAL" if pen >= ESC_PENALTY_S
                                                 else "no from-rest penalty above threshold"))
    doc["note"] = ("delay from the SERVO_OUTPUT_RAW report removes the link latency; from the command includes it. "
                   "The step on a vertical thruster rides on the restoring moment, so 'steady rate' is the PEAK rate")
    pr.log(f"  from-rest penalty {_fmt(pen, 3)} s -> {doc['verdict']}")
    pr.write_json("esc_latency.json", doc)
    pr.results["D"] = {"from_rest_penalty_s": pen, "verdict": doc["verdict"]}
    return doc


# ============================================================================= sequence E — rest trim
def run_seq_e(pr: ProbeRunner, seconds: float = REST_S) -> dict:
    pr.log(f"\n== E: rest trim, {seconds:.0f} s of neutral pulses, jaw EMPTY ==")
    st = pr.step("E", "rest", NEUTRAL_PULSES, seconds)
    att = pr.tel.att_between(st.t_start + 2.0, st.t_end)
    dep = pr.tel.depth_between(st.t_start + 2.0, st.t_end)
    bat = pr.tel.batt_between(st.t_start + 2.0, st.t_end)
    slope, _, nd = _linfit([e[0] for e in dep], [e[1] for e in dep])
    if slope is None:
        weight = "unknown"
    elif slope > REST_DRIFT_MIN_MPS:
        weight = "heavy (sinks)"
    elif slope < -REST_DRIFT_MIN_MPS:
        weight = "buoyant (rises)"
    else:
        weight = "neutral within resolution"
    doc = dict(pr.header())
    doc.update({"sequence": "E", "seconds": seconds, "n_att": len(att), "n_depth": nd,
                "roll_mean_deg": _mean(e[1] for e in att), "roll_std_deg": _std(e[1] for e in att),
                "pitch_mean_deg": _mean(e[2] for e in att), "pitch_std_deg": _std(e[2] for e in att),
                "yaw_mean_deg": _mean(e[3] for e in att), "yaw_drift_deg": (att[-1][3] - att[0][3]) if att else None,
                "depth_start_m": (dep[0][1] if dep else None), "depth_end_m": (dep[-1][1] if dep else None),
                "depth_rate_mps": slope, "net_weight_sign": weight,
                "net_weight_n_est_down_positive": (None if slope is None else slope * D_HEAVE_NSM),
                "batt_a_mean": _mean(e[2] for e in bat), "batt_a_std": _std(e[2] for e in bat),
                "batt_v_mean": _mean(e[1] for e in bat),
                "convention": "NED: +pitch = nose UP, +roll = starboard down; depth positive down",
                "note": f"net weight = drift x {D_HEAVE_NSM} N*s/m [예측]; drift below {REST_DRIFT_MIN_MPS} m/s is "
                        f"'neutral' [예측 Bar30 resolution]; jaw-empty is the operator's statement (operator_notes)",
                "operator_notes": pr.notes})
    pr.log(f"  roll {_fmt(doc['roll_mean_deg'])} +- {_fmt(doc['roll_std_deg'], 2)} deg, pitch {_fmt(doc['pitch_mean_deg'])} "
           f"+- {_fmt(doc['pitch_std_deg'], 2)} deg (NED, +nose up), depth {_fmt(doc['depth_start_m'], 3)} -> "
           f"{_fmt(doc['depth_end_m'], 3)} m ({_fmt(None if slope is None else slope * 1000, 1)} mm/s: {weight}), "
           f"{_fmt(doc['batt_a_mean'], 2)} A hotel, {_fmt(doc['batt_v_mean'], 2)} V")
    pr.write_json("rest_trim.json", doc)
    pr.results["E"] = {"pitch_mean_deg": doc["pitch_mean_deg"], "net_weight_sign": weight}
    return doc


# ============================================================================= orchestration
def parse_seq(s: str) -> list:
    letters = [c for c in str(s or "A").upper() if c.isalpha()]
    bad = [c for c in letters if c not in "ABCDE"]
    if bad:
        raise ValueError(f"--seq: unknown sequence(s) {bad} (A..E)")
    out = []
    for c in letters:
        if c not in out:
            out.append(c)
    return out


def run_all(pr: ProbeRunner, seqs: list, *, free_decay_s: float = 0.0, esc_motor: int = ESC_MOTOR_DEFAULT,
            esc_levels=ESC_LEVELS_DEFAULT,
            esc_on_s: float = ESC_ON_S, stiffness_nm: float = STIFF_TORQUE_NM, nv_passes: int = 2,
            rest_s: float = REST_S, argv_record: dict | None = None, free_decay_before=None, nv_hold_us="0") -> int:
    """preflight -> engage -> the sequences in order -> neutral. Exit codes: 0 done (and A all_pass when A ran),
    1 done with a non-PASS verdict / a finding, 2 refused before any pulse, 3 aborted mid-run, 130 Ctrl-C."""
    code = 0
    try:
        try:
            pr.preflight()
        except ProbeAbort as e:
            pr.abort_reason = f"REFUSED: {e}"
            pr.log(f"REFUSED: {e}")
            return 2
        pr.engage()
        for s in seqs:
            if s == "A":
                doc = run_seq_a(pr)
                if not doc["all_pass"]:
                    code = 1
            elif s == "B":
                doc = run_seq_b(pr, passes=nv_passes, hold_us=nv_hold_us)
                if doc["pooled_regression"].get("findings"):
                    code = max(code, 1)
            elif s == "C":
                if free_decay_s > 0:
                    run_free_decay(pr, free_decay_s, before=free_decay_before)
                else:
                    run_seq_c(pr, torque_nm=stiffness_nm)
            elif s == "D":
                run_seq_d(pr, motor=esc_motor, levels=esc_levels, on_s=esc_on_s)
            elif s == "E":
                run_seq_e(pr, rest_s)
        pr.neutral("exit", 0.5, "exit")
        return code
    except ProbeAbort as e:
        pr.abort_reason = str(e)
        pr.log(f"\nABORT: {e} — neutral sent")
        return 3
    except KeyboardInterrupt:
        pr.abort_reason = "operator Ctrl-C"
        pr.log("\nABORT: operator Ctrl-C — neutral sent")
        return 130
    except Exception as e:                                       # noqa: BLE001
        pr.abort_reason = f"{type(e).__name__}: {e}"
        pr.log(f"\nABORT on exception: {pr.abort_reason} — neutral sent")
        raise
    finally:
        pr.neutral_burst()
        try:
            pr.finish(code if pr.abort_reason is None else (2 if str(pr.abort_reason).startswith("REFUSED") else 3),
                      argv_record)
        except Exception as e:                                   # noqa: BLE001
            pr.log(f"could not write probe_run.json: {e}")
        pr.link.close()


def build_dry_run(export: ExportModel, clock, **veh_kw) -> Link:
    return Link(SyntheticVehicle(export, clock, **veh_kw), target_sys=int(veh_kw.get("sysid", 1)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="A", help="sequences to run, in order: A,B,C,D,E (default A)")
    ap.add_argument("--policy-dir", default=str(DEFAULT_POLICY_DIR))
    ap.add_argument("--mav", default="udpin:0.0.0.0:14550", help="the vehicle link (BlueOS 'GCS Client Link'; close QGC)")
    ap.add_argument("--cmd-sysid", type=int, default=STATION_SYSID, help="our sysid = SYSID_MYGCS (default 255)")
    ap.add_argument("--target-sysid", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data"), help="dated run root (rov_gui/runstore.py)")
    ap.add_argument("--note", default="", help="operator notes written into every artefact (jaw empty? pack? ballast?)")
    ap.add_argument("--gyro", choices=("attitude", "raw_imu"), default="attitude")
    ap.add_argument("--free-decay", type=float, default=0.0, metavar="S", help="with C: record N s of free decay instead")
    ap.add_argument("--stiffness-nm", type=float, default=STIFF_TORQUE_NM)
    ap.add_argument("--esc-motor", type=int, default=ESC_MOTOR_DEFAULT,
                    help="D: the motor stepped (default 1 = horizontal/yaw; 5 = the spec's vertical/roll, which lifts "
                         "the vehicle out of the depth band within three repeats [유도 dry run])")
    ap.add_argument("--esc-levels", default=",".join(f"{a}:{b}" for a, b in ESC_LEVELS_DEFAULT),
                    help="pre:on pulse pairs for D (default %(default)s)")
    ap.add_argument("--esc-on-s", type=float, default=ESC_ON_S)
    ap.add_argument("--nv-passes", type=int, default=2, help="altitude passes of B (default 2)")
    ap.add_argument("--nv-hold-us", default="0", help="B: depth-hold pattern level [us] under the common mode: 0 = none "
                                                      "(default, ballast the vehicle near neutral first — E tells), 41 = "
                                                      "the 2026-10 mixer holds [측정], auto = calibrated to today's ballast")
    ap.add_argument("--rest-s", type=float, default=REST_S)
    for k, v in LIMITS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v, help=f"abort limit (default {v})")
    ap.add_argument("--dry-run", action="store_true", help="synthetic vehicle, fast clock, no hardware")
    ap.add_argument("--dry-run-realtime", action="store_true", help="dry run at wall-clock speed")
    ap.add_argument("--swap", default="", help="dry run: miswire two motors, e.g. 5,8")
    ap.add_argument("--reverse", default="", help="dry run: reversed physical thrusters, e.g. 3")
    ap.add_argument("--dry-k", type=float, default=K_NOMINAL_NM_PER_RAD, help="dry run: stiffness N*m/rad")
    ap.add_argument("--dry-tilt-at", type=float, default=2.0, help="dry run free decay: release time [s] of a 15 deg tilt")
    ap.add_argument("--dry-weight-n", type=float, default=0.02, help="dry run: net weight [N], + = sinks")
    a = ap.parse_args(argv)
    try:
        seqs = parse_seq(a.seq)
        levels = tuple((int(x.split(":")[0]), int(x.split(":")[1])) for x in a.esc_levels.split(",") if x.strip())
    except ValueError as e:
        ap.error(str(e))
    if not (1 <= a.esc_motor <= N_MOTORS):
        ap.error("--esc-motor must be 1..8")
    limits = {k: float(getattr(a, k)) for k in LIMITS}
    export = ExportModel(a.policy_dir)
    argv_record = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in vars(a).items()}

    if a.dry_run:
        clock = time.monotonic if a.dry_run_realtime else FakeClock()
        sleep = time.sleep if a.dry_run_realtime else clock.sleep
        kw = {"sysid": a.target_sysid, "mygcs": a.cmd_sysid, "k_nm_per_rad": a.dry_k, "net_weight_n": a.dry_weight_n}
        if a.swap:
            kw["swap"] = tuple(int(x) for x in a.swap.split(","))
        if a.reverse:
            kw["reverse"] = tuple(int(x) for x in a.reverse.split(","))
        link = build_dry_run(export, clock, **kw)
        out = run_dir(a.out, kind="probedry", join=False)
        operator = lambda prompt, timeout: True                  # noqa: E731 — the dry-run operator is always ready
        print(f"DRY RUN: synthetic vehicle (k {a.dry_k} N*m/rad, lag 0.25 s, ESC start 0.12 s"
              f"{', swap ' + a.swap if a.swap else ''}{', reverse ' + a.reverse if a.reverse else ''}) -> {out}")
        pr = ProbeRunner(link, export, out, dry_run=True, cmd_sysid=a.cmd_sysid, clock=clock, sleep=sleep,
                         limits=limits, operator=operator, gyro_source=a.gyro, notes=a.note)
        veh = link.master

        def free_decay_before():
            # the synthetic operator: a 15 deg pitch + 10 deg roll tilt released --dry-tilt-at s into the recording
            veh.tilt_impulse = (clock() + a.dry_tilt_at, 10.0, 15.0)
            veh._impulsed = False
    else:
        os.environ.setdefault("MAVLINK20", "1")
        from pymavlink import mavutil
        master = mavutil.mavlink_connection(a.mav, source_system=a.cmd_sysid, source_component=190)
        link = Link(master, target_sys=a.target_sysid)
        out = run_dir(a.out, kind="probe", join=False)
        print(f"link {a.mav} (sysid {a.cmd_sysid} -> vehicle {a.target_sysid}); artefacts -> {out}")
        print("the vehicle must be ARMED in MANUAL, props ON, mid-pool, tether slack, nobody in the water")
        pr = ProbeRunner(link, export, out, dry_run=False, cmd_sysid=a.cmd_sysid, limits=limits,
                         gyro_source=a.gyro, notes=a.note)
        free_decay_before = None
    code = run_all(pr, seqs, free_decay_s=a.free_decay, esc_motor=a.esc_motor, esc_levels=levels, esc_on_s=a.esc_on_s,
                   stiffness_nm=a.stiffness_nm, nv_passes=a.nv_passes, rest_s=a.rest_s, argv_record=argv_record,
                   free_decay_before=free_decay_before, nv_hold_us=a.nv_hold_us)
    print(f"exit {code}: {'ok' if code == 0 else pr.abort_reason or 'see the artefacts'} -> {out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
