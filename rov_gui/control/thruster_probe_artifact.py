#!/usr/bin/env python3
"""thruster_probe_artifact.py — the per-thruster sign-probe artefact: its schema, its constants, and the check an
engage gate runs on it before any rl_pwm pulse leaves (2026-10-10).

WHAT THE ARTEFACT IS. ``rov_gui/tools/thruster_probe.py --seq A`` (README §10.3 item 4 of RL_controller, the pool
"probe block" of the 2026-10-10 fix plan) gives each thruster +100 us for 1 s and -100 us for 1 s, armed in MANUAL,
mid-pool, through the SAME transport the policy flies on (RC_CHANNELS_OVERRIDE ch 9..16 -> rl_pwm_override.lua), and
judges each motor from the mean body rate on its DOMINANT axis — computed from the export's B_flu, never hardcoded —
over 0.15..0.55 s after the vehicle's own SERVO_OUTPUT_RAW first showed the pulse, minus the mean rate of the 0.5 s
before the command (yaw has no restoring moment: the previous pulse's rotation is still there 3 s later). It writes
``data/<YYYYMMDD>/<MMDD_HHMMSS>_probe/thruster_sign_probe.json`` with exactly the keys below (schema 1).

PURE DATA ON PURPOSE. No numpy, no MAVLink, no Qt: the gate that refuses to fly a policy without a passing probe must
be importable from anywhere (the command sink, the worker, a CLI) and testable without either. The tool imports the
same ``expected_axis_sign`` / ``verdict`` below, so what the tool wrote and what the gate recomputes cannot drift.

WHAT ``load_and_check`` REFUSES (in this order, first reason wins): a file that is missing / not JSON / not this
schema or tool; a DRY-RUN artefact unless the caller says ``allow_dry_run=True`` — the synthetic vehicle of
``--dry-run`` is not a measurement and must never stand in for one at the engage gate (that default is the safety
decision of this module; a test that exercises the dry-run path passes the flag); a motors list that is not motors
1..8 in order; a verdict that does not follow from the numbers stored beside it (a hand-edited "PASS" is caught);
any motor not PASS / ``all_pass`` false; an export sigma or a vehicle MOT_1..8_DIRECTION different from what the
caller is about to fly with; an artefact older than ``max_age_s`` when one is given.

Schema (every key present; floats are plain JSON numbers, never NaN):

    {"schema": 1, "written": "<ISO 8601 local time>", "tool": "rov_gui/tools/thruster_probe.py", "dry_run": bool,
     "vehicle": {"sysid": int, "fw": "<AUTOPILOT_VERSION a.b.c or ''>", "mot_direction": [8 ints, read at probe time]},
     "export": {"policy_dir": "<path>", "sigma": [8 floats], "B_flu": [[6 x 8]], "obs_spec_sha1": "<sha1>"},
     "pulse_us": 100, "on_s": 1.0, "off_s": 3.0,
     "motors": [{"motor": 1..8, "axis": "p|q|r", "expected_sign_pos": +1|-1,
                 "mean_rate_pos_dps": float, "mean_rate_neg_dps": float,
                 "verdict": "PASS|WEAK|FAIL", "n_samples_pos": int, "n_samples_neg": int}, ... x 8],
     "all_pass": bool, "operator_notes": ""}
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Iterable, Sequence

SCHEMA = 1
TOOL = "rov_gui/tools/thruster_probe.py"
FILENAME = "thruster_sign_probe.json"
N_MOTORS = 8
#: The probe pulse: +-100 us = +-0.25 throttle, the first-water cap of RL_controller run_policy.py --pwm-cap, just
#: above the +-25 us ESC deadband [스펙 BlueRobotics Basic ESC] and the +-30 us of the exported T200 model (deadband
#: 0.075 x 400 us). +100 us ~ 4.2 N, -100 us ~ 3.4 N at voltage_scale 0.72 [유도 pwm10_s1 obs_spec t200 curve].
PULSE_US = 100
ON_S = 1.0
OFF_S = 3.0
#: The judging window after the vehicle's first non-neutral SERVO_OUTPUT_RAW report of the pulse: late enough for the
#: ESC start-up (110-130 ms from rest [스펙 BR forum / BLHeli_S manual]) and the thrust lag, early enough that the
#: hydrostatic restoring moment (k 4-11 N*m/rad [측정 REPORT.md §1]) has not reversed the rate [예측].
WINDOW_S = (0.15, 0.55)
#: |mean rate| that counts as a clear response: ~0.9 N*m on I_roll 0.5-1.05 kg m^2 gives 20-40 deg/s after 0.4 s with
#: no damping; ATTITUDE gyro noise is ~1 deg/s [예측 — unmeasured on this vehicle, the first artefact will say].
RATE_THRESHOLD_DPS = 5.0
#: Fewer gyro samples than this inside the window is not a verdict either way (ATTITUDE at the default 4 Hz gives 1-2;
#: the tool asks for 25 Hz, so ~10 are expected).
MIN_SAMPLES = 3
AXES = ("p", "q", "r")
VERDICT_PASS, VERDICT_WEAK, VERDICT_FAIL = "PASS", "WEAK", "FAIL"
VERDICTS = (VERDICT_PASS, VERDICT_WEAK, VERDICT_FAIL)
#: Body wrench rows [X, Y, Z, K, M, N]: FLU -> FRD/NED sign per row (rl_pwm.FLU_TO_FRD, deploy/pwm.py).
FLU_TO_FRD = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)


# ----------------------------------------------------------------------------- the expected response
def expected_axis_sign(B_flu: Sequence[Sequence[float]], column: int) -> tuple:
    """(axis, sign) a POSITIVE pulse on thruster ``column`` (0-based) should produce: the FRD moment row of largest
    magnitude in that B_flu column (K -> "p", M -> "q", N -> "r") and the sign of the body rate that moment starts.

    Only the geometry decides: positive thrust along the column's direction, so the magnitude of the thrust (the T200
    curve, the voltage) cancels. A column with no moment at all raises — it cannot be judged by a gyro."""
    if len(B_flu) != 6:
        raise ValueError(f"B_flu has {len(B_flu)} rows, expected 6")
    mom = []
    for row in (3, 4, 5):
        v = float(B_flu[row][column]) * FLU_TO_FRD[row]
        if not math.isfinite(v):
            raise ValueError(f"B_flu[{row}][{column}] is not finite")
        mom.append(v)
    k = max(range(3), key=lambda i: abs(mom[i]))
    if abs(mom[k]) <= 1e-9:
        raise ValueError(f"thruster {column + 1} produces no body moment in B_flu — a gyro cannot judge it")
    return AXES[k], (1 if mom[k] > 0.0 else -1)


def expected_table(B_flu) -> list:
    """[(motor 1..8, axis, sign)] for every column."""
    n = len(B_flu[0])
    return [(m + 1,) + expected_axis_sign(B_flu, m) for m in range(n)]


def verdict(expected_sign_pos: int, mean_rate_pos_dps, mean_rate_neg_dps, n_samples_pos: int, n_samples_neg: int,
            threshold_dps: float = RATE_THRESHOLD_DPS, min_samples: int = MIN_SAMPLES) -> str:
    """PASS: both pulses answered with the expected sign AND |mean rate| >= threshold, with enough samples.
    WEAK: both signs agree but a rate is below threshold (or a window is thin) — not proof, not a contradiction.
    FAIL: a sign disagrees, a rate is zero, a window is empty, or a number is missing."""
    s = 1 if int(expected_sign_pos) >= 0 else -1
    try:
        mp, mn = float(mean_rate_pos_dps), float(mean_rate_neg_dps)
        n_p, n_n = int(n_samples_pos), int(n_samples_neg)
    except (TypeError, ValueError):
        return VERDICT_FAIL
    if not (math.isfinite(mp) and math.isfinite(mn)) or n_p <= 0 or n_n <= 0:
        return VERDICT_FAIL
    if mp * s <= 0.0 or mn * s >= 0.0:
        return VERDICT_FAIL
    if abs(mp) >= threshold_dps and abs(mn) >= threshold_dps and n_p >= min_samples and n_n >= min_samples:
        return VERDICT_PASS
    return VERDICT_WEAK


def motor_record(motor: int, axis: str, expected_sign_pos: int, mean_rate_pos_dps, mean_rate_neg_dps,
                 n_samples_pos: int, n_samples_neg: int, **extra) -> dict:
    """One ``motors[]`` entry, verdict computed here and nowhere else. ``extra`` keys (diagnostics) ride along."""
    mp = None if mean_rate_pos_dps is None else float(mean_rate_pos_dps)
    mn = None if mean_rate_neg_dps is None else float(mean_rate_neg_dps)
    rec = {"motor": int(motor), "axis": str(axis), "expected_sign_pos": (1 if int(expected_sign_pos) >= 0 else -1),
           "mean_rate_pos_dps": (0.0 if mp is None or not math.isfinite(mp) else mp),
           "mean_rate_neg_dps": (0.0 if mn is None or not math.isfinite(mn) else mn),
           "n_samples_pos": int(n_samples_pos), "n_samples_neg": int(n_samples_neg)}
    rec["verdict"] = verdict(rec["expected_sign_pos"], rec["mean_rate_pos_dps"], rec["mean_rate_neg_dps"],
                             rec["n_samples_pos"], rec["n_samples_neg"])
    rec.update(extra)
    return rec


def build(*, dry_run: bool, sysid: int, fw: str, mot_direction: Iterable[int], policy_dir: str,
          sigma: Iterable[float], B_flu, obs_spec_sha1: str, motors: list, operator_notes: str = "",
          written: str | None = None) -> dict:
    """The artefact dict (schema 1). ``motors`` are :func:`motor_record` dicts for motors 1..8 in order."""
    motors = [dict(m) for m in motors]
    return {"schema": SCHEMA, "written": written or time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool": TOOL,
            "dry_run": bool(dry_run),
            "vehicle": {"sysid": int(sysid), "fw": str(fw or ""), "mot_direction": [int(v) for v in mot_direction]},
            "export": {"policy_dir": str(policy_dir), "sigma": [float(v) for v in sigma],
                       "B_flu": [[float(v) for v in row] for row in B_flu], "obs_spec_sha1": str(obs_spec_sha1)},
            "pulse_us": PULSE_US, "on_s": ON_S, "off_s": OFF_S, "motors": motors,
            "all_pass": bool(len(motors) == N_MOTORS and all(m.get("verdict") == VERDICT_PASS for m in motors)),
            "operator_notes": str(operator_notes or "")}


# ----------------------------------------------------------------------------- the gate's check
def _written_age_s(written, now: float | None = None) -> float | None:
    """Seconds since ``written`` (the ISO string :func:`build` writes), None when unparseable."""
    if not isinstance(written, str) or not written:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            t = time.strptime(written, fmt)
        except ValueError:
            continue
        try:
            stamp = time.mktime(t)
        except (OverflowError, ValueError):
            return None
        return (time.time() if now is None else float(now)) - stamp
    return None


def check(doc, expected_sigma: Iterable[float], expected_mot_direction: Iterable[int], *,
          allow_dry_run: bool = False, max_age_s: float | None = None, now: float | None = None) -> tuple:
    """(ok, reason) for an artefact already loaded as a dict. ``reason`` is "ok" or the FIRST thing that fails, in
    words an operator can act on."""
    if not isinstance(doc, dict):
        return False, "artefact is not a JSON object"
    if doc.get("schema") != SCHEMA:
        return False, f"schema {doc.get('schema')!r} != {SCHEMA}"
    if doc.get("tool") != TOOL:
        return False, f"tool {doc.get('tool')!r} != {TOOL!r}"
    if doc.get("dry_run", True) and not allow_dry_run:
        return False, "dry_run artefact (synthetic vehicle) — not a measurement; run the probe on the vehicle"
    if doc.get("dry_run", True) not in (True, False):
        return False, "dry_run is not a boolean"
    motors = doc.get("motors")
    if not isinstance(motors, list) or len(motors) != N_MOTORS:
        return False, f"motors must list all {N_MOTORS} thrusters (got {0 if not isinstance(motors, list) else len(motors)})"
    for i, m in enumerate(motors):
        if not isinstance(m, dict):
            return False, f"motors[{i}] is not an object"
        if m.get("motor") != i + 1:
            return False, f"motors[{i}].motor is {m.get('motor')!r}, expected {i + 1} (motors 1..8 in order)"
        if m.get("axis") not in AXES:
            return False, f"motor {i + 1}: axis {m.get('axis')!r} not in {AXES}"
        if m.get("expected_sign_pos") not in (1, -1):
            return False, f"motor {i + 1}: expected_sign_pos {m.get('expected_sign_pos')!r} not +1/-1"
        v = m.get("verdict")
        if v not in VERDICTS:
            return False, f"motor {i + 1}: verdict {v!r} not in {VERDICTS}"
        recomputed = verdict(m["expected_sign_pos"], m.get("mean_rate_pos_dps"), m.get("mean_rate_neg_dps"),
                             m.get("n_samples_pos", 0), m.get("n_samples_neg", 0))
        if recomputed != v:
            return False, (f"motor {i + 1}: stored verdict {v} does not follow from its numbers "
                           f"(pos {m.get('mean_rate_pos_dps')} / neg {m.get('mean_rate_neg_dps')} deg/s, "
                           f"n {m.get('n_samples_pos')}/{m.get('n_samples_neg')} -> {recomputed})")
    bad = [m for m in motors if m["verdict"] != VERDICT_PASS]
    if bad:
        detail = ", ".join("T%d %s" % (m["motor"], m["verdict"]) for m in bad)
        return False, (f"thruster(s) {[m['motor'] for m in bad]} did not PASS the sign probe ({detail}) — fix the "
                       f"export's sigma / B or the wiring, re-probe, never fly until all_pass")
    if doc.get("all_pass") is not True:
        return False, "all_pass is not true"
    exp = doc.get("export") if isinstance(doc.get("export"), dict) else {}
    sig = exp.get("sigma")
    want = [float(v) for v in expected_sigma]
    if not isinstance(sig, list) or len(sig) != len(want):
        return False, f"export.sigma missing or not {len(want)} values"
    try:
        diff = [abs(float(a) - b) for a, b in zip(sig, want)]
    except (TypeError, ValueError):
        return False, "export.sigma is not numeric"
    if any(d > 1e-6 for d in diff):
        return False, f"export.sigma {sig} != the policy's sigma {want}: probed with a different thrust-sign model"
    B = exp.get("B_flu")
    if isinstance(B, list) and len(B) == 6:
        try:
            for i, m in enumerate(motors):
                ax, sg = expected_axis_sign(B, i)
                if ax != m["axis"] or sg != m["expected_sign_pos"]:
                    return False, (f"motor {i + 1}: stored axis/sign {m['axis']}{m['expected_sign_pos']:+d} != "
                                   f"{ax}{sg:+d} recomputed from the stored B_flu")
        except (ValueError, TypeError, IndexError) as e:
            return False, f"export.B_flu unusable: {e}"
    else:
        return False, "export.B_flu missing or not 6 rows"
    veh = doc.get("vehicle") if isinstance(doc.get("vehicle"), dict) else {}
    mot = veh.get("mot_direction")
    want_mot = [int(v) for v in expected_mot_direction]
    if not isinstance(mot, list) or len(mot) != len(want_mot):
        return False, f"vehicle.mot_direction missing or not {len(want_mot)} values"
    try:
        got_mot = [int(round(float(v))) for v in mot]
    except (TypeError, ValueError):
        return False, "vehicle.mot_direction is not numeric"
    if got_mot != want_mot:
        return False, (f"vehicle.mot_direction {got_mot} at probe time != MOT_1..8_DIRECTION the policy expects "
                       f"{want_mot}: the vehicle was probed in another configuration")
    if doc.get("pulse_us") != PULSE_US:
        return False, f"pulse_us {doc.get('pulse_us')!r} != {PULSE_US}"
    if max_age_s is not None:
        age = _written_age_s(doc.get("written"), now)
        if age is None:
            return False, f"written {doc.get('written')!r} unreadable, cannot judge its age"
        if age > float(max_age_s):
            return False, f"artefact is {age / 3600.0:.1f} h old (> {float(max_age_s) / 3600.0:.1f} h)"
    return True, "ok"


def load_and_check(path, expected_sigma: Iterable[float], expected_mot_direction: Iterable[int], *,
                   allow_dry_run: bool = False, max_age_s: float | None = None) -> tuple:
    """(ok, reason): read ``path`` and run :func:`check`. Never raises on a bad file — a gate wants a sentence."""
    p = str(path or "")
    if not p or not os.path.isfile(p):
        return False, f"no sign-probe artefact at {p!r} (run rov_gui/tools/thruster_probe.py --seq A)"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        return False, f"{p}: not readable JSON ({e})"
    ok, why = check(doc, expected_sigma, expected_mot_direction, allow_dry_run=allow_dry_run, max_age_s=max_age_s)
    return ok, (why if ok else f"{os.path.basename(p)}: {why}")


def summary(doc) -> str:
    """One line for a log: ``T1 r- PASS (+12.3/-9.8) | ...``."""
    if not isinstance(doc, dict) or not isinstance(doc.get("motors"), list):
        return "no motors"
    parts = []
    for m in doc["motors"]:
        try:
            parts.append(f"T{m['motor']} {m['axis']}{m['expected_sign_pos']:+d} {m['verdict']} "
                         f"({m['mean_rate_pos_dps']:+.1f}/{m['mean_rate_neg_dps']:+.1f})")
        except (KeyError, TypeError, ValueError):
            parts.append("?")
    return " | ".join(parts)


__all__ = ["SCHEMA", "TOOL", "FILENAME", "N_MOTORS", "PULSE_US", "ON_S", "OFF_S", "WINDOW_S", "RATE_THRESHOLD_DPS",
           "MIN_SAMPLES", "AXES", "VERDICTS", "VERDICT_PASS", "VERDICT_WEAK", "VERDICT_FAIL", "FLU_TO_FRD",
           "expected_axis_sign", "expected_table", "verdict", "motor_record", "build", "check", "load_and_check",
           "summary"]
