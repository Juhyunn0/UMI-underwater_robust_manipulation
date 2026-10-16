#!/usr/bin/env python3
"""rl_pwm_interlock_replay.py — replay the recorded pwm engagements through the station's per-thruster interlocks.

    cd <repo> && ~/miniforge3/envs/robust/bin/python -m rov_gui.tools.rl_pwm_interlock_replay
        [--data data] [--days 20261001 20261003 20261005] [--null-us 100 120]
        [--out data/20261010/1010_interlock_replay_offline/report.txt] [--config config/hw_mpc.yaml]
        [csv ...]

WHAT IT DOES (2026-10-10 b, REPORT.md §3 step 5 "replay-tested on the 14 engagements"). Every mpc_*.csv under the
named day folders whose ``mode`` column is ``rl_pwm`` or ``rl_pwm_7d`` (the 14 recorded pwm engagements of
2026-10-01 / 03 / 05) — and, as the control group, every other LOW's file there (mpc_tuned / rl / pid: mixer-allocated,
which must never trip) — is fed tick by tick through control/rl_pwm.PwmInterlocks, the SAME class workers.
_runtime_fault runs on the vehicle, with the thresholds of ``engage.per_thruster`` (the shipped config, or the
defaults) and the null monitor at each of ``--null-us`` (by default the shipped null_us plus 100 and 120 us: REPORT §3
step 5 names 100-150 us, pwm10_s1's squeeze was +82..+107 us, and the shipped value is 80 us since the first replay
of 2026-10-10 showed 120 us / 1.0 s missing it). Per engagement it prints which interlock would have tripped FIRST and
when (seconds after the first non-neutral pulse), the peaks each interlock saw, and when the policy-side tag gate
(rl_pwm.obs_guard -> HOLD pulses, control/rl_pwm.PwmObsGuard) would first have switched the policy out.

WHAT THE RECORD CAN AND CANNOT SAY. The interlocks' inputs are taken from the CSV as the station logged them:
roll_deg / pitch_deg (ATTITUDE), batt_v / batt_a (BATTERY_STATUS, ~1 Hz, held), pwm1..8 and n_tags / tag_age_s.
  * the pulses are the vehicle's SERVO_OUTPUT_RAW report (pwm1..8), ~0.14 s behind the sent frame — not the sent
    pulses the live interlock reads (CSV i_pred_a / null_* exist only from schema 19 on, and no schema-19 pwm row was
    ever flown). A row with EMPTY pwm cells (no new report yet) carries the previous row's pulses forward, as the
    live interlock judges the last frame that left until the next one;
  * the attitude rate is the vehicle's gyro (``*_ekf_inputs.jsonl`` imu records ``pqr``, aligned through the
    header's ``t0_csv``) where that sidecar exists (10-03, 10-05) and the FINITE DIFFERENCE of the logged roll /
    pitch angles (np.gradient, no smoothing) where it does not (10-01) — the column ``rate`` says which. On
    1003_180835 the two agree to 0.96 correlation and within 3 deg/s at the peaks [측정: this tool's development
    check, not persisted];
  * the cap per file is meta ``controller.pwm_cap`` (0.25 on 10-01, 1.0 after) and the pinned item judges the
    vehicle's report against it;
  * rows after the disengage (``engaged`` 0) are skipped — the live interlock never saw them either;
  * AGE (2026-10-15) is judged only on a file that RECORDED the attitude's vehicle-clock age: meta schema_version >=
    21 with the CSV column ``att_age_v_s`` (an empty / nan cell there is "unknown", which the live item counts as
    over). Every earlier file never logged it, so the item is skipped (``age_judged`` false) — the 10-13 ages exist
    only as the offline floor fit of RL_controller/logs/eval/pool_trips_1013_rootcause (its fix-design replay).
Nothing here is a measurement of what the interlocks WILL do on the vehicle: it is the recorded signals judged by the
code that will judge the live ones. Numbers in the report are [측정: the named CSV] through [예측] thresholds.

The report goes to ``--out`` (a NEW file in a new folder: data folders are append-only; an existing report is never
overwritten — a stamped sibling is written instead). The table is also printed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rov_gui.control.rl_pwm import (GUARD_HOLD, PER_THRUSTER_DEFAULTS, PwmInterlocks,  # noqa: E402
                                    PwmObsGuard, parse_obs_guard_block, parse_per_thruster_block)

PWM_MODES = ("rl_pwm", "rl_pwm_7d")
DEFAULT_DAYS = ("20261001", "20261003", "20261005")
DEFAULT_OUT = "data/20261010/1010_interlock_replay_offline/report.txt"
#: a vehicle-reported pulse this far from 1500 us is "non-neutral" (SERVO_OUTPUT_RAW is integer us; the mixer idles
#: at exactly 1500)
NONNEUTRAL_US = 2


# ----------------------------------------------------------------------------- loading
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def load_csv(path: str) -> dict:
    """The columns the interlocks need, as arrays (nan where the CSV has '' / 'nan'). ``mode`` is the first row's."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    d = {"path": path, "n": n, "mode": (rows[0].get("mode", "") if rows else ""), "rows": rows}
    d["t"] = np.array([_f(r.get("t")) for r in rows])
    d["roll"] = np.radians([_f(r.get("roll_deg")) for r in rows])
    d["pitch"] = np.radians([_f(r.get("pitch_deg")) for r in rows])          # CSV pitch_deg = -pitch_NED; |.| only
    d["pwm"] = np.array([[_f(r.get(f"pwm{i}")) for i in range(1, 9)] for r in rows]).reshape(n, 8)
    d["batt_v"] = np.array([_f(r.get("batt_v")) for r in rows])
    d["batt_a"] = np.array([_f(r.get("batt_a")) for r in rows])
    d["n_tags"] = np.array([_f(r.get("n_tags")) for r in rows])
    d["tag_age"] = np.array([_f(r.get("tag_age_s")) for r in rows])
    d["engaged"] = np.array([_f(r.get("engaged", 1)) for r in rows])
    d["p_xyz"] = np.array([[_f(r.get("px")), _f(r.get("py")), _f(r.get("pz"))] for r in rows]).reshape(n, 3)
    # schema 21: the attitude's vehicle-clock age (nan = unknown, or the column is absent: see has_att_age)
    d["has_att_age"] = bool(rows) and "att_age_v_s" in rows[0]
    d["att_age"] = np.array([_f(r.get("att_age_v_s")) for r in rows])
    return d


def load_meta(csv_path: str) -> dict:
    p = Path(csv_path).with_suffix("")
    mp = Path(str(p) + ".meta.json")
    if not mp.is_file():
        return {}
    try:
        with open(mp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def gyro_from_jsonl(csv_path: str, t_csv: np.ndarray):
    """(p, q) [rad/s] at the CSV rows from the EKF-input sidecar's imu records, aligned by the header's t0_csv;
    None when there is no sidecar / no imu records / no t0_csv."""
    jp = Path(str(Path(csv_path).with_suffix("")) + "_ekf_inputs.jsonl")
    if not jp.is_file():
        return None
    t0 = None
    ta, pq = [], []
    try:
        with open(jp, "r", encoding="utf-8") as fh:
            for line in fh:
                if '"imu"' not in line and '"header"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                k = rec.get("k")
                if k == "header":
                    t0 = rec.get("t0_csv")
                elif k == "imu":
                    pqr = rec.get("pqr")
                    if pqr is None or len(pqr) < 2 or rec.get("ta") is None:
                        continue
                    ta.append(float(rec["ta"]))
                    pq.append((float(pqr[0]), float(pqr[1])))
    except OSError:
        return None
    if t0 is None or len(ta) < 2:
        return None
    ta = np.asarray(ta) - float(t0)
    pq = np.asarray(pq)
    order = np.argsort(ta)
    ta, pq = ta[order], pq[order]
    p = np.interp(t_csv, ta, pq[:, 0])
    q = np.interp(t_csv, ta, pq[:, 1])
    return p, q


def rates_for(d: dict):
    """(p, q, source): the gyro where the sidecar has it, else the finite difference of the logged angles."""
    g = gyro_from_jsonl(d["path"], d["t"])
    if g is not None:
        return g[0], g[1], "gyro"
    t = d["t"]
    if len(t) < 2:
        return np.zeros_like(t), np.zeros_like(t), "fd"
    tt = t.copy()
    # a repeated stamp would make the gradient infinite: nudge by a microsecond
    for i in range(1, len(tt)):
        if tt[i] <= tt[i - 1]:
            tt[i] = tt[i - 1] + 1e-6
    p = np.gradient(d["roll"], tt)
    q = np.gradient(d["pitch"], tt)
    return p, q, "fd"


# ----------------------------------------------------------------------------- the replay
def thresholds_from_config(path) -> dict:
    """``engage.per_thruster`` of a hw_mpc.yaml through the station's own loader; the defaults when it cannot load."""
    if path:
        try:
            from rov_gui.control.geometry import MpcConfig
            cfg = MpcConfig.load(str(path))
            return parse_per_thruster_block(cfg.engage.get("per_thruster"))
        except Exception as e:                                   # noqa: BLE001
            print(f"warning: {path}: {type(e).__name__}: {e} — using PER_THRUSTER_DEFAULTS", file=sys.stderr)
    return dict(PER_THRUSTER_DEFAULTS)


def _first_pulse_index(d: dict):
    dev = np.nanmax(np.abs(d["pwm"] - 1500.0), axis=1) if d["n"] else np.array([])
    idx = np.where(dev > NONNEUTRAL_US)[0]
    return int(idx[0]) if idx.size else None


def replay_file(csv_path: str, thresholds: dict, null_us_values=(100.0, 120.0), obs_guard_cfg=None) -> dict:
    """One file through PwmInterlocks once per null threshold (+ the obs-guard tag gate). Returns the row dict the
    table and the tests read."""
    d = load_csv(csv_path)
    meta = load_meta(csv_path)
    ctrl = meta.get("controller") or {}
    run = meta.get("run") or {}
    cap = float(ctrl.get("pwm_cap", 1.0) or 1.0)
    hz = float(run.get("log_hz") or ctrl.get("ctrl_hz") or 20.0)
    p, q, rate_src = rates_for(d)
    i0 = _first_pulse_index(d)
    t_ref = float(d["t"][i0]) if i0 is not None else (float(d["t"][0]) if d["n"] else 0.0)
    live = d["engaged"] >= 0.5 if d["n"] else np.array([], dtype=bool)
    # AGE only where the file recorded it (schema >= 21 + the column); see the module docstring
    try:
        schema = int(meta.get("schema_version") or 0)
    except (TypeError, ValueError):
        schema = 0
    judge_age = bool(schema >= 21 and d["has_att_age"])
    out = {"file": csv_path, "mode": d["mode"], "cap": cap, "hz": hz, "rate_src": rate_src, "rows": int(d["n"]),
           "duration_s": (float(d["t"][-1] - d["t"][0]) if d["n"] else 0.0),
           "first_pulse_s": (None if i0 is None else float(d["t"][i0] - d["t"][0])),
           "t_ref": t_ref, "trips": {}, "peak": None, "hold_guard_s": None, "hold_guard_frac": None,
           "age_judged": judge_age}
    for nu in null_us_values:
        thr = dict(thresholds)
        thr["null_us"] = float(nu)
        ilk = PwmInterlocks(thr, hz, cap, judge_age=judge_age)
        trip = None
        pwm_us = None
        for i in range(d["n"]):
            if not live[i]:
                continue
            pw = d["pwm"][i]
            if not np.any(np.isnan(pw)):
                pwm_us = tuple(int(round(v)) for v in pw)       # else: the previous report, carried forward
            why = ilk.step(d["roll"][i], d["pitch"][i], p[i], q[i], batt_a=d["batt_a"][i], batt_v=d["batt_v"][i],
                           pwm_us=pwm_us, att_age_s=d["att_age"][i])
            if why:
                lt = ilk.last_trip or {}
                trip = {"name": lt.get("name"), "t_s": float(d["t"][i] - t_ref), "row": i, "why": why,
                        "values": lt.get("values")}
                break
        out["trips"][float(nu)] = trip
        out["peak"] = dict(ilk.peak)          # the last pass's peaks (identical across null thresholds up to the trip)
    # the policy-side tag gate (rl_pwm.obs_guard): when would the HOLD pulses have replaced the policy's?
    og = PwmObsGuard(obs_guard_cfg, dt=1.0 / hz)
    first_hold = None
    n_hold = 0
    n_live = 0
    for i in range(d["n"]):
        if not live[i]:
            continue
        n_live += 1
        nt = d["n_tags"][i]
        eta = np.array([d["p_xyz"][i, 0], d["p_xyz"][i, 1], d["p_xyz"][i, 2], 0.0, 0.0, 0.0])
        _e, _n, code = og.apply(eta, np.zeros(6), None if math.isnan(nt) else int(nt), d["tag_age"][i])
        if code == GUARD_HOLD:
            n_hold += 1
            if first_hold is None:
                first_hold = float(d["t"][i] - t_ref)
    out["hold_guard_s"] = first_hold
    out["hold_guard_frac"] = (n_hold / n_live) if n_live else None
    return out


def discover(data_root: str, days) -> list:
    files = []
    for day in days:
        files += sorted(glob.glob(os.path.join(data_root, str(day), "*", "mpc_*.csv")))
    return files


# ----------------------------------------------------------------------------- the report
def _trip_cell(trip) -> str:
    if trip is None:
        return "-"
    return f"{trip['name']} @{trip['t_s']:+.2f}s"


def _peak_cell(pk) -> str:
    if not pk:
        return "-"
    a = pk.get("batt_a_max")
    v = pk.get("batt_v_min")
    return (f"{pk.get('tilt_deg', 0.0):.0f}/{pk.get('rate_dps', 0.0):.0f}/"
            f"{'-' if a is None else f'{a:.0f}'}/{'-' if v is None else f'{v:.1f}'}/"
            f"{pk.get('null_us_max', 0.0):.0f}/{int(pk.get('pinned_n_max', 0))}")


def format_table(results: list, null_us_values, rel_to: str | None = None) -> str:
    nus = [float(x) for x in null_us_values]
    head = ["file", "mode", "cap", "rate", "T s", "pulse@"] + [f"first trip @null{int(x)}" for x in nus] + \
           ["peaks tilt/rate/A/Vmin/null/pin", "tag gate HOLD@ (frac)"]
    rows = []
    for r in results:
        f = r["file"]
        if rel_to:
            try:
                f = os.path.relpath(f, rel_to)
            except ValueError:
                pass
        hg = ("-" if r["hold_guard_s"] is None
              else f"{r['hold_guard_s']:+.2f}s ({100.0 * (r['hold_guard_frac'] or 0.0):.0f}%)")
        rows.append([f, r["mode"], f"{r['cap']:.2f}", r["rate_src"], f"{r['duration_s']:.1f}",
                     ("-" if r["first_pulse_s"] is None else f"{r['first_pulse_s']:.2f}")]
                    + [_trip_cell(r["trips"].get(x)) for x in nus] + [_peak_cell(r["peak"]), hg])
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(head)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*head), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*row) for row in rows]
    return "\n".join(lines)


def summarize(results: list, null_us_values) -> str:
    nus = [float(x) for x in null_us_values]
    pwm = [r for r in results if r["mode"] in PWM_MODES]
    mix = [r for r in results if r["mode"] not in PWM_MODES]
    lines = []
    for nu in nus:
        tripped = [r for r in pwm if r["trips"].get(nu)]
        names = {}
        for r in tripped:
            names[r["trips"][nu]["name"]] = names.get(r["trips"][nu]["name"], 0) + 1
        lines.append(f"null {int(nu)} us: {len(tripped)}/{len(pwm)} pwm engagements trip "
                     f"({', '.join(f'{k} {v}' for k, v in sorted(names.items())) or 'none'}); "
                     f"mixer-mode files tripping: {sum(1 for r in mix if r['trips'].get(nu))}/{len(mix)}")
    return "\n".join(lines)


def write_report(text: str, out_path: str) -> str:
    """A NEW file: the folder is created, an existing report is never overwritten (a stamped sibling instead)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p = p.with_name(f"{p.stem}_{time.strftime('%H%M%S')}{p.suffix}")
    p.write_text(text, encoding="utf-8")
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="*", help="explicit CSV files (default: discover under --data/--days)")
    ap.add_argument("--data", default=str(REPO_ROOT / "data"))
    ap.add_argument("--days", nargs="*", default=list(DEFAULT_DAYS))
    ap.add_argument("--null-us", nargs="*", type=float, default=None,
                    help="null monitors to replay (default: the shipped engage.per_thruster.null_us plus 100 and 120)")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "hw_mpc.yaml"),
                    help="hw_mpc.yaml whose engage.per_thruster / rl_pwm.obs_guard are replayed ('' = defaults)")
    ap.add_argument("--out", default=str(REPO_ROOT / DEFAULT_OUT), help="report path ('' = print only)")
    ap.add_argument("--pwm-only", action="store_true", help="skip the mixer-mode control group")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one engage.per_thruster threshold for a sensitivity pass (repeatable), "
                         "e.g. --set rate_dps=40 --set null_s=0.5")
    ap.add_argument("--title", default="", help="a line naming this pass in the report header")
    a = ap.parse_args(argv)

    thr = thresholds_from_config(a.config)
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not _ or k not in thr:
            ap.error(f"--set {kv!r}: expected KEY=VALUE with KEY one of {sorted(thr)}")
        thr[k] = float(v)
    thr = parse_per_thruster_block(thr)
    if not a.null_us:
        a.null_us = sorted({float(thr["null_us"]), 100.0, 120.0})
    og_cfg = None
    if a.config:
        try:
            from rov_gui.control.geometry import MpcConfig
            og_cfg = parse_obs_guard_block((MpcConfig.load(str(a.config)).rl_pwm or {}).get("obs_guard"))
        except Exception:                                        # noqa: BLE001
            og_cfg = None
    files = list(a.csv) if a.csv else discover(a.data, a.days)
    results = []
    for f in files:
        try:
            r = replay_file(f, thr, a.null_us, og_cfg)
        except Exception as e:                                   # noqa: BLE001
            print(f"skip {f}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if r["rows"] == 0:
            continue
        if a.pwm_only and r["mode"] not in PWM_MODES:
            continue
        results.append(r)
    results.sort(key=lambda r: (r["mode"] not in PWM_MODES, r["file"]))
    pwm_tab = format_table([r for r in results if r["mode"] in PWM_MODES], a.null_us, rel_to=str(REPO_ROOT))
    mix = [r for r in results if r["mode"] not in PWM_MODES]
    mix_tab = format_table(mix, a.null_us, rel_to=str(REPO_ROOT)) if mix else "(none)"
    thr_s = ", ".join(f"{k} {v:g}" for k, v in thr.items())
    text = "\n".join([
        f"rl_pwm_interlock_replay — {time.strftime('%Y-%m-%d %H:%M:%S')}" + (f" — {a.title}" if a.title else ""),
        f"tool: rov_gui/tools/rl_pwm_interlock_replay.py (control/rl_pwm.PwmInterlocks, the class workers._runtime_fault "
        f"runs live; control/rl_pwm.PwmObsGuard for the tag gate column)",
        f"thresholds (engage.per_thruster{' of ' + a.config if a.config else ' defaults'}"
        f"{', overrides ' + ' '.join(a.set) if a.set else ''}): {thr_s}; null_us replayed at "
        f"{', '.join(f'{int(x)}' for x in a.null_us)} us",
        f"obs_guard (rl_pwm.obs_guard): {og_cfg if og_cfg else 'defaults'}",
        "inputs [측정: each CSV]: roll_deg/pitch_deg (ATTITUDE), batt_v/batt_a (BATTERY_STATUS ~1 Hz, held), pwm1..8 "
        "(the vehicle's SERVO_OUTPUT_RAW, ~0.14 s behind the sent frame — NOT the sent pulses; an empty row carries the "
        "previous report forward), n_tags/tag_age_s; "
        "rate = gyro (ekf_inputs.jsonl pqr, aligned by t0_csv) or fd (np.gradient of the logged angles, 10-01: no sidecar); "
        "AGE = att_age_v_s, judged only on schema >= 21 files "
        f"({sum(1 for r in results if r.get('age_judged'))}/{len(results)} here; earlier files never logged it)",
        "columns: trip = first interlock to reach its debounce and seconds after the first non-neutral vehicle pulse "
        "(pulse@ = that pulse's time from the file's first row); peaks = max tilt deg / max |p|,|q| deg/s / max batt_a / "
        "min batt_v / max |null| us / max thrusters at the cap; tag gate HOLD@ = when rl_pwm.obs_guard would first have "
        "replaced the policy's pulses with the hold pulses (fraction of engaged rows held)",
        "every threshold [예측]; cap per file from meta controller.pwm_cap; rows with engaged 0 skipped",
        "",
        "== the pwm engagements (rl_pwm / rl_pwm_7d) ==",
        pwm_tab,
        "",
        "== control group: mixer-allocated LOWs (mpc_tuned / rl / pid) — must never trip ==",
        mix_tab,
        "",
        "== summary ==",
        summarize(results, a.null_us),
    ]) + "\n"
    print(text)
    if a.out:
        where = write_report(text, a.out)
        print(f"report: {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
