#!/usr/bin/env python3
"""ekf_replay.py — re-run the robot_localization EKF on a recorded run, with
the configuration it flew or another one, and compare it with the tag state.

    python -m rov_gui.tools.ekf_replay <run>/<stem>_mpc.csv
    python -m rov_gui.tools.ekf_replay <run>/<stem>_mpc.csv \\
        --set smooth_lagged_data=true --set history_length_s=1.0
    python -m rov_gui.tools.ekf_replay <csv> --set dvl_fuse='[]' --out r.csv

Inputs: the controller CSV and ``<stem>_ekf_inputs.jsonl`` beside it (written
by MpcWorker since 2026-10-02: every tag fix, IMU message, depth and DVL
report exactly as the live EKF got them, with their arrival stamps, plus the
datum the CSV's px/py are in). The node is stepped at the CSV's own tick
times, so the replay sees the same arrivals between ticks as the live one.

What it prints, per configuration (the flown one = "live config" and each
``--set`` variant):

    |EKF - tag| horizontal rms / p95 (mm), yaw rms (deg), over rows with a
    fresh tag state — the TAG STATE IS NOT TRUTH (it is the assembler's
    own estimate, fix-age bridged), so this is a disagreement, not an error;
    the same against the CSV's live ekf_* columns (the replay's self-check:
    with the live config it should be ~0 once both have converged — the
    replay starts COLD at the CSV's first row, the live node was warm — and
    a single row can differ when an input arrived within the CSV ``t``
    column's 0.1 ms rounding of a tick). Compare a VARIANT with the
    replayed live config, not with the live columns: both start cold.

``--out`` writes the per-row replayed pose beside t, px, py, yaw_deg.

``--dvl-all`` lets the variant fuse every DVL report, including those the
live EKF was not given because the DVL switch was OUT or tripped (the live
EKF obeys the DVL's own gates; r_l as published has none).

Offline, Qt-free; needs only numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import rl_ekf as EK                           # noqa: E402


def _inputs_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_ekf_inputs.jsonl")


def load_inputs(path: Path):
    """-> (header dict, [record dicts in file order])."""
    hdr, recs = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                     # a torn last line
            if r.get("k") == "header":
                hdr = r
            else:
                recs.append(r)
    if hdr is None:
        raise SystemExit(f"{path}: no header line")
    return hdr, recs


def _parse_value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def build_node(hdr: dict, overrides: dict) -> EK.RlEkfNode:
    cfg = dict(hdr["config"])
    z_source = cfg.pop("z_source", "pressure")
    cfg.update(overrides)
    mount = None
    tr = hdr.get("dvl_transform")
    if tr:
        mount = (np.asarray(tr["R_frd_dvl"], float),
                 np.asarray(tr["r_frd_m"], float), True)
    return EK.RlEkfNode(cfg, dvl_mount=mount, z_source=z_source)


def _feed(node, r, dvl_all: bool = False) -> None:
    k, ta = r["k"], float(r["ta"])
    if k == "fix":
        node.feed_tag(SimpleNamespace(ok=True, t_capture=r["tc"],
                                      p_ned=r["p"], R_ned_body=r["R"]), ta)
    elif k == "imu":
        rpy, pqr, acc = r["rpy"], r["pqr"], r["acc"]
        node.feed_imu(SimpleNamespace(
            roll=rpy[0], pitch=rpy[1], yaw=rpy[2], p=pqr[0], q=pqr[1],
            r=pqr[2], ax=acc[0], ay=acc[1], az=acc[2], t_att=r["t_att"],
            # a log without t_imu (none were written before it existed —
            # but be safe): treat the accelerometer as current, as then
            t_imu=r["t_imu"] if "t_imu" in r else r["t_att"],
            stamp=r["stamp"]), ta)
    elif k == "depth":
        node.feed_depth(r["z"], r["t"], ta)
    elif k == "dvl":
        if not (dvl_all or r.get("fed", True)):
            return          # the live EKF did not get it (DVL switch OUT)
        node.feed_dvl(SimpleNamespace(
            t_valid=r["tv"], v_dvl=r["v"], valid=r["valid"], kind=r["kind"],
            covariance=r["cov"]), ta)


def _datumize(eta, d):
    if d is None:
        return eta
    p0, yaw0 = np.asarray(d["p0"], float), float(d["yaw0"])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.array(eta, float)
    out[0:3] = Rz @ (out[0:3] - p0)
    out[5] = math.atan2(math.sin(out[5] - yaw0), math.cos(out[5] - yaw0))
    return out


def replay(csv_path: Path, overrides: dict, dvl_all: bool = False):
    """-> (rows: list of dicts with t, tag and replayed poses, node meta)."""
    hdr, recs = load_inputs(_inputs_path(csv_path))
    node = build_node(hdr, overrides)
    if hdr.get("timer_phase") is not None:
        # on the live node's 50 Hz grid: which cycle sees an input first
        # changes the answer by millimetres in motion
        node.align_timer(float(hdr["timer_phase"]))
    t0 = float(hdr["t0_csv"])
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    i = 0
    datum = None
    for row in rows:
        try:
            tick = t0 + float(row["t"])
        except (KeyError, ValueError):
            continue
        while i < len(recs) and float(recs[i]["ta"]) <= tick + 1e-9:
            r = recs[i]
            i += 1
            if r["k"] == "datum":
                datum = r
            else:
                _feed(node, r, dvl_all)
        node.advance(tick)
        o = node.output()
        rec = {"t": float(row["t"]), "row": row, "eta": None}
        if o is not None and EK.finite(o):
            eta = _datumize(o["x"][0:6], datum)
            rec["eta"] = eta
            rec["flu"] = (float(eta[0]), -float(eta[1]), -float(eta[2]),
                          -math.degrees(float(eta[5])))
        out.append(rec)
    return out, node.meta()


def _f(row, k):
    try:
        v = float(row.get(k, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return v


def score(rows):
    """|replay - tag| and |replay - live ekf_*| over rows where both exist
    and the tag state was fresh (engaged, est column present)."""
    d_tag, y_tag, d_live = [], [], []
    for r in rows:
        if "flu" not in r:
            continue
        row = r["row"]
        px, py, yd = _f(row, "px"), _f(row, "py"), _f(row, "yaw_deg")
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        if row.get("bridge_tier", "none") not in ("", "none"):
            continue                          # a bridged row is an estimate
        x, y, _z, yaw = r["flu"]
        d_tag.append(math.hypot(x - px, y - py))
        if math.isfinite(yd):
            y_tag.append(abs((yaw - yd + 180.0) % 360.0 - 180.0))
        ex, ey = _f(row, "ekf_px"), _f(row, "ekf_py")
        if math.isfinite(ex) and math.isfinite(ey):
            d_live.append(math.hypot(x - ex, y - ey))

    def rms(a):
        return math.sqrt(float(np.mean(np.square(a)))) if a else float("nan")

    def p95(a):
        return float(np.percentile(a, 95)) if a else float("nan")
    return {"n": len(d_tag), "tag_rms_mm": 1e3 * rms(d_tag),
            "tag_p95_mm": 1e3 * p95(d_tag), "yaw_rms_deg": rms(y_tag),
            "live_rms_mm": 1e3 * rms(d_live), "n_live": len(d_live)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override one ekf: key for a VARIANT run (JSON "
                         "values: true, 1.0, '[\"x\",\"y\"]'); repeat for "
                         "several keys in one variant")
    ap.add_argument("--dvl-all", action="store_true",
                    help="the VARIANT also fuses the DVL reports the live EKF "
                         "was not given (DVL switch OUT / tripped) — the "
                         "ungated, r_l-as-published behaviour, offline only")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the variant's per-row replay to this CSV")
    a = ap.parse_args(argv)
    if not _inputs_path(a.csv).exists():
        print(f"no {_inputs_path(a.csv).name} beside {a.csv} — runs before "
              f"2026-10-02, or the EKF was off (--ekf off)", file=sys.stderr)
        return 2
    over = {}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set needs KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        over[k.strip()] = _parse_value(v.strip())
    configs = [("live config", {}, False)]
    if over or a.dvl_all:
        configs.append(("variant " + json.dumps(over)
                        + (" + all DVL" if a.dvl_all else ""), over,
                        a.dvl_all))
    last = None
    for label, ov, dall in configs:
        rows, meta = replay(a.csv, ov, dvl_all=dall)
        sc = score(rows)
        late = meta["late_mean_s"].get(EK.TOPIC_TAG)
        print(f"{label}: |EKF - tag| rms {sc['tag_rms_mm']:.1f} mm, p95 "
              f"{sc['tag_p95_mm']:.1f} mm, yaw rms {sc['yaw_rms_deg']:.2f} "
              f"deg over {sc['n']} rows; vs live ekf_* "
              f"{sc['live_rms_mm']:.1f} mm ({sc['n_live']} rows); tag fused "
              f"{'' if late is None else f'{1e3 * late:.0f} ms '}late on "
              f"average, {meta['counts'][EK.TOPIC_TAG]['used']} tags, "
              f"{meta['counts'][EK.TOPIC_DVL]['used']} DVL")
        last = rows
    if a.out is not None and last is not None:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "px", "py", "yaw_deg", "rep_px", "rep_py",
                        "rep_pz", "rep_yaw_deg"])
            for r in last:
                row = r["row"]
                fl = r.get("flu")
                w.writerow([f"{r['t']:.4f}", row.get("px"), row.get("py"),
                            row.get("yaw_deg")]
                           + ([f"{v:.5f}" for v in fl] if fl else
                              ["nan"] * 4))
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
