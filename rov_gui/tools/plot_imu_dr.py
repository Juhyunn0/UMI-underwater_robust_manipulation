#!/usr/bin/env python3
"""plot_imu_dr.py — how far the IMU carried the vehicle before it was lost.

    python -m rov_gui.tools.plot_imu_dr sessions/low_level_controller_data/20260817/0817_143002
    python -m rov_gui.tools.plot_imu_dr <run>/mpc_143002.csv --show

A sibling of ``plot_runs.py`` rather than a mode of it, deliberately. That tool
is organised around comparing CONTROLLER runs to each other (radial RMS, cross
RMS, which mode tracked better); this one's unit of analysis is a WINDOW of
dead reckoning and its question is how an error grows with elapsed time. Same
argument as ``plot_nav_run.py``: a different artifact and a different question
earns its own entry point. The frame transform IS imported from plot_runs, so
the two cannot disagree about where the pool is.

Four figures and a summary:

  1. map overlay      tag trail vs dead-reckoned trail, MAP frame, with rungs
                      joining them every few seconds so the error reads as a
                      ladder rather than as two lines that happen to diverge
  2. error vs elapsed THE figure. |p_dr - p_tag| against time since the anchor
  3. decomposition    along / cross / vertical / YAW, in the DR's own heading
                      frame — this is what says whether to chase the
                      accelerometer, the mounting angle or the gyro
  4. health           dr_hz, tag_age, row gaps. A drift curve without this
                      panel is unfalsifiable: a starved estimator draws a
                      beautifully flat one

  summary.json        time_to_exceed at 5/10/25/50/100 cm, plus fits of
                      e = 0.5*b_eff*t^2 and e = (1/6)*g*beta_eff*t^3. Those two
                      effective numbers are what feed back into the bench
                      calibration.

RE-ESTIMATION
-------------
    python -m rov_gui.tools.plot_imu_dr <run> --from-jsonl --attitude gyro,ahrs
    python -m rov_gui.tools.plot_imu_dr <run> --from-jsonl --restart 20

``--from-jsonl`` re-runs the estimator over the run's raw ``*_c3_imu.jsonl``
instead of reading the flown ``dr_*`` columns, so one pool session answers
questions it was not flown for: a different attitude mode, a different
calibration, or ``--restart N`` to re-anchor every N seconds. That last one is
what turns a single continuous curve (n = 1, and drift is not stationary) into
dozens of independent windows with a p50 and a p95 — the operator chose to
anchor ONCE in flight, and this is where the statistics come from instead.

Offline only; matplotlib is fine here (same precedent as plot_nav_run.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.control.imu_dr import (            # noqa: E402
    G, ImuCalibration, ImuDeadReckoner, euler_zyx)
from rov_gui.control.state_assembler import rot_zyx   # noqa: E402
from rov_gui.tools.plot_runs import _to_map, read_csv   # noqa: E402

THRESH_CM = (5.0, 10.0, 25.0, 50.0, 100.0)
C_TAG, C_DR, C_RUNG = "#22c55e", "#f59e0b", "#94a3b8"


# =============================================================================
# input
# =============================================================================
def find_run(target: Path) -> tuple[Path, Path, Path | None]:
    """(csv, meta, c3_jsonl) from a run folder or a CSV path."""
    if target.is_dir():
        csvs = sorted(target.glob("mpc_*.csv"))
        if not csvs:
            raise SystemExit(f"no mpc_*.csv in {target}")
        csv = csvs[-1]
    else:
        csv = target
    meta = csv.with_suffix(".meta.json")
    if not meta.exists():
        raise SystemExit(f"no sidecar beside {csv} — a run without its "
                         f"meta.json cannot say which estimator produced it")
    stem = csv.with_suffix("")
    jl = Path(f"{stem}_c3_imu.jsonl")
    return csv, meta, (jl if jl.exists() else None)


def datum_of(meta: dict):
    d = ((meta.get("hardware") or {}).get("datum_tag_frame")) or None
    if not d:
        return None
    p0 = [float(v) for v in d["p0"]]
    return p0[0], p0[1], math.radians(float(d["yaw0_deg"]))


# =============================================================================
# segmentation
# =============================================================================
def segments(t: np.ndarray, dr_t: np.ndarray, ok: np.ndarray) -> list:
    """Split into dead-reckoning windows on the elapsed-since-anchor clock.

    A window ends when ``dr_t`` jumps backwards (a re-anchor) or when the
    estimate goes not-ok for a while. Written against the clock rather than a
    row counter so a re-estimation with ``--restart`` and a flown run with one
    anchor come out in the same shape.
    """
    out, start = [], None
    for i in range(len(t)):
        live = bool(ok[i]) and np.isfinite(dr_t[i])
        restart = live and start is not None and dr_t[i] < dr_t[i - 1] - 1e-6
        if live and start is None:
            start = i
        elif restart:
            out.append((start, i))
            start = i
        elif not live and start is not None:
            if i - start > 5:
                out.append((start, i))
            start = None
    if start is not None and len(t) - start > 5:
        out.append((start, len(t)))
    return out


def anchor_velocity_floor(t: np.ndarray, px: np.ndarray,
                          py: np.ndarray) -> float:
    """Sigma of the tag velocity's high-frequency content, m/s.

    A mid-flight re-anchor has no settle behind it, so it must seed velocity
    from the tag's finite difference — and THAT noise integrates linearly,
    which at short horizons buries the quadratic term this analysis exists to
    measure. Measured 2026-08-18 on a 46 s run: 26 mm/s, i.e. 0.21 m by 8 s,
    against re-anchored windows that were crossing 25 cm at a p50 of 8.3 s
    [측정: .../20260818/0818_163342/mpc_163342.csv]. The windows were grading
    the anchor, not the IMU. Printed beside every --restart summary so nobody
    reads those percentiles as an IMU result.
    """
    m = np.isfinite(px) & np.isfinite(py) & np.isfinite(t)
    if m.sum() < 40:
        return float("nan")
    tt, vx, vy = t[m], np.gradient(px[m], t[m]), np.gradient(py[m], t[m])
    k = max(3, int(round(1.0 / max(float(np.median(np.diff(tt))), 1e-6))))
    if vx.size < 3 * k:
        return float("nan")
    box = np.ones(k) / k
    hf = np.hypot(vx - np.convolve(vx, box, "same"),
                  vy - np.convolve(vy, box, "same"))[k:-k]
    return float(np.std(hf))


def time_to_exceed(segs, dr_t, err) -> dict:
    """First elapsed time each threshold is crossed, per window.

    ``None`` for a threshold no window ever crossed, with the window count
    beside it — "p50 = never after 12 windows of 40 s" is a result and must
    not be silently dropped for having no number.
    """
    out = {}
    for cm in THRESH_CM:
        hits, missed = [], 0
        for i0, i1 in segs:
            e, tt = err[i0:i1], dr_t[i0:i1]
            k = np.nonzero(np.isfinite(e) & (e * 100.0 >= cm))[0]
            if k.size:
                hits.append(float(tt[k[0]]))
            else:
                missed += 1
        out[f"{cm:g}cm"] = {
            "p50_s": (float(np.percentile(hits, 50)) if hits else None),
            "p95_s": (float(np.percentile(hits, 95)) if hits else None),
            "windows_reaching_it": len(hits), "windows_not": missed}
    return out


def fit_growth(dr_t, err) -> dict:
    """Least-squares ``0.5*b*t^2`` and ``(1/6)*g*beta*t^3`` through the cloud.

    Single-parameter fits through the origin, because the anchor IS the
    origin: an intercept would let the fit absorb a constant offset that
    cannot exist and would flatter both models. R^2 says which shape the data
    actually has, which is the point — a t^2 that fits better than a t^3 says
    chase the accelerometer, the other way round says chase the gyro.
    """
    m = np.isfinite(dr_t) & np.isfinite(err) & (dr_t > 0.2)
    t, e = dr_t[m], err[m]
    out = {"samples": int(t.size)}
    if t.size < 20:
        return out
    for name, basis in (("b_eff_m_s2", 0.5 * t ** 2),
                        ("beta_eff_rad_s", (1.0 / 6.0) * G * t ** 3)):
        k = float(basis @ e / max(basis @ basis, 1e-30))
        resid = e - k * basis
        ss = float(((e - e.mean()) ** 2).sum())
        out[name] = k
        out[name.replace("_m_s2", "").replace("_rad_s", "") + "_r2"] = (
            float(1.0 - (resid ** 2).sum() / ss) if ss > 0 else None)
    if "beta_eff_rad_s" in out:
        out["beta_eff_deg_s"] = math.degrees(out["beta_eff_rad_s"])
    return out


# =============================================================================
# offline re-estimation
# =============================================================================
def _velocity_ned(t, x, y, z, half_s: float = 0.25) -> np.ndarray:
    """Centred finite difference of the tag position, lightly smoothed.

    Smoothed over +/-half_s rather than differenced sample-to-sample: the tag
    position carries millimetre noise, and at a 0.05 s tick a bare difference
    turns 1 mm into 2 cm/s — which is the same order as the mission speed.
    """
    P = np.stack([np.asarray(x, float), np.asarray(y, float),
                  np.asarray(z, float)], axis=1)
    t = np.asarray(t, float)
    n = max(1, int(round(half_s / max(np.median(np.diff(t)), 1e-6))))
    V = np.full_like(P, np.nan)
    for i in range(len(t)):
        a, b = max(0, i - n), min(len(t) - 1, i + n)
        dt = t[b] - t[a]
        if b > a and dt > 1e-6:
            V[i] = (P[b] - P[a]) / dt
    return V


def reestimate(jsonl: Path, d: dict, meta: dict, attitude: str,
               restart_s: float, calib_path: str | None) -> dict:
    """Re-run the estimator over the raw samples, anchoring off the CSV's tag
    columns. The payoff of logging every sample: a pool session answers
    questions it was not flown for."""
    from rov_gui.tools.calib_c3_imu import read_c3_jsonl

    c3 = read_c3_jsonl(jsonl)
    calib = ImuCalibration.identity()
    if calib_path and Path(calib_path).exists():
        calib = ImuCalibration.from_json(calib_path)
    dr = ImuDeadReckoner(calib=calib, attitude=attitude, z_source="imu")

    # The CSV's t is monotonic from the run start; the IMU's t_device is on
    # the host monotonic clock. Align on the CSV's own first row: the raw log
    # and the CSV are opened together (workers._open_csv), so their zeros are
    # within a tick of each other. Good enough for elapsed-time analysis and
    # stated rather than assumed.
    t_csv = d["t"]
    t0 = float(c3["t"][0])
    px, py, pz = d["px"], d["py"], d["pz"]
    zero = np.zeros_like(t_csv)
    # ALL THREE Euler angles. Anchoring level while the vehicle was actually
    # rolled or pitched leaks g*sin(angle) into the horizontal from the very
    # first sample, which reads afterwards as a huge accelerometer bias — the
    # reason roll_deg was added to the CSV.
    yaw = np.radians(d.get("yaw_deg", zero))
    pit = np.radians(d.get("pitch_deg", zero))
    rol = np.radians(d.get("roll_deg", zero))
    if "roll_deg" not in d:
        print("  NOTE: this CSV predates roll_deg; anchoring with roll = 0, "
              "which inflates the apparent accelerometer bias")
    win = float(dr.static_window_s)
    # World-NED velocity, smoothed, for the anchors. The live flow anchors ONCE
    # at the end of a settle and can honestly call the velocity zero; a
    # mid-flight re-anchor cannot — zeroing a vehicle that is doing 0.1 m/s
    # injects an error that integrates LINEARLY and swamps the quadratic term
    # this analysis exists to measure.
    V = _velocity_ned(t_csv, px, -py, -pz)
    # ...and the vehicle's TRUE body rates, differenced from the recorded
    # attitude. calibrate_static takes the mean gyro over its window as the
    # bias unless it is given a reference, and a re-anchor mid-flight has no
    # settle behind it: without this the vehicle's real 4-5 deg/s of yaw is
    # removed as "bias" and the whole dead-reckoned track comes out ROTATED.
    # This is the offline stand-in for the autopilot rates the live flow
    # passes in (workers._anchor_dr). Small-angle, so roll/pitch rates are
    # their Euler derivatives.
    W = _velocity_ned(t_csv, rol, -pit, -yaw)
    out_t, out_p, out_dt, out_ok, out_zi = [], [], [], [], []
    seg_t0 = None
    j = 0
    for i in range(len(t_csv)):
        if not np.isfinite(px[i]):
            continue
        t_i = t0 + float(t_csv[i])
        # NED from the CSV's world FLU: (x, -y, -z), roll keeps its sign.
        eta = np.array([px[i], -py[i], -pz[i], rol[i], -pit[i], -yaw[i]])
        if seg_t0 is None or (restart_s > 0 and
                              float(t_csv[i]) - seg_t0 >= restart_s):
            # Rebuild the static window from the samples that PRECEDE this
            # anchor, exactly as the live flow does at the end of a settle —
            # the earlier shortcut fed one synthetic sample and so measured a
            # static correction of zero, defeating its whole purpose.
            dr.reset()
            lo = int(np.searchsorted(c3["t"], t_i - win))
            hi = int(np.searchsorted(c3["t"], t_i))
            if hi > lo:
                pre = np.zeros((hi - lo, 7))
                pre[:, 0] = c3["t"][lo:hi]
                pre[:, 1:4] = c3["a"][lo:hi]
                pre[:, 4:7] = c3["g"][lo:hi]
                dr.integrate(pre)
                m = ((t_csv >= float(t_csv[i]) - win) & (t_csv <= t_csv[i])
                     & np.isfinite(W).all(axis=1))
                gref = aref = None
                if m.sum() >= 4:
                    # SERIES, not means, in the c3 clock — same reasons as the
                    # live flow: a constant cannot remove time-varying
                    # rotation, and the accel offset is a per-sample residual
                    # against the attitude actually held (imu_dr explains the
                    # measured cost of getting either one wrong).
                    gref = np.column_stack([t0 + t_csv[m], W[m]])
                    aref = np.column_stack([t0 + t_csv[m], rol[m], -pit[m]])
                dr.calibrate_static(
                    rot_zyx(*[float(v) for v in eta[3:6]]),
                    gyro_ref=gref, att_ref=aref)
            v0 = V[i]
            dr.anchor(eta, nu_world_ned=v0, t=t_i,
                      zero_velocity=not np.all(np.isfinite(v0)))
            seg_t0 = float(t_csv[i])
            j = hi
        k = int(np.searchsorted(c3["t"], t_i))
        if k > j:
            arr = np.zeros((k - j, 7))
            arr[:, 0] = c3["t"][j:k]
            arr[:, 1:4] = c3["a"][j:k]
            arr[:, 4:7] = c3["g"][j:k]
            dr.integrate(arr)
            j = k
        st = dr.state(t_i)
        out_t.append(float(t_csv[i]))
        out_ok.append(bool(st["ok"]))
        out_dt.append(float(st["elapsed"] or 0.0))
        out_p.append(st["p_ned"] if st["ok"] else np.array([np.nan] * 3))
        out_zi.append(float(st["pz_imu"]) if st["ok"] else np.nan)
    P = np.asarray(out_p, float)
    return {"t": np.asarray(out_t), "dr_t_s": np.asarray(out_dt),
            "dr_ok": np.asarray(out_ok, float),
            # back to world FLU, the convention the CSV columns use
            "dr_px": P[:, 0], "dr_py": -P[:, 1], "dr_pz": -P[:, 2],
            "dr_pz_imu": -np.asarray(out_zi, float)}


# =============================================================================
# figures
# =============================================================================
def draw(d, segs, meta, title, out: Path, plt, show: bool) -> list[Path]:
    datum = datum_of(meta)
    tx, ty = _to_map(d["px"], d["py"], datum)
    dx, dy = _to_map(d["dr_px"], d["dr_py"], datum)
    err = np.hypot(d["dr_px"] - d["px"], d["dr_py"] - d["py"])
    dt = d["dr_t_s"]
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_title(f"{title}\ntag (truth) vs IMU dead reckoning", fontsize=10)
    for i0, i1 in segs:
        ax.plot(ty[i0:i1], tx[i0:i1], "-", color=C_TAG, lw=1.6,
                label="tag (truth)" if i0 == segs[0][0] else None)
        ax.plot(dy[i0:i1], dx[i0:i1], "--", color=C_DR, lw=1.6,
                label="IMU only" if i0 == segs[0][0] else None)
        # Rungs every 2 s: the eye reads a ladder as a growing gap far better
        # than it reads two curves drifting apart.
        step = max(1, int(2.0 / max(1e-6, np.median(np.diff(d["t"][i0:i1])))))
        for k in range(i0, i1, step):
            ax.plot([ty[k], dy[k]], [tx[k], dx[k]], "-", color=C_RUNG,
                    lw=0.6, alpha=0.7)
        ax.plot(ty[i0], tx[i0], "o", color=C_TAG, ms=6, mfc="none")
    ax.set_xlabel("y map (m)   [screen right]")
    ax.set_ylabel("x map (m)   [screen up]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    paths += _save(fig, out, "map", plt, show)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_title(f"{title}\nposition error vs time since the anchor",
                 fontsize=10)
    for i0, i1 in segs:
        ax.plot(dt[i0:i1], err[i0:i1] * 100.0, "-", color=C_DR, lw=1.0,
                alpha=0.75 if len(segs) > 1 else 1.0)
    if len(segs) > 3:
        # p50/p95 envelope across windows, on a shared elapsed grid.
        gmax = float(np.nanmax([dt[i1 - 1] for i0, i1 in segs]))
        grid = np.linspace(0.0, gmax, 60)
        stack = np.full((len(segs), grid.size), np.nan)
        for r, (i0, i1) in enumerate(segs):
            stack[r] = np.interp(grid, dt[i0:i1], err[i0:i1] * 100.0,
                                 left=np.nan, right=np.nan)
        with np.errstate(all="ignore"):
            p50 = np.nanpercentile(stack, 50, axis=0)
            p95 = np.nanpercentile(stack, 95, axis=0)
        ax.plot(grid, p50, "-", color="#0f172a", lw=2.0, label="p50")
        ax.plot(grid, p95, ":", color="#0f172a", lw=1.6, label="p95")
        ax.legend(fontsize=8)
    for cm in THRESH_CM:
        ax.axhline(cm, color=C_RUNG, lw=0.6, ls=":")
        ax.annotate(f"{cm:g} cm", (0.0, cm), fontsize=7, color="#64748b",
                    va="bottom")
    ax.set_xlabel("seconds since the anchor")
    ax.set_ylabel("|p_dr - p_tag|  (cm)")
    ax.grid(alpha=0.25)
    paths += _save(fig, out, "error", plt, show)

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    axs[0].set_title(f"{title}\nwhere the error went", fontsize=10)
    for i0, i1 in segs:
        # In the DR's own heading frame, so "along" and "cross" mean what the
        # vehicle would call them.
        psi = np.radians(-d.get("yaw_deg", np.zeros_like(dt))[i0:i1])
        ex = d["dr_px"][i0:i1] - d["px"][i0:i1]
        ey = -(d["dr_py"][i0:i1] - d["py"][i0:i1])
        axs[0].plot(dt[i0:i1],
                    (np.cos(psi) * ex + np.sin(psi) * ey) * 100.0,
                    color=C_DR, lw=1.0, alpha=0.8)
        axs[1].plot(dt[i0:i1],
                    (-np.sin(psi) * ex + np.cos(psi) * ey) * 100.0,
                    color=C_DR, lw=1.0, alpha=0.8)
        # dr_pz_imu, NOT dr_pz. With z_source "pressure" (the default) dr_pz
        # IS the barometer, so dr_pz - pz is the same number minus itself and
        # this panel drew +-0.06 cm of tick-alignment noise while the estimator
        # was really 397 cm out. The IMU's own integrated depth is the only
        # honest vertical here [측정: .../20260818/0818_163342/mpc_163342.csv].
        zc = "dr_pz_imu" if "dr_pz_imu" in d else None
        if zc is not None:
            axs[2].plot(dt[i0:i1],
                        (d[zc][i0:i1] - d["pz"][i0:i1]) * 100.0,
                        color=C_DR, lw=1.0, alpha=0.8)
    for a, lbl in zip(axs, ("along heading (cm)", "cross heading (cm)",
                            "vertical, IMU-only (cm)")):
        a.set_ylabel(lbl, fontsize=9)
        a.grid(alpha=0.25)
        a.axhline(0.0, color=C_RUNG, lw=0.8)
    axs[-1].set_xlabel("seconds since the anchor")
    paths += _save(fig, out, "decomposition", plt, show)

    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axs[0].set_title(f"{title}\nis the drift curve believable?", fontsize=10)
    if "dr_hz" in d:
        axs[0].plot(d["t"], d["dr_hz"], "-", color=C_DR, lw=1.0)
        axs[0].axhline(150.0, color="#ef4444", lw=0.8, ls=":")
    axs[0].set_ylabel("IMU rate (Hz)", fontsize=9)
    if "tag_age_s" in d:
        axs[1].plot(d["t"], d["tag_age_s"], "-", color=C_TAG, lw=1.0)
    axs[1].set_ylabel("tag age (s)", fontsize=9)
    for i0, _i1 in segs:
        for a in axs:
            a.axvline(d["t"][i0], color="#0f172a", lw=0.8, alpha=0.5)
    for a in axs:
        a.grid(alpha=0.25)
    axs[-1].set_xlabel("run time (s)")
    paths += _save(fig, out, "health", plt, show)
    return paths


def _save(fig, out: Path, stem: str, plt, show: bool) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"imu_dr_{stem}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    if not show:
        plt.close(fig)
    return [p]


# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a run folder or an mpc_*.csv")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="figure directory (default: beside the CSV)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--from-jsonl", action="store_true",
                    help="re-run the estimator over the raw IMU log instead "
                         "of reading the dr_* columns the run flew")
    ap.add_argument("--attitude", default="ahrs",
                    help="attitude mode(s) for --from-jsonl, comma separated")
    ap.add_argument("--restart", type=float, default=0.0,
                    help="re-anchor every N s (--from-jsonl only). This is "
                         "where the p50/p95 drift statistics come from: the "
                         "flown run anchors once, by design.")
    ap.add_argument("--calib", default="config/c3_imu_calib.json")
    a = ap.parse_args(argv)

    csv, meta_path, jsonl = find_run(a.target)
    meta = json.loads(meta_path.read_text())
    d = read_csv(csv)
    out_dir = a.out or csv.parent
    imu_meta = meta.get("imu_dr") or {}
    print(f"run   : {csv}")
    print(f"meta  : schema {meta.get('schema_version')}, imu_dr "
          f"{'ON' if imu_meta.get('enabled') else 'OFF'}"
          + (f" ({imu_meta.get('mode')}/{imu_meta.get('attitude')}, calib "
             f"{(imu_meta.get('calibration') or {}).get('sha1') or 'NONE'})"
             if imu_meta.get("enabled") else ""))
    if not imu_meta.get("enabled") and not a.from_jsonl:
        print("\nThis run was flown with the dead reckoner OFF, so there is "
              "nothing to plot. If the raw IMU log is there, --from-jsonl can "
              "estimate one after the fact.")
        return 1

    variants = {}
    if a.from_jsonl:
        if jsonl is None:
            raise SystemExit(f"no *_c3_imu.jsonl beside {csv}")
        for att in [s.strip() for s in a.attitude.split(",") if s.strip()]:
            print(f"re-estimating: attitude={att} restart={a.restart:g}s")
            variants[att] = reestimate(jsonl, d, meta, att, a.restart,
                                       a.calib)
    else:
        variants[str(imu_meta.get("attitude") or "flown")] = {
            k: d[k] for k in ("t", "dr_px", "dr_py", "dr_pz", "dr_t_s",
                              "dr_ok", "dr_hz") if k in d}

    summary = {"csv": str(csv), "meta": imu_meta, "variants": {}}
    figs = []
    for name, v in variants.items():
        base = dict(d)
        base.update(v)
        for k in ("dr_px", "dr_py", "dr_t_s", "dr_ok"):
            if k not in base:
                raise SystemExit(f"the CSV has no {k} column — this run "
                                 f"predates the estimator; use --from-jsonl")
        segs = segments(base["t"], base["dr_t_s"], base["dr_ok"] > 0.5)
        if not segs:
            print(f"  {name}: no live dead-reckoning window in this run")
            continue
        err = np.hypot(base["dr_px"] - base["px"], base["dr_py"] - base["py"])
        spans = [float(base["dr_t_s"][i1 - 1]) for i0, i1 in segs]
        s = {"windows": len(segs),
             "window_s_p50": float(np.percentile(spans, 50)),
             "window_s_max": float(np.max(spans)),
             "time_to_exceed": time_to_exceed(segs, base["dr_t_s"], err),
             "growth_fit": fit_growth(base["dr_t_s"], err)}
        summary["variants"][name] = s
        print(f"\n  [{name}] {len(segs)} window(s), longest "
              f"{s['window_s_max']:.1f} s")
        if len(segs) > 1:
            sig = anchor_velocity_floor(base["t"], base["px"], base["py"])
            if np.isfinite(sig):
                s["anchor_velocity_sigma_m_s"] = sig
                s["anchor_velocity_floor_m"] = sig * s["window_s_p50"]
                print(f"    NOTE: re-anchored windows seed velocity from the "
                      f"tag ({sig * 1000:.0f} mm/s of noise), which alone is "
                      f"{sig * s['window_s_p50']:.2f} m by "
                      f"{s['window_s_p50']:.0f} s — read the rows below "
                      f"against that floor, not as an IMU result")
        for cm, r in s["time_to_exceed"].items():
            p50 = r["p50_s"]
            print(f"    exceeds {cm:>6}: "
                  + (f"p50 {p50:5.1f} s  p95 {r['p95_s']:5.1f} s  "
                     f"({r['windows_reaching_it']} of "
                     f"{r['windows_reaching_it'] + r['windows_not']} windows)"
                     if p50 is not None
                     else f"NEVER in {r['windows_not']} window(s)"))
        g = s["growth_fit"]
        if "b_eff_m_s2" in g:
            print(f"    0.5*b*t^2  -> b    {g['b_eff_m_s2']:.4f} m/s^2 "
                  f"(R2 {g.get('b_eff_r2') or float('nan'):.3f})")
            print(f"    g*beta*t^3/6 -> beta {g['beta_eff_deg_s']:.4f} deg/s "
                  f"(R2 {g.get('beta_eff_r2') or float('nan'):.3f})")
        if len(variants) == 1:
            plt = _mpl(a.show)
            figs += draw(base, segs, meta, f"{csv.parent.name} / {name}",
                         out_dir, plt, a.show)

    sp = out_dir / "imu_dr_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {sp}")
    for p in figs:
        print(f"      {p}")
    if a.show and figs:
        _mpl(True).show()
    return 0


def _mpl(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
