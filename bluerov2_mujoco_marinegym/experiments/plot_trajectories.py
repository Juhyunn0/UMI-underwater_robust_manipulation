#!/usr/bin/env python3
"""Publication-quality overlay of the square-mission trajectories from run_viewer
CSVs (one per controller). Reads traj_square_<mode>_<ctrl>_seed<seed>_dir<deg>.csv
(columns t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap) and draws, in one figure:
  * left  : XY top-down path of PID / MPC / DOB-MPC over the reference square,
  * right : radial tracking error |p - r| vs time.

Usage:
  python -m experiments.plot_trajectories                       # latest square_view, CDW
  python -m experiments.plot_trajectories --mode CDW --dir <square_view dir>
  python -m experiments.plot_trajectories --out fig.png --size 1.0
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# distinct red / blue / green (PID-vs-MPC were both warm hues before); ordered
# worst->best so DOB-MPC draws on top.
CTRLS = ["pid", "mpc", "dobmpc"]
COLOR = {"pid": "#9E2B36", "mpc": "#C9A227", "dobmpc": "#2E8B57"}
LABEL = {"pid": "PID", "mpc": "MPC", "dobmpc": "DOB-MPC"}
ZORDER = {"pid": 3, "mpc": 4, "dobmpc": 5}
ALPHA = {"pid": 0.7, "mpc": 0.85, "dobmpc": 0.95}     # de-emphasise the cluttered PID
ALL_MODES = ["NONE", "C", "CD", "CW", "CDW"]
MODE_TITLE = {"NONE": "NONE  (still water)", "C": "C  (current)",
              "CD": "CD  (current + drift)", "CW": "CW  (current + waves)",
              "CDW": "CDW  (current + drift + waves)"}


def _latest_square_view():
    cands = sorted(glob.glob(os.path.join(HERE, "recordings", "*", "square_view")))
    return cands[-1] if cands else os.path.join(HERE, "recordings")


def _find(d, mode, ctrl, seed, ddeg):
    pat = os.path.join(d, f"traj_square_{mode}_{ctrl}_seed{seed}_dir{ddeg}.csv")
    m = glob.glob(pat) or glob.glob(os.path.join(d, f"traj_square_{mode}_{ctrl}_*.csv"))
    return sorted(m)[-1] if m else None


def _plot_all_modes(d, seed, ddeg, ctrls, S, out):
    """Combined overview: one trajectory panel per disturbance mode (NONE..CDW),
    all laps, with a shared legend. The 6th cell holds the legend."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.linewidth": 0.9, "axes.edgecolor": "#2b2b2b",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 300, "figure.dpi": 130,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), constrained_layout=True)
    axes = axes.ravel()
    sq = np.array([[0, 0], [S, 0], [S, S], [0, S], [0, 0]], float)
    for i, mode in enumerate(ALL_MODES):
        ax = axes[i]
        ax.plot(sq[:, 0], sq[:, 1], ls="--", lw=1.4, color="#555555", zorder=2)
        ax.scatter(sq[:-1, 0], sq[:-1, 1], s=20, facecolor="white",
                   edgecolor="#555555", linewidths=1.0, zorder=2.5)
        xs, ys, rms = [], [], {}
        for c in ctrls:
            f = _find(d, mode, c, seed, ddeg)
            if f is None:
                print(f"[plot] WARN: no CSV for {mode}/{c}")
                continue
            a = np.genfromtxt(f, delimiter=",", names=True)
            ax.plot(a["px"], a["py"], color=COLOR[c], lw=1.5, alpha=ALPHA[c],
                    solid_joinstyle="round", zorder=ZORDER[c])
            xs.append(a["px"]); ys.append(a["py"])
            rms[c] = float(np.sqrt(np.mean(np.hypot(a["px"] - a["rx"],
                                                    a["py"] - a["ry"]) ** 2))) * 100.0
        ax.plot(0, 0, marker="o", ms=7, mfc="#1b1b1b", mec="white", mew=1.0, zorder=6)
        ax.set_aspect("equal", "box")
        ax.grid(True, color="#dddddd", lw=0.7); ax.set_axisbelow(True)
        ax.set_xlabel("x  [m]", fontsize=11); ax.set_ylabel("y  [m]", fontsize=11)
        ax.set_title(MODE_TITLE[mode], fontsize=12, fontweight="bold", pad=6)
        allx = np.concatenate(xs + [sq[:, 0]]); ally = np.concatenate(ys + [sq[:, 1]])
        pad = 0.10 * S
        lo = min(allx.min(), ally.min()) - pad; hi = max(allx.max(), ally.max()) + pad
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        txt = "\n".join(f"{LABEL[c]}: {rms[c]:.1f} cm" for c in ctrls if c in rms)
        ax.text(0.035, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                        ec="#cccccc", alpha=0.92))
    axes[-1].axis("off")                                       # 6th cell -> shared legend
    handles = [Line2D([0], [0], color=COLOR[c], lw=3.0, label=LABEL[c]) for c in ctrls]
    handles.append(Line2D([0], [0], ls="--", lw=1.6, color="#555555", label="Reference square"))
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=15,
                    handlelength=2.2, title="radial RMS shown per panel", title_fontsize=11)
    fig.suptitle("BlueROV2-Heavy — square tracking across disturbance modes  "
                 f"(seed {seed}, current {ddeg}°, all laps)",
                 fontsize=16, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Overlay square trajectories (PID/MPC/DOB-MPC).")
    ap.add_argument("--dir", default=None, help="square_view dir (default: latest).")
    ap.add_argument("--mode", default="CDW", choices=("NONE", "C", "CD", "CW", "CDW"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir-deg", type=int, default=0)
    ap.add_argument("--size", type=float, default=1.0, help="square edge [m].")
    ap.add_argument("--ctrls", default="pid,mpc,dobmpc")
    ap.add_argument("--all-modes", action="store_true",
                    help="one combined figure: a trajectory panel per mode (NONE..CDW).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.dir or _latest_square_view()
    ctrls = [c for c in args.ctrls.split(",") if c in CTRLS]
    S = float(args.size)

    if args.all_modes:
        out = args.out or os.path.join(d, "trajectory_compare_ALLMODES.png")
        _plot_all_modes(d, args.seed, args.dir_deg, ctrls, S, out)
        return

    data = {}
    for c in ctrls:
        f = _find(d, args.mode, c, args.seed, args.dir_deg)
        if f is None:
            print(f"[plot] WARN: no CSV for {c} in {d}")
            continue
        a = np.genfromtxt(f, delimiter=",", names=True)
        rxy = np.hypot(a["px"] - a["rx"], a["py"] - a["ry"])      # radial error [m]
        lap = a["lap"]
        steady = lap >= max(0, int(lap.max()) - 1)               # last 2 laps (steady)
        data[c] = dict(t=a["t"], px=a["px"], py=a["py"], err=rxy, steady=steady,
                       rms=float(np.sqrt(np.mean(rxy ** 2))) * 100.0,           # full-run
                       rms_ss=float(np.sqrt(np.mean(rxy[steady] ** 2))) * 100.0)  # steady, cm
    if not data:
        raise SystemExit(f"[plot] no trajectory CSVs found in {d}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "axes.linewidth": 0.9,
        "axes.edgecolor": "#2b2b2b", "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 10, "legend.framealpha": 0.95,
        "savefig.dpi": 300, "figure.dpi": 130,
    })

    fig, axT = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)

    # reference square
    sq = np.array([[0, 0], [S, 0], [S, S], [0, S], [0, 0]], float)
    axT.plot(sq[:, 0], sq[:, 1], ls="--", lw=1.5, color="#555555", zorder=2)
    axT.scatter(sq[:-1, 0], sq[:-1, 1], s=24, facecolor="white",
                edgecolor="#555555", linewidths=1.1, zorder=2.5)

    # full trajectories (ALL laps); PID de-emphasised so the square + best paths read
    xs, ys = [], []
    for c in ctrls:
        if c not in data:
            continue
        px, py = data[c]["px"], data[c]["py"]
        xs.append(px); ys.append(py)
        axT.plot(px, py, color=COLOR[c], lw=1.6, alpha=ALPHA[c],
                 solid_joinstyle="round", zorder=ZORDER[c])
    axT.plot(0, 0, marker="o", ms=8, mfc="#1b1b1b", mec="white", mew=1.0, zorder=6)
    axT.annotate("start", (0, 0), textcoords="offset points", xytext=(-6, -14),
                 ha="right", fontsize=9, color="#1b1b1b")
    # CCW travel-direction hint
    axT.annotate("", xy=(0.55 * S, -0.05 * S), xytext=(0.30 * S, -0.05 * S),
                 arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.3))

    axT.set_aspect("equal", "box")
    axT.set_xlabel("x  [m]"); axT.set_ylabel("y  [m]")
    axT.grid(True, color="#dddddd", lw=0.7); axT.set_axisbelow(True)
    allx = np.concatenate(xs + [sq[:, 0]]); ally = np.concatenate(ys + [sq[:, 1]])
    pad = 0.10 * S
    lo = min(allx.min(), ally.min()) - pad; hi = max(allx.max(), ally.max()) + pad
    axT.set_xlim(lo, hi); axT.set_ylim(lo, hi)

    handles = [Line2D([0], [0], color=COLOR[c], lw=2.4,
                      label=f"{LABEL[c]}   (radial RMS {data[c]['rms']:.1f} cm)")
               for c in ctrls if c in data]
    handles.append(Line2D([0], [0], ls="--", lw=1.5, color="#555555",
                          label="Reference square"))
    axT.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=2, frameon=False, handlelength=1.8, columnspacing=1.4)

    axT.set_title(f"BlueROV2-Heavy — square trajectory  (mode {args.mode}, "
                  f"current {args.dir_deg}°, seed {args.seed})",
                  fontsize=13, fontweight="bold", pad=10)

    out = args.out or os.path.join(d, f"trajectory_compare_{args.mode}.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    for c in ctrls:
        if c in data:
            print(f"        {LABEL[c]:<8} radial RMS = {data[c]['rms']:.2f} cm")


if __name__ == "__main__":
    main()
