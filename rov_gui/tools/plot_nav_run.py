#!/usr/bin/env python3
"""plot_nav_run.py — replot a REC NAV recording (sessions/nav_runs/<stamp>/).

    python -m rov_gui.tools.plot_nav_run sessions/nav_runs/20260814_101500
    python -m rov_gui.tools.plot_nav_run <run dir> --show
    python -m rov_gui.tools.plot_nav_run <run dir> -o custom.png

The recording is two files the GUI's REC NAV button wrote (window.py):

    map.json    the locked tag map (id -> x, y, z, yaw), tag size, pool
                boundary, geofence — the world as the run knew it
    fixes.csv   every localizer result, RAW tag-frame, misses included

This draws the old tagslam-style top-down view: tags as green squares with
ids, the pool border, and the vehicle path colored by TIME (viridis — same
ramp the GUI trail uses, so the two read alike). Axes follow the GUI plot:
screen up = +x_ned, screen right = +y_ned.

matplotlib is fine HERE (offline analysis, its own process); the no-matplotlib
house rule applies to the live GUI only (widgets/indicators.py:5).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run_dir / "map.json").read_text())
    rows = []
    with open(run_dir / "fixes.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return meta, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path, help="sessions/nav_runs/<stamp>/")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output image (default <run_dir>/trajectory.png)")
    ap.add_argument("--show", action="store_true", help="open a window too")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    meta, rows = load_run(args.run_dir)
    ok = [r for r in rows if r["ok"] == "1"]
    if not ok:
        print(f"{args.run_dir}: {len(rows)} rows, none with a fix — nothing "
              f"to plot")
        return 1
    t = [float(r["t_capture"]) for r in ok]
    x = [float(r["x_ned"]) for r in ok]
    y = [float(r["y_ned"]) for r in ok]
    t0 = t[0]
    ts = [v - t0 for v in t]

    fig, ax = plt.subplots(figsize=(8, 9))

    # Pool boundary + geofence (display frames match the GUI plot: the map
    # frame's y is the horizontal axis, x the vertical one).
    pool = meta.get("pool_ned")
    if pool:
        ax.add_patch(patches.Rectangle(
            (pool["y"][0], pool["x"][0]),
            pool["y"][1] - pool["y"][0], pool["x"][1] - pool["x"][0],
            fill=False, edgecolor="0.35", linewidth=1.8, label="pool"))
    fence = meta.get("geofence_ned")
    if fence:
        ax.add_patch(patches.Rectangle(
            (fence["y"][0], fence["x"][0]),
            fence["y"][1] - fence["y"][0], fence["x"][1] - fence["x"][0],
            fill=False, edgecolor="orange", linestyle="--", linewidth=0.9,
            alpha=0.7, label="geofence"))

    # Tags: oriented green squares with ids — the old tagslam visualization.
    size = float(meta.get("tag_size_m", 0.17))
    used_any = set()
    for r in ok:
        used_any.update(i for i in r["tag_ids"].split(";") if i)
    for tid, tag in (meta.get("tags") or {}).items():
        c, s = math.cos(tag["yaw_rad"]), math.sin(tag["yaw_rad"])
        h = size / 2.0
        cx, cy = tag["x"], tag["y"]
        quad = [(cx + c * dx - s * dy, cy + s * dx + c * dy)
                for dx, dy in ((-h, -h), (-h, h), (h, h), (h, -h))]
        seen = str(tid) in used_any
        ax.add_patch(patches.Polygon(
            [(qy, qx) for qx, qy in quad], closed=True,
            facecolor="mediumseagreen" if seen else "0.85",
            edgecolor="darkgreen" if seen else "0.6", linewidth=0.6,
            alpha=0.9 if seen else 0.6))
        ax.annotate(str(tid), (cy, cx), fontsize=6, ha="center", va="center",
                    color="black" if seen else "0.45")

    # The path, colored by time — viridis, like the GUI trail.
    sc = ax.scatter(y, x, c=ts, cmap="viridis", s=6, zorder=3)
    fig.colorbar(sc, ax=ax, label="t [s] since first fix", shrink=0.8)
    ax.plot(y[0], x[0], "k^", markersize=9, zorder=4)
    ax.annotate("start", (y[0], x[0]), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.plot(y[-1], x[-1], "r*", markersize=12, zorder=4)
    ax.annotate("end", (y[-1], x[-1]), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color="red")

    dur = ts[-1] if ts[-1] > 0 else float("nan")
    rate = (len(ok) - 1) / dur if dur and dur > 0 else float("nan")
    ax.set_title(f"{args.run_dir.name} — {len(ok)}/{len(rows)} fixes, "
                 f"{dur:.0f} s, mean {rate:.1f} Hz "
                 f"(src {ok[-1]['source'] or '?'})", fontsize=10)
    ax.set_xlabel("y_ned [m]  (screen right)")
    ax.set_ylabel("x_ned [m]  (screen up)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    out = args.out or (args.run_dir / "trajectory.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
