#!/usr/bin/env python3
"""sweep_path_cost.py — pick q_along / q_cross for mode ``mpc_tuned`` offline.

    python -m rov_gui.tools.sweep_path_cost                    # the default grid
    python -m rov_gui.tools.sweep_path_cost --along 0.1 0.25 1 --cross 1 4 16
    python -m rov_gui.tools.sweep_path_cost --fillet 0.0       # waypoint square
    python -m rov_gui.tools.sweep_path_cost --plot figures/path_cost.png
    python -m rov_gui.tools.sweep_path_cost --mode dobmpc_tuned --laps 2

WHAT THIS IS FOR. ``mpc_tuned`` rotates the NMPC's 2x2 position weight into
the path frame (rov_gui/control/path_cost.py), which turns one isotropic
number into two: how much it costs to be OFF the line, and how much it costs
to be BEHIND on it. The ratio is the knob against corner cutting, and it has
no obvious right value. Because the weights are a run-time ``cost_set`` on an
already generated OCP, a whole grid costs ONE acados build — so the knob can
be swept instead of guessed.

WHAT IT IS NOT. The plant flown here is the CONTROLLER'S OWN prediction model
(dobmpc.mpc._f_casadi): no ESC deadband, no tether, and by the whole-loop fit
recorded in config/hw_mpc.yaml roughly 8-12x too little horizontal drag. It
therefore ranks GEOMETRY (does the corner get cut) honestly and CANNOT rank
anything that depends on real actuator authority. Treat every number it
prints as [예측], never as a measurement, and confirm the chosen pair with a
pool run before quoting it.

Columns:
    cut_mm    max |cross-track| inside the first corner (its fillet arc plus
              0.2 m either side) — the corner-cutting number
    p95_mm    lap-wide 95th percentile |cross-track|
    lag_mm    mean along-track lag; this is what the split BUYS the cut with,
              and PathCursor's leash bounds it at path_lead_m anyway
    lap_s     wall time to close one lap (0.10 m/s over 3.74 m = 37.4 s ideal)
    |u|       mean horizontal wrench magnitude [N] — effort, and a hint at how
              much of this survives a real thruster deadband
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _plant():
    import casadi as ca
    from dobmpc.mpc import _f_casadi

    xs, us, ws = ca.SX.sym("x", 12), ca.SX.sym("u", 6), ca.SX.sym("w", 6)
    F = ca.Function("F", [xs, us, ws], [_f_casadi(xs, us, ws)])
    z = np.zeros(6)

    def rk4(x, u, dt=0.05, n=5):
        h = dt / n
        for _ in range(n):
            k1 = np.array(F(x, u, z)).ravel()
            k2 = np.array(F(x + h / 2 * k1, u, z)).ravel()
            k3 = np.array(F(x + h / 2 * k2, u, z)).ravel()
            k4 = np.array(F(x + h * k3, u, z)).ravel()
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
    return rk4


def _fly(ctrl, mode, tune, path, s_g, v_g, lead_m, depth, max_ticks):
    """One run, driven exactly the way MpcWorker._advance_path_clock does."""
    from rov_gui.control.path_geometry import PathCursor

    rk4 = _plant()
    cur = PathCursor(path, lead_m, s_g, v_g)
    ctrl.set_path_cost(tune)
    ctrl.mode = mode
    ctrl.reset()
    dt = float(ctrl.path_plan_dt)
    x = np.zeros(12)
    x0, y0, _p, _k = path.sample(np.array([0.0]))
    x[0], x[1], x[2] = float(x0[0]), float(y0[0]), depth
    xy, lag, us, n_fail = [], [], [], 0
    for i in range(max_ticks):
        _tg, _psi, along, _cross, _v = cur.step(x[:3], dt)
        plan = cur.plan(ctrl.path_plan_steps, dt, depth, 0.0, False)
        ctrl.set_path_plan_ned(plan)
        u, info = ctrl.step(x[0:6], x[6:12], np.zeros(6), i * dt)
        n_fail = int(info.get("n_fail", 0))
        x = rk4(x, np.asarray(u, float))
        xy.append((x[0], x[1]))
        lag.append(float(along))
        us.append(float(math.hypot(u[0], u[1])))
        if cur.complete:
            break
    return (np.array(xy), np.array(lag), np.array(us),
            (i + 1) * dt, cur.complete, n_fail)


def _metrics(xy, path, pad=0.20):
    """(cut_m, p95_m, s_of_each_sample). Nearest point on ONE lap."""
    s = np.arange(0.0, path.lap_length, 0.002)
    cx, cy, _p, k = path.sample(s)
    d = np.hypot(xy[:, 0][:, None] - cx[None, :],
                 xy[:, 1][:, None] - cy[None, :])
    j = np.argmin(d, axis=1)
    err, s_at = d[np.arange(len(xy)), j], s[j]
    arc = np.isfinite(k) & (np.abs(k) > 1e-6)
    if arc.any():                       # the FIRST corner's arc only
        i0 = int(np.argmax(arc))
        i1 = i0
        while i1 + 1 < arc.size and arc[i1 + 1]:
            i1 += 1
        a, b = float(s[i0]), float(s[i1])
    else:                               # un-filleted: the vertex itself
        a = b = path.lap_length / 4.0
    m = (s_at > a - pad) & (s_at < b + pad)
    cut = float(err[m].max()) if m.any() else float("nan")
    return cut, float(np.percentile(err, 95)), err, s_at, (a, b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="offline sweep of the mpc_tuned along/cross weights")
    ap.add_argument("--along", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 1.0],
                    help="along_scale values (x the isotropic Q[0:2])")
    ap.add_argument("--cross", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0],
                    help="cross_scale values")
    ap.add_argument("--config", default="config/hw_mpc.yaml")
    ap.add_argument("--mode", default="mpc_tuned",
                    choices=("mpc_tuned", "dobmpc_tuned"))
    ap.add_argument("--size", type=float, default=1.0, help="square side [m]")
    ap.add_argument("--speed", type=float, default=None,
                    help="path speed [m/s]; default = the config's square")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--fillet", type=float, default=None,
                    help="corner fillet [m]; default = the config's")
    ap.add_argument("--lead", type=float, default=None,
                    help="path_lead_m; default = the config's")
    ap.add_argument("--terminal", dest="terminal", action="store_true",
                    default=True)
    ap.add_argument("--no-terminal", dest="terminal", action="store_false",
                    help="leave QN isotropic (ablation)")
    ap.add_argument("--split-velocity", action="store_true",
                    help="also split the linear-velocity weight")
    ap.add_argument("--plot", default=None, help="write a PNG here")
    a = ap.parse_args(argv)

    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.mpc_bridge import HwDobMpc
    from rov_gui.control.path_geometry import path_from_scenario

    cfg = MpcConfig.load(str(ROOT / a.config))
    fillet = cfg.path_fillet_m if a.fillet is None else a.fillet
    lead = cfg.path_lead_m if a.lead is None else a.lead
    speed = float(cfg.square.get("speed", 0.1)) if a.speed is None else a.speed
    scen = {"kind": "square", "origin_ned": (0.0, 0.0), "size": a.size,
            "size_y": a.size, "rot_deg": 0.0, "laps": a.laps, "speed": speed}
    path = path_from_scenario(scen, fillet_m=fillet,
                              turn_radius_m=cfg.path_turn_radius_m)
    s_g, v_g = path.speed_profile(speed, cfg.path_lat_accel_m_s2,
                                  cfg.path_long_accel_m_s2,
                                  v_creep=cfg.path_creep_m_s)
    ideal = path.total_length / max(1e-6, speed)
    ticks = int(6.0 * ideal / 0.05)

    print(f"plant: dobmpc prediction model — NO deadband, NO tether, "
          f"~8-12x too little drag. Everything below is [예측].")
    print(f"square {a.size:.2f} m x {a.laps} lap(s), fillet {fillet:.2f} m, "
          f"speed {speed:.3f} m/s, lead {lead:.2f} m, "
          f"model {cfg.rov_model}, mode {a.mode}")
    print("building the solver once (every grid point reuses it) ...")
    ctrl = HwDobMpc("mpc", cfg, log=lambda m: print("  " + m))
    baseline_mode = "dobmpc" if a.mode.startswith("dobmpc") else "mpc"

    rows = []
    runs = {}
    combos = ([("baseline", None, None)]
              + [(f"{al:g}/{cr:g}", al, cr)
                 for al in a.along for cr in a.cross])
    for label, al, cr in combos:
        if al is None:
            mode, tune = baseline_mode, None
        else:
            mode = a.mode
            tune = {"along_scale": al, "cross_scale": cr,
                    "apply_terminal": a.terminal,
                    "split_velocity": a.split_velocity}
        xy, lag, um, secs, done, n_fail = _fly(
            ctrl, mode, tune, path, s_g, v_g, lead, 0.5, ticks)
        cut, p95, _e, _s, arc = _metrics(xy, path)
        rows.append({"label": label, "along": al, "cross": cr,
                     "cut": cut, "p95": p95, "lag": float(np.mean(np.abs(lag))),
                     "secs": secs, "u": float(np.mean(um)),
                     "done": done, "n_fail": n_fail})
        runs[label] = xy

    base = rows[0]
    print(f"\nfirst corner arc: s = [{arc[0]:.2f}, {arc[1]:.2f}] m   "
          f"ideal lap {ideal / max(1, a.laps):.1f} s")
    print(f"{'along/cross':>12} {'ratio':>7} {'cut_mm':>8} {'vs base':>8} "
          f"{'p95_mm':>8} {'lag_mm':>8} {'lap_s':>7} {'|u|_N':>7}  flags")
    for r in rows:
        ratio = "" if r["along"] is None else f"{r['cross'] / r['along']:.0f}x"
        rel = ("" if r is base or not np.isfinite(base["cut"])
               else f"{100 * (r['cut'] - base['cut']) / base['cut']:+7.0f}%")
        flags = ("" if r["done"] else " INCOMPLETE") + (
            f" n_fail={r['n_fail']}" if r["n_fail"] else "")
        print(f"{r['label']:>12} {ratio:>7} {r['cut'] * 1e3:8.1f} {rel:>8} "
              f"{r['p95'] * 1e3:8.1f} {r['lag'] * 1e3:8.1f} "
              f"{r['secs'] / max(1, a.laps):7.1f} {r['u']:7.2f} {flags}")

    good = [r for r in rows[1:] if r["done"] and np.isfinite(r["cut"])]
    if good:
        best = min(good, key=lambda r: r["cut"])
        print(f"\nsmallest corner cut: along_scale {best['along']:g} / "
              f"cross_scale {best['cross']:g} -> {best['cut'] * 1e3:.1f} mm "
              f"({100 * (best['cut'] - base['cut']) / base['cut']:+.0f} % vs "
              f"baseline), lag {best['lag'] * 1e3:.0f} mm, "
              f"lap {best['secs'] / max(1, a.laps):.1f} s")
        print("Put it in config/hw_mpc.yaml under mpc_tuned:, then fly it — "
              "this plant cannot rank anything the thrusters decide.")

    if a.plot:
        _plot(a.plot, path, runs, rows)
    return 0


def _plot(out, path, runs, rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.arange(0.0, path.lap_length + 0.002, 0.002)
    cx, cy, _p, _k = path.sample(s)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(cy, cx, "k--", lw=1.2, label="path", zorder=3)
    keep = ["baseline"] + [r["label"] for r in rows[1:]][:6]
    for lab in keep:
        xy = runs[lab]
        ax[0].plot(xy[:, 1], xy[:, 0], lw=1.4,
                   label=lab, alpha=0.9)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("y [m]")
    ax[0].set_ylabel("x [m]")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title("trajectory (offline prediction-model plant)")
    ax[0].grid(alpha=0.3)

    b = rows[0]
    xs = [r["cut"] * 1e3 for r in rows[1:]]
    ys = [r["lag"] * 1e3 for r in rows[1:]]
    ax[1].scatter(xs, ys, c=[r["cross"] / r["along"] for r in rows[1:]],
                  cmap="viridis", s=60)
    for r, x, y in zip(rows[1:], xs, ys):
        ax[1].annotate(r["label"], (x, y), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].scatter([b["cut"] * 1e3], [b["lag"] * 1e3], marker="*", s=220,
                  c="crimson", label="baseline (isotropic)", zorder=5)
    ax[1].set_xlabel("corner cut [mm]  (lower = follows the corner)")
    ax[1].set_ylabel("mean along-track lag [mm]  (the price)")
    ax[1].set_title("what the split buys, and with what")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("mpc_tuned along/cross sweep — [예측], offline plant "
                 "(no deadband, no tether)", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
