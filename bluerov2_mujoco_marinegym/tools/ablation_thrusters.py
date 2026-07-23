#!/usr/bin/env python3
"""Actuator-realism ablation: ideal force path vs realistic T200 thrusters.

Station-keeping (DP at the origin, disturbance ON) for PID / MPC / DOB-MPC under
three actuator stages, with everything else identical:
  - ideal           : commanded per-thruster force == realized (the baseline used
                      in every prior experiment)
  - realistic       : T200 inverse + deadband (sub-0.7 N lost, 0.7-2 N snap to the
                      ~1.44 N minimum-spin) + fwd/rev asymmetry + saturation + motor
                      lag, at nominal voltage
  - realistic-LV    : the above PLUS a 15 % multiplicative thrust loss (battery sag
                      / wear / load) -- the part an additive DOB cannot fully cancel

Quantifies the sim-to-real actuator gap and which controller tolerates it best.
The simulator is unchanged; the realism is the opt-in ThrusterModel. Run in `robust`.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root
from dobmpc import eval_dp as E
from thrusters import ThrusterModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRLS = ["pid", "mpc", "dobmpc"]
SCEN = [("ideal", None),
        ("realistic", dict(lag=True, voltage_scale=1.0)),
        ("realistic-LV", dict(lag=True, voltage_scale=0.85))]
SEEDS = [0, 1, 2, 3, 4]               # average over disturbance realizations
T, START = 25.0, [0.1, 0.05, 0.0]


def _agg(c, name, akw):
    """Run all seeds for one (controller, actuator scenario); return mean metrics."""
    rr, jit, dc, pmx, nf = [], [], [], [], 0
    for s in SEEDS:
        act = ThrusterModel(**akw) if akw else None
        L, _, nfail = E.run_one(c, T, s, START, actuator=act)
        m = E.metrics(L)
        rr.append(m["radial_rms"])
        jit.append(np.hypot(m["std_x"], m["std_y"]))      # jitter (deadband signature)
        dc.append(np.hypot(m["dc_ex"], m["dc_ey"]))       # DC bias (voltage signature)
        pmx.append(m["pitch_max"]); nf += nfail
    return dict(radial=np.mean(rr), radial_sd=np.std(rr), jitter=np.mean(jit),
                dc=np.mean(dc), pitch_max=np.mean(pmx), nfail=nf)


def main():
    res = {}
    print(f"=== actuator-realism ablation: {T:.0f}s DP, disturbance ON, "
          f"mean over seeds {SEEDS} ===")
    print("(deadband floor ~1.44 N/thruster; realistic-LV adds x0.85 voltage loss)\n")
    for c in CTRLS:
        for name, akw in SCEN:
            res[(c, name)] = _agg(c, name, akw)
            r = res[(c, name)]
            print(f"[{c:7s} | {name:13s}] radial {r['radial']:5.2f}±{r['radial_sd']:4.2f}cm  "
                  f"jitter {r['jitter']:5.2f}cm  DC_bias {r['dc']:5.2f}cm  "
                  f"pitch_max {r['pitch_max']:5.1f}deg  n_fail {r['nfail']}")

    # ---- table: radial RMS and degradation vs each controller's own ideal
    print(f"\n=== station-keeping radial RMS [cm], mean over {len(SEEDS)} seeds "
          f"(Δ vs that controller's ideal) ===")
    hdr = f"{'ctrl':8s}" + "".join(f"{s[0]:>16s}" for s in SCEN)
    print(hdr); print("-" * len(hdr))
    for c in CTRLS:
        base = res[(c, "ideal")]["radial"]
        cells = []
        for name, _ in SCEN:
            v = res[(c, name)]["radial"]
            cells.append(f"{v:6.2f} ({v-base:+5.2f})" if name != "ideal" else f"{v:6.2f}        ")
        print(f"{c:8s}" + "".join(f"{x:>16s}" for x in cells))
    print("\njitter [cm] = position std (deadband limit-cycle signature):")
    for c in CTRLS:
        print(f"  {c:7s}: " + "  ".join(f"{s[0]} {res[(c,s[0])]['jitter']:.2f}" for s in SCEN))

    # ---- plot: grouped bars
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(CTRLS)); w = 0.26
    colors = ["#4c72b0", "#dd8452", "#c44e52"]
    for i, (name, _) in enumerate(SCEN):
        vals = [res[(c, name)]["radial"] for c in CTRLS]
        ax.bar(x + (i-1)*w, vals, w, label=name, color=colors[i])
        for xi, v in zip(x + (i-1)*w, vals):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([c.upper() for c in CTRLS])
    ax.set_ylabel("station-keeping radial RMS [cm]")
    ax.set_title(f"Actuator realism ablation ({T:.0f}s DP, disturbance ON): "
                 "ideal vs realistic T200")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    out = os.path.join(HERE, "docs", "figs", "ablation_thrusters.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
