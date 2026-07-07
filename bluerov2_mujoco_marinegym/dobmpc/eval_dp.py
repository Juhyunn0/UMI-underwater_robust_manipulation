#!/usr/bin/env python3
"""Dynamic-positioning (station-keeping) comparison: PID vs MPC vs DOB-MPC.

Runs the three controllers on the SAME marinegym plant, SAME disturbance seed
(current + irregular JONSWAP waves + Poisson kicks) and SAME start offset, holding
the global origin. Reports the DP hold metrics our baseline-PID analysis used --
radial error, DC bias (current rejection), wave-band residual, pitch -- plus the
EAOB disturbance estimate vs the true current. Saves a 6-panel comparison figure.

Expected (control-theory-advisor): DOB-MPC's EAOB rejects the DC current almost
completely (radial bias ~0), strongly beating PID; the wave band is only partially
rejected (w_dot=0 model) and 0.15 s kicks are not rejected -- the same residual the
PID analysis found, smaller.

Usage:  python dobmpc/eval_dp.py [--T 60] [--seed 0] [--start 0.1,0.05,0] [--ctrls pid,mpc,dobmpc]
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # marinegym dir
sys.path.insert(0, HERE)

import hydro as H
import disturbances as D
import rov_model as RM
from controller import PoseController
from dobmpc_controller import DOBMPCController

XML = RM.XML_PATH          # bluerov.xml (6 thr) or bluerov_heavy.xml (8 thr) per ROV_MODEL


def _build(seed):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    field = D.DisturbanceField(waves=D.jonswap_wave_specs(seed=seed), seed=seed)
    field.enabled = True
    hydro = H.Hydrodynamics(model, disturbance=field).install()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return model, data, hydro, field, bid


def run_one(mode, T, seed, start, log_hz=20.0, actuator=None):
    model, data, hydro, field, bid = _build(seed)
    if mode == "pid":
        ctrl = PoseController(model, mode="pid", buoyancy_ff=hydro, actuator=actuator)
    else:
        ctrl = DOBMPCController(model, hydro=hydro, mode=mode, actuator=actuator)
    ctrl.set_target((0.0, 0.0, 0.0), yaw_ref=0.0)
    data.qpos[:3] = list(start)
    mujoco.mj_forward(model, data)

    log_dt = 1.0 / log_hz
    L = {k: [] for k in ("t", "px", "py", "pz", "pitch",
                         "wx", "wy", "wz", "cur_x", "cur_y")}
    t0 = time.time(); last = -log_dt
    while data.time < T:
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)
        if data.time - last >= log_dt:
            last = data.time
            p = np.asarray(data.xpos[bid], float)
            R = np.asarray(data.xmat[bid], float).reshape(3, 3)
            L["t"].append(data.time)
            L["px"].append(p[0]); L["py"].append(p[1]); L["pz"].append(p[2])
            L["pitch"].append(np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1))))
            w = ctrl.w_world_flu()[:3] if mode != "pid" else np.zeros(3)
            L["wx"].append(w[0]); L["wy"].append(w[1]); L["wz"].append(w[2])
            cur = hydro.water["current"][1]
            L["cur_x"].append(cur[0]); L["cur_y"].append(cur[1])
    wall = time.time() - t0
    nfail = getattr(ctrl, "n_fail", 0)
    H.Hydrodynamics.uninstall()
    return {k: np.asarray(v) for k, v in L.items()}, wall, nfail


def metrics(L, settle=10.0):
    m = L["t"] >= settle                                   # steady window
    px, py, pz = L["px"][m], L["py"][m], L["pz"][m]
    rxy = np.hypot(px, py)
    return dict(
        radial_rms=np.sqrt((rxy ** 2).mean()) * 100,       # cm
        radial_max=rxy.max() * 100,
        dc_ex=px.mean() * 100, dc_ey=py.mean() * 100,      # cm (current rejection)
        std_x=px.std() * 100, std_y=py.std() * 100,        # cm (wave residual)
        depth_std=pz.std() * 100,
        pitch_mean=L["pitch"][m].mean(), pitch_max=np.abs(L["pitch"]).max(),
        w_x=L["wx"][m].mean(), w_y=L["wy"][m].mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=str, default="0.1,0.05,0.0")
    ap.add_argument("--ctrls", type=str, default="pid,mpc,dobmpc")
    args = ap.parse_args()
    start = [float(v) for v in args.start.split(",")]
    ctrls = args.ctrls.split(",")

    logs, mets = {}, {}
    for c in ctrls:
        print(f"[eval_dp] running {c} ({args.T:.0f}s DP, seed {args.seed}) ...", flush=True)
        L, wall, nfail = run_one(c, args.T, args.seed, start)
        logs[c] = L; mets[c] = metrics(L)
        print(f"   wall {wall:.1f}s  solve_fail {nfail}  radial_rms {mets[c]['radial_rms']:.1f} cm",
              flush=True)

    # ---- table
    cur = logs[ctrls[0]]["cur_x"][-1], logs[ctrls[0]]["cur_y"][-1]
    print(f"\n=== DP comparison (T={args.T:.0f}s, seed {args.seed}, "
          f"true current ({cur[0]:+.2f},{cur[1]:+.2f}) m/s, disturb ON) ===")
    hdr = f"{'ctrl':8s} {'radial_rms':>10s} {'DC_ex':>7s} {'DC_ey':>7s} {'std_x':>6s} {'std_y':>6s} {'pitch_mn':>8s} {'pitch_mx':>8s} {'w_x':>6s}"
    print(hdr); print("-" * len(hdr))
    for c in ctrls:
        m = mets[c]
        print(f"{c:8s} {m['radial_rms']:9.1f}c {m['dc_ex']:+6.1f}c {m['dc_ey']:+6.1f}c "
              f"{m['std_x']:5.1f}c {m['std_y']:5.1f}c {m['pitch_mean']:+7.1f} "
              f"{m['pitch_max']:7.1f} {m['w_x']:+5.1f}")
    print("(c = cm, deg for pitch, N for w_x)")

    # ---- plot
    col = {"pid": "tab:red", "mpc": "tab:blue", "dobmpc": "tab:green"}
    fig = plt.figure(figsize=(15, 9))
    # 1 XY scatter
    ax = fig.add_subplot(2, 3, 1)
    for c in ctrls:
        ax.plot(logs[c]["px"] * 100, logs[c]["py"] * 100, lw=.5, color=col.get(c), alpha=.8, label=c)
    ax.scatter([0], [0], c="k", marker="+", s=120, zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    ax.set_title("DP hold scatter"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 2 radial error vs time
    ax = fig.add_subplot(2, 3, 2)
    for c in ctrls:
        ax.plot(logs[c]["t"], np.hypot(logs[c]["px"], logs[c]["py"]) * 100, lw=.6, color=col.get(c), label=c)
    ax.set_xlabel("t [s]"); ax.set_ylabel("radial-xy [cm]"); ax.set_title("Hold error vs time")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 3 DC bias + wave-residual bar
    ax = fig.add_subplot(2, 3, 3)
    xb = np.arange(len(ctrls)); wbar = 0.35
    ax.bar(xb - wbar / 2, [abs(mets[c]["dc_ex"]) for c in ctrls], wbar, label="|DC bias x|", color="tab:blue")
    ax.bar(xb + wbar / 2, [mets[c]["std_x"] for c in ctrls], wbar, label="std x (wave)", color="tab:cyan")
    ax.set_xticks(xb); ax.set_xticklabels(ctrls); ax.set_ylabel("cm")
    ax.set_title("DC bias vs wave residual (x)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 4 pitch vs time
    ax = fig.add_subplot(2, 3, 4)
    for c in ctrls:
        ax.plot(logs[c]["t"], logs[c]["pitch"], lw=.6, color=col.get(c), label=c)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pitch [deg]"); ax.set_title("Pitch (underactuated trim)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 5 EAOB w_hat vs true current (dobmpc)
    ax = fig.add_subplot(2, 3, 5)
    if "dobmpc" in ctrls:
        L = logs["dobmpc"]
        ax.plot(L["t"], L["wx"], lw=.7, color="tab:green", label=r"$\hat w_x$ (EAOB)")
        ax.plot(L["t"], L["wy"], lw=.7, color="tab:olive", label=r"$\hat w_y$")
        ax.axhline(0, color="k", lw=.5)
        ax.set_title("DOB-MPC disturbance estimate (FLU world)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 6 radial RMS summary bar
    ax = fig.add_subplot(2, 3, 6)
    ax.bar(xb, [mets[c]["radial_rms"] for c in ctrls], color=[col.get(c) for c in ctrls])
    for i, c in enumerate(ctrls):
        ax.text(i, mets[c]["radial_rms"], f"{mets[c]['radial_rms']:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xb); ax.set_xticklabels(ctrls); ax.set_ylabel("radial RMS [cm]")
    ax.set_title("Station-keeping error (steady)"); ax.grid(alpha=.3)
    fig.suptitle(f"DP comparison: PID vs MPC vs DOB-MPC  (JONSWAP+current+kicks, T={args.T:.0f}s)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    day = time.strftime("%Y%m%d")
    out_dir = os.path.join(HERE, "recordings", day)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"dp_compare_{time.strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
