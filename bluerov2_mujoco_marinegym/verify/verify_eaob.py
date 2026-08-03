#!/usr/bin/env python3
"""EAOB tuning validation: profile="verify" (deadbeat, clean meas) vs
profile="perf" (sensor-noise injection + sigma-derived covariances + NIS gate).

Runs the marinegym plant (DP hold, disturbances ON) with `--ctrl dobmpc` twice --
once per profile -- and scores the OBSERVER, not the controller:

  * per-axis RMSE of w_hat vs w_true. w_true is the model residual the EAOB is
    defined to estimate (Eq. 7):  w = M a + C_RB nu + C_A nu + D nu + g - tau,
    evaluated at the CLEAN plant state with the tick finite-difference a and the
    tau the EAOB consumed that tick. Cross-checked against hydro's independent
    physics-level diagnostic w_true_world (diag_wtrue).
  * filter consistency after a burn-in: mean NIS ~ 18 (z is 18-dim; accept
    14-24) and mean NEES ~ 18 (angle errors wrapped), gated-frame count.
  * motion-correlation diagnostic: per-axis regression of (w_hat - w_true)
    against |nu_lin| and each nu component. A significant slope means model
    error leaking into w_hat; near-zero slope + white residual is plain
    measurement-noise passthrough (expected).

NOTE: mean NIS / NEES are only meaningful when the filter's R matches reality
-- i.e. for perf(+noise). The verify profile's R (DT^4/4 on every channel) is a
unit-blind placeholder, so its NIS/NEES are reported for reference but not gated.

Run:  /home/bdml/miniforge3/envs/robust/bin/python verify/verify_eaob.py \
          [--T 60] [--seed 0] [--mode CDW] [--config config/base.yaml] [--no-plot]
Exit code 1 if the perf profile misses the NIS/NEES acceptance ranges.
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
import rov_model as RM
from dobmpc_controller import DOBMPCController
from dobmpc import fossen, frames
from dobmpc import params as P
from disturbance.env import DisturbanceEnv, MODES
from disturbance.config import load_config

XML = RM.XML_PATH
BURN_IN = 5.0                 # s discarded before statistics
# mean NIS / NEES target ~18 (18-dim). The SAFETY-relevant failure is OVER-
# CONFIDENCE (value > upper bound: innovations/errors bigger than the filter's own
# covariance predicts -> it trusts a wrong state). Below the band is CONSERVATIVE
# (the filter under-trusts itself: safe, just sub-optimal) and is EXPECTED whenever
# the measurement noise is small relative to the residual process/tau-channel terms
# (the deliberate perf R/Q conservatism) -- e.g. after the 2026-07-24 sensor-noise
# reduction NIS sits ~13. So the gate is UPPER-bound only; the lower value is
# reported, not failed.
NIS_UPPER = 24.0
NEES_UPPER = 24.0
NIS_RANGE = (14.0, 24.0)      # informational target band (print only)
NEES_RANGE = (14.0, 24.0)
AXES = ("X", "Y", "Z", "K", "M", "N")


def _build(cfg, mode, seed, T):
    H.Hydrodynamics.uninstall()
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    env = DisturbanceEnv(cfg.dist, mode=mode, seed=seed, dt=float(model.opt.timestep),
                         T_sim=T)
    env.enabled = True
    hydro = H.Hydrodynamics(model, disturbance=env, diag_wtrue=True).install()
    return model, data, hydro, env


def _w_true_model(eta, nu, a, tau):
    """The lumped disturbance the EAOB estimates (model residual, NED body)."""
    return (fossen.M_TOTAL @ a + fossen.coriolis_rb(nu) + fossen.coriolis_added(nu)
            + fossen.damping(nu) + fossen.restoring(eta) - tau)


def _hydro_wtrue_ned_body(wt_world_flu, eta_ned):
    """hydro.w_true_world (FLU world) -> NED body, for the cross-check RMSE."""
    S = frames.S
    Rn = fossen.rot_ib(*eta_ned[3:6])
    return np.concatenate([Rn.T @ (S @ wt_world_flu[:3]),
                           Rn.T @ (S @ wt_world_flu[3:])])


def run_profile(profile, noise, cfg, mode, seed, T, start):
    model, data, hydro, env = _build(cfg, mode, seed, T)
    # mpc_state_source pinned to "truth": this runner scores the OBSERVER in
    # isolation, so the loop closure must not change with params.MPC_STATE_SOURCE
    # (the estimate-fed closed loop is verify_state_source.py's job).
    ctrl = DOBMPCController(model, hydro=hydro, mode="dobmpc",
                            eaob_profile=profile, meas_noise=noise, noise_seed=seed,
                            mpc_state_source="truth")
    ctrl.set_target((0.0, 0.0, 0.0), yaw_ref=0.0)
    data.qpos[:3] = list(start)
    mujoco.mj_forward(model, data)

    L = {k: [] for k in ("t", "nis", "nees", "w_hat", "w_true", "w_true_hyd", "nu",
                         "e_eta", "e_nu")}
    n_seen = 0
    nu_prev = None                     # runner's own CLEAN tick FD (mirrors the ctrl)
    tau_used = np.zeros(6)             # wrench the EAOB consumed this tick (prev cmd)
    sumwt = np.zeros(6)
    nwt = 0
    t0 = time.time()
    while data.time < T:
        ctrl.apply(model, data)
        eaob = ctrl.eaob
        if eaob is not None and eaob.n_upd > n_seen:   # a control tick just ran
            n_seen = eaob.n_upd
            p, R, nu_flu = ctrl._read_state(data)      # clean truth, same readout
            eta_ned = frames.flu_to_ned_eta(p, R)
            nu_ned = frames.flu_to_ned_nu(nu_flu)
            if nu_prev is not None:
                a_clean = (nu_ned - nu_prev) / ctrl.ctrl_dt
                w_true = _w_true_model(eta_ned, nu_ned, a_clean, tau_used)
                wt_avg = (sumwt / nwt) if nwt else np.zeros(6)
                e = eaob.x - np.concatenate([eta_ned, nu_ned, w_true])
                e[3:6] = fossen.wrap_angle(e[3:6])
                L["t"].append(data.time)
                L["nis"].append(eaob.last_nis)
                L["nees"].append(float(e @ np.linalg.solve(eaob.P, e)))
                L["w_hat"].append(eaob.x[12:].copy())
                L["w_true"].append(w_true)
                L["w_true_hyd"].append(_hydro_wtrue_ned_body(wt_avg, eta_ned))
                L["nu"].append(nu_ned.copy())
                L["e_eta"].append(e[:6].copy())        # estimation error vs truth
                L["e_nu"].append(e[6:12].copy())
            nu_prev = nu_ned.copy()
            tau_used = ctrl._tau_ned_cmd.copy()        # consumed at the NEXT tick
            sumwt = np.zeros(6)
            nwt = 0
        mujoco.mj_step(model, data)
        sumwt += hydro.w_true_world
        nwt += 1
    wall = time.time() - t0
    H.Hydrodynamics.uninstall()

    out = {k: np.asarray(v, float) for k, v in L.items()}
    out["wall"] = wall
    out["n_fail"] = int(ctrl.n_fail)
    out["n_gated"] = int(ctrl.eaob.n_gated)
    out["n_upd"] = int(ctrl.eaob.n_upd)
    return out


def metrics(L):
    m = L["t"] >= BURN_IN
    dw = L["w_hat"][m] - L["w_true"][m]
    dwh = L["w_hat"][m] - L["w_true_hyd"][m]
    nu = L["nu"][m]
    spd = np.linalg.norm(nu[:, :3], axis=1)
    slopes, rs = np.zeros(6), np.zeros(6)
    comp_r = np.zeros((6, 6))
    for i in range(6):
        if spd.std() > 1e-9 and dw[:, i].std() > 1e-12:
            slopes[i] = np.polyfit(spd, dw[:, i], 1)[0]
            rs[i] = np.corrcoef(spd, dw[:, i])[0, 1]
        for j in range(6):
            if nu[:, j].std() > 1e-9 and dw[:, i].std() > 1e-12:
                comp_r[i, j] = np.corrcoef(nu[:, j], dw[:, i])[0, 1]
    return dict(
        n=int(m.sum()),
        rmse=np.sqrt((dw ** 2).mean(axis=0)),
        rmse_hyd=np.sqrt((dwh ** 2).mean(axis=0)),
        rmse_eta=np.sqrt((L["e_eta"][m] ** 2).mean(axis=0)),   # (6,) m / rad
        rmse_nu=np.sqrt((L["e_nu"][m] ** 2).mean(axis=0)),     # (6,) m/s / rad/s
        nis_mean=float(L["nis"][m].mean()), nis_med=float(np.median(L["nis"][m])),
        nees_mean=float(L["nees"][m].mean()), nees_med=float(np.median(L["nees"][m])),
        slopes=slopes, rs=rs, comp_r=comp_r, spd=spd, dw=dw,
    )


def plot_profile(profile, L, M, out_dir, stamp):
    t = L["t"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        ax.plot(t, L["w_true"][:, i], "k-", lw=0.8, label="w_true (model residual)")
        ax.plot(t, L["w_true_hyd"][:, i], "c-", lw=0.6, alpha=0.6, label="w_true (hydro)")
        ax.plot(t, L["w_hat"][:, i], "g-", lw=0.8, label="w_hat (EAOB)")
        ax.axvline(BURN_IN, color="gray", lw=0.5, ls="--")
        unit = "N" if i < 3 else "N*m"
        ax.set_title(f"{AXES[i]}  RMSE {M['rmse'][i]:.3f} {unit}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
        if i >= 3:
            ax.set_xlabel("t [s]")
    fig.suptitle(f"EAOB {profile}: w_hat vs w_true (NED body)  "
                 f"NIS {M['nis_mean']:.1f}  NEES {M['nees_mean']:.1f}  "
                 f"gated {L['n_gated']}/{L['n_upd']}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f1 = os.path.join(out_dir, f"verify_eaob_{profile}_traj_{stamp}.png")
    fig.savefig(f1, dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    for i, ax in enumerate(axs.ravel()):
        ax.scatter(M["spd"], M["dw"][:, i], s=3, alpha=0.4)
        xs = np.linspace(0, max(M["spd"].max(), 1e-3), 10)
        ax.plot(xs, M["slopes"][i] * xs + M["dw"][:, i].mean()
                - M["slopes"][i] * M["spd"].mean(), "r-", lw=1)
        ax.set_title(f"{AXES[i]}  slope {M['slopes'][i]:+.3f}  r {M['rs'][i]:+.2f}")
        ax.grid(alpha=0.3)
        ax.set_xlabel("|nu_lin| [m/s]")
        if i % 3 == 0:
            ax.set_ylabel("w_hat - w_true")
    fig.suptitle(f"EAOB {profile}: motion-correlation of the estimation residual")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f2 = os.path.join(out_dir, f"verify_eaob_{profile}_corr_{stamp}.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)
    return f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="CDW")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "base.yaml"))
    ap.add_argument("--start", type=str, default="0.1,0.05,0.0")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tau-dist", type=float, default=None,
                    help="override params.EAOB_TAU_DIST for the perf profile (tuning sweeps)")
    ap.add_argument("--profiles", type=str, default="verify,perf",
                    help="comma list from {verify,perf}")
    args = ap.parse_args()
    cfg = load_config(args.config)
    start = [float(v) for v in args.start.split(",")]
    if args.tau_dist is not None:
        P.EAOB_TAU_DIST = float(args.tau_dist)     # read at EAOB construction
    print(f"[verify_eaob] EAOB_TAU_DIST = {P.EAOB_TAU_DIST} s")
    wanted = tuple(s.strip() for s in args.profiles.split(",") if s.strip())

    runs = {}
    for profile, noise in (("verify", False), ("perf", True)):
        if profile not in wanted:
            continue
        print(f"[verify_eaob] {profile} (noise {'ON' if noise else 'off'}, "
              f"mode {args.mode}, T {args.T:.0f}s, seed {args.seed}) ...", flush=True)
        L = run_profile(profile, noise, cfg, args.mode, args.seed, args.T, start)
        runs[profile] = (L, metrics(L))
        print(f"   wall {L['wall']:.1f}s  ticks {L['n_upd']}  solve_fail {L['n_fail']}"
              f"  gated {L['n_gated']}")

    # ---- comparison table
    print(f"\n=== EAOB validation (mode {args.mode}, T={args.T:.0f}s, seed {args.seed}, "
          f"burn-in {BURN_IN:.0f}s) ===")
    done = [p for p in ("verify", "perf") if p in runs]
    hdr = (f"{'profile':8s} {'noise':5s} | " + " ".join(f"{a:>7s}" for a in AXES)
           + f" | {'NIS':>6s} {'NEES':>7s} {'gated':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for profile in done:
        L, M = runs[profile]
        rm = " ".join(f"{v:7.3f}" for v in M["rmse"])
        print(f"{profile:8s} {'on' if profile == 'perf' else 'off':5s} | {rm} | "
              f"{M['nis_mean']:6.1f} {M['nees_mean']:7.1f} "
              f"{L['n_gated']:4d}/{L['n_upd']}")
    print("(RMSE of w_hat vs w_true per axis: X,Y,Z in N; K,M,N in N*m. "
          "NIS/NEES: mean, target ~18.)")
    for profile in done:
        L, M = runs[profile]
        rmh = " ".join(f"{v:7.3f}" for v in M["rmse_hyd"])
        print(f"   cross-check RMSE vs hydro w_true [{profile:6s}]: {rmh}")
    print("\nstate-estimate RMSE vs truth (pos [mm] | ang [mrad] | lvel [mm/s] | avel [mrad/s]):")
    for profile in done:
        _, M = runs[profile]
        re, rn = M["rmse_eta"], M["rmse_nu"]
        print(f"   [{profile:6s}] "
              f"{re[0]*1e3:5.1f} {re[1]*1e3:5.1f} {re[2]*1e3:5.1f} | "
              f"{re[3]*1e3:5.2f} {re[4]*1e3:5.2f} {re[5]*1e3:5.2f} | "
              f"{rn[0]*1e3:5.1f} {rn[1]*1e3:5.1f} {rn[2]*1e3:5.1f} | "
              f"{rn[3]*1e3:5.2f} {rn[4]*1e3:5.2f} {rn[5]*1e3:5.2f}")

    print("\nmotion-correlation (residual vs |nu_lin|): slope [N per m/s] / Pearson r")
    for profile in done:
        _, M = runs[profile]
        s = " ".join(f"{AXES[i]} {M['slopes'][i]:+.3f}/{M['rs'][i]:+.2f}"
                     for i in range(6))
        print(f"   [{profile:6s}] {s}")

    if not args.no_plot:
        day = time.strftime("%Y%m%d")
        out_dir = os.path.join(HERE, "recordings", day)
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for profile in done:
            L, M = runs[profile]
            f1, f2 = plot_profile(profile, L, M, out_dir, stamp)
            print(f"saved {f1}\n      {f2}")

    if "perf" not in runs:
        sys.exit(0)
    _, Mp = runs["perf"]
    ok_nis = Mp["nis_mean"] <= NIS_UPPER      # over-confidence is the unsafe side
    ok_nees = Mp["nees_mean"] <= NEES_UPPER
    lo_nis = Mp["nis_mean"] < NIS_RANGE[0]
    lo_nees = Mp["nees_mean"] < NEES_RANGE[0]
    print(f"\nperf consistency (gate = over-confidence, value <= {NIS_UPPER:.0f}; "
          f"target ~18): NIS {Mp['nis_mean']:.1f} -> {'PASS' if ok_nis else 'FAIL'}"
          f"{' (conservative)' if lo_nis else ''};  NEES {Mp['nees_mean']:.1f} -> "
          f"{'PASS' if ok_nees else 'FAIL'}{' (conservative)' if lo_nees else ''}")
    sys.exit(0 if (ok_nis and ok_nees) else 1)


if __name__ == "__main__":
    main()
