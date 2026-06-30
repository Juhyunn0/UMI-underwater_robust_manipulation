#!/usr/bin/env python3
"""Batch disturbance comparison: PID vs MPC vs DOB-MPC x 4 modes x N seeds.

Builds the marinegym plant, injects the finite-depth DisturbanceEnv (current +
drift + finite-depth directional waves + Froude-Krylov inertia; NO kicks), and runs
each controller under each mode with a SHARED seed (bit-identical disturbance) for a
fair comparison. Two scenarios:
  * primary   = DP station-keeping (rejection quantification)
  * secondary = square trajectory (tracking robustness; nu_ref=0 structural lag)

Outputs under recordings/<YYYYMMDD>/compare_<ts>/:
  results.csv (aggregated mean+-std), results_raw.csv (per run), figures/*.png,
  config snapshot. Reuses hydro / controllers / recorder / Disturbance.to_meta().

Usage:
  python -m experiments.run_compare --config config/base.yaml
  python -m experiments.run_compare --config config/base.yaml --smoke
  python -m experiments.run_compare --config config/base.yaml --ctrls pid --seeds 0 --T 20
"""
import argparse
import csv
import dataclasses
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
from controller import PoseController
from dobmpc_controller import DOBMPCController
from recorder import Recorder, record_row, build_run_meta
from disturbance.env import DisturbanceEnv, MODES
from disturbance.config import load_config

XML = RM.XML_PATH
COL = {"pid": "tab:red", "mpc": "tab:orange", "dobmpc": "tab:green"}


# --------------------------------------------------------------------- build
def build(cfg, mode, seed, ctrl_name, t_sim_env, dist):
    """Fresh model/data/hydro/env. Uninstalls any stale passive callback FIRST
    (from_xml_path runs an internal forward that fires the global mjcb_passive).
    t_sim_env = the actual run duration, so the env's precomputed GM drift sequence
    always covers the run (DP uses cfg.sim.T_sim; square is laps-bounded).
    `dist` is the (possibly direction-rotated) DistConfig for this run."""
    H.Hydrodynamics.uninstall()
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    env = DisturbanceEnv(dist, mode=mode, seed=seed, dt=dt, T_sim=t_sim_env)
    env.enabled = True
    hydro = H.Hydrodynamics(model, disturbance=env,
                            diag_wtrue=(ctrl_name == "dobmpc")).install()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return model, data, hydro, env, bid


def make_controller(ctrl_name, model, hydro):
    if ctrl_name == "pid":
        return PoseController(model, mode="pid", buoyancy_ff=hydro, actuator=None)
    if ctrl_name in ("mpc", "dobmpc"):
        return DOBMPCController(model, hydro=hydro, mode=ctrl_name, actuator=None)
    raise ValueError(ctrl_name)


# ------------------------------------------------------------- square scenario
def square_setpoint(t, size, speed):
    """(point2, tangent2) at arclength s=speed*t around the CCW square (origin corner)."""
    P = 4.0 * size
    s = (speed * t) % P
    S = size
    if s < S:
        return (s, 0.0), (1.0, 0.0)
    if s < 2 * S:
        return (S, s - S), (0.0, 1.0)
    if s < 3 * S:
        return (3 * S - s, S), (-1.0, 0.0)
    return (0.0, 4 * S - s), (0.0, -1.0)


# --------------------------------------------------------------------- one run
def run_one(ctrl_name, cfg, mode, seed, scenario, scen, dist):
    """Run one (controller, mode, seed, scenario, dist). Returns a log dict of arrays."""
    # run duration: DP = fixed sim time (cfg.sim.T_sim); square = exactly `laps` laps.
    depth = float(scen.get("depth", 0.0))
    if scenario == "square":
        size, speed = float(scen["size"]), float(scen["speed"])
        laps = int(scen.get("laps", 10))
        T = laps * (4.0 * size) / speed              # perimeter 4*size; time = laps*P/speed
    else:
        T = float(cfg.sim["T_sim"])

    model, data, hydro, env, bid = build(cfg, mode, seed, ctrl_name, T, dist)
    ctrl = make_controller(ctrl_name, model, hydro)
    ctrl.reset()
    nu_act = model.nu
    ctrlrange = np.asarray(model.actuator_ctrlrange[:nu_act], float)   # (nu,2) [lo,hi]

    if scenario == "dp":
        start = np.asarray(scen.get("start", [0.1, 0.05, 0.0]), float)
        data.qpos[:3] = start
        ctrl.set_target((0.0, 0.0, depth), yaw_ref=0.0, v_ref=(0.0, 0.0, 0.0))
    else:                                            # square: start at the origin corner
        data.qpos[:3] = [0.0, 0.0, depth]
    mujoco.mj_forward(model, data)

    log_dt = 1.0 / float(cfg.sim["log_hz"])
    keys = ("t", "px", "py", "pz", "rx", "ry", "pitch",
            "wt0", "wt1", "wt2", "wh0", "wh1", "wh2")
    L = {k: [] for k in keys}
    U = []                                            # per-thruster forces at log times
    sumwt = np.zeros(6); nwt = 0                       # tick-average of w_true_world
    last = -log_dt
    t0 = time.time()
    while data.time < T:
        if scenario == "square":
            (rx, ry), (tx, ty) = square_setpoint(data.time, size, speed)
            ctrl.set_target((rx, ry, depth), yaw_ref=0.0,
                            v_ref=(speed * tx, speed * ty, 0.0))
        else:
            rx, ry = 0.0, 0.0
        ctrl.apply(model, data)
        mujoco.mj_step(model, data)
        if hydro.diag_wtrue:
            sumwt += hydro.w_true_world; nwt += 1
        if data.time - last >= log_dt:
            last = data.time
            p = np.asarray(data.xpos[bid], float)
            R = np.asarray(data.xmat[bid], float).reshape(3, 3)
            L["t"].append(data.time)
            L["px"].append(p[0]); L["py"].append(p[1]); L["pz"].append(p[2])
            L["rx"].append(rx); L["ry"].append(ry)
            L["pitch"].append(np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1))))
            wt = (sumwt / nwt) if nwt else np.zeros(6)
            sumwt = np.zeros(6); nwt = 0
            wh = ctrl.w_world_flu()[:3] if ctrl_name == "dobmpc" else np.zeros(3)
            L["wt0"].append(wt[0]); L["wt1"].append(wt[1]); L["wt2"].append(wt[2])
            L["wh0"].append(wh[0]); L["wh1"].append(wh[1]); L["wh2"].append(wh[2])
            U.append(np.asarray(data.ctrl[:nu_act], float).copy())
    wall = time.time() - t0
    n_fail = int(getattr(ctrl, "n_fail", 0))
    H.Hydrodynamics.uninstall()
    out = {k: np.asarray(v, float) for k, v in L.items()}
    out["U"] = np.asarray(U, float)                   # (n_log, nu)
    out["ctrlrange"] = ctrlrange
    out["log_dt"] = log_dt
    out["wall"] = wall
    out["n_fail"] = n_fail
    out["T"] = T                                      # actual sim duration of this run
    return out


# --------------------------------------------------------------------- metrics
def _band_rms(x, log_dt, lo, hi):
    """RMS contribution of x(t) in the angular-frequency band [lo, hi) rad/s (cm if
    x is in cm). One-sided power via rfft with Parseval weighting."""
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 0.0
    X = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(n, d=log_dt)              # Hz
    omega = 2.0 * np.pi * freqs
    w = np.full(X.size, 2.0); w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    power = w * (np.abs(X) ** 2) / (n ** 2)           # per-bin variance contribution
    mask = (omega >= lo) & (omega < hi)
    return float(np.sqrt(power[mask].sum()))


def metrics(L, cfg, scenario):
    t = L["t"]
    band = float(cfg.experiment.get("settle_band_cm", 5.0)) / 100.0
    # steady-state window: settle_s, but never past half the run (keeps short/smoke
    # runs non-empty); fall back to the whole record if still too few samples.
    t_end = float(t[-1]) if t.size else 0.0
    settle = min(float(cfg.experiment.get("settle_s", 10.0)), 0.5 * t_end)
    ex = L["px"] - L["rx"]                            # tracking error (m)
    ey = L["py"] - L["ry"]
    rxy = np.hypot(ex, ey)
    m = t >= settle
    if m.sum() < 3:
        m = np.ones_like(t, dtype=bool)
    U = L["U"]; log_dt = L["log_dt"]
    ranges = L["ctrlrange"]
    lo = ranges[:, 0]; hi = ranges[:, 1]

    # settling time: last time rxy >= band, then the next sample (0 if always in band)
    above = np.where(rxy >= band)[0]
    settling = float(t[above[-1] + 1]) if (above.size and above[-1] + 1 < t.size) else 0.0

    # control effort / slew / saturation
    effort = float((U ** 2).sum(axis=1).sum() * log_dt)
    slew = float(np.abs(np.diff(U, axis=0)).sum()) if U.shape[0] > 1 else 0.0
    sat = (U >= 0.98 * hi[None, :]) | (U <= 0.98 * lo[None, :])
    sat_freq = float(sat.any(axis=1).mean())

    # estimation error (dobmpc only): ||w_true - w_hat|| over the steady window
    wt = np.stack([L["wt0"], L["wt1"], L["wt2"]], axis=1)
    wh = np.stack([L["wh0"], L["wh1"], L["wh2"]], axis=1)
    we = np.linalg.norm((wt - wh)[m], axis=1)
    est_err_rms = float(np.sqrt((we ** 2).mean())) if m.any() else 0.0

    out = dict(
        radial_rms=float(np.sqrt((rxy[m] ** 2).mean())) * 100,
        radial_max=float(rxy.max()) * 100,
        dc_ex=float(ex[m].mean()) * 100, dc_ey=float(ey[m].mean()) * 100,
        dc_radial=float(np.hypot(ex[m].mean(), ey[m].mean())) * 100,  # direction-agnostic DC bias
        std_x=float(ex[m].std()) * 100, std_y=float(ey[m].std()) * 100,
        depth_std=float((L["pz"] - 0.0)[m].std()) * 100,
        pitch_mean=float(L["pitch"][m].mean()), pitch_max=float(np.abs(L["pitch"]).max()),
        IAE=float(np.trapz(rxy, t)) * 100,
        IAE_norm=float(rxy[m].mean()) * 100,             # mean |error| in steady window
        settling_time=settling,
        ss_error=float(rxy[m].mean()) * 100,
        control_effort=effort,
        slew=slew,
        sat_freq=sat_freq,
        band_dc=_band_rms(ex[m] * 100, log_dt, 0.0, 0.05),
        band_wave=_band_rms(ex[m] * 100, log_dt, 0.3, 1.7),
        band_high=_band_rms(ex[m] * 100, log_dt, 5.0, 1e3),
        est_err_rms=est_err_rms,
        w_x=float(wh[m, 0].mean()), w_y=float(wh[m, 1].mean()),
        n_fail=L["n_fail"],
    )
    return out


# --------------------------------------------------------------------- figures
def fig_timehistory(out_dir, mode, logs, mets):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(2, 2, 1)
    for c, L in logs.items():
        ax.plot(L["t"], np.hypot(L["px"] - L["rx"], L["py"] - L["ry"]) * 100,
                lw=.7, color=COL.get(c), label=c)
    ax.set_xlabel("t [s]"); ax.set_ylabel("radial error [cm]")
    ax.set_title(f"[{mode}] tracking error"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 2)
    for c, L in logs.items():
        ax.plot(L["t"], L["pitch"], lw=.7, color=COL.get(c), label=c)
    ax.set_xlabel("t [s]"); ax.set_ylabel("pitch [deg]")
    ax.set_title(f"[{mode}] pitch"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 3)
    for c, L in logs.items():
        ax.plot(L["t"], (L["U"] ** 2).sum(axis=1), lw=.6, color=COL.get(c), label=c)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\sum u^2$ [N$^2$]")
    ax.set_title(f"[{mode}] control effort"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 4)
    if "dobmpc" in logs:
        L = logs["dobmpc"]
        ax.plot(L["t"], L["wt0"], lw=.7, color="k", label=r"$w_{true,x}$")
        ax.plot(L["t"], L["wh0"], lw=.7, color="tab:green", label=r"$\hat w_x$")
        ax.plot(L["t"], L["wt1"], lw=.5, color="gray", label=r"$w_{true,y}$")
        ax.plot(L["t"], L["wh1"], lw=.5, color="tab:olive", label=r"$\hat w_y$")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "(DOB-MPC only)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]")
    ax.set_title(f"[{mode}] EAOB estimate vs ground truth (FLU world)")
    ax.grid(alpha=.3)

    fig.suptitle(f"Disturbance mode {mode}: PID vs MPC vs DOB-MPC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"timehistory_{mode}.png"), dpi=110)
    plt.close(fig)


def fig_bars(out_dir, agg, modes, ctrls, metric, ylabel, tag):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(modes)); width = 0.8 / max(1, len(ctrls))
    for i, c in enumerate(ctrls):
        means = [agg[(mode, c)][metric][0] for mode in modes]
        stds = [agg[(mode, c)][metric][1] for mode in modes]
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=c, color=COL.get(c))
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (mean +- std over seeds)")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bar_{tag}.png"), dpi=110)
    plt.close(fig)


def fig_selfcheck(out_dir, cfg, dist=None, mode="CDW", seed=0):
    """Controller-independent disturbance self-check. Sampled over a fixed long window
    (>= a few swell periods) so Hs / spectra are representative regardless of T_sim.
    `dist` (default cfg.dist) lets the caller show a direction-rotated current."""
    dt = 0.05
    T = max(float(cfg.sim["T_sim"]), 300.0)
    env = DisturbanceEnv(dist if dist is not None else cfg.dist,
                         mode=mode, seed=seed, dt=dt, T_sim=T)
    pos = np.array([0.0, 0.0, float(cfg.experiment["primary"].get("depth", 0.0))])
    ts = np.arange(0.0, T, dt)
    cur = np.array([env.current.current_velocity(t) for t in ts])
    eta = np.array([env.waves.elevation(t, pos) for t in ts])
    wv = np.array([env.waves.velocity(t, pos) for t in ts])
    F = np.array([env.external_wrench(t, pos)[0] for t in ts])

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(2, 2, 1)
    for i, lab in enumerate(("x", "y", "z")):
        ax.plot(ts, cur[:, i], lw=.7, label=f"current_{lab}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[m/s]"); ax.set_title("current velocity (mean+drift)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(ts, eta, lw=.6, color="tab:blue")
    ax.set_xlabel("t [s]"); ax.set_ylabel("eta [m]")
    ax.set_title(f"wave elevation (4*std={4*eta.std():.2f} m vs Hs={cfg.dist.Hs})")
    ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 3)
    for i, lab in enumerate(("x", "y", "z")):
        ax.plot(ts, wv[:, i], lw=.6, label=f"u_{lab}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[m/s]"); ax.set_title("wave particle velocity @ ROV")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(ts, np.linalg.norm(F, axis=1), lw=.6, color="tab:purple")
    ax.set_xlabel("t [s]"); ax.set_ylabel("|F| [N]")
    ax.set_title("Froude-Krylov inertia force magnitude")
    ax.grid(alpha=.3)

    fig.suptitle(f"Disturbance self-check (mode {mode}, seed {seed})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "selfcheck_disturbance.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------- driver
def parse_directions(cfg):
    """Current-heading sweep (deg). `experiment.directions.headings_deg` takes
    precedence; else `n_random` headings drawn from `direction_seed`; else [0.0]
    (single direction = legacy behaviour). Only the CURRENT heading rotates."""
    d = cfg.experiment.get("directions")
    if not d:
        return [0.0]
    if d.get("headings_deg"):
        return [float(h) for h in d["headings_deg"]]
    k = int(d.get("n_random", 1))
    rng = np.random.default_rng(int(d.get("direction_seed", 0)))
    return [round(float(h), 1) for h in rng.uniform(0.0, 360.0, k)]


def _aggregate(metric_dicts):
    """mean+-std over a list of per-run metric dicts -> {metric: (mean, std)}."""
    keys = list(metric_dicts[0].keys())
    return {k: (float(np.mean([m[k] for m in metric_dicts])),
                float(np.std([m[k] for m in metric_dicts]))) for k in keys}


def fig_direction_summary(out_dir, scenario, modes, ctrls, directions, agg_dir):
    """radial_rms vs current heading (one line per controller), one panel per mode.
    The headline plot for the direction sweep. Skipped for a single direction."""
    if len(directions) < 2:
        return
    dirs = sorted(directions)
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for j, mode in enumerate(modes):
        ax = axes[0][j]
        for c in ctrls:
            mu = [agg_dir[(h, mode, c)]["radial_rms"][0] for h in dirs]
            sd = [agg_dir[(h, mode, c)]["radial_rms"][1] for h in dirs]
            ax.errorbar(dirs, mu, yerr=sd, marker="o", ms=4, capsize=3,
                        color=COL.get(c), label=c)
        ax.set_xlabel("current heading [deg]"); ax.set_ylabel("radial RMS [cm]")
        ax.set_title(f"[{mode}]"); ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{scenario}: directional sensitivity (radial RMS vs current heading)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, f"direction_summary_{scenario}.png"), dpi=110)
    plt.close(fig)


def run_block(cfg, scenario, scen, ctrls, raw_rows, dist, direction_deg):
    """Run one (direction, scenario block): modes x seeds x ctrls with the rotated
    `dist`. Returns per_run[(mode,seed,ctrl)] -> metrics and fig_logs[mode][ctrl]
    (seed0 logs, for the time-history figures)."""
    modes = list(scen["modes"]); seeds = list(scen["seeds"])
    per_run, fig_logs = {}, {}
    for mode in modes:
        for seed in seeds:
            for c in ctrls:
                tag = f"{scenario}/dir{direction_deg:.0f}/{mode}/seed{seed}/{c}"
                print(f"[run] {tag} ...", flush=True)
                L = run_one(c, cfg, mode, seed, scenario, scen, dist)
                M = metrics(L, cfg, scenario)
                per_run[(mode, seed, c)] = M
                raw_rows.append(dict(scenario=scenario, direction_deg=round(direction_deg, 1),
                                     mode=mode, seed=seed, controller=c,
                                     wall=round(L["wall"], 2), **M))
                print(f"      radial_rms={M['radial_rms']:.1f}cm  dc_radial={M['dc_radial']:.1f}cm"
                      f"  est_err={M['est_err_rms']:.1f}N  fail={M['n_fail']}  "
                      f"(sim {L['T']:.0f}s, wall {L['wall']:.1f}s)", flush=True)
                if seed == seeds[0]:
                    fig_logs.setdefault(mode, {})[c] = L
    return per_run, fig_logs, modes, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config", "base.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: T=5s, seed 0, pid only, dp only, single direction")
    ap.add_argument("--ctrls", default=None, help="override controllers, e.g. pid,mpc")
    ap.add_argument("--seeds", default=None, help="override seeds, e.g. 0,1")
    ap.add_argument("--dirs", type=int, default=None,
                    help="override the number of random current headings")
    ap.add_argument("--T", type=float, default=None, help="override DP T_sim [s]")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.T is not None:
        cfg.sim["T_sim"] = args.T
    ctrls = cfg.experiment.get("controllers", ["pid", "mpc", "dobmpc"])
    if args.ctrls:
        ctrls = args.ctrls.split(",")
    if args.smoke:
        cfg.sim["T_sim"] = args.T if args.T is not None else 5.0
        ctrls = args.ctrls.split(",") if args.ctrls else ["pid"]
        cfg.experiment["primary"]["seeds"] = [0]
        cfg.experiment["secondary"] = None
        cfg.experiment["directions"] = None              # single direction unless --dirs
    if args.dirs is not None:                            # overrides smoke's single dir
        ds = (cfg.experiment.get("directions") or {}).get("direction_seed", 100)
        cfg.experiment["directions"] = {"n_random": args.dirs, "direction_seed": ds}
    seed_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    directions = parse_directions(cfg)
    day = time.strftime("%Y%m%d"); ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "recordings", day, f"compare_{ts}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"[run_compare] {RM.MODEL} plant | ctrls={ctrls} | "
          f"current headings(deg)={directions} | out={out_dir}", flush=True)

    raw_rows = []
    blocks = [("primary", cfg.experiment.get("primary"))]
    if cfg.experiment.get("secondary"):
        blocks.append(("secondary", cfg.experiment["secondary"]))

    results = {}                                          # name -> (scenario, modes, agg_dir, agg_overall)
    for name, scen in blocks:
        if scen is None:
            continue
        if seed_override is not None:
            scen = dict(scen); scen["seeds"] = seed_override
        scenario = scen["scenario"]
        modes = list(scen["modes"]); seeds = list(scen["seeds"])
        print(f"\n=== block {name}: scenario={scenario} modes={modes} "
              f"seeds={seeds} dirs={directions} ===", flush=True)

        per_run_dir, fig_logs_rep = {}, None             # direction -> per_run; rep-dir logs
        for di, h in enumerate(directions):
            dist = dataclasses.replace(cfg.dist, theta_c=float(np.radians(h)))
            per_run, fig_logs, modes, seeds = run_block(
                cfg, scenario, scen, ctrls, raw_rows, dist, h)
            per_run_dir[h] = per_run
            if di == 0:
                fig_logs_rep = fig_logs

        # aggregate: per (direction, mode, ctrl) over seeds; overall over directions x seeds
        agg_dir, agg_overall = {}, {}
        for mode in modes:
            for c in ctrls:
                for h in directions:
                    agg_dir[(h, mode, c)] = _aggregate(
                        [per_run_dir[h][(mode, s, c)] for s in seeds])
                agg_overall[(mode, c)] = _aggregate(
                    [per_run_dir[h][(mode, s, c)] for h in directions for s in seeds])
        results[name] = (scenario, modes, agg_dir, agg_overall)

        # figures: representative-direction time histories, overall radial bar, direction summary
        for mode in modes:
            fig_timehistory(fig_dir, f"{scenario}_{mode}", fig_logs_rep[mode], agg_overall)
        fig_bars(fig_dir, agg_overall, modes, ctrls, "radial_rms",
                 "radial RMS [cm]", f"{scenario}_radial_rms")
        fig_direction_summary(fig_dir, scenario, modes, ctrls, directions, agg_dir)

    # self-check for the first heading (shows the rotated current x/y)
    rep_dist = dataclasses.replace(cfg.dist, theta_c=float(np.radians(directions[0])))
    fig_selfcheck(fig_dir, cfg, dist=rep_dist)

    # ---- write CSVs
    if raw_rows:
        raw_path = os.path.join(out_dir, "results_raw.csv")
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader(); w.writerows(raw_rows)
        # aggregated long-format: per-direction (if >1) + overall (all directions) + DRR
        agg_path = os.path.join(out_dir, "results.csv")
        with open(agg_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "direction", "mode", "controller", "metric", "mean", "std"])
            for name, (scenario, modes, agg_dir, agg_overall) in results.items():
                if len(directions) > 1:
                    for h in directions:
                        for mode in modes:
                            for c in ctrls:
                                for k, (mu, sd) in agg_dir[(h, mode, c)].items():
                                    w.writerow([scenario, f"{h:.1f}", mode, c, k,
                                                f"{mu:.4f}", f"{sd:.4f}"])
                for mode in modes:
                    for c in ctrls:
                        for k, (mu, sd) in agg_overall[(mode, c)].items():
                            w.writerow([scenario, "all", mode, c, k, f"{mu:.4f}", f"{sd:.4f}"])
                    if "mpc" in ctrls and "dobmpc" in ctrls:    # pure DOB contribution
                        rm = agg_overall[(mode, "mpc")]["radial_rms"][0]
                        rd = agg_overall[(mode, "dobmpc")]["radial_rms"][0]
                        drr = rm / rd if rd > 1e-9 else float("nan")
                        w.writerow([scenario, "all", mode, "mpc/dobmpc", "DRR", f"{drr:.4f}", "0"])
        print(f"\n[run_compare] wrote {agg_path}\n               {raw_path}", flush=True)

    # config snapshot
    try:
        import shutil
        shutil.copy(cfg.path, os.path.join(out_dir, "config.yaml"))
    except Exception as e:
        print(f"[run_compare] WARN copy config: {e}")
    print(f"[run_compare] figures in {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
