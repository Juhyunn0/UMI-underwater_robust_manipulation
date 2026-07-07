#!/usr/bin/env python3
"""Batch disturbance comparison: PID vs MPC vs DOB-MPC x 4 modes x N seeds.

Builds the marinegym plant, injects the finite-depth DisturbanceEnv (current +
drift + finite-depth directional waves + Froude-Krylov inertia; NO kicks), and runs
each controller under each mode with a SHARED seed (bit-identical disturbance) for a
fair comparison. Two scenarios:
  * primary   = DP station-keeping (rejection quantification)
  * secondary = square trajectory (tracking robustness; nu_ref=0 structural lag)

Outputs under recordings/<YYYYMMDD>/compare_<ts>/:
  results.csv (aggregated mean+-std), results_raw.csv (per run), figures/*.png
  (incl. trajectory_compare_<MODE>/_ALLMODES with every sweep heading overlaid),
  runs/traj_*.csv + runs/meta_*.json (per-run trajectory + manifest, run_viewer-
  compatible schema; gate with experiment.record_runs), config snapshot.
  Reuses hydro / controllers / recorder / Disturbance.to_meta().

Usage:
  python -m experiments.run_compare --config config/base.yaml
  python -m experiments.run_compare --config config/base.yaml --smoke
  python -m experiments.run_compare --config config/base.yaml --ctrls pid --seeds 0 --T 20
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time

# Runs are embarrassingly parallel (see --jobs): keep each worker's BLAS/OpenMP
# single-threaded so N processes don't oversubscribe the cores. setdefault -> the
# user can still override. MUST precede `import numpy` to bind the BLAS thread pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # marinegym dir
sys.path.insert(0, HERE)

import hydro as H
import rov_model as RM
from controller import PoseController, DEFAULT_GAINS
from dobmpc_controller import DOBMPCController
from recorder import Recorder, record_row, build_run_meta
from disturbance.env import DisturbanceEnv, MODES
from disturbance.config import load_config

XML = RM.XML_PATH
COL = {"pid": "tab:red", "mpc": "tab:blue", "dobmpc": "tab:green"}
# refined, print-friendly palette for the bar charts (hue-matched to COL, softer)
BAR_COL = {"pid": "#C2444D", "mpc": "#4C86C0", "dobmpc": "#3E9D6F"}
BAR_LABEL = {"pid": "PID", "mpc": "MPC", "dobmpc": "DOB-MPC"}
# trajectory-compare palette (same hues as experiments/plot_trajectories.py); draw
# order worst->best via zorder so DOB-MPC reads on top.
TRAJ_COLOR = {"pid": "#9E2B36", "mpc": "#2B5F9E", "dobmpc": "#2E8B57"}
TRAJ_ZORDER = {"pid": 3, "mpc": 4, "dobmpc": 5}
TRAJ_ALPHA = {"pid": 0.7, "mpc": 0.85, "dobmpc": 0.95}   # single-run alpha
# disturbance-mode key (see disturbance/env.py MODES): C=current only, D=drift, W=waves
MODE_DESC = {"NONE": "still water", "C": "current", "CD": "current + drift",
             "CW": "current + waves", "CDW": "current + drift + waves"}


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


def slew_heading(yaw_ref, tx, ty, rate_rad, dt):
    """Slew the held yaw reference toward the path tangent atan2(ty,tx) at most
    `rate_rad` per second (shortest angle). Smooths the 90-deg corner steps so the
    heading reference is continuous -- the ROV faces its travel direction without an
    instantaneous jump. The POSITION path stays the sharp square (heading only)."""
    target = np.arctan2(ty, tx)
    d = np.arctan2(np.sin(target - yaw_ref), np.cos(target - yaw_ref))   # wrap (-pi,pi]
    step = rate_rad * dt
    return yaw_ref + float(np.clip(d, -step, step))


# --------------------------------------------------------------------- one run
def run_one(ctrl_name, cfg, mode, seed, scenario, scen, dist):
    """Run one (controller, mode, seed, scenario, dist). Returns a log dict of arrays."""
    # run duration: DP = fixed sim time (cfg.sim.T_sim); square = exactly `laps` laps.
    depth = float(scen.get("depth", 0.0))
    if scenario == "square":
        size, speed = float(scen["size"]), float(scen["speed"])
        laps = int(scen.get("laps", 10))
        heading_follow = bool(scen.get("heading_follow", False))   # face travel dir
        yaw_rate = np.radians(float(scen.get("yaw_rate_deg_s", 60.0)))   # corner smoothing
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
    keys = ("t", "px", "py", "pz", "rx", "ry", "yaw", "pitch",
            "wt0", "wt1", "wt2", "wh0", "wh1", "wh2")
    L = {k: [] for k in keys}
    U = []                                            # per-thruster forces at log times
    sumwt = np.zeros(6); nwt = 0                       # tick-average of w_true_world
    last = -log_dt
    yaw_cmd = 0.0                                       # slewed heading ref (square starts +x)
    t0 = time.time()
    while data.time < T:
        if scenario == "square":
            (rx, ry), (tx, ty) = square_setpoint(data.time, size, speed)
            r_cmd = 0.0
            if heading_follow:
                yaw_new = slew_heading(yaw_cmd, tx, ty, yaw_rate, model.opt.timestep)
                r_cmd = (yaw_new - yaw_cmd) / model.opt.timestep   # slew rate = yaw-rate ref
                yaw_cmd = yaw_new
            ctrl.set_target((rx, ry, depth), yaw_ref=(yaw_cmd if heading_follow else 0.0),
                            v_ref=(speed * tx, speed * ty, 0.0), r_ref=r_cmd)
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
            L["yaw"].append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            L["pitch"].append(np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1))))
            wt = (sumwt / nwt) if nwt else np.zeros(6)
            sumwt = np.zeros(6); nwt = 0
            wh = ctrl.w_world_flu()[:3] if ctrl_name == "dobmpc" else np.zeros(3)
            L["wt0"].append(wt[0]); L["wt1"].append(wt[1]); L["wt2"].append(wt[2])
            L["wh0"].append(wh[0]); L["wh1"].append(wh[1]); L["wh2"].append(wh[2])
            U.append(np.asarray(data.ctrl[:nu_act], float).copy())
    wall = time.time() - t0
    n_fail = int(getattr(ctrl, "n_fail", 0))
    try:                                              # rotated theta_c/beta live in here
        env_meta = env.to_meta()
    except Exception as e:
        env_meta = {"error": str(e)}
    H.Hydrodynamics.uninstall()
    out = {k: np.asarray(v, float) for k, v in L.items()}
    out["U"] = np.asarray(U, float)                   # (n_log, nu)
    out["ctrlrange"] = ctrlrange
    out["log_dt"] = log_dt
    out["wall"] = wall
    out["n_fail"] = n_fail
    out["T"] = T                                      # actual sim duration of this run
    out["dt"] = float(model.opt.timestep)
    out["env_meta"] = env_meta                        # for the per-run meta sidecar
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
    """Grouped bar chart: one cluster per disturbance mode, one bar per controller.
    Bars show mean +- std pooled over seeds x current headings."""
    n = len(ctrls)
    x = np.arange(len(modes)); width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.85", lw=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color="0.92", lw=0.5, zorder=0)
    ax.minorticks_on(); ax.tick_params(axis="x", which="minor", bottom=False)

    top = 0.0
    for i, c in enumerate(ctrls):
        means = np.array([agg[(mode, c)][metric][0] for mode in modes])
        stds = np.array([agg[(mode, c)][metric][1] for mode in modes])
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, means, width * 0.90, yerr=stds, capsize=3,
               color=BAR_COL.get(c, COL.get(c)), edgecolor="white",
               linewidth=0.8, zorder=3, label=BAR_LABEL.get(c, c),
               error_kw=dict(ecolor="0.25", elinewidth=1.1, capthick=1.1,
                             alpha=0.9, zorder=4))
        top = max(top, float((means + stds).max()))
        for xb, m, s in zip(x + off, means, stds):
            ax.annotate(f"{m:.1f}", (xb, m + s), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=8, color="0.35")

    ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    xt_labels = [f"{m}\n({MODE_DESC[m]})" if m in MODE_DESC else m for m in modes]
    ax.set_xticks(x); ax.set_xticklabels(xt_labels, fontsize=11)
    ax.set_xlabel("disturbance mode", fontsize=11, color="0.25", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11.5, color="0.2", labelpad=8)
    ax.tick_params(colors="0.35", labelsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("0.6")

    scen = tag.split("_")[0].capitalize()
    metric_name = ylabel.split("[")[0].strip()        # drop unit (kept on y-axis)
    ax.set_title(f"{scen} tracking  —  {metric_name}", fontsize=14,
                 fontweight="bold", color="0.13", pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=n,
              frameon=False, fontsize=10.5, handlelength=1.3, columnspacing=2.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bar_{tag}.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# per-environment disturbance figures -- clean, presentation-quality x/y/z component
# palette (blue/orange/green), used by fig_environment below.
_ENV_COMP = (("x", "#2C6FBB"), ("y", "#E08D2B"), ("z", "#4C9A6E"))


def _env_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.7, zorder=0)
    ax.axhline(0, color="0.7", lw=0.8, zorder=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("0.55")
    ax.tick_params(colors="0.3", labelsize=11)
    ax.set_xlabel("time [s]", fontsize=12, color="0.2", labelpad=6)
    ax.set_ylabel("velocity [m/s]", fontsize=12, color="0.2", labelpad=6)


def _env_finish(fig, ax, title, path):
    ax.set_title(title, fontsize=15, fontweight="bold", color="0.12", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
              fontsize=12, handlelength=1.5, columnspacing=2.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_environment(cur_dir, wav_dir, raw_dir, cfg, theta, beta, stem,
                    mode="CDW", seed=0, wave_window=200.0, save_raw=True):
    """Two clean per-environment figures (controller-independent): (1) ocean CURRENT
    velocity over the full window, (2) WAVE particle velocity over the first
    `wave_window` s. Disturbance rotated to (theta_c=theta, beta=beta). Optionally
    saves the raw time series as a compressed .npz. (Replaces the old 4-panel
    selfcheck; wave elevation + FK-force panels dropped.)"""
    dt = 0.05
    T = max(float(cfg.sim["T_sim"]), 300.0)
    d = dataclasses.replace(cfg.dist, theta_c=float(np.radians(theta)),
                            beta_bar=float(np.radians(beta)))
    env = DisturbanceEnv(d, mode=mode, seed=seed, dt=dt, T_sim=T)
    pos = np.array([0.0, 0.0, float(cfg.experiment["primary"].get("depth", 0.0))])
    ts = np.arange(0.0, T, dt)
    cur = np.array([env.current.current_velocity(t) for t in ts])
    wv = np.array([env.waves.velocity(t, pos) for t in ts])

    # (1) ocean current velocity -- full window
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    _env_axes(ax)
    for i, (lab, col) in enumerate(_ENV_COMP):
        ax.plot(ts, cur[:, i], lw=1.4, color=col, label=f"$v_{lab}$", zorder=3)
    _env_finish(fig, ax, "Ocean current velocity", os.path.join(cur_dir, stem + ".png"))

    # (2) wave particle velocity -- first wave_window seconds (readable on a slide)
    m = ts <= wave_window
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    _env_axes(ax)
    for i, (lab, col) in enumerate(_ENV_COMP):
        ax.plot(ts[m], wv[m, i], lw=1.5, color=col, label=f"$u_{lab}$", zorder=3)
    _env_finish(fig, ax, "Wave-induced particle velocity at ROV",
                os.path.join(wav_dir, stem + ".png"))

    # (3) raw time series (compressed, float32): current over the full window, wave over
    # the plotted window only -- keeps the .npz small (~0.1 MB/env vs ~0.8 MB for full).
    if save_raw:
        np.savez_compressed(
            os.path.join(raw_dir, stem + ".npz"),
            t=ts.astype(np.float32), current=cur.astype(np.float32),
            t_wave=ts[m].astype(np.float32), wave=wv[m].astype(np.float32),
            theta_c_deg=float(theta), beta_deg=float(beta), mode=mode, seed=seed,
            wave_window=float(wave_window))


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


def parse_sweep(cfg):
    """Sweep points as (theta_c_deg, beta_deg) pairs: theta_c = CURRENT heading
    (parse_directions), beta = WAVE heading.

    `directions.pairing: grid` -> the FULL cross product: every current heading x
    every wave heading (explicit `wave_headings_deg`, else `n_random_wave` random
    draws, else the single fixed waves.beta_bar_deg), theta-major order.

    `pairing: paired` (or absent = legacy): one beta per theta. If
    directions.sweep_wave_heading is set, beta is drawn INDEPENDENTLY from its own
    seed -> random (current, wave) pairs with the SAME run count; else beta stays at
    the fixed waves.beta_bar_deg for every point. (NONE/C/CD ignore beta -- only
    CW/CDW have waves -- so the beta draw only adds variety to the wave-active
    modes.)"""
    thetas = parse_directions(cfg)
    d = cfg.experiment.get("directions") or {}
    pairing = str(d.get("pairing", "paired")).lower()
    if pairing not in ("grid", "paired"):
        raise ValueError(f"directions.pairing must be grid|paired, got {pairing!r}")
    if pairing == "grid":
        if d.get("sweep_wave_heading"):
            raise ValueError("directions: pairing=grid and sweep_wave_heading are "
                             "mutually exclusive (grid already sweeps the wave heading)")
        if d.get("wave_headings_deg"):
            betas = [float(b) for b in d["wave_headings_deg"]]
        elif d.get("n_random_wave"):
            rng = np.random.default_rng(int(d.get("wave_heading_seed", 200)))
            betas = [round(float(b), 1)
                     for b in rng.uniform(0.0, 360.0, int(d["n_random_wave"]))]
        else:
            betas = [float((cfg.raw.get("waves") or {}).get("beta_bar_deg", 0.0))]
        return [(t, b) for t in thetas for b in betas]
    if d.get("sweep_wave_heading"):
        if d.get("wave_headings_deg"):                    # explicit list (cycled to length)
            wl = [float(b) for b in d["wave_headings_deg"]]
            betas = [round(wl[i % len(wl)], 1) for i in range(len(thetas))]
        else:
            rng = np.random.default_rng(int(d.get("wave_heading_seed", 200)))
            betas = [round(float(b), 1) for b in rng.uniform(0.0, 360.0, len(thetas))]
    else:
        beta0 = float((cfg.raw.get("waves") or {}).get("beta_bar_deg", 0.0))
        betas = [beta0] * len(thetas)
    return list(zip(thetas, betas))


def _aggregate(metric_dicts):
    """mean+-std over a list of per-run metric dicts -> {metric: (mean, std)}."""
    keys = list(metric_dicts[0].keys())
    return {k: (float(np.mean([m[k] for m in metric_dicts])),
                float(np.std([m[k] for m in metric_dicts]))) for k in keys}


def fig_direction_summary(out_dir, scenario, modes, ctrls, sweep, agg_k, wave_swept=False):
    """radial_rms vs current heading (one line per controller), one panel per mode.
    Points are the sweep indices (each = a (current, wave) heading pair), plotted
    against current heading. Skipped for a single sweep point."""
    if len(sweep) < 2:
        return
    order = sorted(range(len(sweep)), key=lambda i: sweep[i][0])   # by current heading
    xs = [sweep[i][0] for i in order]
    # grid sweeps put several points at the SAME current heading -- connecting them
    # with a line would be misleading, so fall back to markers only
    ls = "none" if len({t for t, _ in sweep}) < len(sweep) else "-"
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for j, mode in enumerate(modes):
        ax = axes[0][j]
        for c in ctrls:
            mu = [agg_k[(i, mode, c)]["radial_rms"][0] for i in order]
            sd = [agg_k[(i, mode, c)]["radial_rms"][1] for i in order]
            ax.errorbar(xs, mu, yerr=sd, marker="o", ms=4, capsize=3, ls=ls,
                        color=COL.get(c), label=c)
        ax.set_xlabel("current heading [deg]"); ax.set_ylabel("radial RMS [cm]")
        ax.set_title(f"[{mode}]"); ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8)
    sub = "  (wave heading also swept per point)" if wave_swept else ""
    fig.suptitle(f"{scenario}: directional sensitivity (radial RMS vs current heading){sub}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, f"direction_summary_{scenario}.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------- trajectory-compare figures
def _sweep_desc(sweep, seeds):
    """Human-readable sweep summary for figure titles."""
    seed_s = f"seed {seeds[0]}" if len(seeds) == 1 else f"{len(seeds)} seeds"
    if len(sweep) == 1:
        t, b = sweep[0]
        return f"current {t:g}°, wave {b:g}°, {seed_s}"
    nc = len({t for t, _ in sweep}); nw = len({b for _, b in sweep})
    if nc * nw == len(sweep):                            # full grid
        return f"{nc} current × {nw} wave headings, {seed_s}"
    return f"{len(sweep)} (current, wave) heading pairs, {seed_s}"


def _load_run_traj(run_dir, scenario, mode, ctrl, seed, theta, beta):
    """Load one runs/traj_<tag>.csv as a named array; None + WARN if missing OR
    unreadable/degenerate (partial file from a failed worker write must degrade to
    a skipped line, never crash the figure stage after the whole batch ran)."""
    path = os.path.join(run_dir,
                        f"traj_{_run_tag(scenario, mode, ctrl, seed, theta, beta)}.csv")
    if not os.path.isfile(path):
        print(f"[fig] WARN: no per-run CSV {os.path.basename(path)}", flush=True)
        return None
    try:
        a = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    except Exception as e:
        print(f"[fig] WARN: unreadable per-run CSV {os.path.basename(path)}: {e}",
              flush=True)
        return None
    return a if a.size else None


def _draw_traj_panel(ax, run_dir, scenario, mode, ctrls, sweep, seeds, S):
    """One mode panel: reference square + EVERY (sweep point x seed) run overlaid
    (controller = color, individual runs = thin translucent lines). Returns
    {ctrl: [per-run full-run radial RMS in cm]} over the runs actually loaded."""
    sq = np.array([[0, 0], [S, 0], [S, S], [0, S], [0, 0]], float)
    ax.plot(sq[:, 0], sq[:, 1], ls="--", lw=1.4, color="#555555", zorder=2)
    ax.scatter(sq[:-1, 0], sq[:-1, 1], s=20, facecolor="white",
               edgecolor="#555555", linewidths=1.0, zorder=2.5)
    N = len(sweep) * len(seeds)                          # runs per controller
    lw = 1.5 if N == 1 else 0.9
    xs, ys, rms = [], [], {}
    for c in ctrls:
        alpha = (TRAJ_ALPHA.get(c, 0.9) if N == 1
                 else float(np.clip(2.0 / np.sqrt(N), 0.15, 0.6)))
        vals = []
        for theta, beta in sweep:
            for seed in seeds:
                a = _load_run_traj(run_dir, scenario, mode, c, seed, theta, beta)
                if a is None:
                    continue
                stride = max(1, a["px"].size // 4000)    # plot-speed decimation
                ax.plot(a["px"][::stride], a["py"][::stride],
                        color=TRAJ_COLOR.get(c, COL.get(c)), lw=lw, alpha=alpha,
                        solid_joinstyle="round", zorder=TRAJ_ZORDER.get(c, 3))
                xs.append(a["px"][::stride]); ys.append(a["py"][::stride])
                vals.append(float(np.sqrt(np.mean(np.hypot(
                    a["px"] - a["rx"], a["py"] - a["ry"]) ** 2))) * 100.0)
        if vals:
            rms[c] = vals
    ax.plot(0, 0, marker="o", ms=7, mfc="#1b1b1b", mec="white", mew=1.0, zorder=6)
    ax.set_aspect("equal", "box")
    ax.grid(True, color="#dddddd", lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("x  [m]", fontsize=11); ax.set_ylabel("y  [m]", fontsize=11)
    ax.set_title(f"{mode}  ({MODE_DESC.get(mode, mode)})",
                 fontsize=12, fontweight="bold", pad=6)
    allx = np.concatenate(xs + [sq[:, 0]]); ally = np.concatenate(ys + [sq[:, 1]])
    pad = 0.10 * S
    lo = min(allx.min(), ally.min()) - pad; hi = max(allx.max(), ally.max()) + pad
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if rms:
        txt = "\n".join((f"{BAR_LABEL.get(c, c)}: {np.mean(rms[c]):.1f} cm" if len(rms[c]) == 1
                         else f"{BAR_LABEL.get(c, c)}: {np.mean(rms[c]):.1f} ± {np.std(rms[c]):.1f} cm")
                        for c in ctrls if c in rms)
        ax.text(0.035, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                        ec="#cccccc", alpha=0.92))
    else:
        ax.text(0.5, 0.5, "(no per-run CSVs)", transform=ax.transAxes,
                ha="center", va="center", color="#888888")
    return rms


def _traj_handles(ctrls, have, N):
    """Legend handles: one thick line per controller that actually has runs,
    the reference square, and (for overlays) a thin-line explainer."""
    handles = [Line2D([0], [0], color=TRAJ_COLOR.get(c, COL.get(c)), lw=3.0,
                      label=BAR_LABEL.get(c, c)) for c in ctrls if c in have]
    handles.append(Line2D([0], [0], ls="--", lw=1.6, color="#555555",
                          label="Reference square"))
    if N > 1:
        handles.append(Line2D([0], [0], color="#999999", lw=0.9, alpha=0.7,
                              label="thin line = one (heading, seed) run"))
    return handles


def _n_txt(rms_dicts, N):
    """'n=N' for the legend, honest when some per-run CSVs failed to load."""
    loaded = [len(v) for r in rms_dicts for v in r.values()]
    return f"n={N}" if loaded and min(loaded) == N else f"n≤{N}"


def fig_trajectory_compare(fig_dir, run_dir, scenario, mode, ctrls, sweep, seeds, S):
    """Single-panel trajectory compare for one mode, ALL sweep directions overlaid
    -> trajectory_compare_<MODE>.png (like the old square_view per-mode figure)."""
    fig, ax = plt.subplots(figsize=(7.6, 7.6), constrained_layout=True)
    rms = _draw_traj_panel(ax, run_dir, scenario, mode, ctrls, sweep, seeds, S)
    N = len(sweep) * len(seeds)
    ax.set_title(f"{RM.MODEL} — square trajectory  (mode {mode}, "
                 f"{_sweep_desc(sweep, seeds)})", fontsize=12, fontweight="bold", pad=10)
    ax.legend(handles=_traj_handles(ctrls, rms, N), loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False,
              handlelength=1.8, columnspacing=1.4, fontsize=9,
              title=f"boxed RMS = full-run radial RMS (all laps), "
                    f"mean ± std over runs ({_n_txt([rms], N)})", title_fontsize=8)
    fig.savefig(os.path.join(fig_dir, f"trajectory_compare_{mode}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_trajectory_allmodes(fig_dir, run_dir, scenario, modes, ctrls, sweep, seeds, S):
    """2x3 overview: one panel per disturbance mode (up to 5) + shared legend in the
    6th cell; every sweep direction x seed overlaid per panel
    -> trajectory_compare_ALLMODES.png (like the old square_view composite)."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), constrained_layout=True)
    axes = axes.ravel()
    shown = list(modes)[:5]
    have, rms_all = set(), []
    for i, mode in enumerate(shown):
        rms = _draw_traj_panel(axes[i], run_dir, scenario, mode, ctrls, sweep, seeds, S)
        have.update(rms.keys())
        rms_all.append(rms)
    for j in range(len(shown), 6):
        axes[j].axis("off")
    N = len(sweep) * len(seeds)
    axes[-1].legend(handles=_traj_handles(ctrls, have, N), loc="center", frameon=False,
                    fontsize=15, handlelength=2.2,
                    title=f"radial RMS per panel: full-run mean ± std "
                          f"over runs ({_n_txt(rms_all, N)})",
                    title_fontsize=11)
    fig.suptitle(f"{RM.MODEL} — square tracking across disturbance modes  "
                 f"({_sweep_desc(sweep, seeds)}, all laps)",
                 fontsize=16, fontweight="bold")
    fig.savefig(os.path.join(fig_dir, "trajectory_compare_ALLMODES.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] trajectory_compare figures ({'+'.join(shown)} + ALLMODES), "
          f"all headings overlaid", flush=True)


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


# ----------------------------------------------------- per-run CSV + meta
def _deg_tag(x):
    """Filename tag for a heading: '045' for integral degrees (grid style),
    '123.4' for random draws."""
    x = float(x)
    return f"{int(round(x)):03d}" if abs(x - round(x)) < 0.05 else f"{x:05.1f}"


def _run_tag(scenario, mode, ctrl, seed, theta, beta):
    return f"{scenario}_{mode}_{ctrl}_seed{seed}_c{_deg_tag(theta)}_w{_deg_tag(beta)}"


def _write_run_outputs(run_dir, scenario, scen, mode, seed, ctrl, theta, beta, L, cfg):
    """Per-run trajectory CSV + meta sidecar under runs/, run_viewer-compatible
    schema (t,px,py,pz,rx,ry,yaw_deg,pitch_deg,lap) so plot_trajectories.py can read
    them standalone. Filenames are unique per task -> parallel workers never collide."""
    tag = _run_tag(scenario, mode, ctrl, seed, theta, beta)
    t = L["t"]
    if scenario == "square":
        size, speed = float(scen["size"]), float(scen["speed"])
        lap = ((speed * t) // (4.0 * size)).astype(int)
    else:
        lap = np.zeros(t.size, dtype=int)
    csv_path = os.path.join(run_dir, f"traj_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "px", "py", "pz", "rx", "ry", "yaw_deg", "pitch_deg", "lap"])
        for i in range(t.size):
            w.writerow([f"{t[i]:.4f}", f"{L['px'][i]:.5f}", f"{L['py'][i]:.5f}",
                        f"{L['pz'][i]:.5f}", f"{L['rx'][i]:.5f}", f"{L['ry'][i]:.5f}",
                        f"{L['yaw'][i]:.3f}", f"{L['pitch'][i]:.3f}", int(lap[i])])
    if scenario == "square":
        traj_meta = dict(kind="square", size=float(scen["size"]), speed=float(scen["speed"]),
                         laps=int(scen.get("laps", 10)), depth=float(scen.get("depth", 0.0)),
                         heading_follow=bool(scen.get("heading_follow", False)),
                         yaw_rate_deg_s=float(scen.get("yaw_rate_deg_s", 60.0)))
    else:
        traj_meta = dict(kind="dp", start=list(scen.get("start", [0.1, 0.05, 0.0])),
                         depth=float(scen.get("depth", 0.0)))
    ctrl_meta = dict(type=ctrl)
    if ctrl == "pid":
        ctrl_meta["pid_gains"] = dict(DEFAULT_GAINS)
    meta = build_run_meta(
        disturbance=L.get("env_meta"),               # rotated theta_c/beta inside
        controller=ctrl_meta,
        trajectory=traj_meta,
        run=dict(mode=mode, seed=int(seed), theta_c_deg=float(theta),
                 beta_deg=float(beta), T=float(L["T"]), dt=float(L["dt"]),
                 log_hz=float(cfg.sim["log_hz"]), wall_s=round(float(L["wall"]), 2),
                 n_fail=int(L["n_fail"]), config=os.path.abspath(cfg.path),
                 csv=os.path.basename(csv_path)),
    )
    with open(os.path.join(run_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ----------------------------------------------------- parallel run driver
def _run_task(task):
    """Worker: run ONE (block, direction, mode, seed, ctrl). Picklable and process-
    safe -- each worker owns its own MuJoCo passive callback + acados solver, so the
    runs are independent. Returns metrics (+ the full log only when a figure needs it)."""
    cfg, name, scenario, scen, mode, seed, ctrl, theta, beta, k, need_log, run_dir = task
    dist = dataclasses.replace(cfg.dist, theta_c=float(np.radians(theta)),
                               beta_bar=float(np.radians(beta)))
    L = run_one(ctrl, cfg, mode, seed, scenario, scen, dist)
    M = metrics(L, cfg, scenario)
    if run_dir:
        try:
            _write_run_outputs(run_dir, scenario, scen, mode, seed, ctrl, theta, beta, L, cfg)
        except Exception as e:                       # metrics still stand; don't kill the run
            tag = _run_tag(scenario, mode, ctrl, seed, theta, beta)
            print(f"[run] WARN per-run CSV/meta failed for {tag}: {e}", flush=True)
            for fn in (f"traj_{tag}.csv", f"meta_{tag}.json"):   # no partial files
                try:
                    os.remove(os.path.join(run_dir, fn))
                except OSError:
                    pass
    r = dict(block=name, k=k, theta=theta, beta=beta, mode=mode, seed=seed, ctrl=ctrl,
             M=M, wall=round(L["wall"], 2), radial=M["radial_rms"], nfail=M["n_fail"])
    if need_log:
        r["log"] = L                                     # only the rep point/seed sends logs back
    return r


def _prebuild_acados(cfg, ctrls):
    """Build the acados solver ONCE in the parent so parallel workers can load it
    (build=False) instead of racing to recompile into the shared _acados_gen dir.
    mpc and dobmpc share the SAME solver (make_nmpc is mode-independent), so either
    triggers the prebuild. Returns True iff the solver is on disk for workers."""
    pre = next((c for c in ("dobmpc", "mpc") if c in ctrls), None)
    if pre is None:
        return False
    try:
        dist0 = dataclasses.replace(cfg.dist, theta_c=0.0)
        m, _d, hy, _en, _bid = build(cfg, "NONE", 0, pre, 10.0, dist0)
        make_controller(pre, m, hy)                      # generate + compile into _acados_gen
        H.Hydrodynamics.uninstall()
        print("[run_compare] acados solver pre-built; workers load it (no rebuild)",
              flush=True)
        return True
    except Exception as e:
        H.Hydrodynamics.uninstall()
        print(f"[run_compare] WARN acados pre-build failed ({type(e).__name__}: {e}); "
              f"workers build per-process", flush=True)
        return False


def _run_all_tasks(tasks, jobs):
    """Run all tasks sequentially (jobs<=1) or across `jobs` forked processes.
    Returns the result dicts (completion order); the caller re-indexes them."""
    n = len(tasks)

    def _report(i, r):
        print(f"  [{i}/{n}] {r['block']}/{r['mode']}/c{r['theta']:.0f}w{r['beta']:.0f}/"
              f"{r['ctrl']} radial={r['radial']:.1f}cm wall={r['wall']:.1f}s "
              f"fail={r['nfail']}", flush=True)

    if jobs <= 1:
        out = []
        for i, t in enumerate(tasks, 1):
            r = _run_task(t); _report(i, r); out.append(r)
        return out

    import multiprocessing as mp
    ctx = mp.get_context("fork")                          # inherit imports; OMP=1 => fork-safe
    out = []
    with ctx.Pool(processes=jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            _report(i, r); out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config", "base.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: T=5s, seed 0, pid only, dp only, single direction")
    ap.add_argument("--ctrls", default=None, help="override controllers, e.g. pid,mpc")
    ap.add_argument("--seeds", default=None, help="override seeds, e.g. 0,1")
    ap.add_argument("--dirs", type=int, default=None,
                    help="override with N random current headings (legacy paired "
                         "sweep; replaces any directions block incl. pairing: grid)")
    ap.add_argument("--T", type=float, default=None, help="override DP T_sim [s]")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes (default: min(cpu,16); 1 = sequential)")
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
        d0 = cfg.experiment.get("directions") or {}
        nd = {"n_random": args.dirs, "direction_seed": d0.get("direction_seed", 100)}
        if d0.get("sweep_wave_heading"):                 # keep the wave-heading sweep on
            nd["sweep_wave_heading"] = True
            nd["wave_heading_seed"] = d0.get("wave_heading_seed", 200)
        cfg.experiment["directions"] = nd
    seed_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    sweep = parse_sweep(cfg)                              # [(theta_c_deg, beta_deg), ...]
    wave_swept = len({b for _, b in sweep}) > 1           # >1 distinct wave heading
    day = time.strftime("%Y%m%d"); ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(HERE, "recordings", day, f"compare_{ts}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    record = bool(cfg.experiment.get("record_runs", True))
    run_dir = os.path.join(out_dir, "runs") if record else None
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
    else:
        print("[run_compare] record_runs=false -> skipping per-run CSVs + "
              "trajectory-compare figures", flush=True)
    _sw = "current+wave headings" if wave_swept else "current headings"
    print(f"[run_compare] {RM.MODEL} plant | ctrls={ctrls} | {len(sweep)} sweep pt(s) | "
          f"{_sw}(deg)="
          f"{[(round(t,1), round(b,1)) for t, b in sweep] if wave_swept else [t for t, _ in sweep]}"
          f" | out={out_dir}", flush=True)

    blocks = [("primary", cfg.experiment.get("primary"))]
    if cfg.experiment.get("secondary"):
        blocks.append(("secondary", cfg.experiment["secondary"]))
    # runs/ CSVs, meta_*.json and figure filenames are scenario-keyed: two enabled
    # blocks with the SAME scenario would silently overwrite (and, in the fork pool,
    # race on) each other's files -- fail loudly instead.
    _scens = [s["scenario"] for _, s in blocks if s]
    if len(set(_scens)) != len(_scens):
        raise ValueError(f"primary and secondary both use scenario {_scens[0]!r}; "
                         "per-run CSVs and figures are scenario-keyed and would "
                         "overwrite each other -- give the blocks distinct scenarios")

    # ---- flat task list: every (block x sweep-point x mode x seed x ctrl) run
    block_meta = {}                                      # name -> (scenario, scen, modes, seeds)
    tasks = []
    est_mb = 0.0                                         # per-run CSV disk estimate
    for name, scen in blocks:
        if scen is None:
            continue
        if seed_override is not None:
            scen = dict(scen); scen["seeds"] = seed_override
        scenario = scen["scenario"]
        modes = list(scen["modes"]); seeds = list(scen["seeds"])
        block_meta[name] = (scenario, scen, modes, seeds)
        if scenario == "square":                         # duration-aware (dp >> square)
            T_est = int(scen.get("laps", 10)) * 4.0 * float(scen["size"]) / float(scen["speed"])
        else:
            T_est = float(cfg.sim["T_sim"])
        est_mb += (len(sweep) * len(modes) * len(seeds) * len(ctrls)
                   * T_est * float(cfg.sim["log_hz"]) * 70e-6)   # ~70 B/row
        for k, (theta, beta) in enumerate(sweep):
            for mode in modes:
                for seed in seeds:
                    for c in ctrls:
                        need_log = (k == 0 and seed == seeds[0])    # rep point/seed -> timehistory
                        tasks.append((cfg, name, scenario, scen, mode, seed, c,
                                      theta, beta, k, need_log, run_dir))

    # ---- pick worker count; pre-build acados once if going parallel
    jobs = args.jobs if args.jobs is not None else min(os.cpu_count() or 1, 16)
    jobs = max(1, min(jobs, len(tasks)))
    if jobs > 1 and _prebuild_acados(cfg, ctrls):
        os.environ["DOBMPC_ACADOS_BUILD"] = "0"          # forked workers load, no rebuild
    print(f"[run_compare] {len(tasks)} runs on {jobs} worker(s)", flush=True)
    if run_dir:
        print(f"[run_compare] record_runs=true -> ~{est_mb:.0f} MB of per-run CSVs "
              f"under {run_dir}", flush=True)
    if len(tasks) > 1000:
        print(f"[run_compare] WARN: {len(tasks)} runs is a LOT (grid sweeps multiply) "
              f"-- expect a long wall time", flush=True)

    # Run-level manifest (meta.json): plant variant + the exact PID gains in effect.
    # make_controller passes no gains= override, so DEFAULT_GAINS *is* the effective
    # set — if an override is ever added there, it must be reflected here too.
    meta = build_run_meta(
        controller=dict(controllers=list(ctrls), pid_gains=dict(DEFAULT_GAINS)),
        run=dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), xml=RM.XML_NAME,
                 n_thrusters=RM.N_THRUSTERS, fully_actuated=RM.FULLY_ACTUATED,
                 config=os.path.abspath(cfg.path), T_sim=float(cfg.sim["T_sim"]),
                 smoke=bool(args.smoke), jobs=jobs),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t_wall0 = time.time()
    res_list = _run_all_tasks(tasks, jobs)
    print(f"[run_compare] all runs done in {time.time() - t_wall0:.0f}s wall", flush=True)

    # ---- re-index results (by sweep-point index k) -> raw_rows + per-point structures
    by_key = {(r["block"], r["k"], r["mode"], r["seed"], r["ctrl"]): r for r in res_list}
    fig_logs = {}                                        # name -> {mode: {ctrl: L}} (rep point/seed)
    for r in res_list:
        if "log" in r:
            fig_logs.setdefault(r["block"], {}).setdefault(r["mode"], {})[r["ctrl"]] = r["log"]

    K = len(sweep)
    raw_rows = []
    results = {}                                         # name -> (scenario, modes, agg_k, agg_overall, sweep)
    for name, (scenario, scen, modes, seeds) in block_meta.items():
        per_run_k = {k: {} for k in range(K)}
        for k, (theta, beta) in enumerate(sweep):
            for mode in modes:
                for seed in seeds:
                    for c in ctrls:
                        r = by_key[(name, k, mode, seed, c)]
                        per_run_k[k][(mode, seed, c)] = r["M"]
                        raw_rows.append(dict(scenario=scenario, direction_deg=round(theta, 1),
                                             wave_deg=round(beta, 1), mode=mode, seed=seed,
                                             controller=c, wall=r["wall"], **r["M"]))
        # aggregate: per sweep-point over seeds; overall over all points x seeds
        agg_k, agg_overall = {}, {}
        for mode in modes:
            for c in ctrls:
                for k in range(K):
                    agg_k[(k, mode, c)] = _aggregate(
                        [per_run_k[k][(mode, s, c)] for s in seeds])
                agg_overall[(mode, c)] = _aggregate(
                    [per_run_k[k][(mode, s, c)] for k in range(K) for s in seeds])
        results[name] = (scenario, modes, agg_k, agg_overall, sweep)

        # figures: representative-point time histories, overall radial bar, direction summary
        flog = fig_logs.get(name, {})
        for mode in modes:
            fig_timehistory(fig_dir, f"{scenario}_{mode}", flog[mode], agg_overall)
        fig_bars(fig_dir, agg_overall, modes, ctrls, "radial_rms",
                 "radial RMS [cm]", f"{scenario}_radial_rms")
        fig_direction_summary(fig_dir, scenario, modes, ctrls, sweep, agg_k, wave_swept)
        # trajectory-compare figures (per mode + ALLMODES composite) from the per-run
        # CSVs, all sweep directions overlaid. Square only -- dp has no trajectory.
        # Guarded: a figure failure must never cost the results.csv written below.
        if scenario == "square" and run_dir:
            try:
                S = float(scen.get("size", 1.0))
                for mode in modes:
                    fig_trajectory_compare(fig_dir, run_dir, scenario, mode, ctrls,
                                           sweep, seeds, S)
                fig_trajectory_allmodes(fig_dir, run_dir, scenario, modes, ctrls,
                                        sweep, seeds, S)
            except Exception as e:
                print(f"[fig] WARN: trajectory-compare figures failed ({e}); "
                      f"continuing to results CSVs", flush=True)

    # per-environment disturbance figures (controller-independent; depend only on each
    # sweep point's (theta_c, beta), seed 0, mode CDW): ocean current + wave particle
    # velocity split into two folders, plus raw .npz time series.
    cur_dir = os.path.join(fig_dir, "selfcheck", "current")
    wav_dir = os.path.join(fig_dir, "selfcheck", "wave")
    raw_dir = os.path.join(fig_dir, "selfcheck", "raw")
    for d in (cur_dir, wav_dir, raw_dir):
        os.makedirs(d, exist_ok=True)
    for k, (theta, beta) in enumerate(sweep):
        fig_environment(cur_dir, wav_dir, raw_dir, cfg, theta, beta,
                        stem=f"env_{k:02d}_c{theta:.1f}_w{beta:.1f}")
    print(f"[run_compare] {len(sweep)} per-environment current/wave figures (+raw) "
          f"under {os.path.join(fig_dir, 'selfcheck')}", flush=True)

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
            for name, (scenario, modes, agg_k, agg_overall, sweep) in results.items():
                if len(sweep) > 1:
                    # wave heading swept too -> label both, else rows with equal
                    # theta (grid) would collide/ambiguate
                    multi_w = len({b for _, b in sweep}) > 1
                    for ki, (theta, beta) in enumerate(sweep):
                        lbl = f"c{theta:.1f}/w{beta:.1f}" if multi_w else f"{theta:.1f}"
                        for mode in modes:
                            for c in ctrls:
                                for mk, (mu, sd) in agg_k[(ki, mode, c)].items():
                                    w.writerow([scenario, lbl, mode, c, mk,
                                                f"{mu:.4f}", f"{sd:.4f}"])
                for mode in modes:
                    for c in ctrls:
                        for mk, (mu, sd) in agg_overall[(mode, c)].items():
                            w.writerow([scenario, "all", mode, c, mk, f"{mu:.4f}", f"{sd:.4f}"])
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
