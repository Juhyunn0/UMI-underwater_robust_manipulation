#!/usr/bin/env python3
"""Attitude-reference validation of the sim DOB-MPC twin (design S2-S7, 2026-09-26).

The Heavy NMPC now takes a roll/pitch REFERENCE (dobmpc_controller.set_target
roll_ref/pitch_ref, FLU rad) and commands the K/M torques the 4 vertical thrusters
realize. This script closes the loop on the MuJoCo plant and writes one artefact
folder per invocation so every attitude number in the docs can cite a path:

    recordings/<YYYYMMDD>/attitude_hold_<HHMMSS>/
        results.csv          one row per run (metrics below + accept flag)
        runs/traj_<tag>.csv  20 Hz tick log: t, roll/pitch/yaw_deg, roll/pitch_ref_deg,
                             px,py,pz, X,Y,Z,K,M,N (commanded FLU), K_app,M_app (after the
                             scenario plugin), u0..u7 (allocated thruster N, pre-clip),
                             sum_abs_f, sat, wr_hat, wp_hat (EAOB w_hat[3:5] NED), status
        runs/meta_<tag>.json per-run manifest (recorder.build_run_meta schema)
        meta.json            folder manifest (args, rov_model, u_max, mpc_q, eaob, git)

Scenarios (--scenarios, comma list; every one ROV_MODEL-aware, read at import):
  hold      S2  (roll,pitch) grid x ctrl x mode, T s, steady window t >= settle (the
                run_compare.steady_mask definition). Metrics: e_roll/e_pitch mean/RMS/max
                (deg), radial + z position RMS (coupling, cm), commanded K/M mean (N*m),
                per-thruster max|f| + saturation ticks (ctrlrange), mean sum|f_i| (THRUST
                COST) and its delta vs the same ctrl's level hold, w_hat[3:5] mean,
                n_fail/last_status. accept = |e|_rms < 1 deg AND n_fail 0 AND sat 0 [예측].
  step      S3  level -> (0, step_deg) at t_step: overshoot %, settling time (1 deg band),
                damping ratio from the overshoot [유도].
  ramp      S3  level -> step_deg at pq_max rad/s through the SETPOINT body (no preview,
                = the station's hold path): peak error during the ramp, lag = mean(e)/w.
  dropped   S4  K/M zeroed AFTER the solver while the EAOB is credited the commanded
                K/M (the unknown-firmware case): reference (0, dropped_deg) from t = 0,
                attitude stays passive, position drift, t_div_rp_s = the tick on which
                |e_rp| > div_deg has held for 0.5 s (the station's div_rp interlock trip,
                config policy.div_max_rp_deg 15). dropped_deg (default 20 = the station's
                rp_max_deg clip) MUST exceed div_deg -- with the two equal, |e| of a
                passive hull sits ON the threshold and the trip time is a noise
                crossing (the 2026-09-26 174607 artefact); build_specs refuses that.
  signinv   S5  K/M sign negated: t_abort = first |rp| > abort_deg, then the wrench is
                zeroed (neutral) and t_recover = time back inside 2 deg [예측].
  gain      S5  applied K/M scaled 0.5x / 2x (the DOB multiplicative-uncertainty check).
  deadband  S5  per-thruster |f| < knee -> 0 (T200 deadband plugin): limit-cycle
                amplitude (steady peak-to-peak pitch error) + dominant frequency.
  headroom  S6  (0,0) and (0, step_deg) under CDW: max |X|,|Z|, ticks at the 30 N box,
                mean X/Z, predicted heave-trim residual NET_BUOYANCY*sin(theta) [유도].
  rate      S7  6-tuple SAMPLER ramp at rate_rad_s to step_deg, att_rate_source none vs
                fd_horizon_T_inv: peak error, lag, K/M chatter (mean |dM| per tick).

Which samples a metric scores (results.csv columns):
  steady window  e_roll_*/e_pitch_* (mean/rms/max), rad_rms/z_rms/yaw_rms, K_mean/M_mean,
                 sum_f_mean (+ thrust_cost_vs_level), wr_hat/wp_hat mean, and accept.
                 Scenarios whose reference is constant from t = 0 (hold, dropped, signinv,
                 gain, deadband, headroom): steady_mask(t, settle) = t >= settle.
                 Scenarios whose reference MOVES at t_step (step, ramp, rate): the same
                 steady_mask rule applied to t - t_step, i.e. t >= t_step + settle, clamped
                 by steady_mask to the second half of the post-step record when the run is
                 too short (whole record if < 3 samples). The window start actually used
                 is written to `extra` as steady_from_s and to meta run.steady_from_s.
                 (Before 2026-09-26 (2) these three scored t >= settle = t_step, so the
                 transient sat inside the "steady" window: step e_pitch_rms 1.65 deg,
                 accept 0 in 174607/results.csv -- do not compare those rows' e_* with new.)
  whole run      max_abs_f_n, sat_ticks, e_rp_max_all_deg, n_fail/last_status, finite,
                 and every transient number in `extra` (overshoot, settle, ramp peak/lag,
                 t_div_rp_s, t_abort/t_recover, rad_max_cm).

Run (robust env; heavy is the default ROV_MODEL):
    python verify/verify_attitude_hold.py                       # S2 grid, NONE, mpc+dobmpc
    ROV_MODEL=heavy_gripper python verify/verify_attitude_hold.py --grid 0,20 --scenarios hold
    python verify/verify_attitude_hold.py --scenarios hold,step,ramp,dropped,signinv,gain,deadband,headroom,rate
    python verify/verify_attitude_hold.py --scenarios dropped --dropped-deg 20 --div-deg 15
Exit 1 only on a non-finite run or an NMPC solve failure (n_fail > 0; the signinv
tumble is exempt -- its solver failures are the scenario working); the per-row
`accept` flag is reported, not gated (the heavy_gripper (0,20) plain-mpc steady tilt is
an EXPECTED miss -- the observer-evidence prediction the artefact is for).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # forked workers must not oversubscribe BLAS
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # marinegym dir
sys.path.insert(0, HERE)

import hydro as H
import rov_model as RM
import thrusters as T
from dobmpc import params as P
from dobmpc_controller import DOBMPCController, ATT_RATE_SOURCES
from disturbance.env import MODES
from disturbance.config import load_config
from experiments.run_compare import build, steady_mask
from recorder import build_run_meta

SCENARIOS = ("hold", "step", "ramp", "dropped", "signinv", "gain", "deadband",
             "headroom", "rate")
DEFAULT_GRID = "0,0;15,0;0,15;0,-15;10,10"
# SEPARATE fieldnames list (recorder.RECORD_FIELDS untouched; verify_meta.py pins it)
ATT_FIELDS = (["t", "roll_deg", "pitch_deg", "yaw_deg", "roll_ref_deg", "pitch_ref_deg",
               "px", "py", "pz", "X", "Y", "Z", "K", "M", "N", "K_app", "M_app"]
              + [f"u{i}" for i in range(8)]
              + ["sum_abs_f", "sat", "wr_hat", "wp_hat", "status"])
RESULT_FIELDS = ["scenario", "ctrl", "mode", "roll_ref_deg", "pitch_ref_deg", "variant",
                 "rate_source", "seed", "T", "n_tick",
                 "e_roll_mean_deg", "e_roll_rms_deg", "e_roll_max_deg",
                 "e_pitch_mean_deg", "e_pitch_rms_deg", "e_pitch_max_deg",
                 "e_rp_max_all_deg", "rad_rms_cm", "z_rms_cm", "yaw_rms_deg",
                 "K_mean_nm", "M_mean_nm", "max_abs_f_n", "sat_ticks",
                 "sum_f_mean_n", "thrust_cost_vs_level_n", "wr_hat_mean", "wp_hat_mean",
                 "n_fail", "last_status", "finite", "accept", "extra", "wall_s", "csv"]


# ------------------------------------------------------------ controller hooks
class AttCtrl(DOBMPCController):
    """DOBMPCController + verify-only actuation plugins. The EAOB keeps seeing the
    COMMANDED wrench (_tau_ned_cmd, set inside super()._control_step) -- exactly what
    the station does when its wire drops / inverts / scales K/M without telling the
    observer -- while the plant gets the plugin's version of the FLU wrench.
    km_plugin: callable(tau_flu (6,), self) -> tau_flu applied (None = identity).
    deadband_n: per-thruster |f| below this (N) is zeroed at allocation (None = off)."""
    km_plugin = None
    deadband_n = None

    def _control_step(self, data):
        super()._control_step(data)                 # sets commanded (pre-plugin) + EAOB input
        if self.km_plugin is not None:
            self._tau_flu = np.asarray(self.km_plugin(self._tau_flu.copy(), self), float)

    def apply(self, model, data):
        if self.deadband_n is None:
            return super().apply(model, data)
        if self._k % self.decim == 0:
            self._control_step(data)
        self._k += 1
        forces = T.allocate(self.B, self._tau_flu)
        forces = np.where(np.abs(forces) < float(self.deadband_n), 0.0, forces)
        T.set_thruster_forces(model, data, forces)
        self.realized = np.asarray(self.B @ forces, float)
        return forces, self.realized


def _plugin_drop(tau, ctrl):
    tau[3:5] = 0.0
    return tau


def _plugin_signinv(tau, ctrl):
    if getattr(ctrl, "neutral", False):
        return np.zeros(6)
    tau[3:5] *= -1.0
    return tau


def _plugin_gain(g):
    def f(tau, ctrl):
        tau[3:5] *= g
        return tau
    return f


# ------------------------------------------------------------------- one run
def _rp_deg(R):
    return (np.degrees(np.arctan2(R[2, 1], R[2, 2])),
            np.degrees(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0))),
            np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def run_one(spec, cfg, out_runs):
    """spec: dict(scenario, ctrl, mode, roll_deg, pitch_deg, variant, rate_source, seed,
    T, settle, t_step, step_deg, rate_rad_s, div_deg, abort_deg, knee_n, tag)."""
    scen, ctrl_name, mode = spec["scenario"], spec["ctrl"], spec["mode"]
    T_run, seed = float(spec["T"]), int(spec["seed"])
    roll0, pitch0 = np.radians(spec["roll_deg"]), np.radians(spec["pitch_deg"])
    t_step, step = float(spec["t_step"]), np.radians(spec["step_deg"])
    w_ramp = float(spec["rate_rad_s"])
    t_wall = time.time()

    model, data, hydro, env, bid = build(cfg, mode, seed, ctrl_name, T_run, cfg.dist)
    ctrl = AttCtrl(model, hydro=hydro, mode=ctrl_name, actuator=None,
                   att_rate_source=spec.get("rate_source") or "fd_horizon_T_inv")
    ctrl.reset()
    variant = spec.get("variant", "")
    if scen == "dropped":
        ctrl.km_plugin = _plugin_drop
    elif scen == "signinv":
        ctrl.km_plugin = _plugin_signinv
    elif scen == "gain":
        ctrl.km_plugin = _plugin_gain(float(variant))
    elif scen == "deadband":
        ctrl.deadband_n = float(spec["knee_n"])

    # reference program (world-FLU roll/pitch rad as a function of sim time)
    if scen in ("step",):
        def ref_at(t):
            return (0.0, step if t >= t_step else 0.0)
    elif scen == "ramp":
        def ref_at(t):
            return (0.0, float(np.clip(w_ramp * (t - t_step), 0.0, step)) if t >= t_step else 0.0)
    else:
        def ref_at(t):
            return (roll0, pitch0)

    if scen == "rate":                                   # tracking mode, 6-tuple sampler
        def sampler(ts):
            ts = np.asarray(ts, float); K = ts.size
            pitch = np.clip(w_ramp * (ts - t_step), 0.0, step) * (ts >= t_step)
            return (np.zeros((3, K)), np.zeros(K), np.zeros((3, K)), np.zeros(K),
                    np.zeros(K), pitch)
        ctrl.set_reference_traj(sampler)
        ctrl.set_target((0.0, 0.0, 0.0), yaw_ref=0.0)    # PID-parity / logging only
    else:
        r0, p0 = ref_at(0.0)
        ctrl.set_target((0.0, 0.0, 0.0), yaw_ref=0.0, v_ref=(0.0, 0.0, 0.0),
                        roll_ref=r0, pitch_ref=p0)
    data.qpos[:3] = [0.1, 0.05, 0.0]                      # the DP start offset (run_compare)
    mujoco.mj_forward(model, data)

    idx = T._ctrl_index(model)                           # thruster actuators (jaw excluded)
    lo = np.asarray(model.actuator_ctrlrange[idx, 0], float)
    hi = np.asarray(model.actuator_ctrlrange[idx, 1], float)
    rows = []
    tripped_at = None
    while data.time < T_run:
        t = float(data.time)
        if scen in ("step", "ramp"):
            r_ref, p_ref = ref_at(t)
            ctrl.set_target(roll_ref=r_ref, pitch_ref=p_ref)
        forces, _realized = ctrl.apply(model, data)
        if (ctrl._k - 1) % ctrl.decim == 0:               # a control tick just ran
            p = np.asarray(data.xpos[bid], float)
            R = np.asarray(data.xmat[bid], float).reshape(3, 3)
            roll, pitch, yaw = _rp_deg(R)
            if scen == "rate":
                p_ref = float(sampler(np.array([t]))[5][0]); r_ref = 0.0
            else:
                r_ref, p_ref = ref_at(t)
            f = np.asarray(forces, float)[:len(idx)]
            sat = int(np.any(f > hi + 1e-9) or np.any(f < lo - 1e-9))
            cmd = ctrl.commanded                          # pre-plugin FLU wrench
            app = ctrl._tau_flu                           # applied (post-plugin) FLU wrench
            u8 = list(f) + [0.0] * (8 - len(f))
            rows.append([t, roll, pitch, yaw, np.degrees(r_ref), np.degrees(p_ref),
                         p[0], p[1], p[2], cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5],
                         app[3], app[4]] + u8 +
                        [float(np.abs(f).sum()), sat, ctrl.w_hat[3], ctrl.w_hat[4],
                         int(getattr(ctrl.nmpc, "last_status", 0))])
            if scen == "signinv" and tripped_at is None and \
                    max(abs(roll), abs(pitch)) > float(spec["abort_deg"]):
                tripped_at = t
                ctrl.neutral = True                       # disengage: whole wrench -> 0
        mujoco.mj_step(model, data)
    H.Hydrodynamics.uninstall()
    wall = time.time() - t_wall

    A = np.asarray(rows, float)
    L = {k: A[:, i] for i, k in enumerate(ATT_FIELDS)}
    tt = L["t"]
    m = _steady(scen, tt, spec)
    e_r = L["roll_deg"] - L["roll_ref_deg"]
    e_p = L["pitch_deg"] - L["pitch_ref_deg"]
    rad = np.hypot(L["px"], L["py"]) * 100.0
    finite = bool(np.all(np.isfinite(A)))
    res = dict(scenario=scen, ctrl=ctrl_name, mode=mode, roll_ref_deg=spec["roll_deg"],
               pitch_ref_deg=spec["pitch_deg"], variant=variant,
               rate_source=(ctrl.att_rate_source if scen == "rate" else "none"),
               seed=seed, T=T_run, n_tick=int(tt.size),
               e_roll_mean_deg=float(e_r[m].mean()), e_roll_rms_deg=float(np.sqrt((e_r[m] ** 2).mean())),
               e_roll_max_deg=float(np.abs(e_r[m]).max()),
               e_pitch_mean_deg=float(e_p[m].mean()), e_pitch_rms_deg=float(np.sqrt((e_p[m] ** 2).mean())),
               e_pitch_max_deg=float(np.abs(e_p[m]).max()),
               e_rp_max_all_deg=float(max(np.abs(e_r).max(), np.abs(e_p).max())),
               rad_rms_cm=float(np.sqrt((rad[m] ** 2).mean())),
               z_rms_cm=float(np.sqrt((L["pz"][m] ** 2).mean()) * 100.0),
               yaw_rms_deg=float(np.sqrt((L["yaw_deg"][m] ** 2).mean())),
               K_mean_nm=float(L["K"][m].mean()), M_mean_nm=float(L["M"][m].mean()),
               max_abs_f_n=float(np.abs(A[:, 17:25]).max()),
               sat_ticks=int(L["sat"].sum()),
               sum_f_mean_n=float(L["sum_abs_f"][m].mean()),
               thrust_cost_vs_level_n=float("nan"),
               wr_hat_mean=float(L["wr_hat"][m].mean()), wp_hat_mean=float(L["wp_hat"][m].mean()),
               n_fail=int(ctrl.n_fail), last_status=int(getattr(ctrl.nmpc, "last_status", 0)),
               finite=int(finite), accept=0, extra="", wall_s=round(wall, 1), csv="")
    res["accept"] = int(finite and res["n_fail"] == 0 and res["sat_ticks"] == 0
                        and res["e_roll_rms_deg"] < 1.0 and res["e_pitch_rms_deg"] < 1.0)
    res["extra"] = _scenario_extra(scen, spec, L, e_r, e_p, m, tripped_at)

    # per-run artefacts
    tag = spec["tag"]
    csv_path = os.path.join(out_runs, f"traj_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ATT_FIELDS)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])
    res["csv"] = os.path.relpath(csv_path, os.path.dirname(out_runs))
    ctrl_meta = dict(type=ctrl_name, solver=type(ctrl.nmpc).__name__,
                     u_max=list(map(float, ctrl.nmpc.u_max)),
                     mpc_q=list(map(float, P.MPC_Q)), mpc_r=list(map(float, P.MPC_R)),
                     mpc_state_source=ctrl.mpc_state_source,
                     attitude_ref=ctrl.attitude_meta(),
                     plugin=dict(scenario=scen, variant=variant,
                                 deadband_n=ctrl.deadband_n, tripped_at=tripped_at))
    if ctrl_name == "dobmpc":
        ctrl_meta["eaob"] = ctrl.eaob_meta()
    meta = build_run_meta(
        disturbance=env.to_meta(), controller=ctrl_meta,
        trajectory=dict(kind="attitude_" + scen, roll_ref_deg=spec["roll_deg"],
                        pitch_ref_deg=spec["pitch_deg"], t_step=t_step,
                        step_deg=spec["step_deg"], rate_rad_s=w_ramp,
                        div_deg=float(spec["div_deg"]), dropped_deg=float(spec["dropped_deg"]),
                        attitude_ref_deg=dict(roll=spec["roll_deg"], pitch=spec["pitch_deg"])),
        run=dict(mode=mode, seed=seed, T=T_run, dt=float(model.opt.timestep), log_hz=20.0,
                 settle_s=float(spec["settle"]),
                 # the steady-window rule + its effective start (module docstring)
                 steady_rule=("t>=settle" if scen not in STEP_SCENARIOS
                              else "steady_mask(t-t_step, settle)"),
                 steady_from_s=float(tt[m][0]),
                 wall_s=round(wall, 2), n_fail=res["n_fail"],
                 csv=os.path.basename(csv_path), fields=ATT_FIELDS))
    with open(os.path.join(out_runs, f"meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return res


STEP_SCENARIOS = ("step", "ramp", "rate")        # reference moves at t_step


def _steady(scen, tt, spec):
    """Steady-window mask (module docstring, 'Which samples a metric scores').
    Constant-reference scenarios: steady_mask(t, settle). step/ramp/rate: the same
    rule on t - t_step (the reference moves at t_step), so the window starts at
    t_step + settle instead of containing the transient; steady_mask's own
    half-record clamp is the short-run fallback (whole record if < 3 samples)."""
    tt = np.asarray(tt, float)
    if scen in STEP_SCENARIOS:
        return steady_mask(tt - float(spec["t_step"]), float(spec["settle"]))
    return steady_mask(tt, float(spec["settle"]))


def _scenario_extra(scen, spec, L, e_r, e_p, m, tripped_at):
    """Scenario-specific numbers, packed as 'k=v;k=v' into results.csv `extra`.
    Everything here is TRANSIENT / whole-run (t >= t_step etc.), not steady-window,
    except steady_from_s which records where the steady window started."""
    tt = L["t"]; dt = float(np.median(np.diff(tt))) if tt.size > 2 else 0.05
    t_step, step = float(spec["t_step"]), float(spec["step_deg"])
    ex = {}
    if scen in STEP_SCENARIOS:
        ex["steady_from_s"] = float(tt[m][0])
    if scen == "step":
        a = tt >= t_step
        pk = float(L["pitch_deg"][a].max()) if step > 0 else float(L["pitch_deg"][a].min())
        os_pct = (pk - step) / step * 100.0 if step else float("nan")
        out = np.where(a & (np.abs(e_p) > 1.0))[0]
        ex["overshoot_pct"] = os_pct
        ex["settle_1deg_s"] = (float(tt[out[-1]] - t_step) if out.size else 0.0)
        if os_pct > 0:
            ln = np.log(os_pct / 100.0)
            ex["zeta_from_os"] = float(-ln / np.sqrt(np.pi ** 2 + ln ** 2))   # [유도]
        else:
            ex["zeta_from_os"] = 1.0
    elif scen in ("ramp", "rate"):
        w = float(spec["rate_rad_s"])
        t_end = t_step + np.radians(step) / w
        r = (tt >= t_step) & (tt <= t_end)
        ex["ramp_e_peak_deg"] = float(np.abs(e_p[r]).max()) if r.any() else float("nan")
        ex["ramp_lag_s"] = (float(np.abs(e_p[r]).mean() / np.degrees(w)) if r.any()
                            else float("nan"))                              # [유도] mean(e)/w
        ex["post_e_rms_deg"] = float(np.sqrt((e_p[tt > t_end + 3.0] ** 2).mean())) \
            if np.any(tt > t_end + 3.0) else float("nan")
        dM = np.abs(np.diff(L["M"]))
        ex["chat_M_nm_per_tick"] = float(dM.mean())
        ex["chat_M_ramp_nm_per_tick"] = float(dM[r[1:]].mean()) if r[1:].any() else float("nan")
    elif scen == "dropped":
        div = float(spec["div_deg"])
        assert div < float(spec["dropped_deg"]), "degenerate dropped spec (build_specs guards this)"
        over = (np.maximum(np.abs(e_r), np.abs(e_p)) > div)
        # station rule (workers.py div_rp_streak): halt on the tick the streak reaches
        # round(0.5 * ctrl_hz) consecutive over-threshold samples -> t_div is THAT tick
        # (pre-2026-09-26 (2) it was the streak's first tick, 0.45 s earlier)
        need = max(2, int(round(0.5 / dt)))
        t_div = float("nan")
        run = 0
        for i, o in enumerate(over):
            run = run + 1 if o else 0
            if run >= need:
                t_div = float(tt[i]); break
        ex["t_div_rp_s"] = t_div
        ex["rp_meas_mean_deg"] = f"{L['roll_deg'][m].mean():.2f}/{L['pitch_deg'][m].mean():.2f}"
        ex["rad_max_cm"] = float((np.hypot(L["px"], L["py"]) * 100.0).max())
    elif scen == "signinv":
        ex["t_abort_s"] = float("nan") if tripped_at is None else float(tripped_at)
        ex["rp_max_deg"] = float(np.maximum(np.abs(L["roll_deg"]), np.abs(L["pitch_deg"])).max())
        if tripped_at is not None:
            after = tt >= tripped_at
            inside = after & (np.maximum(np.abs(L["roll_deg"]), np.abs(L["pitch_deg"])) < 2.0)
            ex["t_recover_s"] = (float(tt[inside][0] - tripped_at) if inside.any()
                                 else float("nan"))
        else:
            ex["t_recover_s"] = float("nan")
    elif scen == "deadband":
        e = e_p[m] - e_p[m].mean()
        ex["lc_p2p_deg"] = float(e.max() - e.min())
        if e.size > 8 and (e.max() - e.min()) > 0.1:
            F = np.abs(np.fft.rfft(e)); F[0] = 0.0
            ex["lc_freq_hz"] = float(np.fft.rfftfreq(e.size, dt)[int(F.argmax())])
        else:
            ex["lc_freq_hz"] = 0.0
        ex["knee_n"] = float(spec["knee_n"])
    elif scen == "headroom":
        box = float(P.U_MAX[0])
        ex["X_max_n"] = float(np.abs(L["X"][m]).max()); ex["Z_max_n"] = float(np.abs(L["Z"][m]).max())
        ex["box_ticks"] = int(np.sum((np.abs(L["X"][m]) >= 0.999 * box) | (np.abs(L["Z"][m]) >= 0.999 * box)))
        ex["X_mean_n"] = float(L["X"][m].mean()); ex["Z_mean_n"] = float(L["Z"][m].mean())
        ex["net_buoy_sin_theta_n"] = float(P.NET_BUOYANCY * np.sin(np.radians(spec["pitch_deg"])))
    return ";".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in ex.items())


# ------------------------------------------------------------------- driver
def _run_task(task):
    spec, cfg_path, out_runs = task
    cfg = load_config(cfg_path)
    try:
        return run_one(spec, cfg, out_runs)
    except Exception as e:                                # keep the batch alive
        H.Hydrodynamics.uninstall()
        return dict(scenario=spec["scenario"], ctrl=spec["ctrl"], mode=spec["mode"],
                    roll_ref_deg=spec["roll_deg"], pitch_ref_deg=spec["pitch_deg"],
                    variant=spec.get("variant", ""), finite=0, accept=0, n_fail=-1,
                    extra=f"error={type(e).__name__}:{e}", seed=spec["seed"], T=spec["T"])


def _prebuild(cfg):
    """Generate/compile the acados solver ONCE so forked workers load it (run_compare
    pattern); returns True iff workers may use DOBMPC_ACADOS_BUILD=0."""
    try:
        model, _d, hy, _e, _b = build(cfg, "NONE", 0, "mpc", 5.0, cfg.dist)
        DOBMPCController(model, hydro=hy, mode="mpc")
        H.Hydrodynamics.uninstall()
        return True
    except Exception as e:
        H.Hydrodynamics.uninstall()
        print(f"[verify_attitude_hold] WARN acados pre-build failed ({e}); workers build",
              flush=True)
        return False


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_specs(args):
    grid = []
    for cell in args.grid.split(";"):
        r, p = cell.split(",")
        grid.append((float(r), float(p)))
    common = dict(seed=args.seed, T=args.T, settle=args.settle, t_step=args.t_step,
                  step_deg=args.step_deg, rate_rad_s=args.rate, div_deg=args.div_deg,
                  dropped_deg=args.dropped_deg, abort_deg=args.abort_deg, knee_n=args.knee_n,
                  rate_source=None)
    if "dropped" in args.scenarios and float(args.div_deg) >= float(args.dropped_deg):
        # a passive hull sits at |e| = dropped_deg; div_deg >= that never trips or trips
        # on noise -> the scenario would measure nothing (2026-09-26 174607 artefact)
        raise SystemExit(f"[verify_attitude_hold] refusing 'dropped': --div-deg {args.div_deg} "
                         f">= --dropped-deg {args.dropped_deg} (interlock threshold must be "
                         f"below the reference the passive hull cannot reach)")
    if any(s in STEP_SCENARIOS for s in args.scenarios) and float(args.T) <= float(args.t_step) + float(args.settle):
        print(f"[verify_attitude_hold] WARN T {args.T} <= t_step + settle "
              f"{args.t_step + args.settle}: step/ramp/rate steady window falls back to "
              f"steady_mask's half-record clamp (see extra steady_from_s)", flush=True)
    specs = []

    def add(scen, ctrl, mode, roll, pitch, variant="", **kw):
        s = dict(common, scenario=scen, ctrl=ctrl, mode=mode, roll_deg=roll, pitch_deg=pitch,
                 variant=variant, **kw)
        vt = f"_{variant}" if variant else ""
        s["tag"] = f"{scen}_{ctrl}_{mode}_r{roll:+g}_p{pitch:+g}{vt}".replace("+", "p").replace("-", "m")
        specs.append(s)

    for scen in args.scenarios:
        for ctrl in args.ctrls:
            if scen == "hold":
                for mode in args.modes:
                    for r, p in grid:
                        add(scen, ctrl, mode, r, p)
            elif scen in ("step", "ramp", "signinv"):
                add(scen, ctrl, args.modes[0], 0.0, args.step_deg)
            elif scen == "dropped":
                add(scen, ctrl, args.modes[0], 0.0, args.dropped_deg)
            elif scen == "gain":
                for g in ("0.5", "2.0"):
                    add(scen, ctrl, args.modes[0], 0.0, args.step_deg, variant=g)
            elif scen == "deadband":
                add(scen, ctrl, args.modes[0], 0.0, args.step_deg)
            elif scen == "headroom":
                for p in (0.0, args.step_deg):
                    add(scen, ctrl, "CDW", 0.0, p)
            elif scen == "rate":
                for rs in ATT_RATE_SOURCES:
                    add(scen, ctrl, args.modes[0], 0.0, args.step_deg, variant=rs, rate_source=rs)
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="hold", help=f"comma list of {SCENARIOS}")
    ap.add_argument("--ctrls", default="mpc,dobmpc")
    ap.add_argument("--modes", default="NONE", help="disturbance modes for hold; first = the rest")
    ap.add_argument("--grid", default=DEFAULT_GRID, help="roll,pitch deg cells separated by ';'")
    ap.add_argument("--T", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=10.0, help="steady window start [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-step", type=float, default=10.0, help="step/ramp start time [s]")
    ap.add_argument("--step-deg", type=float, default=15.0, help="step/ramp/S4-S7 pitch target [deg] [예측]")
    ap.add_argument("--rate", type=float, default=0.35, help="ramp rate [rad/s] (pq_max [예측])")
    ap.add_argument("--div-deg", type=float, default=15.0,
                    help="div_rp interlock [deg] (= config/hw_mpc.yaml policy.div_max_rp_deg 15)")
    ap.add_argument("--dropped-deg", type=float, default=20.0,
                    help="dropped-scenario pitch reference [deg] (= policy.rp_max_deg clip 20); "
                         "must exceed --div-deg or the scenario is refused")
    ap.add_argument("--abort-deg", type=float, default=35.0, help="attitude ceiling [deg] [예측]")
    ap.add_argument("--knee-n", type=float, default=1.44,
                    help="per-thruster deadband knee [N]: 0.096 x 15 N per vertical [유도]")
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--config", default=os.path.join(HERE, "config", "base.yaml"))
    ap.add_argument("--out", default=None, help="artefact folder (default recordings/<day>/attitude_hold_<ts>)")
    args = ap.parse_args()
    args.scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    args.ctrls = [c.strip() for c in args.ctrls.split(",") if c.strip()]
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    assert all(s in SCENARIOS for s in args.scenarios), args.scenarios
    assert all(c in ("mpc", "dobmpc") for c in args.ctrls), args.ctrls
    assert all(m in MODES for m in args.modes), args.modes

    specs = build_specs(args)                    # may refuse (degenerate dropped) BEFORE any folder exists
    ts = time.strftime("%Y%m%d_%H%M%S")
    day, hms = ts.split("_")
    out = args.out or os.path.join(HERE, "recordings", day, f"attitude_hold_{hms}")
    out_runs = os.path.join(out, "runs")
    os.makedirs(out_runs, exist_ok=True)
    cfg = load_config(args.config)
    print(f"[verify_attitude_hold] ROV_MODEL={RM.MODEL}  {len(specs)} runs  jobs={args.jobs}"
          f"  -> {out}", flush=True)

    t0 = time.time()
    tasks = [(s, args.config, out_runs) for s in specs]
    if args.jobs > 1 and len(tasks) > 1 and _prebuild(cfg):
        import multiprocessing as mp
        os.environ["DOBMPC_ACADOS_BUILD"] = "0"
        with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
            results = []
            for i, r in enumerate(pool.imap(_run_task, tasks)):
                results.append(r)
                print(f"  [{i + 1}/{len(tasks)}] {specs[i]['tag']}: e_rms r/p "
                      f"{r.get('e_roll_rms_deg', float('nan')):.2f}/{r.get('e_pitch_rms_deg', float('nan')):.2f} deg"
                      f"  sat {r.get('sat_ticks', '?')}  fail {r.get('n_fail', '?')}  {r.get('extra', '')}",
                      flush=True)
    else:
        results = []
        for i, task in enumerate(tasks):
            r = _run_task(task)
            results.append(r)
            print(f"  [{i + 1}/{len(tasks)}] {specs[i]['tag']}: e_rms r/p "
                  f"{r.get('e_roll_rms_deg', float('nan')):.2f}/{r.get('e_pitch_rms_deg', float('nan')):.2f} deg"
                  f"  sat {r.get('sat_ticks', '?')}  fail {r.get('n_fail', '?')}  {r.get('extra', '')}",
                  flush=True)

    # thrust cost vs the same ctrl/mode LEVEL hold (S2's IL-relevant number)
    level = {(r["ctrl"], r["mode"]): r.get("sum_f_mean_n") for r in results
             if r.get("scenario") == "hold" and r["roll_ref_deg"] == 0 and r["pitch_ref_deg"] == 0}
    for r in results:
        base = level.get((r.get("ctrl"), r.get("mode")))
        if base is not None and r.get("sum_f_mean_n") is not None:
            r["thrust_cost_vs_level_n"] = float(r["sum_f_mean_n"] - base)

    res_path = os.path.join(out, "results.csv")
    with open(res_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v)
                        for k, v in {f: r.get(f, "") for f in RESULT_FIELDS}.items()})
    meta = dict(schema="attitude_hold/1", rov_model=RM.MODEL, started=ts, git=_git_head(),
                args=vars(args), config=os.path.abspath(args.config),
                u_max=list(map(float, P.U_MAX)), mpc_q=list(map(float, P.MPC_Q)),
                mpc_r=list(map(float, P.MPC_R)), net_buoyancy_n=float(P.NET_BUOYANCY),
                zg=float(P.ZG), eaob_tau_dist=float(P.EAOB_TAU_DIST),
                mpc_state_source=P.MPC_STATE_SOURCE, fields=ATT_FIELDS,
                result_fields=RESULT_FIELDS, wall_s=round(time.time() - t0, 1),
                acceptance="accept = e_roll_rms < 1 deg AND e_pitch_rms < 1 deg (steady window) "
                           "AND n_fail 0 AND sat_ticks 0 (whole run) [예측]",
                steady_window="t >= settle; step/ramp/rate: steady_mask(t - t_step, settle) "
                              "(effective start in extra steady_from_s / run.steady_from_s)")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    hdr = (f"{'scenario':9s} {'ctrl':6s} {'mode':4s} {'ref r/p':>9s} {'var':>14s} | "
           f"{'e_r rms':>7s} {'e_p rms':>7s} {'e_p mean':>8s} {'e max':>6s} | {'radRMS':>6s} "
           f"{'K':>6s} {'M':>6s} {'max|f|':>6s} {'sat':>4s} {'sum|f|':>6s} {'d.lvl':>6s} | "
           f"{'w_r':>5s} {'w_p':>5s} {'fail':>4s} {'ok':>2s}")
    print(f"\n=== attitude hold ({RM.MODEL}, seed {args.seed}, T {args.T} s, steady t>={args.settle} s "
          f"[step/ramp/rate: t>=t_step+settle={args.t_step + args.settle} s, extra steady_from_s]; "
          f"max|f|/sat whole run; deg / cm / N / N*m) ===\n{hdr}\n" + "-" * len(hdr))
    for r in results:
        if not r.get("finite"):
            print(f"{r['scenario']:9s} {r['ctrl']:6s} {r['mode']:4s}  ERROR {r.get('extra')}")
            continue
        print(f"{r['scenario']:9s} {r['ctrl']:6s} {r['mode']:4s} "
              f"{r['roll_ref_deg']:+4.0f}/{r['pitch_ref_deg']:+4.0f} {str(r['variant']):>14s} | "
              f"{r['e_roll_rms_deg']:7.3f} {r['e_pitch_rms_deg']:7.3f} {r['e_pitch_mean_deg']:+8.3f} "
              f"{r['e_rp_max_all_deg']:6.2f} | {r['rad_rms_cm']:6.2f} {r['K_mean_nm']:+6.2f} "
              f"{r['M_mean_nm']:+6.2f} {r['max_abs_f_n']:6.1f} {r['sat_ticks']:4d} "
              f"{r['sum_f_mean_n']:6.1f} {r['thrust_cost_vs_level_n']:+6.1f} | "
              f"{r['wr_hat_mean']:+5.2f} {r['wp_hat_mean']:+5.2f} {r['n_fail']:4d} {r['accept']:2d}")
        if r.get("extra"):
            print(f"{'':9s} extra: {r['extra']}")
    print(f"\nartefact: {res_path}")
    # signinv is DESIGNED to tumble the vehicle (solver failures at 60-90 deg are the
    # scenario working, not a regression) -> excluded from the exit gate, still recorded.
    bad = [r for r in results if not r.get("finite")
           or (r.get("scenario") != "signinv" and int(r.get("n_fail", 0)) != 0)]
    print(f"gate (finite + n_fail 0, signinv exempt): {'PASS' if not bad else 'FAIL'}   "
          f"accept rows: {sum(int(r.get('accept', 0)) for r in results)}/{len(results)}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
