#!/usr/bin/env python3
"""sim_pid7d_attitude.py — SYNTHETIC evidence for LOW pid_7d's roll/pitch loop (2026-10-03).

Two things, both [유도] (models, not the vehicle):

1. Linear margins of the shipped gains (config/hw_mpc.yaml pid_7d) — the re-check the control-theory-advisor consult
   asked for: L(s) = g (Kp + Kd s + Ki/s) e^{-sT} / (I s^2 + d s + k), d = 0.07 (the linear placeholder), over the
   restoring stiffness k (1.35 NMPC ZG*W .. 20 RL lumped fit), actuator-gain error g and total delay T.
2. The REAL ``HwPid7d`` class closed around a per-axis nonlinear plant
       I a'' + d a' + d2 a'|a'| + k sin(a) = g * tau(t - T_act)
   at 1 kHz, the station's 20 Hz tick, a measurement 1 tick old (ATTITUDE ~52 ms), the worker's allocation (cap in
   force, 1.5 axis/s slew) and note_applied with the NOMINAL realized torque — a pitch 0 -> 10 deg step held by a tracked
   one-stage plan, and a pitch ramp at 0.35 rad/s. No thruster deadband, no coupling, no translation: a statement about
   the loop's arithmetic and its delay margin, not about the hull.

Run (robust env):  python rov_gui/tests/sim_pid7d_attitude.py [--out PATH]
Writes the tables to stdout and (default) rov_gui/tests/fixtures/sim_pid7d_attitude_20261003.txt.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from rov_gui.control.geometry import MpcConfig  # noqa: E402
from rov_gui.control.path_geometry import NedPlan  # noqa: E402
from rov_gui.control.pid import HwPid7d  # noqa: E402

D_LIN = 0.07          # N m s/rad, marinegym placeholder (config/hw_mpc.yaml plant)
D_QUAD = 1.55         # N m s^2/rad^2, placeholder


def margins(kp, kd, ki, I, k, g, T, d=D_LIN):
    """(crossover rad/s, PM deg, GM x) of L(jw) on a log grid; GM = 1/|L| at the first -180 deg crossing."""
    w = np.logspace(-2.5, 2.0, 200000)
    s = 1j * w
    L = g * (kp + kd * s + ki / s) * np.exp(-s * T) / (I * s * s + d * s + k)
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))
    pm, wc = None, None
    idx = np.where((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    if len(idx):
        i = idx[-1]                       # the LAST downward crossing (resonant plants cross more than once)
        wc = float(w[i])
        pm = float(np.degrees(ph[i]) + 180.0)
        while pm > 180.0:
            pm -= 360.0
        while pm < -180.0:
            pm += 360.0
    gm = None
    cross = np.where((ph[:-1] > -np.pi) & (ph[1:] <= -np.pi))[0]   # the first -180 deg crossing (phase falls with w)
    if len(cross):
        gm = float(1.0 / mag[cross[0]])
    return wc, pm, gm


def closed_loop(axis, k, g, T_act, caps, ref, t_end=40.0, payload_nm=0.0):
    """The REAL HwPid7d on one axis (0 roll, 1 pitch). ref(t) -> (angle, rate) NED rad, installed each tick as a
    one-stage plan. Returns dict of metrics."""
    cfg = MpcConfig.load(os.path.join(ROOT, "config", "hw_mpc.yaml"))
    ctrl = HwPid7d(cfg, log=lambda m: None)
    gains = dict(cfg.axis_gain)
    nm = float(gains["roll_nm" if axis == 0 else "pitch_nm"])
    cap = float(caps[axis])
    ctrl.set_attitude_axes(True, u_max_wire_nm=[caps[0] * float(gains["roll_nm"]), caps[1] * float(gains["pitch_nm"])])
    ctrl.set_target_ned(np.zeros(3), 0.0)
    I = float(ctrl.rp_inertia[axis])
    dt_sim, dt = 0.001, 1.0 / float(cfg.ctrl_hz)
    n_sub = int(round(dt / dt_sim))
    delay_steps = int(round(T_act / dt_sim))
    a, ad = 0.0, 0.0
    tau_hist = [0.0] * (delay_steps + 1)
    meas_prev = (0.0, 0.0)              # the measurement one tick old (ATTITUDE age ~ one tick)
    axis_prev = 0.0
    log_t, log_a, log_r, log_ax = [], [], [], []
    tau_cmd_applied = 0.0
    steps = int(t_end / dt)
    for n in range(steps):
        t = n * dt
        r, rd = ref(t)
        rp = np.zeros((2, 1)); rpr = np.zeros((2, 1))
        rp[axis, 0], rpr[axis, 0] = r, rd
        plan = NedPlan(p_ned=np.zeros((3, 1)), yaw_ned=np.zeros(1), v_ned=np.zeros((3, 1)), r_ned=np.zeros(1),
                       psi_path=np.zeros(1), rp_ned=rp, rp_rate_ned=rpr)
        ctrl.set_path_plan_ned(plan)
        eta = np.zeros(6); nu = np.zeros(6)
        eta[3 + axis], nu[3 + axis] = meas_prev
        u, _info = ctrl.step(eta, nu, np.zeros(6), t)
        # the worker: wrench_to_axes(attitude=True) cap, slew_axes(rate_rp 1.5/s), note_applied(axes_to_wrench)
        ax_cmd = float(np.clip(u[3 + axis] / nm, -cap, cap))
        st = 1.5 * dt
        ax_cmd = axis_prev + float(np.clip(ax_cmd - axis_prev, -st, st))
        axis_prev = ax_cmd
        realized = np.zeros(6)
        realized[3 + axis] = ax_cmd * nm
        ctrl.note_applied(realized)
        tau_cmd_applied = g * ax_cmd * nm
        meas_prev = (a, ad)
        for _ in range(n_sub):
            tau_hist.append(tau_cmd_applied)
            tau = tau_hist.pop(0)
            add = (tau + payload_nm - D_LIN * ad - D_QUAD * ad * abs(ad) - k * math.sin(a)) / I
            ad += add * dt_sim
            a += ad * dt_sim
            if not math.isfinite(a) or abs(a) > 3.0:
                return {"diverged": True}
        log_t.append(t); log_a.append(a); log_r.append(r); log_ax.append(ax_cmd)
    t_arr, a_arr, r_arr, ax_arr = map(np.asarray, (log_t, log_a, log_r, log_ax))
    last = t_arr > t_end - 10.0
    err = r_arr - a_arr
    return {"diverged": False,
            "e_last10_mean_deg": float(np.degrees(err[last].mean())),
            "e_last10_pp_deg": float(np.degrees(err[last].max() - err[last].min())),
            "overshoot_deg": float(np.degrees(max(0.0, (a_arr - r_arr.max()).max()))),
            "sat_frac": float(np.mean(np.abs(ax_arr) >= cap - 1e-9)),
            "e_rms_deg": float(np.degrees(np.sqrt(np.mean(err ** 2))))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "sim_pid7d_attitude_20261003.txt"))
    args = ap.parse_args()
    cfg = MpcConfig.load(os.path.join(ROOT, "config", "hw_mpc.yaml"))
    c = HwPid7d(cfg, log=lambda m: None)
    lines = ["# sim_pid7d_attitude.py — SYNTHETIC [유도] (models, not the vehicle); config/hw_mpc.yaml pid_7d as shipped",
             f"# kp {c.rp_kp.round(4).tolist()} kd {c.rp_kd.round(4).tolist()} ki {c.rp_ki.round(4).tolist()} "
             f"k_ff {c.rp_k_ff} I {c.rp_inertia.tolist()} d_lin {D_LIN}",
             "", "## 1. linear margins: PM deg / GM x (crossover rad/s); g = actuator-gain error, T = total delay s"]
    for ax, name in ((0, "roll"), (1, "pitch")):
        lines.append(f"### {name}")
        lines.append("k     g    | T 0.1               | T 0.2               | T 0.3")
        for k in (1.35, 2.3, 20.0):
            for g in (0.4, 1.0, 2.5):
                cells = []
                for T in (0.1, 0.2, 0.3):
                    wc, pm, gm = margins(c.rp_kp[ax], c.rp_kd[ax], c.rp_ki[ax], c.rp_inertia[ax], k, g, T)
                    cells.append(f"{pm:6.1f} / {gm if gm is not None else float('nan'):5.2f} ({wc if wc else float('nan'):4.2f})")
                lines.append(f"{k:5.2f} {g:3.1f}  | " + " | ".join(cells))
    lines += ["", "## 2. REAL HwPid7d closed loop, pitch axis, first-water caps (0.1, 0.15), T_act = actuation delay",
              "##    (+ one tick of measurement age); step 0 -> 10 deg at t=1 s held 40 s / ramp 0.35 rad/s to 10 deg",
              "k     g    T_act | step: e_mean_last10 / pp_last10 / overshoot / sat% | ramp: e_rms / pp_last10"]
    step = lambda t: ((math.radians(10.0), 0.0) if t >= 1.0 else (0.0, 0.0))  # noqa: E731

    def ramp(t):
        th = min(max(0.0, (t - 1.0) * 0.35), math.radians(10.0))
        return th, (0.35 if 0.0 < th < math.radians(10.0) else 0.0)

    for k in (1.35, 20.0):
        for g in (0.4, 1.0, 2.5):
            for T in (0.05, 0.15):
                rs = closed_loop(1, k, g, T, (0.1, 0.15), step)
                rr = closed_loop(1, k, g, T, (0.1, 0.15), ramp)
                if rs.get("diverged") or rr.get("diverged"):
                    lines.append(f"{k:5.2f} {g:3.1f} {T:4.2f}  | DIVERGED")
                    continue
                lines.append(f"{k:5.2f} {g:3.1f} {T:4.2f}  | {rs['e_last10_mean_deg']:6.2f} / {rs['e_last10_pp_deg']:5.2f} / "
                             f"{rs['overshoot_deg']:5.2f} / {100 * rs['sat_frac']:5.1f} | {rr['e_rms_deg']:5.2f} / "
                             f"{rr['e_last10_pp_deg']:5.2f}")
    lines += ["", "## reading: k = the hull's restoring stiffness (unknown, 1.35 .. ~20 N m/rad); a 10 deg hold needs",
              "## k*0.175 N m against the first-water pitch cap 1.08 N m -> unreachable at k 20: the mean error stays",
              "## 4-9 deg with the AXIS BELOW its cap (sat% column), because the integral is clamped to 0.8 x the wire",
              "## torque. This harness has no interlocks. In the station the follower reports that state as 'authority",
              "## exhausted' (integral at its limit with > 1 deg error, or error beyond the 20 deg gate) and interlock (iii)",
              "## judges it like a pinned axis (safety audit 2026-10-03) — a static unreachable tilt then disengages after",
              "## sat_ineffective_s. Not a measurement of the vehicle."]
    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
