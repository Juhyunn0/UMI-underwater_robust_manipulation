#!/usr/bin/env python3
"""pid.py — the PID baseline, same harness surface as the MPC bridge.

The gain SET is the simulator's pole-placed design, copied verbatim from
``bluerov2_mujoco_marinegym/controller.py`` ``GAINS_HEAVY_GRIPPER`` (ωn=2.0
translation / 3.0 yaw, ζ=0.9, α=0.2, re-evaluated at the heavy+gripper+C3
payload masses — provenance in that file). Structure ported likewise:
world-frame position error rotated into the body, per-axis PID with gated
(+clamped) integrals, wrench saturation and a slew limit; yaw its own PID on
the wrapped angle. Roll/pitch terms are NOT ported: MANUAL_CONTROL carries no
roll/pitch axis, so they would be dropped downstream anyway.

Two hardware deltas, both deliberate:
  * ``omega_derate`` scales the design frequency down (kp·d², kd·d, ki·d³ —
    the pole-placement scalings) because the sim's ωn assumed clean 20 Hz
    state and ideal thrust; the sim's own docs say "derate ωn for hardware".
    [예측] until a pool session tunes it.
  * no buoyancy feed-forward — the real net buoyancy is unmeasured; the
    heave integral carries it instead.

Interface-compatible with :class:`~rov_gui.control.mpc_bridge.HwDobMpc` as
far as MpcWorker cares (step / set_target_ned / set_square_ned / ref_ned_at /
note_applied / reset / meta / solver_kind / realtime_ok), so the worker
swaps controllers without special cases. No casadi, no acados — this module
imports instantly, which is also why the PID is always available even when
the solver build fails.
"""

from __future__ import annotations

import math

import numpy as np

from .state_assembler import rot_zyx

# GAINS_HEAVY_GRIPPER, bluerov2_mujoco_marinegym/controller.py:90 (2026-08-12
# copy; pole placement at the payload masses). Wrench-space units: N, N·m.
SIM_GAINS = {
    "kp": (143.7, 143.7, 153.9),
    "kd": (99.5, 99.5, 108.0),
    "ki": (42.3, 42.3, 45.3),
    "i_max": (4.0, 5.0, 5.0),
    "f_max": (30.0, 30.0, 30.0),
    "slew": 120.0,
    "yaw_kp": 9.92, "yaw_kd": 4.79, "yaw_ki": 4.38,
    "yaw_i_max": 2.0, "mz_max": 10.0,
    "e_gate": 0.15, "yaw_gate": 0.2,
}


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class HwPid:
    """The sim's pose PID, driven by the same NED datum-frame state the MPC
    consumes. One instance per MpcWorker; ``reset()`` between engagements."""

    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.mode = "pid"
        self.solver_kind = "pid"
        self.realtime_ok = True
        self.probe_ms = 0.0
        self.build_s = 0.0
        self.n_fail = 0
        self.w_hat = np.zeros(6)
        self.scenario: dict | None = None
        g = dict(SIM_GAINS)
        raw = dict(getattr(cfg, "pid", None) or {})
        d = float(raw.pop("omega_derate", 0.6))
        for k, v in raw.items():
            g[k] = v
        # pole-placement scalings: kp ~ ωn², kd ~ ωn, ki ~ ωn³
        self.kp = np.asarray(g["kp"], float) * d * d
        self.kd = np.asarray(g["kd"], float) * d
        self.ki = np.asarray(g["ki"], float) * d ** 3
        self.i_max = np.asarray(g["i_max"], float)
        self.f_max = np.asarray(g["f_max"], float)
        self.slew = float(g["slew"])
        self.yaw_kp = float(g["yaw_kp"]) * d * d
        self.yaw_kd = float(g["yaw_kd"]) * d
        self.yaw_ki = float(g["yaw_ki"]) * d ** 3
        self.yaw_i_max = float(g["yaw_i_max"])
        self.mz_max = float(g["mz_max"])
        self.e_gate = float(g["e_gate"])
        self.yaw_gate = float(g["yaw_gate"])
        self.dt = 1.0 / float(cfg.ctrl_hz)
        self.omega_derate = d
        log(f"pid: sim GAINS_HEAVY_GRIPPER x omega_derate {d} -> "
            f"kp {self.kp.round(1).tolist()} yaw_kp {self.yaw_kp:.2f} [예측]")
        self.reset()

    # ------------------------------------------------------------- reference
    # World-FLU sampler, same convention as the MPC bridge (reference.py is a
    # sim copy) — the PID consumes only the CURRENT sample (moving setpoint),
    # which is exactly how the sim benchmark drives its PID (run_compare
    # feeds set_target each tick; no preview).
    def set_target_ned(self, p_ned, yaw_ned, v_ned=None, r_ned=0.0):
        self._p_ref = np.asarray(p_ned, float).copy()
        self._yaw_ref = float(yaw_ned)
        self._ref_traj = None

    def set_square_ned(self, square: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        from .reference import make_square_ref_world

        size = float(square.get("size", 1.0))
        speed = float(square.get("speed", 0.12))
        laps = int(square.get("laps", 3))
        follow = bool(square.get("heading_follow", False))
        yaw_rate = math.radians(float(square.get("yaw_rate_deg_s", 60.0)))
        rot_ned = math.radians(float(square.get("rot_deg", 0.0)))
        T_run = laps * 4.0 * size / max(1e-6, speed)
        self._ref_traj = make_square_ref_world(
            size=size, speed=speed, depth=-float(depth_ned),
            heading_follow=follow, yaw_rate=yaw_rate, dt=self.dt,
            T_total=T_run + 1.0,
            origin_xy=(float(origin_ned_xy[0]), -float(origin_ned_xy[1])),
            rot_rad=-rot_ned, yaw_fixed=-float(yaw_fixed_ned))
        self.scenario = {"kind": "square", "size": size, "speed": speed,
                         "laps": laps, "depth_ned": float(depth_ned),
                         "heading_follow": follow,
                         "yaw_rate_deg_s": float(square.get("yaw_rate_deg_s", 60.0)),
                         "origin_ned": [float(origin_ned_xy[0]),
                                        float(origin_ned_xy[1])],
                         "rot_deg": float(square.get("rot_deg", 0.0)),
                         "yaw_fixed_ned_deg": math.degrees(yaw_fixed_ned),
                         "T_run_s": T_run}
        return self.scenario

    def ref_ned_at(self, t: float):
        if self._ref_traj is None:
            return self._p_ref.copy(), self._yaw_ref
        p, yaw, _v, _r = self._ref_traj(np.array([float(t)]))
        # sim-FLU -> NED mirror (S = diag(1,-1,-1); yaw negates)
        return (np.array([p[0, 0], -p[1, 0], -p[2, 0]]), -float(yaw[0]))

    # ------------------------------------------------------------------ tick
    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        p_ref, yaw_ref = self.ref_ned_at(t)

        # world error -> body frame (the frame the thrusters act in)
        R = rot_zyx(eta[3], eta[4], eta[5])
        e_b = R.T @ (p_ref - eta[:3])
        # gated integral + clamp (anti-windup, sim structure)
        for i in range(3):
            if abs(e_b[i]) < self.e_gate:
                self._i_xyz[i] += e_b[i] * self.dt
            self._i_xyz[i] = float(np.clip(
                self._i_xyz[i],
                -self.i_max[i] / max(1e-9, self.ki[i]),
                self.i_max[i] / max(1e-9, self.ki[i])))
        f = self.kp * e_b - self.kd * nu[:3] + self.ki * self._i_xyz
        f = np.clip(f, -self.f_max, self.f_max)

        e_psi = _wrap(yaw_ref - eta[5])
        if abs(e_psi) < self.yaw_gate:
            self._i_yaw += e_psi * self.dt
        self._i_yaw = float(np.clip(
            self._i_yaw, -self.yaw_i_max / max(1e-9, self.yaw_ki),
            self.yaw_i_max / max(1e-9, self.yaw_ki)))
        mz = self.yaw_kp * e_psi - self.yaw_kd * nu[5] + self.yaw_ki * self._i_yaw
        mz = float(np.clip(mz, -self.mz_max, self.mz_max))

        u = np.array([f[0], f[1], f[2], 0.0, 0.0, mz])
        # uniform slew limit (N/s and N·m/s, sim convention)
        du = np.clip(u - self._u_prev, -self.slew * self.dt, self.slew * self.dt)
        u = self._u_prev + du
        self._u_prev = u.copy()
        return u.copy(), {"w_hat": self.w_hat, "solve_ms": 0.05, "status": 0,
                          "n_fail": 0, "nis": 0.0}

    def note_applied(self, tau_ned_est) -> None:
        # The PID's own anti-windup (gates + clamps + slew) already bounds the
        # integrals; the realized-wrench estimate is not needed.
        pass

    def reset(self) -> None:
        self._p_ref = np.zeros(3)
        self._yaw_ref = 0.0
        self._ref_traj = None
        self.scenario = None
        self._i_xyz = np.zeros(3)
        self._i_yaw = 0.0
        self._u_prev = np.zeros(6)

    def meta(self) -> dict:
        return {"type": "pid", "solver": "pid",
                "ctrl_hz": float(self.cfg.ctrl_hz),
                "gains_provenance": "sim GAINS_HEAVY_GRIPPER "
                                    "(controller.py pole placement) x "
                                    f"omega_derate {self.omega_derate} [예측]",
                "kp": self.kp.tolist(), "kd": self.kd.tolist(),
                "ki": self.ki.tolist(),
                "yaw_kp": self.yaw_kp, "yaw_kd": self.yaw_kd,
                "yaw_ki": self.yaw_ki,
                "f_max": self.f_max.tolist(), "mz_max": self.mz_max,
                "ref_preview": False}
