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
far as MpcWorker cares (including the shared geometric-path plan API), so the
worker swaps controllers without special cases. No casadi, no acados — this
module imports instantly, which is also why the PID is always available even
when the solver build fails.
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

    #: May a ``follow`` mission be armed on this controller? Yes — the PID
    #: consumes the CURRENT sample every tick (a moving setpoint is its
    #: native mode) and it uses the reference velocity as a feedforward.
    follow_ok = True

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
        self._v_ref = (np.zeros(3) if v_ned is None
                       else np.asarray(v_ned, float).copy())
        self._r_ref = float(r_ned)
        self._ref_traj = None
        self._path_plan = None

    # PLACEMENT IS NOT THE CONTROLLER'S BUSINESS. Both arming methods delegate
    # to reference.place_*_ned, the single definition HwDobMpc also uses, so a
    # PID run and an MPC run of the same mission fly the SAME geometry — which
    # is the entire premise of comparing them. This is not hypothetical: while
    # this module owned its own copy it silently ignored `size_y`, so an
    # operator who entered 2.00 x 0.50 m got a rectangle under MPC and a
    # 2.00 x 2.00 m square under PID (2026-08-14). The only thing the PID
    # supplies is its own dt and no preview — it consumes the CURRENT sample
    # only (moving setpoint), exactly as the sim benchmark drives its PID.
    def set_square_ned(self, square: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        from .reference import place_square_ned

        self._ref_traj, self.scenario = place_square_ned(
            square, origin_ned_xy, yaw_fixed_ned, depth_ned, dt=self.dt)
        self._path_plan = None
        return self.scenario

    def set_line_ned(self, line: dict, origin_ned_xy, yaw_fixed_ned: float,
                     depth_ned: float):
        """Same contract as HwDobMpc.set_line_ned. Its absence is what made a
        PID line mission reach its start point and then never begin
        (2026-08-14): the worker armed, the attribute did not exist."""
        from .reference import place_line_ned

        self._ref_traj, self.scenario = place_line_ned(
            line, origin_ned_xy, yaw_fixed_ned, depth_ned, dt=self.dt)
        self._path_plan = None
        return self.scenario

    def set_circle_ned(self, circle: dict, origin_ned_xy, yaw_fixed_ned: float,
                       depth_ned: float):
        """Same contract as HwDobMpc.set_circle_ned, same placement call. It
        exists at all because of the lesson in the block comment above: a
        shape the MPC can arm and the PID cannot is a mission that dies
        silently at START, and that is how a PID line run reached its start
        point and then never began (2026-08-14)."""
        from .reference import place_circle_ned

        self._ref_traj, self.scenario = place_circle_ned(
            circle, origin_ned_xy, yaw_fixed_ned, depth_ned, dt=self.dt)
        self._path_plan = None
        return self.scenario

    @property
    def path_plan_steps(self) -> int:
        """PID consumes stage 0 of the same plan the MPC previews."""
        return 1

    @property
    def path_plan_dt(self) -> float:
        return self.dt

    def set_path_plan_ned(self, plan) -> None:
        """Install one geometry-driven NED plan made by the worker follower."""
        if plan is None:
            self._path_plan = None
            return
        p = np.asarray(plan.p_ned, float)
        yaw = np.asarray(plan.yaw_ned, float).ravel()
        v = np.asarray(plan.v_ned, float)
        r = np.asarray(plan.r_ned, float).ravel()
        if (p.shape != (3, 1) or v.shape != (3, 1)
                or yaw.size != 1 or r.size != 1):
            raise ValueError("PID path plan must contain exactly one stage")
        self._path_plan = plan

    def _reference_ned_at(self, t: float):
        """Internal position/yaw/world-velocity/yaw-rate NED reference."""
        if self._path_plan is not None:
            plan = self._path_plan
            return (np.asarray(plan.p_ned[:, 0], float).copy(),
                    float(plan.yaw_ned[0]),
                    np.asarray(plan.v_ned[:, 0], float).copy(),
                    float(plan.r_ned[0]))
        if self._ref_traj is None:
            return (self._p_ref.copy(), self._yaw_ref, self._v_ref.copy(),
                    self._r_ref)
        p, yaw, v, r = self._ref_traj(np.array([float(t)]))
        # sim-FLU -> NED mirror (S = diag(1,-1,-1); yaw negates)
        return (np.array([p[0, 0], -p[1, 0], -p[2, 0]]),
                -float(yaw[0]),
                np.array([v[0, 0], -v[1, 0], -v[2, 0]]),
                -float(r[0]))

    def ref_ned_at(self, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        """Current ``(position, yaw, world velocity)`` reference in NED."""
        p, yaw, v, _r = self._reference_ned_at(t)
        return p, yaw, v

    # ------------------------------------------------------------------ tick
    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        p_ref, yaw_ref, v_ref_ned, r_ref_ned = self._reference_ned_at(t)

        # world error -> body frame (the frame the thrusters act in)
        R = rot_zyx(eta[3], eta[4], eta[5])
        e_b = R.T @ (p_ref - eta[:3])
        v_ref_b = R.T @ v_ref_ned
        # gated integral + clamp (anti-windup, sim structure)
        for i in range(3):
            if abs(e_b[i]) < self.e_gate:
                self._i_xyz[i] += e_b[i] * self.dt
            self._i_xyz[i] = float(np.clip(
                self._i_xyz[i],
                -self.i_max[i] / max(1e-9, self.ki[i]),
                self.i_max[i] / max(1e-9, self.ki[i])))
        # Track velocity in the same body frame as nu. For a stationary
        # reference this reduces exactly to the previous ``-kd * nu`` term.
        f = (self.kp * e_b + self.kd * (v_ref_b - nu[:3])
             + self.ki * self._i_xyz)
        f = np.clip(f, -self.f_max, self.f_max)

        e_psi = _wrap(yaw_ref - eta[5])
        if abs(e_psi) < self.yaw_gate:
            self._i_yaw += e_psi * self.dt
        self._i_yaw = float(np.clip(
            self._i_yaw, -self.yaw_i_max / max(1e-9, self.yaw_ki),
            self.yaw_i_max / max(1e-9, self.yaw_ki)))
        mz = (self.yaw_kp * e_psi + self.yaw_kd * (r_ref_ned - nu[5])
              + self.yaw_ki * self._i_yaw)
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
        self._v_ref = np.zeros(3)
        self._r_ref = 0.0
        self._ref_traj = None
        self._path_plan = None
        self.scenario = None
        self._i_xyz = np.zeros(3)
        self._i_yaw = 0.0
        self._u_prev = np.zeros(6)

    def meta(self) -> dict:
        """EVERY constant this controller was flown with, not just the gains.

        A run is only reproducible from what is written down, and an anti-windup
        gate or a slew limit changes the trajectory as surely as kp does — the
        2026-08-14 operator request ("내가 조정하는 것들을 적어두는거지") is the
        whole list, so the whole list is here. ``omega_derate`` is recorded
        separately from the scaled gains because it is the knob that is actually
        turned; the kp/kd/ki below are already scaled by it."""
        return {"type": "pid", "solver": "pid",
                "ctrl_hz": float(self.cfg.ctrl_hz), "dt_s": float(self.dt),
                "gains_provenance": "sim GAINS_HEAVY_GRIPPER "
                                    "(controller.py pole placement) x "
                                    f"omega_derate {self.omega_derate} [예측]",
                "omega_derate": float(self.omega_derate),
                "gain_scaling": "kp ~ wn^2, kd ~ wn, ki ~ wn^3 — the values "
                                "below are ALREADY scaled by omega_derate",
                "kp": self.kp.tolist(), "kd": self.kd.tolist(),
                "ki": self.ki.tolist(),
                "yaw_kp": self.yaw_kp, "yaw_kd": self.yaw_kd,
                "yaw_ki": self.yaw_ki,
                # anti-windup + authority: the rest of what makes the loop
                "i_max_n": self.i_max.tolist(),
                "yaw_i_max_nm": float(self.yaw_i_max),
                "e_gate_m": float(self.e_gate),
                "yaw_gate_rad": float(self.yaw_gate),
                "f_max": self.f_max.tolist(), "mz_max": self.mz_max,
                "slew_n_per_s": float(self.slew),
                "axes_note": "u = [X, Y, Z, 0, 0, N] — MANUAL_CONTROL carries "
                             "no roll/pitch axis, so K and M are not commanded",
                "ref_preview": False,
                "path_reference": "shared spatial plan, stage 0"}
