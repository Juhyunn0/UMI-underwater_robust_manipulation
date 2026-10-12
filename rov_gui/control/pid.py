#!/usr/bin/env python3
"""pid.py — the PID baseline, same harness surface as the MPC bridge.

The gain SET is the simulator's pole-placed design, copied verbatim from
``bluerov2_mujoco_marinegym/controller.py`` ``GAINS_HEAVY_GRIPPER`` (ωn=2.0
translation / 3.0 yaw, ζ=0.9, α=0.2, re-evaluated at the heavy+gripper+C3
payload masses — provenance in that file). Structure ported likewise:
world-frame position error rotated into the body, per-axis PID with gated
(+clamped) integrals, wrench saturation and a slew limit; yaw its own PID on
the wrapped angle. Roll/pitch terms are NOT in HwPid: the 4-DoF MANUAL_CONTROL
frame carries no roll/pitch axis, so LOW ``pid`` emits K = M = 0 and stays a
4-DoF follower whatever ``engage.attitude_axes`` says. The roll/pitch loop
lives in :class:`HwPid7d` (LOW ``pid_7d``, 2026-10-03) at the bottom of this
file: the same translation/yaw PID plus a roll/pitch PID on the plan's
``rp_ned`` whose K/M ride the MANUAL_CONTROL s/t extension axes.

Two hardware deltas, both deliberate:
  * ``omega_derate`` scales the design frequency down (kp·d², kd·d, ki·d³ —
    the pole-placement scalings) because the sim's ωn assumed clean 20 Hz
    state and ideal thrust; the sim's own docs say "derate ωn for hardware".
    [예측] until a pool session tunes it.
  * heave trim (2026-10-01): ``hw_mpc.yaml vehicle_net_buoyancy_n`` (B - W of
    the vehicle as flown, N, + floats) is added to the heave force as a
    constant feed-forward, the same knob the NMPC consumes. Before this the
    heave integral alone carried the net buoyancy, and it is clamped at
    ``i_max`` 5 N — with the peg in the jaw the vehicle needs ~12 N up and
    sank to the floor under pid exactly as under mpc_tuned (2026-09-30/10-01
    runs). The integral still carries the residual. 0 = off.

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
        # Constant heave feed-forward: NED body z is down-positive, so a
        # vehicle that sinks (B - W < 0) needs a thrust of exactly B - W on
        # z (negative = up). Added before the f_max clip so it competes for
        # authority like any other term.
        self.f_z_trim = float(getattr(cfg, "vehicle_net_buoyancy_n", 0.0) or 0.0)
        if self.f_z_trim:
            log(f"pid: heave trim f_z {self.f_z_trim:+.2f} N (config "
                f"vehicle_net_buoyancy_n; the integral carries the rest, "
                f"i_max {float(self.i_max[2]):.0f} N)")
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
        tb = self._trim_force_body(eta)
        if tb is None:
            f[2] += self.f_z_trim
        else:                                   # HwPid7d: the trim rotated with the measured attitude
            f = f + tb
        f = np.clip(f, -self.f_max, self.f_max)

        e_psi = _wrap(yaw_ref - eta[5])
        if abs(e_psi) < self.yaw_gate:
            self._i_yaw += e_psi * self.dt
        self._i_yaw = float(np.clip(
            self._i_yaw, -self.yaw_i_max / max(1e-9, self.yaw_ki),
            self.yaw_i_max / max(1e-9, self.yaw_ki)))
        mz = (self.yaw_kp * e_psi + self.yaw_kd * (r_ref_ned - self._yaw_rate_meas(eta, nu))
              + self.yaw_ki * self._i_yaw)
        mz = float(np.clip(mz, -self.mz_max, self.mz_max))

        u = np.array([f[0], f[1], f[2], 0.0, 0.0, mz])
        # uniform slew limit (N/s and N·m/s, sim convention)
        du = np.clip(u - self._u_prev, -self.slew * self.dt, self.slew * self.dt)
        u = self._u_prev + du
        self._u_prev = u.copy()
        return u.copy(), {"w_hat": self.w_hat, "solve_ms": 0.05, "status": 0,
                          "n_fail": 0, "nis": 0.0}

    # Two hooks for HwPid7d (2026-10-03). Here they return exactly what the
    # inline code did — None keeps `f[2] += f_z_trim`, nu[5] is the body r —
    # so LOW pid is byte-identical.
    def _trim_force_body(self, eta):
        return None

    def _yaw_rate_meas(self, eta, nu) -> float:
        return nu[5]

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
                "heave_trim": {"applied": bool(self.f_z_trim),
                               "vehicle_net_buoyancy_n": float(self.f_z_trim),
                               "f_z_trim_n": float(self.f_z_trim),
                               "note": "constant feed-forward on body z, "
                                       "2026-10-01; 0 = off (pre-10-01 runs)"},
                "axes_note": "u = [X, Y, Z, 0, 0, N] — this controller has no "
                             "roll/pitch loop, so K and M are never commanded "
                             "(attitude_axes / attitude_track refused on pid)",
                "ref_preview": False,
                "path_reference": "shared spatial plan, stage 0"}


# ============================================================================ pid_7d
from .attitude_ref import AttitudeRefMixin, body_rates_from_euler  # noqa: E402

#: LOW ``pid_7d`` roll/pitch loop design (2026-10-03, control-theory-advisor
#: consult, .claude/journal/consults.md). Per-axis plant
#: I_eff*a'' + d*a' + k*a = g*tau(t - T). Kp/Kd by the SAME 3rd-order pole
#: placement as the translation/yaw set (CONTROL_METHODOLOGY.md "PID analytic
#: pole placement", k = d = 0 in the formula):
#:     Kp = I(1 + 2 zeta alpha) wn^2,  Kd = I(2 zeta + alpha) wn
#: but Ki is set APART (0.15 N m/(rad s)): the formula's I*alpha*wn^3 assumes
#: k = 0, and the integral actually works against k + Kp — at the formula's
#: 0.0125/0.022 its time constant at k 1.35 is 70-120 s [유도].
#: wn = 0.5 rad/s is a HARDWARE value set by the DELAY, not the inertia
#: (ATTITUDE ~52 ms old at the tick [측정 data/20261001/1001_154825/
#: mpc_154825.csv imu_age_s p50] + 25 ms ZOH + 25 ms mean sink phase + thruster
#: lag); omega_derate is NOT applied to it. Margins [유도, linear, d 0.07]:
#: T 0.2 s, actuator gain x0.4..x2.5 -> PM >= ~50 deg while the restoring
#: stiffness k <= ~2.3 (roll) / 3.6 (pitch) N m/rad (free-decay period >= 3.2
#: s); at k ~20 (the RL repo's lumped log fit) PM falls to 3-41 deg — the hull
#: resonance plus delay, which no gain fixes: a pulse-and-release test must
#: measure k/I, d, g and T BEFORE this loop is closed in water.
#: restoring FF tau_ff = k_ff*[cos(theta_r) sin(phi_r), sin(theta_r)] on the
#: REFERENCE angles (on the measured ones it is a negative stiffness), k_ff =
#: 1.35 = the NMPC's ZG*W, the low end of the k range, so it never
#: over-compensates while k >= 1.35; the integral fills the rest.
#: I_eff = rigid + rotational added mass of the heavy_gripper model [유도:
#: rov_model IX/IY + 0.12 placeholder]. Every number [예측] until the attitude
#: pulse/step test (docs/DP_6DOF_PLAN.ko.md §3).
PID7D_DEFAULTS = {
    "rp_wn": 0.5,                    # rad/s [예측] (delay-limited; NOT scaled by pid.omega_derate)
    "rp_zeta": 0.9,
    "rp_alpha": 0.2,
    "rp_inertia": (0.50154, 0.8978),  # kg m^2 roll, pitch [유도]
    "rp_k_model": (0.0, 0.0),        # N m/rad subtracted from the formula's Kp (0 = ignore, the 2026-10-03 design)
    "rp_d_model": (0.0, 0.0),        # N m s/rad subtracted from the formula's Kd
    "rp_kp": None, "rp_kd": None,    # explicit overrides (roll, pitch); None = the formula
    "rp_ki": (0.15, 0.15),           # N m/(rad s) — set apart from the formula (see above) [예측]
    "rp_k_ff": 1.35,                 # N m/rad restoring feed-forward on the reference angles; 0 = off [예측]
    "rp_i_max": (2.64, 2.16),        # |I| ceiling, N m (the wire share below usually binds)
    "rp_i_cap_frac": 0.8,            # |I| <= this x the wire torque in force (the rest is the P/D share)
    "rp_gate_rad": 0.35,             # integrate only while |e| < this (= the 20 deg reference clip)
    "rp_max_nm": (2.64, 2.16),       # |K|,|M| ceiling = full caps x roll_nm/pitch_nm [유도]; the caps in force win
    "trim_rotated": True,            # heave trim rotated with the measured attitude (world-vertical net buoyancy)
    # AUTHORITY EXHAUSTED (safety audit 2026-10-03, HIGH): with I clamped below the cap and a small Kp, the AXIS never
    # reaches its cap, so interlock (iii) — which judges cap-pinned ticks — could not see an inverted wire sign (the PID
    # settles at a WRONG tilt ~ i_lim/(k - kp) [유도]) nor a stiff hull it cannot tilt. The integral at its limit with an
    # error left counts as pinned for (iii): rp_pin_err_rad is that error [예측].
    "rp_pin_err_rad": 0.0175,        # 1 deg
    # PULSE-AND-RELEASE GATE (same audit, MEDIUM): on a stiff hull (free-decay period < ~3.2 s) this loop's phase margin
    # collapses at 0.2 s delay [유도] into a ~1 Hz limit cycle no interlock catches. A vehicle engage on pid_7d needs the
    # pinned test record {"free_decay_period_s": [roll, pitch], ...} with min >= min_free_decay_period_s, or the RECORDED
    # opt-out require_pulse_test: false.
    "require_pulse_test": True,
    "pulse_test": None,
    "min_free_decay_period_s": 3.2,
}


def _pair(v, name):
    a = np.asarray(v, float).ravel()
    if a.size == 1:
        a = np.repeat(a, 2)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError(f"pid_7d.{name} must be a number or a (roll, pitch) pair, got {v!r}")
    return a


class HwPid7d(AttitudeRefMixin, HwPid):
    """LOW ``pid_7d`` (2026-10-03): HwPid's translation + yaw loops UNCHANGED
    (same gains, same omega_derate) plus a roll/pitch PID that tracks the
    plan's ``rp_ned`` (policy 7-dim / replay track_attitude), else the D5 hold
    ramp, else level — active levelling. K/M leave the station only through
    the worker's attitude path (engage.attitude_axes, MANUAL_CONTROL s/t; the
    worker refuses to engage this LOW without it on a vehicle).

    Loop per axis (NED/FRD, +roll starboard-down, +pitch nose-up):
        e   = wrap(ref - meas)                          ATTITUDE roll/pitch
        de  = T^-1(ref angles)(ref Euler rates) - (p, q) ATTITUDE rollspeed/pitchspeed (gyro)
        tau = k_ff [cos(theta_r) sin(phi_r), sin(theta_r)] + kp e + kd de + I,
              clipped to min(rp_max_nm, wire torque in force)
    I (N m) += ki e dt only while |e| < rp_gate, clamped to min(rp_i_max,
    rp_i_cap_frac x wire torque); it is FROZEN while the previous tick's command
    was cut by the cap / slew in the direction the error pushes (conditional
    integration on the REALIZED K/M from note_applied) — otherwise it would pin
    the capped axis and trip interlock (iii) on a recoverable transient.
    """

    def __init__(self, cfg, log=print):
        self._att_init(cfg)
        super().__init__(cfg, log=log)
        self.mode = "pid_7d"
        raw = dict(getattr(cfg, "pid_7d", None) or {})
        unknown = set(raw) - set(PID7D_DEFAULTS)
        if unknown:
            raise ValueError(f"pid_7d: unknown key(s) {sorted(unknown)} (allowed: {sorted(PID7D_DEFAULTS)})")
        g = dict(PID7D_DEFAULTS)
        g.update(raw)
        self.rp_wn = float(g["rp_wn"])
        self.rp_zeta = float(g["rp_zeta"])
        self.rp_alpha = float(g["rp_alpha"])
        if not (self.rp_wn > 0 and self.rp_zeta > 0 and self.rp_alpha >= 0):
            raise ValueError("pid_7d: rp_wn and rp_zeta must be > 0 and rp_alpha >= 0")
        I = _pair(g["rp_inertia"], "rp_inertia")
        k = _pair(g["rp_k_model"], "rp_k_model")
        d = _pair(g["rp_d_model"], "rp_d_model")
        wn, z, al = self.rp_wn, self.rp_zeta, self.rp_alpha
        kp = I * (1.0 + 2.0 * z * al) * wn ** 2 - k
        kd = I * (2.0 * z + al) * wn - d
        src = []
        src.append("kp explicit" if g["rp_kp"] is not None else "kp = I(1+2 zeta alpha) wn^2 - k_model")
        src.append("kd explicit" if g["rp_kd"] is not None else "kd = I(2 zeta + alpha) wn - d_model")
        src.append("ki explicit (set apart from the formula)" if g["rp_ki"] is not None
                   else "ki = I alpha wn^3")
        self.rp_gains_source = "; ".join(src) + " [예측]"
        self.rp_kp = _pair(g["rp_kp"], "rp_kp") if g["rp_kp"] is not None else kp
        self.rp_kd = _pair(g["rp_kd"], "rp_kd") if g["rp_kd"] is not None else kd
        self.rp_ki = (_pair(g["rp_ki"], "rp_ki") if g["rp_ki"] is not None else I * al * wn ** 3)
        if np.any(self.rp_kp <= 0) or np.any(self.rp_kd < 0) or np.any(self.rp_ki < 0):
            raise ValueError(f"pid_7d: gains must be kp > 0, kd >= 0, ki >= 0 (kp {self.rp_kp}, kd {self.rp_kd}, "
                             f"ki {self.rp_ki}) — lower rp_k_model / rp_d_model or raise rp_wn")
        self.rp_inertia, self.rp_k_model, self.rp_d_model = I, k, d
        self.rp_k_ff = float(g["rp_k_ff"])
        if not (math.isfinite(self.rp_k_ff) and self.rp_k_ff >= 0.0):
            raise ValueError(f"pid_7d.rp_k_ff must be >= 0, got {g['rp_k_ff']!r}")
        self.rp_i_max = _pair(g["rp_i_max"], "rp_i_max")
        self.rp_i_cap_frac = float(g["rp_i_cap_frac"])
        if not (0.0 <= self.rp_i_cap_frac <= 1.0):
            raise ValueError(f"pid_7d.rp_i_cap_frac must be in [0, 1], got {g['rp_i_cap_frac']!r}")
        self.rp_gate = float(g["rp_gate_rad"])
        self.rp_max_nm = _pair(g["rp_max_nm"], "rp_max_nm")
        self.trim_rotated = bool(g["trim_rotated"])
        self.rp_pin_err = float(g["rp_pin_err_rad"])
        self.require_pulse_test = bool(g["require_pulse_test"])
        self.pulse_test = g["pulse_test"]
        self.min_free_decay_period_s = float(g["min_free_decay_period_s"])
        #: what the worker's engage gate read (path, sha1, periods, or the opt-out) — meta controller.rp_loop.pulse_test
        self.pulse_test_record = None
        log(f"pid_7d: roll/pitch PID wn {wn} zeta {z} alpha {al} -> kp {self.rp_kp.round(3).tolist()} "
            f"kd {self.rp_kd.round(3).tolist()} ki {self.rp_ki.round(3).tolist()}, restoring FF {self.rp_k_ff} N m/rad "
            f"[예측]; translation/yaw = the pid set. NOT FLOWN: measure k/I, d, gain and delay (pulse-and-release) first")
        self._rp_reset()

    # ------------------------------------------------------------------ loops
    def _rp_reset(self) -> None:
        self.n_fail = 0
        self._rp_exhausted = (False, False)
        self._i_rp = np.zeros(2)
        self._km_cmd = np.zeros(2)            # what the loop asked for last tick
        self._km_applied = None               # what the worker says left (note_applied), None = not told yet
        self._rp_err = np.zeros(2)
        self._rp_clip_ticks = 0

    def _trim_force_body(self, eta):
        """The heave trim is a WORLD-vertical net buoyancy: rotate it into the measured body frame (the NMPC's
        heave_trim_attitude_rotated form). Level -> exactly HwPid's body-z constant. trim_rotated false = HwPid's."""
        if not self.trim_rotated or not self.f_z_trim:
            return None
        phi, theta = float(eta[3]), float(eta[4])
        wz = float(self.f_z_trim)
        return np.array([-wz * math.sin(theta), wz * math.cos(theta) * math.sin(phi),
                         wz * math.cos(theta) * math.cos(phi)])

    def _yaw_rate_meas(self, eta, nu) -> float:
        """The yaw D term compares against an EULER yaw-rate reference; under a commanded tilt the body r is not it:
        psi_dot = (q sin(phi) + r cos(phi)) / cos(theta). Level -> r exactly."""
        phi, theta = float(eta[3]), float(eta[4])
        ct = math.cos(theta)
        if abs(ct) < 0.2:                     # |pitch| > 78 deg: outside every gate anyway
            return float(nu[5])
        return float((nu[4] * math.sin(phi) + nu[5] * math.cos(phi)) / ct)

    def _rp_limits(self) -> np.ndarray:
        lim = self.rp_max_nm.copy()
        if self._u_max_wire_nm is not None:
            lim = np.minimum(lim, np.abs(np.asarray(self._u_max_wire_nm, float)))
        return lim

    def _rp_torque(self, eta, nu, t, r_ref_ned: float) -> np.ndarray:
        phi, theta = float(eta[3]), float(eta[4])
        phr, thr = self.ref_attitude_ned_at(t)
        phd, thd = self.ref_attitude_rate_ned_at(t)
        p_ref, q_ref, _ = body_rates_from_euler(phr, thr, phd, thd, float(r_ref_ned))
        e = np.array([_wrap(phr - phi), _wrap(thr - theta)])
        de = np.array([p_ref - float(nu[3]), q_ref - float(nu[4])])
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(de))):
            raise ValueError("non-finite attitude / rate")       # step() turns it into a counted failure
        lim = self._rp_limits()
        i_lim = np.minimum(self.rp_i_max, self.rp_i_cap_frac * lim)
        for i in range(2):
            frozen = False
            if self._km_applied is not None:
                cut = self._km_cmd[i] - self._km_applied[i]          # >0: the wire carried less than asked (upward)
                frozen = abs(cut) > 1e-6 and np.sign(cut) == np.sign(e[i])
            if abs(e[i]) < self.rp_gate and not frozen:
                self._i_rp[i] += self.rp_ki[i] * e[i] * self.dt      # stored in N m: bumpless under a gain change
            self._i_rp[i] = float(np.clip(self._i_rp[i], -i_lim[i], i_lim[i]))
        # restoring feed-forward on the REFERENCE angles (g(eta) of the NMPC model, ZG*W -> k_ff)
        ff = self.rp_k_ff * np.array([math.cos(thr) * math.sin(phr), math.sin(thr)])
        tau = ff + self.rp_kp * e + self.rp_kd * de + self._i_rp
        clipped = np.clip(tau, -lim, lim)
        if np.any(clipped != tau):
            self._rp_clip_ticks += 1
        # authority exhausted per axis: the command cut at the ceiling, the integral at its limit with an error left,
        # or an error beyond the integration gate (the integral has stopped helping — an inverted wire sign drives the
        # hull AWAY and lands exactly here, below the cap, 2026-10-03 test). Interlock (iii) then judges progress.
        self._rp_exhausted = tuple(
            bool(clipped[i] != tau[i]
                 or (i_lim[i] > 0.0 and abs(self._i_rp[i]) >= i_lim[i] - 1e-9 and abs(e[i]) > self.rp_pin_err)
                 or abs(e[i]) >= self.rp_gate)
            for i in range(2))
        self._rp_err = e
        self._km_cmd = clipped.copy()
        return clipped

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        status = 0
        self._att_decay()                     # D5 FIRST: the reference read after the tick (CSV, (iii)) = the one used
        try:
            u, info = super().step(eta, nu, nudot_ned, t)
            _p, _yaw, _v, r_ref = self._reference_ned_at(t)
            km = self._rp_torque(eta, nu, t, r_ref)
            u = np.asarray(u, float).copy()
            u[3], u[4] = float(km[0]), float(km[1])
            if not np.all(np.isfinite(u)):
                raise ValueError("non-finite wrench")
        except (ValueError, FloatingPointError):
            # A NaN must never reach the allocation: max(-c, min(c, nan)) is +c (safety audit 2026-10-03). A counted
            # failure with a ZERO wrench instead — the worker disengages after engage.max_solver_fails in a row.
            self.n_fail += 1
            status = 1
            u = np.zeros(6)
            info = {"w_hat": self.w_hat, "solve_ms": 0.05, "nis": 0.0}
            self._rp_exhausted = (False, False)
        self._u_prev[3:5] = 0.0               # HwPid's uniform slew memory never carries K/M (the worker slews them)
        if not np.all(np.isfinite(self._u_prev)):
            self._u_prev = np.zeros(6)
        info = dict(info)
        info["status"] = status
        info["n_fail"] = self.n_fail
        info["rp_err"] = self._rp_err.copy()
        return u, info

    def attitude_authority_exhausted(self) -> tuple[bool, bool]:
        """Per axis (roll, pitch): has this loop run out of authority? The worker's interlock (iii) treats it as
        cap-pinned (safety audit 2026-10-03): this PID's integral ceiling sits BELOW the wire cap, so its axis alone
        would never pin."""
        return tuple(bool(v) for v in self._rp_exhausted)

    def note_applied(self, tau_ned_est) -> None:
        tau = np.asarray(tau_ned_est, float)
        self._km_applied = np.array([float(tau[3]), float(tau[4])])

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_rp_hold"):
            self._att_reset()
        if hasattr(self, "rp_kp"):
            self._rp_reset()

    def meta(self) -> dict:
        m = super().meta()
        m["type"] = "pid_7d"
        m["axes_note"] = ("u = [X, Y, Z, K, M, N]; X/Y/Z/N = the pid set unchanged, K/M = the roll/pitch PID below — they "
                          "reach the wire only with engage.attitude_axes live (allocation.attitude)")
        m["rp_loop"] = {
            "gains_source": self.rp_gains_source,
            "rp_wn": self.rp_wn, "rp_zeta": self.rp_zeta, "rp_alpha": self.rp_alpha,
            "rp_inertia_kgm2": self.rp_inertia.tolist(), "rp_k_model_nm_per_rad": self.rp_k_model.tolist(),
            "rp_d_model_nms_per_rad": self.rp_d_model.tolist(),
            "kp_nm_per_rad": self.rp_kp.tolist(), "kd_nms_per_rad": self.rp_kd.tolist(),
            "ki_nm_per_rad_s": self.rp_ki.tolist(),
            "k_ff_nm_per_rad": self.rp_k_ff,
            "ff": "k_ff*[cos(theta_r) sin(phi_r), sin(theta_r)] on the REFERENCE angles",
            "i_max_nm": self.rp_i_max.tolist(), "i_cap_frac": self.rp_i_cap_frac,
            "gate_rad": self.rp_gate, "max_nm": self.rp_max_nm.tolist(),
            "integral_units": "N m (ki applied inside the integral)",
            "anti_windup": "gate |e| + clamp to min(i_max, i_cap_frac x wire torque) + frozen while the realized K/M "
                           "was cut toward e (conditional integration)",
            "omega_derate_applied": False,
            "d_term": "T^-1(ref angles)(ref Euler rates, ref yaw rate) - ATTITUDE (p, q)",
            "clip_ticks": int(self._rp_clip_ticks),
            "pin_err_rad": self.rp_pin_err,
            "authority_exhausted_feeds_interlock_iii": True,
            "pulse_test": self.pulse_test_record,
            "require_pulse_test": self.require_pulse_test,
            "min_free_decay_period_s": self.min_free_decay_period_s,
            "yaw_d_term": "Euler yaw rate (q sin phi + r cos phi)/cos theta (pid: body r)",
        }
        m["heave_trim"]["attitude_rotated"] = bool(self.trim_rotated and self.f_z_trim)
        m.update(self._att_meta())
        return m
