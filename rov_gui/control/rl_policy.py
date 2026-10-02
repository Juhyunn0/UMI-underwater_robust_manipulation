#!/usr/bin/env python3
"""rl_policy.py — the RL low-level controller (LOW mode ``rl``), same harness surface as HwPid / HwDobMpc.

Policy provenance: /home/bdml/Desktop/RL_controller (2026-09-21, "Learning to Swim" reproduction for this vehicle).
Trained in Isaac Lab on the heavy+C3+gripper model with the ArduSub vectored-6DOF mixer, tag-PnP/IMU sensing simulated
and the station's own velocity estimator (state_assembler: finite-difference + LPF alpha 0.6) in the loop. The exported
actor (numpy weights + ``obs_spec.json``) is shipped under ``rov_gui/control/rl_policies/<name>/``.

Contract with the worker (identical to HwPid): the reference API (set_target_ned / set_square_ned / set_line_ned /
set_circle_ned / set_path_plan_ned / ref_ned_at) is INHERITED unchanged, so an RL run flies the SAME placed geometry as a PID
or MPC run. ``step`` returns a wrench-shaped vector u = [X, Y, Z, 0, 0, N]; the worker's ``wrench_to_axes`` divides it by
``cfg.axis_gain`` and caps it, so u is built as ``axis * gain`` (a PSEUDO-wrench) and the policy's normalized axes come out
the other end exactly (up to the cap). ``note_applied`` receives ``axes_to_wrench(sent axes)`` and inverts it: the policy
was trained with the APPLIED command fed back as its previous action, so the capped/slewed value must be what it sees.

Frames. The policy was trained in the simulator's world z-up / body FLU frames with yaw counter-clockwise positive; the
station's datum frame is NED/FRD. The conversion is the mirror C = diag(1,-1,-1) applied to positions and velocities, with
roll unchanged and pitch / yaw / yaw-rate negated — the same mapping the sim's PID reference sampler uses (pid.py
_reference_ned_at). Body-frame vectors: (u, v, w)_FRD -> (u, -v, -w)_FLU, (p, q, r) -> (p, -q, -r).

Observation frame (25 floats, newest-first history of ``history_len`` frames):
    [q_des(w,x,y,z), offset_b(3), q_est(w,x,y,z), v_b(3), gyro_b(3), goal_v_b(3), goal_yaw_rate(1), prev_action(4)]
    q_* canonicalised to w >= 0 (obs_spec.obs_quat_unique); offset clipped to +-obs_offset_clip [m].
Action: [surge(+fwd), sway(+right), heave(+up), yaw(+clockwise from above)] in [-1, 1] == MANUAL_CONTROL convention.

Assumptions the policy carries (recorded in meta()): vehicle JS_GAIN_DEFAULT 0.5 (obs_spec.pilot_gain_assumed) — a
different pilot gain rescales every command; ArduSub MANUAL mode; 20 Hz tick (obs_spec.control_dt).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .pid import HwPid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POLICY_DIR = os.path.join("rov_gui", "control", "rl_policies", "traj5_s2")
C_NED_TO_UP = np.diag([1.0, -1.0, -1.0])
#: axes6 (2026-09-27): the policy's pitch/roll actions are fractions of the +-400 us channel (ArduSub adds MANUAL_CONTROL s/t as
#: raw microseconds), while the station's attitude axes are 1000 us per unit (hardware.py packs s = pitch * 1000). So one
#: policy unit = 0.4 station units; K/M in the pseudo-wrench use the station's roll_nm / pitch_nm on the STATION unit.
STATION_ATT_AXIS_PER_ACTION = 0.4


# ----------------------------------------------------------------------------- vendored helpers (RL_controller/deploy)
def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


_ACTS = {"elu": _elu, "relu": lambda x: np.maximum(x, 0.0), "tanh": np.tanh}


class NumpyPolicy:
    """Exported actor MLP (RL_controller/deploy/policy_numpy.py, verbatim behaviour): obs -> mean action, clipped to [-1, 1]."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=False)
        n = int(d["n_layers"])
        self.layers = [(d[f"W{i}"].astype(np.float64), d[f"b{i}"].astype(np.float64)) for i in range(n)]
        self.activation = _ACTS[str(d["activation"])]
        self.obs_mean = d["obs_mean"].astype(np.float64) if "obs_mean" in d else None
        self.obs_std = np.sqrt(d["obs_var"].astype(np.float64)) if "obs_var" in d else None
        self.obs_dim = self.layers[0][0].shape[1]
        self.act_dim = self.layers[-1][0].shape[0]

    def act(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64).reshape(1, self.obs_dim)
        if self.obs_mean is not None:
            x = (x - self.obs_mean) / self.obs_std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return np.clip(x[0], -1.0, 1.0)


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w,x,y,z) body->world for Rz(yaw) Ry(pitch) Rx(roll) — Isaac Lab's quat_from_euler_xyz."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return np.array([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                     cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp])


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """world -> body for a body quaternion (w,x,y,z)."""
    xyz = q[1:]
    t = np.cross(xyz, v) * 2
    return v - q[0] * t + np.cross(xyz, t)


def _unique(q: np.ndarray) -> np.ndarray:
    return -q if q[0] < 0 else q


# ----------------------------------------------------------------------------- the controller
class HwRl(HwPid):
    """RL follower with HwPid's harness surface (reference API inherited)."""

    follow_ok = True
    #: Which config block names the policy, and which action modes this class flies. HwRlPwm (LOW mode ``rl_pwm``,
    #: 2026-09-30) overrides both; nothing else differs in how the export is read.
    CFG_KEY = "rl"
    ACCEPTED_MODES = ("axes", "axes6")
    DEFAULT_DIR = DEFAULT_POLICY_DIR

    def __init__(self, cfg, log=print):
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        pdir = str(raw.get("policy_dir", self.DEFAULT_DIR))
        if not os.path.isabs(pdir):
            pdir = os.path.join(REPO_ROOT, pdir)
        self.policy_dir = pdir
        with open(os.path.join(pdir, "obs_spec.json"), "r", encoding="utf-8") as f:
            self.spec = json.load(f)
        self.policy = NumpyPolicy(os.path.join(pdir, "policy_weights.npz"))
        # 2026-09-27: RL_controller also trains a "pwm" policy (8 per-thruster throttles, mixer bypassed). The station's only
        # transport is MANUAL_CONTROL (allocation.py), which cannot carry per-thruster pulses, so such a policy is refused
        # HERE, at build time, with the reason — never silently mapped onto the 4 axes. It flies through
        # RL_controller/deploy/run_policy.py (pwm_lua_override: RC_CHANNELS_OVERRIDE ch 9..16 read by a Navigator Lua
        # script that forces the motor outputs with a 100 ms timeout while armed + MANUAL; motors stay on the mixer, so
        # ARM/DISARM and every failsafe keep working — RCIN passthrough was rejected for bypassing them).
        self.action_mode = str(self.spec.get("action_mode", "axes"))
        if self.action_mode == "pwm" and "pwm" not in self.ACCEPTED_MODES:
            raise ValueError(f"rl: policy {os.path.relpath(pdir, REPO_ROOT)} is a '{self.action_mode}' policy "
                             f"({self.spec.get('num_actions')} outputs); LOW mode rl has only the MANUAL_CONTROL 4-axis transport "
                             "— fly it with RL_controller/deploy/run_policy.py (transport pwm_lua_override: RC_CHANNELS_OVERRIDE "
                             "ch 9..16 + the Navigator Lua timeout override; NOT the station's reserved rc_override ch 1..6), "
                             "name it under `rl_pwm:` and pick LOW RL_PWM, or export an axes policy")
        if self.action_mode != "pwm" and self.ACCEPTED_MODES == ("pwm",):
            raise ValueError(f"rl_pwm: policy {os.path.relpath(pdir, REPO_ROOT)} is a '{self.action_mode}' policy — LOW "
                             f"mode rl_pwm flies per-thruster 'pwm' exports only; name an axes policy under `rl:`")
        # axes6 (2026-09-27): the policy also commands pitch (a4) and roll (a5). They ride the 6-DoF variant's extension
        # axes (engage.attitude_axes, 2026-09-26) as K = a5*roll_nm, M = a4*pitch_nm in the pseudo-wrench; without that
        # switch the worker would DROP K/M and the policy would fly with two of its outputs silently ignored -> refuse.
        att = (getattr(cfg, "engage", None) or {}).get("attitude_axes") or {}
        if self.action_mode == "axes6" and not bool(att.get("enabled", False)):
            raise ValueError(f"rl: policy {os.path.relpath(pdir, REPO_ROOT)} is a 6-axis policy (surge, sway, heave, yaw, pitch, "
                             "roll) but engage.attitude_axes.enabled is false: its pitch/roll outputs would be dropped by "
                             "allocation.wrench_to_axes. Enable engage.attitude_axes (with its probe / sign gates) or use a 4-axis policy")
        if self.action_mode not in self.ACCEPTED_MODES:
            raise ValueError(f"rl: unknown action_mode {self.action_mode!r} in {os.path.relpath(pdir, REPO_ROOT)}")
        self.K = int(self.spec["history_len"])
        self.frame_dim = int(self.spec["frame_dim"])
        self.n_act = int(self.spec["num_actions"])
        self.clip_m = float(self.spec.get("obs_offset_clip", 1.0))
        self.quat_unique = bool(self.spec.get("obs_quat_unique", True))
        self.goal_velocity = bool(self.spec.get("obs_goal_velocity", False))
        if self.policy.obs_dim != self.frame_dim * self.K or self.policy.act_dim != self.n_act:
            raise ValueError(f"rl: policy dims {self.policy.obs_dim}->{self.policy.act_dim} do not match obs_spec "
                             f"({self.frame_dim}x{self.K} -> {self.n_act})")
        exp_dt = float(self.spec.get("control_dt", 1.0 / float(cfg.ctrl_hz)))
        if abs(exp_dt - 1.0 / float(cfg.ctrl_hz)) > 1e-6:
            raise ValueError(f"rl: policy trained at {1/exp_dt:.1f} Hz but ctrl_hz is {cfg.ctrl_hz}")
        self._hist = None
        self._prev_action = np.zeros(self.n_act)
        self._gains = None
        super().__init__(cfg, log=lambda m: None)      # HwPid gains are irrelevant; its reference API is what we inherit
        self.mode = "rl"
        self.solver_kind = "rl"
        self.realtime_ok = True
        self.build_s = 0.0
        self.n_fail = 0
        #: The worker honours this instead of cfg.axis_slew_per_s: the policy was trained WITHOUT a command slew and
        #: with its own previous command fed back, so an external rate limit would be an unmodelled actuator.
        self.axis_slew_per_s = float(raw.get("axis_slew_per_s", 0.0))
        #: |axis| ceiling while this controller flies (the worker takes min(axis_cap, this)). None = the config's
        #: axis_cap. 0.3 caps the steady-state speed near 0.14 m/s by the fitted speed law (0.692*(|axis|-0.096)).
        _cap = raw.get("axis_cap", 0.3)
        self.axis_cap = None if _cap is None else float(_cap)
        #: Goal leash [m]: the policy never saw goal offsets beyond obs_offset_clip (1.0 m, where the clip destroys the
        #: bearing), so a far target (station hold from metres away) is pulled to this distance along the true bearing.
        self.leash_m = float(raw.get("leash_m", 0.8))
        #: The K=1 path plan carries r_ned == 0 even while the heading slews (path_geometry.py), so the yaw-rate
        #: feed-forward is recovered by differencing the reference yaw between ticks, clipped to the trained range.
        self.yaw_rate_ff = bool(raw.get("yaw_rate_ff", True))
        self.yaw_rate_max = float(raw.get("yaw_rate_max", 0.6))
        self._yaw_ref_prev = None
        self._gains = dict(cfg.axis_gain)
        # inference cost (numpy, 3-layer MLP): measured once so MpcStatus shows a real probe
        t0 = time.perf_counter()
        for _ in range(20):
            self.policy.act(np.zeros(self.policy.obs_dim))
        self.probe_ms = 1e3 * (time.perf_counter() - t0) / 20.0
        # a pwm export carries pilot_gain_assumed: null (no mixer, no pilot gain); HwRlPwm reports None in its meta
        _pg = self.spec.get("pilot_gain_assumed", 0.5)
        self.pilot_gain_assumed = 0.5 if _pg is None else float(_pg)
        log(f"rl: policy {os.path.relpath(pdir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), obs {self.policy.obs_dim} "
            f"(K={self.K}, feed-forward={'on' if self.goal_velocity else 'off'}), act {self.n_act}, probe {self.probe_ms:.2f} ms; "
            f"assumes JS_GAIN_DEFAULT {self.pilot_gain_assumed} and no axis slew [스펙 obs_spec.json]")

    # ------------------------------------------------------------------ observation
    def _frame(self, eta, nu, p_ref, yaw_ref, v_ref_ned, r_ref_ned) -> np.ndarray:
        """One observation frame in the simulator's frames (world z-up / body FLU) from the datum-NED state."""
        p_up = C_NED_TO_UP @ eta[:3]
        roll, pitch, yaw = float(eta[3]), -float(eta[4]), -float(eta[5])
        q_est = quat_from_euler_xyz(roll, pitch, yaw)
        v_b = np.array([nu[0], -nu[1], -nu[2]])
        gyro = np.array([nu[3], -nu[4], -nu[5]])
        goal_p = C_NED_TO_UP @ np.asarray(p_ref, float)
        d = goal_p - p_up
        n = float(np.linalg.norm(d))
        if n > self.leash_m:                       # norm-preserving leash: same bearing, in-distribution distance
            d = d * (self.leash_m / n)
        q_des = quat_from_euler_xyz(0.0, 0.0, -float(yaw_ref))
        offset_b = np.clip(quat_apply_inverse(q_est, d), -self.clip_m, self.clip_m)
        q_d, q_o = (_unique(q_des), _unique(q_est)) if self.quat_unique else (q_des, q_est)
        parts = [q_d, offset_b, q_o, v_b, gyro]
        if self.goal_velocity:
            goal_v = C_NED_TO_UP @ np.asarray(v_ref_ned, float)
            parts += [quat_apply_inverse(q_est, goal_v), np.array([-float(r_ref_ned)])]
        return np.concatenate(parts + [self._prev_action])

    def observation(self, eta_ned, nu_ned, t: float) -> np.ndarray:
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        p_ref, yaw_ref, v_ref, r_ref = self._reference_ned_at(t)
        if self.yaw_rate_ff and r_ref == 0.0:
            if self._yaw_ref_prev is not None:
                r_ref = math.atan2(math.sin(yaw_ref - self._yaw_ref_prev), math.cos(yaw_ref - self._yaw_ref_prev)) / self.dt
                r_ref = float(np.clip(r_ref, -self.yaw_rate_max, self.yaw_rate_max))
            self._yaw_ref_prev = float(yaw_ref)
        frame = self._frame(eta, nu, p_ref, yaw_ref, v_ref, r_ref)
        if self._hist is None:
            self._hist = np.tile(frame, (self.K, 1))
        else:
            self._hist = np.roll(self._hist, 1, axis=0)
            self._hist[0] = frame
        return self._hist.reshape(-1)

    # ------------------------------------------------------------------ tick
    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        t0 = time.perf_counter()
        obs = self.observation(eta_ned, nu_ned, t)
        a = self.policy.act(obs)
        self.last_action = a.copy()
        g = self._gains
        # pseudo-wrench: wrench_to_axes(u) == a (before the cap); sway +right = +Y_ned, heave +up = -Z_ned, yaw +cw = +N_ned;
        # axes6: M = a4*pitch_nm, K = a5*roll_nm (wrench_to_axes(attitude=True) recovers them, with its own caps)
        k_nm = STATION_ATT_AXIS_PER_ACTION * a[5] * float(g.get("roll_nm", 13.2)) if self.n_act == 6 else 0.0
        m_nm = STATION_ATT_AXIS_PER_ACTION * a[4] * float(g.get("pitch_nm", 7.2)) if self.n_act == 6 else 0.0
        u = np.array([a[0] * g["surge_n"], a[1] * g["sway_n"], -a[2] * g["heave_n"], k_nm, m_nm, a[3] * g["yaw_nm"]])
        ms = 1e3 * (time.perf_counter() - t0)
        return u, {"w_hat": self.w_hat, "solve_ms": ms, "status": 0, "n_fail": 0, "nis": 0.0}

    def note_applied(self, tau_ned_est) -> None:
        """Inverse of allocation.axes_to_wrench: the axes that actually LEFT the station (cap, yaw hold, slew) become
        the policy's previous action, exactly as the applied action was fed back during training."""
        tau = np.asarray(tau_ned_est, float)
        g = self._gains
        prev = [tau[0] / g["surge_n"], tau[1] / g["sway_n"], -tau[2] / g["heave_n"], tau[5] / g["yaw_nm"]]
        if self.n_act == 6:
            prev += [tau[4] / float(g.get("pitch_nm", 7.2)) / STATION_ATT_AXIS_PER_ACTION,
                     tau[3] / float(g.get("roll_nm", 13.2)) / STATION_ATT_AXIS_PER_ACTION]
        self._prev_action = np.clip(np.array(prev), -1.0, 1.0)

    def reset(self) -> None:
        super().reset()
        self._hist = None
        self._yaw_ref_prev = None
        self._prev_action = np.zeros(getattr(self, "n_act", 4))
        self.last_action = np.zeros(getattr(self, "n_act", 4))

    def meta(self) -> dict:
        return {"type": "rl", "solver": "rl", "ctrl_hz": float(self.cfg.ctrl_hz), "dt_s": float(self.dt),
                "policy_dir": os.path.relpath(self.policy_dir, REPO_ROOT), "action_mode": self.action_mode,
                "checkpoint": self.spec.get("checkpoint"), "obs_dim": int(self.policy.obs_dim),
                "history_len": self.K, "frame_layout": self.spec.get("frame_layout"),
                "obs_quat_unique": self.quat_unique, "obs_goal_velocity": self.goal_velocity,
                "obs_offset_clip_m": self.clip_m, "dr_preset": self.spec.get("dr_preset"),
                "pilot_gain_assumed": self.pilot_gain_assumed,
                "axis_gain_for_pseudo_wrench": dict(self._gains),
                "axis_slew_per_s": self.axis_slew_per_s, "axis_cap": self.axis_cap, "leash_m": self.leash_m,
                "yaw_rate_ff": self.yaw_rate_ff, "yaw_rate_max": self.yaw_rate_max,
                "axes_note": "u = axes * axis_gain (pseudo-wrench); the worker's wrench_to_axes returns the policy axes, "
                             "capped by axis_cap; roll/pitch not commanded",
                "frames": "policy world z-up / body FLU; station NED/FRD mirrored with diag(1,-1,-1)",
                "ref_preview": False, "path_reference": "shared spatial plan, stage 0 (+ its velocity as feed-forward)",
                "provenance": "RL_controller (Isaac Lab, PPO), see rl_policies/<name>/RL_controller_README.md"}


# ----------------------------------------------------------------------------- per-thruster PWM (LOW mode ``rl_pwm``)
class HwRlPwm(HwRl):
    """RL follower whose 8 outputs are per-thruster throttles (RL_controller ``action_mode: pwm``) — LOW mode ``rl_pwm``.

    Everything about the OBSERVATION is HwRl's: the same frame, the same reference API, the same NED -> z-up mirror; the
    frame is only wider (the previous action is 8 throttles, not 4 axes). What differs is the output. There is no
    wrench to hand the worker's ``wrench_to_axes``: ``step`` puts the pulses in ``info["pwm_us"]`` and the worker sends
    THOSE on ``bus.cmd_pwm`` (rc override ch 9..16 -> the vehicle's Lua timeout override, control/rl_pwm.py) beside a
    NEUTRAL MANUAL_CONTROL. The ``u`` it returns is the nominal steady-state wrench of the capped throttles, for the
    CSV's uX..uN — comparable with the other controllers' columns only as an estimate, never as a command.

    ``pwm_cap`` is the |throttle| ceiling on the wire (fraction of +-400 us). The policy was trained at 1.0; the shipped
    config says 0.25 because this path has never moved the vehicle (rl_pwm.PWM_CAP_FIRST_WATER). The capped pulse is
    what the network is fed back as its previous action, as in training.

    UNTESTED ON THE VEHICLE (2026-09-30) — see control/rl_pwm.py for the gate that has to pass first.
    """

    CFG_KEY = "rl_pwm"
    ACCEPTED_MODES = ("pwm",)
    DEFAULT_DIR = os.path.join("rov_gui", "control", "rl_policies", "pwm10_s1")
    #: The worker reads this to pick the transport: "pwm" = pulses on bus.cmd_pwm, anything else = axes on cmd_pilot.
    transport = "pwm"

    def __init__(self, cfg, log=print):
        from .rl_pwm import PWM_CAP_FIRST_WATER, PWM_CHANNELS_DEFAULT, PwmModel
        super().__init__(cfg, log=lambda m: None)
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        if self.n_act != 8:
            raise ValueError(f"rl_pwm: policy has {self.n_act} outputs, the transport carries 8 thrusters")
        out = self.spec.get("output") or {}
        if str(out.get("kind")) != "pwm" or not isinstance(out.get("pwm_model"), dict):
            raise ValueError("rl_pwm: obs_spec.json has no output.pwm_model (re-export with RL_controller >= 2026-09-27)")
        self.model = PwmModel(out["pwm_model"])
        _cap = raw.get("pwm_cap", PWM_CAP_FIRST_WATER)
        if isinstance(_cap, bool) or not isinstance(_cap, (int, float)) \
                or not (math.isfinite(float(_cap)) and 0.0 < float(_cap) <= 1.0):
            raise ValueError(f"rl_pwm.pwm_cap must be a number in (0, 1], got {_cap!r}")
        cap = float(_cap)
        #: never above what the policy was trained with (the env's own cap), whatever the config says
        self.pwm_cap = min(cap, self.model.cap_trained)
        #: fixed: the vehicle's script maps RC channel 8+m to motor m (control/rl_pwm._fixed_channels)
        self.pwm_channels = tuple(PWM_CHANNELS_DEFAULT)
        self.mode = "rl_pwm"
        self.solver_kind = "rl_pwm"
        self.axis_cap = None                       # no axes leave under this controller
        self.last_pwm_us = None
        log(f"rl_pwm: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), obs "
            f"{self.policy.obs_dim} (K={self.K}), 8 thruster throttles, |throttle| <= {self.pwm_cap:.2f} on the wire "
            f"(trained {self.model.cap_trained:.2f}), rc channels {list(self.pwm_channels)}, expects MOT_1..8_DIRECTION "
            f"{self.model.motor_direction}; transport {self.model.transport} — UNTESTED on the vehicle [예측]")

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """Pulses for this tick. Anything that goes wrong on the way — a non-finite state, a network output that is
        not a number — yields eight 1500s and counts as a solver failure (the worker disengages after
        engage.max_solver_fails in a row), never an exception that leaves the engagement up with nothing sent."""
        from .rl_pwm import NEUTRAL_PULSES, throttles_to_pwm
        t0 = time.perf_counter()
        try:
            obs = self.observation(eta_ned, nu_ned, t)
            if not np.all(np.isfinite(obs)):
                raise ValueError("non-finite observation")
            a = self.policy.act(obs)
            thr = np.clip(self.model.throttles(a), -self.pwm_cap, self.pwm_cap)
            pwm = throttles_to_pwm(thr, self.pwm_cap)
            u = self.model.wrench_ned_from_throttles(thr)
            if not np.all(np.isfinite(u)):
                raise ValueError("non-finite wrench estimate")
            self.last_action = np.asarray(a, float).copy()
            status = 0
        except Exception:                                        # noqa: BLE001
            self.n_fail += 1
            thr = np.zeros(8)
            pwm = tuple(NEUTRAL_PULSES)
            u = np.zeros(6)
            status = 1
        ms = 1e3 * (time.perf_counter() - t0)
        return u, {"w_hat": self.w_hat, "solve_ms": ms, "status": status, "n_fail": self.n_fail, "nis": 0.0,
                   "pwm_us": pwm, "throttle": [float(v) for v in thr]}

    def note_applied(self, tau_ned_est) -> None:
        """The axes path's feedback does not apply: nothing this controller asked for left as an axis (the worker sends
        a neutral MANUAL_CONTROL). The previous action comes from ``note_applied_pwm``."""
        return None

    def note_applied_pwm(self, pwm_us) -> None:
        """The 8 pulses that actually LEFT become the policy's previous action (None = nothing left: neutral)."""
        from .rl_pwm import NEUTRAL_PULSES, pwm_to_action
        pw = NEUTRAL_PULSES if pwm_us is None else pwm_us
        self.last_pwm_us = tuple(int(p) for p in pw)
        self._prev_action = pwm_to_action(pw, self.model.throttle_max)

    def reset(self) -> None:
        super().reset()
        self.last_pwm_us = None

    def meta(self) -> dict:
        d = super().meta()
        d.update({"type": "rl_pwm", "solver": "rl_pwm", "pwm_cap": float(self.pwm_cap),
                  "pwm_cap_trained": float(self.model.cap_trained), "pwm_throttle_max": float(self.model.throttle_max),
                  "pwm_channels": list(self.pwm_channels), "motor_direction_expected": list(self.model.motor_direction),
                  "transport": self.model.transport,
                  "pilot_gain_assumed": None, "axis_cap": None, "axis_slew_per_s": None,
                  "axes_note": "NO axes: 8 pulses on RC_CHANNELS_OVERRIDE ch 9..16, forced by the vehicle's Lua timeout "
                               "override; MANUAL_CONTROL is sent NEUTRAL. CSV ax_* are 0, pwm1..8 are the vehicle's "
                               "own SERVO_OUTPUT_RAW",
                  "wrench_note": "uX..uN = nominal steady-state wrench of the capped throttles (T200 curve x voltage "
                                 "scale x B) — an estimate for the log, not a command and not a measurement",
                  "status_note": "UNTESTED on the vehicle as of 2026-09-30 [예측]"})
        d.pop("axis_gain_for_pseudo_wrench", None)
        return d

