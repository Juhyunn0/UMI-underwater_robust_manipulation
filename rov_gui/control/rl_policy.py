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
    #: 2026-10-03: "axes6" moved to LOW rl_7d (HwRl7d). Under ``rl`` its pitch/roll outputs could never reach the
    #: wire (the worker never makes the attitude axes live for a 4-DoF LOW), so it was a policy flying with two of its
    #: outputs silently dropped — refused below with the reason.
    ACCEPTED_MODES = ("axes",)
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
        if self.action_mode == "axes6" and "axes6" not in self.ACCEPTED_MODES:
            raise ValueError(f"{self.CFG_KEY}: policy {os.path.relpath(pdir, REPO_ROOT)} is a 6-axis policy (surge, sway, heave, "
                             "yaw, pitch, roll); LOW rl is 4-DoF (its pitch/roll outputs would be dropped by "
                             "allocation.wrench_to_axes) — a 6-axis policy TRAINED WITH A ROLL/PITCH REFERENCE flies on LOW "
                             "rl_7d (name it under `rl_7d:`); a level-trained one (sea6_*) flies nowhere on the station")
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
        q_des = self._q_des(yaw_ref)
        offset_b = np.clip(quat_apply_inverse(q_est, d), -self.clip_m, self.clip_m)
        q_d, q_o = (_unique(q_des), _unique(q_est)) if self.quat_unique else (q_des, q_est)
        parts = [q_d, offset_b, q_o, v_b, gyro]
        if self.goal_velocity:
            goal_v = C_NED_TO_UP @ np.asarray(v_ref_ned, float)
            parts += [quat_apply_inverse(q_est, goal_v), np.array([-float(r_ref_ned)])]
        return np.concatenate(parts + [self._prev_action])

    def _q_des(self, yaw_ref) -> np.ndarray:
        """The goal quaternion: LEVEL with the reference yaw — what every export before att7 was trained on. HwRl7d
        (LOW rl_7d) overrides it with the plan's roll/pitch."""
        return quat_from_euler_xyz(0.0, 0.0, -float(yaw_ref))

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

    ``pwm_cap`` is the |throttle| ceiling on the wire (fraction of +-400 us), the TRANSPORT's single value
    ``rl_pwm.pwm_cap`` (PWM_CAP_KEY; the command sink refuses any pulse outside the same window). The policy was trained
    at 1.0; the code default is the first-water 0.25 (rl_pwm.PWM_CAP_FIRST_WATER) and the shipped config says 1.0 since
    the operator's 2026-10-01 decision (the 0.25 run limit-cycled, data/20261001/1001_172910). The capped pulse is what
    the network is fed back as its previous action, as in training.

    2026-10-10 (REPORT.md §1 cause 4 / §3 step 5, RL_controller/logs/eval/pwm_cocontraction_analysis_20261010):
    the cap is applied as a THRUST-SPACE SCALE (``rl_pwm.pwm_cap_mode: scale``, rl_pwm.scale_throttles — one scalar
    on the thrust vector, allocation direction kept, max|throttle| exactly at the cap), the per-thruster clip of
    2026-09-30..10-09 survives only as ``clip`` for an A/B and as the refuse-if-exceeded assertion on the way out; a
    config cap BELOW the trained cap is REFUSED at build unless ``rl_pwm.allow_caps_below_trained: true`` (the 10-01
    runs flew a cap-1.0 policy through a 0.25 clip); and the per-thruster sign-probe artefact
    (``rl_pwm.require_thruster_probe``, rl_pwm.ThrusterProbeCheck) must pass before the worker engages.

    UNTESTED ON THE VEHICLE (2026-09-30) — see control/rl_pwm.py for the gate that has to pass first.
    """

    CFG_KEY = "rl_pwm"
    #: The block the |throttle| cap is read from — the pwm TRANSPORT's, whatever the follower (2026-10-03): the command
    #: sink builds its one cap from `rl_pwm.pwm_cap` at launch, so a second pwm follower (HwRlPwm7d) reading its own block
    #: would default to 0.25 against the sink's value and fail the worker's follower-vs-sink check on every engage.
    PWM_CAP_KEY = "rl_pwm"
    ACCEPTED_MODES = ("pwm",)
    DEFAULT_DIR = os.path.join("rov_gui", "control", "rl_policies", "pwm10_s1")
    #: The worker reads this to pick the transport: "pwm" = pulses on bus.cmd_pwm, anything else = axes on cmd_pilot.
    transport = "pwm"

    def __init__(self, cfg, log=print):
        from .rl_pwm import (PWM_CHANNELS_DEFAULT, PwmModel, PwmObsGuard, ThrusterProbeCheck, hold_throttles,
                             parse_rl_pwm_block, scale_throttles, throttles_to_pwm)
        super().__init__(cfg, log=lambda m: None)
        raw = dict(getattr(cfg, self.PWM_CAP_KEY, None) or {})
        if self.n_act != 8:
            raise ValueError(f"rl_pwm: policy has {self.n_act} outputs, the transport carries 8 thrusters")
        out = self.spec.get("output") or {}
        if str(out.get("kind")) != "pwm" or not isinstance(out.get("pwm_model"), dict):
            raise ValueError("rl_pwm: obs_spec.json has no output.pwm_model (re-export with RL_controller >= 2026-09-27)")
        self.model = PwmModel(out["pwm_model"])
        blk = parse_rl_pwm_block(raw)            # pwm_cap and the 2026-10-10 keys, validated, with their defaults
        cap = blk["pwm_cap"]
        self.pwm_cap_config = cap
        #: never above what the policy was trained with (the env's own cap), whatever the config says
        self.pwm_cap = min(cap, self.model.cap_trained)
        #: 2026-10-10: how the cap is applied — "scale" (thrust-space scale, rl_pwm.scale_throttles) or "clip" (the
        #: per-thruster clip of 2026-09-30..10-09, kept for an A/B only). RECORD BOUNDARY: meta controller.pwm_cap_mode.
        self.pwm_cap_mode = blk["pwm_cap_mode"]
        #: A config cap BELOW the trained one feeds the policy a previous action it never trained on — the 2026-10-01
        #: runs flew a cap-1.0 policy through a 0.25 clip and limit-cycled (data/20261001/1001_172910, KNOWN_ISSUES
        #: 'LOW RL_PWM'). Refused at build (the wire value keeps the min) unless the LOW's own RECORDED opt-out says
        #: otherwise: rl_pwm.allow_caps_below_trained here, rl_pwm_7d.allow_caps_below_trained for HwRlPwm7d.
        self.caps_below_trained = bool(self.pwm_cap < self.model.cap_trained - 1e-9)
        self.allow_caps_below_trained, opt_key = self._caps_below_trained_opt_out(cfg, blk)
        if self.caps_below_trained and not self.allow_caps_below_trained:
            raise ValueError(
                f"{self.CFG_KEY}: the config cap rl_pwm.pwm_cap {cap:.2f} is below the policy's trained cap "
                f"{self.model.cap_trained:.2f} ({os.path.relpath(self.policy_dir, REPO_ROOT)}) — a policy capped below "
                f"its training is fed back a previous action it never saw and limit-cycles (rl_pwm 2026-10-01: cap 0.25 "
                f"on a cap-1.0 policy); raise rl_pwm.pwm_cap to >= {self.model.cap_trained:.2f}, or set {opt_key}: "
                f"true (recorded in meta)")
        #: The per-thruster sign-probe artefact gate (2026-10-10, rl_pwm.ThrusterProbeCheck): the worker asks it at
        #: engage (probe_refusal) and the command sink's PwmGate every frame. None = the recorded opt-out
        #: rl_pwm.require_thruster_probe: false.
        self.require_thruster_probe = blk["require_thruster_probe"]
        self.thruster_probe = None
        if self.require_thruster_probe:
            self.thruster_probe = ThrusterProbeCheck(self.model.sigma, blk["thruster_probe_path"],
                                                     blk["thruster_probe_max_age_days"],
                                                     data_root=self._probe_data_root(cfg))
        self.last_pwm_scale = 1.0
        #: 2026-10-10 (b), REPORT.md §3 step 5: the policy-side observation guards (``rl_pwm.obs_guard`` — the
        #: TRANSPORT block's, for both pwm followers; rl_pwm.PwmObsGuard: velocity slew, fix-jump reject, tag gate)
        #: and the HOLD pulses the tag gate switches to: Z = hold_wrench_z_n [N, NED] through the export's
        #: minimum-norm vertical allocation + T200 inverse (rl_pwm.hold_throttles), horizontals neutral, then capped
        #: like any frame (scale / clip) so the strict conversion and the sink's window always admit them. Applied
        #: INSIDE step to the state the network sees; the worker's state, the CSV and every other LOW are untouched.
        self.obs_guard = PwmObsGuard(raw.get("obs_guard"), dt=self.dt)
        hold_thr = hold_throttles(self.model, self.obs_guard.cfg["hold_wrench_z_n"])
        if self.pwm_cap_mode == "scale":
            hold_thr = scale_throttles(hold_thr, self.pwm_cap, self.model)
        else:
            hold_thr = np.clip(hold_thr, -self.pwm_cap, self.pwm_cap)
        self.hold_throttles = np.asarray(hold_thr, dtype=float)
        self.hold_pwm_us = throttles_to_pwm(self.hold_throttles, self.pwm_cap, strict=True)
        #: the fix quality of the current tick (note_fix, from the worker): None = never told
        self._fix_n_tags = None
        self._fix_age_s = None
        #: fixed: the vehicle's script maps RC channel 8+m to motor m (control/rl_pwm._fixed_channels)
        self.pwm_channels = tuple(PWM_CHANNELS_DEFAULT)
        self.mode = "rl_pwm"
        self.solver_kind = "rl_pwm"
        self.axis_cap = None                       # no axes leave under this controller
        self.last_pwm_us = None
        og = self.obs_guard.cfg
        log(f"rl_pwm: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), obs "
            f"{self.policy.obs_dim} (K={self.K}), 8 thruster throttles, |throttle| <= {self.pwm_cap:.2f} on the wire "
            f"as a thrust-space {self.pwm_cap_mode} (trained {self.model.cap_trained:.2f}"
            f"{', BELOW trained — recorded opt-out' if self.caps_below_trained else ''}), rc channels "
            f"{list(self.pwm_channels)}, expects MOT_1..8_DIRECTION {self.model.motor_direction}; sign probe "
            f"{'required' if self.require_thruster_probe else 'NOT required (recorded opt-out)'}; obs guard: "
            f"v slew {og['vel_slew_mps2']:g} m/s^2, fix jump {og['jump_m']:g} m under {og['min_tags']} tags, HOLD "
            f"after {og['hold_after_s']:g} s of < {og['min_tags']} tags / age > {og['age_s']:g} s = Z "
            f"{og['hold_wrench_z_n']:+.1f} N NED -> pulses {list(self.hold_pwm_us)} us; transport "
            f"{self.model.transport} — UNTESTED on the vehicle [예측]")

    @staticmethod
    def _probe_data_root(cfg) -> str:
        """Where the newest sign-probe artefact is looked for: <log_dir>/<day>/<run>_probe/ (hw_mpc.yaml log_dir)."""
        d = str(getattr(cfg, "log_dir", None) or "data")
        return d if os.path.isabs(d) else os.path.join(REPO_ROOT, d)

    def _caps_below_trained_opt_out(self, cfg, blk: dict) -> tuple:
        """(allowed, the config key that records it) for THIS LOW: rl_pwm.allow_caps_below_trained."""
        return bool(blk.get("allow_caps_below_trained", False)), "rl_pwm.allow_caps_below_trained"

    def probe_refusal(self, expected_mot_direction=None) -> str:
        """The sign-probe artefact gate at engage: "" when it holds or is not required (recorded opt-out), else the
        reason, prefixed "thruster probe:". The directions are the export's unless the caller passes what the
        vehicle said (the sink's gate has already matched the two when it reports ok)."""
        if self.thruster_probe is None:
            return ""
        mot = self.model.motor_direction if expected_mot_direction is None else list(expected_mot_direction)
        ok, why = self.thruster_probe.check(mot)
        return "" if ok else f"thruster probe: {why}"

    def note_fix(self, n_tags, age_s) -> None:
        """The fix quality of THIS tick, from the worker before ``step`` (2026-10-10 b): tag count and fix age [s] —
        what the observation guards (rl_pwm.obs_guard) judge and the state assembler does not pass on. Kept until
        the next report or ``reset``; never told (None) = the tag gate and the jump reject stay inactive (recorded
        in meta obs_guard.fix_reports)."""
        self._fix_n_tags = None if n_tags is None else int(n_tags)
        self._fix_age_s = age_s

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """Pulses for this tick. Anything that goes wrong on the way — a non-finite state, a network output that is
        not a number — yields eight 1500s and counts as a solver failure (the worker disengages after
        engage.max_solver_fails in a row), never an exception that leaves the engagement up with nothing sent.

        2026-10-10 (b): the observation guards run FIRST on the (eta, nu) handed in (rl_pwm.PwmObsGuard: velocity
        slew, fix-jump reject under a thin fix, tag gate), and under the tag gate's HOLD the eight pulses are the
        hold pulses, not the network's (the network is still stepped so its history stays continuous and
        ``last_action`` says what it asked for; what it is fed back as its previous action is what LEFT, through
        ``note_applied_pwm``). ``info["pwm_guard"]`` = the CSV code (0 none, 1 slew, 2 jump, 3 hold)."""
        from .rl_pwm import CAP_TOL, GUARD_HOLD, GUARD_NONE, NEUTRAL_PULSES, scale_throttles, throttles_to_pwm
        t0 = time.perf_counter()
        scale = 1.0
        guard = GUARD_NONE
        try:
            eta_g, nu_g, guard = self.obs_guard.apply(eta_ned, nu_ned, self._fix_n_tags, self._fix_age_s)
            obs = self.observation(eta_g, nu_g, t)
            if not np.all(np.isfinite(obs)):
                raise ValueError("non-finite observation")
            a = self.policy.act(obs)
            thr = self.model.throttles(a)                        # action -> throttle, clipped to the TRAINED cap
            if self.pwm_cap_mode == "scale":
                # 2026-10-10: ONE scalar on the thrust vector (direction kept), never a per-thruster clip. What is
                # beyond the cap after this is a bug upstream, and is REFUSED (neutral + n_fail), not clipped.
                thr, scale = scale_throttles(thr, self.pwm_cap, self.model, return_scale=True)
                if float(np.max(np.abs(thr))) > self.pwm_cap + CAP_TOL:
                    raise ValueError("scaled throttles exceed the cap")
            else:                                                # "clip": the 2026-09-30..10-09 behaviour (A/B only)
                thr = np.clip(thr, -self.pwm_cap, self.pwm_cap)
            if guard == GUARD_HOLD:
                # the tag gate: the HOLD pulses leave instead of the policy's (already capped at build)
                thr = self.hold_throttles.copy()
                scale = 1.0
            pwm = throttles_to_pwm(thr, self.pwm_cap, strict=True)
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
            scale = float("nan")
        self.last_pwm_scale = scale
        ms = 1e3 * (time.perf_counter() - t0)
        return u, {"w_hat": self.w_hat, "solve_ms": ms, "status": status, "n_fail": self.n_fail, "nis": 0.0,
                   "pwm_us": pwm, "throttle": [float(v) for v in thr], "pwm_scale": scale,
                   "pwm_cap_mode": self.pwm_cap_mode, "pwm_guard": int(guard),
                   "guard_why": str(self.obs_guard.last.get("why", ""))}

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
        # the worker zeroes its _nfail_prev at engage; a count left from the previous engagement read as a fresh
        # "solver FAILURE" (and a fail-streak of 1) on the first tick (2026-10-03 survey, also fixed in HwRl7d)
        self.n_fail = 0
        # 2026-10-10 (b): slew memory, jump memory, hold state (HwPid.__init__ calls reset before the guard exists)
        og = getattr(self, "obs_guard", None)
        if og is not None:
            og.reset()
        self._fix_n_tags = None
        self._fix_age_s = None

    def meta(self) -> dict:
        d = super().meta()
        d.update({"type": "rl_pwm", "solver": "rl_pwm", "pwm_cap": float(self.pwm_cap),
                  "pwm_cap_trained": float(self.model.cap_trained), "pwm_throttle_max": float(self.model.throttle_max),
                  # 2026-10-10 (RECORD BOUNDARY: never pool across pwm_cap_mode or caps_below_trained)
                  "pwm_cap_config": float(self.pwm_cap_config), "pwm_cap_mode": self.pwm_cap_mode,
                  "caps_below_trained": bool(self.caps_below_trained),
                  "allow_caps_below_trained": bool(self.allow_caps_below_trained),
                  "require_thruster_probe": bool(self.require_thruster_probe),
                  "thruster_probe": ({"required": False} if self.thruster_probe is None
                                     else self.thruster_probe.state()),
                  "sigma": (None if self.model.sigma is None else list(self.model.sigma)),
                  # 2026-10-10 (b): the observation guards and the hold they switch to (RECORD BOUNDARY: never
                  # pool across obs_guard.config; the CSV pwm_guard column says which guard acted per row)
                  "obs_guard": self.obs_guard.meta(),
                  "hold_pwm_us": [int(v) for v in self.hold_pwm_us],
                  "hold_throttles": [float(v) for v in self.hold_throttles],
                  "hold_wrench_ned_est": [float(v) for v in self.model.wrench_ned_from_throttles(self.hold_throttles)],
                  "cap_note": ("scale: one scalar on the thrust vector (rl_pwm.scale_throttles), max|throttle| at "
                               "the cap, direction kept; clip: per-thruster clip (A/B only). A pulse beyond the cap "
                               "is refused (neutral + n_fail), never clipped — REPORT 2026-10-10 §1 cause 4"),
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



# ----------------------------------------------------------------------------- roll/pitch reference (LOW mode ``rl_7d``)
from .attitude_ref import AttitudeRefMixin  # noqa: E402

#: The frame an rl_7d export must declare, element for element (RL_controller rov_rl/obs_spec.frame_layout(True, 6)).
RL7D_FRAME_LAYOUT = ["q_des(w,x,y,z)", "offset_b(3)", "q_est(w,x,y,z)", "v_b_est(3)", "gyro_b(3)", "goal_v_b(3)",
                     "goal_yaw_rate(1)", "prev_action(6)"]


class HwRl7d(AttitudeRefMixin, HwRl):
    """LOW ``rl_7d`` (2026-10-03): a 6-axis RL policy (``action_mode: axes6`` — surge, sway, heave, yaw, pitch, roll on
    MANUAL_CONTROL x/y/z/r/s/t) TRAINED WITH A ROLL/PITCH REFERENCE in its goal quaternion (RL_controller ``att7``
    recipe: dr preset traj10att, scripts/_pipeline_att7_seed.sh). The station builds

        q_des = quat_from_euler_xyz(roll_ref, -pitch_ref, -yaw_ref)       (NED -> the sim's z-up/FLU, as q_est)

    from the plan's stage-0 ``rp_ned`` (policy 7-dim / replay track_attitude), else the D5 hold ramp, else LEVEL, clipped
    to the envelope the export says it was trained on. Fail-closed: an export without ``attitude_reference.trained: true``
    is REFUSED — its layout is identical to a level-trained one (q_des is 4 floats either way), so that key is the only
    thing that tells them apart, and a level-trained policy fed a tilted q_des is out of distribution. As of 2026-10-03
    no such export exists (training deferred by the operator), so the shipped config names none and the mode refuses.

    Everything else is HwRl: the reference API, the leash, the obs layout, K/M = 0.4 * a * roll_nm|pitch_nm (one policy
    unit = +-400 us on s/t = 0.4 station units), the applied-axes feedback. The worker additionally slews the roll/pitch
    axes at engage.attitude_axes.slew_per_s (the translation axes stay unslewed, as trained).
    """

    CFG_KEY = "rl_7d"
    ACCEPTED_MODES = ("axes6",)
    DEFAULT_DIR = None

    def __init__(self, cfg, log=print):
        self._att_init(cfg)
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        unknown = set(raw) - {"policy_dir", "axis_cap", "leash_m", "yaw_rate_ff", "yaw_rate_max",
                              "allow_caps_below_trained"}
        if unknown:
            raise ValueError(f"rl_7d: unknown key(s) {sorted(unknown)}")
        pdir = raw.get("policy_dir")
        if not pdir:
            raise ValueError("rl_7d: no policy_dir — no export trained with a roll/pitch reference exists yet "
                             "(RL_controller scripts/_pipeline_att7_seed.sh, dr preset traj10att; not trained as of "
                             "2026-10-03). Name one under `rl_7d: policy_dir:`")
        pabs = pdir if os.path.isabs(str(pdir)) else os.path.join(REPO_ROOT, str(pdir))
        with open(os.path.join(pabs, "obs_spec.json"), "r", encoding="utf-8") as f:
            spec = json.load(f)
        mode = str(spec.get("action_mode", "axes"))
        if mode != "axes6":
            raise ValueError(f"rl_7d: policy {pdir} is an '{mode}' policy — rl_7d flies 6-axis (axes6) exports only "
                             f"(roll/pitch go out as MANUAL_CONTROL s/t)")
        ar = spec.get("attitude_reference") or {}
        if not bool(ar.get("trained", False)):
            raise ValueError(f"rl_7d: policy {pdir} was trained with a LEVEL goal (obs_spec attitude_reference.trained is "
                             f"not true) — a tilted q_des would be out of its distribution; train the att7 recipe")
        if list(spec.get("frame_layout") or []) != RL7D_FRAME_LAYOUT:
            raise ValueError(f"rl_7d: policy {pdir} frame_layout {spec.get('frame_layout')} != {RL7D_FRAME_LAYOUT}")
        super().__init__(cfg, log=lambda m: None)
        self.mode = "rl_7d"
        self.solver_kind = "rl_7d"
        try:
            self.att_roll_max = float(ar["roll_max_rad"])
            self.att_pitch_max = float(ar["pitch_max_rad"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("rl_7d: attitude_reference.trained is true but roll_max_rad / pitch_max_rad are missing — "
                             "the trained envelope is unknown (a 0 default would silently level every reference)")
        if not (math.isfinite(self.att_roll_max) and math.isfinite(self.att_pitch_max)
                and self.att_roll_max > 0.0 and self.att_pitch_max > 0.0):
            raise ValueError("rl_7d: attitude_reference roll_max_rad / pitch_max_rad must be finite and > 0")
        self.att_rate_max = self._trained_rate_max(ar, "rl_7d")
        self.attitude_reference = dict(ar)
        cap_tr = spec.get("attitude_cap_trained") or [0.5, 0.75]
        #: the trained roll/pitch action caps in STATION axis units (0.4 per policy unit), what the worker compares
        #: against the caps in force at engage (a capped policy is fed back an out-of-distribution previous action —
        #: the 2026-10-01 rl_pwm cap-0.25 run limit-cycled on exactly that)
        self.attitude_cap_trained_station = (STATION_ATT_AXIS_PER_ACTION * float(cap_tr[0]),
                                             STATION_ATT_AXIS_PER_ACTION * float(cap_tr[1]))
        self.allow_caps_below_trained = bool(raw.get("allow_caps_below_trained", False))
        self._rp_clip_ticks = 0
        self._rp_rate_ticks = 0
        self._rp_flown = (0.0, 0.0)
        log(f"rl_7d: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), axes6, "
            f"trained roll/pitch envelope +-{math.degrees(self.att_roll_max):.1f} / +-{math.degrees(self.att_pitch_max):.1f} "
            f"deg, trained caps (station units) {self.attitude_cap_trained_station}")

    @staticmethod
    def _trained_rate_max(ar: dict, who: str) -> float:
        """attitude_reference.rate_max_rad_s — the fastest roll/pitch goal the policy saw in training (goals.py
        att_rate_max); required, a missing rate would let a plan step reach q_des in one tick (review 2026-10-03)."""
        try:
            v = float(ar["rate_max_rad_s"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{who}: attitude_reference.rate_max_rad_s is missing — the trained goal rate is unknown, "
                             f"a plan step would reach q_des in one tick (re-export with scripts/export_policy.py)")
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{who}: attitude_reference.rate_max_rad_s must be finite and > 0")
        return v

    def _advance_rp_flown(self) -> None:
        """Once per tick, BEFORE the policy reads q_des: the plan / hold / level reference clipped to the trained
        envelope, then moved toward by at most rate_max_rad_s * dt per axis — training never showed the policy a goal
        faster than that (goals.py att_rate_max; review 2026-10-03: a 20 deg plan step otherwise lands in one tick).
        A fresh engagement starts from level, as the training goal does (attitude_reference.starts_level)."""
        raw = AttitudeRefMixin.ref_attitude_ned_at(self)
        r = float(np.clip(raw[0], -self.att_roll_max, self.att_roll_max))
        p = float(np.clip(raw[1], -self.att_pitch_max, self.att_pitch_max))
        if (r, p) != (float(raw[0]), float(raw[1])):
            self._rp_clip_ticks += 1
        pr, pp = self._rp_flown
        st = self.att_rate_max * self.dt
        r2 = pr + float(np.clip(r - pr, -st, st))
        p2 = pp + float(np.clip(p - pp, -st, st))
        if abs(r2 - r) > 1e-12 or abs(p2 - p) > 1e-12:
            self._rp_rate_ticks += 1
        self._rp_flown = (r2, p2)

    def ref_attitude_ned_at(self, t: float | None = None) -> tuple[float, float]:
        """The reference this follower FLIES this tick: the plan / hold / level reference clipped to the trained
        envelope and rate-limited to the trained goal rate (_advance_rp_flown) — what the CSV rroll/rpitch and interlock
        (iii) judge (safety audit 2026-10-03: judging the unclipped plan would blame the follower for a tilt it was never
        given). NOTE div_rp is NOT judged on it: the worker compares the stitcher's reference (T1-clipped at
        policy.rp_max_deg), the conservative choice — a plan beyond the envelope eats into the 15 deg divergence budget
        even when this follower flies its clipped reference perfectly."""
        return float(self._rp_flown[0]), float(self._rp_flown[1])

    def _q_des(self, yaw_ref) -> np.ndarray:
        r, p = self._rp_flown
        return quat_from_euler_xyz(r, -p, -float(yaw_ref))

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        """HwRl.step, guarded like HwRlPwm.step: a non-finite observation or action is a counted solver failure with a
        neutral command (the worker disengages after engage.max_solver_fails), never an exception that leaves the
        engagement up with nothing sent."""
        t0 = time.perf_counter()
        self._att_decay()                                        # D5 first: the reference read after the tick = the one used
        self._advance_rp_flown()
        try:
            u, info = super().step(eta_ned, nu_ned, nudot_ned, t)
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(self.last_action))):
                raise ValueError("non-finite action")
            # never more roll/pitch torque than the policy produced in training: the env clamps a4/a5 to
            # attitude_cap_trained INSIDE training (rov_env _a_cap), so an output beyond it was never flown there
            # (safety audit 2026-10-03; HwRlPwm's min(cap, cap_trained) twin). The caps in force cut further.
            g = self._gains
            u = np.asarray(u, float).copy()
            k_max = self.attitude_cap_trained_station[0] * float(g.get("roll_nm", 13.2))
            m_max = self.attitude_cap_trained_station[1] * float(g.get("pitch_nm", 7.2))
            u[3] = float(np.clip(u[3], -k_max, k_max))
            u[4] = float(np.clip(u[4], -m_max, m_max))
            info = dict(info)
            info["n_fail"] = self.n_fail
        except Exception:                                        # noqa: BLE001
            self.n_fail += 1
            u = np.zeros(6)
            info = {"w_hat": self.w_hat, "solve_ms": 1e3 * (time.perf_counter() - t0), "status": 1,
                    "n_fail": self.n_fail, "nis": 0.0}
        return u, info

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_rp_hold"):
            self._att_reset()
        self._rp_clip_ticks = 0
        self._rp_rate_ticks = 0
        self._rp_flown = (0.0, 0.0)
        self.n_fail = 0           # the worker zeroes its _nfail_prev at engage; a stale count read as a fresh failure

    def meta(self) -> dict:
        d = super().meta()
        d.update({"type": "rl_7d", "solver": "rl_7d",
                  "wire_sign_note": "a4/a5 are PHYSICAL (+a4 = +M nose-up, +a5 = +K, the training mixer's convention, "
                                    "obs_spec action_convention); the sink's engage.attitude_axes.sign maps physical "
                                    "to wire exactly as for the model-based followers",
                  "attitude_reference_trained": self.attitude_reference,
                  "rp_envelope_clip_ticks": int(self._rp_clip_ticks),
                  "rp_rate_limit_ticks": int(self._rp_rate_ticks),
                  "rp_rate_max_rad_s": float(self.att_rate_max),
                  "attitude_cap_trained_station": list(self.attitude_cap_trained_station),
                  "allow_caps_below_trained": self.allow_caps_below_trained,
                  "axes_note": "u = axes * axis_gain (pseudo-wrench) incl. K = 0.4*a5*roll_nm, M = 0.4*a4*pitch_nm; "
                               "q_des carries the plan's roll/pitch clipped to the trained envelope; roll/pitch slewed "
                               "at engage.attitude_axes.slew_per_s"})
        d.update(self._att_meta())
        return d


# ----------------------------------------------------------------------------- roll/pitch reference on PULSES (LOW ``rl_pwm_7d``)
#: The frame an rl_pwm_7d export must declare (RL_controller rov_rl/obs_spec.frame_layout(True, 8)).
RLPWM7D_FRAME_LAYOUT = RL7D_FRAME_LAYOUT[:-1] + ["prev_action(8)"]


class HwRlPwm7d(AttitudeRefMixin, HwRlPwm):
    """LOW ``rl_pwm_7d`` (2026-10-03): a per-thruster RL policy (``action_mode: pwm``) TRAINED WITH A ROLL/PITCH REFERENCE
    in q_des (RL_controller ``att7pwm``: dr preset traj10att, still water, scripts/_pipeline_att7pwm_seed.sh). Everything
    about the pulses is HwRlPwm's — the transport (``transport = "pwm"``: RC_CHANNELS_OVERRIDE ch 9..16 + the vehicle's Lua
    override, MANUAL_CONTROL neutral), the vehicle-read gate, the guarded step, the applied-pulse feedback, and the ONE
    |throttle| cap of the transport (``rl_pwm.pwm_cap``, PWM_CAP_KEY). What differs is HwRl7d's: q_des carries the plan's
    roll/pitch (``quat_from_euler_xyz(roll, -pitch, -yaw)``), clipped to the envelope the export says it was trained on,
    fail-closed on ``obs_spec.attitude_reference.trained`` (a level-trained pwm export — pwm10_s1 — has the identical
    layout). Roll/pitch leave as PULSES, never as s/t: the worker keeps the s/t gates (engage.attitude_axes, firmware,
    probe, sign) away from this LOW and arms the attitude SAFETY interlocks for it (tilt ceiling, div_rp) through its
    pulse bit (geometry.LOW_7D_PULSE_BASE)."""

    CFG_KEY = "rl_pwm_7d"
    DEFAULT_DIR = None
    #: The envelope-clipped, rate-limited reference and the q_des built from it — HwRl7d's, verbatim (same attributes:
    #: att_roll_max, att_pitch_max, att_rate_max, _rp_flown, _rp_clip_ticks, _rp_rate_ticks).
    ref_attitude_ned_at = HwRl7d.ref_attitude_ned_at
    _q_des = HwRl7d._q_des
    _advance_rp_flown = HwRl7d._advance_rp_flown

    def _caps_below_trained_opt_out(self, cfg, blk: dict) -> tuple:
        """This LOW's recorded opt-out is its OWN block's (rl_pwm_7d.allow_caps_below_trained, 2026-10-03), not the
        transport's: the decision to fly a capped policy is per LOW. Read in HwRlPwm.__init__, before this class's
        own attributes exist."""
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        return bool(raw.get("allow_caps_below_trained", False)), "rl_pwm_7d.allow_caps_below_trained"

    def __init__(self, cfg, log=print):
        from .geometry import validate_rl_pwm_7d_block
        self._att_init(cfg)
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        validate_rl_pwm_7d_block(raw)
        pdir = raw.get("policy_dir")
        if not pdir:
            raise ValueError("rl_pwm_7d: no policy_dir — name a pwm export trained with a roll/pitch reference "
                             "(RL_controller att7pwm_s*, scripts/_pipeline_att7pwm_seed.sh) under `rl_pwm_7d: policy_dir:`")
        pabs = pdir if os.path.isabs(str(pdir)) else os.path.join(REPO_ROOT, str(pdir))
        with open(os.path.join(pabs, "obs_spec.json"), "r", encoding="utf-8") as f:
            spec = json.load(f)
        mode = str(spec.get("action_mode", "axes"))
        if mode != "pwm":
            raise ValueError(f"rl_pwm_7d: policy {pdir} is an '{mode}' policy — rl_pwm_7d flies per-thruster 'pwm' "
                             f"exports only (a 6-axis roll/pitch policy flies on rl_7d)")
        ar = spec.get("attitude_reference") or {}
        if not bool(ar.get("trained", False)):
            raise ValueError(f"rl_pwm_7d: policy {pdir} was trained with a LEVEL goal (obs_spec attitude_reference.trained "
                             f"is not true) — a tilted q_des would be out of its distribution; fly it on rl_pwm")
        if list(spec.get("frame_layout") or []) != RLPWM7D_FRAME_LAYOUT:
            raise ValueError(f"rl_pwm_7d: policy {pdir} frame_layout {spec.get('frame_layout')} != {RLPWM7D_FRAME_LAYOUT}")
        super().__init__(cfg, log=lambda m: None)
        self.mode = "rl_pwm_7d"
        self.solver_kind = "rl_pwm_7d"
        try:
            self.att_roll_max = float(ar["roll_max_rad"])
            self.att_pitch_max = float(ar["pitch_max_rad"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("rl_pwm_7d: attitude_reference.trained is true but roll_max_rad / pitch_max_rad are missing — "
                             "the trained envelope is unknown")
        if not (math.isfinite(self.att_roll_max) and math.isfinite(self.att_pitch_max)
                and self.att_roll_max > 0.0 and self.att_pitch_max > 0.0):
            raise ValueError("rl_pwm_7d: attitude_reference roll_max_rad / pitch_max_rad must be finite and > 0")
        self.att_rate_max = HwRl7d._trained_rate_max(ar, "rl_pwm_7d")
        self.attitude_reference = dict(ar)
        self.allow_caps_below_trained = bool(raw.get("allow_caps_below_trained", False))
        # A policy whose throttles are CLIPPED below what it was trained with is fed back an out-of-distribution previous
        # action — the 2026-10-01 rl_pwm cap-0.25 run limit-cycled on exactly that (data/20261001/1001_172910). The cap is
        # the transport's (rl_pwm.pwm_cap); refused here unless the recorded opt-out says otherwise.
        self.caps_below_trained = bool(self.pwm_cap < self.model.cap_trained - 1e-9)
        if self.caps_below_trained and not self.allow_caps_below_trained:
            raise ValueError(f"rl_pwm_7d: the transport cap rl_pwm.pwm_cap {self.pwm_cap:.2f} is below the policy's trained "
                             f"cap {self.model.cap_trained:.2f} — a clipped policy limit-cycles (rl_pwm 2026-10-01); raise "
                             f"rl_pwm.pwm_cap or set rl_pwm_7d.allow_caps_below_trained: true (recorded)")
        self._rp_clip_ticks = 0
        self._rp_rate_ticks = 0
        self._rp_flown = (0.0, 0.0)
        log(f"rl_pwm_7d: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), pwm, "
            f"trained roll/pitch envelope +-{math.degrees(self.att_roll_max):.1f} / +-{math.degrees(self.att_pitch_max):.1f} "
            f"deg, |throttle| <= {self.pwm_cap:.2f} (transport rl_pwm.pwm_cap; trained {self.model.cap_trained:.2f}) — "
            f"roll/pitch as PULSES, never s/t; UNTESTED on the vehicle [예측]")

    def step(self, eta_ned, nu_ned, nudot_ned, t: float):
        self._att_decay()                                        # D5 first: the reference read after the tick = the one used
        self._advance_rp_flown()                                 # envelope clip + trained goal rate
        return super().step(eta_ned, nu_ned, nudot_ned, t)       # HwRlPwm: neutral pulses + n_fail on any failure

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_rp_hold"):
            self._att_reset()
        self._rp_clip_ticks = 0
        self._rp_rate_ticks = 0
        self._rp_flown = (0.0, 0.0)

    def meta(self) -> dict:
        d = super().meta()
        d.update({"type": "rl_pwm_7d", "solver": "rl_pwm_7d",
                  "attitude_reference_trained": self.attitude_reference,
                  "rp_envelope_clip_ticks": int(self._rp_clip_ticks),
                  "rp_rate_limit_ticks": int(self._rp_rate_ticks),
                  "rp_rate_max_rad_s": float(self.att_rate_max),
                  "allow_caps_below_trained": self.allow_caps_below_trained,
                  "caps_below_trained": self.caps_below_trained,
                  "pwm_cap_source": "rl_pwm.pwm_cap (the transport's single cap)",
                  "sea_state_training": self.spec.get("sea_state_training"),
                  "training": self.spec.get("training"),
                  "axes_note": "NO axes, NO s/t: 8 pulses on RC_CHANNELS_OVERRIDE ch 9..16 (Lua override); roll/pitch are "
                               "commanded THROUGH the pulses, the reference rides q_des (plan roll/pitch clipped to the "
                               "trained envelope)"})
        d.update(self._att_meta())
        d["allocation"] = {"attitude": bool(self._attitude_axes), "via": "per_thruster_pulses"}
        return d
