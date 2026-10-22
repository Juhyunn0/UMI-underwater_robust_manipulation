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

Observation frame (newest-first history of ``history_len`` frames; 17 + 4 feed-forward + n_actions floats = 25 for an
axes export, 29 for a pwm one, + 5 with the integral-error slots = 34 for a pwm "PI" export):
    [q_des(w,x,y,z), offset_b(3), q_est(w,x,y,z), v_b(3), gyro_b(3), goal_v_b(3), goal_yaw_rate(1),
     (int_pos_b(3), int_att_rp(2),) prev_action(n)]
    q_* canonicalised to w >= 0 (obs_spec.obs_quat_unique); offset clipped to +-obs_offset_clip [m].
    Goal frame (2026-10-15, obs_spec.obs_goal_frame: true — RL_controller pwm11att7 and later): q_des and q_est are first
    rotated by -goal yaw (the ZYX yaw of q_des), THEN canonicalised, so q_des carries no yaw and q_est only the heading
    error (goal_frame_quats); offset_b and goal_v_b stay in the body frame of the WORLD q_est. Such an export names the
    two slots q_des_goalframe / q_est_goalframe, and every RL LOW accepts that layout ONLY with obs_goal_frame true and
    the plain one ONLY without it (check_frame_layout) — anything else is refused at build.
    Integral-error slots (2026-10-18, obs_spec.obs_integral.enabled: true — RL_controller pwm12 contract C6, the "PI"
    variant; flown ONLY on rl_pwm / rl_pwm_7d): a leaky, clamped integral of the error the frame already shows
    (StationErrorIntegrator = RL_controller deploy/obs_builder.ErrorIntegratorNp transcribed operation for operation,
    plus the round-2 options of its dev copy logs/r2dev_20261018/src). Per tick, after the observation guard and the
    goal update and before the frame is assembled: I_pos = I_att = 0 when the history is refilled (engage) BEFORE the
    update, so the first frame already carries one update; e_pos_w = clip(d, +-obs_offset_clip) per world z-up axis with
    d = the LEASHED goal offset offset_b uses (a station convention: the training env has no leash — identical while
    |goal - p| <= leash_m); e_att = 2 sign(w) (x, y) of conj(q_est_f) (x) q_des_f ("quat_xy", round 1) or the wrapped
    ZYX roll / pitch differences of the two frame quaternions ("euler_rp", round 2); I = clip(I (1 - dt/tau) + dt e,
    +-max) per component, the add masked to |e| < cond_pos_m / cond_att_rad when the export sets them; slots =
    [R(q_est_WORLD)^T I_pos / pos_max, I_att / att_max]. int_pos_b is NOT bounded to [-1, 1] (a per-world-axis clamp,
    rotated: up to sqrt(2) level, sqrt(3) tilted [유도]) and is never clipped. The integral is updated on every tick the
    observation is built, the tag gate's HOLD included, as the network history is — but (fix round 2026-10-18, review
    F1) the update that follows a command which was NOT the policy's own masks its add and keeps the leak (the law's own
    masked-add step, as the round-2 cond mask does): after the tag gate's HOLD pulses, a failed tick's neutral, or
    nothing sent at all (note_applied_pwm(None): the worker's station-bridge coast, the pwm-gate grace window, a
    disengage). Training integrates through its own 1-2 tick actuation dropouts only and has no HOLD / coast / grace;
    without the mask an override of seconds winds the integral up with an error the policy could not act on. This is a
    STATION convention [예측], recorded in meta controller.obs_integral.override_rule. Per tick the slots the network saw
    and the mask leave in step's info (obs_int_slots, obs_int_add_masked) — NOT in the CSV (workers.py has no column
    for them); meta controller.obs_integral keeps the law, the conventions and per-engagement counts that survive a
    re-engage. Any key, law or law TEXT the station does not implement is refused at build (check_obs_integral,
    check_frame_layout).
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


def yaw_from_q(q: np.ndarray) -> float:
    """ZYX yaw of a (w,x,y,z) quaternion — RL_controller deploy/obs_builder.yaw_from_q (= rov_rl.utils.math.yaw_from_quat)."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, (w,x,y,z) — RL_controller deploy/obs_builder.quat_mul_np, verbatim."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def goal_frame_quats(q_des: np.ndarray, q_est: np.ndarray) -> tuple:
    """obs_goal_frame (2026-10-15): (q_des, q_est) rotated by -goal yaw, the goal yaw being the ZYX yaw of q_des —
    RL_controller rov_env._get_observations / deploy/obs_builder.py, transcribed operation for operation. The caller
    canonicalises AFTER this (as training does): the canonical yaw quaternion flips sign at +-180 deg of heading, and the
    att6 actors answered that flip with a yaw-command jump [측정 RL_controller logs/eval/att6_verify_20261014/robustness];
    in the goal frame q_des has no yaw and q_est carries only the heading ERROR, so the flip sits at an error of 180 deg."""
    yg = yaw_from_q(q_des)
    q_rot = np.array([math.cos(-0.5 * yg), 0.0, 0.0, math.sin(-0.5 * yg)])
    return quat_mul(q_rot, q_des), quat_mul(q_rot, q_est)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """(w, -x, -y, -z) — RL_controller deploy/obs_builder.quat_conjugate_np, verbatim."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def roll_pitch_zyx(q: np.ndarray) -> tuple:
    """ZYX Euler roll, pitch [rad] of one (w,x,y,z) quaternion, FLU (+ roll = right-side-down, + pitch = nose DOWN) —
    RL_controller logs/r2dev_20261018/src/deploy/obs_builder._roll_pitch_zyx_np (round-2 dev copy), verbatim."""
    w, x, y, z = (float(v) for v in q)
    return (np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


#: The two spellings of an export's quaternion slots (RL_controller rov_rl/obs_spec.frame_layout): world frame (every
#: export before 2026-10-15) and goal-heading frame (obs_goal_frame: true). The rename is the export's own "do not fly me
#: without the transform": a station that compares frame_layout slot for slot refuses it until it implements it.
Q_SLOTS_WORLD = ("q_des(w,x,y,z)", "q_est(w,x,y,z)")
Q_SLOTS_GOAL = ("q_des_goalframe(w,x,y,z)", "q_est_goalframe(w,x,y,z)")
#: 2026-10-18 (RL_controller pwm12 contract C6, rov_rl/obs_spec.INTEGRAL_SLOTS): the integral-error slots an export with
#: obs_integral.enabled: true carries between the goal feed-forward and prev_action (5 floats; check_obs_integral).
INTEGRAL_SLOTS = ("int_pos_b(3)", "int_att_rp(2)")
INTEGRAL_DIM = 5


def frame_layout(goal_velocity: bool, n_actions: int, goal_frame: bool = False, integral: bool = False) -> list:
    """The frame_layout an export with these keys declares — RL_controller rov_rl/obs_spec.frame_layout, transcribed
    (2026-10-15); the goal-frame layout renames ONLY the two quaternion slots. ``integral`` (2026-10-18, obs_integral
    enabled): INTEGRAL_SLOTS after the goal feed-forward, before prev_action."""
    q_des, q_est = Q_SLOTS_GOAL if goal_frame else Q_SLOTS_WORLD
    core = [q_des, "offset_b(3)", q_est, "v_b_est(3)", "gyro_b(3)"]
    if goal_velocity:
        core += ["goal_v_b(3)", "goal_yaw_rate(1)"]
    if integral:
        core += list(INTEGRAL_SLOTS)
    return core + [f"prev_action({int(n_actions)})"]


def check_frame_layout(spec: dict, who: str, where: str, goal_velocity: bool | None = None,
                       n_actions: int | None = None) -> bool:
    """The layout gate every RL LOW passes at build (2026-10-15) -> the export's obs_goal_frame.

    The goal-frame layout (q_des_goalframe / q_est_goalframe) is accepted ONLY with ``obs_goal_frame: true``, the
    world-frame layout ONLY with it false or absent. A non-boolean obs_goal_frame, a layout that is neither, or a
    frame_dim that is not the layout's width is REFUSED (ValueError naming the export and what is wrong).
    ``goal_velocity`` / ``n_actions`` pin what the LOW flies (rl_7d / rl_pwm_7d: the feed-forward and 6 / 8 outputs);
    None = the export's own obs_goal_velocity / num_actions (rl, rl_pwm — whose action mode is checked before this).

    2026-10-18 (obs_integral, RL_controller pwm12 C6): the layout with the integral-error slots (INTEGRAL_SLOTS before
    prev_action, frame_dim + 5) is accepted ONLY with ``obs_integral.enabled: true``, the plain one ONLY without it
    (no block, or enabled false); a block that is not a dict or a non-boolean ``enabled`` is refused
    (obs_integral_enabled). The integral LAW is checked by check_obs_integral (HwRl.__init__). The return value is
    unchanged: the export's obs_goal_frame."""
    gf = spec.get("obs_goal_frame", False)
    if not isinstance(gf, bool):
        raise ValueError(f"{who}: policy {where} has obs_goal_frame {gf!r} — expected true or false (absent = false); "
                         f"refused, the frame its quaternion slots are in is unknown")
    integ = obs_integral_enabled(spec, who, where)
    gv = bool(spec.get("obs_goal_velocity", False)) if goal_velocity is None else bool(goal_velocity)
    na = int(spec.get("num_actions", -1)) if n_actions is None else int(n_actions)
    want = frame_layout(gv, na, goal_frame=gf, integral=integ)
    got = spec.get("frame_layout")
    if got != want:
        if got == frame_layout(gv, na, goal_frame=not gf, integral=integ):
            if gf:
                raise ValueError(f"{who}: policy {where} says obs_goal_frame: true but its frame_layout names the "
                                 f"WORLD-frame quaternion slots {list(Q_SLOTS_WORLD)} — an inconsistent export; refused "
                                 f"(re-export with RL_controller scripts/export_policy.py)")
            raise ValueError(f"{who}: policy {where} frame_layout names the GOAL-FRAME quaternion slots "
                             f"{list(Q_SLOTS_GOAL)} but obs_goal_frame is "
                             f"{'false' if 'obs_goal_frame' in spec else 'absent'} — refused: q_des / q_est are rotated "
                             f"into the goal frame only for an export that says obs_goal_frame: true")
        if got == frame_layout(gv, na, goal_frame=gf, integral=not integ):
            if integ:
                raise ValueError(f"{who}: policy {where} says obs_integral.enabled: true but its frame_layout has no "
                                 f"integral-error slots {list(INTEGRAL_SLOTS)} — an inconsistent export; refused "
                                 f"(re-export with RL_controller scripts/export_policy.py)")
            raise ValueError(f"{who}: policy {where} frame_layout carries the integral-error slots "
                             f"{list(INTEGRAL_SLOTS)} but obs_integral.enabled is "
                             f"{'false' if 'obs_integral' in spec else 'absent'} — refused: the integral is built only "
                             f"for an export that says obs_integral.enabled: true")
        raise ValueError(f"{who}: policy {where} frame_layout {got} != {want} (the layout of obs_goal_frame "
                         f"{str(gf).lower()}, obs_goal_velocity {str(gv).lower()}, {na} actions"
                         f"{', obs_integral enabled' if integ else ''}); refused")
    width = 17 + (4 if gv else 0) + (INTEGRAL_DIM if integ else 0) + na
    if spec.get("frame_dim") != width:
        raise ValueError(f"{who}: policy {where} frame_dim {spec.get('frame_dim')!r} != {width}, the width of its "
                         f"frame_layout; refused")
    return gf


# ----------------------------------------------------------------------------- what velocity the export was trained on
#: 2026-10-16 (dvl.selection; RL_controller logs/eval/dvl_trip_fix_20261015 SPEC.md §8): an export's
#: obs_spec.velocity_source constants (RL_controller scripts/export_policy.py, pwm11att13 and later) and the station's
#: dvl: key that implements each. The layout gate cannot tell such an export apart — its slot is v_b_est(3) like every
#: other's — so these keys are the only thing that can.
VELOCITY_SOURCE_KEYS = (("hold_s", "blend_hold_s"), ("xfade_s", "blend_xfade_s"), ("gross_m_s", "blend_gross_m_s"),
                        ("gross_tau_s", "blend_gross_tau_s"), ("stale_s", "stale_s"))
#: the export's obs_guard.thin_proxy ("the env's low-tag spell stands for n_tags < min_tags") was calibrated at 2 [스펙]
THIN_PROXY_MIN_TAGS = 2


class ExportRefused(ValueError):
    """An export that says what it was trained on (obs_spec velocity_source / obs_guard) and this station does not
    implement it the same way: refused at build, and logged at warn (a refusal, not a missing file)."""

    export_refused = True


def _num(who: str, where: str, what: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ExportRefused(f"{who}: policy {where} {what} is {v!r}, not a number — refused")
    return float(v)


def station_velocity_source(dvl_raw, vel_lp_alpha) -> dict:
    """What THIS station implements, in the export's terms. ``dvl_raw`` = cfg.dvl (the resolved block when a DVL was
    built, else the yaml's: resolved again here, same values either way)."""
    from . import dvl as DV
    c = DV.resolve(dict(dvl_raw or {}))
    return {"selection": c["selection"], "hold_s": c["blend_hold_s"], "xfade_s": c["blend_xfade_s"],
            "gross_m_s": c["blend_gross_m_s"], "gross_tau_s": c["blend_gross_tau_s"], "stale_s": c["stale_s"],
            "dvl_w": False, "vel_lp_alpha": None if vel_lp_alpha is None else float(vel_lp_alpha)}


def check_velocity_source(spec: dict, who: str, where: str, dvl_raw, vel_lp_alpha) -> dict | None:
    """R7 (2026-10-16): an export whose obs_spec says ``velocity_source.trained: true`` was trained on the DVL-primary
    blend (RL_controller deploy/obs_builder.DvlBlendNp) and is accepted ONLY by a station that implements the same blend:
    ``dvl.selection: primary`` with equal hold / fade / gross / gross time constant / stale clock (1e-9), no DVL heave,
    and — when the export records it — the same tag-velocity low-pass (estimator.vel_lp_alpha). Anything else raises
    ExportRefused naming the difference. No block -> None (accepted exactly as before); ``trained: false`` -> the block
    (accepted; the engage note reads it)."""
    if "velocity_source" not in spec:
        return None
    vs = spec["velocity_source"]
    if not isinstance(vs, dict) or not isinstance(vs.get("trained"), bool):
        raise ExportRefused(f"{who}: policy {where} has a velocity_source block that is not {{trained: true|false, ...}} "
                            f"({vs!r:.80}) — refused, the velocity it was trained on is unknown")
    if not vs["trained"]:
        return vs
    try:
        st = station_velocity_source(dvl_raw, vel_lp_alpha)
    except ValueError as e:
        raise ExportRefused(f"{who}: policy {where} was trained on the DVL-primary blend, and this station's dvl: block "
                            f"does not resolve ({e}) — refused") from e
    if st["selection"] != "primary":
        raise ExportRefused(f"{who}: policy {where} was trained on the DVL-primary velocity blend (velocity_source.trained "
                            f"true: hold {vs.get('hold_s')} s, fade {vs.get('xfade_s')} s, gross {vs.get('gross_m_s')} "
                            f"m/s) but this station runs dvl.selection: {st['selection']} (hard switches) — refused; set "
                            f"dvl.selection: primary (and restart), or fly an export without velocity_source")
    for k, key in VELOCITY_SOURCE_KEYS:
        v = _num(who, where, f"velocity_source.{k}", vs.get(k))
        s = float(st[k])
        if abs(v - s) > 1e-9:
            raise ExportRefused(f"{who}: policy {where} was trained with velocity_source.{k} {v:g}; this station "
                                f"implements {s:g} (dvl.{key}) — refused: the policy would be fed a blend it never saw")
    if vs.get("dvl_w") is not False:
        raise ExportRefused(f"{who}: policy {where} was trained with the DVL's heave (velocity_source.dvl_w "
                            f"{vs.get('dvl_w')!r}); this station never feeds the DVL's w (heave stays on the tag finite "
                            f"difference) — refused")
    est = spec.get("estimator")
    if isinstance(est, dict) and "vel_lp_alpha" in est:
        v = _num(who, where, "estimator.vel_lp_alpha", est["vel_lp_alpha"])
        if vel_lp_alpha is None:
            raise ExportRefused(f"{who}: policy {where} was trained on a tag finite difference low-passed with "
                                f"estimator.vel_lp_alpha {v:g}; this station's vel_lp_alpha is unknown — refused")
        s = float(vel_lp_alpha)
        if abs(v - s) > 1e-9:
            raise ExportRefused(f"{who}: policy {where} was trained on a tag finite difference low-passed with "
                                f"estimator.vel_lp_alpha {v:g}; this station's vel_lp_alpha is {s:g} — refused: both "
                                f"the fallback velocity and the gross check's reference would differ")
    return vs


def check_obs_guard(spec: dict, who: str, where: str, guard_cfg) -> dict | None:
    """R7 (2026-10-16): an export whose obs_spec says ``obs_guard.trained: true`` was trained behind the pwm LOWs'
    observation guard (rl_pwm.PwmObsGuard rules 1-2) and is accepted ONLY by a LOW that applies it (``guard_cfg`` =
    its validated rl_pwm.obs_guard block; None = an axes LOW, which has none) with the same velocity slew and fix-jump
    limits and a thin fix = n_tags < the trained min_tags (the export's, else THIN_PROXY_MIN_TAGS). No block -> None;
    ``trained: false`` -> the block."""
    if "obs_guard" not in spec:
        return None
    og = spec["obs_guard"]
    if not isinstance(og, dict) or not isinstance(og.get("trained"), bool):
        raise ExportRefused(f"{who}: policy {where} has an obs_guard block that is not {{trained: true|false, ...}} "
                            f"({og!r:.80}) — refused, the observation guard it was trained behind is unknown")
    if not og["trained"]:
        return og
    if guard_cfg is None:
        raise ExportRefused(f"{who}: policy {where} was trained behind the station's observation guard (obs_guard.trained "
                            f"true: velocity slew {og.get('vel_slew_mps2')} m/s^2, fix jump {og.get('jump_m')} m) — LOW "
                            f"{who} does not apply it (only rl_pwm / rl_pwm_7d do) — refused")
    for k in ("vel_slew_mps2", "jump_m"):
        v = _num(who, where, f"obs_guard.{k}", og.get(k))
        s = float(guard_cfg[k])
        if abs(v - s) > 1e-9:
            raise ExportRefused(f"{who}: policy {where} obs_guard.{k} {v:g} != the station's rl_pwm.obs_guard.{k} "
                                f"{s:g} — refused")
    m = og.get("min_tags", THIN_PROXY_MIN_TAGS)
    if isinstance(m, bool) or not isinstance(m, int) or guard_cfg.get("min_tags") != m:
        raise ExportRefused(f"{who}: policy {where} its fix-jump rule was trained on a thin fix = n_tags < {m}; "
                            f"rl_pwm.obs_guard.min_tags is {guard_cfg.get('min_tags')} — refused")
    return og


def rl_velocity_note(mode: str, vs, armed: bool, run_why: str, estimator: str) -> tuple | None:
    """The engage-time line about the velocity an RL LOW is fed this run -> (level, text) or None. ``vs`` = the
    follower's ``velocity_source`` (the export's block, None without one), ``armed`` = the DVL is IN and this run is
    armed for it, ``run_why`` = why not, ``estimator`` = "tag" | "ekf" | "imu_dr". An export without the block gets the
    shipped warning (armed) or nothing — exactly as before 2026-10-16."""
    trained = isinstance(vs, dict) and vs.get("trained") is True
    if not trained:
        if armed:
            return ("warn", f"ctrl: {mode} was trained on a simulated tag-finite-difference velocity, not a DVL's — "
                            f"this run feeds it a different observation distribution (meta dvl.use_at_engage)")
        return None
    if str(estimator) != "tag":
        return None                 # refused at engage (_engage_refusal): never flies
    if armed:
        return ("info", f"ctrl: {mode} flies the DVL-primary blend it was trained on (hold {float(vs['hold_s']):g} s, "
                        f"fade {float(vs['xfade_s']):g} s, gross {float(vs['gross_m_s']):g} m/s)")
    p = vs.get("p")
    pct = isinstance(p, (list, tuple)) and len(p) >= 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in p[:2])
    return ("warn", f"ctrl: {mode} was trained mostly on the DVL-primary blend"
                    + (f" ({float(p[1]) * 100:.0f} % of its episodes)" if pct else "")
                    + f"; this run is NOT armed for the DVL ({run_why}) and flies on the tag finite difference — inside "
                      f"its training" + (f" ({float(p[0]) * 100:.0f} % tag-FD episodes)" if pct else "")
                    + " but at tag-FD accuracy")


# ----------------------------------------------------------------------------- the integral-error observation (2026-10-18)
#: Every key an obs_spec ``obs_integral`` block may carry: RL_controller scripts/export_policy._integral_block of the
#: live tree (round 1: enabled, tau_s, pos_max, att_max, slots, update_law, units, implementations) and of the round-2
#: dev copy logs/r2dev_20261018/src (+ att_error, cond_pos_m, cond_att_rad). Any other key is a law field this station
#: does not know: refused (check_obs_integral) — a future option must never be silently ignored.
OBS_INTEGRAL_KEYS = ("enabled", "tau_s", "pos_max", "att_max", "att_error", "cond_pos_m", "cond_att_rad", "slots",
                     "update_law", "units", "implementations")
#: the attitude-error laws StationErrorIntegrator implements: "quat_xy" (round 1; absent = this) and "euler_rp" (round 2)
OBS_INTEGRAL_ATT_ERRORS = ("quat_xy", "euler_rp")
#: Fix round 2026-10-18 (review F4): an enabled block's ``update_law`` and ``units`` are part of the contract, not prose.
#: The station accepts exactly the texts RL_controller writes for a law it implements: obs_integral_law_text transcribes
#: rov_rl/obs_spec.integral_update_law of the live tree (round 1) and of the dev copy logs/r2dev_20261018/src (round 2:
#: + one clause per option), OBS_INTEGRAL_UNITS_TEXT the units string of scripts/export_policy._integral_block (the same
#: in both trees). A law changed only in its text (another order, another frame, no new key) is refused at build instead
#: of flying on the old law; a mere rewording is refused too, and is answered by transcribing the new text here after
#: checking the law against StationErrorIntegrator. rov_gui/tests/test_rl_integral_obs.py compares both with the RL
#: generators (fixtures/rl_integral_law_texts_20261018.json always, the RL_controller sources when they are here).
#: ``implementations`` names where the RL side keeps the law (provenance, no law content): it must be a string and is
#: recorded in meta, not compared.
OBS_INTEGRAL_UNITS_TEXT = ("I_pos [m*s] world z-up, I_att [rad*s] body FLU (roll, pitch). int_att_rp = I_att / att_max "
                           "lies in [-1, 1]; int_pos_b = R(q_est)^T I_pos / pos_max where I_pos is clamped PER WORLD "
                           "AXIS, so a body component can reach sqrt(2) level and sqrt(3) tilted — do NOT clip or "
                           "range-assert it to [-1, 1] at deployment")
#: Fix round 2026-10-18 (review F2): how many finished engagements meta controller.obs_integral keeps (oldest dropped,
#: counted) — one REC CSV can span several engagements, and reset() at each engage starts fresh counts.
OBS_INTEGRAL_ENGAGEMENTS_KEPT = 64


def obs_integral_law_text(tau_s: float, pos_max: float, att_max: float, att_error: str = "quat_xy",
                          cond_pos_m: float = 0.0, cond_att_rad: float = 0.0) -> str:
    """The ``update_law`` text RL_controller writes into obs_spec.json for this law (fix round 2026-10-18, review F4):
    rov_rl/obs_spec.integral_update_law of the live tree (round 1 — the text for the defaults) and its round-2 dev copy
    (logs/r2dev_20261018/src: ``_integral_update_law_r1`` + " ROUND-2 OPTIONS: " and one clause per option), transcribed
    verbatim. ValueError for an att_error neither implements."""
    base = (f"per control tick (dt = control_dt), AFTER the estimator / observation guard / goal update and BEFORE the "
            f"frame is assembled: e_pos_w = clip(p_goal_w - p_est_w, -obs_offset_clip, +obs_offset_clip) per world axis "
            f"(world z-up; p_est_w = exactly the position offset_b uses); q_err = conj(q_est_f) (x) q_des_f with q_des_f / "
            f"q_est_f = the frame's two quaternion slots (goal-heading frame when obs_goal_frame, w >= 0 canonicalization "
            f"applied); e_att = 2 sign(q_err.w) (q_err.x, q_err.y) (roll, pitch error, body FLU: + = the hull must roll "
            f"right-side-down / pitch nose-DOWN to reach the goal); I_pos = clip(I_pos (1 - dt / {float(tau_s):g}) + dt "
            f"e_pos_w, +-{float(pos_max):g}); I_att = clip(I_att (1 - dt / {float(tau_s):g}) + dt e_att, "
            f"+-{float(att_max):g}) (leak, then add, then clip, per component). Slots: int_pos_b(3) = R(q_est_w)^T I_pos / "
            f"{float(pos_max):g} (q_est_w = the world-frame estimated attitude offset_b uses, not the goal-frame slot), "
            f"int_att_rp(2) = I_att / {float(att_max):g}. I_pos = I_att = 0 at engage / episode reset (whenever the "
            f"history is refilled), BEFORE that tick's update, so the first frame already carries dt e.")
    extra = []
    if att_error == "euler_rp":
        extra.append("e_att REPLACED by the heading-invariant (wrap(roll_des - roll_est), pitch_des - pitch_est) with ZYX "
                     "Euler roll = atan2(2(wx + yz), 1 - 2(x^2 + y^2)), pitch = asin(clip(2(wy - zx), -1, 1)) of the same "
                     "two frame quaternions (FLU: + roll = right-side-down, + pitch = nose-DOWN), wrap = atan2(sin, cos)")
    elif att_error != "quat_xy":
        raise ValueError(f"obs_integral_law_text: att_error must be one of {OBS_INTEGRAL_ATT_ERRORS}, got {att_error!r}")
    if float(cond_pos_m) > 0.0 or float(cond_att_rad) > 0.0:
        extra.append(f"CONDITIONAL integration per component: the add term dt e is used only while |e| < threshold "
                     f"(position {float(cond_pos_m):g} m, attitude {float(cond_att_rad):g} rad; 0 = always), else 0; the "
                     f"leak and the clip always run (leak, then the masked add, then clip)")
    return base if not extra else base + " ROUND-2 OPTIONS: " + "; ".join(extra) + "."


def _text_mismatch(got, want: str) -> str:
    """Where an export's text field departs from the station's (for the refusal message)."""
    if not isinstance(got, str):
        return f"it is {'missing' if got is None else type(got).__name__ + ' ' + repr(got)[:60]}"
    i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    lo = max(0, i - 40)
    return (f"first difference at character {i} of {len(got)} (station: {len(want)}): export "
            f"{got[lo:i + 60]!r} vs station {want[lo:i + 60]!r}")


def obs_integral_enabled(spec: dict, who: str, where: str) -> bool:
    """obs_spec ``obs_integral.enabled`` (2026-10-18): False without the block (every export before 2026-10-18). A block
    that is not a dict, or an ``enabled`` that is not a bool (absent included), is REFUSED (ExportRefused): whether the
    export's frame carries the integral-error slots would be unknown."""
    if "obs_integral" not in spec:
        return False
    oi = spec["obs_integral"]
    if not isinstance(oi, dict):
        raise ExportRefused(f"{who}: policy {where} has an obs_integral block that is not {{enabled: true|false, ...}} "
                            f"({oi!r:.80}) — refused, whether its frame carries the integral-error slots is unknown")
    en = oi.get("enabled")
    if not isinstance(en, bool):
        raise ExportRefused(f"{who}: policy {where} has obs_integral.enabled {en!r} — expected true or false; refused, "
                            f"whether its frame carries the integral-error slots is unknown")
    return en


def check_obs_integral(spec: dict, who: str, where: str, control_dt: float) -> dict | None:
    """The integral-error law an export was trained with (2026-10-18, obs_spec ``obs_integral``, RL_controller pwm12
    contract C6) -> None (no block, or ``enabled: false``: the frame of before, no state at all) or the law this station
    flies: {tau_s, pos_max, att_max, att_error, cond_pos_m, cond_att_rad, slots, control_dt, offset_clip, defaulted}
    (``defaulted`` = the round-2 keys the export does not carry, read as the round-1 law: quat_xy, 0, 0).

    REFUSED (ExportRefused, naming the export and the reason) — anything this station does not implement exactly:
    a block that is not a dict or an ``enabled`` that is not a bool (obs_integral_enabled); a key outside
    OBS_INTEGRAL_KEYS, enabled or not; and for an enabled block: tau_s / pos_max / att_max missing, not a number,
    non-finite or <= 0; tau_s <= control_dt (the leak 1 - dt/tau must lie in (0, 1)); att_error other than quat_xy /
    euler_rp; cond_pos_m / cond_att_rad not a number, non-finite or negative; slots other than INTEGRAL_SLOTS in that
    order; obs_offset_clip (the e_pos clamp) not a finite number > 0; and (fix round 2026-10-18, review F4) an
    ``update_law`` that is not obs_integral_law_text of the law the block declares, ``units`` other than
    OBS_INTEGRAL_UNITS_TEXT, or ``implementations`` that is not a non-empty string. The law then also carries
    ``law_round`` (1: the round-1 text, 2: with the ROUND-2 OPTIONS clause) and the export's ``implementations``."""
    enabled = obs_integral_enabled(spec, who, where)
    if "obs_integral" not in spec:
        return None
    oi = spec["obs_integral"]
    unknown = sorted(str(k) for k in set(oi) - set(OBS_INTEGRAL_KEYS))
    if unknown:
        raise ExportRefused(f"{who}: policy {where} obs_integral carries key(s) {unknown} this station does not "
                            f"implement (known: {list(OBS_INTEGRAL_KEYS)}) — refused: a law field must never be "
                            f"silently ignored")
    if not enabled:
        return None
    law = {}
    for k in ("tau_s", "pos_max", "att_max"):
        if k not in oi:
            raise ExportRefused(f"{who}: policy {where} obs_integral.{k} is missing — refused, the integral law is "
                                f"unknown")
        v = _num(who, where, f"obs_integral.{k}", oi[k])
        if not v > 0.0:
            raise ExportRefused(f"{who}: policy {where} obs_integral.{k} {v:g} is not > 0 — refused")
        law[k] = v
    dt = float(control_dt)
    if not (math.isfinite(dt) and dt > 0.0 and law["tau_s"] > dt):
        raise ExportRefused(f"{who}: policy {where} obs_integral.tau_s {law['tau_s']:g} s does not exceed the control "
                            f"step {dt:g} s — refused (the leak 1 - dt/tau must lie in (0, 1))")
    defaulted = []
    ae = oi.get("att_error", "quat_xy")
    if "att_error" not in oi:
        defaulted.append("att_error")
    if not isinstance(ae, str) or ae not in OBS_INTEGRAL_ATT_ERRORS:
        raise ExportRefused(f"{who}: policy {where} obs_integral.att_error {ae!r} — this station implements "
                            f"{list(OBS_INTEGRAL_ATT_ERRORS)} only; refused")
    law["att_error"] = ae
    for k in ("cond_pos_m", "cond_att_rad"):
        if k not in oi:
            defaulted.append(k)
            law[k] = 0.0
            continue
        v = _num(who, where, f"obs_integral.{k}", oi[k])
        if v < 0.0:
            raise ExportRefused(f"{who}: policy {where} obs_integral.{k} {v:g} is negative — refused (0 = always "
                                f"integrate, > 0 = integrate only while |e| < it)")
        law[k] = v
    slots = oi.get("slots")
    if slots != list(INTEGRAL_SLOTS):
        raise ExportRefused(f"{who}: policy {where} obs_integral.slots {slots!r} != {list(INTEGRAL_SLOTS)} — refused, "
                            f"the station builds exactly those slots in that order")
    clip = _num(who, where, "obs_offset_clip", spec.get("obs_offset_clip", 1.0))
    if not clip > 0.0:
        raise ExportRefused(f"{who}: policy {where} obs_offset_clip {clip:g} is not > 0 — refused (it is the "
                            f"integral's position-error clamp)")
    # fix round 2026-10-18 (review F4): the texts are the contract — a law changed only in its update_law / units text
    # must never fly on the law this station implements
    want = obs_integral_law_text(law["tau_s"], law["pos_max"], law["att_max"], ae, law["cond_pos_m"],
                                 law["cond_att_rad"])
    for k, w in (("update_law", want), ("units", OBS_INTEGRAL_UNITS_TEXT)):
        if oi.get(k) != w:
            raise ExportRefused(f"{who}: policy {where} obs_integral.{k} is not the text RL_controller writes for the law "
                                f"this block declares (tau {law['tau_s']:g} s, clamps {law['pos_max']:g} / "
                                f"{law['att_max']:g}, att_error {ae}, cond {law['cond_pos_m']:g} / "
                                f"{law['cond_att_rad']:g}) — {_text_mismatch(oi.get(k), w)}; refused: a law change "
                                f"written only into the text must never fly on the old law (if RL_controller merely "
                                f"reworded it, transcribe the new text into rl_policy.py after checking the law)")
    impl = oi.get("implementations")
    if not isinstance(impl, str) or not impl.strip():
        raise ExportRefused(f"{who}: policy {where} obs_integral.implementations {impl!r} — expected the RL side's "
                            f"implementation names (a non-empty string); refused")
    law.update({"slots": list(INTEGRAL_SLOTS), "control_dt": dt, "offset_clip": clip, "defaulted": defaulted,
                "law_round": 2 if " ROUND-2 OPTIONS: " in want else 1, "implementations": impl})
    return law


class StationErrorIntegrator:
    """The integral-error observation of an obs_integral export (2026-10-18, RL_controller pwm12 contract C6) for one
    vehicle, float64: RL_controller deploy/obs_builder.ErrorIntegratorNp (round 1) with the round-2 options of its dev
    copy logs/r2dev_20261018/src/deploy/obs_builder.py (att_error "euler_rp", cond_pos_m / cond_att_rad), transcribed
    operation for operation — the slots equal that twin's (rov_gui/tests/test_rl_integral_obs.py, <= 1e-9).

    ``update(e_src_w, q_des_f, q_est_f, q_est_w)`` = one control tick -> the 5 frame slots [int_pos_b(3), int_att_rp(2)]:
        e_pos = clip(e_src_w, +-offset_clip)                    (world z-up; HwRl passes its LEASHED goal offset d)
        e_att = 2 sign(q_err.w) (q_err.x, q_err.y), q_err = conj(q_est_f) (x) q_des_f       ("quat_xy")
              | (wrap(roll(q_des_f) - roll(q_est_f)), pitch(q_des_f) - pitch(q_est_f))  ("euler_rp", ZYX Euler)
        add   = dt e, zeroed per component where |e| >= cond (only when cond > 0; STRICT <, as trained)
        I     = clip(I (1 - dt/tau) + add, +-max)               (leak, then the masked add, then the clip)
        slots = [quat_apply_inverse(q_est_w, I_pos) / pos_max, I_att / att_max]   (q_est_w = the WORLD estimate)
    int_pos_b is not bounded to [-1, 1] (I_pos is clamped per WORLD axis; rotated, a body component reaches sqrt(2)
    level and sqrt(3) tilted [유도]) and is never clipped here.

    A tick whose NEW state is not finite is not committed: the slots it returns are non-finite (so HwRlPwm.step refuses
    the observation: neutral pulses + a counted solver failure) and the integral resumes from its last finite value at
    the next tick. Training never has a non-finite state, so nothing it trained on differs; committing it would leave
    every later observation of the engagement non-finite. ``reset`` = a fresh integral (history refill), ``clear_counts``
    = fresh per-engagement counts (HwRl.reset); both run at engage.

    ``update(..., add=False)`` (fix round 2026-10-18, review F1) = an update whose add is masked as a whole: the leak and
    the clip run, nothing is added — the station's override rule (HwRl._frame: the update after a command that was not
    the policy's own). Counted per engagement: ticks_add_masked, overrides (masked runs), override_ticks_max."""

    IMPLEMENTATION = ("rov_gui/control/rl_policy.py StationErrorIntegrator: numpy twin of RL_controller "
                      "deploy/obs_builder.ErrorIntegratorNp (round 1) + the round-2 options of "
                      "logs/r2dev_20261018/src/deploy/obs_builder.py (att_error euler_rp, cond_pos_m / cond_att_rad)")

    def __init__(self, dt: float, tau_s: float, pos_max: float, att_max: float, offset_clip: float,
                 att_error: str = "quat_xy", cond_pos_m: float = 0.0, cond_att_rad: float = 0.0):
        self.dt, self.tau = float(dt), float(tau_s)
        self.pos_max, self.att_max, self.clip = float(pos_max), float(att_max), float(offset_clip)
        self.att_error = str(att_error)
        self.cond_pos, self.cond_att = float(cond_pos_m), float(cond_att_rad)
        if self.att_error not in OBS_INTEGRAL_ATT_ERRORS:
            raise ValueError(f"StationErrorIntegrator: att_error must be one of {OBS_INTEGRAL_ATT_ERRORS}, "
                             f"got {att_error!r}")
        vals = (self.dt, self.tau, self.pos_max, self.att_max, self.clip, self.cond_pos, self.cond_att)
        if not (all(math.isfinite(v) for v in vals) and self.dt > 0.0 and self.tau > self.dt and self.pos_max > 0.0
                and self.att_max > 0.0 and self.clip > 0.0 and self.cond_pos >= 0.0 and self.cond_att >= 0.0):
            raise ValueError(f"StationErrorIntegrator: need finite tau_s > dt > 0, positive clamps and cond >= 0, got dt "
                             f"{self.dt} tau {self.tau} pos_max {self.pos_max} att_max {self.att_max} clip {self.clip} "
                             f"cond {self.cond_pos} / {self.cond_att}")
        self.leak = 1.0 - self.dt / self.tau
        self.reset()
        self.clear_counts()

    def reset(self) -> None:
        """A fresh integral: the history is refilled at this tick (engage), BEFORE that tick's update."""
        self.I_pos = np.zeros(3)          # [m*s] world z-up
        self.I_att = np.zeros(2)          # [rad*s] body FLU roll, pitch

    def clear_counts(self) -> None:
        """Fresh per-engagement counts (meta controller.obs_integral.counts)."""
        self.ticks = 0
        self.ticks_any_saturated = 0
        self.ticks_nonfinite = 0
        self.ticks_add_masked = 0           # fix round (F1): committed updates whose add was masked (override rule)
        self.overrides = 0                  # runs of consecutive masked updates
        self.override_ticks_max = 0         # the longest such run [ticks]
        self._masked_run = 0
        self.max_abs_slot = np.zeros(INTEGRAL_DIM)
        self.last_slots = None

    def error(self, e_src_w, q_des_f, q_est_f) -> tuple:
        """(e_pos (3,), e_att (2,)) of one tick, before the conditional mask."""
        e_pos = np.clip(np.asarray(e_src_w, dtype=float), -self.clip, self.clip)
        if self.att_error == "quat_xy":
            q_err = quat_mul(quat_conjugate(q_est_f), q_des_f)
            e_att = 2.0 * np.sign(q_err[0]) * q_err[1:3]
        else:                                          # "euler_rp": ZYX roll / pitch differences (heading-invariant)
            r_d, p_d = roll_pitch_zyx(q_des_f)
            r_e, p_e = roll_pitch_zyx(q_est_f)
            e_att = np.array([np.arctan2(np.sin(r_d - r_e), np.cos(r_d - r_e)), p_d - p_e])
        return e_pos, e_att

    def update(self, e_src_w, q_des_f, q_est_f, q_est_w, add: bool = True) -> np.ndarray:
        """One control tick -> the 5 slots (see the class docstring). ``add`` False = the whole add masked (leak only)."""
        e_pos, e_att = self.error(e_src_w, q_des_f, q_est_f)
        add_pos, add_att = self.dt * e_pos, self.dt * e_att
        if not add:                                    # fix round (F1): the override rule — leak and clip only
            add_pos, add_att = np.zeros(3), np.zeros(2)
        if self.cond_pos > 0.0:                        # conditional integration per component
            add_pos = np.where(np.abs(e_pos) < self.cond_pos, add_pos, 0.0)
        if self.cond_att > 0.0:
            add_att = np.where(np.abs(e_att) < self.cond_att, add_att, 0.0)
        i_pos = np.clip(self.I_pos * self.leak + add_pos, -self.pos_max, self.pos_max)
        i_att = np.clip(self.I_att * self.leak + add_att, -self.att_max, self.att_max)
        slots = np.concatenate([quat_apply_inverse(np.asarray(q_est_w, dtype=float), i_pos) / self.pos_max,
                                i_att / self.att_max])
        if bool(np.all(np.isfinite(i_pos))) and bool(np.all(np.isfinite(i_att))):
            self.I_pos, self.I_att = i_pos, i_att
            self.ticks += 1
            if add:
                self._masked_run = 0
            else:
                self.ticks_add_masked += 1
                self._masked_run += 1
                if self._masked_run == 1:
                    self.overrides += 1
                self.override_ticks_max = max(self.override_ticks_max, self._masked_run)
            if bool(np.any(np.abs(i_pos) >= self.pos_max)) or bool(np.any(np.abs(i_att) >= self.att_max)):
                self.ticks_any_saturated += 1
            if bool(np.all(np.isfinite(slots))):
                self.max_abs_slot = np.maximum(self.max_abs_slot, np.abs(slots))
                self.last_slots = slots.copy()
        else:
            self.ticks_nonfinite += 1
        return slots


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
    #: 2026-10-16 (R7): does this LOW apply the pwm observation guard (rl_pwm.PwmObsGuard)? An export trained behind it
    #: (obs_spec obs_guard.trained) is refused by a LOW that does not. HwRlPwm (and HwRlPwm7d) set it.
    APPLIES_OBS_GUARD = False
    #: 2026-10-18 (obs_integral, RL_controller pwm12 C6): may this LOW fly an export with the integral-error slots? Only
    #: the pwm LOWs (HwRlPwm, HwRlPwm7d): theirs is the observation parity-tested against RL_controller
    #: deploy/obs_builder.py with the integral (rov_gui/tests/test_rl_integral_obs.py), and no axes / axes6 export with
    #: the integral exists. rl / rl_7d refuse such an export at build.
    OBS_INTEGRAL_LOW = False

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
        #: 2026-10-15 (obs_spec.obs_goal_frame, RL_controller pwm11att7 and later): q_des / q_est in the goal-heading frame
        #: (goal_frame_quats, BEFORE the canonicalization) for every LOW that loads such an export. The layout gate refuses
        #: an export whose frame_layout and obs_goal_frame disagree, whatever the LOW (rl / rl_pwm compared no layout
        #: before this — a goal-frame export would have flown there on world-frame quaternions).
        self.goal_frame = check_frame_layout(self.spec, self.CFG_KEY, os.path.relpath(pdir, REPO_ROOT))
        #: 2026-10-16 (R7, RL_controller logs/eval/dvl_trip_fix_20261015 SPEC.md §8): the velocity the export was
        #: trained on (obs_spec velocity_source; None = no block, exactly as before) — refused at build unless this
        #: station implements the same blend — and, for a LOW without the pwm observation guard, an export trained
        #: behind that guard is refused here too (HwRlPwm checks its own guard against the export).
        rel = os.path.relpath(pdir, REPO_ROOT)
        _alpha = getattr(cfg, "vel_lp_alpha", None)
        self.velocity_source = check_velocity_source(self.spec, self.CFG_KEY, rel, getattr(cfg, "dvl", None), _alpha)
        self._vs_station = (station_velocity_source(getattr(cfg, "dvl", None), _alpha)
                            if isinstance(self.velocity_source, dict) and self.velocity_source.get("trained") else None)
        if not self.APPLIES_OBS_GUARD:
            check_obs_guard(self.spec, self.CFG_KEY, rel, None)
        if self.policy.obs_dim != self.frame_dim * self.K or self.policy.act_dim != self.n_act:
            raise ValueError(f"rl: policy dims {self.policy.obs_dim}->{self.policy.act_dim} do not match obs_spec "
                             f"({self.frame_dim}x{self.K} -> {self.n_act})")
        exp_dt = float(self.spec.get("control_dt", 1.0 / float(cfg.ctrl_hz)))
        if abs(exp_dt - 1.0 / float(cfg.ctrl_hz)) > 1e-6:
            raise ValueError(f"rl: policy trained at {1/exp_dt:.1f} Hz but ctrl_hz is {cfg.ctrl_hz}")
        #: 2026-10-18 (obs_spec obs_integral, RL_controller pwm12 contract C6): the integral law the export was trained
        #: with — None (no block, or enabled false) = the frame, meta and build log of before, no integrator at all.
        #: Anything the station does not implement is refused (check_obs_integral), and only the pwm LOWs fly it.
        self.obs_integral = check_obs_integral(self.spec, self.CFG_KEY, rel, exp_dt)
        if self.obs_integral is not None and not self.OBS_INTEGRAL_LOW:
            raise ExportRefused(f"{self.CFG_KEY}: policy {rel} was trained with the integral-error observation "
                                f"(obs_integral.enabled true, slots {list(INTEGRAL_SLOTS)}) — LOW {self.CFG_KEY} does not "
                                f"fly it (only rl_pwm / rl_pwm_7d, whose observation is parity-tested with the integral "
                                f"against RL_controller deploy/obs_builder.py); refused")
        oi = self.obs_integral
        self._integ = None if oi is None else StationErrorIntegrator(
            oi["control_dt"], oi["tau_s"], oi["pos_max"], oi["att_max"], oi["offset_clip"], att_error=oi["att_error"],
            cond_pos_m=oi["cond_pos_m"], cond_att_rad=oi["cond_att_rad"])
        #: fix round 2026-10-18 (review F1, obs_integral exports only): True = the command that left after the previous
        #: observation was NOT the policy's own (HwRlPwm.step: HOLD pulses / a failed tick; note_applied_pwm(None)) ->
        #: the next integral update masks its add (leak only). (review F2) _int_tick = (slots, masked) of this tick's
        #: update for step's info; the engagement record (index, time span) and the finished engagements' records.
        self._int_mask_next = False
        self._int_tick = None
        self._int_eng_index = 1
        self._int_span = {"t_traj_s": [None, None], "wall_s": [None, None]}
        self._int_history = []
        self._int_history_dropped = 0
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
        #: A K=1 reference with r_ned == 0 (a station / approach setpoint, a crab path) gets its yaw-rate feed-forward by
        #: differencing the reference yaw between ticks; a heading-follow plan carries its own (2026-10-14). Either is
        #: clipped to yaw_rate_max. NOTE the shipped 0.6 matches neither export number: moving_goal.yaw_rate_max 0.4
        #: (in-place / independent yaw drift) and turn_rate_max 1.0 (heading-follow arcs) in traj10_s1, traj12_s2 and
        #: pwm11k0.2_s0 obs_spec.json [스펙].
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
            f"(K={self.K}, feed-forward={'on' if self.goal_velocity else 'off'}"
            f"{', goal-frame quaternions' if self.goal_frame else ''}), act {self.n_act}, probe {self.probe_ms:.2f} ms; "
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
        q_d, q_o = q_des, q_est
        if self.goal_frame:                        # 2026-10-15: rotated by -goal yaw BEFORE the canonicalization
            q_d, q_o = goal_frame_quats(q_des, q_est)
        q_d, q_o = (_unique(q_d), _unique(q_o)) if self.quat_unique else (q_d, q_o)
        parts = [q_d, offset_b, q_o, v_b, gyro]
        if self.goal_velocity:                     # body frame of the WORLD q_est, goal frame or not (as trained)
            goal_v = C_NED_TO_UP @ np.asarray(v_ref_ned, float)
            parts += [quat_apply_inverse(q_est, goal_v), np.array([-float(r_ref_ned)])]
        integ = getattr(self, "_integ", None)
        if integ is not None:
            # 2026-10-18 (obs_integral, RL_controller rov_env order): a history refilled at this tick (engage) starts a
            # fresh integral BEFORE the update; then one update with the error THIS frame shows. e_pos_w = clip(d,
            # +-obs_offset_clip) of the SAME leashed d offset_b uses — a station convention: training has no leash, so
            # the two agree exactly while |goal - p| <= leash_m; beyond it the station integrates (goal - p)_i * leash_m /
            # |goal - p| where training integrates clip((goal - p)_i, +-obs_offset_clip). A component above pos_max / tau
            # in both (0.1 m with the pwm12 constants [유도: fixed point I = tau e]) drives I_pos into its clamp either
            # way. q_d / q_o = this frame's
            # quaternion slots; the slots rotate I_pos by the WORLD q_est (as offset_b). Never clipped: int_pos_b may
            # exceed 1 (up to sqrt(3) [유도]).
            # Fix round 2026-10-18 (review F1): the update after a command that was not the policy's own (HOLD pulses,
            # a failed tick, nothing sent — _int_mask_next) masks its add and keeps the leak: an error the policy could
            # not act on is not integrated (a station convention [예측]; training integrates through 1-2 tick
            # dropouts only). A refill (engage) clears the mask with the integral.
            if self._hist is None:
                integ.reset()
                self._int_mask_next = False
            masked = bool(self._int_mask_next)
            slots = integ.update(d, q_d, q_o, q_est, add=not masked)
            self._int_tick = (slots, masked)
            parts.append(slots)
        return np.concatenate(parts + [self._prev_action])

    def _q_des(self, yaw_ref) -> np.ndarray:
        """The goal quaternion: LEVEL with the reference yaw — what every export before att7 was trained on. HwRl7d
        (LOW rl_7d) overrides it with the plan's roll/pitch."""
        return quat_from_euler_xyz(0.0, 0.0, -float(yaw_ref))

    def observation(self, eta_ned, nu_ned, t: float) -> np.ndarray:
        eta = np.asarray(eta_ned, float)
        nu = np.asarray(nu_ned, float)
        p_ref, yaw_ref, v_ref, r_ref = self._reference_ned_at(t)
        if self.yaw_rate_ff and r_ref == 0.0 and self._yaw_ref_prev is not None:
            r_ref = math.atan2(math.sin(yaw_ref - self._yaw_ref_prev), math.cos(yaw_ref - self._yaw_ref_prev)) / self.dt
        # Since 2026-10-14 a heading-follow K=1 plan CARRIES its yaw rate (analytic, PathCursor.plan): the plan's value is
        # used as is, and every r_ref — recovered or carried — is clipped to yaw_rate_max. The previous yaw is kept
        # EVERY tick: while a plan carried r the old code froze it, so the first r == 0 tick after a vertex turn would
        # difference against the heading from before the turn — a one-tick (clipped) rate spike.
        r_ref = float(np.clip(r_ref, -self.yaw_rate_max, self.yaw_rate_max))
        self._yaw_ref_prev = float(yaw_ref)
        frame = self._frame(eta, nu, p_ref, yaw_ref, v_ref, r_ref)
        if getattr(self, "_integ", None) is not None:   # fix round (F2): the engagement's time span, for its record
            sp = self._int_span
            sp["t_traj_s"] = [float(t) if sp["t_traj_s"][0] is None else sp["t_traj_s"][0], float(t)]
            now = time.time()
            sp["wall_s"] = [now if sp["wall_s"][0] is None else sp["wall_s"][0], now]
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
        integ = getattr(self, "_integ", None)      # 2026-10-18: a fresh integral and fresh per-engagement counts
        if integ is not None:
            # fix round (F2): the finished engagement's record survives into meta (bounded) — one REC CSV can span
            # several engagements, and its closing meta used to describe only the last
            if integ.ticks or integ.ticks_nonfinite:
                self._int_history.append(self._obs_integral_engagement())
                if len(self._int_history) > OBS_INTEGRAL_ENGAGEMENTS_KEPT:
                    del self._int_history[0]
                    self._int_history_dropped += 1
                self._int_eng_index += 1
            integ.reset()
            integ.clear_counts()
            self._int_mask_next = False
            self._int_tick = None
            self._int_span = {"t_traj_s": [None, None], "wall_s": [None, None]}

    def _obs_integral_log_line(self) -> str:
        """The one build line of an obs_integral export (2026-10-18; "" without one — the build log of before)."""
        oi = self.obs_integral if getattr(self, "_integ", None) is not None else None
        if oi is None:
            return ""
        cond = ("" if oi["cond_pos_m"] == 0.0 and oi["cond_att_rad"] == 0.0 else
                f", integrating only while |e| < {oi['cond_pos_m']:g} m / {oi['cond_att_rad']:g} rad")
        # fix round (review F3): the round-2 keys the export does NOT carry were read as the round-1 law — said here, at
        # the pool edge, because the live RL_controller exporter writes none of them even for a round-2 run
        dflt = ("" if not oi["defaulted"] else
                f" (DEFAULTED: {', '.join(oi['defaulted'])} absent from the export, read as the round-1 law; a round-2 "
                f"run exported by the live RL_controller exporter looks exactly like this — check the run's "
                f"params/env.yaml obs_int_*)")
        return (f"{self.CFG_KEY}: integral-error slots {list(INTEGRAL_SLOTS)} (obs_integral, RL_controller pwm12 C6, "
                f"round-{oi['law_round']} law text): tau {oi['tau_s']:g} s, clamps {oi['pos_max']:g} m*s / "
                f"{oi['att_max']:g} rad*s, e_att {oi['att_error']}{cond}{dflt}, dt {oi['control_dt']:g} s [스펙 "
                f"obs_spec.json]; reset at engage before the first update; e_pos = the LEASHED goal offset (leash "
                f"{self.leash_m:g} m — training has none) clipped to +-{oi['offset_clip']:g} m; int_pos_b unclipped (up "
                f"to sqrt(3)); the add is MASKED (leak only) after a command that was not the policy's own (tag-gate "
                f"HOLD pulses, a failed tick, nothing sent: coast / pwm-gate grace) — a station convention [예측]; "
                f"per-tick slots in step info obs_int_slots, NOT in the CSV (meta controller.obs_integral) — never "
                f"flown [예측]")

    def _obs_integral_counts(self) -> dict:
        """The current engagement's counts (meta controller.obs_integral.counts)."""
        g = self._integ
        return {"ticks": int(g.ticks), "ticks_any_saturated": int(g.ticks_any_saturated),
                "ticks_nonfinite": int(g.ticks_nonfinite), "ticks_add_masked": int(g.ticks_add_masked),
                "overrides": int(g.overrides), "override_ticks_max": int(g.override_ticks_max),
                "max_abs_slot": [float(v) for v in g.max_abs_slot],
                "last_slots": None if g.last_slots is None else [float(v) for v in g.last_slots]}

    def _obs_integral_engagement(self) -> dict:
        """One engagement's record (fix round 2026-10-18, review F2): its index since build, the t handed to step (the
        CSV t_traj) and the wall clock (epoch s) of its first and last observation, its counts and its state."""
        g = self._integ
        return {"engagement": int(self._int_eng_index), "t_traj_s": list(self._int_span["t_traj_s"]),
                "wall_s": list(self._int_span["wall_s"]), "counts": self._obs_integral_counts(),
                "state": {"I_pos_w": [float(v) for v in g.I_pos], "I_att_rp": [float(v) for v in g.I_att]}}

    def _obs_integral_meta(self) -> dict:
        """meta controller.obs_integral (2026-10-18, only for an obs_integral export): the export's law, what implements
        it here, the leash and override conventions, the current engagement's counts and state at the time of the
        call, and (fix round, review F2) the finished engagements' records."""
        g, oi = self._integ, self.obs_integral
        return {"enabled": True, "tau_s": oi["tau_s"], "pos_max": oi["pos_max"], "att_max": oi["att_max"],
                "att_error": oi["att_error"], "cond_pos_m": oi["cond_pos_m"], "cond_att_rad": oi["cond_att_rad"],
                "slots": list(oi["slots"]), "control_dt": oi["control_dt"], "offset_clip_m": oi["offset_clip"],
                "defaulted": list(oi["defaulted"]), "leak": float(g.leak),
                "law_round": int(oi["law_round"]),
                "law_text_note": ("update_law / units equal RL_controller's generators for this law (refused otherwise, "
                                  "review F4); 'defaulted' = round-2 keys absent from the export and read as the "
                                  "round-1 law — the live RL_controller exporter writes none of them even for a round-2 "
                                  "run (review F3)"),
                "export_implementations": oi["implementations"],
                "implementation": StationErrorIntegrator.IMPLEMENTATION,
                "leash_note": (f"e_pos_w = clip(d, +-{oi['offset_clip']:g} m) per world z-up axis with d = the LEASHED goal "
                               f"offset offset_b uses (leash_m {self.leash_m:g}; a station convention, training has no "
                               f"leash): equal to training while |goal - p| <= leash_m; beyond it the station integrates "
                               f"(goal - p)_i * leash_m / |goal - p| where training integrates clip((goal - p)_i, "
                               f"+-obs_offset_clip); a component above pos_max / tau in both drives I_pos into its clamp "
                               f"either way [유도: fixed point I = tau e]"),
                "override_rule": ("the update that follows a command which was not the policy's own masks its add "
                                  "(leak and clip only, the law's masked-add step): after the tag gate's HOLD pulses "
                                  "(CSV pwm_guard 3), a failed tick's neutral (status != 0), or nothing sent "
                                  "(note_applied_pwm(None): the worker's station-bridge coast, the pwm-gate grace "
                                  "window, a disengage). Training integrates through its own 1-2 tick actuation "
                                  "dropouts (dr.act_dropout_ticks) and has no HOLD / coast / grace. A STATION "
                                  "convention (review F1, 2026-10-18) [예측], not a trained behaviour: the RL side "
                                  "decides whether to keep it, reset at release, or model the overrides in the env"),
                "engagement": int(self._int_eng_index),
                "t_traj_s": list(self._int_span["t_traj_s"]), "wall_s": list(self._int_span["wall_s"]),
                "counts": self._obs_integral_counts(),
                "state": {"I_pos_w": [float(v) for v in g.I_pos], "I_att_rp": [float(v) for v in g.I_att]},
                "previous_engagements": [dict(r) for r in self._int_history],
                "previous_engagements_dropped": int(self._int_history_dropped),
                "per_tick": ("step info obs_int_slots (the 5 slots of the frame the network saw this tick; None when "
                             "no frame was built) and obs_int_add_masked (the override rule masked this tick's add) — "
                             "NOT in the CSV until workers.py logs them"),
                "note": ("counts are per engagement (reset at engage; the finished engagements' records stay in "
                         "previous_engagements, the last " + str(OBS_INTEGRAL_ENGAGEMENTS_KEPT) + ", with the t_traj "
                         "and wall-clock span that places them in a CSV): ticks = committed updates (one per "
                         "observation, the tag gate's HOLD ticks included), ticks_any_saturated = ticks with any |I| at "
                         "its clamp (training's saturated()), ticks_nonfinite = ticks whose new state was not finite (not "
                         "committed; that observation was refused), ticks_add_masked / overrides / override_ticks_max = "
                         "the override rule's masked updates, their runs and the longest run, max_abs_slot = per slot "
                         "[int_pos_b x, y, z, int_att_rp roll, pitch]; int_pos_b may exceed 1 (per-world-axis clamp, "
                         "rotated: up to sqrt(3) [유도]) and is never clipped. The integral state is NOT in the CSV "
                         "(workers.py has no column): per tick it is in step's info only. Never flown [예측]")}

    def meta(self) -> dict:
        d = {"type": "rl", "solver": "rl", "ctrl_hz": float(self.cfg.ctrl_hz), "dt_s": float(self.dt),
             "policy_dir": os.path.relpath(self.policy_dir, REPO_ROOT), "action_mode": self.action_mode,
             "checkpoint": self.spec.get("checkpoint"), "obs_dim": int(self.policy.obs_dim),
             "history_len": self.K, "frame_layout": self.spec.get("frame_layout"),
             "obs_quat_unique": self.quat_unique, "obs_goal_velocity": self.goal_velocity,
             "obs_goal_frame": self.goal_frame,
             "obs_offset_clip_m": self.clip_m, "dr_preset": self.spec.get("dr_preset"),
             "pilot_gain_assumed": self.pilot_gain_assumed,
             "axis_gain_for_pseudo_wrench": dict(self._gains),
             "axis_slew_per_s": self.axis_slew_per_s, "axis_cap": self.axis_cap, "leash_m": self.leash_m,
             "yaw_rate_ff": self.yaw_rate_ff, "yaw_rate_max": self.yaw_rate_max,
             "axes_note": "u = axes * axis_gain (pseudo-wrench); the worker's wrench_to_axes returns the policy axes, "
                          "capped by axis_cap; roll/pitch not commanded",
             "frames": "policy world z-up / body FLU; station NED/FRD mirrored with diag(1,-1,-1)"
                       + ("; q_des / q_est rotated by -goal yaw (ZYX yaw of q_des) before the w>=0 canonicalization "
                          "(obs_goal_frame)" if self.goal_frame else ""),
             "ref_preview": False, "path_reference": "shared spatial plan, stage 0 (+ its velocity as feed-forward)",
             # 2026-10-16 (R7): what velocity the export says it was trained on, and what this station implements
             # (only for an export trained on the blend; checked equal at build)
             "velocity_source": {"export": self.velocity_source, "station": self._vs_station},
             "provenance": "RL_controller (Isaac Lab, PPO), see rl_policies/<name>/RL_controller_README.md"}
        if getattr(self, "_integ", None) is not None:  # 2026-10-18: only an obs_integral export (else the meta of before)
            d["obs_integral"] = self._obs_integral_meta()
        return d


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

    2026-10-18 — the HEAVE TRIM (``rl_pwm.heave_trim_n``; HwRlPwm7d reads its own ``rl_pwm_7d.heave_trim_n``): a constant
    force, physical N, + = UP in the world, in the export's nominal thrust model (the model of obs_guard.hold_wrench_z_n,
    with the opposite sign: that one is NED, -4 = 4 N up). Each tick it is rotated into the body frame with the
    estimated roll/pitch and allocated over the four verticals (rl_pwm.heave_trim_thrust — the HOLD's allocation), added
    to the policy's thrust, and the cap is applied to the TOTAL (rl_pwm.trim_and_cap: scale = one scalar on the total
    thrust vector). The policy is fed back its OWN component after that same scale (note_applied_pwm), never the trim;
    while the tag gate HOLDs, the HOLD pulses leave unchanged (obs_guard.hold_wrench_z_n is the HOLD's own absolute
    heave — set it for the payload too). 0 = the code path of before, bit for bit. Refused at build beyond
    +-rl_pwm.HEAVE_TRIM_MAX_N (10 N) and beyond rl_pwm.HEAVE_TRIM_MAX_CAP_FRACTION of one vertical's capped range (the
    cap budget, review D-R2). The added wrench is the requested pure heave only where every vertical total clears its
    deadband edge (review D-R1): the build line prints the trim's pattern with the policy's verticals at rest, info
    ``trim_wrench_ned`` / ``trim_n_edge`` say what each tick delivered (nominal model), meta controller.heave_trim
    counts and averages it. The CSV (workers.py) has no column for the policy's own part: on a run with a trim, pwm1..8
    and uX..uN are policy + trim (review D-R3; RECORD BOUNDARY meta controller.heave_trim_n). NEVER FLOWN [예측].

    2026-10-18 — the INTEGRAL-ERROR observation (obs_spec ``obs_integral.enabled: true``, RL_controller pwm12 contract
    C6, the "PI" variant): this LOW and HwRlPwm7d are the only ones that fly such an export (OBS_INTEGRAL_LOW). The
    integral lives in HwRl._frame (StationErrorIntegrator, see the module docstring); here it only matters that
    ``step`` builds the observation every tick, after the observation guard and including the tag gate's HOLD, so the
    integral is updated with the guarded state on every tick, as the network history is; a non-finite integral update
    surfaces as a non-finite observation (neutral + n_fail). Fix round 2026-10-18 (review F1): ``step`` marks a tick
    whose pulses are not the policy's own (the HOLD pulses, a failed tick's neutral) and ``note_applied_pwm(None)`` one
    on which nothing left (coast, pwm-gate grace, disengage); the NEXT integral update then masks its add (leak only).
    (review F2) info ``obs_int_slots`` / ``obs_int_add_masked`` carry the slots and the mask per tick — NOT in the CSV;
    meta controller.obs_integral. NEVER FLOWN [예측].

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
    #: 2026-10-16 (R7): this LOW applies rl_pwm.PwmObsGuard, so an export trained behind it is checked against its config
    APPLIES_OBS_GUARD = True
    #: 2026-10-18: this LOW (and HwRlPwm7d) flies an export with the integral-error slots (obs_integral)
    OBS_INTEGRAL_LOW = True

    def __init__(self, cfg, log=print):
        from .rl_pwm import (HEAVE_TRIM_MAX_CAP_FRACTION, PWM_CHANNELS_DEFAULT, PwmModel, PwmObsGuard,
                             ThrusterProbeCheck, heave_trim_at_rest, heave_trim_budget, heave_trim_thrust, hold_throttles,
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
        #: 2026-10-16 (R7): an export trained behind the guard (obs_spec obs_guard.trained) flies only with THIS guard
        #: configured as it was trained (slew, fix jump, thin = n_tags < min_tags); None = no block, as before.
        self.obs_guard_trained = check_obs_guard(self.spec, self.CFG_KEY, os.path.relpath(self.policy_dir, REPO_ROOT),
                                                 self.obs_guard.cfg)
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
        #: 2026-10-18: the heave trim of THIS LOW (its own block's heave_trim_n; physical N, + = up in the world, refused
        #: beyond +-10 N) and, for the meta / the build line, its allocation at LEVEL attitude (the per-tick one is
        #: rotated by the estimated roll/pitch). RECORD BOUNDARY: meta controller.heave_trim_n.
        self.heave_trim_n, self.heave_trim_key = self._heave_trim_cfg(cfg)
        self.heave_trim_thrust_level = heave_trim_thrust(self.model, self.heave_trim_n)
        #: review D-R2: the cap acts on policy + trim, so a trim that fills a vertical's capped range would scale the
        #: policy's WHOLE command down whenever it drives that thruster the same way — refused at build, never clipped
        self.heave_trim_budget = heave_trim_budget(self.model, self.heave_trim_n, self.pwm_cap)
        b = self.heave_trim_budget
        if b["max"] > HEAVE_TRIM_MAX_CAP_FRACTION:
            raise ValueError(
                f"{self.heave_trim_key} {self.heave_trim_n:+.2f} N takes {100.0 * b['max']:.1f} % of thruster "
                f"{b['thruster']}'s thrust range at the wire cap {self.pwm_cap:.2f} (limit "
                f"{100.0 * HEAVE_TRIM_MAX_CAP_FRACTION:.0f} %, rl_pwm.HEAVE_TRIM_MAX_CAP_FRACTION): the cap acts on policy "
                f"+ trim, so all eight of the policy's thrusters would be scaled down whenever it drove thruster "
                f"{b['thruster']} the same way — lower the trim or fly a higher rl_pwm.pwm_cap (refused, not clipped)")
        #: review D-R1: what the trim alone sends with the policy's verticals at rest (the deadband's worst case)
        self.heave_trim_rest = heave_trim_at_rest(self.model, self.heave_trim_n, self.pwm_cap, self.pwm_cap_mode)
        #: review D-R5: HwRlPwm7d records here what the TRANSPORT block's rl_pwm.heave_trim_n says (never applied there)
        self.heave_trim_rl_pwm_block = None
        self._trim_fb = None
        self._trim_reset_counts()
        #: fixed: the vehicle's script maps RC channel 8+m to motor m (control/rl_pwm._fixed_channels)
        self.pwm_channels = tuple(PWM_CHANNELS_DEFAULT)
        self.mode = "rl_pwm"
        self.solver_kind = "rl_pwm"
        self.axis_cap = None                       # no axes leave under this controller
        self.last_pwm_us = None
        og = self.obs_guard.cfg
        log(f"rl_pwm: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), obs "
            f"{self.policy.obs_dim} (K={self.K}{', goal-frame quaternions' if self.goal_frame else ''}"
            f"{', integral-error slots' if self._integ is not None else ''}), 8 thruster "
            f"throttles, |throttle| <= {self.pwm_cap:.2f} on the wire "
            f"as a thrust-space {self.pwm_cap_mode} (trained {self.model.cap_trained:.2f}"
            f"{', BELOW trained — recorded opt-out' if self.caps_below_trained else ''}), rc channels "
            f"{list(self.pwm_channels)}, expects MOT_1..8_DIRECTION {self.model.motor_direction}; sign probe "
            f"{'required' if self.require_thruster_probe else 'NOT required (recorded opt-out)'}; obs guard: "
            f"v slew {og['vel_slew_mps2']:g} m/s^2, fix jump {og['jump_m']:g} m under {og['min_tags']} tags, HOLD "
            f"after {og['hold_after_s']:g} s of < {og['min_tags']} tags / age > {og['age_s']:g} s = Z "
            f"{og['hold_wrench_z_n']:+.1f} N NED -> pulses {list(self.hold_pwm_us)} us; transport "
            f"{self.model.transport} — UNTESTED on the vehicle [예측]")
        _trim_line = self._heave_trim_log_line()
        if _trim_line:
            log(_trim_line)
        _int_line = self._obs_integral_log_line()        # 2026-10-18: "" without obs_integral (the log of before)
        if _int_line:
            log(_int_line)

    def _heave_trim_cfg(self, cfg) -> tuple:
        """(heave_trim_n, the config key) for THIS LOW, from its OWN block (2026-10-18): rl_pwm.heave_trim_n here,
        rl_pwm_7d.heave_trim_n for HwRlPwm7d (CFG_KEY) — like allow_caps_below_trained, the decision is per LOW; absent =
        0.0 = off. Validated by rl_pwm.parse_heave_trim_n (a number, finite, |v| <= 10 N; anything else REFUSES the
        build)."""
        from .rl_pwm import parse_heave_trim_n
        raw = dict(getattr(cfg, self.CFG_KEY, None) or {})
        key = f"{self.CFG_KEY}.heave_trim_n"
        return parse_heave_trim_n(raw.get("heave_trim_n", 0.0), key), key

    def _heave_trim_log_line(self) -> str:
        """The one mission-log line of a non-zero heave trim at build ("" when the trim is off: the build log of
        before)."""
        if self.heave_trim_n == 0.0:
            return ""
        lv = self.heave_trim_thrust_level
        b, rest = self.heave_trim_budget, self.heave_trim_rest
        w = rest["wrench_ned"]
        side = "up" if self.heave_trim_n > 0.0 else "down"
        ep = rest["edge_pattern"][side]
        we = ep["wrench_ned"]
        return (f"{self.CFG_KEY}: heave trim {self.heave_trim_n:+.2f} N world-up ({self.heave_trim_key}; export nominal "
                f"thrust model, T200 x voltage_scale {self.model.voltage_scale:g}; + = up, unlike the NED "
                f"hold_wrench_z_n) = thrusters 5..8 {[round(float(v), 3) for v in lv[4:8]]} N at level, rotated by the "
                f"estimated roll/pitch every tick; added in thrust space BEFORE the {self.pwm_cap_mode} cap (applied to "
                f"the TOTAL; the trim takes {100.0 * b['max']:.1f} % of thruster {b['thruster']}'s capped range, limit "
                f"{100.0 * b['limit']:.0f} %), left out of the policy's previous action, NOT added to the obs-guard HOLD "
                f"pulses. A zero-moment heave only where every vertical total clears its deadband edge [유도 nominal "
                f"model]: with the policy's verticals at rest this trim leaves as pulses 5..8 {rest['pulses_us']} us = "
                f"Z {w[2]:+.2f} N NED, K {w[3]:+.3f} / M {w[4]:+.3f} N*m FRD (asked Z {rest['requested_z_ned']:+.2f}, "
                f"no moment); every trim below {rest['all_edge_below_n'][side] or 0.0:.2f} N {side} leaves as the one "
                f"edge pattern {ep['pulses_us']} us = Z {we[2]:+.2f} N, K {we[3]:+.3f} / M {we[4]:+.3f} N*m whatever "
                f"its value. CSV pwm1..8 / uX..uN are policy + trim (no policy-part column) — never flown [예측]")

    def _trim_reset_counts(self) -> None:
        self._trim_counts = {"ticks_trimmed": 0, "ticks_scaled": 0, "ticks_scaled_by_trim": 0,
                             "ticks_scale_tightened": 0, "ticks_edge": 0, "ticks_hold_untrimmed": 0}
        self._trim_last_thrust = None
        #: per-engagement sums of what the trim delivered (review D-R1 / D-R3: the CSV cannot say it)
        self._trim_sums = {"n": 0, "wrench_ned": [0.0] * 6, "z_requested": 0.0, "scale": 0.0, "scale_min": None}

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
        from .rl_pwm import (CAP_TOL, GUARD_HOLD, GUARD_NONE, NEUTRAL_PULSES, heave_trim_thrust, scale_throttles,
                             throttles_to_pwm, trim_and_cap)
        t0 = time.perf_counter()
        scale = 1.0
        guard = GUARD_NONE
        trimmed = self.heave_trim_n != 0.0
        self._trim_fb = None
        trim_thr = thr_pol = trim_det = trim_w = None
        trim_z_req = 0.0
        integ_on = self._integ is not None              # fix round 2026-10-18: obs_integral exports only
        if integ_on:
            self._int_tick = None
        try:
            eta_g, nu_g, guard = self.obs_guard.apply(eta_ned, nu_ned, self._fix_n_tags, self._fix_age_s)
            obs = self.observation(eta_g, nu_g, t)
            if not np.all(np.isfinite(obs)):
                raise ValueError("non-finite observation")
            a = self.policy.act(obs)
            thr = self.model.throttles(a)                        # action -> throttle, clipped to the TRAINED cap
            if trimmed and guard != GUARD_HOLD:
                # 2026-10-18 heave trim: world-up rotated by the estimated roll/pitch onto the four verticals (the HOLD's
                # allocation), added in THRUST space; the cap (scale: one scalar) is applied to the TOTAL, and the
                # policy's own part after the same scale is what it will be fed back (note_applied_pwm)
                trim_thr = heave_trim_thrust(self.model, self.heave_trim_n, float(eta_g[3]), float(eta_g[4]))
                thr, thr_pol, scale, trim_det = trim_and_cap(thr, trim_thr, self.pwm_cap, self.model,
                                                             self.pwm_cap_mode, return_detail=True)
                if float(np.max(np.abs(thr))) > self.pwm_cap + CAP_TOL:
                    raise ValueError("trimmed throttles exceed the cap")
                # the pure heave asked for (NED body Z; zero moment) — what was delivered is judged against it below
                trim_z_req = -self.heave_trim_n * math.cos(float(eta_g[3])) * math.cos(float(eta_g[4]))
            elif self.pwm_cap_mode == "scale":
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
            if thr_pol is not None:
                # what LEFT (with the trim) and the policy's own part on the same 1-us grid the feedback reads
                self._trim_fb = (tuple(int(p) for p in pwm), throttles_to_pwm(thr_pol, 1.0))
                # review D-R1: the wrench the trim actually added on top of the policy's own part (nominal model of the
                # throttles, the uX..uN convention) — the edge quantization and the cap scale included
                trim_w = u - self.model.wrench_ned_from_throttles(thr_pol)
        except Exception:                                        # noqa: BLE001
            self.n_fail += 1
            thr = np.zeros(8)
            pwm = tuple(NEUTRAL_PULSES)
            u = np.zeros(6)
            status = 1
            scale = float("nan")
            self._trim_fb = None
            trim_thr = thr_pol = trim_det = trim_w = None
        self.last_pwm_scale = scale
        ms = 1e3 * (time.perf_counter() - t0)
        info = {"w_hat": self.w_hat, "solve_ms": ms, "status": status, "n_fail": self.n_fail, "nis": 0.0,
                "pwm_us": pwm, "throttle": [float(v) for v in thr], "pwm_scale": scale,
                "pwm_cap_mode": self.pwm_cap_mode, "pwm_guard": int(guard),
                "guard_why": str(self.obs_guard.last.get("why", ""))}
        if integ_on:
            # fix round 2026-10-18 (review F1): the NEXT integral update masks its add when what leaves this tick is not
            # the policy's own command — the HOLD pulses or a failed tick's neutral (note_applied_pwm(None) adds the
            # worker's "nothing sent"); (review F2) what the network saw this tick, for a CSV column the worker lacks
            self._int_mask_next = bool(status != 0 or guard == GUARD_HOLD)
            it = self._int_tick
            info["obs_int_slots"] = None if it is None else [float(v) for v in it[0]]
            info["obs_int_add_masked"] = None if it is None else bool(it[1])
        n_edge = None
        if trimmed:
            # 2026-10-18: only with a trim configured (trim 0 = the info of before); None = not applied this tick
            # (tag-gate HOLD, or a failed tick)
            if status == 0:
                if trim_thr is not None:
                    c = self._trim_counts
                    c["ticks_trimmed"] += 1
                    s_pol = float(trim_det["s_policy"])
                    n_edge = int(np.count_nonzero(trim_det["edge"]))
                    if scale < 1.0:
                        c["ticks_scaled"] += 1
                        if s_pol >= 1.0:
                            c["ticks_scaled_by_trim"] += 1        # the policy alone would not have been scaled
                    if scale < s_pol:
                        c["ticks_scale_tightened"] += 1           # the trim made the scale bind harder
                    if n_edge:
                        c["ticks_edge"] += 1                      # a vertical total sent AT its deadband edge
                    self._trim_last_thrust = [float(v) for v in trim_thr]
                    sm = self._trim_sums
                    sm["n"] += 1
                    sm["wrench_ned"] = [acc + float(w) for acc, w in zip(sm["wrench_ned"], trim_w)]
                    sm["z_requested"] += float(trim_z_req)
                    sm["scale"] += float(scale)
                    sm["scale_min"] = float(scale) if sm["scale_min"] is None else min(sm["scale_min"], float(scale))
                elif guard == GUARD_HOLD:
                    self._trim_counts["ticks_hold_untrimmed"] += 1
            info["heave_trim_n"] = float(self.heave_trim_n)
            info["trim_thrust_n"] = None if trim_thr is None else [float(v) for v in trim_thr]
            info["throttle_policy"] = None if thr_pol is None else [float(v) for v in thr_pol]
            # review D-R1 / D-R2: what the trim delivered this tick (nominal model), how many verticals left at their
            # deadband edge, and the scale the policy alone would have had
            info["trim_wrench_ned"] = None if trim_w is None else [float(v) for v in trim_w]
            info["trim_n_edge"] = n_edge
            info["pwm_scale_policy"] = None if trim_det is None else float(trim_det["s_policy"])
        return u, info

    def note_applied(self, tau_ned_est) -> None:
        """The axes path's feedback does not apply: nothing this controller asked for left as an axis (the worker sends
        a neutral MANUAL_CONTROL). The previous action comes from ``note_applied_pwm``."""
        return None

    def note_applied_pwm(self, pwm_us) -> None:
        """The 8 pulses that actually LEFT become the policy's previous action (None = nothing left: neutral).

        2026-10-18 (heave trim): when the pulses reported are exactly the ones this tick's step produced WITH the trim,
        the policy is fed back its OWN part of them instead — its command after the same cap scale (rl_pwm.trim_and_cap
        thr_policy_part) on the same 1-us grid — so the trim never enters its previous action. Any other report (None,
        neutral, foreign pulses) is fed back as it always was; with no trim configured nothing here changes."""
        from .rl_pwm import NEUTRAL_PULSES, pwm_to_action
        if pwm_us is None and getattr(self, "_integ", None) is not None:
            # fix round 2026-10-18 (review F1): nothing left (coast, pwm-gate grace, disengage) — the next integral
            # update masks its add (obs_integral exports only)
            self._int_mask_next = True
        pw = NEUTRAL_PULSES if pwm_us is None else pwm_us
        self.last_pwm_us = tuple(int(p) for p in pw)
        fb = self._trim_fb
        if fb is not None and pwm_us is not None and self.last_pwm_us == fb[0]:
            pw = fb[1]
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
        # 2026-10-18: no trim feedback carried across an engagement; the per-engagement trim counters
        self._trim_fb = None
        if hasattr(self, "heave_trim_n"):
            self._trim_reset_counts()

    def _heave_trim_meta(self) -> dict:
        """meta controller.heave_trim (2026-10-18): the key, the level allocation, what the build line said, and how
        often the trim was applied / scaled / skipped under the HOLD this engagement."""
        from .rl_pwm import FLU_TO_FRD, HEAVE_TRIM_MAX_N
        lv = np.asarray(self.heave_trim_thrust_level, float)
        w_level = (self.model.B_flu @ lv) * FLU_TO_FRD
        vs = float(self.model.voltage_scale) if float(self.model.voltage_scale) > 0.0 else 1.0
        sm = self._trim_sums
        n = int(sm["n"])
        delivered = {"n_ticks": n,
                     "mean_wrench_ned": (None if n == 0 else [float(v) / n for v in sm["wrench_ned"]]),
                     "mean_requested_z_ned": (None if n == 0 else float(sm["z_requested"]) / n),
                     "scale_mean": (None if n == 0 else float(sm["scale"]) / n), "scale_min": sm["scale_min"],
                     "note": ("over this engagement's trimmed ticks: the wrench the trim added on top of the policy's "
                              "own part (nominal model of the throttles, the uX..uN convention; the edge quantization "
                              "and the cap scale included) against the pure heave asked for (-trim cos roll cos pitch, "
                              "zero moment) — a model number, NOT a measurement")}
        return {"key": self.heave_trim_key, "heave_trim_n": float(self.heave_trim_n),
                "applied": bool(self.heave_trim_n != 0.0), "max_abs_n": float(HEAVE_TRIM_MAX_N),
                "thrust_level_n": [float(v) for v in lv],
                "thrust_level_curve": [float(v) for v in lv / vs],
                "wrench_ned_level": [float(v) for v in w_level],
                "cap_budget": dict(self.heave_trim_budget),
                "at_rest": dict(self.heave_trim_rest),
                "counts": dict(self._trim_counts), "last_thrust_n": self._trim_last_thrust,
                "delivered": delivered,
                "rl_pwm_block": (None if self.heave_trim_rl_pwm_block is None else dict(self.heave_trim_rl_pwm_block)),
                "csv_note": ("the CSV (workers.py schema <= 23) has NO column for the policy's own part, pwm_scale or the "
                             "trim: on a run with heave_trim_n != 0, pwm1..8 (the vehicle's SERVO_OUTPUT_RAW) and uX..uN "
                             "are policy + trim — never read them as the policy's actions and never pool the run with "
                             "untrimmed ones; this block's counts / delivered are the only per-engagement record"),
                "note": ("physical N, + = UP in the world (obs_guard.hold_wrench_z_n is NED: the same thrust model, the "
                         "opposite sign), the export's nominal thrust model (T200 x voltage_scale); per tick world-up is "
                         "rotated into the body frame by the ESTIMATED roll/pitch and allocated minimum-norm over "
                         "thrusters 5..8 (rl_pwm.heave_trim_thrust, the HOLD's allocation — a zero-moment heave REQUEST, "
                         "the tilted part of world-up dropped), added in thrust space, then the cap on the TOTAL "
                         "(rl_pwm.trim_and_cap); the delivered wrench is that pure heave only where every vertical total "
                         "clears its deadband edge — a total inside the gap leaves AT the edge (counts.ticks_edge; "
                         "at_rest = the pattern with the policy's verticals at rest); a trim above cap_budget.limit of "
                         "one vertical's capped range is refused at build; prev_action = the policy's own part after "
                         "the same scale; the HOLD pulses are never trimmed. thrust_level_* / wrench_ned_level = the "
                         "request at level attitude. 0 = off (the code path of before). NEVER FLOWN [예측]")}

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
                  # 2026-10-16 (R7): the export's obs_guard block (None = it has none)
                  "obs_guard_trained": self.obs_guard_trained,
                  "hold_pwm_us": [int(v) for v in self.hold_pwm_us],
                  "hold_throttles": [float(v) for v in self.hold_throttles],
                  "hold_wrench_ned_est": [float(v) for v in self.model.wrench_ned_from_throttles(self.hold_throttles)],
                  # 2026-10-18 (RECORD BOUNDARY: never pool across heave_trim_n)
                  "heave_trim_n": float(self.heave_trim_n),
                  "heave_trim": self._heave_trim_meta(),
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

#: The frame an rl_7d export must declare, element for element (RL_controller rov_rl/obs_spec.frame_layout(True, 6)) —
#: or, since 2026-10-15, the same with the q_*_goalframe slots together with obs_goal_frame: true (check_frame_layout).
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
        # RL7D_FRAME_LAYOUT (world-frame quaternions) or its goal-frame twin with obs_goal_frame true (2026-10-15)
        check_frame_layout(spec, "rl_7d", pdir, goal_velocity=True, n_actions=6)
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
        log(f"rl_7d: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), axes6"
            f"{', goal-frame quaternions' if self.goal_frame else ''}, "
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
#: The frame an rl_pwm_7d export must declare (RL_controller rov_rl/obs_spec.frame_layout(True, 8)) — or its goal-frame
#: twin with obs_goal_frame: true (2026-10-15, pwm11att7 and later; check_frame_layout).
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
        # RLPWM7D_FRAME_LAYOUT (world-frame quaternions) or its goal-frame twin with obs_goal_frame true (2026-10-15)
        check_frame_layout(spec, "rl_pwm_7d", pdir, goal_velocity=True, n_actions=8)
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
        log(f"rl_pwm_7d: policy {os.path.relpath(self.policy_dir, REPO_ROOT)} ({self.spec.get('checkpoint', '?')}), pwm"
            f"{', goal-frame quaternions' if self.goal_frame else ''}"
            f"{', integral-error slots' if self._integ is not None else ''}, "
            f"trained roll/pitch envelope +-{math.degrees(self.att_roll_max):.1f} / +-{math.degrees(self.att_pitch_max):.1f} "
            f"deg, |throttle| <= {self.pwm_cap:.2f} (transport rl_pwm.pwm_cap; trained {self.model.cap_trained:.2f}) — "
            f"roll/pitch as PULSES, never s/t; UNTESTED on the vehicle [예측]")
        _trim_line = self._heave_trim_log_line()          # its OWN rl_pwm_7d.heave_trim_n (HwRlPwm._heave_trim_cfg)
        if _trim_line:
            log(_trim_line)
        # review D-R5: rl_pwm.pwm_cap and rl_pwm.obs_guard ARE this LOW's (the transport block), rl_pwm.heave_trim_n is
        # NOT — an operator who set it there and flies RL_PWM_7D is told at build, and the meta says it (no behaviour
        # change: this LOW flies its own rl_pwm_7d.heave_trim_n only)
        self.heave_trim_rl_pwm_block, _note = self._rl_pwm_block_trim_note(cfg)
        if _note:
            log(_note)
        _int_line = self._obs_integral_log_line()        # 2026-10-18: "" without obs_integral (the log of before)
        if _int_line:
            log(_int_line)

    def _rl_pwm_block_trim_note(self, cfg) -> tuple:
        """(meta dict | None, build line | "") for the TRANSPORT block's rl_pwm.heave_trim_n as seen from this LOW
        (2026-10-18, review D-R5): None / "" when it is absent; a line only when it is set, non-zero and differs from
        this LOW's own rl_pwm_7d.heave_trim_n. Never applied here; an invalid value is reported, not raised (the rl_pwm
        LOW refuses it at its own build)."""
        from .rl_pwm import parse_heave_trim_n
        raw = dict(getattr(cfg, "rl_pwm", None) or {})
        if "heave_trim_n" not in raw:
            return None, ""
        v = raw.get("heave_trim_n")
        try:
            f = parse_heave_trim_n(v, "rl_pwm.heave_trim_n")
        except ValueError:
            f = None
        noted = f is None or (f != 0.0 and f != float(self.heave_trim_n))
        meta = {"key": "rl_pwm.heave_trim_n", "value": f, "valid": f is not None, "applied_here": False,
                "noted": bool(noted)}
        if f is None:
            meta["raw"] = repr(v)
        if not noted:
            return meta, ""
        what = f"{v!r} (not a valid trim — the rl_pwm LOW refuses it)" if f is None else f"{f:+.2f} N"
        return meta, (f"{self.CFG_KEY}: NOTE rl_pwm.heave_trim_n is {what} but this LOW flies ONLY its own "
                      f"{self.heave_trim_key} ({self.heave_trim_n:+.2f} N): the transport block's trim belongs to the "
                      f"rl_pwm LOW and is NOT applied here (unlike rl_pwm.pwm_cap / rl_pwm.obs_guard, which are) — set "
                      f"{self.heave_trim_key} to fly a trim on RL_PWM_7D")

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
