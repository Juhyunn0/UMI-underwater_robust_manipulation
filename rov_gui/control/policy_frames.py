#!/usr/bin/env python3
"""policy_frames.py — the frame algebra between a UMI diffusion policy and the
station's NED plan stream (spec DP_LIVE_POLICY_SPEC_V2 §B, agent A).

WHAT THE POLICY SPEAKS. Everything the trained policy sees and emits is
RELATIVE to its own tool-centre-point (TCP) pose. Two action encodings exist
(``state.ACTION_REPR_*``, chosen by the checkpoint and told apart by width):

  * ``pos_yaw_width`` (16, 5) -- the 2026-09-07 retrain:
    ``[dx, dy, dz, dyaw, width]``. ``dp`` is the translation of the relative
    TCP transform in the CURRENT (tilted) camera frame, ``dyaw`` the ZYX yaw a
    LEVEL vehicle carrying the camera on the C3 mount would turn to reproduce
    the relative rotation, ``yaw_of(R_bt @ R_rel @ R_bt^T)`` with ``R_bt`` the
    mount constant ``R_BT_C3`` (NED sign, + = right turn). Transcribed from
    ``umi/common/yaw_action.py`` (parity-tested by path). Knot k decodes as
    ``p_tcp_k = p_a + R_ned_tcp_a @ dp_k``, ``yaw_k = yaw_anchor + dyaw_k``.
  * ``pos_rpy_width`` (16, 7) -- the 6-DoF variant (2026-09-26):
    ``[dx, dy, dz, dyaw, droll, dpitch, width]``; columns 0:4 and 6 are
    BIT-IDENTICAL to ``pos_yaw_width``'s 0:4 and 4, and ``(droll, dpitch,
    dyaw)`` are the ZYX Euler angles of the SAME matrix ``R_bt @ R_rel @
    R_bt^T`` (``encode_pos_rpy``, transcribed from ``yaw_action.py``). Whether
    the attitude columns are FLOWN is the caller's ``track_rp`` (config
    policy.attitude_track, default off = "dropped-and-logged": 5-dim
    arithmetic on cols 0:4, the requested attitude only recorded).
  * ``pose10d`` (16, 10) -- legacy (the 2026-09-01 checkpoint), in the
    ``umi.common.pose_util`` encoding ``[pos(3), rot6d(6), width(1)]`` where
    rot6d is the first TWO ROWS of the rotation matrix (Gram-Schmidt back to a
    matrix on decode): ``T_k = T_latest @ pose10d_to_mat(a_k)``. Kept so old
    plans.jsonl still recompose; never flown (state.POLICY_ACTION_REPR).

The observation is two rows of proprio (66.7 ms apart, the last row expressed
relative to itself, so it is identity/zero; rotation still as rot6d) and the
action is 16 knots with knot 0 at the obs time.

WHAT THE STATION SPEAKS. ``MpcWorker`` holds ``eta = [x, y, z, roll, pitch,
yaw]`` in the engage-datum NED frame with ``rot_zyx(roll, pitch, yaw)`` =
R_ned_body (body FRD), and consumes ``plan_stream.PlanMsg`` — positions and
YAW only; roll/pitch are what the NMPC levels. This module is the bridge, and
every decision below exists because a naive bridge was wrong in review:

  * TCP-space composition (spec A2). Compose the action in the TCP frame,
    ``T_ned_tcp_k = (T_ned_body_a @ T_bt) @ pose10d_to_mat(a_k)`` (legacy) or
    ``p_tcp_k = p_a + R_ned_tcp_a @ dp_k`` (5-dim), take the implied body
    rotation's ZYX yaw (5-dim: ``yaw_anchor + dyaw_k`` -- exact, because
    ``rot_zyx(r, p, y + d) = Rz(d) @ rot_zyx(r, p, y)`` turns every body-fixed
    vector's azimuth by exactly d for ANY constant roll/pitch), and then LEVEL
    the body position:
    ``p_body_k = p_tcp_k - rot_zyx(roll_lvl, pitch_lvl, yaw_k) @ t_bt`` with
    ``(roll_lvl, pitch_lvl)`` the anchor's measured attitude (what the NMPC
    will hold, ≈ level). Dropping roll/pitch AFTER composing in body space
    would have moved the jaw by the lever arm ``|t_bt|·sin(pitch)`` every
    time the policy pitched its camera — the jaw, not the hull, is what the
    policy positions. The dropped roll/pitch magnitude is reported so the
    log shows how much attitude the policy asked for and did not get.
  * Body-z composition for the tracked 7-dim branch (D12, 2026-09-26).
    ``R_body_k = R_body_a @ rot_zyx(droll_k, dpitch_k, dyaw_k)`` composes
    the relative attitude about the ANCHOR's body axes (what the label
    measured: the level-mount vehicle's own axes), then ``(roll_k, pitch_k,
    yaw_k) = rpy_of(R_body_k)``. At a level anchor this is exactly ``roll =
    droll, pitch = dpitch, yaw = anchor_yaw + dyaw`` (the 5-dim decode); at
    a tilted anchor it differs from the 5-dim WORLD-z composition by
    O(rp * dyaw) (~6.6e-4 rad at 5 deg / 0.3 rad, test (c)) -- which is why
    it is a THIRD branch and the two existing branches are untouched. The
    hull attitude is then clipped per axis to +-rp_max_rad (D3 tier T1,
    recorded as rp_clipped_deg / rp_clipped_n) and ``p_body_k = p_tcp_k -
    R_body_k @ t_bt`` with the CLIPPED attitude: the jaw stays where the
    policy put it, the hull carries the attitude it is allowed.
  * The TCP is the ROV's OWN jaw (spec A3), never the handheld's camera
    offset: ``tcp_offset_from_body`` turns a body-frame jaw position into the
    camera-frame offset ``T_body_tcp`` wants, so the config can carry the jaw
    in body coordinates (= cam_t_flu + lens->grip [0.196, 0, -0.275]: x
    measured 2026-09-08, z the CAD vertical [유도] — hw_mpc.yaml
    ``policy.tcp_body_flu_m``) and the handheld number stays a comparison
    value in the meta, not a live input. The jaw-vs-handheld mismatch this
    implies (jaw - handheld fingertips = [-0.036, -0.062, +0.012] m in the
    camera frame, i.e. ~5 cm past / ~4 cm above / ~3.6 cm left of the object
    in body FLU [유도]) is NOT compensated; the note beside that key says why.
  * Leashed anchor (spec A4). Anchoring every replan at the MEASURED pose
    resets the tracking error each period (the plan doc's B1 failure); at
    the REFERENCE it lets the reference run away from a vehicle that cannot
    follow. ``leash_anchor`` starts at the measurement and moves toward the
    reference by at most ``leash_m`` / ``leash_yaw``.
  * Resample to a 0.2 s knot grid (spec A5). The filter's finite-difference
    acceleration gate at dt = 1/15 s amplifies 1.5 mm of knot jitter into a
    rejection (gain 551·σ [유도: control review]); windowed means on a
    0.2 s grid cut that to 61·σ. Knot 0 is PINNED to raw knot 0 so the plan
    still starts exactly where the policy said "now" is.
  * Proprio history keyed on the FIX (spec A6). The station's eta is a 20 Hz
    staircase of ~10 Hz tag fixes; rows appended per tick would quantise the
    66.7 ms motion cue to {0, 1, 2} fixes. ``EtaHistory`` takes one row per
    fix stamp and tells the caller when the two obs rows collapse onto one
    fix (degenerate — skip inference rather than feed a zero motion cue).
  * Gripper width is open-loop (spec A12): the jaw has no feedback, so the
    width the policy is told is an integrator of the drive levels the jaw
    was sent, clamped to the dataset's open/closed levels.

Pure numpy + stdlib. Imports only ``.plan_stream.PlanMsg`` and
``.state_assembler.rot_zyx`` so it loads (and is testable) without Qt, cv2,
torch or acados — the same discipline as ``plan_stream.py``.

Conventions: metres, radians, seconds; datum NED for positions; body FRD for
``t_bt``; camera optical frame (x right, y down, z forward) for the TCP offset.
All angle outputs are wrapped to (-pi, pi] unless the docstring says
"unwrapped".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .plan_stream import PlanMsg
from .state_assembler import rot_zyx

from ..state import (ACTION_REPR_BY_DIM, ACTION_REPR_POSE10D,
                     ACTION_REPR_POS_YAW_WIDTH)

# The 6-DoF vocabulary (state.py, 2026-09-26); the getattr fallback lets this
# module import against an older state.py (then no (K, 7) action decodes).
try:
    from ..state import ACTION_REPR_POS_RPY_WIDTH
except ImportError:                                   # pragma: no cover
    ACTION_REPR_POS_RPY_WIDTH = "pos_rpy_width"

__all__ = [
    "rot6d_to_mat", "mat_to_rot6d", "pose10d_to_mat", "mat_to_pose10d",
    "YAW_AXIS_CAM_TILT_DEG", "R_BT_C3", "rot_z", "rot_x", "rot_y",
    "encode_pos_yaw", "decode_pos_yaw", "encode_pos_rpy", "decode_pos_rpy",
    "rpy_of_batch", "action_repr_of", "DYAW_MAX_RAD", "DRP_MAX_RAD",
    "T_from_eta", "yaw_of", "rpy_of", "T_body_tcp", "tcp_offset_from_body",
    "EtaHistory", "lowdim_obs", "resample_knots", "resample_knots_att",
    "compose_plan", "leash_anchor", "guard_yaw", "filter_plan_yaw",
    "GripperWidthEstimator", "ROT6D_MIN_NORM", "PITCH_SINGULAR_RAD",
]

#: rot6d columns whose either 3-vector is shorter than this are treated as a
#: degenerate action (spec A23: reject with a reason, never disengage).
ROT6D_MIN_NORM = 1e-6

#: A TRACKED 7-dim plan whose ABSOLUTE pitch (anchor pitch composed with the
#: relative dpitch) reaches this is rejected before its yaw is used:
#: :func:`rpy_of` returns ``atan2(0, 0)`` yaw at |pitch| = 90 deg and the
#: garbage would enter ``p_body`` and ``PlanMsg.yaw`` (the T1 clip repairs
#: roll/pitch only). 85 deg [결정]: far beyond any rp_reject (30 deg) yet
#: short of where cos(pitch) loses the yaw (2026-09-26 review).
PITCH_SINGULAR_RAD = math.radians(85.0)

#: A 5-dim knot whose |dyaw| exceeds this is a schema fault (A23: reject with a
#: reason). A range-normalised clip_sample network cannot emit it -- the real
#: cap is the checkpoint's normaliser input_stats (contract['action_range']).
DYAW_MAX_RAD = math.pi

#: A 7-dim knot whose |droll| or |dpitch| exceeds this is a schema fault (A23:
#: reject with a reason). pi/2 is the ZYX pitch singularity and far beyond
#: any relative attitude one 16-knot chunk can mean; the dataset build asserts
#: |dpitch| < 60 deg [예측] and the flown cap is policy.rp_max_deg.
DRP_MAX_RAD = math.pi / 2.0

#: The obs contract's key order (spec v1 §0). Kept here so the worker and the
#: tests spell the keys the same way.
LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_rot_axis_angle",
               "robot0_gripper_width", "robot0_eef_rot_axis_angle_wrt_start")


def _wrap(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(math.atan2(math.sin(a), math.cos(a)))


# =============================================================================
# rot6d / pose10d — umi.common.pose_util semantics, transcribed
# =============================================================================
def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return (vec.T / norm).T


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3), ``umi.common.pose_util.rot6d_to_mat`` verbatim.

    ROWS convention: ``b1 = normalize(a1)``, ``b2 = normalize(a2 - <b1,a2> b1)``,
    ``b3 = b1 x b2`` and ``out[..., i, :] = b_i`` (``np.stack(..., axis=-2)``).
    The parity test imports the upstream module by path and compares.
    """
    d6 = np.asarray(d6, dtype=float)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two ROWS, flattened (upstream)."""
    mat = np.asarray(mat, dtype=float)
    batch_dim = mat.shape[:-2]
    return mat[..., :2, :].copy().reshape(batch_dim + (6,))


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """(..., 4, 4) -> (..., 9) = [pos(3), rot6d(6)] (upstream; the width is
    the caller's 10th number)."""
    mat = np.asarray(mat, dtype=float)
    pos = mat[..., :3, 3]
    d6 = mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([pos, d6], axis=-1)


def pose10d_to_mat(d10: np.ndarray) -> np.ndarray:
    """(..., >=9) -> (..., 4, 4): pos from [:3], rot6d from [3:9]. A trailing
    width column (the 10th action number) is ignored, as upstream's
    ``d10[..., 3:]`` would NOT be — upstream slices the action to 9 columns
    before calling this; we accept both so the caller need not."""
    d10 = np.asarray(d10, dtype=float)
    pos = d10[..., :3]
    d6 = d10[..., 3:9]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=float)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


# =============================================================================
# pos_yaw_width -- umi.common.yaw_action semantics, transcribed (2026-09-07)
# =============================================================================
#: The mount tilt the label's yaw axis is defined for [deg down]. The
#: checkpoint records it (shape_meta.action.yaw_axis_cam_tilt_deg) and the
#: backend refuses to arm when its NavConfig disagrees.
YAW_AXIS_CAM_TILT_DEG = 43.3

#: body(FRD) <- camera(optical) rotation of the C3 on its 43.3 deg pool mount,
#: == NavConfig(cam_tilt_deg=43.3).R_t_frd_cam("main")[0] at full precision
#: (config/hw_nav.yaml:187 + geometry._C3_XYAXES; optical axis 42.98 deg below
#: body x). VERBATIM of umi/common/yaw_action.py:R_BT_C3 -- the training label
#: is defined with this constant, so it must not follow a config value.
R_BT_C3 = np.array([
    [0.0, -0.6817320540699252, 0.7316019453593604],
    [1.0, 0.0, 0.0],
    [0.0, 0.7316019453593604, 0.6817320540699253],
], dtype=np.float64)


def rot_z(psi) -> np.ndarray:
    """Rz(psi) as (..., 3, 3) float64 (upstream ``yaw_action.rot_z``)."""
    psi = np.asarray(psi, dtype=np.float64)
    c, s = np.cos(psi), np.sin(psi)
    out = np.zeros(psi.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def encode_pos_yaw(T_rel, R_bt=R_BT_C3) -> Tuple[np.ndarray, np.ndarray]:
    """(..., 4, 4) relative TCP transform(s) -> (dp (..., 3), dyaw (...,)).

    ``dp = T_rel[..., :3, 3]`` (same dtype, bit-identical to the legacy
    ``mat_to_pose10d`` position columns); ``dyaw = yaw_of(R_bt @ R_rel @
    R_bt^T)`` float64 in (-pi, pi]. VERBATIM of ``yaw_action.encode_pos_yaw``.
    """
    T_rel = np.asarray(T_rel)
    dp = T_rel[..., :3, 3]
    R_rel = np.asarray(T_rel[..., :3, :3], dtype=np.float64)
    R_bt = np.asarray(R_bt, dtype=np.float64)
    M = R_bt @ R_rel @ R_bt.T
    dyaw = np.arctan2(M[..., 1, 0], M[..., 0, 0])
    return dp, dyaw


def decode_pos_yaw(dp, dyaw, R_bt=R_BT_C3) -> np.ndarray:
    """Inverse of :func:`encode_pos_yaw` for a yaw-only rotation: the (..., 4, 4)
    relative transform with translation ``dp`` and rotation ``R_bt^T @ Rz(dyaw)
    @ R_bt``. VERBATIM of ``yaw_action.decode_pos_yaw``."""
    dp = np.asarray(dp, dtype=np.float64)
    dyaw = np.asarray(dyaw, dtype=np.float64)
    R_bt = np.asarray(R_bt, dtype=np.float64)
    R_rel = R_bt.T @ rot_z(dyaw) @ R_bt
    out = np.zeros(dyaw.shape + (4, 4), dtype=np.float64)
    out[..., :3, :3] = R_rel
    out[..., :3, 3] = dp
    out[..., 3, 3] = 1.0
    return out


def rot_x(phi) -> np.ndarray:
    """Rx(phi) as (..., 3, 3) float64 (upstream ``yaw_action.rot_x``)."""
    phi = np.asarray(phi, dtype=np.float64)
    c, s = np.cos(phi), np.sin(phi)
    out = np.zeros(phi.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = c
    out[..., 1, 2] = -s
    out[..., 2, 1] = s
    out[..., 2, 2] = c
    return out


def rot_y(theta) -> np.ndarray:
    """Ry(theta) as (..., 3, 3) float64 (upstream ``yaw_action.rot_y``)."""
    theta = np.asarray(theta, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(theta.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def rpy_of_batch(M) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched ZYX Euler ``(roll, pitch, yaw)`` of (..., 3, 3) rotations,
    the inverse of ``rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)`` (=
    ``rot_zyx``) away from |pitch| = 90 deg: ``yaw = arctan2(M[1,0],
    M[0,0])`` (bit-identical to :func:`encode_pos_yaw`'s dyaw on the same
    M), ``pitch = arcsin(clip(-M[2,0], -1, 1))``, ``roll = arctan2(M[2,1],
    M[2,2])``. VERBATIM of upstream ``yaw_action.rpy_of`` (named _batch here
    because the scalar :func:`rpy_of` predates it)."""
    M = np.asarray(M)
    yaw = np.arctan2(M[..., 1, 0], M[..., 0, 0])
    pitch = np.arcsin(np.clip(-M[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(M[..., 2, 1], M[..., 2, 2])
    return roll, pitch, yaw


def encode_pos_rpy(T_rel, R_bt=R_BT_C3
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(..., 4, 4) relative TCP transform(s) -> ``(dp, dyaw, droll, dpitch)``.

    ``dp`` as in :func:`encode_pos_yaw` (same dtype and bits); ``(droll,
    dpitch, dyaw) = rpy_of(R_bt @ R_rel @ R_bt^T)`` -- the ZYX Euler angles
    of the SAME matrix whose yaw the 5-dim label takes, so ``dyaw`` is
    bit-identical to :func:`encode_pos_yaw` and the 5-dim label is a column
    subset of the 7-dim one. Body FRD signs: +dyaw = CW from above, +dpitch
    = nose-up, +droll = starboard-down. VERBATIM of
    ``yaw_action.encode_pos_rpy``.
    """
    T_rel = np.asarray(T_rel)
    dp = T_rel[..., :3, 3]
    R_rel = np.asarray(T_rel[..., :3, :3], dtype=np.float64)
    R_bt = np.asarray(R_bt, dtype=np.float64)
    M = R_bt @ R_rel @ R_bt.T
    droll, dpitch, dyaw = rpy_of_batch(M)
    return dp, dyaw, droll, dpitch


def decode_pos_rpy(dp, dyaw, droll, dpitch, R_bt=R_BT_C3) -> np.ndarray:
    """Inverse of :func:`encode_pos_rpy`: the (..., 4, 4) relative transform
    with translation ``dp`` and rotation ``R_bt^T @ Rz(dyaw) @ Ry(dpitch) @
    Rx(droll) @ R_bt``. With ``droll = dpitch = 0`` it equals
    :func:`decode_pos_yaw` bit for bit (Ry(0) and Rx(0) are exact
    identities). VERBATIM of ``yaw_action.decode_pos_rpy``."""
    dp = np.asarray(dp, dtype=np.float64)
    dyaw = np.asarray(dyaw, dtype=np.float64)
    droll = np.asarray(droll, dtype=np.float64)
    dpitch = np.asarray(dpitch, dtype=np.float64)
    R_bt = np.asarray(R_bt, dtype=np.float64)
    # Parenthesised exactly as upstream: the inner ZYX product first, so
    # with Ry(0) = Rx(0) = I the outer product is evaluated in the SAME order
    # as decode_pos_yaw (R_bt.T @ Rz @ R_bt) and the bytes match.
    R_rel = R_bt.T @ (rot_z(dyaw) @ rot_y(dpitch) @ rot_x(droll)) @ R_bt
    shape = np.broadcast_shapes(dyaw.shape, droll.shape, dpitch.shape)
    out = np.zeros(shape + (4, 4), dtype=np.float64)
    out[..., :3, :3] = R_rel
    out[..., :3, 3] = dp
    out[..., 3, 3] = 1.0
    return out


def action_repr_of(width: int) -> str:
    """The action representation an array width implies (5 -> pos_yaw_width,
    7 -> pos_rpy_width, 10 -> pose10d); ``ValueError`` for any other
    width."""
    try:
        return ACTION_REPR_BY_DIM[int(width)]
    except (KeyError, TypeError, ValueError):
        raise ValueError("compose_plan: action must be (K, 5) [pos_yaw_width], "
                         "(K, 7) [pos_rpy_width] or (K, 10) [pose10d], got "
                         f"width {width!r}") from None


# =============================================================================
# eta <-> homogeneous transforms
# =============================================================================
def T_from_eta(eta6) -> np.ndarray:
    """``[x, y, z, roll, pitch, yaw]`` (datum NED, body FRD) -> (4, 4)
    T_ned_body with ``rot_zyx(roll, pitch, yaw)`` — the one rotation the
    whole station uses (state_assembler / dobmpc.fossen.rot_ib)."""
    e = np.asarray(eta6, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = rot_zyx(float(e[3]), float(e[4]), float(e[5]))
    T[:3, 3] = e[:3]
    return T


def yaw_of(R: np.ndarray) -> float:
    """ZYX yaw of a rotation (or 4x4 transform): atan2(R[1,0], R[0,0])."""
    R = np.asarray(R, dtype=float)
    return float(math.atan2(R[1, 0], R[0, 0]))


def rpy_of(R: np.ndarray) -> Tuple[float, float, float]:
    """ZYX Euler (roll, pitch, yaw) of R = Rz(yaw) Ry(pitch) Rx(roll) — the
    inverse of :func:`rot_zyx` away from the pitch = +-90 deg singularity."""
    R = np.asarray(R, dtype=float)
    pitch = float(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll = float(math.atan2(R[2, 1], R[2, 2]))
    yaw = float(math.atan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def T_body_tcp(R_bc: np.ndarray, t_bc: np.ndarray,
               tcp_offset_cam: np.ndarray) -> np.ndarray:
    """T_body_tcp = [[R_bc, t_bc], [0, 1]] @ [[I, off_cam], [0, 1]].

    ``(R_bc, t_bc)`` is the station's camera extrinsic (``x_body = R x_cam +
    t``, ``NavConfig.R_t_frd_cam``), ``tcp_offset_cam`` the TCP position in
    the camera optical frame (rotation identity — the dataset's TCP
    convention, spec v1 §0). The TCP therefore shares the camera's axes:
    +z forward along the optical axis, +y down, +x right.
    """
    R_bc = np.asarray(R_bc, dtype=float).reshape(3, 3)
    t_bc = np.asarray(t_bc, dtype=float).reshape(3)
    off = np.asarray(tcp_offset_cam, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R_bc
    T[:3, 3] = t_bc + R_bc @ off
    return T


def tcp_offset_from_body(R_bc: np.ndarray, t_bc: np.ndarray,
                         tcp_body_frd: np.ndarray) -> np.ndarray:
    """The camera-frame TCP offset that puts the TCP at ``tcp_body_frd``
    (body FRD): ``off_cam = R_bc.T @ (t_bt - t_bc)`` — the exact inverse of
    :func:`T_body_tcp`'s translation, so
    ``T_body_tcp(R_bc, t_bc, tcp_offset_from_body(R_bc, t_bc, p))[:3, 3] == p``.
    Lets the config name the ROV's own jaw in body coordinates (spec A3)."""
    R_bc = np.asarray(R_bc, dtype=float).reshape(3, 3)
    t_bc = np.asarray(t_bc, dtype=float).reshape(3)
    t_bt = np.asarray(tcp_body_frd, dtype=float).reshape(3)
    return R_bc.T @ (t_bt - t_bc)


# =============================================================================
# proprio history keyed on the fix stamp (spec A6)
# =============================================================================
class EtaHistory:
    """Time-stamped eta rows, one per FIX, with yaw unwrapped along the axis.

    ``append`` ignores a row whose stamp does not ADVANCE (the assembler
    holds a fix across ticks — the same fix must not become several rows),
    keeps only the last ``keep_s`` seconds, and stores yaw unwrapped relative
    to the previous row so interpolation across +-pi turns the short way.
    ``interp`` clamps outside the covered span (no extrapolation — a policy
    obs built from an extrapolated pose is a guess dressed as a measurement)
    and raises on an empty history. ``rows_at`` gives the two obs rows and
    the DEGENERATE flag: both rows resolve to the same single fix (clamped
    to one end, or exactly on one stamp) or the history spans less than
    ``obs_dt`` — the motion cue would be zero by construction, not by
    measurement, so the caller skips inference (spec A6).
    """

    def __init__(self, keep_s: float = 3.0):
        self.keep_s = float(keep_s)
        self._t: List[float] = []
        self._eta: List[np.ndarray] = []     # yaw UNWRAPPED

    # ------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._t)

    def clear(self) -> None:
        self._t.clear()
        self._eta.clear()

    @property
    def latest_t(self) -> Optional[float]:
        return self._t[-1] if self._t else None

    @property
    def stamps(self) -> np.ndarray:
        return np.asarray(self._t, dtype=float)

    def span(self) -> float:
        """Covered time [s]; 0 with fewer than two rows."""
        return float(self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0

    # ------------------------------------------------------------ append
    def append(self, t: float, eta6) -> bool:
        """Add a row stamped ``t``. Returns False (and stores nothing) when
        ``t`` does not advance past the last stamp — the dedup rule."""
        t = float(t)
        if not math.isfinite(t):
            return False
        if self._t and t <= self._t[-1]:
            return False
        e = np.array(np.asarray(eta6, dtype=float).reshape(6), copy=True)
        if not np.all(np.isfinite(e)):
            return False
        if self._eta:
            prev_yaw = self._eta[-1][5]
            e[5] = prev_yaw + _wrap(e[5] - prev_yaw)      # unwrap along the axis
        else:
            e[5] = _wrap(e[5])
        self._t.append(t)
        self._eta.append(e)
        cutoff = t - self.keep_s
        while len(self._t) > 2 and self._t[0] < cutoff:
            self._t.pop(0)
            self._eta.pop(0)
        return True

    # ------------------------------------------------------------ query
    def _locate(self, t: float) -> Tuple[int, int, float]:
        """(i0, i1, frac): the bracketing row indices and the interpolation
        fraction; ``i0 == i1`` when clamped to an end or exactly on a stamp."""
        if not self._t:
            raise ValueError("EtaHistory is empty")
        ts = self._t
        if t <= ts[0]:
            return 0, 0, 0.0
        if t >= ts[-1]:
            n = len(ts) - 1
            return n, n, 0.0
        i1 = int(np.searchsorted(np.asarray(ts), t, side="left"))
        if ts[i1] == t:
            return i1, i1, 0.0
        i0 = i1 - 1
        frac = (t - ts[i0]) / (ts[i1] - ts[i0])
        return i0, i1, float(frac)

    def _row(self, loc: Tuple[int, int, float]) -> np.ndarray:
        i0, i1, f = loc
        if i0 == i1:
            e = self._eta[i0].copy()
        else:
            e = (1.0 - f) * self._eta[i0] + f * self._eta[i1]
        e[5] = _wrap(e[5])
        return e

    def interp(self, t: float) -> np.ndarray:
        """eta6 at ``t`` (linear on all six, yaw along the unwrapped axis,
        returned wrapped); clamps outside the span; raises when empty."""
        return self._row(self._locate(float(t)))

    def fix_stamps(self, t: float) -> Tuple[float, ...]:
        """The fix stamp(s) an ``interp(t)`` draws on — one when clamped or
        exact, two when bracketed. For the plan record (``obs_fix_t``)."""
        i0, i1, _ = self._locate(float(t))
        return (self._t[i0],) if i0 == i1 else (self._t[i0], self._t[i1])

    def rows_at(self, t_obs: float, obs_dt: float
                ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """(eta_prev, eta_now, degenerate) for rows at ``t_obs - obs_dt`` and
        ``t_obs``. See the class docstring for the degenerate rule.

        NOTE: when ``t_obs`` is NEWER than the newest fix the 'now' row clamps
        to that fix while the 'prev' row is still placed at ``t_obs - obs_dt``,
        so the two rows are LESS than ``obs_dt`` apart and the motion cue is
        under-scaled without being flagged (verify 2026-09-02). Use
        :meth:`rows_for` for a policy observation; this one is kept unchanged
        for callers that want the literal stamps.
        """
        t_obs = float(t_obs)
        obs_dt = float(obs_dt)
        loc_prev = self._locate(t_obs - obs_dt)
        loc_now = self._locate(t_obs)
        same_single = (loc_prev[0] == loc_prev[1] and loc_now[0] == loc_now[1]
                       and loc_prev[0] == loc_now[0])
        degenerate = bool(same_single or self.span() < obs_dt)
        return self._row(loc_prev), self._row(loc_now), degenerate

    def rows_for(self, t_now: float, spacing: float
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """The two proprio rows for an observation stamped ``t_now``, EXACTLY
        ``spacing`` apart, never extrapolated (verify 2026-09-02).

        ``t_now_eff = min(t_now, latest_t)`` — a depth stamp newer than the
        newest fix is served by the newest fix, and BOTH rows shift back with
        it (``t_prev = t_now_eff - spacing``) so the displacement between them
        is the ``spacing`` the policy was trained on, not a fraction of it.
        Returns ``(eta_prev, eta_now, info)`` with ``info = {"t_prev",
        "t_now" (the effective row stamps — record THESE as obs_rows_t),
        "fix_lag_s" (``max(0, t_now - latest_t)``, how far the rows lag the
        depth frame), "degenerate" (both rows on one fix, or the history
        spans less than ``spacing``)}``. Raises on an empty history.
        """
        if not self._t:
            raise ValueError("EtaHistory is empty")
        t_now = float(t_now)
        spacing = float(spacing)
        latest = float(self._t[-1])
        t_now_eff = min(t_now, latest)
        t_prev = t_now_eff - spacing
        loc_prev = self._locate(t_prev)
        loc_now = self._locate(t_now_eff)
        same_single = (loc_prev[0] == loc_prev[1] and loc_now[0] == loc_now[1]
                       and loc_prev[0] == loc_now[0])
        degenerate = bool(same_single or self.span() < spacing)
        info: Dict[str, object] = {
            "t_prev": float(t_prev), "t_now": float(t_now_eff),
            "fix_lag_s": float(max(0.0, t_now - latest)),
            "degenerate": degenerate,
        }
        return self._row(loc_prev), self._row(loc_now), info


# =============================================================================
# low-dim observation (spec v1 §0 / D3, v2 A6)
# =============================================================================
def lowdim_obs(eta_prev, eta_now, eta_start, T_bt: np.ndarray,
               w_prev: float, w_now: float) -> Dict[str, np.ndarray]:
    """The policy's proprio rows from two eta rows (older first).

    ``T_i = T_from_eta(eta_i) @ T_bt`` is the TCP pose in datum NED; the
    dataset's ``obs_pose_repr: relative`` is ``T_rel_i = inv(T_latest) @ T_i``
    (upstream ``convert_pose_mat_rep(..., 'relative')``), so the LAST row is
    identity / zero by construction. ``wrt_start`` is the rotation part of
    ``inv(T_start) @ T_i`` with ``T_start`` the TCP pose at policy START
    (training jittered that start pose with N(0, 0.05) noise — a small
    error in ``eta_start`` is inside the distribution). Output float32 in
    the contract's shapes: pos (2,3), rot6d (2,6), width (2,1), wrt_start
    (2,6). Frame invariance is the whole point: the dict is the same for any
    datum, only relative motion and the start-relative rotation survive.
    """
    if eta_start is None:
        raise ValueError("lowdim_obs: eta_start is None (policy not started)")
    T_bt = np.asarray(T_bt, dtype=float).reshape(4, 4)
    T_prev = T_from_eta(eta_prev) @ T_bt
    T_now = T_from_eta(eta_now) @ T_bt
    T_start = T_from_eta(eta_start) @ T_bt
    inv_latest = _inv_T(T_now)
    inv_start = _inv_T(T_start)
    rows = (T_prev, T_now)
    pos = np.zeros((2, 3))
    rot = np.zeros((2, 6))
    wrt = np.zeros((2, 6))
    for i, T_i in enumerate(rows):
        rel = inv_latest @ T_i
        pos[i] = rel[:3, 3]
        rot[i] = mat_to_rot6d(rel[:3, :3])
        wrt[i] = mat_to_rot6d((inv_start @ T_i)[:3, :3])
    width = np.array([[float(w_prev)], [float(w_now)]])
    return {
        "robot0_eef_pos": pos.astype(np.float32),
        "robot0_eef_rot_axis_angle": rot.astype(np.float32),
        "robot0_gripper_width": width.astype(np.float32),
        "robot0_eef_rot_axis_angle_wrt_start": wrt.astype(np.float32),
    }


def _inv_T(T: np.ndarray) -> np.ndarray:
    """Rigid-transform inverse (R^T, -R^T t) — no np.linalg.inv on a 4x4."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


# =============================================================================
# knot resampling (spec A5)
# =============================================================================
def _window_mean(vals: np.ndarray, obs_dt: float, knot_dt: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (D, K) raw knots at ``k*obs_dt`` onto ``j*knot_dt`` by the
    mean of the raw knots inside ``[tau_j - knot_dt/2, tau_j + knot_dt/2]``
    (clamped to ``[0, T_end]``); knot 0 pinned to raw knot 0 and, when the
    grid's last knot lands ON ``T_end``, the last knot pinned to raw knot
    K-1 (verify 2026-09-02: a one-sided window mean stopped the plan short
    of the policy's endpoint); returns (out (D, J), tau)."""
    vals = np.asarray(vals, dtype=float)
    D, K = vals.shape
    T_end = (K - 1) * obs_dt
    eps = 1e-9
    J = int(math.floor(T_end / knot_dt + 1e-6)) + 1
    tau = np.arange(J) * knot_dt
    t_raw = np.arange(K) * obs_dt
    out = np.zeros((D, J))
    out[:, 0] = vals[:, 0]                                  # PINNED
    pin_last = bool(J >= 2 and abs(tau[J - 1] - T_end) <= 1e-6 * knot_dt)
    if pin_last:
        out[:, J - 1] = vals[:, K - 1]                      # PINNED (endpoint)
    for j in range(1, J - 1 if pin_last else J):
        lo = max(0.0, tau[j] - 0.5 * knot_dt)
        hi = min(T_end, tau[j] + 0.5 * knot_dt)
        sel = (t_raw >= lo - eps) & (t_raw <= hi + eps)
        if sel.any():
            out[:, j] = vals[:, sel].mean(axis=1)
        else:                                               # knot_dt < obs_dt
            for d in range(D):
                out[d, j] = np.interp(tau[j], t_raw, vals[d])
    return out, tau


def resample_knots(p: np.ndarray, yaw: np.ndarray, g: np.ndarray,
                   obs_dt: float, knot_dt: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Raw policy knots (3,K)/(K,)/(K,) at ``obs_dt`` -> (p (3,J), yaw (J,),
    g (J,), jitter_rms_mm) on the ``knot_dt`` grid, spec A5.

    Knot j sits at ``tau_j = j*knot_dt`` for ``j = 0..floor(T_end/knot_dt)``
    (``T_end = (K-1)*obs_dt``; 16 knots @ 1/15 s, 0.2 s grid -> 6 knots at
    0..1.0 s) and is the MEAN of the raw knots inside ``tau_j +- knot_dt/2``
    (clamped to the plan); knot 0 is PINNED to raw knot 0 so the plan still
    begins at the policy's "now", and the LAST knot is pinned to raw knot
    K-1 whenever the grid ends on ``T_end`` (it does for the shipped 16 @
    1/15 s -> 0.2 s contract) so the plan ENDS where the policy said —
    the one-sided window mean over raw knots 14..15 used to stop it half a
    raw step short (verify 2026-09-02). Yaw is unwrapped BEFORE averaging
    (a mean across a +-pi seam would point backwards) and returned unwrapped.

    ``jitter_rms_mm`` = RMS over the raw knots of the second-difference
    VECTOR norm, in mm — the quantity the filter's finite-difference accel
    gate amplifies by 1/dt^2 [유도: control review], recorded per plan so
    a rejection can be traced to jitter rather than intent.
    """
    p = np.asarray(p, dtype=float)
    yaw = np.asarray(yaw, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)
    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"resample_knots: p must be (3, K), got {p.shape}")
    K = p.shape[1]
    if yaw.shape[0] != K or g.shape[0] != K:
        raise ValueError("resample_knots: yaw/g length must match p")
    obs_dt = float(obs_dt)
    knot_dt = float(knot_dt)
    if not (obs_dt > 0.0 and knot_dt > 0.0):
        raise ValueError("resample_knots: obs_dt and knot_dt must be > 0")
    if K >= 3:
        d2 = np.diff(p, n=2, axis=1)                        # (3, K-2)
        jitter_rms_mm = float(np.sqrt(np.mean(np.sum(d2 * d2, axis=0))) * 1e3)
    else:
        jitter_rms_mm = 0.0
    yaw_u = np.unwrap(yaw)
    stacked = np.vstack([p, yaw_u[None, :], g[None, :]])   # (5, K)
    out, _tau = _window_mean(stacked, obs_dt, knot_dt)
    return out[:3], out[3], out[4], jitter_rms_mm


def resample_knots_att(rp: np.ndarray, obs_dt: float, knot_dt: float
                       ) -> np.ndarray:
    """Raw roll/pitch knots (2, K) at ``obs_dt`` -> (2, J) on the ``knot_dt``
    grid: the SAME windowed mean, knot-0 and last-knot pinning as
    :func:`resample_knots` (one ``_window_mean`` call, so the attitude
    knots sit on exactly the positions' grid). No unwrap: roll/pitch are
    bounded far below pi by the compose clip."""
    rp = np.asarray(rp, dtype=float)
    if rp.ndim != 2 or rp.shape[0] != 2:
        raise ValueError(f"resample_knots_att: rp must be (2, K), got {rp.shape}")
    obs_dt = float(obs_dt)
    knot_dt = float(knot_dt)
    if not (obs_dt > 0.0 and knot_dt > 0.0):
        raise ValueError("resample_knots_att: obs_dt and knot_dt must be > 0")
    out, _tau = _window_mean(rp, obs_dt, knot_dt)
    return out


# =============================================================================
# plan composition (spec A2 + A5 + A12)
# =============================================================================
def compose_plan(action: np.ndarray, anchor_eta6, rp_level, T_bt: np.ndarray,
                 obs_dt: float, knot_dt: float, t0: float, plan_id: int,
                 obs_t_rel: float, w_open: float, w_closed: float,
                 *, action_repr: Optional[str] = None,
                 track_rp: bool = False, rp_max_rad: Optional[float] = None,
                 rp_reject_rad: Optional[float] = None
                 ) -> Tuple[PlanMsg, Dict[str, object]]:
    """A (K, 5), (K, 7) or (K, 10) policy action -> ``PlanMsg`` in datum NED
    (+ info).

    The representation is the array WIDTH (``action_repr_of``); a caller that
    knows the checkpoint contract passes ``action_repr`` and a disagreement is
    a ``ValueError`` (the plan lands in reject_compose with the reason).

    ``pos_yaw_width`` (K, 5), the flown contract (2026-09-07): anchor TCP pose
    ``T_ned_tcp_a = T_from_eta(anchor) @ T_bt``; ``p_tcp_k = p_a + R_ned_tcp_a
    @ dp_k``; ``yaw_k = anchor_yaw + dyaw_k`` (unwrapped) -- exact, since
    ``R_ned_tcp_a @ R_bt^T = R_ned_body_a`` makes the anchor's body yaw
    ``anchor_eta6[5]`` and ``rot_zyx(r, p, y + d) = Rz(d) @ rot_zyx(r, p, y)``
    turns every body-fixed vector (the optical axis the label was defined on
    included) by exactly ``d`` for any constant roll/pitch. Nothing is
    dropped (``dropped_rp_deg`` is 0.0 by construction). Rejects a knot with
    ``|dyaw| > DYAW_MAX_RAD``.

    ``pose10d`` (K, 10), legacy: per knot ``T_ned_tcp_k = T_ned_tcp_a @
    pose10d_to_mat(a_k)``; ``yaw_k`` = ZYX yaw of the implied body rotation
    ``R_ned_tcp_k @ R_bt^T``; roll/pitch replaced by the anchor's measured
    ``rp_level`` and the difference recorded. For a yaw-only relative rotation
    both branches give the same ``p_body`` / ``yaw`` (test_policy_frames).

    ``pos_rpy_width`` (K, 7), the 6-DoF variant (2026-09-26, D12): cols 0:4
    exactly as the 5-dim branch (``p_tcp`` is the policy's request and is
    NEVER altered); ``R_body_k = R_body_a @ rot_zyx(droll_k, dpitch_k,
    dyaw_k)`` with ``R_body_a = rot_zyx(*anchor[3:6])`` (body-z
    composition, module docstring) and ``(roll_k, pitch_k, yaw_k) =
    rpy_of(R_body_k)``. Two sub-modes on ``track_rp``:
      * False (DEFAULT, "dropped-and-logged"): the 5-dim arithmetic on cols
        0:4 with ``rp_level`` levelling -- byte-identical to what the same
        action's cols [0:4, 6] give on the 5-dim path; ``dropped_rp_deg`` =
        max |rp_k - rp_level| is MEANINGFUL again; ``PlanMsg.rp`` is None.
      * True: ``rp_raw[:, k] = (roll_k, pitch_k)``; FIRST two rejections
        on the RAW (unclipped) attitude, both ``ValueError`` (the worker
        rejects the plan with the message, A23), BEFORE ``yaw_k`` is used:
        the singularity guard ``|pitch_k| >= PITCH_SINGULAR_RAD`` (85 deg,
        ALWAYS on: past it :func:`rpy_of`'s yaw is ``atan2(0, 0)`` garbage
        that the T1 clip would not repair), then T2 ``max |rp_raw| >
        rp_reject_rad`` when ``rp_reject_rad`` is given (None = no T2
        here; on the compose path T2 was DEAD before 2026-09-26 because T1
        clipped first). Then the survivors are clipped per axis to
        +-``rp_max_rad`` (T1; ``rp_clipped_deg`` / ``rp_clipped_n``
        recorded), ``yaw_raw = unwrap(yaw_k)``, ``p_body_k = p_tcp_k -
        rot_zyx(rp_clip_k, yaw_k) @ t_bt`` (NO levelling: the hull carries
        the clipped commanded attitude), ``rp_dropped = 0``, and
        ``PlanMsg.rp`` = :func:`resample_knots_att` of the clipped knots.
        ``droll = dpitch = 0`` holds the anchor's attitude (handheld
        semantics: keep what you have); the anchor's roll/pitch should be
        the LEASHED attitude (:func:`leash_anchor` ``rp_ref``).
    Rejects |droll| or |dpitch| > ``DRP_MAX_RAD`` like |dyaw| > pi;
    ``track_rp`` with a 5- or 10-dim action is a ``ValueError`` (the arm
    gate refuses that pairing before any plan exists). ``rp_reject_rad``
    is validated (finite, >= 0) on every call but consulted ONLY on the
    tracked branch: the dropped-and-logged and the 5-/10-dim paths are
    byte-identical with or without it.

    Both: leveled body position ``p_body_k = p_tcp_k - rot_zyx(roll_lvl,
    pitch_lvl, yaw_k) @ t_bt`` (``rp_level`` = the anchor's MEASURED
    roll/pitch, what the NMPC holds); ``g_k = clip((w_k - w_closed) /
    (w_open - w_closed), 0, 1)`` (spec A12) from the last column; then
    :func:`resample_knots` onto the ``knot_dt`` grid (spec A5); the
    message's ``dt`` is ``knot_dt``, ``t0`` and ``obs_t`` are the caller's
    (mission-relative, spec A1 -- this function does no clock conversion).

    Raises ``ValueError`` for a non-finite action, a wrong width / contract
    mismatch, a rot6d with a vanishing 3-vector (norm < ``ROT6D_MIN_NORM``),
    ``|dyaw| > pi``, ``w_open <= w_closed``, and on the tracked 7-dim path
    an absolute pitch at/over ``PITCH_SINGULAR_RAD`` or a raw attitude over
    ``rp_reject_rad``. The caller rejects the plan with the message (spec
    A23); nothing here disengages.

    ``info`` carries: ``action_repr``, ``dropped_rp_deg`` (max over raw knots
    of |roll_k - roll_lvl|, |pitch_k - pitch_lvl|; 0.0 on the 5-dim path),
    ``rp_dropped_deg`` (2, K), ``jitter_rms_mm``, ``p_tcp`` (3, J) resampled
    TCP positions, ``p_tcp_raw`` (3, K), ``p_body_raw`` (3, K), ``yaw_raw``
    (K,) unwrapped, ``anchor_tcp`` (p (3,), yaw_of(T_ned_tcp_a) -- NOTE: the
    azimuth of the camera x axis, not the body yaw), ``n_raw``, ``n_knots``,
    ``rp_tracked`` (bool, always). On the 7-dim path additionally ``rp_raw``
    (2, K) rad (decoded absolute roll/pitch per raw knot, BEFORE the clip),
    ``rp`` (2, J) rad (the message's knots; None when not tracked),
    ``rp_clipped_deg`` (float) and ``rp_clipped_n`` (int).
    """
    a = np.asarray(action, dtype=float)
    if a.ndim != 2:
        raise ValueError("compose_plan: action must be (K, 5), (K, 7) or "
                         f"(K, 10), got {a.shape}")
    repr_ = action_repr_of(a.shape[1])
    if action_repr is not None and str(action_repr) != repr_:
        raise ValueError(f"compose_plan: action_repr {str(action_repr)!r} (checkpoint "
                         f"contract) but the action is {a.shape} ({repr_})")
    if not np.all(np.isfinite(a)):
        raise ValueError("compose_plan: action has non-finite entries")
    w_open = float(w_open)
    w_closed = float(w_closed)
    if not (w_open > w_closed):
        raise ValueError("compose_plan: w_open must exceed w_closed")
    T_bt = np.asarray(T_bt, dtype=float).reshape(4, 4)
    R_bt = T_bt[:3, :3]
    t_bt = T_bt[:3, 3]
    rp = np.asarray(rp_level, dtype=float).reshape(2)
    roll_lvl, pitch_lvl = float(rp[0]), float(rp[1])

    track_rp = bool(track_rp)
    if track_rp and repr_ != ACTION_REPR_POS_RPY_WIDTH:
        raise ValueError(f"compose_plan: track_rp requires a (K, 7) "
                         f"{ACTION_REPR_POS_RPY_WIDTH} action, got {a.shape} "
                         f"({repr_})")
    if rp_max_rad is not None:
        rp_max_rad = float(rp_max_rad)
        if not (math.isfinite(rp_max_rad) and rp_max_rad >= 0.0):
            raise ValueError("compose_plan: rp_max_rad must be finite and >= 0")
    if rp_reject_rad is not None:
        rp_reject_rad = float(rp_reject_rad)
        if not (math.isfinite(rp_reject_rad) and rp_reject_rad >= 0.0):
            raise ValueError("compose_plan: rp_reject_rad must be finite and >= 0")

    K = a.shape[0]
    anchor6 = np.asarray(anchor_eta6, dtype=float).reshape(6)
    T_ned_tcp_a = T_from_eta(anchor6) @ T_bt
    p_body = np.zeros((3, K))
    rp_raw: Optional[np.ndarray] = None       # 7-dim only
    rp_r: Optional[np.ndarray] = None         # 7-dim tracked only
    rp_clipped_deg = 0.0
    rp_clipped_n = 0
    if repr_ == ACTION_REPR_POSE10D:
        n1 = np.linalg.norm(a[:, 3:6], axis=1)
        n2 = np.linalg.norm(a[:, 6:9], axis=1)
        if np.any(n1 < ROT6D_MIN_NORM) or np.any(n2 < ROT6D_MIN_NORM):
            raise ValueError("compose_plan: degenerate rot6d (norm < "
                             f"{ROT6D_MIN_NORM:g}) at knot "
                             f"{int(np.argmax((n1 < ROT6D_MIN_NORM) | (n2 < ROT6D_MIN_NORM)))}")
        T_k = T_ned_tcp_a[None] @ pose10d_to_mat(a[:, :9])      # (K, 4, 4)
        p_tcp = T_k[:, :3, 3].T                                 # (3, K)
        R_body = T_k[:, :3, :3] @ R_bt.T                        # (K, 3, 3)
        yaw_raw = np.zeros(K)
        rp_dropped = np.zeros((2, K))
        for k in range(K):
            roll_k, pitch_k, yaw_k = rpy_of(R_body[k])
            yaw_raw[k] = yaw_k
            rp_dropped[0, k] = math.degrees(_wrap(roll_k - roll_lvl))
            rp_dropped[1, k] = math.degrees(_wrap(pitch_k - pitch_lvl))
            p_body[:, k] = p_tcp[:, k] - rot_zyx(roll_lvl, pitch_lvl, yaw_k) @ t_bt
        yaw_raw = np.unwrap(yaw_raw)
        w_col = 9
    elif repr_ == ACTION_REPR_POS_RPY_WIDTH:
        dp = a[:, :3]
        dyaw = a[:, 3]
        droll = a[:, 4]
        dpitch = a[:, 5]
        bad = np.abs(dyaw) > DYAW_MAX_RAD
        if bad.any():
            raise ValueError(f"compose_plan: |dyaw| > pi ({DYAW_MAX_RAD:.6g} rad) at "
                             f"knot {int(np.argmax(bad))}")
        bad = (np.abs(droll) > DRP_MAX_RAD) | (np.abs(dpitch) > DRP_MAX_RAD)
        if bad.any():
            raise ValueError(f"compose_plan: |droll| or |dpitch| > pi/2 "
                             f"({DRP_MAX_RAD:.6g} rad) at knot {int(np.argmax(bad))}")
        R_a = T_ned_tcp_a[:3, :3]
        p_a = T_ned_tcp_a[:3, 3]
        p_tcp = p_a[:, None] + R_a @ dp.T                       # (3, K) -- the request
        # Body-z composition (D12): the relative attitude about the ANCHOR's
        # own body axes, the axes the label measured through R_bt.
        R_body_a = rot_zyx(float(anchor6[3]), float(anchor6[4]), float(anchor6[5]))
        rp_raw = np.zeros((2, K))
        yaw_body = np.zeros(K)
        for k in range(K):
            R_body_k = R_body_a @ rot_zyx(float(droll[k]), float(dpitch[k]),
                                          float(dyaw[k]))
            roll_k, pitch_k, yaw_k = rpy_of(R_body_k)
            rp_raw[0, k] = _wrap(roll_k)
            rp_raw[1, k] = _wrap(pitch_k)
            yaw_body[k] = yaw_k
        rp_dropped = np.zeros((2, K))
        if not track_rp:
            # Dropped-and-logged: the 5-dim arithmetic on cols 0:4 and the
            # anchor's measured attitude for levelling, verbatim; only the
            # record knows what attitude the policy asked for.
            yaw_raw = np.unwrap(anchor6[5] + dyaw)              # exact (see docstring)
            for k in range(K):
                p_body[:, k] = p_tcp[:, k] - rot_zyx(roll_lvl, pitch_lvl, yaw_raw[k]) @ t_bt
                rp_dropped[0, k] = math.degrees(_wrap(rp_raw[0, k] - roll_lvl))
                rp_dropped[1, k] = math.degrees(_wrap(rp_raw[1, k] - pitch_lvl))
        else:
            # Singularity guard, ALWAYS, before yaw_body is consumed: the
            # absolute pitch is anchor pitch (+) relative dpitch (<= pi/2),
            # so it can reach 90 deg where rpy_of's yaw is atan2(0, 0).
            sing = np.abs(rp_raw[1]) >= PITCH_SINGULAR_RAD
            if sing.any():
                k_bad = int(np.argmax(sing))
                raise ValueError(
                    f"compose_plan: absolute pitch "
                    f"{math.degrees(float(rp_raw[1, k_bad])):.1f} deg at knot "
                    f"{k_bad} is at/over the {math.degrees(PITCH_SINGULAR_RAD):.0f} "
                    f"deg singularity guard (yaw undefined)")
            # T2 (D3) on the RAW attitude -- dead on this path while T1
            # clipped first; the worker's filter never saw the excess.
            if rp_reject_rad is not None:
                worst_k = int(np.argmax(np.abs(rp_raw).max(axis=0))) if K else 0
                worst = float(np.abs(rp_raw).max()) if K else 0.0
                if worst > rp_reject_rad:
                    raise ValueError(
                        f"compose_plan: requested attitude {math.degrees(worst):.1f} "
                        f"deg at knot {worst_k} exceeds rp_reject "
                        f"{math.degrees(rp_reject_rad):.1f} deg")
            yaw_raw = np.unwrap(yaw_body)
            if rp_max_rad is not None:
                rp_clip = np.clip(rp_raw, -rp_max_rad, rp_max_rad)   # T1 (D3)
            else:
                rp_clip = rp_raw.copy()
            d_clip = np.abs(rp_raw - rp_clip)
            rp_clipped_deg = float(math.degrees(d_clip.max())) if K else 0.0
            rp_clipped_n = int(np.count_nonzero(d_clip.max(axis=0) > 0.0))
            for k in range(K):
                # NO levelling: the jaw stays where the policy put it and
                # the hull carries the clipped commanded attitude.
                p_body[:, k] = p_tcp[:, k] - rot_zyx(float(rp_clip[0, k]),
                                                     float(rp_clip[1, k]),
                                                     float(yaw_raw[k])) @ t_bt
            rp_r = resample_knots_att(rp_clip, obs_dt, knot_dt)
        w_col = 6
    else:
        dp = a[:, :3]
        dyaw = a[:, 3]
        bad = np.abs(dyaw) > DYAW_MAX_RAD
        if bad.any():
            raise ValueError(f"compose_plan: |dyaw| > pi ({DYAW_MAX_RAD:.6g} rad) at "
                             f"knot {int(np.argmax(bad))}")
        R_a = T_ned_tcp_a[:3, :3]
        p_a = T_ned_tcp_a[:3, 3]
        p_tcp = p_a[:, None] + R_a @ dp.T                       # (3, K)
        yaw_raw = np.unwrap(anchor6[5] + dyaw)                  # exact (see docstring)
        rp_dropped = np.zeros((2, K))
        for k in range(K):
            p_body[:, k] = p_tcp[:, k] - rot_zyx(roll_lvl, pitch_lvl, yaw_raw[k]) @ t_bt
        w_col = 4
    g_raw = np.clip((a[:, w_col] - w_closed) / (w_open - w_closed), 0.0, 1.0)

    p_r, yaw_r, g_r, jitter = resample_knots(p_body, yaw_raw, g_raw,
                                             obs_dt, knot_dt)
    p_tcp_r, _ = _window_mean(p_tcp, float(obs_dt), float(knot_dt))
    msg = PlanMsg(plan_id=int(plan_id), t0=float(t0), dt=float(knot_dt),
                  p_ned=p_r, yaw=yaw_r, g=g_r, obs_t=float(obs_t_rel),
                  rp=rp_r)
    info: Dict[str, object] = {
        "action_repr": repr_,
        "dropped_rp_deg": float(np.abs(rp_dropped).max()) if K else 0.0,
        "rp_dropped_deg": rp_dropped,
        "jitter_rms_mm": float(jitter),
        "p_tcp": p_tcp_r,
        "p_tcp_raw": p_tcp,
        "p_body_raw": p_body,
        "yaw_raw": yaw_raw,
        "anchor_tcp": (T_ned_tcp_a[:3, 3].copy(), yaw_of(T_ned_tcp_a)),
        "n_raw": int(K),
        "n_knots": int(p_r.shape[1]),
        "rp_tracked": bool(track_rp),
    }
    if rp_raw is not None:
        info["rp_raw"] = rp_raw
        info["rp"] = rp_r
        info["rp_clipped_deg"] = float(rp_clipped_deg)
        info["rp_clipped_n"] = int(rp_clipped_n)
    return msg, info


# =============================================================================
# leashed anchor (spec A4)
# =============================================================================
def leash_anchor(eta_meas6, p_ref3, yaw_ref: float, leash_m: float,
                 leash_yaw: float, *, rp_ref=None, leash_rp: float = 0.0
                 ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Anchor = measurement + the reference offset clipped to a leash.

    ``p_anchor = p_meas + clip_norm(p_ref - p_meas, leash_m)``,
    ``yaw_anchor = yaw_meas + clip(wrap(yaw_ref - yaw_meas), +-leash_yaw)``;
    roll/pitch are the MEASURED ones (they become ``rp_level`` for
    :func:`compose_plan`) -- unless ``rp_ref`` (the flown attitude
    reference (roll, pitch) at the obs time) is given AND ``leash_rp > 0``
    (2026-09-26, the attitude-tracking variant): then ``rp_anchor =
    rp_meas + clip(wrap(rp_ref - rp_meas), +-leash_rp)`` per axis, the
    same leash idea on the attitude channel (a replan re-anchored to a
    0.5 s-stale measurement is the 2026-09-14 yaw-wobble mechanism), and
    ``info`` additionally carries ``drp`` (2,), ``drp_applied`` (2,),
    ``clipped_rp``. With no installed plan the caller passes the
    measurement as the reference and both branches coincide. Returns the
    anchor eta6 and ``info``: ``offset_ned`` (raw ``p_ref - p_meas``),
    ``offset_m`` (its norm), ``dyaw`` (raw wrapped yaw offset), plus
    ``offset_applied_m``, ``dyaw_applied``, ``clipped_pos``, ``clipped_yaw``.
    """
    e = np.asarray(eta_meas6, dtype=float).reshape(6)
    p_ref = np.asarray(p_ref3, dtype=float).reshape(3)
    leash_m = float(leash_m)
    leash_yaw = float(leash_yaw)
    off = p_ref - e[:3]
    off_m = float(np.linalg.norm(off))
    clipped_pos = off_m > leash_m
    off_applied = off * (leash_m / off_m) if clipped_pos and off_m > 0.0 else off.copy()
    if leash_m <= 0.0:
        off_applied = np.zeros(3)
        clipped_pos = off_m > 0.0
    dyaw = _wrap(float(yaw_ref) - e[5])
    dyaw_applied = max(-leash_yaw, min(leash_yaw, dyaw))
    clipped_yaw = abs(dyaw) > leash_yaw
    anchor = e.copy()
    anchor[:3] = e[:3] + off_applied
    anchor[5] = _wrap(e[5] + dyaw_applied)
    info: Dict[str, object] = {
        "offset_ned": off,
        "offset_m": off_m,
        "dyaw": dyaw,
        "offset_applied_m": float(np.linalg.norm(off_applied)),
        "dyaw_applied": float(dyaw_applied),
        "clipped_pos": bool(clipped_pos),
        "clipped_yaw": bool(clipped_yaw),
    }
    leash_rp = float(leash_rp)
    if rp_ref is not None and leash_rp > 0.0:
        rr = np.asarray(rp_ref, dtype=float).reshape(2)
        drp = np.array([_wrap(float(rr[0]) - e[3]), _wrap(float(rr[1]) - e[4])])
        drp_applied = np.clip(drp, -leash_rp, leash_rp)
        anchor[3] = _wrap(e[3] + float(drp_applied[0]))
        anchor[4] = _wrap(e[4] + float(drp_applied[1]))
        info["drp"] = drp
        info["drp_applied"] = drp_applied
        info["clipped_rp"] = bool(np.any(np.abs(drp) > leash_rp))
    return anchor, info


# =============================================================================
# open-loop gripper width (spec A12)
# =============================================================================
class GripperWidthEstimator:
    """Jaw width from the drive levels the jaw was sent — there is no feedback.

    ``drive(level, t)`` integrates the PREVIOUS level up to ``t`` then latches
    the new one (+1 opens, -1 closes, 0 holds — the ``cmd_gripper_drive``
    convention; fractional levels scale the rate). The rate is
    ``(w_open - w_closed) / travel_s`` per unit level per second and the
    width is clamped to ``[w_closed, w_open]`` (the DATASET's open/closed
    levels, spec A12 — the Newton's 62 mm [스펙] max opening is outside that
    band, which is fine: the policy never saw it). ``width(t)`` advances the
    integrator to ``t`` and returns the estimate; ``release(t)`` = drive 0
    at ``t`` (called from every path that zeroes the jaw command). A
    non-advancing ``t`` neither integrates nor un-latches. ``w_init`` is
    an ASSUMPTION (jaw open at process start) — the caller logs it.
    """

    def __init__(self, w_open: float, w_closed: float, w_init: float,
                 travel_s: float):
        self.w_open = float(w_open)
        self.w_closed = float(w_closed)
        if not (self.w_open > self.w_closed):
            raise ValueError("GripperWidthEstimator: w_open must exceed w_closed")
        self.travel_s = float(travel_s)
        if not (self.travel_s > 0.0):
            raise ValueError("GripperWidthEstimator: travel_s must be > 0")
        self._w = min(self.w_open, max(self.w_closed, float(w_init)))
        self._level = 0.0
        self._t: Optional[float] = None
        self.n_drives = 0

    @property
    def rate(self) -> float:
        """[m/s] at |level| = 1."""
        return (self.w_open - self.w_closed) / self.travel_s

    @property
    def level(self) -> float:
        return self._level

    def _advance(self, t: float) -> None:
        if self._t is None:
            self._t = t
            return
        dt = t - self._t
        if dt <= 0.0:
            return
        self._w = min(self.w_open, max(self.w_closed,
                                       self._w + self._level * self.rate * dt))
        self._t = t

    def drive(self, level: float, t: float) -> None:
        """Latch a new drive level at ``t`` (after integrating the old one).
        ``n_drives`` counts level CHANGES only — the bus copy of a drive this
        process emitted itself re-latches the same level and must not count
        twice (verify 2026-09-02)."""
        t = float(t)
        self._advance(t)
        lv = float(level)
        if not math.isfinite(lv):
            lv = 0.0
        lv = max(-1.0, min(1.0, lv))
        if lv != self._level:
            self.n_drives += 1
        self._level = lv

    def release(self, t: float) -> None:
        """Drive released (level 0) at ``t`` — the jaw holds its position."""
        self.drive(0.0, t)

    def width(self, t: float) -> float:
        """The estimate at ``t`` [m], integrating the latched level up to it."""
        self._advance(float(t))
        return float(self._w)

    def g(self, t: float) -> float:
        """The same estimate as the policy's [0, 1] level (0 = closed)."""
        return float((self.width(t) - self.w_closed) / (self.w_open - self.w_closed))


# =============================================================================
# yaw reference filter (2026-09-14)
# =============================================================================
def guard_yaw(yaw: float, yaw_meas: float, guard: float) -> Tuple[float, bool]:
    """``yaw`` pulled to within ``guard`` [rad] of ``yaw_meas`` (wrapped
    difference, unwrapped branch of ``yaw`` kept). Returns (yaw, clamped).
    The ONE place the measured yaw reaches the filtered yaw reference."""
    yaw = float(yaw)
    d = _wrap(yaw - float(yaw_meas))
    if abs(d) > guard:
        return yaw - d + math.copysign(guard, d), True
    return yaw, False


def filter_plan_yaw(msg: PlanMsg, p_tcp_knots, yaw_ref0: Optional[float],
                    yaw_meas: float, rp_level, T_bt: np.ndarray,
                    omega_prev: float, dt_since_prev: float, *,
                    rate: float, tau: float, guard: float
                    ) -> Tuple[PlanMsg, Dict[str, object]]:
    """Replace a composed plan's yaw knots with a SLOW reference that never
    restarts from the measured yaw (2026-09-14, the wobble amplifier).

    Why: the station yaw-step test (data/20260914/0914_190226) showed the
    NMPC yaw loop is stable but underdamped (overshoot 18-45 %, ring period
    ~1.4-2 s, settles in 1-2 cycles). The policy runs kept it ringing
    because every 0.5 s plan re-anchored the yaw reference to the yaw
    measured at obs_t (0.5 s stale, a 0.8 s-delayed copy of the hull's own
    swing: run 2 of 0914_181425 anti-phase, eyaw_bp 4.8 deg > yaw_bp 3.5) or,
    with a yaw leash, to the policy's alternating dyaw. Either way the
    reference carried content near the ring period. This function makes the
    yaw reference (1) continuous with what is being FLOWN, (2) turn at a
    low-passed version of the policy's requested turn rate, capped, and (3)
    stay within ``guard`` of the measured yaw — the only place the
    measurement enters, and only in a gross-deviation case.

    Inputs: ``msg`` = the plan :func:`compose_plan` built (its yaw knots are
    ``anchor_yaw + dyaw_k``, unwrapped); ``p_tcp_knots`` (3, K) = the TCP
    positions at the same knots (``info["p_tcp"]``); ``yaw_ref0`` = the
    stitched reference's yaw at ``msg.t0`` (None = no plan flying yet: start
    at the plan's own first yaw, i.e. the anchor); ``yaw_meas`` = the
    FRESHEST measured yaw (for the guard); ``rp_level`` = the roll/pitch the
    plan was projected at; ``T_bt`` = body->TCP; ``omega_prev`` = the
    filter's turn-rate state from the previous plan [rad/s] (0 at arm);
    ``dt_since_prev`` = seconds since that plan's t0 (the period for the
    first plan). ``rate`` [rad/s], ``tau`` [s], ``guard`` [rad].

    COMPOSE AGAINST THE REFERENCE YAW (safety review 2026-09-14, HIGH): the
    caller must pass compose_plan an anchor whose yaw is ``yaw_ref0`` (guard-
    clamped with :func:`guard_yaw`) and hand the same value here. With the
    plan's TCP knots laid out on the MEASURED heading, re-deriving the body
    knots from them would move knot 0 off the anchor by ``[R(yaw_meas) -
    R(yaw_ref0)] t_bt`` — 0.13 m sideways at a 15 deg guard on the 0.5 m
    TCP lever — which puts the yaw tracking error straight back into the
    SWAY reference, the very coupling this filter exists to cut. Composed
    against the reference yaw instead, body knot 0 is exactly the anchor
    and the only cost is the chunk's dp being rotated by the tracking error
    (|dp| * dyaw, ~1 cm at 5 deg over 0.1 m). ``body_shift_max_m`` in the
    info then measures only the filtered-vs-requested dyaw ramp.

    A non-finite requested turn rate (a NaN anchor — that plan is refused
    by the PlanFilter anyway) counts as 0 and is flagged
    ``omega_policy_nonfinite`` instead of poisoning the state: Python's
    ``max/min`` would otherwise pin a NaN to ``+rate``.

    The filter: ``omega_policy`` = the chunk's mean turn rate (last knot
    minus first over its span); ``omega_ref = omega_prev + (omega_policy -
    omega_prev) * (1 - exp(-dt_since_prev / tau))`` clipped to ``+-rate``;
    ``yaw_k = yaw_0 + omega_ref * (t_k - t_0)`` with ``yaw_0`` the flown
    reference at t0. So a policy that keeps asking the same turn converges
    to it in ~tau (then turns at most ``rate``), and one that alternates
    every plan averages out instead of swinging the hull. The BODY knots are
    re-derived from the TCP knots with the filtered yaw (``p_body_k =
    p_tcp_k - rot_zyx(roll, pitch, yaw_k) @ t_bt``) so the TCP still goes
    where the policy asked; the heading is what is filtered. A plan that
    CARRIES attitude knots (``msg.rp``, the tracked 7-dim branch) is
    re-derived with its own per-knot ``rot_zyx(rp[0,k], rp[1,k], yaw_k)``
    and ``rp`` is forwarded unchanged -- otherwise this constructor would
    silently re-level the plan; ``msg.rp`` None -> ``rp_level``. ``info``
    records every number (the plan record's ``yaw_ref_filter``): the raw and
    filtered end-to-end dyaw, both omegas, the guard clamps, and the largest
    body-knot shift the re-derivation caused.
    """
    p_tcp = np.asarray(p_tcp_knots, dtype=float)
    yaw_in = np.unwrap(np.asarray(msg.yaw, dtype=float))
    K = int(yaw_in.size)
    if p_tcp.shape != (3, K):
        raise ValueError(f"filter_plan_yaw: p_tcp_knots {p_tcp.shape} vs "
                         f"{K} yaw knots")
    rate = float(rate)
    tau = float(tau)
    guard = float(guard)
    if not (rate > 0.0 and tau > 0.0 and guard > 0.0):
        raise ValueError("filter_plan_yaw: rate, tau and guard must be > 0")
    dt = float(msg.dt)
    span = (K - 1) * dt
    omega_policy = float((yaw_in[-1] - yaw_in[0]) / span) if span > 0.0 else 0.0
    omega_nonfinite = not math.isfinite(omega_policy)
    if omega_nonfinite:
        omega_policy = 0.0
    omega_prev = float(omega_prev)
    if not math.isfinite(omega_prev):
        omega_prev = 0.0
    dtp = float(dt_since_prev)
    if not math.isfinite(dtp) or dtp < 0.0:
        dtp = 0.0
    dtp = min(dtp, 4.0 * tau)          # a long gap converges, it does not jump
    alpha = 1.0 - math.exp(-dtp / tau)
    omega_ref = float(omega_prev) + alpha * (omega_policy - float(omega_prev))
    omega_ref = max(-rate, min(rate, omega_ref))
    rate_clipped = abs(float(omega_prev) + alpha * (omega_policy - float(omega_prev))) > rate

    yaw_meas = float(yaw_meas)
    yaw0 = float(yaw_in[0]) if yaw_ref0 is None else float(yaw_ref0)
    yaw0, clamped0 = guard_yaw(yaw0, yaw_meas, guard)
    t_rel = np.arange(K) * dt
    yaw_out = yaw0 + omega_ref * t_rel
    clamps = 0
    for k in range(K):
        yaw_out[k], c = guard_yaw(float(yaw_out[k]), yaw_meas, guard)
        clamps += int(c)
    T_bt = np.asarray(T_bt, dtype=float).reshape(4, 4)
    t_bt = T_bt[:3, 3]
    rp = np.asarray(rp_level, dtype=float).reshape(2)
    rp_msg = getattr(msg, "rp", None)
    if rp_msg is not None:
        rp_msg = np.asarray(rp_msg, dtype=float)
        if rp_msg.shape != (2, K):
            raise ValueError(f"filter_plan_yaw: msg.rp {rp_msg.shape} vs "
                             f"{K} yaw knots")
    p_body = np.zeros((3, K))
    for k in range(K):
        if rp_msg is not None:
            R_k = rot_zyx(float(rp_msg[0, k]), float(rp_msg[1, k]),
                          float(yaw_out[k]))
        else:
            R_k = rot_zyx(float(rp[0]), float(rp[1]), float(yaw_out[k]))
        p_body[:, k] = p_tcp[:, k] - R_k @ t_bt
    p_old = np.asarray(msg.p_ned, dtype=float)
    shift = float(np.linalg.norm(p_body - p_old, axis=0).max()) if K else 0.0
    out = PlanMsg(plan_id=msg.plan_id, t0=msg.t0, dt=msg.dt, p_ned=p_body,
                  yaw=yaw_out, g=msg.g, obs_t=msg.obs_t,
                  arrival_t=msg.arrival_t, rp=rp_msg)
    info: Dict[str, object] = {
        "rate_deg_s": math.degrees(rate), "tau_s": tau,
        "guard_deg": math.degrees(guard),
        "dt_since_prev_s": dtp, "alpha": float(alpha),
        "omega_policy_deg_s": math.degrees(omega_policy),
        "omega_policy_nonfinite": bool(omega_nonfinite),
        "omega_ref_deg_s": math.degrees(omega_ref),
        "rate_clipped": bool(rate_clipped),
        "yaw_ref0_deg": (None if yaw_ref0 is None else math.degrees(float(yaw_ref0))),
        "yaw0_deg": math.degrees(yaw0),
        "yaw_meas_deg": math.degrees(yaw_meas),
        "yaw_anchor_deg": math.degrees(float(yaw_in[0])),
        "dyaw_raw_deg": math.degrees(float(yaw_in[-1] - yaw_in[0])),
        "dyaw_filt_deg": math.degrees(float(yaw_out[-1] - yaw_out[0])),
        "guard_clamped_start": bool(clamped0),
        "guard_clamped_knots": int(clamps),
        "body_shift_max_m": shift,
    }
    return out, info
