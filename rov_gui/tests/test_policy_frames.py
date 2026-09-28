#!/usr/bin/env python3
"""
test_policy_frames.py — the policy <-> NED frame algebra, offline.

    ~/miniforge3/envs/robust/bin/python rov_gui/tests/test_policy_frames.py
    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_policy_frames.py

Pure numpy: no Qt, no torch, no acados. What it pins down (spec v2 §B,
control/policy_frames.py) — 22 tests + the pos_rpy_width set (2026-09-26):

* rot6d / pose10d round-trip and PARITY with the upstream
  ``umi.common.pose_util`` (imported by path when scipy is present — the
  policy was trained on that encoding, a transposed convention would be
  silently wrong);
* pos_yaw_width — the flown 5-dim ``[dx, dy, dz, dyaw, width]`` contract
  (2026-09-07): bit PARITY of ``R_BT_C3`` / ``rot_z`` / ``encode_pos_yaw`` /
  ``decode_pos_yaw`` with the upstream ``umi.common.yaw_action`` (imported
  by path; a UMI checkout WITHOUT that file is a FAILURE, not a skip — the
  label the checkpoint learned and the decode the station flies would be
  unverified against each other; only a missing checkout skips);
  ``decode -> encode`` round trip; and the SIGN / MOUNT fixtures that need no
  upstream: ``R_bt^T Rz(+10 deg) R_bt`` encodes to +10 deg and turns the
  composed plan clockwise seen from above (NED +), a hand roll of 5 deg
  about the heading axis with the camera 55 deg down leaks < 1.5 deg
  (the rejected optical-axis-azimuth label would leak ~7 deg), a pure camera
  pitch is exactly 0, and ``R_BT_C3`` IS ``NavConfig(cam_tilt_deg=43.3)``'s
  mount rotation;
* ``T_from_eta`` is the station's ``rot_zyx``;
* the low-dim obs is ``inv(T_latest) @ T_i`` (upstream
  ``convert_pose_mat_rep('relative')``, imported by path when present) with
  the last row identity/zero, and ``wrt_start`` is rotation-only;
* A2: a pure-pitch TCP delta leaves the projected TCP position unchanged
  and the LEVELED body position at the anchor, while the un-leveled body
  composition would have moved by |t_bt|·sin(pitch); a +z (forward) TCP
  delta through the tilted C3 moves the body forward-and-down;
* A3: ``tcp_offset_from_body`` is the exact inverse of ``T_body_tcp``;
* ``compose_plan`` — legacy (K, 10) AND the flown (K, 5) — yields a
  ``PlanMsg`` with obs_t set, dt == knot_dt, g in [0, 1] from the LAST
  column, knot 0 == the anchor's TCP projection, yaw unwrapped across +-pi,
  and rejects NaN / degenerate rot6d / |dyaw| > pi / a wrong width / an
  ``action_repr`` that disagrees with the width, with a ValueError (A23).
  On the 5-dim path ``yaw_k == anchor_yaw + dyaw_k`` EXACTLY and the
  optical axis' azimuth moves by exactly ``dyaw_k`` even at a 5 deg
  roll/pitch anchor (the label definition made concrete), and nothing is
  dropped (``dropped_rp_deg == 0.0`` structurally). The 5-dim decode equals
  the legacy decode to 1e-12 for a yaw-only rotation at a level anchor and,
  at a tilted LEASHED anchor, when the legacy rotation is about the WORLD
  vertical; the mount-vertical construction at a tilted anchor is NOT equal
  (legacy reads a body-z rotation as roll/pitch) — asserted, as a boundary;
* A5: ``resample_knots`` pins knot 0 and cuts the finite-difference
  acceleration of jittery knots by more than 3x;
* A4: ``leash_anchor`` clips a 0.30 m offset to the leash and a 40 deg yaw
  offset to leash_yaw, and passes through inside the leash;
* A6: ``EtaHistory`` dedups a repeated fix stamp, interpolates yaw the
  short way across +-pi, clamps, and flags degenerate rows;
* A12: ``GripperWidthEstimator`` integrates +-1 drives at
  (open-closed)/travel_s, clamps, and ``release`` stops it.
* pos_rpy_width (2026-09-26, the 6-DoF variant): bit PARITY of ``rot_x`` /
  ``rot_y`` / ``rpy_of_batch`` / ``encode_pos_rpy`` / ``decode_pos_rpy``
  with upstream ``yaw_action`` by path, the dyaw column bit-identical to the
  5-dim label, a camera pitch labelled as dpitch with dyaw EXACTLY 0;
  ``compose_plan``'s THIRD branch in both sub-modes (dropped-and-logged ==
  the 5-dim arithmetic byte for byte with a MEANINGFUL dropped_rp_deg;
  tracked: msg.rp (2, 6), knot-0 attitude == the anchor's, the T1 clip
  recorded with p_tcp untouched, body-z composition == the legacy pose10d
  decode at a tilted anchor and != the 5-dim world-z one); the pure-pitch
  twin; ``resample_knots_att``; the attitude leash (default byte-identical);
  ``filter_plan_yaw`` forwarding rp with per-knot attitude; the tracked
  branch's two REJECTIONS before yaw is used (the 85 deg pitch singularity
  guard, always; T2 ``rp_reject_rad``, when given — dead on this path
  before 2026-09-26 because T1 clipped first) with the dropped-and-logged
  and 4-DoF paths byte-identical either way; and the GOLDEN byte-hashes of
  every 4-DoF path recorded on the pre-variant code
  (``test_four_dof_paths_byte_identical_to_pre_variant``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rov_gui.control.geometry import NavConfig, S_FLU_FRD
from rov_gui.control.plan_stream import PlanMsg
from rov_gui.control.policy_frames import (
    DRP_MAX_RAD, DYAW_MAX_RAD, R_BT_C3, YAW_AXIS_CAM_TILT_DEG, EtaHistory,
    GripperWidthEstimator, PITCH_SINGULAR_RAD, T_body_tcp, T_from_eta,
    action_repr_of,
    compose_plan, decode_pos_rpy, decode_pos_yaw, encode_pos_rpy,
    encode_pos_yaw, filter_plan_yaw, leash_anchor, lowdim_obs,
    mat_to_pose10d, mat_to_rot6d, pose10d_to_mat, resample_knots,
    resample_knots_att, rot6d_to_mat, rot_x, rot_y, rot_z, rpy_of,
    rpy_of_batch, tcp_offset_from_body, yaw_of)
from rov_gui.control.state_assembler import rot_zyx
from rov_gui.state import (ACTION_DIM_BY_REPR, ACTION_REPR_POSE10D,
                           ACTION_REPR_POS_RPY_WIDTH,
                           ACTION_REPR_POS_YAW_WIDTH, POLICY_UMI_REPO)

#: The vendored UMI checkout (2026-09-09). ONE source: rov_gui.state.
UMI_REPO = Path(POLICY_UMI_REPO)
OBS_DT = 2.0 / 30.0          # down_sample_steps 2 @ 30 fps (spec v1 §0)
KNOT_DT = 0.2                # policy.knot_dt_s [예측] (spec A5)
W_OPEN, W_CLOSED = 0.069, 0.042   # spec A12 [측정: actions_summary.json]
IDENT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])   # rot6d of I (rows)
#: compose_plan's clock / gripper kwargs for the tests that do not vary them.
_KW = dict(obs_dt=OBS_DT, knot_dt=KNOT_DT, t0=0.0, plan_id=1, obs_t_rel=0.0,
           w_open=W_OPEN, w_closed=W_CLOSED)


# --------------------------------------------------------------- fixtures
def _load_by_path(path: Path, name: str):
    """Import a single upstream module file without touching sys.path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skip(msg: str) -> None:
    """Print-and-return skip: the plain runner counts it as ok, pytest too."""
    print(f"        skip: {msg}")


def _rand_R(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-ish random rotation via QR."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rand_T(rng: np.random.Generator) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rand_R(rng)
    T[:3, 3] = rng.uniform(-1, 1, 3)
    return T


def _c3_extrinsic(tilt_deg: float = 43.3):
    """The station's C3 extrinsic with the pool tilt (config/hw_nav.yaml
    cam_tilt_deg: 43.3) — NavConfig() alone is level, so set it here."""
    return NavConfig(cam_tilt_deg=tilt_deg).R_t_frd_cam("main")


def _T_bt(tilt_deg: float = 43.3) -> np.ndarray:
    """T_body_tcp for the ROV's own jaw (spec A3 tcp_body_flu_m [유도])."""
    R_bc, t_bc = _c3_extrinsic(tilt_deg)
    # = cam_t_flu + lens->grip [0.196, 0, -0.275] (2026-09-08, jaw anchored to
    # the lens; hw_mpc.yaml policy.tcp_body_flu_m). Was the sim JAW_POS 0.4165.
    t_bt = S_FLU_FRD @ np.array([0.502, 0.0, -0.17])
    return T_body_tcp(R_bc, t_bc, tcp_offset_from_body(R_bc, t_bc, t_bt))


def _ident_action(K: int = 16, width: float = W_OPEN) -> np.ndarray:
    a = np.zeros((K, 10))
    a[:, 3:9] = IDENT6
    a[:, 9] = width
    return a


def _delta_action(dT: np.ndarray, K: int = 16, width: float = W_OPEN) -> np.ndarray:
    """Every knot = the same TCP delta (a hold at the displaced pose)."""
    a = np.zeros((K, 10))
    a[:, :9] = mat_to_pose10d(dT)
    a[:, 9] = width
    return a


def _ident_action5(K: int = 16, width: float = W_OPEN) -> np.ndarray:
    """The (K, 5) pos_yaw_width hold: dp = 0, dyaw = 0, width in column 4."""
    a = np.zeros((K, 5))
    a[:, 4] = width
    return a


def _ident_action7(K: int = 16, width: float = W_OPEN) -> np.ndarray:
    """The (K, 7) pos_rpy_width hold: dp = 0, dyaw = droll = dpitch = 0,
    width in column 6."""
    a = np.zeros((K, 7))
    a[:, 6] = width
    return a


def _as5(a7: np.ndarray) -> np.ndarray:
    """The 5-dim action a 7-dim one CONTAINS: cols [0:4, 6]."""
    return a7[:, [0, 1, 2, 3, 6]]


def _Rx(a: float) -> np.ndarray:
    """Rotation about x (a camera-frame x rotation == a camera pitch)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _wrap_arr(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


# =============================================================================
# rot6d / pose10d
# =============================================================================
def test_rot6d_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = _rand_R(rng)
        d6 = mat_to_rot6d(R)
        assert d6.shape == (6,)
        assert np.allclose(d6, R[:2, :].reshape(6))        # ROWS
        R2 = rot6d_to_mat(d6)
        assert np.allclose(R2, R, atol=1e-12), (R, R2)
        # Gram-Schmidt: a scaled / skewed input still decodes to SO(3)
        R3 = rot6d_to_mat(d6 * 3.0 + np.array([0, 0, 0, 0.1, 0, 0]))
        assert np.allclose(R3 @ R3.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R3) - 1.0) < 1e-12
        assert np.allclose(R3[0], R[0])                    # b1 = normalize(a1)
    # batched and pose10d
    Ts = np.stack([_rand_T(rng) for _ in range(7)])
    d9 = mat_to_pose10d(Ts)
    assert d9.shape == (7, 9)
    assert np.allclose(pose10d_to_mat(d9), Ts, atol=1e-12)
    # a 10th (width) column is tolerated and ignored
    d10 = np.concatenate([d9, np.full((7, 1), 0.05)], axis=1)
    assert np.allclose(pose10d_to_mat(d10), Ts, atol=1e-12)
    assert np.allclose(rot6d_to_mat(IDENT6), np.eye(3))


def test_rot6d_parity_upstream():
    """Bit-level parity with umi.common.pose_util (the training encoding)."""
    src = UMI_REPO / "umi" / "common" / "pose_util.py"
    if not src.exists():
        return _skip(f"upstream pose_util.py not found at {src}")
    try:
        import scipy  # noqa: F401  (pose_util imports scipy.spatial.transform)
    except Exception as e:                                   # noqa: BLE001
        return _skip(f"scipy unavailable ({e}); upstream parity not run")
    up = _load_by_path(src, "umi_pose_util_for_parity")
    rng = np.random.default_rng(7)
    d6 = rng.standard_normal((32, 6))
    assert np.array_equal(rot6d_to_mat(d6), up.rot6d_to_mat(d6))
    Ts = np.stack([_rand_T(rng) for _ in range(32)])
    assert np.array_equal(mat_to_rot6d(Ts[:, :3, :3]), up.mat_to_rot6d(Ts[:, :3, :3]))
    assert np.array_equal(mat_to_pose10d(Ts), up.mat_to_pose10d(Ts))
    d9 = up.mat_to_pose10d(Ts)
    assert np.array_equal(pose10d_to_mat(d9), up.pose10d_to_mat(d9))
    # single (unbatched) inputs too
    assert np.array_equal(rot6d_to_mat(d6[0]), up.rot6d_to_mat(d6[0]))
    print("        parity vs upstream pose_util: exact")


# =============================================================================
# pos_yaw_width: label parity, sign, mount, round trip (2026-09-07)
# =============================================================================
def test_pos_yaw_parity_upstream():
    """Bit-level parity with ``umi.common.yaw_action`` — the 5-dim LABEL the
    checkpoint was trained on. A UMI checkout WITHOUT that file FAILS here
    rather than skipping: the label and the flown decode would then be
    unverified against each other. Only a missing checkout skips."""
    if not UMI_REPO.is_dir():
        return _skip(f"UMI repo not found at {UMI_REPO}")
    src = UMI_REPO / "umi" / "common" / "yaw_action.py"
    if not src.exists():
        raise AssertionError("upstream yaw_action.py missing: label/decoder "
                             "agreement unverified")
    up = _load_by_path(src, "umi_yaw_action_for_parity")
    # the constants are the same numbers and the same vocabulary
    assert np.array_equal(up.R_BT_C3, R_BT_C3)
    assert up.R_BT_C3.dtype == R_BT_C3.dtype == np.float64
    assert up.YAW_AXIS_CAM_TILT_DEG == YAW_AXIS_CAM_TILT_DEG == 43.3
    assert up.ACTION_REPR_DIM == ACTION_DIM_BY_REPR, (up.ACTION_REPR_DIM, ACTION_DIM_BY_REPR)
    rng = np.random.default_rng(23)
    # rot_z: single and batched
    psi = rng.uniform(-2.0 * math.pi, 2.0 * math.pi, 32)
    assert np.array_equal(rot_z(psi), up.rot_z(psi))
    assert np.array_equal(rot_z(psi[0]), up.rot_z(psi[0]))
    # encode: both outputs, single and batched, float64 and float32 inputs
    Ts = np.stack([_rand_T(rng) for _ in range(32)])
    for T in (Ts, Ts.astype(np.float32)):
        dp, dyaw = encode_pos_yaw(T)
        dp_u, dyaw_u = up.encode_pos_yaw(T)
        assert dp.dtype == dp_u.dtype == T.dtype and np.array_equal(dp, dp_u)
        assert dyaw.dtype == dyaw_u.dtype == np.float64
        assert dyaw.shape == (32,) and np.array_equal(dyaw, dyaw_u)
        dp1, dyaw1 = encode_pos_yaw(T[0])
        dp1_u, dyaw1_u = up.encode_pos_yaw(T[0])
        assert np.array_equal(dp1, dp1_u) and np.array_equal(dyaw1, dyaw1_u)
        assert np.shape(dyaw1) == ()
        # the position columns ARE the legacy pose10d position columns
        assert np.array_equal(dp, T[:, :3, 3])
        assert np.array_equal(dp, mat_to_pose10d(T)[:, :3])
    # decode: 32 random (dp, dyaw) and the encodings of 32 random T_rel
    dp_r = rng.uniform(-1.0, 1.0, (32, 3))
    dyaw_r = rng.uniform(-math.pi, math.pi, 32)
    assert np.array_equal(decode_pos_yaw(dp_r, dyaw_r), up.decode_pos_yaw(dp_r, dyaw_r))
    dp_e, dyaw_e = up.encode_pos_yaw(Ts)
    assert np.array_equal(decode_pos_yaw(dp_e, dyaw_e), up.decode_pos_yaw(dp_e, dyaw_e))
    assert np.array_equal(decode_pos_yaw(dp_r[0], dyaw_r[0]),
                          up.decode_pos_yaw(dp_r[0], dyaw_r[0]))
    print("        parity vs upstream yaw_action: exact")


def test_pos_yaw_sign_and_mount():
    """The sign and the mount, pinned WITHOUT the upstream checkout:
    (a) a vehicle right turn expressed in the camera frame encodes +;
    (b) compose_plan turns the plan clockwise seen from above (NED +);
    (c) a hand roll about the heading axis at the handheld's 55 deg camera
        pitch leaks < 1.5 deg into dyaw (the rejected optical-axis-azimuth
        label would leak tan(55 deg) x 5 deg = 7.1 deg);
    (d) a pure camera pitch is EXACTLY zero yaw;
    (e) the label constant is the deploy-side mount constant."""
    d = math.radians(10.0)
    # (a) R_rel = R_bt^T Rz(+10 deg) R_bt -> dyaw = +10 deg
    T = np.eye(4)
    T[:3, :3] = R_BT_C3.T @ rot_z(d) @ R_BT_C3
    dp, dyaw = encode_pos_yaw(T)
    assert np.all(dp == 0.0) and abs(float(dyaw) - d) < 1e-12, math.degrees(dyaw)
    # (b) [[0,0,0,0,w],[0,0,0,+10deg,w]*15] at a level anchor heading north
    T_bt = _T_bt()
    anchor = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    a = _ident_action5()
    a[1:, 3] = d
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    assert msg.yaw[0] == 0.0 and abs(msg.yaw[-1] - d) < 1e-12
    assert np.allclose(info["yaw_raw"][1:], d, atol=1e-12)
    heading = rot_zyx(0.0, 0.0, msg.yaw[-1])[:, 0]           # body x in NED
    assert heading[0] > 0.0 and heading[1] > 0.0, "north AND east = clockwise from above"
    assert abs(math.degrees(math.atan2(heading[1], heading[0])) - 10.0) < 1e-9
    # a yaw-only knot holds the JAW and swings the hull about it (A2)
    assert np.allclose(info["p_tcp_raw"], info["p_tcp_raw"][:, :1], atol=1e-12)
    assert np.linalg.norm(msg.p_ned[:, -1] - anchor[:3]) > 0.01
    assert info["dropped_rp_deg"] == 0.0
    # (c) hand roll 5 deg about the heading with the camera pitched 55 deg down
    R_level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # world(Z-up) <- cam, optical axis = world x
    assert np.allclose(R_level @ R_level.T, np.eye(3)) and abs(np.linalg.det(R_level) - 1.0) < 1e-12
    R0 = R_level @ _Rx(math.radians(-55.0))                  # camera pitched down (about cam x)
    assert abs(math.degrees(math.asin(R0[2, 2])) + 55.0) < 1e-9   # optical axis 55 deg below the horizon
    R1 = _Rx(math.radians(5.0)) @ R0                         # the hand rolls about world x = heading
    T = np.eye(4)
    T[:3, :3] = R0.T @ R1
    _dp, dyaw_roll = encode_pos_yaw(T)
    assert abs(math.degrees(dyaw_roll)) < 1.5, math.degrees(dyaw_roll)
    assert abs(math.degrees(dyaw_roll) + 1.04) < 0.01, math.degrees(dyaw_roll)   # [유도: plan 2026-09-07]
    def azimuth(R):
        return math.atan2(R[1, 2], R[0, 2])
    leak_az = math.degrees(azimuth(R1) - azimuth(R0))
    assert abs(leak_az) > 5.0, leak_az                       # ~7.1 deg: why that label was rejected
    print(f"        hand-roll 5 deg @ 55 deg down: dyaw {math.degrees(dyaw_roll):+.3f} deg "
          f"(azimuth label would give {leak_az:+.2f} deg)")
    # (d) a pure camera pitch (about cam x) -> exactly 0
    for deg in (-40.0, -12.0, 12.0, 40.0):
        T = np.eye(4)
        T[:3, :3] = _Rx(math.radians(deg))
        assert encode_pos_yaw(T)[1] == 0.0
    # (e) R_BT_C3 == NavConfig(cam_tilt_deg=43.3).R_t_frd_cam("main")[0]
    R_nav = NavConfig(cam_tilt_deg=YAW_AXIS_CAM_TILT_DEG).R_t_frd_cam("main")[0]
    assert np.allclose(R_nav, R_BT_C3, rtol=0.0, atol=1e-15)
    assert YAW_AXIS_CAM_TILT_DEG == 43.3
    assert np.allclose(R_BT_C3 @ R_BT_C3.T, np.eye(3), atol=1e-14)
    assert abs(np.linalg.det(R_BT_C3) - 1.0) < 1e-12
    elev = math.degrees(math.atan2(-R_BT_C3[2, 2], R_BT_C3[0, 2]))   # optical axis, FRD (z down)
    assert abs(elev + 42.98) < 0.01, elev                    # 43.3 - 0.32 deg built-in cam pitch
    assert R_BT_C3[1, 2] == 0.0                              # no lateral component -> azimuth == body yaw when level
    # ...and the deploy-side mount used by the tests' T_bt is that same matrix
    assert np.array_equal(T_bt[:3, :3], R_BT_C3)


def test_encode_decode_round_trip():
    """decode_pos_yaw then encode_pos_yaw reproduces (dp, wrap(dyaw))."""
    rng = np.random.default_rng(29)
    dp = rng.uniform(-1.0, 1.0, (16, 3))
    dyaw = rng.uniform(-2.0 * math.pi, 2.0 * math.pi, 16)   # beyond +-pi: comes back WRAPPED
    T = decode_pos_yaw(dp, dyaw)
    assert T.shape == (16, 4, 4) and T.dtype == np.float64
    assert np.allclose(T[:, 3], [0.0, 0.0, 0.0, 1.0])
    R = T[:, :3, :3]
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=1e-14)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    # the decoded rotation is a pure yaw about the MOUNT's vertical
    assert np.allclose(R_BT_C3 @ R @ R_BT_C3.T, rot_z(dyaw), atol=1e-14)
    dp2, dyaw2 = encode_pos_yaw(T)
    assert np.array_equal(dp2, dp)                           # copied through bit-identical
    assert np.allclose(dyaw2, _wrap_arr(dyaw), atol=1e-12)
    assert np.all(np.abs(dyaw2) <= math.pi)
    # single input, and zero is the identity
    T1 = decode_pos_yaw(dp[0], dyaw[0])
    assert T1.shape == (4, 4)
    d1, y1 = encode_pos_yaw(T1)
    assert np.array_equal(d1, dp[0]) and abs(float(y1) - _wrap_arr(dyaw)[0]) < 1e-12
    assert np.allclose(decode_pos_yaw(np.zeros(3), 0.0), np.eye(4))


# =============================================================================
# eta / transforms
# =============================================================================
def test_T_from_eta_matches_rot_zyx():
    rng = np.random.default_rng(3)
    for _ in range(20):
        eta = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-1.2, 1.2, 3)])
        T = T_from_eta(eta)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], rot_zyx(eta[3], eta[4], eta[5]))
        assert np.allclose(T[:3, 3], eta[:3])
        assert np.allclose(T[3], [0, 0, 0, 1])
        r, p, y = rpy_of(T[:3, :3])
        assert np.allclose([r, p, y], eta[3:], atol=1e-12), (r, p, y, eta[3:])
        assert abs(yaw_of(T) - eta[5]) < 1e-12


def test_tcp_offset_from_body_roundtrip():
    """A3: the camera-frame offset that lands the TCP on the body-frame jaw."""
    R_bc, t_bc = _c3_extrinsic()
    jaw_frd = S_FLU_FRD @ np.array([0.502, 0.0, -0.17])   # cam_t_flu + [0.196,0,-0.275] (2026-09-08)
    off = tcp_offset_from_body(R_bc, t_bc, jaw_frd)
    T = T_body_tcp(R_bc, t_bc, off)
    assert np.allclose(T[:3, 3], jaw_frd, atol=1e-12)
    assert np.allclose(T[:3, :3], R_bc)                        # TCP shares camera axes
    # the handheld's offset [0.0355, 0.1293, 0.3186] is NOT this one
    hand = T_body_tcp(R_bc, t_bc, [0.0355, 0.1293, 0.3186])
    assert np.linalg.norm(hand[:3, 3] - jaw_frd) > 0.05
    # T_body_tcp is the stated product [[R,t],[0,1]] @ [[I,off],[0,1]]
    A = np.eye(4); A[:3, :3] = R_bc; A[:3, 3] = t_bc
    B = np.eye(4); B[:3, 3] = off
    assert np.allclose(T, A @ B)


# =============================================================================
# low-dim observation
# =============================================================================
def test_lowdim_obs_last_row_identity_and_relative():
    rng = np.random.default_rng(11)
    T_bt = _T_bt()
    eta_prev = np.array([1.0, 0.5, 0.3, 0.02, -0.03, 0.4])
    eta_now = eta_prev + np.array([0.01, -0.004, 0.002, 0.01, 0.005, 0.03])
    eta_start = np.array([0.8, 0.4, 0.3, 0.0, 0.0, 0.1])
    obs = lowdim_obs(eta_prev, eta_now, eta_start, T_bt, 0.060, 0.058)
    assert set(obs) == {"robot0_eef_pos", "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start"}
    assert obs["robot0_eef_pos"].shape == (2, 3)
    assert obs["robot0_eef_rot_axis_angle"].shape == (2, 6)
    assert obs["robot0_gripper_width"].shape == (2, 1)
    assert obs["robot0_eef_rot_axis_angle_wrt_start"].shape == (2, 6)
    assert all(v.dtype == np.float32 for v in obs.values())
    # last row: relative to itself -> zero / identity
    assert np.allclose(obs["robot0_eef_pos"][1], 0.0)
    assert np.allclose(obs["robot0_eef_rot_axis_angle"][1], IDENT6)
    assert np.allclose(obs["robot0_gripper_width"][:, 0], [0.060, 0.058])
    # first row == mat_to_pose10d(inv(T_latest) @ T_prev), computed independently
    T_prev = T_from_eta(eta_prev) @ T_bt
    T_now = T_from_eta(eta_now) @ T_bt
    rel = np.linalg.inv(T_now) @ T_prev
    assert np.allclose(obs["robot0_eef_pos"][0], rel[:3, 3], atol=1e-6)
    assert np.allclose(obs["robot0_eef_rot_axis_angle"][0],
                       mat_to_rot6d(rel[:3, :3]), atol=1e-6)
    assert np.linalg.norm(rel[:3, 3]) > 1e-3        # the motion cue is not zero
    # upstream convert_pose_mat_rep('relative') when the checkout is present
    src = UMI_REPO / "diffusion_policy" / "common" / "pose_repr_util.py"
    if src.exists():
        up = _load_by_path(src, "dp_pose_repr_util_for_parity")
        ref = up.convert_pose_mat_rep(np.stack([T_prev, T_now]), T_now,
                                      pose_rep="relative")
        d9 = mat_to_pose10d(ref)
        assert np.allclose(obs["robot0_eef_pos"], d9[:, :3], atol=1e-6)
        assert np.allclose(obs["robot0_eef_rot_axis_angle"], d9[:, 3:], atol=1e-6)
        print("        parity vs upstream convert_pose_mat_rep('relative'): ok")
    else:
        _skip(f"{src} not found; relative-rep parity computed locally only")
    # datum invariance: shifting/rotating the whole world changes nothing
    T_shift = T_from_eta([2.0, -1.0, 0.5, 0.0, 0.0, 1.1])
    def shifted(e):
        T = T_shift @ T_from_eta(e)
        r, p, y = rpy_of(T[:3, :3])
        return np.array([*T[:3, 3], r, p, y])
    obs2 = lowdim_obs(shifted(eta_prev), shifted(eta_now), shifted(eta_start),
                      T_bt, 0.060, 0.058)
    for k in obs:
        assert np.allclose(obs[k], obs2[k], atol=1e-5), k
    # no start pose -> refuse (the worker must not infer before START)
    try:
        lowdim_obs(eta_prev, eta_now, None, T_bt, 0.06, 0.06)
        assert False, "eta_start=None must raise"
    except ValueError:
        pass
    del rng


def test_lowdim_obs_wrt_start_rotation_only():
    T_bt = _T_bt()
    eta_prev = np.array([1.0, 0.5, 0.3, 0.0, 0.0, 0.4])
    eta_now = np.array([1.02, 0.5, 0.3, 0.0, 0.0, 0.45])
    start_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
    start_b = start_a + np.array([5.0, -3.0, 1.0, 0.0, 0.0, 0.0])   # moved, same attitude
    start_c = start_a + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])    # same place, yawed
    oa = lowdim_obs(eta_prev, eta_now, start_a, T_bt, 0.06, 0.06)
    ob = lowdim_obs(eta_prev, eta_now, start_b, T_bt, 0.06, 0.06)
    oc = lowdim_obs(eta_prev, eta_now, start_c, T_bt, 0.06, 0.06)
    k = "robot0_eef_rot_axis_angle_wrt_start"
    assert np.allclose(oa[k], ob[k]), "start POSITION leaked into wrt_start"
    assert not np.allclose(oa[k], oc[k]), "start ROTATION must matter"
    # value = rot6d of (inv(T_start) @ T_i)[:3,:3] for each row
    T_start = T_from_eta(start_a) @ T_bt
    for i, e in enumerate((eta_prev, eta_now)):
        T_i = T_from_eta(e) @ T_bt
        want = mat_to_rot6d((np.linalg.inv(T_start) @ T_i)[:3, :3])
        assert np.allclose(oa[k][i], want, atol=1e-6)
    # the relative rows do not depend on the start at all
    for kk in ("robot0_eef_pos", "robot0_eef_rot_axis_angle"):
        assert np.allclose(oa[kk], oc[kk])


# =============================================================================
# A2: TCP-space composition + leveled body position
# =============================================================================
def test_pure_pitch_delta_keeps_tcp_position():
    """A pure-pitch TCP delta: p_tcp unchanged; the LEVELED body position
    stays at the anchor; the un-leveled body composition (v1 D3) would have
    moved the body by |t_bt|·2·sin(pitch/2) ≈ |t_bt|·sin(pitch)."""
    T_bt = _T_bt()
    R_bt, t_bt = T_bt[:3, :3], T_bt[:3, 3]
    anchor = np.array([1.0, -0.5, 0.8, 0.0, 0.0, 0.7])       # level, yawed
    rp_level = anchor[3:5]
    theta = math.radians(10.0)
    # pitch in the TCP frame = about the camera x axis (right)... expressed
    # so that the IMPLIED body rotation is a pure body pitch Ry(theta):
    Ry = rot_zyx(0.0, theta, 0.0)
    dT = np.eye(4)
    dT[:3, :3] = R_bt.T @ Ry @ R_bt
    a = _delta_action(dT)
    msg, info = compose_plan(a, anchor, rp_level, T_bt, OBS_DT, KNOT_DT,
                             t0=0.0, plan_id=1, obs_t_rel=0.0,
                             w_open=W_OPEN, w_closed=W_CLOSED)
    T_tcp_a = T_from_eta(anchor) @ T_bt
    p_tcp_a = T_tcp_a[:3, 3]
    # (1) the TCP does not move (rotation-only delta about the TCP origin)
    assert np.allclose(info["p_tcp_raw"], p_tcp_a[:, None], atol=1e-12)
    assert np.allclose(info["p_tcp"], p_tcp_a[:, None], atol=1e-12)
    # (2) yaw is unchanged and the leveled body position == the anchor body
    assert np.allclose(info["yaw_raw"], anchor[5], atol=1e-12)
    assert np.allclose(msg.p_ned, anchor[:3, None], atol=1e-12)
    # (3) the dropped attitude is reported: 10 deg of pitch, no roll
    assert abs(info["dropped_rp_deg"] - 10.0) < 1e-9, info["dropped_rp_deg"]
    assert np.allclose(info["rp_dropped_deg"][0], 0.0, atol=1e-9)
    assert np.allclose(info["rp_dropped_deg"][1], 10.0, atol=1e-9)
    # (4) what leveling absorbed: the un-leveled body would sit at
    #     p_tcp - R_body_k @ t_bt, |t_bt|·2·sin(theta/2) from the anchor
    R_body_k = T_tcp_a[:3, :3] @ dT[:3, :3] @ R_bt.T
    p_body_unleveled = p_tcp_a - R_body_k @ t_bt
    moved = np.linalg.norm(p_body_unleveled - anchor[:3])
    lever = math.hypot(t_bt[0], t_bt[2])                     # component ⊥ pitch axis
    assert abs(moved - lever * 2.0 * math.sin(theta / 2.0)) < 1e-12, (moved, lever)
    assert abs(moved - np.linalg.norm(t_bt) * math.sin(theta)) < 0.01 * moved
    assert moved > 0.03, moved                                # ~0.079 m at 10 deg [유도]


def test_tcp_forward_delta_with_tilted_c3():
    """+z in the TCP (camera forward) with the C3 tilted 43.3 deg down moves
    both the TCP and the body forward AND down, by the same vector."""
    T_bt = _T_bt(43.3)
    R_bc, _t_bc = _c3_extrinsic(43.3)
    anchor = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])       # level, heading north
    dz = 0.10
    dT = np.eye(4); dT[2, 3] = dz
    a = _delta_action(dT)
    a[0, :9] = mat_to_pose10d(np.eye(4))                     # knot 0 = "now", then a step
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, OBS_DT, KNOT_DT,
                             t0=0.0, plan_id=2, obs_t_rel=0.0,
                             w_open=W_OPEN, w_closed=W_CLOSED)
    d_tcp = info["p_tcp_raw"][:, -1] - info["p_tcp_raw"][:, 0]
    d_body = msg.p_ned[:, -1] - msg.p_ned[:, 0]
    want = dz * R_bc[:, 2]                                   # camera +z in body == NED here
    assert np.allclose(d_tcp, want, atol=1e-12), (d_tcp, want)
    assert np.allclose(d_body, d_tcp, atol=1e-12), "rotation-free delta: body follows the TCP"
    assert d_body[0] > 0.0 and d_body[2] > 0.0 and abs(d_body[1]) < 1e-9, d_body   # forward, DOWN
    ang = math.degrees(math.atan2(d_body[2], d_body[0]))
    assert abs(ang - 43.3) < 0.5, ang                        # -0.32 deg built-in cam pitch
    assert abs(np.linalg.norm(d_body) - dz) < 1e-12
    assert np.allclose(msg.yaw, 0.0) and info["dropped_rp_deg"] < 1e-9
    # knot 0 (pinned) is the anchor body position; the plan is a hold
    # at the displaced pose from knot 1 on (all raw knots equal)
    assert np.allclose(msg.p_ned[:, 0], anchor[:3])
    assert np.allclose(msg.p_ned[:, 1:], (anchor[:3] + want)[:, None])


def test_tcp_forward_delta_with_tilted_c3_5d():
    """The (K, 5) mirror: a[:, 2] = dz (camera +z), a[0, 2] = 0 -> the same
    forward-AND-down 43.3 deg step, yaw 0, nothing dropped."""
    T_bt = _T_bt(43.3)
    R_bc, _t_bc = _c3_extrinsic(43.3)
    anchor = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])       # level, heading north
    dz = 0.10
    a = _ident_action5()
    a[:, 2] = dz
    a[0, 2] = 0.0                                            # knot 0 = "now", then a step
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **{**_KW, "plan_id": 2})
    d_tcp = info["p_tcp_raw"][:, -1] - info["p_tcp_raw"][:, 0]
    d_body = msg.p_ned[:, -1] - msg.p_ned[:, 0]
    want = dz * R_bc[:, 2]
    assert np.allclose(d_tcp, want, atol=1e-12), (d_tcp, want)
    assert np.allclose(d_body, d_tcp, atol=1e-12), "rotation-free delta: body follows the TCP"
    assert d_body[0] > 0.0 and d_body[2] > 0.0 and abs(d_body[1]) < 1e-9, d_body   # forward, DOWN
    ang = math.degrees(math.atan2(d_body[2], d_body[0]))
    assert abs(ang - 43.3) < 0.5, ang
    assert abs(np.linalg.norm(d_body) - dz) < 1e-12
    assert np.allclose(msg.yaw, 0.0) and info["dropped_rp_deg"] == 0.0
    assert info["action_repr"] == ACTION_REPR_POS_YAW_WIDTH
    assert np.allclose(msg.p_ned[:, 0], anchor[:3])
    assert np.allclose(msg.p_ned[:, 1:], (anchor[:3] + want)[:, None])
    # the same step through the legacy encoding lands on the same plan
    dT = np.eye(4)
    dT[2, 3] = dz
    a10 = _delta_action(dT)
    a10[0, :9] = mat_to_pose10d(np.eye(4))
    msg10, _ = compose_plan(a10, anchor, anchor[3:5], T_bt, **{**_KW, "plan_id": 2})
    assert np.allclose(msg10.p_ned, msg.p_ned, atol=1e-12)
    assert np.allclose(msg10.yaw, msg.yaw, atol=1e-12)


# =============================================================================
# compose_plan contract
# =============================================================================
def test_compose_plan_contract_legacy10():
    """The legacy (K, 10) pose10d contract (the 2026-09-01 checkpoint)."""
    T_bt = _T_bt()
    anchor = np.array([0.3, -0.2, 0.9, 0.03, -0.02, -2.9])   # slightly un-level
    a = _ident_action()
    # a forward drift along TCP +z at 0.05 m/s and a width ramp open -> closed
    a[:, 2] = 0.05 * np.arange(16) * OBS_DT
    a[:, 9] = np.linspace(W_OPEN + 0.01, W_CLOSED - 0.01, 16)
    a[8, 9] = 0.5 * (W_OPEN + W_CLOSED)
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, OBS_DT, KNOT_DT,
                             t0=12.5, plan_id=42, obs_t_rel=12.5,
                             w_open=W_OPEN, w_closed=W_CLOSED)
    assert isinstance(msg, PlanMsg)
    assert msg.plan_id == 42 and msg.t0 == 12.5
    assert msg.obs_t == 12.5, "obs_t must be set (a live plan without it is refused)"
    assert msg.dt == KNOT_DT
    assert msg.n_knots == 6 and info["n_knots"] == 6 and info["n_raw"] == 16
    assert msg.p_ned.shape == (3, 6) and msg.yaw.shape == (6,) and msg.g.shape == (6,)
    assert abs(msg.t_end - (12.5 + 1.0)) < 1e-9
    assert np.all(np.isfinite(msg.p_ned)) and np.all(np.isfinite(msg.yaw))
    # g in [0, 1] from widths: above open -> 1, below closed -> 0, mid -> 0.5
    assert np.all(msg.g >= 0.0) and np.all(msg.g <= 1.0)
    assert msg.g[0] == 1.0                                   # pinned raw knot 0 (> open)
    assert msg.g[-1] < 0.05                                  # tail below closed -> ~0
    raw_g = np.clip((a[:, 9] - W_CLOSED) / (W_OPEN - W_CLOSED), 0, 1)
    assert abs(raw_g[8] - 0.5) < 1e-12
    # knot 0 == the anchor TCP projection: p_tcp[:,0] is the anchor's TCP
    # pose and (rp_level == anchor roll/pitch) the body knot is the anchor
    T_tcp_a = T_from_eta(anchor) @ T_bt
    assert np.allclose(info["p_tcp"][:, 0], T_tcp_a[:3, 3], atol=1e-12)
    assert np.allclose(info["anchor_tcp"][0], T_tcp_a[:3, 3])
    assert np.allclose(msg.p_ned[:, 0], anchor[:3], atol=1e-12)
    assert abs(math.atan2(math.sin(msg.yaw[0] - anchor[5]),
                          math.cos(msg.yaw[0] - anchor[5]))) < 1e-12
    assert info["dropped_rp_deg"] < 1e-9                     # identity rotations
    assert info["p_tcp"].shape == (3, 6)
    # the plan yaw is continuous (unwrapped) although the anchor is near -pi
    assert np.max(np.abs(np.diff(msg.yaw))) < 1e-9
    # a different rp_level moves knot 0 off the anchor by the lever arm (A2)
    msg2, _ = compose_plan(a, anchor, (0.0, 0.0), T_bt, OBS_DT, KNOT_DT,
                           t0=0.0, plan_id=43, obs_t_rel=0.0,
                           w_open=W_OPEN, w_closed=W_CLOSED)
    assert np.linalg.norm(msg2.p_ned[:, 0] - anchor[:3]) > 1e-4
    # ...but the TCP it implies is the same one (only the leveling changed)
    t_bt = T_bt[:3, 3]
    p_tcp2 = np.stack([msg2.p_ned[:, k] + rot_zyx(0.0, 0.0, msg2.yaw[k]) @ t_bt
                       for k in range(msg2.n_knots)], axis=1)
    assert np.allclose(p_tcp2, info["p_tcp"], atol=1e-12)


def test_compose_plan_contract_5d():
    """The (K, 5) pos_yaw_width twin of the legacy contract."""
    T_bt = _T_bt()
    anchor = np.array([0.3, -0.2, 0.9, 0.03, -0.02, -2.9])   # slightly un-level
    a = _ident_action5()
    # a forward drift along TCP +z at 0.05 m/s, a yaw ramp 0 -> -0.5 rad that
    # carries anchor + dyaw ACROSS -pi, and a width ramp open -> closed
    a[:, 2] = 0.05 * np.arange(16) * OBS_DT
    a[:, 3] = np.linspace(0.0, -0.5, 16)
    a[:, 4] = np.linspace(W_OPEN + 0.01, W_CLOSED - 0.01, 16)
    a[8, 4] = 0.5 * (W_OPEN + W_CLOSED)
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, OBS_DT, KNOT_DT,
                             t0=12.5, plan_id=42, obs_t_rel=12.5,
                             w_open=W_OPEN, w_closed=W_CLOSED)
    assert isinstance(msg, PlanMsg)
    assert msg.plan_id == 42 and msg.t0 == 12.5
    assert msg.obs_t == 12.5, "obs_t must be set (a live plan without it is refused)"
    assert msg.dt == KNOT_DT
    assert msg.n_knots == 6 and info["n_knots"] == 6 and info["n_raw"] == 16
    assert msg.p_ned.shape == (3, 6) and msg.yaw.shape == (6,) and msg.g.shape == (6,)
    assert abs(msg.t_end - (12.5 + 1.0)) < 1e-9
    assert np.all(np.isfinite(msg.p_ned)) and np.all(np.isfinite(msg.yaw))
    # the 5-dim specifics: named, and nothing dropped -- structurally
    assert info["action_repr"] == ACTION_REPR_POS_YAW_WIDTH == "pos_yaw_width"
    assert info["dropped_rp_deg"] == 0.0
    assert info["rp_dropped_deg"].shape == (2, 16) and np.all(info["rp_dropped_deg"] == 0)
    # g in [0, 1] from column 4: above open -> 1, below closed -> 0, mid -> 0.5
    assert np.all(msg.g >= 0.0) and np.all(msg.g <= 1.0)
    assert msg.g[0] == 1.0                                   # pinned raw knot 0 (> open)
    assert msg.g[-1] == 0.0                                  # pinned raw knot 15 (< closed)
    raw_g = np.clip((a[:, 4] - W_CLOSED) / (W_OPEN - W_CLOSED), 0, 1)
    assert abs(raw_g[8] - 0.5) < 1e-12
    # knot 0 == the anchor TCP projection: p_tcp[:,0] is the anchor's TCP
    # pose and (rp_level == anchor roll/pitch) the body knot is the anchor
    T_tcp_a = T_from_eta(anchor) @ T_bt
    assert np.allclose(info["p_tcp"][:, 0], T_tcp_a[:3, 3], atol=1e-12)
    assert np.allclose(info["anchor_tcp"][0], T_tcp_a[:3, 3])
    assert np.allclose(msg.p_ned[:, 0], anchor[:3], atol=1e-12)
    assert msg.yaw[0] == anchor[5]                           # pinned; dyaw_0 = 0
    assert info["p_tcp"].shape == (3, 6)
    # yaw is anchor + dyaw, UNWRAPPED: the ramp carries the plan through -pi
    # without a jump to +pi (last knot pinned to raw knot 15)
    assert np.allclose(info["yaw_raw"], anchor[5] + a[:, 3], atol=1e-12)
    assert np.all(np.diff(msg.yaw) < 0.0) and np.max(np.abs(np.diff(msg.yaw))) < 0.11
    assert msg.yaw[-1] < -math.pi and abs(msg.yaw[-1] - (anchor[5] - 0.5)) < 1e-12
    # a different rp_level moves knot 0 off the anchor by the lever arm (A2)
    msg2, info2 = compose_plan(a, anchor, (0.0, 0.0), T_bt, OBS_DT, KNOT_DT,
                               t0=0.0, plan_id=43, obs_t_rel=0.0,
                               w_open=W_OPEN, w_closed=W_CLOSED)
    assert np.linalg.norm(msg2.p_ned[:, 0] - anchor[:3]) > 1e-4
    # ...but the TCP it implies is the same one (only the leveling changed).
    # Checked on the RAW knots and the pinned knot 0: with a yaw RAMP the
    # window mean of the leveled body knots is not the leveling of the mean
    # (the legacy test's reconstruction of every resampled knot works only
    # because its yaw is constant).
    t_bt = T_bt[:3, 3]
    p_tcp2_raw = info2["p_body_raw"] + np.stack(
        [rot_zyx(0.0, 0.0, info2["yaw_raw"][k]) @ t_bt for k in range(16)], axis=1)
    assert np.allclose(p_tcp2_raw, info["p_tcp_raw"], atol=1e-12)
    assert np.array_equal(info2["p_tcp"], info["p_tcp"])     # the TCP track ignores rp_level
    assert np.allclose(msg2.p_ned[:, 0] + rot_zyx(0.0, 0.0, msg2.yaw[0]) @ t_bt,
                       info["p_tcp"][:, 0], atol=1e-12)


def test_compose_plan_5d_yaw_is_anchor_plus_dyaw():
    """5-dim: yaw_k = anchor_yaw + dyaw_k EXACTLY at a rolled/pitched anchor
    -- and that is the label's definition made concrete: the azimuth of the
    optical axis of rot_zyx(roll_lvl, pitch_lvl, yaw_k) @ R_bc moves by
    exactly dyaw_k, because rot_zyx(r, p, y + d) = Rz(d) @ rot_zyx(r, p, y)."""
    T_bt = _T_bt()
    R_bc = T_bt[:3, :3]
    assert np.array_equal(R_bc, R_BT_C3)
    anchor = np.array([0.3, -0.2, 0.9, math.radians(5.0), math.radians(5.0), 0.4])
    dyaw = np.linspace(0.0, 0.3, 16)
    a = _ident_action5()
    a[:, 3] = dyaw
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    assert np.allclose(info["yaw_raw"], anchor[5] + dyaw, atol=1e-12)
    assert abs(msg.yaw[0] - anchor[5]) < 1e-12               # pinned knots
    assert abs(msg.yaw[-1] - (anchor[5] + 0.3)) < 1e-12
    assert info["dropped_rp_deg"] == 0.0

    def azimuth(R):
        return math.atan2(R[1, 2], R[0, 2])                  # of the optical axis (column 2)

    az = np.array([azimuth(rot_zyx(anchor[3], anchor[4], info["yaw_raw"][k]) @ R_bc)
                   for k in range(16)])
    assert np.allclose(_wrap_arr(az - az[0]), dyaw, atol=1e-12)
    # at the tilted anchor the azimuth itself is NOT the yaw (the 5 deg
    # attitude bends the tilted optical axis sideways) -- only its CHANGE is
    # dyaw, which is all the label and the decode need...
    assert abs(az[0] - anchor[5]) > 0.01
    # ...whereas at a level anchor the azimuth IS the body yaw (R_bc has no
    # lateral component), so azimuth_k == yaw_k
    level = anchor.copy()
    level[3:5] = 0.0
    _msg_l, info_l = compose_plan(a, level, level[3:5], T_bt, **_KW)
    az_l = np.array([azimuth(rot_zyx(0.0, 0.0, info_l["yaw_raw"][k]) @ R_bc) for k in range(16)])
    assert np.allclose(az_l, info_l["yaw_raw"], atol=1e-12)


def test_compose_plan_5d_equals_legacy_for_yaw_only_rotation():
    """The 5-dim decode IS the legacy pose10d decode for a yaw-only relative
    rotation: (a) at a LEVEL anchor with the label's own construction
    (decode_pos_yaw: R_bt^T Rz(d) R_bt); (b) at a tilted, LEASHED anchor when
    the legacy rotation is about the WORLD vertical expressed in the anchor
    camera frame (R_a^T Rz(d) R_a, R_a = R_ned_tcp_a). (c) The mount-vertical
    construction at a tilted anchor is NOT equal: the legacy path reads a
    rotation about the (tilted) body z as roll/pitch and drops it -- the
    boundary the 5-dim label draws, documented here as an inequality."""
    T_bt = _T_bt()
    K = 16
    dyaw = np.linspace(0.0, 0.3, K)
    dp = np.zeros((K, 3))
    dp[:, 0] = 0.01 * np.arange(K) * OBS_DT
    dp[:, 2] = 0.05 * np.arange(K) * OBS_DT
    width = np.linspace(W_OPEN, W_CLOSED, K)
    a5 = np.concatenate([dp, dyaw[:, None], width[:, None]], axis=1)

    def legacy_from(T_rel):
        return np.concatenate([mat_to_pose10d(T_rel), width[:, None]], axis=1)

    def same(anchor, a10):
        m5, i5 = compose_plan(a5, anchor, anchor[3:5], T_bt, **_KW)
        m10, i10 = compose_plan(a10, anchor, anchor[3:5], T_bt, **_KW)
        assert i5["action_repr"] == ACTION_REPR_POS_YAW_WIDTH
        assert i10["action_repr"] == ACTION_REPR_POSE10D
        for key in ("p_body_raw", "yaw_raw", "p_tcp_raw"):
            assert np.allclose(i5[key], i10[key], atol=1e-12), key
        assert np.allclose(m5.p_ned, m10.p_ned, atol=1e-12)
        assert np.allclose(m5.yaw, m10.yaw, atol=1e-12)
        assert np.allclose(m5.g, m10.g, atol=1e-12)
        assert i10["dropped_rp_deg"] < 1e-9, i10["dropped_rp_deg"]
        assert i5["dropped_rp_deg"] == 0.0
        return i5, i10

    # (a) level anchor, the label's own construction
    level = np.array([0.3, -0.2, 0.9, 0.0, 0.0, -2.9])
    a10_mount = legacy_from(decode_pos_yaw(dp, dyaw))
    same(level, a10_mount)
    # (b) tilted, LEASHED anchor (rp = 5 deg, yaw 2.9: the ramp crosses +pi),
    #     legacy rotation about the WORLD vertical in the anchor camera frame
    meas = np.array([0.3, -0.2, 0.9, math.radians(5.0), math.radians(5.0), 2.9])
    anchor, linfo = leash_anchor(meas, meas[:3] + np.array([0.3, 0.1, 0.0]), meas[5],
                                 0.05, math.radians(10.0))
    assert linfo["clipped_pos"] and np.allclose(anchor[3:], meas[3:])
    assert np.linalg.norm(anchor[:3] - meas[:3]) > 0.04
    R_a = (T_from_eta(anchor) @ T_bt)[:3, :3]
    T_rel = np.zeros((K, 4, 4))
    T_rel[:, 3, 3] = 1.0
    for k in range(K):
        T_rel[k, :3, :3] = R_a.T @ rot_z(dyaw[k]) @ R_a
        T_rel[k, :3, 3] = dp[k]
    i5b, _i10b = same(anchor, legacy_from(T_rel))
    assert i5b["yaw_raw"][-1] > math.pi                      # unwrapped past +pi in BOTH
    # (c) the mount-vertical construction at the tilted anchor: NOT equal
    m5c, i5c = compose_plan(a5, anchor, anchor[3:5], T_bt, **_KW)
    m10c, i10c = compose_plan(a10_mount, anchor, anchor[3:5], T_bt, **_KW)
    yaw_diff = float(np.max(np.abs(i10c["yaw_raw"] - i5c["yaw_raw"])))
    assert yaw_diff > 1e-4, yaw_diff                         # ~6.6e-4 rad at 5 deg / 0.3 rad
    assert i10c["dropped_rp_deg"] > 1.0, i10c["dropped_rp_deg"]   # ~1.7 deg read as roll/pitch
    assert np.max(np.abs(m10c.p_ned - m5c.p_ned)) > 1e-5    # lever arm ~0.3 mm
    print(f"        (c) mount-vertical at a 5 deg anchor: yaw diff "
          f"{math.degrees(yaw_diff):.3f} deg, legacy dropped {i10c['dropped_rp_deg']:.2f} deg")


def test_compose_plan_rejects_degenerate_action():
    T_bt = _T_bt()
    anchor = np.zeros(6)
    kw = dict(obs_dt=OBS_DT, knot_dt=KNOT_DT, t0=0.0, plan_id=1, obs_t_rel=0.0,
              w_open=W_OPEN, w_closed=W_CLOSED)
    bad = _ident_action(); bad[5, 1] = np.nan
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "NaN must raise"
    except ValueError as e:
        assert "non-finite" in str(e)
    bad = _ident_action(); bad[3, 3:6] = 0.0                 # a1 vanishes
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "zero rot6d must raise"
    except ValueError as e:
        assert "degenerate rot6d" in str(e) and "knot 3" in str(e)
    try:
        compose_plan(_ident_action()[:, :9], anchor, (0, 0), T_bt, **kw)
        assert False, "(K, 9) must raise"
    except ValueError:
        pass
    try:
        compose_plan(_ident_action(), anchor, (0, 0), T_bt,
                     **{**kw, "w_open": W_CLOSED})
        assert False, "w_open <= w_closed must raise"
    except ValueError:
        pass
    # ---- (K, 5) pos_yaw_width
    bad = _ident_action5(); bad[5, 1] = np.nan
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "(K, 5) NaN must raise"
    except ValueError as e:
        assert "non-finite" in str(e)
    bad = _ident_action5(); bad[4, 3] = 3.5                  # |dyaw| > pi at knot 4
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "|dyaw| > pi must raise"
    except ValueError as e:
        assert "dyaw" in str(e) and "knot 4" in str(e), str(e)
    ok = _ident_action5(); ok[4, 3] = DYAW_MAX_RAD           # the bound is inclusive
    compose_plan(ok, anchor, (0, 0), T_bt, **kw)
    for w in (4, 6, 8):                                      # 7 is pos_rpy_width now
        try:
            compose_plan(np.zeros((16, w)), anchor, (0, 0), T_bt, **kw)
            assert False, f"(K, {w}) must raise"
        except ValueError as e:
            assert "(K, 5)" in str(e) and "(K, 7)" in str(e) and "(K, 10)" in str(e), str(e)
    # ---- (K, 7) pos_rpy_width (2026-09-26)
    bad = _ident_action7(); bad[5, 1] = np.nan
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "(K, 7) NaN must raise"
    except ValueError as e:
        assert "non-finite" in str(e)
    bad = _ident_action7(); bad[4, 3] = 3.5                  # |dyaw| > pi at knot 4
    try:
        compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "|dyaw| > pi must raise"
    except ValueError as e:
        assert "dyaw" in str(e) and "knot 4" in str(e), str(e)
    for col in (4, 5):                                       # |droll| / |dpitch| > pi/2
        bad = _ident_action7(); bad[6, col] = -(DRP_MAX_RAD + 1e-6)
        try:
            compose_plan(bad, anchor, (0, 0), T_bt, **kw); assert False, "|drp| > pi/2 must raise"
        except ValueError as e:
            assert "droll" in str(e) and "dpitch" in str(e) and "knot 6" in str(e), str(e)
    ok = _ident_action7(); ok[6, 4] = DRP_MAX_RAD; ok[7, 5] = -DRP_MAX_RAD   # inclusive
    compose_plan(ok, anchor, (0, 0), T_bt, **kw)
    # track_rp needs the 7-dim action -- the arm gate refuses the pairing,
    # and compose refuses it too (a 5-dim ckpt has no attitude columns)
    for arr in (_ident_action5(), _ident_action()):
        try:
            compose_plan(arr, anchor, (0, 0), T_bt, **kw, track_rp=True)
            assert False, "track_rp with a non-7-dim action must raise"
        except ValueError as e:
            assert "track_rp" in str(e), str(e)
    try:
        compose_plan(_ident_action7(), anchor, (0, 0), T_bt, **kw, track_rp=True,
                     rp_max_rad=-0.1)
        assert False, "negative rp_max_rad must raise"
    except ValueError:
        pass
    try:
        compose_plan(_ident_action7(), anchor, (0, 0), T_bt, **kw, action_repr="pos_yaw_width")
        assert False, "7-dim array under a pos_yaw_width contract must raise"
    except ValueError as e:
        assert "action_repr" in str(e), str(e)
    _, i7 = compose_plan(_ident_action7(), anchor, (0, 0), T_bt, **kw,
                         action_repr=ACTION_REPR_POS_RPY_WIDTH)
    assert i7["action_repr"] == ACTION_REPR_POS_RPY_WIDTH == "pos_rpy_width"
    assert action_repr_of(7) == ACTION_REPR_POS_RPY_WIDTH
    # contract mismatch: the checkpoint says one thing, the array another
    try:
        compose_plan(_ident_action5(), anchor, (0, 0), T_bt, **kw, action_repr="pose10d")
        assert False, "5-dim array under a pose10d contract must raise"
    except ValueError as e:
        assert "action_repr" in str(e), str(e)
    try:
        compose_plan(_ident_action(), anchor, (0, 0), T_bt, **kw, action_repr="pos_yaw_width")
        assert False, "10-dim array under a pos_yaw_width contract must raise"
    except ValueError as e:
        assert "action_repr" in str(e), str(e)
    # ...and agreement passes on both widths
    _, i5 = compose_plan(_ident_action5(), anchor, (0, 0), T_bt, **kw,
                         action_repr=ACTION_REPR_POS_YAW_WIDTH)
    _, i10 = compose_plan(_ident_action(), anchor, (0, 0), T_bt, **kw,
                          action_repr=ACTION_REPR_POSE10D)
    assert i5["action_repr"] == ACTION_REPR_POS_YAW_WIDTH and i10["action_repr"] == ACTION_REPR_POSE10D
    # the width vocabulary is state's
    assert action_repr_of(5) == ACTION_REPR_POS_YAW_WIDTH and action_repr_of(10) == ACTION_REPR_POSE10D
    for w in (4, 6, 8, 9):
        try:
            action_repr_of(w); assert False, w
        except ValueError:
            pass
    # a 1-D array is not an action
    try:
        compose_plan(np.zeros(5), anchor, (0, 0), T_bt, **kw); assert False, "1-D must raise"
    except ValueError:
        pass


# =============================================================================
# A5: resampling
# =============================================================================
def _accel_peak(p: np.ndarray, dt: float) -> float:
    """The filter's finite-difference accel gate (plan_stream.py step 5)."""
    v = np.diff(p, axis=1) / dt
    a = np.diff(v, axis=1) / dt
    return float(np.linalg.norm(a, axis=0).max())


def test_resample_knots_pins_and_smooths():
    rng = np.random.default_rng(5)
    K = 16
    t = np.arange(K) * OBS_DT
    p_clean = np.vstack([0.06 * t, 0.02 * t, np.zeros(K)])   # 0.063 m/s straight
    sigma = 1.5e-3                                           # [유도: control review]
    p = p_clean + rng.normal(0.0, sigma, p_clean.shape)
    yaw = 0.3 * t + rng.normal(0.0, 0.01, K)
    g = np.linspace(1.0, 0.0, K)
    p_r, yaw_r, g_r, jitter = resample_knots(p, yaw, g, OBS_DT, KNOT_DT)
    assert p_r.shape == (3, 6) and yaw_r.shape == (6,) and g_r.shape == (6,)
    assert np.array_equal(p_r[:, 0], p[:, 0]) and yaw_r[0] == yaw[0] and g_r[0] == g[0], \
        "knot 0 must be PINNED"
    a_raw = _accel_peak(p, OBS_DT)
    a_res = _accel_peak(p_r, KNOT_DT)
    print(f"        accel peak raw {a_raw:.3f} -> resampled {a_res:.3f} m/s^2 "
          f"(x{a_raw / a_res:.1f}), jitter {jitter:.2f} mm")
    assert a_raw > 3.0 * a_res, (a_raw, a_res)
    assert a_raw > 0.3, "the raw grid would have been REJECTED (a_max 0.20)"
    # the resampled track still follows the clean line (windowed mean is unbiased)
    clean_r, _, _, _ = resample_knots(p_clean, yaw, g, OBS_DT, KNOT_DT)
    assert np.max(np.abs(p_r - clean_r)) < 4 * sigma
    # jitter: white noise sigma -> second-diff vector RMS = sigma*sqrt(6*3)
    assert 0.5 * sigma * math.sqrt(18) * 1e3 < jitter < 1.5 * sigma * math.sqrt(18) * 1e3, jitter
    # clean input -> zero jitter, exact line through the grid
    _, _, _, j0 = resample_knots(p_clean, yaw, g, OBS_DT, KNOT_DT)
    assert j0 < 1e-9
    # yaw is unwrapped BEFORE averaging: a track crossing +-pi stays continuous
    yaw_wrap = np.array([math.atan2(math.sin(y), math.cos(y))
                         for y in np.linspace(math.pi - 0.2, math.pi + 0.4, K)])
    _, yw, _, _ = resample_knots(p_clean, yaw_wrap, g, OBS_DT, KNOT_DT)
    assert np.all(np.diff(yw) > 0.0) and abs(yw[-1] - (math.pi + 0.4)) < 0.05, yw
    # knot grid: 6 knots at 0..1.0 s for 16 raw knots (T_end = 1.0 s)
    assert p_r.shape[1] == 6
    # the LAST knot is PINNED to raw knot K-1 (verify 2026-09-02): the old
    # one-sided window [0.9, 1.0] averaged knots 14..15 and stopped the plan
    # half a raw step short of the policy's endpoint
    assert np.array_equal(p_r[:, -1], p[:, -1]) and yaw_r[-1] == yaw[-1] \
        and g_r[-1] == g[-1], "last knot must be PINNED"
    assert not np.allclose(clean_r[:, -1], p_clean[:, 14:16].mean(axis=1))
    # interior knots are symmetric-window means, so a constant-velocity
    # chunk lands EXACTLY on the line at every knot, endpoint included
    t_grid = np.arange(6) * KNOT_DT
    line = np.vstack([0.06 * t_grid, 0.02 * t_grid, np.zeros(6)])
    assert np.allclose(clean_r, line, atol=1e-12), clean_r - line
    assert np.allclose(clean_r[:, -1], p_clean[:, -1], atol=1e-12)
    # a grid whose last knot does NOT land on T_end (15 raw knots: T_end =
    # 0.933 s, last grid knot 0.8 s) is not pinned to a knot at another time
    t15 = np.arange(15) * OBS_DT
    p15 = np.vstack([0.06 * t15, 0.02 * t15, np.zeros(15)])
    p15_r, _, _, _ = resample_knots(p15, np.zeros(15), np.ones(15), OBS_DT, KNOT_DT)
    assert p15_r.shape[1] == 5
    assert np.allclose(p15_r[:, -1], [0.06 * 0.8, 0.02 * 0.8, 0.0], atol=1e-12)
    assert not np.allclose(p15_r[:, -1], p15[:, -1])


# =============================================================================
# A4: leash
# =============================================================================
def test_leash_anchor_clips():
    eta = np.array([1.0, 2.0, 0.5, 0.02, -0.01, 0.3])
    leash_m, leash_yaw = 0.05, math.radians(10.0)
    # a 0.30 m reference offset -> anchor 0.05 m toward it, same direction
    off = np.array([0.3, 0.0, 0.0])
    anchor, info = leash_anchor(eta, eta[:3] + off, eta[5] + math.radians(40.0),
                                leash_m, leash_yaw)
    assert np.allclose(anchor[:3] - eta[:3], [0.05, 0.0, 0.0], atol=1e-12)
    assert np.allclose(info["offset_ned"], off) and abs(info["offset_m"] - 0.30) < 1e-12
    assert info["clipped_pos"] and abs(info["offset_applied_m"] - 0.05) < 1e-12
    # a 40 deg yaw offset -> +10 deg
    assert abs(anchor[5] - (eta[5] + leash_yaw)) < 1e-12
    assert abs(info["dyaw"] - math.radians(40.0)) < 1e-12 and info["clipped_yaw"]
    assert abs(info["dyaw_applied"] - leash_yaw) < 1e-12
    # roll / pitch are the measured ones
    assert np.allclose(anchor[3:5], eta[3:5])
    # inside the leash: pass-through (reference == anchor)
    p_ref = eta[:3] + np.array([0.0, 0.03, 0.0])
    anchor2, info2 = leash_anchor(eta, p_ref, eta[5] - math.radians(4.0),
                                  leash_m, leash_yaw)
    assert np.allclose(anchor2[:3], p_ref) and not info2["clipped_pos"]
    assert abs(anchor2[5] - (eta[5] - math.radians(4.0))) < 1e-12 and not info2["clipped_yaw"]
    # no plan installed: r = x_meas -> identical branches, zero offset
    anchor3, info3 = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw)
    assert np.allclose(anchor3, eta) and info3["offset_m"] == 0.0 and info3["dyaw"] == 0.0
    # the yaw offset is WRAPPED before clipping (ref at -179 deg, meas at +179)
    e4 = eta.copy(); e4[5] = math.radians(179.0)
    anchor4, info4 = leash_anchor(e4, e4[:3], math.radians(-179.0), leash_m, leash_yaw)
    assert abs(info4["dyaw"] - math.radians(2.0)) < 1e-12 and not info4["clipped_yaw"]
    assert abs(math.atan2(math.sin(anchor4[5] - math.radians(-179.0)),
                          math.cos(anchor4[5] - math.radians(-179.0)))) < 1e-12


# =============================================================================
# A6: eta history keyed on fix stamps
# =============================================================================
def test_eta_history_dedup_interp_degenerate():
    h = EtaHistory(keep_s=3.0)
    try:
        h.interp(0.0); assert False, "empty history must raise"
    except ValueError:
        pass
    e0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 3.0])            # yaw 171.9 deg
    e1 = np.array([0.1, 0.0, 1.0, 0.0, 0.0, -3.0])           # yaw -171.9 deg (short way: +0.283)
    assert h.append(10.0, e0)
    # the same fix re-emitted by later ticks is NOT a new row
    assert not h.append(10.0, e0) and not h.append(9.99, e0)
    assert len(h) == 1 and h.span() == 0.0
    assert h.append(10.1, e1) and len(h) == 2 and abs(h.span() - 0.1) < 1e-12
    # interpolation: linear position, yaw the SHORT way across +-pi
    e = h.interp(10.025)
    assert abs(e[0] - 0.025) < 1e-12
    want_yaw = 3.0 + 0.25 * (2 * math.pi - 6.0)              # 3.0708, not 1.5
    assert abs(e[5] - want_yaw) < 1e-12, (e[5], want_yaw)
    assert abs(e[5]) <= math.pi
    # clamps outside the span, exact on a stamp
    assert np.allclose(h.interp(5.0), h.interp(10.0)) and abs(h.interp(10.0)[5] - 3.0) < 1e-12
    assert np.allclose(h.interp(99.0), h.interp(10.1))
    assert h.fix_stamps(10.05) == (10.0, 10.1) and h.fix_stamps(10.1) == (10.1,)
    assert h.fix_stamps(50.0) == (10.1,)
    # rows_at: both rows past the newest fix -> both clamp to ONE fix -> degenerate
    prev, now, degen = h.rows_at(10.5, OBS_DT)
    assert degen and np.allclose(prev, now)
    # both rows exactly on the same stamp is degenerate too
    h2 = EtaHistory(); h2.append(0.0, e0); h2.append(0.5, e1)
    assert h2.rows_at(0.5, 0.0)[2]
    # a history shorter than obs_dt is degenerate even if bracketed
    h3 = EtaHistory(); h3.append(0.0, e0); h3.append(0.03, e1)
    assert h3.rows_at(0.03, OBS_DT)[2]
    # a proper 10 Hz history around t_obs -> NOT degenerate, rows differ
    h4 = EtaHistory()
    for i in range(12):
        h4.append(100.0 + 0.1 * i, np.array([0.05 * i, 0, 1, 0, 0, 0.1 * i]))
    prev, now, degen = h4.rows_at(100.95, OBS_DT)
    assert not degen
    assert abs(now[0] - 0.475) < 1e-12 and abs(prev[0] - (0.475 - 0.05 * OBS_DT / 0.1)) < 1e-12
    assert h4.fix_stamps(100.95) == (100.9, 101.0)
    # two obs rows between the SAME pair of fixes is a valid (interpolated)
    # motion cue, not degenerate
    prev, now, degen = h4.rows_at(100.99, 0.05)
    assert not degen and h4.fix_stamps(100.99) == h4.fix_stamps(100.94)
    # keep window: rows older than 3 s fall off (the two newest always stay)
    h5 = EtaHistory(keep_s=3.0)
    for i in range(50):
        h5.append(float(i) * 0.1, e0)
    assert h5.stamps[0] >= 4.9 - 3.0 - 1e-9 and len(h5) <= 32
    h5.clear(); assert len(h5) == 0 and h5.latest_t is None
    # yaw stays continuous along the axis over many wraps (no drift)
    h6 = EtaHistory(keep_s=100.0)
    for i in range(200):
        y = 0.1 * i
        h6.append(float(i), [0, 0, 0, 0, 0, math.atan2(math.sin(y), math.cos(y))])
    e = h6.interp(150.5)
    assert abs(math.atan2(math.sin(e[5] - 15.05), math.cos(e[5] - 15.05))) < 1e-9


def test_eta_history_rows_for_never_half_scales_the_motion_cue():
    """MAJOR (verify 2026-09-02): fixes at 10 Hz, a depth stamp half an
    obs_dt NEWER than the newest fix. `rows_at` clamps the 'now' row and
    leaves the 'prev' row where it was (rows < obs_dt apart -> a displacement
    scaled by a factor in (0, 1], unflagged); `rows_for` shifts BOTH rows
    back so they are exactly `spacing` apart and reports the lag."""
    v = 0.5                                                  # m/s along x
    h = EtaHistory()
    for i in range(12):
        h.append(100.0 + 0.1 * i, np.array([v * 0.1 * i, 0, 1, 0, 0, 0.1 * i]))
    last = h.latest_t
    assert abs(last - 101.1) < 1e-12
    t_now = last + 0.5 * OBS_DT
    # the frozen interface: (eta_prev, eta_now, info)
    prev, now, info = h.rows_for(t_now, OBS_DT)
    assert set(info) == {"t_prev", "t_now", "fix_lag_s", "degenerate"}
    assert abs(info["fix_lag_s"] - 0.5 * OBS_DT) < 1e-12, info
    assert abs(info["t_now"] - last) < 1e-12 and abs(info["t_prev"] - (last - OBS_DT)) < 1e-12
    assert info["degenerate"] is False
    assert abs((now[0] - prev[0]) - v * OBS_DT) < 1e-12, "rows must be spacing apart"
    assert abs(now[0] - v * 1.1) < 1e-12                     # the newest fix
    # ...whereas rows_at (unchanged) returns the half-scaled cue
    prev_a, now_a, degen_a = h.rows_at(t_now, OBS_DT)
    assert not degen_a
    assert abs((now_a[0] - prev_a[0]) - v * 0.5 * OBS_DT) < 1e-12
    assert np.allclose(now_a, now)
    # inside the span the two agree and the lag is 0
    prev2, now2, info2 = h.rows_for(100.95, OBS_DT)
    prev3, now3, _ = h.rows_at(100.95, OBS_DT)
    assert np.allclose(prev2, prev3) and np.allclose(now2, now3)
    assert info2["fix_lag_s"] == 0.0 and abs(info2["t_now"] - 100.95) < 1e-12
    # a stamp far ahead is served by the newest fix with the full lag
    _, now4, info4 = h.rows_for(last + 3.0, OBS_DT)
    assert abs(info4["fix_lag_s"] - 3.0) < 1e-12 and np.allclose(now4, now)
    assert not info4["degenerate"]
    # degenerate: both rows clamp to the oldest fix / span < spacing / empty
    assert h.rows_for(50.0, OBS_DT)[2]["degenerate"]
    h2 = EtaHistory(); h2.append(0.0, np.zeros(6)); h2.append(0.03, np.ones(6) * 0.01)
    assert h2.rows_for(0.03, OBS_DT)[2]["degenerate"]
    try:
        EtaHistory().rows_for(0.0, OBS_DT); assert False, "empty must raise"
    except ValueError:
        pass


# =============================================================================
# A12: gripper width estimator
# =============================================================================
def test_gripper_width_estimator():
    est = GripperWidthEstimator(W_OPEN, W_CLOSED, W_OPEN, travel_s=2.0)
    rate = (W_OPEN - W_CLOSED) / 2.0
    assert abs(est.rate - rate) < 1e-15
    assert est.width(0.0) == W_OPEN                          # init OPEN (assumption, logged)
    # holding CLOSE (-1) for 1 s -> half way
    est.drive(-1.0, 0.0)
    assert abs(est.width(1.0) - (W_OPEN - rate)) < 1e-12
    # release stops it
    est.release(1.0)
    assert abs(est.width(5.0) - (W_OPEN - rate)) < 1e-12
    assert est.level == 0.0
    # close for 10 s -> clamps at closed
    est.drive(-1.0, 5.0)
    assert abs(est.width(15.0) - W_CLOSED) < 1e-12
    # open (+1) for 1 s -> closed + rate; then clamps at open
    est.drive(+1.0, 15.0)
    assert abs(est.width(16.0) - (W_CLOSED + rate)) < 1e-12
    assert abs(est.width(40.0) - W_OPEN) < 1e-12
    # a non-advancing clock neither integrates nor loses the latch
    est.drive(-1.0, 40.0)
    assert abs(est.width(39.0) - W_OPEN) < 1e-12 and est.level == -1.0
    assert abs(est.width(40.5) - (W_OPEN - 0.5 * rate)) < 1e-12
    # width -> g mapping matches compose_plan's clip rule
    assert abs(est.g(40.5) - 0.75) < 1e-12
    # fractional / non-finite levels: scaled / treated as released
    est.drive(0.5, 41.0)          # 40.5->41.0 at -1 (-rate), then +0.5 OPENS at rate/2
    assert abs(est.width(42.0) - (W_OPEN - 1.0 * rate + 0.5 * rate)) < 1e-12
    est.drive(float("nan"), 42.0); assert est.level == 0.0
    # init outside the band is clamped; bad params raise
    assert GripperWidthEstimator(W_OPEN, W_CLOSED, 0.5, 2.0).width(0.0) == W_OPEN
    for bad in ((W_CLOSED, W_OPEN, W_OPEN, 2.0), (W_OPEN, W_CLOSED, W_OPEN, 0.0)):
        try:
            GripperWidthEstimator(*bad); assert False, bad
        except ValueError:
            pass
    assert est.n_drives == 7
    # n_drives counts level CHANGES only (verify 2026-09-02): the bus copy
    # of a drive this process emitted re-latches the same level
    est.drive(0.0, 43.0)                                     # already 0
    assert est.n_drives == 7
    est.drive(-1.0, 44.0); est.drive(-1.0, 45.0); est.release(45.0)
    assert est.n_drives == 9 and est.level == 0.0
    assert abs(est.width(46.0) - (W_OPEN - 1.0 * rate + 0.5 * rate - 1.0 * rate)) < 1e-12


# =============================================================================
# yaw reference filter (2026-09-14)
# =============================================================================
def _turn_action5(dyaw_end: float, K: int = 16) -> np.ndarray:
    """A chunk asking for a uniform turn to ``dyaw_end`` over the chunk,
    with a small forward motion so the TCP knots are distinct."""
    a = _ident_action5(K)
    a[:, 2] = 0.05 * np.arange(K) * OBS_DT
    a[:, 3] = np.linspace(0.0, dyaw_end, K)
    return a


def _filt(msg, info, anchor, T_bt, yaw_ref0, omega_prev, dt_prev, yaw_meas=None,
          rate=3.0, tau=2.0, guard=15.0):
    return filter_plan_yaw(
        msg, info["p_tcp"], yaw_ref0,
        anchor[5] if yaw_meas is None else yaw_meas, anchor[3:5], T_bt,
        omega_prev, dt_prev, rate=math.radians(rate), tau=tau,
        guard=math.radians(guard))


def test_filter_plan_yaw_first_plan_low_passes_the_requested_turn():
    """First plan of a mission (no reference flying, omega state 0): the
    chunk asks for +10 deg over its 1 s span (10 deg/s); the filtered yaw
    starts AT the anchor (there is nothing else to continue) and turns at
    omega_policy * (1 - exp(-period/tau)) — 0.22 of the request at the
    shipped 0.5 s / 2 s — never at the raw rate."""
    T_bt = _T_bt()
    anchor = np.array([0.3, -0.2, 0.9, 0.0, 0.0, 0.4])
    a = _turn_action5(math.radians(10.0))
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    span = (msg.yaw.size - 1) * msg.dt
    omega_pol = (msg.yaw[-1] - msg.yaw[0]) / span
    out, yi = _filt(msg, info, anchor, T_bt, yaw_ref0=None, omega_prev=0.0,
                    dt_prev=0.5)
    alpha = 1.0 - math.exp(-0.5 / 2.0)
    assert abs(out.yaw[0] - anchor[5]) < 1e-12
    assert abs(math.radians(yi["omega_ref_deg_s"]) - alpha * omega_pol) < 1e-12
    assert np.allclose(np.diff(out.yaw) / msg.dt, alpha * omega_pol, atol=1e-12)
    assert abs(yi["dyaw_raw_deg"] - 10.0) < 1e-9
    assert abs(yi["dyaw_filt_deg"] - alpha * 10.0) < 1e-9
    assert not yi["rate_clipped"] and not yi["guard_clamped_start"]
    assert yi["guard_clamped_knots"] == 0
    # the plan's knot count, dt, t0 and gripper are untouched
    assert out.n_knots == msg.n_knots and out.dt == msg.dt and out.t0 == msg.t0
    assert np.array_equal(out.g, msg.g)


def test_filter_plan_yaw_continues_the_flown_reference_not_the_measurement():
    """With a reference flying, the filtered yaw starts at ITS yaw at t0 —
    the anchor (measured, stale) yaw the plan was composed against does not
    reach the knots. That is the whole point: no plan-to-plan copy of the
    hull's own swing."""
    T_bt = _T_bt()
    anchor = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.40])     # measured 0.40 rad
    a = _turn_action5(0.0)                                  # no turn asked
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    out, yi = _filt(msg, info, anchor, T_bt, yaw_ref0=0.46, omega_prev=0.0,
                    dt_prev=0.5)
    assert np.allclose(out.yaw, 0.46, atol=1e-12)           # the flown ref, 3.4 deg off the fix
    assert abs(yi["yaw_anchor_deg"] - math.degrees(0.40)) < 1e-9
    assert abs(yi["yaw0_deg"] - math.degrees(0.46)) < 1e-9


def test_filter_plan_yaw_converges_and_averages_out_alternation():
    """Plan after plan asking the same +2 deg/s turn (under the 3 deg/s
    cap): the turn-rate state converges to it (1 - e^-4 after 4 tau). A
    +30 deg/s request is capped at rate. Alternating +-2 deg/s every plan:
    the state settles on a small oscillation whose amplitude is
    alpha/(2-alpha) of the request — the swing that used to reach the hull
    is gone, and it is an average, not a freeze."""
    T_bt = _T_bt()
    anchor = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    a = _turn_action5(math.radians(2.0))                    # 2 deg over 1 s
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    omega = 0.0
    for _ in range(16):                                     # 8 s of plans at 0.5 s
        _o, yi = _filt(msg, info, anchor, T_bt, yaw_ref0=0.0, omega_prev=omega,
                       dt_prev=0.5)
        omega = math.radians(yi["omega_ref_deg_s"])
    assert abs(math.degrees(omega) - 2.0 * (1 - math.exp(-8.0 / 2.0))) < 1e-6
    assert not yi["rate_clipped"]
    # a bigger request is capped at the rate
    big = _turn_action5(math.radians(30.0))
    msg_b, info_b = compose_plan(big, anchor, anchor[3:5], T_bt, **_KW)
    omega = 0.0
    for _ in range(40):
        _o, yi = _filt(msg_b, info_b, anchor, T_bt, yaw_ref0=0.0,
                       omega_prev=omega, dt_prev=0.5)
        omega = math.radians(yi["omega_ref_deg_s"])
    assert abs(math.degrees(omega) - 3.0) < 1e-9 and yi["rate_clipped"]
    # alternation
    neg = _turn_action5(math.radians(-2.0))
    msg_n, info_n = compose_plan(neg, anchor, anchor[3:5], T_bt, **_KW)
    omega = 0.0
    seen = []
    for i in range(40):
        m, inf = (msg, info) if i % 2 == 0 else (msg_n, info_n)
        _o, yi = _filt(m, inf, anchor, T_bt, yaw_ref0=0.0, omega_prev=omega,
                       dt_prev=0.5)
        omega = math.radians(yi["omega_ref_deg_s"])
        seen.append(math.degrees(omega))
    alpha = 1.0 - math.exp(-0.5 / 2.0)
    amp = 2.0 * alpha / (2.0 - alpha)
    assert max(abs(v) for v in seen[-6:]) < amp + 1e-3    # converging from above
    assert max(abs(v) for v in seen[-6:]) > 0.9 * amp     # an average, not a freeze


def test_filter_plan_yaw_guard_clamps_and_the_tcp_is_preserved():
    """The guard is the ONLY place the measurement enters: a reference 20
    deg from the fix is pulled back to the 15 deg guard and says so. And the
    body knots are re-derived from the TCP knots: with the filtered yaw the
    TCP (body + R t_bt) still sits exactly where compose_plan put it."""
    T_bt = _T_bt()
    t_bt = T_bt[:3, 3]
    anchor = np.array([0.1, 0.2, 0.5, math.radians(3.0), math.radians(-2.0), 0.0])
    a = _turn_action5(math.radians(4.0))
    msg, info = compose_plan(a, anchor, anchor[3:5], T_bt, **_KW)
    out, yi = _filt(msg, info, anchor, T_bt, yaw_ref0=math.radians(20.0),
                    omega_prev=0.0, dt_prev=0.5, yaw_meas=0.0)
    assert yi["guard_clamped_start"]
    assert abs(out.yaw[0] - math.radians(15.0)) < 1e-12
    for k in range(out.n_knots):
        tcp = out.p_ned[:, k] + rot_zyx(anchor[3], anchor[4], float(out.yaw[k])) @ t_bt
        assert np.allclose(tcp, info["p_tcp"][:, k], atol=1e-12), k
    assert yi["body_shift_max_m"] > 0.05      # 15 deg on a 0.5 m lever moves the body
    # a wrapped comparison: a fix at +179 deg and a reference at -179 deg are
    # 2 deg apart, not 358, so no clamp
    out2, yi2 = _filt(msg, info, anchor, T_bt, yaw_ref0=math.radians(-179.0),
                      omega_prev=0.0, dt_prev=0.5, yaw_meas=math.radians(179.0))
    assert not yi2["guard_clamped_start"] and yi2["guard_clamped_knots"] == 0


def test_filter_plan_yaw_rejects_bad_inputs():
    T_bt = _T_bt()
    anchor = np.zeros(6)
    msg, info = compose_plan(_turn_action5(0.1), anchor, anchor[3:5], T_bt, **_KW)
    for bad in (dict(p=info["p_tcp"][:, :2], rate=0.05),
                dict(p=info["p_tcp"], rate=0.0)):
        try:
            filter_plan_yaw(msg, bad["p"], None, 0.0, anchor[3:5], T_bt,
                            0.0, 0.5, rate=bad["rate"], tau=2.0, guard=0.3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"filter_plan_yaw accepted {bad}")
    # a long gap converges instead of jumping past the request
    out, yi = filter_plan_yaw(msg, info["p_tcp"], None, 0.0, anchor[3:5], T_bt,
                              0.0, 1e9, rate=1.0, tau=2.0, guard=0.3)
    assert yi["dt_since_prev_s"] == 8.0
    assert abs(yi["omega_ref_deg_s"] - yi["omega_policy_deg_s"] * (1 - math.exp(-4.0))) < 1e-9



# =============================================================================
# pos_rpy_width (2026-09-26): label parity, encode/decode, the third branch
# =============================================================================
def test_pos_rpy_parity_upstream():
    """Bit-level parity of the 7-dim label helpers with upstream
    ``umi.common.yaw_action`` (imported by path): ``rot_x`` / ``rot_y`` /
    ``rpy_of`` (station ``rpy_of_batch``) / ``encode_pos_rpy`` /
    ``decode_pos_rpy``; the dyaw column bit-identical to the 5-dim label;
    the vocabulary {10, 5, 7}. A checkout whose yaw_action.py lacks the
    7-dim functions FAILS (the label the 7-dim checkpoint learns and the
    decode the station flies would be unverified against each other)."""
    if not UMI_REPO.is_dir():
        return _skip(f"UMI repo not found at {UMI_REPO}")
    src = UMI_REPO / "umi" / "common" / "yaw_action.py"
    if not src.exists():
        raise AssertionError("upstream yaw_action.py missing")
    up = _load_by_path(src, "umi_yaw_action_for_rpy_parity")
    for name in ("rot_x", "rot_y", "rpy_of", "encode_pos_rpy", "decode_pos_rpy"):
        assert hasattr(up, name), f"upstream yaw_action.{name} missing (training group)"
    assert up.ACTION_REPR_DIM == ACTION_DIM_BY_REPR == {"pose10d": 10, "pos_yaw_width": 5,
                                                        "pos_rpy_width": 7}
    rng = np.random.default_rng(31)
    ang = rng.uniform(-2.0 * math.pi, 2.0 * math.pi, 32)
    assert np.array_equal(rot_x(ang), up.rot_x(ang)) and np.array_equal(rot_x(ang[0]), up.rot_x(ang[0]))
    assert np.array_equal(rot_y(ang), up.rot_y(ang)) and np.array_equal(rot_y(ang[0]), up.rot_y(ang[0]))
    # rot_x / rot_y / rot_z compose to the station's rot_zyx
    for i in range(5):
        r, p, y = rng.uniform(-1.2, 1.2, 3)
        assert np.allclose(rot_z(y) @ rot_y(p) @ rot_x(r), rot_zyx(r, p, y), atol=1e-15)
    Ms = np.stack([_rand_R(rng) for _ in range(32)])
    for M in (Ms, Ms[0]):
        got = rpy_of_batch(M)
        want = up.rpy_of(M)
        assert len(got) == len(want) == 3
        for g_, w_ in zip(got, want):
            assert np.array_equal(g_, w_) and g_.dtype == w_.dtype
    Ts = np.stack([_rand_T(rng) for _ in range(32)])
    for T in (Ts, Ts.astype(np.float32)):
        dp, dyaw, droll, dpitch = encode_pos_rpy(T)
        dp_u, dyaw_u, droll_u, dpitch_u = up.encode_pos_rpy(T)
        assert dp.dtype == dp_u.dtype == T.dtype and np.array_equal(dp, dp_u)
        for a_, b_ in ((dyaw, dyaw_u), (droll, droll_u), (dpitch, dpitch_u)):
            assert a_.dtype == b_.dtype == np.float64 and a_.shape == (32,)
            assert np.array_equal(a_, b_)
        # the 5-dim label is a column subset: dp and dyaw BIT-identical
        dp5, dyaw5 = encode_pos_yaw(T)
        assert np.array_equal(dp, dp5) and np.array_equal(dyaw, dyaw5)
        assert np.array_equal(dyaw, up.encode_pos_yaw(T)[1])
        one = encode_pos_rpy(T[0])
        one_u = up.encode_pos_rpy(T[0])
        assert all(np.array_equal(a_, b_) for a_, b_ in zip(one, one_u))
        assert np.shape(one[1]) == np.shape(one[2]) == np.shape(one[3]) == ()
    # decode: random (dp, dyaw, droll, dpitch), encodings of random T, single
    dp_r = rng.uniform(-1.0, 1.0, (32, 3))
    dy_r = rng.uniform(-math.pi, math.pi, 32)
    dr_r = rng.uniform(-1.0, 1.0, 32)
    dpi_r = rng.uniform(-1.0, 1.0, 32)
    assert np.array_equal(decode_pos_rpy(dp_r, dy_r, dr_r, dpi_r),
                          up.decode_pos_rpy(dp_r, dy_r, dr_r, dpi_r))
    enc = up.encode_pos_rpy(Ts)
    assert np.array_equal(decode_pos_rpy(*enc), up.decode_pos_rpy(*enc))
    assert np.array_equal(decode_pos_rpy(dp_r[0], dy_r[0], dr_r[0], dpi_r[0]),
                          up.decode_pos_rpy(dp_r[0], dy_r[0], dr_r[0], dpi_r[0]))
    # zero roll/pitch decodes EXACTLY as the 5-dim decode (Ry(0), Rx(0) exact)
    z = np.zeros(32)
    assert np.array_equal(decode_pos_rpy(dp_r, dy_r, z, z), decode_pos_yaw(dp_r, dy_r))
    assert np.array_equal(up.decode_pos_rpy(dp_r, dy_r, z, z), up.decode_pos_yaw(dp_r, dy_r))
    print("        parity vs upstream yaw_action (7-dim): exact")


def test_pos_rpy_sign_mount_and_round_trip():
    """Without the upstream: (a) a pure vehicle roll / pitch / yaw through
    R_BT_C3 is labelled exactly on its own column, the other two 0 (to fp);
    (b) a pure CAMERA pitch (about cam x, the (d) case of the 5-dim test)
    gives dyaw EXACTLY 0 and a non-zero dpitch under pos_rpy_width; (c)
    decode -> encode round trip; (d) rpy_of_batch(rot_zyx) round trip to
    +-60 deg and the scalar rpy_of agree; FRD signs: +dpitch = nose-up."""
    d = math.radians(10.0)
    for col, R in (("roll", rot_zyx(d, 0.0, 0.0)), ("pitch", rot_zyx(0.0, d, 0.0)),
                   ("yaw", rot_zyx(0.0, 0.0, d))):
        T = np.eye(4)
        T[:3, :3] = R_BT_C3.T @ R @ R_BT_C3
        dp, dyaw, droll, dpitch = encode_pos_rpy(T)
        got = {"roll": float(droll), "pitch": float(dpitch), "yaw": float(dyaw)}
        for k, v in got.items():
            want = d if k == col else 0.0
            assert abs(v - want) < 1e-12, (col, got)
    # (b) camera pitch about cam x: 5-dim says 0 (exactly), 7-dim labels it
    for deg in (-40.0, -12.0, 12.0, 40.0):
        T = np.eye(4)
        T[:3, :3] = _Rx(math.radians(deg))
        dp, dyaw, droll, dpitch = encode_pos_rpy(T)
        assert dyaw == 0.0 and encode_pos_yaw(T)[1] == 0.0
        assert abs(float(dpitch)) > 0.1, math.degrees(dpitch)
        # cam +x is body +y (R_BT_C3 col 0 = [0, 1, 0]): a rotation about
        # it is a pure BODY pitch, so droll is exactly 0 and |dpitch| = |deg|
        assert abs(float(droll)) < 1e-12 and abs(abs(math.degrees(dpitch)) - abs(deg)) < 1e-9
        # camera x-rotation by +a tips the optical axis ... nose-DOWN in
        # FRD for +a (cam x = body y, right-hand about +y is nose-down? no:
        # rot about body +y by +a IS a nose-up pitch by the ZYX convention)
        assert (float(dpitch) > 0.0) == (deg > 0.0)
    # (c) round trip
    rng = np.random.default_rng(37)
    dp = rng.uniform(-1.0, 1.0, (16, 3))
    dyaw = rng.uniform(-math.pi, math.pi, 16)
    droll = rng.uniform(-1.0, 1.0, 16)
    dpitch = rng.uniform(-1.0, 1.0, 16)
    T = decode_pos_rpy(dp, dyaw, droll, dpitch)
    assert T.shape == (16, 4, 4) and T.dtype == np.float64
    R = T[:, :3, :3]
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=1e-14)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(R_BT_C3 @ R @ R_BT_C3.T, rot_z(dyaw) @ rot_y(dpitch) @ rot_x(droll), atol=1e-14)
    dp2, dyaw2, droll2, dpitch2 = encode_pos_rpy(T)
    assert np.array_equal(dp2, dp)
    assert np.allclose(dyaw2, dyaw, atol=1e-12) and np.allclose(droll2, droll, atol=1e-12)
    assert np.allclose(dpitch2, dpitch, atol=1e-12)
    assert np.allclose(decode_pos_rpy(np.zeros(3), 0.0, 0.0, 0.0), np.eye(4))
    # (d) rpy_of_batch(rot_zyx) round trip to +-60 deg, and == the scalar rpy_of
    for _ in range(64):
        r, p, y = rng.uniform(-math.radians(60.0), math.radians(60.0), 3)
        y = rng.uniform(-math.pi, math.pi)
        R = rot_zyx(r, p, y)
        rb, pb, yb = rpy_of_batch(R)
        assert abs(rb - r) < 1e-12 and abs(pb - p) < 1e-12 and abs(yb - y) < 1e-12
        rs, ps, ys = rpy_of(R)
        assert abs(rb - rs) < 1e-15 and abs(pb - ps) < 1e-15 and abs(yb - ys) < 1e-15
    Rs = np.stack([rot_zyx(*rng.uniform(-1.0, 1.0, 3)) for _ in range(8)])
    rb, pb, yb = rpy_of_batch(Rs)
    assert rb.shape == pb.shape == yb.shape == (8,)


def _bytes_equal(a, b) -> bool:
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    b = np.ascontiguousarray(np.asarray(b, dtype=float))
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def test_compose_plan_contract_7d():
    """The (K, 7) pos_rpy_width contract (2026-09-26), both sub-modes.

    Dropped-and-logged (track_rp False, the DEFAULT): msg and every
    pre-existing info array BYTE-identical to the 5-dim action the 7-dim
    one contains (cols [0:4, 6]); msg.rp None; dropped_rp_deg MEANINGFUL
    (max |rp_k - rp_level|); rp_tracked False.
    Tracked: msg.rp (2, 6) on the knot grid; knot-0 attitude == the anchor's
    (droll_0 = dpitch_0 = 0 holds what you have); p_tcp BYTE-identical to
    the 5-dim path (the request is never altered); TCP consistency of every
    raw body knot with its own attitude; the T1 clip recorded
    (rp_clipped_deg / rp_clipped_n) with p_tcp still untouched and the
    clipped hull attitude in the knots."""
    T_bt = _T_bt()
    t_bt = T_bt[:3, 3]
    anchor = np.array([0.3, -0.2, 0.9, math.radians(5.0), math.radians(-3.0), 0.4])
    a7 = _ident_action7()
    a7[:, 2] = 0.05 * np.arange(16) * OBS_DT
    a7[:, 3] = np.linspace(0.0, 0.3, 16)
    a7[:, 4] = np.linspace(0.0, math.radians(-8.0), 16)
    a7[:, 5] = np.linspace(0.0, math.radians(30.0), 16)      # ramps past a 20 deg cap
    a7[:, 6] = np.linspace(W_OPEN + 0.01, W_CLOSED - 0.01, 16)
    a5 = _as5(a7)
    kw = dict(_KW, t0=12.5, plan_id=42, obs_t_rel=12.5)
    m5, i5 = compose_plan(a5, anchor, anchor[3:5], T_bt, **kw)
    # ---- dropped-and-logged ------------------------------------------
    m7, i7 = compose_plan(a7, anchor, anchor[3:5], T_bt, **kw)
    assert i7["action_repr"] == ACTION_REPR_POS_RPY_WIDTH and i7["rp_tracked"] is False
    assert m7.rp is None and i7["rp"] is None
    for f in ("plan_id", "t0", "dt", "obs_t", "arrival_t"):
        assert getattr(m7, f) == getattr(m5, f)
    for f in ("p_ned", "yaw", "g"):
        assert _bytes_equal(getattr(m7, f), getattr(m5, f)), f
    for key in ("p_tcp", "p_tcp_raw", "p_body_raw", "yaw_raw"):
        assert _bytes_equal(i7[key], i5[key]), key
    assert i7["jitter_rms_mm"] == i5["jitter_rms_mm"]
    assert i7["n_raw"] == 16 and i7["n_knots"] == 6 and m7.n_knots == 6
    assert np.array_equal(i7["anchor_tcp"][0], i5["anchor_tcp"][0])
    assert i5["rp_tracked"] is False and "rp_raw" not in i5
    # dropped_rp_deg is MEANINGFUL again: at a level anchor the requested
    # attitude is exactly (droll, dpitch) and the drop is measured from
    # rp_level; here (tilted anchor) it is the body-z composed attitude
    assert i7["rp_raw"].shape == (2, 16)
    assert np.allclose(i7["rp_raw"][:, 0], anchor[3:5], atol=1e-12)
    assert i7["dropped_rp_deg"] > 20.0, i7["dropped_rp_deg"]
    assert abs(i7["dropped_rp_deg"] - np.abs(i7["rp_dropped_deg"]).max()) < 1e-12
    assert i7["rp_clipped_deg"] == 0.0 and i7["rp_clipped_n"] == 0
    lvl = anchor.copy(); lvl[3:5] = 0.0
    _m, il = compose_plan(a7, lvl, lvl[3:5], T_bt, **kw)
    assert np.allclose(il["rp_raw"][0], a7[:, 4], atol=1e-12)
    assert np.allclose(il["rp_raw"][1], a7[:, 5], atol=1e-12)
    assert abs(il["dropped_rp_deg"] - 30.0) < 1e-9
    assert np.allclose(il["rp_dropped_deg"][1], np.degrees(a7[:, 5]), atol=1e-9)
    # ---- tracked -----------------------------------------------------
    cap = math.radians(20.0)
    mt, it = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap, **kw)
    assert it["rp_tracked"] is True and mt.rp is not None
    assert mt.rp.shape == (2, 6) and it["rp"].shape == (2, 6) and np.array_equal(mt.rp, it["rp"])
    assert np.all(np.isfinite(mt.rp)) and np.all(np.abs(mt.rp) <= cap + 1e-15)
    assert np.allclose(mt.rp[:, 0], anchor[3:5], atol=1e-12)           # knot 0 = anchor attitude
    assert np.allclose(it["rp_raw"][:, 0], anchor[3:5], atol=1e-12)
    assert _bytes_equal(it["p_tcp"], i5["p_tcp"]) and _bytes_equal(it["p_tcp_raw"], i5["p_tcp_raw"])
    assert np.all(it["rp_dropped_deg"] == 0.0) and it["dropped_rp_deg"] == 0.0
    assert mt.n_knots == 6 and mt.dt == KNOT_DT and mt.obs_t == 12.5
    assert np.array_equal(mt.g, m5.g)                                   # width column 6
    # the T1 clip: the 30 deg pitch ramp exceeds the 20 deg cap on its tail
    assert it["rp_raw"][1].max() > cap and abs(it["rp_clipped_deg"] - math.degrees(it["rp_raw"][1].max() - cap)) < 1e-9
    assert it["rp_clipped_n"] == int(np.count_nonzero(np.abs(it["rp_raw"]).max(axis=0) > cap))
    assert 1 <= it["rp_clipped_n"] < 16
    assert abs(mt.rp[1, -1] - cap) < 1e-12                              # last knot pinned, clipped
    # TCP consistency on the RAW knots with the CLIPPED attitude
    rp_clip = np.clip(it["rp_raw"], -cap, cap)
    for k in range(16):
        tcp = it["p_body_raw"][:, k] + rot_zyx(rp_clip[0, k], rp_clip[1, k], it["yaw_raw"][k]) @ t_bt
        assert np.allclose(tcp, it["p_tcp_raw"][:, k], atol=1e-12), k
    # knot 0: the body knot is the anchor (attitude == anchor, dp_0 = 0)
    assert np.allclose(mt.p_ned[:, 0], anchor[:3], atol=1e-12)
    # no cap: nothing clipped, the hull carries the full request
    mu, iu = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **kw)
    assert iu["rp_clipped_deg"] == 0.0 and iu["rp_clipped_n"] == 0
    assert np.array_equal(iu["rp_raw"], it["rp_raw"])
    assert mu.rp[1, -1] > cap
    # the tracked body knots differ from the levelled ones (the hull tilts)
    assert np.max(np.abs(mt.p_ned - m5.p_ned)) > 0.01
    # level anchor, tracked: roll = droll, pitch = dpitch, yaw = anchor + dyaw
    ml, ilt = compose_plan(a7, lvl, lvl[3:5], T_bt, track_rp=True, **kw)
    assert np.allclose(ilt["rp_raw"][0], a7[:, 4], atol=1e-12)
    assert np.allclose(ilt["rp_raw"][1], a7[:, 5], atol=1e-12)
    assert np.allclose(ilt["yaw_raw"], lvl[5] + a7[:, 3], atol=1e-12)
    assert np.allclose(ml.rp[:, 0], 0.0, atol=1e-12)
    # attitude knots ride the SAME grid as the positions (one _window_mean)
    assert np.array_equal(ml.rp, resample_knots_att(ilt["rp_raw"], OBS_DT, KNOT_DT))


def test_compose_plan_7d_body_z_composition_vs_legacy_and_5d():
    """(d) At a tilted LEASHED anchor the tracked 7-dim decode with droll =
    dpitch = 0 IS the legacy pose10d BODY-frame decode of the mount-vertical
    construction (R_body_a @ Rz(d) both ways): same yaw_raw, and the
    roll/pitch the legacy path DROPPED is exactly what the 7-dim path
    tracks; and it differs from the 5-dim WORLD-z composition by the (c)
    amount (~6.6e-4 rad at 5 deg / 0.3 rad)."""
    T_bt = _T_bt()
    K = 16
    dyaw = np.linspace(0.0, 0.3, K)
    dp = np.zeros((K, 3))
    dp[:, 0] = 0.01 * np.arange(K) * OBS_DT
    dp[:, 2] = 0.05 * np.arange(K) * OBS_DT
    width = np.linspace(W_OPEN, W_CLOSED, K)
    a5 = np.concatenate([dp, dyaw[:, None], width[:, None]], axis=1)
    a7 = np.concatenate([dp, dyaw[:, None], np.zeros((K, 2)), width[:, None]], axis=1)
    a10 = np.concatenate([mat_to_pose10d(decode_pos_yaw(dp, dyaw)), width[:, None]], axis=1)
    meas = np.array([0.3, -0.2, 0.9, math.radians(5.0), math.radians(5.0), 2.9])
    anchor, _ = leash_anchor(meas, meas[:3] + np.array([0.3, 0.1, 0.0]), meas[5],
                             0.05, math.radians(10.0))
    m5, i5 = compose_plan(a5, anchor, anchor[3:5], T_bt, **_KW)
    m10, i10 = compose_plan(a10, anchor, anchor[3:5], T_bt, **_KW)
    m7, i7 = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **_KW)
    # 7-dim tracked == legacy body-frame decode
    assert np.allclose(i7["yaw_raw"], i10["yaw_raw"], atol=1e-12)
    assert np.allclose(i7["p_tcp_raw"], i10["p_tcp_raw"], atol=1e-12)
    rp_legacy = anchor[3:5, None] + np.radians(i10["rp_dropped_deg"])   # what legacy dropped
    assert np.allclose(i7["rp_raw"], rp_legacy, atol=1e-12)
    assert i10["dropped_rp_deg"] > 1.0                                  # ~1.7 deg read as roll/pitch
    # ...and != the 5-dim world-z composition by the (c) amount
    yaw_diff = float(np.max(np.abs(i7["yaw_raw"] - i5["yaw_raw"])))
    assert 1e-4 < yaw_diff < 1e-2, yaw_diff
    assert i7["yaw_raw"][-1] > math.pi                                  # unwrapped past +pi
    # the 7-dim DROPPED sub-mode of the same action is the 5-dim path
    m7d, i7d = compose_plan(a7, anchor, anchor[3:5], T_bt, **_KW)
    assert _bytes_equal(m7d.p_ned, m5.p_ned) and _bytes_equal(m7d.yaw, m5.yaw)
    assert abs(i7d["dropped_rp_deg"] - i10["dropped_rp_deg"]) < 1e-9
    print(f"        (d) body-z vs world-z at a 5 deg anchor: yaw diff {math.degrees(yaw_diff):.3f} deg")


def test_compose_plan_7d_pure_pitch_twin():
    """The 7-dim twin of test_pure_pitch_delta_keeps_tcp_position: a hold
    at dpitch = theta (dp = 0). Dropped mode: p_tcp and the levelled body
    knots identical to the 5-dim hold, dropped_rp_deg == theta. Tracked:
    p_tcp identical, the body knot moves |t_bt|·2·sin(theta/2) from the
    anchor -- the hull tilts around the jaw."""
    T_bt = _T_bt()
    t_bt = T_bt[:3, 3]
    anchor = np.array([1.0, -0.5, 0.8, 0.0, 0.0, 0.7])       # level, yawed
    theta = math.radians(10.0)
    a7 = _ident_action7()
    a7[:, 5] = theta
    a5 = _as5(a7)
    T_tcp_a = T_from_eta(anchor) @ T_bt
    p_tcp_a = T_tcp_a[:3, 3]
    m5, i5 = compose_plan(a5, anchor, anchor[3:5], T_bt, **_KW)
    md, idr = compose_plan(a7, anchor, anchor[3:5], T_bt, **_KW)
    assert _bytes_equal(md.p_ned, m5.p_ned) and _bytes_equal(idr["p_tcp"], i5["p_tcp"])
    assert np.allclose(md.p_ned, anchor[:3, None], atol=1e-12)
    assert abs(idr["dropped_rp_deg"] - 10.0) < 1e-9
    assert np.allclose(idr["rp_dropped_deg"][0], 0.0, atol=1e-9)
    assert np.allclose(idr["rp_dropped_deg"][1], 10.0, atol=1e-9)
    mt, it = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **_KW)
    assert _bytes_equal(it["p_tcp"], i5["p_tcp"])
    assert np.allclose(it["p_tcp_raw"], p_tcp_a[:, None], atol=1e-12)
    assert np.allclose(mt.rp[1], theta, atol=1e-12) and np.allclose(mt.rp[0], 0.0, atol=1e-12)
    moved = np.linalg.norm(mt.p_ned[:, 1] - anchor[:3])
    lever = math.hypot(t_bt[0], t_bt[2])
    assert abs(moved - lever * 2.0 * math.sin(theta / 2.0)) < 1e-12, (moved, lever)
    assert moved > 0.03, moved                               # ~0.079 m at 10 deg [유도]
    # knot 0 has dpitch = theta as well (a hold), so it moved too; the plan
    # is a hold at the tilted pose
    assert np.allclose(mt.p_ned, mt.p_ned[:, :1], atol=1e-12)
    # the same tilt in the legacy encoding lands on the same tracked knots
    R_bt = T_bt[:3, :3]
    dT = np.eye(4)
    dT[:3, :3] = R_bt.T @ rot_zyx(0.0, theta, 0.0) @ R_bt
    _m10, i10 = compose_plan(_delta_action(dT), anchor, anchor[3:5], T_bt, **_KW)
    R_body_k = T_tcp_a[:3, :3] @ dT[:3, :3] @ R_bt.T
    assert np.allclose(mt.p_ned[:, 0], p_tcp_a - R_body_k @ t_bt, atol=1e-12)
    assert abs(i10["dropped_rp_deg"] - 10.0) < 1e-9


def test_compose_plan_7d_tracked_rejects_the_pitch_singularity():
    """(N2) anchor pitch 25 deg + dpitch 65 deg = 90 deg absolute: rpy_of's
    yaw is atan2(0, 0) there and would enter p_body / PlanMsg.yaw (the T1
    clip repairs roll/pitch only) — the tracked branch raises BEFORE the
    yaw is used, with or without a cap; just under the guard it composes;
    the dropped-and-logged sub-mode of the same action is unchanged (it
    never uses that yaw) and so is the 5-dim twin."""
    T_bt = _T_bt()
    anchor = np.array([0.2, 0.1, 0.8, 0.0, math.radians(25.0), 0.0])
    a7 = _ident_action7()                                    # dyaw = 0: absolute
    a7[8:, 5] = math.radians(65.0)                           # pitch = 25 + dpitch
    assert PITCH_SINGULAR_RAD == math.radians(85.0)          # exactly; knots 8.. at 90
    for extra in ({}, {"rp_max_rad": math.radians(20.0)}, {"rp_reject_rad": math.radians(30.0)}):
        try:
            compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **extra, **_KW)
            assert False, f"90 deg absolute pitch must raise (tracked, {extra})"
        except ValueError as e:
            assert "singularity" in str(e) and "knot 8" in str(e), str(e)
    # the guard sits at 85 deg absolute: 60.1 deg relative (85.1 abs)
    # raises, 59.9 (84.9 abs) composes (and is then T1-clipped / T2-rejected
    # as usual). The exact edge is not asserted: the composed asin lands an
    # ulp either side of 85 deg. (A nonzero dyaw BETWEEN the two pitches
    # would make the absolute pitch less than the sum -- kept at 0 here.)
    a7[8:, 5] = math.radians(60.1)
    try:
        compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **_KW)
        assert False, "85.1 deg absolute pitch must raise"
    except ValueError as e:
        assert "singularity" in str(e), str(e)
    a7[8:, 5] = math.radians(59.9)
    m_ok, i_ok = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **_KW)
    assert np.all(np.isfinite(m_ok.yaw)) and np.all(np.isfinite(m_ok.p_ned))
    assert abs(i_ok["rp_raw"][1, 8] - math.radians(84.9)) < 1e-9
    # dropped-and-logged: never raises on the pitch, byte-identical to the
    # 5-dim twin as before (the yaw it flies is anchor + dyaw, exact)
    a7[8:, 5] = math.radians(65.0)
    m5, i5 = compose_plan(_as5(a7), anchor, anchor[3:5], T_bt, **_KW)
    md, idr = compose_plan(a7, anchor, anchor[3:5], T_bt, **_KW)
    assert _bytes_equal(md.p_ned, m5.p_ned) and _bytes_equal(md.yaw, m5.yaw)
    assert _bytes_equal(idr["yaw_raw"], i5["yaw_raw"]) and md.rp is None
    assert idr["dropped_rp_deg"] > 60.0
    # ...and rp_reject_rad is inert on the dropped-and-logged and 5-dim paths
    md2, idr2 = compose_plan(a7, anchor, anchor[3:5], T_bt, rp_reject_rad=math.radians(30.0), **_KW)
    assert _bytes_equal(md2.p_ned, md.p_ned) and _bytes_equal(md2.yaw, md.yaw)
    assert _bytes_equal(idr2["rp_raw"], idr["rp_raw"]) and _bytes_equal(idr2["rp_dropped_deg"], idr["rp_dropped_deg"])
    m5b, i5b = compose_plan(_as5(a7), anchor, anchor[3:5], T_bt, rp_reject_rad=math.radians(30.0), **_KW)
    assert _bytes_equal(m5b.p_ned, m5.p_ned) and _bytes_equal(m5b.yaw, m5.yaw) and _bytes_equal(i5b["p_body_raw"], i5["p_body_raw"])
    m10, i10 = compose_plan(_ident_action(), anchor, anchor[3:5], T_bt, **_KW)
    m10b, i10b = compose_plan(_ident_action(), anchor, anchor[3:5], T_bt, rp_reject_rad=math.radians(30.0), **_KW)
    assert _bytes_equal(m10b.p_ned, m10.p_ned) and _bytes_equal(i10b["yaw_raw"], i10["yaw_raw"])


def test_compose_plan_7d_tracked_rp_reject_before_the_t1_clip():
    """(N5) T2 on the compose path: a knot at 40 deg with rp_reject_rad 30
    deg raises (the excess was invisible to the worker's filter while T1
    clipped first); with rp_reject_rad None the T1 clip applies exactly as
    before (same bytes as the pre-fix call); the bound is inclusive; a
    negative / non-finite rp_reject_rad is refused."""
    T_bt = _T_bt()
    anchor = np.array([0.3, -0.2, 0.9, math.radians(2.0), math.radians(-1.0), 0.4])
    a7 = _ident_action7()
    a7[:, 3] = np.linspace(0.0, 0.1, 16)
    a7[5, 4] = math.radians(40.0)                             # one roll knot at ~40 deg
    cap, rej = math.radians(20.0), math.radians(30.0)
    try:
        compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap,
                     rp_reject_rad=rej, **_KW)
        assert False, "40 deg raw attitude over a 30 deg rp_reject must raise"
    except ValueError as e:
        assert "rp_reject" in str(e) and "knot 5" in str(e), str(e)
    try:
        compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_reject_rad=rej, **_KW)
        assert False, "T2 must fire without a T1 cap too"
    except ValueError as e:
        assert "rp_reject" in str(e), str(e)
    # None: the T1 clip as before -- the raw knot is recorded and clipped
    mt, it = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap, **_KW)
    assert it["rp_raw"][0, 5] > rej and it["rp_clipped_n"] >= 1
    assert np.all(np.abs(mt.rp) <= cap + 1e-15)
    assert abs(it["rp_clipped_deg"] - math.degrees(np.abs(it["rp_raw"]).max() - cap)) < 1e-9
    mt2, it2 = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap,
                            rp_reject_rad=None, **_KW)
    assert _bytes_equal(mt2.p_ned, mt.p_ned) and _bytes_equal(mt2.yaw, mt.yaw) and _bytes_equal(mt2.rp, mt.rp)
    # inclusive: a reject bound AT the worst raw knot passes, just under it raises
    worst = float(np.abs(it["rp_raw"]).max())
    m_edge, _ = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap,
                             rp_reject_rad=worst, **_KW)
    assert _bytes_equal(m_edge.rp, mt.rp)
    try:
        compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap,
                     rp_reject_rad=worst - 1e-9, **_KW)
        assert False, "just under the worst raw knot must raise"
    except ValueError as e:
        assert "rp_reject" in str(e), str(e)
    # a plan inside the reject bound composes with or without it, same bytes
    a_ok = _ident_action7(); a_ok[:, 5] = np.linspace(0.0, math.radians(25.0), 16)
    m_a, i_a = compose_plan(a_ok, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap, **_KW)
    m_b, i_b = compose_plan(a_ok, anchor, anchor[3:5], T_bt, track_rp=True, rp_max_rad=cap,
                            rp_reject_rad=rej, **_KW)
    assert _bytes_equal(m_a.p_ned, m_b.p_ned) and _bytes_equal(m_a.rp, m_b.rp) and i_a["rp_clipped_n"] == i_b["rp_clipped_n"] > 0
    for bad in (-0.1, float("nan"), float("inf")):
        try:
            compose_plan(a_ok, anchor, anchor[3:5], T_bt, track_rp=True, rp_reject_rad=bad, **_KW)
            assert False, f"rp_reject_rad {bad} must raise"
        except ValueError as e:
            assert "rp_reject_rad" in str(e), str(e)


def test_resample_knots_att():
    """(2, K) roll/pitch onto the knot grid: knot 0 and the last knot
    PINNED, interior knots the same window means as resample_knots (a
    constant-rate ramp lands exactly on the line), no unwrap (roll/pitch
    are bounded), shape / positivity checks."""
    K = 16
    t = np.arange(K) * OBS_DT
    rp = np.vstack([0.10 * t, -0.05 * t])
    out = resample_knots_att(rp, OBS_DT, KNOT_DT)
    assert out.shape == (2, 6)
    assert np.array_equal(out[:, 0], rp[:, 0]) and np.array_equal(out[:, -1], rp[:, -1])
    t_grid = np.arange(6) * KNOT_DT
    assert np.allclose(out, np.vstack([0.10 * t_grid, -0.05 * t_grid]), atol=1e-12)
    # identical to resample_knots' treatment of the position rows
    p = np.vstack([rp, np.zeros(K)])
    p_r, _y, _g, _j = resample_knots(p, np.zeros(K), np.ones(K), OBS_DT, KNOT_DT)
    assert np.array_equal(p_r[:2], out)
    rng = np.random.default_rng(9)
    noisy = rp + rng.normal(0.0, 0.01, rp.shape)
    out_n = resample_knots_att(noisy, OBS_DT, KNOT_DT)
    p_n, _y, _g, _j = resample_knots(np.vstack([noisy, np.zeros(K)]), np.zeros(K), np.ones(K), OBS_DT, KNOT_DT)
    assert np.array_equal(p_n[:2], out_n)
    for bad in (np.zeros((3, K)), np.zeros(K)):
        try:
            resample_knots_att(bad, OBS_DT, KNOT_DT); assert False, bad.shape
        except ValueError:
            pass
    try:
        resample_knots_att(rp, OBS_DT, 0.0); assert False, "knot_dt 0"
    except ValueError:
        pass


def test_leash_anchor_rp():
    """The attitude leash (2026-09-26): with rp_ref and leash_rp > 0 the
    anchor's roll/pitch move from the measurement toward the reference by
    at most leash_rp per axis (wrapped), info gains drp / drp_applied /
    clipped_rp; without them (the default, and leash_rp = 0) the anchor
    keeps the MEASURED roll/pitch BYTE for byte and info has no rp keys."""
    eta = np.array([1.0, 2.0, 0.5, math.radians(2.0), math.radians(-1.0), 0.3])
    leash_m, leash_yaw, leash_rp = 0.05, math.radians(10.0), math.radians(3.0)
    base, binfo = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw)
    assert not any(k in binfo for k in ("drp", "drp_applied", "clipped_rp"))
    assert _bytes_equal(base[3:5], eta[3:5])
    # rp_ref given but leash_rp 0 -> measured, and no rp keys (byte-identical)
    z, zinfo = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw,
                            rp_ref=(0.5, 0.5), leash_rp=0.0)
    assert _bytes_equal(z, base) and "drp" not in zinfo
    # a +10 deg roll / -1 deg pitch reference offset -> +3 deg / -1 deg
    rp_ref = eta[3:5] + np.array([math.radians(10.0), math.radians(-1.0)])
    anc, info = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw,
                             rp_ref=rp_ref, leash_rp=leash_rp)
    assert abs(anc[3] - (eta[3] + leash_rp)) < 1e-12
    assert abs(anc[4] - (eta[4] + math.radians(-1.0))) < 1e-12
    assert np.allclose(info["drp"], [math.radians(10.0), math.radians(-1.0)], atol=1e-12)
    assert np.allclose(info["drp_applied"], [leash_rp, math.radians(-1.0)], atol=1e-12)
    assert info["clipped_rp"]
    assert np.array_equal(anc[:3], base[:3]) and anc[5] == base[5]      # p / yaw untouched
    # inside the leash: pass-through, not clipped
    rp_ref2 = eta[3:5] + np.array([math.radians(1.0), math.radians(2.0)])
    anc2, info2 = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw,
                               rp_ref=rp_ref2, leash_rp=leash_rp)
    assert np.allclose(anc2[3:5], rp_ref2, atol=1e-12) and not info2["clipped_rp"]
    # wrapped: a reference at -179 deg roll vs a measurement at +179 deg is 2 deg
    e3 = eta.copy(); e3[3] = math.radians(179.0)
    anc3, info3 = leash_anchor(e3, e3[:3], e3[5], leash_m, leash_yaw,
                               rp_ref=(math.radians(-179.0), e3[4]), leash_rp=leash_rp)
    assert abs(info3["drp"][0] - math.radians(2.0)) < 1e-12 and not info3["clipped_rp"]
    assert abs(math.atan2(math.sin(anc3[3] - math.radians(-179.0)),
                          math.cos(anc3[3] - math.radians(-179.0)))) < 1e-12
    # no plan: reference == measurement -> zero offset, nothing clipped
    anc4, info4 = leash_anchor(eta, eta[:3], eta[5], leash_m, leash_yaw,
                               rp_ref=eta[3:5], leash_rp=leash_rp)
    assert np.allclose(anc4, eta) and np.all(info4["drp"] == 0.0) and not info4["clipped_rp"]


def test_filter_plan_yaw_forwards_rp_and_uses_per_knot_attitude():
    """A tracked 7-dim plan through the yaw filter: ``rp`` is forwarded
    unchanged and the body knots are re-derived with EACH knot's own
    roll/pitch (not rp_level) so the TCP still sits where compose_plan put
    it; a plan without rp is re-derived with rp_level exactly as before and
    its output has rp None."""
    T_bt = _T_bt()
    t_bt = T_bt[:3, 3]
    anchor = np.array([0.1, 0.2, 0.5, math.radians(3.0), math.radians(-2.0), 0.0])
    a7 = _ident_action7()
    a7[:, 2] = 0.05 * np.arange(16) * OBS_DT
    a7[:, 3] = np.linspace(0.0, math.radians(4.0), 16)
    a7[:, 4] = np.linspace(0.0, math.radians(6.0), 16)
    a7[:, 5] = np.linspace(0.0, math.radians(-9.0), 16)
    msg, info = compose_plan(a7, anchor, anchor[3:5], T_bt, track_rp=True, **_KW)
    assert msg.rp is not None
    out, yi = _filt(msg, info, anchor, T_bt, yaw_ref0=math.radians(20.0),
                    omega_prev=0.0, dt_prev=0.5, yaw_meas=0.0)
    assert out.rp is not None and np.array_equal(out.rp, msg.rp)
    assert yi["guard_clamped_start"]
    for k in range(out.n_knots):
        tcp = out.p_ned[:, k] + rot_zyx(float(out.rp[0, k]), float(out.rp[1, k]),
                                        float(out.yaw[k])) @ t_bt
        assert np.allclose(tcp, info["p_tcp"][:, k], atol=1e-12), k
    # re-levelling with rp_level instead would have moved the knots
    wrong = np.stack([info["p_tcp"][:, k] - rot_zyx(anchor[3], anchor[4], float(out.yaw[k])) @ t_bt
                      for k in range(out.n_knots)], axis=1)
    assert np.max(np.abs(wrong - out.p_ned)) > 1e-3
    # the 5-dim plan of the same action: rp None in, rp None out, and the
    # body knots re-derived with rp_level (the pre-variant behaviour)
    m5, i5 = compose_plan(_as5(a7), anchor, anchor[3:5], T_bt, **_KW)
    o5, _ = _filt(m5, i5, anchor, T_bt, yaw_ref0=math.radians(20.0),
                  omega_prev=0.0, dt_prev=0.5, yaw_meas=0.0)
    assert o5.rp is None
    for k in range(o5.n_knots):
        tcp = o5.p_ned[:, k] + rot_zyx(anchor[3], anchor[4], float(o5.yaw[k])) @ t_bt
        assert np.allclose(tcp, i5["p_tcp"][:, k], atol=1e-12), k
    # a wrong-shaped rp is refused
    bad = PlanMsg(plan_id=msg.plan_id, t0=msg.t0, dt=msg.dt, p_ned=msg.p_ned,
                  yaw=msg.yaw, g=msg.g, obs_t=msg.obs_t, rp=msg.rp[:, :3])
    try:
        _filt(bad, info, anchor, T_bt, yaw_ref0=None, omega_prev=0.0, dt_prev=0.5)
        assert False, "msg.rp shape mismatch must raise"
    except ValueError:
        pass


# =============================================================================
# 4-DoF byte-identity pin (recorded on the pre-variant code, 2026-09-26)
# =============================================================================
def _sha(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


#: sha256[:16] of the exact output bytes of each 4-DoF path, computed with
#: the SAME fixtures on the code as it stood before the pos_rpy_width change
#: (scratchpad frames/golden_before.json, conda env `robust`, numpy < 2).
#: A mismatch means the variant changed a 4-DoF number. A different BLAS may
#: round a matmul differently -- then re-pin on the pre-variant commit, never
#: on the current one.
_GOLDEN_4DOF = {
    "compose_5d": "9186248b385ce6c0",
    "compose_10d": "0bb0733ba54687ff",
    "leash_default": "e76a02f1c26e050b",
    "filter_plan_yaw": "8f54374d90347445",
}


def _reference_5d_branch(a5, anchor6, rp_level, T_bt, obs_dt, knot_dt, w_open, w_closed):
    """The pre-variant 5-dim arithmetic, transcribed line for line (the
    portable half of the byte-identity proof: same ops in the same order)."""
    a = np.asarray(a5, dtype=float)
    T_bt = np.asarray(T_bt, dtype=float).reshape(4, 4)
    t_bt = T_bt[:3, 3]
    rp = np.asarray(rp_level, dtype=float).reshape(2)
    roll_lvl, pitch_lvl = float(rp[0]), float(rp[1])
    K = a.shape[0]
    anchor6 = np.asarray(anchor6, dtype=float).reshape(6)
    T_ned_tcp_a = T_from_eta(anchor6) @ T_bt
    p_body = np.zeros((3, K))
    dp = a[:, :3]
    dyaw = a[:, 3]
    R_a = T_ned_tcp_a[:3, :3]
    p_a = T_ned_tcp_a[:3, 3]
    p_tcp = p_a[:, None] + R_a @ dp.T
    yaw_raw = np.unwrap(anchor6[5] + dyaw)
    for k in range(K):
        p_body[:, k] = p_tcp[:, k] - rot_zyx(roll_lvl, pitch_lvl, yaw_raw[k]) @ t_bt
    g_raw = np.clip((a[:, 4] - float(w_closed)) / (float(w_open) - float(w_closed)), 0.0, 1.0)
    p_r, yaw_r, g_r, _jitter = resample_knots(p_body, yaw_raw, g_raw, obs_dt, knot_dt)
    return p_r, yaw_r, g_r, p_tcp, p_body, yaw_raw


def test_four_dof_paths_byte_identical_to_pre_variant():
    """Every existing compose_plan branch, the default leash and the yaw
    filter produce the SAME BYTES as before the pos_rpy_width change: (1)
    golden sha256 pins recorded on the pre-variant code; (2) a line-for-line
    transcription of the old 5-dim branch compared with np.array_equal;
    (3) the new trailing fields are None / absent on these paths."""
    T_bt = _T_bt()
    KW = dict(obs_dt=OBS_DT, knot_dt=KNOT_DT, t0=12.5, plan_id=42, obs_t_rel=12.5,
              w_open=W_OPEN, w_closed=W_CLOSED)
    rng = np.random.default_rng(2026)
    anchor = np.array([0.3, -0.2, 0.9, 0.03, -0.02, -2.9])
    a5 = np.zeros((16, 5))
    a5[:, :3] = rng.normal(0, 0.03, (16, 3))
    a5[:, 3] = np.linspace(0, -0.5, 16)
    a5[:, 4] = np.linspace(W_OPEN + 0.01, W_CLOSED - 0.01, 16)
    m5, i5 = compose_plan(a5, anchor, anchor[3:5], T_bt, **KW)
    got = _sha(m5.p_ned, m5.yaw, m5.g, i5["p_tcp"], i5["p_tcp_raw"], i5["p_body_raw"],
               i5["yaw_raw"], i5["rp_dropped_deg"], [i5["dropped_rp_deg"], i5["jitter_rms_mm"]])
    assert got == _GOLDEN_4DOF["compose_5d"], (got, "5-dim compose bytes changed")
    assert m5.rp is None and i5["rp_tracked"] is False and "rp_raw" not in i5
    # (2) the transcribed pre-variant arithmetic, bit for bit
    p_r, yaw_r, g_r, p_tcp, p_body, yaw_raw = _reference_5d_branch(
        a5, anchor, anchor[3:5], T_bt, OBS_DT, KNOT_DT, W_OPEN, W_CLOSED)
    assert np.array_equal(m5.p_ned, p_r) and np.array_equal(m5.yaw, yaw_r)
    assert np.array_equal(m5.g, g_r) and np.array_equal(i5["p_tcp_raw"], p_tcp)
    assert np.array_equal(i5["p_body_raw"], p_body) and np.array_equal(i5["yaw_raw"], yaw_raw)
    # legacy 10-dim branch
    a10 = np.concatenate([mat_to_pose10d(decode_pos_yaw(a5[:, :3], a5[:, 3])), a5[:, 4:5]], axis=1)
    m10, i10 = compose_plan(a10, anchor, anchor[3:5], T_bt, **KW)
    got = _sha(m10.p_ned, m10.yaw, m10.g, i10["p_tcp"], i10["p_tcp_raw"], i10["p_body_raw"],
               i10["yaw_raw"], i10["rp_dropped_deg"], [i10["dropped_rp_deg"], i10["jitter_rms_mm"]])
    assert got == _GOLDEN_4DOF["compose_10d"], (got, "10-dim compose bytes changed")
    assert m10.rp is None and i10["rp_tracked"] is False
    # default leash
    anc, li = leash_anchor(anchor + np.array([0.02, 0.03, 0, 0.01, 0.02, 0.1]), anchor[:3],
                           anchor[5], 0.05, math.radians(10))
    got = _sha(anc, [li["offset_m"], li["dyaw"], li["offset_applied_m"], li["dyaw_applied"]])
    assert got == _GOLDEN_4DOF["leash_default"], got
    assert "drp" not in li
    # yaw filter on the 5-dim plan
    fo, fi = filter_plan_yaw(m5, i5["p_tcp"], -2.85, -2.9, anchor[3:5], T_bt, 0.02, 0.5,
                             rate=math.radians(3), tau=2.0, guard=math.radians(15))
    got = _sha(fo.p_ned, fo.yaw, fo.g, [fi["omega_ref_deg_s"], fi["body_shift_max_m"], fi["yaw0_deg"]])
    assert got == _GOLDEN_4DOF["filter_plan_yaw"], got
    assert fo.rp is None
    # PlanMsg: the new field is trailing and defaults to None (positional
    # constructors in the codebase are untouched)
    pm = PlanMsg(1, 0.0, 0.2, np.zeros((3, 2)), np.zeros(2))
    assert pm.rp is None and pm.g is None and pm.obs_t is None and pm.arrival_t == 0.0


# =============================================================================
# runner (same shape as test_replay.py — works with or without pytest)
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
