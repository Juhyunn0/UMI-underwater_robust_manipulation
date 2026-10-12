#!/usr/bin/env python3
"""rl_ekf.py — robot_localization's EKF, ported, as a second estimator to
compare against the state assembler (2026-10-02; HARDWARE-UNVERIFIED).

WHY "robot_localization". Learning to Swim (arXiv:2410.00120v2, Fig. 2)
fuses "IMU, DVL, April Tag" into a "Fused state estimation (50Hz)" that the
text calls an EKF "implemented in ROS"; the vehicle paper it deploys on
(CUREE, arXiv:2303.03943) says the same in one sentence. Neither publishes
the state vector, the noise or the delay handling, and the released code
(github.com/warplab/isaac-auv-env) is the simulator only. The lab's own
state-estimation repository (github.com/warplab/robot-state-estimation) is
empty but describes itself as "A state estimation codebase for marine robots
using the ROS localization package" — i.e. ``robot_localization``'s
``ekf_localization_node``. So "the paper's way" here is THAT node, ported
from cra-ros-pkg/robot_localization noetic-devel @ 9ef26a57 (2025-01-24):

    filter_base.cpp  reset, processMeasurement, validateDelta,
                     checkMahalanobisThreshold, computeDynamicProcess...
    ekf.cpp          predict (15-state kinematic model + Jacobian),
                     correct (Joseph form, angle-wrapped innovation)
    ros_filter.cpp   integrateMeasurements, periodicUpdate (incl. the
                     "corrected data" publication suppression),
                     differentiateMeasurements, per-topic stamp ordering,
                     imuCallback's three-way split, prepareTwist (lever arm
                     with the STATE's body rate), prepareAcceleration
                     (gravity removed with the MESSAGE's orientation),
                     revertTo / clearExpiredHistory (smooth_lagged_data)

Not ported: 2-D mode, the control input, differential/relative inputs,
set_pose, diagnostics, a full 15x15 process noise (diagonal only), the
subscribers' queue_size 1 (r_l keeps only the newest message per topic when
its spinner falls behind; here every input is delivered), TF lookups — the
three transforms are fixed here (the
tag pose is already the vehicle in the world, the autopilot's IMU frame IS
the body, the DVL mount comes from ``dvl.mount_from_cfg``; an unmeasured
DVL mount is "no transform", and r_l drops a message it cannot transform).

WHAT IS NOT FROM THE PAPER — every number in ``DEFAULTS`` that is not an r_l
default is a CHOICE ([예측] in config/hw_mpc.yaml): which fields each sensor
feeds, the tag/depth/DVL-fallback noise, the IMU noise (mavros' imu plugin
defaults, as the stack a BlueROV2 + ROS user would get). Two choices are
deliberate deviations FOR THE COMPARISON: the tag feeds x, y, yaw (+ z only
when the station's z source is the tag) and the DVL feeds (vx, vy) — the
same information the assembler gets, so a difference between the two is the
FUSION, not extra data. Both are one config edit away from "everything".

FRAMES. r_l assumes REP-103 (ENU world, FLU body). This port runs in the
station's NED world / FRD body. Every formula in predict/correct holds for
any right-handed pair with ZYX Euler angles; the one sign that changes is
gravity: a level accelerometer reads (0, 0, -g) in FRD, so removal subtracts
R^T (0, 0, -g) where r_l subtracts R^T (0, 0, +g).

TIME. r_l's timer fires at ``frequency`` and integrates every measurement
that has ARRIVED (callbacks run between timer cycles) and whose stamp is
<= now. Here the station's 20 Hz tick calls ``advance(t_now)``, which
replays the timer cycles t0 + k/f <= t_now and delivers each input, in
arrival order, before the first cycle at or after its arrival. A tick
therefore sees exactly what a 50 Hz node would have published by then
(minus real-time jitter). Stamps: tag = ``t_capture``, IMU = ATTITUDE host
arrival, depth = baro host arrival, DVL = ``t_valid``; arrival = when the
worker's slot ran.

THE DELAY BEHAVIOUR THIS REPRODUCES (r_l defaults: smooth_lagged_data false,
predict_to_current_time false). A tag frame ~0.1-0.3 s old reaches the queue
after IMU messages stamped later than it; the filter has already predicted
past it, so ``processMeasurement`` sees delta < 0, skips the prediction and
CORRECTS THE CURRENT STATE WITH THE OLD POSE. That is r_l's documented
behaviour without smoothing, and it is the most likely configuration of an
unpublished launch file — so it is the default here. ``smooth_lagged_data:
true`` (with ``history_length_s``) is r_l's own fix and is ported too.

Plain numpy, no Qt — importable and testable on its own.
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------- state layout
# filter_common.h StateMembers, same order.
X, Y, Z, ROLL, PITCH, YAW, VX, VY, VZ, VROLL, VPITCH, VYAW, AX, AY, AZ = \
    range(15)
N = 15
NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
         "vroll", "vpitch", "vyaw", "ax", "ay", "az")
_IDX = {n: i for i, n in enumerate(NAMES)}

RL_SOURCE = ("cra-ros-pkg/robot_localization noetic-devel @ 9ef26a57 "
             "(2025-01-24)")

# filter_base.cpp FilterBase::reset — r_l's default process noise.
RL_PROCESS_NOISE = (0.05, 0.05, 0.06, 0.03, 0.03, 0.06, 0.025, 0.025, 0.04,
                    0.01, 0.01, 0.02, 0.01, 0.01, 0.015)
RL_INITIAL_COV = 1e-9                  # FilterBase::reset
RL_GRAVITY = 9.80665                   # ros_filter.cpp gravitational_acceleration
DBL_MAX = 1.7976931348623157e308       # r_l's default rejection threshold

# mavros imu plugin defaults (the variances it writes into sensor_msgs/Imu):
# linear_acceleration_stdev 0.0003, angular_velocity_stdev 0.02 deg,
# orientation_stdev 1.0 [스펙: mavros_plugins imu.cpp parameter defaults —
# the analogue of "a BlueROV2 on ROS"; the paper does not say].
MAVROS_ACC_STDEV = 0.0003
MAVROS_GYRO_STDEV = 0.02 * math.pi / 180.0
MAVROS_ORIENT_STDEV = 1.0

DEFAULTS = {
    "enabled": True,                   # shadow on every tick
    # --- r_l node parameters (r_l default unless noted)
    "frequency_hz": 50.0,              # paper Fig. 2 (r_l default 30)
    "sensor_timeout_s": None,          # None = 1/frequency (r_l default)
    "smooth_lagged_data": False,
    "history_length_s": 0.0,
    "permit_corrected_publication": False,
    "dynamic_process_noise_covariance": False,
    "process_noise_diag": list(RL_PROCESS_NOISE),
    "initial_estimate_covariance": RL_INITIAL_COV,   # scalar or 15 values
    "gravitational_acceleration": RL_GRAVITY,
    # --- inputs (CHOICES, see the module docstring)
    "tag_fuse": ["x", "y", "yaw"],     # + "z" when nav z_source == tag
    "tag_pos_sigma_m": 0.01,           # [예측]
    "tag_rp_sigma_deg": 3.0,           # [예측] used only if roll/pitch fused
    "tag_yaw_sigma_deg": 1.0,          # [예측]
    "tag_rejection_threshold": None,   # None = r_l default (never reject)
    "depth_sigma_m": 0.01,             # [예측]
    "depth_rejection_threshold": None,
    "imu_fuse": ["roll", "pitch", "vroll", "vpitch", "vyaw",
                 "ax", "ay", "az"],    # compass yaw NOT fused (state_assembler)
    # r_l's default is FALSE; true is what fusing specific force as ax..az
    # physically needs (every r_l IMU launch file sets it).
    "imu_remove_gravitational_acceleration": True,
    "imu_orientation_stdev": MAVROS_ORIENT_STDEV,
    "imu_angular_velocity_stdev": MAVROS_GYRO_STDEV,
    "imu_linear_acceleration_stdev": MAVROS_ACC_STDEV,
    "imu_pose_rejection_threshold": None,
    "imu_twist_rejection_threshold": None,
    "imu_linear_acceleration_rejection_threshold": None,
    "dvl_fuse": ["vx", "vy"],
    "dvl_sigma_m_s": 0.01,             # [예측] only when a report has no covariance
    "dvl_rejection_threshold": None,
    # --- station glue (not r_l)
    "max_cycles_per_advance": 100,     # a stalled worker skips, not floods
    # The accelerometer rides on the ATTITUDE message (mavros' imu/data);
    # older than this, the reading is left out of that message instead of
    # being re-fused as new (a dead SCALED_IMU2 under a live ATTITUDE).
    "imu_accel_stale_s": 0.3,          # [예측]
    # Flying on the EKF only: how far it may sit from a FRESH tag state
    # (debounced 0.25 s) before the run ends, and how close it must be to
    # be switched in or engaged on (the targets are set from the tag state).
    "control_max_tag_divergence_m": 0.15,      # [예측] x, y
    "control_max_z_divergence_m": 0.15,        # [예측]
    "control_max_yaw_divergence_deg": 10.0,    # [예측]
    "control_max_rp_divergence_deg": 10.0,     # [예측] roll, pitch
    "control_max_velocity_divergence_m_s": 0.15,  # [예측] |nu_ekf - nu_tag|
    "control_engage_max_divergence_m": 0.05,   # [예측] x, y, z
    "control_engage_max_yaw_deg": 3.0,         # [예측]
}
KEYS = tuple(DEFAULTS)

TOPIC_TAG = "pose0"
TOPIC_DEPTH = "pose1"
TOPIC_IMU_POSE = "imu0_pose"
TOPIC_IMU_TWIST = "imu0_twist"
TOPIC_IMU_ACC = "imu0_acceleration"
TOPIC_DVL = "twist0"
TOPICS = (TOPIC_TAG, TOPIC_DEPTH, TOPIC_IMU_POSE, TOPIC_IMU_TWIST,
          TOPIC_IMU_ACC, TOPIC_DVL)


def resolve(raw: dict | None, z_source: str = "pressure") -> dict:
    """Merge an ``ekf:`` block over DEFAULTS, rejecting typos (the imu_dr /
    dvl rule: a silently ignored key flies a filter the file does not
    describe). ``z_source`` adds "z" to the tag's fields when the station's
    height comes from the tag, so both estimators see the same z."""
    cfg = {k: (list(v) if isinstance(v, list) else v)
           for k, v in DEFAULTS.items()}
    if raw:
        # A resolved config carries the z source it was resolved for; the
        # ARGUMENT decides (re-resolving must not be a typo error).
        raw = {k: v for k, v in dict(raw).items() if k != "z_source"}
        unknown = set(raw) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown ekf keys {sorted(unknown)}; "
                             f"known: {sorted(KEYS)}")
        cfg.update(raw)
    for k in ("enabled", "smooth_lagged_data", "permit_corrected_publication",
              "dynamic_process_noise_covariance",
              "imu_remove_gravitational_acceleration"):
        if not isinstance(cfg[k], bool):
            raise ValueError(f"ekf.{k} must be true or false, got {cfg[k]!r}")
    for k in ("frequency_hz", "tag_pos_sigma_m", "tag_rp_sigma_deg",
              "tag_yaw_sigma_deg", "depth_sigma_m", "imu_orientation_stdev",
              "imu_angular_velocity_stdev", "imu_linear_acceleration_stdev",
              "dvl_sigma_m_s", "gravitational_acceleration",
              "imu_accel_stale_s", "control_max_tag_divergence_m",
              "control_max_z_divergence_m", "control_max_yaw_divergence_deg",
              "control_max_rp_divergence_deg",
              "control_max_velocity_divergence_m_s",
              "control_engage_max_divergence_m",
              "control_engage_max_yaw_deg"):
        v = cfg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError(f"ekf.{k} must be a positive number, got {v!r}")
        cfg[k] = float(v)
    st = cfg["sensor_timeout_s"]
    if st is None:
        cfg["sensor_timeout_s"] = 1.0 / cfg["frequency_hz"]
    elif isinstance(st, bool) or not isinstance(st, (int, float)) \
            or not float(st) > 0.0:
        raise ValueError(f"ekf.sensor_timeout_s must be null or > 0, "
                         f"got {st!r}")
    else:
        cfg["sensor_timeout_s"] = float(st)
    h = cfg["history_length_s"]
    if isinstance(h, bool) or not isinstance(h, (int, float)) \
            or not math.isfinite(float(h)):
        raise ValueError(f"ekf.history_length_s must be a number, got {h!r}")
    cfg["history_length_s"] = abs(float(h))     # ros_filter.cpp: fabs()
    n = cfg["max_cycles_per_advance"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"ekf.max_cycles_per_advance must be an integer "
                         f">= 1, got {n!r}")
    q = np.asarray(cfg["process_noise_diag"], float).reshape(-1)
    if q.shape != (N,) or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("ekf.process_noise_diag must be 15 non-negative "
                         "numbers")
    cfg["process_noise_diag"] = [float(x) for x in q]
    p0 = cfg["initial_estimate_covariance"]
    p0a = np.asarray(p0, float).reshape(-1)
    if p0a.shape not in ((1,), (N,)) or not np.all(np.isfinite(p0a)) \
            or np.any(p0a <= 0):
        raise ValueError("ekf.initial_estimate_covariance must be one or "
                         "15 positive numbers")
    cfg["initial_estimate_covariance"] = (float(p0a[0]) if p0a.shape == (1,)
                                          else [float(x) for x in p0a])
    for k, allowed in (("tag_fuse", NAMES[:6]), ("imu_fuse", NAMES[3:6]
                       + NAMES[9:15]), ("dvl_fuse", NAMES[6:9])):
        v = cfg[k]
        if not isinstance(v, (list, tuple)) or any(
                str(x) not in allowed for x in v):
            raise ValueError(f"ekf.{k} must be a list drawn from "
                             f"{list(allowed)}, got {v!r}")
        cfg[k] = [str(x) for x in v]
    if z_source == "tag" and "z" not in cfg["tag_fuse"]:
        cfg["tag_fuse"] = cfg["tag_fuse"] + ["z"]
    for k in ("tag_rejection_threshold", "depth_rejection_threshold",
              "imu_pose_rejection_threshold", "imu_twist_rejection_threshold",
              "imu_linear_acceleration_rejection_threshold",
              "dvl_rejection_threshold"):
        v = cfg[k]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not float(v) > 0.0:
            raise ValueError(f"ekf.{k} must be null or > 0, got {v!r}")
        cfg[k] = float(v)
    cfg["z_source"] = str(z_source)
    return cfg


def _thresh(v) -> float:
    return DBL_MAX if v is None else float(v)


def _mask(fields) -> np.ndarray:
    m = np.zeros(N, bool)
    for f in fields:
        m[_IDX[f]] = True
    return m


def _wrap(a: float) -> float:
    """FilterUtilities::clampRotation — on noetic that is
    angles::normalize_angle (fmod; -pi maps to +pi), not the older while
    loop (fidelity review 2026-10-02: ~1e-16 either way)."""
    r = math.fmod(a + math.pi, 2.0 * math.pi)
    if r <= 0.0:
        return r + math.pi
    return r - math.pi


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world, Rz(yaw) Ry(pitch) Rx(roll) — tf2 Matrix3x3::setRPY."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rpy_of(R) -> tuple:
    """ZYX roll, pitch, yaw of a body->world rotation (tf2 getRPY)."""
    R = np.asarray(R, float).reshape(3, 3)
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


class Measurement:
    """filter_common.h / filter_base.h Measurement."""

    __slots__ = ("time", "topic", "z", "R", "update", "thresh")

    def __init__(self, time, topic, z, R, update, thresh):
        self.time = float(time)
        self.topic = topic
        self.z = z
        self.R = R
        self.update = update
        self.thresh = float(thresh)


# =============================================================================
# FilterBase + Ekf
# =============================================================================
class Ekf:
    """filter_base.cpp + ekf.cpp. Public state: ``x`` (15), ``P`` (15x15),
    ``initialized``, ``last_meas_time``."""

    def __init__(self, process_noise=RL_PROCESS_NOISE,
                 initial_cov=RL_INITIAL_COV, sensor_timeout=0.033333333,
                 dynamic_process_noise=False):
        self.Q = np.diag(np.asarray(process_noise, float))
        self._p0 = initial_cov
        self.sensor_timeout = float(sensor_timeout)
        self.dynamic = bool(dynamic_process_noise)
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.x = np.zeros(N)
        p0 = np.asarray(self._p0, float).reshape(-1)
        self.P = (np.eye(N) * float(p0[0]) if p0.size == 1
                  else np.diag(p0))
        self.last_meas_time = 0.0
        self._F = np.eye(N)

    # ---------------------------------------------------------------- predict
    def predict(self, reference_time: float, delta: float) -> None:
        x = self.x
        roll, pitch, yaw = x[ROLL], x[PITCH], x[YAW]
        xv, yv, zv = x[VX], x[VY], x[VZ]
        pv, yawv = x[VPITCH], x[VYAW]
        xa, ya, za = x[AX], x[AY], x[AZ]
        sp, cp = math.sin(pitch), math.cos(pitch)
        cpi = 1.0 / cp
        tp = sp * cpi
        sr, cr = math.sin(roll), math.cos(roll)
        sy, cy = math.sin(yaw), math.cos(yaw)
        d = delta

        F = self._F                    # transferFunction_ (identity + these)
        F[X, VX] = cy * cp * d
        F[X, VY] = (cy * sp * sr - sy * cr) * d
        F[X, VZ] = (cy * sp * cr + sy * sr) * d
        F[X, AX] = 0.5 * F[X, VX] * d
        F[X, AY] = 0.5 * F[X, VY] * d
        F[X, AZ] = 0.5 * F[X, VZ] * d
        F[Y, VX] = sy * cp * d
        F[Y, VY] = (sy * sp * sr + cy * cr) * d
        F[Y, VZ] = (sy * sp * cr - cy * sr) * d
        F[Y, AX] = 0.5 * F[Y, VX] * d
        F[Y, AY] = 0.5 * F[Y, VY] * d
        F[Y, AZ] = 0.5 * F[Y, VZ] * d
        F[Z, VX] = -sp * d
        F[Z, VY] = cp * sr * d
        F[Z, VZ] = cp * cr * d
        F[Z, AX] = 0.5 * F[Z, VX] * d
        F[Z, AY] = 0.5 * F[Z, VY] * d
        F[Z, AZ] = 0.5 * F[Z, VZ] * d
        F[ROLL, VROLL] = d
        F[ROLL, VPITCH] = sr * tp * d
        F[ROLL, VYAW] = cr * tp * d
        F[PITCH, VPITCH] = cr * d
        F[PITCH, VYAW] = -sr * d
        F[YAW, VPITCH] = sr * cpi * d
        F[YAW, VYAW] = cr * cpi * d
        F[VX, AX] = d
        F[VY, AY] = d
        F[VZ, AZ] = d

        h = 0.5 * d * d
        yC, zC = cy * sp * cr + sy * sr, -cy * sp * sr + sy * cr
        dFx_dR = (yC * yv + zC * zv) * d + (yC * ya + zC * za) * h
        dFR_dR = 1.0 + (cr * tp * pv - sr * tp * yawv) * d
        xC, yC, zC = -cy * sp, cy * cp * sr, cy * cp * cr
        dFx_dP = (xC * xv + yC * yv + zC * zv) * d + \
            (xC * xa + yC * ya + zC * za) * h
        dFR_dP = (cpi * cpi * sr * pv + cpi * cpi * cr * yawv) * d
        xC, yC, zC = -sy * cp, -sy * sp * sr - cy * cr, -sy * sp * cr + cy * sr
        dFx_dY = (xC * xv + yC * yv + zC * zv) * d + \
            (xC * xa + yC * ya + zC * za) * h
        yC, zC = sy * sp * cr - cy * sr, -sy * sp * sr - cy * cr
        dFy_dR = (yC * yv + zC * zv) * d + (yC * ya + zC * za) * h
        dFP_dR = (-sr * pv - cr * yawv) * d
        xC, yC, zC = -sy * sp, sy * cp * sr, sy * cp * cr
        dFy_dP = (xC * xv + yC * yv + zC * zv) * d + \
            (xC * xa + yC * ya + zC * za) * h
        xC, yC, zC = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
        dFy_dY = (xC * xv + yC * yv + zC * zv) * d + \
            (xC * xa + yC * ya + zC * za) * h
        yC, zC = cp * cr, -cp * sr
        dFz_dR = (yC * yv + zC * zv) * d + (yC * ya + zC * za) * h
        dFY_dR = (cr * cpi * pv - sr * cpi * yawv) * d
        xC, yC, zC = -cp, -sp * sr, -sp * cr
        dFz_dP = (xC * xv + yC * yv + zC * zv) * d + \
            (xC * xa + yC * ya + zC * za) * h
        dFY_dP = (sr * tp * cpi * pv + cr * tp * cpi * yawv) * d

        J = F.copy()                   # transferFunctionJacobian_
        J[X, ROLL], J[X, PITCH], J[X, YAW] = dFx_dR, dFx_dP, dFx_dY
        J[Y, ROLL], J[Y, PITCH], J[Y, YAW] = dFy_dR, dFy_dP, dFy_dY
        J[Z, ROLL], J[Z, PITCH] = dFz_dR, dFz_dP
        J[ROLL, ROLL], J[ROLL, PITCH] = dFR_dR, dFR_dP
        J[PITCH, ROLL] = dFP_dR
        J[YAW, ROLL], J[YAW, PITCH] = dFY_dR, dFY_dP

        Q = self.Q
        if self.dynamic:
            Q = self._dynamic_q()
        # (no control term: use_control false)
        self.x = F @ x
        self._wrap_state()
        self.P = J @ self.P @ J.T + d * Q

    def _dynamic_q(self) -> np.ndarray:
        """computeDynamicProcessNoiseCovariance."""
        Q = self.Q.copy()
        s = float(np.linalg.norm(self.x[VX:VYAW + 1]))
        V = np.eye(6) * s
        Q[0:6, 0:6] = V @ self.Q[0:6, 0:6] @ V.T
        return Q

    # ---------------------------------------------------------------- correct
    def correct(self, m: Measurement) -> bool | None:
        """-> True applied, False rejected by the Mahalanobis gate, None
        nothing to update (every flagged field NaN/inf) or singular S."""
        idx = [i for i in range(N) if m.update[i] and math.isfinite(m.z[i])]
        if not idx:
            return None
        idx = np.asarray(idx)
        z = m.z[idx]
        xs = self.x[idx]
        R = m.R[np.ix_(idx, idx)].copy()
        for i in range(len(idx)):
            if R[i, i] < 0:
                R[i, i] = abs(R[i, i])
            if R[i, i] < 1e-9:
                R[i, i] = 1e-9
        H = np.zeros((len(idx), N))
        H[np.arange(len(idx)), idx] = 1.0
        pht = self.P @ H.T
        S = H @ pht + R
        if not np.all(np.isfinite(S)):
            # A NaN covariance (e.g. in a device report): Eigen would carry it
            # into the state for good; refused instead (deviation, counted).
            return None
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Eigen would hand back inf/nan here and poison the state; this
            # port refuses the update instead (deviation, counted by caller).
            return None
        K = pht @ S_inv
        innov = z - xs
        for i, j in enumerate(idx):
            if j in (ROLL, PITCH, YAW):
                while innov[i] < -math.pi:
                    innov[i] += 2.0 * math.pi
                while innov[i] > math.pi:
                    innov[i] -= 2.0 * math.pi
        sq = float(innov @ (S_inv @ innov))
        if sq >= m.thresh * m.thresh:
            return False
        self.x = self.x + K @ innov
        G = np.eye(N) - K @ H
        self.P = G @ self.P @ G.T + K @ R @ K.T
        self._wrap_state()
        return True

    def _wrap_state(self) -> None:
        for i in (ROLL, PITCH, YAW):
            self.x[i] = _wrap(float(self.x[i]))

    @staticmethod
    def validate_delta(delta: float) -> float:
        return 0.01 if delta > 100000.0 else delta

    # ------------------------------------------------------- processMeasurement
    def process(self, m: Measurement):
        """-> (result of correct | "init", delta)."""
        delta = 0.0
        if self.initialized:
            delta = m.time - self.last_meas_time
            if delta > 0:
                delta = self.validate_delta(delta)
                self.predict(m.time, delta)
            res = self.correct(m)
        else:
            u = m.update
            self.x = np.where(u, m.z, self.x)
            both = np.outer(u, u)
            self.P = np.where(both, m.R, self.P)
            self.initialized = True
            res = "init"
        if delta >= 0.0:
            self.last_meas_time = m.time
        return res, delta


# =============================================================================
# RosFilter: queue, timer, inputs
# =============================================================================
class RlEkfNode:
    """``ekf_localization_node`` minus ROS. Feed inputs with their arrival
    time, call ``advance(t_now)`` from the control tick, read ``output()``.
    """

    def __init__(self, cfg: dict | None = None, dvl_mount=None,
                 z_source: str = "pressure"):
        """``dvl_mount`` = (R_frd_dvl, r_frd, measured) from
        ``dvl.mount_from_cfg``; None or unmeasured = no transform, and the
        DVL is not fused (r_l drops what it cannot transform)."""
        self.cfg = c = resolve(cfg, z_source=z_source)
        self.period = 1.0 / c["frequency_hz"]
        self.ekf = Ekf(process_noise=c["process_noise_diag"],
                       initial_cov=c["initial_estimate_covariance"],
                       sensor_timeout=c["sensor_timeout_s"],
                       dynamic_process_noise=c["dynamic_process_noise_covariance"])
        self.smooth = c["smooth_lagged_data"]
        self.history_length = c["history_length_s"]
        self.g = c["gravitational_acceleration"]
        self._m_tag = _mask(c["tag_fuse"])
        self._m_depth = _mask(["z"])
        f = set(c["imu_fuse"])
        self._m_imu_pose = _mask([n for n in ("roll", "pitch", "yaw") if n in f])
        self._m_imu_twist = _mask([n for n in ("vroll", "vpitch", "vyaw")
                                   if n in f])
        self._m_imu_acc = _mask([n for n in ("ax", "ay", "az") if n in f])
        self._m_dvl = _mask(c["dvl_fuse"])
        self.dvl_R = self.dvl_r = None
        if dvl_mount is not None and bool(dvl_mount[2]):
            self.dvl_R = np.asarray(dvl_mount[0], float).reshape(3, 3)
            self.dvl_r = np.asarray(dvl_mount[1], float).reshape(3)
        self.reset()

    def reset(self) -> None:
        self.ekf.reset()
        self._queue: list = []         # heap of (time, seq, Measurement)
        self._seq = 0
        self._inbox: list = []         # (arrival, kind, payload), arrival order
        self._seen: dict = {}          # kind -> recent stamps handed in (dedupe)
        self._last_msg_time: dict = {}
        self._t_cycle = None           # the last timer cycle run
        self._state_hist: deque = deque()   # (x, P, last_meas_time)
        self._meas_hist: deque = deque()
        self._ang_acc = np.zeros(3)
        self._last_diff_time = 0.0
        self._last_twist_rot = np.zeros(3)
        self._last_published_stamp = 0.0
        self._published = None
        self.n = {t: {"in": 0, "used": 0, "rejected": 0, "late": 0,
                      "order_drop": 0, "void": 0} for t in TOPICS}
        self.n_cycles = 0
        self.n_timeout_predicts = 0
        self.n_reverts = 0
        self.n_skipped_cycles = 0
        self.n_corrected_suppressed = 0
        self.n_dvl_no_tf = 0
        self.late_max_s = {t: 0.0 for t in TOPICS}
        self.last_tag_used = None      # (stamp, lateness s) of the last fused tag
        self.last_used_t: dict = {}    # topic -> stamp of its last FUSED message
        self._late_sum = {t: 0.0 for t in TOPICS}

    # ------------------------------------------------------------------ inputs
    # Each feed_* is a ROS callback that has not run yet: it records the
    # arrival and is delivered by ``advance`` before the first timer cycle at
    # or after that arrival — prepare* can depend on the state (prepareTwist
    # reads the state's body rate), so it must run THEN, not now.
    #
    # The station re-delivers SNAPSHOTS (the same NavFix / VehicleImu object
    # can reach a slot more than once); a ROS topic delivers each message
    # once. So a stamp equal to the last one handed in for that input is the
    # same message again and is not queued — r_l's own ordering rule passes
    # equal stamps and would fuse it twice.
    def _push(self, kind: str, stamp: float, t_arrival: float,
              payload) -> None:
        # The last few stamps, not only the last one: an A, B, A, B
        # re-delivery would otherwise queue B twice (fidelity review).
        seen = self._seen.setdefault(kind, deque(maxlen=16))
        if stamp in seen:
            return
        seen.append(stamp)
        item = (float(t_arrival), kind, payload)
        if self._inbox and item[0] < self._inbox[-1][0]:
            k = len(self._inbox)
            while k > 0 and self._inbox[k - 1][0] > item[0]:
                k -= 1
            self._inbox.insert(k, item)
        else:
            self._inbox.append(item)

    def feed_tag(self, fix, t_arrival: float) -> None:
        if fix is None or not getattr(fix, "ok", False) \
                or fix.p_ned is None or fix.R_ned_body is None:
            return
        tc = float(fix.t_capture)
        self._push("tag", tc, t_arrival,
                   (tc, tuple(float(v) for v in fix.p_ned),
                    tuple(float(v) for v in fix.R_ned_body)))

    def feed_depth(self, z_world: float, t_stamp: float,
                   t_arrival: float) -> None:
        self._push("depth", float(t_stamp), t_arrival,
                   (float(t_stamp), float(z_world)))

    def feed_imu(self, imu, t_arrival: float) -> None:
        """One ATTITUDE-stamped IMU message (mavros imu/data: orientation +
        rates from ATTITUDE, the latest accelerometer reading beside it)."""
        if imu is None or imu.roll is None or imu.p is None:
            return
        t = float(imu.t_att if imu.t_att is not None else imu.stamp)
        t_imu = getattr(imu, "t_imu", None)
        acc = None
        if imu.ax is not None and t_imu is not None \
                and t - float(t_imu) <= self.cfg["imu_accel_stale_s"]:
            # (station glue) only a CURRENT accelerometer reading rides on
            # this message; a frozen one would be fused as new every time.
            acc = (float(imu.ax), float(imu.ay), float(imu.az))
        # NaN, not 0, for a missing compass: correct() skips NaN fields.
        yaw = float(imu.yaw) if imu.yaw is not None else float("nan")
        self._push("imu", t, t_arrival,
                   (t, float(imu.roll), float(imu.pitch), yaw,
                    (float(imu.p), float(imu.q), float(imu.r)), acc))

    def feed_dvl(self, sample, t_arrival: float) -> None:
        """A bottom-track report. Only ``valid`` velocity reports, as a ROS
        DVL driver would publish a TwistWithCovariance."""
        if sample is None or not bool(getattr(sample, "valid", False)) \
                or getattr(sample, "kind", "velocity") != "velocity" \
                or sample.v_dvl is None:
            return
        cov = getattr(sample, "covariance", None)
        tv = float(sample.t_valid)
        self._push("dvl", tv, t_arrival,
                   (tv, tuple(float(v) for v in sample.v_dvl),
                    None if cov is None else tuple(float(v) for v in cov)))

    # --------------------------------------------------------------- callbacks
    def _accept_stamp(self, topic: str, stamp: float) -> bool:
        """The per-topic ordering rule every r_l callback applies: a message
        older than the last one on its topic is dropped (equal passes);
        stamps <= the last set_pose time (0 here) are ignored."""
        self.n[topic]["in"] += 1
        if stamp <= 0.0:
            self.n[topic]["order_drop"] += 1
            return False
        last = self._last_msg_time.setdefault(topic, stamp)
        if stamp < last:
            self.n[topic]["order_drop"] += 1
            return False
        self._last_msg_time[topic] = stamp
        return True

    def _enqueue(self, topic, stamp, z, R, update, thresh) -> None:
        m = Measurement(stamp, topic, z, R, update, thresh)
        heapq.heappush(self._queue, (m.time, self._seq, m))
        self._seq += 1

    def _deliver(self, kind: str, payload) -> None:
        c = self.cfg
        if kind == "tag":
            stamp, p, Rb = payload
            if not self._accept_stamp(TOPIC_TAG, stamp):
                return
            roll, pitch, yaw = rpy_of(Rb)
            z = np.zeros(N)
            z[X:Z + 1] = p
            z[ROLL], z[PITCH], z[YAW] = roll, pitch, yaw
            R = np.zeros((N, N))
            sp = c["tag_pos_sigma_m"] ** 2
            srp = math.radians(c["tag_rp_sigma_deg"]) ** 2
            sy = math.radians(c["tag_yaw_sigma_deg"]) ** 2
            R[X, X] = R[Y, Y] = R[Z, Z] = sp
            R[ROLL, ROLL] = R[PITCH, PITCH] = srp
            R[YAW, YAW] = sy
            self._enqueue(TOPIC_TAG, stamp, z, R, self._m_tag.copy(),
                          _thresh(c["tag_rejection_threshold"]))
        elif kind == "depth":
            stamp, zw = payload
            if not self._accept_stamp(TOPIC_DEPTH, stamp):
                return
            z = np.zeros(N)
            z[Z] = zw
            R = np.zeros((N, N))
            R[Z, Z] = c["depth_sigma_m"] ** 2
            self._enqueue(TOPIC_DEPTH, stamp, z, R, self._m_depth.copy(),
                          _thresh(c["depth_rejection_threshold"]))
        elif kind == "imu":
            self._deliver_imu(payload)
        elif kind == "dvl":
            self._deliver_dvl(payload)

    def _deliver_imu(self, payload) -> None:
        """imuCallback: orientation -> poseCallback(imuData=true), rates ->
        twistCallback, acceleration -> accelerationCallback; each on its own
        topic name with its own ordering memory and threshold. The IMU frame
        is the body (identity transform, zero offset)."""
        c = self.cfg
        stamp, roll, pitch, yaw, gyro, acc = payload
        if self._m_imu_pose.any() and self._accept_stamp(TOPIC_IMU_POSE, stamp):
            z = np.zeros(N)
            z[ROLL], z[PITCH] = _wrap(roll), _wrap(pitch)
            upd = self._m_imu_pose.copy()
            if math.isfinite(yaw):
                z[YAW] = _wrap(yaw)
            else:
                # no compass in this message: the field is not offered at
                # all (a NaN would initialize the state to NaN — r_l never
                # sees one from mavros, so this input is the port's own)
                upd[YAW] = False
            R = np.zeros((N, N))
            v = c["imu_orientation_stdev"] ** 2
            R[ROLL, ROLL] = R[PITCH, PITCH] = R[YAW, YAW] = v
            self._enqueue(TOPIC_IMU_POSE, stamp, z, R, upd,
                          _thresh(c["imu_pose_rejection_threshold"]))
        if self._m_imu_twist.any() and self._accept_stamp(TOPIC_IMU_TWIST,
                                                          stamp):
            z = np.zeros(N)
            z[VROLL:VYAW + 1] = gyro
            # prepareTwist: linear = R*0 + origin x w_state = 0 (zero offset)
            R = np.zeros((N, N))
            v = c["imu_angular_velocity_stdev"] ** 2
            R[VROLL, VROLL] = R[VPITCH, VPITCH] = R[VYAW, VYAW] = v
            self._enqueue(TOPIC_IMU_TWIST, stamp, z, R,
                          self._m_imu_twist.copy(),
                          _thresh(c["imu_twist_rejection_threshold"]))
        if acc is not None and self._m_imu_acc.any() \
                and self._accept_stamp(TOPIC_IMU_ACC, stamp):
            a = np.asarray(acc, float)
            if c["imu_remove_gravitational_acceleration"]:
                # prepareAcceleration, with the MESSAGE's orientation:
                # acc -= R^-1 * normAcc. normAcc is (0,0,+g) in r_l's ENU and
                # (0,0,-g) here — what a level FRD accelerometer reads.
                Rm = rot_zyx(roll, pitch, yaw if math.isfinite(yaw) else 0.0)
                a = a - Rm.T @ np.array([0.0, 0.0, -self.g])
            # + origin x alpha - (origin x w) x w = 0 (zero offset)
            z = np.zeros(N)
            z[AX:AZ + 1] = a
            R = np.zeros((N, N))
            v = c["imu_linear_acceleration_stdev"] ** 2
            R[AX, AX] = R[AY, AY] = R[AZ, AZ] = v
            self._enqueue(TOPIC_IMU_ACC, stamp, z, R, self._m_imu_acc.copy(),
                          _thresh(c["imu_linear_acceleration_rejection_threshold"]))

    def _deliver_dvl(self, payload) -> None:
        """twistCallback -> prepareTwist with the DVL's mount transform."""
        c = self.cfg
        stamp, v, cov = payload
        if self.dvl_R is None:
            # lookupTransformSafe failed: r_l drops the message.
            self.n_dvl_no_tf += 1
            return
        if not self._accept_stamp(TOPIC_DVL, stamp):
            return
        Rm, r = self.dvl_R, self.dvl_r
        w_state = self.ekf.x[VROLL:VYAW + 1].copy()
        # twistLin = R * v + origin.cross(stateTwistRot)
        lin = Rm @ np.asarray(v, float) + np.cross(r, w_state)
        mask = self._m_dvl[VX:VZ + 1].astype(float)
        mrot = Rm @ np.diag(mask)
        upd = np.zeros(N, bool)
        upd[VX:VZ + 1] = np.linalg.norm(mrot, axis=1) >= 1e-6
        C = np.zeros((6, 6))
        if cov is not None:
            C[0:3, 0:3] = np.asarray(cov, float).reshape(3, 3)
        else:
            C[0:3, 0:3] = np.eye(3) * c["dvl_sigma_m_s"] ** 2
        R6 = np.eye(6)
        R6[0:3, 0:3] = Rm
        R6[3:6, 3:6] = Rm
        C = R6 @ C @ R6.T
        z = np.zeros(N)
        z[VX:VZ + 1] = lin
        Rfull = np.zeros((N, N))
        Rfull[VX:VYAW + 1, VX:VYAW + 1] = C
        self._enqueue(TOPIC_DVL, stamp, z, Rfull, upd,
                      _thresh(c["dvl_rejection_threshold"]))

    # ------------------------------------------------------------------- timer
    @property
    def timer_phase(self):
        """The last timer cycle run (None before the first) — recorded so an
        offline replay can put its cycles where the live node's were."""
        return self._t_cycle

    def align_timer(self, t_cycle: float) -> None:
        """Before the first ``advance``: run the cycles on the grid
        t_cycle + k/f instead of starting the grid at the first call."""
        if self._t_cycle is None:
            self._t_cycle = float(t_cycle)

    def advance(self, t_now: float) -> int:
        """Run every timer cycle due by ``t_now``; -> cycles run."""
        t_now = float(t_now)
        if self._t_cycle is None:
            nxt = t_now                # the timer's phase starts here
        else:
            nxt = self._t_cycle + self.period
            if nxt < t_now - 1e3 * self.period:
                # far in the past (an aligned replay grid, or a long stall):
                # jump along the grid, keeping its phase, and count it
                k = math.floor((t_now - nxt) / self.period)
                nxt += k * self.period
                self.n_skipped_cycles += k
        owed = int(math.floor((t_now - nxt) / self.period + 1e-9)) + 1
        cap = int(self.cfg["max_cycles_per_advance"])
        if owed > cap:
            # A stalled station would otherwise replay seconds of cycles in
            # one tick. Skip to the last `cap` (the inputs are still all
            # delivered, in order, before the first cycle that runs).
            skip = owed - cap
            self.n_skipped_cycles += skip
            nxt += skip * self.period
        ran = 0
        while nxt <= t_now + 1e-9:
            k = 0
            while k < len(self._inbox) and self._inbox[k][0] <= nxt + 1e-9:
                _, kind, payload = self._inbox[k]
                self._deliver(kind, payload)
                k += 1
            if k:
                del self._inbox[:k]
            self._cycle(nxt)
            self._t_cycle = nxt
            nxt += self.period
            ran += 1
        return ran

    def _cycle(self, tc: float) -> None:
        """periodicUpdate (toggled on, filter enabled)."""
        self.n_cycles += 1
        self._integrate(tc)
        self._differentiate(tc)
        self._publish(tc)
        if self.smooth:
            self._clear_expired(self.ekf.last_meas_time - self.history_length)

    def _integrate(self, tc: float) -> None:
        """integrateMeasurements."""
        f = self.ekf
        predict_now = False            # predict_to_current_time: false
        if self._queue:
            first = self._queue[0][2]
            if self.smooth and first.time < f.last_meas_time:
                self.n_reverts += 1
                self._revert_to(first.time - 1e-9)
            while self._queue:
                m = self._queue[0][2]
                if m.time > tc:
                    break
                heapq.heappop(self._queue)
                self._process(m)
                if self.smooth:
                    self._meas_hist.append(m)
                    if not self._queue or abs(self._queue[0][2].time
                                              - f.last_meas_time) > 1e-9:
                        self._state_hist.append(
                            (f.x.copy(), f.P.copy(), f.last_meas_time))
        elif f.initialized:
            if tc - f.last_meas_time >= f.sensor_timeout:
                predict_now = True
        if f.initialized and predict_now:
            delta = tc - f.last_meas_time
            delta = f.validate_delta(delta)
            f.predict(tc, delta)
            f.last_meas_time = f.last_meas_time + delta
            self.n_timeout_predicts += 1

    def _process(self, m: Measurement) -> None:
        f = self.ekf
        res, delta = f.process(m)
        n = self.n[m.topic]
        if res == "init" or res is True:
            n["used"] += 1
            self.last_used_t[m.topic] = max(
                m.time, self.last_used_t.get(m.topic, m.time))
        elif res is False:
            n["rejected"] += 1
        else:
            n["void"] += 1
        if delta < 0.0:
            # Corrected into a state that had already been predicted PAST it
            # (no smoothing): this is the lag the comparison is about.
            n["late"] += 1
            self._late_sum[m.topic] += -delta
            self.late_max_s[m.topic] = max(self.late_max_s[m.topic], -delta)
        if m.topic == TOPIC_TAG and (res is True or res == "init"):
            self.last_tag_used = (m.time, max(0.0, -delta))

    def _differentiate(self, tc: float) -> None:
        """differentiateMeasurements: the angular acceleration r_l publishes
        on accel/filtered."""
        f = self.ekf
        if not f.initialized:
            return
        dt = tc - self._last_diff_time
        w = f.x[VROLL:VYAW + 1].copy()
        if dt > 0:
            self._ang_acc = (w - self._last_twist_rot) / dt
        self._last_twist_rot = w
        self._last_diff_time = tc

    def _publish(self, tc: float) -> None:
        f = self.ekf
        if not f.initialized:
            return
        stamp = f.last_meas_time
        corrected = (not self.cfg["permit_corrected_publication"]
                     and self._last_published_stamp >= stamp)
        if corrected:
            self.n_corrected_suppressed += 1
        else:
            self._published = {"x": f.x.copy(), "P": f.P.copy(),
                               "stamp": stamp, "t_cycle": tc,
                               "ang_acc": self._ang_acc.copy()}
        self._last_published_stamp = stamp

    def _revert_to(self, time: float) -> bool:
        """revertTo."""
        f = self.ekf
        last = None
        while self._state_hist and self._state_hist[-1][2] > time:
            last = self._state_hist.pop()
        ok = False
        if self._state_hist:
            ok = True
            last = self._state_hist[-1]
        if last is not None:
            f.x, f.P, f.last_meas_time = last[0].copy(), last[1].copy(), last[2]
            while self._meas_hist and self._meas_hist[-1].time > time:
                m = self._meas_hist.pop()
                if last[2] <= m.time:
                    heapq.heappush(self._queue, (m.time, self._seq, m))
                    self._seq += 1
        return ok

    def _clear_expired(self, cutoff: float) -> None:
        while self._meas_hist and self._meas_hist[0].time < cutoff:
            self._meas_hist.popleft()
        while self._state_hist and self._state_hist[0][2] < cutoff:
            self._state_hist.popleft()

    # ------------------------------------------------------------------ output
    def output(self):
        """The last PUBLISHED odometry (+ accel), or None before the first
        measurement. A dict of copies: x, P, stamp (lastMeasurementTime),
        t_cycle, ang_acc."""
        return self._published

    def stale_inputs(self, t_now: float, tag_s: float, imu_s: float,
                     depth_s: float = 1.5, acc_s: float | None = None) -> str:
        """"" if every input the filter is configured to fuse was FUSED
        recently enough, else why not. This, not the published stamp, is
        the EKF's freshness: with nothing queued r_l predicts to now and
        stamps it now, so its output never looks old (safety review
        2026-10-02). The DVL is optional and not required."""
        need = [(TOPIC_TAG, tag_s, "tag pose")]
        if self._m_imu_pose.any():
            need.append((TOPIC_IMU_POSE, imu_s, "IMU attitude"))
        if self._m_imu_twist.any():
            need.append((TOPIC_IMU_TWIST, imu_s, "IMU rate"))
        if self._m_imu_acc.any():
            # SCALED_IMU2 can arrive at 2-3 Hz (ArduSub's default rate, no
            # --mavlink-set-rates): a looser limit, or the gate flaps.
            need.append((TOPIC_IMU_ACC, imu_s if acc_s is None else acc_s,
                         "accelerometer"))
        if self.cfg["z_source"] == "pressure":
            need.append((TOPIC_DEPTH, depth_s, "depth"))
        for topic, lim, what in need:
            ts = self.last_used_t.get(topic)
            if ts is None:
                return f"no {what} fused yet"
            if t_now - ts > lim:
                return f"no {what} fused for {t_now - ts:.2f} s (> {lim:g} s)"
        return ""

    def late_mean_s(self, topic: str) -> float | None:
        k = self.n[topic]["late"]
        return self._late_sum[topic] / k if k else None

    def meta(self) -> dict:
        c = dict(self.cfg)
        return {
            "source": RL_SOURCE,
            "config": c,
            "dvl_transform": (None if self.dvl_R is None else
                              {"R_frd_dvl": self.dvl_R.round(6).tolist(),
                               "r_frd_m": self.dvl_r.round(4).tolist()}),
            "counts": {t: dict(v) for t, v in self.n.items()},
            "late_max_s": {t: round(v, 4) for t, v in self.late_max_s.items()},
            "late_mean_s": {t: (None if self.late_mean_s(t) is None
                                else round(self.late_mean_s(t), 4))
                            for t in TOPICS},
            "cycles": int(self.n_cycles),
            "timeout_predicts": int(self.n_timeout_predicts),
            "reverts": int(self.n_reverts),
            "skipped_cycles": int(self.n_skipped_cycles),
            "corrected_suppressed": int(self.n_corrected_suppressed),
            "dvl_dropped_no_transform": int(self.n_dvl_no_tf),
        }


# =============================================================================
# station glue
# =============================================================================
def to_meas(out) -> dict | None:
    """Published EKF output -> the {eta, nu, nudot} the controller eats.

    eta and nu are the state as published (world NED pose, body FRD twist).
    nudot is NOT an r_l output in the EAOB's sense: r_l's linear
    acceleration state is driven by gravity-removed specific force, i.e. the
    inertial acceleration in the body frame, while the EAOB wants d(nu)/dt;
    the two differ by w x v (the transport term state_assembler subtracts
    too). The angular part is r_l's own differentiated body rate."""
    if out is None:
        return None
    x = out["x"]
    nu = x[VX:VYAW + 1].copy()
    a_lin = x[AX:AZ + 1] - np.cross(x[VROLL:VYAW + 1], x[VX:VZ + 1])
    return {"eta": x[X:YAW + 1].copy(), "nu": nu,
            "nudot": np.concatenate([a_lin, np.asarray(out["ang_acc"],
                                                       float)])}


def finite(out) -> bool:
    return bool(out is not None and np.all(np.isfinite(out["x"]))
                and np.all(np.isfinite(out["P"])))


def sigma(out, idx) -> float:
    """sqrt of the summed variances of ``idx`` (e.g. (X, Y))."""
    if out is None:
        return float("nan")
    return math.sqrt(max(0.0, float(sum(out["P"][i, i] for i in idx))))
