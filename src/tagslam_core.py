"""
Camera-agnostic TagSLAM core: AprilTag detection, refractive PnP, GTSAM iSAM2
backend, run-folder recorder, and trajectory IO/visualization writers.

Anything ZED-SDK-specific (pyzed, IMU gravity helper, parse_args/main) lives in
the per-camera entry script (e.g. ``zed2_underwater_tagslam.py``); this module
is intended to be imported by multiple front-end scripts so they share the same
detection pipeline, optimizer, and on-disk output format.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable
import warnings

import cv2
import gtsam
from gtsam import BetweenFactorPose3, NonlinearFactorGraph, Pose3, PriorFactorPose3
from gtsam import Point3, Rot3, Values
from gtsam.symbol_shorthand import L, X
import numpy as np
from pupil_apriltags import Detector

from tagslam.visualization import (
    compute_pool_geometry,
    normalize_pool_config,
    pool_geometry_json,
)


WATER_SCALE_FACTOR = 3.6
DEFAULT_ANCHOR_TAG_ID = 1
DEFAULT_TAG_SIZE_M = 0.085
DEFAULT_POOL_DEPTH_M = 1.143
DEFAULT_WATER_REFRACTIVE_INDEX = 1.333
DEFAULT_AIR_REFRACTIVE_INDEX = 1.0
PLOT_Z_SCALE = 0.5


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    corners: np.ndarray
    center: tuple[int, int]
    rvec: np.ndarray
    raw_tvec_m: np.ndarray
    scaled_tvec_m: np.ndarray
    camera_T_tag: Pose3
    decision_margin: float
    hamming: int
    tag_area_px: float
    off_nadir_deg: float
    image_eccentricity: float
    tag_tilt_deg: float
    reprojection_error_px: float
    applied_scale: float


@dataclass
class RefractiveContext:
    water_cfg: dict[str, object]
    backend: object | None = None
    warned_bootstrap_fallback: bool = False
    pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    solve_count: int = 0
    fallback_count: int = 0
    total_s: float = 0.0
    frame_count: int = 0
    frame_total_s: float = 0.0
    last_frame_s: float = 0.0
    last_frame_tag_count: int = 0
    # Unit gravity (down) direction expressed in the camera frame for the current
    # frame. When non-None, the refractive interface normal is taken as -gravity
    # instead of being derived from the anchor frame / near-nadir bootstrap.
    imu_gravity_camera: np.ndarray | None = None


@dataclass(frozen=True)
class RawTagCandidate:
    tag_id: int
    corners: np.ndarray
    center: tuple[int, int]
    raw_rvec: np.ndarray
    raw_tvec_m: np.ndarray
    raw_camera_T_tag: Pose3
    decision_margin: float
    hamming: int
    tag_area_px: float
    off_nadir_deg: float
    image_eccentricity: float
    tag_tilt_deg: float
    reprojection_error_px: float


@dataclass(frozen=True)
class BackendUpdate:
    optimized: bool
    status: str
    camera_pose: Pose3 | None
    tag_poses: dict[int, Pose3]
    camera_index: int | None
    used_observation_count: int = 0
    camera_position_std_cm: float | None = None
    anchor_tag_id: int = DEFAULT_ANCHOR_TAG_ID


@dataclass(frozen=True)
class TrajectorySample:
    camera_index: int
    elapsed_s: float
    detected_tag_ids: tuple[int, ...]
    image_path: str | None = None
    timestamp_unix: float | None = None
    timestamp_monotonic: float | None = None
    extra: dict[str, float] | None = None


@dataclass(frozen=True)
class FittedPlane:
    point: np.ndarray
    normal: np.ndarray


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return vector / norm


def normalize_water_config(
    water_cfg: dict[str, object] | None,
    pool_cfg: dict[str, object] | None,
) -> dict[str, object]:
    pool = normalize_pool_config(pool_cfg)
    cfg = dict(water_cfg or {})
    cfg.setdefault("surface_height_m", pool.get("water_depth_m", pool.get("depth_m", DEFAULT_POOL_DEPTH_M)))
    cfg.setdefault("n_water", DEFAULT_WATER_REFRACTIVE_INDEX)
    cfg.setdefault("n_air", DEFAULT_AIR_REFRACTIVE_INDEX)
    # Existing TagSLAM data has the camera above the floor at negative world Z,
    # so the configurable "up" direction defaults to -Z. Change this vector
    # when an IMU/gravity calibration supplies the true anchor-frame up axis.
    cfg.setdefault("up_axis_world", [0.0, 0.0, -1.0])

    cfg["surface_height_m"] = float(cfg["surface_height_m"])
    cfg["n_water"] = float(cfg["n_water"])
    cfg["n_air"] = float(cfg["n_air"])
    cfg["up_axis_world"] = [float(value) for value in cfg["up_axis_world"]]
    if cfg["surface_height_m"] <= 0:
        raise ValueError("water.surface_height_m must be positive")
    if cfg["n_air"] <= 0:
        raise ValueError("water.n_air must be positive")
    if cfg["n_water"] <= cfg["n_air"]:
        raise ValueError("water.n_water must be greater than water.n_air")
    if np.linalg.norm(np.asarray(cfg["up_axis_world"], dtype=np.float64)) < 1e-9:
        raise ValueError("water.up_axis_world must be nonzero")
    return cfg


def parse_simple_yaml(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    current_section: dict[str, object] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            result[section_name] = current_section
            continue

        target = current_section if current_section is not None else result
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        target[key.strip()] = parse_simple_yaml_value(value.strip())

    return result


def parse_simple_yaml_value(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_simple_yaml_value(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def make_detector(args: argparse.Namespace) -> Detector:
    return Detector(
        families=args.tag_family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=1,
        decode_sharpening=args.decode_sharpening,
        debug=0,
    )


def tag_object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    # pupil-apriltags reports corners in the same ideal-tag order documented
    # for its homography: (-1,1), (1,1), (1,-1), (-1,-1). The object frame
    # here uses OpenCV camera handedness: +x right, +y down, +z forward.
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def pose3_from_rvec_tvec(rvec: np.ndarray, tvec_m: np.ndarray) -> Pose3:
    rotation_matrix, _jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return Pose3(
        Rot3(rotation_matrix.astype(np.float64)),
        Point3(float(tvec_m[0]), float(tvec_m[1]), float(tvec_m[2])),
    )


def quadrilateral_area_px(corners: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))))


def off_nadir_angle_deg(tvec_m: np.ndarray) -> float:
    tvec = np.asarray(tvec_m, dtype=np.float64).reshape(3)
    lateral = float(np.hypot(tvec[0], tvec[1]))
    forward = max(float(abs(tvec[2])), 1e-9)
    return float(np.degrees(np.arctan2(lateral, forward)))


def image_eccentricity(center: np.ndarray, intrinsics: CameraIntrinsics, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    cx = float(intrinsics.camera_matrix[0, 2])
    cy = float(intrinsics.camera_matrix[1, 2])
    radius = float(np.hypot(center[0] - cx, center[1] - cy))
    max_radius = float(np.hypot(max(cx, width - cx), max(cy, height - cy)))
    return radius / max(max_radius, 1e-9)


def tag_tilt_deg_from_rvec(rvec: np.ndarray) -> float:
    rotation_matrix, _jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    # Fronto-parallel tags have their plane normal nearly collinear with the
    # camera optical axis. Use abs() because solvePnP tag normal sign can flip
    # with tag coordinate conventions.
    normal_z = float(np.clip(abs(rotation_matrix[2, 2]), 0.0, 1.0))
    return float(np.degrees(np.arccos(normal_z)))


def reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float:
    projected, _jacobian = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec_m, dtype=np.float64).reshape(3, 1),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    residuals = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def transform_object_points(object_points: np.ndarray, rvec: np.ndarray, tvec_m: np.ndarray) -> np.ndarray:
    rotation_matrix, _jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    return np.asarray(object_points, dtype=np.float64) @ rotation_matrix.T + np.asarray(
        tvec_m,
        dtype=np.float64,
    ).reshape(1, 3)


def rotation_error_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rot_a, _jacobian = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3))
    rot_b, _jacobian = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3))
    delta = rot_a @ rot_b.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def solve_interface_lateral_newton(
    lateral_m: np.ndarray,
    air_depth_m: float,
    water_depth_m: np.ndarray,
    n_air: float,
    n_water: float,
    max_iterations: int = 10,
    tolerance: float = 1e-11,
) -> np.ndarray:
    b = np.asarray(lateral_m, dtype=np.float64)
    a = np.asarray(water_depth_m, dtype=np.float64)
    d = float(air_depth_m)
    q = np.zeros_like(b)
    active = b > 1e-12
    if not np.any(active):
        return q

    b_active = b[active]
    a_active = a[active]
    q_active = b_active * (n_water * d) / np.maximum(n_air * a_active + n_water * d, 1e-12)
    q_active = np.clip(q_active, 0.0, b_active)
    low = np.zeros_like(q_active)
    high = b_active.copy()

    for _ in range(max_iterations):
        air_len = np.sqrt(d * d + q_active * q_active)
        water_lateral = b_active - q_active
        water_len = np.sqrt(a_active * a_active + water_lateral * water_lateral)
        value = n_air * q_active / air_len - n_water * water_lateral / water_len
        derivative = (
            n_air * d * d / np.maximum(air_len**3, 1e-18)
            + n_water * a_active * a_active / np.maximum(water_len**3, 1e-18)
        )

        high = np.where(value >= 0.0, q_active, high)
        low = np.where(value < 0.0, q_active, low)
        step = value / np.maximum(derivative, 1e-18)
        candidate = q_active - step
        midpoint = 0.5 * (low + high)
        candidate = np.where((candidate > low) & (candidate < high), candidate, midpoint)
        max_delta = float(np.max(np.abs(candidate - q_active)))
        q_active = candidate
        if max_delta <= tolerance or float(np.max(np.abs(value))) <= tolerance:
            break

    q[active] = q_active
    return q


def project_refractive(
    points_cam: np.ndarray,
    n_hat: np.ndarray,
    d_air: float,
    n_air: float,
    n_water: float,
    intrinsics: CameraIntrinsics,
    ray_max_iterations: int = 10,
    ray_tolerance: float = 1e-11,
) -> np.ndarray:
    """
    Project underwater camera-frame points through one flat air-water interface.

    The interface plane is n_hat dot x = d_air in the camera frame. n_hat points
    from the camera-side air into the water. Each point must lie on the water
    side of the interface. The refraction point is solved by Fermat/Snell in
    the 2D plane spanned by n_hat and the underwater point, then the in-air
    segment is projected with the existing OpenCV pinhole + distortion model.
    """
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    normal = normalize_vector(np.asarray(n_hat, dtype=np.float64).reshape(3))
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normal)):
        raise ValueError("invalid refractive projection input")
    if float(np.linalg.norm(normal)) < 1e-9:
        raise ValueError("interface normal is zero")
    d = float(d_air)
    if d <= 1e-6:
        raise ValueError("camera is not on the air side of the interface")

    h = points @ normal
    water_depth = h - d
    if np.any(water_depth <= 1e-6):
        raise ValueError("underwater point is not behind the interface")

    lateral = points - h[:, None] * normal.reshape(1, 3)
    b = np.linalg.norm(lateral, axis=1)
    q = solve_interface_lateral_newton(
        b,
        d,
        water_depth,
        n_air,
        n_water,
        max_iterations=ray_max_iterations,
        tolerance=ray_tolerance,
    )

    q_points = np.tile((d * normal).reshape(1, 3), (points.shape[0], 1))
    active = b > 1e-12
    if np.any(active):
        q_points[active] += lateral[active] * (q[active] / b[active]).reshape(-1, 1)
    if np.any(q_points[:, 2] <= 1e-9):
        raise ValueError("interface refraction point projects behind the camera")

    projected, _jacobian = cv2.projectPoints(
        q_points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    return projected.reshape(-1, 2)


def refractive_reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_hat: np.ndarray,
    d_air: float,
    n_air: float,
    n_water: float,
    ray_max_iterations: int = 10,
    ray_tolerance: float = 1e-11,
) -> float:
    points_cam = transform_object_points(object_points, rvec, tvec_m)
    projected = project_refractive(
        points_cam,
        n_hat,
        d_air,
        n_air,
        n_water,
        intrinsics,
        ray_max_iterations=ray_max_iterations,
        ray_tolerance=ray_tolerance,
    )
    residuals = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def refractive_residual_vector(
    params: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_hat: np.ndarray,
    d_air: float,
    n_air: float,
    n_water: float,
) -> np.ndarray:
    try:
        points_cam = transform_object_points(object_points, params[:3], params[3:])
        projected = project_refractive(points_cam, n_hat, d_air, n_air, n_water, intrinsics)
        return (projected - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)).reshape(-1)
    except Exception:
        return np.full(8, 1e6, dtype=np.float64)


def refine_refractive_pose_lm(
    object_points: np.ndarray,
    image_points: np.ndarray,
    initial_rvec: np.ndarray,
    initial_tvec_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_hat: np.ndarray,
    d_air: float,
    n_air: float,
    n_water: float,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    params = np.concatenate(
        (
            np.asarray(initial_rvec, dtype=np.float64).reshape(3),
            np.asarray(initial_tvec_m, dtype=np.float64).reshape(3),
        )
    )
    damping = 1e-3
    residual = refractive_residual_vector(
        params,
        object_points,
        image_points,
        intrinsics,
        n_hat,
        d_air,
        n_air,
        n_water,
    )
    best_cost = float(residual @ residual)
    if not np.isfinite(best_cost) or best_cost > 1e11:
        raise ValueError("initial refractive pose is geometrically invalid")

    converged = False
    iterations = 0
    eps = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5], dtype=np.float64)
    for iterations in range(1, max_iterations + 1):
        jacobian = np.empty((residual.size, params.size), dtype=np.float64)
        for col in range(params.size):
            step = np.zeros_like(params)
            step[col] = eps[col]
            plus = refractive_residual_vector(
                params + step,
                object_points,
                image_points,
                intrinsics,
                n_hat,
                d_air,
                n_air,
                n_water,
            )
            minus = refractive_residual_vector(
                params - step,
                object_points,
                image_points,
                intrinsics,
                n_hat,
                d_air,
                n_air,
                n_water,
            )
            jacobian[:, col] = (plus - minus) / (2.0 * eps[col])

        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diag = np.maximum(np.diag(normal_matrix), 1e-9)
        accepted = False
        for _attempt in range(8):
            try:
                step = -np.linalg.solve(
                    normal_matrix + damping * np.diag(diag),
                    gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            if not np.all(np.isfinite(step)):
                damping *= 10.0
                continue
            candidate = params + step
            candidate_residual = refractive_residual_vector(
                candidate,
                object_points,
                image_points,
                intrinsics,
                n_hat,
                d_air,
                n_air,
                n_water,
            )
            candidate_cost = float(candidate_residual @ candidate_residual)
            if np.isfinite(candidate_cost) and candidate_cost < best_cost:
                params = candidate
                residual = candidate_residual
                if best_cost - candidate_cost < 1e-10 or float(np.linalg.norm(step)) < 1e-9:
                    converged = True
                best_cost = candidate_cost
                damping = max(damping / 3.0, 1e-9)
                accepted = True
                break
            damping *= 10.0
        if converged:
            break
        if not accepted:
            break

    rms = float(np.sqrt(best_cost / max(1, residual.size // 2)))
    # Accept the result as "converged" if either the pose-tol convergence flag
    # is set or the reprojection RMS is below ~1.5 px. The earlier 0.25 px
    # threshold rejected partially-converged-but-physically-reasonable solves
    # in handheld pool runs, falling back to the raw in-air PnP — which gives
    # the APPARENT underwater position (real / n_water ~ 0.75 x real) and
    # produces a systematic distance underestimation. 1.5 px is still subpixel
    # and well within underwater AprilTag detection noise.
    return params[:3].copy(), params[3:].copy(), rms, converged or rms < 1.5, iterations


def undistorted_rays_from_pixels(
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    normalized = cv2.undistortPoints(
        np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    rays = np.column_stack((normalized[:, 0], normalized[:, 1], np.ones(normalized.shape[0])))
    return rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)


def refract_air_rays_to_water(
    air_dirs: np.ndarray,
    n_hat: np.ndarray,
    n_air: float,
    n_water: float,
) -> np.ndarray:
    normal = normalize_vector(np.asarray(n_hat, dtype=np.float64).reshape(3))
    dirs = np.asarray(air_dirs, dtype=np.float64).reshape(-1, 3)
    cos_air = dirs @ normal
    if np.any(cos_air <= 1e-9):
        raise ValueError("observed air ray does not intersect the front side of the interface")

    tangent = dirs - cos_air[:, None] * normal.reshape(1, 3)
    sin_air = np.linalg.norm(tangent, axis=1)
    sin_water = (float(n_air) / float(n_water)) * sin_air
    if np.any(sin_water >= 1.0):
        raise ValueError("total internal refraction geometry is invalid")

    water_dirs = np.empty_like(dirs)
    active = sin_air > 1e-12
    if np.any(active):
        tangent_unit = tangent[active] / sin_air[active].reshape(-1, 1)
        cos_water = np.sqrt(np.maximum(1.0 - sin_water[active] * sin_water[active], 0.0))
        water_dirs[active] = (
            cos_water.reshape(-1, 1) * normal.reshape(1, 3)
            + sin_water[active].reshape(-1, 1) * tangent_unit
        )
    if np.any(~active):
        water_dirs[~active] = normal.reshape(1, 3)
    return water_dirs / np.maximum(np.linalg.norm(water_dirs, axis=1, keepdims=True), 1e-12)


def corrected_pixels_for_refractive_fixed_point_batch(
    image_points_batch: np.ndarray,
    points_cam_batch: np.ndarray,
    n_hat_batch: np.ndarray,
    d_air_batch: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_air: float,
    n_water: float,
) -> np.ndarray:
    image_points = np.asarray(image_points_batch, dtype=np.float64)
    points_cam = np.asarray(points_cam_batch, dtype=np.float64)
    normals = np.asarray(n_hat_batch, dtype=np.float64)
    d_air = np.asarray(d_air_batch, dtype=np.float64).reshape(-1)
    tag_count, corner_count = image_points.shape[:2]

    flat_rays = undistorted_rays_from_pixels(image_points.reshape(-1, 2), intrinsics)
    rays = flat_rays.reshape(tag_count, corner_count, 3)
    corrected_points = np.empty_like(points_cam)
    for tag_index in range(tag_count):
        normal = normalize_vector(normals[tag_index])
        d = float(d_air[tag_index])
        if d <= 1e-6:
            raise ValueError("camera is not on the air side of the interface")
        air_dirs = rays[tag_index]
        cos_air = air_dirs @ normal
        if np.any(cos_air <= 1e-9):
            raise ValueError("observed ray misses the front side of the interface")
        interface_points = air_dirs * (d / cos_air).reshape(-1, 1)
        water_dirs = refract_air_rays_to_water(air_dirs, normal, n_air, n_water)
        water_depth = points_cam[tag_index] @ normal - d
        if np.any(water_depth <= 1e-6):
            raise ValueError("estimated point is not behind the interface")
        water_cos = water_dirs @ normal
        if np.any(water_cos <= 1e-9):
            raise ValueError("refracted water ray is parallel to the interface")
        corrected_points[tag_index] = (
            interface_points + water_dirs * (water_depth / water_cos).reshape(-1, 1)
        )

    projected, _jacobian = cv2.projectPoints(
        corrected_points.reshape(-1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    return projected.reshape(tag_count, corner_count, 2)


def solve_refractive_pose_fixed_point(
    object_points: np.ndarray,
    image_points: np.ndarray,
    initial_rvec: np.ndarray,
    initial_tvec_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_hat: np.ndarray,
    d_air: float,
    n_air: float,
    n_water: float,
    max_iterations: int,
    convergence_tol_m: float,
    convergence_tol_deg: float,
    ray_max_iterations: int,
    ray_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    rvec = np.asarray(initial_rvec, dtype=np.float64).reshape(3).copy()
    tvec = np.asarray(initial_tvec_m, dtype=np.float64).reshape(3).copy()
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        points_cam = transform_object_points(object_points, rvec, tvec).reshape(1, -1, 3)
        corrected_pixels = corrected_pixels_for_refractive_fixed_point_batch(
            np.asarray(image_points, dtype=np.float64).reshape(1, -1, 2),
            points_cam,
            np.asarray(n_hat, dtype=np.float64).reshape(1, 3),
            np.asarray([d_air], dtype=np.float64),
            intrinsics,
            n_air,
            n_water,
        )[0]
        success, new_rvec, new_tvec = cv2.solvePnP(
            object_points,
            corrected_pixels.astype(np.float32),
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            flags=method,
        )
        if not success:
            raise ValueError("fixed-point solvePnP failed")
        new_rvec = np.asarray(new_rvec, dtype=np.float64).reshape(3)
        new_tvec = np.asarray(new_tvec, dtype=np.float64).reshape(3)
        trans_delta = float(np.linalg.norm(new_tvec - tvec))
        rot_delta = rotation_error_deg(new_rvec, rvec)
        rvec, tvec = new_rvec, new_tvec
        if trans_delta <= convergence_tol_m and rot_delta <= convergence_tol_deg:
            converged = True
            break

    rms = refractive_reprojection_error_px(
        object_points,
        image_points,
        rvec,
        tvec,
        intrinsics,
        n_hat,
        d_air,
        n_air,
        n_water,
        ray_max_iterations=ray_max_iterations,
        ray_tolerance=ray_tolerance,
    )
    # See ``refine_refractive_pose_lm``: 1.5 px is still subpixel for AprilTag
    # detection noise and avoids the systematic-underestimation fallback path.
    return rvec, tvec, rms, converged or rms < 1.5, iterations


def solve_refractive_pose_fixed_point_batch(
    object_points: np.ndarray,
    image_points_batch: np.ndarray,
    initial_rvecs: np.ndarray,
    initial_tvecs_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    n_hat_batch: np.ndarray,
    d_air_batch: np.ndarray,
    n_air: float,
    n_water: float,
    max_iterations: int,
    convergence_tol_m: float,
    convergence_tol_deg: float,
    ray_max_iterations: int,
    ray_tolerance: float,
) -> list[tuple[np.ndarray, np.ndarray, float, bool, int] | None]:
    tag_count = int(image_points_batch.shape[0])
    rvecs = np.asarray(initial_rvecs, dtype=np.float64).reshape(tag_count, 3).copy()
    tvecs = np.asarray(initial_tvecs_m, dtype=np.float64).reshape(tag_count, 3).copy()
    active = np.ones(tag_count, dtype=bool)
    converged = np.zeros(tag_count, dtype=bool)
    iterations_used = np.zeros(tag_count, dtype=np.int32)
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )

    for iteration in range(1, max_iterations + 1):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        points_cam_batch = np.asarray(
            [transform_object_points(object_points, rvecs[index], tvecs[index]) for index in active_indices],
            dtype=np.float64,
        )
        try:
            corrected_batch = corrected_pixels_for_refractive_fixed_point_batch(
                image_points_batch[active_indices],
                points_cam_batch,
                n_hat_batch[active_indices],
                d_air_batch[active_indices],
                intrinsics,
                n_air,
                n_water,
            )
        except Exception:
            break

        for local_index, tag_index in enumerate(active_indices):
            success, new_rvec, new_tvec = cv2.solvePnP(
                object_points,
                corrected_batch[local_index].astype(np.float32),
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
                flags=method,
            )
            if not success:
                active[tag_index] = False
                continue
            new_rvec = np.asarray(new_rvec, dtype=np.float64).reshape(3)
            new_tvec = np.asarray(new_tvec, dtype=np.float64).reshape(3)
            trans_delta = float(np.linalg.norm(new_tvec - tvecs[tag_index]))
            rot_delta = rotation_error_deg(new_rvec, rvecs[tag_index])
            rvecs[tag_index] = new_rvec
            tvecs[tag_index] = new_tvec
            iterations_used[tag_index] = iteration
            if trans_delta <= convergence_tol_m and rot_delta <= convergence_tol_deg:
                converged[tag_index] = True
                active[tag_index] = False

    results: list[tuple[np.ndarray, np.ndarray, float, bool, int] | None] = []
    for tag_index in range(tag_count):
        if iterations_used[tag_index] == 0:
            results.append(None)
            continue
        try:
            rms = refractive_reprojection_error_px(
                object_points,
                image_points_batch[tag_index],
                rvecs[tag_index],
                tvecs[tag_index],
                intrinsics,
                n_hat_batch[tag_index],
                float(d_air_batch[tag_index]),
                n_air,
                n_water,
                ray_max_iterations=ray_max_iterations,
                ray_tolerance=ray_tolerance,
            )
        except Exception:
            results.append(None)
            continue
        results.append(
            (
                rvecs[tag_index].copy(),
                tvecs[tag_index].copy(),
                float(rms),
                # Same rationale as the non-batch ``solve_refractive_pose_fixed_point``:
                # 1.5 px is still subpixel and avoids the raw-fallback path that
                # otherwise leaves underwater tags at apparent (n_water-scaled) positions.
                bool(converged[tag_index] or rms < 1.5),
                int(iterations_used[tag_index]),
            )
        )
    return results


def water_up_axis_world(water_cfg: dict[str, object]) -> np.ndarray:
    return normalize_vector(np.asarray(water_cfg["up_axis_world"], dtype=np.float64).reshape(3))


def interface_plane_from_camera_pose(
    world_T_camera: Pose3,
    water_cfg: dict[str, object],
    gravity_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    up_world = water_up_axis_world(water_cfg)
    air_to_water_world = -up_world
    surface_point_world = up_world * float(water_cfg["surface_height_m"])
    camera_position_world = pose_translation(world_T_camera)
    d_air = float(air_to_water_world @ (surface_point_world - camera_position_world))
    if gravity_camera is not None:
        # The refractive solver's n_hat is defined to point FROM AIR INTO WATER
        # (see ``project_refractive``). For a horizontal water surface, that
        # direction in the camera frame is exactly the gravity direction
        # ``gravity_camera`` (down). The original derivation gives the same:
        #   n_camera = R_camera_world @ air_to_water_world
        #            = R_camera_world @ gravity_world = gravity_camera.
        # Replacing it directly with IMU gravity removes the dependence on the
        # anchor-tilted world rotation. The air gap d_air is still derived
        # from the tracked camera pose and the configured water surface.
        n_camera = np.asarray(gravity_camera, dtype=np.float64).reshape(3)
    else:
        rotation_world_camera = np.asarray(world_T_camera.rotation().matrix(), dtype=np.float64)
        n_camera = rotation_world_camera.T @ air_to_water_world
    return normalize_vector(n_camera), d_air


def fallback_near_nadir_interface_from_raw_pose(
    raw_tvec_m: np.ndarray,
    water_cfg: dict[str, object],
    args: argparse.Namespace,
    gravity_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    raw_tvec = np.asarray(raw_tvec_m, dtype=np.float64).reshape(3)
    if gravity_camera is not None:
        # Bootstrap path: when the camera pose is not yet tracked, prefer IMU
        # gravity over the near-nadir guess so the bootstrap interface normal
        # is still true horizontal in the camera frame. Same convention as
        # ``interface_plane_from_camera_pose``: n_hat points FROM AIR INTO
        # WATER, which is the gravity direction in the camera frame.
        n_camera = normalize_vector(np.asarray(gravity_camera, dtype=np.float64).reshape(3))
    else:
        n_camera = normalize_vector(raw_tvec)
    apparent_water_m = (
        float(water_cfg["surface_height_m"])
        * float(water_cfg["n_air"])
        / float(water_cfg["n_water"])
    )
    d_air = float(np.linalg.norm(raw_tvec) - apparent_water_m)
    if not np.isfinite(d_air) or d_air <= 0.01:
        d_air = max(float(args.surface_distance_m), 0.01)
    return n_camera, d_air


def refractive_pose_from_raw_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    raw_tvec_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    args: argparse.Namespace,
    context: RefractiveContext,
    raw_camera_T_tag: Pose3,
    tag_id: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    world_T_camera = None
    if context.backend is not None and hasattr(context.backend, "camera_pose_hint_for_measurement"):
        world_T_camera = context.backend.camera_pose_hint_for_measurement(tag_id, raw_camera_T_tag)

    used_fallback = False
    gravity_camera = context.imu_gravity_camera
    if world_T_camera is not None:
        n_camera, d_air = interface_plane_from_camera_pose(
            world_T_camera,
            context.water_cfg,
            gravity_camera=gravity_camera,
        )
        if d_air <= 1e-6:
            used_fallback = True
            n_camera, d_air = fallback_near_nadir_interface_from_raw_pose(
                raw_tvec_m,
                context.water_cfg,
                args,
                gravity_camera=gravity_camera,
            )
    else:
        used_fallback = True
        n_camera, d_air = fallback_near_nadir_interface_from_raw_pose(
            raw_tvec_m,
            context.water_cfg,
            args,
            gravity_camera=gravity_camera,
        )

    if used_fallback and not context.warned_bootstrap_fallback:
        print(
            "Refractive PnP: using near-nadir bootstrap interface for initial "
            "unlocalized frame; later frames derive air gap from tracked pose.",
            flush=True,
        )
        context.warned_bootstrap_fallback = True

    start_s = time.monotonic()
    try:
        initial = context.pose_cache.get(tag_id)
        if initial is not None:
            initial_rvec, initial_tvec = initial
        else:
            initial_rvec, initial_tvec = rvec, raw_tvec_m
        refined_rvec, refined_tvec, rms, converged, _iterations = solve_refractive_pose_fixed_point(
            object_points,
            image_points,
            initial_rvec,
            initial_tvec,
            intrinsics,
            n_camera,
            d_air,
            float(context.water_cfg["n_air"]),
            float(context.water_cfg["n_water"]),
            int(args.refractive_max_iterations),
            float(args.refractive_convergence_tol_m),
            float(args.refractive_convergence_tol_deg),
            int(args.refractive_ray_max_iterations),
            float(args.refractive_ray_tol),
        )
        # See the batched path for the rationale: prefer a partially-converged
        # but finite/low-rms refractive result over the raw in-air PnP, which
        # produces a systematic ~25% underestimation of underwater distance.
        if not converged and (not np.isfinite(rms) or rms > 5.0):
            raise ValueError(f"refractive solve diverged, rms={rms:.3f}px")
        context.pose_cache[tag_id] = (refined_rvec.copy(), refined_tvec.copy())
        context.solve_count += 1
        context.total_s += time.monotonic() - start_s
        if used_fallback or not converged:
            context.fallback_count += 1
        return refined_rvec, refined_tvec, rms, True
    except Exception as exc:
        context.solve_count += 1
        context.fallback_count += 1
        context.total_s += time.monotonic() - start_s
        print(
            f"WARNING: refractive PnP failed for tag {tag_id}; using in-air pose ({exc})",
            flush=True,
        )
        context.pose_cache.pop(tag_id, None)
        return (
            np.asarray(rvec, dtype=np.float64).reshape(3),
            np.asarray(raw_tvec_m, dtype=np.float64).reshape(3),
            reprojection_error_px(object_points, image_points, rvec, raw_tvec_m, intrinsics),
            False,
        )


def corrected_underwater_tvec(raw_tvec_m: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if args.water_correction_mode in {"none", "refractive"}:
        scale = 1.0
    else:
        scale = float(args.water_scale)
    return np.asarray(raw_tvec_m, dtype=np.float64).reshape(3) * scale, scale


def estimate_raw_tag_candidate(
    detection,
    intrinsics: CameraIntrinsics,
    object_points: np.ndarray,
    image_shape: tuple[int, int],
) -> RawTagCandidate | None:
    image_points = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=method,
    )
    if not success:
        return None

    raw_tvec_m = np.asarray(tvec, dtype=np.float64).reshape(3)
    raw_rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    raw_camera_T_tag = pose3_from_rvec_tvec(raw_rvec, raw_tvec_m)
    reproj_error = reprojection_error_px(object_points, image_points, raw_rvec, raw_tvec_m, intrinsics)
    center = np.asarray(detection.center, dtype=np.float64).reshape(2)
    return RawTagCandidate(
        tag_id=int(detection.tag_id),
        corners=image_points,
        center=(int(round(center[0])), int(round(center[1]))),
        raw_rvec=raw_rvec,
        raw_tvec_m=raw_tvec_m,
        raw_camera_T_tag=raw_camera_T_tag,
        decision_margin=float(detection.decision_margin),
        hamming=int(detection.hamming),
        tag_area_px=quadrilateral_area_px(image_points),
        off_nadir_deg=off_nadir_angle_deg(raw_tvec_m),
        image_eccentricity=image_eccentricity(center, intrinsics, image_shape),
        tag_tilt_deg=tag_tilt_deg_from_rvec(raw_rvec),
        reprojection_error_px=reproj_error,
    )


def observation_from_candidate(
    candidate: RawTagCandidate,
    corrected_rvec: np.ndarray,
    corrected_tvec_m: np.ndarray,
    reprojection_error: float,
    applied_scale: float,
) -> TagObservation:
    corrected_rvec = np.asarray(corrected_rvec, dtype=np.float64).reshape(3)
    corrected_tvec_m = np.asarray(corrected_tvec_m, dtype=np.float64).reshape(3)
    return TagObservation(
        tag_id=candidate.tag_id,
        corners=candidate.corners,
        center=candidate.center,
        rvec=corrected_rvec,
        raw_tvec_m=candidate.raw_tvec_m,
        scaled_tvec_m=corrected_tvec_m,
        camera_T_tag=pose3_from_rvec_tvec(corrected_rvec, corrected_tvec_m),
        decision_margin=candidate.decision_margin,
        hamming=candidate.hamming,
        tag_area_px=candidate.tag_area_px,
        off_nadir_deg=candidate.off_nadir_deg,
        image_eccentricity=candidate.image_eccentricity,
        tag_tilt_deg=tag_tilt_deg_from_rvec(corrected_rvec),
        reprojection_error_px=float(reprojection_error),
        applied_scale=applied_scale,
    )


def estimate_tag_observation(
    detection,
    intrinsics: CameraIntrinsics,
    object_points: np.ndarray,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    refractive_context: RefractiveContext | None = None,
) -> TagObservation | None:
    candidate = estimate_raw_tag_candidate(detection, intrinsics, object_points, image_shape)
    if candidate is None:
        return None

    if args.water_correction_mode == "refractive" and refractive_context is not None:
        corrected_rvec, corrected_tvec_m, reproj_error, used_refractive = refractive_pose_from_raw_pnp(
            object_points,
            candidate.corners,
            candidate.raw_rvec,
            candidate.raw_tvec_m,
            intrinsics,
            args,
            refractive_context,
            candidate.raw_camera_T_tag,
            candidate.tag_id,
        )
        applied_scale = 1.0
    else:
        corrected_rvec = candidate.raw_rvec
        corrected_tvec_m, applied_scale = corrected_underwater_tvec(candidate.raw_tvec_m, args)
        reproj_error = candidate.reprojection_error_px

    return observation_from_candidate(
        candidate,
        corrected_rvec,
        corrected_tvec_m,
        reproj_error,
        applied_scale,
    )


def refractive_observations_from_candidates_batch(
    candidates: list[RawTagCandidate],
    intrinsics: CameraIntrinsics,
    object_points: np.ndarray,
    args: argparse.Namespace,
    context: RefractiveContext,
) -> list[TagObservation]:
    if not candidates:
        return []

    start_s = time.monotonic()
    n_air = float(context.water_cfg["n_air"])
    n_water = float(context.water_cfg["n_water"])
    image_points_batch = np.asarray([candidate.corners for candidate in candidates], dtype=np.float64)
    raw_rvecs = np.asarray([candidate.raw_rvec for candidate in candidates], dtype=np.float64)
    raw_tvecs = np.asarray([candidate.raw_tvec_m for candidate in candidates], dtype=np.float64)
    initial_rvecs = raw_rvecs.copy()
    initial_tvecs = raw_tvecs.copy()
    n_hat_batch = np.empty((len(candidates), 3), dtype=np.float64)
    d_air_batch = np.empty(len(candidates), dtype=np.float64)
    used_fallback = np.zeros(len(candidates), dtype=bool)

    current_ids = {candidate.tag_id for candidate in candidates}
    for stale_id in set(context.pose_cache) - current_ids:
        context.pose_cache.pop(stale_id, None)

    for index, candidate in enumerate(candidates):
        cached_pose = context.pose_cache.get(candidate.tag_id)
        if cached_pose is not None:
            initial_rvecs[index], initial_tvecs[index] = cached_pose

        world_T_camera = None
        if context.backend is not None and hasattr(context.backend, "camera_pose_hint_for_measurement"):
            world_T_camera = context.backend.camera_pose_hint_for_measurement(
                candidate.tag_id,
                candidate.raw_camera_T_tag,
            )
        gravity_camera = context.imu_gravity_camera
        if world_T_camera is not None:
            n_camera, d_air = interface_plane_from_camera_pose(
                world_T_camera,
                context.water_cfg,
                gravity_camera=gravity_camera,
            )
            if d_air <= 1e-6:
                used_fallback[index] = True
                n_camera, d_air = fallback_near_nadir_interface_from_raw_pose(
                    candidate.raw_tvec_m,
                    context.water_cfg,
                    args,
                    gravity_camera=gravity_camera,
                )
        else:
            used_fallback[index] = True
            n_camera, d_air = fallback_near_nadir_interface_from_raw_pose(
                candidate.raw_tvec_m,
                context.water_cfg,
                args,
                gravity_camera=gravity_camera,
            )
        n_hat_batch[index] = n_camera
        d_air_batch[index] = d_air

    if bool(np.any(used_fallback)) and not context.warned_bootstrap_fallback:
        print(
            "Refractive PnP: using near-nadir bootstrap interface for initial "
            "unlocalized frame; later frames derive air gap from tracked pose.",
            flush=True,
        )
        context.warned_bootstrap_fallback = True

    results = solve_refractive_pose_fixed_point_batch(
        object_points,
        image_points_batch,
        initial_rvecs,
        initial_tvecs,
        intrinsics,
        n_hat_batch,
        d_air_batch,
        n_air,
        n_water,
        int(args.refractive_max_iterations),
        float(args.refractive_convergence_tol_m),
        float(args.refractive_convergence_tol_deg),
        int(args.refractive_ray_max_iterations),
        float(args.refractive_ray_tol),
    )

    # rms above this threshold (in pixels) means the refractive solve
    # diverged or landed on a clearly wrong geometry; only then is the raw
    # in-air PnP a safer choice than the refractive result. For anything
    # below it we PREFER the (possibly partially-converged) refractive
    # output, because the raw in-air pose is the APPARENT underwater
    # position (off by ~n_water = 1.333x in distance), which produces a
    # systematic ~25% underestimation of tag distance from the camera.
    PARTIAL_REFRACTIVE_RMS_MAX_PX = 5.0

    observations: list[TagObservation] = []
    for index, (candidate, result) in enumerate(zip(candidates, results)):
        if result is None:
            context.fallback_count += 1
            context.pose_cache.pop(candidate.tag_id, None)
            observations.append(
                observation_from_candidate(
                    candidate,
                    candidate.raw_rvec,
                    candidate.raw_tvec_m,
                    candidate.reprojection_error_px,
                    1.0,
                )
            )
            continue
        rvec, tvec, rms, converged, _iterations = result
        if not converged and (not np.isfinite(rms) or rms > PARTIAL_REFRACTIVE_RMS_MAX_PX):
            # Real failure: rms is garbage, refractive output unusable.
            context.fallback_count += 1
            context.pose_cache.pop(candidate.tag_id, None)
            observations.append(
                observation_from_candidate(
                    candidate,
                    candidate.raw_rvec,
                    candidate.raw_tvec_m,
                    candidate.reprojection_error_px,
                    1.0,
                )
            )
            continue
        if not converged:
            # Partial convergence: solver ran and produced a finite, low-rms
            # pose; this is still much closer to the true underwater pose
            # than the raw in-air PnP, so use it instead of falling back.
            context.fallback_count += 1
        context.pose_cache[candidate.tag_id] = (rvec.copy(), tvec.copy())
        if used_fallback[index]:
            context.fallback_count += 1
        observations.append(observation_from_candidate(candidate, rvec, tvec, rms, 1.0))

    elapsed_s = time.monotonic() - start_s
    context.solve_count += len(candidates)
    context.total_s += elapsed_s
    context.frame_count += 1
    context.frame_total_s += elapsed_s
    context.last_frame_s = elapsed_s
    context.last_frame_tag_count = len(candidates)
    return observations


def detect_observations(
    gray: np.ndarray,
    detector: Detector,
    intrinsics: CameraIntrinsics,
    object_points: np.ndarray,
    args: argparse.Namespace,
    refractive_context: RefractiveContext | None = None,
) -> list[TagObservation]:
    detections = detector.detect(gray, estimate_tag_pose=False)
    observations: list[TagObservation] = []
    refractive_candidates: list[RawTagCandidate] = []
    image_shape = gray.shape[:2]
    for detection in detections:
        tag_id = int(detection.tag_id)
        if args.max_tag_id >= 0 and tag_id > args.max_tag_id:
            continue
        if int(detection.hamming) > args.max_hamming:
            continue
        if float(detection.decision_margin) < args.min_decision_margin:
            continue
        corners = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
        if quadrilateral_area_px(corners) < args.min_tag_area_px:
            continue
        if args.water_correction_mode == "refractive" and refractive_context is not None:
            candidate = estimate_raw_tag_candidate(detection, intrinsics, object_points, image_shape)
            if candidate is not None:
                refractive_candidates.append(candidate)
            continue
        observation = estimate_tag_observation(
            detection,
            intrinsics,
            object_points,
            image_shape,
            args,
            refractive_context,
        )
        if observation is None:
            continue
        if observation.reprojection_error_px > args.max_reprojection_error_px:
            continue
        if observation.off_nadir_deg > args.max_off_nadir_deg:
            continue
        if observation.image_eccentricity > args.max_image_eccentricity:
            continue
        if observation.tag_tilt_deg > args.max_tag_tilt_deg:
            continue
        observations.append(observation)

    if args.water_correction_mode == "refractive" and refractive_context is not None:
        for observation in refractive_observations_from_candidates_batch(
            refractive_candidates,
            intrinsics,
            object_points,
            args,
            refractive_context,
        ):
            if observation.reprojection_error_px > args.max_reprojection_error_px:
                continue
            if observation.off_nadir_deg > args.max_off_nadir_deg:
                continue
            if observation.image_eccentricity > args.max_image_eccentricity:
                continue
            if observation.tag_tilt_deg > args.max_tag_tilt_deg:
                continue
            observations.append(observation)
    return sorted(observations, key=lambda obs: obs.tag_id)


def make_pose_noise(rot_sigma_rad: float, trans_sigma_m: float) -> gtsam.noiseModel.Base:
    # For Pose3, GTSAM's 6D tangent vector is ordered as rotation then
    # translation: [rx, ry, rz, tx, ty, tz].
    sigmas = np.array([rot_sigma_rad] * 3 + [trans_sigma_m] * 3, dtype=np.float64)
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def make_robust_noise(
    base_noise: gtsam.noiseModel.Base,
    kernel: str,
    threshold: float,
) -> gtsam.noiseModel.Base:
    if kernel == "none":
        return base_noise
    estimators = gtsam.noiseModel.mEstimator
    if kernel == "huber":
        estimator = estimators.Huber.Create(threshold)
    elif kernel == "cauchy":
        estimator = estimators.Cauchy.Create(threshold)
    elif kernel == "tukey":
        estimator = estimators.Tukey.Create(threshold)
    else:
        raise ValueError(f"Unsupported robust kernel: {kernel}")
    return gtsam.noiseModel.Robust.Create(estimator, base_noise)


def make_floor_prior_noise(z_sigma_m: float, normal_sigma_rad: float | None = None) -> gtsam.noiseModel.Base:
    """
    Weak floor prior for Pose3 tags.

    When normal_sigma_rad is provided, roll/pitch are softly constrained so a
    tag's local z-axis aligns with the fitted floor plane normal. Yaw and X/Y
    translation remain effectively free. Without normal_sigma_rad this behaves
    like the original Z-only prior.
    """
    LARGE = 1e6
    rot_sigma = LARGE if normal_sigma_rad is None else normal_sigma_rad
    sigmas = np.array(
        [
            rot_sigma,
            rot_sigma,
            LARGE,
            LARGE,
            LARGE,
            z_sigma_m,
        ],
        dtype=np.float64,
    )
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def set_isam2_param(params: gtsam.ISAM2Params, setter_name: str, attr_name: str, value) -> None:
    """Set an iSAM2 parameter across GTSAM Python wrapper versions."""
    setter = getattr(params, setter_name, None)
    if callable(setter):
        setter(value)
        return
    if hasattr(params, attr_name):
        setattr(params, attr_name, value)


def pose_translation(pose: Pose3) -> np.ndarray:
    return np.asarray(pose.translation(), dtype=np.float64).reshape(3)


def pose_rpy(pose: Pose3) -> np.ndarray:
    return np.asarray(pose.rotation().rpy(), dtype=np.float64).reshape(3)


def values_has_pose(values: Values, key: int) -> bool:
    try:
        return bool(values.exists(key))
    except AttributeError:
        try:
            values.atPose3(key)
            return True
        except RuntimeError:
            return False


def fit_plane_svd(points: np.ndarray) -> FittedPlane | None:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return None
    centroid = np.median(points, axis=0)
    centered = points - centroid
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = normalize_vector(vh[-1])
    if float(normal[2]) < 0.0:
        normal = -normal
    return FittedPlane(point=centroid, normal=normal)


def robust_fit_plane(points: np.ndarray, outlier_threshold_m: float) -> FittedPlane | None:
    initial = fit_plane_svd(points)
    if initial is None:
        return None
    distances = np.abs((points - initial.point) @ initial.normal)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    threshold = max(outlier_threshold_m, median + 3.0 * 1.4826 * mad)
    inliers = points[distances <= threshold]
    if inliers.shape[0] < 3:
        inliers = points
    return fit_plane_svd(inliers)


def project_point_to_plane(point: np.ndarray, plane: FittedPlane) -> np.ndarray:
    return point - plane.normal * float((point - plane.point) @ plane.normal)


def rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 rotation that maps unit vector ``source`` onto unit vector
    ``target`` via the shortest geodesic. If the two are already parallel,
    return identity; if antiparallel, rotate 180 degrees about any axis
    orthogonal to ``source``. This is used to roll/pitch-align the world frame
    to IMU gravity without injecting yaw: the rotation axis is the cross
    product, which is purely horizontal when both vectors are near-vertical.
    """
    source = normalize_vector(np.asarray(source, dtype=np.float64).reshape(3))
    target = normalize_vector(np.asarray(target, dtype=np.float64).reshape(3))
    cos_theta = float(np.clip(source @ target, -1.0, 1.0))
    if cos_theta > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cos_theta < -1.0 + 1e-12:
        # 180-degree flip. Pick any axis orthogonal to source.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(helper @ source)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize_vector(np.cross(source, helper))
        # Rodrigues with theta = pi: R = 2*outer(axis, axis) - I.
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = np.cross(source, target)
    sin_theta = float(np.linalg.norm(axis))
    axis = axis / sin_theta
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_theta * K + (1.0 - cos_theta) * (K @ K)


def target_rotation_aligned_to_plane(current_pose: Pose3, plane: FittedPlane) -> Rot3:
    current_rotation = np.asarray(current_pose.rotation().matrix(), dtype=np.float64)
    normal = normalize_vector(plane.normal)
    x_axis = current_rotation[:, 0] - normal * float(current_rotation[:, 0] @ normal)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(fallback @ normal)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = fallback - normal * float(fallback @ normal)
    x_axis = normalize_vector(x_axis)
    y_axis = normalize_vector(np.cross(normal, x_axis))
    x_axis = normalize_vector(np.cross(y_axis, normal))
    rotation_matrix = np.column_stack((x_axis, y_axis, normal))
    if np.linalg.det(rotation_matrix) < 0:
        y_axis = -y_axis
        rotation_matrix = np.column_stack((x_axis, y_axis, normal))
    return Rot3(rotation_matrix.astype(np.float64))


class TagSlamBackend:
    """
    Incremental TagSLAM backend.

    State definitions:
        X(i) = world_T_camera_i, the camera pose in the global frame.
        L(j) = world_T_tag_j, the AprilTag pose in the global frame.

    Measurement definitions:
        solvePnP returns camera_T_tag, i.e. a point in tag coordinates maps to
        camera coordinates as p_c = R_c_t * p_t + t_c_t.

    Factor math:
        GTSAM BetweenFactorPose3(a, b, Z) minimizes
            Log( Z^-1 * (a^-1 * b) )
        For a camera observing a tag, a = X(i), b = L(j), and
            X(i)^-1 * L(j) = camera_T_tag
        so Z is exactly the scaled solvePnP pose.
    """

    def __init__(self, args: argparse.Namespace):
        isam_params = gtsam.ISAM2Params()
        set_isam2_param(isam_params, "setRelinearizeThreshold", "relinearizeThreshold", 0.01)
        set_isam2_param(isam_params, "setRelinearizeSkip", "relinearizeSkip", 1)
        self.isam = gtsam.ISAM2(isam_params)
        self.graph = NonlinearFactorGraph()
        self.current_estimate = Values()

        self.prior_noise = make_pose_noise(args.prior_rot_sigma, args.prior_trans_sigma)
        self.tag_noise_base = make_pose_noise(args.tag_rot_sigma, args.tag_trans_sigma)
        self.tag_noise = make_robust_noise(
            self.tag_noise_base,
            args.tag_robust_kernel,
            args.tag_robust_threshold,
        )
        self.odom_noise = make_pose_noise(args.odom_rot_sigma, args.odom_trans_sigma)
        self.floor_prior_enabled = bool(args.floor_prior_enabled)
        self.floor_z_sigma = float(args.floor_z_sigma)
        self.floor_normal_sigma_rad = float(np.radians(args.floor_normal_sigma_deg))
        self.floor_plane_min_tags = int(args.floor_plane_min_tags)
        self.floor_plane_outlier_threshold = float(args.floor_plane_outlier_threshold)
        strict_coplanar_flag = bool(getattr(args, "strict_coplanar", False))
        # In strict-coplanar mode, drop the rotation-alignment component of the
        # prior (pass normal_sigma_rad=None so make_floor_prior_noise uses LARGE
        # rotation sigmas). The intent is "all tags lie on the same plane in
        # z," not "all tags share the exact same orientation"; constraining
        # rotation here only fights tag-detection rotation information without
        # helping the z-flatness the user actually wants.
        self.floor_prior_noise = (
            make_floor_prior_noise(
                self.floor_z_sigma,
                None if strict_coplanar_flag else self.floor_normal_sigma_rad,
            )
            if self.floor_prior_enabled
            else None
        )
        self.tag_init_min_observations = int(args.tag_init_min_observations)
        self.init_min_observations = int(args.init_min_observations)
        self.init_min_decision_margin = float(args.init_min_decision_margin)
        self.init_min_tag_area_px = float(args.init_min_tag_area_px)
        self.init_max_off_nadir_deg = float(args.init_max_off_nadir_deg)
        self.init_max_image_eccentricity = float(args.init_max_image_eccentricity)
        self.init_max_tag_tilt_deg = float(args.init_max_tag_tilt_deg)
        self.anchor_tag_id = int(args.anchor_tag_id)
        self.use_imu_gravity = bool(getattr(args, "use_imu_gravity", False))
        self.gravity_align_world = bool(getattr(args, "gravity_align_world", False))

        self.initialized = False
        self.next_camera_index = 0
        self.last_camera_index: int | None = None
        self.last_camera_key: int | None = None
        self.last_relative_motion = Pose3()
        self.initialized_tag_ids: set[int] = set()
        self.tag_observation_counts: dict[int, int] = {}
        self.floor_prior_tag_ids: set[int] = set()
        self.strict_coplanar = bool(getattr(args, "strict_coplanar", False))
        self.floor_prior_refresh_frames = int(getattr(args, "floor_prior_refresh_frames", 0))
        self.frames_since_floor_refresh: int = 0
        self.pose_history: deque[np.ndarray] = deque(maxlen=int(args.pose_std_window))
        self.factor_count = 0
        # Latest unit gravity (down) direction in the camera frame, updated by
        # main() once per frame before backend.update(). None disables the
        # gravity references (G2/G3/G4) and reverts to anchor-frame behavior.
        self.imu_gravity_camera: np.ndarray | None = None

    def _add_factor(self, new_graph: NonlinearFactorGraph, factor) -> None:
        new_graph.add(factor)
        self.graph.add(factor)
        self.factor_count += 1

    def set_imu_gravity(self, gravity_camera: np.ndarray | None) -> None:
        if gravity_camera is None:
            self.imu_gravity_camera = None
            return
        self.imu_gravity_camera = np.asarray(gravity_camera, dtype=np.float64).reshape(3)

    def _gravity_up_in_world(self) -> np.ndarray | None:
        """
        Express the IMU 'up' direction in the world frame using the latest
        optimized camera pose, and return it in the same sign convention as
        ``fit_plane_svd`` (positive z component).

        Up in camera = -gravity_camera. Up in world = R_world_camera @ up_camera.
        Returns None when IMU gravity or a camera pose is unavailable.

        The sign flip mirrors what ``fit_plane_svd`` does to the SVD normal so
        that downstream consumers (``target_rotation_aligned_to_plane``,
        ``project_point_to_plane``) behave identically except that the normal
        now references true gravity instead of the anchor-tilted floor.
        """
        if self.imu_gravity_camera is None:
            return None
        if self.last_camera_key is None or not values_has_pose(
            self.current_estimate,
            self.last_camera_key,
        ):
            return None
        rotation_world_camera = np.asarray(
            self.current_estimate.atPose3(self.last_camera_key).rotation().matrix(),
            dtype=np.float64,
        )
        up_world = normalize_vector(rotation_world_camera @ (-self.imu_gravity_camera))
        if float(up_world[2]) < 0.0:
            up_world = -up_world
        return up_world

    def _add_floor_prior(
        self,
        new_graph: NonlinearFactorGraph,
        tag_id: int,
        tag_key: int,
        plane: FittedPlane | None = None,
    ) -> None:
        """
        Add a soft plane prior for a tag pose.

        The anchor tag is skipped because it already has the hard world-frame prior at
        identity. Once enough tags exist, a robust floor plane is fit from the
        current tag map; each tag gets one soft prior to its projection on that
        plane and to a co-oriented normal.
        """
        if not self.floor_prior_enabled or self.floor_prior_noise is None:
            return
        if tag_id == self.anchor_tag_id:
            return
        if plane is None or not values_has_pose(self.current_estimate, tag_key):
            return
        current_pose = self.current_estimate.atPose3(tag_key)
        projected = project_point_to_plane(pose_translation(current_pose), plane)
        target_rotation = target_rotation_aligned_to_plane(current_pose, plane)
        floor_pose = Pose3(
            target_rotation,
            Point3(float(projected[0]), float(projected[1]), float(projected[2])),
        )
        self._add_factor(
            new_graph,
            PriorFactorPose3(tag_key, floor_pose, self.floor_prior_noise),
        )
        self.floor_prior_tag_ids.add(tag_id)

    def _record_observation_counts(self, observations: list[TagObservation]) -> None:
        for obs in observations:
            self.tag_observation_counts[obs.tag_id] = (
                self.tag_observation_counts.get(obs.tag_id, 0) + 1
            )

    def _ready_observations(self, observations: list[TagObservation]) -> list[TagObservation]:
        ready: list[TagObservation] = []
        for obs in observations:
            if obs.tag_id in self.initialized_tag_ids:
                ready.append(obs)
            elif self.tag_observation_counts.get(obs.tag_id, 0) >= self.tag_init_min_observations:
                ready.append(obs)
        return ready

    def _anchor_init_rejection_reason(self, anchor_obs: TagObservation) -> str | None:
        count = self.tag_observation_counts.get(self.anchor_tag_id, 0)
        if count < self.init_min_observations:
            return (
                f"anchor tag {self.anchor_tag_id} seen "
                f"{count}/{self.init_min_observations} accepted frames"
            )
        if anchor_obs.decision_margin < self.init_min_decision_margin:
            return (
                f"anchor tag {self.anchor_tag_id} decision margin {anchor_obs.decision_margin:.1f} "
                f"< {self.init_min_decision_margin:.1f}"
            )
        if anchor_obs.tag_area_px < self.init_min_tag_area_px:
            return (
                f"anchor tag {self.anchor_tag_id} area {anchor_obs.tag_area_px:.0f}px "
                f"< {self.init_min_tag_area_px:.0f}px"
            )
        if anchor_obs.off_nadir_deg > self.init_max_off_nadir_deg:
            return (
                f"anchor tag {self.anchor_tag_id} off-nadir {anchor_obs.off_nadir_deg:.1f}deg "
                f"> {self.init_max_off_nadir_deg:.1f}deg"
            )
        if anchor_obs.image_eccentricity > self.init_max_image_eccentricity:
            return (
                f"anchor tag {self.anchor_tag_id} eccentricity {anchor_obs.image_eccentricity:.2f} "
                f"> {self.init_max_image_eccentricity:.2f}"
            )
        if anchor_obs.tag_tilt_deg > self.init_max_tag_tilt_deg:
            return (
                f"anchor tag {self.anchor_tag_id} tilt {anchor_obs.tag_tilt_deg:.1f}deg "
                f"> {self.init_max_tag_tilt_deg:.1f}deg"
            )
        return None

    def _fit_current_floor_plane(self) -> FittedPlane | None:
        tag_points = []
        for tag_id in sorted(self.initialized_tag_ids):
            tag_key = L(tag_id)
            if values_has_pose(self.current_estimate, tag_key):
                tag_points.append(pose_translation(self.current_estimate.atPose3(tag_key)))
        if len(tag_points) < self.floor_plane_min_tags:
            return None
        return robust_fit_plane(np.asarray(tag_points, dtype=np.float64), self.floor_plane_outlier_threshold)

    def _apply_floor_plane_priors(self) -> None:
        if not self.floor_prior_enabled or self.floor_prior_noise is None:
            return
        if self.strict_coplanar:
            # Pin the floor plane to pass through the anchor with normal along
            # the gravity-aligned world's +z axis. This is the user's actual
            # mental model: "every tag's z is the anchor's z, plus a small
            # error." Skipping the SVD fit avoids the plane drifting away from
            # z=0 whenever a few tags happen to land off-plane (which then
            # biases the plane fit, which then biases later tag priors, etc.).
            # The +z sign matches the SVD ``fit_plane_svd`` positive-z
            # convention and the AprilTag local z-axis direction.
            anchor_key = L(self.anchor_tag_id)
            if values_has_pose(self.current_estimate, anchor_key):
                anchor_point = pose_translation(self.current_estimate.atPose3(anchor_key))
            else:
                anchor_point = np.zeros(3, dtype=np.float64)
            plane = FittedPlane(point=anchor_point, normal=np.array([0.0, 0.0, 1.0]))
        else:
            plane = self._fit_current_floor_plane()
            if plane is None:
                return
            if self.use_imu_gravity:
                gravity_up_world = self._gravity_up_in_world()
                if gravity_up_world is not None:
                    # Keep the offset (plane point) from the SVD fit so tag heights
                    # follow the actual reconstructed floor, but reference the
                    # plane's NORMAL to true gravity instead of the SVD-fitted
                    # (anchor-frame) normal. This removes the tilted-ramp coupling
                    # while keeping the prior soft (the existing sigmas apply).
                    plane = FittedPlane(point=plane.point, normal=gravity_up_world)
        # Optional periodic refresh: drop the per-tag "already primed" tracking
        # every N frames so every tag gets a NEW prior pulling it toward the
        # latest (refitted) plane. Without this, a tag's prior is locked to the
        # plane that existed when the tag first entered the graph — fine for
        # well-converged early scenes, but for handheld pool runs the early
        # plane fit can be noisy and tags initialized late end up off-plane.
        refresh_due = (
            self.floor_prior_refresh_frames > 0
            and self.frames_since_floor_refresh >= self.floor_prior_refresh_frames
        )
        if refresh_due:
            self.floor_prior_tag_ids.clear()
            self.frames_since_floor_refresh = 0
        else:
            self.frames_since_floor_refresh += 1
        new_graph = NonlinearFactorGraph()
        before = self.factor_count
        for tag_id in sorted(self.initialized_tag_ids):
            if tag_id in self.floor_prior_tag_ids:
                continue
            self._add_floor_prior(new_graph, tag_id, L(tag_id), plane)
        if self.factor_count == before:
            return
        self.isam.update(new_graph, Values())
        self.current_estimate = self.isam.calculateEstimate()

    def _camera_position_std_cm(self) -> float | None:
        if len(self.pose_history) < 2:
            return None
        history = np.vstack(tuple(self.pose_history))
        return float(np.linalg.norm(np.std(history, axis=0)) * 100.0)

    def _remember_camera_pose(self, pose: Pose3) -> float | None:
        self.pose_history.append(pose_translation(pose))
        return self._camera_position_std_cm()

    def _insert_pose_once(
        self,
        new_values: Values,
        inserted_keys: set[int],
        key: int,
        pose: Pose3,
    ) -> None:
        if key in inserted_keys or values_has_pose(self.current_estimate, key):
            return
        new_values.insert(key, pose)
        inserted_keys.add(key)

    def _initial_camera_from_observations(self, observations: list[TagObservation]) -> Pose3 | None:
        for obs in observations:
            tag_key = L(obs.tag_id)
            if values_has_pose(self.current_estimate, tag_key):
                world_T_tag = self.current_estimate.atPose3(tag_key)
                return world_T_tag.compose(obs.camera_T_tag.inverse())
        if self.last_camera_key is not None and values_has_pose(
            self.current_estimate, self.last_camera_key
        ):
            previous = self.current_estimate.atPose3(self.last_camera_key)
            return previous.compose(self.last_relative_motion)
        return None

    def camera_pose_hint_for_measurement(
        self,
        tag_id: int,
        raw_camera_T_tag: Pose3,
    ) -> Pose3 | None:
        """
        Initial world_T_camera used by refractive PnP to express the fixed water
        surface in the camera frame.

        This mirrors the backend bootstrapping logic: if the observed tag has an
        optimized world pose, compose that landmark with the inverse raw
        camera_T_tag. Otherwise use the constant-velocity camera prediction. For
        the very first anchor observation, use the in-air anchor pose as a
        near-nadir bootstrap; the refractive solver can then replace it before
        the factor enters the graph.
        """
        tag_key = L(tag_id)
        if values_has_pose(self.current_estimate, tag_key):
            world_T_tag = self.current_estimate.atPose3(tag_key)
            return world_T_tag.compose(raw_camera_T_tag.inverse())
        if self.last_camera_key is not None and values_has_pose(
            self.current_estimate,
            self.last_camera_key,
        ):
            previous = self.current_estimate.atPose3(self.last_camera_key)
            return previous.compose(self.last_relative_motion)
        if tag_id == self.anchor_tag_id:
            return raw_camera_T_tag.inverse()
        return None

    def _tag_initial_pose(self, camera_pose: Pose3, observation: TagObservation) -> Pose3:
        return camera_pose.compose(observation.camera_T_tag)

    def _optimized_tag_poses(self) -> dict[int, Pose3]:
        tag_poses: dict[int, Pose3] = {}
        for tag_id in sorted(self.initialized_tag_ids):
            tag_key = L(tag_id)
            if values_has_pose(self.current_estimate, tag_key):
                tag_poses[tag_id] = self.current_estimate.atPose3(tag_key)
        return tag_poses

    def optimized_camera_pose(self, camera_index: int) -> Pose3 | None:
        camera_key = X(camera_index)
        if not values_has_pose(self.current_estimate, camera_key):
            return None
        return self.current_estimate.atPose3(camera_key)

    def optimized_tag_poses(self) -> dict[int, Pose3]:
        return self._optimized_tag_poses()

    def update(self, observations: list[TagObservation]) -> BackendUpdate:
        self._record_observation_counts(observations)
        if not self.initialized:
            return self._initialize_when_anchor_visible(observations)
        return self._update_incremental(self._ready_observations(observations))

    def _initialize_when_anchor_visible(
        self,
        observations: list[TagObservation],
    ) -> BackendUpdate:
        anchor_obs = next((obs for obs in observations if obs.tag_id == self.anchor_tag_id), None)
        if anchor_obs is None:
            return BackendUpdate(
                optimized=False,
                status=f"Waiting for anchor tag {self.anchor_tag_id} to define the world frame",
                camera_pose=None,
                tag_poses={},
                camera_index=None,
                anchor_tag_id=self.anchor_tag_id,
            )
        rejection_reason = self._anchor_init_rejection_reason(anchor_obs)
        if rejection_reason is not None:
            return BackendUpdate(
                optimized=False,
                status=f"Waiting for robust anchor init: {rejection_reason}",
                camera_pose=None,
                tag_poses={},
                camera_index=None,
                anchor_tag_id=self.anchor_tag_id,
            )

        graph_observations = self._ready_observations(observations)
        if all(obs.tag_id != self.anchor_tag_id for obs in graph_observations):
            graph_observations.append(anchor_obs)

        new_graph = NonlinearFactorGraph()
        new_values = Values()
        inserted_keys: set[int] = set()

        anchor_key = L(self.anchor_tag_id)
        # Start with the original anchor-frame initialization. Without IMU
        # gravity, this branch is byte-for-byte the existing behavior:
        #   world_T_anchor = identity, world_T_camera = camera_T_anchor^-1.
        # With --gravity-align-world AND a valid IMU sample, we rotate the
        # world frame by a roll/pitch-only correction so its up axis matches
        # IMU gravity. The anchor stays the position datum (its world
        # translation remains zero); only the orientation tilt is removed.
        world_T_camera = anchor_obs.camera_T_tag.inverse()
        anchor_pose = Pose3()
        gravity_align_applied = False
        if (
            self.use_imu_gravity
            and self.gravity_align_world
            and self.imu_gravity_camera is not None
        ):
            rotation_world_camera = np.asarray(
                world_T_camera.rotation().matrix(),
                dtype=np.float64,
            )
            gravity_in_world_now = rotation_world_camera @ self.imu_gravity_camera
            # Target gravity direction in world: water_cfg default up_axis_world
            # is [0, 0, -1], i.e. world +Z = down. So gravity should be (0,0,1).
            target_gravity_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            R_correction = rotation_aligning_vectors(
                gravity_in_world_now,
                target_gravity_world,
            )
            new_rotation_world_camera = R_correction @ rotation_world_camera
            # Keep the anchor tag at the origin in the gravity-aligned world.
            # world_T_anchor = world_T_camera * camera_T_anchor; with the
            # anchor at world origin, t_world_camera = -R_world_camera * t_camera_anchor.
            t_camera_anchor = np.asarray(
                anchor_obs.camera_T_tag.translation(),
                dtype=np.float64,
            ).reshape(3)
            new_t_world_camera = -(new_rotation_world_camera @ t_camera_anchor)
            world_T_camera = Pose3(
                Rot3(new_rotation_world_camera),
                Point3(*new_t_world_camera.tolist()),
            )
            anchor_pose = world_T_camera.compose(anchor_obs.camera_T_tag)
            gravity_align_applied = True
            tilt_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(
                                normalize_vector(gravity_in_world_now)
                                @ target_gravity_world
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            print(
                f"Gravity-aligned world at init: anchor-frame tilt vs gravity = {tilt_deg:.2f} deg",
                flush=True,
            )

        # Hard world anchor. L(anchor) is pinned at the (possibly gravity-
        # aligned) anchor_pose. With anchor_pose == identity this is the
        # original behavior; with the IMU correction applied, anchor still
        # sits at the world origin but with a tilt-corrected orientation.
        self._add_factor(new_graph, PriorFactorPose3(anchor_key, anchor_pose, self.prior_noise))
        self._insert_pose_once(new_values, inserted_keys, anchor_key, anchor_pose)
        self.initialized_tag_ids.add(self.anchor_tag_id)

        camera_index = self.next_camera_index
        camera_key = X(camera_index)
        # With anchor_pose pinned and camera_T_anchor measured by solvePnP,
        # world_T_camera = anchor_pose * inverse(camera_T_anchor). When G4 is
        # off this collapses to camera_T_anchor^-1 (the original expression).
        if not gravity_align_applied:
            world_T_camera = anchor_pose.compose(anchor_obs.camera_T_tag.inverse())
        self._insert_pose_once(new_values, inserted_keys, camera_key, world_T_camera)

        for obs in graph_observations:
            tag_key = L(obs.tag_id)
            if obs.tag_id not in self.initialized_tag_ids:
                self._insert_pose_once(
                    new_values,
                    inserted_keys,
                    tag_key,
                    self._tag_initial_pose(world_T_camera, obs),
                )
                self.initialized_tag_ids.add(obs.tag_id)
            # Observation factor: predicted relative pose is
            # X(camera_index)^-1 * L(tag_id). The measurement is the scaled
            # camera_T_tag pose from solvePnP.
            self._add_factor(
                new_graph,
                BetweenFactorPose3(camera_key, tag_key, obs.camera_T_tag, self.tag_noise),
            )

        self.isam.update(new_graph, new_values)
        self.current_estimate = self.isam.calculateEstimate()
        self.initialized = True
        self._apply_floor_plane_priors()
        self.last_camera_index = camera_index
        self.last_camera_key = camera_key
        self.next_camera_index += 1

        camera_pose = self.current_estimate.atPose3(camera_key)
        position_std_cm = self._remember_camera_pose(camera_pose)
        return BackendUpdate(
            optimized=True,
            status=(
                f"Initialized with anchor tag {self.anchor_tag_id} "
                f"and {len(graph_observations)} tag factors"
            ),
            camera_pose=camera_pose,
            tag_poses=self._optimized_tag_poses(),
            camera_index=camera_index,
            used_observation_count=len(graph_observations),
            camera_position_std_cm=position_std_cm,
            anchor_tag_id=self.anchor_tag_id,
        )

    def _update_incremental(self, observations: list[TagObservation]) -> BackendUpdate:
        camera_index = self.next_camera_index
        camera_key = X(camera_index)
        predicted_camera_pose = self._initial_camera_from_observations(observations)
        if predicted_camera_pose is None:
            predicted_camera_pose = Pose3()

        new_graph = NonlinearFactorGraph()
        new_values = Values()
        inserted_keys: set[int] = set()
        self._insert_pose_once(new_values, inserted_keys, camera_key, predicted_camera_pose)

        if self.last_camera_key is not None:
            # This is a weak process factor. It encodes a constant-velocity
            # prediction using the previous optimized relative camera motion,
            # not a hard physical truth. Tag factors are intentionally tighter.
            self._add_factor(
                new_graph,
                BetweenFactorPose3(
                    self.last_camera_key,
                    camera_key,
                    self.last_relative_motion,
                    self.odom_noise,
                ),
            )

        for obs in observations:
            tag_key = L(obs.tag_id)
            if obs.tag_id not in self.initialized_tag_ids:
                self._insert_pose_once(
                    new_values,
                    inserted_keys,
                    tag_key,
                    self._tag_initial_pose(predicted_camera_pose, obs),
                )
                self.initialized_tag_ids.add(obs.tag_id)
            # Same camera-to-tag measurement model as initialization, now added
            # incrementally for the newest camera state X(camera_index).
            self._add_factor(
                new_graph,
                BetweenFactorPose3(camera_key, tag_key, obs.camera_T_tag, self.tag_noise),
            )

        self.isam.update(new_graph, new_values)
        self.current_estimate = self.isam.calculateEstimate()
        self._apply_floor_plane_priors()

        if self.last_camera_key is not None:
            previous_pose = self.current_estimate.atPose3(self.last_camera_key)
            current_pose = self.current_estimate.atPose3(camera_key)
            self.last_relative_motion = previous_pose.between(current_pose)
        else:
            current_pose = self.current_estimate.atPose3(camera_key)

        self.last_camera_index = camera_index
        self.last_camera_key = camera_key
        self.next_camera_index += 1
        position_std_cm = self._remember_camera_pose(current_pose)

        return BackendUpdate(
            optimized=True,
            status=(
                f"Optimized X{camera_index}: {len(observations)} tag factors, "
                f"{self.factor_count} total factors"
            ),
            camera_pose=current_pose,
            tag_poses=self._optimized_tag_poses(),
            camera_index=camera_index,
            used_observation_count=len(observations),
            camera_position_std_cm=position_std_cm,
            anchor_tag_id=self.anchor_tag_id,
        )


def make_run_dir(root: Path, suffix: str) -> Path:
    """
    Create a fresh dated run directory under ``root`` named
    ``YYYYMMDD/YYYYMMDD_HHMMSS_<suffix>[_NN]``. Used by both the ZED2
    TagSLAM script and other front-end scripts so the on-disk layout stays
    consistent across cameras.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = timestamp[:8]
    base = root / date / f"{timestamp}_{suffix}"
    output_dir = base
    nth = 1
    while output_dir.exists():
        output_dir = root / date / f"{timestamp}_{suffix}_{nth:02d}"
        nth += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


class TrajectoryRecorder:
    def __init__(
        self,
        output_root: Path,
        image_width: int,
        pool_cfg: dict[str, object],
        tag_size_m: float,
        plot_z_scale: float,
        anchor_tag_id: int,
        suffix: str = "tagslam_trajectory",
        frames_subdir: str = "zed_frames",
    ):
        self.output_root = output_root
        self.image_width = image_width
        self.pool_cfg = pool_cfg
        self.tag_size_m = tag_size_m
        self.plot_z_scale = plot_z_scale
        self.anchor_tag_id = anchor_tag_id
        self.suffix = suffix
        self.frames_subdir = frames_subdir
        self.active = False
        self.start_monotonic_s = 0.0
        self.samples: list[TrajectorySample] = []
        self.output_dir: Path | None = None
        self.frames_dir: Path | None = None

    def start(self, now_s: float) -> None:
        self.active = True
        self.start_monotonic_s = now_s
        self.samples = []
        self.output_dir = self._new_output_dir()
        self.frames_dir = self.output_dir / self.frames_subdir
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        print(
            "Trajectory recording started. "
            f"Frames are being saved in {self.frames_dir}. "
            "Press q to save, or r to stop and save.",
            flush=True,
        )

    def append(
        self,
        update: BackendUpdate,
        observations: list[TagObservation],
        now_s: float,
        annotated_frame: np.ndarray | None = None,
        *,
        timestamp_unix: float | None = None,
        timestamp_monotonic: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Append a recorded sample.

        ``timestamp_unix``/``timestamp_monotonic``/``extra`` are optional and
        flow into the saved CSV opaquely via ``TrajectorySample``. ZED2-style
        callers that don't pass them get byte-identical CSV output to the
        pre-refactor pipeline; the fisheye+gantry pipeline uses them to record
        the shared clock plus the gantry ground-truth columns.
        """
        if not self.active or update.camera_pose is None or update.camera_index is None:
            return
        if self.samples and self.samples[-1].camera_index == update.camera_index:
            return

        image_path = self._save_frame(annotated_frame, update.camera_index)
        self.samples.append(
            TrajectorySample(
                camera_index=update.camera_index,
                elapsed_s=now_s - self.start_monotonic_s,
                detected_tag_ids=tuple(obs.tag_id for obs in observations),
                image_path=image_path,
                timestamp_unix=timestamp_unix,
                timestamp_monotonic=timestamp_monotonic,
                extra=extra,
            )
        )

    def stop_and_save(self, backend: TagSlamBackend) -> Path | None:
        if not self.active:
            return None
        self.active = False

        if not self.samples:
            print("Trajectory recording stopped, but no optimized camera poses were captured.", flush=True)
            return None

        if self.output_dir is None:
            self.output_dir = self._new_output_dir()
        output_dir = self.output_dir

        camera_rows = self._camera_rows(backend)
        tag_rows = self._tag_rows(backend)
        write_camera_trajectory_csv(output_dir / "camera_trajectory.csv", camera_rows)
        write_tag_poses_csv(output_dir / "tag_poses.csv", tag_rows)
        interactive_path = write_interactive_trajectory_html(
            output_dir,
            camera_rows,
            tag_rows,
            self.pool_cfg,
            self.tag_size_m,
            self.plot_z_scale,
            self.anchor_tag_id,
        )
        plot_path = write_trajectory_plot(
            output_dir,
            camera_rows,
            tag_rows,
            self.pool_cfg,
            self.tag_size_m,
            self.plot_z_scale,
            self.anchor_tag_id,
        )

        print(f"Trajectory saved: {output_dir}", flush=True)
        print(f"Interactive trajectory plot: {interactive_path}", flush=True)
        if plot_path is not None:
            print(f"Static trajectory plot: {plot_path}", flush=True)
        return output_dir

    def _new_output_dir(self) -> Path:
        return make_run_dir(self.output_root, self.suffix)

    def _save_frame(self, frame: np.ndarray | None, camera_index: int) -> str | None:
        if frame is None or self.frames_dir is None or self.output_dir is None:
            return None

        saved = frame
        if self.image_width > 0 and frame.shape[1] > self.image_width:
            scale = self.image_width / frame.shape[1]
            saved = cv2.resize(
                frame,
                (self.image_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        frame_name = f"{len(self.samples):06d}_X{camera_index:06d}.jpg"
        frame_path = self.frames_dir / frame_name
        ok = cv2.imwrite(str(frame_path), saved, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            print(f"Warning: could not save ZED trajectory frame {frame_path}", flush=True)
            return None
        return frame_path.relative_to(self.output_dir).as_posix()

    def _camera_rows(self, backend: TagSlamBackend) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for sample in self.samples:
            pose = backend.optimized_camera_pose(sample.camera_index)
            if pose is None:
                continue
            x_m, y_m, z_m = pose_translation(pose)
            roll, pitch, yaw = np.degrees(pose_rpy(pose))
            row: dict[str, object] = {
                "camera_index": sample.camera_index,
                "time_s": sample.elapsed_s,
                "x_m": float(x_m),
                "y_m": float(y_m),
                "z_m": float(z_m),
                "roll_deg": float(roll),
                "pitch_deg": float(pitch),
                "yaw_deg": float(yaw),
                "detected_tags": " ".join(str(tag_id) for tag_id in sample.detected_tag_ids),
                "has_tag_update": bool(sample.detected_tag_ids),
                "image_path": sample.image_path or "",
            }
            # Optional timestamp / extra columns flow through opaquely; the CSV
            # writer extends the header only when ANY row actually populates them
            # so the ZED2 default output stays byte-for-byte identical.
            if sample.timestamp_unix is not None:
                row["timestamp_unix"] = float(sample.timestamp_unix)
            if sample.timestamp_monotonic is not None:
                row["timestamp_monotonic"] = float(sample.timestamp_monotonic)
            if sample.extra:
                for key, value in sample.extra.items():
                    row[key] = float(value)
            rows.append(row)
        return rows

    def _tag_rows(self, backend: TagSlamBackend) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for tag_id, pose in sorted(backend.optimized_tag_poses().items()):
            x_m, y_m, z_m = pose_translation(pose)
            roll, pitch, yaw = np.degrees(pose_rpy(pose))
            rows.append(
                {
                    "tag_id": int(tag_id),
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "z_m": float(z_m),
                    "roll_deg": float(roll),
                    "pitch_deg": float(pitch),
                    "yaw_deg": float(yaw),
                }
            )
        return rows


def write_camera_trajectory_csv(path: Path, rows: list[dict[str, object]]) -> None:
    base_fields = [
        "camera_index",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "detected_tags",
        "has_tag_update",
        "image_path",
    ]
    # Time-sync prefix: only injected when at least one row actually carries a
    # populated unix timestamp. Without this gate the ZED2 script's header
    # would change shape (would break downstream parsers and tests). When
    # injected, ``timestamp_monotonic`` rides along even on rows that left it
    # unset, because the column either applies to the whole CSV or to none.
    inject_timestamps = any(row.get("timestamp_unix") is not None for row in rows)
    # Extra columns: union of keys any row populated, stable-sorted so the
    # column order is deterministic across runs/rows.
    extra_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in base_fields or key in {"timestamp_unix", "timestamp_monotonic"}:
                continue
            if key not in seen:
                seen.add(key)
                extra_keys.append(key)
    extra_keys.sort()

    fields = list(base_fields)
    if inject_timestamps:
        fields = ["timestamp_unix", "timestamp_monotonic"] + fields
    fields = fields + extra_keys

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output_row: dict[str, object] = {}
            for field_name in fields:
                if field_name in row and row[field_name] is not None:
                    output_row[field_name] = row[field_name]
                else:
                    output_row[field_name] = ""
            writer.writerow(output_row)


def write_tag_poses_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["tag_id", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_interactive_trajectory_html(
    output_dir: Path,
    camera_rows: list[dict[str, object]],
    tag_rows: list[dict[str, object]],
    pool_cfg: dict[str, object],
    tag_size_m: float,
    plot_z_scale: float,
    anchor_tag_id: int,
    *,
    gantry_csv: "Path | None" = None,
    fisheye_calib: "object | None" = None,
    rms_summary: dict | None = None,
) -> Path:
    # When gantry data is supplied (experiment pipeline), emit the consolidated
    # 2-tab dashboard (Trajectory + Velocity) to trajectory_interactive.html.
    # When gantry_csv is None (standalone zed2 pipeline), fall through to the
    # original rich single-tab 3D viewer below, unchanged.
    if gantry_csv is not None:
        R = getattr(fisheye_calib, "R_gantry_to_slam", None)
        T_gc = getattr(fisheye_calib, "T_gantry_camera", None)
        offset = getattr(fisheye_calib, "gantry_anchor_offset_mm", None)
        cam_csv = output_dir / "camera_trajectory.csv"
        tag_csv = output_dir / "tag_poses.csv"
        return write_experiment_dashboard_html(
            output_dir / "trajectory_interactive.html",
            gantry_csv=gantry_csv,
            camera_csv=(cam_csv if cam_csv.exists() else None),
            tag_poses_csv=(tag_csv if tag_csv.exists() else None),
            pool_cfg=pool_cfg,
            anchor_id=anchor_tag_id,
            T_gantry_camera=T_gc,
            gantry_anchor_offset_mm=(list(offset) if offset is not None else None),
            R_gantry_to_slam=R,
            run_name=str(output_dir.name),
            rms_summary=rms_summary,
            tag_size_m=tag_size_m,
            plot_z_scale=plot_z_scale,
        ) or (output_dir / "trajectory_interactive.html")

    html_path = output_dir / "trajectory_interactive.html"
    html_path.write_text(
        _build_trajectory_viewer_html(
            camera_rows, tag_rows, pool_cfg, tag_size_m, plot_z_scale, anchor_tag_id
        ),
        encoding="utf-8",
    )
    return html_path


def _build_trajectory_viewer_html(
    camera_rows: list,
    tag_rows: list,
    pool_cfg: dict,
    tag_size_m: float,
    plot_z_scale: float,
    anchor_tag_id: int,
    *,
    gantry_traj: "list | None" = None,
) -> str:
    """Return the self-contained rich 3D-viewer HTML as a string.

    Shared by the standalone zed2 path (write_interactive_trajectory_html) and
    the experiment dashboard (which embeds the returned string in an <iframe>
    srcdoc as the Trajectory tab). ``gantry_traj`` is an optional list of
    ``{x_m, y_m, z_m, t}`` samples in the SLAM frame (already aligned to the
    camera frame); when supplied the viewer overlays it as a plasma time-coded
    trajectory via DATA.gantry.
    """
    pool_cfg = normalize_pool_config(pool_cfg)
    if anchor_tag_id == 1:
        # In the Tag-1 world frame the physical Tag 1 location is the origin.
        # Keep the fixed pool overlay consistent even if an older config still
        # contains a non-anchor-frame tag1_position_m value.
        pool_cfg["tag1_position_m"] = [0.0, 0.0, 0.0]
    data_json = json.dumps(
        {
            "camera": camera_rows,
            "tags": tag_rows,
            "gantry": list(gantry_traj or []),
            "pool": {
                "config": pool_cfg,
                "geometry": pool_geometry_json(pool_cfg),
            },
            "visualization": {
                "tag_size_m": float(tag_size_m),
                "tag_triad_length_m": 0.10,
                "trajectory_full_opacity_tail": 30,
                "display_x_axis": "short_edge",
                "display_flip_z": True,
                "plot_z_scale": float(plot_z_scale),
                "reference_frame_offset_m": [0.0, 0.0, 0.60],
                "anchor_tag_id": int(anchor_tag_id),
            },
            "reference": {
                "origin": f"Tag {anchor_tag_id} + 0.60 m visual reference",
                "x": f"+X in tag-{anchor_tag_id} frame",
                "y": f"+Y in tag-{anchor_tag_id} frame",
                "z": f"+Z in tag-{anchor_tag_id} frame",
            },
        },
        indent=2,
    )

    html_template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Interactive TagSLAM Trajectory</title>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #f4f6f8;
      color: #17202a;
      font-family: Arial, sans-serif;
    }
    body {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    #toolbar {
      box-sizing: border-box;
      min-height: 86px;
      padding: 10px 14px;
      border-bottom: 1px solid #c8d0d8;
      background: #ffffff;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button, select {
      height: 34px;
      padding: 0 11px;
      border: 1px solid #98a6b3;
      border-radius: 4px;
      background: #f8fafc;
      color: #17202a;
      cursor: pointer;
      font-size: 13px;
    }
    button:hover, select:hover { background: #eef3f7; }
    /* Layer-toggle buttons: solid when the layer is shown, faded when hidden. */
    button.toggle { border-color: #6c8aa6; }
    button.toggle.off { opacity: 0.45; background: #eceff2; }
    #slider {
      flex: 1 1 220px;
      min-width: 180px;
    }
    #frameLabel {
      font-size: 13px;
      white-space: nowrap;
      color: #34495e;
    }
    #hint {
      flex-basis: 100%;
      font-size: 12px;
      color: #5d6d7e;
    }
    #content {
      box-sizing: border-box;
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: minmax(360px, 38%) minmax(520px, 1fr);
      gap: 0;
    }
    #plotPanel {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(320px, 1fr) 230px;
      background: #fbfcfd;
    }
    #plot {
      display: block;
      width: 100%;
      height: 100%;
      background: #fbfcfd;
      cursor: grab;
    }
    #plot:active { cursor: grabbing; }
    #profile {
      display: block;
      width: 100%;
      height: 100%;
      border-top: 1px solid #c8d0d8;
      background: #ffffff;
    }
    #zedPanel {
      box-sizing: border-box;
      height: 100%;
      padding: 12px;
      border-right: 1px solid #c8d0d8;
      background: #eef2f5;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
    }
    #zedTitle {
      font-size: 14px;
      font-weight: 700;
      color: #17202a;
    }
    #zedImageWrap {
      min-height: 0;
      display: grid;
      place-items: center;
      background: #111820;
      border: 1px solid #aab6c2;
    }
    #zedImage {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
    }
    #zedPlaceholder {
      color: #d5dde5;
      font-size: 13px;
    }
    #zedMeta {
      font-size: 12px;
      color: #34495e;
      min-height: 30px;
    }
    @media (max-width: 1000px) {
      #content {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(320px, 1fr) minmax(260px, 42%);
      }
      #zedPanel {
        border-right: 0;
        border-bottom: 1px solid #c8d0d8;
      }
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="play">Play</button>
    <button id="restart">Restart</button>
    <button id="full">Show Full</button>
    <input id="slider" type="range" min="0" value="0">
    <select id="speed" title="Playback speed (real-time = 1x)">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x (real-time)</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <button id="top">Top</button>
    <button id="front">Front</button>
    <button id="side">Side</button>
    <button id="iso">Iso</button>
    <button id="reset">Reset View</button>
    <button id="fit">Fit Data</button>
    <button id="ids">Show IDs</button>
    <button id="zscale">Z 1:1</button>
    <button id="tgCam" class="toggle">Camera</button>
    <button id="tgGantry" class="toggle">Gantry</button>
    <button id="tgTags" class="toggle">Tags</button>
    <button id="tgPool" class="toggle">Pool</button>
    <button id="tgMarkers" class="toggle">Markers</button>
    <div id="hint"></div>
  </div>
  <div id="content">
    <section id="zedPanel">
      <div id="zedTitle">ZED2 View With Detected AprilTags</div>
      <div id="zedImageWrap">
        <img id="zedImage" alt="Annotated ZED2 frame">
        <div id="zedPlaceholder">No recorded ZED frame for this sample.</div>
      </div>
      <div id="zedMeta"></div>
    </section>
    <section id="plotPanel">
      <canvas id="plot"></canvas>
      <canvas id="profile"></canvas>
    </section>
  </div>
  <script>
const DATA = __DATA_JSON__;

const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const profileCanvas = document.getElementById("profile");
const profileCtx = profileCanvas.getContext("2d");
const zedImage = document.getElementById("zedImage");
const zedPlaceholder = document.getElementById("zedPlaceholder");
const zedMeta = document.getElementById("zedMeta");
const slider = document.getElementById("slider");
const playButton = document.getElementById("play");
const restartButton = document.getElementById("restart");
const fullButton = document.getElementById("full");
const topButton = document.getElementById("top");
const frontButton = document.getElementById("front");
const sideButton = document.getElementById("side");
const isoButton = document.getElementById("iso");
const resetButton = document.getElementById("reset");
const fitButton = document.getElementById("fit");
const idsButton = document.getElementById("ids");
const zScaleButton = document.getElementById("zscale");
const speedSelect = document.getElementById("speed");
const hint = document.getElementById("hint");
const frameLabel = document.createElement("div");
frameLabel.id = "frameLabel";
document.getElementById("toolbar").insertBefore(frameLabel, speedSelect);

// Parallel gantry trajectory (SLAM frame, already aligned to the camera frame).
// Empty for the standalone zed2 path; populated by the experiment dashboard.
const GANTRY_TRAJ = Array.isArray(DATA.gantry) ? DATA.gantry : [];

// Layer-visibility flags driven by the toolbar toggle buttons.
const layerShow = { camera: true, gantry: true, tags: true, pool: true, markers: true };

let currentIndex = 0;
let playing = false;
let lastStepMs = 0;
let playAnchorMs = null;   // wall-clock anchor for real-time playback
let playAnchorT = 0.0;     // camera time_s at the playback anchor
let rotX = 0.52;
let rotY = 0.0;
let zoom = 1.0;
let targetRotX = rotX;
let targetRotY = rotY;
let targetZoom = zoom;
let dragging = false;
let dragMode = "rotate";
let panX = 0.0;
let panY = 0.0;
let targetPanX = 0.0;
let targetPanY = 0.0;
let lastPointerX = 0;
let lastPointerY = 0;
let dpr = Math.max(1, window.devicePixelRatio || 1);
let showAllIds = false;
let fitToData = false;
let useCompressedZ = false;
let activePreset = "iso";
const POOL = DATA.pool || {};
const POOL_CFG = POOL.config || {};
const POOL_GEOM = POOL.geometry || null;
const VIS = DATA.visualization || {};
const TAIL_FULL_OPACITY = VIS.trajectory_full_opacity_tail || 30;
const TAG_TRIAD_LENGTH_M = VIS.tag_triad_length_m || 0.10;
const TAG_SIZE_M = VIS.tag_size_m || 0.085;
const ANCHOR_TAG_ID = Number(VIS.anchor_tag_id || 1);
const DISPLAY_X_AXIS = VIS.display_x_axis || "short_edge";
const DISPLAY_FLIP_Z = VIS.display_flip_z !== false;
const COMPRESSED_Z_SCALE = Math.max(0.05, Number(VIS.plot_z_scale || 0.5));
let currentZScale = 1.0;
const REF_FRAME_OFFSET_M = VIS.reference_frame_offset_m || [0.0, 0.0, 0.60];
const POOL_EDGE = "#4A7A9C";
const POOL_WATER = "#6FB3D9";
const POOL_FLOOR = "#315C78";
const POOL_GRID = "#CBD5DD";

slider.max = Math.max(0, (DATA.camera.length || (Array.isArray(DATA.gantry) ? DATA.gantry.length : 0)) - 1);

function allPoints() {
  const points = [[0, 0, 0]];
  if (POOL_GEOM) {
    POOL_GEOM.floor.forEach(p => points.push(p));
    POOL_GEOM.top.forEach(p => points.push(p));
  }
  DATA.camera.forEach(row => points.push([row.x_m, row.y_m, row.z_m]));
  DATA.tags.forEach(row => points.push([row.x_m, row.y_m, row.z_m]));
  return points;
}

function dataPoints() {
  const points = [[0, 0, 0]];
  DATA.camera.forEach(row => points.push([row.x_m, row.y_m, row.z_m]));
  DATA.tags.forEach(row => points.push([row.x_m, row.y_m, row.z_m]));
  return points;
}

function pointBounds(points) {
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  points.forEach(p => {
    for (let i = 0; i < 3; i++) {
      mins[i] = Math.min(mins[i], p[i]);
      maxs[i] = Math.max(maxs[i], p[i]);
    }
  });
  const center = mins.map((v, i) => (v + maxs[i]) / 2);
  const radius = Math.max(0.20, ...mins.map((v, i) => maxs[i] - v)) / 2;
  return {center, radius};
}

function bounds() {
  if (!fitToData && POOL_GEOM && POOL_GEOM.axis_limits) {
    const limits = POOL_GEOM.axis_limits;
    return {
      center: [
        (limits.x[0] + limits.x[1]) / 2,
        (limits.y[0] + limits.y[1]) / 2,
        (limits.z[0] + limits.z[1]) / 2,
      ],
      radius: Math.max(
        0.20,
        limits.x[1] - limits.x[0],
        limits.y[1] - limits.y[0],
        limits.z[1] - limits.z[0],
      ) / 2,
      limits,
    };
  }
  return pointBounds(fitToData ? dataPoints() : allPoints());
}

function displaySwapXY() {
  if (DISPLAY_X_AXIS === "short_edge") {
    const longAxis = (POOL_GEOM && POOL_GEOM.long_axis) || POOL_CFG.pool_long_axis || "x";
    return longAxis === "x";
  }
  return Boolean(VIS.display_swap_xy);
}

function plotCoordinatePoint(point) {
  const z = DISPLAY_FLIP_Z ? -point[2] : point[2];
  return displaySwapXY()
    ? [point[1], point[0], z]
    : [point[0], point[1], z];
}

function scaleDisplayZ(z) {
  return z * currentZScale;
}

function unscaleDisplayZ(z) {
  return z / currentZScale;
}

function displayPoint(point) {
  const display = plotCoordinatePoint(point);
  display[2] = scaleDisplayZ(display[2]);
  return display;
}

function worldPoint(display) {
  const zUnscaled = unscaleDisplayZ(display[2]);
  const z = DISPLAY_FLIP_Z ? -zUnscaled : zUnscaled;
  return displaySwapXY()
    ? [display[1], display[0], z]
    : [display[0], display[1], z];
}

function worldPointFromPlotCoords(display) {
  return worldPoint([display[0], display[1], scaleDisplayZ(display[2])]);
}

function displayBounds(rawBounds) {
  const output = {
    center: displayPoint(rawBounds.center),
    radius: rawBounds.radius,
  };
  if (rawBounds.limits) {
    const zLimits = DISPLAY_FLIP_Z
      ? [-rawBounds.limits.z[1], -rawBounds.limits.z[0]]
      : rawBounds.limits.z;
    output.limits = displaySwapXY()
      ? {x: rawBounds.limits.y, y: rawBounds.limits.x, z: zLimits}
      : {x: rawBounds.limits.x, y: rawBounds.limits.y, z: zLimits};
  }
  return output;
}

function displayAxisLabel(axis) {
  return axis.toUpperCase();
}

function trajectoryColor(index) {
  const n = Math.max(1, DATA.camera.length - 1);
  const u = Math.max(0, Math.min(1, index / n));
  const start = [91, 150, 184];
  const end = [25, 44, 68];
  const rgb = start.map((v, i) => Math.round(v + (end[i] - v) * u));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function rgba(rgb, alpha) {
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function trajectoryRgb(index) {
  const n = Math.max(1, DATA.camera.length - 1);
  const u = Math.max(0, Math.min(1, index / n));
  const start = [91, 150, 184];
  const end = [25, 44, 68];
  return start.map((v, i) => Math.round(v + (end[i] - v) * u));
}

const VIRIDIS = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [32, 144, 140],
  [34, 167, 132],
  [68, 190, 112],
  [121, 209, 81],
  [189, 223, 38],
  [253, 231, 37],
];

function viridisRgbAt(u) {
  const clamped = Math.max(0, Math.min(1, u));
  const scaled = clamped * (VIRIDIS.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(VIRIDIS.length - 1, lo + 1);
  const t = scaled - lo;
  return VIRIDIS[lo].map((v, i) => Math.round(v + (VIRIDIS[hi][i] - v) * t));
}

function viridisColorAt(u) {
  const rgb = viridisRgbAt(u);
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

// Plasma colormap — used for the gantry trajectory so it is time-coded like the
// camera (viridis) yet visually distinct (warm vs. cool ramp).
const PLASMA = [
  [13, 8, 135],
  [84, 2, 163],
  [139, 10, 165],
  [185, 50, 137],
  [219, 92, 104],
  [244, 136, 73],
  [254, 188, 43],
  [240, 249, 33],
];

function plasmaRgbAt(u) {
  const clamped = Math.max(0, Math.min(1, u));
  const scaled = clamped * (PLASMA.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(PLASMA.length - 1, lo + 1);
  const t = scaled - lo;
  return PLASMA[lo].map((v, i) => Math.round(v + (PLASMA[hi][i] - v) * t));
}

function plasmaColorAt(u) {
  const rgb = plasmaRgbAt(u);
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function rgbCss(rgb, alpha = 1.0) {
  return alpha >= 1.0
    ? `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
    : `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function desaturateRgb(rgb, amount = 0.72) {
  const gray = Math.round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]);
  return rgb.map(v => Math.round(v * (1 - amount) + gray * amount));
}

function tagZRange() {
  const values = DATA.tags
    .map(tag => plotCoordinatePoint(tagPoint(tag))[2])
    .filter(value => Number.isFinite(value));
  if (values.length === 0) {
    return {min: -0.05, max: 0.05};
  }
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (Math.abs(maxValue - minValue) < 1e-6) {
    minValue -= 0.05;
    maxValue += 0.05;
  }
  return {min: minValue, max: maxValue};
}

function tagZUnit(tag) {
  const range = tagZRange();
  const z = plotCoordinatePoint(tagPoint(tag))[2];
  return Math.max(0, Math.min(1, (z - range.min) / (range.max - range.min)));
}

function tagRgb(tag) {
  return viridisRgbAt(tagZUnit(tag));
}

function rpyToMatrix(rollDeg, pitchDeg, yawDeg) {
  const r = rollDeg * Math.PI / 180;
  const p = pitchDeg * Math.PI / 180;
  const y = yawDeg * Math.PI / 180;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
}

function addScaledAxis(origin, matrix, column, scale) {
  return [
    origin[0] + matrix[0][column] * scale,
    origin[1] + matrix[1][column] * scale,
    origin[2] + matrix[2][column] * scale,
  ];
}

function tagPoint(tag) {
  return [tag.x_m, tag.y_m, tag.z_m];
}

function currentCameraRow() {
  if (DATA.camera.length === 0) {
    return null;
  }
  return DATA.camera[Math.min(currentIndex, DATA.camera.length - 1)];
}

function detectedTagCount(row) {
  return detectedTagIds(row).length;
}

function detectedTagIds(row) {
  if (!row || !row.detected_tags) {
    return [];
  }
  return row.detected_tags
    .split(/\s+/)
    .filter(Boolean)
    .map(value => Number(value))
    .filter(value => Number.isFinite(value));
}

function activeTagIdSet() {
  return new Set(detectedTagIds(currentCameraRow()));
}

function anchorEstimate() {
  return DATA.tags.find(tag => Number(tag.tag_id) === ANCHOR_TAG_ID) || null;
}

function referenceFrameOrigin() {
  const estimate = anchorEstimate();
  const base = estimate
    ? [estimate.x_m, estimate.y_m, estimate.z_m]
    : [0, 0, 0];
  const display = plotCoordinatePoint(base);
  return worldPointFromPlotCoords([
    display[0] + (REF_FRAME_OFFSET_M[0] || 0),
    display[1] + (REF_FRAME_OFFSET_M[1] || 0),
    display[2] + (REF_FRAME_OFFSET_M[2] || 0),
  ]);
}

function tagZSpanCm() {
  if (!DATA.tags || DATA.tags.length === 0) {
    return null;
  }
  const zValues = DATA.tags.map(tag => tag.z_m).filter(value => Number.isFinite(value));
  if (zValues.length === 0) {
    return null;
  }
  return (Math.max(...zValues) - Math.min(...zValues)) * 100;
}

function cameraPathLengthM(endIndex) {
  const end = Math.min(endIndex, DATA.camera.length - 1);
  let distance = 0.0;
  for (let i = 1; i <= end; i++) {
    const a = DATA.camera[i - 1];
    const b = DATA.camera[i];
    const dx = b.x_m - a.x_m;
    const dy = b.y_m - a.y_m;
    const dz = b.z_m - a.z_m;
    distance += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return distance;
}

function project(point) {
  const b = displayBounds(bounds());
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const display = displayPoint(point);
  let x = display[0] - b.center[0];
  let y = display[1] - b.center[1];
  let z = display[2] - b.center[2];
  const scale = zoom * Math.min(width, height) * 0.42 / Math.max(b.radius, 0.001);

  // World convention for the report: X/Y are the pool-floor footprint,
  // and Z is the vertical water-depth axis. Yaw rotates within the
  // footprint plane; elevation then tilts that plane around the screen X axis.
  const cosYaw = Math.cos(rotY);
  const sinYaw = Math.sin(rotY);
  const footprintX = cosYaw * x - sinYaw * y;
  const footprintY = sinYaw * x + cosYaw * y;
  const cosElev = Math.cos(rotX);
  const sinElev = Math.sin(rotX);
  const screenY = cosElev * z - sinElev * footprintY;
  const depth = sinElev * z + cosElev * footprintY;
  return {
    x: width / 2 + panX + footprintX * scale,
    y: height / 2 + panY - screenY * scale,
    depth: depth,
  };
}

function drawLine3d(a, b, color, width = 2, dash = []) {
  const pa = project(a);
  const pb = project(b);
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.stroke();
  ctx.restore();
}

function drawPoint3d(point, color, radius = 4, label = null, square = false, alpha = 1.0) {
  const p = project(point);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  if (square) {
    ctx.fillRect(p.x - radius, p.y - radius, radius * 2, radius * 2);
    ctx.strokeRect(p.x - radius, p.y - radius, radius * 2, radius * 2);
  } else {
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  if (label) {
    ctx.fillStyle = "#17202a";
    ctx.font = "11px Arial";
    ctx.fillText(label, p.x + radius + 5, p.y - radius - 3);
  }
  ctx.restore();
}

function drawPolygon3d(points, color) {
  if (!points || points.length < 3) {
    return;
  }
  const projected = points.map(project);
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i++) {
    ctx.lineTo(projected[i].x, projected[i].y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLoop3d(points, color, width = 1.5, dash = [], alpha = 1.0) {
  if (!points || points.length === 0) {
    return;
  }
  ctx.save();
  ctx.globalAlpha = alpha;
  for (let i = 0; i < points.length; i++) {
    drawLine3d(points[i], points[(i + 1) % points.length], color, width, dash);
  }
  ctx.restore();
}

function drawPool() {
  if (!layerShow.pool || !POOL_GEOM) {
    return;
  }
  const floor = POOL_GEOM.floor;
  const top = POOL_GEOM.top;
  drawPolygon3d(top, "rgba(111, 179, 217, 0.08)");
  drawFloorGrid();
  drawLoop3d(floor, POOL_FLOOR, 2.0, [], 0.95);
  drawLoop3d(top, POOL_EDGE, 1.5, [7, 5], 0.76);
  for (let i = 0; i < 4; i++) {
    drawLine3d(floor[i], top[i], POOL_EDGE, 0.9, []);
  }
}

function drawFloorGrid() {
  if (!POOL_GEOM) {
    return;
  }
  const floor = POOL_GEOM.floor;
  const p0 = floor[0], p1 = floor[1], p2 = floor[2], p3 = floor[3];
  const divisions = 8;
  for (let i = 1; i < divisions; i++) {
    const u = i / divisions;
    const a = interpPoint(p0, p1, u);
    const b = interpPoint(p3, p2, u);
    const c = interpPoint(p0, p3, u);
    const d = interpPoint(p1, p2, u);
    drawLine3d(a, b, POOL_GRID, 0.7, []);
    drawLine3d(c, d, POOL_GRID, 0.7, []);
  }
}

function interpPoint(a, b, u) {
  return [
    a[0] * (1 - u) + b[0] * u,
    a[1] * (1 - u) + b[1] * u,
    a[2] * (1 - u) + b[2] * u,
  ];
}

function niceTickStep(rangeMeters) {
  if (!Number.isFinite(rangeMeters) || rangeMeters <= 0) {
    return 1.0;
  }
  const rough = rangeMeters / 8.0;
  const exponent = Math.floor(Math.log10(rough));
  const base = Math.pow(10, exponent);
  for (const multiplier of [1, 2, 5, 10]) {
    if (rough <= multiplier * base) {
      return multiplier * base;
    }
  }
  return base;
}

function tickValues(minValue, maxValue, step) {
  const ticks = [];
  const start = Math.ceil(minValue / step) * step;
  for (let value = start; value <= maxValue + step * 0.25; value += step) {
    ticks.push(Math.abs(value) < 1e-9 ? 0 : value);
  }
  return ticks;
}

function tickFormat(step) {
  return step < 1.0 ? 1 : 0;
}

function drawTextAtPoint(point, text, dx = 0, dy = 0, color = "#43515c") {
  const projected = project(point);
  ctx.fillStyle = color;
  ctx.fillText(text, projected.x + dx, projected.y + dy);
}

function drawMeterTicks() {
  if (!POOL_GEOM) {
    return;
  }
  const b = displayBounds(bounds());
  if (!b.limits) {
    return;
  }
  const limits = b.limits;
  const floorZ = displayPoint([0, 0, POOL_GEOM.floor_z || 0.0])[2];
  const xRange = limits.x[1] - limits.x[0];
  const yRange = limits.y[1] - limits.y[0];
  const zMinMeters = unscaleDisplayZ(limits.z[0]);
  const zMaxMeters = unscaleDisplayZ(limits.z[1]);
  const zRange = zMaxMeters - zMinMeters;
  const tickLen = Math.max(0.04, Math.min(xRange, yRange) * 0.025);
  const xStep = niceTickStep(xRange);
  const yStep = niceTickStep(yRange);
  const zStep = niceTickStep(zRange);
  const xDigits = tickFormat(xStep);
  const yDigits = tickFormat(yStep);
  const zDigits = tickFormat(zStep);
  const yEdge = limits.y[0];
  const xEdge = limits.x[0];

  ctx.save();
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.strokeStyle = "#7b8794";
  ctx.fillStyle = "#43515c";

  tickValues(limits.x[0], limits.x[1], xStep).forEach(value => {
    const a = worldPoint([value, yEdge, floorZ]);
    const c = worldPoint([value, yEdge + tickLen, floorZ]);
    drawLine3d(a, c, "#7b8794", 0.8, []);
    drawTextAtPoint(
      worldPoint([value, yEdge - tickLen * 1.9, floorZ]),
      value.toFixed(xDigits),
      -10,
      4,
    );
  });

  tickValues(limits.y[0], limits.y[1], yStep).forEach(value => {
    const a = worldPoint([xEdge, value, floorZ]);
    const c = worldPoint([xEdge + tickLen, value, floorZ]);
    drawLine3d(a, c, "#7b8794", 0.8, []);
    drawTextAtPoint(
      worldPoint([xEdge - tickLen * 2.2, value, floorZ]),
      value.toFixed(yDigits),
      -12,
      4,
    );
  });

  drawTextAtPoint(
    worldPoint([(limits.x[0] + limits.x[1]) / 2, yEdge - tickLen * 4.2, floorZ]),
    `${displayAxisLabel("x")} (m)`,
    -18,
    6,
    "#17202a",
  );
  drawTextAtPoint(
    worldPoint([xEdge - tickLen * 4.4, (limits.y[0] + limits.y[1]) / 2, floorZ]),
    `${displayAxisLabel("y")} (m)`,
    -18,
    6,
    "#17202a",
  );

  if (activePreset !== "top") {
    const zAxisX = xEdge;
    const zAxisY = yEdge;
    drawLine3d(
      worldPoint([zAxisX, zAxisY, limits.z[0]]),
      worldPoint([zAxisX, zAxisY, limits.z[1]]),
      "#7b8794",
      0.9,
      [],
    );
    tickValues(zMinMeters, zMaxMeters, zStep).forEach(value => {
      const displayZ = scaleDisplayZ(value);
      const a = worldPoint([zAxisX, zAxisY, displayZ]);
      const c = worldPoint([zAxisX + tickLen, zAxisY, displayZ]);
      drawLine3d(a, c, "#7b8794", 0.8, []);
      drawTextAtPoint(
        worldPoint([zAxisX - tickLen * 2.8, zAxisY - tickLen * 0.8, displayZ]),
        value.toFixed(zDigits),
        -14,
        4,
      );
    });
    drawTextAtPoint(
      worldPoint([
        zAxisX - tickLen * 4.8,
        zAxisY - tickLen * 1.2,
        (limits.z[0] + limits.z[1]) / 2,
      ]),
      `${displayAxisLabel("z")} (m)`,
      -18,
      6,
      "#17202a",
    );
  }
  ctx.restore();
}

function drawAnchorOrigin() {
  const estimate = anchorEstimate();
  const point = estimate ? tagPoint(estimate) : [0, 0, 0];
  drawPoint3d(point, "#111111", 5, `Tag ${ANCHOR_TAG_ID} world origin`, false);
}

function drawAxes() {
  const b = bounds();
  const origin = referenceFrameOrigin();
  const displayOrigin = displayPoint(origin);
  const axisLen = Math.max(0.25, Math.min(0.70, b.radius * 0.20));
  const xEnd = worldPoint([displayOrigin[0] + axisLen, displayOrigin[1], displayOrigin[2]]);
  const yEnd = worldPoint([displayOrigin[0], displayOrigin[1] + axisLen, displayOrigin[2]]);
  const zEnd = worldPoint([displayOrigin[0], displayOrigin[1], displayOrigin[2] + axisLen]);
  drawLine3d(origin, xEnd, "#D33F49", 4.0);
  drawLine3d(origin, yEnd, "#2E8B57", 4.0);
  drawLine3d(origin, zEnd, "#2F6DB3", 4.0);
  drawPoint3d(xEnd, "#D33F49", 4, "X");
  drawPoint3d(yEnd, "#2E8B57", 4, "Y");
  drawPoint3d(zEnd, "#2F6DB3", 4, "Z");
  drawPoint3d(origin, "#111111", 4, `Tag ${ANCHOR_TAG_ID} +0.60 m ref`);
}

function drawTags() {
  if (!layerShow.tags) {
    return;
  }
  const sortedTags = [...DATA.tags].sort((a, b) => a.tag_id - b.tag_id);
  const activeIds = activeTagIdSet();
  const row = currentCameraRow();
  const cameraPoint = row ? [row.x_m, row.y_m, row.z_m] : null;
  if (cameraPoint) {
    sortedTags.forEach(tag => {
      if (!activeIds.has(Number(tag.tag_id))) {
        return;
      }
      drawLine3d(cameraPoint, tagPoint(tag), "rgba(27, 42, 65, 0.32)", 1.1, [3, 4]);
    });
  }
  sortedTags.forEach(tag => {
    const isActive = activeIds.has(Number(tag.tag_id));
    const label = isActive || showAllIds ? `Tag ${tag.tag_id}` : null;
    const baseRgb = tagRgb(tag);
    const color = isActive ? rgbCss(baseRgb) : rgbCss(desaturateRgb(baseRgb), 0.55);
    const alpha = isActive ? 1.0 : 0.25;
    const point = tagPoint(tag);
    const sizePx = Math.max(5, Math.min(16, projectedMetricSize(point, TAG_SIZE_M)));
    drawPoint3d(point, color, (isActive ? 0.72 : 0.50) * sizePx, label, true, alpha);
    if (isActive || showAllIds) {
      drawTagTriad(tag, isActive ? 1.0 : 0.35);
    }
  });
}

function projectedMetricSize(point, meters) {
  const a = project(point);
  const b = project([point[0] + meters, point[1], point[2]]);
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function drawTagTriad(tag, alpha = 1.0) {
  const origin = tagPoint(tag);
  const rot = rpyToMatrix(tag.roll_deg || 0, tag.pitch_deg || 0, tag.yaw_deg || 0);
  drawLine3d(origin, addScaledAxis(origin, rot, 0, TAG_TRIAD_LENGTH_M), rgba([211, 63, 73], alpha), 1.6);
  drawLine3d(origin, addScaledAxis(origin, rot, 1, TAG_TRIAD_LENGTH_M), rgba([46, 139, 87], alpha), 1.6);
  drawLine3d(origin, addScaledAxis(origin, rot, 2, TAG_TRIAD_LENGTH_M), rgba([47, 109, 179], alpha), 1.6);
}

function drawTrajectory() {
  if (!layerShow.camera || DATA.camera.length === 0) {
    return;
  }
  const end = Math.min(currentIndex, DATA.camera.length - 1);
  for (let i = 1; i <= end; i++) {
    const a = DATA.camera[i - 1];
    const b = DATA.camera[i];
    const dash = b.has_tag_update ? [] : [6, 5];
    const age = end - i;
    const alpha = age < TAIL_FULL_OPACITY
      ? 1.0
      : Math.max(0.12, 1.0 - (age - TAIL_FULL_OPACITY) / Math.max(1, end));
    drawLine3d(
      [a.x_m, a.y_m, a.z_m],
      [b.x_m, b.y_m, b.z_m],
      rgba(trajectoryRgb(i), alpha),
      age < TAIL_FULL_OPACITY ? 3.2 : 2.0,
      dash,
    );
  }

  for (let i = 0; i <= end; i++) {
    if (i % 8 !== 0 && i !== end) {
      continue;
    }
    const row = DATA.camera[i];
    drawPoint3d(
      [row.x_m, row.y_m, row.z_m],
      row.has_tag_update ? trajectoryColor(i) : "#8a8f98",
      i === end ? 5 : 2.5,
    );
  }

  const start = DATA.camera[0];
  const current = DATA.camera[end];
  drawPoint3d([start.x_m, start.y_m, start.z_m], "#2ca02c", 5, "Start");
  drawPoint3d(
    [current.x_m, current.y_m, current.z_m],
    "#1B2A41",
    8,
    `Camera (${(current.x_m * 100).toFixed(1)}, ${(current.y_m * 100).toFixed(1)}, ${(current.z_m * 100).toFixed(1)}) cm`,
  );
}

function gantryAlignedPoint(row, offset) {
  return [
    row.gantry_x_mm / 1000 - offset[0],
    row.gantry_y_mm / 1000 - offset[1],
    row.gantry_z_mm / 1000 - offset[2],
  ];
}

function gantryHasData(row) {
  return row && Number.isFinite(row.gantry_x_mm)
              && Number.isFinite(row.gantry_y_mm)
              && Number.isFinite(row.gantry_z_mm);
}

function hasGantryOverlay() {
  return GANTRY_TRAJ.length > 1
      || (DATA.camera.length > 0 && gantryHasData(DATA.camera[0]));
}

// Elapsed time (s) at the current slider position. Driven by the camera sample
// when present, else (gantry-only) by the gantry sample itself.
function currentTimeS() {
  const row = currentCameraRow();
  if (row && Number.isFinite(row.time_s)) {
    return row.time_s;
  }
  if (GANTRY_TRAJ.length) {
    const g = GANTRY_TRAJ[Math.min(currentIndex, GANTRY_TRAJ.length - 1)];
    return Number.isFinite(g.t) ? g.t : 0;
  }
  return 0;
}

// Largest GANTRY_TRAJ index whose timestamp is <= tNow (i.e. revealed so far).
function gantryEndIndex(tNow) {
  let end = 0;
  for (let i = 0; i < GANTRY_TRAJ.length; i++) {
    if (Number.isFinite(GANTRY_TRAJ[i].t) && GANTRY_TRAJ[i].t <= tNow) {
      end = i;
    }
  }
  return end;
}

function drawGantryTrajectory() {
  if (!layerShow.gantry) {
    return;
  }
  // Preferred path: full-rate, SLAM-frame gantry trajectory (DATA.gantry),
  // time-coded with the plasma colormap and revealed up to the slider time.
  if (GANTRY_TRAJ.length > 1) {
    const tNow = currentTimeS();
    const end = gantryEndIndex(tNow);
    const denom = Math.max(1, GANTRY_TRAJ.length - 1);
    for (let i = 1; i <= end; i++) {
      const a = GANTRY_TRAJ[i - 1];
      const b = GANTRY_TRAJ[i];
      drawLine3d(
        [a.x_m, a.y_m, a.z_m],
        [b.x_m, b.y_m, b.z_m],
        plasmaColorAt(i / denom),
        3.4,
        [],
      );
    }
    const step = Math.max(1, Math.round(GANTRY_TRAJ.length / 60));
    for (let i = 0; i <= end; i += step) {
      const g = GANTRY_TRAJ[i];
      drawPoint3d([g.x_m, g.y_m, g.z_m], plasmaColorAt(i / denom), 2.4);
    }
    const cur = GANTRY_TRAJ[end];
    drawPoint3d(
      [cur.x_m, cur.y_m, cur.z_m], "#f0f921", 7,
      `Gantry (${(cur.x_m * 100).toFixed(1)}, ${(cur.y_m * 100).toFixed(1)}, ${(cur.z_m * 100).toFixed(1)}) cm`,
    );
    // Dashed delta line + |Δ| label between the camera and gantry at this time.
    if (layerShow.markers && layerShow.camera) {
      const crow = currentCameraRow();
      if (crow) {
        const cp = [crow.x_m, crow.y_m, crow.z_m];
        const gp = [cur.x_m, cur.y_m, cur.z_m];
        drawLine3d(cp, gp, "rgba(80, 88, 96, 0.85)", 1.3, [4, 3]);
        const dd = Math.hypot(cp[0] - gp[0], cp[1] - gp[1], cp[2] - gp[2]) * 1000;
        const mid = project([(cp[0] + gp[0]) / 2, (cp[1] + gp[1]) / 2, (cp[2] + gp[2]) / 2]);
        ctx.save();
        ctx.fillStyle = "#17202a";
        ctx.font = "11px Arial";
        ctx.fillText(`|Δ|=${dd.toFixed(1)} mm`, mid.x + 6, mid.y - 4);
        ctx.restore();
      }
    }
    return;
  }

  // Legacy fallback: gantry pose carried on each camera row (older runs);
  // first-sample-zeroed to the camera start, solid orange.
  if (DATA.camera.length === 0 || !gantryHasData(DATA.camera[0])) {
    return;
  }
  const c0 = DATA.camera[0];
  const offset = [
    (c0.gantry_x_mm / 1000) - c0.x_m,
    (c0.gantry_y_mm / 1000) - c0.y_m,
    (c0.gantry_z_mm / 1000) - c0.z_m,
  ];
  const end = Math.min(currentIndex, DATA.camera.length - 1);
  for (let i = 1; i <= end; i++) {
    const a = DATA.camera[i - 1];
    const b = DATA.camera[i];
    if (!gantryHasData(a) || !gantryHasData(b)) continue;
    drawLine3d(gantryAlignedPoint(a, offset), gantryAlignedPoint(b, offset),
               "rgba(255, 165, 0, 0.92)", 2.0, []);
  }
  const cur = DATA.camera[end];
  if (gantryHasData(cur)) {
    const p = gantryAlignedPoint(cur, offset);
    drawPoint3d(p, "#d97500", 7,
      `Gantry (${(p[0] * 100).toFixed(1)}, ${(p[1] * 100).toFixed(1)}, ${(p[2] * 100).toFixed(1)}) cm`);
  }
}

function drawLegend() {
  const width = canvas.clientWidth;
  const hasGantry = hasGantryOverlay();
  const usesPlasmaGantry = GANTRY_TRAJ.length > 1;
  const boxH = hasGantry ? 150 : 112;
  const x = width - 270;
  const y = 18;
  ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
  ctx.strokeStyle = "#c8d0d8";
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, 246, boxH);
  ctx.strokeRect(x, y, 246, boxH);
  ctx.font = "12px Arial";
  ctx.fillStyle = "#17202a";
  ctx.fillText("Camera (AprilTag SLAM, viridis by time)", x + 12, y + 22);
  for (let i = 0; i < 96; i++) {
    ctx.fillStyle = viridisColorAt(i / 95);
    ctx.fillRect(x + 12 + i, y + 34, 1, 9);
  }
  ctx.fillStyle = "#17202a";
  ctx.fillText("Dashed/gray: no tag detection", x + 12, y + 62);
  ctx.fillText("Bright tags/lines: active constraints", x + 12, y + 82);
  ctx.fillText("Faint tags: in graph, inactive", x + 12, y + 102);
  if (hasGantry) {
    if (usesPlasmaGantry) {
      ctx.fillStyle = "#17202a";
      ctx.fillText("Gantry GT (plasma by time)", x + 12, y + 122);
      for (let i = 0; i < 96; i++) {
        ctx.fillStyle = plasmaColorAt(i / 95);
        ctx.fillRect(x + 12 + i, y + 132, 1, 9);
      }
    } else {
      ctx.fillStyle = "rgba(255, 165, 0, 0.92)";
      ctx.fillRect(x + 12, y + 130, 24, 4);
      ctx.fillStyle = "#17202a";
      ctx.fillText("Gantry (first-sample-zeroed)", x + 44, y + 134);
    }
  }
}

function drawInfoPanel() {
  const row = currentCameraRow();
  if (!row) {
    return;
  }
  const displayCamera = plotCoordinatePoint([row.x_m, row.y_m, row.z_m]);
  const zSpan = tagZSpanCm();
  const zSpanText = zSpan === null ? " n/a" : ` ${zSpan.toFixed(1)} cm`;
  const activeIds = detectedTagIds(row);
  const activeText = activeIds.length ? activeIds.join(" ") : "none";
  const lines = [
    `Plot cm:   X ${(displayCamera[0] * 100).toFixed(1)}  Y ${(displayCamera[1] * 100).toFixed(1)}  Z ${(displayCamera[2] * 100).toFixed(1)}`,
    `RPY deg:   R ${row.roll_deg.toFixed(1)}  P ${row.pitch_deg.toFixed(1)}  Y ${row.yaw_deg.toFixed(1)}`,
    `Localizing from ${activeIds.length} tags: [${activeText}]`,
    `Path:      ${cameraPathLengthM(currentIndex).toFixed(2)} m traveled`,
    `Tags:      ${detectedTagCount(row)} observed / ${DATA.tags.length} in graph`,
    `Tag Z span:${zSpanText}`,
    `Water:     ${(POOL_CFG.water_depth_m || 0).toFixed(3)} m`,
  ];
  const x = 18;
  const y = 18;
  const width = 330;
  const height = 22 + lines.length * 17;
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.90)";
  ctx.strokeStyle = "#c8d0d8";
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, width, height);
  ctx.strokeRect(x, y, width, height);
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.fillStyle = "#17202a";
  lines.forEach((line, index) => {
    ctx.fillText(line, x + 12, y + 22 + index * 17);
  });
  ctx.restore();
}

function profilePoint(point) {
  return plotCoordinatePoint(point);
}

function profileDataBounds() {
  const xs = [0], ys = [0], zs = [0];
  DATA.camera.forEach(row => {
    const p = profilePoint([row.x_m, row.y_m, row.z_m]);
    xs.push(p[0]);
    ys.push(p[1]);
    zs.push(p[2]);
  });
  DATA.tags.forEach(tag => {
    const p = profilePoint(tagPoint(tag));
    xs.push(p[0]);
    ys.push(p[1]);
    zs.push(p[2]);
  });
  const expand = (values, padMin = 0.05) => {
    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    const pad = Math.max(padMin, (maxValue - minValue) * 0.08);
    if (Math.abs(maxValue - minValue) < 1e-6) {
      minValue -= padMin;
      maxValue += padMin;
    } else {
      minValue -= pad;
      maxValue += pad;
    }
    return [minValue, maxValue];
  };
  return {x: expand(xs), y: expand(ys), z: expand(zs)};
}

function drawProfileAxes(ctx2, rect, xLabel, xLimits, zLimits) {
  const margin = {left: 46, right: 12, top: 22, bottom: 32};
  const plot = {
    x: rect.x + margin.left,
    y: rect.y + margin.top,
    w: Math.max(20, rect.w - margin.left - margin.right),
    h: Math.max(20, rect.h - margin.top - margin.bottom),
  };
  const xRange = Math.max(1e-6, xLimits[1] - xLimits[0]);
  const zRange = Math.max(1e-6, zLimits[1] - zLimits[0]);
  const scale = Math.min(plot.w / xRange, plot.h / zRange);
  const usedW = xRange * scale;
  const usedH = zRange * scale;
  const originX = plot.x + (plot.w - usedW) / 2;
  const originY = plot.y + (plot.h - usedH) / 2;
  const map = (x, z) => ({
    x: originX + (x - xLimits[0]) * scale,
    y: originY + usedH - (z - zLimits[0]) * scale,
  });

  ctx2.save();
  ctx2.strokeStyle = "#d7dee5";
  ctx2.lineWidth = 1;
  ctx2.strokeRect(originX, originY, usedW, usedH);
  const xStep = niceTickStep(xRange);
  const zStep = niceTickStep(zRange);
  ctx2.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx2.fillStyle = "#52616d";
  ctx2.strokeStyle = "#edf1f4";
  tickValues(xLimits[0], xLimits[1], xStep).forEach(value => {
    const a = map(value, zLimits[0]);
    const b = map(value, zLimits[1]);
    ctx2.beginPath();
    ctx2.moveTo(a.x, a.y);
    ctx2.lineTo(b.x, b.y);
    ctx2.stroke();
    ctx2.fillText(value.toFixed(tickFormat(xStep)), a.x - 10, originY + usedH + 15);
  });
  tickValues(zLimits[0], zLimits[1], zStep).forEach(value => {
    const a = map(xLimits[0], value);
    const b = map(xLimits[1], value);
    ctx2.beginPath();
    ctx2.moveTo(a.x, a.y);
    ctx2.lineTo(b.x, b.y);
    ctx2.stroke();
    ctx2.fillText(value.toFixed(tickFormat(zStep)), rect.x + 8, a.y + 4);
  });
  const z0a = map(xLimits[0], 0);
  const z0b = map(xLimits[1], 0);
  ctx2.strokeStyle = "#7b8794";
  ctx2.setLineDash([5, 4]);
  ctx2.beginPath();
  ctx2.moveTo(z0a.x, z0a.y);
  ctx2.lineTo(z0b.x, z0b.y);
  ctx2.stroke();
  ctx2.setLineDash([]);
  ctx2.fillStyle = "#17202a";
  ctx2.font = "11px Arial";
  ctx2.fillText(`${xLabel} vs Z, true scale`, rect.x + 8, rect.y + 14);
  ctx2.fillText(`${xLabel} (m)`, originX + usedW / 2 - 18, rect.y + rect.h - 8);
  ctx2.save();
  ctx2.translate(rect.x + 14, originY + usedH / 2 + 16);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText("Z (m)", 0, 0);
  ctx2.restore();
  ctx2.restore();
  return map;
}

function drawProfilePanel() {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  profileCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = profileCanvas.clientWidth || 1;
  const height = profileCanvas.clientHeight || 1;
  profileCtx.clearRect(0, 0, width, height);
  profileCtx.fillStyle = "#ffffff";
  profileCtx.fillRect(0, 0, width, height);
  const bounds2 = profileDataBounds();
  const gap = 14;
  const colorbarW = 58;
  const panelW = Math.max(120, (width - colorbarW - gap * 3) / 2);
  const rectX = {x: gap, y: 8, w: panelW, h: height - 16};
  const rectY = {x: gap * 2 + panelW, y: 8, w: panelW, h: height - 16};
  const mapX = drawProfileAxes(profileCtx, rectX, "X", bounds2.x, bounds2.z);
  const mapY = drawProfileAxes(profileCtx, rectY, "Y", bounds2.y, bounds2.z);
  const activeIds = activeTagIdSet();
  const end = Math.min(currentIndex, DATA.camera.length - 1);

  function drawPath(map, coordIndex) {
    if (DATA.camera.length < 2) {
      return;
    }
    profileCtx.save();
    profileCtx.strokeStyle = "#315C78";
    profileCtx.lineWidth = 1.7;
    profileCtx.beginPath();
    for (let i = 0; i <= end; i++) {
      const p = profilePoint([DATA.camera[i].x_m, DATA.camera[i].y_m, DATA.camera[i].z_m]);
      const q = map(p[coordIndex], p[2]);
      if (i === 0) {
        profileCtx.moveTo(q.x, q.y);
      } else {
        profileCtx.lineTo(q.x, q.y);
      }
    }
    profileCtx.stroke();
    profileCtx.restore();
  }

  drawPath(mapX, 0);
  drawPath(mapY, 1);

  DATA.tags.forEach(tag => {
    const p = profilePoint(tagPoint(tag));
    const rgb = tagRgb(tag);
    const active = activeIds.has(Number(tag.tag_id));
    const color = active ? rgbCss(rgb) : rgbCss(desaturateRgb(rgb), 0.45);
    const radius = active ? 5 : 3.2;
    [mapX(p[0], p[2]), mapY(p[1], p[2])].forEach(q => {
      profileCtx.save();
      profileCtx.globalAlpha = active ? 1.0 : 0.35;
      profileCtx.fillStyle = color;
      profileCtx.strokeStyle = active ? "#111820" : "#ffffff";
      profileCtx.lineWidth = active ? 1.4 : 0.8;
      profileCtx.beginPath();
      profileCtx.arc(q.x, q.y, radius, 0, Math.PI * 2);
      profileCtx.fill();
      profileCtx.stroke();
      profileCtx.restore();
    });
  });

  if (DATA.camera.length > 0) {
    const row = DATA.camera[end];
    const p = profilePoint([row.x_m, row.y_m, row.z_m]);
    [mapX(p[0], p[2]), mapY(p[1], p[2])].forEach(q => {
      profileCtx.fillStyle = "#1B2A41";
      profileCtx.beginPath();
      profileCtx.arc(q.x, q.y, 5.5, 0, Math.PI * 2);
      profileCtx.fill();
    });
  }

  const range = tagZRange();
  const barX = width - colorbarW + 18;
  const barY = 34;
  const barH = Math.max(40, height - 76);
  for (let i = 0; i < barH; i++) {
    const u = 1 - i / Math.max(1, barH - 1);
    profileCtx.fillStyle = viridisColorAt(u);
    profileCtx.fillRect(barX, barY + i, 14, 1);
  }
  profileCtx.strokeStyle = "#9aa6b2";
  profileCtx.strokeRect(barX, barY, 14, barH);
  profileCtx.fillStyle = "#17202a";
  profileCtx.font = "11px Arial";
  profileCtx.fillText("Tag Z", barX - 6, 18);
  profileCtx.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  profileCtx.fillText(`${(range.max * 100).toFixed(1)} cm`, barX - 10, barY - 7);
  profileCtx.fillText(`${(range.min * 100).toFixed(1)} cm`, barX - 10, barY + barH + 14);
  profileCtx.fillStyle = "#52616d";
  profileCtx.fillText("z=0 dashed", barX - 16, height - 8);
}

function updateLabel() {
  if (DATA.camera.length === 0) {
    frameLabel.textContent = "No camera trajectory samples";
    zedImage.style.display = "none";
    zedPlaceholder.style.display = "block";
    zedMeta.textContent = "";
    return;
  }
  const row = DATA.camera[currentIndex];
  const tags = row.detected_tags || "none";
  const activeIds = detectedTagIds(row);
  frameLabel.textContent =
    `${currentIndex + 1}/${DATA.camera.length} | X${row.camera_index} | ` +
    `t=${row.time_s.toFixed(2)}s | ` +
    `pos=(${row.x_m.toFixed(3)}, ${row.y_m.toFixed(3)}, ${row.z_m.toFixed(3)}) m | ` +
    `tags=${tags}`;
  slider.value = String(currentIndex);

  if (row.image_path) {
    if (!zedImage.src.endsWith(row.image_path)) {
      zedImage.src = row.image_path;
    }
    zedImage.style.display = "block";
    zedPlaceholder.style.display = "none";
  } else {
    zedImage.removeAttribute("src");
    zedImage.style.display = "none";
    zedPlaceholder.style.display = "block";
  }
  zedMeta.textContent =
    `Camera node X${row.camera_index}, ` +
    `time ${row.time_s.toFixed(2)} s, ` +
    `detected AprilTags: ${tags}`;
  hint.textContent = `Localizing from ${activeIds.length} tags: [${activeIds.length ? activeIds.join(" ") : "none"}]`;
}

function drawCanvasError(error) {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.clientWidth || 1;
  const height = canvas.clientHeight || 1;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#9b2c2c";
  ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.fillText("3D plot error", 18, 28);
  ctx.fillStyle = "#34495e";
  ctx.fillText(String(error && error.message ? error.message : error), 18, 50);
  updateLabel();
}

function drawScene() {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);

  drawPool();
  drawMeterTicks();
  drawAxes();
  drawTags();
  drawAnchorOrigin();
  drawTrajectory();
  drawGantryTrajectory();
  drawLegend();
  drawInfoPanel();
  updateLabel();
}

function draw() {
  try {
    drawScene();
    drawProfilePanel();
  } catch (error) {
    console.error(error);
    drawCanvasError(error);
  }
}

function resize() {
  dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  profileCanvas.width = Math.max(1, Math.floor(profileCanvas.clientWidth * dpr));
  profileCanvas.height = Math.max(1, Math.floor(profileCanvas.clientHeight * dpr));
  draw();
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function setViewTargets(rx, ry, presetName) {
  targetRotX = clamp(rx, -1.55, 1.55);
  targetRotY = ry;
  activePreset = presetName;
}

function snapViewTargets(rx, ry, presetName) {
  setViewTargets(rx, ry, presetName);
  rotX = targetRotX;
  rotY = targetRotY;
  draw();
}

function setIsoView() {
  snapViewTargets(0.58, -0.72, "iso");
}

function setDefaultView() {
  fitToData = false;
  useCompressedZ = false;
  currentZScale = 1.0;
  zScaleButton.textContent = "Z 1:1";
  setViewTargets(0.58, -0.72, "iso");
  rotX = targetRotX;
  rotY = targetRotY;
  zoom = targetZoom = 1.0;
  panX = targetPanX = 0.0;
  panY = targetPanY = 0.0;
}

function resetView() {
  setDefaultView();
  draw();
}

function stepDamping() {
  const before = [rotX, rotY, zoom, panX, panY];
  rotX += (targetRotX - rotX) * 0.22;
  rotY += (targetRotY - rotY) * 0.22;
  zoom += (targetZoom - zoom) * 0.22;
  panX += (targetPanX - panX) * 0.22;
  panY += (targetPanY - panY) * 0.22;
  if (Math.abs(targetRotX - rotX) < 1e-4) rotX = targetRotX;
  if (Math.abs(targetRotY - rotY) < 1e-4) rotY = targetRotY;
  if (Math.abs(targetZoom - zoom) < 1e-4) zoom = targetZoom;
  if (Math.abs(targetPanX - panX) < 0.05) panX = targetPanX;
  if (Math.abs(targetPanY - panY) < 0.05) panY = targetPanY;
  return before.some((value, index) => Math.abs(value - [rotX, rotY, zoom, panX, panY][index]) > 1e-5);
}

function playLength() {
  return DATA.camera.length > 0 ? DATA.camera.length : GANTRY_TRAJ.length;
}

playButton.addEventListener("click", () => {
  if (playLength() === 0) {
    return;
  }
  playing = !playing;
  playAnchorMs = null;  // re-anchor real-time clock on (re)start
  playButton.textContent = playing ? "Pause" : "Play";
});

restartButton.addEventListener("click", () => {
  currentIndex = 0;
  playing = false;
  playAnchorMs = null;
  playButton.textContent = "Play";
  draw();
});

fullButton.addEventListener("click", () => {
  currentIndex = Math.max(0, playLength() - 1);
  playing = false;
  playAnchorMs = null;
  playButton.textContent = "Play";
  draw();
});

resetButton.addEventListener("click", resetView);
topButton.addEventListener("click", () => {
  snapViewTargets(Math.PI / 2, 0.0, "top");
});
frontButton.addEventListener("click", () => {
  snapViewTargets(0.0, 0.0, "front");
});
sideButton.addEventListener("click", () => {
  snapViewTargets(0.0, -Math.PI / 2, "side");
});
isoButton.addEventListener("click", () => {
  setIsoView();
});
fitButton.addEventListener("click", () => {
  fitToData = true;
  targetZoom = 1.15;
  targetPanX = 0.0;
  targetPanY = 0.0;
});
idsButton.addEventListener("click", () => {
  showAllIds = !showAllIds;
  idsButton.textContent = showAllIds ? "Hide IDs" : "Show IDs";
  draw();
});
zScaleButton.addEventListener("click", () => {
  useCompressedZ = !useCompressedZ;
  currentZScale = useCompressedZ ? COMPRESSED_Z_SCALE : 1.0;
  zScaleButton.textContent = useCompressedZ ? `Z ${COMPRESSED_Z_SCALE.toFixed(2)}x` : "Z 1:1";
  draw();
});

slider.addEventListener("input", event => {
  currentIndex = Number(event.target.value);
  playing = false;
  playAnchorMs = null;
  playButton.textContent = "Play";
  draw();
});

// Layer-visibility toggle buttons (camera / gantry / tags / pool / markers).
function wireToggle(id, key) {
  const btn = document.getElementById(id);
  if (!btn) {
    return;
  }
  const sync = () => btn.classList.toggle("off", !layerShow[key]);
  sync();
  btn.addEventListener("click", () => {
    layerShow[key] = !layerShow[key];
    sync();
    draw();
  });
}
wireToggle("tgCam", "camera");
wireToggle("tgGantry", "gantry");
wireToggle("tgTags", "tags");
wireToggle("tgPool", "pool");
wireToggle("tgMarkers", "markers");

canvas.addEventListener("pointerdown", event => {
  event.preventDefault();
  dragging = true;
  dragMode = event.shiftKey || event.button === 1 || event.button === 2
    ? "pan"
    : "rotate";
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", event => {
  if (!dragging) {
    return;
  }
  const dx = event.clientX - lastPointerX;
  const dy = event.clientY - lastPointerY;
  lastPointerX = event.clientX;
  lastPointerY = event.clientY;
  if (dragMode === "pan") {
    targetPanX += dx;
    targetPanY += dy;
  } else {
    targetRotY += dx * 0.007;
    targetRotX = clamp(targetRotX + dy * 0.007, -1.55, 1.55);
    activePreset = "custom";
  }
});

canvas.addEventListener("pointerup", () => {
  dragging = false;
});

canvas.addEventListener("pointercancel", () => {
  dragging = false;
});

canvas.addEventListener("contextmenu", event => {
  event.preventDefault();
});

canvas.addEventListener("dblclick", resetView);

canvas.addEventListener("wheel", event => {
  event.preventDefault();
  targetZoom *= Math.exp(-event.deltaY * 0.0012);
  targetZoom = Math.max(0.12, Math.min(10.0, targetZoom));
}, {passive: false});

// Elapsed time (s) for the playback timeline at a given index — camera time
// when available, else the gantry sample time (gantry-only runs).
function timelineTimeAt(index) {
  if (DATA.camera.length > 0) {
    const row = DATA.camera[Math.min(index, DATA.camera.length - 1)];
    return Number.isFinite(row.time_s) ? row.time_s : index;
  }
  if (GANTRY_TRAJ.length > 0) {
    const g = GANTRY_TRAJ[Math.min(index, GANTRY_TRAJ.length - 1)];
    return Number.isFinite(g.t) ? g.t : index;
  }
  return index;
}

function animate(timestampMs) {
  let needsDraw = stepDamping();
  const n = playLength();
  if (playing && n > 0) {
    const speed = Number(speedSelect.value) || 1.0;  // real-time multiplier
    if (playAnchorMs === null) {
      playAnchorMs = timestampMs;
      playAnchorT = timelineTimeAt(currentIndex);
    }
    // Advance currentIndex so its sample time matches real elapsed wall-clock
    // time (1 s of recording == 1 s of playback at 1x).
    const targetT = playAnchorT + (timestampMs - playAnchorMs) / 1000 * speed;
    let idx = currentIndex;
    while (idx + 1 < n && timelineTimeAt(idx + 1) <= targetT) {
      idx += 1;
    }
    if (idx !== currentIndex) {
      currentIndex = idx;
      needsDraw = true;
    }
    if (currentIndex >= n - 1) {
      currentIndex = n - 1;
      playing = false;
      playAnchorMs = null;
      playButton.textContent = "Play";
    }
    lastStepMs = timestampMs;
  }
  if (needsDraw) {
    draw();
  }
  requestAnimationFrame(animate);
}

window.addEventListener("resize", resize);
setDefaultView();
resize();
requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
    return html_template.replace("__DATA_JSON__", data_json)


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.20)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def set_axes_equal_2d(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.20)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def trajectory_color_tuple(index: int, count: int) -> tuple[float, float, float]:
    denom = max(1, count - 1)
    u = max(0.0, min(1.0, index / denom))
    start = np.array([106, 198, 255], dtype=np.float64)
    end = np.array([3, 20, 74], dtype=np.float64)
    rgb = start + (end - start) * u
    return tuple((rgb / 255.0).tolist())


def rpy_deg_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def write_trajectory_plot(
    output_dir: Path,
    camera_rows: list[dict[str, object]],
    tag_rows: list[dict[str, object]],
    pool_cfg: dict[str, object],
    tag_size_m: float,
    plot_z_scale: float,
    anchor_tag_id: int,
) -> Path | None:
    if not camera_rows:
        (output_dir / "trajectory_plot.txt").write_text(
            "No optimized camera poses were available for plotting.\n",
            encoding="utf-8",
        )
        return None

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        (output_dir / "trajectory_plot.txt").write_text(
            "matplotlib is required to generate trajectory_plot.png.\n"
            "Install it with: python3 -m pip install matplotlib\n"
            f"Import error: {exc}\n",
            encoding="utf-8",
        )
        return None

    camera_xyz = np.asarray(
        [[row["x_m"], row["y_m"], row["z_m"]] for row in camera_rows],
        dtype=np.float64,
    )
    tag_xyz = np.asarray(
        [[row["x_m"], row["y_m"], row["z_m"]] for row in tag_rows],
        dtype=np.float64,
    )
    has_tag_update = np.asarray([bool(row["has_tag_update"]) for row in camera_rows])
    pool_cfg = normalize_pool_config(pool_cfg)
    if anchor_tag_id == 1:
        pool_cfg["tag1_position_m"] = [0.0, 0.0, 0.0]
    pool_geometry = compute_pool_geometry(pool_cfg)

    display_swap_xy = pool_geometry.long_axis == "x"

    def to_display_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if display_swap_xy:
            return np.column_stack((points[:, 1], points[:, 0], -points[:, 2] * plot_z_scale))
        return np.column_stack((points[:, 0], points[:, 1], -points[:, 2] * plot_z_scale))

    def to_display_point(point: np.ndarray | list[float]) -> np.ndarray:
        return to_display_points(np.asarray([point], dtype=np.float64))[0]

    camera_display_xyz = to_display_points(camera_xyz)
    tag_display_xyz = to_display_points(tag_xyz) if tag_xyz.size else tag_xyz
    pool_floor_display = to_display_points(pool_geometry.floor)
    pool_top_display = to_display_points(pool_geometry.top)
    display_axis_limits = {
        "x": pool_geometry.axis_limits["y"] if display_swap_xy else pool_geometry.axis_limits["x"],
        "y": pool_geometry.axis_limits["x"] if display_swap_xy else pool_geometry.axis_limits["y"],
        "z": (
            -pool_geometry.axis_limits["z"][1] * plot_z_scale,
            -pool_geometry.axis_limits["z"][0] * plot_z_scale,
        ),
    }
    all_points = np.vstack((pool_floor_display, pool_top_display))

    fig = plt.figure(figsize=(14, 6), dpi=160)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ax3d.add_collection3d(
            Poly3DCollection(
                [pool_top_display],
                facecolors="#6FB3D9",
                edgecolors="none",
                alpha=0.08,
            )
        )
    except Exception:
        pass
    closed_floor_3d = np.vstack((pool_floor_display, pool_floor_display[0]))
    closed_top_3d = np.vstack((pool_top_display, pool_top_display[0]))
    ax3d.plot(closed_floor_3d[:, 0], closed_floor_3d[:, 1], closed_floor_3d[:, 2], color="#315C78", linewidth=1.8)
    ax3d.plot(closed_top_3d[:, 0], closed_top_3d[:, 1], closed_top_3d[:, 2], color="#4A7A9C", linewidth=1.4, linestyle="--", alpha=0.78)
    for corner_index in range(4):
        ax3d.plot(
            [pool_floor_display[corner_index, 0], pool_top_display[corner_index, 0]],
            [pool_floor_display[corner_index, 1], pool_top_display[corner_index, 1]],
            [pool_floor_display[corner_index, 2], pool_top_display[corner_index, 2]],
            color="#4A7A9C",
            linewidth=0.9,
            alpha=0.58,
        )
    p0_3d, p1_3d, p2_3d, p3_3d = pool_floor_display
    for grid_index in range(1, 8):
        u = grid_index / 8.0
        a = p0_3d * (1.0 - u) + p1_3d * u
        b = p3_3d * (1.0 - u) + p2_3d * u
        c = p0_3d * (1.0 - u) + p3_3d * u
        d = p1_3d * (1.0 - u) + p2_3d * u
        ax3d.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#CBD5DD", linewidth=0.55)
        ax3d.plot([c[0], d[0]], [c[1], d[1]], [c[2], d[2]], color="#CBD5DD", linewidth=0.55)
    ax3d.plot([], [], [], color=trajectory_color_tuple(0, len(camera_xyz)), label="Camera trajectory")
    for index in range(1, len(camera_xyz)):
        ax3d.plot(
            camera_display_xyz[index - 1 : index + 1, 0],
            camera_display_xyz[index - 1 : index + 1, 1],
            camera_display_xyz[index - 1 : index + 1, 2],
            color=trajectory_color_tuple(index, len(camera_xyz)),
            linewidth=1.8,
            linestyle="-" if has_tag_update[index] else "--",
        )
    ax3d.scatter(*camera_display_xyz[0], color="#2ca02c", s=55, label="Start")
    ax3d.scatter(*camera_display_xyz[-1], color="#d62728", s=55, label="End")
    if np.any(~has_tag_update):
        predicted = camera_display_xyz[~has_tag_update]
        ax3d.scatter(
            predicted[:, 0],
            predicted[:, 1],
            predicted[:, 2],
            color="#888888",
            s=14,
            marker="x",
            label="No-tag prediction",
        )

    if tag_rows:
        cmap = plt.get_cmap("viridis")
        sorted_rows = sorted(tag_rows, key=lambda item: int(item["tag_id"]))
        for tag_index, row in enumerate(sorted_rows):
            color = cmap(tag_index / max(1, len(sorted_rows) - 1))
            marker_size = max(35.0, (tag_size_m * 700.0) ** 2 / 100.0)
            display_origin = to_display_point([row["x_m"], row["y_m"], row["z_m"]])
            ax3d.scatter(
                display_origin[0],
                display_origin[1],
                display_origin[2],
                color=color,
                s=marker_size,
                marker="s",
                edgecolor="#ffffff",
                linewidth=0.6,
                label="AprilTags" if tag_index == 0 else None,
            )
            ax3d.text(display_origin[0], display_origin[1], display_origin[2], f"  Tag {row['tag_id']}")
            origin = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=np.float64)
            rot = rpy_deg_to_matrix(row["roll_deg"], row["pitch_deg"], row["yaw_deg"])
            axis_len_tag = 0.10
            for col, color_axis in enumerate(("#D33F49", "#2E8B57", "#2F6DB3")):
                end = origin + rot[:, col] * axis_len_tag
                display_end = to_display_point(end)
                ax3d.plot(
                    [display_origin[0], display_end[0]],
                    [display_origin[1], display_end[1]],
                    [display_origin[2], display_end[2]],
                    color=color_axis,
                    linewidth=1.1,
                )

    anchor_estimate = None
    for row in tag_rows:
        if int(row["tag_id"]) == anchor_tag_id:
            anchor_estimate = [row["x_m"], row["y_m"], row["z_m"]]
            break
    anchor_origin_display = (
        to_display_point(anchor_estimate)
        if anchor_estimate is not None
        else to_display_point([0.0, 0.0, 0.0])
    )
    ax3d.scatter(
        *anchor_origin_display,
        color="#111111",
        s=46,
        marker="o",
        label=f"Tag {anchor_tag_id} world origin",
    )

    reference_origin = (
        to_display_point(anchor_estimate)
        if anchor_estimate is not None
        else anchor_origin_display
    )
    reference_origin = reference_origin + np.array([0.0, 0.0, 0.60 * plot_z_scale], dtype=np.float64)
    axis_len = max(0.25, min(0.70, float(np.max(np.ptp(all_points, axis=0))) * 0.14))
    ax3d.quiver(*reference_origin, axis_len, 0, 0, color="#D33F49", linewidth=2.2, arrow_length_ratio=0.15)
    ax3d.quiver(*reference_origin, 0, axis_len, 0, color="#2E8B57", linewidth=2.2, arrow_length_ratio=0.15)
    ax3d.quiver(*reference_origin, 0, 0, axis_len, color="#2F6DB3", linewidth=2.2, arrow_length_ratio=0.15)
    ax3d.text(reference_origin[0] + axis_len, reference_origin[1], reference_origin[2], "X", color="#D33F49")
    ax3d.text(reference_origin[0], reference_origin[1] + axis_len, reference_origin[2], "Y", color="#2E8B57")
    ax3d.text(reference_origin[0], reference_origin[1], reference_origin[2] + axis_len, "Z", color="#2F6DB3")
    ax3d.text(*reference_origin, f"Tag {anchor_tag_id} +0.60 m ref", color="#111111")
    ax3d.set_title("TagSLAM Camera Trajectory in Display Frame")
    ax3d.set_xlabel("X short edge (m)")
    ax3d.set_ylabel("Y long edge (m)")
    ax3d.set_zlabel(f"Z flipped (m, {plot_z_scale:.2g}x visual)")
    ax3d.set_xlim(*display_axis_limits["x"])
    ax3d.set_ylim(*display_axis_limits["y"])
    ax3d.set_zlim(*display_axis_limits["z"])
    ax3d.zaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value / max(plot_z_scale, 1e-9):g}")
    )
    try:
        ax3d.set_box_aspect(
            (
                display_axis_limits["x"][1] - display_axis_limits["x"][0],
                display_axis_limits["y"][1] - display_axis_limits["y"][0],
                display_axis_limits["z"][1] - display_axis_limits["z"][0],
            )
        )
    except Exception:
        pass
    ax3d.view_init(elev=30, azim=0)
    ax3d.legend(loc="best")

    floor_xy = pool_geometry.floor[:, [1, 0]]
    closed_floor_xy = np.vstack((floor_xy, floor_xy[0]))
    axxy.fill(
        closed_floor_xy[:, 0],
        closed_floor_xy[:, 1],
        color="#6FB3D9",
        alpha=0.08,
        label="Pool footprint",
    )
    axxy.plot(
        closed_floor_xy[:, 0],
        closed_floor_xy[:, 1],
        color="#315C78",
        linewidth=1.8,
    )
    p0, p1, p2, p3 = floor_xy
    for grid_index in range(1, 8):
        u = grid_index / 8.0
        a = p0 * (1.0 - u) + p1 * u
        b = p3 * (1.0 - u) + p2 * u
        c = p0 * (1.0 - u) + p3 * u
        d = p1 * (1.0 - u) + p2 * u
        axxy.plot([a[0], b[0]], [a[1], b[1]], color="#CBD5DD", linewidth=0.55)
        axxy.plot([c[0], d[0]], [c[1], d[1]], color="#CBD5DD", linewidth=0.55)

    for index in range(1, len(camera_xyz)):
        axxy.plot(
            camera_xyz[index - 1 : index + 1, 1],
            camera_xyz[index - 1 : index + 1, 0],
            color=trajectory_color_tuple(index, len(camera_xyz)),
            linewidth=1.8,
            linestyle="-" if has_tag_update[index] else "--",
        )
    axxy.scatter(camera_xyz[0, 1], camera_xyz[0, 0], color="#2ca02c", s=55, label="Start")
    axxy.scatter(camera_xyz[-1, 1], camera_xyz[-1, 0], color="#d62728", s=55, label="End")
    if np.any(~has_tag_update):
        predicted = camera_xyz[~has_tag_update]
        axxy.scatter(
            predicted[:, 1],
            predicted[:, 0],
            color="#888888",
            s=14,
            marker="x",
            label="No-tag prediction",
        )
    if tag_rows:
        axxy.scatter(tag_xyz[:, 1], tag_xyz[:, 0], color="#111111", s=70, marker="s")
        for row in tag_rows:
            axxy.annotate(
                f"Tag {row['tag_id']}",
                (row["y_m"], row["x_m"]),
                textcoords="offset points",
                xytext=(5, 5),
            )
    axxy.set_xlim(*pool_geometry.axis_limits["y"])
    axxy.set_ylim(*pool_geometry.axis_limits["x"])
    axxy.set_aspect("equal", adjustable="box")
    axxy.grid(True, alpha=0.22)
    axxy.set_title("Top View: X-Y Pool Footprint")
    axxy.set_xlabel("X short edge (m)")
    axxy.set_ylabel("Y long edge (m)")
    axxy.legend(loc="best")

    fig.tight_layout()
    plot_path = output_dir / "trajectory_plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def draw_filled_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(int(x1), image.shape[1] - 1))
    y1 = max(0, min(int(y1), image.shape[0] - 1))
    x2 = max(0, min(int(x2), image.shape[1]))
    y2 = max(0, min(int(y2), image.shape[0]))
    if x2 <= x1 or y2 <= y1:
        return
    if alpha >= 1.0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
        return
    roi = image[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def draw_text_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 5,
    alpha: float = 0.72,
) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(0, min(int(x), image.shape[1] - width - padding * 2 - 1))
    y = max(height + padding, min(int(y), image.shape[0] - baseline - padding - 1))
    draw_filled_rect(
        image,
        (x, y - height - padding),
        (x + width + padding * 2, y + baseline + padding),
        bg_color,
        alpha=alpha,
    )
    cv2.putText(
        image,
        text,
        (x + padding, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_observations(image: np.ndarray, observations: Iterable[TagObservation]) -> None:
    for obs in observations:
        points = np.round(obs.corners).astype(np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.drawMarker(
            image,
            obs.center,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = f"ID {obs.tag_id} z={obs.scaled_tvec_m[2]:.2f}m"
        origin = tuple((points[0] + np.array([0, -8])).tolist())
        draw_text_box(
            image,
            label,
            origin,
            scale=0.55,
            bg_color=(0, 80, 0),
            thickness=2,
            padding=4,
        )


def pose_text(pose: Pose3 | None, anchor_tag_id: int) -> str:
    if pose is None:
        return "Global pose: unavailable"
    x_m, y_m, z_m = pose_translation(pose)
    roll, pitch, yaw = np.degrees(pose_rpy(pose))
    return (
        f"Global camera pose from tag{anchor_tag_id}: "
        f"X {x_m:+.3f} Y {y_m:+.3f} Z {z_m:+.3f} m  "
        f"RPY {roll:+.1f} {pitch:+.1f} {yaw:+.1f} deg"
    )


def draw_overlay(
    image: np.ndarray,
    update: BackendUpdate,
    observations: list[TagObservation],
    fps_value: float,
    trajectory_active: bool = False,
    trajectory_samples: int = 0,
) -> None:
    draw_text_box(
        image,
        pose_text(update.camera_pose, update.anchor_tag_id),
        (12, 32),
        scale=0.68,
        bg_color=(15, 20, 26),
    )
    draw_text_box(
        image,
        (
            f"{len(observations)} detections / {update.used_observation_count} used | "
            f"{update.status} | "
            f"std={update.camera_position_std_cm:.1f}cm | " if update.camera_position_std_cm is not None
            else f"{len(observations)} detections / {update.used_observation_count} used | {update.status} | "
        )
        + f"{fps_value:.1f} FPS",
        (12, 66),
        scale=0.56,
        text_color=(210, 235, 255),
        bg_color=(15, 20, 26),
        thickness=2,
        padding=4,
    )

    if trajectory_active:
        draw_text_box(
            image,
            f"TRAJ REC {trajectory_samples}",
            (image.shape[1] - 180, 32),
            scale=0.62,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 180),
            thickness=2,
            padding=5,
            alpha=0.85,
        )

    if update.tag_poses:
        lines = []
        for tag_id, pose in sorted(update.tag_poses.items())[:8]:
            t = pose_translation(pose)
            lines.append(f"L{tag_id}: {t[0]:+.2f} {t[1]:+.2f} {t[2]:+.2f}")
        y = image.shape[0] - 24 - 24 * (len(lines) - 1)
        for line in lines:
            draw_text_box(
                image,
                line,
                (12, y),
                scale=0.52,
                text_color=(230, 255, 230),
                bg_color=(15, 20, 26),
                thickness=1,
                padding=4,
            )
            y += 24


def get_display_scale(frame_shape: tuple[int, int, int], max_width: int) -> float:
    width = frame_shape[1]
    if max_width == 0 or width <= max_width:
        return 1.0
    return max_width / width


def resize_for_display(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def print_backend_update(update: BackendUpdate, observations: list[TagObservation]) -> None:
    tag_ids = ",".join(str(obs.tag_id) for obs in observations) if observations else "none"
    if update.camera_pose is None:
        print(f"{update.status}; detections={tag_ids}", flush=True)
        return
    translation = pose_translation(update.camera_pose)
    rpy_deg = np.degrees(pose_rpy(update.camera_pose))
    std_text = (
        f" std={update.camera_position_std_cm:.1f}cm"
        if update.camera_position_std_cm is not None
        else ""
    )
    print(
        f"X{update.camera_index} "
        f"pos=({translation[0]:+.3f},{translation[1]:+.3f},{translation[2]:+.3f})m "
        f"rpy=({rpy_deg[0]:+.1f},{rpy_deg[1]:+.1f},{rpy_deg[2]:+.1f})deg "
        f"detections={tag_ids} used={update.used_observation_count}{std_text}",
        flush=True,
    )


# =============================================================================
# Experiment comparison visualizations (new — do not touch existing writers)
# =============================================================================

def write_overlay_topdown_plot(
    path: Path,
    gantry_traj_mm: list[dict],
    camera_traj_rows: list[dict],
    tag_pose_rows: list[dict],
    anchor_id: int,
    T_gantry_camera: "np.ndarray | None",
    pool_cfg: dict,
    gantry_anchor_offset_mm: "list[float] | None" = None,
    run_name: str = "",
) -> "Path | None":
    """Top-down (X, Y) matplotlib PNG that overlays:
    - Pool outline (light gray dashed, from pool_cfg)
    - AprilTag positions (numbered markers, anchor highlighted in cyan)
    - Estimated camera trajectory (viridis colormap, time-encoded)
    - Gantry ground-truth trajectory (orange dashed)

    Frame alignment:
    - If *gantry_anchor_offset_mm* is provided, subtract it from gantry XY
      before plotting so both trajectories share the same origin.
    - Otherwise a first-sample offset is applied (approximate alignment).

    Returns the written path or None on error.
    """
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.patches as mpatches
    except ImportError as exc:
        path.with_suffix(".txt").write_text(
            f"matplotlib required for overlay plot.\n{exc}\n", encoding="utf-8"
        )
        return None

    if not camera_traj_rows:
        return None

    # ── camera trajectory (meters, SLAM world frame) ─────────────────────────
    cam_xy = []
    for row in camera_traj_rows:
        try:
            cam_xy.append((float(row["x_m"]), float(row["y_m"])))
        except (KeyError, ValueError):
            pass
    if not cam_xy:
        return None

    # ── gantry trajectory (mm → m, aligned) ─────────────────────────────────
    gantry_xy_m = []
    for row in gantry_traj_mm:
        try:
            gx = float(row.get("x_mm", 0.0)) / 1000.0
            gy = float(row.get("y_mm", 0.0)) / 1000.0
            if T_gantry_camera is not None:
                # Add camera-body offset in gantry frame (same as render_topdown_panel)
                gx += float(T_gantry_camera[0, 3])
                gy += float(T_gantry_camera[1, 3])
            gantry_xy_m.append((gx, gy))
        except (KeyError, ValueError):
            pass

    # Alignment: subtract anchor offset or first-sample align
    if gantry_anchor_offset_mm is not None and len(gantry_anchor_offset_mm) >= 2:
        ox = gantry_anchor_offset_mm[0] / 1000.0
        oy = gantry_anchor_offset_mm[1] / 1000.0
        gantry_xy_m = [(x - ox, y - oy) for x, y in gantry_xy_m]
        aligned_note = "gantry_anchor_offset_mm"
    elif gantry_xy_m and cam_xy:
        ox = gantry_xy_m[0][0] - cam_xy[0][0]
        oy = gantry_xy_m[0][1] - cam_xy[0][1]
        gantry_xy_m = [(x - ox, y - oy) for x, y in gantry_xy_m]
        aligned_note = "first-sample-zeroed"
    else:
        aligned_note = "none"

    # ── tag positions ─────────────────────────────────────────────────────────
    tag_xy = {}
    for row in tag_pose_rows:
        try:
            tid = int(row["tag_id"])
            tag_xy[tid] = (float(row["x_m"]), float(row["y_m"]))
        except (KeyError, ValueError):
            pass

    # ── pool outline ─────────────────────────────────────────────────────────
    pool_cfg = normalize_pool_config(pool_cfg)
    pool_geom = compute_pool_geometry(pool_cfg)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    ax.set_facecolor("#0f0f12")
    fig.patch.set_facecolor("#0f0f12")

    # Pool floor outline
    floor = pool_geom.floor  # shape (4, 3)
    closed = np.vstack([floor, floor[0]])
    ax.plot(closed[:, 0], closed[:, 1],
            color="#444", linestyle="--", linewidth=1.2, label="Pool outline")

    # Tags
    for tid, (tx, ty) in tag_xy.items():
        color = "#00e5e5" if tid == anchor_id else "#aaa"
        ax.scatter(tx, ty, color=color, s=40, zorder=5)
        ax.annotate(str(tid), (tx, ty), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color=color)

    # Camera trajectory (viridis)
    n = len(cam_xy)
    if n >= 2:
        cmap = cm.get_cmap("viridis")
        for i in range(n - 1):
            t_frac = i / max(n - 1, 1)
            ax.plot([cam_xy[i][0], cam_xy[i + 1][0]],
                    [cam_xy[i][1], cam_xy[i + 1][1]],
                    color=cmap(t_frac), linewidth=1.5, alpha=0.9)
        ax.scatter(*cam_xy[0],  color=cmap(0.0),  s=60, zorder=6, marker="^")
        ax.scatter(*cam_xy[-1], color=cmap(1.0),  s=60, zorder=6, marker="s")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("time", color="#ccc", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ccc", fontsize=8)

    # Gantry trajectory
    if len(gantry_xy_m) >= 2:
        gx_arr = [p[0] for p in gantry_xy_m]
        gy_arr = [p[1] for p in gantry_xy_m]
        ax.plot(gx_arr, gy_arr, color="#ff7700", linestyle="--",
                linewidth=1.8, alpha=0.85, zorder=4)
        ax.scatter(gx_arr[0],  gy_arr[0],  color="#ff7700", s=60, zorder=6, marker="^")
        ax.scatter(gx_arr[-1], gy_arr[-1], color="#ff7700", s=60, zorder=6, marker="s")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#555", label="Pool"),
        plt.Line2D([0], [0], color=cm.get_cmap("viridis")(0.5), linewidth=2, label="Camera (viridis)"),
        plt.Line2D([0], [0], color="#ff7700", linestyle="--", linewidth=2, label="Gantry GT"),
    ]
    ax.legend(handles=legend_items, facecolor="#222", edgecolor="#555",
              labelcolor="#ddd", fontsize=8, loc="upper left")

    title = run_name or "Experiment"
    ax.set_title(f"{title} — Top-down (X, Y) [m]  align={aligned_note}",
                 color="#ddd", fontsize=11)
    ax.set_xlabel("X [m]", color="#ccc")
    ax.set_ylabel("Y [m]", color="#ccc")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#444")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_pose_velocity_acceleration_plot(
    path: Path,
    gantry_csv: Path,
    camera_csv: Path,
) -> "Path | None":
    """3×3 matplotlib PNG: rows = Pose / Velocity / Acceleration, cols = X / Y / Z.

    Overlays gantry ground truth (solid blue) and AprilTag estimate (dashed red).
    Velocity and acceleration for the camera trajectory are derived using the
    same 5-sample SMA central-difference as GantryTelemetryLogger.
    Per-axis pose RMSE (mm) annotated in subplot titles.
    """
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        path.with_suffix(".txt").write_text(
            f"matplotlib required.\n{exc}\n", encoding="utf-8"
        )
        return None

    import csv as _csv

    def _load_csv(p: Path) -> list[dict[str, str]]:
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    g_rows = _load_csv(gantry_csv)
    c_rows = _load_csv(camera_csv)

    if not g_rows or not c_rows:
        return None

    # ── extract gantry arrays ─────────────────────────────────────────────────
    def _col(rows: list[dict], key: str) -> "np.ndarray":
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (KeyError, ValueError):
                vals.append(float("nan"))
        return np.array(vals, dtype=np.float64)

    g_t   = _col(g_rows, "elapsed_s")
    g_x   = _col(g_rows, "x_mm")
    g_y   = _col(g_rows, "y_mm")
    g_z   = _col(g_rows, "z_mm")
    g_vx  = _col(g_rows, "vx_mm_s")
    g_vy  = _col(g_rows, "vy_mm_s")
    g_vz  = _col(g_rows, "vz_mm_s")
    g_ax  = _col(g_rows, "ax_mm_s2")
    g_ay  = _col(g_rows, "ay_mm_s2")
    g_az  = _col(g_rows, "az_mm_s2")

    # ── extract camera arrays (m → mm; derive vel/acc) ────────────────────────
    c_t   = _col(c_rows, "elapsed_s")
    c_x_m = _col(c_rows, "x_m")
    c_y_m = _col(c_rows, "y_m")
    c_z_m = _col(c_rows, "z_m")
    c_x   = c_x_m * 1000.0
    c_y   = c_y_m * 1000.0
    c_z   = c_z_m * 1000.0

    def _sma_deriv(t: "np.ndarray", x: "np.ndarray", window: int = 5) -> "np.ndarray":
        """5-sample SMA central difference — matches GantryTelemetryLogger convention."""
        n = len(t)
        dx = np.full(n, np.nan, dtype=np.float64)
        half = window // 2
        for i in range(half, n - half):
            t_front = np.nanmean(t[i - half: i])
            t_back  = np.nanmean(t[i + 1: i + 1 + half])
            x_front = np.nanmean(x[i - half: i])
            x_back  = np.nanmean(x[i + 1: i + 1 + half])
            dt = t_back - t_front
            if dt > 0:
                dx[i] = (x_back - x_front) / dt
        return dx

    # velocity in mm/s, acceleration in mm/s²
    c_vx = _sma_deriv(c_t, c_x)
    c_vy = _sma_deriv(c_t, c_y)
    c_vz = _sma_deriv(c_t, c_z)
    c_ax = _sma_deriv(c_t, c_vx)
    c_ay = _sma_deriv(c_t, c_vy)
    c_az = _sma_deriv(c_t, c_vz)

    # cm/s and cm/s² for display (match panel convention)
    def _mm_to_cm(a: "np.ndarray") -> "np.ndarray":
        return a / 10.0

    # ── RMSE ─────────────────────────────────────────────────────────────────
    def _rmse_mm(gantry_arr: "np.ndarray", g_t_arr: "np.ndarray",
                 cam_arr: "np.ndarray", c_t_arr: "np.ndarray") -> float:
        """Interpolate camera onto gantry time grid; compute RMSE in mm."""
        if len(c_t_arr) < 2:
            return float("nan")
        finite_g = np.isfinite(gantry_arr) & np.isfinite(g_t_arr)
        finite_c = np.isfinite(cam_arr)    & np.isfinite(c_t_arr)
        if not finite_g.any() or not finite_c.any():
            return float("nan")
        c_interp = np.interp(
            g_t_arr[finite_g], c_t_arr[finite_c], cam_arr[finite_c]
        )
        diff = gantry_arr[finite_g] - c_interp
        return float(np.sqrt(np.nanmean(diff ** 2)))

    rmse_x = _rmse_mm(g_x, g_t, c_x, c_t)
    rmse_y = _rmse_mm(g_y, g_t, c_y, c_t)
    rmse_z = _rmse_mm(g_z, g_t, c_z, c_t)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), dpi=130, sharey="row")
    fig.patch.set_facecolor("#0f0f12")
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor("#1a1a1d")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

    rows_data = [
        # (row_label, gantry_arrs, cam_arrs, y_label, scale)
        ("Pose",         [g_x, g_y, g_z],  [c_x, c_y, c_z],  "mm",     1.0),
        ("Velocity",     [g_vx, g_vy, g_vz], [c_vx, c_vy, c_vz], "cm/s", 0.1),
        ("Acceleration", [g_ax, g_ay, g_az], [c_ax, c_ay, c_az], "cm/s²", 0.1),
    ]
    col_labels = ["X", "Y", "Z"]
    rmse_vals = [rmse_x, rmse_y, rmse_z]

    for row_idx, (row_lbl, g_arrs, c_arrs, y_lbl, scale) in enumerate(rows_data):
        for col_idx, (g_arr, c_arr, col_lbl) in enumerate(zip(g_arrs, c_arrs, col_labels)):
            ax = axes[row_idx][col_idx]
            ax.plot(g_t, g_arr * scale, color="#4ea1ff", linewidth=1.2,
                    label="Gantry GT", alpha=0.9)
            ax.plot(c_t, c_arr * scale, color="#ff5555", linestyle="--",
                    linewidth=1.0, label="AprilTag", alpha=0.85)

            if row_idx == 0:  # pose row → annotate RMSE
                rmse = rmse_vals[col_idx]
                rmse_str = f"{rmse:.1f} mm" if np.isfinite(rmse) else "N/A"
                title = f"{col_lbl}  [RMSE={rmse_str}]"
            else:
                title = col_lbl
            ax.set_title(title, color="#ddd", fontsize=9, pad=4)
            ax.set_ylabel(y_lbl, color="#aaa", fontsize=8)
            if row_idx == 2:
                ax.set_xlabel("elapsed [s]", color="#aaa", fontsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(facecolor="#222", edgecolor="#555",
                          labelcolor="#ddd", fontsize=7, loc="upper left")

    fig.text(0.02, 0.5, "Pose / Velocity / Acceleration",
             va="center", rotation="vertical", color="#ccc", fontsize=10)
    fig.suptitle(
        "Gantry GT vs AprilTag Estimate\n"
        "(AprilTag vel/acc: 5-sample SMA + central difference, matching gantry-logger convention)",
        color="#ddd", fontsize=10,
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.93])
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_gantry_only_plot(
    path: Path,
    gantry_csv: Path,
) -> "Path | None":
    """3×3 matplotlib PNG of gantry-only ground truth: rows = Pose / Velocity /
    Acceleration, cols = X / Y / Z. No camera overlay.

    Used by the Experiment runner in 'gantry_only' camera mode where the
    fisheye pipeline is bypassed entirely. Reads gantry_telemetry.csv columns
    written by GantryTelemetryLogger (mm / mm·s / mm·s²); displays in cm/s and
    cm/s² to match the live panel convention.
    """
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        path.with_suffix(".txt").write_text(
            f"matplotlib required.\n{exc}\n", encoding="utf-8"
        )
        return None

    import csv as _csv

    def _load_csv(p: Path) -> list[dict[str, str]]:
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    g_rows = _load_csv(gantry_csv)
    if not g_rows:
        return None

    def _col(rows: list[dict], key: str) -> "np.ndarray":
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (KeyError, ValueError):
                vals.append(float("nan"))
        return np.array(vals, dtype=np.float64)

    g_t   = _col(g_rows, "elapsed_s")
    g_pos = [_col(g_rows, "x_mm"),     _col(g_rows, "y_mm"),     _col(g_rows, "z_mm")]
    g_vel = [_col(g_rows, "vx_mm_s"),  _col(g_rows, "vy_mm_s"),  _col(g_rows, "vz_mm_s")]
    g_acc = [_col(g_rows, "ax_mm_s2"), _col(g_rows, "ay_mm_s2"), _col(g_rows, "az_mm_s2")]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), dpi=130, sharey="row")
    fig.patch.set_facecolor("#0f0f12")
    for ax in axes.flatten():
        ax.set_facecolor("#1a1a1d")
        ax.tick_params(colors="#aaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

    rows_data = [
        ("Pose",         g_pos, "mm",     1.0),
        ("Velocity",     g_vel, "cm/s",   0.1),
        ("Acceleration", g_acc, "cm/s²",  0.1),
    ]
    col_labels = ["X", "Y", "Z"]
    for row_idx, (_row_lbl, arrs, y_lbl, scale) in enumerate(rows_data):
        for col_idx, (arr, col_lbl) in enumerate(zip(arrs, col_labels)):
            ax = axes[row_idx][col_idx]
            ax.plot(g_t, arr * scale, color="#4ea1ff", linewidth=1.2,
                    label="Gantry GT", alpha=0.9)
            ax.set_title(col_lbl, color="#ddd", fontsize=9, pad=4)
            ax.set_ylabel(y_lbl, color="#aaa", fontsize=8)
            if row_idx == 2:
                ax.set_xlabel("elapsed [s]", color="#aaa", fontsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(facecolor="#222", edgecolor="#555",
                          labelcolor="#ddd", fontsize=7, loc="upper left")

    fig.text(0.02, 0.5, "Pose / Velocity / Acceleration",
             va="center", rotation="vertical", color="#ccc", fontsize=10)
    fig.suptitle(
        "Gantry-only Ground Truth (camera disabled)",
        color="#ddd", fontsize=10,
    )
    fig.tight_layout(rect=[0.03, 0, 1, 0.93])
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_pose_velocity_acceleration_html(
    path: Path,
    gantry_csv: Path,
    camera_csv: Path,
) -> Path:
    """Interactive HTML with 9 subplots (3×3: pose/vel/acc × X/Y/Z).

    Features:
    - Plain JS + Canvas — no external libraries.
    - Synchronized hover: hovering one panel highlights the same timestamp on all 9.
    - Toggle buttons to show/hide either curve (gantry or AprilTag).
    """
    import csv as _csv
    import json as _json

    def _load_csv(p: Path) -> list[dict[str, str]]:
        if not p.exists():
            return []
        with p.open(newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    g_rows = _load_csv(gantry_csv)
    c_rows = _load_csv(camera_csv)

    def _col(rows: list[dict], key: str) -> list[float]:
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (KeyError, ValueError):
                out.append(float("nan"))
        return out

    def _sma_deriv(t: list[float], x: list[float], window: int = 5) -> list[float]:
        n = len(t)
        dx = [float("nan")] * n
        half = window // 2
        for i in range(half, n - half):
            t_front = sum(t[max(0, i - half): i]) / half if half > 0 else t[i]
            t_back  = sum(t[i + 1: i + 1 + half]) / half if half > 0 else t[i]
            x_front = sum(x[max(0, i - half): i]) / half if half > 0 else x[i]
            x_back  = sum(x[i + 1: i + 1 + half]) / half if half > 0 else x[i]
            dt = t_back - t_front
            if dt > 0:
                dx[i] = (x_back - x_front) / dt
        return dx

    g_t  = _col(g_rows, "elapsed_s")
    g_x  = _col(g_rows, "x_mm")
    g_y  = _col(g_rows, "y_mm")
    g_z  = _col(g_rows, "z_mm")
    g_vx = _col(g_rows, "vx_mm_s")
    g_vy = _col(g_rows, "vy_mm_s")
    g_vz = _col(g_rows, "vz_mm_s")
    g_ax = _col(g_rows, "ax_mm_s2")
    g_ay = _col(g_rows, "ay_mm_s2")
    g_az = _col(g_rows, "az_mm_s2")

    c_t_raw  = _col(c_rows, "elapsed_s")
    c_x_mm = [v * 1000.0 for v in _col(c_rows, "x_m")]
    c_y_mm = [v * 1000.0 for v in _col(c_rows, "y_m")]
    c_z_mm = [v * 1000.0 for v in _col(c_rows, "z_m")]
    c_vx = _sma_deriv(c_t_raw, c_x_mm)
    c_vy = _sma_deriv(c_t_raw, c_y_mm)
    c_vz = _sma_deriv(c_t_raw, c_z_mm)
    c_ax = _sma_deriv(c_t_raw, c_vx)
    c_ay = _sma_deriv(c_t_raw, c_vy)
    c_az = _sma_deriv(c_t_raw, c_vz)

    def _clean(vals: list[float]) -> list[float | None]:
        return [None if (v != v) else v for v in vals]  # nan → None for JSON

    data_json = _json.dumps({
        "gantry": {
            "t": _clean(g_t),
            "pose": [_clean(g_x), _clean(g_y), _clean(g_z)],
            "vel":  [_clean(g_vx), _clean(g_vy), _clean(g_vz)],
            "acc":  [_clean(g_ax), _clean(g_ay), _clean(g_az)],
        },
        "camera": {
            "t": _clean(c_t_raw),
            "pose": [_clean(c_x_mm), _clean(c_y_mm), _clean(c_z_mm)],
            "vel":  [_clean(c_vx), _clean(c_vy), _clean(c_vz)],
            "acc":  [_clean(c_ax), _clean(c_ay), _clean(c_az)],
        },
        "labels": {
            "rows": ["Pose (mm)", "Velocity (cm/s)", "Acceleration (cm/s²)"],
            "cols": ["X", "Y", "Z"],
            "gantry_color": "#4ea1ff",
            "camera_color": "#ff5555",
        },
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pose / Velocity / Acceleration — Experiment Comparison</title>
<style>
  body {{ background:#0f0f12; color:#ddd; font-family:sans-serif; margin:10px; }}
  h2 {{ color:#4ea1ff; }}
  .controls {{ margin-bottom:10px; }}
  button {{ background:#2a2a2e; color:#ddd; border:1px solid #444; border-radius:4px;
            padding:5px 14px; cursor:pointer; margin-right:6px; font-size:13px; }}
  button.active {{ background:#1a73e8; border-color:#1a73e8; }}
  .grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
  .cell {{ position:relative; }}
  canvas {{ background:#1a1a1d; border:1px solid #333; border-radius:4px; display:block; width:100%; }}
  .row-lbl {{ grid-column:1/-1; color:#888; font-size:11px; margin:4px 0 0 0; padding-left:4px;
              border-left:3px solid #333; }}
  .tooltip {{ position:absolute; pointer-events:none; background:#222; border:1px solid #555;
              padding:4px 8px; font-size:11px; border-radius:4px; white-space:nowrap;
              display:none; z-index:10; }}
</style>
</head>
<body>
<h2>Experiment: Pose / Velocity / Acceleration Comparison</h2>
<p style="font-size:11px;color:#888;">
  AprilTag vel/acc derived via 5-sample SMA + central difference, matching gantry-logger convention.<br>
  Hover any panel to synchronize the crosshair across all 9.
</p>
<div class="controls">
  <button id="btn_gantry" class="active" onclick="toggleSeries('gantry')">Gantry GT</button>
  <button id="btn_camera" class="active" onclick="toggleSeries('camera')">AprilTag</button>
</div>
<div class="grid" id="grid"></div>
<div class="tooltip" id="tooltip"></div>

<script>
const RAW = {data_json};
const SCALES = [1, 0.1, 0.1];  // mm→mm, mm/s→cm/s, mm/s²→cm/s²
const SHOW = {{gantry:true, camera:true}};

const canvases = [];
const contexts = [];
const chartData = [];  // per-cell: {{minT,maxT,minY,maxY,gantry,camera}}

function toggleSeries(name) {{
  SHOW[name] = !SHOW[name];
  document.getElementById('btn_' + name).className = SHOW[name] ? 'active' : '';
  renderAll();
}}

function buildGrid() {{
  const grid = document.getElementById('grid');
  const rowNames = RAW.labels.rows;
  const colNames = RAW.labels.cols;
  for (let r = 0; r < 3; r++) {{
    const lbl = document.createElement('div');
    lbl.className = 'row-lbl';
    lbl.textContent = rowNames[r];
    grid.appendChild(lbl);
    for (let c = 0; c < 3; c++) {{
      const cell = document.createElement('div');
      cell.className = 'cell';
      const canvas = document.createElement('canvas');
      canvas.width = 420; canvas.height = 180;
      cell.appendChild(canvas);
      grid.appendChild(cell);
      canvases.push(canvas);
      contexts.push(canvas.getContext('2d'));

      const idx = r * 3 + c;
      const sc = SCALES[r];
      const gt = (RAW.gantry[['pose','vel','acc'][r]][c] || []).map((v,i) => [RAW.gantry.t[i], v === null ? null : v * sc]);
      const cm_ = (RAW.camera[['pose','vel','acc'][r]][c] || []).map((v,i) => [RAW.camera.t[i], v === null ? null : v * sc]);

      const allY = [...gt.map(p=>p[1]), ...cm_.map(p=>p[1])].filter(v=>v!==null);
      const allT = [...gt.map(p=>p[0]), ...cm_.map(p=>p[0])].filter(v=>v!==null);
      chartData.push({{
        label: colNames[c],
        rowLabel: rowNames[r],
        gantry: gt, camera: cm_,
        minT: allT.length ? Math.min(...allT) : 0,
        maxT: allT.length ? Math.max(...allT) : 1,
        minY: allY.length ? Math.min(...allY) : -1,
        maxY: allY.length ? Math.max(...allY) : 1,
      }});

      canvas.addEventListener('mousemove', e => onHover(e, canvas, idx));
      canvas.addEventListener('mouseleave', () => {{
        renderAll();
        document.getElementById('tooltip').style.display = 'none';
      }});
    }}
  }}
}}

function toPixel(val, min, max, pxMin, pxMax) {{
  if (max === min) return (pxMin + pxMax) / 2;
  return pxMin + (val - min) / (max - min) * (pxMax - pxMin);
}}

function drawChart(ctx, data, crossT) {{
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const L=40, R=8, T=20, B=28;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#1a1a1d'; ctx.fillRect(0,0,W,H);

  const pxL=L, pxR=W-R, pxT=T, pxB=H-B;
  const {{minT,maxT,minY,maxY,label,gantry,camera}} = data;
  const ySpan = maxY-minY || 1;
  const yPad  = ySpan*0.08;
  const yLo   = minY - yPad, yHi = maxY + yPad;

  // Axes
  ctx.strokeStyle='#333'; ctx.lineWidth=1;
  ctx.strokeRect(pxL, pxT, pxR-pxL, pxB-pxT);

  // Grid lines (3 horizontal)
  ctx.setLineDash([2,4]);
  for (let k=0; k<=3; k++) {{
    const yv = yLo + (yHi - yLo) * k / 3;
    const py = toPixel(yv, yHi, yLo, pxT, pxB);
    ctx.strokeStyle='#2a2a2a'; ctx.beginPath();
    ctx.moveTo(pxL, py); ctx.lineTo(pxR, py); ctx.stroke();
    ctx.fillStyle='#777'; ctx.font='9px sans-serif'; ctx.fillText(yv.toFixed(1), 2, py+3);
  }}
  ctx.setLineDash([]);

  // Title
  ctx.fillStyle='#bbb'; ctx.font='bold 10px sans-serif';
  ctx.fillText(label, pxL+4, T-6);

  function drawSeries(pts, color) {{
    if (!pts || pts.length < 2) return;
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=1.4;
    let started = false;
    for (const [t,y] of pts) {{
      if (t === null || y === null) {{ started=false; continue; }}
      const px = toPixel(t, minT, maxT, pxL, pxR);
      const py = toPixel(y, yHi, yLo, pxT, pxB);
      if (!started) {{ ctx.moveTo(px,py); started=true; }} else ctx.lineTo(px,py);
    }}
    ctx.stroke();
  }}

  if (SHOW.gantry) drawSeries(gantry, RAW.labels.gantry_color);
  if (SHOW.camera) drawSeries(camera, RAW.labels.camera_color);

  // Crosshair
  if (crossT !== null) {{
    const px = toPixel(crossT, minT, maxT, pxL, pxR);
    ctx.setLineDash([3,3]); ctx.strokeStyle='rgba(255,255,255,0.4)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(px,pxT); ctx.lineTo(px,pxB); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function renderAll(crossT) {{
  chartData.forEach((data, idx) => drawChart(contexts[idx], data, crossT||null));
}}

function onHover(e, canvas, idx) {{
  const rect = canvas.getBoundingClientRect();
  const L=40, R=8;
  const W=canvas.width, pxL=L, pxR=W-R;
  const rawX = e.clientX - rect.left;
  const frac = (rawX * (W / rect.width) - pxL) / (pxR - pxL);
  const data = chartData[idx];
  const t = data.minT + frac * (data.maxT - data.minT);
  renderAll(t);

  // Tooltip
  const tip = document.getElementById('tooltip');
  // Interpolate gantry value
  function interpVal(pts, tq) {{
    if (!pts || pts.length < 2) return null;
    for (let i=1; i<pts.length; i++) {{
      const [t0,y0]=pts[i-1], [t1,y1]=pts[i];
      if (t0===null||t1===null||y0===null||y1===null) continue;
      if (tq>=t0 && tq<=t1) {{
        return y0 + (y1-y0)*(tq-t0)/(t1-t0);
      }}
    }}
    return null;
  }}
  const gv = interpVal(data.gantry, t);
  const cv = interpVal(data.camera, t);
  const fmt = v => v===null ? 'N/A' : v.toFixed(2);
  tip.innerHTML = `t=${{t.toFixed(2)}}s | Gantry: ${{fmt(gv)}} | AprilTag: ${{fmt(cv)}}`;
  tip.style.display = 'block';
  tip.style.left = (e.pageX + 12) + 'px';
  tip.style.top  = (e.pageY - 20) + 'px';
}}

buildGrid();
renderAll();
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


# =============================================================================
# Unified experiment dashboard (single self-contained HTML, three tabs)
# =============================================================================
_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment Dashboard</title>
<style>
  :root { --bg:#0f0f12; --panel:#1a1a1d; --card:#232327; --line:#33333a;
          --text:#e6e6e6; --muted:#9aa3ad; --accent:#4ea1ff;
          --gantry:#ff8c42; --gt:#4ea1ff; --est:#ff5d5d; --cyan:#26d0e0; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font-family: Arial, Helvetica, sans-serif; }
  body { display:flex; flex-direction:column; overflow:hidden; }
  #header { padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
  #header h1 { margin:0 0 4px 0; font-size:16px; }
  #meta { font-size:12px; color:var(--muted); line-height:1.6; }
  #meta b { color:var(--text); font-weight:600; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
           font-weight:700; margin-left:8px; }
  .badge-gantry { background:#5a3a18; color:#ffb774; border:1px solid #ff8c42; }
  #tabs { display:flex; gap:2px; padding:8px 16px 0 16px; background:var(--panel);
          border-bottom:1px solid var(--line); }
  .tabbtn { padding:8px 18px; border:1px solid var(--line); border-bottom:none;
            border-radius:6px 6px 0 0; background:var(--card); color:var(--muted);
            cursor:pointer; font-size:13px; font-weight:600; }
  .tabbtn.active { background:var(--bg); color:var(--accent); border-color:var(--accent); }
  #content { flex:1; min-height:0; position:relative; }
  .tabpane { position:absolute; inset:0; display:none; }
  .tabpane.active { display:block; }
  .toolbar { padding:8px 16px; display:flex; flex-wrap:wrap; gap:14px; align-items:center;
             font-size:12px; color:var(--muted); border-bottom:1px solid var(--line); }
  .toolbar label { cursor:pointer; user-select:none; }
  .toolbar input { vertical-align:middle; margin-right:4px; }
  .hint { color:#6b7682; font-size:11px; }
  /* Trajectory tab layout */
  /* 3D-viewer mode: the rich interactive viewer fills the pane via an iframe,
     and the classic 2D toolbar/body are hidden. */
  #trajFrame { position:absolute; inset:0; width:100%; height:100%; border:0;
               display:none; background:#f4f6f8; }
  #trajPane.viewer3d #trajToolbar { display:none; }
  #trajPane.viewer3d .body { display:none; }
  #trajPane.viewer3d #trajFrame { display:block; }
  #trajPane .body { position:absolute; top:42px; left:0; right:0; bottom:0;
                    display:grid; grid-template-columns: minmax(0,1fr) 320px; }
  #trajLeft { position:relative; min-width:0; display:flex; flex-direction:column; }
  #trajCanvasWrap { flex:1; min-height:0; position:relative; }
  #trajCanvas { width:100%; height:100%; display:block; cursor:crosshair; }
  #sliderRow { padding:8px 12px; display:flex; gap:10px; align-items:center;
               border-top:1px solid var(--line); background:var(--panel); }
  #slider { flex:1; }
  #sliderLabel { font-size:12px; color:var(--muted); white-space:nowrap; min-width:90px; }
  #zedCard { border-left:1px solid var(--line); background:var(--panel); padding:10px;
             display:flex; flex-direction:column; gap:8px; min-width:0; }
  #zedCard .title { font-size:13px; font-weight:700; }
  #zedImgWrap { flex:1; min-height:0; display:grid; place-items:center; background:#0a0d11;
                border:1px solid var(--line); }
  #zedImg { max-width:100%; max-height:100%; object-fit:contain; display:none; }
  #zedPlaceholder { color:var(--muted); font-size:12px; text-align:center; padding:10px; }
  #zedMeta { font-size:11px; color:var(--muted); line-height:1.5; }
  /* time-series tabs */
  .seriesBody { position:absolute; top:42px; left:0; right:0; bottom:0;
                display:flex; flex-direction:column; }
  .seriesCanvasWrap { flex:1; min-height:0; position:relative; }
  .seriesCanvasWrap canvas { width:100%; height:100%; display:block; cursor:crosshair; }
  .tooltip { position:fixed; pointer-events:none; background:rgba(20,24,30,0.95);
             border:1px solid var(--accent); border-radius:5px; padding:7px 9px;
             font-size:12px; line-height:1.5; color:var(--text); display:none;
             white-space:pre; z-index:50; font-family: "DejaVu Sans Mono", monospace; }
</style>
</head>
<body>
  <div id="header">
    <h1 id="runName">Run</h1>
    <div id="meta"></div>
  </div>
  <div id="tabs">
    <div class="tabbtn active" data-tab="trajectory" onclick="switchTab('trajectory')">Trajectory</div>
    <div class="tabbtn" data-tab="velocity" onclick="switchTab('velocity')">Velocity</div>
  </div>
  <div id="content">
    <!-- Trajectory -->
    <div class="tabpane active" id="trajPane">
      <!-- 3D interactive viewer (camera + gantry + tags + pool + play/slider on
           top); injected via srcdoc when DASH.traj_viewer_html is present. -->
      <iframe id="trajFrame" title="Interactive 3D trajectory viewer"></iframe>
      <!-- Classic 2D top-down fallback (gantry-only runs / viewer build failed). -->
      <div class="toolbar" id="trajToolbar">
        <label><input type="checkbox" id="tgCam" checked> Camera trajectory</label>
        <label><input type="checkbox" id="tgGantry" checked> Gantry GT</label>
        <label><input type="checkbox" id="tgTags" checked> Tags</label>
        <label><input type="checkbox" id="tgPool" checked> Pool outline</label>
        <label><input type="checkbox" id="tgCursor" checked> Time-cursor markers</label>
        <span class="hint">top-down (X,Y) · hover for nearest-sample readout</span>
      </div>
      <div class="body">
        <div id="trajLeft">
          <div id="trajCanvasWrap"><canvas id="trajCanvas"></canvas></div>
          <div id="sliderRow">
            <span id="sliderLabel">t = 0.00 s</span>
            <input type="range" id="slider" min="0" max="1000" value="0">
          </div>
        </div>
        <div id="zedCard">
          <div class="title">Camera view</div>
          <div id="zedImgWrap"><img id="zedImg"><div id="zedPlaceholder">No camera frame</div></div>
          <div id="zedMeta"></div>
        </div>
      </div>
    </div>
    <!-- Velocity -->
    <div class="tabpane" id="velPane">
      <div class="toolbar">
        <label><input type="checkbox" id="vgGantry" checked> Gantry GT</label>
        <label><input type="checkbox" id="vgCam" checked> Camera estimate</label>
        <span class="hint">wheel = zoom X · drag = pan X · double-click = reset</span>
      </div>
      <div class="seriesBody">
        <div class="seriesCanvasWrap"><canvas id="velCanvas0"></canvas></div>
        <div class="seriesCanvasWrap"><canvas id="velCanvas1"></canvas></div>
        <div class="seriesCanvasWrap"><canvas id="velCanvas2"></canvas></div>
      </div>
    </div>
  </div>
  <div class="tooltip" id="tooltip"></div>

<script>
"use strict";
const DASH = __DASHBOARD_JSON__;

// When the experiment pipeline supplied the rich 3D viewer HTML, the Trajectory
// tab is that viewer (in an iframe) and the classic 2D top-down code is dormant.
const USE_VIEWER3D = !!(DASH.traj_viewer_html && DASH.traj_viewer_html.length);

// Shared time cursor across all tabs (seconds).
let currentT = 0;
let activeTab = "trajectory";

const tooltip = document.getElementById("tooltip");
function showTip(px, py, text){ tooltip.textContent = text; tooltip.style.display = "block";
  tooltip.style.left = (px + 14) + "px"; tooltip.style.top = (py + 10) + "px"; }
function hideTip(){ tooltip.style.display = "none"; }
function fmt(v, d){ return (v===null||v===undefined||!isFinite(v)) ? "n/a" : v.toFixed(d===undefined?1:d); }

// ── viridis colormap (approx control points) ────────────────────────────────
const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
  [31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(f){ f=Math.max(0,Math.min(1,f)); const x=f*(VIRIDIS.length-1);
  const i=Math.floor(x), t=x-i, a=VIRIDIS[i], b=VIRIDIS[Math.min(i+1,VIRIDIS.length-1)];
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`; }

// ── header ──────────────────────────────────────────────────────────────────
(function fillHeader(){
  document.getElementById("runName").textContent = "Run: " + DASH.run_name;
  let rms = "RMSE: n/a";
  if (DASH.rms) rms = `RMSE: X=${fmt(DASH.rms.x_mm)}mm  Y=${fmt(DASH.rms.y_mm)}mm  Z=${fmt(DASH.rms.z_mm)}mm`;
  const badge = DASH.gantry_only ? `<span class="badge badge-gantry">Gantry-only run (no camera)</span>` : "";
  const legacy = DASH.legacy_csv ? `<span class="badge badge-gantry">Legacy CSV — SDK velocity only</span>` : "";
  document.getElementById("meta").innerHTML =
    `<b>Duration:</b> ${DASH.duration_s}s &middot; <b>${DASH.n_gantry}</b> gantry samples &middot; ` +
    `<b>${DASH.n_camera}</b> camera samples${badge}${legacy}<br>` +
    `${rms}<br>` +
    `<b>Alignment:</b> ${DASH.alignment} &middot; <b>Anchor tag:</b> ${DASH.anchor_id}`;
})();

// ── tab switching (pure class toggle) ────────────────────────────────────────
function switchTab(name){
  activeTab = name;
  document.querySelectorAll(".tabbtn").forEach(b => b.classList.toggle("active", b.dataset.tab===name));
  document.getElementById("trajPane").classList.toggle("active", name==="trajectory");
  document.getElementById("velPane").classList.toggle("active", name==="velocity");
  hideTip();
  if (name==="trajectory"){ if(!USE_VIEWER3D){ syncSliderToCurrentT(); resizeTraj(); drawTraj(); } }
  else { velTab.resizeAll(); velTab.render(); }
}

function setupCanvas(cv){
  const dpr = Math.max(1, window.devicePixelRatio||1);
  const r = cv.getBoundingClientRect();
  cv.width = Math.max(1, Math.floor(r.width*dpr));
  cv.height = Math.max(1, Math.floor(r.height*dpr));
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w:r.width, h:r.height};
}

/* =========================================================================
 * TAB 1 — Trajectory (top-down X,Y)
 * ========================================================================= */
const GANTRY = DASH.traj.gantry.filter(p => p.x_m!==null && p.y_m!==null);
const CAMERA = DASH.traj.camera.filter(p => p.x_m!==null && p.y_m!==null);
const trajCanvas = document.getElementById("trajCanvas");
let trajView = null;  // {ctx,w,h}
const layers = { cam:true, gantry:true, tags:true, pool:true, cursor:true };

function trajBounds(){
  let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
  const acc = (x,y)=>{ if(x<minx)minx=x; if(x>maxx)maxx=x; if(y<miny)miny=y; if(y>maxy)maxy=y; };
  GANTRY.forEach(p=>acc(p.x_m,p.y_m));
  CAMERA.forEach(p=>acc(p.x_m,p.y_m));
  DASH.tags.forEach(t=>acc(t.x_m,t.y_m));
  // pool centered on data centroid (placement is approximate context only)
  const cx=(minx+maxx)/2||0, cy=(miny+maxy)/2||0;
  const L=DASH.pool.length_m, W=DASH.pool.width_m;
  let px, py;
  if (DASH.pool.long_axis==="x"){ px=L/2; py=W/2; } else { px=W/2; py=L/2; }
  pool_rect = {x0:cx-px, x1:cx+px, y0:cy-py, y1:cy+py};
  acc(pool_rect.x0,pool_rect.y0); acc(pool_rect.x1,pool_rect.y1);
  if(minx>maxx){ minx=-1;maxx=1;miny=-1;maxy=1; }
  const padx=(maxx-minx)*0.08+0.05, pady=(maxy-miny)*0.08+0.05;
  return {minx:minx-padx,maxx:maxx+padx,miny:miny-pady,maxy:maxy+pady};
}
let pool_rect = null;
let trajBnd = null;
function w2s(x,y){
  const b=trajBnd, m=24, w=trajView.w, h=trajView.h;
  const sx=(w-2*m)/(b.maxx-b.minx), sy=(h-2*m)/(b.maxy-b.miny);
  const s=Math.min(sx,sy);
  const ox=m+((w-2*m)-(b.maxx-b.minx)*s)/2, oy=m+((h-2*m)-(b.maxy-b.miny)*s)/2;
  return { x: ox+(x-b.minx)*s, y: h-(oy+(y-b.miny)*s) };  // flip Y up
}
function resizeTraj(){ trajView = setupCanvas(trajCanvas); trajBnd = trajBounds(); }

function nearestByTime(arr, t){
  if(!arr.length) return -1;
  let best=0, bd=1e18;
  for(let i=0;i<arr.length;i++){ const ti=arr[i].t; if(ti===null) continue;
    const d=Math.abs(ti-t); if(d<bd){bd=d;best=i;} }
  return best;
}

function drawTraj(){
  if(!trajView) resizeTraj();
  const ctx=trajView.ctx, w=trajView.w, h=trajView.h;
  ctx.clearRect(0,0,w,h); ctx.fillStyle="#0f0f12"; ctx.fillRect(0,0,w,h);
  // pool
  if(layers.pool && pool_rect){
    const a=w2s(pool_rect.x0,pool_rect.y0), b=w2s(pool_rect.x1,pool_rect.y1);
    ctx.save(); ctx.strokeStyle="#5a6470"; ctx.lineWidth=1.2; ctx.setLineDash([7,5]);
    ctx.strokeRect(Math.min(a.x,b.x),Math.min(a.y,b.y),Math.abs(b.x-a.x),Math.abs(b.y-a.y));
    ctx.restore();
    ctx.fillStyle="#5a6470"; ctx.font="11px Arial";
    ctx.fillText(`pool ${DASH.pool.length_m}x${DASH.pool.width_m} m (approx)`, Math.min(a.x,b.x)+4, Math.min(a.y,b.y)+14);
  }
  // tags
  if(layers.tags){
    DASH.tags.forEach(t=>{
      const p=w2s(t.x_m,t.y_m); const anchor=(t.id===DASH.anchor_id);
      ctx.fillStyle=anchor?"#26d0e0":"#888c93";
      ctx.beginPath(); ctx.arc(p.x,p.y,anchor?6:4,0,Math.PI*2); ctx.fill();
      ctx.fillStyle=anchor?"#26d0e0":"#aeb4bb"; ctx.font=(anchor?"bold ":"")+"11px Arial";
      ctx.fillText(""+t.id, p.x+7, p.y-6);
    });
  }
  // gantry GT (orange dashed) — drawn under camera
  if(layers.gantry && GANTRY.length>1){
    ctx.save(); ctx.strokeStyle="#ff8c42"; ctx.lineWidth=2; ctx.setLineDash([6,4]);
    ctx.beginPath();
    for(let i=0;i<GANTRY.length;i++){ const p=w2s(GANTRY[i].x_m,GANTRY[i].y_m);
      if(i===0)ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y); }
    ctx.stroke(); ctx.restore();
  }
  // camera (viridis, time-coded)
  if(layers.cam && CAMERA.length>1){
    for(let i=1;i<CAMERA.length;i++){
      const a=w2s(CAMERA[i-1].x_m,CAMERA[i-1].y_m), b=w2s(CAMERA[i].x_m,CAMERA[i].y_m);
      ctx.strokeStyle=viridis(i/(CAMERA.length-1)); ctx.lineWidth=CAMERA[i].has_tag?2.4:1.4;
      ctx.setLineDash(CAMERA[i].has_tag?[]:[5,4]);
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  // current-time markers
  if(layers.cursor){
    const gi=nearestByTime(GANTRY,currentT), ci=nearestByTime(CAMERA,currentT);
    let gp=null, cp=null;
    if(layers.gantry && gi>=0){ const g=GANTRY[gi]; gp=w2s(g.x_m,g.y_m);
      ctx.fillStyle="#2ecc71"; ctx.beginPath(); ctx.arc(gp.x,gp.y,6,0,Math.PI*2); ctx.fill(); }
    if(layers.cam && ci>=0 && CAMERA.length){ const c=CAMERA[ci]; cp=w2s(c.x_m,c.y_m);
      ctx.strokeStyle="#2ecc71"; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(cp.x,cp.y,6,0,Math.PI*2); ctx.stroke(); }
    if(gp&&cp){
      ctx.strokeStyle="#888"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(gp.x,gp.y); ctx.lineTo(cp.x,cp.y); ctx.stroke(); ctx.setLineDash([]);
      const g=GANTRY[gi], c=CAMERA[ci];
      const dd=Math.hypot((g.x_m-c.x_m),(g.y_m-c.y_m),(g.z_m||0)-(c.z_m||0))*1000;
      ctx.fillStyle="#cfd6dd"; ctx.font="11px Arial";
      ctx.fillText(`|Δ|=${dd.toFixed(1)} mm`, (gp.x+cp.x)/2+6, (gp.y+cp.y)/2);
    }
  }
  // legend
  ctx.fillStyle="rgba(20,24,30,0.85)"; ctx.fillRect(w-188,10,178,GANTRY.length&&CAMERA.length?72:54);
  ctx.strokeStyle="#33333a"; ctx.strokeRect(w-188,10,178,GANTRY.length&&CAMERA.length?72:54);
  ctx.font="11px Arial"; let ly=28;
  if(CAMERA.length){ ctx.strokeStyle=viridis(0.7); ctx.lineWidth=3; ctx.beginPath();
    ctx.moveTo(w-180,ly-4); ctx.lineTo(w-156,ly-4); ctx.stroke();
    ctx.fillStyle="#cfd6dd"; ctx.fillText("Camera (AprilTag SLAM)", w-150, ly); ly+=20; }
  if(GANTRY.length){ ctx.strokeStyle="#ff8c42"; ctx.lineWidth=2; ctx.setLineDash([6,4]); ctx.beginPath();
    ctx.moveTo(w-180,ly-4); ctx.lineTo(w-156,ly-4); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#cfd6dd"; ctx.fillText("Gantry GT", w-150, ly); ly+=20; }
  ctx.fillStyle="#26d0e0"; ctx.beginPath(); ctx.arc(w-172,ly-4,4,0,Math.PI*2); ctx.fill();
  ctx.fillStyle="#cfd6dd"; ctx.fillText("Anchor tag "+DASH.anchor_id, w-150, ly);
}

// slider <-> currentT
const slider=document.getElementById("slider"), sliderLabel=document.getElementById("sliderLabel");
function tRange(){
  let lo=1e18, hi=-1e18;
  GANTRY.concat(CAMERA).forEach(p=>{ if(p.t!==null){ if(p.t<lo)lo=p.t; if(p.t>hi)hi=p.t; } });
  if(lo>hi){ lo=0; hi=Math.max(1,DASH.duration_s); }
  return [lo,hi];
}
const TR = tRange();
function syncSliderToCurrentT(){
  const f=(currentT-TR[0])/Math.max(1e-6,(TR[1]-TR[0]));
  slider.value = Math.round(Math.max(0,Math.min(1,f))*1000);
  sliderLabel.textContent = `t = ${currentT.toFixed(2)} s`;
  updateZed();
}
slider.addEventListener("input", ()=>{
  const f=slider.value/1000; currentT = TR[0]+f*(TR[1]-TR[0]);
  sliderLabel.textContent = `t = ${currentT.toFixed(2)} s`;
  updateZed(); drawTraj();
});
function updateZed(){
  const img=document.getElementById("zedImg"), ph=document.getElementById("zedPlaceholder"),
        meta=document.getElementById("zedMeta");
  if(!CAMERA.length){ img.style.display="none"; ph.style.display="block";
    ph.textContent = DASH.gantry_only?"Gantry-only run (no camera)":"No camera frame"; meta.textContent=""; return; }
  const ci=nearestByTime(CAMERA,currentT); const c=CAMERA[ci];
  if(c.image){ img.src=c.image; img.style.display="block"; ph.style.display="none";
    img.onerror=()=>{ img.style.display="none"; ph.style.display="block"; ph.textContent="frame not found (HTML moved from run folder)"; }; }
  else { img.style.display="none"; ph.style.display="block"; ph.textContent="no frame for this sample"; }
  meta.textContent = `t = ${(c.t||0).toFixed(2)} s   tags: ${c.has_tag?"yes":"no"}\n`+
    `cam X=${(c.x_m*100).toFixed(1)} Y=${(c.y_m*100).toFixed(1)} Z=${(c.z_m*100).toFixed(1)} cm`;
}
["tgCam","tgGantry","tgTags","tgPool","tgCursor"].forEach((id,k)=>{
  document.getElementById(id).addEventListener("change",e=>{
    layers[["cam","gantry","tags","pool","cursor"][k]] = e.target.checked; drawTraj();
  });
});
trajCanvas.addEventListener("mousemove", e=>{
  if(!trajView) return;
  const r=trajCanvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  // nearest sample by screen distance over the union of gantry+camera
  let best=null, bd=1e18, bt=currentT;
  function scan(arr){ for(let i=0;i<arr.length;i++){ const p=w2s(arr[i].x_m,arr[i].y_m);
    const d=Math.hypot(p.x-mx,p.y-my); if(d<bd){bd=d; best=arr[i]; bt=arr[i].t;} } }
  if(layers.gantry) scan(GANTRY); if(layers.cam) scan(CAMERA);
  if(best===null || bd>40){ hideTip(); return; }
  const gi=nearestByTime(GANTRY,bt), ci=nearestByTime(CAMERA,bt);
  const g=gi>=0?GANTRY[gi]:null, c=ci>=0&&CAMERA.length?CAMERA[ci]:null;
  let txt=`t = ${bt!==null?bt.toFixed(2):"n/a"} s`;
  if(g) txt+=`\nGantry:  X=${(g.x_m*1000).toFixed(1)}  Y=${(g.y_m*1000).toFixed(1)}  Z=${(g.z_m*1000).toFixed(1)}  mm`;
  if(c) txt+=`\nCamera:  X=${(c.x_m*1000).toFixed(1)}  Y=${(c.y_m*1000).toFixed(1)}  Z=${(c.z_m*1000).toFixed(1)}  mm`;
  if(g&&c){ const dd=Math.hypot(g.x_m-c.x_m,g.y_m-c.y_m,(g.z_m||0)-(c.z_m||0))*1000;
    txt+=`\nΔ:       ${((g.x_m-c.x_m)*1000).toFixed(1)}     ${((g.y_m-c.y_m)*1000).toFixed(1)}     ${((g.z_m-c.z_m)*1000).toFixed(1)}      mm  (|Δ|=${dd.toFixed(1)} mm)`; }
  showTip(e.clientX, e.clientY, txt);
});
trajCanvas.addEventListener("mouseleave", hideTip);

/* =========================================================================
 * TABS 2 & 3 — shared time-series renderer
 * ========================================================================= */
function makeSeriesTab(prefix, axisKeys, unit, gantryToggleId, camToggleId){
  // axisKeys: ['vx','vy','vz'] or ['ax','ay','az']; unit: 'cm/s' | 'cm/s²'
  const labels = axisKeys.map(k => k[0].toUpperCase()+k.slice(1).replace(/^./, s=>s.toUpperCase()));
  const niceLabels = (unit.indexOf("s²")>=0)
    ? ["Ax ("+unit+")","Ay ("+unit+")","Az ("+unit+")"]
    : ["Vx ("+unit+")","Vy ("+unit+")","Vz ("+unit+")"];
  const canvases = [0,1,2].map(i=>document.getElementById(prefix+"Canvas"+i));
  const views = [null,null,null];
  const gT = DASH.series.t_gantry, cT = DASH.series.t_camera;
  const gS = axisKeys.map(k => DASH.series.gantry[k]);
  const cS = DASH.series.camera ? axisKeys.map(k => DASH.series.camera[k]) : null;
  const fullRange = (()=>{ let lo=1e18,hi=-1e18;
    [gT, cT].forEach(arr=>{ if(arr) arr.forEach(t=>{ if(t!==null){ if(t<lo)lo=t; if(t>hi)hi=t; } }); });
    if(lo>hi){ lo=0; hi=Math.max(1,DASH.duration_s); } return [lo,hi]; })();
  const st = { xRange:[fullRange[0],fullRange[1]], showG:true, showC:true,
               dragging:false, dragX0:0, range0:null };
  const M = {l:58, r:14, t:10, b:24};

  function resizeAll(){ for(let i=0;i<3;i++) views[i]=setupCanvas(canvases[i]); }
  function yfit(i){
    let lo=1e18, hi=-1e18; const x0=st.xRange[0], x1=st.xRange[1];
    function scan(T,S){ if(!T||!S) return; for(let k=0;k<T.length;k++){ const t=T[k];
      if(t===null||t<x0||t>x1) continue; const v=S[i][k]; if(v===null) continue;
      if(v<lo)lo=v; if(v>hi)hi=v; } }
    if(st.showG) scan(gT,gS); if(st.showC&&cS) scan(cT,cS);
    if(lo>hi){ lo=-1; hi=1; }
    if(hi-lo<1e-6){ lo-=1; hi+=1; }
    const pad=(hi-lo)*0.1; return [lo-pad, hi+pad];
  }
  function ticks(lo,hi,n){ const span=hi-lo; let step=Math.pow(10,Math.floor(Math.log10(span/n)));
    const err=span/(n*step); if(err>5)step*=10; else if(err>2)step*=5; else if(err>1)step*=2;
    const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi; v+=step) out.push(v); return out; }
  function plot2screen(i, t, v, yr){ const view=views[i];
    const x0=st.xRange[0], x1=st.xRange[1];
    const px=M.l+(t-x0)/(x1-x0)*(view.w-M.l-M.r);
    const py=M.t+(yr[1]-v)/(yr[1]-yr[0])*(view.h-M.t-M.b);
    return {x:px,y:py}; }

  function renderOne(i){
    const view=views[i]; if(!view) return; const ctx=view.ctx, w=view.w, h=view.h;
    ctx.clearRect(0,0,w,h); ctx.fillStyle="#1a1a1d"; ctx.fillRect(0,0,w,h);
    const yr=yfit(i), x0=st.xRange[0], x1=st.xRange[1];
    // grid + axes
    ctx.strokeStyle="#2a2a30"; ctx.fillStyle="#9aa3ad"; ctx.font="10px Arial"; ctx.lineWidth=1;
    ticks(yr[0],yr[1],6).forEach(v=>{ const p=plot2screen(i,x0,v,yr);
      ctx.beginPath(); ctx.moveTo(M.l,p.y); ctx.lineTo(w-M.r,p.y); ctx.stroke();
      ctx.fillText(v.toFixed(Math.abs(v)<10?1:0), 6, p.y+3); });
    ticks(x0,x1,8).forEach(t=>{ const p=plot2screen(i,t,yr[1],yr);
      ctx.strokeStyle="#23232a"; ctx.beginPath(); ctx.moveTo(p.x,M.t); ctx.lineTo(p.x,h-M.b); ctx.stroke();
      if(i===2){ ctx.fillStyle="#9aa3ad"; ctx.fillText(t.toFixed(1), p.x-8, h-8); } });
    // y label
    ctx.save(); ctx.translate(12,h/2); ctx.rotate(-Math.PI/2); ctx.fillStyle="#cfd6dd";
    ctx.font="11px Arial"; ctx.textAlign="center"; ctx.fillText(niceLabels[i],0,0); ctx.restore();
    // line drawer
    function line(T,S,color,dash){ if(!T||!S) return; ctx.strokeStyle=color; ctx.lineWidth=1.4;
      ctx.setLineDash(dash); ctx.beginPath(); let pen=false;
      for(let k=0;k<T.length;k++){ const t=T[k], v=S[i][k];
        if(t===null||v===null||t<x0||t>x1){ pen=false; continue; }
        const p=plot2screen(i,t,v,yr); if(!pen){ctx.moveTo(p.x,p.y);pen=true;} else ctx.lineTo(p.x,p.y); }
      ctx.stroke(); ctx.setLineDash([]); }
    if(st.showG) line(gT,gS,"#4ea1ff",[]);
    if(st.showC&&cS) line(cT,cS,"#ff5d5d",[6,4]);
    // time cursor
    if(currentT>=x0 && currentT<=x1){ const p=plot2screen(i,currentT,yr[1],yr);
      ctx.strokeStyle="#8a929c"; ctx.lineWidth=1; ctx.setLineDash([4,3]);
      ctx.beginPath(); ctx.moveTo(p.x,M.t); ctx.lineTo(p.x,h-M.b); ctx.stroke(); ctx.setLineDash([]); }
    // axis frame
    ctx.strokeStyle="#3a3a42"; ctx.strokeRect(M.l,M.t,w-M.l-M.r,h-M.t-M.b);
  }
  function render(){ for(let i=0;i<3;i++) renderOne(i); }

  function valAt(T,S,i,t){ if(!T||!S) return null; let best=null,bd=1e18;
    for(let k=0;k<T.length;k++){ if(T[k]===null||S[i][k]===null) continue;
      const d=Math.abs(T[k]-t); if(d<bd){bd=d;best=S[i][k];} } return best; }

  // interaction (attach to each canvas)
  canvases.forEach((cv,idx)=>{
    cv.addEventListener("mousemove", e=>{
      const view=views[idx]; if(!view) return;
      const r=cv.getBoundingClientRect(), mx=e.clientX-r.left;
      if(st.dragging){ const x0=st.xRange[0], x1=st.xRange[1];
        const dpx=(mx-st.dragX0)/(view.w-M.l-M.r)*(st.range0[1]-st.range0[0]);
        st.xRange=[st.range0[0]-dpx, st.range0[1]-dpx]; render(); return; }
      const x0=st.xRange[0], x1=st.xRange[1];
      const t=x0+(mx-M.l)/(view.w-M.l-M.r)*(x1-x0);
      if(t<x0||t>x1){ hideTip(); return; }
      currentT=t; render();
      // tooltip
      let txt=`t = ${t.toFixed(2)} s`;
      for(let i=0;i<3;i++){ const gv=valAt(gT,gS,i,t), cv2=cS?valAt(cT,cS,i,t):null;
        txt+=`\n${labels[i]}:  Gantry ${gv===null?"n/a":(gv>=0?"+":"")+gv.toFixed(1)} ${unit}`+
             (cS?`   Camera ${cv2===null?"n/a":(cv2>=0?"+":"")+cv2.toFixed(1)} ${unit}`:""); }
      showTip(e.clientX, e.clientY, txt);
    });
    cv.addEventListener("mouseleave", ()=>{ hideTip(); });
    cv.addEventListener("mousedown", e=>{ const r=cv.getBoundingClientRect();
      st.dragging=true; st.dragX0=e.clientX-r.left; st.range0=[st.xRange[0],st.xRange[1]]; });
    window.addEventListener("mouseup", ()=>{ st.dragging=false; });
    cv.addEventListener("wheel", e=>{ e.preventDefault(); const view=views[idx]; if(!view) return;
      const r=cv.getBoundingClientRect(), mx=e.clientX-r.left;
      const x0=st.xRange[0], x1=st.xRange[1];
      const t=x0+(mx-M.l)/(view.w-M.l-M.r)*(x1-x0);
      const f=e.deltaY<0?0.85:1.18;
      st.xRange=[t-(t-x0)*f, t+(x1-t)*f]; render(); }, {passive:false});
    cv.addEventListener("dblclick", ()=>{ st.xRange=[fullRange[0],fullRange[1]]; render(); });
  });

  function setG(v){ st.showG=v; render(); }
  function setC(v){ st.showC=v; render(); }
  document.getElementById(gantryToggleId).addEventListener("change",e=>setG(e.target.checked));
  document.getElementById(camToggleId).addEventListener("change",e=>setC(e.target.checked));
  return { render, resizeAll, st };
}

const velTab = makeSeriesTab("vel", ["vx","vy","vz"], "cm/s", "vgGantry", "vgCam");

// ── init ─────────────────────────────────────────────────────────────────────
window.addEventListener("resize", ()=>{
  if(activeTab==="trajectory"){ if(!USE_VIEWER3D){ resizeTraj(); drawTraj(); } }
  else { velTab.resizeAll(); velTab.render(); }
});
currentT = TR[0];
if (USE_VIEWER3D) {
  // Mount the rich 3D viewer; the classic 2D toolbar/body stay hidden.
  document.getElementById("trajPane").classList.add("viewer3d");
  document.getElementById("trajFrame").srcdoc = DASH.traj_viewer_html;
} else {
  requestAnimationFrame(()=>{ resizeTraj(); syncSliderToCurrentT(); drawTraj(); });
}
</script>
</body>
</html>"""


# Camera-velocity smoothing parameters. Kept identical to gantry_runner's
# SMOOTHING_WINDOW_S / SMOOTHING_POLYORDER so the camera estimate and the gantry
# ground truth share the same temporal smoothing (a fair comparison).
SMOOTHING_WINDOW_S = 0.25
SMOOTHING_POLYORDER = 2


def _savgol_deriv(t: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    """Savitzky-Golay smooth time-derivative for the camera trajectory.

    Delegates to gantry_runner.compute_derivative_savgol with the SAME window
    and polynomial order the gantry logger uses, so camera-derived velocity and
    gantry-derived velocity are computed identically (fair comparison). Falls
    back to a local copy if gantry_runner can't be imported (keeps tagslam_core
    importable on a camera-only checkout).
    """
    try:
        from gantry_runner import compute_derivative_savgol as _cds
        return _cds(t, y, SMOOTHING_WINDOW_S, SMOOTHING_POLYORDER)
    except Exception:
        pass
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < SMOOTHING_POLYORDER + 1:
        return out
    half = SMOOTHING_WINDOW_S / 2.0
    tol = half * 0.5
    for i in range(n):
        if not np.isfinite(t[i]):
            continue
        mask = np.isfinite(t) & np.isfinite(y) & (t >= t[i] - half) & (t <= t[i] + half)
        if int(mask.sum()) < SMOOTHING_POLYORDER + 1:
            continue
        tt = t[mask] - t[i]
        if tt.min() > -tol or tt.max() < tol:
            continue
        try:
            coef = np.polyfit(tt, y[mask], SMOOTHING_POLYORDER)
            out[i] = float(np.polyval(np.polyder(coef), 0.0))
        except Exception:
            continue
    return out


def _dash_velocity_diagnostics(g_t, V_gantry, V_slam, c_t, V_cam, R3, legacy_csv):
    """Stderr diagnostics for the dashboard Velocity tab (Issue #2).

    Traces one sample through the gantry-velocity transform chain, reports the
    per-axis camera-vs-gantry RMS divergence, warns when it is gross (>50 cm/s),
    and — crucially — warns when applying R's transpose would align the curves
    far better (the classic "R_gantry_to_slam stored transposed" mistake).

    Args (all mm/s unless noted):
      g_t        (N,)   gantry elapsed time [s]
      V_gantry   (N,3)  gantry-frame velocity (before R)
      V_slam     (N,3)  gantry velocity rotated into the SLAM frame (plotted)
      c_t        (M,)   camera elapsed time [s]
      V_cam      (M,3)  camera-derived velocity, or None when no camera
      R3         (3,3)  rotation applied to the gantry velocity
      legacy_csv bool   True when only the legacy SDK velocity column existed
    """
    axes = ("X", "Y", "Z")
    if legacy_csv:
        for a in axes:
            print(f"[dashboard] using legacy SDK velocity column for {a} axis — "
                  "derived not available", file=sys.stderr)

    try:
        speeds = np.linalg.norm(np.nan_to_num(V_gantry), axis=1)
        i = int(np.argmax(speeds)) if speeds.size else 0
        vg, vs = V_gantry[i], V_slam[i]
        print("[dashboard] velocity transform (sample %d): gantry (%.2f, %.2f, %.2f) mm/s "
              "--R--> slam (%.2f, %.2f, %.2f) mm/s --/10--> (%.2f, %.2f, %.2f) cm/s"
              % (i, vg[0], vg[1], vg[2], vs[0], vs[1], vs[2],
                 vs[0] / 10, vs[1] / 10, vs[2] / 10), file=sys.stderr)
    except Exception:
        pass

    if V_cam is None or not getattr(V_cam, "size", 0):
        return

    def _div(gv_mm, cv_mm):
        mc = np.isfinite(c_t) & np.isfinite(cv_mm)
        if int(mc.sum()) < 2:
            return float("nan")
        ci = np.interp(g_t, c_t[mc], cv_mm[mc])
        f = np.isfinite(gv_mm) & np.isfinite(ci) & np.isfinite(g_t)
        if int(f.sum()) < 1:
            return float("nan")
        return float(np.sqrt(np.nanmean(((gv_mm[f] - ci[f]) / 10.0) ** 2)))

    div = [_div(V_slam[:, k], V_cam[:, k]) for k in range(3)]
    print("[dashboard] velocity RMS divergence (camera vs gantry): "
          "Vx=%.1f Vy=%.1f Vz=%.1f cm/s" % (div[0], div[1], div[2]), file=sys.stderr)
    worst = max([d for d in div if np.isfinite(d)], default=0.0)
    if worst > 50.0:
        print("[dashboard] WARNING: velocity RMS divergence %.1f cm/s exceeds 50 cm/s — "
              "camera/gantry frames likely misaligned (check R_gantry_to_slam)."
              % worst, file=sys.stderr)

    # Transpose hint: would R.T align the curves substantially better?
    if R3 is not None and not np.allclose(R3, np.eye(3), atol=1e-6):
        V_slam_T = (np.asarray(R3).T @ V_gantry.T).T
        div_T = [_div(V_slam_T[:, k], V_cam[:, k]) for k in range(3)]
        tot, tot_T = float(np.nansum(div)), float(np.nansum(div_T))
        if np.isfinite(tot) and np.isfinite(tot_T) and tot > 3.0 and tot_T < 0.6 * tot:
            print("[dashboard] WARNING: R_gantry_to_slam may be TRANSPOSED — current total "
                  "velocity divergence %.1f cm/s, transpose gives %.1f cm/s. Consider "
                  "replacing R with its transpose in the calibration." % (tot, tot_T),
                  file=sys.stderr)


def write_experiment_dashboard_html(
    path: Path,
    gantry_csv: Path,
    camera_csv: "Path | None",
    tag_poses_csv: "Path | None",
    pool_cfg: dict,
    *,
    anchor_id: int = 1,
    T_gantry_camera: "np.ndarray | None" = None,
    gantry_anchor_offset_mm: "list[float] | None" = None,
    R_gantry_to_slam: "np.ndarray | None" = None,
    run_name: str = "",
    rms_summary: dict | None = None,
    zed_view_image_paths: "list[Path] | None" = None,
    tag_size_m: float = 0.085,
    plot_z_scale: float = 0.5,
) -> "Path | None":
    """Write a single self-contained HTML dashboard with two tabs:
    Trajectory and Velocity.

    The Trajectory tab reuses the visual language of
    write_interactive_trajectory_html (dark theme, dashed pool outline, numbered
    AprilTag markers with the anchor highlighted in cyan, a viridis time-coded
    camera trajectory, a time slider and a camera-frame card) rendered as a
    top-down (X, Y) 2D canvas, and overlays the gantry ground-truth trajectory
    as an orange dashed polyline.

    The Velocity tab shows three stacked subplots (Vx, Vy, Vz in cm/s) over
    elapsed_s, each overlaying gantry GT (solid blue) and camera estimate
    (dashed red). The Acceleration tab is identical in structure, in cm/s².

    Pure HTML + CSS + vanilla JS using <canvas>. No CDN, no external libraries;
    the file renders when copied to any folder/machine (camera-frame thumbnails
    in the Trajectory card are referenced by relative path and are the only part
    that needs the sibling frames/ directory).

    Gantry-only mode (camera_csv is None): Trajectory shows pool + gantry only;
    Velocity/Acceleration show gantry curves only; header shows a gantry-only
    badge.
    """
    import csv as _csv
    import json as _json

    def _load(p: "Path | None") -> list[dict]:
        if p is None or not Path(p).exists():
            return []
        with open(p, newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))

    def _fcol(rows: list[dict], key: str) -> "np.ndarray":
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (KeyError, ValueError, TypeError):
                out.append(float("nan"))
        return np.array(out, dtype=np.float64)

    g_rows = _load(gantry_csv)
    c_rows = _load(camera_csv) if camera_csv is not None else []
    if not g_rows and not c_rows:
        return None
    gantry_only = not c_rows

    # ── shared time base: elapsed = timestamp_monotonic - t0, t0 = min across
    #    both CSVs. Falls back to each CSV's own elapsed column when monotonic
    #    timestamps are absent. ───────────────────────────────────────────────
    g_tmono = _fcol(g_rows, "timestamp_monotonic") if g_rows else np.array([])
    c_tmono = _fcol(c_rows, "timestamp_monotonic") if c_rows else np.array([])
    t0_candidates = []
    for arr in (g_tmono, c_tmono):
        finite = arr[np.isfinite(arr)] if arr.size else arr
        if finite.size:
            t0_candidates.append(float(finite.min()))
    t0 = min(t0_candidates) if t0_candidates else None

    def _elapsed(rows: list[dict], tmono: "np.ndarray", fallback_key: str) -> "np.ndarray":
        if t0 is not None and tmono.size == len(rows) and np.isfinite(tmono).any():
            return tmono - t0
        return _fcol(rows, fallback_key)

    g_t = _elapsed(g_rows, g_tmono, "elapsed_s") if g_rows else np.array([])
    c_t = _elapsed(c_rows, c_tmono, "time_s") if c_rows else np.array([])

    # ── gantry position (mm) + derived velocity (mm/s) ────────────────────────
    g_x = _fcol(g_rows, "x_mm");  g_y = _fcol(g_rows, "y_mm");  g_z = _fcol(g_rows, "z_mm")

    # Velocity source: prefer the new *_derived columns; fall back to the legacy
    # vx_mm_s (SDK) for old recordings (legacy flag drives a banner in the HTML).
    g_cols = set(g_rows[0].keys()) if g_rows else set()
    legacy_csv = ("vx_mm_s_derived" not in g_cols) and ("vx_mm_s" in g_cols)
    if "vx_mm_s_derived" in g_cols:
        g_vx = _fcol(g_rows, "vx_mm_s_derived"); g_vy = _fcol(g_rows, "vy_mm_s_derived"); g_vz = _fcol(g_rows, "vz_mm_s_derived")
    elif "vx_mm_s" in g_cols:  # legacy schema
        g_vx = _fcol(g_rows, "vx_mm_s"); g_vy = _fcol(g_rows, "vy_mm_s"); g_vz = _fcol(g_rows, "vz_mm_s")
    else:
        g_vx = g_vy = g_vz = np.full(len(g_rows), np.nan)

    # ── camera arrays: positions m → mm, derive velocity via Savitzky-Golay,
    #    the SAME window/order as GantryTelemetryLogger (fair comparison). ──────
    c_xm = _fcol(c_rows, "x_m"); c_ym = _fcol(c_rows, "y_m"); c_zm = _fcol(c_rows, "z_m")
    c_x = c_xm * 1000.0; c_y = c_ym * 1000.0; c_z = c_zm * 1000.0
    if c_rows:
        c_vx = _savgol_deriv(c_t, c_x); c_vy = _savgol_deriv(c_t, c_y); c_vz = _savgol_deriv(c_t, c_z)
    else:
        c_vx = c_vy = c_vz = np.array([])

    # ── gantry → SLAM frame transform: rotate by R_gantry_to_slam (default
    #    identity), subtract gantry_anchor_offset_mm. Positions get R+offset;
    #    velocities get R only (no translation). See transform_gantry_to_slam.
    R3 = np.asarray(R_gantry_to_slam, dtype=np.float64).reshape(3, 3) \
        if R_gantry_to_slam is not None else np.eye(3)
    if gantry_anchor_offset_mm is not None and len(gantry_anchor_offset_mm) >= 3:
        off3 = np.array([float(gantry_anchor_offset_mm[0]),
                         float(gantry_anchor_offset_mm[1]),
                         float(gantry_anchor_offset_mm[2])], dtype=np.float64)
    else:
        off3 = np.zeros(3, dtype=np.float64)

    if len(g_rows):
        P = np.stack([g_x, g_y, g_z], axis=1)              # mm, gantry frame
        P_slam_mm = (R3 @ (P - off3).T).T                  # mm, SLAM orientation
        g_xa = P_slam_mm[:, 0] / 1000.0
        g_ya = P_slam_mm[:, 1] / 1000.0
        g_za = P_slam_mm[:, 2] / 1000.0
        V = np.stack([g_vx, g_vy, g_vz], axis=1)           # mm/s, gantry frame
        V_slam = (R3 @ V.T).T                              # mm/s, SLAM orientation
        g_vx, g_vy, g_vz = V_slam[:, 0], V_slam[:, 1], V_slam[:, 2]
        # Issue #2: trace the velocity transform + flag frame/transpose problems.
        _dash_velocity_diagnostics(
            g_t, V, V_slam, c_t,
            (np.stack([c_vx, c_vy, c_vz], axis=1) if (not gantry_only and c_vx.size) else None),
            R3, legacy_csv,
        )
    else:
        g_xa = g_ya = g_za = np.array([])

    if gantry_anchor_offset_mm is not None and len(gantry_anchor_offset_mm) >= 2:
        alignment = "gantry_anchor_offset_mm"
    elif (not gantry_only and g_xa.size and c_xm.size
          and np.isfinite(g_xa[0]) and np.isfinite(c_xm[0])):
        # Pin gantry first sample to camera first sample (approximate).
        g_xa = g_xa - (g_xa[0] - c_xm[0])
        g_ya = g_ya - (g_ya[0] - c_ym[0])
        g_za = g_za - (g_za[0] - c_zm[0])
        alignment = "first-sample-zeroed (approximate)"
    elif g_xa.size:
        # Gantry-only: zero to the gantry's own first sample.
        g_xa = g_xa - g_xa[0]; g_ya = g_ya - g_ya[0]; g_za = g_za - g_za[0]
        alignment = "first-sample-zeroed (approximate)"
    else:
        alignment = "none"

    # ── pose RMSE (mm): interpolate camera onto gantry time grid, on the
    #    aligned positions, so the number reflects tracking drift rather than a
    #    constant frame offset. rms_summary (if provided) overrides. ───────────
    def _rmse(gar: "np.ndarray", gt: "np.ndarray", car: "np.ndarray", ct: "np.ndarray") -> float:
        fg = np.isfinite(gar) & np.isfinite(gt)
        fc = np.isfinite(car) & np.isfinite(ct)
        if fg.sum() < 1 or fc.sum() < 2:
            return float("nan")
        ci = np.interp(gt[fg], ct[fc], car[fc])
        return float(np.sqrt(np.nanmean((gar[fg] - ci) ** 2)))

    rms = rms_summary
    if rms is None and not gantry_only and c_xm.size:
        rms = {
            "x_mm": _rmse(g_xa * 1000.0, g_t, c_x, c_t),
            "y_mm": _rmse(g_ya * 1000.0, g_t, c_y, c_t),
            "z_mm": _rmse(g_za * 1000.0, g_t, c_z, c_t),
        }

    # ── tag markers ───────────────────────────────────────────────────────────
    tag_rows = _load(tag_poses_csv) if tag_poses_csv is not None else []
    tags = []
    for r in tag_rows:
        try:
            tags.append({"id": int(r["tag_id"]), "x_m": float(r["x_m"]), "y_m": float(r["y_m"])})
        except (KeyError, ValueError, TypeError):
            pass

    # ── duration ──────────────────────────────────────────────────────────────
    all_t = [a for a in (g_t, c_t) if a.size]
    duration_s = 0.0
    if all_t:
        finite_max = [np.nanmax(a) for a in all_t if np.isfinite(a).any()]
        duration_s = float(max(finite_max)) if finite_max else 0.0

    # ── camera image paths (relative, for the ZED-view card) ─────────────────
    cam_images = [str(r.get("image_path", "") or "") for r in c_rows]

    def _clean(arr: "np.ndarray") -> list:
        # JSON-safe: NaN/inf -> None
        return [None if not np.isfinite(v) else round(float(v), 5) for v in arr]

    pool = normalize_pool_config(dict(pool_cfg) if pool_cfg else {})

    traj_gantry = []
    for i in range(len(g_rows)):
        traj_gantry.append({
            "t": None if not np.isfinite(g_t[i]) else round(float(g_t[i]), 4),
            "x_m": None if not np.isfinite(g_xa[i]) else round(float(g_xa[i]), 5),
            "y_m": None if not np.isfinite(g_ya[i]) else round(float(g_ya[i]), 5),
            "z_m": None if not np.isfinite(g_za[i]) else round(float(g_za[i]), 5),
        })
    traj_camera = []
    for i in range(len(c_rows)):
        has_tag = str(c_rows[i].get("has_tag_update", "")).strip().lower() in ("true", "1", "yes")
        traj_camera.append({
            "t": None if not np.isfinite(c_t[i]) else round(float(c_t[i]), 4),
            "x_m": None if not np.isfinite(c_xm[i]) else round(float(c_xm[i]), 5),
            "y_m": None if not np.isfinite(c_ym[i]) else round(float(c_ym[i]), 5),
            "z_m": None if not np.isfinite(c_zm[i]) else round(float(c_zm[i]), 5),
            "has_tag": has_tag,
            "image": cam_images[i] if i < len(cam_images) else "",
        })

    # ── rich 3D viewer (Trajectory tab) ───────────────────────────────────────
    #    The Trajectory tab embeds the full interactive 3D viewer (the same one
    #    the standalone zed2 pipeline produces) via an <iframe srcdoc>. We build
    #    camera_rows in the viewer's dict schema, a parallel DATA.gantry array in
    #    the (already camera-aligned) SLAM frame, and full tag rows. Gantry-only
    #    runs (no camera) leave this empty and fall back to the 2D top-down.
    def _f(r, k, default=0.0):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return default

    viewer_gantry = []
    for i in range(len(g_rows)):
        if not (np.isfinite(g_xa[i]) and np.isfinite(g_ya[i]) and np.isfinite(g_za[i])):
            continue
        viewer_gantry.append({
            "x_m": round(float(g_xa[i]), 5),
            "y_m": round(float(g_ya[i]), 5),
            "z_m": round(float(g_za[i]), 5),
            "t": (round(float(g_t[i]), 4) if np.isfinite(g_t[i]) else None),
        })

    viewer_camera_rows = []
    for i, r in enumerate(c_rows):
        htu = str(r.get("has_tag_update", "")).strip().lower() in ("true", "1", "yes")
        viewer_camera_rows.append({
            "camera_index": int(_f(r, "camera_index", i)),
            "time_s": (round(float(c_t[i]), 4) if np.isfinite(c_t[i]) else 0.0),
            "x_m": round(float(c_xm[i]), 5) if np.isfinite(c_xm[i]) else 0.0,
            "y_m": round(float(c_ym[i]), 5) if np.isfinite(c_ym[i]) else 0.0,
            "z_m": round(float(c_zm[i]), 5) if np.isfinite(c_zm[i]) else 0.0,
            "roll_deg": _f(r, "roll_deg"), "pitch_deg": _f(r, "pitch_deg"),
            "yaw_deg": _f(r, "yaw_deg"),
            "detected_tags": str(r.get("detected_tags", "") or ""),
            "has_tag_update": htu,
            "image_path": str(r.get("image_path", "") or ""),
        })

    viewer_tag_rows = []
    for r in tag_rows:
        try:
            viewer_tag_rows.append({
                "tag_id": int(float(r["tag_id"])),
                "x_m": float(r["x_m"]), "y_m": float(r["y_m"]),
                "z_m": float(r.get("z_m", 0.0) or 0.0),
                "roll_deg": float(r.get("roll_deg", 0.0) or 0.0),
                "pitch_deg": float(r.get("pitch_deg", 0.0) or 0.0),
                "yaw_deg": float(r.get("yaw_deg", 0.0) or 0.0),
            })
        except (KeyError, ValueError, TypeError):
            pass

    viewer_html = ""
    if not gantry_only and viewer_camera_rows:
        try:
            viewer_html = _build_trajectory_viewer_html(
                viewer_camera_rows, viewer_tag_rows, pool_cfg,
                float(tag_size_m), float(plot_z_scale), int(anchor_id),
                gantry_traj=viewer_gantry,
            )
        except Exception as exc:  # fall back to the 2D top-down trajectory tab
            print(f"[dashboard] 3D viewer build failed, using 2D top-down: {exc}",
                  file=sys.stderr)
            viewer_html = ""

    payload = {
        "run_name": run_name or (Path(gantry_csv).parent.name if gantry_csv else "run"),
        "alignment": alignment,
        "traj_viewer_html": viewer_html,
        "gantry_only": bool(gantry_only),
        "duration_s": round(duration_s, 2),
        "n_gantry": len(g_rows),
        "n_camera": len(c_rows),
        "anchor_id": int(anchor_id),
        "legacy_csv": bool(legacy_csv),
        "rms": (None if rms is None else {k: (None if (v is None or not np.isfinite(v)) else round(float(v), 1))
                                          for k, v in rms.items()}),
        "pool": {
            "length_m": float(pool.get("length_m", 4.877)),
            "width_m": float(pool.get("width_m", 2.438)),
            "depth_m": float(pool.get("depth_m", 1.143)),
            "long_axis": str(pool.get("pool_long_axis", "x")),
        },
        "tags": tags,
        "traj": {"gantry": traj_gantry, "camera": traj_camera},
        "series": {
            "t_gantry": _clean(g_t),
            "t_camera": _clean(c_t),
            # cm/s for display (mm/s -> cm/s == / 10). Velocity tab only.
            "gantry": {
                "vx": _clean(g_vx / 10.0), "vy": _clean(g_vy / 10.0), "vz": _clean(g_vz / 10.0),
            },
            "camera": (None if gantry_only else {
                "vx": _clean(c_vx / 10.0), "vy": _clean(c_vy / 10.0), "vz": _clean(c_vz / 10.0),
            }),
        },
    }

    # Safe-embed: traj_viewer_html contains its own </script>; escaping "</" as
    # "<\/" keeps the HTML tokenizer from closing the dashboard's <script> early
    # (JSON parses \/ back to /). Standard JSON-in-<script> hardening.
    dash_json = _json.dumps(payload).replace("</", "<\\/")
    html = _DASHBOARD_TEMPLATE.replace("__DASHBOARD_JSON__", dash_json)
    path.write_text(html, encoding="utf-8")
    return path
