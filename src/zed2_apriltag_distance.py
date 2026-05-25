import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pyzed.sl as sl

import ZED2 as zed_utils


WINDOW_NAME = "ZED2 AprilTag distance"
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.085
DEFAULT_TEST_DURATION_S = 10.0
APRILTAG_DICTIONARIES = {
    "16h5": ("DICT_APRILTAG_16h5", "DICT_APRILTAG_16H5"),
    "25h9": ("DICT_APRILTAG_25h9", "DICT_APRILTAG_25H9"),
    "36h10": ("DICT_APRILTAG_36h10", "DICT_APRILTAG_36H10"),
    "36h11": ("DICT_APRILTAG_36h11", "DICT_APRILTAG_36H11"),
}


@dataclass
class TagMeasurement:
    tag_id: int
    corners: np.ndarray
    center_px: tuple[int, int]
    xyz_m: tuple[float, float, float] | None
    range_m: float | None
    z_m: float | None
    valid_pixels: int
    pose_tvec_m: tuple[float, float, float] | None = None
    pose_range_m: float | None = None
    rotation_matrix: np.ndarray | None = None
    euler_deg: tuple[float, float, float] | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect AprilTags with the ZED2 left camera and measure distance "
            "using ZED stereo depth."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print connected ZED cameras and exit.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        help="Open a specific local ZED camera ID.",
    )
    parser.add_argument(
        "--serial",
        type=int,
        help="Open a specific ZED camera serial number.",
    )
    parser.add_argument(
        "--svo",
        help="Read from a ZED SVO file instead of a live camera.",
    )
    parser.add_argument(
        "--stream",
        help="Read from a ZED network stream, as IP or IP:PORT.",
    )
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
        help="Camera resolution; default: HD720.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS; default: 30.",
    )
    parser.add_argument(
        "--depth-mode",
        default="NEURAL",
        choices=["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS"],
        help="ZED depth mode; default: NEURAL.",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=zed_utils.DEFAULT_DEPTH_MIN_M,
        help=f"SDK minimum depth distance in meters; default: {zed_utils.DEFAULT_DEPTH_MIN_M}.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=zed_utils.DEFAULT_DEPTH_MAX_M,
        help=f"SDK maximum depth distance in meters; default: {zed_utils.DEFAULT_DEPTH_MAX_M}.",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=95,
        help="Runtime confidence threshold, 0-100; default: 95.",
    )
    parser.add_argument(
        "--texture-confidence",
        type=int,
        default=100,
        help="Runtime texture confidence threshold, 0-100; default: 100.",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Enable SDK depth fill mode.",
    )
    parser.add_argument(
        "--tag-family",
        default=DEFAULT_TAG_FAMILY,
        choices=sorted(
            list(APRILTAG_DICTIONARIES)
            + [f"tag{name}" for name in APRILTAG_DICTIONARIES]
        ),
        help=f"AprilTag family; default: {DEFAULT_TAG_FAMILY}.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=DEFAULT_TAG_SIZE_M,
        help=(
            "Physical tag edge length in meters. The default is 0.085 "
            "for an 85 mm tag."
        ),
    )
    parser.add_argument(
        "--sample-shrink",
        type=float,
        default=0.35,
        help=(
            "Shrink the tag polygon before sampling depth so edges/background "
            "are ignored. 0 uses the full tag; default: 0.35."
        ),
    )
    parser.add_argument(
        "--center-radius",
        type=int,
        default=5,
        help=(
            "Fallback depth sample radius around the tag center in pixels; "
            "default: 5."
        ),
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=15,
        help="Minimum valid ZED depth pixels required for a tag measurement.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.25,
        help="Seconds between console measurement prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=zed_utils.DEFAULT_DISPLAY_WIDTH,
        help=(
            "Maximum displayed window image width. Use 0 for native size. "
            f"Default: {zed_utils.DEFAULT_DISPLAY_WIDTH}."
        ),
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV window; print measurements only.",
    )
    parser.add_argument(
        "--test-duration",
        type=float,
        default=DEFAULT_TEST_DURATION_S,
        help=f"Seconds to collect samples after pressing s; default: {DEFAULT_TEST_DURATION_S}.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.depth_min <= 0:
        parser.error("--depth-min must be positive")
    if args.depth_max <= 0:
        parser.error("--depth-max must be positive")
    if args.depth_min >= args.depth_max:
        parser.error("--depth-min must be less than --depth-max")
    if not 0 <= args.confidence <= 100:
        parser.error("--confidence must be in 0..100")
    if not 0 <= args.texture_confidence <= 100:
        parser.error("--texture-confidence must be in 0..100")
    if args.tag_size is not None and args.tag_size <= 0:
        parser.error("--tag-size must be positive")
    if not 0 <= args.sample_shrink < 1:
        parser.error("--sample-shrink must be in [0, 1)")
    if args.center_radius < 0:
        parser.error("--center-radius must be >= 0")
    if args.min_valid_pixels <= 0:
        parser.error("--min-valid-pixels must be positive")
    if args.print_every < 0:
        parser.error("--print-every must be >= 0")
    if args.display_width < 0:
        parser.error("--display-width must be >= 0")
    if args.test_duration <= 0:
        parser.error("--test-duration must be positive")

    selected_inputs = sum(
        value is not None
        for value in (args.camera_id, args.serial, args.svo, args.stream)
    )
    if selected_inputs > 1:
        parser.error("choose only one of --camera-id, --serial, --svo, or --stream")

    return args


def normalize_tag_family(tag_family):
    return tag_family[3:] if tag_family.startswith("tag") else tag_family


def make_detector(tag_family):
    tag_family = normalize_tag_family(tag_family)
    dictionary_name = next(
        name for name in APRILTAG_DICTIONARIES[tag_family] if hasattr(cv2.aruco, name)
    )
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters), dictionary, parameters
    return None, dictionary, parameters


def detect_tags(gray, detector, dictionary, parameters):
    if detector is not None:
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return [], []

    return corners, ids.flatten().astype(int)


def xyz_from_zed_mat(xyz_mat):
    xyz = xyz_mat.get_data()
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise RuntimeError(f"Unsupported ZED XYZ shape: {xyz.shape}")
    return xyz[:, :, :3].astype(np.float32, copy=False)


def tag_sample_mask(shape, corners, shrink):
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    if shrink > 0:
        points = center + (points - center) * (1.0 - shrink)

    h, w = shape
    x_min = max(int(np.floor(points[:, 0].min())), 0)
    y_min = max(int(np.floor(points[:, 1].min())), 0)
    x_max = min(int(np.ceil(points[:, 0].max())) + 1, w)
    y_max = min(int(np.ceil(points[:, 1].max())) + 1, h)
    if x_min >= x_max or y_min >= y_max:
        return None, None

    local_points = np.round(points - np.array([x_min, y_min])).astype(np.int32)
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_points, 255)
    return mask.astype(bool), (x_min, y_min, x_max, y_max)


def center_sample_mask(shape, center_px, radius):
    h, w = shape
    cx, cy = center_px
    x_min = max(cx - radius, 0)
    y_min = max(cy - radius, 0)
    x_max = min(cx + radius + 1, w)
    y_max = min(cy + radius + 1, h)
    if x_min >= x_max or y_min >= y_max:
        return None, None
    mask = np.ones((y_max - y_min, x_max - x_min), dtype=bool)
    return mask, (x_min, y_min, x_max, y_max)


def sample_xyz(xyz_m, depth_m, corners, center_px, args):
    mask, bounds = tag_sample_mask(depth_m.shape, corners, args.sample_shrink)
    if mask is None:
        mask, bounds = center_sample_mask(depth_m.shape, center_px, args.center_radius)
    if mask is None:
        return None, None, None, 0

    x_min, y_min, x_max, y_max = bounds
    xyz_roi = xyz_m[y_min:y_max, x_min:x_max]
    depth_roi = depth_m[y_min:y_max, x_min:x_max]

    valid = (
        mask
        & np.isfinite(depth_roi)
        & (depth_roi > 0)
        & np.all(np.isfinite(xyz_roi), axis=2)
        & (xyz_roi[:, :, 2] > 0)
    )

    if int(valid.sum()) < args.min_valid_pixels and args.center_radius > 0:
        mask, bounds = center_sample_mask(depth_m.shape, center_px, args.center_radius)
        if mask is None:
            return None, None, None, int(valid.sum())

        x_min, y_min, x_max, y_max = bounds
        xyz_roi = xyz_m[y_min:y_max, x_min:x_max]
        depth_roi = depth_m[y_min:y_max, x_min:x_max]
        valid = (
            mask
            & np.isfinite(depth_roi)
            & (depth_roi > 0)
            & np.all(np.isfinite(xyz_roi), axis=2)
            & (xyz_roi[:, :, 2] > 0)
        )

    valid_count = int(valid.sum())
    if valid_count < args.min_valid_pixels:
        return None, None, None, valid_count

    xyz_value = np.median(xyz_roi[valid], axis=0)
    depth_value = float(np.median(depth_roi[valid]))
    range_value = float(np.linalg.norm(xyz_value))
    return tuple(float(value) for value in xyz_value), range_value, depth_value, valid_count


def get_left_intrinsics(zed, image_shape):
    info = zed.get_camera_information()
    calibration = info.camera_configuration.calibration_parameters.left_cam
    fx = float(calibration.fx)
    fy = float(calibration.fy)
    cx = float(calibration.cx)
    cy = float(calibration.cy)

    resolution = info.camera_configuration.resolution
    calib_w = float(resolution.width)
    calib_h = float(resolution.height)
    img_h, img_w = image_shape[:2]
    if calib_w > 0 and calib_h > 0 and (calib_w != img_w or calib_h != img_h):
        sx = img_w / calib_w
        sy = img_h / calib_h
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(calibration.disto, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def vector_from_zed(value):
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 3:
        raise RuntimeError(f"Expected a 3D ZED vector, got {value!r}")
    return array[:3]


def get_stereo_center_offset_m(zed):
    calibration = zed.get_camera_information().camera_configuration.calibration_parameters
    try:
        left_to_right_m = vector_from_zed(calibration.stereo_transform.get_translation())
    except Exception:
        baseline_m = float(calibration.get_camera_baseline())
        left_to_right_m = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)

    if not np.all(np.isfinite(left_to_right_m)) or np.linalg.norm(left_to_right_m) <= 0:
        baseline_m = float(calibration.get_camera_baseline())
        left_to_right_m = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)

    return left_to_right_m / 2.0, left_to_right_m


def rotation_matrix_to_euler_deg(rotation_matrix):
    r = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0

    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def rotation_matrix_to_quaternion(rotation_matrix):
    r = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def average_rotation_matrix(rotation_matrices):
    quaternions = []
    for rotation_matrix in rotation_matrices:
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        if quaternions and np.dot(quaternions[0], quaternion) < 0:
            quaternion = -quaternion
        quaternions.append(quaternion)

    if not quaternions:
        return None

    mean_quaternion = np.mean(np.vstack(quaternions), axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    return quaternion_to_rotation_matrix(mean_quaternion)


def estimate_marker_pose(corners, tag_size_m, camera_matrix, dist_coeffs):
    half = tag_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    method = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=method,
    )
    if not success:
        return None, None
    return rvec, tvec.reshape(3)


def measure_tags(corners_list, ids, xyz_m, depth_m, args, pose_context, stereo_center_offset_m):
    measurements = []
    for corners, tag_id in zip(corners_list, ids):
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        center_px = (int(round(center[0])), int(round(center[1])))
        xyz_left_m, _range_left_m, z_value, valid_count = sample_xyz(
            xyz_m,
            depth_m,
            points,
            center_px,
            args,
        )
        xyz_value = None
        range_value = None
        if xyz_left_m is not None:
            xyz_value_array = np.asarray(xyz_left_m, dtype=np.float64) - stereo_center_offset_m
            xyz_value = tuple(float(value) for value in xyz_value_array)
            range_value = float(np.linalg.norm(xyz_value_array))

        pose_tvec = None
        pose_range = None
        rotation_matrix = None
        euler_deg = None
        camera_matrix, dist_coeffs = pose_context
        pose = estimate_marker_pose(
            points,
            args.tag_size,
            camera_matrix,
            dist_coeffs,
        )
        if pose[0] is not None:
            rvec, tvec_left_m = pose
            rotation_matrix, _jacobian = cv2.Rodrigues(rvec)
            euler_deg = rotation_matrix_to_euler_deg(rotation_matrix)
            tvec_m = np.asarray(tvec_left_m, dtype=np.float64) - stereo_center_offset_m
            pose_tvec = tuple(float(value) for value in tvec_m)
            pose_range = float(np.linalg.norm(tvec_m))

        measurements.append(
            TagMeasurement(
                tag_id=int(tag_id),
                corners=points,
                center_px=center_px,
                xyz_m=xyz_value,
                range_m=range_value,
                z_m=z_value,
                valid_pixels=valid_count,
                pose_tvec_m=pose_tvec,
                pose_range_m=pose_range,
                rotation_matrix=rotation_matrix,
                euler_deg=euler_deg,
            )
        )
    return measurements


def draw_filled_rect(image, top_left, bottom_right, color, alpha=1.0):
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
    image,
    text,
    origin,
    scale=0.6,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=2,
    padding=5,
    alpha=0.8,
):
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


def draw_measurements(image, measurements):
    for measurement in measurements:
        points = np.round(measurement.corners).astype(np.int32)
        color = (0, 255, 0) if measurement.range_m is not None else (0, 0, 255)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        cv2.drawMarker(
            image,
            measurement.center_px,
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        label = f"ID {measurement.tag_id}"
        origin = tuple(points[0] + np.array([0, -10]))
        origin = (
            max(8, min(origin[0], image.shape[1] - 20)),
            max(24, min(origin[1], image.shape[0] - 8)),
        )
        draw_text_box(
            image,
            label,
            origin,
            scale=0.55,
            text_color=(255, 255, 255),
            bg_color=(0, 90, 0) if measurement.range_m is not None else (0, 0, 120),
            thickness=2,
            padding=4,
            alpha=0.85,
        )


def measurement_row_text(measurement):
    if measurement.range_m is None or measurement.xyz_m is None:
        return f"{measurement.tag_id:>3}   no depth ({measurement.valid_pixels}px)"

    x_m, y_m, z_m = measurement.xyz_m
    if measurement.euler_deg is None:
        rpy_text = "rpy   --    --    --"
    else:
        roll, pitch, yaw = measurement.euler_deg
        rpy_text = f"rpy {roll:5.0f} {pitch:5.0f} {yaw:5.0f}"

    return (
        f"{measurement.tag_id:>3}  "
        f"d {measurement.range_m:5.3f}  "
        f"x {x_m:6.3f} y {y_m:6.3f} z {z_m:6.3f}  "
        f"{rpy_text}"
    )


def make_measurement_panel(width, measurements, status, status_color, tag_family):
    row_height = 30
    header_height = 90
    visible_rows = min(len(measurements), 12)
    hidden_count = max(0, len(measurements) - visible_rows)
    panel_height = (
        header_height
        + row_height * visible_rows
        + (row_height if hidden_count else 0)
        + 24
    )
    panel = np.full((panel_height, width, 3), (24, 28, 32), dtype=np.uint8)
    draw_measurement_panel(
        panel,
        measurements,
        status,
        status_color,
        tag_family,
        visible_rows,
        hidden_count,
    )
    return panel


def draw_measurement_panel(
    image,
    measurements,
    status,
    status_color,
    tag_family,
    visible_rows,
    hidden_count,
):
    x0 = 12
    y0 = 12
    row_height = 25
    header_height = 88
    visible = sorted(measurements, key=lambda measurement: measurement.tag_id)[:visible_rows]
    panel_height = header_height + row_height * len(visible) + (row_height if hidden_count else 0)

    draw_filled_rect(
        image,
        (x0, y0),
        (image.shape[1] - 12, y0 + panel_height + 12),
        (0, 0, 0),
        alpha=0.55,
    )
    cv2.rectangle(
        image,
        (x0, y0),
        (image.shape[1] - 12, y0 + panel_height + 12),
        (220, 220, 220),
        thickness=1,
    )

    draw_text_box(
        image,
        f"AprilTag {tag_family}: {len(measurements)} detected",
        (x0 + 12, y0 + 28),
        scale=0.7,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness=2,
        padding=1,
        alpha=0.0,
    )
    draw_text_box(
        image,
        status,
        (x0 + 12, y0 + 58),
        scale=0.62,
        text_color=status_color,
        bg_color=(0, 0, 0),
        thickness=2,
        padding=1,
        alpha=0.0,
    )

    header = " ID  dist    x       y       z       roll pitch   yaw"
    draw_text_box(
        image,
        header,
        (x0 + 12, y0 + 84),
        scale=0.6,
        text_color=(210, 230, 255),
        bg_color=(0, 0, 0),
        thickness=1,
        padding=1,
        alpha=0.0,
    )

    y = y0 + header_height + 18
    for measurement in visible:
        row_color = (230, 255, 230) if measurement.range_m is not None else (210, 210, 255)
        draw_text_box(
            image,
            measurement_row_text(measurement),
            (x0 + 12, y),
            scale=0.58,
            text_color=row_color,
            bg_color=(0, 0, 0),
            thickness=1,
            padding=1,
            alpha=0.0,
        )
        y += row_height

    if hidden_count:
        draw_text_box(
            image,
            f"... {hidden_count} more tags hidden",
            (x0 + 12, y),
            scale=0.58,
            text_color=(180, 180, 180),
            bg_color=(0, 0, 0),
            thickness=1,
            padding=1,
            alpha=0.0,
        )


def print_measurements(measurements):
    if not measurements:
        print("tags: none", flush=True)
        return

    parts = []
    for measurement in measurements:
        if measurement.range_m is None or measurement.xyz_m is None:
            parts.append(f"id={measurement.tag_id}: no_depth valid_px={measurement.valid_pixels}")
            continue

        x_m, y_m, z_m = measurement.xyz_m
        text = (
            f"id={measurement.tag_id}: "
            f"x={x_m:.3f} y={y_m:.3f} z={z_m:.3f} "
            f"range={measurement.range_m:.3f}m "
            f"valid_px={measurement.valid_pixels}"
        )
        if measurement.euler_deg is not None:
            roll, pitch, yaw = measurement.euler_deg
            text += f" rpy={roll:.1f},{pitch:.1f},{yaw:.1f}deg"
        parts.append(text)

    print(" | ".join(parts), flush=True)


def measurement_to_record(measurement, frame_index, elapsed_s):
    record = {
        "frame_index": int(frame_index),
        "time_s": float(elapsed_s),
        "tag_id": int(measurement.tag_id),
        "center_x_px": int(measurement.center_px[0]),
        "center_y_px": int(measurement.center_px[1]),
        "valid_depth_pixels": int(measurement.valid_pixels),
        "position_m": None,
        "distance_m": None,
        "z_depth_m": None,
        "pose_position_m": None,
        "pose_distance_m": None,
        "rotation_matrix": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
    }

    if measurement.xyz_m is not None:
        record["position_m"] = [float(value) for value in measurement.xyz_m]
        record["distance_m"] = float(measurement.range_m)
        record["z_depth_m"] = float(measurement.z_m)

    if measurement.pose_tvec_m is not None:
        record["pose_position_m"] = [float(value) for value in measurement.pose_tvec_m]
        record["pose_distance_m"] = float(measurement.pose_range_m)

    if measurement.rotation_matrix is not None:
        record["rotation_matrix"] = np.asarray(
            measurement.rotation_matrix,
            dtype=float,
        ).tolist()

    if measurement.euler_deg is not None:
        roll, pitch, yaw = measurement.euler_deg
        record["roll_deg"] = float(roll)
        record["pitch_deg"] = float(pitch)
        record["yaw_deg"] = float(yaw)

    return record


def sample_records_from_measurements(measurements, frame_index, elapsed_s):
    return [
        measurement_to_record(measurement, frame_index, elapsed_s)
        for measurement in measurements
    ]


def mean_std(values):
    values = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if values.size == 0:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def summarize_tag_records(records, frames_total):
    tag_ids = sorted({int(record["tag_id"]) for record in records})
    summaries = []
    for tag_id in tag_ids:
        tag_records = [record for record in records if int(record["tag_id"]) == tag_id]
        position_records = [
            record for record in tag_records if record["position_m"] is not None
        ]
        rotation_records = [
            record for record in tag_records if record["rotation_matrix"] is not None
        ]

        distances = [record["distance_m"] for record in position_records]
        z_depths = [record["z_depth_m"] for record in position_records]
        distance_mean, distance_std = mean_std(distances)
        z_mean, z_std = mean_std(z_depths)

        position_mean = None
        position_std = None
        if position_records:
            positions = np.asarray(
                [record["position_m"] for record in position_records],
                dtype=np.float64,
            )
            position_mean = np.mean(positions, axis=0).tolist()
            position_std = np.std(positions, axis=0).tolist()

        pose_position_mean = None
        pose_position_std = None
        pose_records = [
            record for record in tag_records if record["pose_position_m"] is not None
        ]
        if pose_records:
            pose_positions = np.asarray(
                [record["pose_position_m"] for record in pose_records],
                dtype=np.float64,
            )
            pose_position_mean = np.mean(pose_positions, axis=0).tolist()
            pose_position_std = np.std(pose_positions, axis=0).tolist()

        rotation_matrix_mean = None
        euler_mean = None
        euler_std = None
        if rotation_records:
            rotations = [
                np.asarray(record["rotation_matrix"], dtype=np.float64)
                for record in rotation_records
            ]
            rotation_matrix_mean = average_rotation_matrix(rotations)
            euler_mean = rotation_matrix_to_euler_deg(rotation_matrix_mean)
            euler_values = np.asarray(
                [
                    [record["roll_deg"], record["pitch_deg"], record["yaw_deg"]]
                    for record in rotation_records
                    if record["roll_deg"] is not None
                ],
                dtype=np.float64,
            )
            if euler_values.size:
                euler_std = np.std(euler_values, axis=0).tolist()

        summaries.append(
            {
                "tag_id": tag_id,
                "detections": len(tag_records),
                "valid_distance_measurements": len(position_records),
                "valid_rotation_measurements": len(rotation_records),
                "detection_rate_per_frame": (
                    len(tag_records) / frames_total if frames_total > 0 else None
                ),
                "distance_m_mean": distance_mean,
                "distance_m_std": distance_std,
                "z_depth_m_mean": z_mean,
                "z_depth_m_std": z_std,
                "position_m_mean": position_mean,
                "position_m_std": position_std,
                "pose_position_m_mean": pose_position_mean,
                "pose_position_m_std": pose_position_std,
                "rotation_matrix_mean": (
                    rotation_matrix_mean.tolist()
                    if rotation_matrix_mean is not None
                    else None
                ),
                "rotation_rpy_deg_mean": (
                    {
                        "roll": float(euler_mean[0]),
                        "pitch": float(euler_mean[1]),
                        "yaw": float(euler_mean[2]),
                    }
                    if euler_mean is not None
                    else None
                ),
                "rotation_rpy_deg_std": (
                    {
                        "roll": float(euler_std[0]),
                        "pitch": float(euler_std[1]),
                        "yaw": float(euler_std[2]),
                    }
                    if euler_std is not None
                    else None
                ),
            }
        )

    return summaries


def test_output_directory(now):
    stem = f"{now.strftime('%Y%m%d_%H%M%S')}_apriltag_test"
    return zed_utils.session_output_directory(stem)


def csv_value(value):
    if value is None:
        return ""
    return value


def write_samples_csv(path, records):
    fields = [
        "frame_index",
        "time_s",
        "tag_id",
        "center_x_px",
        "center_y_px",
        "valid_depth_pixels",
        "x_m",
        "y_m",
        "z_m",
        "distance_m",
        "z_depth_m",
        "pose_x_m",
        "pose_y_m",
        "pose_z_m",
        "pose_distance_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            position = record["position_m"] or [None, None, None]
            pose_position = record["pose_position_m"] or [None, None, None]
            writer.writerow(
                {
                    "frame_index": record["frame_index"],
                    "time_s": f"{record['time_s']:.6f}",
                    "tag_id": record["tag_id"],
                    "center_x_px": record["center_x_px"],
                    "center_y_px": record["center_y_px"],
                    "valid_depth_pixels": record["valid_depth_pixels"],
                    "x_m": csv_value(position[0]),
                    "y_m": csv_value(position[1]),
                    "z_m": csv_value(position[2]),
                    "distance_m": csv_value(record["distance_m"]),
                    "z_depth_m": csv_value(record["z_depth_m"]),
                    "pose_x_m": csv_value(pose_position[0]),
                    "pose_y_m": csv_value(pose_position[1]),
                    "pose_z_m": csv_value(pose_position[2]),
                    "pose_distance_m": csv_value(record["pose_distance_m"]),
                    "roll_deg": csv_value(record["roll_deg"]),
                    "pitch_deg": csv_value(record["pitch_deg"]),
                    "yaw_deg": csv_value(record["yaw_deg"]),
                }
            )


def write_summary_csv(path, tag_summaries):
    fields = [
        "tag_id",
        "avg_x_m",
        "avg_y_m",
        "avg_z_m",
        "avg_roll_deg",
        "avg_pitch_deg",
        "avg_yaw_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for summary in tag_summaries:
            position = summary["position_m_mean"] or [None, None, None]
            rpy = summary["rotation_rpy_deg_mean"] or {}
            writer.writerow(
                {
                    "tag_id": summary["tag_id"],
                    "avg_x_m": csv_value(position[0]),
                    "avg_y_m": csv_value(position[1]),
                    "avg_z_m": csv_value(position[2]),
                    "avg_roll_deg": csv_value(rpy.get("roll")),
                    "avg_pitch_deg": csv_value(rpy.get("pitch")),
                    "avg_yaw_deg": csv_value(rpy.get("yaw")),
                }
            )


def set_axes_equal(ax, points):
    points = np.asarray(points, dtype=np.float64)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def write_3d_plot_matplotlib(path, tag_summaries):
    tag_points = [
        (
            summary["tag_id"],
            np.asarray(summary["position_m_mean"], dtype=np.float64),
            summary["distance_m_mean"],
        )
        for summary in tag_summaries
        if summary["position_m_mean"] is not None
    ]
    if not tag_points:
        path.with_suffix(".txt").write_text(
            "No valid average tag positions were recorded, so no 3D plot was generated.\n",
            encoding="utf-8",
        )
        return False

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        path.with_suffix(".txt").write_text(
            "matplotlib is required to generate plot_3d.png.\n"
            "Install it with: python3 -m pip install matplotlib\n"
            f"Import error: {exc}\n",
            encoding="utf-8",
        )
        return False

    fig = plt.figure(figsize=(9, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Average AprilTag Positions from ZED2 Stereo Center")

    all_points = [np.zeros(3, dtype=np.float64)]
    for tag_id, position, distance_m in tag_points:
        all_points.append(position)
        ax.scatter(
            position[0],
            position[1],
            position[2],
            s=90,
            depthshade=True,
            label=f"ID {tag_id}",
        )
        ax.text(
            position[0],
            position[1],
            position[2],
            f"  ID {tag_id}\n  {distance_m:.3f} m",
            fontsize=8,
        )

    ax.scatter(0, 0, 0, c="black", marker="x", s=120, label="ZED2 center")
    ax.text(0, 0, 0, "  ZED2 center", fontsize=9)

    axis_len = max(0.25, max(np.linalg.norm(point) for point in all_points) * 0.35)
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", arrow_length_ratio=0.12)
    ax.text(axis_len, 0, 0, "+X right", color="r")
    ax.text(0, axis_len, 0, "+Y down", color="g")
    ax.text(0, 0, axis_len, "+Z forward", color="b")

    set_axes_equal(ax, all_points)
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Y down (m)")
    ax.set_zlabel("Z forward (m)")
    ax.view_init(elev=22, azim=-55)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_3d_plot_html(path, tag_summaries):
    tags = [
        {
            "tag_id": int(summary["tag_id"]),
            "position_m": [float(value) for value in summary["position_m_mean"]],
            "distance_m": (
                float(summary["distance_m_mean"])
                if summary["distance_m_mean"] is not None
                else None
            ),
            "rotation_rpy_deg": summary["rotation_rpy_deg_mean"],
        }
        for summary in tag_summaries
        if summary["position_m_mean"] is not None
    ]
    data_json = json.dumps({"tags": tags}, indent=2)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ZED2 AprilTag Average Positions</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #11161c;
      color: #e8eef4;
      font-family: Arial, sans-serif;
    }}
    header {{
      box-sizing: border-box;
      height: 74px;
      padding: 12px 16px;
      border-bottom: 1px solid #2e3843;
      background: #0d1116;
    }}
    #title {{ font-size: 18px; font-weight: 700; }}
    #hint {{ margin-top: 5px; font-size: 13px; color: #aab7c4; }}
    #wrap {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      height: calc(100% - 74px);
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background: #151b21;
    }}
    aside {{
      box-sizing: border-box;
      padding: 12px;
      border-left: 1px solid #2e3843;
      background: #0f141a;
      overflow: auto;
      font-size: 13px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 5px 4px; border-bottom: 1px solid #26313b; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    button {{
      margin-bottom: 10px;
      padding: 6px 10px;
      background: #243447;
      color: #e8eef4;
      border: 1px solid #486079;
      border-radius: 4px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
<header>
  <div id="title">ZED2 AprilTag Average Positions</div>
  <div id="hint">Drag to rotate, wheel to zoom, double-click to reset. One point is plotted per AprilTag ID. Units are meters from the ZED2 stereo center.</div>
</header>
<div id="wrap">
  <canvas id="plot"></canvas>
  <aside>
    <button id="reset">Reset view</button>
    <table id="tagTable"></table>
  </aside>
</div>
<script>
const DATA = {data_json};
</script>
"""
    html += r"""<script>
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const table = document.getElementById("tagTable");
const colors = ["#4da3ff", "#ffcc00", "#35c759", "#ff6b6b", "#b18cff", "#4dd8c8", "#ff9f40", "#d7f75b"];
let rotX = -0.55;
let rotY = 0.72;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;

function resetView() {
  rotX = -0.55;
  rotY = 0.72;
  zoom = 1.0;
  draw();
}

function resize() {
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * window.devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * window.devicePixelRatio));
  draw();
}

function sceneBounds() {
  const pts = [[0, 0, 0]].concat(DATA.tags.map(tag => tag.position_m));
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  pts.forEach(p => {
    for (let i = 0; i < 3; i++) {
      mins[i] = Math.min(mins[i], p[i]);
      maxs[i] = Math.max(maxs[i], p[i]);
    }
  });
  const center = mins.map((v, i) => (v + maxs[i]) / 2);
  const radius = Math.max(0.25, ...mins.map((v, i) => maxs[i] - v)) / 2;
  return {center, radius};
}

function rotate(point) {
  const b = sceneBounds();
  let x = point[0] - b.center[0];
  let y = point[1] - b.center[1];
  let z = point[2] - b.center[2];
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  const y1 = y * cx - z * sx;
  const z1 = y * sx + z * cx;
  const x2 = x * cy + z1 * sy;
  const z2 = -x * sy + z1 * cy;
  return [x2, y1, z2];
}

function project(point) {
  const b = sceneBounds();
  const p = rotate(point);
  const scale = Math.min(canvas.width, canvas.height) * 0.42 * zoom / b.radius;
  return [
    canvas.width / 2 + p[0] * scale,
    canvas.height / 2 + p[1] * scale,
    p[2]
  ];
}

function line(a, b, color, width = 2) {
  const pa = project(a);
  const pb = project(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = width * window.devicePixelRatio;
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.stroke();
  return pb;
}

function label(text, p, color = "#e8eef4", align = "left") {
  ctx.fillStyle = color;
  ctx.font = `${13 * window.devicePixelRatio}px Arial`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, p[0], p[1]);
}

function drawGrid() {
  const b = sceneBounds();
  const step = niceStep(b.radius / 3);
  const limit = Math.ceil(b.radius / step) * step;
  ctx.strokeStyle = "#27333f";
  ctx.lineWidth = window.devicePixelRatio;
  for (let v = -limit; v <= limit + 1e-9; v += step) {
    line([-limit, 0, v], [limit, 0, v], "#27333f", 1);
    line([v, 0, -limit], [v, 0, limit], "#27333f", 1);
  }
}

function niceStep(value) {
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(value, 1e-6))));
  const n = value / pow;
  if (n < 1.5) return pow;
  if (n < 3.5) return 2 * pow;
  if (n < 7.5) return 5 * pow;
  return 10 * pow;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!DATA.tags.length) {
    label("No valid average tag positions were recorded.", [24 * window.devicePixelRatio, 36 * window.devicePixelRatio]);
    return;
  }
  drawGrid();
  const maxDistance = Math.max(0.25, ...DATA.tags.map(tag => Math.hypot(...tag.position_m)));
  const axis = maxDistance * 0.35;
  let p;
  p = line([0, 0, 0], [axis, 0, 0], "#ff6b6b", 3);
  label("+X right", [p[0] + 8, p[1]], "#ff6b6b");
  p = line([0, 0, 0], [0, axis, 0], "#35c759", 3);
  label("+Y down", [p[0] + 8, p[1]], "#35c759");
  p = line([0, 0, 0], [0, 0, axis], "#4da3ff", 3);
  label("+Z forward", [p[0] + 8, p[1]], "#4da3ff");

  const origin = project([0, 0, 0]);
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(origin[0], origin[1], 5 * window.devicePixelRatio, 0, Math.PI * 2);
  ctx.fill();
  label("ZED2 center", [origin[0] + 9, origin[1] - 10], "#ffffff");

  const sorted = DATA.tags
    .map(tag => ({tag, p: project(tag.position_m)}))
    .sort((a, b) => a.p[2] - b.p[2]);
  sorted.forEach(({tag, p}, index) => {
    const color = colors[Math.abs(tag.tag_id) % colors.length];
    line([0, 0, 0], tag.position_m, color + "99", 1.5);
    ctx.fillStyle = color;
    ctx.strokeStyle = "#0d1116";
    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 9 * window.devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    const distance = tag.distance_m == null ? Math.hypot(...tag.position_m) : tag.distance_m;
    label(`ID ${tag.tag_id}  ${distance.toFixed(3)} m`, [p[0] + 12, p[1] - 12], color);
  });
}

function fillTable() {
  table.innerHTML = "<tr><th>ID</th><th>X</th><th>Y</th><th>Z</th><th>D</th></tr>" +
    DATA.tags.map(tag => {
      const p = tag.position_m;
      const d = tag.distance_m == null ? Math.hypot(...p) : tag.distance_m;
      return `<tr><td>${tag.tag_id}</td><td>${p[0].toFixed(3)}</td><td>${p[1].toFixed(3)}</td><td>${p[2].toFixed(3)}</td><td>${d.toFixed(3)}</td></tr>`;
    }).join("");
}

canvas.addEventListener("mousedown", event => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
});
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", event => {
  if (!dragging) return;
  rotY += (event.clientX - lastX) * 0.01;
  rotX += (event.clientY - lastY) * 0.01;
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
});
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  zoom *= event.deltaY < 0 ? 1.1 : 0.9;
  zoom = Math.max(0.2, Math.min(8, zoom));
  draw();
}, {passive: false});
canvas.addEventListener("dblclick", resetView);
document.getElementById("reset").addEventListener("click", resetView);
window.addEventListener("resize", resize);
fillTable();
resize();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def save_test_results(
    records,
    frames_total,
    started_at,
    ended_at,
    args,
    camera_name,
    left_to_right_m,
    stereo_center_offset_m,
):
    now = datetime.now()
    output_dir = test_output_directory(now)
    duration_s = max(0.0, ended_at - started_at)
    tag_summaries = summarize_tag_records(records, frames_total)
    frames_with_detections = len({record["frame_index"] for record in records})

    summary = {
        "created_at": now.isoformat(timespec="seconds"),
        "camera": camera_name,
        "tag_family": args.tag_family,
        "tag_size_m": args.tag_size,
        "test_duration_requested_s": args.test_duration,
        "test_duration_actual_s": duration_s,
        "frames_total": int(frames_total),
        "frames_with_detections": int(frames_with_detections),
        "total_detections": int(len(records)),
        "coordinate_frame": {
            "origin": "midpoint between the ZED left and right camera optical centers",
            "axes": "+X right, +Y down, +Z forward",
            "units": "meters",
            "left_to_right_vector_m": [float(value) for value in left_to_right_m],
            "left_camera_to_stereo_center_offset_m": [
                float(value) for value in stereo_center_offset_m
            ],
            "note": (
                "ZED XYZ is sampled in the rectified left-camera frame, then "
                "left_camera_to_stereo_center_offset_m is subtracted."
            ),
        },
        "tags": tag_summaries,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "samples.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    write_samples_csv(output_dir / "samples.csv", records)
    write_summary_csv(output_dir / "summary.csv", tag_summaries)
    write_3d_plot_matplotlib(output_dir / "plot_3d.png", tag_summaries)
    write_3d_plot_html(output_dir / "plot_3d.html", tag_summaries)
    return output_dir, summary


def main():
    args = parse_args()
    if args.list:
        zed_utils.print_camera_list()
        return 0

    zed = None
    try:
        zed = zed_utils.open_camera(args)
        info = zed.get_camera_information()
        name = zed_utils.camera_label(info)
        stereo_center_offset_m, left_to_right_m = get_stereo_center_offset_m(zed)
        print(f"Connected: {name}")
        print(
            "Distance uses ZED XYZ/depth in meters. "
            f"Press s to start a {args.test_duration:.1f} second test, q to quit."
            if not args.no_window
            else "Distance uses ZED XYZ/depth in meters. Press Ctrl+C to quit."
        )
        print(
            "Reference frame: stereo center, "
            f"left-to-right vector = {left_to_right_m.tolist()} m"
        )

        runtime = zed_utils.make_runtime_parameters(args)
        image_mat = sl.Mat()
        depth_mat = sl.Mat()
        xyz_mat = sl.Mat()
        detector, dictionary, parameters = make_detector(args.tag_family)
        pose_context = None
        last_print = 0.0
        frame_index = 0
        test_active = False
        test_started_at = None
        test_records = []
        test_frames_total = 0
        saved_message = None

        if not args.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while True:
            error = zed.grab(runtime)
            if error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if error != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT_BGR, sl.MEM.CPU)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU)
            zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ, sl.MEM.CPU)

            rgb = zed_utils.bgr_from_zed_mat(image_mat)
            depth_m = zed_utils.depth_from_zed_mat(depth_mat)
            xyz_m = xyz_from_zed_mat(xyz_mat)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            corners, ids = detect_tags(gray, detector, dictionary, parameters)

            if pose_context is None:
                pose_context = get_left_intrinsics(zed, rgb.shape)

            measurements = measure_tags(
                corners,
                ids,
                xyz_m,
                depth_m,
                args,
                pose_context,
                stereo_center_offset_m,
            )

            now = time.monotonic()
            if test_active:
                elapsed_s = now - test_started_at
                test_frames_total += 1
                test_records.extend(
                    sample_records_from_measurements(
                        measurements,
                        frame_index,
                        elapsed_s,
                    )
                )
                if elapsed_s >= args.test_duration:
                    output_dir, _summary = save_test_results(
                        test_records,
                        test_frames_total,
                        test_started_at,
                        now,
                        args,
                        name,
                        left_to_right_m,
                        stereo_center_offset_m,
                    )
                    saved_message = (
                        "Saved: "
                        f"{output_dir}"
                    )
                    print(f"Saved test: {output_dir}")
                    test_active = False
                    test_started_at = None
                    test_records = []
                    test_frames_total = 0

            if args.print_every > 0 and now - last_print >= args.print_every:
                print_measurements(measurements)
                last_print = now

            if args.no_window:
                frame_index += 1
                continue

            display = rgb.copy()
            draw_measurements(display, measurements)
            status = "Press s to start test"
            status_color = (255, 255, 255)
            if test_active:
                remaining_s = max(0.0, args.test_duration - (now - test_started_at))
                status = (
                    f"TEST {remaining_s:.1f}s left, "
                    f"samples {len(test_records)}"
                )
                status_color = (0, 255, 255)
            elif saved_message:
                status = saved_message
                status_color = (0, 255, 0)
            panel = make_measurement_panel(
                display.shape[1],
                measurements,
                status,
                status_color,
                args.tag_family,
            )
            display = np.vstack((display, panel))
            display_scale = zed_utils.get_display_scale(
                display.shape,
                args.display_width,
            )
            cv2.imshow(WINDOW_NAME, zed_utils.resize_for_display(display, display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")) and not test_active:
                test_active = True
                test_started_at = time.monotonic()
                test_records = []
                test_frames_total = 0
                saved_message = None
                print(
                    f"AprilTag test started for {args.test_duration:.1f} seconds."
                )
            if key == ord("q"):
                break
            frame_index += 1
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
