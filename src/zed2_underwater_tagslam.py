#!/usr/bin/env python3
"""
Underwater ZED2 AprilTag SLAM without ROS.

Install core Python dependencies:
    python3 -m pip install opencv-python pupil-apriltags gtsam numpy

GTSAM only:
    python3 -m pip install gtsam

The ZED2 camera input uses Stereolabs' pyzed Python module, which is installed
from the ZED SDK/wheel rather than PyPI on most Linux systems.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pyzed.sl as sl

from tagslam_core import (
    CameraIntrinsics,
    DEFAULT_AIR_REFRACTIVE_INDEX,
    DEFAULT_ANCHOR_TAG_ID,
    DEFAULT_TAG_SIZE_M,
    DEFAULT_WATER_REFRACTIVE_INDEX,
    PLOT_Z_SCALE,
    RefractiveContext,
    TagSlamBackend,
    TrajectoryRecorder,
    detect_observations,
    draw_observations,
    draw_overlay,
    get_display_scale,
    make_detector,
    normalize_vector,
    normalize_water_config,
    parse_simple_yaml,
    pose_rpy,
    pose_translation,
    print_backend_update,
    project_refractive,
    refine_refractive_pose_lm,
    resize_for_display,
    rotation_error_deg,
    solve_refractive_pose_fixed_point,
    solve_refractive_pose_fixed_point_batch,
    tag_object_points,
    transform_object_points,
)


WATER_SCALE_FACTOR = 3.6
DEFAULT_POOL_DEPTH_M = 1.143
WINDOW_NAME = "Underwater ZED2 TagSLAM"
TRAJECTORY_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AprilTag SLAM for a ZED2 left camera using pupil-apriltags, "
            "OpenCV solvePnP, and an incremental GTSAM iSAM2 backend."
        )
    )
    parser.add_argument("--list", action="store_true", help="List connected ZED cameras and exit.")
    parser.add_argument(
        "--refractive-self-test",
        action="store_true",
        help="Run the synthetic flat-interface refractive-PnP round-trip test and exit.",
    )
    parser.add_argument(
        "--refractive-regression-check",
        action="store_true",
        help="Compare the fast refractive solver against the reference LM solver on synthetic poses and exit.",
    )
    parser.add_argument(
        "--refractive-benchmark",
        action="store_true",
        help="Run a synthetic 20-tag refractive solver timing benchmark and exit.",
    )
    parser.add_argument("--camera-id", type=int, help="Open a specific local ZED camera ID.")
    parser.add_argument("--serial", type=int, help="Open a specific ZED serial number.")
    parser.add_argument("--svo", help="Read a ZED SVO file instead of a live camera.")
    parser.add_argument("--stream", help="Read a ZED network stream as IP or IP:PORT.")
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=["HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA"],
        help="ZED camera resolution.",
    )
    parser.add_argument("--fps", type=int, default=30, help="ZED camera FPS.")
    parser.add_argument("--tag-family", default="tag36h11", help="AprilTag family.")
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        help="Physical AprilTag edge length in meters.",
    )
    parser.add_argument(
        "--anchor-tag-id",
        type=int,
        default=DEFAULT_ANCHOR_TAG_ID,
        help="AprilTag ID hard-pinned as the SLAM world-frame origin.",
    )
    parser.add_argument(
        "--max-tag-id",
        type=int,
        default=-1,
        help=(
            "Ignore detections above this ID. Default -1 accepts all IDs; "
            "set 7 to use only tags 0 through 7."
        ),
    )
    parser.add_argument(
        "--water-scale",
        type=float,
        default=WATER_SCALE_FACTOR,
        help=(
            "Empirical translation scale for scalar/trust-region water correction. "
            "Calibrate this from a tape-measured near-nadir static test."
        ),
    )
    parser.add_argument(
        "--water-correction-mode",
        choices=["none", "scalar", "trust-region", "refractive"],
        default="refractive",
        help=(
            "Underwater correction mode. 'refractive' refines pose through the "
            "config-defined flat air-water interface; 'trust-region' filters "
            "oblique detections then applies --water-scale; 'scalar' applies "
            "scale to every detection."
        ),
    )
    parser.add_argument(
        "--surface-distance-m",
        type=float,
        default=0.20,
        help="Legacy approximate air gap used only as a refractive bootstrap fallback.",
    )
    parser.add_argument(
        "--water-refractive-index",
        type=float,
        default=1.333,
        help="Legacy water refractive index metadata; refractive mode reads config water.n_water.",
    )
    parser.add_argument(
        "--refractive-max-iterations",
        type=int,
        default=8,
        help=(
            "Maximum fixed-point solvePnP iterations for refractive mode. "
            "Higher is safer for handheld scenes where the initial in-air pose "
            "is far from the true underwater pose; raise this if you see a "
            "high refractive fallback ratio in the runtime log."
        ),
    )
    parser.add_argument(
        "--refractive-convergence-tol-m",
        type=float,
        default=1e-5,
        help="Translation convergence tolerance in meters for refractive fixed-point PnP.",
    )
    parser.add_argument(
        "--refractive-convergence-tol-deg",
        type=float,
        default=0.01,
        help="Rotation convergence tolerance in degrees for refractive fixed-point PnP.",
    )
    parser.add_argument(
        "--refractive-ray-max-iterations",
        type=int,
        default=10,
        help="Maximum Newton iterations for each flat-interface refraction-point solve.",
    )
    parser.add_argument(
        "--refractive-ray-tol",
        type=float,
        default=1e-11,
        help="Convergence tolerance for the flat-interface refraction-point Newton solve.",
    )
    parser.add_argument(
        "--min-tag-area-px",
        type=float,
        default=120.0,
        help="Reject detections whose image quadrilateral area is below this pixel area.",
    )
    parser.add_argument(
        "--max-off-nadir-deg",
        type=float,
        default=25.0,
        help="Trust-region limit for tag center ray angle from the camera optical axis.",
    )
    parser.add_argument(
        "--max-image-eccentricity",
        type=float,
        default=0.65,
        help="Reject detections too far from the principal point, normalized by image half diagonal.",
    )
    parser.add_argument(
        "--max-tag-tilt-deg",
        type=float,
        default=35.0,
        help="Reject detections whose estimated tag normal is too oblique to the camera optical axis.",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=5.0,
        help="Reject detections whose solvePnP corner reprojection RMS exceeds this value.",
    )
    parser.add_argument("--nthreads", type=int, default=2, help="pupil-apriltags worker threads.")
    parser.add_argument(
        "--quad-decimate",
        type=float,
        default=1.0,
        help="AprilTag quad decimation. 1.0 keeps full image resolution.",
    )
    parser.add_argument(
        "--quad-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for AprilTag segmentation.",
    )
    parser.add_argument(
        "--decode-sharpening",
        type=float,
        default=0.25,
        help="AprilTag decode sharpening parameter.",
    )
    parser.add_argument(
        "--min-decision-margin",
        type=float,
        default=30,
        help="Reject detections below this decision margin.",
    )
    parser.add_argument(
        "--max-hamming",
        type=int,
        default=0,
        help="Reject detections with more corrected bits than this.",
    )
    parser.add_argument(
        "--tag-rot-sigma",
        type=float,
        default=0.08,
        help="Camera-tag rotation sigma in radians for BetweenFactorPose3.",
    )
    parser.add_argument(
        "--tag-trans-sigma",
        type=float,
        default=0.04,
        help="Camera-tag translation sigma in meters for BetweenFactorPose3.",
    )
    parser.add_argument(
        "--tag-robust-kernel",
        choices=["none", "huber", "cauchy", "tukey"],
        default="huber",
        help="Robust m-estimator used around camera-to-tag BetweenFactorPose3 noise.",
    )
    parser.add_argument(
        "--tag-robust-threshold",
        type=float,
        default=1.345,
        help="Robust kernel threshold in whitened residual units.",
    )
    parser.add_argument(
        "--tag-init-min-observations",
        type=int,
        default=3,
        help="Require this many accepted detections before a new tag can enter the graph.",
    )
    parser.add_argument(
        "--pose-std-window",
        type=int,
        default=30,
        help="Window length for reporting optimized camera position standard deviation.",
    )
    parser.add_argument(
        "--odom-rot-sigma",
        type=float,
        default=0.35,
        help="Weak constant-velocity rotation sigma in radians.",
    )
    parser.add_argument(
        "--odom-trans-sigma",
        type=float,
        default=0.30,
        help="Weak constant-velocity translation sigma in meters.",
    )
    parser.add_argument(
        "--prior-rot-sigma",
        type=float,
        default=1e-6,
        help="Rotation sigma for the hard prior on the anchor tag.",
    )
    parser.add_argument(
        "--prior-trans-sigma",
        type=float,
        default=1e-6,
        help="Translation sigma in meters for the hard prior on the anchor tag.",
    )
    parser.set_defaults(floor_prior_enabled=True)
    parser.add_argument(
        "--floor-prior-enabled",
        dest="floor_prior_enabled",
        action="store_true",
        help="Enable weak co-planarity prior that keeps tag Z near the anchor-frame floor plane.",
    )
    parser.add_argument(
        "--no-floor-prior",
        dest="floor_prior_enabled",
        action="store_false",
        help="Disable the weak floor co-planarity prior.",
    )
    parser.add_argument(
        "--floor-z-sigma",
        type=float,
        default=0.02,
        help="Z sigma in meters for the weak floor co-planarity prior.",
    )
    parser.add_argument(
        "--floor-plane-min-tags",
        type=int,
        default=4,
        help="Fit and apply the floor plane prior after at least this many tag estimates exist.",
    )
    parser.add_argument(
        "--floor-normal-sigma-deg",
        type=float,
        default=8.0,
        help="Rotation sigma for aligning tag normals to the fitted floor plane normal.",
    )
    parser.set_defaults(strict_coplanar=False)
    parser.add_argument(
        "--strict-coplanar",
        dest="strict_coplanar",
        action="store_true",
        help=(
            "Strongly enforce that all AprilTags lie on a single flat plane. "
            "Overrides --floor-z-sigma and --floor-normal-sigma-deg with very "
            "tight values (2 mm and 1.5 deg). Useful when the tags are known "
            "to be on a flat pool floor and you accept a small reprojection "
            "tradeoff to remove residual tag-height jitter."
        ),
    )
    parser.add_argument(
        "--no-strict-coplanar",
        dest="strict_coplanar",
        action="store_false",
        help="Use the configured --floor-z-sigma / --floor-normal-sigma-deg as-is.",
    )
    parser.add_argument(
        "--floor-prior-refresh-frames",
        type=int,
        default=0,
        help=(
            "If > 0, refit the floor plane and add a fresh prior to every "
            "initialized tag every N camera frames. Helps converge the tag "
            "layout to a single plane when the initial per-tag prior was "
            "added against an early, noisy plane fit. 0 disables refresh "
            "(current behavior)."
        ),
    )
    parser.add_argument(
        "--floor-plane-outlier-threshold",
        type=float,
        default=0.10,
        help="Meters; robust plane fit ignores tag estimates farther than this from the initial plane.",
    )
    parser.set_defaults(use_imu_gravity=True)
    parser.add_argument(
        "--use-imu-gravity",
        dest="use_imu_gravity",
        action="store_true",
        help=(
            "Use the ZED2 built-in IMU gravity vector as the source of truth for "
            "'which way is up'. The refractive interface normal and the floor "
            "co-planarity prior are referenced to true gravity instead of the "
            "possibly tilted anchor frame."
        ),
    )
    parser.add_argument(
        "--no-imu-gravity",
        dest="use_imu_gravity",
        action="store_false",
        help="Disable IMU gravity; refractive normal and floor prior fall back to anchor-frame up.",
    )
    parser.add_argument(
        "--imu-gravity-smoothing-n",
        type=int,
        default=5,
        help=(
            "Average the IMU gravity vector across the last N CAMERA FRAMES. "
            "Sampling is one IMU read per frame, so N is in frames, not samples."
        ),
    )
    parser.set_defaults(gravity_align_world=True)
    parser.add_argument(
        "--gravity-align-world",
        dest="gravity_align_world",
        action="store_true",
        help=(
            "On anchor-init, rotate the world frame so its up axis matches IMU "
            "gravity (roll/pitch only; yaw stays defined by the anchor). The "
            "anchor remains the position datum."
        ),
    )
    parser.add_argument(
        "--no-gravity-align-world",
        dest="gravity_align_world",
        action="store_false",
        help="Keep the original anchor-aligned world frame even when IMU gravity is available.",
    )
    parser.add_argument(
        "--init-min-observations",
        type=int,
        default=3,
        help="Require this many accepted anchor-tag observations before anchoring the world.",
    )
    parser.add_argument(
        "--init-min-decision-margin",
        type=float,
        default=45.0,
        help="Minimum anchor-tag decision margin for world initialization.",
    )
    parser.add_argument(
        "--init-min-tag-area-px",
        type=float,
        default=250.0,
        help="Minimum anchor-tag image area in pixels for world initialization.",
    )
    parser.add_argument(
        "--init-max-off-nadir-deg",
        type=float,
        default=20.0,
        help="Maximum anchor-tag off-nadir angle for world initialization.",
    )
    parser.add_argument(
        "--init-max-image-eccentricity",
        type=float,
        default=0.45,
        help="Maximum anchor-tag normalized image eccentricity for world initialization.",
    )
    parser.add_argument(
        "--init-max-tag-tilt-deg",
        type=float,
        default=25.0,
        help="Maximum anchor-tag normal tilt for world initialization.",
    )
    parser.add_argument(
        "--plot-z-scale",
        type=float,
        default=PLOT_Z_SCALE,
        help="Visual-only Z scale for saved trajectory plots. 0.5 draws Z at half height.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1600,
        help="Maximum display width; 0 displays native width.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="YAML config file for fixed pool geometry and Tag 1 pool-corner metadata.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=str(TRAJECTORY_DIR),
        help=(
            "Root directory for trajectory outputs. Runs are saved as "
            "<root>/YYYYMMDD/<timestamp>_tagslam_trajectory."
        ),
    )
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Start trajectory recording immediately, useful with --no-window or SVO playback.",
    )
    parser.add_argument(
        "--trajectory-image-width",
        type=int,
        default=960,
        help=(
            "Saved ZED-view image width for the interactive trajectory HTML. "
            "Use 0 to save native frame width."
        ),
    )
    parser.add_argument("--no-window", action="store_true", help="Run headless and print poses.")
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Seconds between console pose prints. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames, useful for testing on SVO files.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.anchor_tag_id < 0:
        parser.error("--anchor-tag-id must be >= 0")
    if args.max_tag_id >= 0 and args.anchor_tag_id > args.max_tag_id:
        parser.error("--anchor-tag-id must be <= --max-tag-id when --max-tag-id is set")
    if args.water_scale <= 0:
        parser.error("--water-scale must be positive")
    if args.surface_distance_m < 0:
        parser.error("--surface-distance-m must be >= 0")
    if args.water_refractive_index <= 1.0:
        parser.error("--water-refractive-index must be > 1.0")
    if args.refractive_max_iterations <= 0:
        parser.error("--refractive-max-iterations must be positive")
    if args.refractive_convergence_tol_m <= 0:
        parser.error("--refractive-convergence-tol-m must be positive")
    if args.refractive_convergence_tol_deg <= 0:
        parser.error("--refractive-convergence-tol-deg must be positive")
    if args.refractive_ray_max_iterations <= 0:
        parser.error("--refractive-ray-max-iterations must be positive")
    if args.refractive_ray_tol <= 0:
        parser.error("--refractive-ray-tol must be positive")
    if args.nthreads <= 0:
        parser.error("--nthreads must be positive")
    if args.quad_decimate <= 0:
        parser.error("--quad-decimate must be positive")
    if args.max_hamming < 0:
        parser.error("--max-hamming must be >= 0")
    if args.min_tag_area_px < 0:
        parser.error("--min-tag-area-px must be >= 0")
    if args.max_off_nadir_deg <= 0:
        parser.error("--max-off-nadir-deg must be positive")
    if args.max_image_eccentricity <= 0:
        parser.error("--max-image-eccentricity must be positive")
    if args.max_tag_tilt_deg <= 0:
        parser.error("--max-tag-tilt-deg must be positive")
    if args.max_reprojection_error_px <= 0:
        parser.error("--max-reprojection-error-px must be positive")
    if args.tag_robust_threshold <= 0:
        parser.error("--tag-robust-threshold must be positive")
    if args.tag_init_min_observations <= 0:
        parser.error("--tag-init-min-observations must be positive")
    if args.pose_std_window <= 1:
        parser.error("--pose-std-window must be > 1")
    for name in (
        "tag_rot_sigma",
        "tag_trans_sigma",
        "odom_rot_sigma",
        "odom_trans_sigma",
        "prior_rot_sigma",
        "prior_trans_sigma",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.floor_z_sigma <= 0:
        parser.error("--floor-z-sigma must be positive")
    if args.floor_plane_min_tags < 3:
        parser.error("--floor-plane-min-tags must be >= 3")
    if args.floor_normal_sigma_deg <= 0:
        parser.error("--floor-normal-sigma-deg must be positive")
    if args.floor_plane_outlier_threshold <= 0:
        parser.error("--floor-plane-outlier-threshold must be positive")
    if args.imu_gravity_smoothing_n < 1:
        parser.error("--imu-gravity-smoothing-n must be >= 1")
    if args.floor_prior_refresh_frames < 0:
        parser.error("--floor-prior-refresh-frames must be >= 0")
    if args.strict_coplanar:
        # Tight z sigma (0.5 mm) so the floor prior's information weight
        # dominates the accumulated tag-observation information; otherwise
        # observations outweigh a soft prior and tags drift back up to their
        # apparent underwater z. Rotation alignment is disabled separately
        # in the backend when strict_coplanar is on (the floor plane is also
        # pinned at the ANCHOR's z, so per-tag rotation alignment is not
        # needed and only fights tag-observation rotation information).
        args.floor_z_sigma = min(float(args.floor_z_sigma), 0.0005)
        # Also enable periodic refresh so the floor prior keeps accumulating
        # over many camera frames; over N refreshes the effective z sigma
        # shrinks by sqrt(N), so even if a single 0.5 mm prior is still soft
        # vs the running tag-observation total information, the accumulated
        # prior chain dominates within a few seconds and locks tags to z=0.
        # ~1 refresh per second at typical FPS.
        if args.floor_prior_refresh_frames == 0:
            args.floor_prior_refresh_frames = 30
    if args.init_min_observations <= 0:
        parser.error("--init-min-observations must be positive")
    if args.init_min_decision_margin < 0:
        parser.error("--init-min-decision-margin must be >= 0")
    if args.init_min_tag_area_px < 0:
        parser.error("--init-min-tag-area-px must be >= 0")
    if args.init_max_off_nadir_deg <= 0:
        parser.error("--init-max-off-nadir-deg must be positive")
    if args.init_max_image_eccentricity <= 0:
        parser.error("--init-max-image-eccentricity must be positive")
    if args.init_max_tag_tilt_deg <= 0:
        parser.error("--init-max-tag-tilt-deg must be positive")
    if args.plot_z_scale <= 0:
        parser.error("--plot-z-scale must be positive")
    if args.display_width < 0:
        parser.error("--display-width must be >= 0")
    if args.trajectory_image_width < 0:
        parser.error("--trajectory-image-width must be >= 0")
    if args.print_every < 0:
        parser.error("--print-every must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    selected_inputs = sum(
        value is not None for value in (args.camera_id, args.serial, args.svo, args.stream)
    )
    if selected_inputs > 1:
        parser.error("choose only one of --camera-id, --serial, --svo, or --stream")

    return args

def load_runtime_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if not config_path.exists():
        print(f"Config file not found, using defaults: {config_path}", flush=True)
        pool = normalize_pool_config({})
        water = normalize_water_config({}, pool)
        pool["water_depth_m"] = float(water["surface_height_m"])
        return {"pool": pool, "water": water}

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
    except ImportError:
        loaded = parse_simple_yaml(text)

    if not isinstance(loaded, dict):
        loaded = {}
    loaded["pool"] = normalize_pool_config(loaded.get("pool", {}))
    loaded["water"] = normalize_water_config(loaded.get("water", {}), loaded["pool"])
    loaded["pool"]["water_depth_m"] = float(loaded["water"]["surface_height_m"])
    return loaded

def resolve_tag_size_m(
    runtime_config: dict[str, object],
    cli_tag_size_m: float | None,
) -> tuple[float, str]:
    if cli_tag_size_m is not None:
        return float(cli_tag_size_m), "cli"

    tags_cfg = runtime_config.get("tags", {})
    if isinstance(tags_cfg, dict) and "tag_size_m" in tags_cfg:
        try:
            return float(tags_cfg["tag_size_m"]), "config"
        except (TypeError, ValueError):
            pass
    return DEFAULT_TAG_SIZE_M, "default"

def parse_stream(value: str) -> tuple[str, int]:
    if ":" not in value:
        return value, 30000
    host, port = value.rsplit(":", 1)
    return host, int(port)

def make_init_parameters(args: argparse.Namespace) -> sl.InitParameters:
    params = sl.InitParameters()
    params.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    params.camera_fps = args.fps
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    # On ZED2, the IMU is exposed through get_sensors_data without enabling
    # depth or positional tracking. We deliberately do NOT set sensors_required
    # so non-ZED2 cameras still open successfully; missing-IMU is handled
    # downstream by ImuGravityTracker (falls back to anchor-frame behavior).

    if args.camera_id is not None:
        params.set_from_camera_id(args.camera_id)
    elif args.serial is not None:
        params.set_from_serial_number(args.serial)
    elif args.svo:
        params.set_from_svo_file(args.svo)
    elif args.stream:
        host, port = parse_stream(args.stream)
        params.set_from_stream(host, port)

    return params

def open_zed(args: argparse.Namespace) -> sl.Camera:
    zed = sl.Camera()
    error = zed.open(make_init_parameters(args))
    if error != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {error}")
    return zed

# Quasi-static gravity gate: when the accelerometer norm is far from ~9.81 m/s^2,
# the camera is being accelerated and the gravity component cannot be cleanly
# isolated from the raw reading. These bounds are wide enough to tolerate
# handheld motion and the brief proper-acceleration spikes that happen as the
# operator positions the camera over the anchor tag; the frame-rate smoothing
# window then averages residual motion noise out. Strong jolts (free-fall,
# sharp shakes) still get rejected.
_IMU_ACCEL_NORM_LO_MS2 = 6.0
_IMU_ACCEL_NORM_HI_MS2 = 13.5
def imu_gravity_in_camera(
    zed: sl.Camera,
    sensors_data: "sl.SensorsData",
) -> np.ndarray | None:
    """
    Return the unit gravity (down) direction expressed in the camera frame for
    the current camera frame, or None if no usable IMU sample is available.

    The ZED2 SDK delivers linear acceleration in the user-selected coordinate
    system (IMAGE here), already mapped from the IMU sensor frame into the
    camera frame via the factory extrinsic. A stationary accelerometer reads
    +g pointing up (proper acceleration counteracts gravity), so the down
    direction is the negative of the normalized reading.
    """
    err = zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
    if err != sl.ERROR_CODE.SUCCESS:
        return None
    try:
        imu_data = sensors_data.get_imu_data()
        accel = np.asarray(imu_data.get_linear_acceleration(), dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(accel)):
        return None
    norm = float(np.linalg.norm(accel))
    if norm < _IMU_ACCEL_NORM_LO_MS2 or norm > _IMU_ACCEL_NORM_HI_MS2:
        return None
    return -accel / norm

class ImuGravityTracker:
    """
    Frame-synchronous smoother for the IMU gravity-in-camera vector.

    One IMU read per camera frame; the smoothing window is N camera frames
    (NOT N IMU samples). A missing IMU sample on a frame keeps the previous
    smoothed gravity. A persistently absent IMU surfaces a one-time warning
    and falls back to anchor-frame behavior.
    """

    def __init__(self, smoothing_n: int):
        self.buffer: deque[np.ndarray] = deque(maxlen=max(1, int(smoothing_n)))
        self.last_smoothed: np.ndarray | None = None
        self.frames_seen: int = 0
        self.frames_with_imu: int = 0
        self.warned_missing: bool = False

    def update(self, gravity_camera: np.ndarray | None) -> np.ndarray | None:
        self.frames_seen += 1
        if gravity_camera is None:
            if (
                self.last_smoothed is None
                and not self.warned_missing
                and self.frames_seen >= 30
            ):
                print(
                    "IMU gravity unavailable for the first 30 frames; "
                    "reverting to anchor-frame up. Check that the camera is a "
                    "ZED2 with the IMU detected.",
                    flush=True,
                )
                self.warned_missing = True
            return self.last_smoothed
        self.frames_with_imu += 1
        self.buffer.append(np.asarray(gravity_camera, dtype=np.float64).reshape(3))
        avg = np.mean(np.stack(self.buffer, axis=0), axis=0)
        norm = float(np.linalg.norm(avg))
        if norm < 1e-9:
            return self.last_smoothed
        self.last_smoothed = avg / norm
        return self.last_smoothed

def print_zed_cameras() -> None:
    print(f"ZED SDK version: {sl.Camera().get_sdk_version()}")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("ZED cameras: none found")
        return
    for device in devices:
        print(
            "ZED "
            f"id={device.id} model={device.camera_model} "
            f"serial={device.serial_number} state={device.camera_state} path={device.path}"
        )

def bgr_from_zed_mat(image_mat: sl.Mat) -> np.ndarray:
    image = image_mat.get_data()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported ZED image shape: {image.shape}")

def get_left_intrinsics(zed: sl.Camera, image_shape: tuple[int, int, int]) -> CameraIntrinsics:
    info = zed.get_camera_information()
    left = info.camera_configuration.calibration_parameters.left_cam
    fx = float(left.fx)
    fy = float(left.fy)
    cx = float(left.cx)
    cy = float(left.cy)

    calibration_resolution = info.camera_configuration.resolution
    calib_w = float(calibration_resolution.width)
    calib_h = float(calibration_resolution.height)
    image_h, image_w = image_shape[:2]

    if calib_w > 0 and calib_h > 0 and (calib_w != image_w or calib_h != image_h):
        sx = image_w / calib_w
        sy = image_h / calib_h
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(left.disto, dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

def run_refractive_self_test() -> int:
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(
            [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
    )
    object_points = tag_object_points(0.170)
    true_rvec = np.array([0.035, -0.025, 0.018], dtype=np.float64)
    true_tvec = np.array([0.030, -0.020, 1.194], dtype=np.float64)
    n_camera = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d_air = 0.170
    n_air = DEFAULT_AIR_REFRACTIVE_INDEX
    n_water = DEFAULT_WATER_REFRACTIVE_INDEX

    true_points_cam = transform_object_points(object_points, true_rvec, true_tvec)
    image_points = project_refractive(
        true_points_cam,
        n_camera,
        d_air,
        n_air,
        n_water,
        intrinsics,
    ).astype(np.float32)
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    success, initial_rvec, initial_tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=method,
    )
    if not success:
        print("Refractive self-test failed: initial solvePnP failed", file=sys.stderr)
        return 1

    refined_rvec, refined_tvec, rms, converged, iterations = solve_refractive_pose_fixed_point(
        object_points,
        image_points,
        initial_rvec,
        initial_tvec,
        intrinsics,
        n_camera,
        d_air,
        n_air,
        n_water,
        max_iterations=8,
        convergence_tol_m=1e-8,
        convergence_tol_deg=1e-5,
        ray_max_iterations=12,
        ray_tolerance=1e-12,
    )
    trans_error_m = float(np.linalg.norm(refined_tvec - true_tvec))
    rot_error = rotation_error_deg(refined_rvec, true_rvec)
    print(
        "Refractive self-test: "
        f"rms={rms:.6f}px, translation_error={trans_error_m * 1000.0:.3f}mm, "
        f"rotation_error={rot_error:.4f}deg, iterations={iterations}",
        flush=True,
    )
    if not converged or trans_error_m > 5e-5 or rot_error > 0.01:
        print("Refractive self-test failed.", file=sys.stderr)
        return 1
    print("Refractive self-test passed.", flush=True)
    return 0

def run_refractive_regression_check() -> int:
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(
            [[690.0, 0.0, 642.0], [0.0, 695.0, 358.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
    )
    object_points = tag_object_points(0.170)
    n_air = DEFAULT_AIR_REFRACTIVE_INDEX
    n_water = DEFAULT_WATER_REFRACTIVE_INDEX
    cases = [
        (np.array([0.02, -0.03, 0.01]), np.array([0.02, -0.01, 1.19]), np.array([0.0, 0.0, 1.0]), 0.17),
        (np.array([0.05, 0.02, -0.04]), np.array([0.10, -0.03, 1.34]), normalize_vector(np.array([0.02, -0.01, 1.0])), 0.22),
        (np.array([-0.04, 0.03, 0.06]), np.array([-0.08, 0.06, 1.05]), normalize_vector(np.array([-0.015, 0.025, 1.0])), 0.14),
        (np.array([0.08, -0.05, 0.03]), np.array([0.16, 0.04, 1.55]), normalize_vector(np.array([0.04, 0.02, 1.0])), 0.28),
    ]
    max_trans_error_m = 0.0
    max_rot_error_deg = 0.0
    max_rms_delta = 0.0
    for index, (true_rvec, true_tvec, n_camera, d_air) in enumerate(cases):
        image_points = project_refractive(
            transform_object_points(object_points, true_rvec, true_tvec),
            n_camera,
            d_air,
            n_air,
            n_water,
            intrinsics,
            ray_max_iterations=12,
            ray_tolerance=1e-12,
        ).astype(np.float32)
        method = (
            cv2.SOLVEPNP_IPPE_SQUARE
            if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
            else cv2.SOLVEPNP_ITERATIVE
        )
        success, initial_rvec, initial_tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            flags=method,
        )
        if not success:
            print(f"Refractive regression failed: initial solvePnP failed for case {index}", file=sys.stderr)
            return 1
        ref_rvec, ref_tvec, ref_rms, ref_ok, _ref_iterations = refine_refractive_pose_lm(
            object_points,
            image_points,
            initial_rvec,
            initial_tvec,
            intrinsics,
            n_camera,
            d_air,
            n_air,
            n_water,
            max_iterations=30,
        )
        fast_rvec, fast_tvec, fast_rms, fast_ok, fast_iterations = solve_refractive_pose_fixed_point(
            object_points,
            image_points,
            initial_rvec,
            initial_tvec,
            intrinsics,
            n_camera,
            d_air,
            n_air,
            n_water,
            max_iterations=8,
            convergence_tol_m=1e-8,
            convergence_tol_deg=1e-5,
            ray_max_iterations=12,
            ray_tolerance=1e-12,
        )
        trans_error = float(np.linalg.norm(fast_tvec - ref_tvec))
        rot_error = rotation_error_deg(fast_rvec, ref_rvec)
        rms_delta = abs(float(fast_rms - ref_rms))
        max_trans_error_m = max(max_trans_error_m, trans_error)
        max_rot_error_deg = max(max_rot_error_deg, rot_error)
        max_rms_delta = max(max_rms_delta, rms_delta)
        print(
            f"Regression case {index}: fast_iter={fast_iterations} "
            f"trans_delta={trans_error * 1000.0:.4f}mm "
            f"rot_delta={rot_error:.5f}deg rms_delta={rms_delta:.6f}px",
            flush=True,
        )
        if not ref_ok or not fast_ok:
            print(f"Refractive regression failed: solver did not converge for case {index}", file=sys.stderr)
            return 1
    print(
        "Refractive regression: "
        f"max_trans_delta={max_trans_error_m * 1000.0:.4f}mm, "
        f"max_rot_delta={max_rot_error_deg:.5f}deg, "
        f"max_rms_delta={max_rms_delta:.6f}px",
        flush=True,
    )
    if max_trans_error_m > 1e-4 or max_rot_error_deg > 0.01:
        print("Refractive regression failed.", file=sys.stderr)
        return 1
    print("Refractive regression passed.", flush=True)
    return 0

def run_refractive_benchmark() -> int:
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(
            [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
    )
    object_points = tag_object_points(0.170)
    n_air = DEFAULT_AIR_REFRACTIVE_INDEX
    n_water = DEFAULT_WATER_REFRACTIVE_INDEX
    tag_count = 20
    repeats = 8
    n_hat_batch = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (tag_count, 1))
    d_air_batch = np.full(tag_count, 0.17, dtype=np.float64)
    true_rvecs = []
    true_tvecs = []
    image_points = []
    method = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )
    initial_rvecs = []
    initial_tvecs = []
    for index in range(tag_count):
        col = index % 5
        row = index // 5
        rvec = np.array([0.015 * row, -0.012 * col, 0.01 * (index % 3 - 1)], dtype=np.float64)
        tvec = np.array([0.055 * (col - 2), 0.045 * (row - 1.5), 1.15 + 0.015 * row], dtype=np.float64)
        pixels = project_refractive(
            transform_object_points(object_points, rvec, tvec),
            n_hat_batch[index],
            d_air_batch[index],
            n_air,
            n_water,
            intrinsics,
        ).astype(np.float32)
        success, initial_rvec, initial_tvec = cv2.solvePnP(
            object_points,
            pixels,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            flags=method,
        )
        if not success:
            print(f"Refractive benchmark failed: initial solvePnP failed for tag {index}", file=sys.stderr)
            return 1
        true_rvecs.append(rvec)
        true_tvecs.append(tvec)
        image_points.append(pixels)
        initial_rvecs.append(np.asarray(initial_rvec, dtype=np.float64).reshape(3))
        initial_tvecs.append(np.asarray(initial_tvec, dtype=np.float64).reshape(3))

    image_points_batch = np.asarray(image_points, dtype=np.float64)
    initial_rvecs_array = np.asarray(initial_rvecs, dtype=np.float64)
    initial_tvecs_array = np.asarray(initial_tvecs, dtype=np.float64)

    start_s = time.perf_counter()
    for tag_index in range(tag_count):
        refine_refractive_pose_lm(
            object_points,
            image_points_batch[tag_index],
            initial_rvecs_array[tag_index],
            initial_tvecs_array[tag_index],
            intrinsics,
            n_hat_batch[tag_index],
            float(d_air_batch[tag_index]),
            n_air,
            n_water,
            max_iterations=20,
        )
    lm_frame_s = time.perf_counter() - start_s

    warm_rvecs = initial_rvecs_array.copy()
    warm_tvecs = initial_tvecs_array.copy()
    fast_times = []
    for _repeat in range(repeats):
        start_s = time.perf_counter()
        results = solve_refractive_pose_fixed_point_batch(
            object_points,
            image_points_batch,
            warm_rvecs,
            warm_tvecs,
            intrinsics,
            n_hat_batch,
            d_air_batch,
            n_air,
            n_water,
            max_iterations=4,
            convergence_tol_m=1e-5,
            convergence_tol_deg=0.01,
            ray_max_iterations=10,
            ray_tolerance=1e-11,
        )
        fast_times.append(time.perf_counter() - start_s)
        for tag_index, result in enumerate(results):
            if result is not None:
                warm_rvecs[tag_index] = result[0]
                warm_tvecs[tag_index] = result[1]

    fast_frame_s = float(np.median(fast_times))
    print("Synthetic refractive benchmark, 20 visible tags")
    print("solver             ms/frame   equivalent fps")
    print(f"reference LM       {lm_frame_s * 1000.0:8.2f}   {1.0 / max(lm_frame_s, 1e-9):8.2f}")
    print(f"fast fixed-point   {fast_frame_s * 1000.0:8.2f}   {1.0 / max(fast_frame_s, 1e-9):8.2f}")
    print(f"speedup: {lm_frame_s / max(fast_frame_s, 1e-9):.1f}x")
    return 0

def main() -> int:
    global WATER_SCALE_FACTOR

    args = parse_args()
    WATER_SCALE_FACTOR = args.water_scale
    runtime_config = load_runtime_config(args.config)
    if args.list:
        print_zed_cameras()
        return 0
    if args.refractive_self_test:
        return run_refractive_self_test()
    if args.refractive_regression_check:
        return run_refractive_regression_check()
    if args.refractive_benchmark:
        return run_refractive_benchmark()

    pool_cfg = runtime_config["pool"]
    water_cfg = runtime_config["water"]
    resolved_tag_size_m, tag_size_source = resolve_tag_size_m(
        runtime_config,
        args.tag_size,
    )
    if not np.isfinite(resolved_tag_size_m) or resolved_tag_size_m <= 0:
        print(
            f"TagSLAM failed: resolved tag size must be positive, got {resolved_tag_size_m}",
            file=sys.stderr,
        )
        return 1
    args.tag_size = resolved_tag_size_m
    viz_tag_size_m = resolved_tag_size_m
    print(
        f"SLAM tag size: {args.tag_size:.3f} m (source: {tag_size_source})",
        flush=True,
    )

    zed: sl.Camera | None = None
    backend: TagSlamBackend | None = None
    trajectory_recorder: TrajectoryRecorder | None = None
    refractive_context: RefractiveContext | None = None
    try:
        zed = open_zed(args)
        runtime = sl.RuntimeParameters()
        image_mat = sl.Mat()
        sensors_data = sl.SensorsData() if args.use_imu_gravity else None
        imu_tracker = (
            ImuGravityTracker(args.imu_gravity_smoothing_n) if args.use_imu_gravity else None
        )

        detector = make_detector(args)
        object_points = tag_object_points(args.tag_size)
        backend = TagSlamBackend(args)
        if args.water_correction_mode == "refractive":
            refractive_context = RefractiveContext(water_cfg=water_cfg, backend=backend)
        print(
            f"World-frame anchor: Tag {args.anchor_tag_id}. "
            f"Initialization waits until Tag {args.anchor_tag_id} passes quality gates.",
            flush=True,
        )
        print(
            "Water correction: "
            f"mode={args.water_correction_mode}, empirical_scale={args.water_scale:.3f}, "
            f"surface_height={float(water_cfg['surface_height_m']):.3f} m, "
            f"n_air={float(water_cfg['n_air']):.3f}, "
            f"n_water={float(water_cfg['n_water']):.3f}, "
            f"up_axis={water_cfg['up_axis_world']}",
            flush=True,
        )
        print(
            "Detection trust region: "
            f"area>={args.min_tag_area_px:.0f}px, "
            f"off_nadir<={args.max_off_nadir_deg:.1f}deg, "
            f"ecc<={args.max_image_eccentricity:.2f}, "
            f"tilt<={args.max_tag_tilt_deg:.1f}deg, "
            f"reproj<={args.max_reprojection_error_px:.1f}px",
            flush=True,
        )
        if args.water_correction_mode == "refractive":
            print(
                "Refractive solver: "
                f"fixed_point_iters<={args.refractive_max_iterations}, "
                f"pose_tol={args.refractive_convergence_tol_m:.1e} m/"
                f"{args.refractive_convergence_tol_deg:.3f} deg, "
                f"ray_newton_iters<={args.refractive_ray_max_iterations}, "
                f"ray_tol={args.refractive_ray_tol:.1e}",
                flush=True,
            )
        print(
            "Tag factors: "
            f"robust={args.tag_robust_kernel}, threshold={args.tag_robust_threshold:.3f}, "
            f"init_min_obs={args.tag_init_min_observations}",
            flush=True,
        )
        if args.floor_prior_enabled:
            print(
                "Floor plane prior: ON, "
                f"min_tags={args.floor_plane_min_tags}, "
                f"z_sigma={args.floor_z_sigma:.4f} m, "
                f"normal_sigma={args.floor_normal_sigma_deg:.2f} deg, "
                f"strict_coplanar={args.strict_coplanar}, "
                f"refresh_frames={args.floor_prior_refresh_frames}",
                flush=True,
            )
        else:
            print("Floor plane prior: OFF", flush=True)
        trajectory_recorder = TrajectoryRecorder(
            Path(args.trajectory_dir),
            args.trajectory_image_width,
            pool_cfg,
            viz_tag_size_m,
            args.plot_z_scale,
            args.anchor_tag_id,
        )

        intrinsics: CameraIntrinsics | None = None
        frame_count = 0
        last_print_s = 0.0
        last_frame_s = time.monotonic()
        fps_value = 0.0

        if not args.no_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        print(
            f"Running TagSLAM. Tag {args.anchor_tag_id} defines the world frame. "
            f"water_scale={WATER_SCALE_FACTOR:.4f}. "
            "Press r to record a trajectory plot, q to quit.",
            flush=True,
        )
        if args.use_imu_gravity:
            print(
                "IMU gravity: ON, "
                f"smoothing_n={args.imu_gravity_smoothing_n} frames, "
                f"gravity_align_world={args.gravity_align_world}",
                flush=True,
            )
        else:
            print("IMU gravity: OFF (anchor-frame up axis)", flush=True)
        if args.record_trajectory:
            trajectory_recorder.start(time.monotonic())

        last_gravity_log_s = 0.0
        while True:
            error = zed.grab(runtime)
            if error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if error != sl.ERROR_CODE.SUCCESS:
                continue

            # Frame-synchronous IMU read: one sample per camera frame using the
            # IMAGE-time reference, BEFORE backend.update() so the latest
            # gravity vector informs G2 (interface normal), G3 (floor prior),
            # and G4 (init-time gravity alignment). Missing samples reuse the
            # last smoothed value; absent IMU degrades silently to anchor-frame.
            smoothed_gravity_camera: np.ndarray | None = None
            if args.use_imu_gravity and sensors_data is not None and imu_tracker is not None:
                raw_gravity = imu_gravity_in_camera(zed, sensors_data)
                smoothed_gravity_camera = imu_tracker.update(raw_gravity)
                backend.set_imu_gravity(smoothed_gravity_camera)
                if refractive_context is not None:
                    refractive_context.imu_gravity_camera = smoothed_gravity_camera

            zed.retrieve_image(image_mat, sl.VIEW.LEFT_BGR, sl.MEM.CPU)
            frame = bgr_from_zed_mat(image_mat)
            if intrinsics is None:
                intrinsics = get_left_intrinsics(zed, frame.shape)
                print(
                    "Left camera intrinsics: "
                    f"fx={intrinsics.camera_matrix[0, 0]:.2f}, "
                    f"fy={intrinsics.camera_matrix[1, 1]:.2f}, "
                    f"cx={intrinsics.camera_matrix[0, 2]:.2f}, "
                    f"cy={intrinsics.camera_matrix[1, 2]:.2f}",
                    flush=True,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observations = detect_observations(
                gray,
                detector,
                intrinsics,
                object_points,
                args,
                refractive_context,
            )
            update = backend.update(observations)

            now_s = time.monotonic()
            dt_s = now_s - last_frame_s
            if dt_s > 0:
                fps_value = 0.9 * fps_value + 0.1 * (1.0 / dt_s) if fps_value > 0 else 1.0 / dt_s
            last_frame_s = now_s

            if (
                args.use_imu_gravity
                and args.print_every > 0
                and now_s - last_gravity_log_s >= args.print_every
            ):
                if smoothed_gravity_camera is not None:
                    # Debug line: gravity-in-camera vector + angle vs the
                    # anchor-frame up axis [0,0,-1]. The angle equals the
                    # current world-frame tilt vs true gravity (without G4)
                    # or the residual camera-vs-world tilt (with G4 applied).
                    anchor_up_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                    cos_a = float(
                        np.clip(
                            (-smoothed_gravity_camera) @ anchor_up_camera,
                            -1.0,
                            1.0,
                        )
                    )
                    tilt_deg = float(np.degrees(np.arccos(cos_a)))
                    g = smoothed_gravity_camera
                    print(
                        "IMU gravity-in-camera: "
                        f"({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}) unit, "
                        f"angle_vs_anchor_up={tilt_deg:.2f} deg, "
                        f"imu_frames={imu_tracker.frames_with_imu}/{imu_tracker.frames_seen}",
                        flush=True,
                    )
                last_gravity_log_s = now_s

            if args.print_every > 0 and now_s - last_print_s >= args.print_every:
                print_backend_update(update, observations)
                if refractive_context is not None and refractive_context.solve_count > 0:
                    avg_tag_ms = 1000.0 * refractive_context.total_s / refractive_context.solve_count
                    avg_frame_ms = (
                        1000.0 * refractive_context.frame_total_s / refractive_context.frame_count
                        if refractive_context.frame_count > 0
                        else 0.0
                    )
                    print(
                        f"Refractive PnP latency: last={refractive_context.last_frame_s * 1000.0:.2f} ms/frame "
                        f"({refractive_context.last_frame_tag_count} tags), "
                        f"avg={avg_frame_ms:.2f} ms/frame, {avg_tag_ms:.2f} ms/tag, "
                        f"fps={fps_value:.1f}, "
                        f"fallbacks={refractive_context.fallback_count}/"
                        f"{refractive_context.solve_count}",
                        flush=True,
                    )
                last_print_s = now_s

            draw_observations(frame, observations)
            draw_overlay(
                frame,
                update,
                observations,
                fps_value,
                trajectory_recorder.active,
                len(trajectory_recorder.samples),
            )
            trajectory_recorder.append(update, observations, now_s, frame)

            if not args.no_window:
                display_scale = get_display_scale(frame.shape, args.display_width)
                cv2.imshow(WINDOW_NAME, resize_for_display(frame, display_scale))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("r"), ord("R")):
                    if trajectory_recorder.active:
                        trajectory_recorder.stop_and_save(backend)
                    else:
                        trajectory_recorder.start(time.monotonic())
                if key == ord("q"):
                    break

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

        if trajectory_recorder.active:
            trajectory_recorder.stop_and_save(backend)

        return 0

    except KeyboardInterrupt:
        if trajectory_recorder is not None and trajectory_recorder.active and backend is not None:
            trajectory_recorder.stop_and_save(backend)
        return 0
    except Exception as exc:
        print(f"TagSLAM failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()


if __name__ == "__main__":
    raise SystemExit(main())
