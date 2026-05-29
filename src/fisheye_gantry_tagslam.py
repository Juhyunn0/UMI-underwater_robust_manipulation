#!/usr/bin/env python3
"""
fisheye_gantry_tagslam.py — Fisheye-camera AprilTag SLAM + FMC4030 gantry
motion + telemetry logger, with a shared monotonic clock so the two CSVs can
be joined post-hoc to compare estimated camera pose against gantry ground
truth.

================================================================================
Frame / sign / extrinsic conventions  (PLEASE CONFIRM PER RIG)
================================================================================
* Gantry world (X, Y, Z) is in millimeters in the controller's native frame.
  Per-axis unit scaling lives in gantry_runner.SCALE_MM_PER_UNIT (X=8.25,
  Y=2.5, Z=0.5 mm/unit).
* The fisheye calibration YAML carries a 4x4 ``T_gantry_camera`` extrinsic
  that maps a point in the camera body frame into the gantry world frame.
  If your rig has +Z down (vs SLAM world +Z up, or vice versa), the rotation
  block of T_gantry_camera absorbs the flip. This module makes NO axis-sign
  assumption — it just consumes the 4x4 as provided.
* The SLAM world is the anchor-AprilTag frame. Without a separate gantry-vs-
  anchor calibration, ``translation_error_mm`` is computed literally as
  ||p_est_world - T_gantry_camera @ p_gantry_world|| and includes the
  (unknown, constant) frame offset. Subtracting the first-sample offset is
  the cheapest way to get a meaningful magnitude in post-processing.
* Fisheye undistortion path: we undistort the full frame using the OpenCV
  fisheye model (``cv2.fisheye.initUndistortRectifyMap``) into a rectified
  pinhole image with intrinsics ``new_K``, then run AprilTag detection and
  ``solvePnP`` on the rectified image with ``new_K``. Downstream this looks
  identical to the ZED pinhole pipeline.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# sys.path shim: import sibling modules (`gantry`, `tagslam_core`, `gantry_runner`)
# from src/ regardless of where this script is invoked from.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from gantry import (  # noqa: E402
    Axis,
    ControllerConfig,
    FMC4030Controller,
    FMC4030Error,
)

from gantry_runner import (  # noqa: E402
    AXES,
    CSV_COLUMNS as GANTRY_CSV_COLUMNS,
    GantrySample,
    GantryTelemetryLogger,
    SCALE_MM_PER_UNIT,
    Waypoint,
    _device_soft_limits_mm,
    _make_run_dir as make_gantry_run_dir,  # not used directly; we have our own
    _parse_waypoints_csv as parse_waypoints_csv,
    _read_current_pos_mm,
    _validate_soft_limits,
    move_to_xyz_mm,
    units_to_mm,
)

from tagslam_core import (  # noqa: E402
    CameraIntrinsics,
    DEFAULT_ANCHOR_TAG_ID,
    DEFAULT_TAG_SIZE_M,
    PLOT_Z_SCALE,
    RefractiveContext,
    TagSlamBackend,
    TrajectoryRecorder,
    detect_observations,
    draw_observations,
    draw_overlay,
    get_display_scale,
    make_detector,
    make_run_dir,
    normalize_water_config,
    parse_simple_yaml,
    pose_rpy,
    pose_translation,
    print_backend_update,
    resize_for_display,
    tag_object_points,
)
from tagslam.visualization import normalize_pool_config  # noqa: E402


WINDOW_NAME = "Fisheye+Gantry TagSLAM"
EMERGENCY_STOP = threading.Event()


def _parse_exclude_ids(value) -> set:
    """Parse args.exclude_tags into a set of ints. Accepts a set/list (from the
    GUI), a comma/space/semicolon-separated string (from the CLI), or None."""
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        return {int(v) for v in value}
    out: set = set()
    for tok in str(value).replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return out


def _filter_excluded(observations, exclude_ids: set):
    """Drop observations of blacklisted tag IDs before they reach the backend."""
    if not exclude_ids:
        return observations
    return [o for o in observations if int(o.tag_id) not in exclude_ids]


# =============================================================================
# FisheyeGantryWorker — callable API for ExperimentRunner
# =============================================================================
class FisheyeGantryWorker:
    """Encapsulates the camera + TagSLAM run loop so it can be invoked from
    both ``main()`` (CLI) and ``ExperimentRunner`` (GUI experiment tab).

    Usage::

        worker = FisheyeGantryWorker(
            args, calib, t0_monotonic=t0,
            run_dir=run_dir,
            gantry_logger=logger,
            abort_event=stop_event,
            sample_queue=queue,   # optional: FisheyeStatsSample emitted ~5 Hz
        )
        worker.run()  # blocking; returns when done or abort_event is set

    After ``run()`` returns:
        ``worker.trajectory_recorder``  — TrajectoryRecorder (call stop_and_save)
        ``worker.backend``              — TagSlamBackend
        ``worker.frame_count``          — int
    """

    def __init__(
        self,
        args,
        calib: "FisheyeCalibration",
        *,
        t0_monotonic: float,
        run_dir: Path,
        gantry_logger: "GantryTelemetryLogger | None" = None,
        abort_event: "threading.Event | None" = None,
        sample_queue: "Any | None" = None,  # Queue[FisheyeStatsSample]
        mock_camera: bool = False,
    ) -> None:
        self._args = args
        self._calib = calib
        self._t0 = t0_monotonic
        self._run_dir = run_dir
        self._gantry_logger = gantry_logger
        self._abort = abort_event or threading.Event()
        self._sample_queue = sample_queue
        self._mock_camera = mock_camera

        self.trajectory_recorder = None
        self.backend = None
        self.frame_count = 0

    def run(self) -> None:
        """Blocking camera + TagSLAM loop. Returns when done or aborted."""
        try:
            self._run_inner()
        except Exception as exc:
            print(f"[FisheyeGantryWorker] error: {exc}", file=sys.stderr)

    def _run_inner(self) -> None:
        args = self._args
        calib = self._calib

        runtime_config: dict = {}
        try:
            config_path = Path(args.config)
            if config_path.exists():
                with config_path.open("r") as fh:
                    runtime_config_text = fh.read()
                runtime_config = parse_simple_yaml(runtime_config_text)
        except Exception:
            pass
        pool_cfg = normalize_pool_config(runtime_config.get("pool", {}))
        water_cfg = normalize_water_config(runtime_config.get("water"), pool_cfg)

        if self._mock_camera:
            from experiment_runner import _MockCamera
            cap = _MockCamera(calib.image_size[0], calib.image_size[1])
        else:
            cap = open_camera(
                args.camera_device,
                tuple(args.camera_resolution) if args.camera_resolution else None,
                args.camera_fps,
            )

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        map1, map2, new_K = build_fisheye_undistort_maps(
            calib.K, calib.D, (cam_w, cam_h), args.fisheye_balance,
        )
        intrinsics = rectified_camera_intrinsics(new_K)

        backend = TagSlamBackend(args)
        self.backend = backend
        detector = make_detector(args)
        object_points = tag_object_points(args.tag_size)
        exclude_ids = _parse_exclude_ids(getattr(args, "exclude_tags", ""))
        if exclude_ids:
            print(f"[exclude] dropping tag IDs {sorted(exclude_ids)} from SLAM "
                  "(blacklisted — e.g. physically duplicated IDs)", file=sys.stderr)
        refractive_context = None
        if args.water_correction_mode == "refractive":
            refractive_context = RefractiveContext(water_cfg=water_cfg, backend=backend)

        trajectory_recorder = TrajectoryRecorder(
            output_root=self._run_dir.parent,
            image_width=args.trajectory_image_width,
            pool_cfg=pool_cfg,
            tag_size_m=args.tag_size,
            plot_z_scale=args.plot_z_scale,
            anchor_tag_id=args.anchor_tag_id,
            suffix="fisheye_gantry",
            frames_subdir="frames",
        )
        trajectory_recorder.output_dir = self._run_dir
        trajectory_recorder.frames_dir = self._run_dir / "frames"
        trajectory_recorder.active = True
        trajectory_recorder.start_monotonic_s = self._t0
        trajectory_recorder.samples = []
        self.trajectory_recorder = trajectory_recorder

        last_stats_t = 0.0
        backend_updates = 0

        try:
            while not self._abort.is_set() and not EMERGENCY_STOP.is_set():
                ok, raw_frame = cap.read()
                if not ok:
                    break

                frame_t_unix = time.time()
                frame_t_mono = time.monotonic()

                frame = cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                observations = detect_observations(
                    gray, detector, intrinsics, object_points, args, refractive_context,
                )
                observations = _filter_excluded(observations, exclude_ids)
                update = backend.update(observations)
                if update.camera_pose is not None:
                    backend_updates += 1

                gantry_sample = (
                    self._gantry_logger.latest_sample()
                    if self._gantry_logger is not None else None
                )
                extra: dict | None = None
                drift_mm = float("nan")
                if gantry_sample is not None and update.camera_pose is not None:
                    cam_est = np.array(pose_translation(update.camera_pose), dtype=np.float64)
                    cam_gt = gantry_to_world_translation_m(gantry_sample, calib.T_gantry_camera)
                    drift_mm = float(np.linalg.norm(cam_est - cam_gt)) * 1000.0
                    gx, gy, gz = gantry_sample.pos_mm
                    extra = {
                        "gantry_x_mm": float(gx),
                        "gantry_y_mm": float(gy),
                        "gantry_z_mm": float(gz),
                        "translation_error_mm": drift_mm,
                    }

                trajectory_recorder.append(
                    update, observations, frame_t_mono, frame,
                    timestamp_unix=frame_t_unix,
                    timestamp_monotonic=frame_t_mono,
                    extra=extra,
                )
                self.frame_count += 1

                # Push stats to queue at ~5 Hz
                now = time.monotonic()
                if self._sample_queue is not None and now - last_stats_t >= 0.2:
                    last_stats_t = now
                    try:
                        from experiment_runner import FisheyeStatsSample
                        self._sample_queue.put_nowait(FisheyeStatsSample(
                            tags_this_frame=len(observations),
                            tags_in_graph=len(backend.optimized_tag_poses()),
                            backend_updates=backend_updates,
                            drift_mm=drift_mm,
                        ))
                    except Exception:
                        pass

                if args.max_frames is not None and self.frame_count >= args.max_frames:
                    break
        finally:
            if hasattr(cap, "release"):
                cap.release()


# =============================================================================
# Calibration loading
# =============================================================================
@dataclass(frozen=True)
class FisheyeCalibration:
    K: np.ndarray              # 3x3 fisheye intrinsics
    D: np.ndarray              # 4x1 fisheye distortion
    image_size: tuple[int, int]  # (width, height)
    T_gantry_camera: np.ndarray  # 4x4, camera-frame point -> gantry-frame point
    gantry_anchor_offset_mm: np.ndarray | None = None  # 3-vec, gantry-frame origin offset
    R_gantry_to_slam: np.ndarray | None = None  # 3x3, gantry frame -> SLAM world frame (default identity)


def load_fisheye_calibration(path: Path) -> FisheyeCalibration:
    """Load fisheye intrinsics + T_gantry_camera from a YAML file. Expected keys:

      K: 3x3 list of lists, or flat 9-list
      D: list of 4 floats (fisheye model: k1, k2, k3, k4)
      image_size: [width, height]
      T_gantry_camera: 4x4 list of lists, or flat 16-list

    Raises SystemExit with a clear message on any structural error.
    """
    if not path.exists():
        raise SystemExit(f"Fisheye calibration YAML not found: {path}")
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["D"], dtype=np.float64).reshape(4, 1)
        w, h = (int(v) for v in data["image_size"])
        T = np.asarray(data["T_gantry_camera"], dtype=np.float64).reshape(4, 4)
    except (KeyError, ValueError, TypeError) as exc:
        raise SystemExit(f"Bad fisheye calibration in {path}: {exc}")
    # Quick sanity checks; not exhaustive.
    if not np.isfinite(K).all() or not np.isfinite(D).all() or not np.isfinite(T).all():
        raise SystemExit(f"Fisheye calibration in {path} contains non-finite values.")
    if abs(T[3, 0]) + abs(T[3, 1]) + abs(T[3, 2]) > 1e-9 or abs(T[3, 3] - 1.0) > 1e-9:
        raise SystemExit(f"T_gantry_camera last row must be [0,0,0,1]; got {T[3].tolist()}")

    # Optional gantry_anchor_offset_mm (3-vector). Absent -> None.
    offset = None
    if data.get("gantry_anchor_offset_mm") is not None:
        try:
            offset = np.asarray(data["gantry_anchor_offset_mm"], dtype=np.float64).reshape(3)
        except (ValueError, TypeError):
            print(f"[calib] gantry_anchor_offset_mm malformed in {path}; ignoring.",
                  file=sys.stderr)
            offset = None

    # Optional R_gantry_to_slam (3x3 rotation, gantry frame -> SLAM world frame).
    # Absent -> None (callers treat as identity). On a malformed / non-orthonormal
    # matrix we warn and fall back to None rather than crashing the pipeline.
    R = None
    if data.get("R_gantry_to_slam") is not None:
        try:
            R = np.asarray(data["R_gantry_to_slam"], dtype=np.float64).reshape(3, 3)
        except (ValueError, TypeError):
            print(f"[calib] R_gantry_to_slam malformed in {path}; using identity.",
                  file=sys.stderr)
            R = None
        if R is not None:
            ortho = np.allclose(R @ R.T, np.eye(3), atol=1e-3)
            det_ok = abs(float(np.linalg.det(R)) - 1.0) < 1e-3
            if not (ortho and det_ok):
                print(f"[calib] R_gantry_to_slam in {path} is not a proper rotation "
                      f"(R@R.T==I: {ortho}, det==+1: {det_ok}); using identity.",
                      file=sys.stderr)
                R = None

    return FisheyeCalibration(
        K=K, D=D, image_size=(w, h), T_gantry_camera=T,
        gantry_anchor_offset_mm=offset, R_gantry_to_slam=R,
    )


def build_fisheye_undistort_maps(
    K: np.ndarray, D: np.ndarray, size_wh: tuple[int, int], balance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (map1, map2, new_K) for cv2.remap-based fisheye rectification.

    ``balance`` 0.0 keeps only valid pixels (tight crop, no black borders);
    1.0 preserves the full source FOV (with black corners). We pass through
    to estimateNewCameraMatrixForUndistortRectify.
    """
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size_wh, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size_wh, cv2.CV_16SC2
    )
    return map1, map2, new_K


# =============================================================================
# Per-frame fisheye intrinsics adapter
# =============================================================================
def rectified_camera_intrinsics(new_K: np.ndarray) -> CameraIntrinsics:
    """Wrap the rectified intrinsics in the tagslam_core dataclass with zero
    distortion (the image is already undistorted before AprilTag detection)."""
    return CameraIntrinsics(
        camera_matrix=np.asarray(new_K, dtype=np.float64),
        dist_coeffs=np.zeros((5, 1), dtype=np.float64),
    )


# =============================================================================
# Motion thread: walks the waypoint list while the camera loop runs.
# =============================================================================
class GantryMotionThread(threading.Thread):
    def __init__(
        self,
        controller: FMC4030Controller,
        waypoints: list[Waypoint],
        *,
        acc_mm_s2: float,
        dec_mm_s2: float,
        mode: str,
        lock: threading.RLock,
        logger: GantryTelemetryLogger,
        on_done: threading.Event,
    ) -> None:
        super().__init__(name="gantry-motion", daemon=True)
        self._controller = controller
        self._waypoints = waypoints
        self._acc = acc_mm_s2
        self._dec = dec_mm_s2
        self._mode = mode
        self._lock = lock
        self._logger = logger
        self._on_done = on_done
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def run(self) -> None:
        try:
            for i, wp in enumerate(self._waypoints):
                if EMERGENCY_STOP.is_set():
                    break
                print(
                    f"\n→ waypoint [{i}] target=({wp.x_mm:.2f}, {wp.y_mm:.2f}, "
                    f"{wp.z_mm:.2f}) mm at {wp.speed_mm_s:.2f} mm/s [mode={self._mode}]",
                    flush=True,
                )
                move_to_xyz_mm(
                    self._controller,
                    (wp.x_mm, wp.y_mm, wp.z_mm),
                    wp.speed_mm_s,
                    self._acc,
                    self._dec,
                    mode=self._mode,
                    lock=self._lock,
                    logger=self._logger,
                    waypoint_index=i,
                )
                if wp.dwell_s > 0 and not EMERGENCY_STOP.is_set():
                    time.sleep(wp.dwell_s)
        except BaseException as exc:  # capture so the main thread can decide
            self._error = exc
            for axis in AXES:
                try:
                    with self._lock:
                        self._controller.stop_axis(axis, mode=2)
                except FMC4030Error:
                    pass
        finally:
            self._on_done.set()


# =============================================================================
# Live top-down side panel
# =============================================================================
def render_topdown_panel(
    width_px: int,
    height_px: int,
    backend: TagSlamBackend,
    cam_trajectory_world: list[tuple[float, float]],
    gantry_trajectory_world_mm: list[tuple[float, float]],
    T_gantry_camera: np.ndarray,
    anchor_tag_id: int,
) -> np.ndarray:
    """Top-down (X,Y) plot of: anchor tag origin, optimized tag positions,
    estimated camera trajectory in SLAM world (m), and gantry trajectory
    transformed into SLAM world via the inverse T_gantry_camera (m).

    The two trajectories share an origin only after some external alignment;
    in v1 we just plot them both and let the eye judge drift.
    """
    canvas = np.full((height_px, width_px, 3), 28, dtype=np.uint8)
    margin = 24

    # Collect every point we want to plot so the auto-fit fits all of them.
    pts_m: list[tuple[float, float]] = [(0.0, 0.0)]  # anchor at origin
    tag_poses = backend.optimized_tag_poses()
    for tag_id, pose in tag_poses.items():
        x, y, _ = pose_translation(pose)
        pts_m.append((float(x), float(y)))
    for x_m, y_m in cam_trajectory_world:
        pts_m.append((float(x_m), float(y_m)))

    # Map gantry trajectory (mm in gantry frame) → camera body point in gantry
    # frame: (gantry_xyz_mm / 1000) + T_gantry_camera[:3,3]. SLAM-world and
    # gantry-frame share an origin only after external calibration; v1 plots
    # both on the same axes and lets the eye judge drift. T_gantry_camera
    # translation is in METERS (matches SLAM units).
    offset_m = T_gantry_camera[:3, 3]
    gantry_pts_m: list[tuple[float, float]] = [
        (gx_mm / 1000.0 + float(offset_m[0]), gy_mm / 1000.0 + float(offset_m[1]))
        for gx_mm, gy_mm in gantry_trajectory_world_mm
    ]
    pts_m.extend(gantry_pts_m)

    if len(pts_m) < 2:
        return canvas

    xs = [p[0] for p in pts_m]
    ys = [p[1] for p in pts_m]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(x_max - x_min, 0.5)
    span_y = max(y_max - y_min, 0.5)
    span = max(span_x, span_y) * 1.15  # 15% padding

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    plot_w = width_px - 2 * margin
    plot_h = height_px - 2 * margin
    scale = min(plot_w, plot_h) / span if span > 0 else 1.0

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        # World X right, world Y up; image y axis flipped.
        u = int(margin + plot_w * 0.5 + (x_m - cx) * scale)
        v = int(margin + plot_h * 0.5 - (y_m - cy) * scale)
        return u, v

    # Axes.
    cv2.line(canvas, to_px(x_min, 0), to_px(x_max, 0), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, to_px(0, y_min), to_px(0, y_max), (60, 60, 60), 1, cv2.LINE_AA)

    # Tag positions: anchor in cyan, others in grey.
    for tag_id, pose in tag_poses.items():
        x, y, _ = pose_translation(pose)
        u, v = to_px(float(x), float(y))
        color = (0, 230, 230) if tag_id == anchor_tag_id else (160, 160, 160)
        cv2.circle(canvas, (u, v), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(tag_id), (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Estimated camera trajectory: white polyline + last-point marker.
    if len(cam_trajectory_world) >= 2:
        pts = np.array([to_px(x, y) for x, y in cam_trajectory_world], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
    if cam_trajectory_world:
        u, v = to_px(*cam_trajectory_world[-1])
        cv2.circle(canvas, (u, v), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # Gantry ground-truth trajectory: contrasting orange.
    if len(gantry_pts_m) >= 2:
        pts = np.array([to_px(x, y) for x, y in gantry_pts_m], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)
    if gantry_pts_m:
        u, v = to_px(*gantry_pts_m[-1])
        cv2.circle(canvas, (u, v), 5, (0, 165, 255), -1, cv2.LINE_AA)

    cv2.putText(canvas, "Top-down (X,Y) [m]", (margin, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "white=est  orange=gantry  cyan=anchor",
                (margin, height_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# =============================================================================
# Error column computation
# =============================================================================
def gantry_to_world_translation_m(
    gantry_sample: GantrySample, T_gantry_camera: np.ndarray
) -> np.ndarray:
    """Apply T_gantry_camera (camera-frame point → gantry-frame point) to map
    the gantry's reported tool-point translation in mm into the *camera body
    point in gantry-frame* expressed in METERS.

    For v1 we treat this as the "ground-truth camera position in world", per
    the user's spec — even though gantry-frame and SLAM-world-frame don't
    share an origin without external calibration. Subtract the first sample
    in post-processing for a drift-free error magnitude.
    """
    # The mapping camera-point -> gantry-point is T_gc. The camera body origin
    # in the camera frame is (0,0,0,1); its image in the gantry frame is
    # T_gc @ (0,0,0,1) = T_gc[:3,3]. We want the camera body's position when
    # the gantry tool is at (gx, gy, gz) mm: that's (gantry_xyz_mm + T_gc[:3,3]).
    # Convert mm → m for compatibility with SLAM units (which are meters).
    gantry_mm = np.asarray(gantry_sample.pos_mm, dtype=np.float64)
    cam_in_gantry_m = gantry_mm / 1000.0 + T_gantry_camera[:3, 3]
    return cam_in_gantry_m


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fisheye AprilTag SLAM + FMC4030 gantry motion + telemetry, with "
            "a shared monotonic clock for post-hoc comparison."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- Camera (fisheye) ---------------------------------------------------
    cam = p.add_argument_group("Camera (fisheye)")
    cam.add_argument("--camera-device", default="0",
                     help="cv2.VideoCapture device: integer index ('0') or path ('/dev/video0').")
    cam.add_argument("--camera-resolution", type=int, nargs=2, metavar=("W", "H"),
                     default=None,
                     help="Request capture resolution WxH. Falls back to camera default if unset.")
    cam.add_argument("--camera-fps", type=float, default=None,
                     help="Request capture FPS. Camera may not honor this.")
    cam.add_argument("--fisheye-calib", type=Path, required=True,
                     help="YAML with keys: K (3x3), D (4x1), image_size [W,H], T_gantry_camera (4x4).")
    cam.add_argument("--fisheye-balance", type=float, default=0.0,
                     help="estimateNewCameraMatrixForUndistortRectify balance: 0=tight crop, 1=full FOV.")

    # ----- TagSLAM (inherited from ZED2 script; CLI surface preserved) -------
    slam = p.add_argument_group("TagSLAM / AprilTag")
    slam.add_argument("--tag-family", default="tag36h11")
    slam.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE_M,
                      help="Physical AprilTag edge length in meters.")
    slam.add_argument("--anchor-tag-id", type=int, default=DEFAULT_ANCHOR_TAG_ID)
    slam.add_argument("--tag-map", type=Path, default=None,
                      help="Survey tag map (config/tag_map.yaml from survey_tags.py). "
                           "When set, SLAM runs in PnP-only mode: every tag is locked "
                           "to its mapped pose (transformed into the runtime anchor "
                           "frame) and the live bootstrap is skipped.")
    slam.add_argument("--exclude-tags", default="",
                      help="Comma-separated tag IDs to drop before SLAM (e.g. physically "
                           "duplicated IDs that warp the map): --exclude-tags 64,65,68,69. "
                           "Applied in both PnP-only and bootstrap modes.")
    slam.add_argument("--max-tag-id", type=int, default=-1)
    slam.add_argument("--water-scale", type=float, default=3.6)
    slam.add_argument("--water-correction-mode",
                      choices=["none", "scalar", "trust-region", "refractive"],
                      default="none",
                      help="Default 'none' (in air). Set to 'refractive' once the rig is submerged.")
    slam.add_argument("--surface-distance-m", type=float, default=0.20)
    slam.add_argument("--water-refractive-index", type=float, default=1.333)
    slam.add_argument("--refractive-max-iterations", type=int, default=8)
    slam.add_argument("--refractive-convergence-tol-m", type=float, default=1e-5)
    slam.add_argument("--refractive-convergence-tol-deg", type=float, default=0.01)
    slam.add_argument("--refractive-ray-max-iterations", type=int, default=10)
    slam.add_argument("--refractive-ray-tol", type=float, default=1e-11)

    slam.add_argument("--min-tag-area-px", type=float, default=120.0)
    slam.add_argument("--max-off-nadir-deg", type=float, default=25.0)
    slam.add_argument("--max-image-eccentricity", type=float, default=0.65)
    slam.add_argument("--max-tag-tilt-deg", type=float, default=35.0)
    slam.add_argument("--max-reprojection-error-px", type=float, default=5.0)
    slam.add_argument("--nthreads", type=int, default=2)
    slam.add_argument("--quad-decimate", type=float, default=1.0)
    slam.add_argument("--quad-sigma", type=float, default=0.0)
    slam.add_argument("--decode-sharpening", type=float, default=0.25)
    slam.add_argument("--min-decision-margin", type=float, default=30.0)
    slam.add_argument("--max-hamming", type=int, default=0)

    slam.add_argument("--tag-rot-sigma", type=float, default=0.08)
    slam.add_argument("--tag-trans-sigma", type=float, default=0.04)
    slam.add_argument("--tag-robust-kernel",
                      choices=["none", "huber", "cauchy", "tukey"], default="huber")
    slam.add_argument("--tag-robust-threshold", type=float, default=1.345)
    slam.add_argument("--tag-init-min-observations", type=int, default=3)
    slam.add_argument("--pose-std-window", type=int, default=30)
    slam.add_argument("--odom-rot-sigma", type=float, default=0.35)
    slam.add_argument("--odom-trans-sigma", type=float, default=0.30)
    slam.add_argument("--prior-rot-sigma", type=float, default=1e-6)
    slam.add_argument("--prior-trans-sigma", type=float, default=1e-6)

    slam.set_defaults(floor_prior_enabled=True)
    slam.add_argument("--floor-prior-enabled", dest="floor_prior_enabled", action="store_true")
    slam.add_argument("--no-floor-prior", dest="floor_prior_enabled", action="store_false")
    slam.add_argument("--floor-z-sigma", type=float, default=0.02)
    slam.add_argument("--floor-plane-min-tags", type=int, default=4)
    slam.add_argument("--floor-normal-sigma-deg", type=float, default=8.0)
    slam.set_defaults(strict_coplanar=False)
    slam.add_argument("--strict-coplanar", dest="strict_coplanar", action="store_true")
    slam.add_argument("--no-strict-coplanar", dest="strict_coplanar", action="store_false")
    slam.add_argument("--floor-prior-refresh-frames", type=int, default=0)
    slam.add_argument("--floor-plane-outlier-threshold", type=float, default=0.10)

    # IMU gravity flags exist in the parse-args surface for API parity but
    # are forced OFF since a USB fisheye has no IMU. They appear in args
    # because TagSlamBackend reads them.
    slam.set_defaults(use_imu_gravity=False, gravity_align_world=False)
    slam.add_argument("--imu-gravity-smoothing-n", type=int, default=5)

    slam.add_argument("--init-min-observations", type=int, default=3)
    slam.add_argument("--init-min-decision-margin", type=float, default=45.0)
    slam.add_argument("--init-min-tag-area-px", type=float, default=250.0)
    slam.add_argument("--init-max-off-nadir-deg", type=float, default=20.0)
    slam.add_argument("--init-max-image-eccentricity", type=float, default=0.45)
    slam.add_argument("--init-max-tag-tilt-deg", type=float, default=25.0)

    slam.add_argument("--plot-z-scale", type=float, default=PLOT_Z_SCALE)
    slam.add_argument("--display-width", type=int, default=1600)
    slam.add_argument("--config", default="config/config.yaml",
                      help="YAML with pool geometry and water surface metadata.")
    slam.add_argument("--trajectory-dir", default="data",
                      help="Root for data/YYYYMMDD/<timestamp>_fisheye_gantry/ output folders.")
    slam.add_argument("--record-trajectory", action="store_true",
                      help="Start trajectory recording immediately (also enables frames/ dir).")
    slam.add_argument("--trajectory-image-width", type=int, default=960)
    slam.add_argument("--no-window", action="store_true")
    slam.add_argument("--print-every", type=float, default=0.5)
    slam.add_argument("--max-frames", type=int, default=None)

    # ----- Gantry ------------------------------------------------------------
    g = p.add_argument_group("Gantry (FMC4030)")
    g.add_argument("--gantry-ip", type=str, default="192.168.0.30")
    g.add_argument("--gantry-port", type=int, default=8088)
    g.add_argument("--gantry-id", type=int, default=1)
    g.add_argument("--no-gantry", action="store_true",
                   help="Skip controller; run camera + tags only (passive mode).")
    g.add_argument("--x-mm", type=float, default=None)
    g.add_argument("--y-mm", type=float, default=None)
    g.add_argument("--z-mm", type=float, default=None)
    g.add_argument("--waypoints-csv", type=Path, default=None,
                   help="CSV columns x_mm,y_mm,z_mm,speed_mm_s,dwell_s.")
    g.add_argument("--speed-mm-s", type=float, default=20.0)
    g.add_argument("--acc-mm-s2", type=float, default=50.0)
    g.add_argument("--dec-mm-s2", type=float, default=50.0)
    g.add_argument("--mode", choices=("line", "sequential"), default="line")
    g.add_argument("--log-hz", type=float, default=100.0)
    g.add_argument("--soft-limit-min-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    g.add_argument("--soft-limit-max-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    g.add_argument("--dry-run", action="store_true",
                   help="Connect, validate, write empty CSV headers, exit (no motion, no camera loop).")

    args = p.parse_args(argv)

    # Cross-flag validation
    if args.tag_size is not None and args.tag_size <= 0:
        p.error("--tag-size must be positive")
    if args.anchor_tag_id < 0:
        p.error("--anchor-tag-id must be >= 0")
    if args.max_tag_id >= 0 and args.anchor_tag_id > args.max_tag_id:
        p.error("--anchor-tag-id must be <= --max-tag-id when --max-tag-id is set")
    if args.print_every < 0:
        p.error("--print-every must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    if args.log_hz <= 0:
        p.error("--log-hz must be > 0")
    if not args.no_gantry:
        if args.waypoints_csv is None:
            n_set = sum(v is not None for v in (args.x_mm, args.y_mm, args.z_mm))
            if 0 < n_set < 3:
                p.error("provide all of --x-mm, --y-mm, --z-mm together, or --waypoints-csv")
            if n_set == 0 and not args.dry_run:
                p.error("provide --waypoints-csv or --x-mm/--y-mm/--z-mm (or --no-gantry)")
    return args


# =============================================================================
# Camera capture
# =============================================================================
def open_camera(device: str, resolution: tuple[int, int] | None, fps: float | None) -> cv2.VideoCapture:
    # Try integer first, else string path.
    try:
        idx = int(device)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera device {device!r}")
    if resolution is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolution[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolution[1]))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


# =============================================================================
# Main
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Shared monotonic epoch for both CSVs.
    t0_mono = time.monotonic()
    t0_unix = time.time()

    # Calibration first (fast, validates before touching hardware).
    calib = load_fisheye_calibration(args.fisheye_calib)

    # Runtime config (pool / water).
    runtime_config_text = ""
    runtime_config: dict = {}
    try:
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r") as fh:
                runtime_config_text = fh.read()
            runtime_config = parse_simple_yaml(runtime_config_text)
        else:
            print(f"! Config {config_path} not found; using defaults.", file=sys.stderr)
    except Exception as exc:
        print(f"! Could not parse {args.config}: {exc}", file=sys.stderr)
        runtime_config = {}
    pool_cfg = normalize_pool_config(runtime_config.get("pool", {}))
    water_cfg = normalize_water_config(runtime_config.get("water"), pool_cfg)

    # Resolve waypoints (if any).
    waypoints: list[Waypoint] = []
    if not args.no_gantry:
        if args.waypoints_csv is not None:
            waypoints = parse_waypoints_csv(args.waypoints_csv, args.speed_mm_s)
        elif args.x_mm is not None:
            waypoints = [Waypoint(args.x_mm, args.y_mm, args.z_mm, args.speed_mm_s, 0.0)]

    # Gantry connect.
    controller: FMC4030Controller | None = None
    gantry_lock = threading.RLock()
    gantry_logger: GantryTelemetryLogger | None = None
    motion_thread: GantryMotionThread | None = None
    motion_done = threading.Event()

    if not args.no_gantry:
        controller = FMC4030Controller()
        try:
            controller.connect(ControllerConfig(
                controller_id=args.gantry_id, ip=args.gantry_ip, port=args.gantry_port,
            ))
        except FMC4030Error as exc:
            print(f"✗ Gantry connect failed ({args.gantry_ip}:{args.gantry_port}): {exc}",
                  file=sys.stderr)
            return 2

    # Output folder.
    run_dir = make_run_dir(Path(args.trajectory_dir), "fisheye_gantry")
    print(f"Run output: {run_dir}", flush=True)

    # Resolve and validate soft limits (gantry mode only).
    if controller is not None:
        try:
            dev_min, dev_max = _device_soft_limits_mm(controller, gantry_lock)
        except FMC4030Error as exc:
            print(f"! get_device_parameters failed: {exc}", file=sys.stderr)
            dev_min = [None, None, None]
            dev_max = [None, None, None]
        cli_min = list(args.soft_limit_min_mm) if args.soft_limit_min_mm else [None, None, None]
        cli_max = list(args.soft_limit_max_mm) if args.soft_limit_max_mm else [None, None, None]
        soft_min = [cli_min[i] if cli_min[i] is not None else dev_min[i] for i in range(3)]
        soft_max = [cli_max[i] if cli_max[i] is not None else dev_max[i] for i in range(3)]
        if waypoints:
            _validate_soft_limits(waypoints, soft_min, soft_max)
        try:
            cur_mm = _read_current_pos_mm(controller, gantry_lock)
        except FMC4030Error:
            cur_mm = (math.nan, math.nan, math.nan)
        print(f"  Gantry: pos (mm)={cur_mm}  soft_min={soft_min}  soft_max={soft_max}",
              flush=True)
    else:
        soft_min = soft_max = [None, None, None]
        dev_min = dev_max = [None, None, None]
        cur_mm = (math.nan, math.nan, math.nan)

    # Write waypoints.csv (planned).
    with (run_dir / "waypoints.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "x_mm", "y_mm", "z_mm", "speed_mm_s", "dwell_s"])
        for i, wp in enumerate(waypoints):
            w.writerow([i, wp.x_mm, wp.y_mm, wp.z_mm, wp.speed_mm_s, wp.dwell_s])

    # Metadata.
    metadata: dict = {
        "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "calibration_path": str(args.fisheye_calib),
        "calibration_image_size": list(calib.image_size),
        "T_gantry_camera": calib.T_gantry_camera.tolist(),
        "K": calib.K.tolist(),
        "D": calib.D.flatten().tolist(),
        "scale_mm_per_unit": {a.name: SCALE_MM_PER_UNIT[a] for a in AXES},
        "soft_limit_min_mm": soft_min,
        "soft_limit_max_mm": soft_max,
        "device_soft_limit_min_mm": dev_min,
        "device_soft_limit_max_mm": dev_max,
        "start_unix": t0_unix,
        "start_monotonic": t0_mono,
    }

    def write_metadata() -> None:
        metadata.setdefault("end_unix", time.time())
        metadata.setdefault("end_monotonic", time.monotonic())
        with (run_dir / "run_metadata.json").open("w") as fh:
            json.dump(metadata, fh, indent=2)

    # Dry-run: connect + validate, write empty CSVs, exit.
    if args.dry_run:
        with (run_dir / "gantry_telemetry.csv").open("w", newline="") as fh:
            csv.writer(fh).writerow(GANTRY_CSV_COLUMNS)
        metadata["dry_run"] = True
        write_metadata()
        print("[dry-run] OK.", flush=True)
        if controller is not None:
            controller.close()
        return 0

    # Start gantry telemetry logger BEFORE motion (so we capture the move kickoff).
    if controller is not None:
        gantry_logger = GantryTelemetryLogger(
            controller,
            run_dir / "gantry_telemetry.csv",
            log_hz=args.log_hz,
            lock=gantry_lock,
            t0_monotonic=t0_mono,
        )
        gantry_logger.start()

    # Install SIGINT handler.
    def sigint_handler(signum, frame):
        del signum, frame
        if EMERGENCY_STOP.is_set():
            return
        EMERGENCY_STOP.set()
        print("\n!!! EMERGENCY STOP !!!", file=sys.stderr)
        if controller is not None:
            for axis in AXES:
                try:
                    with gantry_lock:
                        controller.stop_axis(axis, mode=2)
                except FMC4030Error:
                    pass
    signal.signal(signal.SIGINT, sigint_handler)

    # Camera open.
    cap = open_camera(args.camera_device, tuple(args.camera_resolution) if args.camera_resolution else None,
                      args.camera_fps)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {cam_w}x{cam_h} @ {cam_fps:.1f} fps (device={args.camera_device})", flush=True)

    # Fisheye undistortion maps. Use the size we actually opened at, not the
    # calibration's image_size, so a misconfigured camera resolution at least
    # produces a coherent (if slightly miscalibrated) rectification.
    if (cam_w, cam_h) != calib.image_size:
        print(
            f"! Camera resolution {cam_w}x{cam_h} differs from calibration "
            f"{calib.image_size}; undistortion will be approximate.",
            file=sys.stderr,
        )
    map1, map2, new_K = build_fisheye_undistort_maps(
        calib.K, calib.D, (cam_w, cam_h), args.fisheye_balance,
    )
    intrinsics = rectified_camera_intrinsics(new_K)
    print(
        f"Rectified K: fx={new_K[0,0]:.2f} fy={new_K[1,1]:.2f} cx={new_K[0,2]:.2f} cy={new_K[1,2]:.2f}",
        flush=True,
    )

    # Backend, detector.
    backend = TagSlamBackend(args)
    detector = make_detector(args)
    exclude_ids = _parse_exclude_ids(getattr(args, "exclude_tags", ""))
    if exclude_ids:
        print(f"[exclude] dropping tag IDs {sorted(exclude_ids)} from SLAM "
              "(blacklisted — e.g. physically duplicated IDs)", file=sys.stderr)
    object_points = tag_object_points(args.tag_size)
    refractive_context: RefractiveContext | None = None
    if args.water_correction_mode == "refractive":
        refractive_context = RefractiveContext(water_cfg=water_cfg, backend=backend)

    # Trajectory recorder (frames/ subdir per user spec).
    trajectory_recorder = TrajectoryRecorder(
        output_root=Path(args.trajectory_dir),
        image_width=args.trajectory_image_width,
        pool_cfg=pool_cfg,
        tag_size_m=args.tag_size,
        plot_z_scale=args.plot_z_scale,
        anchor_tag_id=args.anchor_tag_id,
        suffix="fisheye_gantry",
        frames_subdir="frames",
    )
    # Reuse our existing run_dir instead of creating a new one (so all artifacts
    # for this run land in one folder).
    trajectory_recorder.output_dir = run_dir
    trajectory_recorder.frames_dir = run_dir / "frames"
    if args.record_trajectory:
        trajectory_recorder.frames_dir.mkdir(parents=True, exist_ok=True)
        trajectory_recorder.active = True
        trajectory_recorder.start_monotonic_s = t0_mono
        trajectory_recorder.samples = []

    paused = threading.Event()
    record_toggle_pending = threading.Event()  # not used; r key handled inline

    if not args.no_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Kick off gantry motion (if any).
    if controller is not None and waypoints:
        motion_thread = GantryMotionThread(
            controller, waypoints,
            acc_mm_s2=args.acc_mm_s2, dec_mm_s2=args.dec_mm_s2,
            mode=args.mode, lock=gantry_lock, logger=gantry_logger,
            on_done=motion_done,
        )
        motion_thread.start()

    # Camera loop.
    cam_trajectory_world: list[tuple[float, float]] = []
    gantry_trajectory_mm: list[tuple[float, float]] = []
    frame_count = 0
    last_print_s = 0.0
    last_frame_s = time.monotonic()
    fps_value = 0.0
    motion_done_at: float | None = None  # monotonic timestamp when motion thread finished

    try:
        while True:
            if EMERGENCY_STOP.is_set():
                break

            ok, raw_frame = cap.read()
            if not ok:
                print("! Camera read failed; stopping loop.", file=sys.stderr)
                break
            frame_t_unix = time.time()
            frame_t_mono = time.monotonic()

            # Undistort full frame, then detect on the rectified pinhole image.
            frame = cv2.remap(raw_frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            observations = detect_observations(
                gray, detector, intrinsics, object_points, args, refractive_context,
            )
            observations = _filter_excluded(observations, exclude_ids)
            update = backend.update(observations)

            # FPS.
            now_s = time.monotonic()
            dt_s = now_s - last_frame_s
            if dt_s > 0:
                fps_value = (0.9 * fps_value + 0.1 / dt_s) if fps_value > 0 else 1.0 / dt_s
            last_frame_s = now_s

            # Pull latest gantry sample for the live error column.
            gantry_sample = gantry_logger.latest_sample() if gantry_logger is not None else None
            extra: dict[str, float] | None = None
            if gantry_sample is not None and update.camera_pose is not None:
                gx_mm, gy_mm, gz_mm = gantry_sample.pos_mm
                cam_world_est = np.array(pose_translation(update.camera_pose), dtype=np.float64)
                cam_world_gantry = gantry_to_world_translation_m(gantry_sample, calib.T_gantry_camera)
                err_m = float(np.linalg.norm(cam_world_est - cam_world_gantry))
                extra = {
                    "gantry_x_mm": float(gx_mm),
                    "gantry_y_mm": float(gy_mm),
                    "gantry_z_mm": float(gz_mm),
                    "translation_error_mm": err_m * 1000.0,
                }
                gantry_trajectory_mm.append((gx_mm, gy_mm))

            if update.camera_pose is not None:
                x, y, _ = pose_translation(update.camera_pose)
                cam_trajectory_world.append((float(x), float(y)))

            # Console print.
            if args.print_every > 0 and now_s - last_print_s >= args.print_every:
                print_backend_update(update, observations)
                last_print_s = now_s

            # Recorder append (with timestamps + extras if available).
            trajectory_recorder.append(
                update, observations, now_s, frame,
                timestamp_unix=frame_t_unix,
                timestamp_monotonic=frame_t_mono,
                extra=extra,
            )

            # Render & display.
            if not args.no_window:
                draw_observations(frame, observations)
                draw_overlay(
                    frame, update, observations, fps_value,
                    trajectory_recorder.active, len(trajectory_recorder.samples),
                )
                panel = render_topdown_panel(
                    width_px=480, height_px=max(360, frame.shape[0] // 2),
                    backend=backend,
                    cam_trajectory_world=cam_trajectory_world,
                    gantry_trajectory_world_mm=gantry_trajectory_mm,
                    T_gantry_camera=calib.T_gantry_camera,
                    anchor_tag_id=args.anchor_tag_id,
                )
                # Side-by-side: scale frame to display width, scale panel to match height.
                disp_scale = get_display_scale(frame.shape, args.display_width)
                disp = resize_for_display(frame, disp_scale)
                panel_h = disp.shape[0]
                panel_w = int(panel.shape[1] * (panel_h / panel.shape[0]))
                panel_disp = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                combined = np.hstack([disp, panel_disp])
                cv2.imshow(WINDOW_NAME, combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
                if key in (ord("r"), ord("R")):
                    if trajectory_recorder.active:
                        # We do NOT call stop_and_save mid-run for this script;
                        # we save once at the end into our shared run_dir.
                        trajectory_recorder.active = False
                        print("Recording paused.", flush=True)
                    else:
                        trajectory_recorder.frames_dir.mkdir(parents=True, exist_ok=True)
                        trajectory_recorder.active = True
                        if trajectory_recorder.start_monotonic_s == 0.0:
                            trajectory_recorder.start_monotonic_s = t0_mono
                        print("Recording resumed.", flush=True)
                if key == ord(" "):
                    if paused.is_set():
                        paused.clear()
                        print("Resumed.", flush=True)
                    else:
                        paused.set()
                        print("Paused (press space to resume).", flush=True)
                while paused.is_set() and not EMERGENCY_STOP.is_set():
                    k = cv2.waitKey(50) & 0xFF
                    if k == ord(" "):
                        paused.clear()
                        break
                    if k in (ord("q"), 27):
                        EMERGENCY_STOP.set()
                        break

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            # After the motion thread finishes, drain ~1 s of extra frames so
            # the post-stop ringdown lands in the logger, then exit. Without
            # a motion thread (passive --no-gantry mode), the loop runs until
            # the user quits or --max-frames is hit.
            if motion_done.is_set() and motion_thread is not None and motion_thread.error is None:
                if motion_done_at is None:
                    motion_done_at = now_s
                elif now_s - motion_done_at >= 1.0:
                    break

    finally:
        # Stop motion (if running) and capture any error.
        if motion_thread is not None:
            EMERGENCY_STOP.set()
            motion_thread.join(timeout=5.0)
            if motion_thread.error is not None:
                metadata["motion_error"] = repr(motion_thread.error)

        # Stop logger and write final CSV/HTML/plot via recorder.
        if gantry_logger is not None:
            gantry_logger.stop()

        # Save trajectory artifacts in the SAME run_dir (recorder uses
        # output_dir we pre-set).
        if trajectory_recorder.samples:
            trajectory_recorder.active = False
            try:
                trajectory_recorder.stop_and_save(backend)
            except Exception as exc:
                print(f"! Trajectory save error: {exc}", file=sys.stderr)
                metadata["trajectory_save_error"] = str(exc)
        else:
            print("No optimized camera poses captured; skipping CSV/HTML/plot writes.",
                  flush=True)

        metadata["end_unix"] = time.time()
        metadata["end_monotonic"] = time.monotonic()
        metadata["frame_count"] = frame_count
        metadata["recorded_samples"] = len(trajectory_recorder.samples)
        write_metadata()

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if controller is not None:
            try:
                controller.close()
            except FMC4030Error:
                pass

    return 0


if __name__ == "__main__":
    # No CLI args -> launch the Tkinter GUI; otherwise run the camera+gantry
    # pipeline directly. The GUI re-invokes this same script as a subprocess
    # with the chosen args, so the CLI path is always authoritative.
    if len(sys.argv) == 1:
        from gui_launcher import launch_fisheye_gui
        raise SystemExit(launch_fisheye_gui(Path(__file__).resolve()))
    raise SystemExit(main())
