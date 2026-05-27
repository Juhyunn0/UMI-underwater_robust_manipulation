#!/usr/bin/env python3
"""
experiment_runner.py — End-to-end experiment orchestration for the UMI gantry.

Coordinates:
  • GantryTelemetryLogger (from gantry_runner)
  • FisheyeGantryWorker  (from fisheye_gantry_tagslam)
  • Post-run comparison plots (from tagslam_core)

State machine:
    IDLE → COUNTDOWN → MOTION → SETTLE → POSTPROCESS → DONE
    Any state → POSTPROCESS (via stop_experiment() or abort_event)

Usage (from gantry_panel.py):
    runner = ExperimentRunner(parent=panel)
    runner.state_changed.connect(my_slot)
    runner.finished.connect(my_result_slot)
    runner.start_experiment(config)
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any

# sys.path shim
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal

from gantry_runner import (
    AXES,
    GantryTelemetryLogger,
    Waypoint,
    move_to_xyz_mm,
)
from gantry import Axis, FMC4030Error


# ─────────────────────────────────────────────────────────────────────────────
# Phase enum
# ─────────────────────────────────────────────────────────────────────────────
class Phase(Enum):
    IDLE        = "IDLE"
    COUNTDOWN   = "COUNTDOWN"
    MOTION      = "MOTION"
    SETTLE      = "SETTLE"
    POSTPROCESS = "POSTPROCESS"
    DONE        = "DONE"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass (passed to start_experiment)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    # Gantry
    controller: Any                      # FMC4030Controller (real or mock)
    controller_lock: threading.RLock
    waypoints: list[Waypoint]
    soft_min_mm: list[float | None]
    soft_max_mm: list[float | None]
    move_mode: str = "line"              # "line" or "sequential"
    acc_mm_s2: float = 50.0
    dec_mm_s2: float = 50.0
    log_hz: float = 100.0

    # Timing
    countdown_s: float = 2.0
    settle_s: float = 2.0
    tag_detection_while_idle: bool = True  # start fisheye during countdown

    # Output
    output_root: Path = field(default_factory=lambda: Path("data"))
    run_name: str = ""

    # Fisheye / TagSLAM (None → skip fisheye)
    fisheye_args: Any = None             # argparse.Namespace from fisheye_gantry_tagslam
    fisheye_calib: Any = None            # FisheyeCalibration dataclass

    # Abort hook
    abort_event: threading.Event = field(default_factory=threading.Event)

    # Mock camera (no real device)
    mock_camera: bool = False

    # Persistent camera session from the panel (shared; no second VideoCapture opened).
    # When set, FisheyeWorkerThread reads from its worker queue instead of opening cv2.
    camera_session: Any = None

    # ── Experiment-mode metadata (gantry-only vs full pipeline) ──
    # "fisheye" runs the full camera pipeline; "gantry_only" skips fisheye
    # entirely and writes a reduced output set.
    camera_mode: str = "fisheye"
    # Per-axis direction sign at panel↔SDK boundary (panel field), recorded for
    # the run_metadata.json so post-hoc analysis can reproduce user-frame.
    axis_sign: dict | None = None
    # Snapshot of user-frame waypoints AT START — preserves the values the user
    # actually entered (waypoints field above is firmware-frame for the SDK).
    waypoints_user_frame: list | None = None
    # Per-axis absolute-mm home reference at experiment start. Lets post-hoc
    # analysis reproduce user-frame coordinates without ambiguity.
    home_reference_abs_mm: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Fisheye stats sample emitted by FisheyeWorkerThread via queue
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FisheyeStatsSample:
    tags_this_frame: int
    tags_in_graph: int
    backend_updates: int
    drift_mm: float     # last ||p_cam - T(gantry)||  (nan if unavailable)


# ─────────────────────────────────────────────────────────────────────────────
# FisheyeWorkerThread — wraps FisheyeGantryWorker in a QThread
# ─────────────────────────────────────────────────────────────────────────────
class FisheyeWorkerThread(QThread):
    """Runs the fisheye+TagSLAM loop in a background QThread.

    Emits stats_ready at ~5 Hz via a Qt signal so the GUI can update without
    polling the worker from the GUI thread.
    """

    stats_ready = pyqtSignal(object)   # FisheyeStatsSample

    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        gantry_logger: GantryTelemetryLogger | None,
        t0_monotonic: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._run_dir = run_dir
        self._gantry_logger = gantry_logger
        self._t0 = t0_monotonic
        self._stop_event = threading.Event()
        self._stats_queue: Queue[FisheyeStatsSample] = Queue(maxsize=10)
        self._trajectory_recorder: Any = None   # set after run() creates it
        self._backend: Any = None               # set after run() creates it

    def stop(self) -> None:
        self._stop_event.set()

    def get_trajectory_recorder(self) -> Any:
        return self._trajectory_recorder

    def get_backend(self) -> Any:
        return self._backend

    def run(self) -> None:  # type: ignore[override]
        try:
            self._run_inner()
        except Exception as exc:
            print(f"[fisheye-worker] unexpected error: {exc}", file=sys.stderr)

    def _run_inner(self) -> None:
        config = self._config
        if config.fisheye_args is None or config.fisheye_calib is None:
            return  # fisheye not configured

        try:
            import cv2
            import numpy as np
            from fisheye_gantry_tagslam import (
                build_fisheye_undistort_maps,
                gantry_to_world_translation_m,
                rectified_camera_intrinsics,
            )
            from tagslam_core import (
                RefractiveContext,
                TagSlamBackend,
                TrajectoryRecorder,
                detect_observations,
                draw_observations,
                draw_overlay,
                get_display_scale,
                make_detector,
                normalize_water_config,
                parse_simple_yaml,
                pose_translation,
                print_backend_update,
                resize_for_display,
                tag_object_points,
            )
            from tagslam.visualization import normalize_pool_config
        except ImportError as exc:
            print(f"[fisheye-worker] import error: {exc}", file=sys.stderr)
            return

        args = config.fisheye_args
        calib = config.fisheye_calib

        # Pool / water config
        runtime_config: dict = {}
        try:
            from pathlib import Path as _Path
            config_path = _Path(getattr(args, "config", "config/config.yaml"))
            if config_path.exists():
                with config_path.open() as fh:
                    text = fh.read()
                runtime_config = parse_simple_yaml(text)
        except Exception:
            pass
        pool_cfg = normalize_pool_config(runtime_config.get("pool", {}))
        water_cfg = normalize_water_config(runtime_config.get("water"), pool_cfg)

        # Determine the frame source:
        #   A) panel camera session  → shared Queue(maxsize=2), no new VideoCapture
        #   B) mock_camera flag      → _MockCamera (no hardware)
        #   C) default               → open cv2.VideoCapture directly
        frame_queue: Queue | None = None

        if config.camera_session is not None and config.camera_session.is_open:
            # Path A: attach to the panel's grab thread via a drop-oldest queue.
            frame_queue = Queue(maxsize=2)
            config.camera_session.attach_worker_queue(frame_queue)
            # Derive resolution from the session's config.
            sess_cfg = config.camera_session.device_config
            cam_w = int(sess_cfg.get("width", 1280))
            cam_h = int(sess_cfg.get("height", 720))
            cap = None
        elif config.mock_camera:
            # Path B: synthetic frames.
            cap = _MockCamera(calib.image_size[0], calib.image_size[1])
            cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            # Path C: open our own VideoCapture.
            try:
                device = getattr(args, "camera_device", "0")
                resolution = getattr(args, "camera_resolution", None)
                fps = getattr(args, "camera_fps", None)
                try:
                    idx = int(device)
                    cap = cv2.VideoCapture(idx)
                except ValueError:
                    cap = cv2.VideoCapture(device)
                if not cap.isOpened():
                    print(f"[fisheye-worker] cannot open camera {device!r}", file=sys.stderr)
                    return
                if resolution:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolution[0]))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolution[1]))
                if fps:
                    cap.set(cv2.CAP_PROP_FPS, float(fps))
            except Exception as exc:
                print(f"[fisheye-worker] camera open error: {exc}", file=sys.stderr)
                return
            cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        map1, map2, new_K = build_fisheye_undistort_maps(
            calib.K, calib.D, (cam_w, cam_h),
            getattr(args, "fisheye_balance", 0.0),
        )
        intrinsics = rectified_camera_intrinsics(new_K)

        backend = TagSlamBackend(args)
        self._backend = backend
        detector = make_detector(args)
        object_points = tag_object_points(args.tag_size)
        refractive_context = None
        if getattr(args, "water_correction_mode", "none") == "refractive":
            refractive_context = RefractiveContext(water_cfg=water_cfg, backend=backend)

        trajectory_recorder = TrajectoryRecorder(
            output_root=config.output_root,
            image_width=getattr(args, "trajectory_image_width", 960),
            pool_cfg=pool_cfg,
            tag_size_m=args.tag_size,
            plot_z_scale=getattr(args, "plot_z_scale", 1.0),
            anchor_tag_id=args.anchor_tag_id,
            suffix="fisheye_gantry",
            frames_subdir="frames",
        )
        trajectory_recorder.output_dir = self._run_dir
        trajectory_recorder.frames_dir = self._run_dir / "frames"
        # TrajectoryRecorder.start() normally creates frames_dir, but we're
        # bypassing it here by setting fields directly — so we must mkdir
        # ourselves or every cv2.imwrite below silently fails and spams
        # "Warning: could not save ZED trajectory frame ..." to stderr.
        trajectory_recorder.frames_dir.mkdir(parents=True, exist_ok=True)
        trajectory_recorder.active = True
        trajectory_recorder.start_monotonic_s = self._t0
        trajectory_recorder.samples = []
        self._trajectory_recorder = trajectory_recorder

        last_stats_emit = 0.0
        backend_update_count = 0
        frame_count = 0

        try:
            while not self._stop_event.is_set():
                if frame_queue is not None:
                    # Path A: pull from shared queue (drop-oldest semantics on
                    # the producer side, so this never blocks the grab thread).
                    try:
                        raw_frame, frame_t_mono = frame_queue.get(timeout=0.2)
                    except Empty:
                        continue
                    frame_t_unix = time.time()
                else:
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
                update = backend.update(observations)
                if update.camera_pose is not None:
                    backend_update_count += 1

                now_s = time.monotonic()

                # Drift from gantry
                drift_mm = math.nan
                gantry_sample = (
                    self._gantry_logger.latest_sample()
                    if self._gantry_logger is not None else None
                )
                extra: dict | None = None
                if gantry_sample is not None and update.camera_pose is not None:
                    import numpy as np
                    gx, gy, gz = gantry_sample.pos_mm
                    cam_est = np.array(pose_translation(update.camera_pose), dtype=np.float64)
                    cam_gt = gantry_to_world_translation_m(gantry_sample, calib.T_gantry_camera)
                    drift_mm = float(np.linalg.norm(cam_est - cam_gt)) * 1000.0
                    extra = {
                        "gantry_x_mm": float(gx),
                        "gantry_y_mm": float(gy),
                        "gantry_z_mm": float(gz),
                        "translation_error_mm": drift_mm,
                    }

                trajectory_recorder.append(
                    update, observations, now_s, frame,
                    timestamp_unix=frame_t_unix,
                    timestamp_monotonic=frame_t_mono,
                    extra=extra,
                )
                frame_count += 1

                # Emit stats at ~5 Hz without blocking GUI
                if now_s - last_stats_emit >= 0.2:
                    last_stats_emit = now_s
                    sample = FisheyeStatsSample(
                        tags_this_frame=len(observations),
                        tags_in_graph=len(backend.optimized_tag_poses()),
                        backend_updates=backend_update_count,
                        drift_mm=drift_mm,
                    )
                    self.stats_ready.emit(sample)

        finally:
            # Detach worker queue before the session grabs any more frames.
            if frame_queue is not None and config.camera_session is not None:
                try:
                    config.camera_session.detach_worker_queue()
                except Exception:
                    pass
            if cap is not None and hasattr(cap, "release"):
                cap.release()
            # NOTE: do NOT set trajectory_recorder.active = False here.
            # stop_and_save() short-circuits on `if not self.active: return None`,
            # so flipping it early makes the postprocess write nothing while
            # silently filling the metadata with paths that don't exist.


# ─────────────────────────────────────────────────────────────────────────────
# Mock camera (generates synthetic gray frames when no device is attached)
# ─────────────────────────────────────────────────────────────────────────────
class _MockCamera:
    """Minimal cv2.VideoCapture-compatible mock that produces blank frames."""

    def __init__(self, width: int, height: int) -> None:
        import numpy as np
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._w = float(width)
        self._h = float(height)

    def get(self, prop: int) -> float:
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._h
        return 0.0

    def read(self) -> tuple[bool, Any]:
        import numpy as np
        # Simulate ~30 fps
        time.sleep(1.0 / 30.0)
        return True, self._frame.copy()

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MotionWorkerThread — runs the waypoint sequence in a background QThread
# ─────────────────────────────────────────────────────────────────────────────
class MotionWorkerThread(QThread):
    waypoint_started = pyqtSignal(int, int)   # (index, total)
    waypoint_done    = pyqtSignal(int, str)   # (index, "" or error)
    motion_done      = pyqtSignal(str)        # "" or error message

    def __init__(
        self,
        config: ExperimentConfig,
        gantry_logger: GantryTelemetryLogger | None,
        abort_event: threading.Event,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._logger = gantry_logger
        self._abort = abort_event

    def _aborted(self) -> bool:
        return self._abort.is_set()

    def run(self) -> None:  # type: ignore[override]
        cfg = self._config
        total = len(cfg.waypoints)
        try:
            for i, wp in enumerate(cfg.waypoints):
                if self._aborted():
                    self.motion_done.emit("aborted")
                    return
                self.waypoint_started.emit(i, total)
                try:
                    move_to_xyz_mm(
                        cfg.controller,
                        (wp.x_mm, wp.y_mm, wp.z_mm),
                        wp.speed_mm_s,
                        cfg.acc_mm_s2,
                        cfg.dec_mm_s2,
                        mode=cfg.move_mode,
                        lock=cfg.controller_lock,
                        logger=self._logger,
                        waypoint_index=i,
                    )
                except Exception as exc:
                    import traceback
                    err = f"waypoint[{i}]: {exc}"
                    print(f"[motion-worker] {err}", file=sys.stderr)
                    traceback.print_exc()
                    self.waypoint_done.emit(i, err)
                    self.motion_done.emit(err)
                    return
                self.waypoint_done.emit(i, "")
                # Dwell
                t_end = time.monotonic() + max(0.0, wp.dwell_s)
                while time.monotonic() < t_end:
                    if self._aborted():
                        self.motion_done.emit("aborted")
                        return
                    time.sleep(min(0.05, t_end - time.monotonic()))
            self.motion_done.emit("")
        except Exception as exc:
            self.motion_done.emit(f"unexpected: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PostprocessThread — writes plots in a background QThread
# ─────────────────────────────────────────────────────────────────────────────
class PostprocessThread(QThread):
    progress = pyqtSignal(str)            # step description
    done = pyqtSignal(dict, str)          # (result_dict, "" or error)

    def __init__(
        self,
        run_dir: Path,
        config: ExperimentConfig,
        gantry_logger: GantryTelemetryLogger | None,
        fisheye_thread: FisheyeWorkerThread | None,
        t0_monotonic: float,
        aborted: bool,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_dir = run_dir
        self._config = config
        self._gantry_logger = gantry_logger
        self._fisheye = fisheye_thread
        self._t0 = t0_monotonic
        self._aborted = aborted
        self._stop_event = threading.Event()
        # Pre-extract data while the fisheye thread is still a valid C++ object.
        # After stop()+wait() the thread may be deleted via deleteLater before
        # run() tries to access it, causing RuntimeError and silently killing the
        # thread without emitting done.
        self._fisheye_backend  = fisheye_thread.get_backend()             if fisheye_thread is not None else None
        self._fisheye_recorder = fisheye_thread.get_trajectory_recorder() if fisheye_thread is not None else None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # type: ignore[override]
        run_dir = self._run_dir
        config = self._config
        result: dict[str, Any] = {
            "run_dir": str(run_dir),
            "aborted": self._aborted,
        }
        errors: list[str] = []
        try:
            self._run_body(run_dir, config, result, errors)
        except Exception as exc:
            import traceback
            print(f"[postprocess] unhandled: {traceback.format_exc()}", file=sys.stderr)
            errors.append(f"unhandled: {exc}")
            result["errors"] = errors
            self.done.emit(result, f"unhandled: {exc}")

    def _run_body(self, run_dir, config, result, errors) -> None:  # type: ignore[override]
        # 1) Stop loggers
        self.progress.emit("Stopping loggers…")
        if self._gantry_logger is not None:
            try:
                self._gantry_logger.stop()
            except Exception as exc:
                errors.append(f"gantry_logger.stop: {exc}")

        if self._fisheye is not None:
            try:
                self._fisheye.stop()
                self._fisheye.wait(5000)
            except RuntimeError:
                pass  # C++ object already deleted — thread already finished naturally

        if self._stop_event.is_set():
            self.done.emit(result, "Stopped by user")
            return

        gantry_only = getattr(config, "camera_mode", "fisheye") == "gantry_only"

        # 2) Save fisheye trajectory (skipped entirely in gantry-only mode)
        backend  = self._fisheye_backend
        recorder = self._fisheye_recorder

        if not gantry_only:
            self.progress.emit("Writing camera trajectory CSV + HTML…")
            if recorder is not None and recorder.samples and backend is not None:
                try:
                    out = recorder.stop_and_save(backend)
                    if out is None:
                        errors.append(
                            "recorder.stop_and_save returned None — recorder "
                            "was inactive or had no optimized camera poses; "
                            "trajectory CSV/HTML were not written"
                        )
                    else:
                        # Only advertise the outputs that actually exist on disk.
                        for key, fname in (
                            ("camera_trajectory_csv", "camera_trajectory.csv"),
                            ("tag_poses_csv",          "tag_poses.csv"),
                            ("trajectory_html",        "trajectory_interactive.html"),
                        ):
                            p = run_dir / fname
                            if p.exists():
                                result[key] = str(p)
                except Exception as exc:
                    errors.append(f"recorder.stop_and_save: {exc}")

        gantry_csv  = run_dir / "gantry_telemetry.csv"
        camera_csv  = run_dir / "camera_trajectory.csv"
        tag_csv     = run_dir / "tag_poses.csv"

        result["gantry_telemetry_csv"] = str(gantry_csv)

        # 2b) Always write waypoints.csv (user-frame if available, otherwise
        # firmware-frame as-fed-to-SDK). Cheap and useful for both modes; the
        # gantry-only mode requirement also makes it a hard guarantee.
        try:
            waypoints_csv_path = run_dir / "waypoints.csv"
            with waypoints_csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["x_mm", "y_mm", "z_mm", "speed_mm_s", "dwell_s"])
                user_wps = getattr(config, "waypoints_user_frame", None)
                if user_wps:
                    for wp in user_wps:
                        w.writerow([wp["x_mm"], wp["y_mm"], wp["z_mm"],
                                    wp["speed_mm_s"], wp["dwell_s"]])
                else:
                    for wp in config.waypoints:
                        w.writerow([wp.x_mm, wp.y_mm, wp.z_mm, wp.speed_mm_s, wp.dwell_s])
            result["waypoints_csv"] = str(waypoints_csv_path)
        except Exception as exc:
            errors.append(f"waypoints_csv: {exc}")

        if self._stop_event.is_set():
            self.done.emit(result, "Stopped by user")
            return

        # In gantry-only mode, write the single-mode plot and skip the overlay
        # and tag-overlay plots entirely.
        if gantry_only:
            if gantry_csv.exists():
                self.progress.emit("Writing gantry pose/vel/acc plot…")
                try:
                    import tagslam_core as _tc
                    png_path = run_dir / "gantry_pose_velocity_acceleration.png"
                    if hasattr(_tc, "write_gantry_only_plot"):
                        out = _tc.write_gantry_only_plot(png_path, gantry_csv)
                    else:
                        out = None
                        errors.append(
                            "write_gantry_only_plot helper missing in tagslam_core"
                        )
                    if out is not None:
                        result["gantry_pose_velocity_acceleration_plot"] = str(out)
                except Exception as exc:
                    errors.append(f"gantry_only_plot: {exc}")

        # 3) Overlay top-down plot (full pipeline only)
        if not gantry_only and gantry_csv.exists() and camera_csv.exists() and tag_csv.exists():
            self.progress.emit("Writing overlay top-down plot…")
            try:
                import tagslam_core as _tc
                gantry_rows = _read_csv(gantry_csv)
                camera_rows = _read_csv(camera_csv)
                tag_rows    = _read_csv(tag_csv)

                T_gc = None
                if config.fisheye_calib is not None:
                    T_gc = config.fisheye_calib.T_gantry_camera

                # Load pool config for overlay
                runtime_cfg: dict = {}
                try:
                    from tagslam_core import parse_simple_yaml
                    from tagslam.visualization import normalize_pool_config
                    cfg_p = Path(getattr(config.fisheye_args, "config", "config/config.yaml"))
                    if cfg_p.exists():
                        runtime_cfg = parse_simple_yaml(cfg_p.read_text())
                except Exception:
                    pass
                from tagslam.visualization import normalize_pool_config
                pool_cfg = normalize_pool_config(runtime_cfg.get("pool", {}))

                # gantry_anchor_offset_mm — check calibration YAML
                anchor_offset = None
                if config.fisheye_args is not None:
                    calib_path = Path(getattr(config.fisheye_args, "fisheye_calib", ""))
                    if calib_path.exists():
                        try:
                            import yaml as _yaml
                            with calib_path.open() as fh:
                                calib_data = _yaml.safe_load(fh) or {}
                            if "gantry_anchor_offset_mm" in calib_data:
                                anchor_offset = [float(v) for v in calib_data["gantry_anchor_offset_mm"]]
                        except Exception:
                            pass

                anchor_id = getattr(config.fisheye_args, "anchor_tag_id", 1)
                plot_path = run_dir / "comparison_topdown.png"
                out = _tc.write_overlay_topdown_plot(
                    plot_path, gantry_rows, camera_rows, tag_rows,
                    anchor_id, T_gc, pool_cfg,
                    gantry_anchor_offset_mm=anchor_offset,
                    run_name=config.run_name or run_dir.name,
                )
                if out is not None:
                    result["topdown_plot"] = str(out)
            except Exception as exc:
                errors.append(f"overlay_topdown_plot: {exc}")

        # 4) Pose/velocity/acceleration plots (full pipeline only)
        if not gantry_only and gantry_csv.exists() and camera_csv.exists():
            self.progress.emit("Writing pose/velocity/acceleration plot…")
            try:
                import tagslam_core as _tc
                png_path = run_dir / "comparison_plot.png"
                out = _tc.write_pose_velocity_acceleration_plot(png_path, gantry_csv, camera_csv)
                if out is not None:
                    result["comparison_plot"] = str(out)
            except Exception as exc:
                errors.append(f"pva_plot: {exc}")

            self.progress.emit("Writing interactive comparison HTML…")
            try:
                import tagslam_core as _tc
                html_path = run_dir / "pose_velocity_acceleration.html"
                out = _tc.write_pose_velocity_acceleration_html(html_path, gantry_csv, camera_csv)
                result["comparison_html"] = str(out)
            except Exception as exc:
                errors.append(f"pva_html: {exc}")

        # 5) Metadata
        self.progress.emit("Writing run_metadata.json…")
        try:
            metadata: dict[str, Any] = {
                "run_dir": str(run_dir),
                "aborted": self._aborted,
                "t0_monotonic": self._t0,
                "end_monotonic": time.monotonic(),
                "end_unix": time.time(),
                "run_name": config.run_name or run_dir.name,
                "camera_mode": getattr(config, "camera_mode", "fisheye"),
                "axis_sign": getattr(config, "axis_sign", None),
                "home_reference_abs_mm": getattr(config, "home_reference_abs_mm", None),
                "outputs": {k: v for k, v in result.items() if k not in ("run_dir", "aborted")},
            }
            if not gantry_only:
                metadata["alignment"] = (
                    "first-sample-zeroed (calibrate gantry_anchor_offset_mm to remove this fallback)"
                    if not self._has_anchor_offset() else "gantry_anchor_offset_mm"
                )
            if errors:
                metadata["postprocess_errors"] = errors
            with (run_dir / "run_metadata.json").open("w") as fh:
                json.dump(metadata, fh, indent=2)
            result["metadata"] = str(run_dir / "run_metadata.json")
        except Exception as exc:
            errors.append(f"metadata: {exc}")

        result["errors"] = errors
        err_str = "; ".join(errors) if errors else ""
        self.done.emit(result, err_str)

    def _has_anchor_offset(self) -> bool:
        if self._config.fisheye_args is None:
            return False
        try:
            import yaml as _yaml
            calib_path = Path(getattr(self._config.fisheye_args, "fisheye_calib", ""))
            if not calib_path.exists():
                return False
            with calib_path.open() as fh:
                data = _yaml.safe_load(fh) or {}
            return "gantry_anchor_offset_mm" in data
        except Exception:
            return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentRunner — main Qt orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class ExperimentRunner(QObject):
    """Orchestrates a single end-to-end experiment run.

    Phases: IDLE → COUNTDOWN → MOTION → SETTLE → POSTPROCESS → DONE.

    Signals (all emitted on the GUI thread):
        state_changed(phase_str, message)
        countdown_tick(remaining_s)
        waypoint_progress(current, total)
        fisheye_stats(FisheyeStatsSample)
        error(message)
        finished(result_dict)
    """

    state_changed     = pyqtSignal(str, str)    # (Phase.value, human message)
    countdown_tick    = pyqtSignal(float)        # seconds remaining
    waypoint_progress = pyqtSignal(int, int)     # (current_idx+1, total)
    fisheye_stats     = pyqtSignal(object)       # FisheyeStatsSample
    error             = pyqtSignal(str)
    finished          = pyqtSignal(dict)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._phase = Phase.IDLE
        self._config: ExperimentConfig | None = None
        self._run_dir: Path | None = None
        self._t0: float = 0.0

        self._gantry_logger: GantryTelemetryLogger | None = None
        self._fisheye_thread: FisheyeWorkerThread | None = None
        self._motion_thread: MotionWorkerThread | None = None
        self._postprocess_thread: PostprocessThread | None = None

        self._countdown_timer: QTimer | None = None
        self._settle_timer: QTimer | None = None
        self._countdown_remaining: float = 0.0
        self._aborted = False
        self._motion_error: str = ""

    # ---- public API --------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    def is_idle(self) -> bool:
        return self._phase == Phase.IDLE

    def is_running(self) -> bool:
        return self._phase not in (Phase.IDLE, Phase.DONE)

    def start_experiment(self, config: ExperimentConfig) -> None:
        if not self.is_idle():
            self.error.emit("Cannot start: experiment already running")
            return
        self._config = config
        self._aborted = False
        self._motion_error = ""
        self._enter_countdown()

    def start_recording(self, config: ExperimentConfig) -> None:
        """Manual recording session: start the gantry telemetry logger and (if
        configured) the fisheye AprilTag SLAM worker, and sit in MOTION phase
        until ``stop_experiment()`` is called. No countdown, no motion thread,
        no settle. The user moves the gantry by hand from the Control tab while
        this is running; stop triggers the standard postprocess pipeline which
        writes the same plots/CSVs/HTML as a full experiment."""
        if not self.is_idle():
            self.error.emit("Cannot start: experiment already running")
            return
        self._config = config
        self._aborted = False
        self._motion_error = ""

        from tagslam_core import make_run_dir
        self._run_dir = make_run_dir(config.output_root, "recording")
        if config.run_name:
            new_dir = self._run_dir.parent / (self._run_dir.name + "_" + config.run_name)
            try:
                self._run_dir.rename(new_dir)
                self._run_dir = new_dir
            except Exception:
                pass

        self._t0 = time.monotonic()
        self._gantry_logger = GantryTelemetryLogger(
            config.controller,
            self._run_dir / "gantry_telemetry.csv",
            log_hz=config.log_hz,
            lock=config.controller_lock,
            t0_monotonic=self._t0,
        )
        self._gantry_logger.start()

        if config.fisheye_args is not None:
            self._start_fisheye_thread()

        # Sit in MOTION until the user stops. stop_experiment() already
        # handles MOTION → _stop_all_workers() → _enter_postprocess().
        self._set_phase(
            Phase.MOTION,
            "Recording — move gantry manually, click Stop to finish",
        )

    def stop_experiment(self) -> None:
        if self._phase in (Phase.IDLE, Phase.DONE):
            return
        self._aborted = True
        self._config.abort_event.set()
        if self._phase == Phase.POSTPROCESS:
            if self._postprocess_thread is not None:
                try:
                    self._postprocess_thread.stop()
                except RuntimeError:
                    pass  # C++ object already deleted
            return
        self._stop_all_workers()
        self._enter_postprocess()

    # ---- private: phase transitions ----------------------------------------

    def _enter_countdown(self) -> None:
        cfg = self._config
        assert cfg is not None

        # Create output directory
        from tagslam_core import make_run_dir
        self._run_dir = make_run_dir(cfg.output_root, "experiment")
        if cfg.run_name:
            # Rename to include user-specified name
            new_dir = self._run_dir.parent / (self._run_dir.name + "_" + cfg.run_name)
            try:
                self._run_dir.rename(new_dir)
                self._run_dir = new_dir
            except Exception:
                pass  # keep original if rename fails

        # Shared epoch
        self._t0 = time.monotonic()

        # Start gantry logger
        self._gantry_logger = GantryTelemetryLogger(
            cfg.controller,
            self._run_dir / "gantry_telemetry.csv",
            log_hz=cfg.log_hz,
            lock=cfg.controller_lock,
            t0_monotonic=self._t0,
        )
        self._gantry_logger.start()

        # Start fisheye early if tag_detection_while_idle
        if cfg.tag_detection_while_idle and cfg.fisheye_args is not None:
            self._start_fisheye_thread()

        # Countdown timer
        self._countdown_remaining = cfg.countdown_s
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(100)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)
        self._countdown_timer.start()

        self._set_phase(Phase.COUNTDOWN, f"Starting in {cfg.countdown_s:.1f}s…")

    def _on_countdown_tick(self) -> None:
        self._countdown_remaining -= 0.1
        self.countdown_tick.emit(max(0.0, self._countdown_remaining))
        if self._countdown_remaining <= 0.0:
            if self._countdown_timer is not None:
                self._countdown_timer.stop()
                self._countdown_timer = None
            self._enter_motion()

    def _enter_motion(self) -> None:
        cfg = self._config
        assert cfg is not None

        # Start fisheye now if not already started
        if self._fisheye_thread is None and cfg.fisheye_args is not None:
            self._start_fisheye_thread()

        self._motion_thread = MotionWorkerThread(
            cfg, self._gantry_logger, cfg.abort_event, self,
        )
        self._motion_thread.waypoint_started.connect(
            lambda i, t: self.waypoint_progress.emit(i + 1, t)
        )
        self._motion_thread.motion_done.connect(self._on_motion_done)
        self._motion_thread.finished.connect(self._motion_thread.deleteLater)
        self._motion_thread.start()

        self._set_phase(Phase.MOTION, f"Running {len(cfg.waypoints)} waypoints…")

    def _on_motion_done(self, err: str) -> None:
        if err and err != "aborted":
            # Motion failed (controller exception, disconnect, etc). Don't
            # pretend everything is fine and proceed through settle — capture
            # the error, skip settle, and surface it on the result.
            self._motion_error = err
            self._aborted = True
            self.error.emit(f"Motion error: {err}")
            self._enter_postprocess()
            return
        if self._aborted:
            self._enter_postprocess()
            return
        self._enter_settle()

    def _enter_settle(self) -> None:
        cfg = self._config
        assert cfg is not None
        self._set_phase(Phase.SETTLE, f"Settling {cfg.settle_s:.1f}s…")
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(int(cfg.settle_s * 1000))
        self._settle_timer.timeout.connect(self._enter_postprocess)
        self._settle_timer.start()

    def _enter_postprocess(self) -> None:
        if self._phase == Phase.POSTPROCESS:
            return  # guard against double-entry
        # Stop timers
        if self._countdown_timer is not None:
            self._countdown_timer.stop()
            self._countdown_timer = None
        if self._settle_timer is not None:
            self._settle_timer.stop()
            self._settle_timer = None

        self._set_phase(Phase.POSTPROCESS, "Post-processing…")

        self._postprocess_thread = PostprocessThread(
            self._run_dir, self._config,
            self._gantry_logger, self._fisheye_thread,
            self._t0, self._aborted, self,
        )
        self._postprocess_thread.progress.connect(
            lambda msg: self._set_phase(Phase.POSTPROCESS, msg)
        )
        self._postprocess_thread.done.connect(self._on_postprocess_done)
        self._postprocess_thread.finished.connect(self._postprocess_thread.deleteLater)
        self._postprocess_thread.finished.connect(self._on_postprocess_thread_finished)
        self._postprocess_thread.start()

    def _on_postprocess_done(self, result: dict, err: str) -> None:
        if self._phase != Phase.POSTPROCESS:
            return  # already handled by finished-signal fallback
        if err:
            self.error.emit(f"Post-process warnings: {err}")
        if self._motion_error:
            result["motion_error"] = self._motion_error
            self._set_phase(Phase.DONE, f"FAILED — motion error: {self._motion_error}")
        else:
            self._set_phase(Phase.DONE, "Experiment complete.")
        self.finished.emit(result)
        # Reset to IDLE so a new experiment can be started
        self._phase = Phase.IDLE

    def _on_postprocess_thread_finished(self) -> None:
        """Fallback: force DONE if run() exited without emitting done (e.g. crash)."""
        if self._phase == Phase.POSTPROCESS:
            result = {"aborted": self._aborted}
            if self._motion_error:
                result["motion_error"] = self._motion_error
                self._set_phase(Phase.DONE, f"FAILED — motion error: {self._motion_error}")
            else:
                self._set_phase(Phase.DONE, "Experiment complete.")
            self.finished.emit(result)
            self._phase = Phase.IDLE

    # ---- helpers -----------------------------------------------------------

    def _set_phase(self, phase: Phase, msg: str) -> None:
        self._phase = phase
        self.state_changed.emit(phase.value, msg)

    def _start_fisheye_thread(self) -> None:
        cfg = self._config
        assert cfg is not None and self._run_dir is not None
        self._fisheye_thread = FisheyeWorkerThread(
            cfg, self._run_dir, self._gantry_logger, self._t0, self,
        )
        self._fisheye_thread.stats_ready.connect(self.fisheye_stats.emit)
        self._fisheye_thread.finished.connect(self._fisheye_thread.deleteLater)
        self._fisheye_thread.start()

    def _stop_all_workers(self) -> None:
        if self._motion_thread is not None and self._motion_thread.isRunning():
            try:
                with self._config.controller_lock:
                    self._config.controller.stop_run()
            except Exception:
                pass

        if self._fisheye_thread is not None:
            self._fisheye_thread.stop()
