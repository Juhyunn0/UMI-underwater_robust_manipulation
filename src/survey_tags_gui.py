#!/usr/bin/env python3
"""survey_tags_gui.py — interactive PyQt5 GUI for end-to-end AprilTag surveying.

Launched by ``python -m src.survey_tags`` (no args). Connect a fisheye camera +
FMC4030 gantry, drive the gantry by hand while live incremental SLAM builds a tag
map (color-coded by uncertainty), then "Stop & Finalize" runs the exact batch
optimization from survey_tags.py and writes config/tag_map.yaml.

PyQt is imported ONLY in this module so the CLI path in survey_tags.py stays
headless-safe. All heavy work runs off the GUI thread:

  GUI (main)            widget updates, slot handlers
  _FisheyeGrabThread    camera read loop (in fisheye_camera.py)
  DetectionSlamWorker   undistort + detect + SLAM + CSV (sole owner of backend)
  GantryPollThread      position/velocity readout @10 Hz   (under _controller_lock)
  GantryMotionThread    Move Abs                            (under _controller_lock)
  GantryTelemetryLogger telemetry CSV @100 Hz               (under _controller_lock)
  BatchOptimizerThread  LM optimization on a frozen snapshot

UI labels, comments, and logs are English by project convention.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_REPO_ROOT = _SRC_DIR.parent

from PyQt5.QtCore import Qt, QObject, QThread, QTimer, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QPushButton,
    QShortcut, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

import cv2  # noqa: E402

import survey_tags as st  # noqa: E402 — reuse the CLI's optimize / YAML / plot
from fisheye_camera import FisheyeCameraSession  # noqa: E402
from fisheye_gantry_tagslam import (  # noqa: E402
    build_fisheye_undistort_maps,
    load_fisheye_calibration,
    rectified_camera_intrinsics,
)
from tagslam_core import (  # noqa: E402
    TagSlamBackend,
    detect_observations,
    make_detector,
    make_run_dir,
    parse_simple_yaml,
    pose_rpy,
    pose_translation,
    tag_object_points,
)

try:
    from tagslam.visualization import normalize_pool_config
except Exception:  # pragma: no cover
    def normalize_pool_config(cfg):
        return dict(cfg or {})

from gantry import Axis, ControllerConfig, FMC4030Controller, FMC4030Error  # noqa: E402
from gantry_panel import MockFMC4030Controller  # noqa: E402
from gantry_runner import (  # noqa: E402
    EMERGENCY_STOP,
    GantryTelemetryLogger,
    SCALE_MM_PER_UNIT,
    _read_current_pos_mm,
    make_gantry_run_dir,
    mm_to_units,
    move_to_xyz_mm,
)

# ── constants ─────────────────────────────────────────────────────────────────
DETECTION_FPS_CAP = 15
MIN_OBSERVATIONS_PER_TAG = st.MIN_OBSERVATIONS_PER_TAG  # 10

JUMP_WARNING_THRESHOLD_MM = 20.0
JUMP_CRITICAL_THRESHOLD_MM = 100.0

OPTIMIZER_MAX_ITERATIONS = st.OPTIMIZER_MAX_ITERATIONS
OPTIMIZER_RELATIVE_TOL = st.OPTIMIZER_RELATIVE_TOL
OPTIMIZER_ABSOLUTE_TOL = st.OPTIMIZER_ABSOLUTE_TOL
ANCHOR_PRIOR_SIGMA = st.ANCHOR_PRIOR_SIGMA
TAG_BETWEEN_ROT_SIGMA_RAD = st.TAG_BETWEEN_ROT_SIGMA_RAD
TAG_BETWEEN_TRANS_SIGMA_M = st.TAG_BETWEEN_TRANS_SIGMA_M

TRAIL_LENGTH = 500
UI_UPDATE_HZ = 5.0
DEFAULT_WINDOW_SIZE = (1500, 950)
MIN_WINDOW_SIZE = (1300, 800)

ANCHOR_TIMEOUT_S = 30.0
_STATE_FILE = Path.home() / ".umi_gui_state.json"

# Uncertainty color thresholds (mm) and palette.
_COL_GREEN = "#34d058"
_COL_YELLOW = "#ffa726"
_COL_RED = "#ef5350"
_COL_GRAY = "#666666"
_COL_ANCHOR = "#26d0e0"


def _color_for_uncertainty(unc_mm: float, n_obs: int) -> str:
    if n_obs < MIN_OBSERVATIONS_PER_TAG:
        return _COL_GRAY
    if not math.isfinite(unc_mm):
        return _COL_GRAY
    if unc_mm < 5.0:
        return _COL_GREEN
    if unc_mm < 15.0:
        return _COL_YELLOW
    return _COL_RED


# viridis control points (RGB) for the camera-trajectory trail.
_VIRIDIS = [
    (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141), (41, 120, 142),
    (32, 144, 140), (34, 167, 132), (68, 190, 112), (121, 209, 81),
    (189, 223, 38), (253, 231, 37),
]


def _viridis(frac: float) -> QColor:
    frac = max(0.0, min(1.0, frac))
    x = frac * (len(_VIRIDIS) - 1)
    i = int(math.floor(x))
    j = min(i + 1, len(_VIRIDIS) - 1)
    t = x - i
    a, b = _VIRIDIS[i], _VIRIDIS[j]
    return QColor(int(a[0] + (b[0] - a[0]) * t),
                  int(a[1] + (b[1] - a[1]) * t),
                  int(a[2] + (b[2] - a[2]) * t))


def _ndarray_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """BGR uint8 ndarray -> QPixmap (RGB)."""
    if frame is None or frame.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def load_gui_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def save_gui_state(state: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as exc:  # pragma: no cover
        print(f"[survey-gui] could not save state: {exc}", file=sys.stderr)


def build_slam_args(anchor_tag_id: int, tag_size: float, tag_family: str):
    """One Namespace feeding TagSlamBackend, make_detector, and
    detect_observations — mirrors gantry_panel._exp_build_fisheye_args defaults.
    The survey builds a map, so tag_map stays None (normal bootstrap)."""
    import argparse
    a = argparse.Namespace()
    a.tag_family = tag_family or "tag36h11"
    a.tag_size = float(tag_size)
    a.anchor_tag_id = int(anchor_tag_id)
    a.max_tag_id = -1
    a.water_correction_mode = "none"
    a.water_scale = 3.6
    a.min_tag_area_px = 120.0
    a.max_off_nadir_deg = 25.0
    a.max_image_eccentricity = 0.65
    a.max_tag_tilt_deg = 35.0
    a.max_reprojection_error_px = 5.0
    a.nthreads = 2
    a.quad_decimate = 1.0
    a.quad_sigma = 0.0
    a.decode_sharpening = 0.25
    a.min_decision_margin = 30.0
    a.max_hamming = 0
    a.tag_rot_sigma = TAG_BETWEEN_ROT_SIGMA_RAD
    a.tag_trans_sigma = TAG_BETWEEN_TRANS_SIGMA_M
    a.tag_robust_kernel = "huber"
    a.tag_robust_threshold = 1.345
    a.tag_init_min_observations = 3
    a.pose_std_window = 30
    a.odom_rot_sigma = 0.35
    a.odom_trans_sigma = 0.30
    a.prior_rot_sigma = ANCHOR_PRIOR_SIGMA
    a.prior_trans_sigma = ANCHOR_PRIOR_SIGMA
    a.floor_prior_enabled = True
    a.floor_z_sigma = 0.02
    a.floor_plane_min_tags = 4
    a.floor_normal_sigma_deg = 8.0
    a.strict_coplanar = False
    a.floor_prior_refresh_frames = 0
    a.floor_plane_outlier_threshold = 0.10
    a.use_imu_gravity = False
    a.gravity_align_world = False
    a.init_min_observations = 3
    a.init_min_decision_margin = 45.0
    a.init_min_tag_area_px = 250.0
    a.init_max_off_nadir_deg = 20.0
    a.init_max_image_eccentricity = 0.45
    a.init_max_tag_tilt_deg = 25.0
    a.tag_map = None
    return a


# ══════════════════════════════════════════════════════════════════════════════
# DetectionSlamWorker — owns the camera frame queue + TagSlamBackend
# ══════════════════════════════════════════════════════════════════════════════
class DetectionSlamWorker(QThread):
    """Pulls (frame, t) from the camera's drop-oldest Queue(maxsize=2),
    undistorts, detects AprilTags, steps the live iSAM2 backend, writes the live
    CSV, detects backend jumps, and emits coalesced (5 Hz) snapshots. Sole owner
    of the TagSlamBackend — no other thread touches it."""

    frame_overlay = pyqtSignal(object, int, int)        # (bgr frame, fps, tags_in_graph)
    tag_map_update = pyqtSignal(object)                 # {tag_id: dict}
    metrics_update = pyqtSignal(object)                 # dict
    jump_detected = pyqtSignal(int, float, int)         # (tag_id, residual_mm, frame_idx)
    anchor_selected = pyqtSignal(int, int)              # (tag_id, frame_idx)
    error = pyqtSignal(str)
    camera_position = pyqtSignal(object)                # (x,y,z) m or None

    def __init__(self, frame_queue: Queue, calib, anchor_mode,
                 tag_size: float, tag_family: str, run_dir: Path,
                 telemetry_logger=None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._q = frame_queue
        self._calib = calib
        self._anchor_mode = anchor_mode  # None == auto, else int tag id
        self._tag_size = tag_size
        self._tag_family = tag_family
        self._run_dir = run_dir
        self._telemetry = telemetry_logger

        self._stop = threading.Event()
        self._paused = threading.Event()

        self.backend: TagSlamBackend | None = None
        self._args = None
        self._detector = None
        self._object_points = None
        self._intrinsics = None
        self._map1 = self._map2 = None

        # finalize buffers (consumed only after the worker has stopped)
        self.observations: list[tuple] = []          # (frame_idx, tag_id, camera_T_tag)
        self.frame_init: dict[int, object] = {}       # frame_idx -> world_T_camera Pose3
        self.anchor_tag_id: int | None = None

        # jump-detection state
        self._p_prev = None
        self._p_prev2 = None
        self._t_prev = None
        self._t_prev2 = None
        self._last_jump_mm = 0.0

        # metrics / coalescing
        self._frame_idx = -1
        self._frames_processed = 0
        self._frames_with_2tags = 0
        self._dropped = 0
        self._last_proc_t = 0.0
        self._last_emit_t = 0.0
        self._last_residual_px = float("nan")
        self._t0 = time.monotonic()
        self._cam_csv_fh = None
        self._cam_csv_writer = None
        self._last_fps = 0

    # — control —
    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def set_fps(self, fps: int) -> None:
        self._last_fps = int(fps)

    def stop(self) -> None:
        self._stop.set()

    # — lifecycle —
    def _ensure_pipeline(self, frame: np.ndarray) -> None:
        if self._map1 is not None:
            return
        h, w = frame.shape[:2]
        self._map1, self._map2, new_K = build_fisheye_undistort_maps(
            self._calib.K, self._calib.D, (w, h), 0.0)
        self._intrinsics = rectified_camera_intrinsics(new_K)
        anchor_seed = self._anchor_mode if self._anchor_mode is not None else 1
        self._args = build_slam_args(anchor_seed, self._tag_size, self._tag_family)
        self._detector = make_detector(self._args)
        self._object_points = tag_object_points(self._args.tag_size)
        # camera_trajectory.csv (live append)
        self._cam_csv_fh = open(self._run_dir / "camera_trajectory.csv", "w",
                                newline="", encoding="utf-8")
        import csv as _csv
        self._cam_csv_writer = _csv.writer(self._cam_csv_fh)
        self._cam_csv_writer.writerow([
            "timestamp_unix", "timestamp_monotonic", "camera_index", "time_s",
            "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg",
            "detected_tags", "has_tag_update", "image_path",
            "gantry_x_mm", "gantry_y_mm", "gantry_z_mm", "translation_error_mm",
        ])

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def _select_anchor(self, obs_list, w: int, h: int, frame_idx: int) -> int | None:
        """Auto: tag nearest image center. Specific: that id if present."""
        if not obs_list:
            return None
        if self._anchor_mode is not None:
            ids = {int(o.tag_id) for o in obs_list}
            return self._anchor_mode if self._anchor_mode in ids else None
        cx, cy = w / 2.0, h / 2.0
        best, best_d = None, float("inf")
        for o in obs_list:
            c = np.asarray(o.center, dtype=np.float64).reshape(2)
            d = math.hypot(c[0] - cx, c[1] - cy)
            if d < best_d:
                best, best_d = int(o.tag_id), d
        if best is not None:
            print(f"[anchor-auto] selected tag {best} at frame {frame_idx} "
                  f"(centroid distance {best_d:.0f} px from image center)",
                  file=sys.stderr)
        return best

    def run(self) -> None:  # type: ignore[override]
        while not self._stop.is_set() and not EMERGENCY_STOP.is_set():
            try:
                frame, t_mono = self._q.get(timeout=0.2)
            except Empty:
                continue
            if self._paused.is_set():
                continue
            # 15 fps processing cap
            if (t_mono - self._last_proc_t) < (1.0 / DETECTION_FPS_CAP):
                self._dropped += 1
                continue
            self._last_proc_t = t_mono
            try:
                self._process(frame, t_mono)
            except Exception as exc:  # never let the worker die silently
                print(f"[survey-gui] detection/SLAM error: {exc}", file=sys.stderr)
        self._close_csv()

    def _process(self, frame: np.ndarray, t_mono: float) -> None:
        self._ensure_pipeline(frame)
        und = self._undistort(frame)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        obs_list = detect_observations(gray, self._detector, self._intrinsics,
                                       self._object_points, self._args, None)
        self._frames_processed += 1
        if len(obs_list) >= 2:
            self._frames_with_2tags += 1

        # Backend creation deferred until the anchor is known.
        if self.backend is None:
            anchor = self._select_anchor(obs_list, w, h, self._frames_processed)
            if anchor is None:
                if (time.monotonic() - self._t0) > ANCHOR_TIMEOUT_S:
                    mode = "auto" if self._anchor_mode is None else f"id {self._anchor_mode}"
                    self.error.emit(f"No usable anchor tag ({mode}) within "
                                    f"{ANCHOR_TIMEOUT_S:.0f}s — aborting survey.")
                    self._stop.set()
                self._emit_overlay(und, obs_list, t_mono)
                return
            self.anchor_tag_id = anchor
            self._args.anchor_tag_id = anchor
            self.backend = TagSlamBackend(self._args)
            self.anchor_selected.emit(anchor, self._frames_processed)

        update = self.backend.update(obs_list)
        self._frame_idx += 1
        # median reprojection residual this frame
        res = [float(getattr(o, "reprojection_error_px", float("nan"))) for o in obs_list]
        res = [r for r in res if math.isfinite(r)]
        self._last_residual_px = float(np.median(res)) if res else float("nan")

        if update.optimized and update.camera_pose is not None:
            p = np.asarray(update.camera_pose.translation(), dtype=np.float64).reshape(3)
            self.camera_position.emit((float(p[0]), float(p[1]), float(p[2])))
            self.frame_init[self._frame_idx] = update.camera_pose
            for o in obs_list:
                self.observations.append((self._frame_idx, int(o.tag_id), o.camera_T_tag))
            self._detect_jump(p, t_mono, obs_list)
            self._write_csv_row(update, obs_list, p, t_mono)

        self._maybe_emit(und, obs_list, t_mono, update)

    def _detect_jump(self, p_curr: np.ndarray, t_curr: float, obs_list) -> None:
        if self._p_prev is not None and self._t_prev is not None:
            dt = max(1e-3, t_curr - self._t_prev)
            actual = float(np.linalg.norm(p_curr - self._p_prev)) * 1000.0  # mm
            if self._p_prev2 is not None and self._t_prev2 is not None:
                dt_prev = max(1e-3, self._t_prev - self._t_prev2)
                v_prev = float(np.linalg.norm(self._p_prev - self._p_prev2)) * 1000.0 / dt_prev
            else:
                v_prev = 0.0
            expected = v_prev * dt
            residual = actual - expected
            self._last_jump_mm = residual
            if residual > JUMP_WARNING_THRESHOLD_MM:
                # Attribute to the newest in-graph tag this frame, if any.
                newest = obs_list[-1].tag_id if obs_list else -1
                self.jump_detected.emit(int(newest), float(residual), self._frame_idx)
        self._p_prev2, self._t_prev2 = self._p_prev, self._t_prev
        self._p_prev, self._t_prev = p_curr, t_curr

    def _gantry_sample(self):
        if self._telemetry is None:
            return None
        try:
            return self._telemetry.latest_sample()
        except Exception:
            return None

    def _write_csv_row(self, update, obs_list, p, t_mono) -> None:
        if self._cam_csv_writer is None:
            return
        rpy = np.degrees(pose_rpy(update.camera_pose))
        tags = " ".join(str(int(o.tag_id)) for o in obs_list)
        gs = self._gantry_sample()
        gx = gy = gz = ""
        if gs is not None:
            gx, gy, gz = (f"{gs.pos_mm[0]:.4f}", f"{gs.pos_mm[1]:.4f}", f"{gs.pos_mm[2]:.4f}")
        self._cam_csv_writer.writerow([
            f"{time.time():.6f}", f"{t_mono:.6f}", self._frame_idx,
            f"{t_mono - self._t0:.4f}",
            f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}",
            f"{rpy[0]:.4f}", f"{rpy[1]:.4f}", f"{rpy[2]:.4f}",
            tags, "True" if obs_list else "False", "",
            gx, gy, gz, "",
        ])

    def _close_csv(self) -> None:
        if self._cam_csv_fh is not None:
            try:
                self._cam_csv_fh.flush()
                self._cam_csv_fh.close()
            except Exception:
                pass
            self._cam_csv_fh = None

    def write_tag_poses_csv(self) -> None:
        """Write final live tag poses to tag_poses.csv (called at finalize)."""
        if self.backend is None:
            return
        import csv as _csv
        with open(self._run_dir / "tag_poses.csv", "w", newline="", encoding="utf-8") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["tag_id", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"])
            for tid, pose in sorted(self.backend.optimized_tag_poses().items()):
                t = pose_translation(pose)
                r = np.degrees(pose_rpy(pose))
                wr.writerow([tid, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                             f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}"])

    def live_tag_states(self) -> dict:
        return self._tag_states()

    def _tag_states(self) -> dict:
        if self.backend is None:
            return {}
        counts = self.backend.tag_observation_counts
        out = {}
        for tid, pose in self.backend.optimized_tag_poses().items():
            t = pose_translation(pose)
            q = st._quat_wxyz(pose.rotation())
            out[int(tid)] = {
                "position_m": [float(t[0]), float(t[1]), float(t[2])],
                "quaternion_wxyz": q,
                "n_observations": int(counts.get(tid, 0)),
                "uncertainty_mm": float("nan"),  # live: no marginals (cost); batch fills it
            }
        return out

    def _emit_overlay(self, und, obs_list, t_mono) -> None:
        tags_in_graph = len(self.backend.optimized_tag_poses()) if self.backend else 0
        self.frame_overlay.emit(self._draw_overlay(und, obs_list), self._last_fps, tags_in_graph)

    def _maybe_emit(self, und, obs_list, t_mono, update) -> None:
        if (time.monotonic() - self._last_emit_t) < (1.0 / UI_UPDATE_HZ):
            return
        self._last_emit_t = time.monotonic()
        self._emit_overlay(und, obs_list, t_mono)
        self.tag_map_update.emit(self._tag_states())
        n_qual = sum(1 for s in self._tag_states().values()
                     if s["n_observations"] >= MIN_OBSERVATIONS_PER_TAG)
        self.metrics_update.emit({
            "frames_processed": self._frames_processed,
            "frames_with_2tags": self._frames_with_2tags,
            "dropped": self._dropped,
            "tags_in_graph": len(self._tag_states()),
            "tags_qualified": n_qual,
            "last_jump_mm": self._last_jump_mm,
            "median_residual_px": self._last_residual_px,
            "anchor_tag_id": self.anchor_tag_id,
            "elapsed_s": time.monotonic() - self._t0,
        })

    def _draw_overlay(self, und: np.ndarray, obs_list) -> np.ndarray:
        img = und.copy()
        for o in obs_list:
            corners = np.asarray(o.corners, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(img, [corners], True, (0, 230, 230), 2)
            c = np.asarray(o.center, dtype=np.int32).reshape(2)
            cv2.putText(img, str(int(o.tag_id)), (int(c[0]) + 4, int(c[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
        return img


# ══════════════════════════════════════════════════════════════════════════════
# CameraPreview — left pane
# ══════════════════════════════════════════════════════════════════════════════
class CameraPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self._pix = QPixmap()
        self._fps = 0
        self._tags = 0
        self.setStyleSheet("background:#0a0d11;")

    def on_frame(self, frame: np.ndarray, fps: int, tags_in_graph: int) -> None:
        self._pix = _ndarray_to_qpixmap(frame)
        self._fps = fps
        self._tags = tags_in_graph
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor("#0a0d11"))
            if not self._pix.isNull():
                scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                p.drawPixmap(x, y, scaled)
            else:
                p.setPen(QColor("#9aa3ad"))
                p.drawText(self.rect(), Qt.AlignCenter, "No camera frame")
            p.setPen(QColor("#e6e6e6"))
            p.setFont(QFont("Arial", 10))
            p.drawText(8, 18, f"FPS: {self._fps}")
            p.drawText(8, 36, f"Tags in graph: {self._tags}")
        finally:
            p.end()


# ══════════════════════════════════════════════════════════════════════════════
# TagMapWidget — center pane (custom QPainter top-down 2D map)
# ══════════════════════════════════════════════════════════════════════════════
class TagMapWidget(QWidget):
    def __init__(self, pool_cfg: dict, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#101317;")
        self._pool = normalize_pool_config(pool_cfg or {})
        self._tags: dict = {}
        self._ghost: dict = {}     # live positions shown faded after finalize
        self._anchor_id: int | None = None
        self._trail: list = []     # list of (x_m, y_m)
        self._cam_pos = None
        # view transform
        self._scale = 200.0        # px per meter
        self._cx = 0.0             # world center x (m)
        self._cy = 0.0
        self._panning = False
        self._last_mouse = None
        self._fitted = False

    # — data —
    def set_tags(self, tags: dict, anchor_id, cam_pos):
        self._tags = tags or {}
        self._anchor_id = anchor_id
        self._cam_pos = cam_pos
        if cam_pos is not None:
            self._trail.append((cam_pos[0], cam_pos[1]))
            if len(self._trail) > TRAIL_LENGTH:
                self._trail = self._trail[-TRAIL_LENGTH:]
        if not self._fitted and self._tags:
            self.fit_to_data()
        self.update()

    def set_refined(self, refined: dict, ghost_live: dict):
        self._tags = refined or {}
        self._ghost = ghost_live or {}
        self.fit_to_data()
        self.update()

    # — transforms —
    def _w2s(self, x: float, y: float) -> QPointF:
        sx = self.width() / 2.0 + (x - self._cx) * self._scale
        sy = self.height() / 2.0 - (y - self._cy) * self._scale  # flip Y up
        return QPointF(sx, sy)

    def fit_to_data(self):
        xs, ys = [], []
        for s in self._tags.values():
            xs.append(s["position_m"][0]); ys.append(s["position_m"][1])
        L = float(self._pool.get("length_m", 4.877))
        W = float(self._pool.get("width_m", 1.8))
        if not xs:
            xs, ys = [-L / 2, L / 2], [-W / 2, W / 2]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        self._cx = (minx + maxx) / 2.0
        self._cy = (miny + maxy) / 2.0
        span_x = max(0.5, (maxx - minx) * 1.3, L * 0.6)
        span_y = max(0.5, (maxy - miny) * 1.3, W * 0.6)
        self._scale = min(self.width() / span_x, self.height() / span_y) if span_x and span_y else 200.0
        self._fitted = True
        self.update()

    # — paint —
    def paintEvent(self, _ev):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.fillRect(self.rect(), QColor("#101317"))
            self._draw_pool(p)
            self._draw_trail(p)
            self._draw_ghost(p)
            self._draw_tags(p)
            self._draw_camera(p)
            self._draw_legend(p)
        finally:
            p.end()

    def _draw_pool(self, p: QPainter):
        L = float(self._pool.get("length_m", 4.877))
        W = float(self._pool.get("width_m", 1.8))
        if str(self._pool.get("pool_long_axis", "x")) == "x":
            hx, hy = L / 2.0, W / 2.0
        else:
            hx, hy = W / 2.0, L / 2.0
        a = self._w2s(self._cx - hx, self._cy - hy)
        b = self._w2s(self._cx + hx, self._cy + hy)
        pen = QPen(QColor("#3a4450")); pen.setStyle(Qt.DashLine); pen.setWidthF(1.2)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(min(a.x(), b.x()), min(a.y(), b.y()),
                          abs(b.x() - a.x()), abs(b.y() - a.y())))

    def _draw_trail(self, p: QPainter):
        if len(self._trail) < 2:
            return
        n = len(self._trail)
        for i in range(1, n):
            c = _viridis(i / max(1, n - 1))
            pen = QPen(c); pen.setWidthF(2.0)
            p.setPen(pen)
            a = self._w2s(*self._trail[i - 1]); b = self._w2s(*self._trail[i])
            p.drawLine(a, b)

    def _draw_ghost(self, p: QPainter):
        for tid, s in self._ghost.items():
            pt = self._w2s(s["position_m"][0], s["position_m"][1])
            col = QColor("#888888"); col.setAlpha(110)
            p.setPen(QPen(col, 1)); p.setBrush(col)
            p.drawEllipse(pt, 7, 7)

    def _draw_tags(self, p: QPainter):
        f = QFont("Arial", 9, QFont.Bold)
        p.setFont(f)
        for tid, s in self._tags.items():
            pt = self._w2s(s["position_m"][0], s["position_m"][1])
            is_anchor = (tid == self._anchor_id)
            col = QColor(_color_for_uncertainty(s.get("uncertainty_mm", float("nan")),
                                                s.get("n_observations", 0)))
            r = 11 if is_anchor else 8
            dropped = s.get("n_observations", 0) < MIN_OBSERVATIONS_PER_TAG
            pen = QPen(QColor("#0a0d11"), 1.5)
            if dropped:
                pen = QPen(QColor("#aaaaaa"), 1.2); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.setBrush(col)
            p.drawEllipse(pt, r, r)
            if is_anchor:
                self._draw_star(p, pt, r + 7, QColor(_COL_ANCHOR))
            p.setPen(QColor("#e6e6e6"))
            p.drawText(QPointF(pt.x() + r + 2, pt.y() + 4), str(tid))

    def _draw_star(self, p: QPainter, c: QPointF, R: float, color: QColor):
        poly = QPolygonF()
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            rr = R if k % 2 == 0 else R * 0.45
            poly.append(QPointF(c.x() + rr * math.cos(ang), c.y() + rr * math.sin(ang)))
        pen = QPen(color, 1.5); p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

    def _draw_camera(self, p: QPainter):
        if self._cam_pos is None:
            return
        pt = self._w2s(self._cam_pos[0], self._cam_pos[1])
        p.setPen(QPen(QColor("#0a0d11"), 1.5)); p.setBrush(QColor("#2ecc71"))
        p.drawEllipse(pt, 7, 7)

    def _draw_legend(self, p: QPainter):
        p.setFont(QFont("Arial", 8))
        items = [("< 5 mm", _COL_GREEN), ("5-15 mm", _COL_YELLOW),
                 (">= 15 mm", _COL_RED), ("dropped", _COL_GRAY)]
        x, y = 8, self.height() - 8 - 16 * len(items)
        for label, col in items:
            p.setBrush(QColor(col)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x + 6, y + 6), 5, 5)
            p.setPen(QColor("#cfd6dd"))
            p.drawText(x + 16, y + 10, label)
            y += 16

    # — interaction —
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._panning = True
            self._last_mouse = ev.pos()

    def mouseReleaseEvent(self, ev):
        self._panning = False

    def mouseMoveEvent(self, ev):
        if self._panning and self._last_mouse is not None:
            d = ev.pos() - self._last_mouse
            self._last_mouse = ev.pos()
            self._cx -= d.x() / self._scale
            self._cy += d.y() / self._scale
            self.update()
            return
        # hover tooltip
        from PyQt5.QtWidgets import QToolTip
        hit = self._tag_at(ev.pos())
        if hit is not None:
            tid, s = hit
            unc = s.get("uncertainty_mm", float("nan"))
            unc_s = "n/a" if not math.isfinite(unc) else f"{unc:.1f} mm"
            QToolTip.showText(ev.globalPos(),
                              f"tag {tid} · obs={s.get('n_observations', 0)} · unc={unc_s}",
                              self)
        else:
            QToolTip.hideText()

    def _tag_at(self, pos):
        for tid, s in self._tags.items():
            pt = self._w2s(s["position_m"][0], s["position_m"][1])
            if math.hypot(pt.x() - pos.x(), pt.y() - pos.y()) <= 12:
                return (tid, s)
        return None

    def wheelEvent(self, ev):
        factor = 1.15 if ev.angleDelta().y() > 0 else 1.0 / 1.15
        self._scale *= factor
        self.update()

    def mouseDoubleClickEvent(self, ev):
        self.fit_to_data()


# ══════════════════════════════════════════════════════════════════════════════
# Gantry threads
# ══════════════════════════════════════════════════════════════════════════════
class GantryPollThread(QThread):
    """10 Hz position readout (mm) + finite-difference velocity (cm/s). Uses the
    664-tolerant _read_current_pos_mm under the shared controller lock."""

    status = pyqtSignal(object, object)  # (pos_mm tuple, vel_cms tuple)
    error = pyqtSignal(str)

    def __init__(self, controller, lock: threading.RLock, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._lock = lock
        self._stop = threading.Event()
        self._prev = None
        self._prev_t = None

    def stop(self):
        self._stop.set()

    def run(self):  # type: ignore[override]
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                pos = _read_current_pos_mm(self._controller, self._lock)
            except Exception as exc:
                self.error.emit(str(exc))
                time.sleep(0.2)
                continue
            vel = (0.0, 0.0, 0.0)
            if self._prev is not None and self._prev_t is not None:
                dt = max(1e-3, t - self._prev_t)
                vel = tuple((pos[i] - self._prev[i]) / dt / 10.0 for i in range(3))  # cm/s
            self._prev, self._prev_t = pos, t
            self.status.emit(tuple(pos), vel)
            time.sleep(0.1)


class GantryMotionThread(QThread):
    """Runs move_to_xyz_mm off the GUI thread."""

    done = pyqtSignal(str)  # "" on success else error

    def __init__(self, controller, target_mm, speed_mm_s, acc_mm_s2, dec_mm_s2,
                 lock, logger=None, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._target = tuple(target_mm)
        self._speed = speed_mm_s
        self._acc = acc_mm_s2
        self._dec = dec_mm_s2
        self._lock = lock
        self._logger = logger

    def run(self):  # type: ignore[override]
        try:
            move_to_xyz_mm(self._controller, self._target, self._speed, self._acc,
                           self._dec, mode="line", lock=self._lock, logger=self._logger)
            self.done.emit("")
        except Exception as exc:
            self.done.emit(str(exc))


class BatchOptimizerThread(QThread):
    """Runs the exact CLI batch optimization (survey_tags.optimize) on a frozen
    snapshot of the live survey's observations."""

    progress = pyqtSignal(str)
    done = pyqtSignal(object)   # result dict
    failed = pyqtSignal(str)

    def __init__(self, observations, frame_init, tag_init, anchor_id, parent=None):
        super().__init__(parent)
        self._obs = observations
        self._frame_init = frame_init
        self._tag_init = tag_init
        self._anchor = anchor_id

    def run(self):  # type: ignore[override]
        try:
            self.progress.emit("Optimizing… (Levenberg-Marquardt, up to "
                               f"{OPTIMIZER_MAX_ITERATIONS} iters)")
            anchor_init = self._tag_init.get(self._anchor, None)
            if anchor_init is None:
                self.failed.emit(f"anchor tag {self._anchor} missing from live map")
                return
            frame_init, tag_init = st.reframe_to_anchor(
                anchor_init, dict(self._frame_init), dict(self._tag_init))
            result = st.optimize(self._obs, frame_init, tag_init, self._anchor,
                                 MIN_OBSERVATIONS_PER_TAG, OPTIMIZER_MAX_ITERATIONS,
                                 floor_coplanar=False)
            # st.optimize's dict has no anchor_id; the GUI needs it for YAML/plot.
            result["anchor_id"] = self._anchor
            self.done.emit(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# SurveyWindow — the main application window
# ══════════════════════════════════════════════════════════════════════════════
_STATES = ("IDLE", "SURVEYING", "PAUSED", "FINALIZING", "DONE", "ERROR")


def _hline():
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setStyleSheet("color:#33333a;")
    return f


class SurveyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tag Survey — live SLAM + batch finalize")
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*MIN_WINDOW_SIZE)

        self._state = "IDLE"
        self._gui_state = load_gui_state()

        # sessions / controllers
        self._camera = FisheyeCameraSession(self)
        self._controller = None
        self._controller_lock = threading.RLock()
        self._calib = None
        self._pool_cfg = self._load_pool_cfg()

        # threads
        self._frame_queue: Queue | None = None
        self._worker: DetectionSlamWorker | None = None
        self._poll: GantryPollThread | None = None
        self._motion: GantryMotionThread | None = None
        self._logger: GantryTelemetryLogger | None = None
        self._batch: BatchOptimizerThread | None = None
        self._run_dir: Path | None = None

        self._batch_result = None
        self._live_tag_states = {}
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(lambda: self._set_banner("", None))

        self._build_ui()
        self._wire_camera()
        self._apply_saved_settings()
        self._set_state("IDLE")

    # ── config helpers ────────────────────────────────────────────────────────
    def _load_pool_cfg(self) -> dict:
        try:
            cfg = parse_simple_yaml((_REPO_ROOT / "config/config.yaml").read_text())
            return normalize_pool_config(cfg.get("pool", {}))
        except Exception:
            return normalize_pool_config({})

    # ── UI construction ─────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        root.addWidget(self._build_connection_bar())

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(8)
        self._splitter.setStyleSheet(
            "QSplitter::handle{background:#2a2f37;} QSplitter::handle:hover{background:#4ea1ff;}")
        self._camera_preview = CameraPreview()
        self._tag_map = TagMapWidget(self._pool_cfg)
        self._splitter.addWidget(self._wrap("Live Camera", self._camera_preview))
        self._splitter.addWidget(self._wrap("Live Tag Map (top-down)", self._tag_map))
        self._splitter.addWidget(self._build_gantry_pane())
        self._splitter.setSizes(self._gui_state.get("survey_tags.splitter_sizes",
                                                     [450, 600, 450]))
        root.addWidget(self._splitter, 1)

        root.addWidget(self._build_status_panel())
        root.addWidget(self._build_global_controls())

        QShortcut(QKeySequence(Qt.Key_Escape), self,
                  activated=self._emergency_stop, context=Qt.ApplicationShortcut)

    def _wrap(self, title: str, w: QWidget) -> QWidget:
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(2, 2, 2, 2)
        lab = QLabel(title); lab.setStyleSheet("font-weight:700; color:#cfd6dd;")
        lay.addWidget(lab); lay.addWidget(w, 1)
        return box

    def _build_connection_bar(self) -> QWidget:
        f = QFrame(); f.setFrameShape(QFrame.StyledPanel)
        g = QGridLayout(f); g.setContentsMargins(8, 6, 8, 6); g.setHorizontalSpacing(8)
        # camera row
        self._cam_device = QLineEdit("0"); self._cam_device.setMaximumWidth(50)
        self._cam_res = QComboBox(); [self._cam_res.addItem(r) for r in ("1280x720", "1920x1080", "640x480")]
        self._cam_fps = QSpinBox(); self._cam_fps.setRange(1, 120); self._cam_fps.setValue(30)
        self._btn_cam = QPushButton("Connect Camera"); self._btn_cam.clicked.connect(self._toggle_camera)
        self._cam_mock = QCheckBox("mock")
        self._cam_ind = QLabel("● Idle"); self._cam_ind.setStyleSheet("color:#9aa3ad;")
        g.addWidget(QLabel("Camera Device"), 0, 0); g.addWidget(self._cam_device, 0, 1)
        g.addWidget(QLabel("Res"), 0, 2); g.addWidget(self._cam_res, 0, 3)
        g.addWidget(QLabel("FPS"), 0, 4); g.addWidget(self._cam_fps, 0, 5)
        g.addWidget(self._cam_mock, 0, 6); g.addWidget(self._btn_cam, 0, 7); g.addWidget(self._cam_ind, 0, 8)
        # gantry row
        self._g_ip = QLineEdit("192.168.0.30"); self._g_ip.setMaximumWidth(120)
        self._g_port = QLineEdit("8088"); self._g_port.setMaximumWidth(60)
        self._g_id = QLineEdit("1"); self._g_id.setMaximumWidth(40)
        self._g_mock = QCheckBox("mock")
        self._btn_gantry = QPushButton("Connect Gantry"); self._btn_gantry.clicked.connect(self._toggle_gantry)
        self._g_ind = QLabel("● Idle"); self._g_ind.setStyleSheet("color:#9aa3ad;")
        g.addWidget(QLabel("Gantry IP"), 1, 0); g.addWidget(self._g_ip, 1, 1)
        g.addWidget(QLabel("Port"), 1, 2); g.addWidget(self._g_port, 1, 3)
        g.addWidget(QLabel("ID"), 1, 4); g.addWidget(self._g_id, 1, 5)
        g.addWidget(self._g_mock, 1, 6); g.addWidget(self._btn_gantry, 1, 7); g.addWidget(self._g_ind, 1, 8)
        # calib + anchor row
        self._calib_edit = QLineEdit("config/fisheye_calibration.yaml")
        self._anchor_edit = QLineEdit("auto"); self._anchor_edit.setMaximumWidth(80)
        self._anchor_edit.setToolTip("'auto' = tag nearest image center at start, or a tag id.")
        self._tagsize_spin = QDoubleSpinBox(); self._tagsize_spin.setRange(0.01, 1.0)
        self._tagsize_spin.setDecimals(3); self._tagsize_spin.setValue(0.170); self._tagsize_spin.setSuffix(" m")
        self._tagfam_edit = QLineEdit("tag36h11"); self._tagfam_edit.setMaximumWidth(100)
        g.addWidget(QLabel("Calib"), 2, 0); g.addWidget(self._calib_edit, 2, 1, 1, 3)
        g.addWidget(QLabel("Anchor tag id"), 2, 4); g.addWidget(self._anchor_edit, 2, 5)
        g.addWidget(QLabel("Tag size"), 2, 6); g.addWidget(self._tagsize_spin, 2, 7)
        g.addWidget(self._tagfam_edit, 2, 8)
        return f

    def _build_gantry_pane(self) -> QWidget:
        # NO soft-limit validation here by design (user preference): every jog and
        # Move Abs command is sent to the controller as-is. Be careful in hardware.
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(6)
        lay.addWidget(QLabel("Gantry Control"))
        sp = QGridLayout()
        self._spd = QDoubleSpinBox(); self._spd.setRange(0.1, 100); self._spd.setValue(10.0); self._spd.setSuffix(" cm/s")
        self._acc = QDoubleSpinBox(); self._acc.setRange(0.1, 100); self._acc.setValue(5.0); self._acc.setSuffix(" cm/s²")
        self._dec = QDoubleSpinBox(); self._dec.setRange(0.1, 100); self._dec.setValue(5.0); self._dec.setSuffix(" cm/s²")
        sp.addWidget(QLabel("Speed:"), 0, 0); sp.addWidget(self._spd, 0, 1)
        sp.addWidget(QLabel("Accel:"), 1, 0); sp.addWidget(self._acc, 1, 1)
        sp.addWidget(QLabel("Decel:"), 2, 0); sp.addWidget(self._dec, 2, 1)
        lay.addLayout(sp)

        jog = QGridLayout()
        self._jog_buttons = []
        specs = [("X+", Axis.X, +1, 0, 0), ("X-", Axis.X, -1, 0, 1),
                 ("Y+", Axis.Y, +1, 1, 0), ("Y-", Axis.Y, -1, 1, 1),
                 ("Z+", Axis.Z, +1, 2, 0), ("Z-", Axis.Z, -1, 2, 1)]
        for label, axis, sign, r, c in specs:
            b = QPushButton(label)
            b.pressed.connect(lambda ax=axis, sg=sign: self._jog_start(ax, sg))
            b.released.connect(lambda ax=axis: self._jog_stop(ax))
            jog.addWidget(b, r, c)
            self._jog_buttons.append(b)
        lay.addWidget(QLabel("Per-Axis Jog (hold):")); lay.addLayout(jog)

        mv = QGridLayout()
        self._mv_x = QDoubleSpinBox(); self._mv_y = QDoubleSpinBox(); self._mv_z = QDoubleSpinBox()
        for s in (self._mv_x, self._mv_y, self._mv_z):
            s.setRange(-100000, 100000); s.setDecimals(3); s.setSuffix(" mm")
        self._btn_move = QPushButton("Move"); self._btn_move.clicked.connect(self._move_abs)
        mv.addWidget(QLabel("X"), 0, 0); mv.addWidget(self._mv_x, 0, 1)
        mv.addWidget(QLabel("Y"), 1, 0); mv.addWidget(self._mv_y, 1, 1)
        mv.addWidget(QLabel("Z"), 2, 0); mv.addWidget(self._mv_z, 2, 1)
        mv.addWidget(self._btn_move, 3, 0, 1, 2)
        lay.addWidget(QLabel("Move Abs (mm):")); lay.addLayout(mv)

        lay.addWidget(_hline())
        self._pos_label = QLabel("Current position:\n X: -- mm\n Y: -- mm\n Z: -- mm")
        self._pos_label.setStyleSheet("font-family: monospace;")
        self._vel_label = QLabel("Velocity:\n X: 0.00 cm/s\n Y: 0.00 cm/s\n Z: 0.00 cm/s")
        self._vel_label.setStyleSheet("font-family: monospace;")
        lay.addWidget(self._pos_label); lay.addWidget(self._vel_label)
        lay.addStretch(1)
        return box

    def _build_status_panel(self) -> QWidget:
        f = QFrame(); f.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(f); lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(3)
        self._state_label = QLabel("State: IDLE   Elapsed: 00:00")
        self._state_label.setStyleSheet("font-weight:700; color:#4ea1ff;")
        self._counts_label = QLabel("Frames processed: 0   With ≥2 tags: 0   "
                                    "Tags in graph: 0 / 0 qualified")
        self._quality_label = QLabel("Last backend jump: 0.0 mm    Median residual: -- px")
        self._worst_label = QLabel("Worst tag: --")
        self._banner = QLabel("")
        self._banner.setVisible(False)
        self._banner.setWordWrap(True)
        for w in (self._state_label, self._counts_label, self._quality_label,
                  self._worst_label, self._banner):
            lay.addWidget(w)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Tag ID", "Observations", "Uncertainty (mm)", "Color"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSortingEnabled(True)
        self._table.setMaximumHeight(180)
        lay.addWidget(self._table)
        self._delta_label = QLabel("")  # populated at DONE
        self._delta_label.setVisible(False)
        lay.addWidget(self._delta_label)
        return f

    def _build_global_controls(self) -> QWidget:
        f = QFrame()
        lay = QHBoxLayout(f); lay.setContentsMargins(8, 4, 8, 4)
        self._btn_start = QPushButton("● Start Survey"); self._btn_start.clicked.connect(self._start_survey)
        self._btn_pause = QPushButton("Pause"); self._btn_pause.clicked.connect(self._toggle_pause)
        self._btn_stop = QPushButton("■ Stop && Finalize"); self._btn_stop.clicked.connect(self._stop_finalize)
        self._btn_save = QPushButton("Save tag_map.yaml…"); self._btn_save.clicked.connect(self._save_map)
        self._btn_estop = QPushButton("EMERGENCY STOP (Esc)")
        self._btn_estop.setStyleSheet("background:#a02020; color:white; font-weight:700;")
        self._btn_estop.clicked.connect(self._emergency_stop)
        self._btn_reset_estop = QPushButton("Reset E-Stop"); self._btn_reset_estop.clicked.connect(self._reset_estop)
        for b in (self._btn_start, self._btn_pause, self._btn_stop, self._btn_save):
            lay.addWidget(b)
        lay.addStretch(1)
        lay.addWidget(self._btn_estop); lay.addWidget(self._btn_reset_estop)
        return f

    # ── settings persistence ────────────────────────────────────────────────────
    def _apply_saved_settings(self):
        cam = self._gui_state.get("survey_tags.camera", {})
        if cam:
            self._cam_device.setText(str(cam.get("device", "0")))
            self._cam_fps.setValue(int(cam.get("fps", 30)))
            idx = self._cam_res.findText(cam.get("res", "1280x720"))
            if idx >= 0:
                self._cam_res.setCurrentIndex(idx)
        gan = self._gui_state.get("survey_tags.gantry", {})
        if gan:
            self._g_ip.setText(str(gan.get("ip", "192.168.0.30")))
            self._g_port.setText(str(gan.get("port", "8088")))
            self._g_id.setText(str(gan.get("id", "1")))

    def _persist_settings(self):
        self._gui_state["survey_tags.splitter_sizes"] = self._splitter.sizes()
        self._gui_state["survey_tags.camera"] = {
            "device": self._cam_device.text(), "fps": self._cam_fps.value(),
            "res": self._cam_res.currentText()}
        self._gui_state["survey_tags.gantry"] = {
            "ip": self._g_ip.text(), "port": self._g_port.text(), "id": self._g_id.text()}
        save_gui_state(self._gui_state)

    # ── camera ───────────────────────────────────────────────────────────────────
    def _wire_camera(self):
        self._camera.state_changed.connect(self._on_camera_state)
        self._camera.stats.connect(lambda fps, ms: self._worker and self._worker.set_fps(fps))

    def _toggle_camera(self):
        if self._camera.is_open:
            self._camera.close(); return
        try:
            w, h = (int(v) for v in self._cam_res.currentText().split("x"))
        except ValueError:
            w, h = 1280, 720
        dev = self._cam_device.text().strip()
        try:
            dev = int(dev)
        except ValueError:
            pass
        self._camera.open(dev, w, h, self._cam_fps.value(),
                          calib_path=self._resolve(self._calib_edit.text()),
                          mock=self._cam_mock.isChecked())

    def _on_camera_state(self, state: str):
        if state in ("connected", "connected_mock"):
            self._cam_ind.setText("● Connected"); self._cam_ind.setStyleSheet("color:#34d058;")
            self._btn_cam.setText("Disconnect Camera")
            if self._calib is None:
                self._try_load_calib()
        elif state == "connecting":
            self._cam_ind.setText("● Connecting…"); self._cam_ind.setStyleSheet("color:#ffa726;")
        else:
            self._cam_ind.setText("● Idle"); self._cam_ind.setStyleSheet("color:#9aa3ad;")
            self._btn_cam.setText("Connect Camera")
            if self._state == "SURVEYING":
                self._set_banner("⚠ Camera disconnected — pausing. Reconnect + Resume.", "warn")
                self._toggle_pause(force_pause=True)
        self._refresh_controls()

    def _try_load_calib(self):
        path = self._resolve(self._calib_edit.text())
        try:
            if path and Path(path).exists():
                self._calib = load_fisheye_calibration(Path(path))
        except Exception as exc:
            self._calib = None
            self._set_banner(f"Calibration load failed: {exc}", "warn")

    def _resolve(self, text: str):
        t = (text or "").strip()
        if not t:
            return None
        p = Path(t)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── gantry connect ────────────────────────────────────────────────────────────
    def _toggle_gantry(self):
        if self._controller is not None:
            self._teardown_gantry(); return
        try:
            if self._g_mock.isChecked():
                self._controller = MockFMC4030Controller()
            else:
                self._controller = FMC4030Controller()
            cfg = ControllerConfig(controller_id=int(self._g_id.text()),
                                   ip=self._g_ip.text().strip(),
                                   port=int(self._g_port.text()))
            with self._controller_lock:
                self._controller.connect(cfg)
        except Exception as exc:
            self._controller = None
            self._g_ind.setText("● Error"); self._g_ind.setStyleSheet("color:#ef5350;")
            self._set_banner(f"Gantry connect failed: {exc}", "warn")
            return
        self._g_ind.setText("● Connected"); self._g_ind.setStyleSheet("color:#34d058;")
        self._btn_gantry.setText("Disconnect Gantry")
        self._poll = GantryPollThread(self._controller, self._controller_lock, self)
        self._poll.status.connect(self._on_gantry_status)
        self._poll.start()
        self._refresh_controls()

    def _teardown_gantry(self):
        if self._poll is not None:
            self._poll.stop(); self._poll.wait(1000); self._poll = None
        try:
            if self._controller is not None:
                with self._controller_lock:
                    self._controller.disconnect()
        except Exception:
            pass
        self._controller = None
        self._g_ind.setText("● Idle"); self._g_ind.setStyleSheet("color:#9aa3ad;")
        self._btn_gantry.setText("Connect Gantry")
        self._refresh_controls()

    def _on_gantry_status(self, pos_mm, vel_cms):
        self._pos_label.setText(
            f"Current position:\n X: {pos_mm[0]:+8.2f} mm\n Y: {pos_mm[1]:+8.2f} mm\n Z: {pos_mm[2]:+8.2f} mm")
        self._vel_label.setText(
            f"Velocity:\n X: {vel_cms[0]:6.2f} cm/s\n Y: {vel_cms[1]:6.2f} cm/s\n Z: {vel_cms[2]:6.2f} cm/s")

    # ── jog / move ────────────────────────────────────────────────────────────────
    def _can_drive(self) -> bool:
        return (self._controller is not None and not EMERGENCY_STOP.is_set()
                and self._state in ("SURVEYING", "PAUSED", "IDLE"))

    def _jog_start(self, axis, sign: int):
        if not self._can_drive():
            return
        spd_u = mm_to_units(self._spd.value() * 10.0, axis)
        acc_u = mm_to_units(self._acc.value() * 10.0, axis)
        dec_u = mm_to_units(self._dec.value() * 10.0, axis)
        try:
            with self._controller_lock:
                self._controller.jog_single_axis(axis, 999999.0 * sign, spd_u, acc_u, dec_u)
        except Exception as exc:
            print(f"[survey-gui] jog error: {exc}", file=sys.stderr)
            self._set_banner(f"Jog error: {exc}", "warn")

    def _jog_stop(self, axis):
        if self._controller is None:
            return
        try:
            with self._controller_lock:
                self._controller.stop_axis(axis, mode=1)  # soft stop
        except Exception as exc:
            print(f"[survey-gui] stop_axis error: {exc}", file=sys.stderr)

    def _move_abs(self):
        if not self._can_drive():
            return
        target = (self._mv_x.value(), self._mv_y.value(), self._mv_z.value())
        self._motion = GantryMotionThread(
            self._controller, target, self._spd.value() * 10.0,
            self._acc.value() * 10.0, self._dec.value() * 10.0,
            self._controller_lock, self._logger, self)
        self._motion.done.connect(lambda err: err and self._set_banner(f"Move failed: {err}", "warn"))
        self._motion.start()

    # ── survey lifecycle ────────────────────────────────────────────────────────
    def _start_survey(self):
        if self._state not in ("IDLE",):
            return
        if not (self._camera.is_open and self._controller is not None):
            self._set_banner("Connect camera AND gantry first.", "warn"); return
        self._try_load_calib()
        if self._calib is None:
            self._set_banner("Calibration not loaded — cannot start.", "warn"); return
        # output folder + telemetry logger
        self._run_dir = make_run_dir(_REPO_ROOT / "data", "survey")
        t0 = time.monotonic()
        try:
            self._logger = GantryTelemetryLogger(
                self._controller, self._run_dir / "gantry_telemetry.csv",
                log_hz=100.0, lock=self._controller_lock, t0_monotonic=t0)
            self._logger.start()
        except Exception as exc:
            print(f"[survey-gui] telemetry logger unavailable: {exc}", file=sys.stderr)
            self._logger = None
        # anchor mode
        atext = self._anchor_edit.text().strip().lower()
        anchor_mode = None if atext in ("", "auto") else int(atext)
        # frame queue + worker
        self._frame_queue = Queue(maxsize=2)
        self._camera.attach_worker_queue(self._frame_queue)
        self._worker = DetectionSlamWorker(
            self._frame_queue, self._calib, anchor_mode,
            self._tagsize_spin.value(), self._tagfam_edit.text().strip() or "tag36h11",
            self._run_dir, telemetry_logger=self._logger, parent=self)
        self._worker.frame_overlay.connect(self._camera_preview.on_frame)
        self._worker.tag_map_update.connect(self._on_tag_map)
        self._worker.metrics_update.connect(self._on_metrics)
        self._worker.jump_detected.connect(self._on_jump)
        self._worker.anchor_selected.connect(self._on_anchor_selected)
        self._worker.camera_position.connect(self._on_camera_position)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_state("SURVEYING")

    def _toggle_pause(self, force_pause: bool = False):
        if self._state == "SURVEYING" or force_pause:
            if self._worker:
                self._worker.set_paused(True)
            self._set_state("PAUSED")
        elif self._state == "PAUSED":
            if self._worker:
                self._worker.set_paused(False)
            self._set_state("SURVEYING")

    def _stop_finalize(self):
        if self._state not in ("SURVEYING", "PAUSED"):
            return
        self._set_state("FINALIZING")
        self._camera.detach_worker_queue()
        if self._worker:
            self._worker.stop(); self._worker.wait(3000)
            try:
                self._worker.write_tag_poses_csv()
            except Exception as exc:
                print(f"[survey-gui] tag_poses.csv write failed: {exc}", file=sys.stderr)
        if self._logger:
            try:
                self._logger.stop()
            except Exception:
                pass
        if self._worker is None or self._worker.backend is None or not self._worker.observations:
            self._set_banner("No observations captured — nothing to finalize.", "warn")
            self._set_state("IDLE"); return
        self._live_tag_states = self._worker.live_tag_states()
        obs = list(self._worker.observations)
        frame_init = dict(self._worker.frame_init)
        tag_init = dict(self._worker.backend.optimized_tag_poses())
        anchor = self._worker.anchor_tag_id
        self._batch = BatchOptimizerThread(obs, frame_init, tag_init, anchor, self)
        self._batch.progress.connect(lambda m: self._set_banner(m, "info"))
        self._batch.done.connect(self._on_batch_done)
        self._batch.failed.connect(self._on_batch_failed)
        self._batch.start()

    def _on_batch_done(self, result):
        self._batch_result = result
        refined = result.get("tags", {})
        self._tag_map.set_refined(refined, self._live_tag_states)
        self._update_delta_table(refined, self._live_tag_states)
        self._populate_table(refined, result.get("anchor_id"))
        self._set_banner(f"Finalized: {result['n_qualified']} tags, "
                         f"converged={result['converged']}.", "info")
        self._set_state("DONE")

    def _on_batch_failed(self, msg):
        self._set_banner(f"Batch optimization failed: {msg}", "crit")
        self._set_state("ERROR")

    def _update_delta_table(self, refined: dict, live: dict):
        lines = ["Per-tag Δ (live → batch), mm:"]
        for tid in sorted(refined):
            if tid in live:
                a = refined[tid]["position_m"]; b = live[tid]["position_m"]
                d = math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) * 1000.0
                lines.append(f"  tag {tid}: Δ={d:.1f} mm")
        self._delta_label.setText("\n".join(lines))
        self._delta_label.setVisible(True)

    def _save_map(self):
        if self._state != "DONE" or self._batch_result is None:
            return
        from PyQt5.QtWidgets import QFileDialog
        default = str(_REPO_ROOT / "config/tag_map.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save tag map", default, "YAML (*.yaml)")
        if not path:
            return
        path = Path(path)
        r = self._batch_result
        from datetime import datetime, timezone
        metadata = [
            ("source", str(self._run_dir)),
            ("used_frames_redetection", False),
            ("n_frames_processed", len(self._worker.frame_init) if self._worker else 0),
            ("n_tags_qualified", int(r["n_qualified"])),
            ("n_tags_dropped", int(r["n_dropped"])),
            ("min_observations", MIN_OBSERVATIONS_PER_TAG),
            ("optimizer_iterations", int(r["iterations"])),
            ("initial_error", round(float(r["initial_error"]), 3)),
            ("final_error", round(float(r["final_error"]), 3)),
            ("converged", bool(r["converged"])),
            ("fisheye_calib_path", str(self._resolve(self._calib_edit.text()))),
            ("tool_version", st.TOOL_VERSION + " (gui)"),
            ("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ]
        st.write_tag_map_yaml(path, r.get("anchor_id"), r["tags"], metadata)
        try:
            st.plot_layout(path.with_name(path.stem + "_layout.png"),
                           r.get("anchor_id"), r["tags"], self._pool_cfg)
        except Exception as exc:
            print(f"[survey-gui] layout plot failed: {exc}", file=sys.stderr)
        self._set_banner(f"Saved {path}", "info")

    # ── worker slots ──────────────────────────────────────────────────────────────
    def _on_tag_map(self, tags: dict):
        self._tag_map.set_tags(tags, self._worker.anchor_tag_id if self._worker else None,
                               self._last_cam_pos if hasattr(self, "_last_cam_pos") else None)
        self._populate_table(tags, self._worker.anchor_tag_id if self._worker else None)

    def _on_camera_position(self, pos):
        self._last_cam_pos = pos

    def _on_metrics(self, m: dict):
        elapsed = int(m.get("elapsed_s", 0))
        self._state_label.setText(f"State: {self._state}   Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}")
        self._counts_label.setText(
            f"Frames processed: {m['frames_processed']}   With ≥2 tags: {m['frames_with_2tags']}   "
            f"Tags in graph: {m['tags_in_graph']} / {m['tags_qualified']} qualified   "
            f"Dropped: {m['dropped']}")
        jm = m.get("last_jump_mm", 0.0)
        jcol = "#ef5350" if jm > JUMP_CRITICAL_THRESHOLD_MM else "#e6e6e6"
        res = m.get("median_residual_px", float("nan"))
        res_s = "--" if not math.isfinite(res) else f"{res:.2f}"
        self._quality_label.setText(f"Last backend jump: {jm:.1f} mm    Median residual: {res_s} px")
        self._quality_label.setStyleSheet(f"color:{jcol};")

    def _on_jump(self, tag_id, residual_mm, frame_idx):
        if residual_mm > JUMP_CRITICAL_THRESHOLD_MM:
            self._set_banner(f"⚠ tag {tag_id} — camera shifted {residual_mm:.0f} mm "
                             "(critical). Consider re-recording this region more slowly.", "crit")
        else:
            self._set_banner(f"⚠ tag {tag_id} added — camera shifted {residual_mm:.0f} mm "
                             "(expected small).", "warn")
            self._banner_timer.start(3000)

    def _on_anchor_selected(self, tag_id, frame_idx):
        self._anchor_edit.setText(str(tag_id))
        self._set_banner(f"Anchor: tag {tag_id} (frame {frame_idx})", "info")

    def _on_worker_error(self, msg):
        self._set_banner(msg, "crit")
        self._set_state("ERROR")

    def _populate_table(self, tags: dict, anchor_id):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(tags))
        order = sorted(tags, key=lambda t: (t != anchor_id, t))
        for row, tid in enumerate(order):
            s = tags[tid]
            unc = s.get("uncertainty_mm", float("nan"))
            unc_s = "n/a" if not math.isfinite(unc) else f"{unc:.1f}"
            col = _color_for_uncertainty(unc, s.get("n_observations", 0))
            star = " ★" if tid == anchor_id else ""
            items = [QTableWidgetItem(f"{tid}{star}"),
                     QTableWidgetItem(str(s.get("n_observations", 0))),
                     QTableWidgetItem(unc_s), QTableWidgetItem("")]
            items[3].setBackground(QColor(col))
            for c, it in enumerate(items):
                self._table.setItem(row, c, it)
        self._table.setSortingEnabled(True)
        # worst tag
        worst = None
        for tid, s in tags.items():
            u = s.get("uncertainty_mm", float("nan"))
            if math.isfinite(u) and (worst is None or u > worst[1]):
                worst = (tid, u)
        self._worst_label.setText(f"Worst tag: {worst[0]} ({worst[1]:.1f} mm)" if worst else "Worst tag: --")

    # ── E-stop ───────────────────────────────────────────────────────────────────
    def _emergency_stop(self):
        EMERGENCY_STOP.set()
        try:
            if self._controller is not None:
                with self._controller_lock:
                    self._controller.stop_run()
                    for i in range(3):
                        self._controller.stop_axis(Axis(i), mode=2)
        except Exception as exc:
            print(f"[survey-gui] E-stop SDK error: {exc}", file=sys.stderr)
        if self._worker:
            self._worker.stop()
        self._set_banner("⛔ EMERGENCY STOP engaged. Press 'Reset E-Stop' to clear.", "crit")
        self._set_state("ERROR")

    def _reset_estop(self):
        EMERGENCY_STOP.clear()
        self._set_banner("E-Stop cleared.", "info")
        self._set_state("IDLE")

    # ── state machine + control enablement ────────────────────────────────────────
    def _set_state(self, state: str):
        self._state = state
        self._state_label.setText(f"State: {state}")
        self._refresh_controls()

    def _set_banner(self, text: str, level):
        if not text:
            self._banner.setVisible(False); return
        colors = {"info": "#1f3a5f", "warn": "#5a4a18", "crit": "#5a1818"}
        self._banner.setStyleSheet(
            f"background:{colors.get(level, '#1f3a5f')}; color:#fff; padding:4px; border-radius:4px;")
        self._banner.setText(text); self._banner.setVisible(True)

    def _refresh_controls(self):
        connected = self._camera.is_open and self._controller is not None
        can_start = (self._state == "IDLE" and connected and not EMERGENCY_STOP.is_set())
        self._btn_start.setEnabled(can_start)
        self._btn_pause.setEnabled(self._state in ("SURVEYING", "PAUSED"))
        self._btn_pause.setText("Resume" if self._state == "PAUSED" else "Pause")
        self._btn_stop.setEnabled(self._state in ("SURVEYING", "PAUSED"))
        self._btn_save.setEnabled(self._state == "DONE")
        for b in getattr(self, "_jog_buttons", []):
            b.setEnabled(self._controller is not None and not EMERGENCY_STOP.is_set())
        self._btn_move.setEnabled(self._controller is not None and not EMERGENCY_STOP.is_set())

    def closeEvent(self, ev):
        self._persist_settings()
        try:
            if self._worker:
                self._worker.stop(); self._worker.wait(1500)
            if self._logger:
                self._logger.stop()
            self._teardown_gantry()
            self._camera.close()
        except Exception:
            pass
        super().closeEvent(ev)


def gui_main(argv=None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5"))
    except Exception:
        app.setStyleSheet("QWidget{background:#1a1a1d; color:#e6e6e6;}")
    win = SurveyWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(gui_main())
