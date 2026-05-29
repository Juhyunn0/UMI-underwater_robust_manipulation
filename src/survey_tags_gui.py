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
import gtsam  # noqa: E402 — periodic batch re-optimization + iSAM2 rebuild
from gtsam.symbol_shorthand import L, X  # noqa: E402

import survey_tags as st  # noqa: E402 — reuse the CLI's optimize / YAML / plot
from survey_diagnostics import SurveyDiagnosticsRecorder  # noqa: E402
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
    set_isam2_param,
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
JUMP_SUSPECT_THRESHOLD_MM = 10.0   # a tag whose addition jumped the camera > this is flagged suspect

# Frames with fewer than this many simultaneously detected tags give no
# triangulation constraint and are rejected from the graph.
MIN_SIMULTANEOUS_TAGS = 2

# Warmup: for the first WARMUP_DURATION_S, only let a tag enter the graph once it
# has been confirmed by >= WARMUP_MIN_CONFIRM_FRAMES frames that each saw
# >= WARMUP_MIN_SIMULTANEOUS_TAGS tags (prevents bad early triangulation).
WARMUP_DURATION_S = 30.0
WARMUP_MIN_CONFIRM_FRAMES = 5
WARMUP_MIN_SIMULTANEOUS_TAGS = 3

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

# Periodic batch re-optimization (drift fix). Between frame processing the worker
# re-balances the whole graph with Levenberg-Marquardt so well-observed early tags
# can't lock in and warp the map as the camera travels away from the anchor.
PERIODIC_BATCH_INTERVAL_FRAMES = 200
PERIODIC_BATCH_INTERVAL_S = 30.0
PERIODIC_BATCH_MIN_SHIFT_MM_TO_REBUILD = 5.0   # rebuild iSAM2 only if it actually moved tags
PERIODIC_BATCH_MAX_ITERATIONS = 50
PERIODIC_BATCH_REL_ERROR_TOL = 1e-5

# Duplicate-tag guard. If two physical tags share one AprilTag ID, the backend has
# a single landmark L(id) but two real locations, so every frame drags L(id)
# between them and warps the whole rigid graph (residuals can stay low per-frame).
# Detection is pose-independent: inter-tag DISTANCES are invariant for a rigid
# scene, so a duplicate's observed distance to its established neighbors stops
# matching the map. A tag that disagrees with the majority of its co-observed,
# already-mapped neighbors over several frames is flagged + auto-excluded.
DUPLICATE_DIST_ABS_TOL_M = 0.05         # base tolerance on one inter-tag distance
DUPLICATE_DIST_REL_TOL = 0.05           # + 5 % of that distance
DUPLICATE_MIN_NEIGHBORS = 3             # need >=3 established co-observed tags to vote
DUPLICATE_INCONSISTENT_FRAC = 0.5       # tag disagrees with > this fraction of them
DUPLICATE_CONFIRM_HITS = 5              # this many inconsistent frames -> confirmed

# Loop-closure analysis (revisit detection + a non-blocking "go back" hint).
LOOP_CLOSURE_REVISIT_DISTANCE_M = 0.30
LOOP_CLOSURE_MIN_AGE_S = 30.0
LOOP_HINT_IDLE_S = 60.0          # no revisit within this long -> suggest closing a loop
LOOP_HINT_REEMIT_S = 30.0        # re-surface the (dismissible) hint this often while idle

# Diagnostic recording cadence (see survey_diagnostics.py).
DIAGNOSTICS_FLUSH_INTERVAL_S = 1.0
SNAPSHOT_INTERVAL_S = 30.0
TAG_HISTORY_OBSERVATION_LOG_EVERY_NTH = 50
SLAM_INTERNALS_SAMPLE_EVERY_NTH_FRAME = 60

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
    # Tightened survey gates (reject far/small, high-reprojection, steep detections).
    a.min_tag_area_px = 200.0
    a.max_off_nadir_deg = 30.0
    a.max_image_eccentricity = 0.65
    a.max_tag_tilt_deg = 35.0
    a.max_reprojection_error_px = 3.0
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
    # Floor co-planarity prior ON by default in survey mode: all tags pinned to
    # the z=0 plane (strict = plane through the anchor with +z normal) at sigma 5 mm.
    a.floor_prior_enabled = True
    a.floor_z_sigma = 0.005
    a.floor_plane_min_tags = 4
    a.floor_normal_sigma_deg = 8.0
    a.strict_coplanar = True
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
    batch_completed = pyqtSignal(object)                # BatchEvent dict (periodic re-opt)
    loop_hint = pyqtSignal(bool)                        # show/hide the revisit hint
    duplicate_detected = pyqtSignal(int, int)           # (tag_id, frame_idx) — duplicate ID

    def __init__(self, frame_queue: Queue, calib, anchor_mode,
                 tag_size: float, tag_family: str, run_dir: Path,
                 telemetry_logger=None, t0_monotonic: float | None = None,
                 diagnostics=None, exclude_tag_ids=None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._q = frame_queue
        self._calib = calib
        self._anchor_mode = anchor_mode  # None == auto, else int tag id
        self._tag_size = tag_size
        self._tag_family = tag_family
        self._run_dir = run_dir
        self._telemetry = telemetry_logger
        self._diag = diagnostics  # SurveyDiagnosticsRecorder | None (black-box logging)

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

        # robustness state
        self._rejected_single = 0                       # frames dropped for < 2 tags
        self._confirm: dict[int, int] = {}              # tag -> #frames seen with >=3 tags (warmup)
        self._suspect: set[int] = set()                 # tags that caused a > 10 mm jump on add
        # User-blacklisted IDs are pre-excluded so they never enter the graph (live
        # or batch). Auto-detected duplicate IDs get added here at runtime too.
        self._excluded: set[int] = set(int(t) for t in (exclude_tag_ids or set()))
        self._blacklist: set[int] = set(self._excluded)  # the static, user-supplied set
        self._excl_lock = threading.Lock()

        # duplicate-ID guard state
        self._dup_hits: dict[int, int] = {}             # tag -> #frames inconsistent with neighbors
        self._duplicate_ids: set[int] = set()           # auto-confirmed duplicate IDs

        # metrics / coalescing
        self._frame_idx = -1
        self._frames_processed = 0
        self._frames_with_2tags = 0
        self._dropped = 0
        self._last_proc_t = 0.0
        self._last_emit_t = 0.0
        self._last_residual_px = float("nan")
        self._worst_residual_px = float("nan")
        self._last_jump_residual_mm = 0.0
        self._last_actual_jump_mm = 0.0
        self._t0 = time.monotonic() if t0_monotonic is None else float(t0_monotonic)
        self._cam_csv_fh = None
        self._cam_csv_writer = None
        self._last_fps = 0
        self._last_backend_ms = 0.0
        self._anchor_observed = False

        # periodic-batch drift fix
        self._frames_since_batch = 0
        self._t_last_batch = self._t0
        self._batch_max_iters = PERIODIC_BATCH_MAX_ITERATIONS
        self._n_batch_events = 0
        self._n_isam_rebuilds = 0
        self._last_batch_info = None

        # loop-closure tracking (downsampled camera history with co-visible tags)
        self._cam_hist: list[tuple] = []     # (elapsed_s, x, y, z, frozenset(tag_ids))
        self._last_cam_hist_t = -1.0
        self._last_near_old_t = 0.0
        self._last_revisit_log_t = -1e9
        self._loop_hint_active = False
        self._last_hint_emit_t = -1e9

        # iSAM2 health sampling (rolling per-update wallclock for p50/p95)
        self._isam_ms_window: list[float] = []
        self._relin_count = 0

        # tag-event bookkeeping for tag_history.csv
        self._tag_first_seen: dict[int, float] = {}
        self._tag_promoted: dict[int, float] = {}
        self._tag_last_logged_pos: dict[int, np.ndarray] = {}
        self._tag_obs_logged: dict[int, int] = {}
        self._suspect_logged: set[int] = set()
        self._excluded_logged: set[int] = set()
        self._anchor_t0_pose = None          # anchor Pose3 at first optimized frame
        self._t_last_snapshot = -1e9

    # — control —
    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def set_fps(self, fps: int) -> None:
        self._last_fps = int(fps)

    def toggle_exclude(self, tag_id: int) -> bool:
        """User-driven exclusion of a suspect tag. Excluded tags stop receiving
        new constraints live and are dropped from the finalize observation set.
        Returns the new excluded state."""
        with self._excl_lock:
            if tag_id in self._excluded:
                self._excluded.discard(tag_id)
                state = False
            else:
                self._excluded.add(tag_id)
                state = True
        if self._diag:
            poses = self.backend.optimized_tag_poses() if self.backend else {}
            counts = self.backend.tag_observation_counts if self.backend else {}
            self._diag.log_tag_event("excluded" if state else "suspect_cleared",
                                     int(tag_id), time.monotonic(), self._frame_idx,
                                     pose=poses.get(int(tag_id)),
                                     n_obs=int(counts.get(int(tag_id), 0)))
        return state

    def stop(self) -> None:
        self._stop.set()

    # — lifecycle —
    def _ensure_pipeline(self, frame: np.ndarray) -> None:
        if self._map1 is not None:
            return
        h, w = frame.shape[:2]
        # calib.K is in CALIBRATION-resolution pixels. If the camera delivers a
        # different resolution than the calibration (cameras often ignore the
        # requested size and keep their native one), K must be rescaled or the
        # fisheye undistortion is geometrically wrong and the image stays curved.
        cal_w, cal_h = int(self._calib.image_size[0]), int(self._calib.image_size[1])
        K = np.asarray(self._calib.K, dtype=np.float64).copy()
        if (w, h) != (cal_w, cal_h):
            sx, sy = w / float(cal_w), h / float(cal_h)
            K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
            print(f"[survey-gui] WARNING: camera delivered {w}x{h} but calibration is "
                  f"{cal_w}x{cal_h}; rescaled K by ({sx:.3f}, {sy:.3f}). For best "
                  "undistortion, capture at the calibration resolution or recalibrate.",
                  file=sys.stderr)
        self._map1, self._map2, new_K = build_fisheye_undistort_maps(
            K, self._calib.D, (w, h), 0.0)
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
                if self.backend is not None and not self._paused.is_set():
                    self._maybe_periodic_batch(t_mono)
            except Exception as exc:  # never let the worker die silently
                print(f"[survey-gui] detection/SLAM error: {exc}", file=sys.stderr)
        if self._rejected_single:
            print(f"[survey] rejected {self._rejected_single} single-tag frames "
                  "(no triangulation constraint)", file=sys.stderr)
        self._write_final_diagnostics()
        self._close_csv()

    # — periodic batch re-optimization (drift fix) ——————————————————————————————
    def _maybe_periodic_batch(self, t_mono: float) -> None:
        due_frames = self._frames_since_batch >= PERIODIC_BATCH_INTERVAL_FRAMES
        due_time = (t_mono - self._t_last_batch) >= PERIODIC_BATCH_INTERVAL_S
        if not (due_frames or due_time):
            return
        if len(self.backend.initialized_tag_ids) < 2:
            self._frames_since_batch = 0
            self._t_last_batch = t_mono
            return
        self._run_periodic_batch("frame_count" if due_frames else "time", t_mono)

    def _run_periodic_batch(self, reason: str, t_mono: float) -> None:
        t_start = time.monotonic()
        graph = self.backend.graph
        init = self.backend.current_estimate
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(self._batch_max_iters))
        params.setRelativeErrorTol(PERIODIC_BATCH_REL_ERROR_TOL)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
        e0 = float(graph.error(init))
        result = optimizer.optimize()
        e1 = float(graph.error(result))
        try:
            iters = int(optimizer.iterations())
        except Exception:
            iters = -1

        shift_by_id: dict[int, float] = {}
        new_pose_by_id: dict[int, object] = {}
        anchor_drift_mm = 0.0
        for tid in list(self.backend.initialized_tag_ids):
            tid = int(tid)
            key = L(tid)
            try:
                old = np.asarray(init.atPose3(key).translation(), dtype=np.float64)
                new_pose = result.atPose3(key)
                new = np.asarray(new_pose.translation(), dtype=np.float64)
            except Exception:
                continue
            d = float(np.linalg.norm(new - old)) * 1000.0
            shift_by_id[tid] = d
            new_pose_by_id[tid] = new_pose
            if tid == self.anchor_tag_id:
                anchor_drift_mm = float(np.linalg.norm(new)) * 1000.0
        shifts = list(shift_by_id.values())
        n_shifted = sum(1 for d in shifts if d > 1.0)
        max_shift = max(shifts) if shifts else 0.0
        median_shift = float(np.median(shifts)) if shifts else 0.0

        rebuilt = False
        if max_shift >= PERIODIC_BATCH_MIN_SHIFT_MM_TO_REBUILD:
            p = gtsam.ISAM2Params()
            set_isam2_param(p, "setRelinearizeThreshold", "relinearizeThreshold", 0.001)
            set_isam2_param(p, "setRelinearizeSkip", "relinearizeSkip", 1)
            new_isam = gtsam.ISAM2(p)
            new_isam.update(graph, result)
            self.backend.isam = new_isam
            self.backend.current_estimate = new_isam.calculateEstimate()
            rebuilt = True
            self._n_isam_rebuilds += 1

        wall_ms = (time.monotonic() - t_start) * 1000.0
        mmss = f"{int(t_mono - self._t0) // 60:02d}:{int(t_mono - self._t0) % 60:02d}"
        if rebuilt:
            print(f"[periodic-batch] frame={self._frame_idx}, t={mmss}, optimized in "
                  f"{iters} iterations, max shift {max_shift:.1f} mm, median shift "
                  f"{median_shift:.1f} mm, iSAM2 rebuilt", file=sys.stderr)
        else:
            print(f"[periodic-batch] frame={self._frame_idx}, max shift only "
                  f"{max_shift:.1f} mm, kept iSAM2 state", file=sys.stderr)
        if wall_ms > 2000.0:
            self._batch_max_iters = max(10, self._batch_max_iters // 2)
            print(f"[periodic-batch] WARNING: took {wall_ms:.0f} ms (>2s); reducing "
                  f"max iterations to {self._batch_max_iters} for next time",
                  file=sys.stderr)
            if self._diag:
                self._diag.log_warning(f"periodic batch took {wall_ms:.0f} ms")

        self._n_batch_events += 1
        self._frames_since_batch = 0
        self._t_last_batch = t_mono
        ev = {
            "elapsed_s": t_mono - self._t0,
            "frame_idx": self._frame_idx,
            "trigger_reason": reason,
            "n_tags_in_graph": len(self.backend.initialized_tag_ids),
            "n_camera_poses_in_graph": int(self.backend.next_camera_index),
            "batch_iterations": iters,
            "batch_initial_error": e0,
            "batch_final_error": e1,
            "max_tag_shift_mm": max_shift,
            "median_tag_shift_mm": median_shift,
            "n_shifted": n_shifted,
            "isam2_rebuilt": rebuilt,
            "batch_wallclock_ms": wall_ms,
            "anchor_drifted_mm": anchor_drift_mm,
        }
        self._last_batch_info = ev
        self.batch_completed.emit(ev)
        if self._diag:
            self._diag.log_batch_event(ev)
            self._log_anchor_stability(t_mono)
            if rebuilt:
                counts = self.backend.tag_observation_counts
                for tid, d in shift_by_id.items():
                    if d > 1.0:
                        self._diag.log_tag_event(
                            "shifted", tid, t_mono, self._frame_idx,
                            pose=new_pose_by_id.get(tid),
                            n_obs=int(counts.get(tid, 0)), shift_mm=d)

    # — loop-closure tracking (revisit detection + go-back hint) ————————————————
    def _update_loop_tracking(self, p: np.ndarray, t_mono: float, obs_used) -> None:
        elapsed = t_mono - self._t0
        cur_tags = frozenset(int(o.tag_id) for o in obs_used)
        if (elapsed - self._last_cam_hist_t) >= 0.5:
            self._cam_hist.append((elapsed, float(p[0]), float(p[1]), float(p[2]), cur_tags))
            self._last_cam_hist_t = elapsed
        near_old = False
        for (te, x, y, z, old_tags) in self._cam_hist:
            if (elapsed - te) < LOOP_CLOSURE_MIN_AGE_S:
                continue
            d = math.sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2)
            if d <= LOOP_CLOSURE_REVISIT_DISTANCE_M:
                near_old = True
                if (elapsed - self._last_revisit_log_t) > 5.0:
                    self._last_revisit_log_t = elapsed
                    if self._diag:
                        self._diag.log_loop_closure(
                            elapsed, self._frame_idx, revisit_target_t_s=te,
                            distance_m=d, n_tags_co_observed=len(cur_tags & old_tags))
                break
        if near_old:
            self._last_near_old_t = elapsed
            if self._loop_hint_active:
                self._loop_hint_active = False
                self.loop_hint.emit(False)
        elif (elapsed - self._last_near_old_t) > LOOP_HINT_IDLE_S:
            if (elapsed - self._last_hint_emit_t) > LOOP_HINT_REEMIT_S:
                self._last_hint_emit_t = elapsed
                self._loop_hint_active = True
                self.loop_hint.emit(True)

    # — diagnostics helpers (all no-op when self._diag is None) ————————————————
    def _log_anchor_stability(self, t_mono: float) -> None:
        if not self._diag or self.backend is None or self.anchor_tag_id is None:
            return
        poses = self.backend.optimized_tag_poses()
        ap = poses.get(self.anchor_tag_id)
        if ap is None:
            return
        if self._anchor_t0_pose is None:
            self._anchor_t0_pose = ap
        t = pose_translation(ap)
        rel = self._anchor_t0_pose.between(ap)
        drift_mm = float(np.linalg.norm(pose_translation(rel))) * 1000.0
        rot_deg = float(np.degrees(np.linalg.norm(np.asarray(rel.rotation().rpy(),
                                                              dtype=np.float64))))
        self._diag.log_anchor_stability(
            t_mono - self._t0, int(self.anchor_tag_id),
            [float(t[0]), float(t[1]), float(t[2])], drift_mm, rot_deg, float("nan"))

    def _maybe_snapshot(self, t_mono: float) -> None:
        if not self._diag or self.backend is None:
            return
        elapsed = t_mono - self._t0
        if (elapsed - self._t_last_snapshot) < SNAPSHOT_INTERVAL_S:
            return
        self._t_last_snapshot = elapsed
        recent_batch = bool(self._last_batch_info
                            and (elapsed - self._last_batch_info["elapsed_s"]) <= 5.0)
        self._diag.log_snapshot(elapsed, self._frame_idx, self.anchor_tag_id,
                                self._tag_states(), recent_batch)

    def _write_final_diagnostics(self) -> None:
        if not self._diag or self.backend is None:
            return
        try:
            self._log_anchor_stability(time.monotonic())
            self._maybe_snapshot(time.monotonic() + SNAPSHOT_INTERVAL_S)  # force one last
        except Exception as exc:  # pragma: no cover
            print(f"[survey-gui] final diagnostics failed: {exc}", file=sys.stderr)

    def _note_first_seen(self, obs_list, t_mono: float) -> None:
        for o in obs_list:
            tid = int(o.tag_id)
            if tid not in self._tag_first_seen:
                self._tag_first_seen[tid] = t_mono - self._t0
                self._diag.log_tag_event("first_seen", tid, t_mono, self._frame_idx,
                                         pose=None, n_obs=0)

    def _log_tag_lifecycle(self, obs_used, new_tags, suspect_before, t_mono: float) -> None:
        counts = self.backend.tag_observation_counts
        poses = self.backend.optimized_tag_poses()
        for tid in new_tags:
            tid = int(tid)
            self._tag_promoted.setdefault(tid, t_mono - self._t0)
            self._diag.log_tag_event("promoted", tid, t_mono, self._frame_idx,
                                     pose=poses.get(tid), n_obs=int(counts.get(tid, 0)))
        for o in obs_used:
            tid = int(o.tag_id)
            n = int(counts.get(tid, 0))
            if n - self._tag_obs_logged.get(tid, 0) >= TAG_HISTORY_OBSERVATION_LOG_EVERY_NTH:
                self._tag_obs_logged[tid] = n
                self._diag.log_tag_event("observation", tid, t_mono, self._frame_idx,
                                         pose=poses.get(tid), n_obs=n)
        for tid in (self._suspect - suspect_before):
            self._diag.log_tag_event("suspect_set", int(tid), t_mono, self._frame_idx,
                                     pose=poses.get(int(tid)),
                                     n_obs=int(counts.get(int(tid), 0)))

    def _log_frame_diag(self, tags_detected: int, tags_used: int, t_mono: float,
                        optimized: bool, p=None, update=None) -> None:
        if not self._diag:
            return
        cam = [float("nan")] * 3
        quat = [float("nan")] * 4
        if optimized and update is not None and update.camera_pose is not None:
            if p is not None:
                cam = [float(p[0]), float(p[1]), float(p[2])]
            quat = st._quat_wxyz(update.camera_pose.rotation())
        gs = self._gantry_sample()
        gmm = list(gs.pos_mm) if gs is not None else [float("nan")] * 3
        gvel = list(gs.vel_mm_s) if gs is not None else [float("nan")] * 3
        tags_in_graph = len(self.backend.optimized_tag_poses()) if self.backend else 0
        qualified = sum(1 for s in self._tag_states().values()
                        if s["n_observations"] >= MIN_OBSERVATIONS_PER_TAG)
        self._diag.log_frame({
            "elapsed_s": t_mono - self._t0,
            "frame_idx": self._frame_idx,
            "state": "SURVEYING",
            "tags_detected": tags_detected,
            "tags_used_after_gating": tags_used,
            "tags_in_graph": tags_in_graph,
            "qualified_tags": qualified,
            "single_tag_frames_rejected_total": self._rejected_single,
            "median_residual_px": self._last_residual_px,
            "worst_residual_px": self._worst_residual_px,
            "last_jump_mm": self._last_actual_jump_mm,
            "last_jump_residual_mm": self._last_jump_residual_mm,
            "camera": cam,
            "quat": quat,
            "anchor_observed_this_frame": self._anchor_observed,
            "gantry_mm": gmm,
            "gantry_vel_mm_s": gvel,
            "in_warmup": (t_mono - self._t0) < WARMUP_DURATION_S,
            "backend_update_ms": self._last_backend_ms,
        })

    def _maybe_sample_slam_internals(self, t_mono: float) -> None:
        if (self._frame_idx % SLAM_INTERNALS_SAMPLE_EVERY_NTH_FRAME) != 0:
            return
        w = self._isam_ms_window
        p50 = float(np.percentile(w, 50)) if w else float("nan")
        p95 = float(np.percentile(w, 95)) if w else float("nan")
        n_vars = len(self.backend.initialized_tag_ids) + int(self.backend.next_camera_index)
        due_in = max(0.0, PERIODIC_BATCH_INTERVAL_S - (t_mono - self._t_last_batch))
        self._diag.log_slam_internals({
            "elapsed_s": t_mono - self._t0,
            "frame_idx": self._frame_idx,
            "n_variables": n_vars,
            "n_factors": int(getattr(self.backend, "factor_count", 0)),
            # GTSAM's Python ISAM2 wrapper doesn't expose a relinearization count;
            # reported as 0 (best-effort). p50/p95 cover per-update wallclock.
            "n_relin": self._relin_count,
            "p50": p50,
            "p95": p95,
            "relin_threshold": 0.001,
            "batch_due_in_s": due_in,
        })

    def _process(self, frame: np.ndarray, t_mono: float) -> None:
        self._ensure_pipeline(frame)
        und = self._undistort(frame)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        obs_list = detect_observations(gray, self._detector, self._intrinsics,
                                       self._object_points, self._args, None)
        self._frames_processed += 1
        n_simul = len(obs_list)
        if n_simul >= 2:
            self._frames_with_2tags += 1
        self._anchor_observed = (self.anchor_tag_id is not None
                                 and any(int(o.tag_id) == self.anchor_tag_id for o in obs_list))
        if self._diag:
            self._note_first_seen(obs_list, t_mono)

        # Backend creation deferred until the anchor is known (auto = nearest the
        # image center on the first frame with any detection).
        if self.backend is None:
            anchor = self._select_anchor(obs_list, w, h, self._frames_processed)
            if anchor is None:
                if (time.monotonic() - self._t0) > ANCHOR_TIMEOUT_S:
                    mode = "auto" if self._anchor_mode is None else f"id {self._anchor_mode}"
                    self.error.emit(f"No usable anchor tag ({mode}) within "
                                    f"{ANCHOR_TIMEOUT_S:.0f}s — aborting survey.")
                    self._stop.set()
                self._log_frame_diag(n_simul, 0, t_mono, optimized=False)
                self._emit_overlay(und, obs_list, t_mono)
                return
            self.anchor_tag_id = anchor
            self._args.anchor_tag_id = anchor
            self.backend = TagSlamBackend(self._args)
            self.anchor_selected.emit(anchor, self._frames_processed)

        # (1) Reject single-tag frames: one tag gives no simultaneous triangulation
        # constraint, so it would only re-assert the existing solution.
        if n_simul < MIN_SIMULTANEOUS_TAGS:
            if n_simul == 1:
                self._rejected_single += 1
            self._log_frame_diag(n_simul, 0, t_mono, optimized=False)
            self._emit_overlay(und, obs_list, t_mono)
            return

        # (5) Warmup: count confirmations on rich (>=3 tag) frames; during the first
        # WARMUP_DURATION_S only confirmed tags (>= 5 such frames) + the anchor may
        # enter the graph, so bad early triangulation never gets baked in.
        if n_simul >= WARMUP_MIN_SIMULTANEOUS_TAGS:
            for o in obs_list:
                tid = int(o.tag_id)
                self._confirm[tid] = self._confirm.get(tid, 0) + 1
        in_warmup = (time.monotonic() - self._t0) < WARMUP_DURATION_S
        with self._excl_lock:
            excluded = set(self._excluded)

        def _allowed(tid: int) -> bool:
            if tid in excluded:                      # (6) user-excluded suspect tags
                return False
            if (in_warmup and tid != self.anchor_tag_id
                    and self._confirm.get(tid, 0) < WARMUP_MIN_CONFIRM_FRAMES):
                return False
            return True

        obs_used = [o for o in obs_list if _allowed(int(o.tag_id))]
        # Reject observations of duplicate-ID tags whose geometry disagrees with the
        # established rigid scene (two physical tags sharing one ID warp the map).
        obs_used = self._reject_duplicate_observations(obs_used, t_mono)
        if len(obs_used) < MIN_SIMULTANEOUS_TAGS:
            # Not enough confirmed/un-excluded tags this frame to constrain anything.
            self._log_frame_diag(n_simul, len(obs_used), t_mono, optimized=False)
            self._emit_overlay(und, obs_list, t_mono)
            return

        prev_init = set(self.backend.initialized_tag_ids)
        suspect_before = set(self._suspect)
        t_upd = time.monotonic()
        update = self.backend.update(obs_used)
        self._last_backend_ms = (time.monotonic() - t_upd) * 1000.0
        self._isam_ms_window.append(self._last_backend_ms)
        if len(self._isam_ms_window) > 60:
            self._isam_ms_window = self._isam_ms_window[-60:]
        self._frame_idx += 1
        new_tags = set(self.backend.initialized_tag_ids) - prev_init
        res = [float(getattr(o, "reprojection_error_px", float("nan"))) for o in obs_used]
        res = [r for r in res if math.isfinite(r)]
        self._last_residual_px = float(np.median(res)) if res else float("nan")
        self._worst_residual_px = float(max(res)) if res else float("nan")

        if update.optimized and update.camera_pose is not None:
            p = np.asarray(update.camera_pose.translation(), dtype=np.float64).reshape(3)
            self.camera_position.emit((float(p[0]), float(p[1]), float(p[2])))
            self.frame_init[self._frame_idx] = update.camera_pose
            for o in obs_used:
                self.observations.append((self._frame_idx, int(o.tag_id), o.camera_T_tag))
            self._detect_jump(p, t_mono, obs_used, new_tags)
            self._write_csv_row(update, obs_used, p, t_mono)
            self._frames_since_batch += 1
            self._update_loop_tracking(p, t_mono, obs_used)
            if self._diag:
                self._log_tag_lifecycle(obs_used, new_tags, suspect_before, t_mono)
                self._log_frame_diag(n_simul, len(obs_used), t_mono,
                                     optimized=True, p=p, update=update)
                self._maybe_snapshot(t_mono)
                self._maybe_sample_slam_internals(t_mono)
        else:
            self._log_frame_diag(n_simul, len(obs_used), t_mono, optimized=False)

        self._maybe_emit(und, obs_list, t_mono, update)

    def _reject_duplicate_observations(self, obs_used, t_mono: float):
        """Drop this frame's observations of tags whose inter-tag geometry is
        inconsistent with the established map — the signature of a duplicate ID.

        Distances between tags are rigid-scene invariants (independent of the
        possibly-drifting camera pose), so for each already-mapped tag co-observed
        this frame we compare its observed distance to every other established
        co-observed tag against the map distance. A tag that disagrees with the
        majority of its established neighbors is mis-associated. After
        DUPLICATE_CONFIRM_HITS such frames the ID is confirmed a duplicate and
        permanently excluded (added to self._excluded)."""
        if self.backend is None or len(obs_used) < DUPLICATE_MIN_NEIGHBORS + 1:
            return obs_used
        est = self.backend.initialized_tag_ids
        mp = self.backend.optimized_tag_poses()
        obspos = {int(o.tag_id): np.asarray(o.camera_T_tag.translation(),
                                            dtype=np.float64).reshape(3) for o in obs_used}
        ids = [tid for tid in obspos if tid in est and tid in mp]
        if len(ids) < DUPLICATE_MIN_NEIGHBORS:
            return obs_used
        mappos = {tid: pose_translation(mp[tid]) for tid in ids}
        inconsistent: set[int] = set()
        for i in ids:
            bad = tot = 0
            for j in ids:
                if i == j:
                    continue
                d_obs = float(np.linalg.norm(obspos[i] - obspos[j]))
                d_map = float(np.linalg.norm(mappos[i] - mappos[j]))
                tol = DUPLICATE_DIST_ABS_TOL_M + DUPLICATE_DIST_REL_TOL * d_map
                tot += 1
                if abs(d_obs - d_map) > tol:
                    bad += 1
            if tot and (bad / tot) > DUPLICATE_INCONSISTENT_FRAC:
                inconsistent.add(i)
        # If (almost) everyone disagrees, the MAP is the problem this frame, not a
        # single tag — don't punish anyone (avoids cascading false positives).
        if not inconsistent or len(inconsistent) >= len(ids) - 1:
            return obs_used
        for tid in inconsistent:
            self._dup_hits[tid] = self._dup_hits.get(tid, 0) + 1
            if (self._dup_hits[tid] >= DUPLICATE_CONFIRM_HITS
                    and tid not in self._duplicate_ids):
                self._duplicate_ids.add(tid)
                with self._excl_lock:
                    self._excluded.add(tid)
                print(f"[duplicate] tag {tid} confirmed DUPLICATE ID after "
                      f"{self._dup_hits[tid]} inconsistent frames — auto-excluded "
                      "(two physical tags appear to share this ID)", file=sys.stderr)
                self.duplicate_detected.emit(int(tid), self._frame_idx)
                if self._diag:
                    self._diag.log_tag_event(
                        "duplicate_detected", tid, t_mono, self._frame_idx,
                        pose=mp.get(tid),
                        n_obs=int(self.backend.tag_observation_counts.get(tid, 0)))
        return [o for o in obs_used if int(o.tag_id) not in inconsistent]

    def _detect_jump(self, p_curr: np.ndarray, t_curr: float, obs_list, new_tags) -> None:
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
            self._last_jump_mm = residual            # velocity-residual (UI semantics)
            self._last_actual_jump_mm = actual       # raw frame-to-frame displacement
            self._last_jump_residual_mm = residual   # jump beyond constant-velocity predict
            # (6) A newly-added tag that jumped the camera > 10 mm is likely a bad
            # triangulation — flag it suspect so the UI highlights it for exclusion.
            if residual > JUMP_SUSPECT_THRESHOLD_MM and new_tags:
                self._suspect |= {int(t) for t in new_tags}
            if residual > JUMP_WARNING_THRESHOLD_MM:
                newest = (next(iter(new_tags)) if new_tags
                          else (obs_list[-1].tag_id if obs_list else -1))
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
        with self._excl_lock:
            excluded = set(self._excluded)
        suspect = set(self._suspect)
        out = {}
        for tid, pose in self.backend.optimized_tag_poses().items():
            tid = int(tid)
            t = pose_translation(pose)
            q = st._quat_wxyz(pose.rotation())
            out[tid] = {
                "position_m": [float(t[0]), float(t[1]), float(t[2])],
                "quaternion_wxyz": q,
                "n_observations": int(counts.get(tid, 0)),
                "uncertainty_mm": float("nan"),  # live: no marginals (cost); batch fills it
                "suspect": tid in suspect,
                "excluded": tid in excluded,
                "duplicate": tid in self._duplicate_ids,
            }
        return out

    def finalize_observations(self):
        """Observations / inits for finalize, with user-excluded tags removed."""
        with self._excl_lock:
            excluded = set(self._excluded)
        obs = [(f, t, p) for (f, t, p) in self.observations if t not in excluded]
        tag_init = {t: p for t, p in self.backend.optimized_tag_poses().items()
                    if int(t) not in excluded}
        return obs, dict(self.frame_init), tag_init, excluded

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
            "rejected_single": self._rejected_single,
            "suspect": len(self._suspect),
            "duplicates": sorted(self._duplicate_ids),
            "in_warmup": (time.monotonic() - self._t0) < WARMUP_DURATION_S,
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
    tag_right_clicked = pyqtSignal(int)  # right-click a tag -> toggle exclude

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
        self._flash_on = False     # 200 ms blue flash on a periodic-batch reopt

    def flash(self):
        """Brief blue flash to mark a periodic batch re-optimization."""
        self._flash_on = True
        self.update()
        QTimer.singleShot(200, self._clear_flash)

    def _clear_flash(self):
        self._flash_on = False
        self.update()

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
            if self._flash_on:
                flash = QColor("#4ea1ff"); flash.setAlpha(60)
                p.fillRect(self.rect(), flash)
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
            excluded = s.get("excluded", False)
            suspect = s.get("suspect", False)
            duplicate = s.get("duplicate", False)
            col = QColor(_color_for_uncertainty(s.get("uncertainty_mm", float("nan")),
                                                s.get("n_observations", 0)))
            if excluded:
                col = QColor("#555555")
            r = 11 if is_anchor else 8
            dropped = s.get("n_observations", 0) < MIN_OBSERVATIONS_PER_TAG
            pen = QPen(QColor("#0a0d11"), 1.5)
            if dropped:
                pen = QPen(QColor("#aaaaaa"), 1.2); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.setBrush(col)
            p.drawEllipse(pt, r, r)
            # (6) suspect: magenta ring (likely-bad triangulation; right-click to exclude)
            if suspect and not excluded:
                ring = QPen(QColor("#ff37ff"), 2.5); ring.setStyle(Qt.DashLine)
                p.setPen(ring); p.setBrush(Qt.NoBrush)
                p.drawEllipse(pt, r + 5, r + 5)
                p.setPen(QColor("#ff37ff"))
                p.drawText(QPointF(pt.x() - r - 10, pt.y() - r - 6), "!")
            if excluded:  # crossed out
                p.setPen(QPen(QColor("#ff5555"), 2))
                p.drawLine(QPointF(pt.x() - r, pt.y() - r), QPointF(pt.x() + r, pt.y() + r))
                p.drawLine(QPointF(pt.x() - r, pt.y() + r), QPointF(pt.x() + r, pt.y() - r))
            if duplicate:  # auto-detected duplicate ID — orange "DUP" tag
                p.setPen(QColor("#ff9800"))
                p.drawText(QPointF(pt.x() - r - 4, pt.y() + r + 12), "DUP")
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
                 (">= 15 mm", _COL_RED), ("dropped", _COL_GRAY),
                 ("suspect (right-click to exclude)", "#ff37ff"),
                 ("DUP = auto-detected duplicate ID", "#ff9800")]
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
        elif ev.button() == Qt.RightButton:
            hit = self._tag_at(ev.pos())
            if hit is not None:
                self.tag_right_clicked.emit(int(hit[0]))

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
            # Survey tags sit on the pool floor: pin them to z=0 in the batch too.
            result = st.optimize(self._obs, frame_init, tag_init, self._anchor,
                                 MIN_OBSERVATIONS_PER_TAG, OPTIMIZER_MAX_ITERATIONS,
                                 floor_coplanar=True)
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
        self._diag = None  # SurveyDiagnosticsRecorder (set when SURVEYING begins)
        self._run_dir: Path | None = None

        self._batch_result = None
        self._live_tag_states = {}
        self._last_cam_pos = None
        self._last_gantry_mm = (float("nan"), float("nan"), float("nan"))
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
        self._tag_map.tag_right_clicked.connect(self._on_tag_right_clicked)
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
        QShortcut(QKeySequence("Ctrl+N"), self,
                  activated=lambda: self._note_edit.setFocus(),
                  context=Qt.ApplicationShortcut)

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
        # Exclude-IDs blacklist: physically duplicated tag IDs (two tags share an ID)
        # corrupt the map — drop them entirely. Auto-detected duplicates are appended
        # here at runtime so the list is ready to reuse on the next clean run.
        self._exclude_edit = QLineEdit()
        self._exclude_edit.setPlaceholderText("e.g. 64,65,68,69 — duplicate / bad IDs to drop")
        self._exclude_edit.setToolTip(
            "Comma-separated AprilTag IDs to exclude from the graph entirely "
            "(live + batch). Use for physically duplicated IDs that warp the map.")
        g.addWidget(QLabel("Exclude tag IDs"), 3, 0)
        g.addWidget(self._exclude_edit, 3, 1, 1, 8)
        return f

    def _parse_exclude_ids(self) -> set:
        out = set()
        for tok in (self._exclude_edit.text() or "").replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    out.add(int(tok))
                except ValueError:
                    pass
        return out

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
        self._batch_label = QLabel("Last batch: --")
        self._batch_label.setStyleSheet("color:#9aa3ad;")
        self._worst_label = QLabel("Worst tag: --")
        # Non-blocking, click-to-dismiss loop-closure suggestion.
        self._hint_label = QPushButton(
            "💡 Consider revisiting an earlier area to close the loop "
            "(improves global map consistency)   ✕")
        self._hint_label.setFlat(True)
        self._hint_label.setStyleSheet(
            "QPushButton{background:#5a4a18; color:#ffd970; padding:3px; "
            "border-radius:4px; text-align:left;}")
        self._hint_label.setVisible(False)
        self._hint_label.clicked.connect(lambda: self._hint_label.setVisible(False))
        self._banner = QLabel("")
        self._banner.setVisible(False)
        self._banner.setWordWrap(True)
        for w in (self._state_label, self._counts_label, self._quality_label,
                  self._batch_label, self._worst_label, self._hint_label, self._banner):
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
        # Add-Note field: the single most valuable diagnostic annotation — lets the
        # user mark "this is the moment it broke" (Ctrl+N focuses, Enter submits).
        lay.addSpacing(12)
        lay.addWidget(QLabel("Note:"))
        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("annotate the run (Ctrl+N), Enter to log…")
        self._note_edit.setMinimumWidth(220)
        self._note_edit.returnPressed.connect(self._add_note)
        self._btn_note = QPushButton("Add Note"); self._btn_note.clicked.connect(self._add_note)
        lay.addWidget(self._note_edit, 1); lay.addWidget(self._btn_note)
        lay.addStretch(0)
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
        self._exclude_edit.setText(str(self._gui_state.get("survey_tags.exclude_ids", "")))

    def _persist_settings(self):
        self._gui_state["survey_tags.splitter_sizes"] = self._splitter.sizes()
        self._gui_state["survey_tags.camera"] = {
            "device": self._cam_device.text(), "fps": self._cam_fps.value(),
            "res": self._cam_res.currentText()}
        self._gui_state["survey_tags.gantry"] = {
            "ip": self._g_ip.text(), "port": self._g_port.text(), "id": self._g_id.text()}
        self._gui_state["survey_tags.exclude_ids"] = self._exclude_edit.text()
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
        self._last_gantry_mm = (float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2]))
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
            self._log_action("jog_start", {"axis": axis.name, "direction": int(sign),
                                           "speed_cm_s": self._spd.value()})
        except Exception as exc:
            print(f"[survey-gui] jog error: {exc}", file=sys.stderr)
            self._set_banner(f"Jog error: {exc}", "warn")

    def _jog_stop(self, axis):
        if self._controller is None:
            return
        try:
            with self._controller_lock:
                self._controller.stop_axis(axis, mode=1)  # soft stop
            self._log_action("jog_stop", {"axis": axis.name})
        except Exception as exc:
            print(f"[survey-gui] stop_axis error: {exc}", file=sys.stderr)

    def _move_abs(self):
        if not self._can_drive():
            return
        target = (self._mv_x.value(), self._mv_y.value(), self._mv_z.value())
        self._log_action("move_abs_start", {"target_xyz_mm": list(target)})
        self._motion = GantryMotionThread(
            self._controller, target, self._spd.value() * 10.0,
            self._acc.value() * 10.0, self._dec.value() * 10.0,
            self._controller_lock, self._logger, self)
        self._motion.done.connect(lambda err: self._on_move_done(err, target))
        self._motion.start()

    def _on_move_done(self, err, target):
        self._log_action("move_abs_complete",
                         {"target_xyz_mm": list(target), "error": err or ""})
        if err:
            self._set_banner(f"Move failed: {err}", "warn")

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
        # diagnostic "black box" recorder (daemon thread; shares t0 with the logger)
        try:
            self._diag = SurveyDiagnosticsRecorder(
                self._run_dir, t0_monotonic=t0,
                flush_interval_s=DIAGNOSTICS_FLUSH_INTERVAL_S)
            self._diag.start()
        except Exception as exc:
            print(f"[survey-gui] diagnostics recorder unavailable: {exc}", file=sys.stderr)
            self._diag = None
        # anchor mode
        atext = self._anchor_edit.text().strip().lower()
        anchor_mode = None if atext in ("", "auto") else int(atext)
        # frame queue + worker
        self._frame_queue = Queue(maxsize=2)
        self._camera.attach_worker_queue(self._frame_queue)
        exclude_ids = self._parse_exclude_ids()
        if exclude_ids:
            print(f"[survey-gui] excluding blacklisted tag IDs: {sorted(exclude_ids)}",
                  file=sys.stderr)
        self._worker = DetectionSlamWorker(
            self._frame_queue, self._calib, anchor_mode,
            self._tagsize_spin.value(), self._tagfam_edit.text().strip() or "tag36h11",
            self._run_dir, telemetry_logger=self._logger, t0_monotonic=t0,
            diagnostics=self._diag, exclude_tag_ids=exclude_ids, parent=self)
        self._worker.frame_overlay.connect(self._camera_preview.on_frame)
        self._worker.tag_map_update.connect(self._on_tag_map)
        self._worker.metrics_update.connect(self._on_metrics)
        self._worker.jump_detected.connect(self._on_jump)
        self._worker.anchor_selected.connect(self._on_anchor_selected)
        self._worker.camera_position.connect(self._on_camera_position)
        self._worker.error.connect(self._on_worker_error)
        self._worker.batch_completed.connect(self._on_batch_event)
        self._worker.loop_hint.connect(self._on_loop_hint)
        self._worker.duplicate_detected.connect(self._on_duplicate_detected)
        self._worker.start()
        self._set_state("SURVEYING")
        self._log_action("start_survey", {"anchor_mode": atext or "auto",
                                          "tag_size_m": self._tagsize_spin.value()})

    def _toggle_pause(self, force_pause: bool = False):
        if self._state == "SURVEYING" or force_pause:
            if self._worker:
                self._worker.set_paused(True)
            self._set_state("PAUSED")
            self._log_action("pause", {"forced": bool(force_pause)})
        elif self._state == "PAUSED":
            if self._worker:
                self._worker.set_paused(False)
            self._set_state("SURVEYING")
            self._log_action("resume")

    def _stop_finalize(self):
        if self._state not in ("SURVEYING", "PAUSED"):
            return
        self._log_action("stop_finalize")
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
            self._close_diagnostics(None)
            self._set_banner("No observations captured — nothing to finalize.", "warn")
            self._set_state("IDLE"); return
        self._live_tag_states = self._worker.live_tag_states()
        # User-excluded suspect tags are dropped from the finalize observation set.
        obs, frame_init, tag_init, excluded = self._worker.finalize_observations()
        if excluded:
            print(f"[survey] finalize excludes user-flagged tags: "
                  f"{sorted(excluded)}", file=sys.stderr)
        anchor = self._worker.anchor_tag_id
        self._batch = BatchOptimizerThread(obs, frame_init, tag_init, anchor, self)
        self._batch.progress.connect(lambda m: self._set_banner(m, "info"))
        self._batch.done.connect(self._on_batch_done)
        self._batch.failed.connect(self._on_batch_failed)
        self._batch.start()

    def _on_batch_done(self, result):
        self._batch_result = result
        self._close_diagnostics(result)
        refined = result.get("tags", {})
        self._tag_map.set_refined(refined, self._live_tag_states)
        self._update_delta_table(refined, self._live_tag_states)
        self._populate_table(refined, result.get("anchor_id"))
        self._set_banner(f"Finalized: {result['n_qualified']} tags, "
                         f"converged={result['converged']}.", "info")
        self._set_state("DONE")

    def _on_batch_failed(self, msg):
        self._close_diagnostics(None)
        self._set_banner(f"Batch optimization failed: {msg}", "crit")
        self._set_state("ERROR")

    def _log_action(self, action_type: str, detail: dict | None = None):
        if self._diag is not None:
            self._diag.log_user_action(action_type, detail, self._last_gantry_mm)

    def _add_note(self):
        text = self._note_edit.text().strip()
        if not text:
            return
        self._note_edit.clear()
        fi = self._worker._frame_idx if self._worker else -1
        if self._diag is not None:
            self._diag.log_user_note(text, fi, self._last_cam_pos, self._last_gantry_mm)
            self._log_action("user_note_added", {"note": text})
            self._set_banner(f"Note logged: {text}", "info")
            self._banner_timer.start(2500)
        else:
            self._set_banner("Note ignored — not surveying (no recorder active).", "warn")
            self._banner_timer.start(2500)

    def _git_commit(self):
        try:
            import subprocess
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO_ROOT),
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    def _close_diagnostics(self, result=None):
        """Write diagnostics_summary.json and shut the recorder down. Safe to call
        more than once (no-op once the recorder is gone)."""
        if self._diag is None:
            return
        meta = {}
        try:
            w = self._worker
            anchor = w.anchor_tag_id if w else None
            dur = (time.monotonic() - w._t0) if w else None
            nframes = len(w.frame_init) if w else None
            states = self._live_tag_states or (w.live_tag_states() if w else {})
            nqual = sum(1 for s in states.values()
                        if s.get("n_observations", 0) >= MIN_OBSERVATIONS_PER_TAG)
            if result is not None:
                nqual = int(result.get("n_qualified", nqual))
                anchor = result.get("anchor_id", anchor)
                self._diag.set_final_tag_stats(result.get("tags", {}))
            meta = {
                "anchor_tag_id": anchor,
                "survey_duration_s": round(dur, 2) if dur else None,
                "n_frames_processed": nframes,
                "n_tags_qualified": nqual,
                "fisheye_calib_path": str(self._resolve(self._calib_edit.text())),
                "constants": {
                    "PERIODIC_BATCH_INTERVAL_FRAMES": PERIODIC_BATCH_INTERVAL_FRAMES,
                    "PERIODIC_BATCH_INTERVAL_S": PERIODIC_BATCH_INTERVAL_S,
                    "WARMUP_DURATION_S": WARMUP_DURATION_S,
                    "min_tag_area_px": 200.0,
                    "max_off_nadir_deg": 30.0,
                    "max_reprojection_error_px": 3.0,
                },
                "git_commit": self._git_commit(),
                "tool_version": st.TOOL_VERSION + " (gui 2.0)",
            }
        except Exception as exc:
            print(f"[survey-gui] diagnostics meta failed: {exc}", file=sys.stderr)
        try:
            self._diag.close(meta)
        finally:
            self._diag = None

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

    def _on_tag_right_clicked(self, tag_id: int):
        """Right-click a (suspect) tag in the map to exclude/include it."""
        if self._worker is None:
            return
        excluded = self._worker.toggle_exclude(int(tag_id))
        self._log_action("tag_excluded", {"tag_id": int(tag_id), "excluded": bool(excluded)})
        self._set_banner(
            f"tag {tag_id} {'EXCLUDED' if excluded else 're-included'} "
            f"({'dropped from' if excluded else 'restored to'} the graph + final map).",
            "warn" if excluded else "info")

    def _on_metrics(self, m: dict):
        elapsed = int(m.get("elapsed_s", 0))
        warm = "  [WARMUP]" if m.get("in_warmup") else ""
        self._state_label.setText(
            f"State: {self._state}{warm}   Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}")
        dups = m.get("duplicates", [])
        dup_s = f"   DUPLICATE IDs: {dups}" if dups else ""
        self._counts_label.setText(
            f"Frames processed: {m['frames_processed']}   With ≥2 tags: {m['frames_with_2tags']}   "
            f"Tags in graph: {m['tags_in_graph']} / {m['tags_qualified']} qualified   "
            f"Dropped: {m['dropped']}   Single-tag rejected: {m.get('rejected_single', 0)}   "
            f"Suspect: {m.get('suspect', 0)}{dup_s}")
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
        self._log_action("anchor_changed", {"tag_id": int(tag_id), "frame_idx": int(frame_idx)})
        self._set_banner(f"Anchor: tag {tag_id} (frame {frame_idx})", "info")

    def _on_batch_event(self, ev: dict):
        self._tag_map.flash()
        s = int(ev.get("elapsed_s", 0))
        mmss = f"{s // 60:02d}:{s % 60:02d}"
        self._batch_label.setText(
            f"Last batch: {mmss} · max tag shift {ev.get('max_tag_shift_mm', 0.0):.1f} mm"
            f"{' · iSAM2 rebuilt' if ev.get('isam2_rebuilt') else ''}")
        self._set_banner(f"Batch reoptimization at {mmss} — "
                         f"{ev.get('n_shifted', 0)} tags shifted >1 mm", "info")
        self._banner_timer.start(5000)

    def _on_loop_hint(self, show: bool):
        self._hint_label.setVisible(bool(show))

    def _on_duplicate_detected(self, tag_id: int, frame_idx: int):
        # Append to the Exclude-IDs field so the user can reuse the full blacklist on
        # the next run, and warn loudly — this is the map-warp root cause.
        cur = self._parse_exclude_ids()
        cur.add(int(tag_id))
        self._exclude_edit.setText(",".join(str(t) for t in sorted(cur)))
        self._log_action("duplicate_detected", {"tag_id": int(tag_id), "frame_idx": int(frame_idx)})
        self._set_banner(
            f"⚠ tag {tag_id} looks like a DUPLICATE ID (two physical tags share it) — "
            f"auto-excluded at frame {frame_idx}. It was warping the map. "
            "Re-survey with it blacklisted for a clean map.", "crit")

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
        self._log_action("emergency_stop")
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
        self._close_diagnostics(None)
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
            self._close_diagnostics(None)
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
