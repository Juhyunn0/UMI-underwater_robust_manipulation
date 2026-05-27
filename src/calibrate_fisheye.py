#!/usr/bin/env python3
"""
calibrate_fisheye.py — Fisheye camera intrinsic calibration GUI (v4.0).

Disk-record + folder-select workflow:
  1. Connect camera.
  2. Click Record — frames with a detected calibration pattern are saved
     continuously to a timestamped session folder under ~/calib_sessions/.
  3. Click Stop — the new session folder is added to the folder list.
  4. (Optionally) Add more folders from previous sessions via "Add Folder".
  5. Click Calibrate — images in all listed folders are loaded, corners
     re-detected, and cv2.fisheye.calibrate runs.
  6. Inspect RMS and per-image errors.
  7. Save YAML (loadable by load_fisheye_calibration in fisheye_gantry_tagslam.py).

Usage:
    python -m src.calibrate_fisheye
    python -m src.calibrate_fisheye --mock-camera
    python -m src.calibrate_fisheye --device 1 --output config/my_calib.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from PyQt5.QtCore import (
    QMetaObject, QObject, Qt, QThread,
    pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QRadioButton, QScrollArea,
    QShortcut, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

# --- sys.path shim (supports both `python src/calibrate_fisheye.py` and -m) --
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR   = _THIS_FILE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# --- optional dependencies ---------------------------------------------------
try:
    import numpy as np
    _HAVE_NP = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAVE_NP = False

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAVE_CV2 = False

try:
    import yaml as _yaml
    _HAVE_YAML = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _HAVE_YAML = False

try:
    import qdarkstyle
    _HAVE_DARK = True
except ImportError:
    qdarkstyle = None  # type: ignore[assignment]
    _HAVE_DARK = False

try:
    from fisheye_camera import FisheyeCameraSession
    _HAVE_CAM_SESSION = True
except ImportError:
    FisheyeCameraSession = None  # type: ignore[assignment]
    _HAVE_CAM_SESSION = False

# =============================================================================
# Tunable constants
# =============================================================================
MIN_SHARPNESS     = 50.0   # Laplacian-variance floor; below = too blurry to save
DEDUP_CENTROID_PX = 40.0   # Centroid dedup distance (pixels)
DEDUP_TIME_MS     = 300    # Min interval (ms) between consecutive saves

_SESSION_ROOT   = Path.home() / "calib_sessions"  # default root for recorded sessions
_DISPLAY_FPS    = 15
_DEFAULT_OUTPUT = "config/fisheye_calibration.yaml"
_SETTINGS_PATH  = Path.home() / ".umi_gui_state.json"
_SETTINGS_KEY   = "calibrate_fisheye"

# --- dark QSS ----------------------------------------------------------------
_QSS = """
QMainWindow, QDialog, QWidget { background-color: #1a1a1d; color: #e6e6e6; }
QScrollArea, QScrollArea > QWidget > QWidget { background-color: #1a1a1d; border: 0; }
QLabel { color: #e6e6e6; }
QSplitter::handle:horizontal {
    width: 8px; background-color: #2a2a2e;
    border-left: 1px solid #3f3f46; border-right: 1px solid #3f3f46;
}
QSplitter::handle:horizontal:hover    { background-color: #4ea1ff; }
QSplitter::handle:horizontal:pressed  { background-color: #1a73e8; }
QSplitter::handle:vertical {
    height: 8px; background-color: #2a2a2e;
    border-top: 1px solid #3f3f46; border-bottom: 1px solid #3f3f46;
}
QSplitter::handle:vertical:hover   { background-color: #4ea1ff; }
QSplitter::handle:vertical:pressed { background-color: #1a73e8; }
QFrame#SectionCard {
    background-color: #2b2b2b; border: 1px solid #3f3f46;
    border-radius: 10px; padding: 12px;
}
QPushButton {
    background-color: #3a3a40; border: 1px solid #4a4a52;
    border-radius: 6px; padding: 6px 12px; color: #e6e6e6;
}
QPushButton:hover { background-color: #45454c; border-color: #4ea1ff; }
QPushButton:pressed { background-color: #2e2e34; }
QPushButton:disabled { background-color: #2a2a2e; color: #555; border-color: #2e2e2e; }
QPushButton#PrimaryBtn {
    background-color: #1a73e8; border: 1px solid #1a73e8;
    color: white; font-weight: 600;
}
QPushButton#PrimaryBtn:hover { background-color: #2589ff; }
QPushButton#PrimaryBtn:disabled {
    background-color: #1a1a1d; color: #555; border-color: #2e2e2e;
}
QPushButton#RecordBtn {
    background-color: #388e3c; border: 1px solid #4caf50;
    color: white; font-weight: 700; font-size: 13px;
}
QPushButton#RecordBtn:hover { background-color: #4caf50; }
QPushButton#RecordBtn:disabled { background-color: #2a2a2e; color: #555; border-color: #2e2e2e; }
QPushButton#StopBtn {
    background-color: #c62828; border: 1px solid #ef5350;
    color: white; font-weight: 700; font-size: 13px;
}
QPushButton#StopBtn:hover { background-color: #ef5350; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background-color: #1f1f22; border: 1px solid #3a3a40;
    border-radius: 5px; padding: 4px 6px; color: #e6e6e6; min-height: 22px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {
    border: 1px solid #4ea1ff;
}
QCheckBox { color: #e6e6e6; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border: 1px solid #4a4a52;
    border-radius: 3px; background: #1f1f22;
}
QCheckBox::indicator:checked { background: #1a73e8; }
QRadioButton { color: #e6e6e6; spacing: 6px; }
QListWidget {
    background-color: #1f1f22; border: 1px solid #3a3a40;
    border-radius: 5px; color: #e6e6e6;
}
QListWidget::item:selected { background-color: #1a73e8; }
QListWidget::item:hover { background-color: #2d2d32; }
QStatusBar { background-color: #1a1a1d; color: #c0c0c0; border-top: 1px solid #2f2f33; }
QToolTip { background-color: #1f1f22; color: #e6e6e6; border: 1px solid #4ea1ff; padding: 4px; }
QScrollBar:vertical { background: #1a1a1d; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3a40; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1a1a1d; height: 8px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a3a40; border-radius: 4px; min-width: 20px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


# =============================================================================
# Settings helpers
# =============================================================================
def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text()).get(_SETTINGS_KEY, {})
    except (OSError, ValueError):
        return {}


def _save_settings(d: dict) -> None:
    try:
        full: dict = {}
        try:
            full = json.loads(_SETTINGS_PATH.read_text())
        except (OSError, ValueError):
            pass
        full[_SETTINGS_KEY] = d
        _SETTINGS_PATH.write_text(json.dumps(full, indent=2))
    except OSError:
        pass


# =============================================================================
# Data types
# =============================================================================
@dataclass
class FrameRecord:
    frame:               Any    # full-resolution BGR ndarray
    corners:             Any    # refined corners, shape (N, 1, 2) float32
    sharpness:           float  # Laplacian variance over corner bbox
    centroid:            tuple  # (cx, cy) mean of corners
    timestamp_monotonic: float


class CalibState(Enum):
    IDLE        = auto()
    RECORDING   = auto()
    LOADING     = auto()
    CALIBRATING = auto()
    DONE        = auto()
    ERROR       = auto()


# =============================================================================
# Helper functions
# =============================================================================
def compute_sharpness(gray: Any, corners: Any) -> float:
    """Laplacian variance over the axis-aligned bounding rect of corners ±10 px."""
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    x0 = max(0, int(xs.min()) - 10)
    x1 = min(gray.shape[1], int(xs.max()) + 10)
    y0 = max(0, int(ys.min()) - 10)
    y1 = min(gray.shape[0], int(ys.max()) + 10)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


# =============================================================================
# Mock chessboard helpers (synthetic frames when --mock-camera is active)
# =============================================================================
def _make_mock_board(cols: int, rows: int, sq_px: int = 46) -> Any:
    """Build a flat BGR chessboard image with cols×rows inner corners."""
    border = sq_px
    w = (cols + 1) * sq_px + 2 * border
    h = (rows + 1) * sq_px + 2 * border
    img: Any = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y1 = r * sq_px + border
                x1 = c * sq_px + border
                img[y1 : y1 + sq_px, x1 : x1 + sq_px] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _animated_mock_frame(board: Any, t: float, tw: int, th: int) -> Any:
    """Render board with time-varying affine transform onto a gray canvas."""
    h, w = board.shape[:2]
    angle = math.sin(t) * 17.0
    scale = 0.83 + math.cos(t * 0.65) * 0.12
    dx    = math.sin(t * 0.55) * 88
    dy    = math.cos(t * 0.80) * 52
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += (tw - w) / 2.0 + dx
    M[1, 2] += (th - h) / 2.0 + dy
    canvas = np.full((th, tw, 3), 112, dtype=np.uint8)
    cv2.warpAffine(board, M, (tw, th), dst=canvas,
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return canvas


# =============================================================================
# RecordWorker — saves detected frames to disk on a background thread
# =============================================================================
class RecordWorker(QObject):
    """Saves calibration board frames to a session folder while recording."""

    stats_update       = pyqtSignal(int, float, float)  # (saved, sharpness, elapsed_s)
    recording_finished = pyqtSignal(str, int)            # (save_dir, n_saved)

    def __init__(
        self,
        cols:         int,
        rows:         int,
        sq_mm:        float,
        pattern_type: int,
        save_dir:     Path,
        parent:       Any = None,
    ) -> None:
        super().__init__(parent)
        self._cols         = cols
        self._rows         = rows
        self._sq_mm        = sq_mm
        self._pattern_type = pattern_type
        self._save_dir     = Path(save_dir)

        self._running:          bool        = False
        self._n_saved:          int         = 0
        self._frame_idx:        int         = 0
        self._last_sharpness:   float       = 0.0
        self._last_save_t:      float       = -1e9
        self._recent_centroids: list[tuple] = []
        self._start_time:       float       = 0.0
        self._last_stats_t:     float       = 0.0

    @pyqtSlot()
    def start_recording(self) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._n_saved          = 0
        self._frame_idx        = 0
        self._last_sharpness   = 0.0
        self._last_save_t      = -1e9
        self._recent_centroids.clear()
        self._start_time   = time.monotonic()
        self._last_stats_t = 0.0
        self._running = True

    @pyqtSlot()
    def stop_recording(self) -> None:
        self._running = False
        self.recording_finished.emit(str(self._save_dir), self._n_saved)

    @pyqtSlot(object, float)
    def on_frame(self, frame: Any, _t_mono: float) -> None:
        if not self._running or not _HAVE_CV2 or not _HAVE_NP:
            return

        self._frame_idx += 1
        if self._frame_idx % 2 != 0:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = self._detect(gray)
        now = time.monotonic()

        if not ok or corners is None:
            self._last_sharpness = 0.0
            self._maybe_emit_stats(now)
            return

        if self._pattern_type == 0:  # chessboard: refine sub-pixel
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)

        sharpness = compute_sharpness(gray, corners)
        self._last_sharpness = sharpness

        if sharpness < MIN_SHARPNESS:
            self._maybe_emit_stats(now)
            return

        cx_f = float(corners[:, 0, 0].mean())
        cy_f = float(corners[:, 0, 1].mean())
        centroid = (cx_f, cy_f)

        # Time-based dedup
        if (now - self._last_save_t) * 1000 < DEDUP_TIME_MS:
            self._maybe_emit_stats(now)
            return

        # Centroid-based dedup (last 3 saved frames)
        for prev_c in self._recent_centroids:
            if math.hypot(centroid[0] - prev_c[0], centroid[1] - prev_c[1]) < DEDUP_CENTROID_PX:
                self._maybe_emit_stats(now)
                return

        # Write to disk
        fname = self._save_dir / f"frame_{self._n_saved:04d}.png"
        try:
            cv2.imwrite(str(fname), frame)
        except Exception:
            self._maybe_emit_stats(now)
            return

        self._n_saved += 1
        self._last_save_t = now
        self._recent_centroids.append(centroid)
        if len(self._recent_centroids) > 3:
            self._recent_centroids.pop(0)

        self._maybe_emit_stats(now)

    def _detect(self, gray: Any) -> tuple:
        cols, rows = self._cols, self._rows
        pt = self._pattern_type
        if pt == 0:
            ok, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        elif pt == 1:
            ok, corners = cv2.findCirclesGrid(gray, (cols, rows), None)
        else:
            ok, corners = cv2.findCirclesGrid(
                gray, (cols, rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID
            )
        return bool(ok), corners if ok else None

    def _maybe_emit_stats(self, now: float) -> None:
        if now - self._last_stats_t >= 0.2:
            self.stats_update.emit(
                self._n_saved,
                self._last_sharpness,
                now - self._start_time,
            )
            self._last_stats_t = now


# =============================================================================
# ImageLoadThread — loads images from selected folders and detects corners
# =============================================================================
class ImageLoadThread(QThread):
    """Loads images from folders and detects calibration corners in background."""

    progress = pyqtSignal(int, int)   # (processed, total)
    finished = pyqtSignal(list)       # list[FrameRecord]
    failed   = pyqtSignal(str)

    def __init__(
        self,
        folder_paths: list,
        cols:         int,
        rows:         int,
        sq_mm:        float,
        pattern_type: int,
        parent:       Any = None,
    ) -> None:
        super().__init__(parent)
        self._folders      = list(folder_paths)
        self._cols         = cols
        self._rows         = rows
        self._sq_mm        = sq_mm
        self._pattern_type = pattern_type

    def run(self) -> None:  # type: ignore[override]
        if not _HAVE_CV2 or not _HAVE_NP:
            self.failed.emit("OpenCV / NumPy not available.")
            return

        # Collect all image files across all folders
        all_paths: list[Path] = []
        for folder in self._folders:
            p = Path(folder)
            if not p.is_dir():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                all_paths.extend(sorted(p.glob(ext)))

        # Deduplicate by resolved path
        seen: set[str] = set()
        unique_paths: list[Path] = []
        for p in all_paths:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)
        all_paths = unique_paths

        if not all_paths:
            self.failed.emit(
                "No image files (PNG/JPG) found in the selected folders.\n"
                "Record a session first or add a folder that contains images."
            )
            return

        records: list[FrameRecord] = []
        total = len(all_paths)

        for i, img_path in enumerate(all_paths):
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, corners = self._detect(gray)
                if not ok or corners is None:
                    continue
                if self._pattern_type == 0:
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
                sharpness = compute_sharpness(gray, corners)
                cx = float(corners[:, 0, 0].mean())
                cy = float(corners[:, 0, 1].mean())
                records.append(FrameRecord(
                    frame=frame,
                    corners=corners,
                    sharpness=sharpness,
                    centroid=(cx, cy),
                    timestamp_monotonic=0.0,
                ))
            except Exception:
                pass
            self.progress.emit(i + 1, total)

        self.finished.emit(records)

    def _detect(self, gray: Any) -> tuple:
        cols, rows = self._cols, self._rows
        pt = self._pattern_type
        if pt == 0:
            ok, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        elif pt == 1:
            ok, corners = cv2.findCirclesGrid(gray, (cols, rows), None)
        else:
            ok, corners = cv2.findCirclesGrid(
                gray, (cols, rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID
            )
        return bool(ok), corners if ok else None


# =============================================================================
# CalibrationThread — final cv2.fisheye.calibrate off the GUI thread
# =============================================================================
class CalibrationThread(QThread):
    progress  = pyqtSignal(int)
    succeeded = pyqtSignal(object, object, float, object)
    failed    = pyqtSignal(str)

    def __init__(
        self, obj_pts: list, img_pts: list, image_size: tuple,
        flags: int, parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._obj_pts    = obj_pts
        self._img_pts    = img_pts
        self._image_size = image_size
        self._flags      = flags

    def run(self) -> None:  # type: ignore[override]
        try:
            K = np.zeros((3, 3), dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            self.progress.emit(5)
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                self._obj_pts, self._img_pts, self._image_size,
                K, D, flags=self._flags, criteria=crit,
            )
            self.progress.emit(80)
            per_img: list[float] = []
            for rvec, tvec, ipts, opts in zip(rvecs, tvecs, self._img_pts, self._obj_pts):
                proj, _ = cv2.fisheye.projectPoints(opts, rvec, tvec, K, D)
                err = float(np.sqrt(np.mean(np.sum((ipts - proj) ** 2, axis=2))))
                per_img.append(err)
            self.progress.emit(100)
            self.succeeded.emit(K, D, float(rms), per_img)
        except cv2.error as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Calibration failed: {exc}")


# =============================================================================
# SectionFrame — card-style section matching gantry_panel visual style
# =============================================================================
class SectionFrame(QWidget):
    def __init__(self, title: str, parent: Any = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #4ea1ff; padding: 0 4px;"
        )
        self._card = QFrame()
        self._card.setObjectName("SectionCard")
        outer.addWidget(self._title_lbl)
        outer.addWidget(self._card)

    def content(self) -> QWidget:
        return self._card

    def set_title(self, title: str) -> None:
        self._title_lbl.setText(title)


# =============================================================================
# ThumbLabel — read-only gallery thumbnail (calibrated frames panel)
# =============================================================================
class ThumbLabel(QLabel):
    """Read-only thumbnail in the calibrated-frames gallery."""

    def __init__(self, pixmap: QPixmap, parent: Any = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(124, 88)
        self.setPixmap(
            pixmap.scaled(120, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.setStyleSheet(
            "border: 2px solid #3a3a40; border-radius: 4px; background: #1f1f22;"
        )

    def highlight(self, on: bool = True) -> None:
        col = "#4ea1ff" if on else "#3a3a40"
        self.setStyleSheet(
            f"border: 2px solid {col}; border-radius: 4px; background: #1f1f22;"
        )


# =============================================================================
# MatrixEditorDialog — 4×4 T_gantry_camera editor
# =============================================================================
class MatrixEditorDialog(QDialog):
    def __init__(self, matrix: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit T_gantry_camera (4×4)")
        self.setModal(True)
        v = QVBoxLayout(self)

        note = QLabel(
            "Rotation + translation: upper-left 3×3 should be a rotation matrix.\n"
            "Translation column units = same as SLAM output (metres).\n"
            "Last row is fixed [0, 0, 0, 1]."
        )
        note.setStyleSheet("color: #aaa; font-size: 11px;")
        v.addWidget(note)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._spins: list[list[QDoubleSpinBox]] = []
        labels = [
            "r00", "r01", "r02", "tx",
            "r10", "r11", "r12", "ty",
            "r20", "r21", "r22", "tz",
            "0",   "0",   "0",   "1",
        ]
        k = 0
        for r in range(4):
            row_spins: list[QDoubleSpinBox] = []
            for c in range(4):
                spin = QDoubleSpinBox()
                spin.setRange(-1e6, 1e6)
                spin.setDecimals(6)
                spin.setSingleStep(0.001)
                spin.setValue(float(matrix[r, c]))
                spin.setMinimumWidth(90)
                if r == 3:
                    spin.setReadOnly(True)
                    spin.setStyleSheet("background-color: #111114; color: #666;")
                spin.setToolTip(labels[k])
                k += 1
                grid.addWidget(spin, r, c)
                row_spins.append(spin)
            self._spins.append(row_spins)
        v.addLayout(grid)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _on_accept(self) -> None:
        M  = self.get_matrix()
        R  = M[:3, :3]
        ed = float(np.max(np.abs(R @ R.T - np.eye(3))))
        dd = abs(float(np.linalg.det(R)) - 1.0)
        if ed > 0.05 or dd > 0.05:
            ans = QMessageBox.warning(
                self, "Non-orthogonal rotation",
                f"Upper-left 3×3 does not look like a rotation matrix\n"
                f"(|R·Rᵀ − I|∞ = {ed:.4f}, |det(R)−1| = {dd:.4f}).\n\n"
                "Accept anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        self.accept()

    def get_matrix(self) -> Any:
        M = np.zeros((4, 4))
        for r in range(4):
            for c in range(4):
                M[r, c] = self._spins[r][c].value()
        M[3] = [0.0, 0.0, 0.0, 1.0]
        return M


# =============================================================================
# Main window
# =============================================================================
class CalibrateWindow(QMainWindow):
    """Single-window fisheye calibration tool (disk-record + folder-select)."""

    _frame_processed = pyqtSignal(object, float)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args
        self.setWindowTitle("UMI Fisheye Calibration")
        self.resize(1400, 900)
        self.setMinimumSize(1280, 800)

        # Camera session
        if _HAVE_CAM_SESSION:
            self._cam_session: Any = FisheyeCameraSession(parent=self)
            self._cam_session.frame_ready.connect(self._on_frame)
            self._cam_session.state_changed.connect(self._on_cam_state)
            self._cam_session.error.connect(self._on_cam_error)
        else:
            self._cam_session = None

        # Frame display state
        self._latest_frame:  Any   = None
        self._last_render_t: float = 0.0
        self._frame_count:   int   = 0
        self._detected:      bool  = False
        self._current_corners: Any = None
        self._preview_fps:   float = 0.0
        self._fps_count:     int   = 0
        self._fps_t0:        float = time.monotonic()

        # Mock board (lazily built when --mock-camera active)
        self._mock_board: Any = None

        # Recording state
        self._state:               CalibState          = CalibState.IDLE
        self._record_thread:       QThread | None      = None
        self._record_worker:       RecordWorker | None = None
        self._frame_to_worker:     bool                = False
        self._record_start_time:   float               = 0.0
        self._n_recorded:          int                 = 0
        self._current_session_dir: Path | None         = None

        # Folder list state
        self._folder_paths: list[str] = []

        # Image loading thread
        self._load_thread: ImageLoadThread | None = None

        # Calibration result
        self._calib_K:             Any              = None
        self._calib_D:             Any              = None
        self._calib_rms:           float | None     = None
        self._calib_per_image:     list[float]      = []
        self._calib_image_size:    tuple | None     = None
        self._calib_thread:        CalibrationThread | None = None
        self._final_calib_records: list[FrameRecord] = []
        self._final_calib_retry:   int               = 0

        # Gallery thumbnails
        self._picked_thumb_widgets: list[ThumbLabel] = []

        # T_gantry_camera (4×4, default identity)
        self._T: Any = np.eye(4) if _HAVE_NP else None

        # Undistortion maps
        self._undist_map1: Any  = None
        self._undist_map2: Any  = None
        self._show_undist: bool = False

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        root.addWidget(self._build_connection_bar())

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(8)
        self._splitter.addWidget(self._build_left_pane())
        self._splitter.addWidget(self._build_right_pane())
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)
        self._splitter.setSizes([840, 560])
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        root.addWidget(self._splitter, stretch=1)

        self._build_status_bar()
        self._setup_shortcuts()
        self._restore_settings()
        self._transition(CalibState.IDLE)

    # =========================================================================
    # UI builders
    # =========================================================================

    def _build_connection_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("SectionCard")
        bar.setStyleSheet("QFrame#SectionCard { padding: 6px 12px; }")
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        cam_lbl = QLabel("Camera")
        cam_lbl.setStyleSheet("font-weight: 600; color: #aaa; min-width: 52px;")
        h.addWidget(cam_lbl)
        h.addWidget(QLabel("Device"))

        self._dev_spin = QSpinBox()
        self._dev_spin.setRange(0, 20)
        self._dev_spin.setValue(0)
        self._dev_spin.setMinimumWidth(55)
        self._dev_spin.setMinimumHeight(24)
        h.addWidget(self._dev_spin)

        self._res_combo = QComboBox()
        for r in ["1280×720", "1920×1080", "640×480"]:
            self._res_combo.addItem(r)
        self._res_combo.setMinimumHeight(24)
        h.addWidget(self._res_combo)

        h.addWidget(QLabel("FPS"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(30)
        self._fps_spin.setMinimumWidth(55)
        self._fps_spin.setMinimumHeight(24)
        h.addWidget(self._fps_spin)
        h.addStretch()

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("PrimaryBtn")
        self._connect_btn.setMinimumHeight(28)
        self._connect_btn.setMinimumWidth(100)
        self._connect_btn.clicked.connect(self._toggle_camera)
        if not _HAVE_CAM_SESSION:
            self._connect_btn.setEnabled(False)
            self._connect_btn.setToolTip("fisheye_camera module not found.")
        h.addWidget(self._connect_btn)

        self._cam_status_lbl = QLabel("● Disconnected")
        self._cam_status_lbl.setStyleSheet("color: #ef5350; font-weight: bold;")
        h.addWidget(self._cam_status_lbl)
        return bar

    def _build_left_pane(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(420)
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Live Preview
        preview_sec = SectionFrame("Live Preview")
        pv = QVBoxLayout(preview_sec.content())
        pv.setContentsMargins(6, 6, 6, 6)
        pv.setSpacing(4)

        self._preview = QLabel("Camera disconnected")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(280)
        self._preview.setStyleSheet(
            "background-color: #0e0e0e; color: #888; border-radius: 4px;"
        )
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pv.addWidget(self._preview, stretch=1)

        det_row = QHBoxLayout()
        self._detect_lbl = QLabel("No pattern")
        self._detect_lbl.setStyleSheet("color: #888; font-size: 12px;")
        det_row.addWidget(self._detect_lbl)
        det_row.addStretch()
        self._undist_btn = QPushButton("▶ Undistortion preview")
        self._undist_btn.setCheckable(True)
        self._undist_btn.setEnabled(False)
        self._undist_btn.setMinimumHeight(26)
        self._undist_btn.clicked.connect(self._toggle_undist)
        det_row.addWidget(self._undist_btn)
        pv.addLayout(det_row)
        v.addWidget(preview_sec, stretch=1)

        # Calibrated frames gallery (hidden until calibration done)
        self._gallery_sec = SectionFrame("Calibrated Frames (0)")
        gv = QVBoxLayout(self._gallery_sec.content())
        gv.setContentsMargins(4, 4, 4, 4)
        gv.setSpacing(2)

        gallery_scroll = QScrollArea()
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll.setFrameShape(QFrame.NoFrame)
        gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gallery_scroll.setFixedHeight(104)

        self._gallery_inner  = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_inner)
        self._gallery_layout.setContentsMargins(2, 2, 2, 2)
        self._gallery_layout.setSpacing(4)
        self._gallery_layout.addStretch()
        gallery_scroll.setWidget(self._gallery_inner)
        gv.addWidget(gallery_scroll)

        self._gallery_sec.setVisible(False)
        v.addWidget(self._gallery_sec)
        return w

    def _build_right_pane(self) -> QWidget:
        outer = QWidget()
        outer.setMinimumWidth(340)
        outer_v = QVBoxLayout(outer)
        outer_v.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(4, 0, 4, 0)
        v.setSpacing(8)
        v.addWidget(self._build_pattern_section())
        v.addWidget(self._build_record_section())
        v.addWidget(self._build_folder_section())
        v.addWidget(self._build_results_section())
        v.addWidget(self._build_save_section())
        v.addStretch()

        scroll.setWidget(inner)
        outer_v.addWidget(scroll)
        return outer

    def _build_pattern_section(self) -> SectionFrame:
        sec = SectionFrame("Pattern")
        v = QVBoxLayout(sec.content())
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems(
            ["Chessboard", "Symmetric circles", "Asymmetric circles"]
        )
        self._pattern_combo.setMinimumHeight(24)
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern_changed)
        type_row.addWidget(self._pattern_combo, stretch=1)
        v.addLayout(type_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Inner corners:"))
        size_row.addWidget(QLabel("cols"))
        tip = "Number of *inner* corners — for a 10×7 squares board, enter 9×6."
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 30)
        self._cols_spin.setValue(9)
        self._cols_spin.setMinimumHeight(24)
        self._cols_spin.setToolTip(tip)
        self._cols_spin.valueChanged.connect(self._on_pattern_changed)
        size_row.addWidget(self._cols_spin)
        size_row.addWidget(QLabel("rows"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 30)
        self._rows_spin.setValue(6)
        self._rows_spin.setMinimumHeight(24)
        self._rows_spin.setToolTip(tip)
        self._rows_spin.valueChanged.connect(self._on_pattern_changed)
        size_row.addWidget(self._rows_spin)
        size_row.addStretch()
        v.addLayout(size_row)

        sq_row = QHBoxLayout()
        sq_row.addWidget(QLabel("Square size (mm):"))
        self._sq_spin = QDoubleSpinBox()
        self._sq_spin.setRange(0.1, 1000.0)
        self._sq_spin.setValue(25.0)
        self._sq_spin.setDecimals(2)
        self._sq_spin.setMinimumHeight(24)
        sq_row.addWidget(self._sq_spin)
        sq_row.addStretch()
        v.addLayout(sq_row)

        flag_lbl = QLabel("Calibration flags:")
        flag_lbl.setStyleSheet("color: #aaa; font-size: 12px; margin-top: 4px;")
        v.addWidget(flag_lbl)
        self._flag_recompute  = QCheckBox("Recompute extrinsic  (CALIB_RECOMPUTE_EXTRINSIC)")
        self._flag_check_cond = QCheckBox("Check condition  (CALIB_CHECK_COND)")
        self._flag_fix_skew   = QCheckBox("Fix skew  (CALIB_FIX_SKEW)")
        self._flag_recompute.setChecked(True)
        self._flag_check_cond.setChecked(True)
        self._flag_fix_skew.setChecked(True)
        for chk in (self._flag_recompute, self._flag_check_cond, self._flag_fix_skew):
            v.addWidget(chk)

        return sec

    def _build_record_section(self) -> SectionFrame:
        self._record_sec = SectionFrame("Record")
        v = QVBoxLayout(self._record_sec.content())
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("State:"))
        self._state_lbl = QLabel("IDLE")
        self._state_lbl.setStyleSheet("font-weight: 600; color: #888;")
        state_row.addWidget(self._state_lbl)
        state_row.addStretch()
        v.addLayout(state_row)

        self._record_btn = QPushButton("● Record  [R]")
        self._record_btn.setObjectName("RecordBtn")
        self._record_btn.setMinimumHeight(38)
        self._record_btn.setEnabled(False)
        self._record_btn.clicked.connect(self._on_record_btn_clicked)
        v.addWidget(self._record_btn)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(4)

        stats_grid.addWidget(QLabel("Frames saved:"), 0, 0)
        self._buffered_lbl = QLabel("0")
        self._buffered_lbl.setStyleSheet("font-weight: 600;")
        stats_grid.addWidget(self._buffered_lbl, 0, 1)

        stats_grid.addWidget(QLabel("Recording time:"), 1, 0)
        self._rec_time_lbl = QLabel("00:00")
        stats_grid.addWidget(self._rec_time_lbl, 1, 1)

        stats_grid.addWidget(QLabel("Live sharpness:"), 2, 0)
        self._sharp_lbl = QLabel("—")
        stats_grid.addWidget(self._sharp_lbl, 2, 1)

        stats_grid.addWidget(QLabel("Save folder:"), 3, 0)
        self._session_folder_lbl = QLabel("—")
        self._session_folder_lbl.setStyleSheet("color: #4ea1ff; font-size: 10px;")
        self._session_folder_lbl.setWordWrap(True)
        stats_grid.addWidget(self._session_folder_lbl, 3, 1)

        v.addLayout(stats_grid)
        return self._record_sec

    def _build_folder_section(self) -> SectionFrame:
        self._folder_sec = SectionFrame("Calibration Images")
        v = QVBoxLayout(self._folder_sec.content())
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        hint = QLabel(
            "Session folders are added here automatically after recording.\n"
            "You can also add past session folders manually."
        )
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self._folder_list = QListWidget()
        self._folder_list.setMaximumHeight(150)
        self._folder_list.setToolTip("Folders whose images will be used for calibration.")
        v.addWidget(self._folder_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._add_folder_btn = QPushButton("Add Folder…")
        self._add_folder_btn.setMinimumHeight(26)
        self._add_folder_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(self._add_folder_btn)

        self._remove_folder_btn = QPushButton("Remove")
        self._remove_folder_btn.setMinimumHeight(26)
        self._remove_folder_btn.setEnabled(False)
        self._remove_folder_btn.clicked.connect(self._remove_selected_folder)
        btn_row.addWidget(self._remove_folder_btn)

        btn_row.addStretch()

        self._total_images_lbl = QLabel("0 images")
        self._total_images_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        btn_row.addWidget(self._total_images_lbl)

        v.addLayout(btn_row)

        self._calib_btn = QPushButton("Calibrate")
        self._calib_btn.setObjectName("PrimaryBtn")
        self._calib_btn.setMinimumHeight(38)
        self._calib_btn.setEnabled(False)
        self._calib_btn.clicked.connect(self._start_calibration_from_folders)
        v.addWidget(self._calib_btn)

        self._folder_list.currentRowChanged.connect(self._on_folder_selection_changed)

        return self._folder_sec

    def _build_results_section(self) -> SectionFrame:
        sec = SectionFrame("Results")
        v = QVBoxLayout(sec.content())
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        self._rms_lbl = QLabel("RMS: —")
        rms_font = QFont()
        rms_font.setPointSize(14)
        rms_font.setBold(True)
        self._rms_lbl.setFont(rms_font)
        v.addWidget(self._rms_lbl)

        mono = QFont("DejaVu Sans Mono, Courier New, monospace")
        mono.setPointSize(10)
        self._K_lbl = QLabel("K: —")
        self._K_lbl.setFont(mono)
        self._K_lbl.setStyleSheet("color: #c0c0c0;")
        v.addWidget(self._K_lbl)

        self._D_lbl = QLabel("D: —")
        self._D_lbl.setFont(mono)
        self._D_lbl.setStyleSheet("color: #c0c0c0;")
        v.addWidget(self._D_lbl)

        sep = QLabel("Per-image reprojection errors  (click row to highlight):")
        sep.setStyleSheet("color: #888; font-size: 11px; margin-top: 4px;")
        v.addWidget(sep)

        self._per_image_list = QListWidget()
        self._per_image_list.setMaximumHeight(140)
        self._per_image_list.currentRowChanged.connect(self._on_result_row_changed)
        v.addWidget(self._per_image_list)

        self._undist_btn2 = QPushButton("▶ Undistortion preview")
        self._undist_btn2.setCheckable(True)
        self._undist_btn2.setEnabled(False)
        self._undist_btn2.setMinimumHeight(26)
        self._undist_btn2.clicked.connect(self._toggle_undist)
        v.addWidget(self._undist_btn2)

        return sec

    def _build_save_section(self) -> SectionFrame:
        sec = SectionFrame("Save YAML")
        v = QVBoxLayout(sec.content())
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Output path:"))
        self._out_path = QLineEdit(_DEFAULT_OUTPUT)
        self._out_path.setMinimumHeight(24)
        path_row.addWidget(self._out_path, stretch=1)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.setMinimumHeight(24)
        browse_btn.clicked.connect(self._browse_output)
        path_row.addWidget(browse_btn)
        v.addLayout(path_row)

        t_lbl = QLabel("T_gantry_camera:")
        t_lbl.setStyleSheet("color: #aaa; font-size: 12px; margin-top: 4px;")
        v.addWidget(t_lbl)

        self._t_identity_rb = QRadioButton("Identity (default — measure + edit later)")
        self._t_identity_rb.setChecked(True)
        self._t_load_rb = QRadioButton("Load from YAML…")
        self._t_edit_rb = QRadioButton("Edit 4×4 manually…")
        v.addWidget(self._t_identity_rb)
        v.addWidget(self._t_load_rb)
        v.addWidget(self._t_edit_rb)
        self._t_load_rb.clicked.connect(self._load_T_from_yaml)
        self._t_edit_rb.clicked.connect(self._edit_T_manually)

        self._save_btn = QPushButton("Save YAML  [Ctrl+S]")
        self._save_btn.setObjectName("PrimaryBtn")
        self._save_btn.setMinimumHeight(34)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_yaml)
        v.addWidget(self._save_btn)

        return sec

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self._sb_conn  = QLabel("● Disconnected")
        self._sb_conn.setStyleSheet("color: #ef5350;")
        self._sb_mid   = QLabel("Preview: 0 fps")
        self._sb_calib = QLabel("Not calibrated")
        sb.addWidget(self._sb_conn)
        sep = QLabel("  │  ")
        sep.setStyleSheet("color: #555;")
        sb.addWidget(sep)
        sb.addWidget(self._sb_mid, 1)
        sb.addPermanentWidget(self._sb_calib)

    # =========================================================================
    # State machine
    # =========================================================================

    def _transition(self, new_state: CalibState) -> None:
        self._state = new_state
        colors = {
            CalibState.IDLE:        "#888",
            CalibState.RECORDING:   "#4caf50",
            CalibState.LOADING:     "#ffd740",
            CalibState.CALIBRATING: "#ffd740",
            CalibState.DONE:        "#4caf50",
            CalibState.ERROR:       "#ef5350",
        }
        label_text = {
            CalibState.IDLE:        "IDLE",
            CalibState.RECORDING:   "RECORDING",
            CalibState.LOADING:     "LOADING…",
            CalibState.CALIBRATING: "CALIBRATING…",
            CalibState.DONE:        "DONE",
            CalibState.ERROR:       "ERROR",
        }
        col = colors.get(new_state, "#888")
        self._state_lbl.setText(label_text.get(new_state, str(new_state)))
        self._state_lbl.setStyleSheet(f"font-weight: 600; color: {col};")

        is_connected = (
            self._cam_session is not None and self._cam_session.is_open
        )
        if new_state == CalibState.IDLE:
            self._record_btn.setText("● Record  [R]")
            self._record_btn.setObjectName("RecordBtn")
            self._record_btn.setEnabled(is_connected)
            self._record_btn.style().unpolish(self._record_btn)
            self._record_btn.style().polish(self._record_btn)
        elif new_state == CalibState.RECORDING:
            self._record_btn.setText("■ Stop  [R]")
            self._record_btn.setObjectName("StopBtn")
            self._record_btn.setEnabled(True)
            self._record_btn.style().unpolish(self._record_btn)
            self._record_btn.style().polish(self._record_btn)
        else:
            self._record_btn.setEnabled(False)

        if new_state in (CalibState.IDLE, CalibState.DONE, CalibState.ERROR):
            self._update_calib_btn()
        else:
            self._calib_btn.setEnabled(False)

        self._update_status_bar()

    # =========================================================================
    # Camera connection
    # =========================================================================

    def _toggle_camera(self) -> None:
        if self._cam_session is None:
            QMessageBox.critical(self, "Missing dependency",
                                 "fisheye_camera module not found in sys.path.")
            return
        if self._cam_session.is_open:
            self._cam_session.close()
            self._connect_btn.setText("Connect")
        else:
            dev, w, h = self._parse_cam_settings()
            fps = self._fps_spin.value()
            self._cam_session.open(
                device=dev, width=w, height=h, fps=fps,
                mock=bool(getattr(self._args, "mock_camera", False)),
            )
            self._connect_btn.setText("Disconnect")

    def _parse_cam_settings(self) -> tuple:
        dev = self._dev_spin.value()
        res = self._res_combo.currentText().replace("×", "x")
        try:
            w_s, h_s = res.split("x")
            return dev, int(w_s), int(h_s)
        except ValueError:
            return dev, 1280, 720

    def _on_cam_state(self, state: str) -> None:
        _colors = {
            "disconnected":   "#ef5350",
            "connecting":     "#ffd740",
            "connected":      "#4caf50",
            "connected_mock": "#4caf50",
            "error":          "#ef5350",
        }
        _labels = {
            "disconnected":   "● Disconnected",
            "connecting":     "● Connecting…",
            "connected":      "● Connected",
            "connected_mock": "● Connected (mock)",
            "error":          "● Error",
        }
        color = _colors.get(state, "#e6e6e6")
        self._cam_status_lbl.setText(_labels.get(state, state))
        self._cam_status_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._sb_conn.setText(_labels.get(state, state))
        self._sb_conn.setStyleSheet(f"color: {color};")

        is_connected = state in ("connected", "connected_mock")
        if self._state == CalibState.IDLE:
            self._record_btn.setEnabled(is_connected)
        if is_connected:
            self._preview.setStyleSheet(
                "background-color: #0e0e0e; border-radius: 4px;"
            )
        elif state == "disconnected":
            self._preview.setText("Camera disconnected")
            self._preview.setStyleSheet(
                "background-color: #0e0e0e; color: #888; border-radius: 4px;"
            )
            self._detect_lbl.setText("No pattern")
            self._detect_lbl.setStyleSheet("color: #888; font-size: 12px;")
            self._detected = False
            if self._state == CalibState.RECORDING:
                self._stop_recording()

    def _on_cam_error(self, msg: str) -> None:
        self._cam_status_lbl.setText("● Error")
        self._cam_status_lbl.setStyleSheet("color: #ef5350; font-weight: bold;")
        self._preview.setText(f"Camera error:\n{msg}")
        self._preview.setStyleSheet(
            "background-color: #0e0e0e; color: #ef5350; border-radius: 4px;"
        )

    # =========================================================================
    # Frame handling
    # =========================================================================

    def _on_frame(self, frame: Any, t_mono: float) -> None:
        if getattr(self._args, "mock_camera", False) and _HAVE_CV2 and _HAVE_NP:
            frame = self._get_mock_frame(frame)
        self._latest_frame = frame

        now = time.monotonic()
        self._fps_count += 1
        if now - self._fps_t0 >= 1.0:
            self._preview_fps = self._fps_count / (now - self._fps_t0)
            self._fps_count   = 0
            self._fps_t0      = now
            self._update_status_bar()

        self._frame_count += 1
        if self._frame_count % 2 == 0 and _HAVE_CV2:
            self._detect_corners(frame)

        if now - self._last_render_t >= 1.0 / _DISPLAY_FPS:
            self._last_render_t = now
            self._render_frame(frame)

        self._frame_processed.emit(frame, t_mono)

    def _get_mock_frame(self, blank: Any) -> Any:
        h, w = blank.shape[:2]
        if self._mock_board is None:
            self._mock_board = _make_mock_board(
                self._cols_spin.value(), self._rows_spin.value()
            )
        return _animated_mock_frame(self._mock_board, time.monotonic() * 0.38, w, h)

    def _on_pattern_changed(self) -> None:
        self._mock_board = None

    def _detect_corners(self, frame: Any) -> None:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cols  = self._cols_spin.value()
        rows  = self._rows_spin.value()
        ptype = self._pattern_combo.currentIndex()

        if ptype == 0:
            ok, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ok and corners is not None:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        elif ptype == 1:
            ok, corners = cv2.findCirclesGrid(gray, (cols, rows), None)
        else:
            ok, corners = cv2.findCirclesGrid(
                gray, (cols, rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID
            )

        self._detected        = bool(ok)
        self._current_corners = corners if ok else None

        if ok:
            self._detect_lbl.setText("Detected ✓")
            self._detect_lbl.setStyleSheet(
                "color: #4caf50; font-size: 12px; font-weight: 600;"
            )
        else:
            self._detect_lbl.setText("No pattern")
            self._detect_lbl.setStyleSheet("color: #888; font-size: 12px;")

    def _render_frame(self, frame: Any) -> None:
        if not _HAVE_CV2:
            return
        try:
            display = frame.copy()
            if self._detected and self._current_corners is not None:
                cols, rows = self._cols_spin.value(), self._rows_spin.value()
                if self._pattern_combo.currentIndex() == 0:
                    cv2.drawChessboardCorners(
                        display, (cols, rows), self._current_corners, True
                    )
                cv2.putText(
                    display, "Detected ✓",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (50, 220, 80), 2, cv2.LINE_AA,
                )

            if self._show_undist and self._undist_map1 is not None:
                undist = cv2.remap(
                    display, self._undist_map1, self._undist_map2, cv2.INTER_LINEAR
                )
                h, w = display.shape[:2]
                uh, uw = undist.shape[:2]
                if uh != h or uw != w:
                    undist = cv2.resize(undist, (w, h))
                display = np.hstack([display, undist])

            h_px, w_px = display.shape[:2]
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            qimg  = QImage(rgb.data, w_px, h_px, w_px * 3, QImage.Format_RGB888)
            pix   = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self._preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._preview.setPixmap(scaled)
        except Exception:
            pass

    # =========================================================================
    # Recording
    # =========================================================================

    def _on_record_btn_clicked(self) -> None:
        if self._state == CalibState.IDLE:
            self._start_recording()
        elif self._state == CalibState.RECORDING:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self._cam_session is None or not self._cam_session.is_open:
            self._toast("Connect camera first.")
            return

        self._cleanup_record_thread()

        session_dir = _SESSION_ROOT / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._current_session_dir = session_dir
        self._session_folder_lbl.setText(session_dir.name)
        self._buffered_lbl.setText("0")

        self._record_worker = RecordWorker(
            cols=self._cols_spin.value(),
            rows=self._rows_spin.value(),
            sq_mm=float(self._sq_spin.value()),
            pattern_type=self._pattern_combo.currentIndex(),
            save_dir=session_dir,
        )
        self._record_thread = QThread(self)
        self._record_worker.moveToThread(self._record_thread)

        self._frame_processed.connect(self._record_worker.on_frame)
        self._frame_to_worker = True
        self._record_worker.stats_update.connect(self._on_record_stats)
        self._record_worker.recording_finished.connect(self._on_recording_stopped)

        self._record_thread.start()
        QMetaObject.invokeMethod(
            self._record_worker, "start_recording", Qt.QueuedConnection
        )

        self._record_start_time = time.monotonic()
        self._transition(CalibState.RECORDING)

    def _stop_recording(self) -> None:
        if self._record_worker is None:
            self._transition(CalibState.IDLE)
            return
        QMetaObject.invokeMethod(
            self._record_worker, "stop_recording", Qt.QueuedConnection
        )

    def _on_record_stats(
        self, saved: int, sharpness: float, elapsed_s: float
    ) -> None:
        self._buffered_lbl.setText(str(saved))

        mins = int(elapsed_s) // 60
        secs = int(elapsed_s) % 60
        self._rec_time_lbl.setText(f"{mins:02d}:{secs:02d}")

        if sharpness > 0:
            if sharpness > 150:
                col = "#4caf50"
            elif sharpness > 50:
                col = "#ffd740"
            else:
                col = "#ef5350"
            self._sharp_lbl.setText(f"{sharpness:.1f}")
            self._sharp_lbl.setStyleSheet(f"color: {col}; font-weight: 600;")
        else:
            self._sharp_lbl.setText("—")
            self._sharp_lbl.setStyleSheet("color: #888;")

    @pyqtSlot(str, int)
    def _on_recording_stopped(self, save_dir: str, n_saved: int) -> None:
        if self._frame_to_worker and self._record_worker is not None:
            try:
                self._frame_processed.disconnect(self._record_worker.on_frame)
            except (TypeError, RuntimeError):
                pass
            self._frame_to_worker = False

        self._cleanup_record_thread()
        self._n_recorded = n_saved

        if n_saved == 0:
            self._toast("No frames saved — keep the board visible while recording.")
            self._transition(CalibState.IDLE)
            return

        # Auto-add new session folder to folder list
        if save_dir not in self._folder_paths:
            self._folder_paths.append(save_dir)
            count = self._count_images_in_folder(save_dir)
            item = QListWidgetItem(f"{Path(save_dir).name}  ({count} images)")
            item.setToolTip(save_dir)
            self._folder_list.addItem(item)

        self._toast(f"Saved {n_saved} frames → {Path(save_dir).name}")
        self._update_calib_btn()
        self._transition(CalibState.IDLE)

    def _cleanup_record_thread(self) -> None:
        if self._record_thread is not None and self._record_thread.isRunning():
            self._record_thread.quit()
            self._record_thread.wait(2000)
        self._record_thread = None
        self._record_worker = None

    def _reset_record_ui(self) -> None:
        self._buffered_lbl.setText("0")
        self._rec_time_lbl.setText("00:00")
        self._sharp_lbl.setText("—")
        self._sharp_lbl.setStyleSheet("color: #888;")

    # =========================================================================
    # Folder management
    # =========================================================================

    def _add_folder(self) -> None:
        start = (
            str(_SESSION_ROOT) if _SESSION_ROOT.exists()
            else str(Path.home())
        )
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing calibration images", start
        )
        if not path:
            return
        if path in self._folder_paths:
            self._toast(f"Already in list: {Path(path).name}")
            return
        self._folder_paths.append(path)
        count = self._count_images_in_folder(path)
        item = QListWidgetItem(f"{Path(path).name}  ({count} images)")
        item.setToolTip(path)
        self._folder_list.addItem(item)
        self._update_calib_btn()

    def _remove_selected_folder(self) -> None:
        row = self._folder_list.currentRow()
        if row < 0:
            return
        self._folder_list.takeItem(row)
        del self._folder_paths[row]
        self._remove_folder_btn.setEnabled(self._folder_list.count() > 0)
        self._update_calib_btn()

    def _on_folder_selection_changed(self, row: int) -> None:
        self._remove_folder_btn.setEnabled(row >= 0)

    def _count_images_in_folder(self, path: str) -> int:
        p = Path(path)
        if not p.is_dir():
            return 0
        count = 0
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            count += len(list(p.glob(ext)))
        return count

    def _update_calib_btn(self) -> None:
        total = sum(self._count_images_in_folder(p) for p in self._folder_paths)
        self._total_images_lbl.setText(f"{total} images")
        can_calib = (
            total >= 12 and
            self._state in (CalibState.IDLE, CalibState.DONE, CalibState.ERROR)
        )
        self._calib_btn.setEnabled(can_calib)
        if total >= 12:
            self._calib_btn.setText(f"Calibrate  ({total} images)")
        elif total > 0:
            self._calib_btn.setText(f"Calibrate  (need ≥ 12)")
        else:
            self._calib_btn.setText("Calibrate")

    # =========================================================================
    # Calibration from folders
    # =========================================================================

    def _start_calibration_from_folders(self) -> None:
        if not self._folder_paths:
            self._toast("Add folders first.")
            return

        # Clear previous results
        self._gallery_sec.setVisible(False)
        self._calib_K         = None
        self._calib_D         = None
        self._calib_rms       = None
        self._calib_per_image = []
        self._save_btn.setEnabled(False)
        self._undist_btn.setEnabled(False)
        self._undist_btn2.setEnabled(False)
        self._show_undist = False
        self._undist_map1 = None
        self._per_image_list.clear()
        self._rms_lbl.setText("RMS: —")
        self._K_lbl.setText("K: —")
        self._D_lbl.setText("D: —")

        self._transition(CalibState.LOADING)

        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait(1000)

        self._load_thread = ImageLoadThread(
            folder_paths=list(self._folder_paths),
            cols=self._cols_spin.value(),
            rows=self._rows_spin.value(),
            sq_mm=float(self._sq_spin.value()),
            pattern_type=self._pattern_combo.currentIndex(),
            parent=self,
        )
        self._load_thread.progress.connect(self._on_load_progress)
        self._load_thread.finished.connect(self._on_load_finished)
        self._load_thread.failed.connect(self._on_load_failed)
        self._load_thread.start()

    def _on_load_progress(self, loaded: int, total: int) -> None:
        self._state_lbl.setText(f"LOADING {loaded}/{total}…")
        self._sb_mid.setText(f"Loading images: {loaded} / {total}")

    def _on_load_finished(self, records: list) -> None:
        n = len(records)
        if n < 12:
            total_imgs = sum(
                self._count_images_in_folder(p) for p in self._folder_paths
            )
            QMessageBox.warning(
                self, "Too few valid frames",
                f"Only {n} of {total_imgs} images contained a detectable "
                f"calibration pattern.\n\n"
                "Need at least 12 frames. Check:\n"
                "• Pattern type and size settings match the recorded board.\n"
                "• Images are not too blurry or cut off.",
            )
            self._transition(CalibState.IDLE)
            return

        self._n_recorded          = n
        self._final_calib_records = list(records)
        self._show_picked_gallery(records)
        self._start_final_calibration(records)

    def _on_load_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Load failed", msg)
        self._transition(CalibState.IDLE)

    # =========================================================================
    # Final calibration (with CALIB_CHECK_COND retry)
    # =========================================================================

    def _start_final_calibration(self, records: list) -> None:
        self._final_calib_records = list(records)
        self._final_calib_retry   = 0
        self._transition(CalibState.CALIBRATING)
        self._do_calib_attempt()

    def _do_calib_attempt(self) -> None:
        if self._calib_thread is not None and self._calib_thread.isRunning():
            return
        records = self._final_calib_records
        obj_template = self._build_object_points()
        obj_pts  = [obj_template] * len(records)
        img_pts  = [r.corners.astype(np.float32) for r in records]
        h0, w0   = records[0].frame.shape[:2]

        self._calib_thread = CalibrationThread(
            obj_pts, img_pts, (w0, h0), self._get_flags(), parent=self
        )
        self._calib_thread.succeeded.connect(self._on_calib_succeeded)
        self._calib_thread.failed.connect(self._on_calib_failed_retry)
        self._calib_thread.start()

    def _on_calib_failed_retry(self, msg: str) -> None:
        m = re.search(r"\b(\d+)\b", msg)
        if m and self._final_calib_retry < 5:
            drop = int(m.group(1))
            if 0 <= drop < len(self._final_calib_records):
                self._final_calib_records.pop(drop)
                if len(self._final_calib_records) >= 12:
                    self._final_calib_retry += 1
                    self._toast(
                        f"CALIB_CHECK_COND retry {self._final_calib_retry}: dropped frame {drop}"
                    )
                    self._do_calib_attempt()
                    return
        self._on_calib_final_error(msg)

    def _on_calib_final_error(self, msg: str) -> None:
        self._transition(CalibState.ERROR)
        frame_hint = ""
        m = re.search(r"\b(\d+)\b", msg)
        if m:
            frame_hint = (
                f"\n\nPossibly related to frame index ~{m.group(1)}. "
                "Try removing problematic images from the folder."
            )
        QMessageBox.critical(
            self, "Calibration failed",
            f"{msg[:600]}{frame_hint}\n\n"
            "Tips:\n"
            "• Uncheck CALIB_CHECK_COND and try again.\n"
            "• Ensure the board fills > 30% of the image in some frames.\n"
            "• Try better lighting to improve sharpness.",
        )
        self._transition(CalibState.IDLE)

    def _on_calib_succeeded(self, K: Any, D: Any, rms: float, per_img: list) -> None:
        self._calib_K         = K
        self._calib_D         = D
        self._calib_rms       = rms
        self._calib_per_image = list(per_img)
        h0, w0                = self._final_calib_records[0].frame.shape[:2]
        self._calib_image_size = (w0, h0)

        try:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, self._calib_image_size, np.eye(3), balance=0.0
            )
            self._undist_map1, self._undist_map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, self._calib_image_size, cv2.CV_16SC2
            )
        except Exception:
            self._undist_map1 = None

        self._update_results_display()
        self._undist_btn.setEnabled(True)
        self._undist_btn2.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._transition(CalibState.DONE)

    # =========================================================================
    # Calibrated frames gallery
    # =========================================================================

    def _show_picked_gallery(self, records: list) -> None:
        for w in self._picked_thumb_widgets:
            self._gallery_layout.removeWidget(w)
            w.deleteLater()
        self._picked_thumb_widgets.clear()

        if not _HAVE_CV2:
            return

        n = len(records)
        self._gallery_sec.set_title(f"Calibrated Frames ({n})")
        self._gallery_sec.setVisible(True)

        for rec in records:
            try:
                h, w  = rec.frame.shape[:2]
                rgb   = cv2.cvtColor(rec.frame, cv2.COLOR_BGR2RGB)
                qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                pix   = QPixmap.fromImage(qimg)
                lbl   = ThumbLabel(pix, self._gallery_inner)
                pos   = max(0, self._gallery_layout.count() - 1)
                self._gallery_layout.insertWidget(pos, lbl)
                self._picked_thumb_widgets.append(lbl)
            except Exception:
                pass

    # =========================================================================
    # Results display
    # =========================================================================

    def _update_results_display(self) -> None:
        K, D, rms = self._calib_K, self._calib_D, self._calib_rms
        if K is None or rms is None:
            return

        if rms < 0.5:
            rms_color, tip = "#4caf50", ""
        elif rms < 1.0:
            rms_color, tip = "#ffd740", ""
        else:
            rms_color = "#ef5350"
            tip = (
                "High reprojection error — check pattern dimensions, "
                "coverage, or record again."
            )
        self._rms_lbl.setText(f"RMS: {rms:.4f} px")
        self._rms_lbl.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {rms_color};"
        )
        self._rms_lbl.setToolTip(tip)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        self._K_lbl.setText(f"fx={fx:.1f}  fy={fy:.1f}\ncx={cx:.1f}  cy={cy:.1f}")
        k1, k2, k3, k4 = D.flatten().tolist()
        self._D_lbl.setText(
            f"D=[{k1:.4f}, {k2:.4f},\n   {k3:.4f}, {k4:.4f}]"
        )

        self._per_image_list.clear()
        for i, err in enumerate(self._calib_per_image):
            c = "#4caf50" if err < 0.5 else "#ffd740" if err < 1.0 else "#ef5350"
            item = QListWidgetItem(f"  Image {i + 1:2d}:  {err:.4f} px")
            item.setForeground(QColor(c))
            self._per_image_list.addItem(item)

    def _on_result_row_changed(self, row: int) -> None:
        for thumb in self._picked_thumb_widgets:
            thumb.highlight(False)
        if 0 <= row < len(self._picked_thumb_widgets):
            self._picked_thumb_widgets[row].highlight(True)

    def _toggle_undist(self) -> None:
        on = bool(self.sender().isChecked())  # type: ignore[union-attr]
        self._show_undist = on
        for btn in (self._undist_btn, self._undist_btn2):
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)

    # =========================================================================
    # Object points & flags
    # =========================================================================

    def _build_object_points(self) -> Any:
        cols = self._cols_spin.value()
        rows = self._rows_spin.value()
        sq   = float(self._sq_spin.value())
        obj  = np.zeros((cols * rows, 1, 3), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                obj[r * cols + c, 0] = [c * sq, r * sq, 0.0]
        return obj

    def _get_flags(self) -> int:
        flags = 0
        if self._flag_recompute.isChecked():
            flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        if self._flag_check_cond.isChecked():
            flags |= cv2.fisheye.CALIB_CHECK_COND
        if self._flag_fix_skew.isChecked():
            flags |= cv2.fisheye.CALIB_FIX_SKEW
        return flags

    # =========================================================================
    # Save YAML
    # =========================================================================

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save calibration YAML", self._out_path.text(),
            "YAML (*.yaml *.yml)",
        )
        if path:
            self._out_path.setText(path)

    def _load_T_from_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load T_gantry_camera from YAML", "", "YAML (*.yaml *.yml)"
        )
        if not path:
            self._t_identity_rb.setChecked(True)
            return
        if not _HAVE_YAML or not _HAVE_NP:
            QMessageBox.warning(self, "Missing deps", "yaml and numpy required.")
            self._t_identity_rb.setChecked(True)
            return
        try:
            with open(path, "r") as fh:
                data = _yaml.safe_load(fh) or {}
            self._T = np.asarray(data["T_gantry_camera"], dtype=np.float64).reshape(4, 4)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            self._t_identity_rb.setChecked(True)

    def _edit_T_manually(self) -> None:
        if not _HAVE_NP:
            QMessageBox.warning(self, "Missing dep", "numpy required.")
            self._t_identity_rb.setChecked(True)
            return
        dlg = MatrixEditorDialog(self._T, self)
        if dlg.exec_() == QDialog.Accepted:
            self._T = dlg.get_matrix()

    def _get_T_for_save(self) -> Any:
        return np.eye(4) if self._t_identity_rb.isChecked() else self._T

    def _save_yaml(self) -> None:
        if not _HAVE_YAML:
            QMessageBox.critical(self, "Missing dep",
                                 "pyyaml required.  pip install pyyaml")
            return
        if not _HAVE_NP:
            QMessageBox.critical(self, "Missing dep", "numpy required.")
            return
        if self._calib_K is None:
            self._toast("Run calibration first.")
            return

        out_path = Path(self._out_path.text().strip() or _DEFAULT_OUTPUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        K = self._calib_K
        D = self._calib_D.flatten()
        T = self._get_T_for_save()
        w, h = self._calib_image_size  # type: ignore[misc]

        pnames = ["chessboard", "symmetric_circles", "asymmetric_circles"]
        flag_names: list[str] = []
        if self._flag_recompute.isChecked():
            flag_names.append("RECOMPUTE_EXTRINSIC")
        if self._flag_check_cond.isChecked():
            flag_names.append("CHECK_COND")
        if self._flag_fix_skew.isChecked():
            flag_names.append("FIX_SKEW")

        doc = {
            "K":              K.tolist(),
            "D":              D.tolist(),
            "image_size":     [w, h],
            "T_gantry_camera": T.tolist(),
            "metadata": {
                "created_at":      datetime.now().isoformat(timespec="seconds"),
                "rms_error_px":    round(float(self._calib_rms), 6),  # type: ignore
                "workflow":        "disk_record_folder_select_v4",
                "session_folders": list(self._folder_paths),
                "frames_used":     len(self._final_calib_records),
                "pattern":         (
                    f"{pnames[self._pattern_combo.currentIndex()]} "
                    f"{self._cols_spin.value()}x{self._rows_spin.value()}"
                ),
                "square_size_mm":  float(self._sq_spin.value()),
                "flags":           flag_names,
                "tool_version":    "calibrate_fisheye.py 4.0",
            },
        }

        try:
            with out_path.open("w") as fh:
                _yaml.dump(doc, fh, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        # Round-trip verification
        try:
            from fisheye_gantry_tagslam import load_fisheye_calibration
            calib = load_fisheye_calibration(out_path)
            assert calib.K.shape == (3, 3)
            assert calib.D.shape == (4, 1)
            assert len(calib.image_size) == 2
        except SystemExit as exc:
            QMessageBox.critical(
                self, "Verification failed",
                f"File written but round-trip check failed:\n{exc}",
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "Verification failed",
                f"File written but round-trip check raised:\n{exc}",
            )
            return

        QMessageBox.information(
            self, "Saved ✓",
            f"Calibration saved and verified:\n{out_path.resolve()}\n\n"
            f"RMS: {self._calib_rms:.4f} px  |  "
            f"{len(self._final_calib_records)} frames used",
        )
        self._update_status_bar()

    # =========================================================================
    # Status bar, shortcuts, settings
    # =========================================================================

    def _update_status_bar(self) -> None:
        fps = f"{self._preview_fps:.1f}" if self._preview_fps > 0 else "0"
        det = "Detected ✓" if self._detected else "No pattern"

        if self._state == CalibState.RECORDING:
            elapsed = time.monotonic() - self._record_start_time
            m = int(elapsed) // 60
            s = int(elapsed) % 60
            mid = f"RECORDING ({m:02d}:{s:02d}) • {fps} fps • {det}"
        elif self._state == CalibState.LOADING:
            mid = f"Preview: {fps} fps • LOADING images…"
        elif self._state == CalibState.CALIBRATING:
            mid = f"Preview: {fps} fps • CALIBRATING…"
        else:
            mid = f"Preview: {fps} fps  •  {det}"

        self._sb_mid.setText(mid)

        if self._calib_rms is not None:
            self._sb_calib.setText(f"RMS: {self._calib_rms:.4f} px")

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("R"),      self).activated.connect(self._on_record_btn_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_yaml)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.close)

    def _toast(self, msg: str, ms: int = 3000) -> None:
        self.statusBar().showMessage(msg, ms)

    def _on_splitter_moved(self, _pos: int, _idx: int) -> None:
        self._save_session_settings()

    def _restore_settings(self) -> None:
        s = _load_settings()
        cam = s.get("camera", {})
        if cam.get("device") is not None:
            self._dev_spin.setValue(int(cam["device"]))
        if cam.get("resolution"):
            idx = self._res_combo.findText(cam["resolution"])
            if idx >= 0:
                self._res_combo.setCurrentIndex(idx)
        if cam.get("fps"):
            self._fps_spin.setValue(int(cam["fps"]))
        pat = s.get("pattern", {})
        if pat.get("type") is not None:
            self._pattern_combo.setCurrentIndex(int(pat["type"]))
        if pat.get("cols"):
            self._cols_spin.setValue(int(pat["cols"]))
        if pat.get("rows"):
            self._rows_spin.setValue(int(pat["rows"]))
        if pat.get("square_mm"):
            self._sq_spin.setValue(float(pat["square_mm"]))
        if s.get("output_path"):
            self._out_path.setText(s["output_path"])
        if s.get("splitter_sizes"):
            sizes = s["splitter_sizes"]
            if isinstance(sizes, list) and len(sizes) == 2:
                self._splitter.setSizes([int(x) for x in sizes])
        # CLI overrides
        if getattr(self._args, "device", None) is not None:
            self._dev_spin.setValue(int(self._args.device))
        if getattr(self._args, "output", None) is not None:
            self._out_path.setText(str(self._args.output))

    def _save_session_settings(self) -> None:
        _save_settings({
            "camera": {
                "device":     self._dev_spin.value(),
                "resolution": self._res_combo.currentText(),
                "fps":        self._fps_spin.value(),
            },
            "pattern": {
                "type":      self._pattern_combo.currentIndex(),
                "cols":      self._cols_spin.value(),
                "rows":      self._rows_spin.value(),
                "square_mm": self._sq_spin.value(),
            },
            "output_path":    self._out_path.text(),
            "splitter_sizes": self._splitter.sizes(),
        })

    def closeEvent(self, ev: Any) -> None:  # type: ignore[override]
        self._save_session_settings()
        if self._state == CalibState.RECORDING:
            self._record_worker and QMetaObject.invokeMethod(
                self._record_worker, "stop_recording", Qt.DirectConnection
            )
        self._cleanup_record_thread()
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait(2000)
        if self._calib_thread is not None and self._calib_thread.isRunning():
            self._calib_thread.wait(2000)
        if self._cam_session is not None and self._cam_session.is_open:
            self._cam_session.close()
        super().closeEvent(ev)


# =============================================================================
# Entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fisheye camera intrinsic calibration GUI.")
    p.add_argument("--device", type=int, default=None,
                   help="Camera device index (overrides saved setting).")
    p.add_argument("--mock-camera", action="store_true",
                   help="Use mock camera with animated chessboard frames.")
    p.add_argument("--output", type=Path, default=None,
                   help=f"Output YAML path (default: {_DEFAULT_OUTPUT}).")
    p.add_argument("--light", action="store_true", help="Skip dark theme.")
    return p.parse_args()


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    args = _parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    if not args.light:
        if _HAVE_DARK:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5"))
        else:
            app.setStyleSheet(_QSS)

    missing = []
    if not _HAVE_CV2:
        missing.append("opencv-python")
    if not _HAVE_NP:
        missing.append("numpy")
    if missing:
        QMessageBox.critical(
            None, "Missing dependencies",
            "Required packages not installed:\n  pip install " + " ".join(missing),
        )
        return
    if not _HAVE_YAML:
        QMessageBox.warning(
            None, "Missing pyyaml",
            "pyyaml not installed — YAML save will fail.\n  pip install pyyaml",
        )

    win = CalibrateWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
