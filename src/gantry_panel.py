#!/usr/bin/env python3
"""
gantry_panel.py — PyQt5 live control panel for the FMC4030 gantry.

Run:
    python src/gantry_panel.py            # real controller
    python src/gantry_panel.py --mock     # in-process mock for smoke tests

================================================================================
Frame / sign / unit conventions
================================================================================
* The panel speaks MILLIMETERS to the user everywhere except the Homing group,
  whose Speed/Acc/Fall-step fields use raw controller UNITS because the SDK's
  ``home_axis()`` takes units directly. The unit difference is mirrored in the
  field labels and tooltips so the user never has to guess.
* mm <-> units conversion uses ``gantry_runner.SCALE_MM_PER_UNIT`` (X=8.25,
  Y=2.5, Z=0.5 mm/unit), copied verbatim from
  ``src/gantry/demos/whisker_dragging.py``. Same values, same axis ordering.
* This module makes NO Z-axis-sign assumption. If your rig has +Z down vs +Z
  up, downstream T_gantry_camera (in the fisheye calibration) absorbs the flip.
* All motion commands and the GantryTelemetryLogger come from gantry_runner.py;
  this file is purely the UI + threading layer. The CLI in gantry_runner.py is
  untouched and remains authoritative.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

# sys.path shim: import sibling modules from src/ regardless of cwd.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _resolve_calib_path(text: str) -> "Path | None":
    """Resolve a calibration-path string to a Path.

    Absolute paths are used as-is. Relative paths resolve against the repo root
    (parent of src/), so the default 'config/fisheye_calibration.yaml' works
    whether the panel is launched from src/, the repo root, or elsewhere.
    """
    text = (text or "").strip()
    if not text:
        return None
    p = Path(text)
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p)

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt5.QtCore import (  # noqa: E402
    QObject, Qt, QThread, QTimer, QUrl, pyqtSignal,
)
from PyQt5.QtGui import (  # noqa: E402
    QColor, QDesktopServices, QFont, QKeySequence,
)
from PyQt5.QtWidgets import (  # noqa: E402
    QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QStatusBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

# Optional dependencies — degrade gracefully if missing.
try:
    import pyqtgraph as pg
    HAVE_PYQTGRAPH = True
except ImportError:
    pg = None  # type: ignore[assignment]
    HAVE_PYQTGRAPH = False

try:
    import qdarkstyle
    HAVE_QDARKSTYLE = True
except ImportError:
    qdarkstyle = None  # type: ignore[assignment]
    HAVE_QDARKSTYLE = False

try:
    import qtawesome as qta
    HAVE_QTAWESOME = True
except ImportError:
    qta = None  # type: ignore[assignment]
    HAVE_QTAWESOME = False

from gantry import (  # noqa: E402
    Axis, ControllerConfig, DeviceParameters, FMC4030Controller, FMC4030Error,
)
from gantry_runner import (  # noqa: E402
    AXES, AXIS_NAMES, EMERGENCY_STOP, SCALE_MM_PER_UNIT,
    GantryTelemetryLogger, Waypoint,
    _read_current_pos_mm,
    make_gantry_run_dir, mm_to_units, move_to_xyz_mm, units_to_mm,
)

# ExperimentRunner — optional; only used when Experiment tab is active.
try:
    from experiment_runner import ExperimentConfig, ExperimentRunner, Phase
    HAVE_EXPERIMENT_RUNNER = True
except ImportError:
    HAVE_EXPERIMENT_RUNNER = False

# FisheyeCameraSession — optional; enables persistent camera connection.
try:
    from fisheye_camera import FisheyeCameraSession
    HAVE_FISHEYE_CAMERA = True
except ImportError:
    FisheyeCameraSession = None  # type: ignore[assignment,misc]
    HAVE_FISHEYE_CAMERA = False


# =============================================================================
# Constants
# =============================================================================
HOME_SPEED_LIMIT_UNITS = 20.0       # hard upper bound on home speed (units/s)
STATUS_POLL_MS = 100                # 10 Hz live readout

# --- Unit conversion helpers (centralized — single source of truth) ---
# SCALE_MM_PER_UNIT imported from gantry_runner (mm per controller unit, per axis).

def cm_s_to_units_s(cm_per_s: float, axis: "Axis") -> float:
    """cm/s → controller units/s for `axis`."""
    return (cm_per_s * 10.0) / SCALE_MM_PER_UNIT[axis]

def units_s_to_cm_s(units_per_s: float, axis: "Axis") -> float:
    """controller units/s → cm/s for `axis`."""
    return (units_per_s * SCALE_MM_PER_UNIT[axis]) / 10.0

def cm_s2_to_units_s2(cm_per_s2: float, axis: "Axis") -> float:
    """cm/s² → controller units/s² for `axis`."""
    return (cm_per_s2 * 10.0) / SCALE_MM_PER_UNIT[axis]

def units_s2_to_cm_s2(units_per_s2: float, axis: "Axis") -> float:
    """controller units/s² → cm/s² for `axis`."""
    return (units_per_s2 * SCALE_MM_PER_UNIT[axis]) / 10.0
# (mm_to_units / units_to_mm already imported from gantry_runner)
PROGRESS_NEAR_LIMIT_PCT = 10.0      # within X% of either soft limit -> yellow
LIVE_PLOT_WINDOW_S = 30.0
FINITE_DIFF_WINDOW = 5              # 5-sample SMA central diff for live accel
POLL_INDICATOR_STALE_MS = 1000      # readout label turns red beyond this age

# Verbose status-poll diagnostics (one-per-second to stderr) — toggle to True
# for an interactive debug session; default False keeps logs quiet.
DEBUG_STATUS_POLL = False

WINDOW_TITLE = "UMI Gantry Control Panel"
DEFAULT_HOME_ORDER = ("Z", "X", "Y")


# =============================================================================
# Stylesheet (used when qdarkstyle is unavailable)
# =============================================================================
FALLBACK_DARK_QSS = """
/* ---- base canvas ---- */
QMainWindow, QWidget { background-color: #1a1a1d; color: #e6e6e6; }
QScrollArea, QScrollArea > QWidget > QWidget { background-color: #1a1a1d; border: 0; }
QLabel { color: #e6e6e6; }
QSplitter::handle { background-color: #3a3a42; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 10px; border-top: 1px solid #505058; border-bottom: 1px solid #505058; }
QSplitter::handle:hover { background-color: #4ea1ff; }

/* ---- card-style sections (label+frame replaces QGroupBox — no title clipping) ---- */
QFrame#SectionCard {
    background-color: #2b2b2b;
    border: 1px solid #3f3f46;
    border-radius: 10px;
    padding: 12px;
}

/* ---- legacy QGroupBox (kept for any external widgets; not used in panel sections) ---- */
QGroupBox {
    background-color: #2b2b2b;
    border: 1px solid #3f3f46;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 28px;
    font-weight: 600;
    font-size: 14px;
    color: #e6e6e6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    top: -10px;
    left: 12px;
    padding: 0 8px;
    background-color: #2b2b2b;
    color: #4ea1ff;
    font-size: 14px;
    font-weight: 600;
}

/* ---- tabs ---- */
QTabWidget::pane {
    border: 1px solid #3f3f46;
    border-radius: 8px;
    background-color: #232327;
    top: -1px;
}
QTabBar::tab {
    background: #2a2a2e;
    color: #c0c0c0;
    padding: 8px 18px;
    border: 1px solid #3a3a40;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #1a73e8;
    color: white;
    font-weight: 600;
}
QTabBar::tab:hover:!selected { background: #38383d; }

/* ---- per-axis cards (interactive jog/move/home + live readout) ---- */
QFrame#AxisCard {
    background-color: #1f1f22;
    border: 1px solid #3a3a40;
    border-radius: 12px;
    padding: 10px;
}
QFrame#AxisCard:hover {
    border: 1px solid #4ea1ff;
}

/* ---- buttons ---- */
QPushButton {
    background-color: #3a3a40;
    border: 1px solid #4a4a52;
    border-radius: 6px;
    padding: 6px 12px;
    color: #e6e6e6;
}
QPushButton:hover { background-color: #45454c; border-color: #4ea1ff; }
QPushButton:pressed { background-color: #2e2e34; }
QPushButton:disabled { background-color: #2a2a2e; color: #666; border-color: #333; }
QPushButton#PrimaryButton {
    background-color: #1a73e8;
    border: 1px solid #1a73e8;
    color: white;
    font-weight: 600;
    font-size: 14px;
}
QPushButton#PrimaryButton:hover { background-color: #2589ff; }
QPushButton#EmergencyButton {
    background-color: #d93025;
    border: 2px solid #ff5147;
    color: white;
    font-weight: 700;
    font-size: 13px;
    padding: 4px 10px;
}
QPushButton#IconButton { text-align: left; padding-left: 32px; }
QPushButton#EmergencyButton:hover { background-color: #ff3b30; }
QPushButton#record:checked {
    background-color: #d93025; color: white; border-color: #ff5147;
}

/* ---- inputs ---- */
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background-color: #1f1f22;
    border: 1px solid #3a3a40;
    border-radius: 5px;
    padding: 4px 6px;
    color: #e6e6e6;
    selection-background-color: #4ea1ff;
    min-height: 22px;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {
    border: 1px solid #4ea1ff;
}

/* ---- position readouts (the big green-on-black numbers) ---- */
QLabel#PositionReadout {
    font-family: "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 22px;
    color: #34d058;
    background-color: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 8px 10px;
    qproperty-alignment: AlignRight;
}
QLabel#UnitsHint { color: #8a8a8a; font-size: 10px; }
QLabel#axis-letter { font-size: 28px; font-weight: 700; color: #4ea1ff; }

/* ---- progress, table, menu, status bar, tooltip ---- */
QProgressBar {
    border: 1px solid #3a3a40; border-radius: 5px; background-color: #131316;
    text-align: center; min-height: 18px; color: #e6e6e6;
}
QProgressBar::chunk { background-color: #1a73e8; border-radius: 4px; }
QTableWidget {
    background-color: #1f1f22; alternate-background-color: #25252a;
    gridline-color: #2f2f33; selection-background-color: #1a73e8; color: #e6e6e6;
    border: 1px solid #3a3a40; border-radius: 6px;
}
QHeaderView::section {
    background-color: #2b2b2b; color: #e6e6e6; padding: 6px;
    border: 0; border-right: 1px solid #1a1a1d;
}
QMenuBar { background-color: #1a1a1d; color: #e6e6e6; }
QMenuBar::item:selected { background-color: #1a73e8; }
QMenu { background-color: #2b2b2b; color: #e6e6e6; border: 1px solid #3f3f46; }
QMenu::item:selected { background-color: #1a73e8; }
QStatusBar { background-color: #1a1a1d; color: #c0c0c0; border-top: 1px solid #2f2f33; }
QToolTip { background-color: #1f1f22; color: #e6e6e6; border: 1px solid #4ea1ff; padding: 4px; }
"""


def _icon(name: str) -> QtGui.QIcon | None:
    """Return a qtawesome icon if installed, else None."""
    if HAVE_QTAWESOME:
        try:
            return qta.icon(name, color="#e8e8e8")
        except Exception:
            return None
    return None


# =============================================================================
# Mock SDK (for --mock smoke tests)
# =============================================================================
class _MockArr(list):
    """List subclass that mimics the c_float*3 / c_int*3 layouts used by the
    real ctypes Structures (indexable + assignable)."""


class MockMachineStatus:
    def __init__(self) -> None:
        self.realPos = _MockArr([0.0, 0.0, 0.0])
        self.realSpeed = _MockArr([0.0, 0.0, 0.0])
        self.inputStatus = 0
        self.outputStatus = 0
        self.limitNStatus = 0
        self.limitPStatus = 0
        self.machineRunStatus = 0
        self.axisStatus = _MockArr([0, 0, 0])
        self.homeStatus = 0


class MockDeviceParameters:
    """Mirrors gantry.DeviceParameters but with mutable per-axis lists."""

    def __init__(self) -> None:
        self.id = 1
        self.bound232 = 115200
        self.bound485 = 115200
        self.ip = "192.168.0.30"
        self.port = 8088
        self.div = _MockArr([10000, 10000, 10000])
        self.lead = _MockArr([1, 1, 1])
        # ~2.4 m on X (8.25*300), ~1 m on Y (2.5*400), ~0.1 m on Z (0.5*200)
        self.soft_limit_min = _MockArr([-1500, -800, -100])
        self.soft_limit_max = _MockArr([1500, 800, 100])
        self.home_time = _MockArr([10, 10, 10])


class MockVersionInfo:
    firmware = 9999
    library = 9999
    serial = 12345


class MockFMC4030Controller:
    """In-process simulation. Each axis moves linearly at the commanded speed
    toward its target; ``get_status`` interpolates based on monotonic time so
    a polling GUI sees realistic position/velocity traces.

    Implements every method the real controller exposes that the panel or
    gantry_runner.py touches; everything else is a no-op or returns a stub.
    """

    def __init__(self) -> None:
        self._connected = False
        # Per-axis state.
        self._axes: dict[int, dict[str, Any]] = {
            i: {
                "pos": 0.0, "target": 0.0,
                "speed": 0.0, "direction": 0,
                "last_update": time.monotonic(),
                "stopped": True,
            }
            for i in range(3)
        }
        self._params = MockDeviceParameters()
        self._version = MockVersionInfo()

    def _advance(self) -> None:
        now = time.monotonic()
        for state in self._axes.values():
            dt = now - state["last_update"]
            state["last_update"] = now
            if state["stopped"] or state["direction"] == 0:
                continue
            step = state["speed"] * state["direction"] * dt
            new_pos = state["pos"] + step
            if (state["direction"] > 0 and new_pos >= state["target"]) or \
               (state["direction"] < 0 and new_pos <= state["target"]):
                state["pos"] = state["target"]
                state["stopped"] = True
                state["speed"] = 0.0
                state["direction"] = 0
            else:
                state["pos"] = new_pos

    # ---- connection
    def connect(self, config: ControllerConfig) -> None:
        time.sleep(0.05)  # tiny simulated handshake
        self._connected = True

    def close(self) -> None:
        self._connected = False

    # ---- status reads
    def get_status(self) -> MockMachineStatus:
        self._advance()
        s = MockMachineStatus()
        for i, state in self._axes.items():
            s.realPos[i] = float(state["pos"])
            s.realSpeed[i] = float(state["speed"] * state["direction"])
            s.axisStatus[i] = 0 if state["stopped"] else 1
        return s

    def get_axis_position(self, axis: Axis) -> float:
        self._advance()
        return float(self._axes[int(axis)]["pos"])

    def get_axis_speed(self, axis: Axis) -> float:
        self._advance()
        s = self._axes[int(axis)]
        return float(s["speed"] * s["direction"])

    def is_axis_stopped(self, axis: Axis) -> bool:
        self._advance()
        return bool(self._axes[int(axis)]["stopped"])

    # ---- device parameters
    def get_device_parameters(self) -> Any:
        return deepcopy(self._params)

    def set_device_parameters(self, params: Any) -> None:
        self._params = deepcopy(params)

    def get_version_info(self) -> Any:
        return self._version

    # ---- motion
    def jog_single_axis(self, axis: Axis, position_units: float, speed_units: float,
                        acc_units: float, dec_units: float, *, relative: bool = False) -> None:
        self._advance()
        idx = int(axis)
        cur = self._axes[idx]["pos"]
        target = cur + position_units if relative else position_units
        direction = 1 if target > cur else -1 if target < cur else 0
        self._axes[idx].update({
            "target": target,
            "speed": max(abs(float(speed_units)), 0.001),
            "direction": direction,
            "stopped": direction == 0,
            "last_update": time.monotonic(),
        })

    def line_move_3d(self, axes, end_x: float, end_y: float, end_z: float,
                     speed: float, acc: float, dec: float) -> None:
        self._advance()
        for i, tgt in enumerate((end_x, end_y, end_z)):
            cur = self._axes[i]["pos"]
            direction = 1 if tgt > cur else -1 if tgt < cur else 0
            self._axes[i].update({
                "target": tgt,
                "speed": max(abs(float(speed)), 0.001),
                "direction": direction,
                "stopped": direction == 0,
                "last_update": time.monotonic(),
            })

    def line_move_2d(self, axes, end_x: float, end_y: float, speed, acc, dec) -> None:
        self.line_move_3d(axes, end_x, end_y, self._axes[2]["pos"], speed, acc, dec)

    def stop_axis(self, axis: Axis, mode: int = 2) -> None:
        self._advance()
        s = self._axes[int(axis)]
        s["stopped"] = True
        s["speed"] = 0.0
        s["direction"] = 0
        s["target"] = s["pos"]

    def stop_run(self) -> None:
        for i in range(3):
            self.stop_axis(Axis(i))

    def pause_run(self, mask: int = 0x07) -> None:
        pass

    def resume_run(self, mask: int = 0x07) -> None:
        pass

    def home_axis(self, axis: Axis, speed: float, acc_dec: float, fall_step: float,
                  *, positive_limit: bool = True) -> None:
        # Simulate moving to "home" = 0.
        self.jog_single_axis(axis, position_units=0.0, speed_units=speed,
                             acc_units=acc_dec, dec_units=acc_dec, relative=False)

    # ---- IO (unused by the panel but kept for compat)
    def set_output(self, channel: int, value: int) -> None:
        pass

    def get_input(self, channel: int) -> int:
        return 1


# =============================================================================
# Worker threads
# =============================================================================
class _PartialStatusShim:
    """Mimics the subset of MachineStatus the panel reads (``realPos[3]``,
    ``realSpeed[3]``) so the snapshot pipeline can use per-axis fallback
    reads when ``get_status()`` errors with FMC4030 code 664 ("machine
    status unavailable", usually because one or more axes are not enabled
    or not yet homed). Per-axis SDK calls (``Get_Axis_Current_Pos``,
    ``Get_Axis_Current_Speed``) are separate from ``Get_Machine_Status``
    and routinely succeed in the 664 state, which is exactly the strategy
    ``manual_pad.py`` uses (see gantry/demos/manual_pad.py:_handle_status_error).
    """

    def __init__(self, pos: list[float], spd: list[float]) -> None:
        self.realPos = list(pos)
        self.realSpeed = list(spd)


class StatusPollThread(QThread):
    """Single-shot per tick: try ``get_status()`` for one SDK round-trip; if
    that errors (the common case is code 664 when an axis is not enabled or
    not homed yet), fall back to per-axis ``get_axis_position`` /
    ``get_axis_speed`` so the live readout doesn't go silent."""

    snapshot_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    # Class-level rate-limit timestamps so DEBUG prints stay readable.
    _last_dbg_t: float = 0.0

    def __init__(self, controller, lock: threading.RLock, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._lock = lock

    def run(self) -> None:  # type: ignore[override]
        if DEBUG_STATUS_POLL:
            now = time.monotonic()
            if now - StatusPollThread._last_dbg_t >= 1.0:
                StatusPollThread._last_dbg_t = now
                print(f"[status-poll] tick t={now:.2f}", file=sys.stderr, flush=True)
        # --- Fast path: one SDK round-trip via get_status().
        get_status_error: str | None = None
        try:
            with self._lock:
                status = self._controller.get_status()
            self.snapshot_ready.emit(status)
            return
        except FMC4030Error as exc:
            get_status_error = str(exc)
        except Exception as exc:
            self.error_occurred.emit(f"Unexpected: {exc}")
            return

        # --- Per-axis fallback (typical when get_status returns 664).
        partial_pos: list[float] = [0.0, 0.0, 0.0]
        partial_spd: list[float] = [0.0, 0.0, 0.0]
        per_axis_errors: list[str] = []
        any_ok = False
        for i in range(3):
            try:
                with self._lock:
                    p = self._controller.get_axis_position(Axis(i))
                partial_pos[i] = float(p)
                try:
                    with self._lock:
                        s = self._controller.get_axis_speed(Axis(i))
                    partial_spd[i] = float(s)
                except Exception:
                    pass  # speed alone failing is fine — leave 0.
                any_ok = True
            except FMC4030Error as exc:
                per_axis_errors.append(f"{Axis(i).name}: {exc}")
            except Exception as exc:
                per_axis_errors.append(f"{Axis(i).name}: {exc}")

        if any_ok:
            # Surface the underlying status error once so the user knows the
            # fallback is engaged (the GUI also shows a 664-specific hint).
            self.error_occurred.emit(f"get_status() failed ({get_status_error}); "
                                     "using per-axis fallback")
            self.snapshot_ready.emit(_PartialStatusShim(partial_pos, partial_spd))
        else:
            joined = "; ".join(per_axis_errors) if per_axis_errors else "no per-axis data"
            self.error_occurred.emit(f"{get_status_error}  |  fallback also failed: {joined}")


class MoveToTargetThread(QThread):
    """Wraps gantry_runner.move_to_xyz_mm so the GUI thread stays responsive."""

    finished_with = pyqtSignal(str)  # "" on success, error message otherwise

    def __init__(self, controller, target_mm, speed_mm_s: float, acc_mm_s2: float,
                 dec_mm_s2: float, mode: str, lock: threading.RLock,
                 logger: GantryTelemetryLogger | None = None,
                 waypoint_index: int = 0) -> None:
        super().__init__()
        self._controller = controller
        self._target = tuple(target_mm)
        self._speed = float(speed_mm_s)
        self._acc = float(acc_mm_s2)
        self._dec = float(dec_mm_s2)
        self._mode = mode
        self._lock = lock
        self._logger = logger
        self._wp_idx = waypoint_index

    def run(self) -> None:  # type: ignore[override]
        try:
            move_to_xyz_mm(
                self._controller, self._target,
                self._speed, self._acc, self._dec,
                mode=self._mode, lock=self._lock,
                logger=self._logger, waypoint_index=self._wp_idx,
            )
            self.finished_with.emit("")
        except Exception as exc:
            self.finished_with.emit(str(exc))


class SequenceThread(QThread):
    """Runs a waypoint list top-to-bottom on one background thread."""

    row_started = pyqtSignal(int)
    row_done = pyqtSignal(int, str)
    sequence_done = pyqtSignal(str)

    def __init__(self, controller, waypoints, acc_mm_s2: float, dec_mm_s2: float,
                 mode: str, lock: threading.RLock,
                 logger: GantryTelemetryLogger | None = None,
                 abort_event: threading.Event | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._waypoints = list(waypoints)
        self._acc = float(acc_mm_s2)
        self._dec = float(dec_mm_s2)
        self._mode = mode
        self._lock = lock
        self._logger = logger
        self._abort_event = abort_event or threading.Event()

    def _aborted(self) -> bool:
        return EMERGENCY_STOP.is_set() or self._abort_event.is_set()

    def run(self) -> None:  # type: ignore[override]
        try:
            for i, wp in enumerate(self._waypoints):
                if self._aborted():
                    self.sequence_done.emit("Interrupted")
                    return
                self.row_started.emit(i)
                try:
                    move_to_xyz_mm(
                        self._controller,
                        (wp.x_mm, wp.y_mm, wp.z_mm),
                        wp.speed_mm_s, self._acc, self._dec,
                        mode=self._mode, lock=self._lock, logger=self._logger,
                        waypoint_index=i,
                    )
                except Exception as exc:
                    err = f"Row {i}: {exc}"
                    self.row_done.emit(i, err)
                    self.sequence_done.emit(err)
                    return
                self.row_done.emit(i, "")
                # Honor dwell, but check abort in small chunks so E-Stop
                # interrupts a long dwell quickly.
                t_end = time.monotonic() + max(0.0, wp.dwell_s)
                while time.monotonic() < t_end:
                    if self._aborted():
                        self.sequence_done.emit("Interrupted")
                        return
                    time.sleep(min(0.05, t_end - time.monotonic()))
            self.sequence_done.emit("")
        except Exception as exc:
            self.sequence_done.emit(f"Unexpected: {exc}")


class AxisAbsMoveThread(QThread):
    """Single-axis absolute move on a background QThread, used by the Per-Axis
    Control cards' Move Abs buttons. Issues ``jog_single_axis(..., relative=
    False)`` then polls ``is_axis_stopped`` until True (or EMERGENCY_STOP).
    """

    finished_with = pyqtSignal(str)  # "" on success, error message otherwise

    def __init__(self, controller, axis: Axis, target_units: float,
                 speed_units: float, acc_units: float, dec_units: float,
                 lock: threading.RLock,
                 abort_event: threading.Event | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._axis = axis
        self._target_units = float(target_units)
        self._speed = max(float(speed_units), 0.001)
        self._acc = max(float(acc_units), 0.001)
        self._dec = max(float(dec_units), 0.001)
        self._lock = lock
        self._abort_event = abort_event or threading.Event()

    def _aborted(self) -> bool:
        return EMERGENCY_STOP.is_set() or self._abort_event.is_set()

    def run(self) -> None:  # type: ignore[override]
        try:
            with self._lock:
                self._controller.jog_single_axis(
                    self._axis,
                    position_units=self._target_units,
                    speed_units=self._speed,
                    acc_units=self._acc,
                    dec_units=self._dec,
                    relative=False,
                )
            while not self._aborted():
                try:
                    with self._lock:
                        stopped = self._controller.is_axis_stopped(self._axis)
                except FMC4030Error as exc:
                    self.finished_with.emit(f"is_axis_stopped failed: {exc}")
                    return
                if stopped:
                    break
                time.sleep(0.05)
            if self._aborted():
                self.finished_with.emit("aborted")
                return
            self.finished_with.emit("")
        except FMC4030Error as exc:
            self.finished_with.emit(str(exc))
        except Exception as exc:
            self.finished_with.emit(f"Unexpected: {exc}")


class GoToHomeThread(QThread):
    """Drive each axis sequentially to its saved 'home reference' position
    using per-axis SDK calls (``jog_single_axis(..., relative=False)``).

    Per-axis SDK calls succeed even when ``get_status()`` returns code 664
    ('axis not enabled / not homed'), so this works without a physical
    limit-switch homing pass. Axes whose home reference is None are skipped.

    Emits ``progress(str)`` between axes and ``finished_with(str)`` at the
    end ("" on success, "aborted", or an error message).
    """

    progress      = pyqtSignal(str)
    finished_with = pyqtSignal(str)

    def __init__(self, controller,
                 targets_mm_abs: dict[Axis, float | None],
                 speed_units_per_s: dict[Axis, float],
                 acc_units_per_s2: dict[Axis, float],
                 dec_units_per_s2: dict[Axis, float],
                 lock: threading.RLock,
                 abort_event: threading.Event | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._targets = dict(targets_mm_abs)
        self._speed = dict(speed_units_per_s)
        self._acc = dict(acc_units_per_s2)
        self._dec = dict(dec_units_per_s2)
        self._lock = lock
        self._abort_event = abort_event or threading.Event()

    def request_abort(self) -> None:
        self._abort_event.set()

    def _aborted(self) -> bool:
        return EMERGENCY_STOP.is_set() or self._abort_event.is_set()

    def run(self) -> None:  # type: ignore[override]
        for axis in AXES:
            if self._aborted():
                self.finished_with.emit("aborted")
                return
            target_mm = self._targets.get(axis)
            if target_mm is None:
                self.progress.emit(f"Axis {axis.name}: no home reference, skipping")
                continue
            self.progress.emit(
                f"Moving axis {axis.name} → home ({target_mm:+.2f} mm)…"
            )
            try:
                target_units = mm_to_units(float(target_mm), axis)
                with self._lock:
                    self._controller.jog_single_axis(
                        axis,
                        position_units=target_units,
                        speed_units=self._speed.get(axis, 5.0),
                        acc_units=self._acc.get(axis, 20.0),
                        dec_units=self._dec.get(axis, 20.0),
                        relative=False,
                    )
            except Exception as exc:
                self.finished_with.emit(f"axis {axis.name}: {exc}")
                return
            # Poll per-axis stopped state. is_axis_stopped is per-axis and
            # works in 664; on transient errors keep polling rather than fail.
            while not self._aborted():
                try:
                    with self._lock:
                        stopped = self._controller.is_axis_stopped(axis)
                except Exception:
                    stopped = False
                if stopped:
                    break
                time.sleep(0.1)
        if self._aborted():
            self.finished_with.emit("aborted")
            return
        self.finished_with.emit("")


# =============================================================================
# Custom widgets
# =============================================================================
class SectionFrame(QWidget):
    """Card-style section using a standalone QLabel title above a QFrame#SectionCard.
    This avoids the QGroupBox::title clipping bug (title rendered inside the border)
    by making the title an ordinary widget ABOVE the card border.

    Callers do ``QLayout(frame.content())``; ``content()`` returns the inner card.
    The outer QWidget (= ``frame`` itself) is what callers pass to parent layouts.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #4ea1ff; padding: 0 4px;"
        )
        self._card = QFrame()
        self._card.setObjectName("SectionCard")
        outer.addWidget(lbl)
        outer.addWidget(self._card)

    def content(self) -> QWidget:
        """Return the inner card frame. Callers set their layout on this widget."""
        return self._card


class AxisStatusCard(QFrame):
    """Per-axis live readout: position (mm + units), velocity, accel, limit bar."""

    def __init__(self, axis: Axis, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.axis = axis
        self.setObjectName("AxisCard")
        self.setFrameShape(QFrame.NoFrame)
        self._soft_min_mm: float | None = None
        self._soft_max_mm: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        letter = QLabel(axis.name)
        letter.setObjectName("axis-letter")
        letter.setAlignment(Qt.AlignCenter)
        layout.addWidget(letter)

        pos_label = QLabel("Position (mm)")
        pos_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(pos_label)
        self.pos_display = QLabel("--")
        self.pos_display.setObjectName("PositionReadout")
        self.pos_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.pos_display.setMinimumHeight(46)
        _fit_label(self.pos_display, "-9999.99")
        layout.addWidget(self.pos_display)

        self.vel_acc_label = QLabel("Vel: -- cm/s\nAcc: -- cm/s²")
        self.vel_acc_label.setToolTip(
            "Velocity and acceleration in cm/s and cm/s².\n"
            "Derived via 5-sample central difference, not an SDK readout."
        )
        _fit_label(self.vel_acc_label, "Acc: 9999.99 cm/s²")
        self.vel_acc_label.setMinimumWidth(
            QtGui.QFontMetrics(self.vel_acc_label.font()).horizontalAdvance("Acc: 9999.99 cm/s²") + 16
        )
        layout.addWidget(self.vel_acc_label)

        # Δ-home line. Gray "—" until a home reference is captured.
        self.home_delta_label = QLabel("Δ home: —")
        self.home_delta_label.setStyleSheet(
            "color: #ffd54f; font-size: 12px; padding-top: 2px;"
        )
        self.home_delta_label.setToolTip(
            "Position relative to the last completed home (or manually-set "
            "reference). Set via the Setup tab → Homing → "
            "'Set Current as Home Reference' or by completing a homing op."
        )
        _fit_label(self.home_delta_label, "Δ home: -99999.99 mm")
        layout.addWidget(self.home_delta_label)

        layout.addSpacing(4)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(500)
        self.progress.setFormat("no soft limits")
        self.progress.setMinimumHeight(20)
        layout.addWidget(self.progress)

        self.setMinimumWidth(180)

    def set_soft_limits(self, lo_mm: float | None, hi_mm: float | None) -> None:
        self._soft_min_mm = lo_mm
        self._soft_max_mm = hi_mm
        if lo_mm is None or hi_mm is None or hi_mm <= lo_mm:
            self.progress.setFormat("no soft limits")
            self.progress.setValue(500)
            self.progress.setStyleSheet("")

    def set_home_reference(self, home_mm: float | None, current_mm: float | None) -> None:
        """Update the Δ-home line from a pre-computed home/current pair. Used
        by the panel when the home reference is captured outside the regular
        snapshot path (e.g. after a homing completion)."""
        if home_mm is None or current_mm is None:
            self.home_delta_label.setText("Δ home: —")
            self.home_delta_label.setStyleSheet(
                "color: #8a8a8a; font-size: 12px; padding-top: 2px;"
            )
        else:
            delta = current_mm - home_mm
            self.home_delta_label.setText(f"Δ home: {delta:+.2f} mm")
            self.home_delta_label.setStyleSheet(
                "color: #ffd54f; font-size: 12px; padding-top: 2px;"
            )

    def update_state(self, pos_units: float, pos_mm: float,
                     vel_mm_s: float, acc_mm_s2: float,
                     *, home_mm: float | None = None) -> None:
        self.pos_display.setText(f"{pos_mm:+.2f}")
        self.vel_acc_label.setText(
            f"Vel: {vel_mm_s / 10.0:.2f} cm/s\nAcc: {acc_mm_s2 / 10.0:.2f} cm/s²"
        )
        # Δ-home is recomputed every tick from the latest mm + the stored ref.
        self.set_home_reference(home_mm, pos_mm)
        lo, hi = self._soft_min_mm, self._soft_max_mm
        if lo is None or hi is None or hi <= lo:
            return
        span = hi - lo
        pct = (pos_mm - lo) / span * 100.0
        if pct < 0 or pct > 100:
            color = "#c62828"  # red: outside limits
        else:
            near = PROGRESS_NEAR_LIMIT_PCT
            color = "#ffa726" if (pct <= near or pct >= 100 - near) else "#1976d2"
        self.progress.setValue(int(max(0.0, min(100.0, pct)) * 10))
        self.progress.setFormat(f"{pos_mm:+.1f}   [{lo:+.0f}, {hi:+.0f}] mm")
        self.progress.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"
        )


# =============================================================================
# Compact status row (replaces the old tall AxisStatusCard in the left pane)
# =============================================================================
class _AxisRow(QFrame):
    """One horizontal row in LiveStatusTable — AxisCard hover styling preserved."""

    def __init__(self, axis: "Axis", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AxisCard")
        self.axis = axis
        self._soft_min: float | None = None
        self._soft_max: float | None = None

        h = QHBoxLayout(self)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)

        # Axis letter.
        letter = QLabel(axis.name)
        letter.setStyleSheet("font-weight: 700; font-size: 16px; color: #4ea1ff;")
        letter.setFixedWidth(24)
        letter.setAlignment(Qt.AlignCenter)
        h.addWidget(letter)

        # Position (mm) — PositionReadout objectName picks up green monospace CSS.
        self.pos_display = QLabel("--")
        self.pos_display.setObjectName("PositionReadout")
        self.pos_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fm = QtGui.QFontMetrics(self.pos_display.font())
        self.pos_display.setMinimumWidth(fm.horizontalAdvance("-9999.99") + 16)
        h.addWidget(self.pos_display)

        # Δ home.
        self.home_delta_label = QLabel("—")
        self.home_delta_label.setStyleSheet("color: #8a8a8a; font-size: 12px;")
        fm2 = QtGui.QFontMetrics(self.home_delta_label.font())
        self.home_delta_label.setMinimumWidth(fm2.horizontalAdvance("+9999.99 mm") + 16)
        h.addWidget(self.home_delta_label)

        # Vel · Acc on one line.
        self.vel_acc_label = QLabel("— cm/s · — cm/s²")
        self.vel_acc_label.setStyleSheet("color: #999; font-size: 12px;")
        fm3 = QtGui.QFontMetrics(self.vel_acc_label.font())
        self.vel_acc_label.setMinimumWidth(
            fm3.horizontalAdvance("999.99 cm/s · 999.99 cm/s²") + 16
        )
        h.addWidget(self.vel_acc_label, stretch=1)

        # Thin soft-limit progress bar.
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(500)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(90)
        self.progress.setStyleSheet(
            "QProgressBar { min-height: 8px; max-height: 8px; border-radius: 3px;"
            " background-color: #131316; border: 1px solid #3a3a40; }"
            "QProgressBar::chunk { background-color: #1976d2; border-radius: 2px; }"
        )
        h.addWidget(self.progress)

        # Single-character limit indicator.
        self.indicator = QLabel("—")
        self.indicator.setAlignment(Qt.AlignCenter)
        self.indicator.setStyleSheet("color: #666; font-size: 14px;")
        self.indicator.setFixedWidth(22)
        h.addWidget(self.indicator)

    # ── public interface (same as AxisStatusCard) ─────────────────────────────

    def update_state(
        self,
        pos_units: float,
        pos_mm: float,
        vel_mm_s: float,
        acc_mm_s2: float,
        *,
        home_mm: float | None = None,
    ) -> None:
        self.pos_display.setText(f"{pos_mm:+.2f}")
        self.vel_acc_label.setText(
            f"{vel_mm_s / 10.0:.2f} cm/s · {acc_mm_s2 / 10.0:.2f} cm/s²"
        )
        self.set_home_reference(home_mm, pos_mm)
        self._update_progress(pos_mm)

    def set_home_reference(
        self,
        home_mm: float | None,
        current_mm: float | None,
    ) -> None:
        if home_mm is None or current_mm is None:
            self.home_delta_label.setText("—")
            self.home_delta_label.setStyleSheet("color: #8a8a8a; font-size: 12px;")
        else:
            delta = current_mm - home_mm
            self.home_delta_label.setText(f"{delta:+.2f} mm")
            self.home_delta_label.setStyleSheet("color: #ffd54f; font-size: 12px;")

    def set_soft_limits(
        self,
        lo_mm: float | None,
        hi_mm: float | None,
    ) -> None:
        self._soft_min = lo_mm
        self._soft_max = hi_mm
        if lo_mm is None or hi_mm is None or hi_mm <= lo_mm:
            self.progress.setValue(500)
            self.progress.setStyleSheet(
                "QProgressBar { min-height: 8px; max-height: 8px; border-radius: 3px;"
                " background-color: #131316; border: 1px solid #3a3a40; }"
                "QProgressBar::chunk { background-color: #1976d2; border-radius: 2px; }"
            )
            self.indicator.setText("—")
            self.indicator.setStyleSheet("color: #666; font-size: 14px;")

    def _update_progress(self, pos_mm: float) -> None:
        lo, hi = self._soft_min, self._soft_max
        if lo is None or hi is None or hi <= lo:
            return
        span = hi - lo
        pct = (pos_mm - lo) / span * 100.0
        near = PROGRESS_NEAR_LIMIT_PCT
        if pct < 0 or pct > 100:
            color, ind_t, ind_c = "#c62828", "✗", "#ef5350"
        elif pct <= near or pct >= 100 - near:
            color, ind_t, ind_c = "#ffa726", "⚠", "#ffa726"
        else:
            color, ind_t, ind_c = "#1976d2", "✓", "#34d058"
        self.progress.setValue(int(max(0.0, min(100.0, pct)) * 10))
        self.progress.setStyleSheet(
            f"QProgressBar {{ min-height: 8px; max-height: 8px; border-radius: 3px;"
            f" background-color: #131316; border: 1px solid #3a3a40; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
        )
        self.indicator.setText(ind_t)
        self.indicator.setStyleSheet(
            f"color: {ind_c}; font-size: 14px; font-weight: bold;"
        )


class LiveStatusTable(QWidget):
    """Compact 3-row status table.  Replaces three side-by-side AxisStatusCards
    in the left pane.  Each row is an _AxisRow (QFrame#AxisCard) so the hover
    border still works.  Target height ≤ 200 px for three axes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self._rows: dict["Axis", _AxisRow] = {}
        for axis in AXES:
            row = _AxisRow(axis, self)
            v.addWidget(row)
            self._rows[axis] = row

    def row(self, axis: "Axis") -> _AxisRow:
        return self._rows[axis]


# =============================================================================
# Background AprilTag detector (runs on its own QThread)
# =============================================================================
class _TagDetectionWorker(QObject):
    """Runs pupil_apriltags.Detector on frames in a background thread.

    Drops frames when busy so the queue never backs up — we always reflect
    the latest available frame, never a stale one.
    """

    # (list of {'id', 'center', 'corners'}, (img_w, img_h))
    detections_ready = pyqtSignal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._detector: Any = None
        self._family: str = "tag36h11"
        self._busy: bool = False
        self._min_decision_margin = 30.0

    def set_family(self, family: str) -> None:
        if family != self._family:
            self._family = family
            self._detector = None  # lazy rebuild on next detect

    def on_frame(self, frame_bgr: Any, _t_mono: float) -> None:
        if self._busy or frame_bgr is None:
            return
        self._busy = True
        try:
            try:
                import cv2
                from pupil_apriltags import Detector
            except ImportError:
                self.detections_ready.emit([], (0, 0))
                return
            if self._detector is None:
                self._detector = Detector(
                    families=self._family,
                    nthreads=2,
                    quad_decimate=2.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0,
                )
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            detections = self._detector.detect(gray, estimate_tag_pose=False)
            tags: list[dict] = []
            for d in detections:
                if int(d.hamming) > 0:
                    continue
                if float(d.decision_margin) < self._min_decision_margin:
                    continue
                cx, cy = float(d.center[0]), float(d.center[1])
                corners = [(float(c[0]), float(c[1])) for c in d.corners]
                tags.append({"id": int(d.tag_id), "center": (cx, cy), "corners": corners})
            self.detections_ready.emit(tags, (w, h))
        except Exception as exc:
            print(f"[tag-detector] {exc}", file=sys.stderr)
            self.detections_ready.emit([], (0, 0))
        finally:
            self._busy = False


# =============================================================================
# Live fisheye preview widget
# =============================================================================
class FisheyePreviewWidget(QWidget):
    """Displays the live camera frame stream inside the left pane.

    On frame_ready: BGR ndarray → QPixmap scaled to fit while preserving
    aspect ratio, drawn at ≤ 15 FPS (configurable).  When a tag detector
    overlay is enabled, corners are drawn with QPainter before the pixmap is
    set (requires the experiment runner to push observations).
    """

    MAX_DISPLAY_FPS = 15.0
    DETECT_HZ = 5.0    # rate at which frames are forwarded to the tag detector

    # Internal signal: GUI thread → worker thread (auto-queued).
    _forward_frame = pyqtSignal(object, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_frame_t: float = 0.0
        self._last_frame: Any = None           # np.ndarray BGR
        self._detector_on: bool = False
        self._current_run_dir: Path | None = None

        # Tag detection state (always running; the overlay toggle only
        # controls whether boxes are drawn on the preview).
        self._last_detect_forward_t: float = 0.0
        self._latest_detections: list[dict] = []
        self._latest_detect_image_size: tuple[int, int] = (0, 0)

        # Restore detector toggle from settings; default ON so newly-installed
        # users immediately see tag boxes drawn over the preview.
        saved = _gp_load_settings().get("gantry_panel", {})
        self._detector_on = bool(saved.get("camera_detector_overlay", True))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Section title (same style as SectionFrame).
        title = QLabel("Fisheye Live")
        title.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #4ea1ff; padding: 0 4px;"
        )
        outer.addWidget(title)

        card = QFrame()
        card.setObjectName("SectionCard")
        outer.addWidget(card, stretch=1)

        v = QVBoxLayout(card)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # Preview label — pixmap goes here.
        self.preview_label = QLabel("Camera not connected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #0e0e0e; color: #555; border-radius: 4px;"
        )
        self.preview_label.setMinimumHeight(160)
        self.preview_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        v.addWidget(self.preview_label, stretch=1)

        # AprilTag detection status (always-on indicator).
        self.tag_status_label = QLabel("● Tags: —")
        self.tag_status_label.setStyleSheet(
            "color: #888; font-size: 12px; font-weight: 600;"
        )
        v.addWidget(self.tag_status_label)

        # Stats strip.
        self.stats_label = QLabel("—")
        self.stats_label.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(self.stats_label)

        # Background tag detector worker.
        self._det_thread = QThread(self)
        self._det_worker = _TagDetectionWorker()
        self._det_worker.moveToThread(self._det_thread)
        self._forward_frame.connect(self._det_worker.on_frame)
        self._det_worker.detections_ready.connect(self._on_detections_ready)
        self._det_thread.start()

        # Buttons.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.setEnabled(False)
        self.snapshot_btn.clicked.connect(self._on_snapshot)
        _size_button(self.snapshot_btn)
        btn_row.addWidget(self.snapshot_btn)

        self.overlay_btn = QPushButton(
            "Detector Overlay: ON" if self._detector_on else "Detector Overlay: OFF"
        )
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setChecked(self._detector_on)
        self.overlay_btn.clicked.connect(self._on_overlay_toggle)
        _size_button(self.overlay_btn)
        btn_row.addWidget(self.overlay_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)

    # ── public API ────────────────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """Update placeholder text when no frame has arrived yet."""
        if state == "disconnected":
            self.preview_label.clear()
            self.preview_label.setText("Camera not connected")
            self.preview_label.setStyleSheet(
                "background-color: #0e0e0e; color: #555; border-radius: 4px;"
            )
            self.snapshot_btn.setEnabled(False)
        elif state == "connecting":
            self.preview_label.clear()
            self.preview_label.setText("Connecting…")
            self.preview_label.setStyleSheet(
                "background-color: #0e0e0e; color: #888; border-radius: 4px;"
            )
        elif state.startswith("connected"):
            self.snapshot_btn.setEnabled(True)
        elif state == "error":
            self.preview_label.clear()
            self.preview_label.setText("Camera error — check connection bar")
            self.preview_label.setStyleSheet(
                "background-color: #0e0e0e; color: #ef5350; border-radius: 4px;"
            )

    def on_frame(self, frame_bgr: Any, t_mono: float) -> None:
        """Slot connected to FisheyeCameraSession.frame_ready; throttled to MAX_DISPLAY_FPS."""
        now = time.monotonic()
        # Always keep _last_frame fresh so snapshots / anchor auto-pick see
        # the very latest frame even when we skip rendering this tick.
        self._last_frame = frame_bgr
        # Forward to background detector at DETECT_HZ. Worker drops anything
        # that arrives while it's still processing the prior frame.
        if now - self._last_detect_forward_t >= 1.0 / self.DETECT_HZ:
            self._last_detect_forward_t = now
            self._forward_frame.emit(frame_bgr, t_mono)
        if now - self._last_frame_t < 1.0 / self.MAX_DISPLAY_FPS:
            return
        self._last_frame_t = now
        self._render_frame(frame_bgr)

    def update_stats(self, fps: int, grab_ms: float, config: dict) -> None:
        device = config.get("device", "?")
        w = config.get("width", "?")
        h = config.get("height", "?")
        mock = config.get("mock", False)
        mock_tag = " (mock)" if mock else ""
        self.stats_label.setText(
            f"Device {device} · {w}×{h} · {fps} FPS"
            f" · last grab {grab_ms:.0f} ms{mock_tag}"
        )

    def set_current_run_dir(self, path: Path | None) -> None:
        self._current_run_dir = path

    @property
    def detector_overlay_on(self) -> bool:
        return self._detector_on

    # ── private ──────────────────────────────────────────────────────────────

    def _render_frame(self, frame_bgr: Any) -> None:
        try:
            import cv2
            import numpy as np
            from PyQt5.QtGui import QImage
            display = frame_bgr
            if self._detector_on and self._latest_detections:
                display = frame_bgr.copy()
                for tag in self._latest_detections:
                    pts = np.asarray(tag["corners"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display, [pts], True, (0, 230, 230), 2, cv2.LINE_AA)
                    cx, cy = tag["center"]
                    cv2.putText(
                        display, f"ID {tag['id']}",
                        (int(cx) - 18, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 230, 230), 2, cv2.LINE_AA,
                    )
            h_px, w_px = display.shape[:2]
            frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                frame_rgb.data, w_px, h_px, w_px * 3, QImage.Format_RGB888
            )
            pix = QtGui.QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            self.preview_label.setPixmap(scaled)
        except Exception:
            pass

    def _on_detections_ready(self, tags: list, image_size: tuple) -> None:
        self._latest_detections = list(tags) if tags else []
        if image_size and image_size[0] > 0:
            self._latest_detect_image_size = (int(image_size[0]), int(image_size[1]))
        n = len(self._latest_detections)
        if n == 0:
            self.tag_status_label.setText("● Tags: none")
            self.tag_status_label.setStyleSheet(
                "color: #888; font-size: 12px; font-weight: 600;"
            )
        else:
            ids = sorted({t["id"] for t in self._latest_detections})
            ids_str = ", ".join(str(i) for i in ids[:8])
            if len(ids) > 8:
                ids_str += "…"
            self.tag_status_label.setText(f"● Tags: {n}  (IDs: {ids_str})")
            self.tag_status_label.setStyleSheet(
                "color: #4caf50; font-size: 12px; font-weight: 600;"
            )
        # If overlay is on, redraw the latest frame with fresh boxes.
        if self._detector_on and self._last_frame is not None:
            self._render_frame(self._last_frame)

    def latest_detections(self) -> tuple[list[dict], tuple[int, int]]:
        """Return the latest tag detections and the image size they apply to.

        Returns ([], (0, 0)) if nothing detected yet.
        """
        return list(self._latest_detections), self._latest_detect_image_size

    def set_tag_family(self, family: str) -> None:
        if family:
            self._det_worker.set_family(family)

    def shutdown(self) -> None:
        """Stop the detector thread cleanly. Called from MainWindow.closeEvent."""
        try:
            self._det_thread.quit()
            self._det_thread.wait(1000)
        except Exception:
            pass

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._last_frame is not None:
            self._render_frame(self._last_frame)

    def _on_snapshot(self) -> None:
        if self._last_frame is None:
            return
        try:
            import cv2
            save_dir = self._current_run_dir or Path.home() / "Pictures"
            save_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"snapshot_{ts}.png"
            cv2.imwrite(str(path), self._last_frame)
            self.stats_label.setText(f"Snapshot saved → {path.name}")
        except Exception as exc:
            self.stats_label.setText(f"Snapshot failed: {exc}")

    def _on_overlay_toggle(self, checked: bool) -> None:
        self._detector_on = checked
        self.overlay_btn.setText(
            "Detector Overlay: ON" if checked else "Detector Overlay: OFF"
        )
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["camera_detector_overlay"] = checked
        _gp_save_section("gantry_panel", payload)


class LivePlotWidget(QWidget):
    """30-second rolling position-vs-time plot. pyqtgraph if available."""

    AXIS_COLORS = {0: "#ef5350", 1: "#66bb6a", 2: "#64b5f6"}  # X red, Y green, Z blue

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAVE_PYQTGRAPH:
            self._plot = pg.PlotWidget()
            self._plot.setBackground("#0e0e0e")
            self._plot.setLabel("left", "Position (mm)", color="#888")
            self._plot.setLabel("bottom", "Time (s)", color="#888")
            self._plot.showGrid(x=True, y=True, alpha=0.15)
            self._plot.addLegend(offset=(10, 10))
            self._curves = {
                i: self._plot.plot([], [], pen=pg.mkPen(self.AXIS_COLORS[i], width=2),
                                   name=Axis(i).name)
                for i in range(3)
            }
            layout.addWidget(self._plot)
        else:
            self._plot = None
            self._curves = {}
            note = QLabel("Install pyqtgraph for live plots\n(pip install pyqtgraph)")
            note.setAlignment(Qt.AlignCenter)
            note.setStyleSheet("color: #888; padding: 24px; border: 1px dashed #444;")
            layout.addWidget(note)

    def update_data(self, t_history: deque, pos_history: dict[int, deque]) -> None:
        if self._plot is None or not t_history:
            return
        t_arr = list(t_history)
        t_now = t_arr[-1]
        cutoff = t_now - LIVE_PLOT_WINDOW_S
        # Trim to the visible window.
        i0 = 0
        for i, t in enumerate(t_arr):
            if t >= cutoff:
                i0 = i
                break
        t_visible = [t - t_now for t in t_arr[i0:]]  # plot as seconds-ago (negative)
        for axis_idx, curve in self._curves.items():
            y = list(pos_history[axis_idx])[i0:]
            if len(y) == len(t_visible):
                curve.setData(t_visible, y)


def _load_pool_config(repo_root: Path) -> dict:
    """Load pool dimensions from config/config.yaml.
    Returns dict with mm-converted dimensions and long-axis assignment.
    Falls back to hardcoded defaults if the file is missing or yaml unavailable."""
    defaults: dict = {
        "length_mm": 4877.0, "width_mm": 1800.0, "depth_mm": 1143.0, "long_axis": "y",
    }
    if _yaml is None:
        return defaults
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        print(f"[gantry_panel] {cfg_path} not found — using default pool size", file=sys.stderr)
        return defaults
    with cfg_path.open() as f:
        cfg = _yaml.safe_load(f) or {}
    pool = cfg.get("pool", {}) or {}
    return {
        "length_mm": float(pool.get("length_m", 4.877)) * 1000.0,
        "width_mm":  float(pool.get("width_m",  1.8))   * 1000.0,
        "depth_mm":  float(pool.get("depth_m",  1.143)) * 1000.0,
        "long_axis": (pool.get("pool_long_axis") or "y").lower(),
    }


def _pick_tick_spacing(span_mm: float) -> float:
    """Return major tick spacing in mm targeting ~5-8 labeled ticks in the span.
    Only major ticks are labelled; no minor level is emitted."""
    if span_mm <= 300:    return 50.0
    if span_mm <= 700:    return 100.0
    if span_mm <= 1500:   return 200.0
    if span_mm <= 3500:   return 500.0
    if span_mm <= 7500:   return 1000.0
    if span_mm <= 15000:  return 2000.0
    if span_mm <= 35000:  return 5000.0
    return 10000.0


# =============================================================================
# Workspace Map (top-down XY + side XZ; pyqtgraph preferred, QPainter fallback)
# =============================================================================
class WorkspaceMap(QWidget):
    """Two stacked 2D plots showing live position, optional target marker,
    a trailing path, and the soft-limit bounding box.

    Public API (called by the panel):
      * update_position(x_mm, y_mm, z_mm)
      * update_target(x_mm, y_mm, z_mm)  /  clear_target()
      * update_soft_limits(min_mm, max_mm)  # each is (x, y, z) of floats|None
      * set_show_trail(bool) / set_show_target(bool)

    Implementation: pyqtgraph if installed (Option A), QPainter fallback
    otherwise (Option B). Both backends share this exact API so callers don't
    care which one is in use.
    """

    TRAIL_MAXLEN = 200
    MIN_GROUP_SIZE = (260, 200)
    AUTO_FIT_MARGIN_MM = 50.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(*self.MIN_GROUP_SIZE)

        self._trail: deque = deque(maxlen=self.TRAIL_MAXLEN)
        self._cur_pos: tuple[float, float, float] | None = None
        self._target: tuple[float, float, float] | None = None
        self._home: tuple[float | None, float | None, float | None] = (None, None, None)
        self._soft_min: tuple[float | None, float | None, float | None] = (None, None, None)
        self._soft_max: tuple[float | None, float | None, float | None] = (None, None, None)
        self._show_trail = True
        self._show_target = True

        # Load pool dimensions from config.yaml (falls back to defaults gracefully).
        self._pool_cfg = _load_pool_config(_THIS_FILE.parent.parent)

        # Outer layout: standalone title label above a QFrame#SectionCard.
        # Title is outside the card border → physically impossible to clip.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        map_title = QLabel("Workspace Map")
        map_title.setStyleSheet(
            "font-size: 14px; font-weight: 600; color: #4ea1ff; padding: 0 4px;"
        )
        outer.addWidget(map_title)
        _map_card = QFrame()
        _map_card.setObjectName("SectionCard")
        outer.addWidget(_map_card, stretch=1)

        v = QVBoxLayout(_map_card)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Always-visible Home header row.
        self.home_header = QLabel("Home: not set")
        self.home_header.setStyleSheet(
            "color: #ffd54f; font-weight: 600; font-size: 12px;"
        )
        v.addWidget(self.home_header)

        # Toolbar: trail/target toggles + 3-mode fit dropdown.
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.trail_chk = QCheckBox("Show Trail")
        self.trail_chk.setChecked(True)
        self.trail_chk.toggled.connect(self.set_show_trail)
        bar.addWidget(self.trail_chk)
        self.target_chk = QCheckBox("Show Target")
        self.target_chk.setChecked(True)
        self.target_chk.toggled.connect(self.set_show_target)
        bar.addWidget(self.target_chk)
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("Fit: Pool",           "pool")
        self.fit_combo.addItem("Fit: Soft Limits",    "soft_limits")
        self.fit_combo.addItem("Fit: Trail + Target", "trail_target")
        saved_mode = _gp_load_settings().get("gantry_panel", {}).get("map_fit_mode", "pool")
        idx = self.fit_combo.findData(saved_mode)
        self.fit_combo.setCurrentIndex(max(0, idx))
        self.fit_combo.currentIndexChanged.connect(self._on_fit_mode_changed)
        self.fit_combo.setToolTip(
            "Fit: Pool — sets view to pool dimensions from config/config.yaml\n"
            "Fit: Soft Limits — sets view to the gantry's configured soft-limit envelope\n"
            "Fit: Trail + Target — auto-fits to the current trail and target position"
        )
        bar.addWidget(self.fit_combo)
        bar.addStretch()
        v.addLayout(bar)

        # Backend.
        self._backend = _PyQtGraphMap() if HAVE_PYQTGRAPH else _PainterMap()
        v.addWidget(self._backend, stretch=1)

    # ---- public API ----------------------------------------------------
    def update_position(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        self._cur_pos = (float(x_mm), float(y_mm), float(z_mm))
        self._trail.append(self._cur_pos)
        self._refresh()

    def update_target(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        self._target = (float(x_mm), float(y_mm), float(z_mm))
        self._refresh()

    def clear_target(self) -> None:
        self._target = None
        self._refresh()

    def update_soft_limits(self, mn: tuple, mx: tuple) -> None:
        self._soft_min = tuple(mn)
        self._soft_max = tuple(mx)
        if hasattr(self._backend, 'reset_view'):
            self._backend.reset_view()
        self._refresh()

    def update_home(self, home_xyz: tuple) -> None:
        """Set the home-reference triplet (Nones allowed per axis)."""
        new_home = tuple(home_xyz)
        home_changed = new_home != self._home
        self._home = new_home
        # Update header text (runs every poll to keep Δ distance fresh).
        if all(v is not None for v in self._home):
            hx, hy, hz = self._home
            if self._cur_pos is not None:
                dx = self._cur_pos[0] - hx
                dy = self._cur_pos[1] - hy
                dz = self._cur_pos[2] - hz
                mag = math.sqrt(dx * dx + dy * dy + dz * dz)
                self.home_header.setText(
                    f"Home: X={hx:+.2f}  Y={hy:+.2f}  Z={hz:+.2f}  "
                    f"(Δ from current: {mag:.1f} mm)"
                )
            else:
                self.home_header.setText(
                    f"Home: X={hx:+.2f}  Y={hy:+.2f}  Z={hz:+.2f}"
                )
            self.home_header.setToolTip(
                f"Home reference (X={hx:.3f}, Y={hy:.3f}, Z={hz:.3f}) mm"
            )
        else:
            self.home_header.setText("Home: not set")
            self.home_header.setToolTip("")
        # Only reset the map view when home actually changes (not every poll).
        if home_changed and hasattr(self._backend, 'reset_view'):
            self._backend.reset_view()
        self._refresh()

    def set_show_trail(self, on: bool) -> None:
        self._show_trail = bool(on)
        self._refresh()

    def set_show_target(self, on: bool) -> None:
        self._show_target = bool(on)
        self._refresh()

    # ---- internal ------------------------------------------------------
    def _on_fit_mode_changed(self) -> None:
        mode = self.fit_combo.currentData()
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["map_fit_mode"] = mode
        _gp_save_section("gantry_panel", payload)
        if hasattr(self._backend, 'reset_view'):
            self._backend.reset_view()
        self._refresh()

    def _pool_bounds(self) -> dict[str, tuple[float, float]]:
        """Compute pool rectangle bounds from config.

        Pool long axis (4.877 m) is always along gantry X; width (1.8 m) along Y.
        When a home reference is set, home is the pool's bottom-left (min X, min Y)
        corner so the pool extends in the +X/+Y direction from home.
        Before homing, corner defaults to (0, 0, 0)."""
        p = self._pool_cfg
        hx = float(self._home[0]) if self._home[0] is not None else 0.0
        hy = float(self._home[1]) if self._home[1] is not None else 0.0
        hz = float(self._home[2]) if self._home[2] is not None else 0.0
        return {
            "x": (hx, hx + p["length_mm"]),
            "y": (hy, hy + p["width_mm"]),
            "z": (hz - p["depth_mm"], hz),
        }

    def _refresh(self) -> None:
        view_bounds = self._compute_view_bounds()
        pool_bounds = self._pool_bounds()
        self._backend.render(
            cur_pos=self._cur_pos,
            target=self._target if self._show_target else None,
            trail=list(self._trail) if self._show_trail else [],
            soft_min=self._soft_min,
            soft_max=self._soft_max,
            home=self._home,
            view_bounds=view_bounds,
            pool_bounds=pool_bounds,
        )

    def _compute_view_bounds(self) -> dict[str, tuple[float, float]]:
        """Return dict with keys 'x', 'y', 'z' -> (lo, hi) for axes."""
        mode = self.fit_combo.currentData()
        if mode == "pool":
            return self._pool_bounds()
        if mode == "soft_limits" and all(v is not None for v in (*self._soft_min, *self._soft_max)):
            return {
                "x": (float(self._soft_min[0]), float(self._soft_max[0])),
                "y": (float(self._soft_min[1]), float(self._soft_max[1])),
                "z": (float(self._soft_min[2]), float(self._soft_max[2])),
            }
        # trail_target mode (or soft_limits fallback when limits unset).
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for p in self._trail:
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        if self._cur_pos is not None:
            xs.append(self._cur_pos[0]); ys.append(self._cur_pos[1]); zs.append(self._cur_pos[2])
        if self._target is not None:
            xs.append(self._target[0]); ys.append(self._target[1]); zs.append(self._target[2])
        if not xs:
            return self._pool_bounds()   # sensible fallback: pool view
        m = self.AUTO_FIT_MARGIN_MM
        return {
            "x": (min(xs) - m, max(xs) + m),
            "y": (min(ys) - m, max(ys) + m),
            "z": (min(zs) - m, max(zs) + m),
        }


class _PyQtGraphMap(QWidget):
    """pyqtgraph backend: two PlotWidgets, shared X axis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # XY (top-down). No aspect lock — lock fights with setRange() on narrow widgets.
        self._xy = pg.PlotWidget(background="#101013")
        self._xy.showGrid(x=True, y=True, alpha=0.18)
        self._xy.setLabel("left", "Y (mm)", color="#888")
        self._xy.setLabel("bottom", "X (mm)", color="#888")
        self._xy.setTitle("Top-down (XY)", color="#bbb", size="10pt")
        self._xy_trail = self._xy.plot([], [], pen=pg.mkPen((76, 175, 80, 200), width=1.5))
        self._xy_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(76, 175, 80, 255), pen=pg.mkPen("k", width=1),
        )
        self._xy.addItem(self._xy_dot)
        self._xy_target = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(None), pen=pg.mkPen(30, 144, 255, width=2),
            symbol="o",
        )
        self._xy.addItem(self._xy_target)
        # Soft-limit envelope (gray, dotted).
        self._xy_sl_box = pg.PlotCurveItem(
            pen=pg.mkPen(120, 120, 120, 180, width=1, style=Qt.DotLine))
        self._xy.addItem(self._xy_sl_box)
        # Pool outline (light blue, dashed).
        self._xy_pool = pg.PlotCurveItem(
            pen=pg.mkPen((0, 180, 216, 200), width=1, style=Qt.DashLine))
        self._xy.addItem(self._xy_pool)
        # Tick label style — suppress auto-text-expansion to prevent label overlap.
        for _ax in (self._xy.getAxis("bottom"), self._xy.getAxis("left")):
            _ax.setStyle(autoExpandTextSpace=False, tickTextOffset=4)
            _ax.setTextPen(QtGui.QPen(QtGui.QColor("#bbb")))
        v.addWidget(self._xy, stretch=1)

        # XZ (side).
        self._xz = pg.PlotWidget(background="#101013")
        self._xz.showGrid(x=True, y=True, alpha=0.18)
        self._xz.setLabel("left", "Z (mm)", color="#888")
        self._xz.setLabel("bottom", "X (mm)", color="#888")
        self._xz.setTitle("Side (XZ)", color="#bbb", size="10pt")
        self._xz.setXLink(self._xy)  # shared X axis
        self._xz_trail = self._xz.plot([], [], pen=pg.mkPen((76, 175, 80, 200), width=1.5))
        self._xz_dot = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(76, 175, 80, 255), pen=pg.mkPen("k", width=1),
        )
        self._xz.addItem(self._xz_dot)
        self._xz_target = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(None), pen=pg.mkPen(30, 144, 255, width=2),
            symbol="o",
        )
        self._xz.addItem(self._xz_target)
        # Soft-limit envelope (gray, dotted).
        self._xz_sl_box = pg.PlotCurveItem(
            pen=pg.mkPen(120, 120, 120, 180, width=1, style=Qt.DotLine))
        self._xz.addItem(self._xz_sl_box)
        # Pool outline (light blue, dashed).
        self._xz_pool = pg.PlotCurveItem(
            pen=pg.mkPen((0, 180, 216, 200), width=1, style=Qt.DashLine))
        self._xz.addItem(self._xz_pool)
        # Tick label style for XZ axes.
        for _ax in (self._xz.getAxis("bottom"), self._xz.getAxis("left")):
            _ax.setStyle(autoExpandTextSpace=False, tickTextOffset=4)
            _ax.setTextPen(QtGui.QPen(QtGui.QColor("#bbb")))
        # Home marker (yellow star) + dashed line from home to current dot.
        home_pen = pg.mkPen(255, 213, 79, width=2)
        self._xy_home = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(255, 213, 79, 220), pen=pg.mkPen("k", width=1),
            symbol="star",
        )
        self._xy.addItem(self._xy_home)
        self._xz_home = pg.ScatterPlotItem(
            size=14, brush=pg.mkBrush(255, 213, 79, 220), pen=pg.mkPen("k", width=1),
            symbol="star",
        )
        self._xz.addItem(self._xz_home)
        self._xy_homeline = pg.PlotCurveItem(pen=pg.mkPen(
            255, 213, 79, width=1, style=Qt.DashLine,
        ))
        self._xy.addItem(self._xy_homeline)
        self._xz_homeline = pg.PlotCurveItem(pen=pg.mkPen(
            255, 213, 79, width=1, style=Qt.DashLine,
        ))
        self._xz.addItem(self._xz_homeline)
        # One-time init: disable auto-range and SI-prefix scaling on all axes.
        # Range is driven exclusively by render() via _last_view_bounds tracking.
        self._xy.enableAutoRange(False)
        self._xz.enableAutoRange(False)
        for _a in (self._xy.getAxis("bottom"), self._xy.getAxis("left"),
                   self._xz.getAxis("bottom"), self._xz.getAxis("left")):
            _a.enableAutoSIPrefix(False)
        self._last_view_bounds: dict | None = None

        # Dynamic tick spacing: update whenever pyqtgraph's internal viewport
        # changes (scroll-zoom, drag-pan), not just when the fit-mode bounds change.
        def _update_xy_ticks(_vb, ranges) -> None:
            xspan = abs(ranges[0][1] - ranges[0][0])
            yspan = abs(ranges[1][1] - ranges[1][0])
            self._xy.getAxis("bottom").setTickSpacing(levels=[(_pick_tick_spacing(xspan), 0)])
            self._xy.getAxis("left").setTickSpacing(levels=[(_pick_tick_spacing(yspan), 0)])
            self._xz.getAxis("bottom").setTickSpacing(levels=[(_pick_tick_spacing(xspan), 0)])

        def _update_xz_ticks(_vb, ranges) -> None:
            zspan = abs(ranges[1][1] - ranges[1][0])
            self._xz.getAxis("left").setTickSpacing(levels=[(_pick_tick_spacing(zspan), 0)])

        self._xy.getViewBox().sigRangeChanged.connect(_update_xy_ticks)
        self._xz.getViewBox().sigRangeChanged.connect(_update_xz_ticks)

        v.addWidget(self._xz, stretch=1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Widget was resized (including first show when it goes from 0×0 to real
        # pixel dimensions).  Invalidate the cache so the next render() reapplies
        # the range correctly with the actual widget size.
        self._last_view_bounds = None

    def reset_view(self) -> None:
        """Invalidate cached bounds so next render() reapplies range + ticks."""
        self._last_view_bounds = None

    def render(self, *, cur_pos, target, trail, soft_min, soft_max, home,
               view_bounds, pool_bounds) -> None:
        # Soft-limit envelope (gray dotted).
        if all(v is not None for v in (*soft_min, *soft_max)):
            x0, x1 = float(soft_min[0]), float(soft_max[0])
            y0, y1 = float(soft_min[1]), float(soft_max[1])
            z0, z1 = float(soft_min[2]), float(soft_max[2])
            self._xy_sl_box.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
            self._xz_sl_box.setData([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0])
        else:
            self._xy_sl_box.setData([], [])
            self._xz_sl_box.setData([], [])

        # Pool outline (light blue dashed).
        xp0, xp1 = pool_bounds["x"]
        yp0, yp1 = pool_bounds["y"]
        zp0, zp1 = pool_bounds["z"]
        self._xy_pool.setData([xp0, xp1, xp1, xp0, xp0], [yp0, yp0, yp1, yp1, yp0])
        self._xz_pool.setData([xp0, xp1, xp1, xp0, xp0], [zp0, zp0, zp1, zp1, zp0])

        # Trail.
        if trail:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            zs = [p[2] for p in trail]
            self._xy_trail.setData(xs, ys)
            self._xz_trail.setData(xs, zs)
        else:
            self._xy_trail.setData([], [])
            self._xz_trail.setData([], [])

        # Current position dot.
        if cur_pos is not None:
            self._xy_dot.setData([cur_pos[0]], [cur_pos[1]])
            self._xz_dot.setData([cur_pos[0]], [cur_pos[2]])
        else:
            self._xy_dot.setData([], [])
            self._xz_dot.setData([], [])

        # Target marker.
        if target is not None:
            self._xy_target.setData([target[0]], [target[1]])
            self._xz_target.setData([target[0]], [target[2]])
        else:
            self._xy_target.setData([], [])
            self._xz_target.setData([], [])

        # Home star + dashed connecting line to current dot.
        home_set = all(v is not None for v in home)
        if home_set:
            hx, hy, hz = float(home[0]), float(home[1]), float(home[2])
            self._xy_home.setData([hx], [hy])
            self._xz_home.setData([hx], [hz])
            if cur_pos is not None:
                self._xy_homeline.setData([hx, cur_pos[0]], [hy, cur_pos[1]])
                self._xz_homeline.setData([hx, cur_pos[0]], [hz, cur_pos[2]])
            else:
                self._xy_homeline.setData([], [])
                self._xz_homeline.setData([], [])
        else:
            self._xy_home.setData([], [])
            self._xz_home.setData([], [])
            self._xy_homeline.setData([], [])
            self._xz_homeline.setData([], [])

        # Apply range + tick spacing only when view_bounds changes.
        # This lets the user zoom/pan freely between changes; range is only
        # reset when fit mode, home, or soft limits are updated.
        if view_bounds != self._last_view_bounds:
            self._last_view_bounds = view_bounds
            xr = view_bounds["x"]
            yr = view_bounds["y"]
            zr = view_bounds["z"]
            # setRange() handles aspect-locked viewboxes correctly by computing
            # a bounding rect that contains both axes simultaneously.
            self._xy.plotItem.vb.setRange(
                xRange=xr, yRange=yr, padding=0.05, disableAutoRange=True
            )
            self._xz.plotItem.vb.setRange(
                yRange=zr, padding=0.05, disableAutoRange=True
            )
            # Adaptive tick spacing — major ticks only (no minor level) to prevent
            # label overlap.  ~5-8 labeled ticks per axis.
            xmaj = _pick_tick_spacing(xr[1] - xr[0])
            ymaj = _pick_tick_spacing(yr[1] - yr[0])
            zmaj = _pick_tick_spacing(zr[1] - zr[0])
            self._xy.getAxis("bottom").setTickSpacing(levels=[(xmaj, 0)])
            self._xy.getAxis("left").setTickSpacing(levels=[(ymaj, 0)])
            self._xz.getAxis("bottom").setTickSpacing(levels=[(xmaj, 0)])
            self._xz.getAxis("left").setTickSpacing(levels=[(zmaj, 0)])


class _PainterMap(QWidget):
    """QPainter fallback: same surface as _PyQtGraphMap.render(...)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(360)
        self._cur_pos = None
        self._target = None
        self._home = (None, None, None)
        self._trail: list[tuple[float, float, float]] = []
        self._soft_min = (None, None, None)
        self._soft_max = (None, None, None)
        self._view_bounds = {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "z": (-100.0, 100.0)}
        self._warned_fallback = False

    def render(self, *, cur_pos, target, trail, soft_min, soft_max, home,
               view_bounds, pool_bounds=None) -> None:
        if not self._warned_fallback:
            print("[gantry_panel] pyqtgraph not available — using QPainter map fallback.",
                  file=sys.stderr)
            self._warned_fallback = True
        self._cur_pos = cur_pos
        self._target = target
        self._home = tuple(home)
        self._trail = list(trail)
        self._soft_min = tuple(soft_min)
        self._soft_max = tuple(soft_max)
        self._view_bounds = view_bounds
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()
        half_h = rect.height() // 2
        top = QtCore.QRect(rect.x(), rect.y(), rect.width(), half_h - 2)
        bot = QtCore.QRect(rect.x(), rect.y() + half_h + 2, rect.width(), half_h - 2)
        self._draw_view(p, top, "x", "y", "Top-down (XY)")
        self._draw_view(p, bot, "x", "z", "Side (XZ)")
        p.end()

    def _draw_view(self, p: QtGui.QPainter, rect: QtCore.QRect, ax_h: str, ax_v: str, title: str) -> None:
        p.fillRect(rect, QtGui.QColor("#101013"))
        p.setPen(QtGui.QPen(QtGui.QColor("#3a3a40"), 1))
        p.drawRect(rect)

        h_lo, h_hi = self._view_bounds[ax_h]
        v_lo, v_hi = self._view_bounds[ax_v]
        if h_hi <= h_lo or v_hi <= v_lo:
            return
        # Margin inside rect.
        m = 28
        plot = QtCore.QRect(rect.x() + m, rect.y() + m, rect.width() - 2 * m, rect.height() - 2 * m)

        def to_screen(hv: float, vv: float) -> tuple[int, int]:
            u = plot.left() + int((hv - h_lo) / (h_hi - h_lo) * plot.width())
            # Flip vertical (image y down vs world y up).
            v = plot.bottom() - int((vv - v_lo) / (v_hi - v_lo) * plot.height())
            return u, v

        # Grid every 100 mm.
        p.setPen(QtGui.QPen(QtGui.QColor("#2a2a30"), 1))
        step = 100.0
        start_h = math.floor(h_lo / step) * step
        h = start_h
        while h <= h_hi:
            u, _ = to_screen(h, v_lo)
            p.drawLine(u, plot.top(), u, plot.bottom())
            h += step
        start_v = math.floor(v_lo / step) * step
        vv = start_v
        while vv <= v_hi:
            _, vp = to_screen(h_lo, vv)
            p.drawLine(plot.left(), vp, plot.right(), vp)
            vv += step

        # Soft-limit box.
        idx_h = "xyz".index(ax_h)
        idx_v = "xyz".index(ax_v)
        if self._soft_min[idx_h] is not None and self._soft_max[idx_h] is not None and \
           self._soft_min[idx_v] is not None and self._soft_max[idx_v] is not None:
            x0, y0 = to_screen(self._soft_min[idx_h], self._soft_max[idx_v])
            x1, y1 = to_screen(self._soft_max[idx_h], self._soft_min[idx_v])
            p.setPen(QtGui.QPen(QtGui.QColor("#777"), 1))
            p.drawRect(QtCore.QRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

        # Trail.
        if self._trail and len(self._trail) >= 2:
            n = len(self._trail)
            for i in range(1, n):
                p0 = self._trail[i - 1]
                p1 = self._trail[i]
                u0, v0 = to_screen(p0[idx_h], p0[idx_v])
                u1, v1 = to_screen(p1[idx_h], p1[idx_v])
                alpha = int(60 + (i / n) * 195)
                p.setPen(QtGui.QPen(QtGui.QColor(76, 175, 80, alpha), 1.5))
                p.drawLine(u0, v0, u1, v1)

        # Target marker.
        if self._target is not None:
            u, v = to_screen(self._target[idx_h], self._target[idx_v])
            p.setPen(QtGui.QPen(QtGui.QColor(30, 144, 255), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QtCore.QPoint(u, v), 6, 6)

        # Home star + dashed connecting line.
        home_set = (self._home[idx_h] is not None and self._home[idx_v] is not None)
        if home_set:
            hu, hv = to_screen(float(self._home[idx_h]), float(self._home[idx_v]))
            if self._cur_pos is not None:
                cu, cv = to_screen(self._cur_pos[idx_h], self._cur_pos[idx_v])
                pen = QtGui.QPen(QtGui.QColor(255, 213, 79, 200), 1, Qt.DashLine)
                p.setPen(pen)
                p.drawLine(hu, hv, cu, cv)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 165, 0), 1))
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 213, 79)))
            p.drawEllipse(QtCore.QPoint(hu, hv), 7, 7)
        else:
            # Gray overlay corner text "No home reference set".
            p.setPen(QtGui.QPen(QtGui.QColor("#666"), 1))
            p.drawText(plot.right() - 156, plot.top() + 12, "No home reference set")

        # Position dot.
        if self._cur_pos is not None:
            u, v = to_screen(self._cur_pos[idx_h], self._cur_pos[idx_v])
            p.setPen(QtGui.QPen(QtGui.QColor("#0a0"), 1))
            p.setBrush(QtGui.QBrush(QtGui.QColor(76, 175, 80)))
            p.drawEllipse(QtCore.QPoint(u, v), 5, 5)
            # Crosshair to axes.
            p.setPen(QtGui.QPen(QtGui.QColor(76, 175, 80, 120), 1))
            p.drawLine(u, v, plot.left(), v)
            p.drawLine(u, v, u, plot.bottom())

        # Axis labels.
        p.setPen(QtGui.QPen(QtGui.QColor("#888"), 1))
        p.drawText(plot.right() - 64, plot.bottom() + 14, f"{ax_h.upper()} (mm)")
        p.drawText(plot.left() - 26, plot.top() + 10, f"{ax_v.upper()} (mm)")
        # Title.
        p.setPen(QtGui.QPen(QtGui.QColor("#bbb"), 1))
        p.drawText(rect.x() + 8, rect.y() + 16, title)


# =============================================================================
# Layout / sizing helpers (clipping fixes)
# =============================================================================
def _fit_label(label: QLabel, sample_text: str) -> None:
    """Size a value-bearing label to fit its worst-case text without truncation.

    Adds a 16 px buffer on top of the measured advance to absorb stylesheet
    padding (QSS padding is NOT included in QFontMetrics measurements)."""
    fm = QtGui.QFontMetrics(label.font())
    label.setMinimumWidth(fm.horizontalAdvance(sample_text) + 16)


# ---- minimal JSON settings persistence (active tab) ------------------------
_GP_SETTINGS_PATH = Path.home() / ".umi_gui_state.json"


def _gp_load_settings() -> dict:
    try:
        import json
        return json.loads(_GP_SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _gp_save_section(key: str, payload: dict) -> None:
    try:
        import json
        data = _gp_load_settings()
        data[key] = payload
        _GP_SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def _size_mm_spinbox(spin) -> None:
    spin.setMinimumWidth(110)
    spin.setMinimumHeight(28)


def _size_button(btn, kind: str = "normal") -> None:
    if kind == "primary":
        btn.setMinimumHeight(40)
    elif kind == "emergency":
        btn.setMinimumHeight(52)
    else:
        btn.setMinimumHeight(34)


def _make_note_label(text: str) -> QLabel:
    """Small gray inline-help label used for in-section explanations."""
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
    lbl.setWordWrap(True)
    return lbl


# =============================================================================
# Main window
# =============================================================================
class GantryPanel(QMainWindow):
    """Live PyQt control panel for the FMC4030 gantry."""

    POLL_INTERVAL_MS = STATUS_POLL_MS

    def __init__(self, controller=None, is_mock: bool = False) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE + (" — MOCK MODE" if is_mock else ""))
        self.resize(1500, 900)
        self.setMinimumSize(1200, 700)

        # SDK + lock + state ---------------------------------------------------
        self.controller = controller if controller is not None else FMC4030Controller()
        self.is_mock = is_mock
        self._controller_lock = threading.RLock()
        self.connected = False

        # Per-axis direction flip: +1 = panel matches firmware counter, -1 = inverted.
        # Restored from settings; user toggles in Setup tab → Axis Direction.
        # Applied at the panel↔SDK boundary only (mm_user_to_units / units_to_mm_user).
        self._axis_sign: dict[Axis, int] = {a: 1 for a in AXES}
        try:
            saved_sign = (_gp_load_settings()
                          .get("gantry_panel", {}).get("axis_sign", {}))
            for a in AXES:
                v = int(saved_sign.get(a.name, 1))
                self._axis_sign[a] = 1 if v >= 0 else -1
        except Exception:
            pass

        # Cached ABSOLUTE machine-frame mm of the latest snapshot, alongside
        # the user-frame _last_pos_mm. Used by anything that needs the raw
        # machine value: home-reference capture, the "Abs: …" diagnostic
        # label, and run_metadata.json logging.
        self._last_pos_abs_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Live data for finite-diff accel + live plot.
        self._vel_buffer: deque = deque(maxlen=FINITE_DIFF_WINDOW)
        self._pos_history: dict[int, deque] = {
            i: deque(maxlen=int(LIVE_PLOT_WINDOW_S * 1000 / STATUS_POLL_MS) + 5)
            for i in range(3)
        }
        self._time_history: deque = deque(
            maxlen=int(LIVE_PLOT_WINDOW_S * 1000 / STATUS_POLL_MS) + 5
        )
        self._t0_mono = time.monotonic()

        # Recording state.
        self._logger: GantryTelemetryLogger | None = None
        self._recording_manual = False
        self._recording_auto = False
        self._recording_via_runner = False
        # True while the ExperimentRunner owns the gantry logger; blocks the
        # panel's auto-logger from spawning parallel <ts>_gantry_run folders.
        self._runner_owns_logger = False
        self._autologger_suppress_warned = False
        self._current_run_dir: Path | None = None

        # Worker handles.
        self._status_thread: StatusPollThread | None = None
        self._move_thread: MoveToTargetThread | None = None
        self._sequence_thread: SequenceThread | None = None
        # Panel-scoped abort flag, set together with the gantry_runner module's
        # EMERGENCY_STOP event whenever the E-Stop fires. Workers we own
        # (SequenceThread, AxisAbsMoveThread) check this flag at every loop
        # iteration so they exit quickly on E-Stop without waiting for the
        # SDK lock.
        self._abort_event = threading.Event()
        # Tracks freshness of the last status snapshot for the Polling
        # indicator (set in _on_status_snapshot).
        self._last_snapshot_t: float = 0.0
        # Home reference (absolute machine-frame mm) per axis, set by the
        # "Set Current as Home Reference" button. Restored from settings.
        self._home_position_mm: dict[Axis, float | None] = {a: None for a in AXES}
        try:
            saved_home = _gp_load_settings().get("gantry_panel", {}).get("home_position_mm", {})
            for a in AXES:
                v = saved_home.get(a.name)
                if v is not None:
                    self._home_position_mm[a] = float(v)
        except Exception:
            pass
        # Per-axis Move Abs workers (one possible live thread per axis).
        self._per_axis_threads: dict[int, AxisAbsMoveThread] = {}
        # Widget handles for the Per-Axis Control cards (filled by _build_per_axis_card).
        self.per_axis_cards: dict[Axis, dict[str, Any]] = {}

        # In-progress flags (drive button enable/disable).
        self._move_in_progress = False
        self._sequence_in_progress = False
        self._home_thread: GoToHomeThread | None = None
        # Per-axis Move Abs is in progress for at least one axis.
        self._per_axis_busy: set[int] = set()

        # Experiment runner (wired up in _build_experiment_tab).
        self._experiment_runner: Any = None
        self._experiment_in_progress = False
        # Camera test result cache (None = untested, True = OK, False = FAIL)
        self._camera_test_result: bool | None = None
        self._exp_fisheye_calib_path: Path | None = None
        # Whether the panel was launched with --mock-camera
        self._is_mock_camera = getattr(self, "_cli_mock_camera", False)
        # Panel-level persistent camera session (shared with experiment runner).
        self._camera: Any = None  # FisheyeCameraSession | None

        # Build UI. ------------------------------------------------------------
        self._build_menu()
        self._build_ui()
        self._build_status_bar()
        self._update_all_button_states()

        # Restore + persist the last-active right-pane tab.
        try:
            saved = _gp_load_settings().get("gantry_panel", {})
            idx = int(saved.get("active_tab", 0))
            if 0 <= idx < self.tabs.count():
                self.tabs.setCurrentIndex(idx)
        except Exception:
            pass
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Status poll timer.
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(self.POLL_INTERVAL_MS)
        self._status_timer.timeout.connect(self._request_status_update)
        self._status_timer.start()

        # Polling freshness indicator (independent ~5 Hz tick so it advances
        # the "age" display even when no fresh snapshot has arrived).
        self._poll_indicator_timer = QTimer(self)
        self._poll_indicator_timer.setInterval(200)
        self._poll_indicator_timer.timeout.connect(self._update_poll_indicator)
        self._poll_indicator_timer.start()

        # E-Stop wiring diagnostic — PyQt5 dropped the old QtCore.SIGNAL()
        # shim, so we query receivers via the bound signal directly.
        try:
            n_recv = self.estop_btn.receivers(self.estop_btn.clicked)
            print(f"[estop-debug] estop_btn.clicked receivers: {n_recv}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[estop-debug] could not query receivers: {exc}", file=sys.stderr)

        # One-shot clipping audit (runs after the window paints once).
        QTimer.singleShot(500, self._audit_clipping)

    # ------------------------------------------------------------------
    # Frame-conversion helpers  (single source of truth)
    # ------------------------------------------------------------------
    # The panel speaks USER-FACING mm everywhere. "User-frame" means:
    #   1. HOME-RELATIVE: zero is the captured home reference for that axis.
    #   2. SIGN-FLIPPED: when axis_sign[axis] == -1, the user's "+" matches
    #      the user's physical intuition (the firmware counter may decrease).
    #
    # Storage invariants (set in __init__; reasserted whenever updated):
    #   - self._home_position_mm[axis]      : ABSOLUTE machine-frame mm
    #   - self._last_pos_mm[axis]           : user-frame mm (home-relative, sign-flipped)
    #   - self._last_pos_abs_mm[axis]       : ABSOLUTE machine-frame mm  (for diagnostics)
    # (Soft-limit fields removed — panel no longer manages soft limits.)
    #
    # Conversion chain (positions):
    #   abs_mm  = (user_mm * sign)  +  home_offset_abs
    #   user_mm = (abs_mm - home_offset_abs) * sign
    #   units   = abs_mm / SCALE_MM_PER_UNIT[axis]
    #
    # For velocities (and any other delta-quantity), the home offset does NOT
    # apply — only the sign flip. See vel_units_to_user_mm_s below.
    #
    # When the user has not yet captured a home reference, home_offset_abs == 0
    # and user-frame collapses to "raw absolute × sign" — graceful degradation.

    def _home_offset_abs(self, axis: Axis) -> float:
        """Absolute machine-frame mm of the captured home reference for `axis`.
        Returns 0.0 when no reference has been captured yet."""
        h = self._home_position_mm.get(axis)
        return float(h) if h is not None else 0.0

    def user_mm_to_abs_mm(self, mm_user: float, axis: Axis) -> float:
        """User-frame (home-relative, sign-flipped) mm → absolute machine-frame mm."""
        return (mm_user * self._axis_sign[axis]) + self._home_offset_abs(axis)

    def abs_mm_to_user_mm(self, mm_abs: float, axis: Axis) -> float:
        """Absolute machine-frame mm → user-frame mm (home-relative, sign-flipped)."""
        return (mm_abs - self._home_offset_abs(axis)) * self._axis_sign[axis]

    def mm_user_to_units(self, mm_user: float, axis: Axis) -> float:
        """User-frame mm → controller units. Applies sign flip AND home offset."""
        return mm_to_units(self.user_mm_to_abs_mm(mm_user, axis), axis)

    def units_to_mm_user(self, units: float, axis: Axis) -> float:
        """Controller units → user-frame mm. Inverse of mm_user_to_units."""
        return self.abs_mm_to_user_mm(units_to_mm(units, axis), axis)

    def vel_units_to_user_mm_s(self, units_per_s: float, axis: Axis) -> float:
        """Velocity in controller units/s → user-frame mm/s. Only the sign
        flip applies; the home offset is a static translation that drops out
        of any rate quantity."""
        return units_to_mm(units_per_s, axis) * self._axis_sign[axis]

    def mm_user_to_mm_firmware(self, mm_user: float, axis: Axis) -> float:
        """User-frame mm → absolute machine-frame mm. Alias for user_mm_to_abs_mm,
        named for use at call sites that feed mm (not units) into the SDK helpers
        (move_to_xyz_mm uses mm and converts via mm_to_units internally)."""
        return self.user_mm_to_abs_mm(mm_user, axis)

    def mm_firmware_to_mm_user(self, mm_fw: float, axis: Axis) -> float:
        """Absolute machine-frame mm → user-frame mm. Inverse alias."""
        return self.abs_mm_to_user_mm(mm_fw, axis)

    def jog_direction_user_to_firmware(self, direction: int, axis: Axis) -> int:
        """User-facing +/-1 jog direction → firmware-frame +/-1."""
        return direction * self._axis_sign[axis]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        mb: QMenuBar = self.menuBar()
        file_menu = mb.addMenu("&File")
        act_open = QAction("Open Waypoints…", self)
        act_open.triggered.connect(self._menu_open_waypoints)
        file_menu.addAction(act_open)
        act_save = QAction("Save Waypoints…", self)
        act_save.triggered.connect(self._menu_save_waypoints)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("&View")
        act_theme = QAction("Toggle Dark / Light", self)
        act_theme.triggered.connect(self._menu_toggle_theme)
        view_menu.addAction(act_theme)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._menu_about)
        help_menu.addAction(act_about)

    def _build_ui(self) -> None:
        """New top-level layout:

            connection bar           (full width, top)
            ┌──────────────┬───────────────────────────────────────┐
            │ left pane:   │ right pane: QTabWidget                │
            │  Live Status │  [Control] [Sequences] [Setup]        │
            │  Workspace   │  [Recording]                          │
            │   Map        │                                       │
            └──────────────┴───────────────────────────────────────┘
            global controls         (full width)
            status bar              (built into QMainWindow)
        """
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        # Top: connection bar.
        outer.addWidget(self._build_connection_bar())

        # Emergency-Stop banner (hidden until E-Stop fires).
        outer.addWidget(self._build_estop_banner())

        # Middle: horizontal splitter.
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_pane())
        splitter.addWidget(self._build_right_tabs())
        # 42/58 split at 1500 wide → roughly [630, 870].
        splitter.setSizes([630, 870])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

    @staticmethod
    def _make_h_rule() -> QFrame:
        """Thin horizontal rule used as a visual separator between pane sections."""
        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setFrameShadow(QFrame.Plain)
        rule.setStyleSheet("color: #3a3a40; margin: 0 4px;")
        rule.setFixedHeight(1)
        return rule

    def _build_left_pane(self) -> QWidget:
        # Inner widget holds the splitter; QScrollArea wraps it so the left
        # pane can scroll vertically when the window is shorter than the content.
        inner = QWidget()
        inner.setMinimumWidth(440)
        inner.setMinimumHeight(400)   # sum of section minimums — triggers scrollbar
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        # Polling freshness indicator sits above the splitter (not resizable).
        self.poll_indicator = QLabel("Polling: — (no data)")
        self.poll_indicator.setStyleSheet(
            "color: #8a8a8a; font-size: 11px; padding: 2px 4px;"
        )
        _fit_label(self.poll_indicator, "Polling: ✗ 9999 ms ago (STALE)")
        v.addWidget(self.poll_indicator)
        v.addWidget(self._make_h_rule())

        # axis_cards no longer maps to a visible table; kept as empty dict so
        # legacy call-sites that iterate over .values() are safe no-ops.
        self.axis_cards = {}

        # Vertical splitter: Fisheye Preview | Workspace Map.
        self._left_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self._left_splitter.setChildrenCollapsible(False)

        # ── section 1: fisheye preview ─────────────────────────────────────────
        self._fisheye_preview = FisheyePreviewWidget()
        self._fisheye_preview.setMinimumHeight(160)
        self._left_splitter.addWidget(self._fisheye_preview)

        # ── section 2: workspace map ───────────────────────────────────────────
        self.workspace_map = WorkspaceMap()
        self.workspace_map.setMinimumHeight(200)
        self._left_splitter.addWidget(self.workspace_map)

        # Restore splitter sizes from last session.
        saved_sizes = _gp_load_settings().get("gantry_panel", {}).get(
            "left_splitter_sizes"
        )
        if saved_sizes and len(saved_sizes) == 2:
            self._left_splitter.setSizes([int(s) for s in saved_sizes])
        else:
            self._left_splitter.setSizes([380, 400])

        self._left_splitter.splitterMoved.connect(self._on_left_splitter_moved)

        v.addWidget(self._left_splitter, stretch=1)

        # Scroll wrapper — horizontal scroll disabled; vertical scroll auto.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inner)
        return scroll

    def _build_right_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(False)
        # Wrap each tab's content in its own QScrollArea so a single tab can
        # scroll without affecting the left pane.
        def _wrap_scroll(*sections: QWidget) -> QScrollArea:
            holder = QWidget()
            inner = QVBoxLayout(holder)
            inner.setContentsMargins(8, 8, 8, 8)
            inner.setSpacing(12)
            for s in sections:
                inner.addWidget(s)
            inner.addStretch()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(holder)
            return scroll

        # Control tab: per-axis cards on top, combined move-to-target below.
        tabs.addTab(_wrap_scroll(self._build_per_axis_group(), self._build_move_panel()),
                    "Control")
        # Sequences tab: waypoint table + its toolbar.
        tabs.addTab(_wrap_scroll(self._build_waypoint_panel()), "Sequences")
        # Setup tab: soft limits + homing.
        tabs.addTab(_wrap_scroll(self._build_home_reference_group()), "Setup")
        # Recording tab: start/stop + CSV path + live 30s plot.
        tabs.addTab(_wrap_scroll(self._build_recording_panel()), "Recording")
        # Experiment tab: end-to-end orchestration.
        tabs.addTab(_wrap_scroll(self._build_experiment_tab()), "Experiment")
        self.tabs = tabs
        return tabs

    def _build_estop_banner(self) -> QFrame:
        """Yellow banner shown across the top whenever the E-Stop fires.
        Hidden by default; the user must click Reset to dismiss it."""
        banner = QFrame()
        banner.setObjectName("EstopBanner")
        banner.setStyleSheet(
            "QFrame#EstopBanner { background-color: #ffd54f; border: 2px solid #ff8f00;"
            " border-radius: 8px; }"
            "QFrame#EstopBanner QLabel { color: #1a1a1d; font-weight: 700;"
            " font-size: 13px; background: transparent; }"
            "QFrame#EstopBanner QPushButton {"
            " background-color: #ff8f00; color: white; border: 1px solid #ff6f00;"
            " border-radius: 5px; padding: 6px 14px; font-weight: 700; }"
            "QFrame#EstopBanner QPushButton:hover { background-color: #ffa726; }"
        )
        h = QHBoxLayout(banner)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(12)
        self._estop_banner_label = QLabel("EMERGENCY STOP triggered — click Reset to resume")
        h.addWidget(self._estop_banner_label, stretch=1)
        reset_btn = QPushButton("Reset E-Stop")
        reset_btn.clicked.connect(self._reset_estop)
        h.addWidget(reset_btn)
        self._estop_banner = banner
        banner.hide()
        return banner

    def _build_connection_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("SectionCard")
        frame.setStyleSheet("QFrame#SectionCard { padding: 6px; }")
        v = QVBoxLayout(frame)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)

        # ── Row 1: Gantry ─────────────────────────────────────────────────────
        gantry_row = QHBoxLayout()
        gantry_row.setSpacing(8)

        gantry_lbl = QLabel("Gantry")
        gantry_lbl.setStyleSheet("font-weight: 600; color: #aaa; min-width: 52px;")
        gantry_row.addWidget(gantry_lbl)

        gantry_row.addWidget(QLabel("IP"))
        self.ip_edit = QLineEdit("192.168.0.30")
        gantry_row.addWidget(self.ip_edit)

        gantry_row.addWidget(QLabel("Port"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8088)
        gantry_row.addWidget(self.port_spin)

        gantry_row.addWidget(QLabel("ID"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 31)
        self.id_spin.setValue(1)
        gantry_row.addWidget(self.id_spin)

        for widget, sample in [
            (self.ip_edit,   "192.168.000.000"),
            (self.port_spin, "65535"),
            (self.id_spin,   "999"),
        ]:
            fm = QtGui.QFontMetrics(widget.font())
            widget.setMinimumWidth(fm.horizontalAdvance(sample) + 32)
            widget.setMinimumHeight(24)

        self.connect_btn = QPushButton("Connect")
        ic = _icon("fa5s.plug")
        if ic is not None:
            self.connect_btn.setIcon(ic)
        self.connect_btn.clicked.connect(self._toggle_connection)
        gantry_row.addWidget(self.connect_btn)

        gantry_row.addSpacing(16)
        gantry_row.addWidget(QLabel("Enabled axes:"))
        self.axis_enable_checks: dict[Axis, QCheckBox] = {}
        for axis in AXES:
            cb = QCheckBox(axis.name)
            cb.setChecked(True)
            self.axis_enable_checks[axis] = cb
            gantry_row.addWidget(cb)

        gantry_row.addStretch()
        self.estop_btn = QPushButton("⚠  EMERGENCY STOP ALL")
        self.estop_btn.setObjectName("EmergencyButton")
        _size_button(self.estop_btn)
        ic_stop = _icon("fa5s.stop-circle")
        if ic_stop is not None:
            self.estop_btn.setIcon(ic_stop)
        self.estop_btn.clicked.connect(self._emergency_stop_all)
        self.estop_btn.setEnabled(False)
        gantry_row.addWidget(self.estop_btn)

        self.conn_status_label = QLabel("● Disconnected")
        self.conn_status_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        gantry_row.addWidget(self.conn_status_label)

        v.addLayout(gantry_row)

        # ── Row 2: Camera ─────────────────────────────────────────────────────
        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)

        cam_lbl = QLabel("Camera")
        cam_lbl.setStyleSheet("font-weight: 600; color: #aaa; min-width: 52px;")
        cam_row.addWidget(cam_lbl)

        cam_row.addWidget(QLabel("Device"))
        self._cam_device_spin = QSpinBox()
        self._cam_device_spin.setRange(0, 10)
        self._cam_device_spin.setValue(0)
        fm_d = QtGui.QFontMetrics(self._cam_device_spin.font())
        self._cam_device_spin.setMinimumWidth(fm_d.horizontalAdvance("99") + 32)
        self._cam_device_spin.setMinimumHeight(24)
        cam_row.addWidget(self._cam_device_spin)

        self._cam_res_combo = QComboBox()
        for res in ["1280×720", "1920×1080", "640×480"]:
            self._cam_res_combo.addItem(res)
        cam_row.addWidget(self._cam_res_combo)

        cam_row.addWidget(QLabel("FPS"))
        self._cam_fps_spin = QSpinBox()
        self._cam_fps_spin.setRange(1, 120)
        self._cam_fps_spin.setValue(30)
        fm_f = QtGui.QFontMetrics(self._cam_fps_spin.font())
        self._cam_fps_spin.setMinimumWidth(fm_f.horizontalAdvance("120") + 32)
        self._cam_fps_spin.setMinimumHeight(24)
        cam_row.addWidget(self._cam_fps_spin)

        cam_row.addWidget(QLabel("Calib:"))
        self._cam_calib_edit = QLineEdit()
        self._cam_calib_edit.setPlaceholderText("fisheye_calibration.yaml")
        self._cam_calib_edit.setMinimumWidth(180)
        self._cam_calib_edit.setMinimumHeight(24)
        cam_row.addWidget(self._cam_calib_edit, stretch=1)
        _cam_browse_btn = QPushButton("…")
        _cam_browse_btn.setFixedWidth(30)
        _cam_browse_btn.setMinimumHeight(24)
        _cam_browse_btn.clicked.connect(self._cam_browse_calib)
        cam_row.addWidget(_cam_browse_btn)

        self._cam_connect_btn = QPushButton("Connect Camera")
        self._cam_connect_btn.clicked.connect(self._toggle_camera)
        cam_row.addWidget(self._cam_connect_btn)

        cam_row.addStretch()
        self._cam_status_label = QLabel("● Disconnected")
        self._cam_status_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        cam_row.addWidget(self._cam_status_label)

        v.addLayout(cam_row)

        # Restore camera settings from last session.
        saved_cam = _gp_load_settings().get("gantry_panel", {}).get("camera", {})
        if saved_cam.get("device") is not None:
            self._cam_device_spin.setValue(int(saved_cam["device"]))
        if saved_cam.get("resolution"):
            idx = self._cam_res_combo.findText(saved_cam["resolution"])
            if idx >= 0:
                self._cam_res_combo.setCurrentIndex(idx)
        if saved_cam.get("fps"):
            self._cam_fps_spin.setValue(int(saved_cam["fps"]))
        if saved_cam.get("calib_path"):
            self._cam_calib_edit.setText(saved_cam["calib_path"])
        else:
            # No saved value: default to the repo's standard calibration file
            # (relative path; resolved against the repo root at use sites).
            self._cam_calib_edit.setText("config/fisheye_calibration.yaml")

        return frame

    def _build_home_reference_group(self) -> SectionFrame:
        """Setup-tab section. Just two things:
          (1) Set Current as Home Reference button — captures the current
              absolute XYZ as the home reference; no motion is commanded.
          (2) Axis Direction toggles — per-axis sign flip for jog/display.
        """
        frame = SectionFrame("Home Reference")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        v.addWidget(_make_note_label(
            "Sets the panel's user-frame origin (0 mm) to wherever the gantry "
            "currently is. After capture, the per-axis position readouts and "
            "all motion targets read as 'distance from home'. No motion is "
            "commanded — manually jog to where you want home to be, then click."
        ))

        set_home_btn = QPushButton("Set Current as Home Reference")
        set_home_btn.setObjectName("PrimaryButton")
        _size_button(set_home_btn, "primary")
        set_home_btn.setToolTip(
            "Mark the current XYZ position as the home reference (no motion).\n"
            "User-frame readouts on each axis card snap to 0.000 mm. The blue\n"
            "'Go Home' buttons on the Control tab return to this reference."
        )
        set_home_btn.clicked.connect(self._set_current_as_home_reference)
        v.addWidget(set_home_btn)

        # ---- Return to Home Reference --------------------------------------
        phys_title = QLabel("Return to Home Reference")
        phys_title.setStyleSheet(
            "color: #4ea1ff; font-weight: 600; padding-top: 12px;"
        )
        v.addWidget(phys_title)
        v.addWidget(_make_note_label(
            "Drives each axis sequentially (X → Y → Z) back to the home "
            "reference set above. Uses per-axis absolute moves at the speed/"
            "acc/dec from the Per-Axis Control card, so it works even when "
            "the controller is in status 664. Axes with no home reference "
            "yet are skipped."
        ))

        home_row = QHBoxLayout()
        home_row.addStretch()
        self._home_all_btn = QPushButton("Go to Home")
        self._home_all_btn.setObjectName("PrimaryButton")
        _size_button(self._home_all_btn, "primary")
        self._home_all_btn.clicked.connect(self._start_go_home)
        home_row.addWidget(self._home_all_btn)
        v.addLayout(home_row)

        # ---- Axis Direction sub-group --------------------------------------
        # Persisted in ~/.umi_gui_state.json under axis_sign.
        dir_title = QLabel("Axis Direction")
        dir_title.setStyleSheet(
            "color: #4ea1ff; font-weight: 600; padding-top: 12px;"
        )
        v.addWidget(dir_title)
        v.addWidget(_make_note_label(
            "If clicking X+ in this panel makes the gantry move in what you "
            "consider the negative direction, toggle that axis to -1. The "
            "panel will invert this axis's user-facing direction. Does NOT "
            "command any test motion — toggle, jog manually, observe."
        ))

        dir_row = QHBoxLayout()
        dir_row.setSpacing(16)
        self.axis_sign_combos: dict[Axis, QComboBox] = {}
        for axis in AXES:
            sub = QHBoxLayout()
            sub.setSpacing(4)
            sub.addWidget(QLabel(f"{axis.name}:"))
            combo = QComboBox()
            combo.addItem("+1 (matches firmware)",  1)
            combo.addItem("-1 (inverted)",         -1)
            combo.setCurrentIndex(0 if self._axis_sign[axis] >= 0 else 1)
            combo.setToolTip(
                f"Direction multiplier for axis {axis.name}.\n"
                "+1: panel X+ jog → firmware counter increases (normal).\n"
                "-1: panel X+ jog → firmware counter decreases (inverted).\n"
                "Applied to jog / Move Abs / Move to Target AND to the\n"
                "position readback before display."
            )
            combo.currentIndexChanged.connect(partial(self._on_axis_sign_changed, axis))
            sub.addWidget(combo)
            dir_row.addLayout(sub)
            self.axis_sign_combos[axis] = combo
        dir_row.addStretch()
        v.addLayout(dir_row)
        return frame

    def _on_axis_sign_changed(self, axis: Axis, idx: int) -> None:
        combo = self.axis_sign_combos.get(axis)
        new_sign = int(combo.itemData(idx)) if combo is not None else 1
        new_sign = 1 if new_sign >= 0 else -1
        if self._axis_sign[axis] == new_sign:
            return
        self._axis_sign[axis] = new_sign
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["axis_sign"] = {a.name: int(self._axis_sign[a]) for a in AXES}
        _gp_save_section("gantry_panel", payload)
        self.sb_op_label.setText(
            f"Axis {axis.name} direction set to {new_sign:+d}. "
            "Jog manually to verify."
        )

    def _build_per_axis_group(self) -> SectionFrame:
        """Per-axis cards: hold-to-jog, Move Abs (mm), and per-axis Home shortcut.
        Shared jog/move parameter row at the top (independent from the combined
        Move-to-Target panel's speed/accel/decel)."""
        frame = SectionFrame("Per-Axis Control  (mm)")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # Shared jog/move parameter row.
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Speed (cm/s)"))
        self.jog_speed_spin = QDoubleSpinBox()
        self.jog_speed_spin.setRange(0.10, 200.00)
        self.jog_speed_spin.setDecimals(2)
        self.jog_speed_spin.setSingleStep(0.50)
        self.jog_speed_spin.setValue(10.0)
        _cx = cm_s_to_units_s(10.0, Axis.X)
        _cy = cm_s_to_units_s(10.0, Axis.Y)
        _cz = cm_s_to_units_s(10.0, Axis.Z)
        self.jog_speed_spin.setToolTip(
            f"Jog/Move speed.\nInternally: X={_cx:.2f} units/s, Y={_cy:.2f} units/s, Z={_cz:.2f} units/s.\n"
            "Calibration: SCALE_MM_PER_UNIT from gantry_runner.py (X=8.25, Y=2.5, Z=0.5 mm/unit)."
        )
        params_row.addWidget(self.jog_speed_spin)

        params_row.addSpacing(8)
        params_row.addWidget(QLabel("Acc (cm/s²)"))
        self.jog_acc_spin = QDoubleSpinBox()
        self.jog_acc_spin.setRange(0.10, 500.00)
        self.jog_acc_spin.setDecimals(2)
        self.jog_acc_spin.setSingleStep(0.50)
        self.jog_acc_spin.setValue(5.0)
        params_row.addWidget(self.jog_acc_spin)

        params_row.addSpacing(8)
        params_row.addWidget(QLabel("Dec (cm/s²)"))
        self.jog_dec_spin = QDoubleSpinBox()
        self.jog_dec_spin.setRange(0.10, 500.00)
        self.jog_dec_spin.setDecimals(2)
        self.jog_dec_spin.setSingleStep(0.50)
        self.jog_dec_spin.setValue(5.0)
        params_row.addWidget(self.jog_dec_spin)
        params_row.addStretch()
        self.refresh_btn = QPushButton("🔄 Refresh")
        _size_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh position readout from controller")
        self.refresh_btn.clicked.connect(self._refresh_position)
        self.refresh_btn.setEnabled(False)
        params_row.addWidget(self.refresh_btn)
        v.addLayout(params_row)

        for spin in (self.jog_speed_spin, self.jog_acc_spin, self.jog_dec_spin):
            spin.setMinimumHeight(28)

        # Three cards side-by-side.
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)
        for axis in AXES:
            cards_row.addWidget(self._build_per_axis_card(axis), stretch=1)
        v.addLayout(cards_row)
        return frame

    def _build_per_axis_card(self, axis: Axis) -> QFrame:
        card = QFrame()
        card.setObjectName("AxisCard")
        card.setFrameShape(QFrame.NoFrame)
        v = QVBoxLayout(card)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)

        # Row 1: axis letter + position readout side by side.
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        letter = QLabel(axis.name)
        letter.setStyleSheet("font-size: 16px; font-weight: 700; color: #4ea1ff;")
        letter.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        header_row.addWidget(letter)
        pos_mm = QLabel("--")
        pos_mm.setObjectName("PositionReadout")
        _fit_label(pos_mm, "-9999.999")
        header_row.addWidget(pos_mm, stretch=1)
        v.addLayout(header_row)

        # Row 2: home status + velocity on one line.
        info_row = QHBoxLayout()
        info_row.setSpacing(4)
        home_label = QLabel("⚠ no home")
        home_label.setStyleSheet("color: #ffa726; font-size: 11px;")
        info_row.addWidget(home_label)
        info_row.addStretch()
        vel = QLabel("Vel: -- cm/s")
        vel.setStyleSheet("color: #aaa; font-size: 11px;")
        _fit_label(vel, "Vel: -9999.99 cm/s")
        info_row.addWidget(vel)
        v.addLayout(info_row)

        # Row 3: hold-to-jog buttons.
        jog_row = QHBoxLayout()
        jog_row.setSpacing(6)
        btn_pos = QPushButton(f"{axis.name}+")
        btn_neg = QPushButton(f"{axis.name}-")
        for jb in (btn_pos, btn_neg):
            jb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_pos.setToolTip(f"Hold to jog {axis.name} in the positive direction")
        btn_neg.setToolTip(f"Hold to jog {axis.name} in the negative direction")
        btn_pos.pressed.connect(partial(self._start_jog, axis, +1))
        btn_pos.released.connect(partial(self._stop_jog, axis))
        btn_neg.pressed.connect(partial(self._start_jog, axis, -1))
        btn_neg.released.connect(partial(self._stop_jog, axis))
        jog_row.addWidget(btn_pos)
        jog_row.addWidget(btn_neg)
        v.addLayout(jog_row)

        # Row 4: abs-move spinbox + Move on one line.
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)
        target_spin = QDoubleSpinBox()
        target_spin.setRange(-100000.0, 100000.0)
        target_spin.setDecimals(3)
        target_spin.setValue(0.0)
        target_spin.setMinimumHeight(28)
        bottom_row.addWidget(target_spin, stretch=1)
        move_btn = QPushButton("Move")
        move_btn.clicked.connect(partial(self._move_axis_abs, axis))
        bottom_row.addWidget(move_btn)
        v.addLayout(bottom_row)

        # Row 5: "Return to home reference (Δ home = 0)" — primary action,
        # promoted with the blue PrimaryButton style + larger size.
        v.addWidget(_make_note_label("Return to home reference (Δ home = 0)"))
        home_btn = QPushButton(f"Go Home  {axis.name} → 0.000 mm")
        home_btn.setObjectName("PrimaryButton")
        _size_button(home_btn, "primary")
        home_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        home_btn.setToolTip(
            f"Move {axis.name} back to its captured home reference (user-frame 0 mm).\n"
            "Uses the same axis-sign rules as Move to Target.\n\n"
            "Requires a home reference to be set first via Setup tab → "
            "'Set Current as Home Reference'."
        )
        ic = _icon("fa5s.home")
        if ic is not None:
            home_btn.setIcon(ic)
        home_btn.clicked.connect(partial(self._go_to_home_axis, axis))
        v.addWidget(home_btn)

        self.per_axis_cards[axis] = {
            "card": card,
            "pos_mm": pos_mm,
            "home_label": home_label,
            "vel": vel,
            "target_spin": target_spin,
            "move_btn": move_btn,
            "home_btn": home_btn,
            "jog_btns": [btn_pos, btn_neg],
        }
        return card

    def _build_move_panel(self) -> SectionFrame:
        frame = SectionFrame("Move to Target  (mm)")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.target_spins: dict[Axis, QDoubleSpinBox] = {}
        for col, axis in enumerate(AXES):
            grid.addWidget(QLabel(f"Target {axis.name} (mm)"), 0, col)
            sp = QDoubleSpinBox()
            sp.setRange(-100000.0, 100000.0)
            sp.setDecimals(3)
            sp.setValue(0.0)
            _size_mm_spinbox(sp)
            grid.addWidget(sp, 1, col)
            self.target_spins[axis] = sp

        grid.addWidget(QLabel("Speed (cm/s)"), 2, 0)
        self.move_speed_spin = QDoubleSpinBox()
        self.move_speed_spin.setRange(0.10, 200.00)
        self.move_speed_spin.setDecimals(2)
        self.move_speed_spin.setSingleStep(0.50)
        self.move_speed_spin.setValue(10.0)
        _mx = cm_s_to_units_s(10.0, Axis.X)
        _my = cm_s_to_units_s(10.0, Axis.Y)
        _mz = cm_s_to_units_s(10.0, Axis.Z)
        self.move_speed_spin.setToolTip(
            f"Internally: X={_mx:.2f} units/s, Y={_my:.2f} units/s, Z={_mz:.2f} units/s.\n"
            "Calibration: SCALE_MM_PER_UNIT from gantry_runner.py (X=8.25, Y=2.5, Z=0.5 mm/unit)."
        )
        _size_mm_spinbox(self.move_speed_spin)
        grid.addWidget(self.move_speed_spin, 3, 0)

        grid.addWidget(QLabel("Accel (cm/s²)"), 2, 1)
        self.move_acc_spin = QDoubleSpinBox()
        self.move_acc_spin.setRange(0.10, 500.00)
        self.move_acc_spin.setDecimals(2)
        self.move_acc_spin.setSingleStep(0.50)
        self.move_acc_spin.setValue(5.0)
        _size_mm_spinbox(self.move_acc_spin)
        grid.addWidget(self.move_acc_spin, 3, 1)

        grid.addWidget(QLabel("Decel (cm/s²)"), 2, 2)
        self.move_dec_spin = QDoubleSpinBox()
        self.move_dec_spin.setRange(0.10, 500.00)
        self.move_dec_spin.setDecimals(2)
        self.move_dec_spin.setSingleStep(0.50)
        self.move_dec_spin.setValue(5.0)
        _size_mm_spinbox(self.move_dec_spin)
        grid.addWidget(self.move_dec_spin, 3, 2)

        # Ensure move speed/accel/decel spinboxes are wide enough.
        for spin in (self.move_speed_spin, self.move_acc_spin, self.move_dec_spin):
            fm = QtGui.QFontMetrics(spin.font())
            spin.setMinimumWidth(fm.horizontalAdvance("9999.99") + 32)
            spin.setMinimumHeight(28)

        grid.addWidget(QLabel("Mode"), 4, 0)
        self.move_mode_combo = QComboBox()
        self.move_mode_combo.addItem("line", "line")
        self.move_mode_combo.addItem("sequential", "sequential")
        self.move_mode_combo.setMinimumHeight(28)
        grid.addWidget(self.move_mode_combo, 5, 0)
        v.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.move_btn = QPushButton("Move to Target")
        self.move_btn.setObjectName("PrimaryButton")
        _size_button(self.move_btn, "primary")
        self.move_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ic_move = _icon("fa5s.arrows-alt")
        if ic_move is not None:
            self.move_btn.setIcon(ic_move)
        self.move_btn.clicked.connect(self._move_to_target)
        btn_row.addWidget(self.move_btn, stretch=2)

        self.use_current_btn = QPushButton("Use Current as Target")
        _size_button(self.use_current_btn)
        self.use_current_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.use_current_btn.clicked.connect(self._use_current_as_target)
        btn_row.addWidget(self.use_current_btn, stretch=1)

        self.cancel_move_btn = QPushButton("Cancel Move")
        _size_button(self.cancel_move_btn)
        self.cancel_move_btn.clicked.connect(self._cancel_move)
        self.cancel_move_btn.setEnabled(False)
        btn_row.addWidget(self.cancel_move_btn, stretch=1)
        v.addLayout(btn_row)
        return frame

    def _build_recording_panel(self) -> SectionFrame:
        frame = SectionFrame("Recording")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        h = QHBoxLayout()
        self.record_btn = QPushButton("● Start Recording")
        self.record_btn.setObjectName("record")
        self.record_btn.setCheckable(True)
        ic_rec = _icon("fa5s.circle")
        if ic_rec is not None:
            self.record_btn.setIcon(ic_rec)
        self.record_btn.toggled.connect(self._toggle_recording)
        h.addWidget(self.record_btn)

        self.csv_path_label = QLabel("(no active CSV)")
        self.csv_path_label.setStyleSheet("color: #888;")
        self.csv_path_label.setCursor(Qt.PointingHandCursor)
        self.csv_path_label.mousePressEvent = self._open_csv_folder  # type: ignore[assignment]
        h.addWidget(self.csv_path_label, stretch=1)
        v.addLayout(h)

        self.plot_widget = LivePlotWidget()
        self.plot_widget.setMinimumHeight(180)
        v.addWidget(self.plot_widget)
        return frame

    # ------------------------------------------------------------------
    # Experiment tab
    # ------------------------------------------------------------------
    def _build_experiment_tab(self) -> QWidget:
        """Build the Experiment orchestration tab."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        from PyQt5.QtWidgets import QRadioButton, QButtonGroup

        # ── (0) Camera mode selector ─────────────────────────────────────────
        # Two operating modes:
        #   - "fisheye": full pipeline; requires camera + calib.
        #   - "gantry_only": telemetry-only test mode; skips fisheye entirely.
        cam_mode_frame = SectionFrame("Camera Mode")
        cm_layout = QVBoxLayout(cam_mode_frame.content())
        cm_layout.setContentsMargins(0, 0, 0, 0)
        cm_layout.setSpacing(4)
        self._exp_cam_mode_fisheye_radio = QRadioButton(
            "With fisheye (full pipeline)"
        )
        self._exp_cam_mode_gantry_radio = QRadioButton(
            "Gantry only (no camera)"
        )
        self._exp_cam_mode_fisheye_radio.setToolTip(
            "Record gantry telemetry + fisheye AprilTag SLAM. Camera + calibration required."
        )
        self._exp_cam_mode_gantry_radio.setToolTip(
            "Record gantry telemetry only. Camera + calibration are not required and "
            "are not used. Useful for motion + telemetry testing."
        )
        cam_mode_bg = QButtonGroup(self)
        cam_mode_bg.addButton(self._exp_cam_mode_fisheye_radio)
        cam_mode_bg.addButton(self._exp_cam_mode_gantry_radio)
        cm_layout.addWidget(self._exp_cam_mode_fisheye_radio)
        cm_layout.addWidget(self._exp_cam_mode_gantry_radio)
        saved_cam_mode = (_gp_load_settings().get("gantry_panel", {})
                          .get("camera_mode", "fisheye"))
        if saved_cam_mode == "gantry_only":
            self._exp_cam_mode_gantry_radio.setChecked(True)
        else:
            self._exp_cam_mode_fisheye_radio.setChecked(True)
        self._exp_cam_mode_fisheye_radio.toggled.connect(self._on_exp_cam_mode_changed)
        self._exp_cam_mode_gantry_radio.toggled.connect(self._on_exp_cam_mode_changed)
        outer.addWidget(cam_mode_frame)

        # ── (a) Pre-flight checklist ──────────────────────────────────────────
        chk_frame = SectionFrame("Pre-flight Checklist")
        chk_layout = QVBoxLayout(chk_frame.content())
        chk_layout.setContentsMargins(0, 0, 0, 0)
        chk_layout.setSpacing(4)

        def _chk_row(label: str) -> tuple[QLabel, QLabel]:
            row = QHBoxLayout()
            status = QLabel("●")
            status.setFixedWidth(18)
            status.setStyleSheet("color: #ef5350; font-weight: bold;")
            lbl = QLabel(label)
            row.addWidget(status)
            row.addWidget(lbl, stretch=1)
            chk_layout.addLayout(row)
            return status, lbl

        self._exp_chk_connected,   _ = _chk_row("Controller connected")
        self._exp_chk_homed,       _ = _chk_row("Home reference set (X, Y, Z)")
        self._exp_chk_path,        _ = _chk_row("Path defined: 0 waypoints")
        self._exp_chk_camera,      _ = _chk_row("Camera connected (see connection bar)")
        self._exp_chk_calib,       _ = _chk_row("Fisheye calibration loaded")
        self._exp_chk_tagsize,     _ = _chk_row("Tag size configured")

        # Camera summary — read-only label showing what the experiment will use.
        self._exp_camera_summary = QLabel("Camera: not connected — use top connection bar")
        self._exp_camera_summary.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 4px;"
        )
        self._exp_camera_summary.setWordWrap(True)
        chk_layout.addWidget(self._exp_camera_summary)

        self._exp_start_btn = QPushButton("▶  Start Experiment")
        self._exp_start_btn.setObjectName("PrimaryButton")
        _size_button(self._exp_start_btn, "primary")
        self._exp_start_btn.setEnabled(False)
        self._exp_start_btn.setToolTip("All checklist items must be green to start.")
        self._exp_start_btn.clicked.connect(self._exp_start)
        chk_layout.addWidget(self._exp_start_btn)

        outer.addWidget(chk_frame)

        # ── (b) Path source ───────────────────────────────────────────────────
        path_frame = SectionFrame("Path Source")
        path_layout = QVBoxLayout(path_frame.content())
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(6)

        self._exp_path_seq_radio = QRadioButton("Use Sequences tab waypoints (current)")
        self._exp_path_csv_radio = QRadioButton("Use CSV file")
        self._exp_path_seq_radio.setChecked(True)
        path_bg = QButtonGroup(self)
        path_bg.addButton(self._exp_path_seq_radio)
        path_bg.addButton(self._exp_path_csv_radio)
        path_layout.addWidget(self._exp_path_seq_radio)

        csv_row = QHBoxLayout()
        csv_row.addWidget(self._exp_path_csv_radio)
        self._exp_path_csv_edit = QLineEdit()
        self._exp_path_csv_edit.setPlaceholderText("x_mm,y_mm,z_mm,speed_cm_s,dwell_s")
        self._exp_path_csv_edit.setEnabled(False)
        csv_row.addWidget(self._exp_path_csv_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        _size_button(browse_btn)
        browse_btn.clicked.connect(self._exp_browse_csv)
        csv_row.addWidget(browse_btn)
        path_layout.addLayout(csv_row)
        self._exp_path_csv_radio.toggled.connect(
            lambda checked: self._exp_path_csv_edit.setEnabled(checked)
        )
        outer.addWidget(path_frame)

        # ── (c) Experiment parameters ─────────────────────────────────────────
        param_frame = SectionFrame("Experiment Parameters")
        param_grid = QGridLayout(param_frame.content())
        param_grid.setContentsMargins(0, 0, 0, 0)
        param_grid.setSpacing(6)

        def _param_row(row: int, label: str, widget: QWidget) -> None:
            param_grid.addWidget(QLabel(label), row, 0)
            param_grid.addWidget(widget, row, 1)

        self._exp_countdown_spin = QDoubleSpinBox()
        self._exp_countdown_spin.setRange(0.0, 10.0)
        self._exp_countdown_spin.setValue(2.0)
        self._exp_countdown_spin.setSuffix(" s")
        _param_row(0, "Pre-motion countdown:", self._exp_countdown_spin)

        self._exp_settle_spin = QDoubleSpinBox()
        self._exp_settle_spin.setRange(0.0, 30.0)
        self._exp_settle_spin.setValue(2.0)
        self._exp_settle_spin.setSuffix(" s")
        _param_row(1, "Post-motion settle time:", self._exp_settle_spin)

        self._exp_idle_detect_chk = QCheckBox("Tag detection during countdown (pre-idle)")
        self._exp_idle_detect_chk.setChecked(True)
        param_grid.addWidget(self._exp_idle_detect_chk, 2, 0, 1, 2)

        self._exp_name_edit = QLineEdit()
        self._exp_name_edit.setPlaceholderText("auto-generated timestamp")
        _param_row(3, "Output folder name:", self._exp_name_edit)

        outer.addWidget(param_frame)

        # ── (d) TagSLAM settings ──────────────────────────────────────────────
        # Camera hardware (device/resolution/FPS/calib) is now in the top
        # connection bar.  Only experiment-specific SLAM knobs live here.
        cam_frame = SectionFrame("TagSLAM Settings")
        cam_grid = QGridLayout(cam_frame.content())
        cam_grid.setContentsMargins(0, 0, 0, 0)
        cam_grid.setSpacing(6)

        def _cam_row(row: int, label: str, widget: QWidget) -> None:
            cam_grid.addWidget(QLabel(label), row, 0)
            cam_grid.addWidget(widget, row, 1)

        self._exp_tag_family_edit = QLineEdit("tag36h11")
        _cam_row(0, "Tag family:", self._exp_tag_family_edit)
        # Keep the live-preview detector aligned with the chosen family.
        self._exp_tag_family_edit.editingFinished.connect(
            lambda: self._fisheye_preview.set_tag_family(
                self._exp_tag_family_edit.text().strip() or "tag36h11"
            ) if hasattr(self, "_fisheye_preview") else None
        )
        if hasattr(self, "_fisheye_preview"):
            self._fisheye_preview.set_tag_family(
                self._exp_tag_family_edit.text().strip() or "tag36h11"
            )

        self._exp_tag_size_spin = QDoubleSpinBox()
        self._exp_tag_size_spin.setRange(0.01, 1.0)
        self._exp_tag_size_spin.setValue(0.170)
        self._exp_tag_size_spin.setDecimals(3)
        self._exp_tag_size_spin.setSuffix(" m")
        _cam_row(1, "Tag size:", self._exp_tag_size_spin)

        self._exp_anchor_spin = QSpinBox()
        self._exp_anchor_spin.setRange(0, 255)
        self._exp_anchor_spin.setValue(1)
        _cam_row(2, "Anchor tag ID:", self._exp_anchor_spin)

        self._exp_water_combo = QComboBox()
        for mode in ["none", "scalar", "refractive"]:
            self._exp_water_combo.addItem(mode)
        _cam_row(3, "Water correction mode:", self._exp_water_combo)

        outer.addWidget(cam_frame)

        # ── (e) Live experiment status ────────────────────────────────────────
        live_frame = SectionFrame("Live Status")
        live_layout = QVBoxLayout(live_frame.content())
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(6)

        self._exp_state_label = QLabel("IDLE")
        self._exp_state_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; color: #4ea1ff;"
        )
        live_layout.addWidget(self._exp_state_label)

        prog_row = QHBoxLayout()
        self._exp_wp_bar = QProgressBar()
        self._exp_wp_bar.setRange(0, 1)
        self._exp_wp_bar.setValue(0)
        self._exp_wp_bar.setFormat("Waypoint %v / %m")
        prog_row.addWidget(self._exp_wp_bar, stretch=1)
        self._exp_time_bar = QProgressBar()
        self._exp_time_bar.setRange(0, 100)
        self._exp_time_bar.setValue(0)
        self._exp_time_bar.setFormat("Elapsed %p%")
        prog_row.addWidget(self._exp_time_bar, stretch=1)
        live_layout.addLayout(prog_row)

        stats_row = QHBoxLayout()
        self._exp_tags_frame_lbl = QLabel("Tags/frame: —")
        self._exp_tags_graph_lbl = QLabel("Tags/graph: —")
        self._exp_updates_lbl    = QLabel("Updates: —")
        self._exp_drift_lbl      = QLabel("Drift: — mm")
        for lbl in (self._exp_tags_frame_lbl, self._exp_tags_graph_lbl,
                    self._exp_updates_lbl, self._exp_drift_lbl):
            lbl.setStyleSheet("color: #aaa; font-size: 12px;")
            stats_row.addWidget(lbl)
        stats_row.addStretch()
        live_layout.addLayout(stats_row)

        stop_row = QHBoxLayout()
        self._exp_stop_btn = QPushButton("■  Stop Experiment")
        self._exp_stop_btn.setEnabled(False)
        self._exp_stop_btn.setStyleSheet(
            "QPushButton { background-color:#8b1a1a; color:white; border:1px solid #c04040;"
            " border-radius:5px; padding:6px 14px; font-weight:600; }"
            "QPushButton:hover { background-color:#b02222; }"
            "QPushButton:disabled { background-color:#2a2a2e; color:#666; }"
        )
        self._exp_stop_btn.clicked.connect(self._exp_stop)
        stop_row.addWidget(self._exp_stop_btn)

        self._exp_result_label = QLabel("")
        self._exp_result_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._exp_result_label.setWordWrap(True)
        stop_row.addWidget(self._exp_result_label, stretch=1)
        live_layout.addLayout(stop_row)

        outer.addWidget(live_frame)
        outer.addStretch()

        # Wire experiment runner if available
        if HAVE_EXPERIMENT_RUNNER:
            self._experiment_runner = ExperimentRunner(self)
            self._experiment_runner.state_changed.connect(self._exp_on_state_changed)
            self._experiment_runner.countdown_tick.connect(self._exp_on_countdown)
            self._experiment_runner.waypoint_progress.connect(self._exp_on_wp_progress)
            self._experiment_runner.fisheye_stats.connect(self._exp_on_fisheye_stats)
            self._experiment_runner.error.connect(
                lambda msg: self._exp_result_label.setText(f"⚠ {msg}")
            )
            self._experiment_runner.finished.connect(self._exp_on_finished)

        # Refresh checklist once UI is built
        QTimer.singleShot(200, self._exp_refresh_checklist)
        return container

    # ── Experiment helpers ────────────────────────────────────────────────────

    def _exp_refresh_checklist(self) -> None:
        """Update the checklist indicators and enable/disable Start.

        When Camera Mode == 'gantry_only', the Camera and Calibration rows show
        '— (skipped)' in gray and do not gate the Start button.
        """
        def _set(indicator: QLabel, ok: bool, text: str = "") -> None:
            indicator.setText("●")
            indicator.setStyleSheet(
                "color: #34d058; font-weight: bold;" if ok
                else "color: #ef5350; font-weight: bold;"
            )

        def _set_skipped(indicator: QLabel) -> None:
            indicator.setText("—")
            indicator.setStyleSheet("color: #888; font-weight: bold;")

        connected = self.connected
        _set(self._exp_chk_connected, connected)

        homed = all(self._home_position_mm.get(a) is not None for a in AXES)
        _set(self._exp_chk_homed, homed)

        n_wp = self._exp_count_waypoints()
        path_ok = n_wp > 0
        self._exp_chk_path.parentWidget()  # just in case
        _set(self._exp_chk_path, path_ok)
        # Update label text to show count
        try:
            self._exp_chk_path.setText("●")
            # Find the sibling QLabel
            for lbl in self._exp_chk_path.parentWidget().findChildren(QLabel):
                if "waypoints" in lbl.text():
                    lbl.setText(f"Path defined: {n_wp} waypoints")
                    break
        except Exception:
            pass

        gantry_only = self._exp_camera_mode() == "gantry_only"

        if gantry_only:
            _set_skipped(self._exp_chk_camera)
            _set_skipped(self._exp_chk_calib)
            cam_ok = True
            calib_ok = True
            # Mark camera/calib labels as '(skipped)' for clarity.
            try:
                for indicator, key in (
                    (self._exp_chk_camera, "Camera"),
                    (self._exp_chk_calib,  "calibration"),
                ):
                    for lbl in indicator.parentWidget().findChildren(QLabel):
                        if key in lbl.text() and "(skipped)" not in lbl.text():
                            lbl.setText(lbl.text() + "  — (skipped)")
                            lbl.setStyleSheet("color: #888;")
            except Exception:
                pass
        else:
            cam_ok = (
                (self._camera is not None and self._camera.is_open)
                or self._is_mock_camera
            )
            _set(self._exp_chk_camera, cam_ok)
            # Effective calib path: the connected session's path if available,
            # else the (repo-root-resolved) Experiment/connection-bar edit text.
            _calib_eff = (
                self._camera.calib_path
                if (self._camera is not None and self._camera.calib_path is not None)
                else _resolve_calib_path(self._cam_calib_edit.text())
            )
            if self._is_mock_camera:
                calib_ok = True
            elif _calib_eff is not None and _calib_eff.exists():
                calib_ok = True
            else:
                calib_ok = False
            _set(self._exp_chk_calib, calib_ok)
            # Clear file-not-found hint when the path is set but missing.
            if not calib_ok and not self._is_mock_camera:
                try:
                    for lbl in self._exp_chk_calib.parentWidget().findChildren(QLabel):
                        if "calibration" in lbl.text().lower():
                            miss = "" if _calib_eff is None else f" — not found: {_calib_eff}"
                            lbl.setText(f"Fisheye calibration{miss}  (run calibrate_fisheye.py)")
                            lbl.setStyleSheet("color: #ef5350;")
                            break
                except Exception:
                    pass
            # Restore base labels (strip any "(skipped)" suffix).
            try:
                for indicator, base in (
                    (self._exp_chk_camera, "Camera connected (see connection bar)"),
                    (self._exp_chk_calib,  "Fisheye calibration loaded"),
                ):
                    for lbl in indicator.parentWidget().findChildren(QLabel):
                        if "(skipped)" in lbl.text() or lbl.text().startswith(base.split()[0]):
                            lbl.setText(base)
                            lbl.setStyleSheet("")
                            break
            except Exception:
                pass

        tag_size_ok = self._exp_tag_size_spin.value() > 0
        _set(self._exp_chk_tagsize, tag_size_ok)

        all_ok = all([connected, homed, path_ok, cam_ok, calib_ok, tag_size_ok])
        self._exp_start_btn.setEnabled(all_ok and not self._experiment_in_progress
                                       and HAVE_EXPERIMENT_RUNNER)
        missing = []
        if not connected:   missing.append("connect controller")
        if not homed:       missing.append("set home reference")
        if not path_ok:     missing.append("add waypoints")
        if not gantry_only and not cam_ok:    missing.append("connect camera (top bar)")
        if not gantry_only and not calib_ok:  missing.append("set calib path (top bar)")
        if all_ok:
            if gantry_only:
                tip = "Will record gantry telemetry only. Camera disabled."
            else:
                tip = ("Will record gantry telemetry + fisheye AprilTag SLAM. "
                       "Camera connected.")
        else:
            tip = "Missing: " + ", ".join(missing)
        self._exp_start_btn.setToolTip(tip)

    def _exp_camera_mode(self) -> str:
        """'fisheye' or 'gantry_only'."""
        if getattr(self, "_exp_cam_mode_gantry_radio", None) is not None \
                and self._exp_cam_mode_gantry_radio.isChecked():
            return "gantry_only"
        return "fisheye"

    def _on_exp_cam_mode_changed(self, _checked: bool = False) -> None:
        # Persist + refresh checklist + update preview placeholder + hide tag stats.
        mode = self._exp_camera_mode()
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["camera_mode"] = mode
        _gp_save_section("gantry_panel", payload)
        gantry_only = mode == "gantry_only"
        # Live experiment stats panel: hide tag-related counters in gantry-only.
        for lbl in (getattr(self, "_exp_tags_frame_lbl", None),
                    getattr(self, "_exp_tags_graph_lbl", None),
                    getattr(self, "_exp_updates_lbl", None),
                    getattr(self, "_exp_drift_lbl", None)):
            if lbl is not None:
                lbl.setVisible(not gantry_only)
        # Fisheye preview placeholder.
        if gantry_only and getattr(self, "_fisheye_preview", None) is not None:
            self._fisheye_preview.preview_label.clear()
            self._fisheye_preview.preview_label.setText(
                "Camera not in use for this experiment"
            )
            self._fisheye_preview.preview_label.setStyleSheet(
                "background-color: #0e0e0e; color: #888; border-radius: 4px;"
            )
        self._exp_refresh_checklist()

    def _exp_count_waypoints(self) -> int:
        if self._exp_path_seq_radio.isChecked():
            return self.waypoint_table.rowCount()
        csv_path = Path(self._exp_path_csv_edit.text().strip())
        if not csv_path.exists():
            return 0
        try:
            import csv as _csv
            with csv_path.open(newline="") as fh:
                return sum(1 for _ in _csv.DictReader(fh))
        except Exception:
            return 0

    def _exp_browse_csv(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Waypoint CSV", "", "CSV Files (*.csv)"
        )
        if p:
            self._exp_path_csv_edit.setText(p)

    def _send_waypoints_to_experiment(self) -> None:
        """Copy Sequences tab waypoints to Experiment tab path source, switch tab."""
        self._exp_path_seq_radio.setChecked(True)
        # Switch to Experiment tab (index 4)
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(4)
        self._exp_refresh_checklist()

    def _exp_autopick_anchor_tag(self) -> None:
        """Pick the AprilTag whose center is closest to the image center and
        write it into the anchor-tag spin box. No-op if no tags are visible
        in the live preview — the spin value the user typed is kept."""
        if not hasattr(self, "_fisheye_preview"):
            return
        tags, image_size = self._fisheye_preview.latest_detections()
        if not tags or image_size == (0, 0):
            self._exp_result_label.setText(
                "⚠ Anchor auto-pick: no AprilTags visible — using manual ID "
                f"{self._exp_anchor_spin.value()}"
            )
            self._exp_result_label.setStyleSheet(
                "color: #ffa726; font-size: 11px;"
            )
            return
        img_cx = image_size[0] / 2.0
        img_cy = image_size[1] / 2.0
        best = min(
            tags,
            key=lambda t: (t["center"][0] - img_cx) ** 2
                          + (t["center"][1] - img_cy) ** 2,
        )
        chosen_id = int(best["id"])
        if chosen_id != int(self._exp_anchor_spin.value()):
            self._exp_anchor_spin.setValue(chosen_id)
        dx = best["center"][0] - img_cx
        dy = best["center"][1] - img_cy
        dist_px = (dx * dx + dy * dy) ** 0.5
        print(
            f"[exp] anchor auto-pick: tag {chosen_id} "
            f"at ({best['center'][0]:.0f}, {best['center'][1]:.0f}), "
            f"{dist_px:.0f}px from center",
            file=sys.stderr,
        )

    def _exp_build_fisheye_args(self):
        """Construct an argparse.Namespace for the fisheye pipeline from the UI fields."""
        import argparse
        args = argparse.Namespace()
        # Camera hardware comes from the top connection bar.
        args.camera_device = str(self._cam_device_spin.value())
        res_text = self._cam_res_combo.currentText().replace("×", "x")
        try:
            w, h = (int(v) for v in res_text.split("x"))
            args.camera_resolution = [w, h]
        except ValueError:
            args.camera_resolution = None
        args.camera_fps = float(self._cam_fps_spin.value())
        # Calibration from active session; fall back to the edit text (mock/no-op path).
        _calib = (
            self._camera.calib_path
            if self._camera is not None and self._camera.calib_path is not None
            else Path(self._cam_calib_edit.text().strip())
        )
        args.fisheye_calib = _calib
        args.fisheye_balance = 0.0
        # TagSLAM
        args.tag_family = self._exp_tag_family_edit.text().strip() or "tag36h11"
        args.tag_size   = float(self._exp_tag_size_spin.value())
        args.anchor_tag_id = int(self._exp_anchor_spin.value())
        args.max_tag_id = -1
        args.water_correction_mode = self._exp_water_combo.currentText()
        args.water_scale = 3.6
        args.surface_distance_m = 0.20
        args.water_refractive_index = 1.333
        args.refractive_max_iterations = 8
        args.refractive_convergence_tol_m = 1e-5
        args.refractive_convergence_tol_deg = 0.01
        args.refractive_ray_max_iterations = 10
        args.refractive_ray_tol = 1e-11
        args.min_tag_area_px = 120.0
        args.max_off_nadir_deg = 25.0
        args.max_image_eccentricity = 0.65
        args.max_tag_tilt_deg = 35.0
        args.max_reprojection_error_px = 5.0
        args.nthreads = 2
        args.quad_decimate = 1.0
        args.quad_sigma = 0.0
        args.decode_sharpening = 0.25
        args.min_decision_margin = 30.0
        args.max_hamming = 0
        args.tag_rot_sigma = 0.08
        args.tag_trans_sigma = 0.04
        args.tag_robust_kernel = "huber"
        args.tag_robust_threshold = 1.345
        args.tag_init_min_observations = 3
        args.pose_std_window = 30
        args.odom_rot_sigma = 0.35
        args.odom_trans_sigma = 0.30
        args.prior_rot_sigma = 1e-6
        args.prior_trans_sigma = 1e-6
        args.floor_prior_enabled = True
        args.floor_z_sigma = 0.02
        args.floor_plane_min_tags = 4
        args.floor_normal_sigma_deg = 8.0
        args.strict_coplanar = False
        args.floor_prior_refresh_frames = 0
        args.floor_plane_outlier_threshold = 0.10
        args.use_imu_gravity = False
        args.gravity_align_world = False
        args.imu_gravity_smoothing_n = 5
        args.init_min_observations = 3
        args.init_min_decision_margin = 45.0
        args.init_min_tag_area_px = 250.0
        args.init_max_off_nadir_deg = 20.0
        args.init_max_image_eccentricity = 0.45
        args.init_max_tag_tilt_deg = 25.0
        args.plot_z_scale = 1.0
        args.trajectory_image_width = 960
        args.config = "config/config.yaml"
        args.max_frames = None
        return args

    def _exp_load_waypoints(self) -> list[Waypoint]:
        if self._exp_path_seq_radio.isChecked():
            return self._collect_waypoints()
        # CSV file
        csv_path = Path(self._exp_path_csv_edit.text().strip())
        if not csv_path.exists():
            return []
        import csv as _csv
        out: list[Waypoint] = []
        with csv_path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                try:
                    # Support both mm/s and cm/s columns
                    speed_raw = float(row.get("speed_mm_s", row.get("speed_cm_s", 20.0)))
                    if "speed_cm_s" in row and "speed_mm_s" not in row:
                        speed_raw *= 10.0  # cm/s → mm/s
                    out.append(Waypoint(
                        x_mm=float(row["x_mm"]),
                        y_mm=float(row["y_mm"]),
                        z_mm=float(row["z_mm"]),
                        speed_mm_s=speed_raw,
                        dwell_s=float(row.get("dwell_s", 0.0)),
                    ))
                except (KeyError, ValueError):
                    continue
        return out

    def _exp_start(self) -> None:
        if not HAVE_EXPERIMENT_RUNNER:
            QMessageBox.critical(self, "Missing Module",
                "experiment_runner.py not found. Place it in the src/ directory.")
            return
        if self._experiment_in_progress:
            return

        # Collect waypoints (user-frame mm).
        try:
            waypoints_user = self._exp_load_waypoints()
        except SystemExit as exc:
            QMessageBox.warning(self, "Bad waypoints", str(exc))
            return
        if not waypoints_user:
            QMessageBox.warning(self, "No waypoints", "Define at least one waypoint first.")
            return

        # Pre-flight: try get_status() once. If it returns code 664 ('axis
        # not enabled / not homed') we let the experiment proceed — the
        # motion/logger paths now fall back to per-axis reads in 664 so they
        # work without a physical limit-switch homing pass. Only block on
        # non-664 errors that we have no fallback for.
        if self.connected and not self.is_mock:
            try:
                with self._controller_lock:
                    self.controller.get_status()
            except FMC4030Error as exc:
                if "664" not in str(exc):
                    QMessageBox.critical(self, "Controller error",
                        f"Pre-flight get_status() failed:\n\n{exc}")
                    return
                # 664 is non-blocking — fall through.
            except Exception as exc:
                QMessageBox.critical(self, "Controller error",
                    f"Pre-flight get_status() raised unexpectedly:\n\n{exc}")
                return

        # Convert waypoints to absolute machine-frame for the SDK.
        waypoints_fw = [
            Waypoint(
                x_mm=self.mm_user_to_mm_firmware(w.x_mm, Axis.X),
                y_mm=self.mm_user_to_mm_firmware(w.y_mm, Axis.Y),
                z_mm=self.mm_user_to_mm_firmware(w.z_mm, Axis.Z),
                speed_mm_s=w.speed_mm_s,
                dwell_s=w.dwell_s,
            )
            for w in waypoints_user
        ]

        camera_mode = self._exp_camera_mode()
        gantry_only = camera_mode == "gantry_only"

        # In gantry-only mode, fisheye is bypassed entirely.
        fisheye_args = None
        fisheye_calib = None
        if not gantry_only:
            # Load calibration from the active camera session (or skip for mock).
            calib_path = (
                self._camera.calib_path
                if self._camera is not None and self._camera.calib_path is not None
                else None
            )
            if not self._is_mock_camera and calib_path is not None:
                try:
                    from fisheye_gantry_tagslam import load_fisheye_calibration
                    fisheye_calib = load_fisheye_calibration(calib_path)
                except SystemExit as exc:
                    QMessageBox.warning(self, "Calibration error", str(exc))
                    return
                except ImportError:
                    pass  # fisheye module optional
            # Auto-pick anchor = tag closest to image center, using the live
            # preview's last detection. Falls back to the spin value if no
            # tags are visible. Updates the spin so the user sees the choice.
            self._exp_autopick_anchor_tag()
            fisheye_args = self._exp_build_fisheye_args()

        EMERGENCY_STOP.clear()
        self._abort_event.clear()

        run_name = self._exp_name_edit.text().strip()

        # In gantry-only mode, do NOT pass the camera session — MotionWorker has
        # no need for it and the fisheye worker is not started.
        config = ExperimentConfig(
            controller=self.controller,
            controller_lock=self._controller_lock,
            waypoints=waypoints_fw,
            soft_min_mm=[None, None, None],
            soft_max_mm=[None, None, None],
            move_mode=getattr(self, "_move_mode", "line"),
            countdown_s=self._exp_countdown_spin.value(),
            settle_s=self._exp_settle_spin.value(),
            tag_detection_while_idle=(
                self._exp_idle_detect_chk.isChecked() and not gantry_only
            ),
            output_root=Path("data"),
            run_name=run_name,
            fisheye_args=fisheye_args,
            fisheye_calib=fisheye_calib,
            abort_event=self._abort_event,
            mock_camera=(self._is_mock_camera and not gantry_only),
            camera_session=(self._camera if not gantry_only else None),
        )
        # Annotate the config with axis_sign + camera_mode + home offset for
        # run_metadata.json. Soft-limit metadata is intentionally omitted —
        # the panel no longer manages or enforces soft limits.
        config.camera_mode = camera_mode
        config.axis_sign = {a.name: int(self._axis_sign[a]) for a in AXES}
        config.waypoints_user_frame = [
            {"x_mm": w.x_mm, "y_mm": w.y_mm, "z_mm": w.z_mm,
             "speed_mm_s": w.speed_mm_s, "dwell_s": w.dwell_s}
            for w in waypoints_user
        ]
        config.home_reference_abs_mm = {
            a.name: (None if self._home_position_mm.get(a) is None
                     else float(self._home_position_mm[a]))
            for a in AXES
        }
        self._experiment_in_progress = True
        self._exp_start_btn.setEnabled(False)
        self._exp_stop_btn.setEnabled(True)
        self._exp_wp_bar.setValue(0)
        self._exp_time_bar.setValue(0)
        self._exp_result_label.setText("")
        self._experiment_runner.start_experiment(config)

    def _exp_stop(self) -> None:
        if self._experiment_runner is not None:
            self._experiment_runner.stop_experiment()
        self._abort_event.set()

    def _exp_on_state_changed(self, phase: str, msg: str) -> None:
        self._exp_state_label.setText(f"{phase}  {msg}")
        color = {
            "IDLE":        "#4ea1ff",
            "COUNTDOWN":   "#ffd54f",
            "MOTION":      "#66bb6a",
            "SETTLE":      "#ff9800",
            "POSTPROCESS": "#ab47bc",
            "DONE":        "#34d058",
        }.get(phase, "#aaa")
        self._exp_state_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {color};"
        )
        if phase == "DONE":
            self._experiment_in_progress = False
            self._exp_stop_btn.setEnabled(False)
            self._exp_refresh_checklist()

    def _exp_on_countdown(self, remaining: float) -> None:
        self._exp_state_label.setText(f"COUNTDOWN  T−{remaining:.1f}s")

    def _exp_on_wp_progress(self, current: int, total: int) -> None:
        self._exp_wp_bar.setRange(0, total)
        self._exp_wp_bar.setValue(current)
        self._exp_wp_bar.setFormat(f"Waypoint {current} / {total}")

    def _exp_on_fisheye_stats(self, sample) -> None:
        drift = f"{sample.drift_mm:.1f} mm" if sample.drift_mm == sample.drift_mm else "N/A"
        self._exp_tags_frame_lbl.setText(f"Tags/frame: {sample.tags_this_frame}")
        self._exp_tags_graph_lbl.setText(f"Tags/graph: {sample.tags_in_graph}")
        self._exp_updates_lbl.setText(f"Updates: {sample.backend_updates}")
        self._exp_drift_lbl.setText(f"Drift: {drift}")

    def _exp_on_finished(self, result: dict) -> None:
        self._experiment_in_progress = False
        self._exp_stop_btn.setEnabled(False)
        was_recording = self._recording_via_runner or self._runner_owns_logger
        self._recording_via_runner = False
        self._runner_owns_logger = False
        self._autologger_suppress_warned = False
        run_dir = result.get("run_dir", "")
        aborted = result.get("aborted", False)
        motion_error = result.get("motion_error", "")
        if motion_error:
            self._exp_result_label.setText(
                f"⚠ Motion FAILED: {motion_error}  (no gantry movement — "
                f"check terminal for traceback)  → {run_dir}"
            )
            self._exp_result_label.setStyleSheet(
                "color: #ef5350; font-size: 11px; font-weight: 600;"
            )
        else:
            if was_recording:
                tag = "Recording saved"
            else:
                tag = "Done (aborted)" if aborted else "Done"
            self._exp_result_label.setText(f"{tag} → {run_dir}")
            self._exp_result_label.setStyleSheet("color: #aaa; font-size: 11px;")
        # Reset the Recording tab's button (whether or not we got here from it).
        if hasattr(self, "record_btn"):
            self.record_btn.blockSignals(True)
            self.record_btn.setChecked(False)
            self.record_btn.setText("● Start Recording")
            self.record_btn.setEnabled(True)
            self.record_btn.blockSignals(False)
            if hasattr(self, "csv_path_label") and was_recording and run_dir:
                self.csv_path_label.setText(str(Path(run_dir) / "gantry_telemetry.csv"))
                self.csv_path_label.setStyleSheet("color: #64b5f6; text-decoration: underline;")
                self._current_run_dir = Path(run_dir)
        self._exp_refresh_checklist()

    def _build_waypoint_panel(self) -> SectionFrame:
        frame = SectionFrame("Waypoint Sequence")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self.waypoint_table = QTableWidget(0, 5)
        self.waypoint_table.setHorizontalHeaderLabels(
            ["X (mm, from home)", "Y (mm, from home)", "Z (mm, from home)", "Speed (mm/s)", "Dwell (s)"]
        )
        hdr = self.waypoint_table.horizontalHeader()
        for i in range(5):
            hdr.setSectionResizeMode(i, QHeaderView.Stretch)
        self.waypoint_table.setAlternatingRowColors(True)
        self.waypoint_table.setMinimumHeight(160)
        v.addWidget(self.waypoint_table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        add_btn = QPushButton("Add Row")
        _size_button(add_btn)
        add_btn.clicked.connect(self._waypoint_add_row)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove Selected")
        _size_button(rm_btn)
        rm_btn.clicked.connect(self._waypoint_remove_selected)
        btn_row.addWidget(rm_btn)
        load_btn = QPushButton("Load CSV…")
        _size_button(load_btn)
        load_btn.clicked.connect(self._menu_open_waypoints)
        btn_row.addWidget(load_btn)
        save_btn = QPushButton("Save CSV…")
        _size_button(save_btn)
        save_btn.clicked.connect(self._menu_save_waypoints)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        self.run_seq_btn = QPushButton("Run Sequence")
        self.run_seq_btn.setObjectName("PrimaryButton")
        _size_button(self.run_seq_btn, "primary")
        self.run_seq_btn.clicked.connect(self._run_sequence)
        btn_row.addWidget(self.run_seq_btn)
        self.stop_seq_btn = QPushButton("Stop Sequence")
        _size_button(self.stop_seq_btn)
        self.stop_seq_btn.clicked.connect(self._stop_sequence)
        self.stop_seq_btn.setEnabled(False)
        btn_row.addWidget(self.stop_seq_btn)
        send_exp_btn = QPushButton("→ Experiment")
        _size_button(send_exp_btn)
        send_exp_btn.setToolTip(
            "Copy these waypoints to the Experiment tab as the motion path, "
            "then switch to the Experiment tab."
        )
        send_exp_btn.clicked.connect(self._send_waypoints_to_experiment)
        btn_row.addWidget(send_exp_btn)
        v.addLayout(btn_row)

        # Run control row — low-level controller pause/resume/stop.
        run_ctrl_row = QHBoxLayout()
        run_ctrl_row.setSpacing(8)
        run_ctrl_row.addWidget(QLabel("Run Control:"))
        self.pause_btn = QPushButton("Pause Run")
        _size_button(self.pause_btn)
        self.pause_btn.clicked.connect(lambda: self._run_global_command("pause"))
        self.pause_btn.setEnabled(False)
        run_ctrl_row.addWidget(self.pause_btn)
        self.resume_btn = QPushButton("Resume Run")
        _size_button(self.resume_btn)
        self.resume_btn.clicked.connect(lambda: self._run_global_command("resume"))
        self.resume_btn.setEnabled(False)
        run_ctrl_row.addWidget(self.resume_btn)
        self.stop_run_btn = QPushButton("Stop Run")
        _size_button(self.stop_run_btn)
        self.stop_run_btn.clicked.connect(lambda: self._run_global_command("stop"))
        self.stop_run_btn.setEnabled(False)
        run_ctrl_row.addWidget(self.stop_run_btn)
        run_ctrl_row.addStretch()
        v.addLayout(run_ctrl_row)
        return frame

    def _build_status_bar(self) -> None:
        sb: QStatusBar = self.statusBar()
        self.sb_conn_label = QLabel("● Disconnected")
        self.sb_conn_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        _fit_label(self.sb_conn_label, "● Disconnected")
        sb.addWidget(self.sb_conn_label)
        self.sb_op_label = QLabel("Idle")
        self.sb_op_label.setStyleSheet("color: #ccc; padding-left: 20px;")
        sb.addWidget(self.sb_op_label, stretch=1)
        self.sb_rec_label = QLabel("")
        self.sb_rec_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        sb.addPermanentWidget(self.sb_rec_label)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _toggle_connection(self) -> None:
        if self.connected:
            self._disconnect()
            return
        if self.is_mock:
            config = ControllerConfig(controller_id=0, ip="mock", port=0)
        else:
            config = ControllerConfig(
                controller_id=int(self.id_spin.value()),
                ip=self.ip_edit.text().strip(),
                port=int(self.port_spin.value()),
            )
        try:
            with self._controller_lock:
                self.controller.connect(config)
        except FMC4030Error as exc:
            QMessageBox.critical(self, "Connect failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Connect failed", f"Unexpected: {exc}")
            return
        self.connected = True
        self.connect_btn.setText("Disconnect")
        self._set_connection_status("Connected", "#66bb6a")
        self._update_all_button_states()

    def _disconnect(self) -> None:
        if self._logger is not None:
            try:
                self._logger.stop()
            except Exception:
                pass
            self._logger = None
        self._status_timer.stop()
        for thread_attr in ("_status_thread", "_move_thread", "_sequence_thread"):
            t = getattr(self, thread_attr, None)
            if t is not None and t.isRunning():
                t.requestInterruption()
                t.wait(500)
            setattr(self, thread_attr, None)
        try:
            with self._controller_lock:
                self.controller.close()
        except Exception:
            pass
        self.connected = False
        self.connect_btn.setText("Connect")
        self._set_connection_status("Disconnected", "#ef5350")
        # Reset per-axis readouts on disconnect.
        for info in self.per_axis_cards.values():
            info["pos_mm"].setText("--")
            info["pos_mm"].setToolTip("")
            info["vel"].setText("Vel: -- cm/s")
            lbl = info.get("home_label")
            if lbl:
                lbl.setText("⚠ no home")
                lbl.setStyleSheet("color: #ffa726; font-size: 11px;")
        self.sb_op_label.setText("Idle")
        self._update_all_button_states()
        self._status_timer.start()

    def _set_connection_status(self, text: str, color: str) -> None:
        self.conn_status_label.setText(f"● {text}")
        self.conn_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.sb_conn_label.setText(f"● {text}")
        self.sb_conn_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------
    def _request_status_update(self) -> None:
        if not self.connected:
            return
        if self._status_thread is not None and self._status_thread.isRunning():
            return
        self._status_thread = StatusPollThread(self.controller, self._controller_lock, self)
        self._status_thread.snapshot_ready.connect(self._on_status_snapshot)
        self._status_thread.error_occurred.connect(self._on_status_error)
        self._status_thread.finished.connect(self._on_status_thread_finished)
        self._status_thread.start()

    def _on_status_thread_finished(self) -> None:
        if self._status_thread is not None:
            self._status_thread.deleteLater()
            self._status_thread = None

    def _on_status_snapshot(self, status) -> None:
        now = time.monotonic()
        pos_units = [float(status.realPos[i]) for i in range(3)]
        vel_units = [float(status.realSpeed[i]) for i in range(3)]
        # Absolute machine-frame mm — raw SDK readback, no sign, no offset.
        # Kept alongside user-frame for home-reference capture and the
        # "Abs: …" diagnostic label.
        pos_abs_mm = [units_to_mm(pos_units[i], AXES[i]) for i in range(3)]
        # User-frame mm — home-relative AND sign-flipped (what every UI shows).
        pos_mm = [self.abs_mm_to_user_mm(pos_abs_mm[i], AXES[i]) for i in range(3)]
        # Velocities: sign flip only (rate; home offset drops out).
        vel_mm = [self.vel_units_to_user_mm_s(vel_units[i], AXES[i]) for i in range(3)]

        # Acceleration: SMA-smoothed central difference over 5 samples.
        self._vel_buffer.append((now, tuple(vel_mm)))
        acc_mm = self._compute_accel_mm_s2()

        # Rate-limited diagnostic.
        if DEBUG_STATUS_POLL:
            if now - getattr(self, "_dbg_snap_t", 0.0) >= 1.0:
                self._dbg_snap_t = now
                print(f"[status-poll] user X={pos_mm[0]:+.3f} Y={pos_mm[1]:+.3f} "
                      f"Z={pos_mm[2]:+.3f} mm  |  abs X={pos_abs_mm[0]:+.3f} "
                      f"Y={pos_abs_mm[1]:+.3f} Z={pos_abs_mm[2]:+.3f}  |  "
                      f"vel(mm/s)=({vel_mm[0]:+.2f},{vel_mm[1]:+.2f},{vel_mm[2]:+.2f})",
                      file=sys.stderr, flush=True)

        # Cache freshness for the Polling indicator.
        self._last_snapshot_t = now
        self._last_pos_mm = tuple(pos_mm)
        self._last_pos_abs_mm = tuple(pos_abs_mm)
        self._last_pos_units = tuple(pos_units)

        # Update per-axis Control card readouts.
        # pos_mm is already user-frame (home-relative); display it directly.
        # The absolute machine value is exposed via the tooltip + the "Abs:"
        # diagnostic label on the Live Status row.
        for i, axis in enumerate(AXES):
            home_mm = self._home_position_mm[axis]
            info = self.per_axis_cards.get(axis)
            if info is not None:
                info["pos_mm"].setText(f"{pos_mm[i]:+8.3f}")
                info["pos_mm"].setToolTip(
                    f"User-frame (home-relative)  ·  abs {pos_abs_mm[i]:+.3f} mm"
                )
                lbl = info.get("home_label")
                if lbl:
                    if home_mm is not None:
                        lbl.setText("⌂ from home")
                        lbl.setStyleSheet("color: #4ea1ff; font-size: 11px;")
                    else:
                        lbl.setText("⚠ no home")
                        lbl.setStyleSheet("color: #ffa726; font-size: 11px;")
                info["vel"].setText(f"Vel: {vel_mm[i] / 10.0:+.2f} cm/s")

        # Update plot history.
        self._time_history.append(now - self._t0_mono)
        for i in range(3):
            self._pos_history[i].append(pos_mm[i])
        self.plot_widget.update_data(self._time_history, self._pos_history)

        # Workspace Map: pass user-frame (home-relative) values so the marker
        # sits at the origin after Set Current as Home Reference.
        if getattr(self, "workspace_map", None) is not None:
            self.workspace_map.update_position(pos_mm[0], pos_mm[1], pos_mm[2])
            home_xyz_user = tuple(
                0.0 if self._home_position_mm[a] is not None else None
                for a in AXES
            )
            self.workspace_map.update_home(home_xyz_user)

    def _on_status_error(self, msg: str) -> None:
        # Special hint for the common 664 ("machine status unavailable")
        # condition — typically means at least one axis is not enabled or
        # not yet homed. The per-axis fallback in StatusPollThread keeps
        # the readouts alive; here we surface the actionable hint.
        if "664" in msg:
            self.sb_op_label.setText(
                "Controller status 664 (axis not enabled / not homed). "
                "Live readouts via per-axis fallback; home axes to clear."
            )
            self.sb_op_label.setStyleSheet("color: #ffa726; font-weight: bold;")
        else:
            self.sb_op_label.setText(f"Status error: {msg[:120]}")

    def _update_poll_indicator(self) -> None:
        if self._last_snapshot_t == 0.0:
            self.poll_indicator.setText("Polling: — (no data)")
            self.poll_indicator.setStyleSheet(
                "color: #8a8a8a; font-size: 11px; padding: 2px 4px;"
            )
            return
        age_ms = (time.monotonic() - self._last_snapshot_t) * 1000.0
        if age_ms > POLL_INDICATOR_STALE_MS:
            self.poll_indicator.setText(f"Polling: ✗ {age_ms:.0f} ms ago (STALE)")
            self.poll_indicator.setStyleSheet(
                "color: #ef5350; font-size: 11px; font-weight: bold; padding: 2px 4px;"
            )
        else:
            self.poll_indicator.setText(f"Polling: ✓ {age_ms:.0f} ms ago")
            self.poll_indicator.setStyleSheet(
                "color: #34d058; font-size: 11px; padding: 2px 4px;"
            )

    def _compute_accel_mm_s2(self) -> tuple[float, float, float]:
        n = len(self._vel_buffer)
        if n < FINITE_DIFF_WINDOW:
            return (0.0, 0.0, 0.0)
        items = list(self._vel_buffer)
        half = FINITE_DIFF_WINDOW // 2
        front = items[:half]
        back = items[-half:]
        t_f = sum(it[0] for it in front) / len(front)
        t_b = sum(it[0] for it in back) / len(back)
        dt = t_b - t_f
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        out = []
        for axis in range(3):
            vf = sum(it[1][axis] for it in front) / len(front)
            vb = sum(it[1][axis] for it in back) / len(back)
            out.append((vb - vf) / dt)
        return tuple(out)  # type: ignore[return-value]

    def _refresh_position(self) -> None:
        if not self.connected:
            return
        self._request_status_update()

    def _set_current_as_home_reference(self) -> None:
        """Capture the current XYZ as the home reference. No motion is
        commanded.

        The home reference is stored in ABSOLUTE machine-frame mm (raw SDK
        readback, no sign flip, no offset). All UI displays then derive
        their user-frame (home-relative, sign-flipped) values from this.

        Source-of-truth precedence:
          1. The latest poll snapshot (``self._last_pos_abs_mm``), if available.
          2. Otherwise, a direct per-axis ``get_axis_position(axis)`` read,
             which works even when ``get_status()`` is returning the 664 error
             (axis not enabled / not homed yet).
          3. If all three per-axis reads also fail, warn and abort.
        """
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        snap_abs = getattr(self, "_last_pos_abs_mm", None)
        if not snap_abs or all(v == 0.0 for v in snap_abs):
            # Fallback: read each axis directly in absolute machine-frame mm.
            captured_abs: list[float | None] = [None, None, None]
            errors: list[str] = []
            for i, axis in enumerate(AXES):
                try:
                    with self._controller_lock:
                        u = float(self.controller.get_axis_position(axis))
                    captured_abs[i] = units_to_mm(u, axis)
                except Exception as exc:
                    errors.append(f"{axis.name}: {exc}")
            if not any(c is not None for c in captured_abs):
                QMessageBox.warning(
                    self, "No position yet",
                    "Could not read the current position from the controller.\n\n"
                    + ("\n".join(errors) if errors else "")
                    + "\n\nTry homing the axes first (status code 664 usually "
                      "means an axis is not enabled / not homed).",
                )
                return
            snap_abs = tuple((c if c is not None else 0.0) for c in captured_abs)
            print(f"[home-ref] captured via per-axis fallback (abs mm): "
                  f"{snap_abs}", file=sys.stderr)

        for i, axis in enumerate(AXES):
            self._home_position_mm[axis] = float(snap_abs[i])
        # Seed the absolute cache AND zero the user-frame cache so the
        # per-axis card flips to +0.000 immediately before the next poll.
        self._last_pos_abs_mm = tuple(float(v) for v in snap_abs)
        self._last_pos_mm = tuple(0.0 for _ in AXES)
        self._persist_home_position()
        self._refresh_home_displays()
        self.sb_op_label.setText(
            f"Home ref captured (abs mm): X={snap_abs[0]:+.2f} "
            f"Y={snap_abs[1]:+.2f} Z={snap_abs[2]:+.2f}. User-frame zeroed."
        )

    def _persist_home_position(self) -> None:
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["home_position_mm"] = {
            a.name: (None if self._home_position_mm[a] is None
                     else float(self._home_position_mm[a]))
            for a in AXES
        }
        _gp_save_section("gantry_panel", payload)

    def _start_go_home(self) -> None:
        """Drive each axis sequentially back to the saved home reference."""
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if getattr(self, "_home_thread", None) is not None and self._home_thread.isRunning():
            return
        if self._move_in_progress or self._sequence_in_progress or self._experiment_in_progress:
            QMessageBox.warning(self, "Busy",
                "Wait for the current move/experiment to finish before going home.")
            return
        if not any(self._home_position_mm.get(a) is not None for a in AXES):
            QMessageBox.warning(
                self, "No home reference",
                "No axis has a home reference yet.\n\n"
                "Click 'Set Current as Home Reference' first to capture the "
                "current position as home.",
            )
            return

        targets = {a: self._home_position_mm.get(a) for a in AXES}
        speed = {a: max(cm_s_to_units_s(self.jog_speed_spin.value(), a), 0.001)
                 for a in AXES}
        acc   = {a: max(cm_s2_to_units_s2(self.jog_acc_spin.value(), a), 0.001)
                 for a in AXES}
        dec   = {a: max(cm_s2_to_units_s2(self.jog_dec_spin.value(), a), 0.001)
                 for a in AXES}

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._home_all_btn.setEnabled(False)
        self._home_all_btn.setText("Going home…")
        self._home_thread = GoToHomeThread(
            self.controller, targets, speed, acc, dec,
            self._controller_lock, self._abort_event,
        )
        self._home_thread.progress.connect(self.sb_op_label.setText)
        self._home_thread.finished_with.connect(self._on_go_home_done)
        self._home_thread.finished.connect(self._home_thread.deleteLater)
        self._home_thread.start()

    def _on_go_home_done(self, err: str) -> None:
        self._home_all_btn.setEnabled(True)
        self._home_all_btn.setText("Go to Home")
        if not err:
            self.sb_op_label.setText("Gantry returned to home reference.")
            self.sb_op_label.setStyleSheet("color: #66bb6a;")
        elif err == "aborted":
            self.sb_op_label.setText("Go-to-home aborted.")
            self.sb_op_label.setStyleSheet("color: #ffa726;")
        else:
            self.sb_op_label.setText(f"Go-to-home failed: {err[:120]}")
            self.sb_op_label.setStyleSheet("color: #ef5350;")
            QMessageBox.critical(self, "Go-to-home failed", err)

    def _refresh_home_displays(self) -> None:
        # Re-render per-axis position readouts using the latest snapshot,
        # and update the workspace map's home marker.
        # _last_pos_mm is already user-frame (home-relative); no subtraction.
        snap_user = getattr(self, "_last_pos_mm", None)
        snap_abs = getattr(self, "_last_pos_abs_mm", None)
        for i, axis in enumerate(AXES):
            home_mm_abs = self._home_position_mm[axis]
            cur_user = snap_user[i] if snap_user is not None else None
            cur_abs = snap_abs[i] if snap_abs is not None else None
            info = self.per_axis_cards.get(axis)
            if info is None or cur_user is None:
                continue
            info["pos_mm"].setText(f"{cur_user:+8.3f}")
            abs_str = f"{cur_abs:+.3f} mm" if cur_abs is not None else "—"
            info["pos_mm"].setToolTip(
                f"User-frame (home-relative)  ·  abs {abs_str}"
            )
            lbl = info.get("home_label")
            if lbl:
                if home_mm_abs is not None:
                    lbl.setText("⌂ from home")
                    lbl.setStyleSheet("color: #4ea1ff; font-size: 11px;")
                else:
                    lbl.setText("⚠ no home")
                    lbl.setStyleSheet("color: #ffa726; font-size: 11px;")
        if getattr(self, "workspace_map", None) is not None:
            # Workspace map operates in user-frame. Home shown at the origin
            # of user-frame, so pass (0,0,0) when home is set; else None.
            home_xyz_user = tuple(
                0.0 if self._home_position_mm[a] is not None else None
                for a in AXES
            )
            self.workspace_map.update_home(home_xyz_user)

    # ------------------------------------------------------------------
    # Move to target
    # ------------------------------------------------------------------
    def _move_to_target(self) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._move_in_progress or self._sequence_in_progress:
            return
        if self._blocked_by_estop():
            return
        target_mm_user = (
            self.target_spins[Axis.X].value(),
            self.target_spins[Axis.Y].value(),
            self.target_spins[Axis.Z].value(),
        )

        # Convert user mm → absolute machine-frame mm for move_to_xyz_mm,
        # which internally uses mm_to_units(...) per axis without any sign.
        target_mm_fw = tuple(
            self.mm_user_to_mm_firmware(target_mm_user[i], AXES[i]) for i in range(3)
        )

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        self._move_in_progress = True
        self._update_all_button_states()
        self.move_btn.setText("Moving…")
        self.cancel_move_btn.setEnabled(True)
        self.sb_op_label.setText(
            f"Moving to ({target_mm_user[0]:+.2f}, {target_mm_user[1]:+.2f}, "
            f"{target_mm_user[2]:+.2f}) mm"
        )
        if getattr(self, "workspace_map", None) is not None:
            self.workspace_map.update_target(*target_mm_user)
        self._move_thread = MoveToTargetThread(
            self.controller, target_mm_fw,
            self.move_speed_spin.value() * 10.0,   # cm/s → mm/s
            self.move_acc_spin.value() * 10.0,
            self.move_dec_spin.value() * 10.0,
            self.move_mode_combo.currentData(),
            self._controller_lock,
            logger=self._logger,
        )
        self._move_thread.finished_with.connect(self._on_move_done)
        self._move_thread.finished.connect(self._move_thread.deleteLater)
        self._move_thread.start()

    def _use_current_as_target(self) -> None:
        # Use the cached absolute position (target spinboxes are in machine mm).
        snap = getattr(self, "_last_pos_mm", None)
        if snap is None:
            return
        for i, axis in enumerate(AXES):
            self.target_spins[axis].setValue(snap[i])

    def _cancel_move(self) -> None:
        # Soft-cancel: stop_run halts coordinated motion. The move thread will
        # observe is_axis_stopped and return; on_move_done then re-enables UI.
        try:
            with self._controller_lock:
                self.controller.stop_run()
        except FMC4030Error:
            pass

    def _on_move_done(self, err: str) -> None:
        self._move_in_progress = False
        # Don't drop ref here — see _on_axis_abs_done docstring (QThread GC race).
        self.move_btn.setText("Move to Target")
        self.cancel_move_btn.setEnabled(False)
        if err:
            self.sb_op_label.setText(f"Move error: {err[:80]}")
            QMessageBox.warning(self, "Move error", err)
        else:
            self.sb_op_label.setText("Move complete")
        # Auto-stop logger after a short tail.
        QTimer.singleShot(500, self._stop_logger_if_auto)
        self._update_all_button_states()

    # ------------------------------------------------------------------
    # Per-Axis Control: jog (hold-to-jog) + Move Abs (mm)
    # ------------------------------------------------------------------
    def _start_jog(self, axis: Axis, direction: int) -> None:
        if not self.connected:
            return
        if (self._sequence_in_progress
                or self._move_in_progress or int(axis) in self._per_axis_busy):
            return
        if self._blocked_by_estop():
            return
        EMERGENCY_STOP.clear()
        self._ensure_logger_started(auto=True)
        # cm/s → units/s conversion for single-axis jog.
        speed_units = max(cm_s_to_units_s(self.jog_speed_spin.value(), axis), 0.001)
        acc_units   = max(cm_s2_to_units_s2(self.jog_acc_spin.value(), axis), 0.001)
        dec_units   = max(cm_s2_to_units_s2(self.jog_dec_spin.value(), axis), 0.001)
        fw_direction = self.jog_direction_user_to_firmware(direction, axis)
        try:
            with self._controller_lock:
                self.controller.jog_single_axis(
                    axis,
                    position_units=999999.0 * fw_direction,
                    speed_units=speed_units,
                    acc_units=acc_units,
                    dec_units=dec_units,
                    relative=True,
                )
        except FMC4030Error as exc:
            QMessageBox.critical(self, "Jog error", str(exc))

    def _stop_jog(self, axis: Axis) -> None:
        if not self.connected:
            return
        try:
            with self._controller_lock:
                # mode=1 = soft stop (decelerate), per manual_pad.py.
                self.controller.stop_axis(axis, mode=1)
        except FMC4030Error as exc:
            QMessageBox.critical(self, "Stop error", str(exc))
        # Tail-stop the auto logger after the ringdown.
        QTimer.singleShot(500, self._stop_logger_if_auto)

    def _go_to_home_axis(self, axis: Axis) -> None:
        """Per-axis 'Go Home' button: move-abs to the captured home reference.

        Home reference is stored in absolute machine-frame mm. In user-frame
        the home position is, by definition, 0.0 mm — so this is a move-abs
        to user-frame 0 on this axis.

        Distinct from physical-limit-switch homing (Setup tab → Homing → Home X).
        Uses the same machinery as Move Abs, so soft-limit + axis-sign + the
        live watchdog all apply unchanged.
        """
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._home_position_mm.get(axis) is None:
            QMessageBox.warning(
                self,
                "No home reference",
                f"{axis.name} has no home reference yet.\n\n"
                "Set one via Setup tab → 'Set Current as Home Reference' "
                "(captures the current XYZ as the user-frame origin; no motion).",
            )
            return
        # Target = user-frame 0 mm on this axis.
        spin = self.per_axis_cards.get(axis, {}).get("target_spin")
        if spin is not None:
            spin.setValue(0.0)
        self._move_axis_abs(axis)

    def _move_axis_abs(self, axis: Axis) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if (self._move_in_progress or self._sequence_in_progress
                or int(axis) in self._per_axis_busy):
            return
        if self._blocked_by_estop():
            return
        # User-facing mm target.
        target_mm_user = self.per_axis_cards[axis]["target_spin"].value()

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        # cm/s → units/s conversion for single-axis move-abs.
        speed_units = max(cm_s_to_units_s(self.jog_speed_spin.value(), axis), 0.001)
        acc_units   = max(cm_s2_to_units_s2(self.jog_acc_spin.value(), axis), 0.001)
        dec_units   = max(cm_s2_to_units_s2(self.jog_dec_spin.value(), axis), 0.001)

        self._per_axis_busy.add(int(axis))
        info = self.per_axis_cards[axis]
        info["move_btn"].setEnabled(False)
        info["move_btn"].setText("Moving…")
        self.sb_op_label.setText(f"Moving {axis.name} → {target_mm_user:+.3f} mm")
        self._update_all_button_states()

        # Show target on the workspace map (user-frame mm; the other two axes
        # keep their current position so the marker lands at the actual destination).
        if getattr(self, "workspace_map", None) is not None and self.workspace_map is not None:
            cur = self.workspace_map._cur_pos or (0.0, 0.0, 0.0)
            full_target = list(cur)
            full_target[int(axis)] = target_mm_user
            self.workspace_map.update_target(*full_target)

        thread = AxisAbsMoveThread(
            self.controller, axis,
            target_units=self.mm_user_to_units(target_mm_user, axis),
            speed_units=speed_units,
            acc_units=acc_units,
            dec_units=dec_units,
            lock=self._controller_lock,
            abort_event=self._abort_event,
        )
        thread.finished_with.connect(partial(self._on_axis_abs_done, axis))
        thread.finished.connect(thread.deleteLater)
        self._per_axis_threads[int(axis)] = thread
        thread.start()

    def _on_axis_abs_done(self, axis: Axis, err: str) -> None:
        # Note: we do NOT pop self._per_axis_threads[int(axis)] here. This slot
        # fires on the worker's ``finished_with`` signal, BEFORE Qt's own
        # ``finished`` signal has propagated. Dropping the last Python ref now
        # would cause CPython to GC the QThread while the underlying C++ thread
        # is still wrapping up, triggering "QThread: Destroyed while thread is
        # still running" + abort. The dict reference is replaced on the next
        # Move Abs for the same axis and cleared in closeEvent.
        self._per_axis_busy.discard(int(axis))
        info = self.per_axis_cards.get(axis)
        if info is not None:
            info["move_btn"].setEnabled(True)
            info["move_btn"].setText("Move Abs")
        if err:
            self.sb_op_label.setText(f"{axis.name} move error: {err[:80]}")
            QMessageBox.warning(self, "Move error", f"{axis.name}: {err}")
        else:
            self.sb_op_label.setText(f"{axis.name} move complete")
        QTimer.singleShot(500, self._stop_logger_if_auto)
        self._update_all_button_states()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _toggle_recording(self, checked: bool) -> None:
        if checked:
            if not self.connected:
                self.record_btn.setChecked(False)
                QMessageBox.warning(self, "Not connected", "Connect first.")
                return
            # If the ExperimentRunner is available, route through it so we get
            # gantry telemetry + fisheye AprilTag SLAM + the same postprocess
            # plots/CSVs/HTML produced by zed2_underwater_tagslam.py and the
            # experiment pipeline. Fall back to the bare telemetry logger only
            # when the runner isn't importable.
            if HAVE_EXPERIMENT_RUNNER and self._experiment_runner is not None:
                started = self._start_recording_via_runner()
                if not started:
                    self.record_btn.setChecked(False)
                    return
                self.record_btn.setText("■ Stop Recording")
                return
            self._recording_manual = True
            self._ensure_logger_started(auto=False)
            self.record_btn.setText("■ Stop Recording")
        else:
            if self._recording_via_runner:
                # The runner's stop_experiment() drives postprocess + finished
                # signal; UI cleanup happens in _exp_on_finished.
                self._experiment_runner.stop_experiment()
                self.record_btn.setText("● Stopping…")
                self.record_btn.setEnabled(False)
                return
            self._recording_manual = False
            self._stop_logger_now()
            self.record_btn.setText("● Start Recording")

    def _start_recording_via_runner(self) -> bool:
        """Build a recording config (empty waypoints, no motion) and hand it
        to ExperimentRunner.start_recording. Returns True on success."""
        if self._experiment_in_progress:
            QMessageBox.warning(self, "Already running",
                "An experiment is already in progress.")
            return False

        # Claim logger ownership SYNCHRONOUSLY, before any motion handler can
        # fire, so the panel's auto-logger never spawns a parallel
        # <ts>_gantry_run folder during this recording. Cleared in
        # _exp_on_finished (or below if we bail out early).
        self._recording_via_runner = True
        self._runner_owns_logger = True
        self._autologger_suppress_warned = False

        def _release_ownership() -> None:
            self._recording_via_runner = False
            self._runner_owns_logger = False

        # Pre-flight: 664 is now non-blocking (telemetry + motion fall back
        # to per-axis reads). Only block on non-664 controller errors.
        if not self.is_mock:
            try:
                with self._controller_lock:
                    self.controller.get_status()
            except FMC4030Error as exc:
                if "664" not in str(exc):
                    QMessageBox.critical(self, "Controller error",
                        f"Pre-flight get_status() failed:\n\n{exc}")
                    _release_ownership()
                    return False
            except Exception as exc:
                QMessageBox.critical(self, "Controller error",
                    f"Pre-flight get_status() raised unexpectedly:\n\n{exc}")
                _release_ownership()
                return False

        # Decide camera_mode based on what's actually open right now.
        camera_open = (
            (self._camera is not None and self._camera.is_open)
            or self._is_mock_camera
        )
        calib_loaded = False
        if self._camera is not None and self._camera.calib_path is not None:
            calib_loaded = self._camera.calib_path.exists()
        if self._is_mock_camera:
            calib_loaded = True

        gantry_only = not (camera_open and calib_loaded)
        camera_mode = "gantry_only" if gantry_only else "fisheye"

        fisheye_args = None
        fisheye_calib = None
        if not gantry_only:
            calib_path = (
                self._camera.calib_path
                if self._camera is not None and self._camera.calib_path is not None
                else None
            )
            if not self._is_mock_camera and calib_path is not None:
                try:
                    from fisheye_gantry_tagslam import load_fisheye_calibration
                    fisheye_calib = load_fisheye_calibration(calib_path)
                except SystemExit as exc:
                    QMessageBox.warning(self, "Calibration error", str(exc))
                    _release_ownership()
                    return False
                except ImportError:
                    pass
            # Auto-pick anchor before locking args.
            self._exp_autopick_anchor_tag()
            fisheye_args = self._exp_build_fisheye_args()

        EMERGENCY_STOP.clear()
        self._abort_event.clear()

        config = ExperimentConfig(
            controller=self.controller,
            controller_lock=self._controller_lock,
            waypoints=[],
            soft_min_mm=[None, None, None],
            soft_max_mm=[None, None, None],
            move_mode=getattr(self, "_move_mode", "line"),
            countdown_s=0.0,
            settle_s=0.0,
            tag_detection_while_idle=False,
            output_root=Path("data"),
            run_name=self._exp_name_edit.text().strip() if hasattr(self, "_exp_name_edit") else "",
            fisheye_args=fisheye_args,
            fisheye_calib=fisheye_calib,
            abort_event=self._abort_event,
            mock_camera=(self._is_mock_camera and not gantry_only),
            camera_session=(self._camera if not gantry_only else None),
        )
        config.camera_mode = camera_mode
        config.axis_sign = {a.name: int(self._axis_sign[a]) for a in AXES}
        config.waypoints_user_frame = []
        config.home_reference_abs_mm = {
            a.name: (None if self._home_position_mm.get(a) is None
                     else float(self._home_position_mm[a]))
            for a in AXES
        }

        self._experiment_in_progress = True
        # _recording_via_runner / _runner_owns_logger already set at top.
        if hasattr(self, "_exp_start_btn"):
            self._exp_start_btn.setEnabled(False)
        if hasattr(self, "_exp_stop_btn"):
            self._exp_stop_btn.setEnabled(True)
        if hasattr(self, "_exp_result_label"):
            self._exp_result_label.setText("Recording manually — click Stop on Recording tab.")
            self._exp_result_label.setStyleSheet("color: #4ea1ff; font-size: 11px;")
        self._experiment_runner.start_recording(config)
        return True

    def _ensure_logger_started(self, auto: bool) -> None:
        if self._recording_via_runner or self._runner_owns_logger:
            # The runner owns the gantry logger for this recording; do not spawn
            # a parallel <ts>_gantry_run logger from jog/move handlers.
            if not self._autologger_suppress_warned:
                print("[recording-folder] suppressed auto-logger spawn while "
                      "runner owns recording", file=sys.stderr)
                self._autologger_suppress_warned = True
            return
        if self._logger is not None:
            return
        run_dir = make_gantry_run_dir(Path("data"), suffix="gantry_run")
        csv_path = run_dir / "gantry_telemetry.csv"
        self._current_run_dir = run_dir
        self._logger = GantryTelemetryLogger(
            self.controller, csv_path,
            log_hz=100.0, lock=self._controller_lock,
            t0_monotonic=self._t0_mono,
        )
        self._logger.start()
        if auto:
            self._recording_auto = True
        self.csv_path_label.setText(str(csv_path))
        self.csv_path_label.setStyleSheet("color: #64b5f6; text-decoration: underline;")
        self.sb_rec_label.setText(f"● RECORDING  {csv_path.name}")

    def _stop_logger_if_auto(self) -> None:
        if self._logger is None:
            return
        if self._recording_manual:
            return
        self._stop_logger_now()

    def _stop_logger_now(self) -> None:
        if self._logger is not None:
            try:
                self._logger.stop()
            except Exception:
                pass
            self._logger = None
        self._recording_auto = False
        self.sb_rec_label.setText("")

    def _open_csv_folder(self, event) -> None:  # mousePressEvent signature
        if self._current_run_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_run_dir)))

    # ------------------------------------------------------------------
    # Waypoint sequence
    # ------------------------------------------------------------------
    def _waypoint_add_row(self) -> None:
        r = self.waypoint_table.rowCount()
        self.waypoint_table.insertRow(r)
        defaults = [0.0, 0.0, 0.0, self.move_speed_spin.value(), 0.0]
        for c, v in enumerate(defaults):
            self.waypoint_table.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def _waypoint_remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.waypoint_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.waypoint_table.removeRow(r)

    def _collect_waypoints(self) -> list[Waypoint]:
        # Waypoints are entered in home-relative mm; convert to absolute here.
        # If home has not been captured for an axis, offset is 0 (absolute passthrough).
        hx = float(self._home_position_mm.get(Axis.X) or 0.0)
        hy = float(self._home_position_mm.get(Axis.Y) or 0.0)
        hz = float(self._home_position_mm.get(Axis.Z) or 0.0)
        out: list[Waypoint] = []
        for r in range(self.waypoint_table.rowCount()):
            try:
                cells = [self.waypoint_table.item(r, c).text() if self.waypoint_table.item(r, c) else "" for c in range(5)]
                wp = Waypoint(
                    x_mm=float(cells[0]) + hx,
                    y_mm=float(cells[1]) + hy,
                    z_mm=float(cells[2]) + hz,
                    speed_mm_s=float(cells[3]) if cells[3].strip() else self.move_speed_spin.value(),
                    dwell_s=float(cells[4]) if cells[4].strip() else 0.0,
                )
                out.append(wp)
            except (ValueError, AttributeError) as exc:
                raise SystemExit(f"Bad waypoint row {r}: {exc}")
        return out

    def _run_sequence(self) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._move_in_progress or self._sequence_in_progress:
            return
        if self._blocked_by_estop():
            return
        try:
            waypoints_user = self._collect_waypoints()
        except SystemExit as exc:
            QMessageBox.warning(self, "Bad waypoints", str(exc))
            return
        if not waypoints_user:
            QMessageBox.information(self, "No waypoints", "Add at least one row first.")
            return

        # Convert user-frame waypoints to absolute machine-frame for SequenceThread.
        waypoints_fw = [
            Waypoint(
                x_mm=self.mm_user_to_mm_firmware(w.x_mm, Axis.X),
                y_mm=self.mm_user_to_mm_firmware(w.y_mm, Axis.Y),
                z_mm=self.mm_user_to_mm_firmware(w.z_mm, Axis.Z),
                speed_mm_s=w.speed_mm_s,
                dwell_s=w.dwell_s,
            )
            for w in waypoints_user
        ]

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        self._sequence_in_progress = True
        self._update_all_button_states()
        self.run_seq_btn.setEnabled(False)
        self.stop_seq_btn.setEnabled(True)
        self._sequence_thread = SequenceThread(
            self.controller, waypoints_fw,
            self.move_acc_spin.value(), self.move_dec_spin.value(),
            self.move_mode_combo.currentData(),
            self._controller_lock, logger=self._logger,
            abort_event=self._abort_event,
        )
        self._sequence_thread.row_started.connect(self._on_sequence_row_started)
        self._sequence_thread.row_done.connect(self._on_sequence_row_done)
        self._sequence_thread.sequence_done.connect(self._on_sequence_done)
        self._sequence_thread.finished.connect(self._sequence_thread.deleteLater)
        self._sequence_thread.start()

    def _stop_sequence(self) -> None:
        EMERGENCY_STOP.set()
        try:
            with self._controller_lock:
                self.controller.stop_run()
        except FMC4030Error:
            pass

    def _on_sequence_row_started(self, idx: int) -> None:
        # Highlight the active row.
        for r in range(self.waypoint_table.rowCount()):
            color = QColor("#1976d2") if r == idx else QColor("#1a1a1a")
            for c in range(self.waypoint_table.columnCount()):
                item = self.waypoint_table.item(r, c)
                if item is not None:
                    item.setBackground(color)
        self.sb_op_label.setText(
            f"Running waypoint {idx + 1} of {self.waypoint_table.rowCount()}"
        )

    def _on_sequence_row_done(self, idx: int, err: str) -> None:
        if err:
            QMessageBox.warning(self, "Sequence error", err)

    def _on_sequence_done(self, err: str) -> None:
        self._sequence_in_progress = False
        # Don't drop ref here — see _on_axis_abs_done docstring (QThread GC race).
        self.run_seq_btn.setEnabled(True)
        self.stop_seq_btn.setEnabled(False)
        # Clear row highlights.
        for r in range(self.waypoint_table.rowCount()):
            for c in range(self.waypoint_table.columnCount()):
                item = self.waypoint_table.item(r, c)
                if item is not None:
                    item.setBackground(QColor("#1a1a1a"))
        if err:
            self.sb_op_label.setText(f"Sequence aborted: {err[:80]}")
        else:
            self.sb_op_label.setText("Sequence complete")
        QTimer.singleShot(500, self._stop_logger_if_auto)
        self._update_all_button_states()

    # ------------------------------------------------------------------
    # Emergency stop & global controls
    # ------------------------------------------------------------------
    def _emergency_stop_all(self) -> None:
        # SAFETY-CRITICAL PATH.
        # ====================================================================
        # Why we DO NOT do ``with self._controller_lock:`` here:
        #   The FMC4030 SDK is thread-safe enough that issuing ``stop_axis``
        #   while another SDK call is in flight on the lock-holding worker is
        #   strictly safer than waiting for the lock and never stopping. If a
        #   worker is mid ``is_axis_stopped`` poll (lock held), blocking E-Stop
        #   on the lock could delay the stop by an entire SDK round-trip per
        #   axis. We try-acquire with a 50 ms timeout for clean locking when
        #   we can get it, then proceed regardless. The two parallel SDK calls
        #   may overlap; the stop wins because the controller serializes
        #   incoming commands in its own queue.
        # ====================================================================
        print("[estop-debug] click handler entered", file=sys.stderr, flush=True)
        t0 = time.monotonic()

        # Set both abort signals BEFORE any SDK contact so workers see them
        # the instant they wake up.
        EMERGENCY_STOP.set()
        self._abort_event.set()

        # Abort any running experiment immediately.
        if self._experiment_runner is not None:
            try:
                self._experiment_runner.stop_experiment()
            except Exception:
                pass

        if not self.connected:
            self._show_estop_banner(t0, [], lock_acquired=False, note="not connected")
            return

        acquired = self._controller_lock.acquire(timeout=0.05)
        per_axis_results: list[tuple[str, str]] = []  # (axis_name, "ok" or err)
        try:
            # 1. stop_run FIRST — only FMC4030_Stop_Run interrupts firmware-level
            #    homing (FMC4030_Home_Single_Axis ignores per-axis stop commands).
            try:
                self.controller.stop_run()
                print("[estop] stop_run issued", file=sys.stderr, flush=True)
            except FMC4030Error as exc:
                print(f"[estop] stop_run FMC4030Error: {exc}", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[estop] stop_run unexpected: {exc}", file=sys.stderr, flush=True)
            # 2. Per-axis immediate stops (mode=2) for jog / move operations.
            for axis in AXES:
                try:
                    self.controller.stop_axis(axis, mode=2)
                    per_axis_results.append((axis.name, "ok"))
                    print(f"[estop] stop_axis {axis.name} issued", file=sys.stderr, flush=True)
                except FMC4030Error as exc:
                    per_axis_results.append((axis.name, f"{exc}"))
                    print(f"[estop] stop_axis {axis.name} FMC4030Error: {exc}", file=sys.stderr, flush=True)
                except Exception as exc:
                    per_axis_results.append((axis.name, f"unexpected: {exc}"))
                    print(f"[estop] stop_axis {axis.name} unexpected: {exc}", file=sys.stderr, flush=True)
        finally:
            if acquired:
                self._controller_lock.release()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        print(f"[estop-debug] completed in {elapsed_ms:.2f} ms "
              f"(lock_acquired={acquired})", file=sys.stderr, flush=True)

        self._stop_logger_now()
        # Verify all axes actually stopped ~200 ms after issuing the commands.
        QTimer.singleShot(200, self._verify_estop_complete)
        self._show_estop_banner(t0, per_axis_results, lock_acquired=acquired)

    def _show_estop_banner(self, t0_mono: float, per_axis_results: list[tuple[str, str]],
                           *, lock_acquired: bool, note: str = "") -> None:
        """Surface the E-Stop outcome: yellow banner + non-blocking info dialog."""
        now_str = datetime.now().strftime("%H:%M:%S")
        if note:
            txt = f"EMERGENCY STOP triggered at {now_str} ({note}). Click Reset to resume."
        else:
            txt = f"EMERGENCY STOP triggered at {now_str} — click Reset to resume."
        if getattr(self, "_estop_banner", None) is not None:
            self._estop_banner_label.setText(txt)
            self._estop_banner.show()
        self.sb_op_label.setText("EMERGENCY STOP — click Reset E-Stop to resume")
        self.sb_op_label.setStyleSheet("color: #ef5350; font-weight: bold;")

        # Non-blocking info dialog so it doesn't freeze the GUI while the user
        # reads it.
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Emergency Stop fired")
        body_lines = [f"Stop issued at {now_str} (lock acquired: {lock_acquired})."]
        if per_axis_results:
            body_lines.append("")
            body_lines.append("Per-axis stop results:")
            for name, result in per_axis_results:
                body_lines.append(f"  {name}: {result}")
        if note:
            body_lines.append(f"\nNote: {note}")
        msg.setText("\n".join(body_lines))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowModality(Qt.NonModal)
        msg.show()  # non-blocking

    def _verify_estop_complete(self) -> None:
        """Best-effort non-blocking verification fired 200 ms after E-Stop.
        Queries is_axis_stopped for each axis and logs the result. A warning
        is printed (but no action taken) if an axis is still moving — the user
        can see this in stderr and repeat the E-Stop."""
        if not self.connected:
            return
        for axis in AXES:
            try:
                stopped = self.controller.is_axis_stopped(axis)
                status = "✓ stopped" if stopped else "⚠ still moving"
                print(f"[estop-verify] {axis.name}: {status}", file=sys.stderr, flush=True)
                if not stopped:
                    print(
                        f"[estop-verify] WARNING: {axis.name} did not stop within 200 ms. "
                        "Check that stop_run() is effective on this controller.",
                        file=sys.stderr, flush=True,
                    )
            except Exception as exc:
                print(f"[estop-verify] {axis.name}: check failed: {exc}", file=sys.stderr, flush=True)

    def _reset_estop(self) -> None:
        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        if getattr(self, "_estop_banner", None) is not None:
            self._estop_banner.hide()
        self.sb_op_label.setText("Idle")
        self.sb_op_label.setStyleSheet("color: #ccc;")
        self._update_all_button_states()

    def _blocked_by_estop(self) -> bool:
        """Refuse to start new motion while the user-acknowledged abort flag
        is still set. Forces an explicit Reset click after every E-Stop."""
        if self._abort_event.is_set():
            QMessageBox.warning(
                self, "Emergency Stop active",
                "Emergency Stop is active. Click 'Reset E-Stop' in the yellow "
                "banner before starting another motion.",
            )
            return True
        return False

    def _run_global_command(self, cmd: str) -> None:
        if not self.connected:
            return
        try:
            with self._controller_lock:
                if cmd == "pause":
                    self.controller.pause_run(0x07)
                elif cmd == "resume":
                    self.controller.resume_run(0x07)
                elif cmd == "stop":
                    self.controller.stop_run()
        except FMC4030Error as exc:
            QMessageBox.critical(self, "Controller error", str(exc))

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------
    def _menu_open_waypoints(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Waypoints CSV", "",
                                              "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception as exc:
            QMessageBox.warning(self, "CSV read error", str(exc))
            return
        self.waypoint_table.setRowCount(0)
        for row in rows:
            r = self.waypoint_table.rowCount()
            self.waypoint_table.insertRow(r)
            try:
                cells = [
                    row.get("x_mm", "0"),
                    row.get("y_mm", "0"),
                    row.get("z_mm", "0"),
                    row.get("speed_mm_s", str(self.move_speed_spin.value())),
                    row.get("dwell_s", "0"),
                ]
            except Exception:
                cells = ["0", "0", "0", str(self.move_speed_spin.value()), "0"]
            for c, val in enumerate(cells):
                self.waypoint_table.setItem(r, c, QTableWidgetItem(val))

    def _menu_save_waypoints(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Waypoints CSV", "",
                                              "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["x_mm", "y_mm", "z_mm", "speed_mm_s", "dwell_s"])
                for r in range(self.waypoint_table.rowCount()):
                    writer.writerow([
                        self.waypoint_table.item(r, c).text() if self.waypoint_table.item(r, c) else ""
                        for c in range(5)
                    ])
        except Exception as exc:
            QMessageBox.warning(self, "CSV write error", str(exc))

    def _menu_toggle_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        cur = app.styleSheet()
        if cur:
            app.setStyleSheet("")
        elif HAVE_QDARKSTYLE:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            app.setStyleSheet(FALLBACK_DARK_QSS)

    def _menu_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<b>UMI Gantry Control Panel</b><br>"
            "PyQt5 live UI on top of <code>gantry_runner.py</code>.<br><br>"
            f"PyQt5: {QtCore.PYQT_VERSION_STR}<br>"
            f"pyqtgraph: {'yes' if HAVE_PYQTGRAPH else 'no'}<br>"
            f"qdarkstyle: {'yes' if HAVE_QDARKSTYLE else 'no'}<br>"
            f"qtawesome: {'yes' if HAVE_QTAWESOME else 'no'}<br>"
            f"Mode: {'MOCK' if self.is_mock else 'real controller'}"
        )

    # ------------------------------------------------------------------
    # Button enable/disable orchestration
    # ------------------------------------------------------------------
    def _update_all_button_states(self) -> None:
        connected = self.connected
        busy = (
            self._move_in_progress
            or self._sequence_in_progress
            or bool(self._per_axis_busy)
        )
        # Motion / record buttons enabled only if connected and not busy.
        self.move_btn.setEnabled(connected and not busy)
        self.use_current_btn.setEnabled(connected)
        self.run_seq_btn.setEnabled(connected and not busy)
        self.record_btn.setEnabled(connected)
        self.refresh_btn.setEnabled(connected)
        self.pause_btn.setEnabled(connected)
        self.resume_btn.setEnabled(connected)
        self.stop_run_btn.setEnabled(connected)
        # Per-Axis Control cards: jog + Move Abs + Go Home buttons — disabled
        # when disconnected or when ANY Move Abs is in flight on any axis.
        for axis in AXES:
            info = self.per_axis_cards.get(axis)
            if info is None:
                continue
            for jog_btn in info["jog_btns"]:
                jog_btn.setEnabled(connected and not busy)
            info["move_btn"].setEnabled(
                connected and not busy and int(axis) not in self._per_axis_busy
            )
            info["home_btn"].setEnabled(connected and not busy)
            info["target_spin"].setEnabled(connected)
        # Emergency stop: always enabled when connected.
        self.estop_btn.setEnabled(connected)

    # ------------------------------------------------------------------
    # Left-pane splitter persistence
    # ------------------------------------------------------------------
    def _on_left_splitter_moved(self, pos: int, index: int) -> None:
        sizes = self._left_splitter.sizes()
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["left_splitter_sizes"] = sizes
        _gp_save_section("gantry_panel", payload)

    # ------------------------------------------------------------------
    # Camera session (persistent connection, shared with experiment runner)
    # ------------------------------------------------------------------
    def _toggle_camera(self) -> None:
        if self._camera is not None and self._camera.is_open:
            self._disconnect_camera()
        else:
            self._connect_camera()

    def _connect_camera(self) -> None:
        if not HAVE_FISHEYE_CAMERA:
            QMessageBox.warning(
                self, "fisheye_camera.py missing",
                "Place fisheye_camera.py in src/ to enable the panel camera session.",
            )
            return
        # Persist current settings.
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["camera"] = {
            "device":      self._cam_device_spin.value(),
            "resolution":  self._cam_res_combo.currentText(),
            "fps":         self._cam_fps_spin.value(),
            "calib_path":  self._cam_calib_edit.text().strip(),
        }
        _gp_save_section("gantry_panel", payload)

        if self._camera is None:
            self._camera = FisheyeCameraSession(self)
            self._camera.state_changed.connect(self._on_camera_state_changed)
            self._camera.frame_ready.connect(self._fisheye_preview.on_frame)
            self._camera.stats.connect(self._on_camera_stats)
            self._camera.error.connect(self._on_camera_error)

        res_text = self._cam_res_combo.currentText().replace("×", "x")
        try:
            w, h = (int(v) for v in res_text.split("x"))
        except ValueError:
            w, h = 1280, 720

        calib_path = _resolve_calib_path(self._cam_calib_edit.text())
        mock = self._is_mock_camera

        self._cam_connect_btn.setEnabled(False)
        self._cam_connect_btn.setText("Connecting…")

        self._camera.open(
            device=self._cam_device_spin.value(),
            width=w, height=h,
            fps=self._cam_fps_spin.value(),
            calib_path=calib_path,
            mock=mock,
        )

    def _disconnect_camera(self) -> None:
        if self._camera is not None:
            self._camera.close()

    def _on_camera_state_changed(self, state: str) -> None:
        if state == "disconnected":
            self._cam_connect_btn.setText("Connect Camera")
            self._cam_connect_btn.setEnabled(True)
            self._cam_status_label.setText("● Disconnected")
            self._cam_status_label.setStyleSheet("color: #ef5350; font-weight: bold;")
            if hasattr(self, "_fisheye_preview"):
                self._fisheye_preview.set_state("disconnected")
        elif state == "connecting":
            self._cam_connect_btn.setText("Connecting…")
            self._cam_connect_btn.setEnabled(False)
            self._cam_status_label.setText("● Connecting…")
            self._cam_status_label.setStyleSheet("color: #ffa726; font-weight: bold;")
            if hasattr(self, "_fisheye_preview"):
                self._fisheye_preview.set_state("connecting")
        elif state in ("connected", "connected_mock"):
            mock_tag = " (mock)" if state == "connected_mock" else ""
            self._cam_connect_btn.setText("Disconnect Camera")
            self._cam_connect_btn.setEnabled(True)
            color = "#ffd54f" if state == "connected_mock" else "#66bb6a"
            self._cam_status_label.setText(f"● Connected{mock_tag}")
            self._cam_status_label.setStyleSheet(
                f"color: {color}; font-weight: bold;"
            )
            if hasattr(self, "_fisheye_preview"):
                self._fisheye_preview.set_state(state)
            self._update_exp_camera_summary()
        elif state == "error":
            self._cam_connect_btn.setText("Connect Camera")
            self._cam_connect_btn.setEnabled(True)
            self._cam_status_label.setText("● Error")
            self._cam_status_label.setStyleSheet("color: #ef5350; font-weight: bold;")
            if hasattr(self, "_fisheye_preview"):
                self._fisheye_preview.set_state("error")
        self._exp_refresh_checklist()

    def _on_camera_stats(self, fps: int, grab_ms: float) -> None:
        if hasattr(self, "_fisheye_preview") and self._camera is not None:
            self._fisheye_preview.update_stats(fps, grab_ms, self._camera.device_config)

    def _on_camera_error(self, msg: str) -> None:
        print(f"[camera-session] {msg}", file=sys.stderr)
        self.sb_op_label.setText(f"Camera error: {msg[:80]}")
        self.sb_op_label.setStyleSheet("color: #ef5350;")

    def _cam_browse_calib(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Fisheye Calibration YAML", "", "YAML Files (*.yaml *.yml)"
        )
        if p:
            self._cam_calib_edit.setText(p)

    def _update_exp_camera_summary(self) -> None:
        """Refresh the read-only camera summary label in the Experiment tab."""
        if not hasattr(self, "_exp_camera_summary"):
            return
        if self._camera is None or not self._camera.is_open:
            self._exp_camera_summary.setText(
                "Camera: not connected — use top connection bar"
            )
            return
        cfg = self._camera.device_config
        w = cfg.get("width", "?")
        h = cfg.get("height", "?")
        fps = cfg.get("fps", "?")
        dev = cfg.get("device", "?")
        mock = cfg.get("mock", False)
        calib_name = ""
        cp = self._camera.calib_path
        if cp:
            calib_name = f" · calib={cp.name}"
        mock_tag = " (mock)" if mock else ""
        self._exp_camera_summary.setText(
            f"Camera: Device {dev} · {w}×{h} · {fps} FPS{calib_name}{mock_tag}"
        )

    # ------------------------------------------------------------------
    # Tab persistence + clipping audit
    # ------------------------------------------------------------------
    def _on_tab_changed(self, idx: int) -> None:
        _gp_save_section("gantry_panel", {"active_tab": int(idx)})

    def _audit_clipping(self) -> None:
        """One-shot sanity walk: flag any QLabel/QPushButton whose sizeHint
        exceeds its actual rect by more than 2 px. Visible-only — we don't
        warn about widgets on inactive tabs."""
        reported = 0
        for w in self.findChildren((QtWidgets.QLabel, QtWidgets.QPushButton)):
            try:
                if not w.isVisible():
                    continue
                hint_w = w.sizeHint().width()
                actual_w = w.width()
                if hint_w > actual_w + 2:
                    text = w.text() if hasattr(w, "text") else ""
                    print(
                        f"[clip-audit] {type(w).__name__} '{text}' "
                        f"hint={hint_w} actual={actual_w}",
                        file=sys.stderr,
                    )
                    reported += 1
            except RuntimeError:
                continue
        if reported == 0:
            print("[clip-audit] OK — no clipped widgets visible.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._status_timer.stop()
        if self._logger is not None:
            try:
                self._logger.stop()
            except Exception:
                pass
            self._logger = None
        # Tell workers to wrap up; they all check EMERGENCY_STOP in their loops.
        EMERGENCY_STOP.set()

        def _safe_wait(t) -> None:
            # The C++ QThread may have already been deleteLater'd; the sip
            # wrapper then raises RuntimeError on any attribute access.
            if t is None:
                return
            try:
                if t.isRunning():
                    t.requestInterruption()
                    t.wait(500)
            except RuntimeError:
                return

        if self._experiment_runner is not None:
            try:
                self._experiment_runner.stop_experiment()
            except Exception:
                pass

        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass

        for thread_attr in ("_status_thread", "_move_thread", "_sequence_thread"):
            _safe_wait(getattr(self, thread_attr, None))
        for t in list(self._per_axis_threads.values()):
            _safe_wait(t)
        if self.connected:
            try:
                with self._controller_lock:
                    self.controller.close()
            except Exception:
                pass
        if hasattr(self, "_fisheye_preview"):
            try:
                self._fisheye_preview.shutdown()
            except Exception:
                pass
        super().closeEvent(event)


# =============================================================================
# main entry
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FMC4030 live control panel (PyQt5).")
    p.add_argument("--mock", action="store_true",
                   help="Use an in-process MockFMC4030Controller (smoke tests).")
    p.add_argument("--mock-camera", action="store_true",
                   help="Inject a synthetic fisheye stream for the Experiment tab "
                        "(no real camera needed; useful with --mock).")
    p.add_argument("--light", action="store_true",
                   help="Skip the dark theme.")
    return p.parse_args(argv)


def _install_sigint(app: QApplication, window: GantryPanel) -> QTimer:
    """Bridge POSIX SIGINT into the Qt event loop.

    Returns a heartbeat QTimer that must be kept alive (don't let it GC).
    """
    def handler(signum, frame):
        del signum, frame
        QTimer.singleShot(0, window._emergency_stop_all)
        QTimer.singleShot(200, app.quit)
    signal.signal(signal.SIGINT, handler)
    # No-op timer keeps the Python interpreter awake periodically so signals
    # can be delivered while Qt is blocked in C event-loop waits.
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(200)
    return heartbeat


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)

    if not args.light:
        if HAVE_QDARKSTYLE:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            app.setStyleSheet(FALLBACK_DARK_QSS)

    if args.mock:
        controller = MockFMC4030Controller()
    else:
        controller = FMC4030Controller()

    window = GantryPanel(controller=controller, is_mock=args.mock)
    window._cli_mock_camera = args.mock_camera       # picked up in __init__
    window._is_mock_camera  = args.mock_camera       # applied immediately
    window.show()

    # Keep a reference to the heartbeat so it doesn't get GC'd.
    _heartbeat = _install_sigint(app, window)
    window._heartbeat = _heartbeat  # type: ignore[attr-defined]

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
