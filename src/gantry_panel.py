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

# sys.path shim: import sibling modules from src/ regardless of cwd.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt5.QtCore import (  # noqa: E402
    QObject, Qt, QThread, QTimer, QUrl, pyqtSignal,
)
from PyQt5.QtGui import (  # noqa: E402
    QColor, QDesktopServices, QFont, QKeySequence,
)
from PyQt5.QtWidgets import (  # noqa: E402
    QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMenuBar, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QShortcut, QSizePolicy, QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
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
    _device_soft_limits_mm, _read_current_pos_mm, _validate_soft_limits,
    make_gantry_run_dir, mm_to_units, move_to_xyz_mm, units_to_mm,
)


# =============================================================================
# Constants
# =============================================================================
HOME_SPEED_LIMIT_UNITS = 20.0       # hard upper bound on home speed (units/s)
STATUS_POLL_MS = 100                # 10 Hz live readout
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
QSplitter::handle { background-color: #2f2f33; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }

/* ---- card-style group boxes (every section uses one) ---- */
QGroupBox {
    background-color: #2b2b2b;
    border: 1px solid #3f3f46;
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 22px;
    padding-left: 12px;
    padding-right: 12px;
    padding-bottom: 12px;
    font-weight: 600;
    font-size: 14px;
    color: #e6e6e6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #2b2b2b;
    color: #4ea1ff;
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
    font-size: 14px;
    padding: 12px;
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
    font-size: 26px;
    color: #34d058;
    background-color: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 10px 14px;
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


class HomingThread(QThread):
    """Sequentially homes one or more axes. Defense-in-depth speed clamping
    is repeated here even though the UI already clamps."""

    axis_started = pyqtSignal(object)
    axis_done = pyqtSignal(object, str)
    all_done = pyqtSignal(str)

    def __init__(self, controller, axes, speed_units: float, acc_dec_units: float,
                 fall_step_units: float, positive_limits: dict[int, bool],
                 lock: threading.RLock,
                 abort_event: threading.Event | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._axes = list(axes)
        self._speed = min(float(speed_units), HOME_SPEED_LIMIT_UNITS)
        self._acc = float(acc_dec_units)
        self._fall = float(fall_step_units)
        # Per-axis direction map, axis index -> True/False (True = positive limit).
        self._positive_limits = dict(positive_limits)
        self._lock = lock
        self._abort_event = abort_event or threading.Event()

    def _aborted(self) -> bool:
        return EMERGENCY_STOP.is_set() or self._abort_event.is_set()

    def run(self) -> None:  # type: ignore[override]
        try:
            for axis in self._axes:
                if self._aborted():
                    self.all_done.emit("Interrupted by emergency stop")
                    return
                self.axis_started.emit(axis)
                try:
                    with self._lock:
                        self._controller.home_axis(
                            axis,
                            speed=self._speed, acc_dec=self._acc,
                            fall_step=self._fall,
                            positive_limit=self._positive_limits.get(
                                int(axis), True),
                        )
                except FMC4030Error as exc:
                    err = f"home_axis({axis.name}) failed: {exc}"
                    self.axis_done.emit(axis, err)
                    self.all_done.emit(err)
                    return
                # Wait for stop, polling at 20 Hz.
                while not self._aborted():
                    try:
                        with self._lock:
                            stopped = self._controller.is_axis_stopped(axis)
                    except FMC4030Error as exc:
                        err = f"is_axis_stopped({axis.name}) failed: {exc}"
                        self.axis_done.emit(axis, err)
                        self.all_done.emit(err)
                        return
                    if stopped:
                        break
                    time.sleep(0.05)
                if self._aborted():
                    self.axis_done.emit(axis, "aborted")
                    self.all_done.emit("Interrupted by emergency stop")
                    return
                self.axis_done.emit(axis, "")
            self.all_done.emit("")
        except Exception as exc:
            self.all_done.emit(f"Unexpected: {exc}")


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


class SoftLimitThread(QThread):
    """Read or read-mutate-write DeviceParameters under the controller lock.

    The whole read-mutate-write happens inside one ``with lock:`` block so any
    concurrent status poll either runs to completion before this starts or
    blocks until the entire update is done.
    """

    result = pyqtSignal(object, str)   # DeviceParameters, msg
    error = pyqtSignal(str)

    def __init__(self, controller, lock: threading.RLock, mode: str,
                 updates_units: dict[int, tuple[int, int]] | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._lock = lock
        self._mode = mode  # "load" or "apply"
        self._updates = updates_units or {}

    def run(self) -> None:  # type: ignore[override]
        try:
            with self._lock:
                params = self._controller.get_device_parameters()
                if self._mode == "apply":
                    for idx, (lo, hi) in self._updates.items():
                        params.soft_limit_min[idx] = int(lo)
                        params.soft_limit_max[idx] = int(hi)
                    self._controller.set_device_parameters(params)
                    params = self._controller.get_device_parameters()
            msg = "Loaded soft limits" if self._mode == "load" else "Applied soft limits"
            self.result.emit(params, msg)
        except FMC4030Error as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"Unexpected: {exc}")


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


# =============================================================================
# Custom widgets
# =============================================================================
class SectionFrame(QtWidgets.QGroupBox):
    """Card-style section. Subclasses QGroupBox so the global stylesheet's
    rounded-card look (``QGroupBox`` + ``QGroupBox::title``) applies uniformly.

    Backwards-compat: callers do ``QHBoxLayout(frame.content())``; ``content()``
    returns ``self`` because QGroupBox accepts a layout directly.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)

    def content(self) -> QWidget:
        return self


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
        self.pos_display.setMinimumHeight(56)
        _fit_label(self.pos_display, "-9999.999")
        layout.addWidget(self.pos_display)

        self.units_label = QLabel("raw: -- units")
        self.units_label.setObjectName("UnitsHint")
        self.units_label.setAlignment(Qt.AlignRight)
        _fit_label(self.units_label, "raw: -99999.99 units")
        layout.addWidget(self.units_label)

        big_font = QFont()
        big_font.setPointSize(13)
        self.vel_label = QLabel("Velocity: -- mm/s")
        self.vel_label.setFont(big_font)
        _fit_label(self.vel_label, "Velocity: -99999.99 mm/s")
        layout.addWidget(self.vel_label)
        self.acc_label = QLabel("Accel: -- mm/s²")
        self.acc_label.setFont(big_font)
        self.acc_label.setToolTip(
            "Derived via 5-sample central difference, not an SDK readout."
        )
        _fit_label(self.acc_label, "Accel: -99999.99 mm/s²")
        layout.addWidget(self.acc_label)

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

        self.setMinimumWidth(220)

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
        self.pos_display.setText(f"{pos_mm:+8.3f}")
        self.units_label.setText(f"raw: {pos_units:+.2f} units")
        self.vel_label.setText(f"Velocity: {vel_mm_s:+.2f} mm/s")
        self.acc_label.setText(f"Accel: {acc_mm_s2:+.2f} mm/s²")
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


# =============================================================================
# Workspace Map (top-down XY + side XZ; pyqtgraph preferred, QPainter fallback)
# =============================================================================
class WorkspaceMap(QtWidgets.QGroupBox):
    """Two stacked 2D plots showing live position, optional target marker,
    a trailing path, and the soft-limit bounding box.

    Public API (called by the panel):
      * update_position(x_mm, y_mm, z_mm)
      * update_target(x_mm, y_mm, z_mm)  /  clear_target()
      * update_soft_limits(min_mm, max_mm)  # each is (x, y, z) of floats|None
      * set_show_trail(bool) / set_show_target(bool) / set_auto_fit(bool)

    Implementation: pyqtgraph if installed (Option A), QPainter fallback
    otherwise (Option B). Both backends share this exact API so callers don't
    care which one is in use.
    """

    TRAIL_MAXLEN = 200
    MIN_GROUP_SIZE = (340, 420)
    AUTO_FIT_MARGIN_MM = 50.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Workspace Map", parent)
        self.setMinimumSize(*self.MIN_GROUP_SIZE)

        self._trail: deque = deque(maxlen=self.TRAIL_MAXLEN)
        self._cur_pos: tuple[float, float, float] | None = None
        self._target: tuple[float, float, float] | None = None
        self._home: tuple[float | None, float | None, float | None] = (None, None, None)
        self._soft_min: tuple[float | None, float | None, float | None] = (None, None, None)
        self._soft_max: tuple[float | None, float | None, float | None] = (None, None, None)
        self._show_trail = True
        self._show_target = True
        self._auto_fit = True

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Always-visible Home header row.
        self.home_header = QLabel("Home: not set")
        self.home_header.setStyleSheet(
            "color: #ffd54f; font-weight: 600; font-size: 12px;"
        )
        v.addWidget(self.home_header)

        # Toolbar: three toggles.
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
        self.fit_chk = QCheckBox("Auto-fit to Soft Limits")
        self.fit_chk.setChecked(True)
        self.fit_chk.toggled.connect(self.set_auto_fit)
        bar.addWidget(self.fit_chk)
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
        self._refresh()

    def update_home(self, home_xyz: tuple) -> None:
        """Set the home-reference triplet (Nones allowed per axis)."""
        self._home = tuple(home_xyz)
        # Update header text.
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
        self._refresh()

    def set_show_trail(self, on: bool) -> None:
        self._show_trail = bool(on)
        self._refresh()

    def set_show_target(self, on: bool) -> None:
        self._show_target = bool(on)
        self._refresh()

    def set_auto_fit(self, on: bool) -> None:
        self._auto_fit = bool(on)
        self._refresh()

    # ---- internal ------------------------------------------------------
    def _refresh(self) -> None:
        view_bounds = self._compute_view_bounds()
        self._backend.render(
            cur_pos=self._cur_pos,
            target=self._target if self._show_target else None,
            trail=list(self._trail) if self._show_trail else [],
            soft_min=self._soft_min,
            soft_max=self._soft_max,
            home=self._home,
            view_bounds=view_bounds,
        )

    def _compute_view_bounds(self) -> dict[str, tuple[float, float]]:
        """Return dict with keys 'x', 'y', 'z' -> (lo, hi) for axes."""
        if self._auto_fit and all(v is not None for v in (*self._soft_min, *self._soft_max)):
            return {
                "x": (float(self._soft_min[0]), float(self._soft_max[0])),
                "y": (float(self._soft_min[1]), float(self._soft_max[1])),
                "z": (float(self._soft_min[2]), float(self._soft_max[2])),
            }
        # Auto-fit OFF (or soft limits unset): bbox of trail ∪ pos ∪ target with margin.
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
            return {"x": (-100.0, 100.0), "y": (-100.0, 100.0), "z": (-100.0, 100.0)}
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

        # XY (top-down). Aspect locked so a 1:1 motion looks square.
        self._xy = pg.PlotWidget(background="#101013")
        self._xy.setAspectLocked(True)
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
        self._xy_box = pg.PlotCurveItem(pen=pg.mkPen(120, 120, 120, 180, width=1))
        self._xy.addItem(self._xy_box)
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
        self._xz_box = pg.PlotCurveItem(pen=pg.mkPen(120, 120, 120, 180, width=1))
        self._xz.addItem(self._xz_box)
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
        v.addWidget(self._xz, stretch=1)

    def render(self, *, cur_pos, target, trail, soft_min, soft_max, home,
               view_bounds) -> None:
        # Soft-limit rectangle.
        if all(v is not None for v in (*soft_min, *soft_max)):
            x0, x1 = float(soft_min[0]), float(soft_max[0])
            y0, y1 = float(soft_min[1]), float(soft_max[1])
            z0, z1 = float(soft_min[2]), float(soft_max[2])
            self._xy_box.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
            self._xz_box.setData([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0])
        else:
            self._xy_box.setData([], [])
            self._xz_box.setData([], [])

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

        # Auto-bounds: only respected when the user keeps auto-fit on; pyqtgraph
        # otherwise handles pan/zoom interactively.
        xr = view_bounds["x"]
        yr = view_bounds["y"]
        zr = view_bounds["z"]
        self._xy.setXRange(*xr, padding=0.05)
        self._xy.setYRange(*yr, padding=0.05)
        self._xz.setYRange(*zr, padding=0.05)


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
               view_bounds) -> None:
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
        self.setMinimumSize(1280, 800)

        # SDK + lock + state ---------------------------------------------------
        self.controller = controller if controller is not None else FMC4030Controller()
        self.is_mock = is_mock
        self._controller_lock = threading.RLock()
        self.connected = False

        # Soft limits (mm) — None means unset.
        self._soft_min_mm: list[float | None] = [None, None, None]
        self._soft_max_mm: list[float | None] = [None, None, None]

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
        self._current_run_dir: Path | None = None

        # Worker handles.
        self._status_thread: StatusPollThread | None = None
        self._move_thread: MoveToTargetThread | None = None
        self._home_thread: HomingThread | None = None
        self._sequence_thread: SequenceThread | None = None
        self._soft_limit_busy = False
        # Panel-scoped abort flag, set together with the gantry_runner module's
        # EMERGENCY_STOP event whenever the E-Stop fires. Workers we own
        # (HomingThread, SequenceThread, AxisAbsMoveThread) check this flag at
        # every loop iteration so they exit quickly on E-Stop without waiting
        # for the SDK lock.
        self._abort_event = threading.Event()
        # Tracks freshness of the last status snapshot for the Polling
        # indicator (set in _on_status_snapshot).
        self._last_snapshot_t: float = 0.0
        # Home reference (mm) per axis, set by successful homing or by the
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
        self._homing_in_progress = False
        self._sequence_in_progress = False
        # Per-axis Move Abs is in progress for at least one axis.
        self._per_axis_busy: set[int] = set()

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

        # ESC for emergency stop — APPLICATION-wide context so it fires even
        # when focus is in a spinbox, table cell, combobox, etc. (default
        # context is WindowShortcut which can be intercepted by focused widgets).
        self._estop_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._estop_shortcut.setContext(Qt.ApplicationShortcut)
        self._estop_shortcut.activated.connect(self._emergency_stop_all)

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
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # Top: connection bar.
        outer.addWidget(self._build_connection_bar())

        # Emergency-Stop banner (hidden until E-Stop fires).
        outer.addWidget(self._build_estop_banner())

        # Middle: horizontal splitter.
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_pane())
        splitter.addWidget(self._build_right_tabs())
        # 38/62 split at 1500 wide → roughly [560, 940].
        splitter.setSizes([560, 940])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # Bottom: global controls bar (Emergency Stop visible at all times).
        outer.addLayout(self._build_global_controls())

    def _build_left_pane(self) -> QWidget:
        pane = QWidget()
        pane.setMinimumWidth(360)
        v = QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)

        # Polling freshness indicator at the very top of the left pane.
        self.poll_indicator = QLabel("Polling: — (no data)")
        self.poll_indicator.setStyleSheet(
            "color: #8a8a8a; font-size: 11px; padding: 2px 4px;"
        )
        _fit_label(self.poll_indicator, "Polling: ✗ 9999 ms ago (STALE)")
        v.addWidget(self.poll_indicator)

        # Live Status section wraps the three AxisStatusCards in a card frame
        # so the section blends with the rest of the GroupBox-styled UI.
        live_section = SectionFrame("Live Status")
        live_inner = QVBoxLayout(live_section.content())
        live_inner.setContentsMargins(0, 0, 0, 0)
        live_inner.setSpacing(8)
        live_inner.addLayout(self._build_status_cards())
        v.addWidget(live_section)

        # Workspace Map.
        self.workspace_map = WorkspaceMap()
        v.addWidget(self.workspace_map, stretch=1)
        return pane

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
            scroll.setWidget(holder)
            return scroll

        # Control tab: per-axis cards on top, combined move-to-target below.
        tabs.addTab(_wrap_scroll(self._build_per_axis_group(), self._build_move_panel()),
                    "Control")
        # Sequences tab: waypoint table + its toolbar.
        tabs.addTab(_wrap_scroll(self._build_waypoint_panel()), "Sequences")
        # Setup tab: soft limits + homing (the "rarely touched" surface).
        tabs.addTab(_wrap_scroll(self._build_soft_limits_group(),
                                 self._build_homing_group()), "Setup")
        # Recording tab: start/stop + CSV path + live 30s plot.
        tabs.addTab(_wrap_scroll(self._build_recording_panel()), "Recording")
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
        frame = SectionFrame("Connection")
        h = QHBoxLayout(frame.content())
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        h.addWidget(QLabel("IP"))
        self.ip_edit = QLineEdit("192.168.0.30")
        self.ip_edit.setMaximumWidth(140)
        h.addWidget(self.ip_edit)

        h.addWidget(QLabel("Port"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8088)
        self.port_spin.setMaximumWidth(80)
        h.addWidget(self.port_spin)

        h.addWidget(QLabel("ID"))
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 31)
        self.id_spin.setValue(1)
        self.id_spin.setMaximumWidth(60)
        h.addWidget(self.id_spin)

        self.connect_btn = QPushButton("Connect")
        ic = _icon("fa5s.plug")
        if ic is not None:
            self.connect_btn.setIcon(ic)
        self.connect_btn.clicked.connect(self._toggle_connection)
        h.addWidget(self.connect_btn)

        h.addSpacing(20)
        h.addWidget(QLabel("Enabled axes:"))
        self.axis_enable_checks: dict[Axis, QCheckBox] = {}
        for axis in AXES:
            cb = QCheckBox(axis.name)
            cb.setChecked(True)
            self.axis_enable_checks[axis] = cb
            h.addWidget(cb)

        h.addStretch()
        self.conn_status_label = QLabel("● Disconnected")
        self.conn_status_label.setStyleSheet("color: #ef5350; font-weight: bold;")
        h.addWidget(self.conn_status_label)

        return frame

    def _build_status_cards(self) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(10)
        self.axis_cards: dict[Axis, AxisStatusCard] = {}
        for axis in AXES:
            card = AxisStatusCard(axis)
            h.addWidget(card, stretch=1)
            self.axis_cards[axis] = card
        return h

    def _build_soft_limits_group(self) -> SectionFrame:
        frame = SectionFrame("Software Limits (mm)")
        grid = QGridLayout(frame.content())
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Axis"), 0, 0)
        grid.addWidget(QLabel("Min (mm)"), 0, 1)
        grid.addWidget(QLabel("Max (mm)"), 0, 2)
        self.soft_limit_spins: dict[Axis, dict[str, QDoubleSpinBox]] = {}
        for row, axis in enumerate(AXES, start=1):
            grid.addWidget(QLabel(axis.name), row, 0)
            mn = QDoubleSpinBox()
            mn.setRange(-100000.0, 100000.0)
            mn.setDecimals(2)
            mn.setValue(0.0)
            _size_mm_spinbox(mn)
            mx = QDoubleSpinBox()
            mx.setRange(-100000.0, 100000.0)
            mx.setDecimals(2)
            mx.setValue(0.0)
            _size_mm_spinbox(mx)
            apply_btn = QPushButton(f"Apply {axis.name}")
            _size_button(apply_btn)
            apply_btn.clicked.connect(partial(self._apply_soft_limits_axis, axis))
            grid.addWidget(mn, row, 1)
            grid.addWidget(mx, row, 2)
            grid.addWidget(apply_btn, row, 3)
            self.soft_limit_spins[axis] = {"min": mn, "max": mx, "apply_btn": apply_btn}

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.load_limits_btn = QPushButton("Load from Controller")
        _size_button(self.load_limits_btn)
        self.load_limits_btn.clicked.connect(self._load_soft_limits)
        btn_row.addWidget(self.load_limits_btn)
        self.apply_all_limits_btn = QPushButton("Apply All")
        _size_button(self.apply_all_limits_btn)
        self.apply_all_limits_btn.clicked.connect(self._apply_all_soft_limits)
        btn_row.addWidget(self.apply_all_limits_btn)
        btn_row.addStretch()
        grid.addLayout(btn_row, len(AXES) + 1, 0, 1, 4)
        return frame

    def _build_homing_group(self) -> SectionFrame:
        frame = SectionFrame("Homing  (UNITS — SDK speaks raw units here)")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        params_row = QGridLayout()
        params_row.addWidget(QLabel("Home Speed (units/s)"), 0, 0)
        self.home_speed_spin = QDoubleSpinBox()
        self.home_speed_spin.setRange(0.1, HOME_SPEED_LIMIT_UNITS)
        self.home_speed_spin.setDecimals(2)
        self.home_speed_spin.setValue(5.0)
        self.home_speed_spin.setToolTip(
            "Homing uses raw controller units because the SDK's home_axis() "
            "takes units directly. Other panels use mm. "
            "Default 5.0 units/s ≈ 41 mm/s on X, 12.5 mm/s on Y, 2.5 mm/s on Z."
        )
        params_row.addWidget(self.home_speed_spin, 0, 1)

        params_row.addWidget(QLabel("Home Accel/Decel (units/s²)"), 0, 2)
        self.home_acc_spin = QDoubleSpinBox()
        self.home_acc_spin.setRange(1.0, 1000.0)
        self.home_acc_spin.setDecimals(1)
        self.home_acc_spin.setValue(20.0)
        params_row.addWidget(self.home_acc_spin, 0, 3)

        params_row.addWidget(QLabel("Fall Step (units)"), 0, 4)
        self.home_fall_spin = QDoubleSpinBox()
        self.home_fall_spin.setRange(0.1, 100.0)
        self.home_fall_spin.setDecimals(2)
        self.home_fall_spin.setValue(5.0)
        params_row.addWidget(self.home_fall_spin, 0, 5)
        v.addLayout(params_row)

        # ---- Per-axis Home Direction row (matches manual_pad.py: each combo
        # has two items, Negative limit (data=False) and Positive limit
        # (data=True); the chosen value is passed as home_axis(...,
        # positive_limit=...) for that axis. Defaults: X=+, Y=+, Z=-.
        dir_row = QHBoxLayout()
        dir_row.setSpacing(12)
        dir_row.addWidget(QLabel("Home Direction:"))
        self.home_dir_combos: dict[Axis, QComboBox] = {}
        # Restore per-axis saved direction (default to Positive for X/Y, Negative for Z).
        saved_dir = _gp_load_settings().get("gantry_panel", {}).get("home_direction", {})
        defaults = {Axis.X: True, Axis.Y: True, Axis.Z: False}
        for axis in AXES:
            sub = QHBoxLayout()
            sub.setSpacing(4)
            sub.addWidget(QLabel(f"{axis.name}:"))
            combo = QComboBox()
            combo.addItem("Negative limit", False)
            combo.addItem("Positive limit", True)
            saved_val = saved_dir.get(axis.name)
            default = saved_val if saved_val is not None else defaults[axis]
            combo.setCurrentIndex(1 if bool(default) else 0)
            combo.setToolTip(
                "'Positive limit' homes toward the +axis end-stop; "
                "'Negative limit' homes toward the −axis end-stop. "
                "Must match the physical limit-switch wiring on this axis."
            )
            combo.currentIndexChanged.connect(self._persist_home_direction)
            sub.addWidget(combo)
            dir_row.addLayout(sub)
            self.home_dir_combos[axis] = combo
        dir_row.addStretch()
        v.addLayout(dir_row)

        # "Set Current as Home Reference" — captures the latest snapshot's mm
        # values for all 3 axes without commanding any motion.
        ref_row = QHBoxLayout()
        ref_row.setSpacing(8)
        set_home_btn = QPushButton("Set Current as Home Reference")
        _size_button(set_home_btn)
        set_home_btn.setToolTip(
            "Mark the current XYZ position as the home reference (no motion). "
            "Used by the Δ-home readouts on each axis card."
        )
        set_home_btn.clicked.connect(self._set_current_as_home_reference)
        ref_row.addWidget(set_home_btn)
        ref_row.addStretch()
        v.addLayout(ref_row)

        btn_row = QHBoxLayout()
        ic_home = _icon("fa5s.home")
        self.home_btns: dict[Axis, QPushButton] = {}
        for axis in AXES:
            b = QPushButton(f"Home {axis.name}")
            if ic_home is not None:
                b.setIcon(ic_home)
            b.clicked.connect(partial(self._home_single, axis))
            btn_row.addWidget(b)
            self.home_btns[axis] = b

        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("Home All order:"))
        self.home_order_combo = QComboBox()
        self.home_order_combo.addItem("Z → X → Y  (safest)", ("Z", "X", "Y"))
        self.home_order_combo.addItem("X → Y → Z", ("X", "Y", "Z"))
        self.home_order_combo.addItem("Y → X → Z", ("Y", "X", "Z"))
        btn_row.addWidget(self.home_order_combo)
        self.home_all_btn = QPushButton("Home All")
        if ic_home is not None:
            self.home_all_btn.setIcon(ic_home)
        self.home_all_btn.clicked.connect(self._home_all)
        btn_row.addWidget(self.home_all_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)
        return frame

    def _build_per_axis_group(self) -> SectionFrame:
        """Per-axis cards: hold-to-jog, Move Abs (mm), and per-axis Home shortcut.
        Shared jog/move parameter row at the top (independent from the combined
        Move-to-Target panel's speed/accel/decel)."""
        frame = SectionFrame("Per-Axis Control  (mm)")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        # Shared jog/move parameter row.
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Jog/Move Speed (mm/s)"))
        self.jog_speed_spin = QDoubleSpinBox()
        self.jog_speed_spin.setRange(0.01, 5000.0)
        self.jog_speed_spin.setDecimals(2)
        self.jog_speed_spin.setValue(20.0)
        params_row.addWidget(self.jog_speed_spin)

        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Accel (mm/s²)"))
        self.jog_acc_spin = QDoubleSpinBox()
        self.jog_acc_spin.setRange(0.01, 10000.0)
        self.jog_acc_spin.setDecimals(2)
        self.jog_acc_spin.setValue(50.0)
        params_row.addWidget(self.jog_acc_spin)

        params_row.addSpacing(12)
        params_row.addWidget(QLabel("Decel (mm/s²)"))
        self.jog_dec_spin = QDoubleSpinBox()
        self.jog_dec_spin.setRange(0.01, 10000.0)
        self.jog_dec_spin.setDecimals(2)
        self.jog_dec_spin.setValue(50.0)
        params_row.addWidget(self.jog_dec_spin)
        params_row.addStretch()
        v.addLayout(params_row)

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
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        # Axis letter.
        letter = QLabel(axis.name)
        letter.setObjectName("axis-letter")
        letter.setAlignment(Qt.AlignCenter)
        v.addWidget(letter)

        # Position readouts (mm primary, units secondary).
        pos_mm = QLabel("--")
        pos_mm.setObjectName("PositionReadout")
        pos_mm.setMinimumHeight(48)
        _fit_label(pos_mm, "-9999.999")
        v.addWidget(pos_mm)
        pos_units = QLabel("-- units")
        pos_units.setObjectName("UnitsHint")
        pos_units.setAlignment(Qt.AlignRight)
        _fit_label(pos_units, "-99999.99 units")
        v.addWidget(pos_units)

        # Velocity.
        vel = QLabel("Vel: -- mm/s")
        vel.setStyleSheet("color: #aaa; font-size: 11px;")
        _fit_label(vel, "Vel: -99999.99 mm/s")
        v.addWidget(vel)

        # Jog row (hold to jog).
        jog_row = QHBoxLayout()
        jog_row.setSpacing(8)
        btn_pos = QPushButton(f"{axis.name}+")
        btn_neg = QPushButton(f"{axis.name}-")
        for jb in (btn_pos, btn_neg):
            _size_button(jb)
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

        # Absolute move row.
        abs_row = QHBoxLayout()
        abs_row.setSpacing(8)
        target_spin = QDoubleSpinBox()
        target_spin.setRange(-100000.0, 100000.0)
        target_spin.setDecimals(3)
        target_spin.setValue(0.0)
        _size_mm_spinbox(target_spin)
        abs_row.addWidget(target_spin, stretch=1)
        move_btn = QPushButton("Move Abs")
        _size_button(move_btn)
        move_btn.clicked.connect(partial(self._move_axis_abs, axis))
        abs_row.addWidget(move_btn)
        v.addLayout(abs_row)

        # Per-axis Home button — reuses the existing _home_single() entry point
        # which spawns the same HomingThread the Homing group uses, consuming
        # the shared home_speed_spin / home_acc_spin / home_fall_spin /
        # home_dir_combo values.
        home_btn = QPushButton(f"Home {axis.name}")
        _size_button(home_btn)
        ic = _icon("fa5s.home")
        if ic is not None:
            home_btn.setIcon(ic)
            home_btn.setObjectName("IconButton")
        home_btn.clicked.connect(partial(self._home_single, axis))
        v.addWidget(home_btn)

        self.per_axis_cards[axis] = {
            "card": card,
            "pos_mm": pos_mm,
            "pos_units": pos_units,
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
            sp.setMinimumHeight(36)
            f = QFont()
            f.setPointSize(13)
            sp.setFont(f)
            grid.addWidget(sp, 1, col)
            self.target_spins[axis] = sp

        grid.addWidget(QLabel("Speed (mm/s)"), 2, 0)
        self.move_speed_spin = QDoubleSpinBox()
        self.move_speed_spin.setRange(0.01, 5000.0)
        self.move_speed_spin.setDecimals(2)
        self.move_speed_spin.setValue(20.0)
        _size_mm_spinbox(self.move_speed_spin)
        grid.addWidget(self.move_speed_spin, 3, 0)

        grid.addWidget(QLabel("Accel (mm/s²)"), 2, 1)
        self.move_acc_spin = QDoubleSpinBox()
        self.move_acc_spin.setRange(0.01, 10000.0)
        self.move_acc_spin.setDecimals(2)
        self.move_acc_spin.setValue(50.0)
        _size_mm_spinbox(self.move_acc_spin)
        grid.addWidget(self.move_acc_spin, 3, 1)

        grid.addWidget(QLabel("Decel (mm/s²)"), 2, 2)
        self.move_dec_spin = QDoubleSpinBox()
        self.move_dec_spin.setRange(0.01, 10000.0)
        self.move_dec_spin.setDecimals(2)
        self.move_dec_spin.setValue(50.0)
        _size_mm_spinbox(self.move_dec_spin)
        grid.addWidget(self.move_dec_spin, 3, 2)

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

    def _build_waypoint_panel(self) -> SectionFrame:
        frame = SectionFrame("Waypoint Sequence")
        v = QVBoxLayout(frame.content())
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self.waypoint_table = QTableWidget(0, 5)
        self.waypoint_table.setHorizontalHeaderLabels(
            ["X (mm)", "Y (mm)", "Z (mm)", "Speed (mm/s)", "Dwell (s)"]
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
        v.addLayout(btn_row)
        return frame

    def _build_global_controls(self) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(8)
        h.setContentsMargins(8, 8, 8, 8)
        self.refresh_btn = QPushButton("🔄 Refresh Position")
        _size_button(self.refresh_btn)
        self.refresh_btn.clicked.connect(self._refresh_position)
        h.addWidget(self.refresh_btn)

        self.pause_btn = QPushButton("Pause Run")
        _size_button(self.pause_btn)
        self.pause_btn.clicked.connect(lambda: self._run_global_command("pause"))
        h.addWidget(self.pause_btn)
        self.resume_btn = QPushButton("Resume Run")
        _size_button(self.resume_btn)
        self.resume_btn.clicked.connect(lambda: self._run_global_command("resume"))
        h.addWidget(self.resume_btn)
        self.stop_run_btn = QPushButton("Stop Run")
        _size_button(self.stop_run_btn)
        self.stop_run_btn.clicked.connect(lambda: self._run_global_command("stop"))
        h.addWidget(self.stop_run_btn)

        h.addStretch()
        self.estop_btn = QPushButton("⚠  EMERGENCY STOP ALL  (Esc)")
        self.estop_btn.setObjectName("EmergencyButton")
        _size_button(self.estop_btn, "emergency")
        self.estop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ic_stop = _icon("fa5s.stop-circle")
        if ic_stop is not None:
            self.estop_btn.setIcon(ic_stop)
        self.estop_btn.clicked.connect(self._emergency_stop_all)
        h.addWidget(self.estop_btn, stretch=2)
        return h

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
        # Auto-load soft limits + populate display.
        self._load_soft_limits()
        self._update_all_button_states()

    def _disconnect(self) -> None:
        if self._logger is not None:
            try:
                self._logger.stop()
            except Exception:
                pass
            self._logger = None
        self._status_timer.stop()
        for thread_attr in ("_status_thread", "_move_thread", "_home_thread", "_sequence_thread"):
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
        # Clear axis cards.
        for card in self.axis_cards.values():
            card.update_state(0.0, 0.0, 0.0, 0.0)
            card.set_soft_limits(None, None)
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
        pos_mm = [units_to_mm(pos_units[i], AXES[i]) for i in range(3)]
        vel_mm = [units_to_mm(vel_units[i], AXES[i]) for i in range(3)]

        # Acceleration: SMA-smoothed central difference over 5 samples.
        self._vel_buffer.append((now, tuple(vel_mm)))
        acc_mm = self._compute_accel_mm_s2()

        # Rate-limited diagnostic.
        if DEBUG_STATUS_POLL:
            if now - getattr(self, "_dbg_snap_t", 0.0) >= 1.0:
                self._dbg_snap_t = now
                print(f"[status-poll] X={pos_mm[0]:+.3f} Y={pos_mm[1]:+.3f} "
                      f"Z={pos_mm[2]:+.3f} mm  vel(mm/s)=({vel_mm[0]:+.2f},"
                      f"{vel_mm[1]:+.2f},{vel_mm[2]:+.2f})",
                      file=sys.stderr, flush=True)

        # Cache freshness for the Polling indicator.
        self._last_snapshot_t = now
        self._last_pos_mm = tuple(pos_mm)
        self._last_pos_units = tuple(pos_units)

        # Update cards (now passing per-axis home reference).
        for i, axis in enumerate(AXES):
            home_mm = self._home_position_mm[axis]
            self.axis_cards[axis].update_state(
                pos_units[i], pos_mm[i], vel_mm[i], acc_mm[i],
                home_mm=home_mm,
            )
            # Mirror onto the Per-Axis Control card readouts.
            info = self.per_axis_cards.get(axis)
            if info is not None:
                info["pos_mm"].setText(f"{pos_mm[i]:+8.3f}")
                info["pos_units"].setText(f"{pos_units[i]:+.2f} units")
                info["vel"].setText(f"Vel: {vel_mm[i]:+.2f} mm/s")

        # Update plot history.
        self._time_history.append(now - self._t0_mono)
        for i in range(3):
            self._pos_history[i].append(pos_mm[i])
        self.plot_widget.update_data(self._time_history, self._pos_history)

        # Workspace Map: same snapshot, no extra SDK call.
        if getattr(self, "workspace_map", None) is not None:
            self.workspace_map.update_position(pos_mm[0], pos_mm[1], pos_mm[2])
            # Keep the home marker visible/updated even when no home is set.
            home_xyz = tuple(self._home_position_mm[a] for a in AXES)
            self.workspace_map.update_home(home_xyz)

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

    # ------------------------------------------------------------------
    # Soft limits
    # ------------------------------------------------------------------
    def _load_soft_limits(self) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._soft_limit_busy:
            return
        self._soft_limit_busy = True
        self.sb_op_label.setText("Loading soft limits…")
        t = SoftLimitThread(self.controller, self._controller_lock, "load")
        t.result.connect(self._on_soft_limits_result)
        t.error.connect(self._on_soft_limits_error)
        t.finished.connect(t.deleteLater)
        t.start()
        # Keep a reference until done (Qt would otherwise GC it).
        self._soft_limit_thread = t

    def _apply_soft_limits_axis(self, axis: Axis) -> None:
        self._apply_soft_limits_for([axis], confirm=False)

    def _apply_all_soft_limits(self) -> None:
        # Build the diff text for the confirmation dialog.
        diff_lines = []
        for i, axis in enumerate(AXES):
            cur_lo = self._soft_min_mm[i]
            cur_hi = self._soft_max_mm[i]
            new_lo = self.soft_limit_spins[axis]["min"].value()
            new_hi = self.soft_limit_spins[axis]["max"].value()
            cur_lo_s = f"{cur_lo:+.2f}" if cur_lo is not None else " unset"
            cur_hi_s = f"{cur_hi:+.2f}" if cur_hi is not None else " unset"
            diff_lines.append(
                f"  {axis.name}: [{cur_lo_s}, {cur_hi_s}] → [{new_lo:+.2f}, {new_hi:+.2f}] mm"
            )
        reply = QMessageBox.question(
            self, "Apply All Soft Limits",
            "Apply these soft limits to the controller?\n\n" + "\n".join(diff_lines),
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._apply_soft_limits_for(list(AXES), confirm=False)

    def _apply_soft_limits_for(self, axes: list[Axis], confirm: bool) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._soft_limit_busy:
            return
        updates: dict[int, tuple[int, int]] = {}
        for axis in axes:
            sp = self.soft_limit_spins[axis]
            mn_mm = sp["min"].value()
            mx_mm = sp["max"].value()
            mn_units = int(round(mm_to_units(mn_mm, axis)))
            mx_units = int(round(mm_to_units(mx_mm, axis)))
            updates[int(axis)] = (mn_units, mx_units)
        self._soft_limit_busy = True
        self.sb_op_label.setText("Applying soft limits…")
        t = SoftLimitThread(self.controller, self._controller_lock,
                            "apply", updates_units=updates)
        t.result.connect(self._on_soft_limits_result)
        t.error.connect(self._on_soft_limits_error)
        t.finished.connect(t.deleteLater)
        t.start()
        self._soft_limit_thread = t

    def _on_soft_limits_result(self, params, msg: str) -> None:
        self._soft_limit_busy = False
        for i, axis in enumerate(AXES):
            lo_units = float(params.soft_limit_min[i])
            hi_units = float(params.soft_limit_max[i])
            if lo_units == 0.0 and hi_units == 0.0:
                lo_mm = hi_mm = None
            else:
                lo_mm = units_to_mm(lo_units, axis)
                hi_mm = units_to_mm(hi_units, axis)
            self._soft_min_mm[i] = lo_mm
            self._soft_max_mm[i] = hi_mm
            sp = self.soft_limit_spins[axis]
            sp["min"].blockSignals(True)
            sp["max"].blockSignals(True)
            sp["min"].setValue(lo_mm if lo_mm is not None else 0.0)
            sp["max"].setValue(hi_mm if hi_mm is not None else 0.0)
            sp["min"].blockSignals(False)
            sp["max"].blockSignals(False)
            self.axis_cards[axis].set_soft_limits(lo_mm, hi_mm)
            # Clamp BOTH the combined Move-to-Target spinbox AND the per-axis
            # card's Move Abs spinbox to the loaded range.
            for target_sp in (self.target_spins[axis],
                              self.per_axis_cards.get(axis, {}).get("target_spin")):
                if target_sp is None:
                    continue
                if lo_mm is not None and hi_mm is not None and hi_mm > lo_mm:
                    target_sp.setRange(lo_mm, hi_mm)
                else:
                    target_sp.setRange(-100000.0, 100000.0)
        # Refresh map's drawn soft-limit bounding box.
        if getattr(self, "workspace_map", None) is not None:
            self.workspace_map.update_soft_limits(
                tuple(self._soft_min_mm), tuple(self._soft_max_mm),
            )
        self.sb_op_label.setText(msg)

    def _on_soft_limits_error(self, msg: str) -> None:
        self._soft_limit_busy = False
        QMessageBox.critical(self, "Soft limit error", msg)
        self.sb_op_label.setText("Soft limit error")

    # ------------------------------------------------------------------
    # Homing
    # ------------------------------------------------------------------
    def _home_single(self, axis: Axis) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        positive = bool(self.home_dir_combos[axis].currentData())
        dir_word = "POSITIVE" if positive else "NEGATIVE"
        speed = self.home_speed_spin.value()
        reply = QMessageBox.question(
            self, "Confirm Home",
            f"Home {axis.name} toward {dir_word} limit at {speed:.2f} units/s.\n"
            f"Make sure the path is clear. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_homing([axis])

    def _home_all(self) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        order_letters = self.home_order_combo.currentData()
        axes = [Axis[letter] for letter in order_letters]
        order_str = " → ".join(a.name for a in axes)
        per_axis_dir = []
        for a in axes:
            d = "POSITIVE" if bool(self.home_dir_combos[a].currentData()) else "NEGATIVE"
            per_axis_dir.append(f"{a.name}: {d}")
        speed = self.home_speed_spin.value()
        reply = QMessageBox.question(
            self, "Confirm Home All",
            f"Home all axes in order {order_str} at {speed:.2f} units/s?\n"
            f"Per-axis direction → " + ", ".join(per_axis_dir) + "\n"
            "Make sure the entire workspace is clear. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_homing(axes)

    def _persist_home_direction(self, _idx: int = 0) -> None:
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["home_direction"] = {
            a.name: bool(self.home_dir_combos[a].currentData()) for a in AXES
        }
        _gp_save_section("gantry_panel", payload)

    def _set_current_as_home_reference(self) -> None:
        """Capture the current XYZ as the home reference. No motion is
        commanded.

        Source-of-truth precedence:
          1. The latest poll snapshot (``self._last_pos_mm``), if available.
          2. Otherwise, a direct per-axis ``get_axis_position(axis)`` read,
             which works even when ``get_status()`` is returning the 664 error
             (axis not enabled / not homed yet).
          3. If all three per-axis reads also fail, warn and abort.
        """
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        snap = getattr(self, "_last_pos_mm", None)
        if snap is None:
            # Fallback: read each axis directly. Captures whatever the SDK
            # reports for axes that respond, leaves the others untouched.
            captured: list[float | None] = [None, None, None]
            errors: list[str] = []
            for i, axis in enumerate(AXES):
                try:
                    with self._controller_lock:
                        u = float(self.controller.get_axis_position(axis))
                    captured[i] = units_to_mm(u, axis)
                except Exception as exc:
                    errors.append(f"{axis.name}: {exc}")
            if not any(c is not None for c in captured):
                QMessageBox.warning(
                    self, "No position yet",
                    "Could not read the current position from the controller.\n\n"
                    + ("\n".join(errors) if errors else "")
                    + "\n\nTry homing the axes first (status code 664 usually "
                      "means an axis is not enabled / not homed).",
                )
                return
            snap = tuple((c if c is not None else 0.0) for c in captured)
            print(f"[home-ref] captured via per-axis fallback: {snap}", file=sys.stderr)

        for i, axis in enumerate(AXES):
            self._home_position_mm[axis] = float(snap[i])
        # Also seed _last_pos_mm so the Δ-home line on each card updates
        # immediately even before the next poll lands.
        self._last_pos_mm = tuple(float(v) for v in snap)
        self._persist_home_position()
        self._refresh_home_displays()
        self.sb_op_label.setText(
            f"Home ref captured: X={snap[0]:+.2f} Y={snap[1]:+.2f} Z={snap[2]:+.2f} mm"
        )

    def _persist_home_position(self) -> None:
        payload = _gp_load_settings().get("gantry_panel", {})
        payload["home_position_mm"] = {
            a.name: (None if self._home_position_mm[a] is None
                     else float(self._home_position_mm[a]))
            for a in AXES
        }
        _gp_save_section("gantry_panel", payload)

    def _refresh_home_displays(self) -> None:
        # Re-render the Δ-home line on each card using the latest position
        # snapshot, and update the workspace map's home marker.
        snap = getattr(self, "_last_pos_mm", None)
        for i, axis in enumerate(AXES):
            card = self.axis_cards.get(axis)
            if card is None:
                continue
            home_mm = self._home_position_mm[axis]
            cur = snap[i] if snap is not None else None
            card.set_home_reference(home_mm, cur)
        if getattr(self, "workspace_map", None) is not None:
            home_xyz = tuple(self._home_position_mm[a] for a in AXES)
            self.workspace_map.update_home(home_xyz)

    def _start_homing(self, axes: list[Axis]) -> None:
        if self._blocked_by_estop():
            return
        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._homing_in_progress = True
        self._update_all_button_states()
        self.sb_op_label.setText(f"Homing {' → '.join(a.name for a in axes)}…")
        self.sb_op_label.setStyleSheet("color: #ffa726; font-weight: bold;")
        # Per-axis direction map: read each axis's combo.
        positive_limits = {
            int(a): bool(self.home_dir_combos[a].currentData()) for a in AXES
        }
        self._home_thread = HomingThread(
            self.controller, axes,
            speed_units=self.home_speed_spin.value(),
            acc_dec_units=self.home_acc_spin.value(),
            fall_step_units=self.home_fall_spin.value(),
            positive_limits=positive_limits,
            lock=self._controller_lock,
            abort_event=self._abort_event,
        )
        self._home_thread.axis_started.connect(self._on_homing_axis_started)
        self._home_thread.axis_done.connect(self._on_homing_axis_done)
        self._home_thread.all_done.connect(self._on_homing_all_done)
        self._home_thread.finished.connect(self._home_thread.deleteLater)
        self._home_thread.start()

    def _on_homing_axis_started(self, axis: Axis) -> None:
        self.sb_op_label.setText(f"Homing {axis.name}…")

    def _on_homing_axis_done(self, axis: Axis, err: str) -> None:
        if err:
            if err != "aborted":
                QMessageBox.warning(self, "Homing error", f"{axis.name}: {err}")
            return
        # Capture the just-homed axis's current mm as the home reference.
        snap = getattr(self, "_last_pos_mm", None)
        if snap is not None:
            self._home_position_mm[axis] = float(snap[int(axis)])
            self._persist_home_position()
            self._refresh_home_displays()

    def _on_homing_all_done(self, err: str) -> None:
        self._homing_in_progress = False
        # Don't drop ref here — see _on_axis_abs_done docstring (QThread GC race).
        self.sb_op_label.setStyleSheet("color: #ccc;")
        if err:
            self.sb_op_label.setText(f"Homing aborted: {err[:80]}")
        else:
            self.sb_op_label.setText("Homing complete")
        # Refresh state.
        self._load_soft_limits()
        self._update_all_button_states()

    # ------------------------------------------------------------------
    # Move to target
    # ------------------------------------------------------------------
    def _move_to_target(self) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if self._move_in_progress or self._sequence_in_progress or self._homing_in_progress:
            return
        if self._blocked_by_estop():
            return
        target_mm = (
            self.target_spins[Axis.X].value(),
            self.target_spins[Axis.Y].value(),
            self.target_spins[Axis.Z].value(),
        )
        # Soft-limit guard.
        wp = Waypoint(target_mm[0], target_mm[1], target_mm[2],
                      self.move_speed_spin.value(), 0.0)
        try:
            _validate_soft_limits([wp], self._soft_min_mm, self._soft_max_mm)
        except SystemExit as exc:
            QMessageBox.warning(self, "Soft limit violation", str(exc))
            return

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        self._move_in_progress = True
        self._update_all_button_states()
        self.move_btn.setText("Moving…")
        self.cancel_move_btn.setEnabled(True)
        self.sb_op_label.setText(
            f"Moving to ({target_mm[0]:+.2f}, {target_mm[1]:+.2f}, {target_mm[2]:+.2f}) mm"
        )
        if getattr(self, "workspace_map", None) is not None:
            self.workspace_map.update_target(*target_mm)
        self._move_thread = MoveToTargetThread(
            self.controller, target_mm,
            self.move_speed_spin.value(),
            self.move_acc_spin.value(),
            self.move_dec_spin.value(),
            self.move_mode_combo.currentData(),
            self._controller_lock,
            logger=self._logger,
        )
        self._move_thread.finished_with.connect(self._on_move_done)
        self._move_thread.finished.connect(self._move_thread.deleteLater)
        self._move_thread.start()

    def _use_current_as_target(self) -> None:
        # Pull the last polled position from the cards' displays.
        for axis in AXES:
            label_text = self.axis_cards[axis].pos_display.text().strip()
            try:
                self.target_spins[axis].setValue(float(label_text))
            except ValueError:
                pass

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
        if (self._homing_in_progress or self._sequence_in_progress
                or self._move_in_progress or int(axis) in self._per_axis_busy):
            return
        if self._blocked_by_estop():
            return
        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        # mm -> units conversion is exact for single-axis jog/move.
        scale = SCALE_MM_PER_UNIT[axis]
        speed_units = max(self.jog_speed_spin.value() / scale, 0.001)
        acc_units = max(self.jog_acc_spin.value() / scale, 0.001)
        dec_units = max(self.jog_dec_spin.value() / scale, 0.001)
        # 999999 * direction = "jog until release" sentinel, same as manual_pad.
        try:
            with self._controller_lock:
                self.controller.jog_single_axis(
                    axis,
                    position_units=999999.0 * direction,
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

    def _move_axis_abs(self, axis: Axis) -> None:
        if not self.connected:
            QMessageBox.warning(self, "Not connected", "Connect first.")
            return
        if (self._move_in_progress or self._sequence_in_progress
                or self._homing_in_progress or int(axis) in self._per_axis_busy):
            return
        if self._blocked_by_estop():
            return
        target_mm = self.per_axis_cards[axis]["target_spin"].value()
        lo = self._soft_min_mm[int(axis)]
        hi = self._soft_max_mm[int(axis)]
        if lo is not None and target_mm < lo:
            QMessageBox.warning(self, "Soft limit",
                                f"{axis.name}={target_mm:.3f} mm < min {lo:.3f} mm")
            return
        if hi is not None and target_mm > hi:
            QMessageBox.warning(self, "Soft limit",
                                f"{axis.name}={target_mm:.3f} mm > max {hi:.3f} mm")
            return

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        scale = SCALE_MM_PER_UNIT[axis]
        speed_units = max(self.jog_speed_spin.value() / scale, 0.001)
        acc_units = max(self.jog_acc_spin.value() / scale, 0.001)
        dec_units = max(self.jog_dec_spin.value() / scale, 0.001)

        self._per_axis_busy.add(int(axis))
        info = self.per_axis_cards[axis]
        info["move_btn"].setEnabled(False)
        info["move_btn"].setText("Moving…")
        self.sb_op_label.setText(f"Moving {axis.name} → {target_mm:+.3f} mm")
        self._update_all_button_states()

        # Show target on the workspace map (the other two axes keep their
        # current position so the marker lands at the actual destination).
        if getattr(self, "workspace_map", None) is not None and self.workspace_map is not None:
            cur = self.workspace_map._cur_pos or (0.0, 0.0, 0.0)
            full_target = list(cur)
            full_target[int(axis)] = target_mm
            self.workspace_map.update_target(*full_target)

        thread = AxisAbsMoveThread(
            self.controller, axis,
            target_units=mm_to_units(target_mm, axis),
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
            self._recording_manual = True
            self._ensure_logger_started(auto=False)
            self.record_btn.setText("■ Stop Recording")
        else:
            self._recording_manual = False
            self._stop_logger_now()
            self.record_btn.setText("● Start Recording")

    def _ensure_logger_started(self, auto: bool) -> None:
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
        out: list[Waypoint] = []
        for r in range(self.waypoint_table.rowCount()):
            try:
                cells = [self.waypoint_table.item(r, c).text() if self.waypoint_table.item(r, c) else "" for c in range(5)]
                wp = Waypoint(
                    x_mm=float(cells[0]),
                    y_mm=float(cells[1]),
                    z_mm=float(cells[2]),
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
        if self._move_in_progress or self._sequence_in_progress or self._homing_in_progress:
            return
        if self._blocked_by_estop():
            return
        try:
            waypoints = self._collect_waypoints()
        except SystemExit as exc:
            QMessageBox.warning(self, "Bad waypoints", str(exc))
            return
        if not waypoints:
            QMessageBox.information(self, "No waypoints", "Add at least one row first.")
            return
        try:
            _validate_soft_limits(waypoints, self._soft_min_mm, self._soft_max_mm)
        except SystemExit as exc:
            QMessageBox.warning(self, "Soft limit violation", str(exc))
            return

        EMERGENCY_STOP.clear()
        self._abort_event.clear()
        self._ensure_logger_started(auto=True)
        self._sequence_in_progress = True
        self._update_all_button_states()
        self.run_seq_btn.setEnabled(False)
        self.stop_seq_btn.setEnabled(True)
        self._sequence_thread = SequenceThread(
            self.controller, waypoints,
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

        if not self.connected:
            self._show_estop_banner(t0, [], lock_acquired=False, note="not connected")
            return

        acquired = self._controller_lock.acquire(timeout=0.05)
        per_axis_results: list[tuple[str, str]] = []  # (axis_name, "ok" or err)
        try:
            for axis in AXES:
                try:
                    self.controller.stop_axis(axis, mode=2)
                    per_axis_results.append((axis.name, "ok"))
                except FMC4030Error as exc:
                    per_axis_results.append((axis.name, f"{exc}"))
                except Exception as exc:
                    per_axis_results.append((axis.name, f"unexpected: {exc}"))
            # Coordinated motion stop in addition to per-axis stops.
            try:
                self.controller.stop_run()
            except FMC4030Error:
                pass
            except Exception:
                pass
        finally:
            if acquired:
                self._controller_lock.release()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        print(f"[estop-debug] completed in {elapsed_ms:.2f} ms "
              f"(lock_acquired={acquired})", file=sys.stderr, flush=True)

        self._stop_logger_now()
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
            or self._homing_in_progress
            or self._sequence_in_progress
            or bool(self._per_axis_busy)
        )
        # Motion / homing / soft-limit / record buttons enabled only if connected
        # and (for motion) not busy.
        for axis in AXES:
            self.home_btns[axis].setEnabled(connected and not busy)
            self.soft_limit_spins[axis]["apply_btn"].setEnabled(connected and not self._soft_limit_busy)
        self.home_all_btn.setEnabled(connected and not busy)
        self.load_limits_btn.setEnabled(connected and not self._soft_limit_busy)
        self.apply_all_limits_btn.setEnabled(connected and not self._soft_limit_busy)
        self.move_btn.setEnabled(connected and not busy)
        self.use_current_btn.setEnabled(connected)
        self.run_seq_btn.setEnabled(connected and not busy)
        self.record_btn.setEnabled(connected)
        self.refresh_btn.setEnabled(connected)
        self.pause_btn.setEnabled(connected)
        self.resume_btn.setEnabled(connected)
        self.stop_run_btn.setEnabled(connected)
        # Per-Axis Control cards: jog buttons + Move Abs + Home — disabled when
        # disconnected, when homing/sequence is running, or when ANY Move Abs
        # is in flight on any axis. Self-axis is also disabled when its own
        # Move Abs is in flight (covered by "busy" via _per_axis_busy).
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

        for thread_attr in ("_status_thread", "_move_thread",
                            "_home_thread", "_sequence_thread"):
            _safe_wait(getattr(self, thread_attr, None))
        for t in list(self._per_axis_threads.values()):
            _safe_wait(t)
        if self.connected:
            try:
                with self._controller_lock:
                    self.controller.close()
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
    window.show()

    # Keep a reference to the heartbeat so it doesn't get GC'd.
    _heartbeat = _install_sigint(app, window)
    window._heartbeat = _heartbeat  # type: ignore[attr-defined]

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
