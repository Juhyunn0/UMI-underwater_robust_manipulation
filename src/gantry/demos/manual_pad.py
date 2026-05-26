#!/usr/bin/env python3
"""PyQt manual jog pad inspired by the original Qt demo."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
from typing import Callable

from PyQt5 import QtCore, QtWidgets

# Allow running the script directly from the repo root
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisker_flow.gantry import (
    Axis,
    ControllerConfig,
    DeviceParameters,
    FMC4030Controller,
    FMC4030Error,
)


@dataclass
class AxisStatusSnapshot:
    positions: dict[Axis, float]
    inputs: dict[int, int | None]
    timestamp: float


class StatusUpdateThread(QtCore.QThread):
    """Background worker that queries positions/speeds without blocking UI."""

    snapshot_ready = QtCore.pyqtSignal(object)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        controller: FMC4030Controller,
        axes: list[Axis],
        input_channels: list[int],
        lock: threading.RLock,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._axes = list(axes)
        self._input_channels = list(input_channels)
        self._lock = lock

    def run(self) -> None:  # type: ignore[override]
        if not self._axes:
            self.snapshot_ready.emit(AxisStatusSnapshot({}, {}, time.monotonic()))
            return

        try:
            with self._lock:
                positions: dict[Axis, float] = {}
                inputs: dict[int, int | None] = {}

                for axis in self._axes:
                    pos = self._controller.get_axis_position(axis)
                    positions[axis] = pos
                    print(f"[DEBUG] StatusUpdateThread: {axis.name} pos={pos:.3f}")

                for channel in self._input_channels:
                    try:
                        inputs[channel] = self._controller.get_input(channel)
                        print(
                            f"[DEBUG] StatusUpdateThread: IN{channel}="
                            f"{'LOW' if inputs[channel] else 'HIGH'}"
                        )
                    except FMC4030Error as exc:
                        inputs[channel] = None
                        print(f"[DEBUG] StatusUpdateThread: Input {channel} error - {exc}")

            self.snapshot_ready.emit(AxisStatusSnapshot(positions, inputs, time.monotonic()))
        except FMC4030Error as exc:
            print(f"[DEBUG] StatusUpdateThread: FMC error - {exc}")
            self.error_occurred.emit(str(exc))
        except Exception as exc:
            print(f"[DEBUG] StatusUpdateThread: Unexpected error - {exc}")
            self.error_occurred.emit(f"Unexpected error: {exc}")


class GantryManualPad(QtWidgets.QWidget):
    soft_limits_result = QtCore.pyqtSignal(object, str)
    soft_limits_error = QtCore.pyqtSignal(str)

    POLL_INTERVAL_MS = 100  # Poll controller ~10x/sec for snappier position/velocity updates
    OUTPUT_CHANNELS = (0, 1, 2, 3)
    # Only IN0 is wired on the current gantry; higher channels return -7
    INPUT_CHANNELS = (0,)
    POLL_INPUTS = False
    HOME_SPEED_LIMIT = 20.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FMC4030 Manual Jog Pad")
        self.controller = FMC4030Controller()
        self._controller_lock = threading.RLock()
        self.connected = False
        self.axis_widgets: dict[Axis, dict[str, QtWidgets.QWidget]] = {}
        self.axis_enable_checks: dict[Axis, QtWidgets.QCheckBox] = {}
        self.soft_limit_spins: dict[Axis, dict[str, QtWidgets.QDoubleSpinBox]] = {}
        self.output_checks: dict[int, QtWidgets.QCheckBox] = {}
        self.input_labels: dict[int, QtWidgets.QLabel] = {}
        self.enabled_axes_mask = 0x01  # Default: only X axis enabled
        self._soft_limit_busy = False
        self._status_thread: StatusUpdateThread | None = None
        self._last_axis_samples: dict[Axis, tuple[float, float]] = {}
        self.soft_limits_result.connect(self._handle_soft_limit_result)
        self.soft_limits_error.connect(self._handle_soft_limit_error)
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._request_status_update)
        # Start timer - it will trigger background status updates
        self._timer.start()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        conn_layout = QtWidgets.QHBoxLayout()
        self.ip_edit = QtWidgets.QLineEdit("192.168.0.30")
        self.port_edit = QtWidgets.QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(8088)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)

        for widget, label in [
            (QtWidgets.QLabel("IP:"), None),
            (self.ip_edit, None),
            (QtWidgets.QLabel("Port:"), None),
            (self.port_edit, None),
        ]:
            conn_layout.addWidget(widget)
        conn_layout.addWidget(self.connect_btn)
        
        # Add axis enable checkboxes
        conn_layout.addWidget(QtWidgets.QLabel("  |  Enabled axes:"))
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            check = QtWidgets.QCheckBox(axis.name)
            check.setChecked(axis == Axis.X)  # Default: only X enabled
            check.stateChanged.connect(self._update_enabled_axes)
            self.axis_enable_checks[axis] = check
            conn_layout.addWidget(check)
        
        main_layout.addLayout(conn_layout)

        motion_layout = QtWidgets.QHBoxLayout()
        speed_widget, self.speed_spin = self._spin_box(
            1.0, 500.0, 20.0, "Speed (units/s)"
        )
        acc_widget, self.acc_spin = self._spin_box(
            1.0, 2000.0, 20.0, "Accel (units/s²)"
        )
        dec_widget, self.dec_spin = self._spin_box(
            1.0, 2000.0, 20.0, "Decel (units/s²)"
        )
        fall_widget, self.home_fall_spin = self._spin_box(
            0.1, 100.0, 5.0, "Home fall step (units)"
        )
        self.home_dir_combo = QtWidgets.QComboBox()
        self.home_dir_combo.addItem("Negative limit", False)
        self.home_dir_combo.addItem("Positive limit", True)
        dir_widget = QtWidgets.QWidget()
        dir_layout = QtWidgets.QVBoxLayout(dir_widget)
        dir_layout.addWidget(QtWidgets.QLabel("Home direction"))
        dir_layout.addWidget(self.home_dir_combo)

        for widget in (speed_widget, acc_widget, dec_widget, fall_widget, dir_widget):
            motion_layout.addWidget(widget)
        main_layout.addLayout(motion_layout)

        axes_container = QtWidgets.QHBoxLayout()
        for axis in [Axis.X, Axis.Y, Axis.Z]:
            axes_container.addWidget(self._build_axis_group(axis))
        main_layout.addLayout(axes_container)
        main_layout.addWidget(self._build_soft_limit_group())

        io_group = QtWidgets.QGroupBox("Digital IO")
        io_layout = QtWidgets.QGridLayout(io_group)
        for col, channel in enumerate(self.OUTPUT_CHANNELS):
            checkbox = QtWidgets.QCheckBox(f"OUT{channel} (low=active)")
            checkbox.stateChanged.connect(partial(self._handle_output_toggle, channel))
            self.output_checks[channel] = checkbox
            io_layout.addWidget(checkbox, 0, col)
        if self.POLL_INPUTS:
            for col, channel in enumerate(self.INPUT_CHANNELS):
                label = QtWidgets.QLabel(f"IN{channel}: --")
                self.input_labels[channel] = label
                io_layout.addWidget(label, 1, col)
        else:
            label = QtWidgets.QLabel("Input polling disabled")
            label.setStyleSheet("color: #888;")
            io_layout.addWidget(label, 1, 0, 1, len(self.OUTPUT_CHANNELS))
        main_layout.addWidget(io_group)

        global_controls = QtWidgets.QHBoxLayout()
        
        # Manual refresh button since auto-polling is disabled
        refresh_btn = QtWidgets.QPushButton("🔄 Refresh Position")
        refresh_btn.setToolTip("Manually update position and velocity displays")
        refresh_btn.clicked.connect(self._refresh_status)
        
        # Emergency stop button - stops ALL axes immediately
        estop_btn = QtWidgets.QPushButton("⚠️ EMERGENCY STOP ALL")
        estop_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold; padding: 10px;")
        estop_btn.clicked.connect(self._emergency_stop_all)
        
        # Other control buttons (for coordinated motion/scripts)
        pause_btn = QtWidgets.QPushButton("Pause Run")
        resume_btn = QtWidgets.QPushButton("Resume Run")
        stop_btn = QtWidgets.QPushButton("Stop Run")
        pause_btn.setToolTip("Pause coordinated motion (line/arc moves)")
        resume_btn.setToolTip("Resume paused coordinated motion")
        stop_btn.setToolTip("Stop coordinated motion")
        pause_btn.clicked.connect(lambda: self._run_global_command("pause"))
        resume_btn.clicked.connect(lambda: self._run_global_command("resume"))
        stop_btn.clicked.connect(lambda: self._run_global_command("stop"))
        
        global_controls.addWidget(refresh_btn)
        global_controls.addWidget(estop_btn)
        global_controls.addWidget(pause_btn)
        global_controls.addWidget(resume_btn)
        global_controls.addWidget(stop_btn)
        main_layout.addLayout(global_controls)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setStyleSheet("color: red;")
        main_layout.addWidget(self.status_label)
        
        # Add helpful note
        note_label = QtWidgets.QLabel(
            "💡 Tips: (1) Position/velocity auto-update in background. Click '🔄 Refresh' for immediate update. "
            "(2) HOLD jog buttons (X+/-) for continuous movement, release to stop. "
            "(3) Use 'Move Abs' for precise positioning. "
            "(4) Lower Speed/Accel for finer control."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        main_layout.addWidget(note_label)

    def _build_axis_group(self, axis: Axis) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(f"{axis.name} Axis")
        layout = QtWidgets.QVBoxLayout(group)

        # Position display with label
        pos_layout = QtWidgets.QVBoxLayout()
        pos_layout.addWidget(QtWidgets.QLabel("Position (units)"))
        pos_label = QtWidgets.QLabel("0.00")
        pos_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        pos_label.setMinimumWidth(120)
        pos_label.setStyleSheet(
            "font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            "font-size: 22px;"
            "padding: 6px;"
            "border: 1px solid #666;"
            "border-radius: 4px;"
            "background-color: #111;"
            "color: #0f0;"
        )
        pos_layout.addWidget(pos_label)
        layout.addLayout(pos_layout)
        
        velocity_label = QtWidgets.QLabel("Vel: 0.00 units/s")

        jog_layout = QtWidgets.QHBoxLayout()
        btn_forward = QtWidgets.QPushButton(f"{axis.name}+")
        btn_reverse = QtWidgets.QPushButton(f"{axis.name}-")
        btn_forward.setToolTip("Hold to jog continuously in positive direction")
        btn_reverse.setToolTip("Hold to jog continuously in negative direction")
        btn_forward.pressed.connect(partial(self._start_jog, axis, +1))
        btn_forward.released.connect(partial(self._stop_jog, axis))
        btn_reverse.pressed.connect(partial(self._start_jog, axis, -1))
        btn_reverse.released.connect(partial(self._stop_jog, axis))
        jog_layout.addWidget(btn_forward)
        jog_layout.addWidget(btn_reverse)

        abs_layout = QtWidgets.QHBoxLayout()
        target_spin = QtWidgets.QDoubleSpinBox()
        target_spin.setRange(-1_000_000.0, 1_000_000.0)
        target_spin.setDecimals(2)
        abs_btn = QtWidgets.QPushButton("Move Abs")
        abs_btn.clicked.connect(partial(self._move_axis_absolute, axis))
        abs_layout.addWidget(target_spin)
        abs_layout.addWidget(abs_btn)

        home_btn = QtWidgets.QPushButton("Home Axis")
        home_btn.clicked.connect(partial(self._home_axis, axis))

        layout.addWidget(velocity_label)
        layout.addLayout(jog_layout)
        layout.addLayout(abs_layout)
        layout.addWidget(home_btn)

        self.axis_widgets[axis] = {
            "position": pos_label,
            "velocity": velocity_label,
            "target": target_spin,
        }
        return group

    def _build_soft_limit_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Software Limits")
        layout = QtWidgets.QGridLayout(group)
        header_axis = QtWidgets.QLabel("Axis")
        header_min = QtWidgets.QLabel("Min (units)")
        header_max = QtWidgets.QLabel("Max (units)")
        layout.addWidget(header_axis, 0, 0)
        layout.addWidget(header_min, 0, 1)
        layout.addWidget(header_max, 0, 2)

        for row, axis in enumerate(Axis, start=1):
            label = QtWidgets.QLabel(axis.name)
            min_spin = QtWidgets.QDoubleSpinBox()
            min_spin.setRange(-1_000_000.0, 1_000_000.0)
            min_spin.setDecimals(2)
            min_spin.setValue(-1_000.0)
            max_spin = QtWidgets.QDoubleSpinBox()
            max_spin.setRange(-1_000_000.0, 1_000_000.0)
            max_spin.setDecimals(2)
            max_spin.setValue(1_000.0)
            apply_btn = QtWidgets.QPushButton(f"Apply {axis.name}")
            apply_btn.setToolTip("Write these limits to the controller for this axis only")
            apply_btn.clicked.connect(partial(self._apply_soft_limits_axis, axis))
            layout.addWidget(label, row, 0)
            layout.addWidget(min_spin, row, 1)
            layout.addWidget(max_spin, row, 2)
            layout.addWidget(apply_btn, row, 3)
            self.soft_limit_spins[axis] = {"min": min_spin, "max": max_spin}

        button_layout = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load from Controller")
        load_btn.clicked.connect(self._load_soft_limits)
        apply_all_btn = QtWidgets.QPushButton("Apply All")
        apply_all_btn.clicked.connect(self._apply_all_soft_limits)
        button_layout.addWidget(load_btn)
        button_layout.addWidget(apply_all_btn)
        layout.addLayout(button_layout, len(Axis) + 1, 0, 1, 4)
        return group

    def _load_soft_limits(self) -> None:
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect first.")
            return

        def worker() -> DeviceParameters:
            with self._controller_lock:
                return self.controller.get_device_parameters()

        self._run_soft_limit_task(worker, "Soft limits loaded", "Loading soft limits...")

    def _apply_soft_limits_axis(self, axis: Axis) -> None:
        self._apply_soft_limits([axis])

    def _apply_all_soft_limits(self) -> None:
        self._apply_soft_limits(list(Axis))

    def _apply_soft_limits(self, axes: list[Axis]) -> None:
        if not axes:
            return
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect first.")
            return

        updates: dict[int, tuple[int, int]] = {}
        for axis in axes:
            spins = self.soft_limit_spins.get(axis)
            if spins is None:
                continue
            min_val = spins["min"].value()
            max_val = spins["max"].value()
            # if min_val >= max_val:
            #     QtWidgets.QMessageBox.warning(
            #         self,
            #         "Invalid Limits",
            #         f"{axis.name}: Min must be less than Max.",
            #     )
            #     return
            updates[int(axis)] = (int(round(min_val)), int(round(max_val)))

        if not updates:
            return

        def worker() -> DeviceParameters:
            with self._controller_lock:
                params: DeviceParameters = self.controller.get_device_parameters()
                for idx, (limit_min, limit_max) in updates.items():
                    params.soft_limit_min[idx] = limit_min
                    params.soft_limit_max[idx] = limit_max
                self.controller.set_device_parameters(params)
                return params

        self._run_soft_limit_task(worker, "Soft limits updated", "Updating soft limits...")

    def _run_soft_limit_task(
        self,
        worker: Callable[[], DeviceParameters],
        success_message: str,
        busy_message: str,
    ) -> None:
        if self._soft_limit_busy:
            QtWidgets.QMessageBox.information(
                self,
                "Soft Limit Busy",
                "Please wait for the current soft limit operation to finish.",
            )
            return
        self._soft_limit_busy = True
        self.status_label.setText(busy_message)
        self.status_label.setStyleSheet("color: orange;")

        def task() -> None:
            try:
                params = worker()
            except FMC4030Error as exc:
                self.soft_limits_error.emit(str(exc))
            except Exception as exc:
                self.soft_limits_error.emit(f"Unexpected error: {exc}")
            else:
                self.soft_limits_result.emit(params, success_message)

        threading.Thread(target=task, daemon=True).start()

    @QtCore.pyqtSlot(object, str)
    def _handle_soft_limit_result(self, params: DeviceParameters, message: str) -> None:
        self._soft_limit_busy = False
        summaries: list[str] = []
        for axis in Axis:
            spins = self.soft_limit_spins.get(axis)
            if not spins:
                continue
            idx = int(axis)
            spins["min"].setValue(float(params.soft_limit_min[idx]))
            spins["max"].setValue(float(params.soft_limit_max[idx]))
            summaries.append(
                f"{axis.name}: min={params.soft_limit_min[idx]}, max={params.soft_limit_max[idx]}"
            )
        if summaries:
            debug_msg = "; ".join(summaries)
            print(f"[DEBUG] Soft limit snapshot -> {debug_msg}")
            self.status_label.setToolTip(debug_msg)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: green;")

    @QtCore.pyqtSlot(str)
    def _handle_soft_limit_error(self, error_msg: str) -> None:
        self._soft_limit_busy = False
        QtWidgets.QMessageBox.critical(self, "Soft Limit Error", error_msg)
        self.status_label.setText("Soft limit error")
        self.status_label.setStyleSheet("color: orange;")

    def _spin_box(
        self, minimum: float, maximum: float, value: float, label_text: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QDoubleSpinBox]:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setValue(value)
        layout = QtWidgets.QVBoxLayout()
        widget = QtWidgets.QWidget()
        layout.addWidget(QtWidgets.QLabel(label_text))
        layout.addWidget(spin)
        widget.setLayout(layout)
        return widget, spin

    def _update_enabled_axes(self) -> None:
        """Update the enabled axes mask based on checkboxes."""
        self.enabled_axes_mask = 0
        for axis, check in self.axis_enable_checks.items():
            if check.isChecked():
                self.enabled_axes_mask |= (1 << int(axis))

    def _enabled_axes(self) -> list[Axis]:
        """Return a list of currently enabled axes."""
        return [axis for axis, check in self.axis_enable_checks.items() if check.isChecked()]

    def _test_connection_with_timeout(self, timeout_sec: float = 2.0) -> tuple[bool, str]:
        """Test if we can query the controller with a timeout.
        
        Returns:
            (success, message) - success is True if test passed, message describes result
        """
        result = {"success": False, "error": None, "value": None}
        
        def test_query():
            try:
                # Try to get position of X axis
                with self._controller_lock:
                    pos = self.controller.get_axis_position(Axis.X)
                result["success"] = True
                result["value"] = pos
            except Exception as e:
                result["error"] = str(e)
        
        # Run test in a thread with timeout
        thread = threading.Thread(target=test_query, daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            return False, f"Timeout after {timeout_sec}s - controller not responding"
        
        if result["success"]:
            return True, f"OK (X pos: {result['value']:.2f})"
        else:
            return False, f"Query failed: {result['error']}"

    # ------------------------------------------------------------------ #
    def toggle_connection(self) -> None:
        if self.connected:
            print("[DEBUG] toggle_connection: Disconnecting...")
            if self._status_thread is not None and self._status_thread.isRunning():
                self._status_thread.requestInterruption()
                self._status_thread.wait(500)
            self._status_thread = None
            try:
                with self._controller_lock:
                    self.controller.close()
            except Exception as e:
                print(f"[DEBUG] Error during close: {e}")
                
            self.connected = False
            self.connect_btn.setText("Connect")
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: red;")
            return
        
        print("[DEBUG] toggle_connection: Connecting...")
        config = ControllerConfig(
            controller_id=0,
            ip=self.ip_edit.text(),
            port=self.port_edit.value(),
        )
        print(f"[DEBUG] Config: IP={config.ip}, Port={config.port}, ID={config.controller_id}")
        
        try:
            with self._controller_lock:
                self.controller.connect(config)
            print("[DEBUG] Connection successful!")
        except FMC4030Error as exc:
            print(f"[DEBUG] FMC4030Error during connect: {exc}")
            QtWidgets.QMessageBox.critical(self, "Connect Failed", str(exc))
            return
        except Exception as exc:
            print(f"[DEBUG] Exception during connect: {exc}")
            QtWidgets.QMessageBox.critical(self, "Connect Failed", f"Unexpected error: {exc}")
            return
        
        # Connection successful!
        self.connected = True
        self.connect_btn.setText("Disconnect")
        
        disabled_count = bin(0x07 ^ self.enabled_axes_mask).count('1')
        print(f"[DEBUG] Enabled axes mask: 0x{self.enabled_axes_mask:02x}, disabled count: {disabled_count}")
        if disabled_count > 0:
            self.status_label.setText(f"Connected ({disabled_count} axis/axes disabled)")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green;")

    def _start_jog(self, axis: Axis, direction: int) -> None:
        if not self.connected:
            return
        # Use large distance for continuous jogging (999999 like Qt demo)
        distance = 999999 * direction
        speed = self.speed_spin.value()
        acc = self.acc_spin.value()
        dec = self.dec_spin.value()
        try:
            with self._controller_lock:
                self.controller.jog_single_axis(
                    axis,
                    position_units=distance,
                    speed_units=speed,
                    acc_units=acc,
                    dec_units=dec,
                    relative=True,
                )
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Jog Error", str(exc))

    def _stop_jog(self, axis: Axis) -> None:
        if not self.connected:
            return
        try:
            with self._controller_lock:
                # Mode 1 = soft stop (decelerate), mode 2 = emergency stop
                self.controller.stop_axis(axis, mode=1)
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Stop Error", str(exc))

    def _move_axis_absolute(self, axis: Axis) -> None:
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        target = self.axis_widgets[axis]["target"].value()
        speed = self.speed_spin.value()
        acc = self.acc_spin.value()
        dec = self.dec_spin.value()
        try:
            with self._controller_lock:
                self.controller.jog_single_axis(
                    axis,
                    position_units=target,
                    speed_units=speed,
                    acc_units=acc,
                    dec_units=dec,
                    relative=False,
                )
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Absolute Move Error", str(exc))

    def _home_axis(self, axis: Axis) -> None:
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        positive = bool(self.home_dir_combo.currentData())
        speed = min(self.speed_spin.value(), self.HOME_SPEED_LIMIT)
        acc_dec = self.acc_spin.value()
        fall_step = self.home_fall_spin.value()
        try:
            with self._controller_lock:
                self.controller.home_axis(
                    axis,
                    speed=speed,
                    acc_dec=acc_dec,
                    fall_step=fall_step,
                    positive_limit=positive,
                )
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Home Error", str(exc))

    def _handle_output_toggle(self, channel: int, state: int) -> None:
        if not self.connected:
            return
        level = 1 if state == QtCore.Qt.Checked else 0
        try:
            with self._controller_lock:
                self.controller.set_output(channel, level)
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Output Error", str(exc))

    def _run_global_command(self, command: str) -> None:
        if not self.connected:
            return
        try:
            with self._controller_lock:
                if command == "pause":
                    self.controller.pause_run(self.enabled_axes_mask)
                elif command == "resume":
                    self.controller.resume_run(self.enabled_axes_mask)
                elif command == "stop":
                    self.controller.stop_run()
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Controller Error", str(exc))

    def _emergency_stop_all(self) -> None:
        """Emergency stop all enabled axes."""
        if not self.connected:
            return
        try:
            with self._controller_lock:
                for axis in Axis:
                    axis_bit = 1 << int(axis)
                    if self.enabled_axes_mask & axis_bit:
                        self.controller.stop_axis(axis, mode=2)
            self.status_label.setText("Emergency stop executed")
            self.status_label.setStyleSheet("color: red;")
        except FMC4030Error as exc:
            QtWidgets.QMessageBox.critical(self, "Emergency Stop Error", str(exc))

    def _request_status_update(self) -> None:
        """Kick off a background status poll if one isn't already running."""
        if not self.connected:
            for widgets in self.axis_widgets.values():
                widgets["position"].setText("0.00")
                widgets["velocity"].setText("Vel: 0.00 units/s")
            self._last_axis_samples.clear()
            for channel, label in self.input_labels.items():
                label.setText(f"IN{channel}: --")
            return

        if self._status_thread is not None and self._status_thread.isRunning():
            print("[DEBUG] _request_status_update: Thread already running, skipping")
            return

        enabled_axes = self._enabled_axes()
        if not enabled_axes:
            print("[DEBUG] _request_status_update: No axes enabled, skipping status thread")
            for axis in Axis:
                widgets = self.axis_widgets[axis]
                widgets["position"].setText("0.00")
                widgets["velocity"].setText("Vel: -- (disabled)")
            self._last_axis_samples.clear()
            self.status_label.setText("Connected (no axes enabled)")
            self.status_label.setStyleSheet("color: orange;")
            return

        input_channels = list(self.INPUT_CHANNELS) if self.POLL_INPUTS else []
        self._status_thread = StatusUpdateThread(
            self.controller,
            enabled_axes,
            input_channels,
            self._controller_lock,
            self,
        )
        self._status_thread.snapshot_ready.connect(self._handle_status_update)
        self._status_thread.error_occurred.connect(self._handle_status_error)
        self._status_thread.finished.connect(self._on_status_thread_finished)
        print("[DEBUG] _request_status_update: Starting new status thread")
        self._status_thread.start()

    def _on_status_thread_finished(self) -> None:
        if self._status_thread is not None:
            self._status_thread.deleteLater()
            self._status_thread = None

    def _handle_status_update(self, snapshot: AxisStatusSnapshot) -> None:
        """Update widgets after a successful status poll."""
        print("[DEBUG] _handle_status_update: Received status snapshot")
        # Update each axis from the status
        for axis in Axis:
            widgets = self.axis_widgets[axis]
            axis_bit = 1 << int(axis)
            
            if not (self.enabled_axes_mask & axis_bit):
                # Axis not enabled, show as inactive
                widgets["position"].setText("0.00")
                widgets["velocity"].setText("Vel: -- (disabled)")
                self._last_axis_samples.pop(axis, None)
            else:
                pos = snapshot.positions.get(axis)
                if pos is None:
                    print(f"[DEBUG] No snapshot data for {axis.name} (likely stale request)")
                    widgets["position"].setText("--")
                    widgets["velocity"].setText("Vel: -- (no data)")
                    self._last_axis_samples.pop(axis, None)
                else:
                    prev = self._last_axis_samples.get(axis)
                    velocity = 0.0
                    if prev is not None:
                        prev_pos, prev_ts = prev
                        dt = snapshot.timestamp - prev_ts
                        if dt > 1e-3:
                            velocity = (pos - prev_pos) / dt
                    self._last_axis_samples[axis] = (pos, snapshot.timestamp)
                    print(f"[DEBUG] Updating {axis.name}: pos={pos:.2f}, vel={velocity:.2f}")
                    widgets["position"].setText(f"{pos:.3f}")
                    widgets["velocity"].setText(f"Vel: {velocity:.2f} units/s")
        
        # Update connection status
        disabled_count = bin(0x07 ^ self.enabled_axes_mask).count('1')
        if disabled_count > 0:
            self.status_label.setText(f"Connected ({disabled_count} axis/axes disabled)")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green;")
        
        # Update inputs using snapshot (already fetched in worker thread)
        self._update_input_labels(snapshot.inputs)
    
    def _handle_status_error(self, error_msg: str) -> None:
        """Handle errors raised while polling status."""
        print(f"[DEBUG] _handle_status_error: {error_msg}")
        # Show error in status but don't freeze
        if "664" in error_msg:
            disabled_count = bin(0x07 ^ self.enabled_axes_mask).count('1')
            self.status_label.setText(f"Connected ({disabled_count} axes disabled)")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText(f"Error: {error_msg[:40]}")
            self.status_label.setStyleSheet("color: orange;")

    def _refresh_status(self) -> None:
        """Manual refresh - kept for compatibility with refresh button."""
        self._request_status_update()

    def _update_input_labels(self, values: dict[int, int | None]) -> None:
        if not self.POLL_INPUTS:
            return
        for channel, label in self.input_labels.items():
            state = values.get(channel)
            if state is None:
                label.setText(f"IN{channel}: error/disabled")
            else:
                label.setText(f"IN{channel}: {'LOW' if state else 'HIGH'}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        if self._status_thread is not None and self._status_thread.isRunning():
            self._status_thread.requestInterruption()
            self._status_thread.wait(500)
            self._status_thread = None
        if self.connected:
            with self._controller_lock:
                self.controller.close()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = GantryManualPad()
    window.resize(600, 250)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
