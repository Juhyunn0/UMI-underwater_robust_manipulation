#!/usr/bin/env python3
"""
gantry_runner.py — Drive the FMC4030 gantry to XYZ targets in mm and log
real-time telemetry to CSV.

================================================================================
Frame & sign conventions (PLEASE CONFIRM AT INTEGRATION TIME)
================================================================================
* Gantry XYZ positions are reported in millimeters in the controller's native
  frame, converted from raw "units" via SCALE_MM_PER_UNIT (copied verbatim from
  src/gantry/demos/whisker_dragging.py — X 8.25, Y 2.50, Z 0.50 mm/unit).
* This module makes NO assumption about Z-axis sign vs "world up". If the
  downstream fisheye pipeline needs +Z-up vs +Z-down handling, the 4x4
  T_gantry_camera in the fisheye calibration YAML absorbs the convention.

================================================================================
line_move_3d() speed conversion choice (dominant-displacement axis)
================================================================================
The FMC4030 SDK's Line_3Axis call takes a SINGLE scalar `speed` in controller
units/s for a coordinated path move, but each axis has a very different
mm-per-unit scale (X 8.25, Y 2.5, Z 0.5 — a 16.5x ratio). One units/s value
cannot simultaneously equal the requested mm/s on all three axes.

We convert the user-facing mm/s using the *dominant-displacement axis*: the
axis whose absolute mm displacement on this leg is largest. Consequences:
  * The realized path speed in mm/s equals the requested speed EXACTLY when
    the leg lies along the dominant axis (the common single-axis case).
  * For mixed-axis legs, the realized speed deviates from the request by the
    SCALE_MM_PER_UNIT ratio between the dominant axis and the off-dominant
    legs. Worst case (e.g. dominant X, off Z): off-axis is up to 16.5x
    faster/slower than its share of the requested path speed. Acc/dec scale
    the same way.
For exact per-axis speed control, use --mode sequential (jog_single_axis),
which lets us convert mm/s to units/s per axis independently.

================================================================================
Acceleration column (derived, not from SDK)
================================================================================
The FMC4030 has no acceleration readout. We derive it as a finite difference
on a sliding window of recent (time, velocity) samples (default window=5):
split the window into front and back halves, average each half (SMA low-pass),
and central-difference between the two centroids. So `a*_mm_s2` is a smoothed
~50 ms-ish estimate, not a direct sensor value.
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
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

# sys.path shim: import `gantry` from src/ regardless of where this script
# is invoked from. (Matches `from gantry import ...` per repo layout.)
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gantry import (  # noqa: E402
    Axis,
    ControllerConfig,
    FMC4030Controller,
    FMC4030Error,
)


SCALE_MM_PER_UNIT: dict[Axis, float] = {Axis.X: 8.25, Axis.Y: 2.5, Axis.Z: 0.5}
AXIS_NAMES: dict[Axis, str] = {Axis.X: "X", Axis.Y: "Y", Axis.Z: "Z"}
AXES: tuple[Axis, Axis, Axis] = (Axis.X, Axis.Y, Axis.Z)

EMERGENCY_STOP = threading.Event()


def units_to_mm(units: float, axis: Axis) -> float:
    return units * SCALE_MM_PER_UNIT[axis]


def mm_to_units(mm: float, axis: Axis) -> float:
    return mm / SCALE_MM_PER_UNIT[axis]


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Waypoint:
    x_mm: float
    y_mm: float
    z_mm: float
    speed_mm_s: float
    dwell_s: float = 0.0


@dataclass
class GantrySample:
    """Snapshot of the gantry state captured by the telemetry logger."""
    timestamp_unix: float
    timestamp_monotonic: float
    elapsed_s: float
    pos_units: tuple[float, float, float]
    pos_mm: tuple[float, float, float]
    vel_units_s: tuple[float, float, float]
    vel_mm_s: tuple[float, float, float]
    acc_mm_s2: tuple[float, float, float]
    target_mm: tuple[float, float, float]
    waypoint_index: int
    is_moving: bool


CSV_COLUMNS: tuple[str, ...] = (
    "timestamp_unix", "timestamp_monotonic", "elapsed_s",
    "x_units", "y_units", "z_units",
    "x_mm", "y_mm", "z_mm",
    "vx_units_s", "vy_units_s", "vz_units_s",
    "vx_mm_s", "vy_mm_s", "vz_mm_s",
    "ax_mm_s2", "ay_mm_s2", "az_mm_s2",
    "target_x_mm", "target_y_mm", "target_z_mm",
    "waypoint_index", "is_moving",
)


# -----------------------------------------------------------------------------
# Telemetry logger
# -----------------------------------------------------------------------------
class GantryTelemetryLogger:
    """Daemon thread that samples controller.get_status() at log_hz and writes
    one CSV row per tick. Position + velocity come from a single SDK call per
    sample (all 3 axes); acceleration is derived (see module docstring)."""

    def __init__(
        self,
        controller: FMC4030Controller,
        csv_path: Path | str,
        *,
        log_hz: float = 100.0,
        lock: threading.RLock | None = None,
        t0_monotonic: float | None = None,
        finite_diff_window: int = 5,
    ) -> None:
        if log_hz <= 0:
            raise ValueError("log_hz must be > 0")
        if finite_diff_window < 3 or finite_diff_window % 2 == 0:
            raise ValueError("finite_diff_window must be odd and >= 3")
        self._controller = controller
        self._csv_path = Path(csv_path)
        self._period = 1.0 / float(log_hz)
        self._lock = lock if lock is not None else threading.RLock()
        self._t0_mono = t0_monotonic if t0_monotonic is not None else time.monotonic()
        self._fd_n = int(finite_diff_window)

        self._target_mm: tuple[float, float, float] = (math.nan, math.nan, math.nan)
        self._waypoint_index: int = -1
        self._target_lock = threading.Lock()

        self._latest: GantrySample | None = None
        self._latest_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_fh = None
        self._csv_writer = None

        self._vel_buf: deque[tuple[float, tuple[float, float, float]]] = deque(maxlen=self._fd_n)

    # ---- public ------------------------------------------------------------
    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def t0_monotonic(self) -> float:
        return self._t0_mono

    def set_target(self, target_mm: Sequence[float], waypoint_index: int) -> None:
        with self._target_lock:
            self._target_mm = (float(target_mm[0]), float(target_mm[1]), float(target_mm[2]))
            self._waypoint_index = int(waypoint_index)

    def latest_sample(self) -> GantrySample | None:
        with self._latest_lock:
            return self._latest

    def start(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_writer.writerow(CSV_COLUMNS)
        self._csv_fh.flush()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gantry-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._csv_fh is not None:
            try:
                self._csv_fh.flush()
                self._csv_fh.close()
            finally:
                self._csv_fh = None
                self._csv_writer = None

    # ---- internal ----------------------------------------------------------
    def _sample_status(self) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float]]:
        """Sample position + velocity (units) for all three axes.

        Tries the bulk ``get_status()`` first. If it errors (typical case:
        firmware status code 664 'axis not enabled / not homed'), falls back
        to per-axis ``get_axis_position`` / ``get_axis_speed`` reads. The
        per-axis SDK calls are independent of the machine-status query and
        routinely succeed in the 664 state, which is what lets us log
        telemetry during manual recordings on a non-physically-homed gantry.
        """
        with self._lock:
            try:
                status = self._controller.get_status()
                pos = (float(status.realPos[0]), float(status.realPos[1]),
                       float(status.realPos[2]))
                vel = (float(status.realSpeed[0]), float(status.realSpeed[1]),
                       float(status.realSpeed[2]))
            except Exception:
                pos = (
                    float(self._controller.get_axis_position(AXES[0])),
                    float(self._controller.get_axis_position(AXES[1])),
                    float(self._controller.get_axis_position(AXES[2])),
                )
                try:
                    vel = (
                        float(self._controller.get_axis_speed(AXES[0])),
                        float(self._controller.get_axis_speed(AXES[1])),
                        float(self._controller.get_axis_speed(AXES[2])),
                    )
                except Exception:
                    vel = (0.0, 0.0, 0.0)
        t_mono = time.monotonic()
        t_unix = time.time()
        return t_unix, t_mono, pos, vel

    def _accel_smoothed_central_diff(self) -> tuple[float, float, float]:
        # SMA-smoothed central difference: average front and back halves of the
        # window, then differentiate between the two half-centroids.
        n = len(self._vel_buf)
        if n < self._fd_n:
            return (0.0, 0.0, 0.0)
        half = self._fd_n // 2
        items = list(self._vel_buf)
        front = items[:half]
        back = items[-half:]
        t_front = sum(it[0] for it in front) / len(front)
        t_back = sum(it[0] for it in back) / len(back)
        dt = t_back - t_front
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        ax = (sum(it[1][0] for it in back) / len(back) - sum(it[1][0] for it in front) / len(front)) / dt
        ay = (sum(it[1][1] for it in back) / len(back) - sum(it[1][1] for it in front) / len(front)) / dt
        az = (sum(it[1][2] for it in back) / len(back) - sum(it[1][2] for it in front) / len(front)) / dt
        return (ax, ay, az)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                t_unix, t_mono, pos_units, vel_units = self._sample_status()
            except Exception as exc:  # don't crash the logger thread
                print(f"[gantry-logger] get_status error: {exc}", file=sys.stderr)
                self._stop.wait(self._period)
                continue

            pos_mm = (
                units_to_mm(pos_units[0], AXES[0]),
                units_to_mm(pos_units[1], AXES[1]),
                units_to_mm(pos_units[2], AXES[2]),
            )
            vel_mm = (
                units_to_mm(vel_units[0], AXES[0]),
                units_to_mm(vel_units[1], AXES[1]),
                units_to_mm(vel_units[2], AXES[2]),
            )

            # Buffer mm-velocity so acceleration comes out in mm/s^2 directly.
            self._vel_buf.append((t_mono, vel_mm))
            acc_mm = self._accel_smoothed_central_diff()

            is_moving = any(abs(v) > 1e-6 for v in vel_units)

            with self._target_lock:
                target_mm = self._target_mm
                wp_idx = self._waypoint_index

            sample = GantrySample(
                timestamp_unix=t_unix,
                timestamp_monotonic=t_mono,
                elapsed_s=t_mono - self._t0_mono,
                pos_units=pos_units,
                pos_mm=pos_mm,
                vel_units_s=vel_units,
                vel_mm_s=vel_mm,
                acc_mm_s2=acc_mm,
                target_mm=target_mm,
                waypoint_index=wp_idx,
                is_moving=is_moving,
            )

            with self._latest_lock:
                self._latest = sample

            if self._csv_writer is not None:
                self._csv_writer.writerow([
                    f"{sample.timestamp_unix:.6f}",
                    f"{sample.timestamp_monotonic:.6f}",
                    f"{sample.elapsed_s:.6f}",
                    f"{pos_units[0]:.6f}", f"{pos_units[1]:.6f}", f"{pos_units[2]:.6f}",
                    f"{pos_mm[0]:.4f}", f"{pos_mm[1]:.4f}", f"{pos_mm[2]:.4f}",
                    f"{vel_units[0]:.6f}", f"{vel_units[1]:.6f}", f"{vel_units[2]:.6f}",
                    f"{vel_mm[0]:.4f}", f"{vel_mm[1]:.4f}", f"{vel_mm[2]:.4f}",
                    f"{acc_mm[0]:.4f}", f"{acc_mm[1]:.4f}", f"{acc_mm[2]:.4f}",
                    f"{target_mm[0]:.4f}", f"{target_mm[1]:.4f}", f"{target_mm[2]:.4f}",
                    wp_idx,
                    int(is_moving),
                ])

            next_tick += self._period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                # Fell behind; resync so we don't burst.
                next_tick = time.monotonic()


# -----------------------------------------------------------------------------
# Motion
# -----------------------------------------------------------------------------
def _dominant_axis_scale(displacement_mm: Sequence[float]) -> tuple[float, int]:
    """Return (SCALE_MM_PER_UNIT[dominant], dominant_axis_index)."""
    abs_disp = [abs(d) for d in displacement_mm]
    if max(abs_disp) <= 0.0:
        return SCALE_MM_PER_UNIT[Axis.X], 0
    idx = abs_disp.index(max(abs_disp))
    return SCALE_MM_PER_UNIT[AXES[idx]], idx


def _read_current_pos_mm(controller: FMC4030Controller, lock: threading.RLock) -> tuple[float, float, float]:
    """Return current (x, y, z) in mm.

    Prefers ``get_status()`` (single SDK call, all three axes). If it errors
    — typically code 664 'axis not enabled / not homed' — falls back to
    per-axis ``get_axis_position()`` reads, which routinely succeed in 664.
    """
    with lock:
        try:
            st = controller.get_status()
            return (
                units_to_mm(float(st.realPos[0]), AXES[0]),
                units_to_mm(float(st.realPos[1]), AXES[1]),
                units_to_mm(float(st.realPos[2]), AXES[2]),
            )
        except Exception:
            return (
                units_to_mm(float(controller.get_axis_position(AXES[0])), AXES[0]),
                units_to_mm(float(controller.get_axis_position(AXES[1])), AXES[1]),
                units_to_mm(float(controller.get_axis_position(AXES[2])), AXES[2]),
            )


def _wait_for_all_stopped(controller: FMC4030Controller, lock: threading.RLock, poll_hz: float = 20.0) -> None:
    period = 1.0 / poll_hz
    while True:
        if EMERGENCY_STOP.is_set():
            return
        with lock:
            all_stopped = all(controller.is_axis_stopped(a) for a in AXES)
        if all_stopped:
            return
        time.sleep(period)


def move_to_xyz_mm(
    controller: FMC4030Controller,
    target_mm: Sequence[float],
    speed_mm_s: float,
    acc_mm_s2: float,
    dec_mm_s2: float,
    *,
    mode: str = "line",
    lock: threading.RLock | None = None,
    logger: GantryTelemetryLogger | None = None,
    waypoint_index: int = 0,
) -> None:
    """Move the gantry to an absolute (x_mm, y_mm, z_mm) target and block until
    every axis reports stopped. See module docstring for the speed-conversion
    rationale."""
    if len(target_mm) != 3:
        raise ValueError("target_mm must be (x, y, z) in mm")
    if speed_mm_s <= 0 or acc_mm_s2 <= 0 or dec_mm_s2 <= 0:
        raise ValueError("speed/acc/dec must be > 0")

    own_lock = lock if lock is not None else threading.RLock()
    cur_mm = _read_current_pos_mm(controller, own_lock)
    disp_mm = (target_mm[0] - cur_mm[0], target_mm[1] - cur_mm[1], target_mm[2] - cur_mm[2])

    if logger is not None:
        logger.set_target(target_mm, waypoint_index)

    if mode == "line":
        scale, _ = _dominant_axis_scale(disp_mm)
        end_units = (
            mm_to_units(target_mm[0], AXES[0]),
            mm_to_units(target_mm[1], AXES[1]),
            mm_to_units(target_mm[2], AXES[2]),
        )
        speed_units = speed_mm_s / scale
        acc_units = acc_mm_s2 / scale
        dec_units = dec_mm_s2 / scale
        with own_lock:
            controller.line_move_3d(
                axes=AXES,
                end_x=end_units[0],
                end_y=end_units[1],
                end_z=end_units[2],
                speed=speed_units,
                acc=acc_units,
                dec=dec_units,
            )
    elif mode == "sequential":
        for i, axis in enumerate(AXES):
            if abs(disp_mm[i]) < 1e-6:
                continue
            scale_i = SCALE_MM_PER_UNIT[axis]
            with own_lock:
                controller.jog_single_axis(
                    axis,
                    position_units=mm_to_units(target_mm[i], axis),
                    speed_units=speed_mm_s / scale_i,
                    acc_units=acc_mm_s2 / scale_i,
                    dec_units=dec_mm_s2 / scale_i,
                    relative=False,
                )
            _wait_for_all_stopped(controller, own_lock)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    _wait_for_all_stopped(controller, own_lock)


# -----------------------------------------------------------------------------
# CLI / waypoints / metadata
# -----------------------------------------------------------------------------
def _parse_waypoints_csv(path: Path, default_speed_mm_s: float) -> list[Waypoint]:
    waypoints: list[Waypoint] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"x_mm", "y_mm", "z_mm"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Waypoints CSV {path} missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            try:
                speed_str = (row.get("speed_mm_s") or "").strip()
                dwell_str = (row.get("dwell_s") or "").strip()
                waypoints.append(Waypoint(
                    x_mm=float(row["x_mm"]),
                    y_mm=float(row["y_mm"]),
                    z_mm=float(row["z_mm"]),
                    speed_mm_s=float(speed_str) if speed_str else default_speed_mm_s,
                    dwell_s=float(dwell_str) if dwell_str else 0.0,
                ))
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"Bad waypoint row {i} in {path}: {exc}")
    if not waypoints:
        raise SystemExit(f"No waypoints in {path}")
    return waypoints


def _resolve_soft_limit(device_val: float | None, cli_val: float | None) -> float | None:
    if cli_val is not None:
        return cli_val
    return device_val


def _device_soft_limits_mm(controller: FMC4030Controller, lock: threading.RLock) -> tuple[list[float | None], list[float | None]]:
    """Return (min_mm[3], max_mm[3]) from controller.get_device_parameters().
    Treat min==max==0 on an axis as 'unconfigured' (returns None for both)."""
    with lock:
        dev = controller.get_device_parameters()
    mins: list[float | None] = []
    maxs: list[float | None] = []
    for i in range(3):
        lo_units = float(dev.soft_limit_min[i])
        hi_units = float(dev.soft_limit_max[i])
        if lo_units == 0.0 and hi_units == 0.0:
            mins.append(None)
            maxs.append(None)
        else:
            mins.append(units_to_mm(lo_units, AXES[i]))
            maxs.append(units_to_mm(hi_units, AXES[i]))
    return mins, maxs


def _validate_soft_limits(
    waypoints: Iterable[Waypoint],
    soft_min_mm: Sequence[float | None],
    soft_max_mm: Sequence[float | None],
) -> None:
    for i, wp in enumerate(waypoints):
        for axis_idx, (mm_val, name) in enumerate(zip((wp.x_mm, wp.y_mm, wp.z_mm), "XYZ")):
            lo = soft_min_mm[axis_idx]
            hi = soft_max_mm[axis_idx]
            if lo is not None and mm_val < lo:
                raise SystemExit(
                    f"Waypoint {i} {name}={mm_val:.3f} mm violates soft-limit min {lo:.3f} mm"
                )
            if hi is not None and mm_val > hi:
                raise SystemExit(
                    f"Waypoint {i} {name}={mm_val:.3f} mm violates soft-limit max {hi:.3f} mm"
                )


def make_gantry_run_dir(root: Path, suffix: str = "gantry_run") -> Path:
    """Return a fresh ``<root>/YYYYMMDD/YYYYMMDD_HHMMSS_<suffix>/`` directory,
    auto-suffixed with ``_NN`` if it collides. Public helper shared by the
    headless CLI and the live PyQt panel."""
    now = datetime.now()
    day_dir = root / now.strftime("%Y%m%d")
    stem = now.strftime("%Y%m%d_%H%M%S") + "_" + suffix
    candidate = day_dir / stem
    n = 0
    while candidate.exists():
        n += 1
        candidate = day_dir / f"{stem}_{n:02d}"
    candidate.mkdir(parents=True)
    return candidate


# Backwards-compat alias used inside this file before the rename.
_make_run_dir = make_gantry_run_dir


def _install_sigint_handler(
    controller: FMC4030Controller,
    lock: threading.RLock,
    logger: GantryTelemetryLogger | None,
) -> None:
    def handler(signum, frame):
        del signum, frame
        if EMERGENCY_STOP.is_set():
            return
        EMERGENCY_STOP.set()
        print("\n" + "!" * 70 + "\n!!! EMERGENCY STOP !!!\n" + "!" * 70, file=sys.stderr)
        for axis in AXES:
            try:
                with lock:
                    controller.stop_axis(axis, mode=2)
            except FMC4030Error:
                pass
        if logger is not None:
            try:
                logger.stop()
            except Exception:
                pass
        try:
            controller.close()
        except Exception:
            pass
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FMC4030 gantry runner: move to XYZ targets in mm and log telemetry to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Target source: either single XYZ or a waypoints CSV.
    p.add_argument("--x-mm", type=float, default=None, help="Single-target X (mm); requires --y-mm and --z-mm.")
    p.add_argument("--y-mm", type=float, default=None)
    p.add_argument("--z-mm", type=float, default=None)
    p.add_argument("--waypoints-csv", type=Path, default=None,
                   help="CSV with columns: x_mm,y_mm,z_mm,speed_mm_s,dwell_s "
                        "(speed_mm_s/dwell_s optional; fall back to --speed-mm-s/0).")

    # Motion shape.
    p.add_argument("--speed-mm-s", type=float, default=20.0,
                   help="Path speed in mm/s (converted via dominant-axis scale; see module docstring).")
    p.add_argument("--acc-mm-s2", type=float, default=50.0)
    p.add_argument("--dec-mm-s2", type=float, default=50.0)
    p.add_argument("--mode", choices=("line", "sequential"), default="line")

    # Logging.
    p.add_argument("--log-hz", type=float, default=100.0, help="Telemetry sample rate (Hz).")

    # Controller connection.
    p.add_argument("--gantry-ip", type=str, default="192.168.0.30")
    p.add_argument("--gantry-port", type=int, default=8088)
    p.add_argument("--gantry-id", type=int, default=1)

    # Safety.
    p.add_argument("--soft-limit-min-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Override device-reported min soft limits (mm). All three required.")
    p.add_argument("--soft-limit-max-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))

    # IO.
    p.add_argument("--trajectory-dir", type=Path, default=Path("data"),
                   help="Root for data/YYYYMMDD/<timestamp>_gantry_run/ output folders.")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect, read state, validate waypoints, write empty CSV header, exit (no motion).")
    p.add_argument("--connect-test", action="store_true",
                   help="Connect, read version + soft limits + current pos, close, exit. "
                        "No target/CSV/output-dir required. Used by the GUI 'Connect' button.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # --connect-test: connect, print version + soft limits + current pos, exit.
    # No target validation, no output dir, no CSVs. The GUI's Connect button
    # uses this to verify a working controller link before any motion is set up.
    if args.connect_test:
        config = ControllerConfig(controller_id=args.gantry_id, ip=args.gantry_ip, port=args.gantry_port)
        controller = FMC4030Controller()
        try:
            controller.connect(config)
        except FMC4030Error as exc:
            print(f"✗ Connect failed ({args.gantry_ip}:{args.gantry_port}): {exc}", file=sys.stderr)
            return 2
        lock = threading.RLock()
        try:
            try:
                with lock:
                    ver = controller.get_version_info()
            except FMC4030Error as exc:
                ver = None
                print(f"! get_version_info failed: {exc}", file=sys.stderr)
            try:
                dev_min, dev_max = _device_soft_limits_mm(controller, lock)
            except FMC4030Error as exc:
                dev_min = [None, None, None]
                dev_max = [None, None, None]
                print(f"! get_device_parameters failed: {exc}", file=sys.stderr)
            try:
                cur_mm = _read_current_pos_mm(controller, lock)
            except FMC4030Error as exc:
                cur_mm = (math.nan, math.nan, math.nan)
                print(f"! position read failed: {exc}", file=sys.stderr)
            print("=" * 70)
            print(f"✓ Connected to FMC4030 at {args.gantry_ip}:{args.gantry_port}")
            if ver is not None:
                print(f"  firmware={ver.firmware}  lib={ver.library}  serial={ver.serial}")
            print(f"  current pos (mm): X={cur_mm[0]:.3f}  Y={cur_mm[1]:.3f}  Z={cur_mm[2]:.3f}")
            print(f"  soft limits (mm): min={dev_min}  max={dev_max}")
            print(f"  scale (mm/unit):  " + "  ".join(
                f"{AXIS_NAMES[a]}={SCALE_MM_PER_UNIT[a]}" for a in AXES))
            print("=" * 70)
        finally:
            try:
                controller.close()
            except FMC4030Error:
                pass
        return 0

    # Resolve target(s).
    if args.waypoints_csv is not None:
        waypoints = _parse_waypoints_csv(args.waypoints_csv, args.speed_mm_s)
    else:
        for v, name in ((args.x_mm, "--x-mm"), (args.y_mm, "--y-mm"), (args.z_mm, "--z-mm")):
            if v is None:
                raise SystemExit(
                    f"Either --waypoints-csv or all of --x-mm/--y-mm/--z-mm must be provided "
                    f"({name} missing)."
                )
        waypoints = [Waypoint(args.x_mm, args.y_mm, args.z_mm, args.speed_mm_s, 0.0)]

    config = ControllerConfig(controller_id=args.gantry_id, ip=args.gantry_ip, port=args.gantry_port)
    controller = FMC4030Controller()
    lock = threading.RLock()

    try:
        controller.connect(config)
    except FMC4030Error as exc:
        print(f"✗ Failed to connect to gantry {args.gantry_ip}:{args.gantry_port}: {exc}", file=sys.stderr)
        return 2

    logger: GantryTelemetryLogger | None = None
    run_dir: Path | None = None
    metadata: dict = {}
    metadata_written = False

    try:
        # Soft limits: device first, CLI overrides.
        try:
            dev_min, dev_max = _device_soft_limits_mm(controller, lock)
        except FMC4030Error as exc:
            print(f"! get_device_parameters failed: {exc} (treating soft limits as unset)", file=sys.stderr)
            dev_min = [None, None, None]
            dev_max = [None, None, None]

        cli_min = list(args.soft_limit_min_mm) if args.soft_limit_min_mm else [None, None, None]
        cli_max = list(args.soft_limit_max_mm) if args.soft_limit_max_mm else [None, None, None]
        soft_min = [_resolve_soft_limit(dev_min[i], cli_min[i]) for i in range(3)]
        soft_max = [_resolve_soft_limit(dev_max[i], cli_max[i]) for i in range(3)]

        _validate_soft_limits(waypoints, soft_min, soft_max)

        # Initial state read.
        try:
            with lock:
                ver = controller.get_version_info()
            cur_mm = _read_current_pos_mm(controller, lock)
        except FMC4030Error as exc:
            print(f"! Initial status read failed: {exc}", file=sys.stderr)
            ver = None
            cur_mm = (math.nan, math.nan, math.nan)

        print("=" * 70)
        print(f"FMC4030 connected at {args.gantry_ip}:{args.gantry_port}")
        if ver is not None:
            print(f"  firmware={ver.firmware}  lib={ver.library}  serial={ver.serial}")
        print(f"  current pos (mm): X={cur_mm[0]:.2f}  Y={cur_mm[1]:.2f}  Z={cur_mm[2]:.2f}")
        print(f"  soft limits (mm): min={soft_min}  max={soft_max}")
        print(f"  waypoints ({len(waypoints)}):")
        for i, wp in enumerate(waypoints):
            print(f"    [{i}] X={wp.x_mm:.2f}  Y={wp.y_mm:.2f}  Z={wp.z_mm:.2f}  "
                  f"v={wp.speed_mm_s:.2f} mm/s  dwell={wp.dwell_s:.2f}s")
        print("=" * 70)

        run_dir = _make_run_dir(Path(args.trajectory_dir), "gantry_run")
        csv_path = run_dir / "gantry_telemetry.csv"

        # waypoints.csv (planned, for reference).
        with open(run_dir / "waypoints.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "x_mm", "y_mm", "z_mm", "speed_mm_s", "dwell_s"])
            for i, wp in enumerate(waypoints):
                w.writerow([i, wp.x_mm, wp.y_mm, wp.z_mm, wp.speed_mm_s, wp.dwell_s])

        metadata = {
            "cli_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "scale_mm_per_unit": {AXIS_NAMES[a]: SCALE_MM_PER_UNIT[a] for a in AXES},
            "soft_limit_min_mm": soft_min,
            "soft_limit_max_mm": soft_max,
            "device_soft_limit_min_mm": dev_min,
            "device_soft_limit_max_mm": dev_max,
            "controller": {
                "ip": args.gantry_ip,
                "port": args.gantry_port,
                "id": args.gantry_id,
                "firmware": getattr(ver, "firmware", None),
                "library": getattr(ver, "library", None),
                "serial": getattr(ver, "serial", None),
            },
            "start_unix": time.time(),
            "start_monotonic": time.monotonic(),
        }

        if args.dry_run:
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(CSV_COLUMNS)
            metadata["dry_run"] = True
            metadata["end_unix"] = time.time()
            metadata["end_monotonic"] = time.monotonic()
            with open(run_dir / "run_metadata.json", "w") as fh:
                json.dump(metadata, fh, indent=2)
            metadata_written = True
            print(f"[dry-run] Validation OK. Output: {run_dir}")
            return 0

        logger = GantryTelemetryLogger(controller, csv_path, log_hz=args.log_hz, lock=lock)
        _install_sigint_handler(controller, lock, logger)

        logger.start()
        for i, wp in enumerate(waypoints):
            if EMERGENCY_STOP.is_set():
                break
            print(f"\n→ waypoint [{i}] target=({wp.x_mm:.2f}, {wp.y_mm:.2f}, {wp.z_mm:.2f}) mm "
                  f"at {wp.speed_mm_s:.2f} mm/s [mode={args.mode}]")
            move_to_xyz_mm(
                controller,
                (wp.x_mm, wp.y_mm, wp.z_mm),
                wp.speed_mm_s,
                args.acc_mm_s2,
                args.dec_mm_s2,
                mode=args.mode,
                lock=lock,
                logger=logger,
                waypoint_index=i,
            )
            if wp.dwell_s > 0 and not EMERGENCY_STOP.is_set():
                time.sleep(wp.dwell_s)

        logger.stop()
        metadata["end_unix"] = time.time()
        metadata["end_monotonic"] = time.monotonic()
        print(f"\n✓ Done. Output: {run_dir}")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n✗ Motion loop error: {exc}", file=sys.stderr)
        for axis in AXES:
            try:
                with lock:
                    controller.stop_axis(axis, mode=2)
            except FMC4030Error:
                pass
        if logger is not None:
            try:
                logger.stop()
            except Exception:
                pass
        if run_dir is not None and not metadata_written:
            metadata.setdefault("error", str(exc))
            metadata.setdefault("end_unix", time.time())
            metadata.setdefault("end_monotonic", time.monotonic())
            with open(run_dir / "run_metadata.json", "w") as fh:
                json.dump(metadata, fh, indent=2)
            metadata_written = True
        raise
    finally:
        if run_dir is not None and not metadata_written:
            with open(run_dir / "run_metadata.json", "w") as fh:
                json.dump(metadata, fh, indent=2)
        try:
            controller.close()
        except FMC4030Error:
            pass


if __name__ == "__main__":
    # No CLI args -> launch the PyQt5 live control panel from gantry_panel.py.
    # Any CLI args -> headless CLI mode (authoritative). This way the gantry
    # tool has one obvious "double-click" entry point while preserving the
    # scriptable interface intact.
    if len(sys.argv) == 1:
        from gantry_panel import main as _panel_main
        sys.exit(_panel_main([]))
    sys.exit(main())
