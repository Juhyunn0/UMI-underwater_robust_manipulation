#!/usr/bin/env python3
"""Run a whisker-dragging motion on the FMC4030 gantry via the Ubuntu SDK."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

# Allow running the script directly without installing the package
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisker_flow.gantry import (
    Axis,
    ControllerConfig,
    FMC4030Controller,
    FMC4030Error,
)


CONTROLLER_CONFIG = ControllerConfig(controller_id=1, ip="192.168.0.30", port=8088)
SCALE_MM_PER_UNIT = {
    Axis.X: 8.25,
    Axis.Y: 2.5,
    Axis.Z: 0.5,
}
AXIS_NAMES = {Axis.X: "X", Axis.Y: "Y", Axis.Z: "Z"}

EMERGENCY_STOP = False
controller = FMC4030Controller()


def units_to_mm(units: float, axis: Axis) -> float:
    return units * SCALE_MM_PER_UNIT[axis]


def mm_to_units(mm: float, axis: Axis) -> float:
    return mm / SCALE_MM_PER_UNIT[axis]


def print_scaling_info() -> None:
    print("\n" + "=" * 70)
    print("⚠  AXIS SCALING (controller units → millimeters)")
    print("=" * 70)
    for axis in Axis:
        print(f"  {AXIS_NAMES[axis]} axis: {SCALE_MM_PER_UNIT[axis]:.2f} mm per unit")
    print("=" * 70 + "\n")


def emergency_stop_handler(signum, frame) -> None:
    del signum, frame
    global EMERGENCY_STOP
    EMERGENCY_STOP = True
    print("\n" + "!" * 70)
    print("!!! EMERGENCY STOP ACTIVATED !!!")
    print("!" * 70)
    for axis in Axis:
        try:
            controller.stop_axis(axis, mode=2)
        except FMC4030Error:
            pass
    sys.exit(1)


signal.signal(signal.SIGINT, emergency_stop_handler)


def print_axis_status() -> None:
    status = controller.get_status()
    print("\nCurrent Axis Status")
    print("-" * 70)
    for axis in Axis:
        idx = int(axis)
        pos_units = status.realPos[idx]
        vel_units = status.realSpeed[idx]
        print(
            f"{AXIS_NAMES[axis]} axis → "
            f"pos: {pos_units:7.2f} units ({units_to_mm(pos_units, axis):7.2f} mm) | "
            f"vel: {vel_units:6.2f} units/s"
        )


def wait_for_axis(axis: Axis) -> None:
    print("Waiting for axis to stop...", end="", flush=True)
    while not controller.is_axis_stopped(axis):
        if EMERGENCY_STOP:
            break
        time.sleep(0.1)
        print(".", end="", flush=True)
    print(" DONE\n")


def run_drag_test() -> None:
    print("\n" + "=" * 70)
    print("TEST: Jog X axis by 10 controller units")
    print("=" * 70)
    distance_units = 10.0
    distance_mm = units_to_mm(distance_units, Axis.X)
    print(f"Expected displacement: {distance_mm:.1f} mm")
    input("Press Enter to start, or Ctrl+C to abort... ")

    controller.jog_single_axis(
        Axis.X,
        position_units=distance_units,
        speed_units=5.0,
        acc_units=10.0,
        dec_units=10.0,
        relative=True,
    )
    wait_for_axis(Axis.X)
    print_axis_status()
    print(
        f"\nMeasure the physical travel to confirm the scaling ({distance_mm:.1f} mm expected)."
    )


def main() -> None:
    print("=" * 70)
    print("FMC4030 Gantry – Whisker Dragging Test (Ubuntu)")
    print("=" * 70)
    print("Press Ctrl+C anytime for emergency stop.\n")
    print_scaling_info()

    try:
        controller.connect(CONTROLLER_CONFIG)
        print("✓ Controller connected\n")
        print_axis_status()
        run_drag_test()
        print("\n✓ Test completed")
    except FMC4030Error as exc:
        print(f"\n✗ Controller error: {exc}")
    finally:
        controller.close()
        print("✓ Controller disconnected")


if __name__ == "__main__":
    main()
