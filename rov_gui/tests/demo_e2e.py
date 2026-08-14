#!/usr/bin/env python3
"""demo_e2e.py — the offline closed-loop check, promoted from a scratchpad.

    ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/demo_e2e.py [pid]

Real MainWindow + demo backend + the REAL controller (acados dobmpc by
default; pass "pid" for the PID). Drives the bus the way an operator would
(mode MANUAL -> ENABLE -> ARM -> cmd_mpc_start) and reports tracking error +
artifact paths. NOT collected by the test runners (name does not start with
test_): it takes ~40 s and builds the solver, so it is run on demand —
after any change to the control stack, before any pool session.
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from rov_gui.qt import QtWidgets, QTimer          # noqa: E402
from rov_gui.window import MainWindow             # noqa: E402
from rov_gui.backends import make_backend         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Opts:
    source = "demo"
    mpc = True
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_rec_")
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"
    nav_config = os.path.join(ROOT, "config/hw_nav.yaml")
    mpc_config = os.path.join(ROOT, "config/hw_mpc.yaml")
    nav_geometry = None
    mpc_mode = "dobmpc"
    rov_model = "heavy_gripper"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("mpc", "dobmpc", "pid"):
        Opts.mpc_mode = sys.argv[1]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses = []
    win.bus.mpc_status.connect(statuses.append)
    win.attach(backend)
    state = {"phase": "boot", "t0": time.monotonic(), "traj_started": 0.0,
             "csv": None}

    def step():
        t = time.monotonic() - state["t0"]
        s = statuses[-1] if statuses else None
        if state["phase"] == "boot" and t > 3.0:
            win.bus.cmd_mode.emit("MANUAL")
            win.bus.cmd_enable.emit(True)
            state["phase"] = "arm"
        elif state["phase"] == "arm" and t > 4.0:
            win.bus.cmd_arm.emit(True)
            win.bus.cmd_mpc_scenario.emit(
                {"size": 0.6, "speed": 0.15, "laps": 1})
            state["phase"] = "engage"
        elif state["phase"] == "engage" and t > 6.0:
            win.bus.cmd_mpc_start.emit()
            state["phase"] = "warmup"
        elif state["phase"] == "warmup" and s is not None and s.traj_on:
            state["phase"] = "traj"
            state["traj_started"] = t
        elif state["phase"] == "warmup" and t > 20.0:
            print("FAIL: START did not reach the square. last:",
                  (s.reason if s else "no status"))
            app.quit()
        elif state["phase"] == "traj":
            if s is not None and s.engaged and not s.traj_on \
                    and t > state["traj_started"] + 5.0:
                state["phase"] = "done"
                mpc = backend.mpc
                state["csv"] = str(mpc._csv_path) if mpc._csv_path else None
                win.bus.cmd_mpc_engage.emit(False)
                QTimer.singleShot(800, app.quit)
            elif s is not None and not s.engaged:
                print("FAIL: disengaged mid-square:", s.reason)
                app.quit()
            elif t > state["traj_started"] + 60.0:
                print("FAIL: square never completed")
                app.quit()

    drv = QTimer()
    drv.setInterval(200)
    drv.timeout.connect(step)
    drv.start()
    QTimer.singleShot(120000, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()

    errs = [s.err_xy for s in statuses if s.traj_on and s.err_xy is not None]
    if not errs:
        print("FAIL: no trajectory samples")
        return 1
    import numpy as np
    e = np.array(errs)
    print(f"[{Opts.mpc_mode}] square samples {len(e)}: err_xy mean "
          f"{e.mean() * 100:.1f} cm, p95 {np.percentile(e, 95) * 100:.1f} cm "
          f"(FABRICATED demo plant — a wiring check, not a performance figure)")
    csv = state["csv"]
    if csv and os.path.exists(csv):
        head = open(csv).readline().strip().split(",")
        print(f"csv {csv}: {sum(1 for _ in open(csv)) - 1} rows, "
              f"last col {head[-1]}")
    if state["phase"] != "done":
        print(f"FAIL: ended in phase {state['phase']}")
        return 1
    print("OK: demo closed loop completed a square with the real controller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
