#!/usr/bin/env python3
"""demo_e2e_dvl.py — the DVL switch, end to end, with no hardware.

    QT_QPA_PLATFORM=offscreen <py> rov_gui/tests/demo_e2e_dvl.py [wrong]

Real MainWindow + demo backend (``--dvl``: a SYNTHETIC DVL) + the real
MpcWorker on the PID follower. Drives the station the way an operator would:

    1. MANUAL -> ENABLE -> ARM, DVL button still OUT (shadow)
    2. a hand-flown pass in surge AND sway with the tags in view
    3. the worker reports the mount PROVEN -> press the DVL button (IN)
    4. START a station hold: the estimator flies on the DVL velocity
    5. DISENGAGE; the CSV and the meta say which instrument each row used

``wrong`` mounts the synthetic DVL 90 degrees out. The pass must then NOT
prove the mount, the button press must be REFUSED with the reason, and the
hold must fly on the tag finite difference.

NOT collected by the test runners (the demo_e2e.py rule: ~35 s, real
threads, real window). Everything it prints is about the PLUMBING — the
demo plant, its tag noise (1 cm) and its DVL (3 mm/s, zero latency) are
fabricated, so no number here says anything about a real DVL.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from rov_gui.backends import make_backend         # noqa: E402
from rov_gui.qt import QTimer, QtWidgets          # noqa: E402
from rov_gui.state import PilotInput              # noqa: E402
from rov_gui.window import MainWindow             # noqa: E402


class Opts:
    source = "demo"
    mpc = True
    dvl = True
    dvl_host = None
    dvl_port = None
    demo_dvl_yaw_deg = 0.0
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_dvl_")
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"
    nav_config = os.path.join(ROOT, "config/hw_nav.yaml")
    mpc_config = os.path.join(ROOT, "config/hw_mpc.yaml")
    nav_geometry = None
    mpc_mode = "pid"
    rov_model = "heavy_gripper"
    imu_dr = None
    imu_dr_attitude = None
    imu_dr_control = False
    c3_imu_rate = 200
    demo_imu_bias = None
    demo_imu_gyro_bias = None
    pose = False
    demo_object = "still"
    replay_session = None
    policy = False
    policy_ckpt = None
    policy_repo = None
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    fstereo = False


PASS_S = 16.0          # the hand-flown pass
HOLD_S = 6.0


def main() -> int:
    wrong = "wrong" in sys.argv[1:]
    if wrong:
        Opts.demo_dvl_yaw_deg = 90.0
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses, logs = [], []
    win.bus.mpc_status.connect(statuses.append)
    win.bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    win.attach(backend)
    # The window's own 20 Hz pilot pump would interleave neutral frames with
    # the pass below; this driver IS the pilot for the next half minute.
    win.cmd_timer.stop()
    st = {"phase": "boot", "t0": time.monotonic(), "asked": False,
          "t_hold": None, "csv": None, "refusal": ""}

    def step():
        t = time.monotonic() - st["t0"]
        s = statuses[-1] if statuses else None
        if st["phase"] == "boot" and t > 3.0:
            win.bus.cmd_mode.emit("MANUAL")
            win.bus.cmd_enable.emit(True)
            st["phase"] = "arm"
        elif st["phase"] == "arm" and t > 4.0:
            win.bus.cmd_arm.emit(True)
            st["phase"], st["t_pass"] = "pass", t
        elif st["phase"] == "pass":
            tp = t - st["t_pass"]
            if tp < PASS_S:
                # forward-and-back, left-and-right: motion in BOTH body axes
                win.bus.cmd_pilot.emit(PilotInput(
                    surge=0.6 * math.sin(2 * math.pi * tp / 5.0),
                    sway=0.6 * math.sin(2 * math.pi * (tp - 1.5) / 6.5),
                    source="e2e"))
            else:
                win.bus.cmd_pilot.emit(PilotInput(source="e2e"))
                st["phase"], st["t_sw"] = "switch", t
        elif st["phase"] == "switch":
            if not st["asked"] and t > st["t_sw"] + 1.5:
                st["proven"] = bool(s and s.dvl_proven)
                st["note"] = s.dvl_note if s else ""
                win._traj_panel.btn_dvl.click()        # the operator's click
                st["asked"] = True
            elif st["asked"] and t > st["t_sw"] + 2.5:
                st["use"] = bool(s and s.dvl_use)
                st["refusal"] = s.reason if s else ""
                st["button"] = win._traj_panel.btn_dvl.text()
                win.bus.cmd_mpc_scenario.emit(
                    {"shape": "station", "origin_tag": None,
                     "yaw_map_deg": 90.0})
                win.bus.cmd_mpc_start.emit()
                st["phase"] = "start"
        elif st["phase"] == "start":
            if s is not None and s.phase == "station":
                st["phase"], st["t_hold"] = "hold", t
            elif t > st["t_sw"] + 30.0:
                print("FAIL: START never reached the station:",
                      s.reason if s else "no status")
                app.quit()
        elif st["phase"] == "hold" and t > st["t_hold"] + HOLD_S:
            mpc = backend.mpc
            st["csv"] = str(mpc._csv_path) if mpc._csv_path else None
            st["chip"] = win._traj_panel.chip.text()
            st["frozen"] = not win._traj_panel.btn_dvl.isEnabled()
            win.bus.cmd_mpc_engage.emit(False)
            st["phase"] = "done"
            QTimer.singleShot(800, app.quit)

    drv = QTimer()
    drv.setInterval(50)
    drv.timeout.connect(step)
    drv.start()
    QTimer.singleShot(90000, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()

    rc = 0

    def check(ok, what):
        nonlocal rc
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        rc |= 0 if ok else 1

    print(f"\n[{'WRONG mount (90 deg)' if wrong else 'correct mount'}] "
          f"FABRICATED demo plant and DVL — a wiring check")
    proof = [m for _l, m in logs if "dvl:" in m and "PROVEN" in m]
    check(any(s.dvl_state for s in statuses), "MpcStatus carries a DVL state")
    check(st["phase"] == "done", f"the run completed (phase {st['phase']})")
    hold = [s for s in statuses if s.engaged and s.phase == "station"]
    if not wrong:
        check(bool(proof), "the pass proved the mount: "
              + (proof[0][:110] if proof else st.get("note", "")))
        check(st.get("use") is True and st.get("button") == "DVL IN",
              f"the button went IN ({st.get('button')}, {st.get('refusal')})")
        n_dvl = sum(1 for s in hold if s.vel_src == "dvl")
        check(hold and n_dvl >= 0.9 * len(hold),
              f"the hold flew on the DVL ({n_dvl}/{len(hold)} ticks)")
        check("v:DVL" in st.get("chip", ""), f"chip: {st.get('chip', '')[:70]}")
    else:
        check(not proof and not st.get("proven"),
              f"the mount was NOT proven ({st.get('note', '')[:100]})")
        check(st.get("use") is False and st.get("button") == "DVL OUT"
              and "refused" in st.get("refusal", ""),
              f"the click was refused: {st.get('refusal', '')[:120]}")
        check(hold and all(s.vel_src == "tag_fd" for s in hold),
              "the hold flew on the tag finite difference")
    check(st.get("frozen") is True, "the button is frozen while engaged")
    path = st["csv"]
    if path and os.path.exists(path):
        rows = list(csv.DictReader(open(path, newline="")))
        src = {k: sum(1 for r in rows if r["vel_src"] == k)
               for k in sorted({r["vel_src"] for r in rows})}
        meta = json.load(open(path[:-4] + ".meta.json"))["dvl"]
        print(f"  csv {path}: {len(rows)} rows, vel_src {src}")
        print(f"  meta dvl: use_at_engage {meta['use_at_engage']}, proven "
              f"{meta['proven']}, rows {meta['rows']}")
        check(meta["enabled"] and meta["source"] == "demo",
              "meta says the DVL was the SYNTHETIC one")
        check(all(r["dvl_u"] != "nan" for r in rows[5:]),
              "the DVL reading is on every row, IN or OUT")
        raw = path[:-4] + "_dvl.jsonl"
        check(not os.path.exists(raw),
              "no raw DVL log in the demo (nothing real to keep)")
    else:
        check(False, "a controller CSV was written")
    print("OK" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
