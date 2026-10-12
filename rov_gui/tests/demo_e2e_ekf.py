#!/usr/bin/env python3
"""demo_e2e_ekf.py — the EST switch (robot_localization EKF), end to end,
with no hardware.

    QT_QPA_PLATFORM=offscreen <py> rov_gui/tests/demo_e2e_ekf.py

Real MainWindow + demo backend (``--dvl``: a SYNTHETIC DVL at the body
origin; its button stays OUT, so the EKF must NOT fuse it) + the real
MpcWorker on the PID follower. Drives
the station the way an operator would:

    1. MANUAL -> ENABLE -> ARM; the EKF runs in shadow from the start
    2. press EST: the worker accepts (the EKF is ready)
    3. START a station hold: the controller is handed the EKF's state
    4. DISENGAGE; the CSV, the meta and the inputs log say so, and
       `rov_gui.tools.ekf_replay` re-runs the EKF on them

NOT collected by the test runners (the demo_e2e.py rule: real threads, real
window). Every number it prints is about the PLUMBING — the demo plant, its
tags and its DVL are fabricated.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from rov_gui.backends import make_backend         # noqa: E402
from rov_gui.qt import QTimer, QtWidgets          # noqa: E402
from rov_gui.window import MainWindow             # noqa: E402


class Opts:
    source = "demo"
    mpc = True
    dvl = True
    dvl_host = None
    dvl_port = None
    demo_dvl_yaw_deg = 0.0
    ekf = None
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_ekf_")
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


HOLD_S = 8.0


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses, logs = [], []
    win.bus.mpc_status.connect(statuses.append)
    win.bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    win.attach(backend)
    st = {"phase": "boot", "t0": time.monotonic(), "t_hold": None,
          "csv": None}

    def step():
        t = time.monotonic() - st["t0"]
        s = statuses[-1] if statuses else None
        if st["phase"] == "boot" and t > 3.0:
            win.bus.cmd_mode.emit("MANUAL")
            win.bus.cmd_enable.emit(True)
            st["phase"] = "arm"
        elif st["phase"] == "arm" and t > 4.0:
            win.bus.cmd_arm.emit(True)
            st["phase"], st["t_arm"] = "warm", t
        elif st["phase"] == "warm" and t > st["t_arm"] + 3.0:
            st["state_before"] = s.ekf_state if s else ""
            st["button_before"] = win._traj_panel.btn_est.text()
            win._traj_panel.btn_est.click()            # the operator's click
            st["phase"], st["t_sw"] = "switch", t
        elif st["phase"] == "switch" and t > st["t_sw"] + 1.0:
            st["ekf_control"] = bool(s and s.ekf_control)
            st["button"] = win._traj_panel.btn_est.text()
            st["reason"] = s.reason if s else ""
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
            st["frozen"] = not win._traj_panel.btn_est.isEnabled()
            st["ring"] = win._traj_panel.view.p_ekf is not None
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

    print("\n[EST switch] FABRICATED demo plant, tags and DVL — a wiring check")
    built = [m for _l, m in logs if m.startswith("ekf:")]
    check(bool(built), "the EKF was built: " + (built[0][:110] if built else ""))
    check(st["phase"] == "done", f"the run completed (phase {st['phase']})")
    check(st.get("button_before") == "EST TAG"
          and st.get("state_before") == "ready",
          f"before the click: {st.get('button_before')}, EKF "
          f"{st.get('state_before')}")
    check(st.get("ekf_control") is True and st.get("button") == "EST EKF",
          f"the click was accepted ({st.get('button')}, {st.get('reason')})")
    hold = [s for s in statuses if s.engaged and s.phase == "station"]
    flew = sum(1 for s in hold if s.ekf_flew)
    check(hold and flew >= 0.9 * len(hold),
          f"the hold flew on the EKF ({flew}/{len(hold)} ticks)")
    errs = [s.ekf_err_m for s in hold if s.ekf_err_m is not None]
    check(errs and max(errs) < 0.05,
          f"EKF - tag stayed small: max {100 * max(errs or [0]):.1f} cm")
    check(st.get("frozen") is True, "the EST button is frozen while engaged")
    check("est:EKF" in st.get("chip", ""), f"chip: {st.get('chip', '')[:70]}")
    check(st.get("ring") is True, "the plot draws the EKF marker")
    path = st["csv"]
    if path and os.path.exists(path):
        rows = list(csv.DictReader(open(path, newline="")))
        est = {k: sum(1 for r in rows if r["est"] == k)
               for k in sorted({r["est"] for r in rows})}
        meta = json.load(open(path[:-4] + ".meta.json"))["ekf"]
        print(f"  csv {path}: {len(rows)} rows, est {est}")
        print(f"  meta ekf: estimator_at_engage "
              f"{meta['estimator_at_engage']}, rows {meta['rows']}, tags "
              f"{meta['node']['counts']['pose0']['used']}, DVL "
              f"{meta['node']['counts']['twist0']['used']}")
        check(meta["estimator_at_engage"] == "ekf", "meta: flown on the EKF")
        check(est.get("ekf", 0) >= 0.9 * len(rows), "CSV est: ekf")
        # The DVL button stayed OUT: the EKF must not have been given the
        # DVL (its own gates apply), and the log keeps every report anyway.
        check(meta["node"]["counts"]["twist0"]["in"] == 0,
              "DVL switch OUT -> the EKF was not given the DVL")
        log = path[:-4] + "_ekf_inputs.jsonl"
        check(os.path.exists(log), "the inputs log is beside the CSV")
        recs = [json.loads(x) for x in open(log) if '"k":"dvl"' in x]
        check(recs and not any(r["fed"] for r in recs),
              f"...and logged all {len(recs)} DVL reports with fed=false")
        from rov_gui.tools import ekf_replay as RP
        rrows, _m = RP.replay(Path(path), {})
        late = [r for r in rrows if "flu" in r and r["row"]["ekf_px"] != "nan"]
        late = late[len(late) // 2:]
        d = [math.hypot(r["flu"][0] - float(r["row"]["ekf_px"]),
                        r["flu"][1] - float(r["row"]["ekf_py"]))
             for r in late]
        # p95, not max: the CSV's t column is rounded to 0.1 ms, so an input
        # that arrived within that of a tick can land on the other side of
        # it in the replay and move ONE row (seen: one row of 7.7 mm, every
        # other row 0.00 mm).
        d.sort()
        p95 = d[int(0.95 * (len(d) - 1))] if d else float("nan")
        check(late and p95 < 1e-3,
              f"replay vs live (second half): p95 {1e3 * p95:.2f} mm, max "
              f"{1e3 * max(d or [0]):.2f} mm")
    else:
        check(False, "a controller CSV was written")
    print("OK" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
