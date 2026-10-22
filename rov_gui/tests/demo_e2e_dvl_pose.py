#!/usr/bin/env python3
"""demo_e2e_dvl_pose.py — --dvl_pose end to end, with no hardware.

    QT_QPA_PLATFORM=offscreen <py> rov_gui/tests/demo_e2e_dvl_pose.py

Real MainWindow + demo backend (a SYNTHETIC DVL, RAW_IMU gyro and Bar30) +
the real MpcWorker on the PID follower, under ``--dvl_pose control``:

    1. MANUAL -> ENABLE -> ARM; a hand-flown surge-and-sway pass with the tags
       in view proves the DVL mount -> the DVL button IN
    2. START a station hold with TAG FIX ON (the pose is the tag state)
    3. a TAG BLACKOUT (demo_tag_blackout, BLACKOUT_S): the run must NOT end —
       the pose is dead-reckoned on the DVL (state dr)
    4. the operator clicks TAG FIX (-> OFF, DVL only), then again: pressed
       while OFF it is ONE correction (TAG FIX ONCE…), then OFF again by
       itself (2026-10-18)
    5. DISENGAGE; the CSV rows and the meta say which pose flew

The temporary config is the shipped config/hw_mpc.yaml with ONE change: the
gyro bias learns from 5 s / 20 fixes instead of 60 s / 200 (an e2e cannot
sit a minute still) — everything else is what the station ships.

NOT collected by the test runners (the demo_e2e.py rule: real threads, real
window, ~80 s). Every number it prints is about the PLUMBING: the demo plant,
its tag noise and its DVL are fabricated.
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

PASS_S = 16.0
BLACKOUT_S = (44.0, 52.0)      # seconds since the backend started
OFF_AT_S, ON_AT_S, END_S = 56.0, 62.0, 68.0


def _config() -> str:
    src = open(os.path.join(ROOT, "config/hw_mpc.yaml")).read()
    a = "  bias_min_span_s: 60.0"
    b = "  bias_min_n: 200"
    assert src.count(a) == 1 and src.count(b) == 1, "dvl_pose block changed"
    src = src.replace(a, "  bias_min_span_s: 5.0").replace(b, "  bias_min_n: 20")
    path = os.path.join(tempfile.mkdtemp(prefix="rov_gui_dvp_cfg_"),
                        "hw_mpc.yaml")
    open(path, "w").write(src)
    return path


class Opts:
    source = "demo"
    mpc = True
    dvl = True
    dvl_pose = "control"
    # DVP_TRUST=1 runs the flow under --dvl-trust (2026-10-19); unset = the
    # YAML's mode (strict as shipped)
    dvl_trust = True if os.environ.get("DVP_TRUST") == "1" else None
    dvl_host = None
    dvl_port = None
    demo_dvl_yaw_deg = 0.0
    demo_tag_blackout = [BLACKOUT_S]
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_dvp_")
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"
    nav_config = os.path.join(ROOT, "config/hw_nav.yaml")
    mpc_config = _config()
    nav_geometry = None
    mpc_mode = "pid"
    rov_model = "heavy_gripper"
    imu_dr = None
    imu_dr_attitude = None
    imu_dr_control = False
    ekf = None
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


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses, logs = [], []
    win.bus.mpc_status.connect(statuses.append)
    win.bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    win.attach(backend)
    # The station's exit path (window.build_and_run). Without it the backend
    # threads outlive main(): the interpreter is torn down under a worker's
    # timer slot that is inside cv2, and the process aborted AFTER "PASS"
    # (exit 134/139 in 4 of 15 runs on 2026-10-18; logs in
    # data/20261017/1017_dvl_pose_feasibility/e2e_exit_race_20261018/).
    app.aboutToQuit.connect(win.shutdown)
    win.cmd_timer.stop()
    st = {"phase": "boot", "t0": time.monotonic(), "clicks": [],
          "t_hold": None, "csv": None}

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
                win.bus.cmd_pilot.emit(PilotInput(
                    surge=0.6 * math.sin(2 * math.pi * tp / 5.0),
                    sway=0.6 * math.sin(2 * math.pi * (tp - 1.5) / 6.5),
                    source="e2e"))
            else:
                win.bus.cmd_pilot.emit(PilotInput(source="e2e"))
                st["phase"], st["t_sw"] = "switch", t
        elif st["phase"] == "switch" and t > st["t_sw"] + 1.5:
            # (trust mode, DVP_TRUST=1: the switch went IN by itself at the
            # first sample — a click would toggle it OUT)
            if not (s and s.dvl_use):
                win._traj_panel.btn_dvl.click()        # DVL IN
            win.bus.cmd_mpc_scenario.emit(
                {"shape": "station", "origin_tag": None, "yaw_map_deg": 90.0})
            st["phase"] = "start_wait"
        elif st["phase"] == "start_wait" and t > st["t_sw"] + 3.0:
            st["dvl_in"] = bool(s and s.dvl_use)
            win.bus.cmd_mpc_start.emit()
            st["phase"] = "start"
        elif st["phase"] == "start":
            if s is not None and s.engaged:
                st["phase"], st["t_hold"] = "hold", t
            elif t > st["t_sw"] + 25.0:
                print("FAIL: never engaged:", s.reason if s else "no status")
                st["phase"] = "failed"
                app.quit()
        elif st["phase"] == "hold":
            if len(st["clicks"]) == 2:
                face = win._traj_panel.btn_tagfix.text()
                if not st["faces"] or st["faces"][-1] != face:
                    st["faces"].append(face)
            if not (s and s.engaged):
                st["phase"] = "dropped"
                st["drop_reason"] = s.reason if s else ""
                QTimer.singleShot(500, app.quit)
            elif (t > OFF_AT_S and not st["clicks"]
                  and s.dvp_state == "tag"):
                # the operator's procedure: OFF once the pose IS the tag
                # state (OFF is refused while a correction still slews in)
                win._traj_panel.btn_tagfix.click()     # -> TAG FIX OFF
                st["clicks"].append(("off", t))
            elif t > ON_AT_S and len(st["clicks"]) == 1:
                st["off_face"] = win._traj_panel.btn_tagfix.text()
                win._traj_panel.btn_tagfix.click()     # -> TAG FIX ONCE…
                st["clicks"].append(("on", t))
                st["faces"] = []
            elif t > END_S and len(st["clicks"]) == 2:
                st["end_face"] = win._traj_panel.btn_tagfix.text()
                st["end_tag_on"] = bool(s.dvp_tag_on)
                mpc = backend.mpc
                st["csv"] = str(mpc._csv_path) if mpc._csv_path else None
                st["chip"] = win._traj_panel.chip.text()
                win.bus.cmd_mpc_engage.emit(False)
                st["phase"] = "done"
                QTimer.singleShot(800, app.quit)

    drv = QTimer()
    drv.setInterval(50)
    drv.timeout.connect(step)
    drv.start()
    QTimer.singleShot(120000, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()

    rc = 0

    def check(ok, what):
        nonlocal rc
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
        rc |= 0 if ok else 1

    print("\n[--dvl_pose control] FABRICATED demo plant, DVL and gyro — a "
          "wiring check")
    check(st["phase"] == "done",
          f"the run completed (phase {st['phase']}"
          + (f", dropped: {st.get('drop_reason')}" if st.get("drop_reason")
             else "") + ")")
    check(st.get("dvl_in") is True, "the DVL went IN after the pass")
    eng = [s for s in statuses if s.engaged]
    check(any(s.dvp_mode == "control" for s in statuses),
          "MpcStatus carries the DVL pose (control)")
    dr = [s for s in eng if s.dvp_state == "dr"]
    check(len(dr) > 100, f"the blackout flew on the dead reckoning "
                         f"({len(dr)} dr ticks engaged)")
    faces = st.get("faces") or []
    check(st.get("off_face") == "TAG FIX OFF" and "TAG FIX ONCE…" in faces
          and st.get("end_face") == "TAG FIX OFF"
          and st.get("end_tag_on") is False,
          f"the TAG FIX button followed the worker ({st.get('off_face')} -> "
          f"{' -> '.join(faces)}): one correction, then OFF by itself")
    path = st["csv"]
    if path and os.path.exists(path):
        rows = list(csv.DictReader(open(path, newline="")))
        src = {k: sum(1 for r in rows if r["dvp_src"] == k)
               for k in sorted({r["dvp_src"] for r in rows})}
        off = sum(1 for r in rows if r["tagfix"] == "0")
        nan_px = sum(1 for r in rows if r["px"] == "nan")
        meta = json.load(open(path[:-4] + ".meta.json"))["dvl_pose"]
        print(f"  csv {path}: {len(rows)} rows, dvp_src {src}, tagfix 0 "
              f"rows {off}, px nan rows {nan_px}")
        print(f"  meta dvl_pose: control {meta.get('control')}, toggles "
              f"{[x['on'] for x in meta.get('toggles', [])]}, corrections "
              f"{[(c['kind'], c['applied'], c['err_m']) for c in meta.get('corrections', [])]}, "
              f"bias {meta.get('bias')}")
        check(src.get("dr", 0) > 100 and off > 60 and nan_px > 100,
              "the CSV records the dead-reckoned rows, TAG FIX OFF rows and "
              "the tagless rows")
        tg = meta.get("toggles", [])
        check([x["on"] for x in tg][-3:] == [False, True, False]
              and tg[-2].get("oneshot") is True
              and tg[-1].get("auto") == "done",
              "the meta records OFF, the one-shot ON and its own OFF")
    else:
        check(False, "a controller CSV was written")
    errs = [s.dvp_err_m for s in eng if s.dvp_state == "dr" and
            s.dvp_err_m is not None]
    if errs:
        print(f"  |pose - tag| while dead-reckoning with tags in view: max "
              f"{max(errs) * 100:.1f} cm (synthetic)")
    print(f"  chip: {st.get('chip', '')[:90]}")
    print("PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
