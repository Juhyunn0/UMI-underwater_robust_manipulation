#!/usr/bin/env python3
"""demo_e2e_dvl_pose_policy.py — the diffusion policy ON the DVL dead
reckoning (--dvl_pose --dvl-trust --policy-dr), end to end, no hardware.

    QT_QPA_PLATFORM=offscreen <py> rov_gui/tests/demo_e2e_dvl_pose_policy.py
    DVP_POLICY_DR=0 QT_QPA_PLATFORM=offscreen <py> rov_gui/tests/demo_e2e_dvl_pose_policy.py

Real MainWindow + demo backend (synthetic DVL, RAW_IMU gyro, Bar30, depth)
+ the real PolicyWorker with the STUB session (canned chunk, no torch) + the
real MpcWorker on the PID follower, under ``--dvl_pose control --dvl-trust``:

    1. MANUAL -> ENABLE -> ARM; a short hand-flown pass with the tags in
       view (trust: the DVL went IN by itself at the first sample)
    2. the operator's procedure: TAG FIX OFF with the tags in view (the pose
       is DVL dead reckoning from here; the tags are recorded as dvp_err_m),
       then START the policy mission (one click)
         --policy-dr (default here)  : the mission flies, plans are
                                       installed, anchored on the DVL pose
         DVP_POLICY_DR=0 (A9, shipped): the mission ends by itself after
                                       policy.stale_s ('bridge')
    3. a TAG BLACKOUT (BLACKOUT_S) mid-flight: dead-reckoned ticks WITHOUT
       a tag fix — the ticks --policy-dr turns into fresh anchors
    4. STOP TRAJ from the driver after FLY_S (the stub's straight-ahead
       chunk leaves the workspace box after ~25 s), DISENGAGE; plans.jsonl /
       CSV / meta say which pose each plan and row flew on

The temporary config is the shipped config/hw_mpc.yaml with the gyro bias
learning from 5 s / 20 fixes (an e2e cannot sit a minute still).

NOT collected by the test runners (the demo_e2e.py rule: real threads, real
window, ~70 s). Every number it prints is about the PLUMBING: the demo plant,
its tag noise, its DVL and the stub policy's chunk are fabricated.
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

KNOB = os.environ.get("DVP_POLICY_DR", "1") != "0"
PASS_S = 8.0
BLACKOUT_S = (22.0, 27.0)      # seconds since the backend started
FLY_S = 12.0                   # == demo_e2e.POLICY_FLY_S


def _config() -> str:
    src = open(os.path.join(ROOT, "config/hw_mpc.yaml")).read()
    a = "  bias_min_span_s: 60.0"
    b = "  bias_min_n: 200"
    assert src.count(a) == 1 and src.count(b) == 1, "dvl_pose block changed"
    src = src.replace(a, "  bias_min_span_s: 5.0").replace(b, "  bias_min_n: 20")
    path = os.path.join(tempfile.mkdtemp(prefix="rov_gui_dvpp_cfg_"),
                        "hw_mpc.yaml")
    open(path, "w").write(src)
    return path


class Opts:
    source = "demo"
    mpc = True
    dvl = True
    dvl_pose = "control"
    dvl_trust = True
    dvl_host = None
    dvl_port = None
    demo_dvl_yaw_deg = 0.0
    demo_tag_blackout = [BLACKOUT_S]
    fps = 15.0
    ui_fps = 30.0
    thrusters = 8
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_dvpp_")
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
    policy = True
    policy_ckpt = "stub"
    policy_repo = None
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    policy_dr = KNOB               # --policy-dr
    fstereo = False


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses, logs, plans, pstates = [], [], [], []
    win.bus.mpc_status.connect(statuses.append)
    win.bus.log.connect(lambda lvl, m: logs.append((lvl, m)))
    win.bus.policy_plan.connect(plans.append)
    win.bus.policy_state.connect(pstates.append)
    win.attach(backend)
    app.aboutToQuit.connect(win.shutdown)      # the station's exit path
    win.cmd_timer.stop()
    st = {"phase": "boot", "t0": time.monotonic(), "clicks": [],
          "csv": None, "ended": None}

    def finish(s):
        mpc = backend.mpc
        st["csv"] = str(mpc._csv_path) if mpc._csv_path else None
        st["end_face"] = win._traj_panel.btn_tagfix.text()
        st["chip"] = win._traj_panel.chip.text()
        win.bus.cmd_mpc_engage.emit(False)
        st["phase"] = "done"
        QTimer.singleShot(800, app.quit)

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
                    surge=0.5 * math.sin(2 * math.pi * tp / 5.0),
                    sway=0.5 * math.sin(2 * math.pi * (tp - 1.5) / 6.5),
                    source="e2e"))
            else:
                win.bus.cmd_pilot.emit(PilotInput(source="e2e"))
                st["phase"], st["t_sw"] = "switch", t
        elif st["phase"] == "switch" and t > st["t_sw"] + 1.5:
            st["dvl_in"] = bool(s and s.dvl_use)
            win.bus.cmd_mpc_scenario.emit({"shape": "policy"})
            st["phase"] = "off"
        elif st["phase"] == "off" and s is not None and s.dvp_state == "tag":
            win._traj_panel.btn_tagfix.click()         # -> TAG FIX OFF
            st["clicks"].append(("off", t))
            st["phase"] = "start_wait"
        elif st["phase"] == "off" and t > st["t_sw"] + 10.0:
            print("FAIL: the pose never became the tag state:",
                  s.reason if s else "no status")
            st["phase"] = "failed"
            app.quit()
        elif st["phase"] == "start_wait" and t > st["clicks"][-1][1] + 1.5:
            st["off_face"] = win._traj_panel.btn_tagfix.text()
            st["off_accepted"] = bool(s and s.dvp_tag_on is False)
            st["off_reason"] = s.reason if s else ""
            win.bus.cmd_mpc_start.emit()          # policy: one click = START
            st["phase"] = "start"
        elif st["phase"] == "start":
            if s is not None and s.engaged and s.traj_on:
                st["phase"], st["t_fly"] = "fly", t
            elif t > st["t_sw"] + 25.0:
                print("FAIL: the policy never started:",
                      s.reason if s else "no status")
                st["phase"] = "failed"
                app.quit()
        elif st["phase"] == "fly":
            if s is not None and not s.engaged:
                st["ended"] = ("disengaged", s.reason)
                st["t_ended"] = t
                finish(s)
            elif s is not None and not s.traj_on:
                # ended by itself (A9's stop without the knob)
                st["ended"] = ("stopped", s.reason)
                st["t_ended"] = t
                finish(s)
            elif t > st["t_fly"] + FLY_S:
                st["n_installed_at_stop"] = backend.mpc.replay["installed"]
                win.bus.cmd_mpc_traj.emit(False)       # STOP TRAJ
                st["phase"], st["t_stop"] = "stopped", t
        elif st["phase"] == "stopped" and t > st["t_stop"] + 1.0:
            finish(s)

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

    mpc = backend.mpc
    last = mpc._replay_last or {}
    print(f"\n[--dvl_pose control --dvl-trust {'--policy-dr' if KNOB else '(A9, no --policy-dr)'}] "
          f"FABRICATED demo plant, DVL, gyro and STUB policy — a wiring check")
    check(st["phase"] == "done", f"the run completed (phase {st['phase']})")
    check(st.get("dvl_in") is True, "the DVL went IN by itself (trust)")
    check(st.get("off_accepted") is True and st.get("off_face") == "TAG FIX OFF",
          f"TAG FIX OFF was accepted before START with the tags in view "
          f"({st.get('off_face')}; {st.get('off_reason')})")
    t_fly, t_end = st.get("t_fly"), st.get("t_ended")
    print(f"  START at {t_fly if t_fly is not None else float('nan'):.1f} s, "
          f"blackout {BLACKOUT_S}, ended {st.get('ended')} at "
          f"{t_end if t_end is not None else float('nan'):.1f} s")
    eng = [s for s in statuses if s.engaged]
    dr = [s for s in eng if s.dvp_state == "dr"]
    dr_anchor = [p for p in pstates if p.active and p.fix_fresh
                 and p.t_fix is not None and p.t_fix == p.t]
    print(f"  {len(plans)} plans emitted, {len(pstates)} PolicyState ticks, "
          f"{len(dr)} dr ticks engaged, {len(dr_anchor)} DR-anchored ticks "
          f"(active); intake received {last.get('received')}, installed "
          f"{last.get('installed')}, skip_bridge {last.get('skip_bridge')}, "
          f"end_reason {last.get('end_reason')!r}, dr_ticks "
          f"{last.get('dr_ticks', 0)}, dr_plans {last.get('dr_plans', 0)}")
    path = st["csv"]
    lines = []
    if path and os.path.exists(path):
        pj = os.path.join(os.path.dirname(path), "plans.jsonl")
        if os.path.exists(pj):
            lines = [json.loads(x) for x in open(pj).read().splitlines()
                     if x.strip()]
    inst = [l for l in lines if l.get("status") in ("accept", "clip")]
    inst_dr = [l for l in inst if l.get("dvp_state") == "dr"]
    skipped = [l for l in lines if l.get("status") == "skip_bridge"]
    print(f"  plans.jsonl: {len(lines)} lines, {len(inst)} installed, "
          f"{len(inst_dr)} of them on a dr pose, {len(skipped)} skip_bridge; "
          f"dvp_state of installed lines "
          f"{sorted({str(l.get('dvp_state')) for l in inst})}")
    if KNOB:
        check(st.get("ended") is None,
              f"the mission was NOT ended by a stop (ended {st.get('ended')})")
        check(len(dr) > 150, f"the whole mission flew on the dead reckoning "
                             f"({len(dr)} dr ticks engaged)")
        check(len(dr_anchor) > 60,
              f"the tagless blackout ticks were offered as fresh anchors "
              f"({len(dr_anchor)} DR-anchored PolicyState ticks)")
        check(len(inst_dr) >= 10 and not skipped,
              f"plans were installed on the dr pose ({len(inst_dr)}), "
              f"none skipped as bridged ({len(skipped)})")
        check(st.get("end_face") == "TAG FIX OFF",
              f"TAG FIX stayed OFF through the run ({st.get('end_face')})")
        check(last.get("end_reason") == "stop",
              f"the run record ends with the driver's STOP "
              f"({last.get('end_reason')!r})")
        check(any("ON THE DVL POSE (--policy-dr)" in m for _l, m in logs),
              "the arm line says the policy flies on the DVL pose")
    else:
        ended = st.get("ended") or (None, "")
        check(ended[0] == "stopped" and last.get("end_reason") == "bridge",
              f"A9: the mission ended by itself during the blackout "
              f"(ended {ended}, end_reason {last.get('end_reason')!r})")
        check(len(skipped) >= 1 and not inst_dr,
              f"plans were skipped as bridged ({len(skipped)}), none "
              f"installed on a dr pose ({len(inst_dr)})")
        check(not dr_anchor, f"no DR tick was offered as a fresh anchor "
                             f"({len(dr_anchor)})")
        t_end = st.get("t_ended")
        check(t_end is not None and t_fly is not None
              and t_end < t_fly + 5.0,
              f"the stop came within stale_s of START (t {t_end}, START "
              f"{t_fly})")
    if path and os.path.exists(path):
        rows = list(csv.DictReader(open(path, newline="")))
        src = {k: sum(1 for r in rows if r["dvp_src"] == k)
               for k in sorted({r["dvp_src"] for r in rows})}
        off = sum(1 for r in rows if r["tagfix"] == "0")
        meta = json.load(open(path[:-4] + ".meta.json"))
        dvp = meta.get("dvl_pose", {})
        pol = meta.get("policy", {})
        print(f"  csv {path}: {len(rows)} rows, dvp_src {src}, tagfix 0 rows "
              f"{off}")
        print(f"  meta dvl_pose: control {dvp.get('control')}, trust "
              f"{dvp.get('trust_dvl')}; policy.dr_pose {pol.get('dr_pose')}; "
              f"gates_enforced.bridge_stop_dr_pose "
              f"{(pol.get('gates_enforced') or {}).get('bridge_stop_dr_pose')}; "
              f"end_reason {pol.get('end_reason', last.get('end_reason'))!r}")
        dp = pol.get("dr_pose") or {}
        if KNOB:
            check(dp.get("allowed") is True and dp.get("active") is True
                  and dp.get("ticks", 0) > 60 and dp.get("plans", 0) >= 10
                  and (pol.get("gates_enforced") or {}).get(
                      "bridge_stop_dr_pose") is False,
                  "the meta records the knob, the DR ticks and plans, and "
                  "the gate that was off")
            check(src.get("dr", 0) > 150 and off > 150,
                  "the CSV records the dead-reckoned rows and the TAG FIX OFF "
                  "rows")
        else:
            check(dp.get("allowed") is False and dp.get("ticks", 0) == 0
                  and dp.get("plans", 0) == 0
                  and (pol.get("gates_enforced") or {}).get(
                      "bridge_stop_dr_pose") is True,
                  "the meta records the shipped gate (no DR ticks, no DR plans)")
    else:
        check(False, "a controller CSV was written")
    print(f"  chip: {st.get('chip', '')[:90]}")
    print("PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
