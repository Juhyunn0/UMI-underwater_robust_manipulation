#!/usr/bin/env python3
"""demo_e2e.py — the offline closed-loop check, promoted from a scratchpad.

    <py> rov_gui/tests/demo_e2e.py [pid|mpc|dobmpc]
                                   [station|line|square|follow|replay] [dr]

Real MainWindow + demo backend + the REAL controller (acados dobmpc by
default; pass "pid" for the PID). Drives the bus the way an operator would
(mode MANUAL -> ENABLE -> ARM -> cmd_mpc_start) and reports tracking error +
artifact paths. NOT collected by the test runners (name does not start with
test_): it takes ~40 s and builds the solver, so it is run on demand —
after any change to the control stack, before any pool session.

``dr`` additionally turns on the IMU dead reckoner and injects a KNOWN
accelerometer bias into the demo plant's synthetic C3 IMU. The demo emits a
specific force the estimator can integrate exactly, so the drift it reports
must come back as 0.5*b*t^2 — which makes this a check of the mechanization
end to end (bus transport, anchoring, frames, CSV) and not just of the wiring.
``dr-control`` does the same with the controller flying ON the estimate.

``follow`` exercises the object-follow mission (control/object_nav.py) against
the demo's SYNTHETIC object — a map-frame object back-projected through the
real camera extrinsic and stamped with the same capture time as the synthetic
NavFix, so the whole composition, the frame pairing and the closed loop run
offline. What it CANNOT show is anything about real object estimation: the
demo's T_cam_obj is built from the demo vehicle's own state, so the round trip
is structural. SAM2 mask quality, FoundationPose latency, dropout statistics
and depth noise appear only at the bench.
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
    # rec_dir is now the ONE root (MpcWorker.setup overrides hw_mpc.yaml's
    # log_dir with it), so this tempdir also keeps the e2e's synthetic CSV and
    # events.log out of the real sessions/ tree — it used to land there.
    rec_dir = tempfile.mkdtemp(prefix="rov_gui_rec_")
    rec_fps = 12.0
    fullscreen = False
    joystick = "none"
    nav_config = os.path.join(ROOT, "config/hw_nav.yaml")
    mpc_config = os.path.join(ROOT, "config/hw_mpc.yaml")
    nav_geometry = None
    mpc_mode = "dobmpc"
    rov_model = "heavy_gripper"
    imu_dr = None                 # set by the "dr" argument
    imu_dr_attitude = None
    imu_dr_control = False
    c3_imu_rate = 200
    # The bias the demo IMU is given, m/s^2 on body x. Chosen to be visible
    # in a ~30 s run without being absurd: 0.5*0.02*30^2 = 9 cm.
    demo_imu_bias = None
    demo_imu_gyro_bias = None
    # The object tracker. In the demo backend this only means "publish a
    # synthetic PoseTrack" (there is no PoseWorker and no torch) plus "let
    # MpcWorker build its object anchor".
    pose = False
    demo_object = "still"
    replay_session = None         # set by the "replay" argument


DR_BIAS = (0.02, 0.0, 0.0)


def _fake_replay_session() -> str:
    """A synthetic extract_pose output for the `replay` mission: an L-shaped
    drive (0.5 m +x, then 0.3 m +y with a 90-degree yaw), so the REAL solver
    consumes a streamed plan whose position AND yaw both move."""
    import json as _json

    import numpy as np

    d = os.path.join(tempfile.mkdtemp(prefix="rov_gui_replay_"), "demo")
    os.makedirs(d)
    pi = 3.141592653589793
    # 0.05 m/s, not the 0.08 the panel default suggests: the FABRICATED demo
    # plant tracks a straight at ~12 cm mean / 26 cm p95 error, and at
    # 0.08 m/s its lag through the yaw ramp crossed the replay divergence
    # guard (2 x anchor_max_m = 0.60 m — real-vehicle sizing) at 0.68 m
    # (2026-08-30 run). The e2e is a wiring check; the guard's sizing is the
    # real vehicle's business, so the DEMO slows down, not the guard.
    hz, v = 10.0, 0.05
    t1, t2 = 0.4 / v, 0.25 / v
    n1, n2 = int(t1 * hz), int(t2 * hz)
    t = np.arange(n1 + n2 + 1) / hz
    x = np.minimum(t, t1) * v
    y = np.maximum(t - t1, 0.0) * v
    # The heading RAMPS through the corner (~0.39 rad/s over 4 s) — a demo
    # whose yaw steps 90 degrees in one knot is not something any vehicle
    # flew, and the PlanFilter rightly rejects it (it did, 2026-08-30, which
    # is how the first version of this fixture was caught).
    yaw = np.clip((t - (t1 - 2.5)) / 5.0, 0.0, 1.0) * (pi / 2.0)
    poses = np.zeros((t.size, 8))
    poses[:, 0] = 1000.0 + t
    poses[:, 1], poses[:, 2] = x, y
    poses[:, 4] = np.cos(yaw / 2.0)       # qw   (yaw about z, w-first)
    poses[:, 7] = np.sin(yaw / 2.0)       # qz
    np.save(os.path.join(d, "poses.npy"), poses)
    with open(os.path.join(d, "poses.json"), "w", encoding="utf-8") as f:
        _json.dump({"schema": "umi_handheld_poses/1", "pose_of": "body_frd",
                    "frame": "map_ned", "note": "demo_e2e synthetic"}, f)
    with open(os.path.join(d, "frames.csv"), "w", encoding="utf-8") as f:
        f.write("idx,t_unix\n")
        for i, ti in enumerate(poses[:, 0]):
            f.write(f"{i},{ti:.6f}\n")
    return d


# The demo plant starts at (0,0) with no tag map behind it, so the missions
# here place themselves at the CURRENT pose (origin_tag None) — the tag
# anchoring is covered by test_line_mission_is_placed_at_a_tag.
SHAPES = {
    "station": {"shape": "station", "origin_tag": None, "yaw_map_deg": 90.0},
    "square": {"shape": "square", "size": 0.6, "size_y": 0.4, "speed": 0.15,
               "laps": 1, "origin_tag": None},
    "line": {"shape": "line", "length": 0.8, "speed": 0.15, "laps": 2,
             "ramp_s": 0.7, "dir_deg": 90.0, "origin_tag": None},
    # No geometry at all: what a follow holds is the offset the vehicle
    # already has, and `speed` is the setpoint's own speed cap.
    "follow": {"shape": "follow", "speed": 0.20},
    # No geometry either: the mission is a recorded demo (a synthetic one is
    # written at startup), anchored at the pose the vehicle has at START and
    # streamed through control/plan_stream.py into the REAL solver — the
    # 61-stage set_path_plan_ned consumption the offline stub cannot cover.
    "replay": {"shape": "replay"},
}
SHAPE = "square"
# The real settle is 10 s (engage.settle_s); the driver has to outwait it or
# it declares failure while the vehicle is doing exactly the right thing.
SETTLE_S = 12.0


def _report_dr(statuses) -> int:
    """Did the estimate drift by the amount the injected bias says it must?

    ``e = 0.5*b*t^2`` is the law the whole error budget rests on, so checking
    it against the LIVE stack (bus transport, anchoring, frames, the datum,
    the CSV) is worth more than checking it again in a unit test.

    But that law assumes the bias points in a FIXED WORLD DIRECTION, and the
    injected one is fixed in the BODY. Hold a heading and the two are the same
    thing; fly a square for half a minute and the bias sweeps through the
    world and partly cancels itself — a 36 s demo square comes in around a
    third of the prediction, which is physics rather than a defect. So the
    tight band is only applied when the heading actually held, and the long
    runs get an order-of-magnitude check that still catches the failure worth
    catching: a wrong frame or transform is out by much more than a factor of
    three, or points the wrong way entirely.
    """
    import numpy as np

    live = [s for s in statuses if s.dr_ok and s.dr_elapsed_s
            and s.dr_err_m is not None]
    if not live:
        notes = [s.dr_note for s in statuses if s.dr_note]
        print("FAIL: the dead reckoner never produced a live estimate. "
              f"last note: {notes[-1] if notes else '(none)'}")
        return 1
    s = live[-1]
    b = float(np.linalg.norm(DR_BIAS))
    want = 0.5 * b * s.dr_elapsed_s ** 2
    hz = np.median([x.dr_hz for x in live if x.dr_hz])
    yaws = [x.yaw_flu_deg for x in live if x.yaw_flu_deg is not None]
    swing = (max(yaws) - min(yaws)) if yaws else 999.0
    held = swing < 20.0
    lo, hi = (0.5, 2.0) if held else (0.15, 3.0)
    print(f"[dr {s.dr_mode}/{s.dr_attitude}] {len(live)} live samples, "
          f"{hz:.0f} Hz, drift {s.dr_err_m * 100:.1f} cm after "
          f"{s.dr_elapsed_s:.1f} s (0.5*b*t^2 predicts {want * 100:.1f} cm "
          f"for the injected {b:.3f} m/s^2 at a FIXED heading; this run swung "
          f"{swing:.0f} deg, band {lo:g}-{hi:g}x)")
    if not (lo * want <= s.dr_err_m <= hi * want):
        print("FAIL: the drift does not follow the injected bias — a frame, "
              "a transform or the anchoring is wrong")
        return 1
    if hz < 150:
        print(f"FAIL: only {hz:.0f} Hz reached the estimator (expect ~200)")
        return 1
    return 0


def _report_follow(statuses, objects) -> int:
    """Did the object reach the map frame, stay paired, and get followed?

    The pairing ratio is the one number worth reporting from a demo run:
    ``pair_exact`` is what says the camera extrinsic cancelled out of the
    composition, and the demo shares one capture stamp between the synthetic
    NavFix and the synthetic pose exactly as the hardware does — so a ratio
    below 1.0 here is a PLUMBING defect, not a physical one.
    """
    live = [o for o in objects if o.p_map is not None]
    if not live:
        notes = [o.note for o in objects if o.note]
        print("FAIL: no object ever reached the map frame. last note: "
              f"{notes[-1] if notes else '(none)'}")
        return 1
    paired = [o for o in live if o.pair_dt_ms is not None]
    exact = sum(1 for o in paired if o.pair_exact)
    ratio = exact / len(paired) if paired else 0.0
    d = [o.distance_m for o in live if o.distance_m is not None]
    print(f"[follow/{Opts.demo_object}] {len(live)} object fixes, "
          f"pair_exact {exact}/{len(paired)} ({ratio * 100:.0f}%), "
          f"range {min(d):.2f}-{max(d):.2f} m "
          f"(SYNTHETIC object — a wiring check, not an estimation figure)")
    if ratio < 0.99:
        print("FAIL: the object and the tag fix stopped sharing a frame — "
              "the camera extrinsic is no longer cancelling")
        return 1
    followed = [s for s in statuses if s.follow_state]
    if not followed:
        print("FAIL: the follow never armed")
        return 1
    lost = [s for s in followed if s.follow_state == "lost"]
    err = [s.follow_err_m for s in followed if s.follow_err_m is not None]
    print(f"[follow] {len(followed)} ticks, states "
          f"{sorted({s.follow_state for s in followed})}, "
          f"setpoint err max {max(err) * 100 if err else 0.0:.1f} cm")
    if lost:
        print("FAIL: the follow lost the object during the run")
        return 1
    return 0


def main() -> int:
    global SHAPE
    for a in sys.argv[1:]:
        if a in ("mpc", "dobmpc", "pid"):
            Opts.mpc_mode = a
        elif a in SHAPES:
            SHAPE = a
        elif a in ("still", "drift", "orbit"):
            Opts.demo_object = a
        elif a in ("dr", "dr-control"):
            Opts.imu_dr = "control" if a == "dr-control" else "shadow"
            Opts.imu_dr_control = a == "dr-control"
            # 'gyro', not the shipped 'ahrs': the point here is to check the
            # arithmetic against a closed form, and levelling deliberately
            # cancels part of a constant accel bias (imu_dr._level), which
            # would turn an exact assertion into a fuzzy one.
            Opts.imu_dr_attitude = "gyro"
            Opts.demo_imu_bias = DR_BIAS
    if SHAPE == "follow":
        Opts.pose = True
    if SHAPE == "replay":
        Opts.replay_session = _fake_replay_session()
        print(f"replay session (synthetic): {Opts.replay_session}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses = []
    win.bus.mpc_status.connect(statuses.append)
    win.attach(backend)
    objects = []
    win.bus.object_fix.connect(objects.append)
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
            if SHAPE == "follow":
                # Turn the (synthetic) tracker on and click the object, the
                # way a pilot would. Both signals reach the same receiver, so
                # they are delivered in order and the click is not dropped.
                win.bus.cmd_pose_enable.emit(True)
                cx, cy = backend.vehicle._demo_object_centre()
                win.bus.cmd_pose_click.emit(cx, cy)
            win.bus.cmd_mpc_scenario.emit(SHAPES[SHAPE])
            state["phase"] = "engage"
        elif state["phase"] == "engage" and t > 6.0:
            win.bus.cmd_mpc_start.emit()
            state["phase"] = "warmup"
        elif state["phase"] == "warmup" and s is not None and s.traj_on:
            state["phase"] = "traj"
            state["traj_started"] = t
        elif state["phase"] == "warmup" and t > 20.0 + SETTLE_S:
            print(f"FAIL: START did not reach the {SHAPE}. last:",
                  (s.reason if s else "no status"))
            app.quit()
        elif state["phase"] == "warmup" and SHAPE in ("station", "follow") \
                and s is not None and s.phase in ("station", "follow"):
            state["phase"] = "traj"
            state["traj_started"] = t
        elif state["phase"] == "traj" and SHAPE in ("station", "follow"):
            if t > state["traj_started"] + 8.0:      # held for 8 s: enough
                state["phase"] = "done"
                mpc = backend.mpc
                state["csv"] = str(mpc._csv_path) if mpc._csv_path else None
                win.bus.cmd_mpc_engage.emit(False)
                QTimer.singleShot(800, app.quit)
            elif s is not None and not s.engaged:
                print(f"FAIL: disengaged during the {SHAPE}:", s.reason)
                app.quit()
        elif state["phase"] == "traj":
            if s is not None and s.engaged and not s.traj_on \
                    and t > state["traj_started"] + 5.0:
                state["phase"] = "done"
                # The WHY the mission ended, captured before the disengage
                # overwrites it: a rejected replay also drops traj_on, and on
                # 2026-08-30 this driver read that as success — the honesty
                # check in main() needs the reason to tell the two apart.
                state["end_reason"] = s.reason
                mpc = backend.mpc
                state["csv"] = str(mpc._csv_path) if mpc._csv_path else None
                win.bus.cmd_mpc_engage.emit(False)
                QTimer.singleShot(800, app.quit)
            elif s is not None and not s.engaged:
                print(f"FAIL: disengaged mid-{SHAPE}:", s.reason)
                app.quit()
            elif t > state["traj_started"] + 60.0:
                print(f"FAIL: {SHAPE} never completed")
                app.quit()

    drv = QTimer()
    drv.setInterval(200)
    drv.timeout.connect(step)
    drv.start()
    QTimer.singleShot(120000, app.quit)
    app.exec_() if hasattr(app, "exec_") else app.exec()

    errs = [s.err_xy for s in statuses
            if (s.traj_on or s.phase in ("station", "follow"))
            and s.err_xy is not None]
    if not errs:
        print("FAIL: no trajectory samples")
        return 1
    import numpy as np
    e = np.array(errs)
    print(f"[{Opts.mpc_mode}/{SHAPE}] samples {len(e)}: err_xy mean "
          f"{e.mean() * 100:.1f} cm, p95 {np.percentile(e, 95) * 100:.1f} cm "
          f"(FABRICATED demo plant — a wiring check, not a performance figure)")
    csv = state["csv"]
    if csv and os.path.exists(csv):
        head = open(csv).readline().strip().split(",")
        print(f"csv {csv}: {sum(1 for _ in open(csv)) - 1} rows, "
              f"last col {head[-1]}")
    rc = 0
    if SHAPE == "replay":
        why = state.get("end_reason", "")
        print(f"[replay] mission ended: {why!r}")
        if "complete" not in why:
            print("FAIL: the replay ended without completing — a rejected or "
                  "diverged plan also drops traj_on, and that is not success")
            rc |= 1
    if SHAPE == "follow":
        rc |= _report_follow(statuses, objects)
    if Opts.imu_dr:
        rc |= _report_dr(statuses)
    if state["phase"] != "done":
        print(f"FAIL: ended in phase {state['phase']}")
        return 1
    if rc:
        return rc
    print(f"OK: demo closed loop completed the {SHAPE} with the real "
          f"controller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
