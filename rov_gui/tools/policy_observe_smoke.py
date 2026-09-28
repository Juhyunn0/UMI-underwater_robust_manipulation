#!/usr/bin/env python3
"""
policy_observe_smoke.py — POLICY OBSERVE, end to end, offline.

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python \
        rov_gui/tools/policy_observe_smoke.py
    ...and with --shots DIR it also saves renders of the trajectory panel,
    which is the only way to look at the vehicle marker without a display.

Builds the REAL MainWindow + DemoBackend + the REAL PolicyWorker (with
``policy.ckpt: stub``, so no torch and no GPU), arms a `policy` mission under
``--policy-observe`` and asserts what the mode promises:

  1. plans reach the trajectory panel — an observe run whose map stays empty
     has measured nothing;
  2. NOTHING is commanded, including that the WINDOW never takes the stick:
     the pilot's own frame must pass through `_pilot_gate` and must not end
     the mission, and the disengage must not inject its neutral;
  3. the record lands in its own tree and says so.

Why this exists as a tool rather than a test: it drives the whole process —
Qt event loop, three workers, real signal/slot delivery across threads — for
half a minute, which is not what the offline suites are for. The unit-level
guarantees are pinned in rov_gui/tests/test_policy.py (`test_observe_*`);
this is the integration check you run before a pool session, and the thing
that caught the two record-layer defects on 2026-09-03 (a demo run claiming
"real vehicle", and the `end_reason` prefix missing on the disengage path).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rov_gui.backends import make_backend                      # noqa: E402
from rov_gui.qt import QtGui, QtWidgets                        # noqa: E402
from rov_gui.state import PilotInput                           # noqa: E402
from rov_gui.window import MainWindow                          # noqa: E402


class Opts:
    source = "demo"
    mpc = True
    policy = True
    policy_ckpt = "stub"
    policy_observe = True
    nav_config = "config/hw_nav.yaml"
    mpc_config = "config/hw_mpc.yaml"
    nav_geometry = None
    mpc_mode = None
    rov_model = None


def _pump(app, sec: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < sec:
        app.processEvents()
        time.sleep(0.005)


def _shots(view, out: Path) -> list:
    """Render the panel in both modes and at a few eye angles. The vehicle
    marker is the one part of this station with no other way to inspect it."""
    from rov_gui.widgets.trajectory import TrajectoryView

    out.mkdir(parents=True, exist_ok=True)
    # A STANDALONE copy, not the docked view. The real one is inside a layout
    # that gives it whatever height is left over — about 100 px in an
    # offscreen window — and `resize` on a laid-out widget does not stick, so
    # rendering it produced a letterbox with the vehicle four pixels tall.
    # The copy carries the live view's state and its own generous size.
    shot = TrajectoryView()
    shot.resize(560, 440)
    for k in ("map_tags", "tag_size_m", "rov_size_m", "cam_tilt_deg", "pool",
              "square_ned", "square_kind", "p_ref", "yaw_ned", "policy_plans",
              "observe", "trail_act", "trail_ref", "z_centre"):
        if hasattr(view, k):
            setattr(shot, k, getattr(view, k))
    shot._set_p_act(view.p_act if view.p_act else (0.0, 0.0, 0.35))
    view = shot
    saved = []
    for name, three_d, az, el, zoom in (
            ("observe_top", False, 0.0, 55.0, 3.0),
            ("observe_3d", True, 0.0, 55.0, 3.0),
            ("observe_3d_low", True, 25.0, 25.0, 3.0),
            ("observe_3d_wide", True, 35.0, 40.0, 1.0)):
        view.three_d, view.azimuth_deg, view.elev_deg = three_d, az, el
        z0 = view.zoom
        view.zoom = zoom
        img = QtGui.QImage(view.size(), QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor("#000000"))
        view.render(img)
        p = out / f"{name}.png"
        img.save(str(p))
        view.zoom = z0
        saved.append(p)
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", metavar="DIR", default=None,
                    help="also save renders of the trajectory panel here")
    ap.add_argument("--run-s", type=float, default=25.0,
                    help="seconds to let the policy mission run (default 25)")
    a = ap.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    # BY SOURCE. Counting ALL cmd_pilot frames was wrong and hid what the
    # mode actually guarantees: with a joystick plugged into the station the
    # pilot's own frames flow at 20 Hz — that is the POINT of observe, the
    # pilot has the stick — and the first version of this tool only passed
    # because no joystick happened to be connected (2026-09-03). Everything
    # MpcWorker emits is tagged `source="mpc"` (allocation.wrench_to_axes and
    # disengage's neutral), so THAT count is the guarantee: it must be zero.
    pilots, grips, viz, stats = [], [], [], []
    win.bus.cmd_pilot.connect(pilots.append)
    win.bus.cmd_gripper_drive.connect(grips.append)
    win.bus.policy_plan_viz.connect(viz.append)
    win.bus.mpc_status.connect(stats.append)
    win.attach(backend)
    mpc = backend.mpc

    fails: list = []

    def check(ok: bool, msg: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {msg}")
        if not ok:
            fails.append(msg)

    _pump(app, 6.0)
    mpc.set_scenario({"shape": "policy"})
    win.bus.cmd_mpc_engage.emit(True)
    _pump(app, 3.0)
    print(f"\nengaged={mpc.engaged} commanding={mpc.commanding} "
          f"reason={mpc.reason!r}")
    if not mpc.engaged:
        print(f"FAIL: never engaged ({mpc.reason})")
        return 2
    win.bus.cmd_mpc_traj.emit(True)
    _pump(app, float(a.run_s))

    s = stats[-1]
    view = win._traj_panel.view
    print(f"\nplans drawn      : {len(viz)} "
          f"(statuses {sorted({v.status for v in viz})})")
    print(f"installed        : {mpc.replay and mpc.replay['installed']}")
    print(f"panel history    : {len(view.policy_plans)} "
          f"(cap {view.policy_plans.maxlen}, observe={view.observe})")
    print(f"cmd_pilot frames : {len(pilots)}   jaw drives: {len(grips)}")
    print(f"status           : engaged={s.engaged} commanding={s.commanding} "
          f"observe={s.observe} u_cmd={s.u_cmd} axes={s.axes}")
    print(f"chip             : {win._traj_panel.chip.text()[:72]}")
    print(f"run_tree         : {mpc._run_tree()}\n")

    def mpc_frames():
        return [c for c in pilots if getattr(c, "source", "") == "mpc"]

    by_src = {}
    for c in pilots:
        k = getattr(c, "source", "?")
        by_src[k] = by_src.get(k, 0) + 1
    print(f"cmd_pilot by source : {by_src or '{}'}")
    check(len(viz) > 0, "the policy's plans reached the panel")
    check(not mpc_frames(),
          f"the CONTROLLER commanded nothing ({len(mpc_frames())} mpc frames "
          f"of {len(pilots)} total)")
    check(grips == [], f"no jaw drive left the station ({len(grips)})")
    check(s.engaged and not s.commanding, "engaged but NOT commanding")
    check(bool(s.observe), "MpcStatus.observe is set")
    check(s.u_cmd == () and s.axes == (), "no wrench/axes published")
    check(s.ref_flu is not None, "the DP reference IS published (it is drawn)")
    check(s.err_xy is None, "tracking error is NOT published (no follower)")
    check(not win._mpc_engaged, "the window left the joystick with the pilot")
    check("OBSERVE" in win._traj_panel.chip.text(),
          "the chip says OBSERVE, never ENGAGED")
    check(mpc._run_tree().kind == "observe", "its own run kind (_observe leaf)")

    # THE PILOT'S OWN STICK: in a control run this frame is swallowed and
    # triggers a takeover that disengages. Here it must pass straight through
    # and the mission must survive it. Identified by a MARKER value rather
    # than by counting, because a connected joystick is pumping frames of its
    # own the whole time and a count comparison would race it.
    marker = 0.37
    win._pilot_gate(PilotInput(surge=marker, source="teleop"))
    _pump(app, 0.3)
    check(any(getattr(c, "source", "") == "teleop"
              and abs(getattr(c, "surge", 0.0) - marker) < 1e-9
              for c in pilots),
          "the pilot's stick passed through the window")
    check(mpc.engaged and mpc.traj_on, "a stick input did not end the mission")

    if a.shots:
        # A pose that puts the vehicle where the marker is worth looking at.
        view._set_p_act((0.0, 0.0, 0.35))
        view.yaw_ned = math.radians(30.0)
        for p in _shots(view, Path(a.shots)):
            print(f"  shot  {p}")

    win.bus.cmd_mpc_engage.emit(False)
    _pump(app, 1.0)
    check(not mpc_frames(),
          "disengage injected no neutral into the pilot's stream")
    end = (mpc._replay_last or {}).get("end_reason", "")
    check(str(end).startswith("observe_"),
          f"end_reason is prefixed ({end!r})")

    backend.stop()
    _pump(app, 0.5)
    print(f"\n{'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
