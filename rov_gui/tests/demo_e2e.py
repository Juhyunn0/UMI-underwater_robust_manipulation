#!/usr/bin/env python3
"""demo_e2e.py — the offline closed-loop check, promoted from a scratchpad.

    <py> rov_gui/tests/demo_e2e.py [pid|mpc|dobmpc|mpc_tuned|dobmpc_tuned]
                                   [station|line|square|follow|replay|policy]
                                   [dr] [rp]

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

``policy`` (2026-09-02, spec v2 A14) drives the REAL PolicyWorker
(rov_gui/backends/policy.py) with the STUB session (``Opts.policy_ckpt =
"stub"`` set directly on the Opts class — the ``--policy-ckpt`` flag is gone
since 2026-09-11, the checkpoint is picked in the panel: a canned
straight-ahead 0.05 m/s chunk, no torch) on the demo's synthetic depth
(identity grid): real PolicyState feed, real depth tap, real clock conversion
/ epoch / estimator, real filter + stitcher + hold-tail mask into the real
acados solver. It asserts that plans flow within 2 s of START, that every
installed plan's obs-age margin was positive, that nothing escalated, that
the run ends HONESTLY (STOP by this driver after a while, never "complete"),
and that "STOP with a plan in flight installs nothing". What it cannot show
is anything about the trained network — that is the dryrun tool's job.

``policy rp`` (2026-09-26) is the 6-DoF VARIANT row: the same real
PolicyWorker with the 7-dim ``stub_rp`` session (a 5 deg pitch ramp per
chunk), a TEMPORARY copy of config/hw_mpc.yaml with policy.action_repr
pos_rpy_width + attitude_track true + engage.attitude_axes.enabled true
(a synthetic but well-formed bench probe JSON pinned — the gate parses it
against the demo's synthetic firmware — and require_sign_probe false, the
recorded opt-out, so first_water_caps are in force), and the demo vehicle's
toy roll/pitch pendulum answering
the K/M the sink would put on the s/t extension axes. It asserts the chain
compose -> PlanFilter -> stitcher -> NedPlan.rp_ned -> HwDobMpc xref[3:5]
-> wrench_to_axes(attitude=True) -> PilotInput.roll/pitch -> plant, and the
records (plans.jsonl rp_tracked/raw.rp, CSV rp_track/ax_pitch/rpitch_deg,
MpcStatus.axes_rp, PolicyPlanViz.rp, meta run.attitude_axes). The follower
is forced to plain ``mpc`` (dobmpc is refused by dobmpc_allowed: false).
The stub's "+5 deg more pitch per chunk" is RELATIVE to the leashed anchor,
so it integrates chunk by chunk up to the rp_max_deg (20 deg [예측]) T1 clip
and the M axis sits at its cap for much of the run: the clip counter and the
cap are exercised on purpose, and no number here is a tracking figure.
What it cannot show: any wire fact (a synthetic firmware string passes the
version gate), any real moment (roll_nm / pitch_nm are [유도]), any real
attitude measurement (the residual gate sees the demo's own attitude).
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
    # events.log out of the real data/ tree — it used to land there.
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
    # The live-policy worker (backends/policy.py). Set by the "policy"
    # argument; the stub session keeps torch out of the e2e.
    policy = False
    policy_ckpt = None
    policy_repo = None
    policy_allow_device_depth = False
    policy_allow_fs_mismatch = False
    fstereo = False


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


def _variant_config() -> str:
    """config/hw_mpc.yaml with the 6-DoF variant switched ON, written to a
    tempdir: the shipped file stays the 4-DoF baseline (its resolved form is
    pinned byte-for-byte by rov_gui/tests/test_attitude_axes.py).

    The bench-probe artefact is a SYNTHETIC but WELL-FORMED probe JSON: since
    the 2026-09-26 safety audit the engage gate PARSES it (tool, pass, wire
    2.0, firmware a.b.c == the vehicle's), so a placeholder no longer arms
    the variant. The demo backend answers a synthetic firmware "4.5.1" /
    "4.5.1 (sim)" — the match is on the parsed a.b.c. No sign_probe is
    pinned (nothing writes one in this cut); the run opts out with
    require_sign_probe false, the RECORDED path, so the wire caps in force
    are first_water_caps (meta run.attitude_axes.caps_in_force
    "first_water", sign_proven false). The e2e proves the station chain,
    not a vehicle."""
    import json as _json

    import yaml

    d = tempfile.mkdtemp(prefix="rov_gui_rp_")
    fp = os.path.join(d, "attitude_axes_probe_SYNTHETIC.json")
    with open(fp, "w", encoding="utf-8") as f:
        _json.dump({"tool": "rov_gui.tools.attitude_axes_probe",
                    "purpose": "demo_e2e SYNTHETIC bench probe — no vehicle "
                               "was probed; well-formed so the parsing gate "
                               "exercises the chain against the demo "
                               "backend's synthetic firmware",
                    "armed_during_probe": False,
                    "firmware_version": "4.5.1 (sim)",
                    "mavlink_wire_version": "2.0",
                    "chan1_moved_under_s": True, "chan2_moved_under_t": True,
                    "returned_to_baseline": True,
                    "pass": True, "verdict": "PASS",
                    "verdict_note": "SYNTHETIC (demo_e2e)",
                    "sign_proven": False}, f, indent=1)
    with open(os.path.join(ROOT, "config/hw_mpc.yaml"), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.setdefault("policy", {})
    raw["policy"]["action_repr"] = "pos_rpy_width"
    raw["policy"]["attitude_track"] = True
    aa = raw.setdefault("engage", {}).setdefault("attitude_axes", {})
    aa["enabled"] = True
    aa["probe"] = fp
    aa["sign_probe"] = None
    # RECORDED opt-out: no armed in-water sign probe exists in this cut, so
    # the demo row flies first_water_caps, as a first water session would.
    aa["require_sign_probe"] = False
    out = os.path.join(d, "hw_mpc_rp.yaml")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# demo_e2e `policy rp`: config/hw_mpc.yaml + the 6-DoF variant ON "
                "(SYNTHETIC, tempdir)\n")
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
    return out


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
    # The LIVE policy seam with the stub session: no geometry, plans arrive
    # from the PolicyWorker at policy.period_s and are anchored at the
    # vehicle by the leash (spec v2 A4), so START moves nothing by itself.
    "policy": {"shape": "policy"},
}
# How long the policy is left flying before this driver presses STOP TRAJ
# (max_run_s is 500 s [결정: operator 2026-09-11 — 120 -> 500 s]; the e2e
# needs an honest end well before that).
POLICY_FLY_S = 12.0
SHAPE = "square"
RP = False          # the `rp` argument: the 6-DoF variant row (policy only)
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


def _report_policy(backend, plans, states, statuses, state) -> int:
    """Did the stub policy's chunks flow through the real seam (A14)?

    Everything here is a WIRING check on a synthetic plant and a canned
    chunk: the numbers say the clocks, the epoch, the filter and the mask
    agree with each other, not that a policy can grasp anything.
    """
    import json as _json

    mpc = backend.mpc
    rp = mpc._replay_last if mpc._replay_last else None
    if not plans:
        st = [s for s in statuses if getattr(s, "note", "")]
        from collections import Counter
        notes = Counter(str(getattr(s, "note", "")) for s in statuses)
        n_active = sum(1 for s in states if getattr(s, "active", False))
        print("FAIL: no PolicyPlan ever reached the bus. last status note: "
              f"{st[-1].note if st else '(none)'}; notes seen {dict(notes)}; "
              f"PolicyState ticks {len(states)} ({n_active} active)")
        return 1
    first_plan_dt = state.get("first_plan_dt")
    print(f"[policy/stub] {len(plans)} plans emitted, first "
          f"{first_plan_dt if first_plan_dt is not None else float('nan'):.2f} s "
          f"after START; {len(states)} PolicyState ticks "
          f"({sum(1 for s in states if s.active)} active)")
    rc = 0
    if first_plan_dt is None or first_plan_dt > 2.0:
        print("FAIL: the first plan did not arrive within 2 s of START")
        rc = 1
    if rp is None or rp.get("kind") != "policy":
        print("FAIL: the policy mission left no run record")
        return 1
    print(f"[policy/stub] intake: received {rp['received']}, installed "
          f"{rp['installed']}, clipped {rp['clipped']}, rejected "
          f"{rp['rejected']}, late {rp['late']}, skip_bridge "
          f"{rp['skip_bridge']}, halted {rp['halted']!r}, end_reason "
          f"{rp['end_reason']!r}, hold_frac(last) {rp['hold_frac']:.2f}")
    if rp["installed"] < 3:
        print("FAIL: fewer than 3 plans were installed")
        rc = 1
    if rp["halted"]:
        print(f"FAIL: the mission escalated/halted: {rp['halted']}")
        rc = 1
    if rp["end_reason"] != "stop":
        print(f"FAIL: the mission ended with {rp['end_reason']!r}, not the "
              f"driver's STOP — a rejected, diverged or silent stream also "
              f"drops traj_on, and that is not success")
        rc = 1
    csv = state.get("csv")
    if csv:
        pj = os.path.join(os.path.dirname(csv), "plans.jsonl")
        if not os.path.exists(pj):
            print("FAIL: plans.jsonl missing")
            return 1
        lines = [_json.loads(x) for x in open(pj) if x.strip()]
        inst = [l for l in lines if l.get("status") in ("accept", "clip")]
        ages = [l["margins"].get("obs_age") for l in inst]
        bad = [a for a in ages if a is None or a <= 0.0]
        # the ACTION contract (2026-09-07): every installed plan's action_raw
        # row is the 5-dim [dx, dy, dz, dyaw, width]; 10 = the legacy
        # pose10d that must no longer reach the seam
        widths = sorted({len(row) for l in inst
                         for row in (l.get("action_raw") or [])})
        print(f"[policy/stub] plans.jsonl: {len(lines)} lines, "
              f"{len(inst)} installed, obs_age margin min "
              f"{min(a for a in ages if a is not None) if ages else float('nan'):.3f} s, "
              f"kinds {sorted({l.get('kind') for l in lines})}, action_raw "
              f"width {widths} action_repr "
              f"{sorted({str(l.get('action_repr')) for l in inst})}")
        if bad or not inst:
            print("FAIL: an installed plan had a non-positive obs-age margin")
            rc = 1
        if any("action_raw" not in l for l in inst):
            print("FAIL: an installed plan line lacks action_raw")
            rc = 1
        want_w, want_repr = ((7, "pos_rpy_width") if RP
                             else (5, "pos_yaw_width"))
        if widths != [want_w]:
            print(f"FAIL: an installed plan's action_raw row is not {want_w} "
                  f"wide (widths {widths}; the flown contract is {want_repr})")
            rc = 1
        if any(l.get("action_repr") != want_repr for l in inst):
            print(f"FAIL: an installed plan line is not action_repr {want_repr}")
            rc = 1
        if RP:
            rc |= _report_policy_rp_lines(inst)
    # STOP with a plan in flight installs nothing: after the STOP the
    # counters are frozen and the inbox is empty.
    if state.get("installed_after_stop", 0) != 0 or mpc._policy_inbox is not None:
        print("FAIL: a plan was installed after STOP")
        rc = 1
    print(f"[policy/stub] after STOP: {state.get('plans_after_stop', 0)} plans "
          f"emitted, 0 installed (inbox empty) — the in-flight chunk was "
          f"dropped by the epoch/inactive rule")
    return rc


def _report_policy_rp_lines(inst) -> int:
    """The plans.jsonl side of the variant row: every installed 7-dim plan
    was TRACKED (rp_tracked, raw.rp (2,J) flown, rp_raw (2,K) decoded) and
    the stub's pitch ramp is visible in it."""
    import math as _math

    import numpy as np
    rc = 0
    if not all(l.get("rp_tracked") is True for l in inst):
        print("FAIL: an installed 7-dim plan line is not rp_tracked")
        rc = 1
    if any("rp" not in (l.get("raw") or {}) for l in inst):
        print("FAIL: an installed tracked plan line lacks raw.rp")
        rc = 1
    if any("rp_raw" not in l for l in inst):
        print("FAIL: an installed 7-dim plan line lacks the top-level rp_raw")
        rc = 1
    pitch_end = [float(np.asarray(l["raw"]["rp"], float)[1, -1])
                 for l in inst if "rp" in (l.get("raw") or {})]
    clipped = [int(l.get("rp_clipped_n", 0)) for l in inst]
    print(f"[policy/rp] plans.jsonl: rp_tracked {sum(1 for l in inst if l.get('rp_tracked'))}"
          f"/{len(inst)}, raw.rp last-knot pitch "
          f"{_math.degrees(float(np.median(pitch_end))) if pitch_end else float('nan'):.2f} deg "
          f"(median; the stub ramps 5 deg [예측] above the leashed anchor), "
          f"rp_clipped_n total {sum(clipped)}")
    return rc


def _report_policy_rp(statuses, viz, state) -> int:
    """The actuation + record side of the variant row: MpcStatus.axes_rp
    carried a (roll, pitch) pair, the PolicyPlanViz carried rp, the mission
    CSV's trailing schema-16 columns say the rows were flown WITH the axes
    and a non-zero M was sent, and meta records the variant."""
    import glob as _glob
    import json as _json
    import math as _math

    import numpy as np
    rc = 0
    pairs = [s.axes_rp for s in statuses if len(getattr(s, "axes_rp", ()) or ()) == 2]
    if not pairs:
        print("FAIL: no MpcStatus carried axes_rp — K/M never left the worker")
        rc = 1
    else:
        a = np.asarray(pairs, float)
        print(f"[policy/rp] MpcStatus.axes_rp on {len(pairs)} statuses: "
              f"max |roll| {np.abs(a[:, 0]).max():.3f}, max |pitch| "
              f"{np.abs(a[:, 1]).max():.3f} (normalised; the caps in force "
              f"are first_water_caps 0.1 / 0.15 on this row — sign not "
              f"proven, see meta run.attitude_axes.caps_in_force)")
        if np.abs(a[:, 0]).max() > 0.1 + 1e-6 or np.abs(a[:, 1]).max() > 0.15 + 1e-6:
            print("FAIL: an attitude axis exceeded first_water_caps although "
                  "the sign is unproven")
            rc = 1
        if np.abs(a[:, 1]).max() <= 0.0:
            print("FAIL: the pitch axis command was identically zero")
            rc = 1
    if not any(getattr(v, "rp", None) is not None for v in viz):
        print("FAIL: no PolicyPlanViz carried rp")
        rc = 1
    csv = state.get("csv")
    if not csv or not os.path.exists(csv):
        print("FAIL: no mission CSV")
        return 1
    with open(csv, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    need = ("rroll_deg", "rpitch_deg", "ax_roll", "ax_pitch", "rp_track")
    if head[-5:] != list(need):
        print(f"FAIL: CSV does not end with {need}: {head[-5:]}")
        return 1
    col = {k: head.index(k) for k in need}
    tracked = [r for r in rows if r[col["rp_track"]] == "1"]
    print(f"[policy/rp] CSV {os.path.basename(csv)}: {len(tracked)}/{len(rows)} rows "
          f"rp_track=1")
    if len(tracked) < 3:
        print("FAIL: fewer than 3 CSV rows were flown with the attitude axes")
        rc = 1
    else:
        axp = np.array([float(r[col["ax_pitch"]]) for r in tracked])
        rp_ref = np.array([float(r[col["rpitch_deg"]]) for r in tracked])
        print(f"[policy/rp] CSV tracked rows: max |ax_pitch| {np.abs(axp).max():.3f}, "
              f"rpitch_deg finite {int(np.isfinite(rp_ref).sum())}/{len(tracked)}, "
              f"range {np.nanmin(rp_ref):.2f}..{np.nanmax(rp_ref):.2f} deg (FLU)")
        if not np.isfinite(axp).all() or np.abs(axp).max() <= 0.0:
            print("FAIL: ax_pitch never left zero on a tracked row")
            rc = 1
        if not np.isfinite(rp_ref).any():
            print("FAIL: rpitch_deg is nan on every tracked row")
            rc = 1
    metas = [m for m in _glob.glob(os.path.join(os.path.dirname(csv), "*.json"))
             if not m.endswith("plans.jsonl")]
    meta = None
    for m in metas:
        try:
            with open(m, encoding="utf-8") as f:
                d = _json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "schema_version" in d:
            meta = d
            break
    if meta is None:
        print("FAIL: no run meta with schema_version beside the CSV")
        return 1
    aa = (meta.get("run") or {}).get("attitude_axes") or {}
    tr = meta.get("trajectory") or {}
    pol = meta.get("policy") or {}
    ctl = (meta.get("controller") or {})
    aref = ctl.get("attitude_ref") or {}
    print(f"[policy/rp] meta schema {meta.get('schema_version')}: run.attitude_axes.enabled "
          f"{aa.get('enabled')}, transport {aa.get('transport')!r}, firmware "
          f"{aa.get('firmware_version')!r}, wire {aa.get('mavlink_wire_version')!r}; "
          f"trajectory.attitude_track {tr.get('attitude_track')}; policy.action_repr "
          f"{pol.get('action_repr')!r}; controller.attitude_ref.tracked "
          f"{aref.get('tracked')} source {aref.get('source')!r}; "
          f"controller.allocation.attitude {(ctl.get('allocation') or {}).get('attitude')}")
    if meta.get("schema_version") != 16:
        print("FAIL: meta schema_version is not 16")
        rc = 1
    if aa.get("enabled") is not True:
        print("FAIL: meta run.attitude_axes.enabled is not true")
        rc = 1
    # the artefact record (safety audit 2026-09-26): the parsed bench probe
    # pinned by sha1, no sign probe, the recorded opt-out, first-water caps
    if aa.get("caps_in_force") != "first_water" or aa.get("sign_proven") is not False:
        print(f"FAIL: meta run.attitude_axes.caps_in_force {aa.get('caps_in_force')!r} / "
              f"sign_proven {aa.get('sign_proven')!r} — expected first_water / False")
        rc = 1
    if not aa.get("probe_sha1") or aa.get("sign_probe_sha1") is not None \
            or aa.get("require_sign_probe") is not False:
        print(f"FAIL: meta run.attitude_axes probe_sha1 {aa.get('probe_sha1')!r}, "
              f"sign_probe_sha1 {aa.get('sign_probe_sha1')!r}, require_sign_probe "
              f"{aa.get('require_sign_probe')!r}")
        rc = 1
    if [aa.get("cap_roll"), aa.get("cap_pitch")] != [0.1, 0.15]:
        print(f"FAIL: meta run.attitude_axes cap_roll/cap_pitch "
              f"{aa.get('cap_roll')}/{aa.get('cap_pitch')} are not first_water_caps")
        rc = 1
    if tr.get("attitude_track") is not True:
        print("FAIL: meta trajectory.attitude_track is not true")
        rc = 1
    if pol.get("action_repr") != "pos_rpy_width":
        print("FAIL: meta policy.action_repr is not pos_rpy_width")
        rc = 1
    if aref and aref.get("tracked") is not True:
        print("FAIL: meta controller.attitude_ref.tracked is not true")
        rc = 1
    return rc


def main() -> int:
    global SHAPE, RP
    for a in sys.argv[1:]:
        if a in ("mpc", "dobmpc", "mpc_tuned", "dobmpc_tuned", "pid", "rl"):
            # every follower the policy/replay stream accepts (the
            # contouring pair mpcc/dobmpcc is refused by design)
            Opts.mpc_mode = a
        elif a in SHAPES:
            SHAPE = a
        elif a in ("still", "drift", "orbit"):
            Opts.demo_object = a
        elif a == "rp":
            RP = True
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
    if RP and SHAPE != "policy":
        print("FAIL: `rp` (the 6-DoF variant row) is a policy-only argument")
        return 2
    if SHAPE == "policy":
        Opts.policy = True
        Opts.policy_ckpt = "stub"
        if RP:
            # THE VARIANT ROW: 7-dim stub with a pitch ramp, a temp config
            # with the variant ON, plain mpc (the attitude gate refuses the
            # dobmpc family without dobmpc_allowed, and pid/rl/mpcc have no
            # K/M path at all).
            Opts.policy_ckpt = "stub_rp"
            if Opts.mpc_mode not in ("mpc", "mpc_tuned"):
                print(f"note: attitude axes are refused under {Opts.mpc_mode} "
                      f"(engage.attitude_axes.dobmpc_allowed false) — flying mpc")
                Opts.mpc_mode = "mpc"
            Opts.mpc_config = _variant_config()
            print(f"variant config (synthetic, temp): {Opts.mpc_config}")
        try:
            import rov_gui.backends.policy  # noqa: F401  (the REAL worker)
        except ImportError as e:
            print(f"FAIL: rov_gui/backends/policy.py is not available ({e}) "
                  f"— the policy e2e drives the real PolicyWorker and does "
                  f"not stub it")
            return 2
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(Opts())
    backend = make_backend("demo", win.bus, win.mailboxes, Opts())
    statuses = []
    win.bus.mpc_status.connect(statuses.append)
    win.attach(backend)
    objects = []
    win.bus.object_fix.connect(objects.append)
    plans, pstates, pstatuses, pviz = [], [], [], []
    win.bus.policy_plan.connect(plans.append)
    win.bus.policy_plan_viz.connect(pviz.append)
    win.bus.policy_state.connect(pstates.append)
    win.bus.policy_status.connect(pstatuses.append)
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
        elif state["phase"] == "traj" and SHAPE == "policy":
            mpc = backend.mpc
            if plans and state.get("first_plan_dt") is None:
                state["first_plan_dt"] = t - state["traj_started"]
            if s is not None and not s.engaged:
                print("FAIL: disengaged during the policy run:", s.reason)
                app.quit()
            elif s is not None and not s.traj_on:
                # Ended by itself: the report reads end_reason and fails it.
                state["phase"] = "stopped"
                state["t_stop"] = t
            elif t > state["traj_started"] + POLICY_FLY_S:
                # STOP TRAJ from the driver, with plans in flight; the
                # counters after this must not move.
                state["n_installed_at_stop"] = mpc.replay["installed"]
                state["n_plans_at_stop"] = len(plans)
                win.bus.cmd_mpc_traj.emit(False)
                state["phase"] = "stopped"
                state["t_stop"] = t
        elif state["phase"] == "stopped":
            if t > state["t_stop"] + 1.5:
                mpc = backend.mpc
                last = mpc._replay_last or {}
                state["installed_after_stop"] = (
                    last.get("installed", 0)
                    - state.get("n_installed_at_stop", last.get("installed", 0)))
                state["plans_after_stop"] = (len(plans)
                                             - state.get("n_plans_at_stop",
                                                         len(plans)))
                state["end_reason"] = last.get("end_reason", "")
                state["csv"] = str(mpc._csv_path) if mpc._csv_path else None
                state["phase"] = "done"
                win.bus.cmd_mpc_engage.emit(False)
                QTimer.singleShot(800, app.quit)
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
    if SHAPE == "policy":
        rc |= _report_policy(backend, plans, pstates, pstatuses, state)
        if RP:
            rc |= _report_policy_rp(statuses, pviz, state)
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
