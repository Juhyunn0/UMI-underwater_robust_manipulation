#!/usr/bin/env python3
"""test_logfilter.py — what the MISSION LOG panel draws (2026-10-28), offline.

Source: the operator's 2026-10-28 session (15:09 startup, 15:24 / 15:26 DVL
lock loss while disengaged): 86 startup lines on the panel, the DVL
temperature lost among them (data/20261028/1028_152309/mission_log.txt). Decision: the panel draws a whitelist
(rov_gui/logfilter.py); every line still reaches mission_log.txt.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_logfilter.py -q -p no:cacheprovider

The lines below are that session's, verbatim up to truncation. A reworded
emitter is NOT caught here (the strings are literals) — that is the price of
matching on wording, the DVP_QUIET_WARN precedent.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.logfilter import mission_shown, panel_text
from rov_gui.state import BANNER_LINE

SHOWN = [
    ("warn", "ARMED (MANUAL)"),
    ("info", "DISARMED (MANUAL)"),
    ("warn", "ARM ignored — COMMAND ENABLE is off"),
    ("warn", "vehicle: Compass performance degraded"),
    ("warn", "dvl: HOT 45.0 °C — out of the water? Submerge it, or turn its "
             "acoustics OFF (web GUI http://192.168.2.95); the vendor shuts "
             "it down at 55 °C"),
    ("warn", "dvl: the DVL itself reports HIGH TEMPERATURE (status bit 0) — "
             "its thermal shutdown comes next"),
    ("info", "dvl: the DVL's high-temperature flag cleared"),
    ("info", "dvl: temperature back to 44.0 °C"),
    ("warn", "ctrl[dobmpc]: DVL IN by itself — --dvl_pose trust mode: no "
             "proving pass"),
    ("info", "policy: READY after 3.8 s (obs_dt 66.7 ms, fps source: ...)"),
    ("warn", "ctrl[dobmpc]: engage refused: flight mode is STABILIZE"),
    ("warn", "ctrl[pid]: START TRAJ refused — no tag fix"),
    ("warn", "ctrl[dobmpc]: TAG FIX ON refused: no tags in view"),
    ("warn", "ctrl: mode change refused while engaged"),
    ("error", "E-STOP pressed — axes zeroed, commands disabled"),
    ("error", "dvl_pose: FAULT — dvl pose stale: no usable DVL for 1.53s"),
]

HIDDEN = [
    # startup configuration
    ("info", "backend hw started"),
    ("info", "c3: target 192.168.2.191 streams=['color', 'left', 'right'] — "
             "71 Mbit/s"),
    ("warn", "command sink OPEN on udpin:0.0.0.0:14552 (sysid 255) — it "
             "transmits only while COMMAND ENABLE is on"),
    ("warn", "dvl: built (192.168.2.95:16171), switch OUT (shadow); mount "
             "rpy [0.0, 0.0, 0.0] deg"),
    ("warn", "dvl_pose: DVL dead-reckoned pose, heading raw_gyro"),
    ("warn", "ctrl[dobmpc]: --policy-dr with NO blind-flight bound"),
    ("info", "pid: heave trim f_z -12.00 N (config vehicle_net_buoyancy_n; "
             "the integral carries the rest, i_max 5 N)"),
    ("info", "ctrl[dobmpc]: RL_7D unavailable (ValueError: rl_7d: no "
             "policy_dir — ...) — rl_7d mode will refuse"),
    ("info", "ctrl[dobmpc]: RL_PWM controller unavailable (ValueError: "
             "rl_pwm: the config cap ...) — rl_pwm mode will refuse"),
    ("warn", "policy: OBS VIEW SHIFT +0.050 m along body x"),
    ("info", "policy: DP policy ready in 3.8 s (warm-up 225 ms, 106 tokens)"),
    ("info", "button 0 = servo_1_max_momentary — gripper_open ok"),
    ("info", "mpc solver: acados, probe median 2.9 ms of 5"),
    ("warn", "requested flight mode MANUAL"),
    ("warn", "sent ARM to sysid 1 — watch MOTORS for the vehicle's own answer"),
    ("warn", "deadman: no input for 500 ms — sending neutral (held drives "
             "released)"),
    # the DVL's first reading: the SENSORS row carries the number
    ("info", "dvl: temperature 45.0 °C (HOT at 45 °C; the vendor shuts the "
             "DVL down at 55 °C)"),
    # the camera, at the operator's request
    ("warn", "c3: DeviceBusyError: could not open 192.168.2.191: Cannot find "
             "any device with given deviceInfo"),
    ("info", "c3: connected"),
    ("warn", "fstereo: no mono pair yet — this run has no other depth source"),
    # 15:24 / 15:26: the DVL doubting its lock, disengaged
    ("warn", "ctrl[dobmpc]: velocity fading to the tag finite difference (DVL "
             "quality flag: fom (6.1 > 3.0 mm/s)) — the DVL returns by itself"),
    ("warn", "dvl_pose: pose: DVL HELD (no usable sample)"),
    ("info", "dvl_pose: pose: DVL dead reckoning"),
    ("warn", "dvl_pose: pose: LOST — dvl pose stale: no usable DVL for 1.53s "
             "(no_lock)"),
    ("warn", "dvl_pose: (disengaged) a START now would be refused — dvl pose: "
             "26 cm of unintegrated motion since the anchor"),
    ("info", "ctrl[dobmpc]: velocity source -> DVL"),
    # a recording's other files
    ("info", "rov sensor log closed: 18027 records"),
    ("info", "mpc csv closed: 1532 rows (data/20261028/1028_152309/x_mpc.csv)"),
    ("info", "recording to data/20261028/1028_152309/ui_152309.mp4"),
    ("debug", "anything at all"),
]


def test_the_session_lines_are_split_as_agreed():
    for level, msg in SHOWN:
        assert panel_text(level, msg) == msg, (level, msg)
    for level, msg in HIDDEN:
        assert panel_text(level, msg) is None, (level, msg)


def test_a_recording_stop_is_shown_as_its_folder():
    run = "data/20261028/1028_152309"
    for msg in (f"C3 FS Depth recording saved: {run}/c3_fs_depth_20261028_"
                f"152309.mp4",
                f"nav recording saved: {run}/nav_152310 (1360 fixes, 1363 "
                f"frames, 26097 detections, 2459 fallback-feed frames)",
                f"recording saved: {run}/ui_152309.mp4"):
        assert panel_text("info", msg) == f"recording saved: {run}/", msg
    # a stop with no file says what it says
    assert panel_text("info", "recording saved: None") == \
        "recording saved: None"


def test_mission_lines_hide_the_session_and_the_estimator_state():
    for msg in ("ready: dobmpc, geometric-path/corner-gate, no geofence, "
                "mission=station @tag37",
                "station_bridge ON: hold every axis 3.0s (xy=imu, calib "
                "687c9ed7), then release x/y/yaw and keep depth+attitude",
                "POSE: DVL dead reckoning (--dvl_pose control, TRUST)",
                "ESTIMATOR EKF (--ekf control)",
                "imu_dr ANCHORED (settle speed 0.4 cm/s, ...)",
                "station_bridge: fix back after 0.84s (tier imu)",
                "dvl_pose: pose: DVL dead reckoning",
                "dvl_pose: the DVL pose was 3.1 cm / 0.4 deg / 0.2 cm z off "
                "the tags",
                "dvl: TRIPPED (latched) — DVL disagreed with the tag velocity",
                f"{BANNER_LINE} (1 'ArduSub V' banner since the last line)",
                "MPC_Tuned profile 20261004_default"):
        assert not mission_shown(msg), msg
    for msg in ("ENGAGED (dobmpc) — holding (0.12, 0.40, 0.55)",
                "SQUARE requested @ tag 37, 1.00 m 90 deg x5",
                "going to tag 37 (1.24 m)",
                "DISENGAGED: released",
                "POLICY requested",
                "TAG FIX ONCE",
                "w_hat railed: DOB diverged"):
        assert mission_shown(msg), msg


def test_the_rl_policy_pick_lines_are_split_as_agreed():
    """The RL policy picker (2026-11-01, MpcWorker.set_rl_policy / setup / the command sink's _setup_pwm_gate). Not a
    recorded session: the emitters' formats, literally (test_rl_policy_pick.py runs the real emitters through the
    same filter). A REFUSED pick is an operator action refused — shown; the pick itself is a mission event the LOW
    row already shows (MISSION_QUIET "RL policy "); the launch provenance and the sink gate's export are
    configuration — file only."""
    for level, msg in (
            ("warn", "ctrl[rl_pwm_7d]: RL policy refused: RL_PWM_7D: not while engaged"),
            ("warn", "ctrl[dobmpc]: RL policy refused: RL_PWM: pwm10_s1 is no-fly — flight ban — the pwm "
                     "pool-failure policy (RL_controller README §13.8 'Never fly again, at any cap'; "
                     "config/hw_mpc.yaml rl_pwm comment)"),
            ("warn", "ctrl[rl_pwm_7d]: RL policy refused: RL_PWM_7D: the thruster sink's gate was built at launch "
                     "from rl_pwm.policy_dir rov_gui/control/rl_policies/pwm11k0.2_s0 (pwm_cap 0.25 ≠ this export's "
                     "0.5) — pick an export with the same pwm_cap, or set rl_pwm.policy_dir in hw_mpc.yaml and "
                     "restart")):
        assert panel_text(level, msg) == msg, (level, msg)
    for level, msg in (
            ("info", "ctrl[dobmpc]: RL policy RL_PWM_7D pwm11att14Rw_s1 (hw_mpc.yaml rl_pwm_7d.policy_dir, id "
                     "d04fb269)"),
            ("info", "rl_pwm transport gate: export rov_gui/control/rl_policies/pwm11k0.2_s0 (id 7e054a14), cap "
                     "0.25"),
            ("info", "ctrl[rl_pwm_7d]: RL policy RL_PWM_7D: pwm16N_s3 (id 98f42c43)"),
            ("warn", "ctrl[rl_pwm_7d]: RL policy RL_PWM_7D: pwm11att9X_s0 (id 777761e2) — WARN: flies at cap 0.25 "
                     "below trained 0.5 (allow_caps_below_trained)")):
        assert panel_text(level, msg) is None, (level, msg)
    assert not mission_shown("RL policy RL_PWM_7D ← pwm16N_s3 (panel pick, id 98f42c43, "
                             "rov_gui/control/rl_policies/pwm16N_s3)")
    assert mission_shown("RL_PWM gate lost — disengaged"), "the transport's own run events stay on the panel"


def test_a_launch_export_on_the_flight_ban_list_reaches_the_panel():
    """Review F1 (2026-11-02): the ban is enforced at the pick only, so a hw_mpc.yaml that names a banned export still
    builds at launch, and its provenance line then ends "— WARN: no-fly — <reason>" (workers._rl_launch_prov). That
    WARN is shown (PANEL_LOG, matched from the line's head) — the id may read "? — not hashed: …" when launch hashing
    failed; while the ordinary launch provenance line, the sink gate's info line, a pick's success lines and the pick
    event stay off the panel. Not a recorded session: the emitter's format, literally (test_rl_policy_pick.py runs
    the real emitter through the same filter); ids are the station copies' rl_library.export_id."""
    for level, msg in (
            ("warn", "ctrl[dobmpc]: RL policy RL_PWM_7D att7pwm_s0 (hw_mpc.yaml rl_pwm_7d.policy_dir, id 3d324330) — "
                     "WARN: no-fly — flight ban — two pool runs failed (co-contraction 26–44 A, sink) (RL_controller "
                     "README §13.8; config/hw_mpc.yaml rl_pwm_7d comment)"),
            ("warn", "ctrl: RL policy RL_PWM pwm10_s1 (hw_mpc.yaml rl_pwm.policy_dir, id ? — not hashed: "
                     "PermissionError: [Errno 13] Permission denied: '/x/obs_spec.json') — WARN: no-fly — flight ban "
                     "— the pwm pool-failure policy (RL_controller README §13.8 'Never fly again, at any cap'; "
                     "config/hw_mpc.yaml rl_pwm comment)")):
        assert panel_text(level, msg) == msg, (level, msg)
    for level, msg in (
            ("info", "ctrl[dobmpc]: RL policy RL_PWM_7D pwm11att14Rw_s1 (hw_mpc.yaml rl_pwm_7d.policy_dir, id "
                     "d04fb269)"),
            ("info", "ctrl[dobmpc]: RL policy RL traj5_s2 (hw_mpc.yaml rl.policy_dir, id ? — not hashed: "
                     "PermissionError: [Errno 13] Permission denied: '/x/obs_spec.json')"),
            ("info", "rl_pwm transport gate: export rov_gui/control/rl_policies/pwm11k0.2_s0 (id 7e054a14), cap "
                     "0.25"),
            ("info", "ctrl[rl_pwm_7d]: RL policy RL_PWM_7D: pwm16N_s3 (id 98f42c43)"),
            ("warn", "ctrl[rl_pwm_7d]: RL policy RL_PWM_7D: pwm11att9X_s0 (id 777761e2) — WARN: flies at cap 0.25 "
                     "below trained 0.5 (allow_caps_below_trained)")):
        assert panel_text(level, msg) is None, (level, msg)
    assert not mission_shown("RL policy RL_PWM_7D ← pwm16N_s3 (panel pick, id 98f42c43, "
                             "rov_gui/control/rl_policies/pwm16N_s3)")


def test_the_rl_pwm_cap_lines_are_split_as_agreed():
    """The LOW row's cap ▾ (2026-11-02, MpcWorker.set_rl_pwm_cap; the cap SPEC's C9). Not a recorded session: the
    emitters' formats, literally (test_rl_pwm_cap.py runs the real emitters through the same filter). A REFUSED change
    is an operator action refused — its head says so ("ctrl[…]: RL_PWM cap refused: …"), PANEL_LOG's refusal pattern
    shows it with no pattern of its own. The change itself is a mission event the combo already shows — MISSION_QUIET
    "RL_PWM cap " — and its summary line is configuration. A pwm LOW the change DROPPED ("ctrl[…]: <LOW> unavailable at
    cap …") is file only: the LOW row shows its ERROR. The quiet prefix is "RL_PWM cap ", NOT "RL_PWM ": the
    transport's own run event "RL_PWM gate lost — disengaged" stays on the panel."""
    for level, msg in (
            ("warn", "ctrl[rl_pwm_7d]: RL_PWM cap refused: not while engaged"),
            ("warn", "ctrl: RL_PWM cap refused: controller still building — pick again when READY"),
            ("warn", "ctrl[dobmpc]: RL_PWM cap refused: the cap applies to RL_PWM / RL_PWM_7D only (asked for 'rl')"),
            ("warn", "ctrl[dobmpc]: RL_PWM cap refused: cap must be a number in (0, 1], got nan"),
            ("warn", "ctrl[rl_pwm_7d]: RL_PWM cap refused: the thruster sink's export "
                     "rov_gui/control/rl_policies/pwm11k0.2_s0 is trained at cap 0.5: the transport cannot exceed "
                     "it"),
            ("warn", "ctrl[dobmpc]: RL_PWM cap refused: RL_PWM cannot fly pwm11k0.2_s0 at cap 0.25: ValueError: "
                     "rl_pwm: the config cap rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50 "
                     "(rov_gui/control/rl_policies/pwm11k0.2_s0) — a policy capped below its training is fed back a "
                     "previous action it never saw and limit-cycles (rl_pwm 2026-10-01: cap 0.25 on a cap-1.0 "
                     "policy); raise rl_pwm.pwm_cap to >= 0.50, or set rl_pwm.allow_caps_below_trained: true "
                     "(recorded in meta)"),
            ("warn", "ctrl[rl_pwm_7d]: RL_PWM cap refused: the thruster sink's gate was built at launch from "
                     "rl_pwm.policy_dir rov_gui/control/rl_policies/pwm11k0.2_s0 (motor_direction [1, 1, 1, 1, 1, 1, "
                     "1, 1] ≠ this export's [-1, -1, 1, 1, -1, 1, 1, -1]) — pick an export with the same "
                     "motor_direction, or set rl_pwm.policy_dir in hw_mpc.yaml and restart")):
        assert panel_text(level, msg) == msg, (level, msg)
    for level, msg in (
            ("warn", "ctrl[rl_pwm_7d]: RL_PWM unavailable at cap 0.25 — ValueError: rl_pwm: the config cap "
                     "rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50 "
                     "(rov_gui/control/rl_policies/pwm11k0.2_s0) — a policy capped below its training is fed back a "
                     "previous action it never saw and limit-cycles (rl_pwm 2026-10-01: cap 0.25 on a cap-1.0 "
                     "policy); raise rl_pwm.pwm_cap to >= 0.50, or set rl_pwm.allow_caps_below_trained: true "
                     "(recorded in meta)"),
            ("warn", "ctrl[dobmpc]: RL_PWM_7D unavailable at cap 0.5 — export files changed while loading"),
            ("info", "ctrl[dobmpc]: RL_PWM cap 0.25 → 0.5 (panel): RL_PWM pwm11k0.2_s0 flies 0.5 (trained 0.5); "
                     "RL_PWM_7D pwm11att14Rw_s1 flies 0.5 (trained 0.5)"),
            ("warn", "ctrl[rl_pwm_7d]: RL_PWM cap 0.5 → 0.25 (panel): RL_PWM → ERROR (ValueError: rl_pwm: the "
                     "config cap rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50 "
                     "(rov_gui/control/rl_policies/pwm11k0.2_s0) — ...); RL_PWM_7D pwm11att14Rw_s1 flies 0.25 "
                     "(trained 0.5)")):
        assert panel_text(level, msg) is None, (level, msg)
    for ev in ("RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5, RL_PWM_7D 0.5)",
               "RL_PWM cap 0.5 → 0.25 (panel; RL_PWM ERROR, RL_PWM_7D 0.25)",
               "RL_PWM cap 0.25 → 1 (panel; RL_PWM_7D 0.5)"):
        assert not mission_shown(ev), ev
    assert mission_shown("RL_PWM gate lost — disengaged"), "NOT 'RL_PWM ' alone: the transport's run events stay"


def test_the_panel_records_every_line_but_draws_the_shown_ones():
    from rov_gui.tests.test_offline import _app
    from rov_gui.widgets.payload import PayloadPanel

    _app()
    p = PayloadPanel()
    run = "data/20261028/1028_152309"
    p.add_event("ARMED (MANUAL)", "warn")
    p.add_event("rov sensor log closed: 18027 records", show=False)
    p.add_event(f"C3 FS Depth recording saved: {run}/a.mp4",
                shown_as=f"recording saved: {run}/")
    p.add_event("dvl raw log closed: 1166 records", show=False)
    p.add_event(f"nav recording saved: {run}/nav_152310 (1360 fixes)",
                shown_as=f"recording saved: {run}/")
    drawn = p.log_view.toPlainText().strip().split("\n")
    assert len(p._log_lines) == 5, p._log_lines
    assert len(drawn) == 2, drawn                    # ARMED + one folder
    assert drawn[1].endswith(f"recording saved: {run}/"), drawn
    assert "nav_152310" in p.log_text() and "dvl raw log" in p.log_text()
    # a repeat rewrites only a block that is on screen
    for _ in range(3):
        p.add_event("dvl_pose: pose: DVL HELD", "warn", show=False)
    assert "(x3)" in p._log_lines[-1]
    assert len(p.log_view.toPlainText().strip().split("\n")) == 2
    for _ in range(2):
        p.add_event("engage refused: no tag fix", "warn")
    drawn = p.log_view.toPlainText().strip().split("\n")
    assert len(drawn) == 3 and "(x2)" in drawn[-1], drawn
    assert drawn[-1] == p._log_lines[-1]
    p.close()


def test_the_window_routes_through_the_filter():
    import tempfile
    from pathlib import Path

    from rov_gui import theme
    from rov_gui.tests.test_offline import Opts, _app
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    win = MainWindow(Opts())
    win.bus.log.emit("info", "mpc solver: acados, probe median 2.9 ms of 5")
    win.bus.log.emit("warn", "dvl: HOT 45.0 °C — out of the water?")
    win.bus.mpc_event.emit("ready: dobmpc, mission=station @tag37")
    win.bus.mpc_event.emit("ENGAGED (dobmpc) — holding")
    drawn = win.payload.log_view.toPlainText()
    assert "mpc solver" not in drawn and "ready: dobmpc" not in drawn
    assert "[WARN] dvl: HOT 45.0" in drawn and "ENGAGED (dobmpc)" in drawn
    with tempfile.TemporaryDirectory() as td:
        win._save_mission_log(td)
        body = (Path(td) / "mission_log.txt").read_text()
    for s in ("mpc solver", "[WARN] dvl: HOT", "ready: dobmpc", "ENGAGED"):
        assert s in body, s
    win.shutdown()
