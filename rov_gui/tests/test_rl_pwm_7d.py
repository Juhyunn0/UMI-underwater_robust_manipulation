#!/usr/bin/env python3
"""LOW mode rl_pwm_7d (2026-10-03): the per-thruster RL policy trained WITH a roll/pitch reference (RL_controller
att7pwm, still water). Roll/pitch go out as PULSES through the rl_pwm transport, never as s/t.

Pinned here:
  * vocabulary: in MODES / MODE_LABELS / the CLI, NOT in LOW_7D_BASE (no s/t gates), in LOW_RP_TRACK_BASE;
  * config: strict `rl_pwm_7d:` block, no pwm_cap key (the transport's single cap is rl_pwm.pwm_cap);
  * HwRlPwm7d: fail-closed build (no export / level-trained / axes6 / layout / envelope / cap below trained), the cap read
    from rl_pwm.pwm_cap, q_des = the RL_controller convention clipped to the envelope, level = HwRlPwm's q_des bitwise,
    non-finite -> neutral pulses + n_fail, n_fail reset;
  * the worker: engages WITHOUT engage.attitude_axes and never reaches the s/t gates; pulse bit + rp_track_live; the
    tilt ceiling (ii) disengages rl_pwm_7d AND plain rl_pwm (2026-10-03: rl_pwm had none) while a 4-DoF pid does not;
    the s/t degrade check (v) does not apply; a 7-dim policy TRACKS on rl_pwm_7d and is dropped-and-logged on rl_pwm;
    CSV rp_track 2 / ax nan; meta rp_transport pwm_pulses, the LOW stamped, the coast sentence corrected.

Run: QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest rov_gui/tests/test_rl_pwm_7d.py -q
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rov_gui.control.geometry import LOW_7D_BASE, LOW_7D_PULSE_BASE, LOW_RP_TRACK_BASE, MpcConfig
from rov_gui.control.path_geometry import NedPlan
from rov_gui.control.rl_pwm import NEUTRAL_PULSES
from rov_gui.state import ACTION_REPR_POS_RPY_WIDTH, Conn, Telemetry, now

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PWM10 = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
ENVELOPE = {"trained": True, "roll_max_rad": 0.17, "pitch_max_rad": 0.35, "rate_max_rad_s": 0.35, "p_att": 0.7}


def _export(tmp, attitude_reference=ENVELOPE, **spec_over) -> Path:
    """A FORMAT export: pwm10_s1's real weights + spec with an attitude_reference injected (never a trained att policy)."""
    d = Path(tmp) / "rlpwm7d_export"
    d.mkdir(exist_ok=True)
    shutil.copy(PWM10 / "policy_weights.npz", d / "policy_weights.npz")
    spec = json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))
    if attitude_reference is not None:
        spec["attitude_reference"] = dict(attitude_reference)
    spec.update(spec_over)
    (d / "obs_spec.json").write_text(json.dumps(spec))
    return d


def _cfg(tmp, cap=1.0, **over) -> MpcConfig:
    cfg = MpcConfig()
    # 2026-10-10: the transport block opts out of the sign-probe gate (offline tests, recorded) and mirrors the cap
    # opt-out for the PLAIN follower (HwRlPwm reads rl_pwm.allow_caps_below_trained; HwRlPwm7d its own block's)
    cfg.rl_pwm = {"pwm_cap": cap, "require_thruster_probe": False,
                  "allow_caps_below_trained": bool(over.get("allow_caps_below_trained", False))}
    cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp)), **over}
    return cfg


def _settle(o, n=60):
    """Advance the flown reference to its target (the trained goal rate moves it 0.35 * 0.05 rad per tick)."""
    for _ in range(n):
        o._advance_rp_flown()


def _plan1(rp=None, p=(0.0, 0.0, 0.8), yaw=0.0):
    kw = {} if rp is None else {"rp_ned": np.asarray(rp, float).reshape(2, 1)}
    return NedPlan(p_ned=np.asarray(p, float).reshape(3, 1), yaw_ned=np.array([yaw]), v_ned=np.zeros((3, 1)),
                   r_ned=np.zeros(1), psi_path=np.zeros(1), **kw)


# =============================================================================
# vocabulary + config
# =============================================================================
def test_the_pulse_low_is_spelled_everywhere_and_kept_out_of_the_st_set():
    from rov_gui.control.workers import MpcWorker
    from rov_gui.widgets.trajectory import MODE_LABELS
    assert LOW_7D_PULSE_BASE == {"rl_pwm_7d": "rl_pwm"}
    assert "rl_pwm_7d" in MpcWorker.MODES and MODE_LABELS["rl_pwm_7d"] == "RL_PWM_7D"
    assert "rl_pwm_7d" not in LOW_7D_BASE and "rl_pwm_7d" not in MpcWorker.ATTITUDE_LOW_ALLOWED
    assert LOW_RP_TRACK_BASE == {**LOW_7D_BASE, "rl_pwm_7d": "rl_pwm"}
    assert '"rl_pwm_7d"' in (ROOT / "rov_gui" / "__main__.py").read_text(encoding="utf-8")


def _block_span(src: str, name: str) -> tuple[int, int]:
    """(start, end) of the top-level YAML block `name:` — header line to the next top-level line."""
    import re
    m = re.search(rf"^{name}:[ \t]*\n", src, flags=re.M)
    assert m, name
    nxt = re.search(r"^\S", src[m.end():], flags=re.M)
    return m.start(), (m.end() + nxt.start()) if nxt else len(src)


def _edit_block(src: str, name: str, fn) -> str:
    a, b = _block_span(src, name)
    return src[:a] + fn(src[a:b]) + src[b:]


def test_config_block_is_strict_and_has_no_cap_of_its_own():
    """Robust to the operator ENABLING the candidate (review 2026-10-03): policy_dir is null or an existing export."""
    import re
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")
    shipped = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    pd = shipped.rl_pwm_7d["policy_dir"]
    assert pd is None or (ROOT / str(pd)).is_dir() or Path(str(pd)).is_dir(), pd
    assert isinstance(shipped.rl_pwm_7d["allow_caps_below_trained"], bool)

    def load(text):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.yaml"
            f.write_text(text)
            return MpcConfig.load(f)

    def add(line):
        return lambda blk: re.sub(r"^(rl_pwm_7d:[ \t]*\n)", rf"\g<1>{line}\n", blk, count=1, flags=re.M)

    with pytest.raises(ValueError, match="unknown key"):
        load(_edit_block(src, "rl_pwm_7d", add("  leash: 1.0")))
    with pytest.raises(ValueError, match="set rl_pwm.pwm_cap"):
        load(_edit_block(src, "rl_pwm_7d", add("  pwm_cap: 0.5")))
    with pytest.raises(ValueError, match="true/false"):
        load(_edit_block(src, "rl_pwm_7d", lambda blk: re.sub(r"^(  allow_caps_below_trained:)[ \t]*\S+",
                                                               r"\g<1> 1", blk, count=1, flags=re.M)))
    assert load(re.sub(r"^mode: \w+", "mode: rl_pwm_7d", src, count=1, flags=re.M)).mode == "rl_pwm_7d"


# =============================================================================
# HwRlPwm7d
# =============================================================================
def test_build_refuses_everything_but_a_roll_pitch_trained_pwm_export():
    from rov_gui.control.rl_policy import HwRlPwm7d
    with pytest.raises(ValueError, match="no policy_dir"):
        HwRlPwm7d(MpcConfig(), log=lambda m: None)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MpcConfig()
        cfg.rl_pwm = {"pwm_cap": 1.0}
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp, attitude_reference=None))}
        with pytest.raises(ValueError, match="LEVEL goal"):
            HwRlPwm7d(cfg, log=lambda m: None)                  # pwm10_s1 as shipped: level-trained
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp, action_mode="axes6"))}
        with pytest.raises(ValueError, match="'axes6' policy"):
            HwRlPwm7d(cfg, log=lambda m: None)
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp, frame_layout=["q_des(w,x,y,z)"]))}
        with pytest.raises(ValueError, match="frame_layout"):
            HwRlPwm7d(cfg, log=lambda m: None)
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp, attitude_reference={"trained": True}))}
        with pytest.raises(ValueError, match="envelope is unknown"):
            HwRlPwm7d(cfg, log=lambda m: None)
        no_rate = {k: v for k, v in ENVELOPE.items() if k != "rate_max_rad_s"}
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp, attitude_reference=no_rate))}
        with pytest.raises(ValueError, match="rate_max_rad_s"):
            HwRlPwm7d(cfg, log=lambda m: None)                  # the trained goal rate is required (review 2026-10-03)
        cfg.rl_pwm = {"pwm_cap": 0.25}                          # the transport cap below the trained 1.0
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp))}
        with pytest.raises(ValueError, match="below the policy's trained"):
            HwRlPwm7d(cfg, log=lambda m: None)
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp)), "allow_caps_below_trained": True}
        r = HwRlPwm7d(cfg, log=lambda m: None)
        assert r.pwm_cap == 0.25 and r.caps_below_trained is True
        m = r.meta()
        assert m["type"] == "rl_pwm_7d" and m["caps_below_trained"] is True and m["allocation"]["via"] == "per_thruster_pulses"
        json.dumps(m)


def test_the_cap_is_the_transports_for_both_pwm_followers():
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp, cap=0.5, allow_caps_below_trained=True)
        assert HwRlPwm7d(cfg, log=lambda m: None).pwm_cap == 0.5
        assert HwRlPwm(cfg, log=lambda m: None).pwm_cap == 0.5


def test_q_des_is_the_training_convention_clipped_and_level_is_bitwise_rl_pwm():
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d, _unique, quat_from_euler_xyz
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        r7, r = HwRlPwm7d(cfg, log=lambda m: None), HwRlPwm(cfg, log=lambda m: None)
        for o in (r7, r):
            o.set_target_ned([0.0, 0.0, 0.8], 0.4)
        assert np.array_equal(r7._q_des(0.4), r._q_des(0.4))         # no rp ever: level, bit for bit
        eta = np.array([0.0, 0.0, 0.8, 0.02, -0.03, 0.4])
        assert np.array_equal(r7.observation(eta, np.zeros(6), 0.0), r.observation(eta, np.zeros(6), 0.0))
        r7.set_path_plan_ned(_plan1(rp=[0.1, 0.2], yaw=0.4))
        _settle(r7)
        assert np.allclose(_unique(r7._q_des(0.4)), _unique(quat_from_euler_xyz(0.1, -0.2, -0.4)))
        r7.set_path_plan_ned(_plan1(rp=[0.3, -0.45], yaw=0.0))
        _settle(r7)
        assert r7.ref_attitude_ned_at() == (pytest.approx(0.17), pytest.approx(-0.35))
        assert np.allclose(r7._q_des(0.0), quat_from_euler_xyz(0.17, 0.35, 0.0))
        # D5: the plan ends -> the reference ramps to level at pq_max per tick inside step
        r7.set_path_plan_ned(_plan1(rp=[0.0, 0.2]))
        _settle(r7)
        r7.set_target_ned([0.0, 0.0, 0.8], 0.0)
        r7.step(np.array([0.0, 0.0, 0.8, 0, 0, 0]), np.zeros(6), np.zeros(6), 0.0)
        assert abs(r7.ref_attitude_ned_at()[1] - (0.2 - 0.35 * 0.05)) < 1e-9


def test_the_flown_reference_never_moves_faster_than_the_trained_goal_rate():
    """Review 2026-10-03: training moved the goal at <= attitude_reference.rate_max_rad_s (goals.py att_rate_max); a plan
    step must reach q_des at that rate, starting level at engage, and what the worker reads after a tick (CSV rroll /
    rpitch, interlock (iii)) is what that tick flew."""
    from rov_gui.control.rl_policy import quat_from_euler_xyz
    from rov_gui.control.rl_policy import HwRlPwm7d
    with tempfile.TemporaryDirectory() as tmp:
        r7 = HwRlPwm7d(_cfg(tmp), log=lambda m: None)
        st = 0.35 * r7.dt
        r7.set_path_plan_ned(_plan1(rp=[0.1, 0.3]))
        eta = np.array([0.0, 0.0, 0.8, 0, 0, 0])
        r7.step(eta, np.zeros(6), np.zeros(6), 0.0)
        assert r7.ref_attitude_ned_at() == (pytest.approx(st), pytest.approx(st))
        assert np.allclose(r7.observation(eta, np.zeros(6), 0.0)[0:4],
                           quat_from_euler_xyz(st, -st, 0.0)) or np.allclose(
            -r7.observation(eta, np.zeros(6), 0.0)[0:4], quat_from_euler_xyz(st, -st, 0.0))
        prev = r7.ref_attitude_ned_at()
        for _ in range(30):
            r7.step(eta, np.zeros(6), np.zeros(6), 0.0)
            cur = r7.ref_attitude_ned_at()
            assert abs(cur[0] - prev[0]) <= st + 1e-12 and abs(cur[1] - prev[1]) <= st + 1e-12
            prev = cur
        assert prev == (pytest.approx(0.1), pytest.approx(0.3)) and r7._rp_rate_ticks == 17     # ceil(0.3 / st) - 1
        m = r7.meta()
        assert m["rp_rate_limit_ticks"] == 17 and m["rp_rate_max_rad_s"] == 0.35
        r7.reset()
        assert r7.ref_attitude_ned_at() == (0.0, 0.0) and r7._rp_rate_ticks == 0


def test_non_finite_is_neutral_pulses_and_n_fail_resets():
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(tmp)
        for cls in (HwRlPwm7d, HwRlPwm):
            r = cls(cfg, log=lambda m: None)
            r.set_target_ned([0.0, 0.0, 0.8], 0.0)
            _u, info = r.step(np.array([0.0, 0.0, np.nan, 0, 0, 0]), np.zeros(6), np.zeros(6), 0.0)
            assert tuple(info["pwm_us"]) == tuple(NEUTRAL_PULSES) and info["status"] == 1 and r.n_fail == 1
            r.reset()
            assert r.n_fail == 0, cls.__name__


# =============================================================================
# the worker
# =============================================================================
def _gate(cap=1.0, **kw):
    from rov_gui.tests.test_rl_pwm import MOT
    g = {"configured": True, "ok": True, "why": "ok", "stamp": now(), "active": False,
         "motor_direction": list(MOT), "pwm_cap": cap}
    g.update(kw)
    return g


def _feed(w, roll=0.0, pitch=0.0, **tel):
    from rov_gui.tests.test_control import _fix, _imu
    t = now()
    w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, t))
    w.on_vehicle_imu(_imu(roll=roll, pitch=pitch, depth=0.8, t=t))
    d = dict(armed=True, mode="MANUAL", conn=Conn.ONLINE, pwm_gate=_gate())
    d.update(tel)
    w.on_telemetry(Telemetry(**d))


def _pwm7d_worker(tmp, mode="rl_pwm_7d", capability=False):
    """A worker on the pwm transport (test sink gate in Telemetry), rl_pwm.pwm_cap 1.0 so the cap matches the trained
    one, the rl_pwm_7d follower built from a FORMAT export; engage.attitude_axes is OFF unless asked."""
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d
    from rov_gui.tests.test_control import _worker
    w, bus, pilots, logs = _worker(tmp)
    # 2026-10-10: the sign-probe gate is opted out (recorded) — no artefact on a test box; its own tests build one
    w.cfg.rl_pwm = {"pwm_cap": 1.0, "require_thruster_probe": False}
    w.cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp))}
    w._rl_pwm = HwRlPwm(w.cfg, log=lambda m: None)
    w._rl_pwm7d = HwRlPwm7d(w.cfg, log=lambda m: None)
    w.cfg.engage["attitude_axes"] = {"enabled": bool(capability)}
    pwms = []
    bus.cmd_pwm.connect(pwms.append)
    w.set_mode(mode)
    assert w.cfg.mode == mode, w.reason
    return w, bus, pilots, pwms, logs


def _engage(w, **tel):
    w.on_enable(True)
    _feed(w, **tel)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):
        _feed(w, **tel)
        w.tick()


def test_engages_without_the_st_capability_and_never_reaches_the_st_gates(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp)
        monkeypatch.setattr(w, "_attitude_refusal", lambda tel: (_ for _ in ()).throw(AssertionError("s/t gate")))
        _engage(w)
        assert w._attitude_axes is False and w._attitude_pulses is True and w.rp_track_live is True
        assert w.attitude_actuated is True and w.ctrl._attitude_axes is True     # told it drives roll/pitch
        assert any("TILT CEILING ARMED" in m and "TRACKED via q_des" in m for _l, m in logs), logs[-4:]
        aa = w._run_meta()["run"]["attitude_axes"]
        assert aa["rp_transport"] == "pwm_pulses" and aa["ceiling_armed"] is True and aa["sat_watch_armed"] is False
        assert aa["enabled"] is False and aa["rp_track_live"] is True and aa["rp_track_low"] == "rl_pwm_7d"
        assert w._run_meta()["controller"]["type"] == "rl_pwm_7d"
        w.disengage("test")
        assert w._attitude_pulses is False and w.rp_track_live is False and w.ctrl._attitude_axes is False
        w.teardown()


@pytest.mark.parametrize("mode", ["rl_pwm_7d", "rl_pwm"])
def test_the_tilt_ceiling_disengages_every_pwm_low(mode):
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp, mode=mode)
        _engage(w)
        for _ in range(4):                                   # 4 ticks over 35 deg: debounce holds
            _feed(w, roll=math.radians(40.0))
            w.tick()
        assert w.engaged, w.reason
        _feed(w, roll=math.radians(40.0))
        w.tick()
        assert not w.engaged and "attitude ceiling" in w.reason, w.reason
        assert pwms and pwms[-1].source in ("stop", "idle") and pwms[-1].pulses == NEUTRAL_PULSES
        w.teardown()


def test_a_4dof_pid_still_has_no_ceiling_and_the_st_degrade_check_skips_pulses():
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp, mode="pid")
        _engage(w)
        for _ in range(8):
            _feed(w, roll=math.radians(40.0))
            w.tick()
        assert w.engaged, w.reason                            # unchanged 4-DoF behaviour
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp)
        _engage(w)
        for _ in range(3):
            _feed(w, attitude_axes_degraded=True)            # an s/t sink fact: irrelevant to pulses
            w.tick()
        assert w.engaged, w.reason
        w.teardown()


def test_pulse_rows_are_rp_track_2_and_rl_pwm_rows_stay_4dof():
    from rov_gui.tests.test_control import _last_csv_row
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp)
        # 2026-10-10 (b): the shipped NULL interlock (engage.per_thruster.null_us 80 us / 0.5 s) is tuned to catch
        # pwm10_s1's n_h squeeze, and this FORMAT export carries pwm10_s1's weights, which emit null_h ~ +144 us on
        # the static synthetic frame — the engagement would end 0.5 s in, as designed (test_rl_pwm_interlocks pins
        # that). The ROWS are the subject here, so the monitor is widened for this test only.
        w.cfg.engage["per_thruster"]["null_us"] = 1000.0
        _engage(w)
        w.ctrl.set_path_plan_ned(_plan1(rp=[0.0, math.radians(8.0)], p=w._eta[:3], yaw=float(w._eta[5])))
        _feed(w)
        w.tick()
        row = _last_csv_row(w)
        assert row["rp_track"] == "2" and row["ax_roll"] == "nan" and row["ax_pitch"] == "nan"
        st = math.degrees(0.35 / w.cfg.ctrl_hz)                     # one tick at the trained goal rate
        assert abs(float(row["rpitch_deg"]) + st) < 2e-3             # FLU sign, the reference it FLIES (rate-limited)
        for _ in range(30):
            _feed(w)
            w.tick()
        assert abs(float(_last_csv_row(w)["rpitch_deg"]) + 8.0) < 2e-3
        # a tick whose policy pulses did NOT leave (the gate-grace window: eight 1500s) is not a tracked row
        _feed(w, pwm_gate=None)
        w.tick()
        assert w.engaged, w.reason
        assert pwms[-1].pulses == NEUTRAL_PULSES
        row = _last_csv_row(w)
        assert row["rp_track"] == "0" and row["rroll_deg"] == "nan" and row["rpitch_deg"] == "nan", row
        sb = w._run_meta()["station_bridge"]
        assert w._bridge is not None and "eight 1500 us" in sb["released_in_coast"]   # the pwm coast, not "attitude kept"
        w.teardown()
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _pwm7d_worker(tmp, mode="rl_pwm")
        _engage(w)
        _feed(w)
        w.tick()
        row = _last_csv_row(w)
        assert row["rp_track"] == "0" and row["rroll_deg"] == "nan"
        aa = w._run_meta()["run"]["attitude_axes"]
        assert aa["rp_transport"] == "pwm_pulses" and aa["ceiling_armed"] is True and aa["rp_track_live"] is False
        w.teardown()


@pytest.mark.parametrize("action_repr", [ACTION_REPR_POS_RPY_WIDTH, "auto"])
def test_a_7dim_policy_tracks_on_rl_pwm_7d_and_is_dropped_on_rl_pwm(action_repr):
    """...identically under policy.action_repr auto (2026-10-03, shipped): the 7-dim PICK alone, no YAML step, tracks
    on RL_PWM_7D (the shipped veto is true) — what the panel's WARN forecast and the chip's "7d rp TRACKED" say."""
    from rov_gui.tests.test_policy import _feed_status
    for mode, tracked in (("rl_pwm_7d", True), ("rl_pwm", False)):
        with tempfile.TemporaryDirectory() as tmp:
            w, bus, pilots, pwms, logs = _pwm7d_worker(tmp, mode=mode)
            w.cfg.policy.update({"action_repr": action_repr, "attitude_track": True})
            w.policy_present = True
            w.set_scenario({"shape": "policy"})
            _engage(w)
            for _ in range(2):
                _feed(w)
                _feed_status(w, action_repr=ACTION_REPR_POS_RPY_WIDTH)
                w.tick()
            w.set_traj(True)
            assert w.traj_on, (mode, w.reason)
            assert w.replay["attitude_track"] is tracked, mode
            m = w._run_meta()
            assert m["trajectory"]["attitude_track"] is tracked
            assert m["trajectory"]["attitude_track_low"] == mode
            assert m["trajectory"]["action_repr"] == ACTION_REPR_POS_RPY_WIDTH
            assert m["policy"]["config"]["action_repr"] == action_repr
            w.teardown()


def test_the_shipped_candidate_export_builds_and_matches_the_transport():
    """rl_policies/att7pwm_s0 (the real att7pwm export, shipped NOT enabled — the test does not pin that): it builds as rl_pwm_7d, its
    envelope is the trained one, and its motor directions are pwm10_s1's — so the sink gate built from rl_pwm's export
    accepts it. The spec is also pinned as fixtures/rlpwm7d_att7pwm_s0_obs_spec.json (the format the exporter writes)."""
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d
    cand = ROOT / "rov_gui" / "control" / "rl_policies" / "att7pwm_s0"
    spec = json.loads((cand / "obs_spec.json").read_text(encoding="utf-8"))
    assert spec == json.loads((HERE / "fixtures" / "rlpwm7d_att7pwm_s0_obs_spec.json").read_text(encoding="utf-8"))
    assert spec["attitude_reference"]["trained"] is True and spec["action_mode"] == "pwm"
    assert (spec.get("sea_state_training") or {}).get("p_mode", [1.0])[0] == 1.0      # still water
    cfg = MpcConfig()
    cfg.rl_pwm = {"pwm_cap": 1.0}
    cfg.rl_pwm_7d = {"policy_dir": str(cand)}
    r7 = HwRlPwm7d(cfg, log=lambda m: None)
    assert abs(r7.att_pitch_max - 0.35) < 1e-9 and abs(r7.att_roll_max - 0.17) < 1e-9
    assert r7.model.motor_direction == HwRlPwm(cfg, log=lambda m: None).model.motor_direction
    pd = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml").rl_pwm_7d["policy_dir"]
    assert pd is None or (ROOT / str(pd)).is_dir() or Path(str(pd)).is_dir(), pd   # shipped null; enabling is the operator's call
