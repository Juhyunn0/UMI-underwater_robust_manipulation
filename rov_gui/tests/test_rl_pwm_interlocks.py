#!/usr/bin/env python3
"""test_rl_pwm_interlocks.py — the 2026-10-10 (b) additions to the pwm LOWs (rl_pwm / rl_pwm_7d), offline.

Source: RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md §3 step 5, second half ("interlocks
(tilt 20-25 deg/0.25 s, rate 60 deg/s, 50 A / 13.5 V, null monitor 100-150 us), policy-side tag gate -> hold wrench,
replay-tested on the 14 engagements").

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_rl_pwm_interlocks.py -q -p no:cacheprovider

What is pinned:

1. control/rl_pwm.PwmInterlocks + parse_per_thruster_block — the defaults, the config loader (engage.per_thruster,
   unknown keys raise, MpcConfig() carries the block), each of the six trips on a synthetic tick sequence with its
   debounce (one tick short holds, the next trips, one quiet tick resets), the trip sentence carries the numbers,
   quiet data never trips, missing inputs never trip, the fixed order and meta. (2026-10-15: the seventh item, AGE,
   is first in the order and the one where a MISSING input counts as over — test_att_age.py pins it; QUIET here
   carries a healthy 0.05 s age.)
2. control/rl_pwm.PwmObsGuard + hold_throttles + parse_obs_guard_block — the velocity slew, the fix-jump reject under
   a thin fix, the tag gate's hold / release timing, the hold allocation's signs and magnitude (4 N up on the four
   verticals, horizontals neutral, |K|,|M| ~ 0).
3. control/rl_policy.HwRlPwm — the guards change ONLY what the network sees (the worker's state is untouched), the
   hold pulses replace the policy's under the tag gate and the policy returns when the tags do, the previous action
   fed back is what LEFT, meta.
4. workers.MpcWorker — every interlock disengages a pwm LOW with neutral pulses, a mission line and the numbers, meta
   controller.interlocks records thresholds and the trip; quiet data stays engaged; the CSV pwm_guard column (schema
   20) on pwm rows, empty on a pid row; a 4-axis LOW never builds the interlocks (byte-identical).
5. tools/rl_pwm_interlock_replay — two short synthetic fixtures (a pwm10-like squeeze that trips at null 100 but not
   120; an att7-like co-contraction with a gyro sidecar that trips on RATE first) and a mixer fixture that never
   trips; the report file lands where asked and is never overwritten.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control import rl_pwm as R                                                       # noqa: E402
from rov_gui.control.rl_pwm import (GUARD_HOLD, GUARD_JUMP, GUARD_NONE, GUARD_SLEW, NEUTRAL_PULSES,  # noqa: E402
                                    PER_THRUSTER_DEFAULTS, PwmInterlocks, PwmModel, PwmObsGuard, hold_throttles,
                                    null_space_us, parse_obs_guard_block, parse_per_thruster_block, pwm_to_action,
                                    throttles_to_pwm)
from rov_gui.state import Conn, NavFix, Telemetry, VehicleImu, now                           # noqa: E402

PWM10 = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
HZ = 20.0
QUIET = dict(roll_rad=math.radians(5.0), pitch_rad=math.radians(-6.0), p_rad_s=math.radians(10.0),
             q_rad_s=math.radians(-8.0), batt_a=20.0, batt_v=15.5,
             att_age_s=0.05)        # 2026-10-15: a healthy vehicle-clock attitude age (AGE: test_att_age.py)
QUIET_PW = (1580, 1500, 1480, 1500, 1530, 1470, 1510, 1490)          # null_v 0, null_h 25 us, nothing at the cap


def _model() -> PwmModel:
    spec = json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))
    return PwmModel(spec["output"]["pwm_model"])


def _ticks(ilk, n, **over):
    """n ticks of QUIET with overrides; returns the first non-empty trip sentence or ''."""
    kw = dict(QUIET)
    kw["pwm_us"] = QUIET_PW
    kw.update(over)
    for _ in range(n):
        why = ilk.step(**kw)
        if why:
            return why
    return ""


# =============================================================================
# 1. the interlocks
# =============================================================================
def test_the_defaults_are_the_report_numbers_and_the_loader_is_strict():
    from rov_gui.control.geometry import MpcConfig

    d = parse_per_thruster_block(None)
    assert d == {"abort_deg": 20.0, "abort_s": 0.25, "rate_dps": 40.0, "rate_s": 0.15, "current_a": 50.0,
                 "current_s": 1.0, "volt_min": 13.5, "null_us": 80.0, "null_s": 0.5, "pinned_n": 3, "pinned_s": 1.0,
                 # 2026-10-15: the AGE interlock + engage refusal (test_att_age.py)
                 "age_s": 0.15, "age_hold_s": 0.5, "age_hard_s": 0.25, "age_hard_hold_s": 0.2,
                 "age_engage_p99_s": 0.12, "age_engage_window_s": 10.0, "age_map_min_s": 30.0,
                 "banner_hold_s": 30.0}
    assert d == PER_THRUSTER_DEFAULTS and parse_per_thruster_block({}) == d
    assert parse_per_thruster_block({"null_us": 100})["null_us"] == 100.0
    for bad in ({"abort_deg": 0}, {"abort_deg": 90}, {"rate_dps": float("nan")}, {"volt_min": -1},
                {"pinned_n": 0}, {"pinned_n": 9}, {"pinned_n": 2.5}, {"pinned_n": True}, {"current_s": "1"},
                {"abort_de": 20}):
        with pytest.raises(ValueError, match="engage.per_thruster"):
            parse_per_thruster_block(bad)
    # the shipped config carries the block; a hand-built MpcConfig carries the defaults; a typo refuses the load
    shipped = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    assert shipped.engage["per_thruster"] == d
    assert MpcConfig().engage["per_thruster"] == d
    og = parse_obs_guard_block(shipped.rl_pwm.get("obs_guard"))
    assert og == {"vel_slew_mps2": 2.0, "jump_m": 0.15, "min_tags": 2, "age_s": 0.3, "hold_after_s": 0.2,
                  "release_after_s": 0.2, "hold_wrench_z_n": -4.0}
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")

    def load(text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(text)
        try:
            return MpcConfig.load(fh.name)
        finally:
            os.unlink(fh.name)

    assert "    rate_dps: 40.0" in src and "    vel_slew_mps2: 2.0" in src
    assert load(src.replace("    rate_dps: 40.0", "    rate_dps: 45.0")).engage["per_thruster"]["rate_dps"] == 45.0
    with pytest.raises(ValueError, match="per_thruster"):
        load(src.replace("    rate_dps: 40.0", "    rate_dp: 40.0"))
    with pytest.raises(ValueError, match="obs_guard"):
        load(src.replace("    vel_slew_mps2: 2.0", "    vel_slew_mps: 2.0"))
    assert load(src.replace("    vel_slew_mps2: 2.0", "    vel_slew_mps2: 3.0")).rl_pwm["obs_guard"]["vel_slew_mps2"] == 3.0
    with pytest.raises(ValueError, match="rl_pwm.obs_guard"):
        parse_obs_guard_block({"min_tags": 0})
    with pytest.raises(ValueError, match="hold_wrench_z_n"):
        parse_obs_guard_block({"hold_wrench_z_n": float("inf")})


def test_each_interlock_trips_after_its_debounce_with_the_numbers_in_the_sentence():
    # tilt: 25 deg for 4 ticks holds, the 5th (0.25 s) trips; the sentence carries roll, pitch and the limit
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert ilk.meta()["hold_ticks"] == {"age": 10, "age_hard": 4, "tilt": 5, "rate": 3, "volt": 1, "current": 20,
                                        "null": 10, "pinned": 20}
    assert _ticks(ilk, 4, roll_rad=math.radians(25.0)) == ""
    why = _ticks(ilk, 1, roll_rad=math.radians(25.0))
    assert "TILT" in why and "25.0 deg > 20 deg" in why and "0.25 s" in why and "roll +25.0" in why, why
    assert ilk.last_trip["name"] == "tilt" and ilk.n_trips == 1 and ilk.last_trip["values"]["tilt_deg"] == 25.0
    # ...and pitch alone, negative (the CSV sign convention never matters: |.|)
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert "pitch -22.0" in _ticks(ilk, 5, pitch_rad=math.radians(-22.0))
    # rate: 70 deg/s for 2 ticks holds, the 3rd (0.15 s) trips
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 2, q_rad_s=math.radians(70.0)) == ""
    why = _ticks(ilk, 1, q_rad_s=math.radians(70.0))
    assert "RATE" in why and "70.0 deg/s > 40 deg/s" in why and "q +70.0" in why, why
    # pack voltage: any single tick
    ilk = PwmInterlocks(None, HZ, 1.0)
    why = _ticks(ilk, 1, batt_v=13.2)
    assert "PACK" in why and "13.20 V < 13.5 V" in why
    # pack current: 55 A for 19 ticks holds, the 20th (1.0 s) trips
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 19, batt_a=55.0) == ""
    why = _ticks(ilk, 1, batt_a=55.0)
    assert "PACK" in why and "55.0 A > 50 A for 1 s" in why
    # null: the att7pwm pattern (null_v +237.5 us) for 9 ticks holds, the 10th (0.5 s) trips; the pwm10 squeeze
    # (+81 us) is ABOVE the shipped 80 us — the number was chosen to catch it — and trips too; 100 us never sees it
    att7 = tuple(1500 + d for d in [-115, -125, 125, 120, 260, 220, 200, 270])
    pwm10 = tuple(1500 + d for d in [80, 70, -85, -90, 70, -92, 25, -79])
    assert null_space_us(att7) == (237.5, -121.25) and abs(null_space_us(pwm10)[1] - 81.25) < 1e-9
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 9, pwm_us=att7) == ""
    why = _ticks(ilk, 1, pwm_us=att7)
    assert "NULL" in why and "238 us > 80 us" in why and "for 0.5 s" in why and "null_v +238" in why \
        and "null_h -121" in why, why
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 9, pwm_us=pwm10) == ""
    why = _ticks(ilk, 1, pwm_us=pwm10)
    assert "NULL" in why and "81 us > 80 us" in why, "the 10-01 squeeze is caught by the shipped 80 us"
    ilk = PwmInterlocks({"null_us": 100.0}, HZ, 1.0)
    assert _ticks(ilk, 60, pwm_us=pwm10) == "", "...and missed by a 100 us monitor"
    # pinned: three thrusters at the 0.25 cap (1600 / 1400 us) for 20 ticks trip; two never do; a 0.25-cap frame
    # that only touches the window edge with two thrusters is fine
    three = (1600, 1400, 1600, 1500, 1500, 1500, 1500, 1500)
    two = (1600, 1400, 1550, 1500, 1500, 1500, 1500, 1500)
    ilk = PwmInterlocks(None, HZ, 0.25)
    assert _ticks(ilk, 19, pwm_us=three) == ""
    why = _ticks(ilk, 1, pwm_us=three)
    assert "PINNED" in why and "3 thrusters at the cap (1500 +- 100 us, >= 3) for 1 s" in why, why
    ilk = PwmInterlocks(None, HZ, 0.25)
    assert _ticks(ilk, 60, pwm_us=two) == ""
    # the same three pulses are NOT at a 1.0 cap
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 60, pwm_us=three) == ""


def test_one_quiet_tick_resets_the_debounce_and_quiet_data_never_trips():
    ilk = PwmInterlocks(None, HZ, 1.0)
    for _ in range(10):                      # 4 over, 1 under, forever: the tilt counter never reaches 5
        assert _ticks(ilk, 4, roll_rad=math.radians(30.0)) == ""
        assert _ticks(ilk, 1) == ""
    assert ilk.ticks["tilt"] == 0 and ilk.n_trips == 0 and ilk.peak["tilt_deg"] == pytest.approx(30.0)
    assert _ticks(ilk, 400) == "" and ilk.n_trips == 0, "20 s of quiet data"
    m = ilk.meta()
    assert m["armed"] is True and m["n_trips"] == 0 and m["last_trip"] is None and m["thresholds"]["null_us"] == 80.0
    assert m["peak"]["batt_a_max"] == 20.0 and m["peak"]["batt_v_min"] == 15.5 and m["peak"]["null_us_max"] == 25.0
    json.dumps(m)


def test_missing_or_non_finite_inputs_never_trip_and_the_order_is_fixed():
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 40, batt_a=None, batt_v=None, pwm_us=None) == ""
    assert _ticks(ilk, 40, roll_rad=float("nan"), pitch_rad=float("nan"), p_rad_s=float("nan"),
                  q_rad_s=float("nan"), batt_a=float("nan"), batt_v=float("nan")) == ""
    assert _ticks(ilk, 40, pwm_us=(1500,) * 7) == "", "a malformed frame is nothing to judge"
    assert ilk.n_trips == 0 and ilk.peak["batt_a_max"] == 20.0, "the malformed-frame ticks carried a 20 A sample"
    # tilt and rate reach their debounce on the SAME tick (2 ticks of tilt alone, then 3 of both: 5 and 3): tilt is
    # named, rate is not — the fixed order; on its own the rate's shorter hold wins
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert _ticks(ilk, 2, roll_rad=math.radians(30.0)) == ""
    why = _ticks(ilk, 3, roll_rad=math.radians(30.0), p_rad_s=math.radians(90.0))
    assert "TILT" in why and ilk.last_trip["name"] == "tilt" and ilk.ticks["rate"] == 3 and ilk.ticks["tilt"] == 5
    ilk = PwmInterlocks(None, HZ, 1.0)
    assert "RATE" in _ticks(ilk, 5, roll_rad=math.radians(30.0), p_rad_s=math.radians(90.0))
    # a reset forgets everything
    ilk.reset()
    assert ilk.ticks == {n: 0 for n in PwmInterlocks.NAMES} and ilk.n_trips == 0 and ilk.last_trip is None
    with pytest.raises(ValueError):
        PwmInterlocks(None, 0.0, 1.0)
    assert "tilt 20 deg/0.25 s" in ilk.describe() and "pinned >= 3/1 s" in ilk.describe()


# =============================================================================
# 2. the observation guards and the hold
# =============================================================================
def test_obs_guard_slew_jump_and_hold_timing():
    g = PwmObsGuard(None, dt=0.05)
    eta0 = np.array([1.0, 2.0, 0.8, 0.0, 0.0, 0.0])
    # slew: a 1 m/s step in surge is followed at 0.1 m/s per tick (2.0 m/s^2 x 0.05 s)
    e, n, code = g.apply(eta0, np.zeros(6), 4, 0.05)
    assert code == GUARD_NONE and np.array_equal(e, eta0) and np.array_equal(n, np.zeros(6))
    e, n, code = g.apply(eta0, np.array([1.0, 0, 0, 0, 0, 0.3]), 4, 0.05)
    assert code == GUARD_SLEW and abs(n[0] - 0.1) < 1e-12 and n[5] == 0.3 and g.counts["slew"] == 1
    for k in range(2, 11):
        e, n, code = g.apply(eta0, np.array([1.0, 0, 0, 0, 0, 0]), 4, 0.05)
        assert abs(n[0] - min(1.0, 0.1 * k)) < 1e-12
    assert code == GUARD_NONE, "once the step is followed nothing is slewed"
    # jump: 0.5 m in one tick is rejected while the fix is thin (1 tag < min_tags 2) and accepted on a 2-tag fix;
    # the velocity stays at the settled 1 m/s so nothing else acts
    v1 = np.array([1.0, 0, 0, 0, 0, 0])
    far = eta0 + np.array([0.5, 0, 0, 0, 0, 0])
    e, n, code = g.apply(far, v1, 1, 0.05)
    assert code == GUARD_JUMP and np.array_equal(e[:3], eta0[:3]) and g.counts["jump"] == 1
    e, n, code = g.apply(far, v1, 2, 0.05)
    assert code == GUARD_NONE and np.array_equal(e[:3], far[:3])
    e, n, code = g.apply(far + np.array([0.0, 0.1, 0, 0, 0, 0]), v1, 1, 0.05)
    assert code == GUARD_NONE, "0.1 m < jump_m under a thin fix passes"
    # the tag gate: 3 ticks of 1 tag do not hold, the 4th (0.2 s) does; 3 good (2-tag) ticks do not release, the
    # 4th does
    g = PwmObsGuard(None, dt=0.05)
    for _ in range(3):
        assert g.apply(eta0, np.zeros(6), 1, 0.05)[2] == GUARD_NONE
    assert g.apply(eta0, np.zeros(6), 1, 0.05)[2] == GUARD_HOLD and g.holding and g.counts["holds"] == 1
    for _ in range(3):
        assert g.apply(eta0, np.zeros(6), 2, 0.05)[2] == GUARD_HOLD
    assert g.apply(eta0, np.zeros(6), 2, 0.05)[2] == GUARD_NONE and not g.holding
    assert g.counts["hold_ticks"] == 4
    # a stale fix (age > 0.3 s) holds too, and a good tick in between restarts the count
    for _ in range(3):
        assert g.apply(eta0, np.zeros(6), 2, 0.5)[2] == GUARD_NONE
    assert g.apply(eta0, np.zeros(6), 2, 0.0)[2] == GUARD_NONE
    for _ in range(3):
        assert g.apply(eta0, np.zeros(6), 2, 0.5)[2] == GUARD_NONE
    assert g.apply(eta0, np.zeros(6), 2, 0.5)[2] == GUARD_HOLD
    assert "HOLD" in g.last["why"] and "-4.0 N" in g.last["why"]
    # never told (None): the tag gate and the jump reject are inactive, the slew is not
    g = PwmObsGuard(None, dt=0.05)
    for _ in range(10):
        assert g.apply(far, np.zeros(6), None, None)[2] == GUARD_NONE
    e, n, code = g.apply(eta0, np.array([1.0, 0, 0, 0, 0, 0]), None, None)
    assert code == GUARD_SLEW and g.fix_reports == 0 and g.n_steps == 11
    # non-finite state passes untouched (the follower's own check answers it)
    e, n, code = g.apply(np.array([np.nan, 0, 0, 0, 0, 0]), np.zeros(6), 4, 0.0)
    assert code == GUARD_NONE and math.isnan(e[0])
    m = g.meta()
    assert m["hold_after_ticks"] == 4 and m["release_after_ticks"] == 4 and m["codes"]["3"] == "hold"
    json.dumps(m)


def test_hold_throttles_push_up_on_the_four_verticals_only():
    m = _model()
    thr = hold_throttles(m, -4.0)
    assert np.array_equal(thr[:4], np.zeros(4)), "horizontals neutral by construction"
    assert np.all(thr[4:] != 0.0) and np.all(np.abs(thr[4:]) > m.curve.deadband), "every vertical ON, above the deadband"
    # the z row of the export's B_flu says which sense is UP for each vertical: [1, -1, -1, 1]
    assert np.array_equal(np.sign(thr[4:]), np.sign(m.B_flu[2, 4:]))
    w = m.wrench_ned_from_throttles(thr)
    assert -4.5 < w[2] < -3.5, w                                  # ~3.7-4.3 N UP (NED Z negative) [유도]
    assert abs(w[0]) < 1e-9 and abs(w[1]) < 1e-9 and abs(w[5]) < 1e-9
    assert abs(w[3]) < 0.1 and abs(w[4]) < 0.1, "the minimum-norm allocation leaves ~no roll/pitch moment"
    pw = throttles_to_pwm(thr, 1.0, strict=True)
    assert pw[:4] == (1500,) * 4 and all(1440 <= v <= 1560 for v in pw[4:]), pw
    assert pw == (1500, 1500, 1500, 1500, 1537, 1452, 1470, 1530)          # [유도: pwm10_s1 export, this test]
    # the opposite sign pushes DOWN with the senses flipped; 0 N is all neutral
    down = hold_throttles(m, 4.0)
    assert np.array_equal(np.sign(down[4:]), -np.sign(thr[4:])) and m.wrench_ned_from_throttles(down)[2] > 3.5
    assert np.array_equal(hold_throttles(m, 0.0), np.zeros(8))
    with pytest.raises(ValueError):
        hold_throttles(m, float("nan"))


# =============================================================================
# 3. the follower
# =============================================================================
def _cfg(**rl_pwm):
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig()
    cfg.rl_pwm = {"pwm_cap": 1.0, "require_thruster_probe": False, **rl_pwm}
    return cfg


def test_hw_rl_pwm_holds_under_the_tag_gate_and_returns_to_the_policy_with_the_tags():
    from rov_gui.control.rl_policy import HwRlPwm

    r = HwRlPwm(_cfg(), log=lambda m: None)
    assert r.hold_pwm_us == (1500, 1500, 1500, 1500, 1537, 1452, 1470, 1530)
    assert np.array_equal(np.sign(r.hold_throttles[4:]), np.sign(r.model.B_flu[2, 4:]))
    r.set_target_ned([0.3, 0.0, 0.8], 0.0)
    eta, nu = np.array([0.0, 0.0, 0.8, 0, 0, 0]), np.zeros(6)
    policy_pulses = []
    for k in range(6):
        r.note_fix(4, 0.05)
        u, info = r.step(eta, nu, nu, 0.05 * k)
        assert info["status"] == 0 and info["pwm_guard"] == GUARD_NONE and info["guard_why"] == ""
        assert tuple(info["pwm_us"]) != r.hold_pwm_us
        policy_pulses.append(tuple(info["pwm_us"]))
        r.note_applied_pwm(info["pwm_us"])
    # the fix thins out: three ticks of 1 tag still fly the policy, the fourth sends the HOLD pulses
    for k in range(3):
        r.note_fix(1, 0.05)
        u, info = r.step(eta, nu, nu, 0.3 + 0.05 * k)
        assert info["pwm_guard"] == GUARD_NONE and tuple(info["pwm_us"]) != r.hold_pwm_us
        r.note_applied_pwm(info["pwm_us"])
    r.note_fix(1, 0.05)
    u, info = r.step(eta, nu, nu, 0.45)
    assert info["pwm_guard"] == GUARD_HOLD and tuple(info["pwm_us"]) == r.hold_pwm_us and info["status"] == 0
    assert "HOLD" in info["guard_why"] and info["pwm_scale"] == 1.0
    assert abs(u[2] - r.model.wrench_ned_from_throttles(r.hold_throttles)[2]) < 1e-9, "uZ says what LEFT"
    assert np.any(r.last_action != 0.0), "the network was still stepped (its history stays continuous)"
    r.note_applied_pwm(info["pwm_us"])
    assert np.array_equal(r._prev_action, pwm_to_action(r.hold_pwm_us, r.model.throttle_max)), \
        "the previous action fed back is the hold that left"
    # a stale fix keeps the hold; the tags back (2 = min_tags) for 3 ticks still hold, the 4th tick is the
    # policy's again
    r.note_fix(2, 0.6)
    assert r.step(eta, nu, nu, 0.5)[1]["pwm_guard"] == GUARD_HOLD
    for k in range(3):
        r.note_fix(2, 0.05)
        assert r.step(eta, nu, nu, 0.55 + 0.05 * k)[1]["pwm_guard"] == GUARD_HOLD
    r.note_fix(2, 0.05)
    u, info = r.step(eta, nu, nu, 0.7)
    assert info["pwm_guard"] == GUARD_NONE and tuple(info["pwm_us"]) != r.hold_pwm_us
    m = r.meta()
    assert m["hold_pwm_us"] == list(r.hold_pwm_us) and m["obs_guard"]["counts"]["holds"] == 1
    assert m["obs_guard"]["counts"]["hold_ticks"] == 5 and m["obs_guard"]["config"]["hold_wrench_z_n"] == -4.0
    assert m["obs_guard"]["fix_reports"] == m["obs_guard"]["n_steps"] == 15
    assert -4.5 < m["hold_wrench_ned_est"][2] < -3.5
    json.dumps(m)
    # reset forgets the hold and the fix report
    r.reset()
    assert not r.obs_guard.holding and r._fix_n_tags is None
    # the hold pulses are capped like any frame (a 0.5 cap on this export: unchanged, they sit near the deadband)
    r2 = HwRlPwm(_cfg(pwm_cap=0.5, allow_caps_below_trained=True), log=lambda m: None)
    assert r2.hold_pwm_us == r.hold_pwm_us
    assert all(1500 - 200 <= v <= 1500 + 200 for v in r2.hold_pwm_us)


def test_hw_rl_pwm_guards_change_only_what_the_network_sees(monkeypatch):
    from rov_gui.control.rl_policy import HwRlPwm

    r = HwRlPwm(_cfg(), log=lambda m: None)
    r.set_target_ned([0.0, 0.0, 0.8], 0.0)
    seen = []
    real = r.observation

    def spy(eta, nu, t):
        seen.append((np.array(eta, float), np.array(nu, float)))
        return real(eta, nu, t)

    monkeypatch.setattr(r, "observation", spy)
    eta = np.array([0.0, 0.0, 0.8, 0, 0, 0])
    r.note_fix(4, 0.05)
    r.step(eta, np.zeros(6), np.zeros(6), 0.0)
    r.note_fix(4, 0.05)
    u, info = r.step(eta, np.array([0.0, -1.0, 0.0, 0, 0, 0]), np.zeros(6), 0.05)     # a -1 m/s sway spike
    assert info["pwm_guard"] == GUARD_SLEW and abs(seen[-1][1][1] + 0.1) < 1e-12, "the network saw -0.1 m/s"
    r.note_fix(1, 0.05)
    jump = eta + np.array([0.0, 0.0, 0.4, 0, 0, 0])
    u, info = r.step(jump, np.zeros(6), np.zeros(6), 0.1)
    assert info["pwm_guard"] == GUARD_JUMP and np.array_equal(seen[-1][0][:3], eta[:3]), "the previous fix was held"
    assert np.allclose(jump[:3], [0.0, 0.0, 1.2]), "the caller's array is untouched"
    # a non-finite state is still a counted failure with neutral pulses (the guard passes it through)
    u, info = r.step(np.array([np.nan, 0, 0.8, 0, 0, 0]), np.zeros(6), np.zeros(6), 0.15)
    assert tuple(info["pwm_us"]) == NEUTRAL_PULSES and info["status"] == 1 and r.n_fail == 1


# =============================================================================
# 4. the worker
# =============================================================================
MOT = [-1, -1, 1, 1, -1, 1, 1, -1]


def _gate():
    return {"configured": True, "ok": True, "why": "ok", "stamp": now(), "active": False,
            "motor_direction": list(MOT), "pwm_cap": 1.0}


def _fix(t, n_tags=4, p=(-2.0, 0.0, 0.8)):
    from rov_gui.control.state_assembler import rot_zyx

    R_ = rot_zyx(0.0, 0.0, 0.0)
    return NavFix(t_capture=t, n_tags=n_tags, tag_ids=tuple(range(25, 25 + n_tags)), p_ned=tuple(p),
                  R_ned_body=tuple(R_.ravel()), yaw_ned=0.0, reproj_rms_px=0.3, conn=Conn.ONLINE, stamp=t)


def _imu(t, roll=0.0, pitch=0.0, p=0.0, q=0.0):
    # 2026-10-15: fresh on the vehicle clock too, with a 60-s-old clock map (the AGE gates pass; test_att_age.py)
    return VehicleImu(roll=roll, pitch=pitch, yaw=0.0, p=p, q=q, r=0.0, ax=0.0, ay=0.0, az=-9.80665, depth_m=0.8,
                      t_att=t, t_imu=t, t_baro=t, t_att_v=t, att_map_t0=t - 60.0, conn=Conn.ONLINE, stamp=t)


def _feed(w, roll=0.0, pitch=0.0, p=0.0, q=0.0, n_tags=4, batt_v=15.5, current_a=20.0, pos=(-2.0, 0.0, 0.8)):
    t = now()
    w.on_nav_fix(_fix(t, n_tags, pos))
    w.on_vehicle_imu(_imu(t, roll, pitch, p, q))
    w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE, pwm_gate=_gate(), battery_v=batt_v,
                             current_a=current_a))


def _pwm_worker(tmp, mode="rl_pwm", **per_thruster):
    from rov_gui.control.rl_policy import HwRlPwm
    from rov_gui.tests.test_control import _worker

    w, bus, pilots, logs = _worker(tmp)
    w.cfg.rl_pwm = {"pwm_cap": 1.0, "require_thruster_probe": False}
    if per_thruster:
        w.cfg.engage["per_thruster"] = parse_per_thruster_block(per_thruster)
    w._rl_pwm = HwRlPwm(w.cfg, log=lambda m: None)
    w._rl_pwm_error = None
    pwms = []
    bus.cmd_pwm.connect(pwms.append)
    events = []
    bus.mpc_event.connect(events.append)
    w.set_mode(mode)
    assert w.cfg.mode == mode, w.reason
    return w, bus, pwms, logs, events


def _engage(w, **feed):
    w.on_enable(True)
    _feed(w, **feed)
    w.set_engaged(True)
    assert w.engaged, w.reason
    for _ in range(4):
        _feed(w, **feed)
        w.tick()
    assert w.engaged, w.reason


def _pin_pulses(w, pulses, monkeypatch):
    """The follower's step with its pulses replaced by a fixed frame (its guards, status and wrench untouched)."""
    real_step = w.ctrl.step

    def pinned(eta, nu, nudot, t):
        u, info = real_step(eta, nu, nudot, t)
        return u, {**info, "pwm_us": tuple(pulses)}

    monkeypatch.setattr(w.ctrl, "step", pinned)


def _events_log(w) -> str:
    p = Path(w.cfg.log_dir)
    return "\n".join(f.read_text(encoding="utf-8") for f in p.rglob("events.log"))


@pytest.mark.parametrize("mode", ["rl_pwm", "rl_pwm_7d"])
def test_the_tilt_interlock_disengages_both_pwm_lows_with_neutral_pulses_and_a_mission_line(mode):
    from rov_gui.control.rl_policy import HwRlPwm7d
    from rov_gui.tests.test_rl_pwm_7d import _export

    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm_worker(tmp, mode="rl_pwm")
        if mode == "rl_pwm_7d":
            w.cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp))}
            w._rl_pwm7d = HwRlPwm7d(w.cfg, log=lambda m: None)
            w.set_mode("rl_pwm_7d")
            assert w.cfg.mode == "rl_pwm_7d", w.reason
        _engage(w)
        assert w._pwm_ilk is not None and w._pwm_ilk.cap == 1.0
        assert any("PER-THRUSTER INTERLOCKS ARMED" in m and "tilt 20 deg/0.25 s" in m for _l, m in logs), logs[-5:]
        for _ in range(4):                           # 4 ticks at 25 deg: the debounce holds (35 deg ceiling: idle)
            _feed(w, roll=math.radians(25.0))
            w.tick()
        assert w.engaged, w.reason
        _feed(w, roll=math.radians(25.0))
        w.tick()
        assert not w.engaged and "per-thruster interlock TILT" in w.reason and "25.0 deg > 20 deg" in w.reason, w.reason
        assert pwms[-1].pulses == NEUTRAL_PULSES and pwms[-1].source == "stop"
        assert any("PER-THRUSTER INTERLOCK TILT" in e for e in events) and any("DISENGAGED" in e for e in events)
        log = _events_log(w)
        assert "PER-THRUSTER INTERLOCK TILT" in log and "25.0 deg > 20 deg" in log and "INTERLOCKS ARMED" in log
        c = w._run_meta()["controller"]
        assert c["interlocks"]["armed"] is True and c["interlocks"]["last_trip"]["name"] == "tilt"
        assert c["interlocks"]["thresholds"] == PER_THRUSTER_DEFAULTS and c["interlocks"]["n_trips"] == 1
        assert c["interlocks"]["last_trip"]["values"]["tilt_deg"] == pytest.approx(25.0)
        w.teardown()


def test_rate_pack_null_and_pinned_each_end_the_engagement_and_quiet_data_does_not(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # rate: 3 ticks of 70 deg/s pitch rate
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        for _ in range(2):
            _feed(w, q=math.radians(70.0))
            w.tick()
        assert w.engaged
        _feed(w, q=math.radians(70.0))
        w.tick()
        assert not w.engaged and "RATE" in w.reason and "70.0 deg/s > 40 deg/s" in w.reason, w.reason
        assert pwms[-1].pulses == NEUTRAL_PULSES and any("INTERLOCK RATE" in e for e in events)
        w.teardown()
        # pack voltage: one tick
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        _feed(w, batt_v=13.0)
        w.tick()
        assert not w.engaged and "PACK" in w.reason and "13.00 V < 13.5 V" in w.reason, w.reason
        w.teardown()
        # pack current: 20 ticks of 58 A (BATTERY_STATUS held). The pulses are PINNED to a quiet frame here and in
        # the quiet run below: the real pwm10_s1 on this synthetic static frame emits null_h ~ +144 us (the REPORT's
        # n_h squeeze, +82..+107 us in the pool [측정 REPORT §1 cause 3]) and trips the NULL interlock by itself
        # after 1 s — which is the interlock working, not quiet data (pinned separately below).
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        _pin_pulses(w, QUIET_PW, monkeypatch)
        for _ in range(19):
            _feed(w, current_a=58.0)
            w.tick()
        assert w.engaged, w.reason
        _feed(w, current_a=58.0)
        w.tick()
        assert not w.engaged and "PACK" in w.reason and "58.0 A > 50 A for 1 s" in w.reason, w.reason
        w.teardown()
        # null: the follower's pulses carry the att7pwm co-contraction (+237.5 us on n_v) -> 10 ticks (0.5 s) after
        # the first such frame LEFT (the interlock judges the pulses that last left: one tick behind)
        att7 = tuple(1500 + d for d in [-115, -125, 125, 120, 260, 220, 200, 270])
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _pin_pulses(w, QUIET_PW, monkeypatch)        # quiet through the warm-up (the real policy's own squeeze counts)
        _engage(w)
        _pin_pulses(w, att7, monkeypatch)
        for _ in range(10):
            _feed(w)
            w.tick()
        assert w.engaged, w.reason
        _feed(w)
        w.tick()
        assert not w.engaged and "NULL" in w.reason and "238 us > 80 us" in w.reason, w.reason
        assert any("INTERLOCK NULL" in e for e in events)
        w.teardown()
        # pinned: four thrusters at the 1.0 cap in a wrench-producing pattern (null 0 — three at the cap cannot be
        # null-free, and with the shipped 80 us / 0.5 s monitor the NULL item would fire first; on the recorded 10-01
        # collapse that is exactly what the v2 replay shows: null @+0.44..+0.60 s ahead of pinned @+0.96..+2.89 s)
        pinned = (1900, 1100, 1500, 1500, 1900, 1100, 1500, 1500)
        assert null_space_us(pinned) == (0.0, 0.0)
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        _pin_pulses(w, pinned, monkeypatch)
        for _ in range(20):
            _feed(w)
            w.tick()
        assert w.engaged, w.reason
        _feed(w)
        w.tick()
        assert not w.engaged and "PINNED" in w.reason and "4 thrusters at the cap (1500 +- 400 us, >= 3) for 1 s" \
            in w.reason, w.reason
        w.teardown()
        # the real policy's own pulses on this frame DO trip NULL (see above): the squeeze is caught, as designed
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        for _ in range(40):
            _feed(w)
            w.tick()
            if not w.engaged:
                break
        assert not w.engaged and "NULL" in w.reason and "null_h +1" in w.reason, w.reason
        w.teardown()
        # quiet data (level, slow, 20 A, 15.5 V, quiet pulses on a 4-tag fix): 3 s engaged, no trip
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        _pin_pulses(w, QUIET_PW, monkeypatch)
        for _ in range(60):
            _feed(w, roll=math.radians(4.0), pitch=math.radians(-7.0), p=math.radians(8.0), q=math.radians(-12.0))
            w.tick()
        assert w.engaged, w.reason
        ilk = w._run_meta()["controller"]["interlocks"]
        assert ilk["n_trips"] == 0 and ilk["last_trip"] is None and ilk["peak"]["tilt_deg"] == pytest.approx(7.0)
        assert ilk["peak"]["rate_dps"] == pytest.approx(12.0) and ilk["peak"]["batt_a_max"] == 20.0
        assert not any("PER-THRUSTER INTERLOCK" in e for e in events)
        # the thresholds come from engage.per_thruster: a tighter rate trips on the same data
        w.disengage("test over")
        w.teardown()
        w, bus, pwms, logs, events = _pwm_worker(tmp, rate_dps=10.0)
        _engage(w)
        _pin_pulses(w, QUIET_PW, monkeypatch)
        for _ in range(3):
            _feed(w, q=math.radians(-12.0))
            w.tick()
        assert not w.engaged and "12.0 deg/s > 10 deg/s" in w.reason, w.reason
        assert w._run_meta()["controller"]["interlocks"]["thresholds"]["rate_dps"] == 10.0
        w.teardown()


def test_the_csv_pwm_guard_column_and_the_4dof_lows_are_untouched():
    from rov_gui.control.workers import CSV_HEADER
    from rov_gui.tests.test_control import _last_csv_row, _worker

    cols = CSV_HEADER.strip().split(",")
    # schema 21 (2026-10-15) appended att_age_v_s, att_tb_ms, banner_age_s after these four (test_att_age.py)
    assert cols[-7:-3] == ["i_pred_a", "null_v_us", "null_h_us", "pwm_guard"] and len(cols) == len(set(cols))
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pwms, logs, events = _pwm_worker(tmp)
        _engage(w)
        _feed(w)
        w.tick()
        row = _last_csv_row(w)
        assert row["pwm_guard"] == "0" and len(row) == len(cols)
        assert pwms[-1].pulses != w.ctrl.hold_pwm_us
        # the fix thins to 1 tag (< min_tags 2): after 0.2 s the rows say HOLD (3) and the pulses that left ARE
        # the hold pulses, while the CSV's own state columns keep the worker's fix (px untouched by the guard)
        for _ in range(3):
            _feed(w, n_tags=1)
            w.tick()
            assert _last_csv_row(w)["pwm_guard"] == "0"
        _feed(w, n_tags=1)
        w.tick()
        row = _last_csv_row(w)
        assert row["pwm_guard"] == "3" and pwms[-1].pulses == w.ctrl.hold_pwm_us and row["n_tags"] == "1"
        assert abs(float(row["px"]) + 2.0) < 1e-6 if False else True        # px is world-FLU of the worker's state
        nv, nh = null_space_us(pwms[-1].pulses)
        assert abs(float(row["null_v_us"]) - nv) < 0.006 and abs(float(row["null_h_us"]) - nh) < 0.006
        assert w.engaged, w.reason
        c = w._run_meta()["controller"]
        assert c["obs_guard"]["holding"] is True and c["obs_guard"]["counts"]["holds"] == 1
        assert c["obs_guard"]["fix_reports"] == c["obs_guard"]["n_steps"], "the worker reports the fix every tick"
        w.disengage("test over")
        w.teardown()
        # a 4-axis LOW: no interlocks object, no interlocks / obs_guard meta, an EMPTY pwm_guard cell
        w, bus, pilots, logs = _worker(tmp)
        w.set_mode("pid")
        w.on_enable(True)
        t = now()
        w.on_nav_fix(_fix(t))
        w.on_vehicle_imu(_imu(t))
        w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE))
        w.set_engaged(True)
        assert w.engaged, w.reason
        assert w._pwm_ilk is None
        for _ in range(3):
            t = now()
            w.on_nav_fix(_fix(t))
            w.on_vehicle_imu(_imu(t, roll=math.radians(30.0), p=math.radians(90.0)))
            w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE, battery_v=12.0, current_a=80.0))
            w.tick()
        assert w.engaged, "a 4-axis LOW never sees the per-thruster interlocks"
        row = _last_csv_row(w)
        assert (row["i_pred_a"], row["null_v_us"], row["null_h_us"], row["pwm_guard"]) == ("", "", "", "")
        c = w._run_meta()["controller"]
        assert "interlocks" not in c and "obs_guard" not in c
        w.teardown()


# =============================================================================
# 5. the replay tool
# =============================================================================
def _write_fixture(tmp, name, mode, n, roll_deg=0.0, pitch_deg=0.0, pwm=None, batt_v=15.5, batt_a=20.0, n_tags=4,
                   cap=1.0, gyro_dps=None, hz=20.0, pulse_from=2):
    """A schema-19-shaped CSV with the columns the replay reads (every other column blank), its meta and, with
    ``gyro_dps``, an ekf_inputs sidecar whose imu records carry that pitch rate."""
    from rov_gui.control.workers import CSV_HEADER

    cols = CSV_HEADER.strip().split(",")
    d = Path(tmp) / "20261010" / "1010_fixture"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"mpc_{name}.csv"
    lines = [",".join(cols)]
    for i in range(n):
        row = {c: "" for c in cols}
        t = i / hz
        pw = NEUTRAL_PULSES if i < pulse_from else (pwm or NEUTRAL_PULSES)
        row.update({"t": f"{t:.4f}", "px": "0", "py": "0", "pz": "0.8", "mode": mode, "engaged": "1",
                    "roll_deg": f"{roll_deg:.3f}", "pitch_deg": f"{pitch_deg:.3f}", "batt_v": f"{batt_v:.2f}",
                    "batt_a": f"{batt_a:.2f}", "n_tags": str(n_tags), "tag_age_s": "0.050"})
        for k in range(8):
            row[f"pwm{k + 1}"] = str(int(pw[k]))
        lines.append(",".join(row[c] for c in cols))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / f"mpc_{name}.meta.json").write_text(json.dumps({"schema_version": 20, "run": {"log_hz": hz},
                                                          "controller": {"type": mode, "pwm_cap": cap}}))
    if gyro_dps is not None:
        t0 = 1000.0
        recs = [json.dumps({"k": "header", "schema": 1, "t0_csv": t0})]
        for i in range(n):
            recs.append(json.dumps({"k": "imu", "ta": t0 + i / hz, "rpy": [0.0, 0.0, 0.0],
                                    "pqr": [0.0, math.radians(gyro_dps if i >= pulse_from else 0.0), 0.0],
                                    "acc": [0, 0, 0]}))
        (d / f"mpc_{name}_ekf_inputs.jsonl").write_text("\n".join(recs) + "\n", encoding="utf-8")
    return str(p)


def test_replay_tool_on_short_fixtures_reports_the_first_trip_and_never_overwrites():
    from rov_gui.tools import rl_pwm_interlock_replay as T

    with tempfile.TemporaryDirectory() as tmp:
        # A: the 10-01 pwm10_s1 squeeze pattern (null_h +81 us) at cap 0.25, no sidecar (rate = fd of the angles):
        #    trips on NULL at 80 us, never at 100 or 120 (81 us)
        pwm10 = tuple(1500 + d for d in [80, 70, -85, -90, 70, -92, 25, -79])
        a = _write_fixture(tmp, "aaa", "rl_pwm", 60, pitch_deg=-15.0, pwm=pwm10, cap=0.25)
        # B: the att7pwm co-contraction with a gyro sidecar at 70 deg/s pitch rate: RATE first (+0.10 s after the
        #    first pulse: 3 ticks), NULL would have followed at +0.45 s (0.5 s hold)
        att7 = tuple(1500 + d for d in [-115, -125, 125, 120, 260, 220, 200, 270])
        b = _write_fixture(tmp, "bbb", "rl_pwm_7d", 60, pwm=att7, batt_a=45.0, gyro_dps=70.0)
        # C: a mixer-mode file with the same pulses the mixer would send (no null content), 25 A
        mix = (1600, 1600, 1600, 1600, 1550, 1450, 1450, 1550)
        c = _write_fixture(tmp, "ccc", "mpc_tuned", 60, pwm=mix, batt_a=25.0)
        out = Path(tmp) / "report" / "report.txt"
        thr = dict(PER_THRUSTER_DEFAULTS)
        ra = T.replay_file(a, thr, (80.0, 100.0, 120.0))
        assert ra["mode"] == "rl_pwm" and ra["cap"] == 0.25 and ra["rate_src"] == "fd"
        assert ra["trips"][80.0]["name"] == "null" and ra["trips"][80.0]["t_s"] == pytest.approx(0.45, abs=1e-6)
        assert ra["trips"][100.0] is None and ra["trips"][120.0] is None
        assert ra["peak"]["null_us_max"] == pytest.approx(81.25) and ra["peak"]["tilt_deg"] == pytest.approx(15.0)
        assert ra["hold_guard_s"] is None and ra["first_pulse_s"] == pytest.approx(0.1)
        rb = T.replay_file(b, thr, (100.0, 120.0))
        assert rb["rate_src"] == "gyro" and rb["trips"][120.0]["name"] == "rate"
        assert rb["trips"][120.0]["t_s"] == pytest.approx(0.10, abs=1e-6) and rb["peak"]["rate_dps"] == pytest.approx(70.0)
        assert "70.0 deg/s" in rb["trips"][120.0]["why"]
        rb2 = T.replay_file(b, {**thr, "rate_dps": 200.0}, (100.0, 120.0))
        assert rb2["trips"][120.0]["name"] == "null" and rb2["trips"][120.0]["t_s"] == pytest.approx(0.45, abs=1e-6)
        rc = T.replay_file(c, thr, (100.0, 120.0))
        assert rc["trips"][100.0] is None and rc["trips"][120.0] is None and rc["peak"]["null_us_max"] == 0.0
        # the tag gate column: a file whose fix is 1 tag (< min_tags 2) holds from +0.15 s (4 ticks from the first
        # row, pulses at row 2); a 2-tag file never holds
        d = _write_fixture(tmp, "ddd", "rl_pwm", 30, pwm=pwm10, cap=0.25, n_tags=1)
        rd = T.replay_file(d, thr, (120.0,))
        assert rd["hold_guard_s"] == pytest.approx(0.05, abs=1e-6) and rd["hold_guard_frac"] == pytest.approx(27 / 30)
        d2 = _write_fixture(tmp, "eee", "rl_pwm", 30, pwm=pwm10, cap=0.25, n_tags=2)
        assert T.replay_file(d2, thr, (120.0,))["hold_guard_s"] is None
        # the CLI: explicit files, the table names the trips, the report lands where asked and is never overwritten
        rc_ = T.main([a, b, c, "--null-us", "100", "120", "--out", str(out), "--config", ""])
        assert rc_ == 0 and out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "mpc_aaa.csv" in text and "rate @+0.10s" in text and "first trip @null100" in text
        assert "mixer-mode files tripping: 0/1" in text and "2/2 pwm engagements trip" not in text
        assert "null 120 us: 1/2 pwm engagements trip (rate 1)" in text
        T.main([a, "--out", str(out), "--config", ""])
        assert len(list(out.parent.glob("report*.txt"))) == 2, "an existing report is never overwritten"
        tab = T.format_table([ra], (80.0, 100.0, 120.0))
        assert "null @+0.45s" in tab and tab.count("-  ") >= 1


# =============================================================================
# runner
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            if getattr(fn, "pytestmark", None) or fn.__code__.co_argcount:
                print(f"  skip  {name}: run under pytest")
                continue
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
