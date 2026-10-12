#!/usr/bin/env python3
"""test_thruster_probe.py — the pool probe block (rov_gui/tools/thruster_probe.py, 2026-10-10), offline.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest \\
        rov_gui/tests/test_thruster_probe.py -q -p no:cacheprovider

No vehicle. What is pinned:

1. the wire: every frame the probe sends is the station's own packer's frame (control/rl_pwm.pack_rc_override: motors
   on RC 9..16 in that order, every other channel 65535) and the synthetic vehicle receives the motors there;
2. the expected axis/sign per motor, recomputed from the shipped pwm10_s1 export, agrees with the physical layout
   (verticals -> roll with T5/T6 one sign and T7/T8 the other; horizontals -> yaw with T1/T3 vs T2/T4) and with an
   independent r x d recomputation from the export's own thrust positions and directions;
3. the T200 inverse round-trips the export's curve and the vertical allocation is wrench-pure;
4. the dry-run synthetic vehicle yields PASS for all 8 with the nominal export, and FAIL with the OPPOSITE sign on
   both motors of a swapped vertical pair (5<->8) and on a reversed thruster: the probe DETECTS a diagonal swap;
5. neutral between steps and at exit, the 5-frame neutral burst on every abort (DISARM, mode, depth), a refusal
   before any frame when the script is not running, exit codes;
6. control/thruster_probe_artifact.load_and_check: the dry-run artefact is accepted only with allow_dry_run, and
   refused on a mismatching sigma / mot_direction / all_pass false / a WEAK verdict / a tampered PASS / age;
7. sequences B, C (+ free decay), D, E produce their artefacts with the numbers the synthetic vehicle was built with.

The vehicle, the Lua script and the thrusters themselves are NOT covered: that is the pool.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control import thruster_probe_artifact as ART  # noqa: E402
from rov_gui.control.rl_pwm import NEUTRAL_PULSES, PWM_CHANNELS_DEFAULT, pack_rc_override  # noqa: E402
from rov_gui.tools import thruster_probe as TP  # noqa: E402

PWM10 = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
MOT = [-1, -1, 1, 1, -1, 1, 1, -1]
SIGMA = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0]


@pytest.fixture(scope="module")
def export() -> TP.ExportModel:
    return TP.ExportModel(PWM10)


def _run(tmp, seqs, veh_kw=None, *, free_decay_s=0.0, tilt=None, limits=None, **run_kw):
    """A whole dry run on a FakeClock: (exit code, runner, synthetic vehicle, out dir)."""
    clock = TP.FakeClock()
    ex = TP.ExportModel(PWM10)
    veh = TP.SyntheticVehicle(ex, clock, **(veh_kw or {}))
    link = TP.Link(veh, target_sys=1)
    out = Path(tmp) / f"probe_{int(time.time() * 1e6) % 10 ** 9}"
    pr = TP.ProbeRunner(link, ex, out, dry_run=True, clock=clock, sleep=clock.sleep, log=lambda *a: None,
                        operator=lambda prompt, timeout: True, limits=limits)
    def release_tilt():
        """The synthetic operator: a tilt released 2 s into the free-decay recording."""
        veh.tilt_impulse = (clock() + 2.0, tilt[0], tilt[1])
        veh._impulsed = False
    code = TP.run_all(pr, list(seqs), free_decay_s=free_decay_s,
                      free_decay_before=(release_tilt if tilt is not None else None), **run_kw)
    return code, pr, veh, out


# =============================================================================
# 1. the wire
# =============================================================================
def test_every_frame_is_the_stations_packer_frame_and_lands_on_rc_9_to_16(tmp_path, export):
    clock = TP.FakeClock()
    veh = TP.SyntheticVehicle(export, clock)
    link = TP.Link(veh, target_sys=1)
    pw = (1600, 1400, 1510, 1490, 1555, 1445, 1500, 1501)
    ch = link.send_pulses(pw, clock())
    assert ch == pack_rc_override(pw)
    assert ch[:8] == [65535] * 8 and tuple(ch[8:16]) == pw and ch[16:] == [65535, 65535]
    assert veh.raw_frames[-1] == pack_rc_override(pw) and veh.frames_rx[-1][1] == pw
    assert tuple(PWM_CHANNELS_DEFAULT) == (9, 10, 11, 12, 13, 14, 15, 16)
    # a probe run: every frame that left is a packer frame of the pulses the runner logged
    code, pr, veh, out = _run(tmp_path, "A")
    assert code == 0
    sent = [p for _, p in pr.link.frames]
    assert len(veh.raw_frames) == len(sent) and all(f == pack_rc_override(p) for f, p in zip(veh.raw_frames, sent))
    assert all(1400 <= p <= 1600 for frame in sent for p in frame), "sequence A never sends more than +-100 us"


# =============================================================================
# 2. the expected response, from the export, against the physical layout
# =============================================================================
def test_expected_axis_and_sign_from_the_shipped_export_match_the_physical_layout(export):
    exp = export.expected
    assert all(exp[m][0] == "p" for m in (5, 6, 7, 8)), "verticals dominate roll"
    assert all(exp[m][0] == "r" for m in (1, 2, 3, 4)), "horizontals dominate yaw"
    assert exp[5][1] == exp[6][1] and exp[7][1] == exp[8][1] and exp[5][1] == -exp[7][1], \
        "front pair one sign, rear pair the other (T5 up / T6 down both roll the same way)"
    assert exp[1][1] == exp[3][1] and exp[2][1] == exp[4][1] and exp[1][1] == -exp[2][1], \
        "right-hand horizontals yaw one way, left-hand the other"
    # physical check from the export's OWN positions / directions (FLU): FRD moment = (r x d) * [1, -1, -1]
    pm = json.loads((PWM10 / "obs_spec.json").read_text())["output"]["pwm_model"]
    for m in range(1, 9):
        r = np.array(pm["thrust_pos_flu"][m - 1])
        d = np.array(pm["thrust_dirs_flu"][m - 1])
        mom = np.cross(r, d) * np.array([1.0, -1.0, -1.0])
        k = int(np.argmax(np.abs(mom)))
        assert exp[m] == ("pqr"[k], int(np.sign(mom[k]))), m
    # the shared helper is what the tool used: T5 pushes up at front-right -> starboard rises -> negative p (FRD)
    assert ART.expected_axis_sign(export.B_flu.tolist(), 4) == ("p", -1)
    assert ART.expected_axis_sign(export.B_flu.tolist(), 7) == ("p", +1)
    with pytest.raises(ValueError):
        ART.expected_axis_sign([[0.0] * 8 for _ in range(6)], 0)


def test_t200_inverse_round_trips_and_the_vertical_allocation_is_wrench_pure(export):
    c = export.curve
    for thr in np.concatenate([np.linspace(-1.0, -0.08, 25), np.linspace(0.08, 1.0, 25)]):
        f = export.thrust_n(float(thr))
        back = export.throttle_for_thrust(f)
        rpm = c.rpm_pos[0] * thr + c.rpm_pos[1] if thr > 0 else c.rpm_neg[0] * thr + c.rpm_neg[1]
        if abs(rpm) <= c.rpm_max:                     # the curve clips rpm at rpm_max: only the thrust round-trips there
            assert abs(back - float(thr)) < 1e-6, (thr, f, back)
        assert abs(export.thrust_n(back) - f) < 1e-6
    assert export.throttle_for_thrust(0.0) == 0.0
    tiny = 0.5 * export.thrust_n(c.deadband + 1e-6)             # below what the deadband edge gives
    assert export.throttle_for_thrust(tiny) == 0.0 and export.throttle_for_thrust(-tiny) == 0.0
    assert export.throttle_for_thrust(1e4) == 1.0 and export.throttle_for_thrust(-1e4) == -1.0
    assert export.thrust_for_pulse(1600) == pytest.approx(4.194, abs=0.01)     # [유도 pwm10_s1 t200 x 0.72]
    assert export.thrust_for_pulse(1400) == pytest.approx(-3.421, abs=0.01)
    pulses, F, thr = export.vertical_pulses_for_moment(0.0, +1.0)             # +1 N*m nose-up (FRD)
    assert pulses[:4] == (1500,) * 4
    w = export.wrench_frd(pulses)
    assert abs(w[4] - 1.0) < 0.02 and abs(w[2]) < 0.02 and abs(w[3]) < 0.02, w    # pure M, no Z, no K
    assert np.allclose(np.abs(F), 2.083, atol=0.01) and F[0] > 0 > F[1] and F[2] > 0 > F[3]
    pulses, F, thr = export.vertical_pulses_for_moment(+1.0, 0.0)             # +1 N*m starboard-down
    w = export.wrench_frd(pulses)
    assert abs(w[3] - 1.0) < 0.05 and abs(w[2]) < 0.05 and abs(w[4]) < 0.05, w
    assert F[0] < 0 and F[1] < 0 and F[2] > 0 and F[3] > 0


# =============================================================================
# 3. the sign probe on the synthetic vehicle: PASS, and a caught swap / reversal
# =============================================================================
def test_dry_run_sign_probe_passes_all_eight_with_the_nominal_export(tmp_path):
    code, pr, veh, out = _run(tmp_path, "A")
    assert code == 0 and pr.abort_reason is None
    doc = json.loads((out / ART.FILENAME).read_text())
    assert doc["schema"] == 1 and doc["tool"] == ART.TOOL and doc["dry_run"] is True
    assert doc["all_pass"] is True and [m["motor"] for m in doc["motors"]] == list(range(1, 9))
    for m in doc["motors"]:
        assert m["verdict"] == "PASS"
        assert m["n_samples_pos"] >= ART.MIN_SAMPLES and m["n_samples_neg"] >= ART.MIN_SAMPLES
        assert m["mean_rate_pos_dps"] * m["expected_sign_pos"] >= ART.RATE_THRESHOLD_DPS
        assert m["mean_rate_neg_dps"] * m["expected_sign_pos"] <= -ART.RATE_THRESHOLD_DPS
    assert doc["vehicle"]["mot_direction"] == MOT and doc["export"]["sigma"] == SIGMA
    assert doc["export"]["obs_spec_sha1"] == pr.ex.obs_spec_sha1 and doc["pulse_us"] == 100
    assert set(doc) >= {"written", "vehicle", "export", "on_s", "off_s", "motors", "all_pass", "operator_notes"}
    # neutral between steps: after every live step the vehicle received neutral frames before the next live one
    frames = [p for _, p in veh.frames_rx]
    live = [i for i, p in enumerate(frames) if p != NEUTRAL_PULSES]
    assert live, "pulses did go out"
    for a, b in zip(live, live[1:]):
        if frames[a] != frames[b]:
            assert any(frames[i] == NEUTRAL_PULSES for i in range(a + 1, b)), "a neutral step separates live steps"
    assert all(p == NEUTRAL_PULSES for p in frames[-TP.NEUTRAL_BURST_FRAMES:]), "the exit burst is neutral"
    # the tick log
    rows = list(csv.DictReader((out / "probe_ticks.csv").open()))
    assert list(rows[0]) == TP.ProbeRunner.CSV_COLUMNS
    assert len(rows) >= 8 * 2 * (1.0 + 3.0) / TP.TICK_S * 0.95
    assert {r["seq"] for r in rows} >= {"A", "engage", "exit"}
    run = json.loads((out / "probe_run.json").read_text())
    assert run["exit_code"] == 0 and run["abort_reason"] is None and run["results"]["A"]["all_pass"] is True


def test_dry_run_sign_probe_detects_a_swapped_vertical_pair_and_a_reversed_thruster(tmp_path):
    code, pr, veh, out = _run(tmp_path, "A", {"swap": (5, 8)})
    assert code == 1
    doc = json.loads((out / ART.FILENAME).read_text())
    assert doc["all_pass"] is False
    v = {m["motor"]: m for m in doc["motors"]}
    for m in (5, 8):
        assert v[m]["verdict"] == "FAIL", m
        # the OPPOSITE sign, clearly: the pulse of motor 5 drove the rear-left thruster
        assert v[m]["mean_rate_pos_dps"] * v[m]["expected_sign_pos"] <= -ART.RATE_THRESHOLD_DPS
        assert v[m]["mean_rate_neg_dps"] * v[m]["expected_sign_pos"] >= ART.RATE_THRESHOLD_DPS
    for m in (1, 2, 3, 4, 6, 7):
        assert v[m]["verdict"] == "PASS", m
    # one reversed thruster
    code, pr, veh, out = _run(tmp_path, "A", {"reverse": (3,)})
    assert code == 1
    doc = json.loads((out / ART.FILENAME).read_text())
    v = {m["motor"]: m for m in doc["motors"]}
    assert v[3]["verdict"] == "FAIL" and v[3]["mean_rate_pos_dps"] * v[3]["expected_sign_pos"] < 0
    assert all(v[m]["verdict"] == "PASS" for m in range(1, 9) if m != 3)
    assert ART.load_and_check(out / ART.FILENAME, SIGMA, MOT, allow_dry_run=True)[0] is False


def test_a_weak_response_is_weak_not_pass(tmp_path):
    """Thrust at 0.4 of the curve (the deadband-region gain the pool logs showed is ~0.5 [측정 REPORT.md §1 item 5], a
    sagging pack less): every sign agrees but the rates stay under 5 deg/s -> WEAK on all eight, never PASS, never
    FAIL — and the gate refuses a WEAK."""
    code, pr, veh, out = _run(tmp_path, "A", {"thrust_gain": 0.4})
    doc = json.loads((out / ART.FILENAME).read_text())
    assert [m["verdict"] for m in doc["motors"]] == ["WEAK"] * 8
    assert all(m["mean_rate_pos_dps"] * m["expected_sign_pos"] > 0 > m["mean_rate_neg_dps"] * m["expected_sign_pos"]
               for m in doc["motors"])
    assert doc["all_pass"] is False and code == 1
    ok, why = ART.load_and_check(out / ART.FILENAME, SIGMA, MOT, allow_dry_run=True)
    assert not ok and "WEAK" in why


# =============================================================================
# 4. abort conditions, refusals, exit codes
# =============================================================================
@pytest.mark.parametrize("veh_kw, expect", [
    ({"disarm_at": 1000.0 + 12.0}, "DISARMED"),
    ({"mode_at": (1000.0 + 12.0, 0)}, "not MANUAL"),
    ({"net_weight_n": 3.0, "depth0_m": 0.6}, "depth"),
])
def test_an_abort_mid_run_ends_in_a_neutral_burst_and_a_failing_artefact(tmp_path, veh_kw, expect):
    code, pr, veh, out = _run(tmp_path, "A", veh_kw)
    assert code == 3 and expect in pr.abort_reason
    frames = [p for _, p in veh.frames_rx]
    assert len(frames) >= TP.NEUTRAL_BURST_FRAMES
    assert all(p == NEUTRAL_PULSES for p in frames[-TP.NEUTRAL_BURST_FRAMES:])
    doc = json.loads((out / ART.FILENAME).read_text())
    assert doc["all_pass"] is False and "ABORTED" in doc["operator_notes"]
    assert any(m["verdict"] == "FAIL" and m["n_samples_pos"] == 0 for m in doc["motors"]), "un-probed motors FAIL"
    assert ART.load_and_check(out / ART.FILENAME, SIGMA, MOT, allow_dry_run=True)[0] is False
    run = json.loads((out / "probe_run.json").read_text())
    assert run["exit_code"] == 3 and expect in run["abort_reason"]


def test_refused_before_any_frame_when_the_script_is_absent_or_a_direction_differs(tmp_path):
    code, pr, veh, out = _run(tmp_path, "A", {"script_running": False})
    assert code == 2 and "RLPWM" in pr.abort_reason
    assert veh.frames_rx == [] and pr.link.n_frames == 0, "nothing on RC 9..16 before the gate holds"
    assert not (out / ART.FILENAME).exists()
    code, pr, veh, out = _run(tmp_path, "A", {"params": {"MOT_6_DIRECTION": -1.0}})
    assert code == 2 and "spin reversed" in pr.abort_reason and veh.frames_rx == []
    code, pr, veh, out = _run(tmp_path, "A", {"batt_v0": 15.2})
    assert code == 2 and "15.5" in pr.abort_reason and veh.frames_rx == []
    # the script answers but never takes over (SCR_USER1 0 on the vehicle, say): neutral frames went out, then abort
    code, pr, veh, out = _run(tmp_path, "A", {"script_enabled": False})
    assert code == 3 and ("did not take over" in pr.abort_reason or "not forcing the outputs" in pr.abort_reason)
    assert veh.frames_rx and all(p == NEUTRAL_PULSES for _, p in veh.frames_rx), "only neutral ever left"


def test_parse_seq():
    assert TP.parse_seq("A") == ["A"] and TP.parse_seq("a,b,e") == ["A", "B", "E"] and TP.parse_seq("CAD") == ["C", "A", "D"]
    with pytest.raises(ValueError):
        TP.parse_seq("A,F")


# =============================================================================
# 5. the artefact reader the engage gate imports
# =============================================================================
def test_load_and_check_accepts_the_dry_run_artefact_only_when_told_and_refuses_every_mismatch(tmp_path):
    code, pr, veh, out = _run(tmp_path, "A")
    path = out / ART.FILENAME
    ok, why = ART.load_and_check(path, SIGMA, MOT)
    assert not ok and "dry_run" in why, "a synthetic vehicle is not a measurement: refused by default"
    assert ART.load_and_check(path, SIGMA, MOT, allow_dry_run=True) == (True, "ok")
    assert ART.load_and_check(path, SIGMA, MOT, allow_dry_run=True, max_age_s=3600.0)[0] is True
    doc = json.loads(path.read_text())

    def variant(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d)
        p = tmp_path / "variant.json"
        p.write_text(json.dumps(d))
        return ART.load_and_check(p, SIGMA, MOT, allow_dry_run=True)

    def refused(mutate, word):
        ok, why = variant(mutate)
        assert not ok and word in why, (word, why)

    refused(lambda d: d["export"].__setitem__("sigma", [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0]), "sigma")
    refused(lambda d: d["vehicle"].__setitem__("mot_direction", [-1, -1, 1, 1, 1, 1, 1, -1]), "mot_direction")
    refused(lambda d: d.__setitem__("all_pass", False), "all_pass")
    refused(lambda d: d["motors"][2].update(verdict="WEAK", mean_rate_pos_dps=-3.0), "T3 WEAK")   # a consistent WEAK
    refused(lambda d: d["motors"][2].__setitem__("verdict", "WEAK"), "does not follow")        # an inconsistent one
    refused(lambda d: d["motors"][0].__setitem__("mean_rate_pos_dps", +20.0), "does not follow")  # tampered PASS (T1 expects -)
    refused(lambda d: d["motors"][0].__setitem__("n_samples_pos", 0), "does not follow")
    refused(lambda d: d.__setitem__("schema", 2), "schema")
    refused(lambda d: d.__setitem__("tool", "something/else.py"), "tool")
    refused(lambda d: d.__setitem__("motors", d["motors"][:7]), "all 8")
    refused(lambda d: d["motors"][4].__setitem__("motor", 9), "in order")
    # a consistently flipped motor 5 (sign AND rates): only the geometry recomputed from the stored B_flu catches it
    refused(lambda d: d["motors"][4].update(expected_sign_pos=+1, mean_rate_pos_dps=9.0, mean_rate_neg_dps=-8.0),
            "recomputed from the stored B_flu")
    refused(lambda d: d["motors"][4].__setitem__("expected_sign_pos", +1), "does not follow")
    refused(lambda d: d.__setitem__("pulse_us", 50), "pulse_us")
    assert variant(lambda d: d.__setitem__("written", "2026-01-01T00:00:00+0900")) == (True, "ok"), "no max_age: any age"
    d_old = json.loads(json.dumps(doc))
    d_old["written"] = "2026-01-01T00:00:00+0900"
    p = tmp_path / "old.json"
    p.write_text(json.dumps(d_old))
    ok, why = ART.load_and_check(p, SIGMA, MOT, allow_dry_run=True, max_age_s=24 * 3600.0)
    assert not ok and "old" in why
    assert not ART.load_and_check(tmp_path / "missing.json", SIGMA, MOT)[0]
    (tmp_path / "junk.json").write_text("{not json")
    assert "JSON" in ART.load_and_check(tmp_path / "junk.json", SIGMA, MOT)[1]
    # the mismatch refusals come BEFORE the dry-run refusal is lifted? no: dry_run is judged first, then the rest
    ok, why = ART.load_and_check(path, [-s for s in SIGMA], MOT)
    assert not ok and "dry_run" in why


def test_artifact_verdict_rules():
    v = ART.verdict
    assert v(+1, 12.0, -9.0, 10, 10) == "PASS" and v(-1, -12.0, 9.0, 10, 10) == "PASS"
    assert v(+1, 3.0, -9.0, 10, 10) == "WEAK" and v(+1, 12.0, -9.0, 2, 10) == "WEAK"
    assert v(+1, -12.0, 9.0, 10, 10) == "FAIL" and v(+1, 12.0, 9.0, 10, 10) == "FAIL"
    assert v(+1, 0.0, -9.0, 10, 10) == "FAIL" and v(+1, 12.0, -9.0, 0, 10) == "FAIL"
    assert v(+1, None, -9.0, 10, 10) == "FAIL" and v(+1, float("nan"), -9.0, 10, 10) == "FAIL"
    rec = ART.motor_record(5, "p", -1, -9.2, 8.1, 10, 10)
    assert rec["verdict"] == "PASS" and rec["motor"] == 5
    assert ART.summary({"motors": [rec]}).startswith("T5 p-1 PASS")


# =============================================================================
# 6. sequences B, C, D, E and the free decay on the synthetic vehicle
# =============================================================================
def test_sequences_b_c_d_e_produce_their_artefacts_with_the_synthetic_numbers(tmp_path):
    code, pr, veh, out = _run(tmp_path, "BCDE")
    assert code == 0, pr.abort_reason
    nv = json.loads((out / "nv_leak.json").read_text())
    assert len(nv["passes"]) == 2 and all(len(p["steps"]) == 4 for p in nv["passes"])
    reg = nv["pooled_regression"]
    assert reg["findings"] == [] and abs(reg["leak_n_per_100us_down_positive"]) < 0.5
    assert abs(reg["leak_pitch_nm_per_100us"]) < 0.05 and abs(reg["leak_roll_nm_per_100us"]) < 0.05
    assert reg["batt_a_per_100us"] > 2.0, "the common mode costs current while producing no wrench"
    last = nv["passes"][0]["steps"][-1]
    assert last["cv_us"] == 300 and 20.0 < last["batt_a_max"] < 30.0 and last["batt_v_min"] > 15.0
    assert max(p for s in nv["passes"][0]["steps"] for p in s["pulses"]) <= 1800
    st = json.loads((out / "stiffness.json").read_text())
    assert all(s["sign_as_expected"] is True for s in st["steps"])
    assert st["k_pitch_nm_per_rad"] == pytest.approx(veh.k, rel=0.3)
    assert st["k_roll_nm_per_rad"] == pytest.approx(veh.k, rel=0.3)
    esc = json.loads((out / "esc_latency.json").read_text())
    assert esc["motor"] == TP.ESC_MOTOR_DEFAULT and len(esc["cases"]) == 3
    assert all(len(c["repeats"]) == 3 for c in esc["cases"])
    pen = esc["from_rest_penalty_s"]
    assert 0.08 <= pen <= 0.35, pen                     # the synthetic ESC start-up delay is 0.12 s (+ the 5 deg/s crossing)
    assert esc["verdict"].endswith("REAL")
    assert esc["cases"][0]["delay_servo_mean_s"] < esc["cases"][0]["delay_cmd_mean_s"], "the servo report removes the lag"
    rest = json.loads((out / "rest_trim.json").read_text())
    assert rest["net_weight_sign"] == "neutral within resolution" and rest["n_att"] > 1000
    assert rest["batt_a_mean"] == pytest.approx(veh.hotel_a, abs=0.05) and abs(rest["pitch_mean_deg"]) < 0.5
    run = json.loads((out / "probe_run.json").read_text())
    assert set(run["results"]) >= {"B", "C", "D", "E"}


def test_rest_trim_reads_the_net_weight_sign_and_the_resting_trim(tmp_path):
    code, pr, veh, out = _run(tmp_path, "E", {"net_weight_n": 1.0, "depth0_m": 0.3}, rest_s=20.0)
    assert code == 0, pr.abort_reason
    rest = json.loads((out / "rest_trim.json").read_text())
    assert rest["net_weight_sign"] == "heavy (sinks)" and rest["depth_rate_mps"] > 0.01
    assert rest["net_weight_n_est_down_positive"] > 0


def test_free_decay_fit_recovers_the_synthetic_period_and_brackets_k(tmp_path):
    code, pr, veh, out = _run(tmp_path, "C", free_decay_s=30.0, tilt=(10.0, 15.0))
    assert code == 0, pr.abort_reason
    fd = json.loads((out / "free_decay.json").read_text())
    for axis, I in (("pitch", veh.I[1]), ("roll", veh.I[0])):
        f = fd["axes"][axis]
        wn = math.sqrt(veh.k / I)
        td = 2.0 * math.pi / (wn * math.sqrt(1.0 - veh.zeta ** 2))
        assert f["n_extrema"] >= 3, (axis, f)
        assert f["period_s"] == pytest.approx(td, rel=0.15), (axis, f["period_s"], td)
        assert f["k_lo_nm_per_rad"] <= veh.k * 1.25 and f["k_hi_nm_per_rad"] >= veh.k * 0.8, (axis, f)
    assert all(p == NEUTRAL_PULSES for _, p in veh.frames_rx), "the free-decay recorder only ever sends neutral"


def test_decay_fit_on_an_analytic_damped_sinusoid():
    wn, zeta = 2.6, 0.12
    wd = wn * math.sqrt(1 - zeta ** 2)
    ts = np.arange(0.0, 30.0, 0.04)
    ys = 2.0 + 15.0 * np.exp(-zeta * wn * ts) * np.cos(wd * ts)
    f = TP.decay_fit(ts, ys)
    assert f["period_s"] == pytest.approx(2 * math.pi / wd, rel=0.03)
    assert f["zeta"] == pytest.approx(zeta, abs=0.04)
    assert f["omega_n_rad_s"] == pytest.approx(wn, rel=0.05)
    assert f["rest_deg"] == pytest.approx(2.0, abs=0.3)
    assert TP.decay_fit(ts, np.full_like(ts, 2.0))["period_s"] is None


def test_nv_hold_calibration_interpolates_todays_ballast(tmp_path):
    """A 4.5 N heavy vehicle (the peg-in-jaw case, 2.5-5.4 N [측정 REPORT.md §1]): neutral sinks, the 41 us pattern
    lifts -> a hold level between the two, the sweep rides on it and the rest between steps is the hold itself."""
    code, pr, veh, out = _run(tmp_path, "B", {"net_weight_n": 4.5, "depth0_m": 0.4}, nv_passes=1, nv_hold_us="auto")
    nv = json.loads((out / "nv_leak.json").read_text()) if (out / "nv_leak.json").exists() else None
    assert nv is not None, pr.abort_reason
    cal = nv["passes"][0]["hold_calibration"]
    assert cal["rate_neutral_mps"] > 0 > cal["rate_pattern_mps"]
    assert TP.NV_HOLD_MIN_US <= nv["passes"][0]["hold_us"] < 60 and code == 0, (cal, pr.abort_reason)
    assert any(s.name == "rest_hold" for s in pr.steps)
    assert nv["pooled_regression"]["findings"] == []
    # a light vehicle: the interpolated level is below the deadband -> the pattern is switched off, not faked
    code, pr, veh, out = _run(tmp_path, "B", {"net_weight_n": 0.5, "depth0_m": 0.4}, nv_passes=1, nv_hold_us="auto")
    nv = json.loads((out / "nv_leak.json").read_text())
    assert nv["passes"][0]["hold_us"] == 0 and "deadband" in nv["passes"][0]["hold_calibration"]["note"]
