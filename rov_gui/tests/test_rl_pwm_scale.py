#!/usr/bin/env python3
"""test_rl_pwm_scale.py — the 2026-10-10 changes to the pwm LOWs (rl_pwm / rl_pwm_7d), offline.

Source: RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md §1 cause 4 ("deployment cap as a
per-thruster CLIP at 0.25 with a cap-1.0 policy -> allocation collapse on 10-01") and §3 step 5 ("station: cap as a
thrust-space SCALE, refuse a config cap below the trained cap, null monitor, predicted current, probe block").

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_rl_pwm_scale.py -q -p no:cacheprovider

What is pinned:

1. control/rl_pwm.T200CurveNp.inverse — the closed-form inverse round-trips the curve within 1e-6 N over the whole
   attainable range of both branches; saturation, the deadband edge and the sub-edge rules.
2. control/rl_pwm.scale_throttles — one scalar on the thrust vector: max|throttle| == cap within 1e-6, direction kept
   (cos 1 where no thruster is pushed under its deadband edge; measured worst case otherwise), the two recorded pool
   patterns exact, the per-thruster clip demonstrably worse; the strict conversion and the sink's window.
3. control/rl_policy.HwRlPwm — scale mode by default (clip for an A/B), the refusal below the trained cap and its
   recorded opt-out (rl_pwm_7d keeps its own key), the config loader's new keys, meta.
4. control/rl_pwm.ThrusterProbeCheck / PwmGate — the sign-probe artefact gate: missing reader, no file, wrong sigma,
   all_pass false, dry_run, too old, other vehicle directions each refuse with the expected reason; a PASS artefact
   (built with the probe tool's own schema module) passes the gate, the sink and the worker.
5. workers.MpcWorker — engage refused by the probe gate; CSV schema 19 columns i_pred_a / null_v_us / null_h_us on
   pwm rows (empty on a pid row; schema 20 appended pwm_guard, test_rl_pwm_interlocks.py), the predicted current
   (physical thrust convention since 2026-10-10 b) and null-space arithmetic on the recorded att7pwm_s0 pattern.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control import rl_pwm as R                                                       # noqa: E402
from rov_gui.control.rl_pwm import (CAP_TOL, NEUTRAL_PULSES, PWM_CAP_MODES, PwmGate, PwmModel,  # noqa: E402
                                    ThrusterProbeCheck, cap_span_us, null_space_us, parse_rl_pwm_block,
                                    predicted_pack_current_a, pulses_within_cap, scale_throttles, throttles_to_pwm)
from rov_gui.state import Conn, PwmCommand, Telemetry, now                                   # noqa: E402

PWM10 = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
MOT = [-1, -1, 1, 1, -1, 1, 1, -1]
SIGMA = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
#: The two recorded pool patterns, mean pulse deviation T1..T8 [us] [측정 REPORT.md §1 cause 3 / scratch hw-decomp]:
#: pwm10_s1 on 2026-10-01 (n_h loaded, no vertical co-contraction) and att7pwm_s0 (n_v +250..+290 us, 26-44 A).
POOL_PWM10 = [80, 70, -85, -90, 70, -92, 25, -79]
POOL_ATT7 = [-115, -125, 125, 120, 260, 220, 200, 270]


def _model() -> PwmModel:
    spec = json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))
    return PwmModel(spec["output"]["pwm_model"])


def _cos(B, F0, F1) -> float:
    w0, w1 = B @ F0, B @ F1
    return float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1)))


# =============================================================================
# 1. the inverse curve
# =============================================================================
def test_inverse_round_trips_the_t200_curve_within_1e6_n_on_both_branches():
    c = _model().curve
    ep, en = c.edge_thrust()
    fp, fn = c.max_thrust()
    # the shipped fit's corners [유도 obs_spec.json output.pwm_model.t200]: +1.44 / -1.10 N at the deadband edge,
    # +64.13 / -51.55 N at rpm_max (the "64 N trained" of KNOWN_ISSUES)
    assert 1.4 < ep < 1.5 and -1.2 < en < -1.0 and 64.0 < fp < 64.3 and -51.7 < fn < -51.4
    for lo, hi in ((ep, fp), (fn, en)):
        F = np.linspace(lo, hi, 20001)
        assert np.max(np.abs(c.thrust(c.inverse(F)) - F)) < 1e-6, (lo, hi)
    # ...and from the throttle side, everything outside the deadband and below rpm saturation comes straight back
    for thr in (np.linspace(c.deadband + 1e-6, 0.97, 5001), np.linspace(-0.97, -c.deadband - 1e-6, 5001)):
        assert np.max(np.abs(c.inverse(c.thrust(thr)) - thr)) < 1e-9
    # beyond rpm_max the inverse is the smallest throttle that reaches the saturated thrust
    assert abs(float(c.thrust(c.inverse([fp + 10.0]))[0]) - fp) < 1e-9
    assert abs(float(c.thrust(c.inverse([fn - 10.0]))[0]) - fn) < 1e-9
    assert float(c.inverse([fp])[0]) < 0.98 and float(c.thrust(1.0)) == fp, "the fit saturates before throttle 1"
    # 0 -> 0; a sub-edge thrust -> the deadband edge with its sign (default) or 0 ("zero")
    assert c.inverse([0.0])[0] == 0.0
    e = c.inverse([0.5, -0.5])
    assert e[0] > c.deadband and e[1] < -c.deadband
    assert abs(float(c.thrust(e)[0]) - ep) < 1e-6 and abs(float(c.thrust(e)[1]) - en) < 1e-6
    assert np.array_equal(c.inverse([0.5, -0.5, 0.0], below_edge="zero"), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        c.inverse([float("nan")])
    with pytest.raises(ValueError):
        c.inverse([1.0], below_edge="clip")


# =============================================================================
# 2. the thrust-space scale
# =============================================================================
@pytest.mark.parametrize("cap", [0.25, 0.5])
def test_scale_puts_the_max_at_the_cap_and_keeps_the_direction_on_10k_random_vectors(cap):
    """10^4 uniform vectors in [-1, 1]^8 plus the two pool patterns. Where the scale pushes no thruster under its
    deadband edge the direction is EXACT (cos 1); where it does, the "stay on at the edge" rule (T200CurveNp.inverse)
    bends it — measured worst case pinned below as a regression number [측정: this test, seed 20261010], and the
    per-thruster clip of 2026-09-30..10-09 is worse on average at both caps."""
    m = _model()
    B = m.B_flu
    ep, en = m.curve.edge_thrust()
    rng = np.random.default_rng(20261010)
    vecs = np.vstack([rng.uniform(-1.0, 1.0, size=(10000, 8)),
                      np.array(POOL_PWM10) / 400.0, np.array(POOL_ATT7) / 400.0])
    n_scaled = n_exact = 0
    worst_any, cos_scale, cos_clip = 1.0, [], []
    for thr in vecs:
        out, s = scale_throttles(thr, cap, m, return_scale=True)
        assert np.all(np.isfinite(out)) and float(np.max(np.abs(out))) <= cap + CAP_TOL
        if float(np.max(np.abs(thr))) <= cap:
            assert s == 1.0 and np.array_equal(out, thr), "nothing exceeds the cap: untouched"
            continue
        n_scaled += 1
        assert 0.0 < s < 1.0 and abs(float(np.max(np.abs(out))) - cap) <= 1e-6, (s, out)
        F, Fo = m.curve.thrust(thr), m.curve.thrust(out)
        cs = _cos(B, F, Fo)
        Fs = s * F
        if not (np.any((Fs > 0) & (Fs < ep)) or np.any((Fs < 0) & (Fs > en))):
            n_exact += 1
            assert cs > 0.999, cs                                 # exact regime: B F' = s B F
            assert np.max(np.abs(Fo - Fs)) < 1e-6
        worst_any = min(worst_any, cs)
        cos_scale.append(cs)
        cos_clip.append(_cos(B, F, m.curve.thrust(np.clip(thr, -cap, cap))))
    assert n_scaled > 0.95 * len(vecs) and n_exact > 400
    assert float(np.mean(cos_scale)) > float(np.mean(cos_clip)) + 0.05
    # the measured worst cases: 0.9815 at cap 0.5; 0.46 at cap 0.25, where a cap-1.0 policy's full-throttle vector
    # (s ~ 0.09) lands seven thrusters under the edge and the edge rule holds them at +-1.4 / -1.1 N — the deadband's
    # doing, not the scale's, and the regime the refusal below the trained cap keeps off the vehicle
    assert worst_any > (0.97 if cap >= 0.5 else 0.40), worst_any


def test_scale_is_exact_on_the_recorded_pool_patterns_and_beats_the_clip():
    m = _model()
    B = m.B_flu
    att7 = np.array(POOL_ATT7) / 400.0
    out, s = scale_throttles(att7, 0.5, m, return_scale=True)
    assert 0.55 < s < 0.60 and abs(float(np.max(np.abs(out))) - 0.5) <= 1e-6
    assert abs(_cos(B, m.curve.thrust(att7), m.curve.thrust(out)) - 1.0) < 1e-9, "all eight stay above the edge"
    assert np.all(np.sign(out) == np.sign(att7)), "no thruster changes sense"
    assert _cos(B, m.curve.thrust(att7), m.curve.thrust(np.clip(att7, -0.5, 0.5))) < 0.999, "the clip bends it"
    pwm10 = np.array(POOL_PWM10) / 400.0
    out, s = scale_throttles(pwm10, 0.25, m, return_scale=True)
    assert s == 1.0 and np.array_equal(out, pwm10), "the 10-01 pattern never exceeded 0.25: untouched"
    out, s = scale_throttles(pwm10, 0.2, m, return_scale=True)
    assert 0.8 < s < 0.85 and abs(float(np.max(np.abs(out))) - 0.2) <= 1e-6
    assert abs(_cos(B, m.curve.thrust(pwm10), m.curve.thrust(out)) - 1.0) < 1e-9
    assert out[6] == 0.0, "T7 at +25 us was inside the deadband (no thrust): 0 under a scale"
    # the rpm-saturated zone (the fit saturates at +0.971 / -0.992 throttle [유도 obs_spec t200 rpm fits]): a cap above
    # both changes only the command, never the thrust
    sat = np.array([1.0, -1.0, 0.0, 0.0, 0.997, -0.997, 0.5, -0.5])
    out, s = scale_throttles(sat, 0.995, m, return_scale=True)
    assert s == 1.0 and float(np.max(np.abs(out))) == 0.995 and np.allclose(m.curve.thrust(out), m.curve.thrust(sat))
    # ...while a cap between the two saturation throttles scales the negative branch only, by its thrust ratio
    out, s = scale_throttles(sat, 0.98, m, return_scale=True)
    assert 0.97 < s < 0.98 and abs(float(np.max(np.abs(out))) - 0.98) <= 1e-6
    # a cap inside the deadband can only mean neutral
    assert np.array_equal(scale_throttles(att7, 0.05, m), np.zeros(8))
    for bad in (np.full(8, np.nan), np.zeros(7)):
        with pytest.raises(ValueError):
            scale_throttles(bad, 0.5, m)
    with pytest.raises(ValueError):
        scale_throttles(att7, 0.0, m)


def test_the_strict_conversion_and_the_sink_window_refuse_instead_of_clipping():
    thr = np.array([0.25, -0.25, 0.2500000001, -0.2500000001, 0.0, 0.1, -0.1, 0.25])
    assert throttles_to_pwm(thr, 0.25, strict=True) == (1600, 1400, 1600, 1400, 1500, 1540, 1460, 1600)
    with pytest.raises(ValueError, match="refused, not clipped"):
        throttles_to_pwm(np.array([0.26, 0, 0, 0, 0, 0, 0, 0]), 0.25, strict=True)
    assert throttles_to_pwm(np.array([0.26, 0, 0, 0, 0, 0, 0, 0]), 0.25)[0] == 1600, "the lenient form still clips"
    assert cap_span_us(0.25) == 100 and cap_span_us(0.5) == 200 and cap_span_us(1.0) == 400
    assert pulses_within_cap((1600, 1400, 1500, 1500, 1500, 1500, 1500, 1500), 0.25)
    assert not pulses_within_cap((1601, 1400, 1500, 1500, 1500, 1500, 1500, 1500), 0.25)
    assert not pulses_within_cap((1500,) * 7, 0.25) and not pulses_within_cap(("x",) * 8, 0.25)


# =============================================================================
# 3. the follower
# =============================================================================
def _cfg(**rl_pwm):
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig()
    cfg.rl_pwm = dict(rl_pwm)
    return cfg


def test_hw_rl_pwm_scales_by_default_clips_only_for_an_ab_and_refuses_an_overshoot():
    from rov_gui.control.rl_policy import HwRlPwm

    cap = 0.5
    base = {"pwm_cap": cap, "allow_caps_below_trained": True, "require_thruster_probe": False}
    r = HwRlPwm(_cfg(**base), log=lambda m: None)
    assert r.pwm_cap_mode == "scale" and r.pwm_cap == cap and r.pwm_cap_config == cap
    rc = HwRlPwm(_cfg(**base, pwm_cap_mode="clip"), log=lambda m: None)
    assert rc.pwm_cap_mode == "clip"
    for o in (r, rc):
        o.set_target_ned([1.0, 0.0, -0.5], 0.0)             # a far goal: the policy saturates
    eta, nu = np.zeros(6), np.zeros(6)
    saw_scale = False
    for k in range(12):
        u, info = r.step(eta, nu, nu, 0.05 * k)
        uc, infoc = rc.step(eta, nu, nu, 0.05 * k)
        pw, pwc = np.array(info["pwm_us"]), np.array(infoc["pwm_us"])
        assert info["status"] == 0 and infoc["status"] == 0
        assert np.max(np.abs(pw - 1500)) <= 400 * cap and np.max(np.abs(pwc - 1500)) <= 400 * cap
        assert info["pwm_cap_mode"] == "scale" and infoc["pwm_cap_mode"] == "clip" and infoc["pwm_scale"] == 1.0
        if info["pwm_scale"] < 1.0:
            saw_scale = True
            assert np.max(np.abs(pw - 1500)) == 400 * cap, "the binding thruster sits exactly at the cap"
            # the same action through the clip: every thruster beyond the cap pinned, the pattern changed
            raw = rc.model.throttles(rc.last_action)
            assert np.array_equal(pwc, throttles_to_pwm(np.clip(raw, -cap, cap), cap))
        r.note_applied_pwm(info["pwm_us"])
        rc.note_applied_pwm(infoc["pwm_us"])
    assert saw_scale, "a far goal must have driven the policy past the cap at least once"
    assert r.meta()["pwm_cap_mode"] == "scale" and rc.meta()["pwm_cap_mode"] == "clip"
    json.dumps(r.meta())
    # a scaled vector that still exceeds the cap is a bug upstream: refused -> neutral + n_fail, never clipped
    r.reset()
    r.set_target_ned([1.0, 0.0, -0.5], 0.0)
    import rov_gui.control.rl_pwm as rl_pwm_mod
    orig = rl_pwm_mod.scale_throttles
    rl_pwm_mod.scale_throttles = lambda thr, c, model, return_scale=False: (np.full(8, 0.6), 0.9)
    try:
        u, info = r.step(eta, nu, nu, 1.0)
    finally:
        rl_pwm_mod.scale_throttles = orig
    assert info["pwm_us"] == NEUTRAL_PULSES and info["status"] == 1 and r.n_fail == 1 and np.allclose(u, 0.0)
    assert math.isnan(info["pwm_scale"])


def test_a_config_cap_below_the_trained_cap_is_refused_unless_recorded_per_low():
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d
    from rov_gui.tests.test_rl_pwm_7d import _export

    with pytest.raises(ValueError, match=r"rl_pwm\.pwm_cap 0\.25 is below the policy's trained cap 1\.00.*"
                                         r"rl_pwm\.allow_caps_below_trained"):
        HwRlPwm(_cfg(pwm_cap=0.25), log=lambda m: None)
    with pytest.raises(ValueError, match="below the policy's trained cap"):
        HwRlPwm(_cfg(pwm_cap=0.999), log=lambda m: None)
    r = HwRlPwm(_cfg(pwm_cap=0.25, allow_caps_below_trained=True), log=lambda m: None)
    assert r.pwm_cap == 0.25 and r.caps_below_trained is True and r.allow_caps_below_trained is True
    m = r.meta()
    assert m["caps_below_trained"] is True and m["allow_caps_below_trained"] is True and m["pwm_cap_config"] == 0.25
    r = HwRlPwm(_cfg(pwm_cap=1.0), log=lambda m: None)
    assert r.caps_below_trained is False and r.meta()["caps_below_trained"] is False
    # rl_pwm_7d records its OWN opt-out (2026-10-03 key): the transport's key does not stand in for it
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(pwm_cap=0.25, allow_caps_below_trained=True)
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp))}
        with pytest.raises(ValueError, match=r"rl_pwm_7d\.allow_caps_below_trained"):
            HwRlPwm7d(cfg, log=lambda m: None)
        cfg.rl_pwm = {"pwm_cap": 0.25}
        cfg.rl_pwm_7d = {"policy_dir": str(_export(tmp)), "allow_caps_below_trained": True}
        assert HwRlPwm7d(cfg, log=lambda m: None).caps_below_trained is True


def test_the_config_loader_knows_the_new_keys_their_defaults_and_their_types():
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.tests.test_rl_pwm import _set_rl_pwm_key

    d = parse_rl_pwm_block({})
    assert d == {"pwm_cap": 0.25, "pwm_cap_mode": "scale", "allow_caps_below_trained": False,
                 "require_thruster_probe": True, "thruster_probe_path": None, "thruster_probe_max_age_days": 30.0}
    assert PWM_CAP_MODES == ("scale", "clip")
    for bad in ({"pwm_cap_mode": "clamp"}, {"pwm_cap_mode": 1}, {"require_thruster_probe": 1},
                {"allow_caps_below_trained": "yes"}, {"thruster_probe_max_age_days": 0},
                {"thruster_probe_max_age_days": True}, {"thruster_probe_path": ""}, {"thruster_probe_path": 3}):
        with pytest.raises(ValueError, match="rl_pwm\\."):
            parse_rl_pwm_block(bad)
    shipped = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml").rl_pwm
    blk = parse_rl_pwm_block(shipped)
    assert blk["pwm_cap_mode"] == "scale" and blk["require_thruster_probe"] is True
    assert blk["allow_caps_below_trained"] is False and blk["thruster_probe_path"] is None
    assert blk["thruster_probe_max_age_days"] == 30.0
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")

    def load(text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(text)
        try:
            return MpcConfig.load(fh.name)
        finally:
            os.unlink(fh.name)

    assert load(_set_rl_pwm_key(src, "pwm_cap_mode", "clip")).rl_pwm["pwm_cap_mode"] == "clip"
    with pytest.raises(ValueError, match="pwm_cap_mode"):
        load(_set_rl_pwm_key(src, "pwm_cap_mode", "clamp"))
    with pytest.raises(ValueError, match="thruster_probe_max_age_days"):
        load(_set_rl_pwm_key(src, "thruster_probe_max_age_days", "-1"))
    with pytest.raises(ValueError, match="unknown key"):
        load(_set_rl_pwm_key(src, "thruster_probe_max_age_day", "3"))


# =============================================================================
# 4. the sign-probe artefact gate
# =============================================================================
def _artifact_module():
    try:
        from rov_gui.control import thruster_probe_artifact as tpa
    except ImportError:
        return None
    return tpa


def _good_doc(sigma=SIGMA, mot=MOT, dry_run=False, written=None, all_pass=True) -> dict:
    """A schema-1 artefact that PASSES the reader's own checks, built with the probe tool's schema module (the
    verdicts must follow from the stored numbers and the axis/sign from the stored B_flu, so the module that owns
    them builds the fixture). None when that module is not in this checkout."""
    tpa = _artifact_module()
    if tpa is None:
        return None
    m = _model()
    B = [[float(v) for v in row] for row in m.B_flu]
    motors = []
    for i in range(8):
        axis, sign = tpa.expected_axis_sign(B, i)
        rate = 20.0 if (all_pass or i != 5) else 1.0        # motor 6 WEAK when all_pass is False
        motors.append(tpa.motor_record(i + 1, axis, sign, sign * rate, -sign * rate, 8, 8))
    return tpa.build(dry_run=dry_run, sysid=1, fw="4.5.0", mot_direction=mot, policy_dir=str(PWM10), sigma=sigma,
                     B_flu=B, obs_spec_sha1="deadbeef", motors=motors,
                     written=written or time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def _write(doc: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / R.THRUSTER_PROBE_FILENAME
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _probe_root(tmp) -> Path:
    return Path(tmp) / "data"


def test_probe_check_refuses_everything_short_of_a_fresh_passing_artefact_for_this_vehicle_and_export():
    with tempfile.TemporaryDirectory() as tmp:
        root = _probe_root(tmp)
        chk = ThrusterProbeCheck(SIGMA, None, 30.0, data_root=str(root))
        ok, why = chk.check(MOT)
        if _artifact_module() is None:
            assert not ok and "not importable" in why
            pytest.skip("rov_gui/control/thruster_probe_artifact.py is not in this checkout yet")
        assert not ok and "no artefact" in why and "thruster_probe" in why
        chk.forget()
        chk2 = ThrusterProbeCheck(SIGMA, str(root / "nowhere" / "x.json"), 30.0, data_root=str(root))
        assert "does not exist" in chk2.check(MOT)[1]
        folder = root / "20261010" / "1010_120000_probe"

        def verdict(doc, sigma=SIGMA, mot=MOT, max_age=30.0, path=None):
            p = _write(doc, folder)
            c = ThrusterProbeCheck(sigma, path or str(p), max_age, data_root=str(root))
            return c.check(mot)

        assert verdict(_good_doc()) == (True, "ok")
        assert "probed for sigma" in verdict(_good_doc(sigma=[-s for s in SIGMA]))[1]
        assert "all_pass is false" in verdict(_good_doc(all_pass=False))[1]
        assert "DRY-RUN" in verdict(_good_doc(dry_run=True))[1]
        old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S")
        why = verdict(_good_doc(written=old))[1]
        assert "days old" in why and "thruster_probe_max_age_days 30" in why
        assert verdict(_good_doc(written=old), max_age=60.0) == (True, "ok"), "the age limit is the config's"
        future = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
        assert "FUTURE" in verdict(_good_doc(written=future))[1]
        assert "not an ISO time" in verdict(_good_doc(written="yesterday"))[1]
        assert "MOT_1..8_DIRECTION" in verdict(_good_doc(), mot=[1] * 8)[1]
        assert "MOT_1..8_DIRECTION" in verdict(_good_doc(mot=[1] * 8))[1]
        bad = _good_doc()
        bad["schema"] = 2
        assert "schema" in verdict(bad)[1]
        bad = _good_doc()
        bad["motors"][2]["verdict"] = "PASS"
        bad["motors"][2]["mean_rate_pos_dps"] = 0.0
        assert not verdict(bad)[0], "a hand-edited verdict is the READER's catch (load_and_check has the last word)"
        (folder / R.THRUSTER_PROBE_FILENAME).write_text("{not json", encoding="utf-8")
        assert "unreadable" in ThrusterProbeCheck(SIGMA, str(folder / R.THRUSTER_PROBE_FILENAME), 30.0,
                                                  data_root=str(root)).check(MOT)[1]
        st = ThrusterProbeCheck(SIGMA, None, 30.0, data_root=str(root)).state()
        assert st["required"] is True and st["ok"] is None and st["max_age_days"] == 30.0
        json.dumps(st)


def test_probe_check_finds_the_newest_artefact_caches_its_verdict_and_refuses_without_its_reader(monkeypatch):
    if _artifact_module() is None:
        pytest.skip("rov_gui/control/thruster_probe_artifact.py is not in this checkout yet")
    with tempfile.TemporaryDirectory() as tmp:
        root = _probe_root(tmp)
        old_p = _write(_good_doc(dry_run=True), root / "20261009" / "1009_090000_probe")
        new_p = _write(_good_doc(), root / "20261010" / "1010_100000_probe")
        os.utime(old_p, (1.0, 1.0))                                # unmistakably the older file
        chk = ThrusterProbeCheck(SIGMA, None, 30.0, data_root=str(root))
        assert Path(chk.resolve_path()) == new_p
        assert chk.check(MOT, now=100.0) == (True, "ok") and chk.state()["path"] == str(new_p)
        # cached for THRUSTER_PROBE_REFRESH_S: deleting the file is noticed only after that (or on forget)
        new_p.unlink()
        assert chk.check(MOT, now=105.0) == (True, "ok")
        assert not chk.check(MOT, now=100.0 + R.THRUSTER_PROBE_REFRESH_S + 0.1)[0]
        _write(_good_doc(), root / "20261010" / "1010_100000_probe")
        assert not chk.check(MOT, now=100.0 + R.THRUSTER_PROBE_REFRESH_S + 1.0)[0], "still the cached refusal"
        chk.forget()
        assert chk.check(MOT, now=100.0 + R.THRUSTER_PROBE_REFRESH_S + 1.0) == (True, "ok")
        # a different direction set is its own question (never the cached answer)
        assert "MOT_1..8_DIRECTION" in chk.check([1] * 8, now=100.0 + R.THRUSTER_PROBE_REFRESH_S + 1.0)[1]
        # no reader module: refused with the module named (never "ok", never an exception)
        monkeypatch.setitem(sys.modules, "rov_gui.control.thruster_probe_artifact", None)
        chk.forget()
        ok, why = chk.check(MOT)
        assert not ok and "thruster_probe_artifact.py is not importable" in why
    with pytest.raises(ValueError, match="no 8-value sigma"):
        ThrusterProbeCheck(None, None, 30.0)
    with pytest.raises(ValueError, match="thruster_probe_max_age_days"):
        ThrusterProbeCheck(SIGMA, None, 0.0)


def test_the_vehicle_gate_carries_the_probe_item_after_the_direction_item():
    from rov_gui.tests.test_rl_pwm import _good_gate

    if _artifact_module() is None:
        pytest.skip("rov_gui/control/thruster_probe_artifact.py is not in this checkout yet")
    t = 100.0
    with tempfile.TemporaryDirectory() as tmp:
        root = _probe_root(tmp)
        p = _write(_good_doc(), root / "20261010" / "1010_100000_probe")
        g = _good_gate(t)
        assert g.check(t) == (True, "ok") and g.state(t)["thruster_probe"] == {"required": False}
        g.thruster_probe = ThrusterProbeCheck(SIGMA, str(p), 30.0, data_root=str(root))
        assert g.check(t) == (True, "ok")
        st = g.state(t)["thruster_probe"]
        assert st["required"] is True and st["ok"] is True and st["path"] == str(p)
        g.thruster_probe = ThrusterProbeCheck(SIGMA, str(root / "none.json"), 30.0, data_root=str(root))
        ok, why = g.check(t)
        assert not ok and why.startswith("thruster probe:") and "does not exist" in why
        # the direction item comes first: a reversed MOT_n_DIRECTION is named, not the probe
        g.note_param("MOT_4_DIRECTION", -MOT[3], t)
        assert "spin reversed" in g.check(t)[1]
        assert PwmGate(MOT, 255, thruster_probe=None).thruster_probe is None


def test_the_sink_refuses_pulses_until_the_probe_passes():
    from rov_gui.tests.test_rl_pwm import PW, _Master, _answer_all, _motors, _set_rl_pwm_cap, _set_rl_pwm_key

    if _artifact_module() is None:
        pytest.skip("rov_gui/control/thruster_probe_artifact.py is not in this checkout yet")
    from rov_gui.backends.hardware import MavlinkCommandSink
    from rov_gui.qt import QtWidgets
    import types

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        root = _probe_root(tmp)
        good = _write(_good_doc(), root / "20261010" / "1010_100000_probe")
        dry = _write(_good_doc(dry_run=True), root / "20261010" / "1010_110000_probe")

        def sink(path):
            text = _set_rl_pwm_key(_set_rl_pwm_cap(src, "0.25"), "thruster_probe_path", json.dumps(str(path)))
            text = _set_rl_pwm_key(text, "require_thruster_probe", "true")
            cfg_path = Path(tmp) / f"cfg_{path.parent.name}.yaml"
            cfg_path.write_text(text, encoding="utf-8")
            log = types.SimpleNamespace(lines=[])
            log.emit = lambda lvl, msg: log.lines.append((lvl, msg))
            bus = types.SimpleNamespace(log=log, payload=types.SimpleNamespace(emit=lambda *_: None))
            opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552", mpc_config=str(cfg_path))
            s = MavlinkCommandSink(bus, opts)
            s.master = _Master(True)
            s.listening = True
            s._setup_pwm_gate()
            assert s._pwm_gate is not None, s._pwm_gate_error
            assert s._pwm_gate.thruster_probe is not None and s._pwm_cap_mode == "scale"
            _answer_all(s)
            s.enabled = True
            return s, log

        s, log = sink(dry)
        s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
        assert s.master.mav.rc == [] and not s._pwm_active
        assert any(lvl == "error" and "thruster probe" in m and "DRY-RUN" in m for lvl, m in log.lines)
        s._pwm_tick(time.monotonic())
        st = s.pwm_gate_state()
        assert st["ok"] is False and "thruster probe" in st["why"] and st["thruster_probe"]["required"] is True
        assert st["pwm_cap_mode"] == "scale"
        s, log = sink(good)
        s.set_pwm(PwmCommand(pulses=PW, source="mpc"))
        assert len(s.master.mav.rc) == 1 and _motors(s.master.mav.rc[0]) == PW and s._pwm_active


# =============================================================================
# 5. the worker: engage gate, CSV columns, meta
# =============================================================================
def _feed(w, gate, batt_v=None):
    from rov_gui.tests.test_control import _fix, _imu

    t = now()
    w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, t))
    w.on_vehicle_imu(_imu(depth=0.8, t=t))
    w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE, pwm_gate=gate, battery_v=batt_v))


def _worker_with(tmp, **rl_pwm):
    from rov_gui.control.rl_policy import HwRlPwm
    from rov_gui.tests.test_control import _worker

    w, bus, pilots, logs = _worker(tmp)
    w.cfg.rl_pwm = {"pwm_cap": 0.25, "allow_caps_below_trained": True, **rl_pwm}
    w._rl_pwm = HwRlPwm(w.cfg, log=lambda m: None)
    w._rl_pwm_error = None
    pwms = []
    bus.cmd_pwm.connect(pwms.append)
    w.set_mode("rl_pwm")
    assert w.cfg.mode == "rl_pwm" and getattr(w.ctrl, "transport", "") == "pwm", w.reason
    return w, bus, pilots, pwms, logs


def test_worker_refuses_to_engage_without_a_passing_probe_and_engages_with_one():
    from rov_gui.tests.test_rl_pwm import _gate

    with tempfile.TemporaryDirectory() as tmp:
        root = _probe_root(tmp)
        # the default: required, nothing probed -> refused with the reason the operator can act on
        w, bus, pilots, pwms, logs = _worker_with(tmp, thruster_probe_path=str(root / "none.json"))
        w.on_enable(True)
        _feed(w, _gate(True))
        w.set_engaged(True)
        assert not w.engaged and "thruster probe:" in w.reason and "does not exist" in w.reason, w.reason
        assert [p for p in pwms if p.pulses != NEUTRAL_PULSES] == []
        w.teardown()
        if _artifact_module() is None:
            pytest.skip("rov_gui/control/thruster_probe_artifact.py is not in this checkout yet")
        dry = _write(_good_doc(dry_run=True), root / "20261010" / "1010_100000_probe")
        w, bus, pilots, pwms, logs = _worker_with(tmp, thruster_probe_path=str(dry))
        w.on_enable(True)
        _feed(w, _gate(True))
        w.set_engaged(True)
        assert not w.engaged and "DRY-RUN" in w.reason
        w.teardown()
        good = _write(_good_doc(), root / "20261010" / "1010_110000_probe")
        w, bus, pilots, pwms, logs = _worker_with(tmp, thruster_probe_path=str(good))
        w.on_enable(True)
        _feed(w, _gate(True))
        w.set_engaged(True)
        assert w.engaged, w.reason
        m = w._run_meta()
        c = m["controller"]
        assert m["schema_version"] == 21 and c["pwm_cap_mode"] == "scale" and c["require_thruster_probe"] is True
        assert c["thruster_probe"]["ok"] is True and c["thruster_probe"]["path"] == str(good)
        assert c["caps_below_trained"] is True and c["allow_caps_below_trained"] is True
        w.teardown()
        # the recorded opt-out: no artefact needed, and meta says so
        w, bus, pilots, pwms, logs = _worker_with(tmp, require_thruster_probe=False)
        w.on_enable(True)
        _feed(w, _gate(True))
        w.set_engaged(True)
        assert w.engaged, w.reason
        c = w._run_meta()["controller"]
        assert c["require_thruster_probe"] is False and c["thruster_probe"] == {"required": False}
        w.teardown()


def test_the_csv_carries_the_predicted_current_and_null_space_of_the_sent_pulses_on_pwm_rows_only():
    from rov_gui.control.workers import CSV_HEADER
    from rov_gui.tests.test_control import _last_csv_row
    from rov_gui.tests.test_rl_pwm import _gate

    cols = CSV_HEADER.strip().split(",")
    # schema 21 (2026-10-15) appended the three attitude-age columns after these four (test_att_age pins them)
    assert cols[-7:-3] == ["i_pred_a", "null_v_us", "null_h_us", "pwm_guard"] and len(cols) == len(set(cols))
    with tempfile.TemporaryDirectory() as tmp:
        w, bus, pilots, pwms, logs = _worker_with(tmp, require_thruster_probe=False)
        w.on_enable(True)
        _feed(w, _gate(True), batt_v=15.0)
        w.set_engaged(True)
        assert w.engaged, w.reason
        for _ in range(4):
            _feed(w, _gate(True), batt_v=15.0)
            w.tick()
        row = _last_csv_row(w)
        sent = pwms[-1].pulses
        assert sent != NEUTRAL_PULSES and len(sent) == 8
        nv, nh = null_space_us(sent)
        assert abs(float(row["null_v_us"]) - nv) < 0.006 and abs(float(row["null_h_us"]) - nh) < 0.006
        assert abs(float(row["i_pred_a"]) - predicted_pack_current_a(sent, w.ctrl.model, 15.0)) < 0.006
        assert float(row["i_pred_a"]) > R.I_PACK_IDLE_A
        # a gate-grace tick sends eight 1500s: the columns say so (2.1 A, 0, 0) — never a repeat of the last pulses
        _feed(w, _gate(False, "RLPWM heartbeat stale (2.4 s) — the Lua script stopped"), batt_v=15.0)
        w.tick()
        assert w.engaged and pwms[-1].pulses == NEUTRAL_PULSES
        row = _last_csv_row(w)
        assert float(row["i_pred_a"]) == R.I_PACK_IDLE_A and float(row["null_v_us"]) == 0.0 \
            and float(row["null_h_us"]) == 0.0
        w.disengage("test over")
        w.teardown()
        # an axes follower: EMPTY, not nan — there are no sent pulses to read them from
        from rov_gui.tests.test_control import _worker

        w, bus, pilots, logs = _worker(tmp)
        w.set_mode("pid")
        w.on_enable(True)
        _feed(w, None)
        w.set_engaged(True)
        assert w.engaged, w.reason
        _feed(w, None)
        w.tick()
        row = _last_csv_row(w)
        assert (row["i_pred_a"], row["null_v_us"], row["null_h_us"], row["pwm_guard"]) == ("", "", "", "")
        assert len(row) == len(cols)
        w.teardown()


# =============================================================================
# 6. the two monitors on the recorded pattern
# =============================================================================
def test_predicted_pack_current_uses_the_physical_thrust_convention_on_the_att7pwm_pattern():
    """2026-10-10 (b): F = T200(thr) * batt_v / 20.5 V — the fit's 1.0 is the ~20 V datasheet curve, 14.8 V -> 0.72
    = the export's voltage_scale [스펙 RL_controller rov_rl/thrusters.py; rov_env _nominal_voltage 0.72 and
    rewards.thruster_current_proxy use exactly this thrust]. Until then the curve was scaled by batt_v / 14.8, i.e.
    taken as the 14.8 V curve (41 A on this pattern). The physical proxy predicts ~24-30 A over 14-16.5 V on the
    att7pwm_s0 pool pattern against a 41.6 A MEASURED mean (202735, 1 Hz) — it under-predicts that co-contracting
    pattern by ~1.6x, which is why the pack interlock judges batt_a and not this column."""
    m = _model()
    pw = tuple(1500 + d for d in POOL_ATT7)
    for v in (14.0, 14.8, 15.0, 16.0, 16.5):
        i = predicted_pack_current_a(pw, m, v)
        assert 23.0 < i < 31.0, (v, i)
    assert 25.5 < predicted_pack_current_a(pw, m, 14.8) < 26.3          # 25.9 A [측정: this test]
    assert 28.5 < predicted_pack_current_a(pw, m, 16.0) < 29.3          # 28.9 A
    assert predicted_pack_current_a(pw, m, 16.0) > predicted_pack_current_a(pw, m, 14.8)
    # the fallback (no voltage reading): the export's 0.72 = the 14.8 V curve, so it sits BESIDE the 14.8 V value now
    fb = predicted_pack_current_a(pw, m, None)
    assert abs(fb - predicted_pack_current_a(pw, m, 14.8)) < 0.15 and fb == predicted_pack_current_a(pw, m, float("nan"))
    assert R.thrust_scale_at_voltage(None, m) == 0.72 and abs(R.thrust_scale_at_voltage(20.5, m) - 1.0) < 1e-12
    assert abs(R.thrust_scale_at_voltage(14.8, m) - 0.722) < 1e-3 and R.PACK_V_CURVE == 20.5
    assert predicted_pack_current_a(NEUTRAL_PULSES, m, 15.0) == R.I_PACK_IDLE_A
    # the fit's own spot check [measured REPORT scratch/design-systems/alloc_numbers.txt]: 26 N forward -> 8.4 A
    c, p = R.I_FIT_FWD
    assert abs(c * 26.0 ** p - 8.4) < 0.1 and abs(c * 2.0 ** p - 0.17) < 0.02
    assert R.I_FIT_REV == (0.0537, 1.673) and R.I_PACK_IDLE_A == 2.1 and R.PACK_V_NOMINAL == 14.8


def test_null_space_monitor_reads_the_two_wrench_free_directions():
    assert null_space_us(NEUTRAL_PULSES) == (0.0, 0.0)
    # pure n_v: all four verticals at the same +200 us -> (200, 0); pure n_h: front +100 / rear -100 -> (0, 100)
    assert null_space_us((1500, 1500, 1500, 1500, 1700, 1700, 1700, 1700)) == (200.0, 0.0)
    assert null_space_us((1600, 1600, 1400, 1400, 1500, 1500, 1500, 1500)) == (0.0, 100.0)
    # wrench-producing patterns have no null content: surge (+ on all four horizontals), heave (+,-,-,+ verticals)
    assert null_space_us((1700, 1700, 1700, 1700, 1500, 1500, 1500, 1500)) == (0.0, 0.0)
    assert null_space_us((1500, 1500, 1500, 1500, 1700, 1300, 1300, 1700)) == (0.0, 0.0)
    # ...and the two directions are exactly the null space of the export's B (rank 6)
    m = _model()
    for n in (np.array([0, 0, 0, 0, 1, 1, 1, 1.0]), np.array([1, 1, -1, -1, 0, 0, 0, 0.0])):
        assert np.max(np.abs(m.B_flu @ n)) < 1e-6              # the export stores B_flu as float32
    assert np.linalg.matrix_rank(m.B_flu) == 6
    # the recorded att7pwm_s0 pattern: +237.5 / -121.25 us (REPORT: c_v +250..+290, c_h -125..-140 across runs)
    nv, nh = null_space_us(tuple(1500 + d for d in POOL_ATT7))
    assert nv == 237.5 and nh == -121.25
    nv, nh = null_space_us(tuple(1500 + d for d in POOL_PWM10))
    assert -45 < nv < -15 and 75 < nh < 110, "pwm10_s1: n_h loaded, no vertical co-contraction (REPORT cause 3)"
    with pytest.raises(ValueError):
        null_space_us((1500,) * 7)


# =============================================================================
# runner
# =============================================================================
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            if getattr(fn, "pytestmark", None):
                print(f"  skip  {name}: parametrized — run under pytest")
                continue
            fn() if fn.__code__.co_argcount == 0 else print(f"  skip  {name}: needs a fixture — run under pytest")
            print(f"  ok    {name}")
        except pytest.skip.Exception as e:
            print(f"  skip  {name}: {e}")
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
