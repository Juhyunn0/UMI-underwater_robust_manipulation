#!/usr/bin/env python3
"""test_rl_velocity_source.py — R7 (2026-10-16): an RL export that SAYS what velocity it was trained on.

RL_controller pwm11att13 and later carry two obs_spec blocks the station did not read before (RL_controller
scripts/export_policy.py): ``velocity_source`` (trained on the DVL-primary blend: hold / fade / gross / stale, no DVL
heave) and ``obs_guard`` (trained behind the pwm LOWs' observation guard: velocity slew, fix jump, thin = n_tags <
min_tags). The layout gate cannot tell such an export apart — its slot is v_b_est(3) like every other's — so these
blocks are the only thing that can. Pinned here (RL_controller logs/eval/dvl_trip_fix_20261015 SPEC.md §8):

    gate     rl_policy.check_velocity_source / check_obs_guard accept an export only when THIS station implements
             the same thing (dvl.selection primary, equal constants, the guard on the same values) and refuse it
             with a message naming the difference; an export WITHOUT the blocks is accepted exactly as before (N23)
    note     the engage-time line (rl_velocity_note) reads the export (N24)
    build    the followers run the gate at build (N25)
    engage   a blend-trained export flies only on the TAG estimator (N26)

(N41 — the RL_PWM export picker refusing a blend-trained export under legacy — waits for the picker merge.)

    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest -q rov_gui/tests/test_rl_velocity_source.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui.control import dvl as DV                         # noqa: E402
from rov_gui.control import rl_policy as RP                   # noqa: E402

POLICIES = ROOT / "rov_gui" / "control" / "rl_policies"
#: The blocks an att13 export carries [스펙: RL_controller logs/eval/dvl_trip_fix_20261015/map/sim_twin_spec.md §6a,
#: generated from the running pwm11att13 env.yaml by export_policy._velocity_blocks].
ATT13 = {
    "estimator": {"vel_lp_alpha": 0.6, "propagate": True, "stale_s": 0.5, "min_fix_dt": 0.02},
    "velocity_source": {
        "trained": True, "modes": ["tag_fd", "dvl_primary", "switching"], "p": [0.15, 0.7, 0.15],
        "hold_s": 0.45, "xfade_s": 0.5, "gross_m_s": 0.5, "gross_tau_s": 1.0, "stale_s": 0.6, "dvl_w": False,
        "switching": {"reentry_s": 0.5, "innov_max_m_s": 0.15},
        "dvl_input": "body-origin velocity, FRD, mount-rotated and lever-arm corrected (dvl.DvlAid v_body)",
        "rule": "newest DVL sample if usable and within gross of LP_1s(tag FD); else hold; else fade"},
    "obs_guard": {"trained": True, "vel_slew_mps2": 2.0, "jump_m": 0.15,
                  "rules": "rl_pwm.PwmObsGuard 1 and 2",
                  "thin_proxy": "training: the env's low-tag spell stands for n_tags < min_tags"},
}
GUARD = {"vel_slew_mps2": 2.0, "jump_m": 0.15, "min_tags": 2, "age_s": 0.3, "hold_after_s": 0.2,
         "release_after_s": 0.2, "hold_wrench_z_n": -4.0}


def _att13(**over) -> dict:
    s = copy.deepcopy(ATT13)
    for k, v in over.items():
        blk, key = k.split("__")
        if v is ...:
            s[blk].pop(key)
        else:
            s[blk][key] = v
    return s


# ======================================================================== gate
def test_rl_velocity_source_gate():
    """N23."""
    vs, og = RP.check_velocity_source, RP.check_obs_guard
    # (a) no blocks: accepted exactly as before
    assert vs({}, "rl_pwm", "p", {}, 0.6) is None
    assert og({}, "rl_pwm", "p", GUARD) is None and og({}, "rl", "p", None) is None
    # (b) trained, the station primary with the same constants: the block
    s = _att13()
    assert vs(s, "rl_pwm", "p", {}, 0.6) is s["velocity_source"]
    assert vs(s, "rl_pwm", "p", DV.resolve({"enabled": True}), 0.6) is s["velocity_source"]  # a resolved block
    # (c) the station runs legacy
    with pytest.raises(ValueError, match="dvl.selection: legacy"):
        vs(s, "rl_pwm", "p", {"selection": "legacy"}, 0.6)
    # (d) a constant the station does not implement — named on both sides
    with pytest.raises(ValueError, match=r"velocity_source\.hold_s 0\.4.*dvl\.blend_hold_s"):
        vs(_att13(velocity_source__hold_s=0.4), "rl_pwm", "p", {}, 0.6)
    with pytest.raises(ValueError, match=r"dvl\.stale_s"):
        vs(s, "rl_pwm", "p", {"stale_s": 0.5}, 0.6)            # stale_s is also the blend's clock
    with pytest.raises(ValueError, match="not a number"):
        vs(_att13(velocity_source__gross_m_s=...), "rl_pwm", "p", {}, 0.6)
    with pytest.raises(ValueError, match="not a number"):
        vs(_att13(velocity_source__xfade_s=True), "rl_pwm", "p", {}, 0.6)
    # (e) the DVL's heave
    with pytest.raises(ValueError, match="never feeds the DVL's w"):
        vs(_att13(velocity_source__dvl_w=True), "rl_pwm", "p", {}, 0.6)
    # (f) a block that does not say trained true / false
    with pytest.raises(ValueError, match=r"not \{trained"):
        vs(_att13(velocity_source__trained="yes"), "rl_pwm", "p", {}, 0.6)
    with pytest.raises(ValueError, match=r"not \{trained"):
        vs({"velocity_source": [1, 2]}, "rl_pwm", "p", {}, 0.6)
    # (g) trained behind the guard, flown on a LOW without one
    with pytest.raises(ValueError, match="only rl_pwm / rl_pwm_7d"):
        og(s, "rl", "p", None)
    # (h) the guard on the trained values: accepted; another slew: refused, named
    assert og(s, "rl_pwm", "p", dict(GUARD)) is s["obs_guard"]
    with pytest.raises(ValueError, match=r"rl_pwm\.obs_guard\.vel_slew_mps2"):
        og(s, "rl_pwm", "p", dict(GUARD, vel_slew_mps2=3.0))
    with pytest.raises(ValueError, match=r"rl_pwm\.obs_guard\.jump_m"):
        og(s, "rl_pwm", "p", dict(GUARD, jump_m=0.2))
    # (i) trained: false — returned, no error, whatever the station runs
    s2 = _att13(velocity_source__trained=False, obs_guard__trained=False)
    assert vs(s2, "rl_pwm", "p", {"selection": "legacy"}, 0.5) is s2["velocity_source"]
    assert og(s2, "rl", "p", None) is s2["obs_guard"]
    # (j) the tag-velocity low-pass it was trained on
    with pytest.raises(ValueError, match="vel_lp_alpha"):
        vs(_att13(estimator__vel_lp_alpha=0.5), "rl_pwm", "p", {}, 0.6)
    with pytest.raises(ValueError, match="vel_lp_alpha"):
        vs(s, "rl_pwm", "p", {}, None)                          # unknown station side
    # (k) the thin-fix definition the jump rule was trained on
    with pytest.raises(ValueError, match="n_tags < 2"):
        og(s, "rl_pwm", "p", dict(GUARD, min_tags=3))
    with pytest.raises(ValueError, match="n_tags < 3"):
        og(_att13(obs_guard__min_tags=3), "rl_pwm", "p", dict(GUARD))
    assert og(_att13(obs_guard__min_tags=3), "rl_pwm", "p", dict(GUARD, min_tags=3))
    # every refusal is an ExportRefused (a ValueError the worker logs at warn)
    with pytest.raises(RP.ExportRefused):
        vs(s, "rl_pwm", "p", {"selection": "legacy"}, 0.6)
    assert RP.station_velocity_source({}, 0.6) == {
        "selection": "primary", "hold_s": 0.45, "xfade_s": 0.5, "gross_m_s": 0.5, "gross_tau_s": 1.0,
        "stale_s": 0.6, "dvl_w": False, "vel_lp_alpha": 0.6}


# ======================================================================== note
def test_the_engage_note_reads_the_export():
    """N24. The shipped warning (every export before 2026-10-16) is unchanged; an export trained on the blend is told
    it flies the blend it was trained on — or, not armed, that it flies on the tag velocity, inside its training but at
    tag-FD accuracy; a non-TAG estimator never gets the "flies the blend" line (it is refused at engage)."""
    vs = ATT13["velocity_source"]
    shipped = ("warn", "ctrl: rl_pwm was trained on a simulated tag-finite-difference velocity, not a DVL's — this run "
                       "feeds it a different observation distribution (meta dvl.use_at_engage)")
    assert RP.rl_velocity_note("rl_pwm", None, True, "", "tag") == shipped
    assert RP.rl_velocity_note("rl_pwm", {"trained": False}, True, "", "tag") == shipped
    assert RP.rl_velocity_note("rl_pwm", None, False, "DVL is OUT (shadow)", "tag") is None
    lvl, text = RP.rl_velocity_note("rl_pwm", vs, True, "", "tag")
    assert lvl == "info" and text == ("ctrl: rl_pwm flies the DVL-primary blend it was trained on (hold 0.45 s, "
                                      "fade 0.5 s, gross 0.5 m/s)")
    lvl, text = RP.rl_velocity_note("rl_pwm_7d", vs, False, "DVL not armed at engage (x)", "tag")
    assert lvl == "warn" and "trained mostly on the DVL-primary blend (70 % of its episodes)" in text
    assert "NOT armed for the DVL (DVL not armed at engage (x))" in text and "(15 % tag-FD episodes)" in text
    assert text.endswith("but at tag-FD accuracy")
    lvl, text = RP.rl_velocity_note("rl_pwm", vs, False, "no DVL built (dvl.enabled false / no --dvl)", "tag")
    assert lvl == "warn" and "no DVL built" in text
    no_p = {k: v for k, v in vs.items() if k != "p"}
    assert "%" not in RP.rl_velocity_note("rl_pwm", no_p, False, "x", "tag")[1]
    for est in ("ekf", "imu_dr"):
        for armed in (True, False):
            assert RP.rl_velocity_note("rl_pwm", vs, armed, "", est) is None


# ======================================================================= build
def _export(tmp, src="pwm11k0.2_s0", **blocks) -> str:
    """A temp policy dir: a shipped export with the att13 blocks (or the given ones) added to its obs_spec."""
    d = Path(tmp) / f"{src}_att13"
    shutil.copytree(POLICIES / src, d)
    spec = json.loads((d / "obs_spec.json").read_text())
    spec.update(copy.deepcopy(blocks or ATT13))
    (d / "obs_spec.json").write_text(json.dumps(spec))
    return str(d)


def test_hw_followers_check_the_export():
    """N25."""
    from rov_gui.control.geometry import MpcConfig

    with tempfile.TemporaryDirectory() as tmp:
        pdir = _export(tmp)
        cfg = MpcConfig()
        cfg.rl_pwm = {"policy_dir": pdir, "pwm_cap": 1.0, "require_thruster_probe": False}
        r = RP.HwRlPwm(cfg, log=lambda m: None)              # primary (the default): builds
        assert r.velocity_source["trained"] is True and r.obs_guard_trained["trained"] is True
        m = r.meta()
        assert m["velocity_source"]["export"]["hold_s"] == 0.45
        assert m["velocity_source"]["station"]["selection"] == "primary"
        assert m["obs_guard_trained"]["vel_slew_mps2"] == 2.0
        cfg.dvl = {"selection": "legacy"}
        with pytest.raises(ValueError, match="dvl.selection: legacy"):
            RP.HwRlPwm(cfg, log=lambda m: None)
        cfg.dvl = {}
        cfg.rl_pwm["obs_guard"] = {"vel_slew_mps2": 3.0}
        with pytest.raises(ValueError, match=r"obs_guard\.vel_slew_mps2"):
            RP.HwRlPwm(cfg, log=lambda m: None)
        # an axes LOW has no observation guard: an export trained behind one is refused there
        adir = _export(tmp, "traj12_s2", obs_guard=ATT13["obs_guard"])
        cfg = MpcConfig()
        cfg.rl = {"policy_dir": adir}
        with pytest.raises(ValueError, match="only rl_pwm / rl_pwm_7d"):
            RP.HwRl(cfg, log=lambda m: None)
    # every SHIPPED export carries neither block and builds as before
    cfg = MpcConfig()
    cfg.rl_pwm = {"allow_caps_below_trained": True, "require_thruster_probe": False}
    r = RP.HwRlPwm(cfg, log=lambda m: None)
    assert r.velocity_source is None and r.obs_guard_trained is None
    assert r.meta()["velocity_source"] == {"export": None, "station": None}
    assert RP.HwRl(MpcConfig(), log=lambda m: None).velocity_source is None
    for d in sorted(POLICIES.iterdir()):
        spec = json.loads((d / "obs_spec.json").read_text())
        assert "velocity_source" not in spec and "obs_guard" not in spec, d.name


# ====================================================================== engage
def test_an_export_trained_on_the_blend_flies_only_on_the_tag_estimator():
    """N26. The blend-trained export saw ONLY the assembler's blend as its velocity: EKF or imu_dr control is refused
    at engage with the reason; the TAG estimator engages, and with no DVL built the engage line says it flies on the
    tag velocity."""
    from rov_gui.tests import test_rl_pwm_interlocks as PI

    with tempfile.TemporaryDirectory() as tmp:
        pdir = _export(tmp, "pwm10_s1")          # the harness sink's export (cap 1.0)
        w, bus, pwms, logs, events = PI._pwm_worker(tmp)
        try:
            w.cfg.rl_pwm = {"policy_dir": pdir, "pwm_cap": 1.0, "require_thruster_probe": False}
            w._rl_pwm = RP.HwRlPwm(w.cfg, log=lambda m: None)
            w.set_mode("rl_pwm")
            assert w.ctrl is w._rl_pwm and w.ctrl.velocity_source["trained"] is True
            w.on_enable(True)
            for est, attr in (("EKF", "ekf_control"), ("imu_dr", "dr_control")):
                setattr(w, attr, True)
                PI._feed(w)
                w.set_engaged(True)
                assert not w.engaged, est
                assert f"ESTIMATOR {est} CONTROL would hand it a different velocity" in w.reason, w.reason
                setattr(w, attr, False)
            PI._engage(w)                                # TAG
            assert w.engaged
            notes = [m for lvl, m in logs if "DVL-primary blend" in m]
            assert notes and "no DVL built" in notes[-1] and "trained mostly" in notes[-1], notes
        finally:
            w.teardown()
