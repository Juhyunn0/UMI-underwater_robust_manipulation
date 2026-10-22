#!/usr/bin/env python3
"""test_rl_pwm_heave_trim.py — the heave trim of the per-thruster RL LOWs (rl_pwm / rl_pwm_7d), 2026-10-18, offline.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python -m pytest \\
        rov_gui/tests/test_rl_pwm_heave_trim.py -q -p no:cacheprovider

Why: the model-based followers carry a net-buoyancy trim (hw_mpc.yaml vehicle_net_buoyancy_n), the pwm RL LOWs had none,
and on 2026-10-16 a payload left rl_pwm on the floor [측정 data/20261016/1016_193523_diag_offline/cross_low/
cross_low_final_table.txt]. ``rl_pwm.heave_trim_n`` / ``rl_pwm_7d.heave_trim_n`` (physical N, + = up in the world, the
export's nominal thrust model) is rotated into the body frame by the estimated roll/pitch, allocated over the four
verticals like the obs-guard HOLD (rl_pwm.heave_trim_thrust), added in thrust space, and the cap acts on the TOTAL
(rl_pwm.trim_and_cap); the policy is fed back only its own part.

What is pinned:

1. trim 0 is the code of before, bit for bit: pulses, pwm_scale, status, guard and the fed-back prev_action against
   fixtures/rl_pwm_trim0_golden_20261018.json (recorded by ``record_golden`` below with rl_policy.py / rl_pwm.py at git
   d70f1150f, BEFORE the key existed — the fixture's meta carries their sha1), and against a frozen copy of the
   pre-trim step on random states, on the recorded 2026-10-17 pool ticks (when that CSV is on this machine), both
   LOWs, scale and clip;
2. trim +-N adds N cos(roll) cos(pitch) of body heave through the export model (within 1e-6 N; through the 1-us
   pulses within their resolution), zero roll / pitch / yaw moment and no horizontal force change, on both LOWs —
   WHERE every vertical total clears its deadband edge (the attainable range; item 8 pins what happens where not);
3. prev_action excludes the trim: a trimmed and an untrimmed follower fed the same states stay in LOCKSTEP (same
   network output every tick) while no scale binds, and the feedback is the policy's own part on the 1-us grid;
4. the cap scale acts on the TOTAL: s is computed on policy + trim, the binding thruster of the total sits at the
   cap, the total's direction is kept, the policy's part is T200^-1(s T200(thr_policy)); clip mode clips the total;
5. the tag-gate HOLD pulses are never trimmed;
6. the build refuses |trim| > 10 N, a bool, a string, null and non-finite values, naming the LOW's own key;
   rl_pwm_7d reads only its own block; the config loader knows both keys and the shipped value is 0.0;
7. meta controller.heave_trim_n + controller.heave_trim (level allocation, counters) and the one build line;
8. (review 2026-10-18 D-R1) the deadband: with the policy's verticals at rest every |trim| < 2.51 N up / 2.58 N down
   leaves as ONE edge pattern (~3.67 N, ~0.109 N*m pitch, whatever the value), checked against the edge thrusts from
   first principles; on the recorded 2026-10-17 tick a +3 N trim delivers ~2.45 N and a -0.083 N*m roll moment; the
   per-tick info / meta / build line say so;
9. (D-R2) the cap budget: a trim taking > 50 % of one vertical's capped range is refused at build (both LOWs), the
   numbers of heave_trim_budget, and the counters of a scale the trim alone made bind;
10. (D-R5) rl_pwm_7d notes (build line + meta) a transport-block rl_pwm.heave_trim_n it does not fly.

Re-record the golden fixture ONLY when an unrelated change has legitimately moved it (with no heave_trim_n anywhere):
    QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/rovgui-pose/bin/python rov_gui/tests/test_rl_pwm_heave_trim.py --record-golden
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rov_gui.control.rl_pwm import (FLU_TO_FRD, GUARD_HOLD, HEAVE_TRIM_MAX_CAP_FRACTION,  # noqa: E402
                                    HEAVE_TRIM_MAX_N, NEUTRAL_PULSES, heave_trim_at_rest, heave_trim_budget,
                                    heave_trim_thrust, parse_heave_trim_n, pwm_to_action, scale_throttles,
                                    throttles_to_pwm, trim_and_cap, vertical_allocation)

PWM10 = ROOT / "rov_gui" / "control" / "rl_policies" / "pwm10_s1"
GOLDEN = HERE / "fixtures" / "rl_pwm_trim0_golden_20261018.json"
#: the 2026-10-17 att14 RL_PWM_7D run at its trained cap 0.5 (165 s, no trip) — data/ is not in git: tests that read it
#: skip when it is absent
POOL_CSV = ROOT / "data" / "20261017" / "1017_204524" / "mpc_204524.csv"
ENVELOPE = {"trained": True, "roll_max_rad": 0.17, "pitch_max_rad": 0.35, "rate_max_rad_s": 0.35, "p_att": 0.7}
N_TICKS = 48
DT = 0.05


def _sha1(*paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _export7d(tmp) -> Path:
    """A FORMAT rl_pwm_7d export (test_rl_pwm_7d._export's recipe): pwm10_s1's real weights + spec with an
    attitude_reference injected — never a trained att policy."""
    d = Path(tmp) / "rlpwm7d_export"
    d.mkdir(exist_ok=True)
    shutil.copy(PWM10 / "policy_weights.npz", d / "policy_weights.npz")
    spec = json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))
    spec["attitude_reference"] = dict(ENVELOPE)
    (d / "obs_spec.json").write_text(json.dumps(spec))
    return d


def _cfg(rl_pwm: dict, rl_pwm_7d: dict | None = None):
    from rov_gui.control.geometry import MpcConfig

    cfg = MpcConfig()
    cfg.rl_pwm = dict(rl_pwm)
    if rl_pwm_7d is not None:
        cfg.rl_pwm_7d = dict(rl_pwm_7d)
    return cfg


#: The golden scenarios: (name, LOW, transport block, own block). Every block opts out of the sign-probe gate (offline,
#: recorded) and pwm10_s1 is a cap-1.0 export, so the 0.5 cap of the two rl_pwm rows needs the recorded opt-out.
SCENARIOS = (
    ("rl_pwm_scale", "rl_pwm", {"pwm_cap": 0.5, "allow_caps_below_trained": True, "require_thruster_probe": False}, None),
    ("rl_pwm_clip", "rl_pwm", {"pwm_cap": 0.5, "pwm_cap_mode": "clip", "allow_caps_below_trained": True,
                               "require_thruster_probe": False}, None),
    ("rl_pwm_7d_scale", "rl_pwm_7d", {"pwm_cap": 1.0, "require_thruster_probe": False}, {}),
    ("rl_pwm_7d_scale05", "rl_pwm_7d", {"pwm_cap": 0.5, "require_thruster_probe": False},
     {"allow_caps_below_trained": True}),
)
#: one seed per scenario (the states differ, so a bug that only shows on one pattern is not averaged away)
SEEDS = {"rl_pwm_scale": 20261018, "rl_pwm_clip": 20261019, "rl_pwm_7d_scale": 20261020, "rl_pwm_7d_scale05": 20261021}
#: cap 1.0, no opt-out: both LOWs on pwm10_s1's trained cap (the scale binds only in rpm saturation)
CAP1 = {"rl_pwm": ({"pwm_cap": 1.0, "require_thruster_probe": False}, None),
        "rl_pwm_7d": ({"pwm_cap": 1.0, "require_thruster_probe": False}, {})}


def _build(low: str, rl_pwm: dict, own: dict | None, tmp, extra_own: dict | None = None, log=lambda m: None):
    """The follower of one scenario; ``extra_own`` lands in the LOW's OWN block (rl_pwm for rl_pwm, rl_pwm_7d for
    rl_pwm_7d)."""
    from rov_gui.control.rl_policy import HwRlPwm, HwRlPwm7d

    extra_own = dict(extra_own or {})
    if low == "rl_pwm":
        return HwRlPwm(_cfg({**rl_pwm, **extra_own}), log=log)
    blk7 = {"policy_dir": str(_export7d(tmp)), **dict(own or {}), **extra_own}
    return HwRlPwm7d(_cfg(rl_pwm, blk7), log=log)


def scenario_ticks(seed: int, n: int = N_TICKS) -> dict:
    """Deterministic synthetic ticks (the golden fixture's inputs and every A/B below): a smooth wander around 0.75 m
    depth with a ~11 deg nose-up pitch (NED +0.20 rad — the pool's rest trim region), measurement noise, a thin fix
    (1 tag) over ticks 18..27 (-> the tag gate HOLDs after 0.2 s and releases 0.2 s after the tags return), a stale
    fix at 36..37, two ticks whose pulses never left (10, 11: note_applied_pwm(None), a coast), and a far goal (the
    policy saturates -> the thrust-space scale acts) replaced by a near one at tick 30."""
    rng = np.random.default_rng(int(seed))
    t = np.arange(n) * DT
    ph = rng.uniform(0.0, 2.0 * np.pi, 6)
    eta = np.stack([0.25 * np.sin(0.8 * t + ph[0]), 0.20 * np.sin(0.6 * t + ph[1]), 0.75 + 0.08 * np.sin(1.1 * t + ph[2]),
                    0.10 * np.sin(1.7 * t + ph[3]), 0.20 + 0.08 * np.sin(1.3 * t + ph[4]), 0.6 * np.sin(0.4 * t + ph[5])],
                   axis=1)
    eta = eta + rng.normal(0.0, 1.0, eta.shape) * np.array([0.004, 0.004, 0.004, 0.003, 0.003, 0.003])
    nu = np.gradient(eta, DT, axis=0) + rng.normal(0.0, 0.02, eta.shape)
    n_tags = np.full(n, 4, dtype=int)
    n_tags[18:28] = 1
    age = np.full(n, 0.05)
    age[36:38] = 0.5
    feed = np.ones(n, dtype=bool)
    feed[10:12] = False
    targets = {0: ([1.2, -0.4, 0.3], 0.5), 30: ([0.05, 0.02, 0.78], 0.1)}
    return {"eta": eta, "nu": nu, "n_tags": n_tags, "age": age, "feed": feed, "targets": targets}


def drive(r, ticks: dict, step=None, note=None) -> dict:
    """Run one follower over ``ticks``: note_fix -> step -> note_applied_pwm(the step's pulses, or None on a coast
    tick), recording what left and what the policy is fed back. ``step`` / ``note`` replace the follower's own
    methods (the frozen pre-trim copies below)."""
    step = r.step if step is None else step
    note = r.note_applied_pwm if note is None else note
    r.reset()
    out = {"pwm_us": [], "pwm_scale": [], "status": [], "guard": [], "prev_action": [], "throttle": [], "u": [],
           "action": []}
    z = np.zeros(6)
    for k in range(len(ticks["n_tags"])):
        if k in ticks["targets"]:
            p, yaw = ticks["targets"][k]
            r.set_target_ned(p, yaw)
        r.note_fix(int(ticks["n_tags"][k]), float(ticks["age"][k]))
        u, info = step(np.asarray(ticks["eta"][k], float), np.asarray(ticks["nu"][k], float), z, DT * k)
        note(info["pwm_us"] if bool(ticks["feed"][k]) else None)
        out["pwm_us"].append([int(p) for p in info["pwm_us"]])
        out["pwm_scale"].append(float(info["pwm_scale"]))
        out["status"].append(int(info["status"]))
        out["guard"].append(int(info["pwm_guard"]))
        out["prev_action"].append([float(v) for v in r._prev_action])
        out["throttle"].append([float(v) for v in info["throttle"]])
        out["u"].append([float(v) for v in np.asarray(u, float)])
        out["action"].append([float(v) for v in np.asarray(r.last_action, float)])
    return out


def record_golden(path: Path = GOLDEN) -> dict:
    """Write the golden fixture: every scenario driven by the follower AS IT IS NOW, with no heave_trim_n key at all.
    Run ONLY when an unrelated change has legitimately moved these numbers (the fixture is the 'current code' that
    trim 0 must reproduce bit for bit), and say so in the commit."""
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:                                        # noqa: BLE001
            return ""

    scen = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, low, blk, own in SCENARIOS:
            r = _build(low, blk, own, tmp)
            d = drive(r, scenario_ticks(SEEDS[name]))
            scen[name] = {k: d[k] for k in ("pwm_us", "pwm_scale", "status", "guard", "prev_action")}
            scen[name]["hold_pwm_us"] = [int(v) for v in r.hold_pwm_us]
    code = [ROOT / "rov_gui" / "control" / f for f in ("rl_policy.py", "rl_pwm.py")]
    doc = {"meta": {"what": "HwRlPwm / HwRlPwm7d pulses, pwm_scale, status, guard and fed-back prev_action on "
                            "test_rl_pwm_heave_trim.scenario_ticks, BEFORE rl_pwm.heave_trim_n existed",
                    "recorded": "2026-10-18",
                    "code_sha1": {p.name: _sha1(p) for p in code},
                    "code_git_head": _git("log", "-1", "--format=%h", "--", *[str(p) for p in code]),
                    "code_dirty": bool(_git("status", "--porcelain", "--", *[str(p) for p in code])),
                    "export_sha1": _sha1(PWM10 / "policy_weights.npz", PWM10 / "obs_spec.json"),
                    "numpy": np.__version__, "n_ticks": N_TICKS, "seeds": SEEDS},
           "scenarios": scen}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return doc


# =============================================================================
# the frozen pre-trim copy (HwRlPwm.step / note_applied_pwm at git d70f1150f, 2026-10-16)
# =============================================================================
def frozen_step(self, eta_ned, nu_ned, nudot_ned, t: float):
    """HwRlPwm.step as it was before the heave trim (git d70f1150f), verbatim but for the imports and ``self``."""
    from rov_gui.control.rl_pwm import CAP_TOL, GUARD_HOLD, GUARD_NONE, NEUTRAL_PULSES, scale_throttles, throttles_to_pwm
    t0 = time.perf_counter()
    scale = 1.0
    guard = GUARD_NONE
    try:
        eta_g, nu_g, guard = self.obs_guard.apply(eta_ned, nu_ned, self._fix_n_tags, self._fix_age_s)
        obs = self.observation(eta_g, nu_g, t)
        if not np.all(np.isfinite(obs)):
            raise ValueError("non-finite observation")
        a = self.policy.act(obs)
        thr = self.model.throttles(a)
        if self.pwm_cap_mode == "scale":
            thr, scale = scale_throttles(thr, self.pwm_cap, self.model, return_scale=True)
            if float(np.max(np.abs(thr))) > self.pwm_cap + CAP_TOL:
                raise ValueError("scaled throttles exceed the cap")
        else:
            thr = np.clip(thr, -self.pwm_cap, self.pwm_cap)
        if guard == GUARD_HOLD:
            thr = self.hold_throttles.copy()
            scale = 1.0
        pwm = throttles_to_pwm(thr, self.pwm_cap, strict=True)
        u = self.model.wrench_ned_from_throttles(thr)
        if not np.all(np.isfinite(u)):
            raise ValueError("non-finite wrench estimate")
        self.last_action = np.asarray(a, float).copy()
        status = 0
    except Exception:                                            # noqa: BLE001
        self.n_fail += 1
        thr = np.zeros(8)
        pwm = tuple(NEUTRAL_PULSES)
        u = np.zeros(6)
        status = 1
        scale = float("nan")
    self.last_pwm_scale = scale
    ms = 1e3 * (time.perf_counter() - t0)
    return u, {"w_hat": self.w_hat, "solve_ms": ms, "status": status, "n_fail": self.n_fail, "nis": 0.0,
               "pwm_us": pwm, "throttle": [float(v) for v in thr], "pwm_scale": scale,
               "pwm_cap_mode": self.pwm_cap_mode, "pwm_guard": int(guard),
               "guard_why": str(self.obs_guard.last.get("why", ""))}


def frozen_step_7d(self, eta_ned, nu_ned, nudot_ned, t: float):
    """HwRlPwm7d.step before the trim: the attitude-reference advance, then the frozen HwRlPwm.step."""
    self._att_decay()
    self._advance_rp_flown()
    return frozen_step(self, eta_ned, nu_ned, nudot_ned, t)


def frozen_note(self, pwm_us) -> None:
    pw = NEUTRAL_PULSES if pwm_us is None else pwm_us
    self.last_pwm_us = tuple(int(p) for p in pw)
    self._prev_action = pwm_to_action(pw, self.model.throttle_max)


def _drive_frozen(r, low: str, ticks: dict) -> dict:
    fs = frozen_step_7d if low == "rl_pwm_7d" else frozen_step
    return drive(r, ticks, step=lambda *a: fs(r, *a), note=lambda pw: frozen_note(r, pw))


def pool_ticks(n: int = 400, start: int = 200) -> dict:
    """Recorded ticks: rows of POOL_CSV (engaged), mapped onto the follower's NED inputs — world FLU position ->
    (x, -y, -z), roll as is, pitch / yaw negated, the CSV's NED-body nu_u/v/w, body rates from the attitude's finite
    difference, the CSV's reference each tick, its n_tags / tag_age_s as the fix quality. An A/B input only (both
    followers get the same ticks), so the mapping's exactness does not enter any assertion."""
    with open(POOL_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("engaged") == "1"][start:start + n]

    def f(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return float("nan")

    eta = np.array([[f(r, "px"), -f(r, "py"), -f(r, "pz"), math.radians(f(r, "roll_deg")),
                     math.radians(-f(r, "pitch_deg")), math.radians(-f(r, "yaw_deg"))] for r in rows])
    rates = np.gradient(np.unwrap(eta[:, 3:6], axis=0), DT, axis=0)
    nu = np.column_stack([[f(r, "nu_u") for r in rows], [f(r, "nu_v") for r in rows], [f(r, "nu_w") for r in rows],
                          rates])
    n_tags = np.array([int(f(r, "n_tags")) if math.isfinite(f(r, "n_tags")) else 0 for r in rows])
    age = np.array([f(r, "tag_age_s") for r in rows])
    targets = {k: ([f(r, "rx"), -f(r, "ry"), -f(r, "rz")], math.radians(-f(r, "ryaw_deg"))) for k, r in enumerate(rows)}
    return {"eta": eta, "nu": nu, "n_tags": n_tags, "age": age, "feed": np.ones(len(rows), dtype=bool),
            "targets": targets}


def _assert_same(a: dict, b: dict, what: str) -> None:
    assert a["pwm_us"] == b["pwm_us"], f"{what}: pulses differ"
    assert np.array_equal(np.array(a["prev_action"]), np.array(b["prev_action"])), f"{what}: prev_action differs"
    for k in ("status", "guard"):
        assert a[k] == b[k], f"{what}: {k} differs"
    for k in ("pwm_scale", "throttle", "u", "action"):
        assert np.array_equal(np.array(a[k]), np.array(b[k]), equal_nan=True), f"{what}: {k} differs"


def _pin_action(r, a) -> None:
    """Replace the network by a constant output (the arithmetic downstream of it is what these tests judge)."""
    a = np.asarray(a, float).copy()
    r.policy.act = lambda obs: a.copy()


def _wrench(model, thr) -> np.ndarray:
    return model.wrench_ned_from_throttles(np.asarray(thr, float))


# =============================================================================
# 1. trim 0 is the code of before
# =============================================================================
def test_trim_zero_reproduces_the_pre_change_golden_bit_for_bit():
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # recorded from COMMITTED code: d70f1150f = rl_policy.py / rl_pwm.py before the trim existed (update this pin, and
    # say why, if the golden is ever re-recorded — see the module docstring)
    assert g["meta"]["code_dirty"] is False and g["meta"]["code_git_head"] == "d70f1150f", g["meta"]
    assert g["meta"]["export_sha1"] == _sha1(PWM10 / "policy_weights.npz", PWM10 / "obs_spec.json"), \
        "pwm10_s1 changed since the golden was recorded — re-record it from the code before the trim (see docstring)"
    with tempfile.TemporaryDirectory() as tmp:
        for name, low, blk, own in SCENARIOS:
            G = g["scenarios"][name]
            assert len(G["pwm_us"]) == N_TICKS
            for extra in (None, {"heave_trim_n": 0.0}, {"heave_trim_n": 0}, {"heave_trim_n": -0.0}):
                logs = []
                r = _build(low, blk, own, tmp, extra_own=extra, log=logs.append)
                d = drive(r, scenario_ticks(SEEDS[name]))
                what = f"{name} with {extra}"
                assert d["pwm_us"] == G["pwm_us"], what
                assert np.array_equal(np.array(d["prev_action"]), np.array(G["prev_action"])), what
                assert d["status"] == G["status"] and d["guard"] == G["guard"], what
                assert np.max(np.abs(np.array(d["pwm_scale"]) - np.array(G["pwm_scale"]))) <= 1e-12, what
                assert [int(v) for v in r.hold_pwm_us] == G["hold_pwm_us"], what
                assert r.heave_trim_n == 0.0 and len(logs) == 1, (what, logs)        # no trim line at 0
                assert not any("heave trim" in m for m in logs)
    # the golden covers what it must: the scale binding, the clip, the HOLD and the coast ticks, on both LOWs
    sc = g["scenarios"]
    assert sum(s < 1.0 for s in sc["rl_pwm_scale"]["pwm_scale"]) >= 20
    assert sum(s < 1.0 for s in sc["rl_pwm_7d_scale05"]["pwm_scale"]) >= 20
    assert all(sc[n]["guard"].count(GUARD_HOLD) >= 5 for n in sc)


@pytest.mark.parametrize("name,low,blk,own", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_trim_zero_matches_the_frozen_pre_trim_step_on_random_states(name, low, blk, own):
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2, 3, 77, 20261018):
            ticks = scenario_ticks(seed)
            r_new = _build(low, blk, own, tmp, extra_own={"heave_trim_n": 0.0})
            r_old = _build(low, blk, own, tmp)
            a = drive(r_new, ticks)
            b = _drive_frozen(r_old, low, ticks)
            _assert_same(a, b, f"{name} seed {seed}")
        # a broken state is the same neutral + counted failure in both
        r_new, r_old = _build(low, blk, own, tmp), _build(low, blk, own, tmp)
        eta = np.zeros(6)
        eta[0] = float("nan")
        un, inew = r_new.step(eta, np.zeros(6), np.zeros(6), 0.0)
        fs = frozen_step_7d if low == "rl_pwm_7d" else frozen_step
        uo, iold = fs(r_old, eta, np.zeros(6), np.zeros(6), 0.0)
        assert inew["pwm_us"] == iold["pwm_us"] == NEUTRAL_PULSES and inew["status"] == iold["status"] == 1
        assert set(inew) == set(iold), "trim 0 adds no info keys"


@pytest.mark.skipif(not POOL_CSV.is_file(), reason=f"recorded pool run {POOL_CSV} not on this machine")
def test_trim_zero_matches_the_frozen_copy_on_recorded_pool_ticks_with_the_shipped_att14_export():
    """The shipped config's rl_pwm_7d (pwm11att14R_s1, cap 0.5 = trained) on 400 recorded ticks of its own 2026-10-17
    pool run: trim 0 == the frozen pre-trim step; and the same ticks with a +3 N trim stay inside the cap, apply the
    trim on every non-HOLD tick and never feed it back."""
    from rov_gui.control.geometry import MpcConfig
    from rov_gui.control.rl_policy import HwRlPwm7d

    cfg = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    if not str(cfg.rl_pwm_7d.get("policy_dir") or "").endswith("pwm11att14R_s1"):
        pytest.skip(f"the shipped rl_pwm_7d export is {cfg.rl_pwm_7d.get('policy_dir')!r}, not the run's att14")
    ticks = pool_ticks()
    assert len(ticks["n_tags"]) == 400
    a = drive(HwRlPwm7d(cfg, log=lambda m: None), ticks)
    b = _drive_frozen(HwRlPwm7d(cfg, log=lambda m: None), "rl_pwm_7d", ticks)
    _assert_same(a, b, "att14 on recorded ticks")
    assert sum(s == 0 for s in a["status"]) >= 390
    cfg.rl_pwm_7d = {**cfg.rl_pwm_7d, "heave_trim_n": 3.0}
    r = HwRlPwm7d(cfg, log=lambda m: None)
    infos = []
    orig = r.step

    def step(*args):
        u, info = orig(*args)
        infos.append(info)
        return u, info

    d = drive(r, ticks, step=step)
    cap = r.pwm_cap
    assert all(abs(p - 1500) <= round(400 * cap) for row in d["pwm_us"] for p in row)
    n_trim = 0
    for k, info in enumerate(infos):
        if info["status"] != 0:
            continue
        if info["pwm_guard"] == GUARD_HOLD:
            assert info["trim_thrust_n"] is None
            continue
        n_trim += 1
        eta = ticks["eta"][k]
        assert np.allclose(info["trim_thrust_n"], heave_trim_thrust(r.model, 3.0, eta[3], eta[4]), atol=1e-12)
        pol_us = throttles_to_pwm(info["throttle_policy"], 1.0)
        assert np.array_equal(d["prev_action"][k], pwm_to_action(pol_us, r.model.throttle_max))
    assert n_trim >= 300 and r.meta()["heave_trim"]["counts"]["ticks_trimmed"] == n_trim
    # (review 2026-10-18 D-R1) in the flown regime the deadband edge is the NORM: on these ticks most +3 N totals put a
    # vertical inside its gap (sent at the edge), and on average the trim delivers ~85 % of the heave asked for, with a
    # small systematic roll / pitch moment [유도: this open-loop replay, nominal model — hw_mpc.yaml rl_pwm.heave_trim_n
    # cites it]
    ht = r.meta()["heave_trim"]
    dl = ht["delivered"]
    assert dl["n_ticks"] == n_trim and ht["counts"]["ticks_edge"] >= 0.85 * n_trim, ht["counts"]
    frac = dl["mean_wrench_ned"][2] / dl["mean_requested_z_ned"]
    assert 0.80 < frac < 0.90, dl
    assert 0.01 < abs(dl["mean_wrench_ned"][3]) < 0.1 and 0.01 < abs(dl["mean_wrench_ned"][4]) < 0.1, dl
    # (D-R2) at the wire cap 0.25 a -4 N trim (about the 2026-10-17 jaw-empty value; budget 36 %, allowed) makes the scale
    # bind on ticks where the policy alone would not have — counted per engagement, never silent
    cfg.rl_pwm = {**cfg.rl_pwm, "pwm_cap": 0.25}
    cfg.rl_pwm_7d = {**cfg.rl_pwm_7d, "heave_trim_n": -4.0, "allow_caps_below_trained": True}
    r25 = HwRlPwm7d(cfg, log=lambda m: None)
    drive(r25, ticks)
    c25 = r25.meta()["heave_trim"]["counts"]
    assert 30 <= c25["ticks_scaled_by_trim"] <= 80 and c25["ticks_scaled_by_trim"] <= c25["ticks_scaled"], c25


def test_trim_and_cap_with_a_zero_trim_is_scale_throttles_bit_for_bit():
    """The library function on its own: a zero trim vector touches no thruster, so the total and the policy part are
    scale_throttles' output (scale) / np.clip (clip) exactly, s included — 10^4 uniform vectors at two caps."""
    from rov_gui.control.rl_pwm import PwmModel

    m = PwmModel(json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))["output"]["pwm_model"])
    rng = np.random.default_rng(181018)
    for cap in (0.25, 0.5, 1.0):
        for v in rng.uniform(-1.0, 1.0, size=(3000, 8)):
            ref, s_ref = scale_throttles(v, cap, m, return_scale=True)
            tot, pol, s = trim_and_cap(v, np.zeros(8), cap, m)
            assert np.array_equal(tot, ref) and np.array_equal(pol, ref) and s == s_ref
            tot, pol, s = trim_and_cap(v, np.zeros(8), cap, m, mode="clip")
            assert np.array_equal(tot, np.clip(v, -cap, cap)) and np.array_equal(pol, tot) and s == 1.0
    with pytest.raises(ValueError):
        trim_and_cap(np.zeros(8), np.full(8, np.nan), 0.5, m)
    with pytest.raises(ValueError):
        trim_and_cap(np.zeros(8), np.zeros(7), 0.5, m)
    with pytest.raises(ValueError):
        trim_and_cap(np.zeros(8), np.zeros(8), 0.5, m, mode="clamp")


# =============================================================================
# 2. what the trim adds
# =============================================================================
#: a synthetic action whose verticals sit well outside the deadband on both branches (|throttle| 0.30-0.35) and below
#: the cap, so every vertical total stays on the attainable range for |trim| <= 10 N
A_PIN = [0.12, -0.15, 0.10, -0.11, 0.35, -0.30, 0.32, -0.34]


@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_trim_adds_n_body_heave_through_the_export_model_with_no_moment(low):
    blk, own = CAP1[low]
    with tempfile.TemporaryDirectory() as tmp:
        r0 = _build(low, blk, own, tmp)
        _pin_action(r0, A_PIN)
        for trim in (3.0, -2.5, HEAVE_TRIM_MAX_N, -HEAVE_TRIM_MAX_N):
            r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": trim})
            assert r.heave_trim_n == trim and r.heave_trim_key == f"{low}.heave_trim_n"
            _pin_action(r, A_PIN)
            for roll, pitch in ((0.0, 0.0), (0.0, 0.2443), (0.08, -0.15), (-0.12, 0.26)):
                eta = np.array([0.1, -0.05, 0.7, roll, pitch, 0.4])
                for o in (r0, r):
                    o.reset()
                    o.set_target_ned([0.2, 0.0, 0.7], 0.4)
                    o.note_fix(5, 0.05)
                u0, i0 = r0.step(eta, np.zeros(6), np.zeros(6), 0.0)
                u1, i1 = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
                assert i0["status"] == i1["status"] == 0 and i0["pwm_scale"] == i1["pwm_scale"] == 1.0
                t0, t1 = np.array(i0["throttle"]), np.array(i1["throttle"])
                assert np.array_equal(t1[:4], t0[:4]), "the horizontals are untouched, bit for bit"
                k = math.cos(roll) * math.cos(pitch)
                dW = _wrench(r.model, t1) - _wrench(r.model, t0)
                assert abs(dW[2] + trim * k) < 1e-6, (trim, roll, pitch, dW)       # Z NED: - = up
                assert np.max(np.abs(dW[[0, 1]])) < 1e-12 and np.max(np.abs(dW[3:])) < 1e-6, dW
                # the world-vertical share of that body force: trim * (cos roll cos pitch)^2 [유도]
                assert abs(-dW[2] * k - trim * k * k) < 1e-6
                # u (the CSV's uX..uN) says what LEFT, trim included
                assert np.allclose(u1, _wrench(r.model, t1), atol=1e-12)
                # on the attainable range the per-tick record says the pure heave was delivered, no vertical at an edge
                assert i1["trim_n_edge"] == 0 and i1["pwm_scale_policy"] == 1.0
                assert np.allclose(i1["trim_wrench_ned"], [0, 0, -trim * k, 0, 0, 0], atol=1e-6), i1["trim_wrench_ned"]
                # the per-tick trim thrust is the attitude-rotated vertical allocation, horizontals exactly 0
                tt = np.array(i1["trim_thrust_n"])
                assert np.array_equal(tt, heave_trim_thrust(r.model, trim, roll, pitch)) and np.all(tt[:4] == 0.0)
                assert np.allclose((r.model.B_flu @ tt) * FLU_TO_FRD, [0, 0, -trim * k, 0, 0, 0], atol=1e-9)
                # ...and through the wire: the 1-us pulses carry it within their resolution
                dWp = (_wrench(r.model, pwm_to_action(i1["pwm_us"], r.model.throttle_max) * r.model.throttle_max)
                       - _wrench(r.model, pwm_to_action(i0["pwm_us"], r.model.throttle_max) * r.model.throttle_max))
                assert abs(dWp[2] + trim * k) < 0.25 and np.max(np.abs(dWp[3:])) < 0.06, dWp


def test_the_trim_helpers_rotate_world_up_and_allocate_like_the_hold():
    from rov_gui.control.rl_pwm import PwmModel, hold_throttles

    m = PwmModel(json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))["output"]["pwm_model"])
    # the HOLD and the trim allocate through the same function: a level trim of +4 N is the HOLD's Z -4 N NED thrust
    lvl = heave_trim_thrust(m, 4.0)
    tau = np.zeros(6)
    tau[2] = 4.0
    assert np.array_equal(lvl[4:], vertical_allocation(m, tau)) and np.all(lvl[:4] == 0.0)
    assert np.array_equal(hold_throttles(m, -4.0)[4:], m.curve.inverse(lvl[4:] / m.voltage_scale, below_edge="edge"))
    # yaw does not enter; roll/pitch scale the body heave by cos roll cos pitch and nothing else is produced
    for roll, pitch in ((0.3, 0.0), (0.0, -0.4), (0.2, 0.25)):
        tt = heave_trim_thrust(m, 5.0, roll, pitch)
        w = (m.B_flu @ tt) * FLU_TO_FRD
        assert np.allclose(w, [0, 0, -5.0 * math.cos(roll) * math.cos(pitch), 0, 0, 0], atol=1e-9)
    assert np.array_equal(heave_trim_thrust(m, 0.0, 0.3, 0.3), np.zeros(8))
    assert np.allclose(heave_trim_thrust(m, -3.0), -heave_trim_thrust(m, 3.0), atol=1e-15)
    with pytest.raises(ValueError):
        heave_trim_thrust(m, 3.0, float("nan"), 0.0)


# =============================================================================
# 3. the policy is never fed the trim
# =============================================================================
@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_prev_action_excludes_the_trim_and_the_network_stays_in_lockstep(low):
    """A trimmed and an untrimmed follower of the same export, fed the same states: while no scale binds, the network
    sees the same observation every tick (its previous action is its OWN part), so it answers the same — any trim in
    the feedback would split them at the first trimmed tick. The pulses differ on the verticals only."""
    blk, own = CAP1[low]
    ticks = scenario_ticks(5)
    ticks["targets"] = {0: ([0.15, -0.05, 0.75], 0.2)}                 # a near goal: the policy does not saturate
    with tempfile.TemporaryDirectory() as tmp:
        r0 = _build(low, blk, own, tmp)
        r1 = _build(low, blk, own, tmp, extra_own={"heave_trim_n": 4.0})
        infos = []
        orig = r1.step

        def step(*args):
            u, info = orig(*args)
            infos.append(info)
            return u, info

        a = drive(r0, ticks)
        b = drive(r1, ticks, step=step)
    n_cmp = n_diff = 0
    for k in range(N_TICKS):
        if a["pwm_scale"][k] < 1.0 or b["pwm_scale"][k] < 1.0:
            break
        n_cmp += 1
        assert np.array_equal(a["action"][k], b["action"][k]), f"tick {k}: the network saw the trim"
        assert np.array_equal(a["prev_action"][k], b["prev_action"][k]), f"tick {k}"
        info = infos[k]
        if b["guard"][k] == GUARD_HOLD:
            assert a["pwm_us"][k] == b["pwm_us"][k] == [int(v) for v in r1.hold_pwm_us]
            assert info["trim_thrust_n"] is None and info["throttle_policy"] is None
            continue
        assert a["pwm_us"][k][:4] == b["pwm_us"][k][:4], f"tick {k}: a horizontal moved"
        n_diff += a["pwm_us"][k][4:] != b["pwm_us"][k][4:]
        # the feedback = the policy's own part on the 1-us grid, which is the untrimmed follower's pulses here
        pol_us = throttles_to_pwm(info["throttle_policy"], 1.0)
        if ticks["feed"][k]:
            assert np.array_equal(b["prev_action"][k], pwm_to_action(pol_us, r1.model.throttle_max))
            assert list(pol_us) == a["pwm_us"][k]
            if b["pwm_us"][k] != a["pwm_us"][k]:
                assert not np.array_equal(b["prev_action"][k], pwm_to_action(b["pwm_us"][k], r1.model.throttle_max))
        else:
            assert np.array_equal(b["prev_action"][k], np.zeros(8)), "nothing left: neutral, as before"
    assert n_cmp == N_TICKS, f"a scale bound at tick {n_cmp}: the lockstep premise does not hold for this scenario"
    assert n_diff >= 30, n_diff
    # a report that is NOT this tick's step (foreign pulses) is fed back as it always was
    foreign = (1510, 1490, 1500, 1500, 1550, 1450, 1530, 1470)
    r1.note_applied_pwm(foreign)
    assert np.array_equal(r1._prev_action, pwm_to_action(foreign, r1.model.throttle_max))


# =============================================================================
# 4. the cap acts on the total
# =============================================================================
@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_the_cap_scale_acts_on_the_total_and_the_policy_part_takes_the_same_scale(low):
    cap = 0.5
    blk = {"pwm_cap": cap, "require_thruster_probe": False}
    own = {}
    if low == "rl_pwm":
        blk["allow_caps_below_trained"] = True
    else:
        own = {"allow_caps_below_trained": True}
    # thruster 5 at the cap pushing UP, 6/7/8 inside it: an up-trim loads 5 past the cap -> the TOTAL binds
    a_pin = [0.10, -0.12, 0.05, -0.08, 0.50, -0.20, -0.15, 0.30]
    eta = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        r0 = _build(low, blk, own, tmp)
        r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": 6.0})
        rc = _build(low, {**blk, "pwm_cap_mode": "clip"}, own, tmp, extra_own={"heave_trim_n": 6.0})
        for o in (r0, r, rc):
            _pin_action(o, a_pin)
            o.set_target_ned([0.2, 0.0, 0.7], 0.0)
            o.note_fix(5, 0.05)
        u0, i0 = r0.step(eta, np.zeros(6), np.zeros(6), 0.0)
        u1, i1 = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
        uc, ic = rc.step(eta, np.zeros(6), np.zeros(6), 0.0)
    m = r.model
    c = m.curve
    thr_pol = m.throttles(a_pin)
    F_pol = c.thrust(thr_pol)
    F_tr = heave_trim_thrust(m, 6.0) / m.voltage_scale
    F_tot = F_pol + F_tr
    s_tot = min(1.0, float(c.thrust(cap)) / float(np.max(F_tot[F_tot > 0])),
                float(c.thrust(-cap)) / float(np.min(F_tot[F_tot < 0])))
    assert i0["pwm_scale"] == 1.0, "the policy alone is inside the cap"
    assert 0.0 < s_tot < 1.0 and abs(i1["pwm_scale"] - s_tot) < 1e-15, (i1["pwm_scale"], s_tot)
    t1 = np.array(i1["throttle"])
    assert abs(float(np.max(np.abs(t1))) - cap) < 1e-9 and int(np.argmax(np.abs(t1))) == 4, "the TOTAL's binder at the cap"
    ep, en = c.edge_thrust()
    ok = (s_tot * F_tot >= ep) | (s_tot * F_tot <= en)
    assert np.all(ok[4:]), "this case keeps every vertical on the attainable range"
    assert np.max(np.abs(c.thrust(t1)[ok] - s_tot * F_tot[ok])) < 1e-6, "the total thrust is s * (policy + trim)"
    pol = np.array(i1["throttle_policy"])
    ref = scale_throttles(thr_pol, cap, m, return_scale=True)
    assert ref[1] == 1.0, "scale_throttles alone would not scale this action"
    attain = (s_tot * F_pol >= ep) | (s_tot * F_pol <= en)
    assert np.max(np.abs(c.thrust(pol)[attain] - s_tot * F_pol[attain])) < 1e-6, "the policy part takes the SAME s"
    assert np.array_equal(pol, c.inverse(s_tot * F_pol, below_edge="edge")), "on every thruster, T200^-1(s T200(thr))"
    r.note_applied_pwm(i1["pwm_us"])
    assert np.array_equal(r._prev_action, pwm_to_action(throttles_to_pwm(pol, 1.0), m.throttle_max))
    counts = r.meta()["heave_trim"]["counts"]
    assert counts["ticks_scaled"] == 1
    # (D-R2) the policy alone was inside the cap: THE TRIM made the scale bind, and the record says so
    assert i1["pwm_scale_policy"] == 1.0 and counts["ticks_scaled_by_trim"] == 1 and counts["ticks_scale_tightened"] == 1
    dl = r.meta()["heave_trim"]["delivered"]
    assert dl["n_ticks"] == 1 and abs(dl["scale_min"] - s_tot) < 1e-15 and abs(dl["scale_mean"] - s_tot) < 1e-15
    # clip mode (A/B only): the TOTAL is clipped per thruster, the policy part is the policy's clipped throttles
    tc = np.array(ic["throttle"])
    assert ic["pwm_scale"] == 1.0 and abs(float(np.max(np.abs(tc))) - cap) < 1e-12
    want = np.where(F_tr != 0.0, np.clip(c.inverse(F_tot, below_edge="edge"), -cap, cap), np.clip(thr_pol, -cap, cap))
    assert np.array_equal(tc, want) and np.array_equal(ic["throttle_policy"], np.clip(thr_pol, -cap, cap))


# =============================================================================
# 5. the HOLD is never trimmed
# =============================================================================
@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_the_hold_pulses_are_never_trimmed(low):
    blk, own = CAP1[low]
    with tempfile.TemporaryDirectory() as tmp:
        for trim in (0.0, 5.0, -5.0):
            r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": trim})
            d = drive(r, scenario_ticks(11))
            holds = [k for k in range(N_TICKS) if d["guard"][k] == GUARD_HOLD]
            assert len(holds) >= 5
            assert all(d["pwm_us"][k] == [int(v) for v in r.hold_pwm_us] for k in holds), trim
            if trim:
                assert r.meta()["heave_trim"]["counts"]["ticks_hold_untrimmed"] == len(holds)


# =============================================================================
# 6. the build refuses, each LOW reads its own key, the loader knows both
# =============================================================================
def test_parse_heave_trim_n_refuses_everything_but_a_finite_number_within_10_n():
    for ok in (0, 0.0, -0.0, 3, 9.99, 10.0, -10.0, np.float64(2.5)):
        assert parse_heave_trim_n(ok) == float(ok)
    for bad in (10.0001, -10.5, 1e9, float("nan"), float("inf"), -float("inf"), True, False, "3", None, [3.0]):
        with pytest.raises(ValueError, match=r"rl_pwm_7d\.heave_trim_n"):
            parse_heave_trim_n(bad, "rl_pwm_7d.heave_trim_n")
    assert HEAVE_TRIM_MAX_N == 10.0


@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_the_build_refuses_a_trim_beyond_10_n_naming_the_lows_own_key(low):
    blk, own = CAP1[low]
    with tempfile.TemporaryDirectory() as tmp:
        for bad in (10.5, -12.0, float("nan"), "4", True, None):
            with pytest.raises(ValueError, match=rf"{low}\.heave_trim_n"):
                _build(low, blk, own, tmp, extra_own={"heave_trim_n": bad})
        for ok in (10.0, -10.0):
            assert _build(low, blk, own, tmp, extra_own={"heave_trim_n": ok}).heave_trim_n == ok


def test_rl_pwm_7d_reads_only_its_own_block_and_rl_pwm_only_its_own():
    with tempfile.TemporaryDirectory() as tmp:
        # the transport block's trim does not stand in for rl_pwm_7d's (review D-R5: it is NOTED, never applied)...
        logs = []
        r7 = _build("rl_pwm_7d", {"pwm_cap": 1.0, "require_thruster_probe": False, "heave_trim_n": 5.0}, {}, tmp,
                    log=logs.append)
        assert r7.heave_trim_n == 0.0 and r7.heave_trim_key == "rl_pwm_7d.heave_trim_n"
        assert r7.meta()["heave_trim"]["applied"] is False
        assert sum("NOTE rl_pwm.heave_trim_n is +5.00 N" in x for x in logs) == 1, logs
        # ...its own does, and a bad own value refuses the build whatever the transport block says
        r7 = _build("rl_pwm_7d", {"pwm_cap": 1.0, "require_thruster_probe": False}, {"heave_trim_n": -2.0}, tmp)
        assert r7.heave_trim_n == -2.0 and r7.meta()["heave_trim_n"] == -2.0
        with pytest.raises(ValueError, match=r"rl_pwm_7d\.heave_trim_n"):
            _build("rl_pwm_7d", {"pwm_cap": 1.0, "require_thruster_probe": False, "heave_trim_n": 1.0},
                   {"heave_trim_n": 11.0}, tmp)
        # and rl_pwm never reads rl_pwm_7d's
        from rov_gui.control.rl_policy import HwRlPwm
        r = HwRlPwm(_cfg({"pwm_cap": 1.0, "require_thruster_probe": False}, {"heave_trim_n": 7.0}), log=lambda m: None)
        assert r.heave_trim_n == 0.0 and r.heave_trim_key == "rl_pwm.heave_trim_n"


def test_the_config_loader_knows_both_keys_and_the_shipped_value_is_off():
    from rov_gui.control.geometry import RL_PWM_7D_KEYS, MpcConfig
    from rov_gui.control.rl_pwm import RL_PWM_KEYS

    assert "heave_trim_n" in RL_PWM_KEYS and "heave_trim_n" in RL_PWM_7D_KEYS
    src = (ROOT / "config" / "hw_mpc.yaml").read_text(encoding="utf-8")
    shipped = MpcConfig.load(ROOT / "config" / "hw_mpc.yaml")
    assert shipped.rl_pwm["heave_trim_n"] == 0.0 and shipped.rl_pwm_7d["heave_trim_n"] == 0.0

    def block(text, name):
        m = re.search(rf"^{name}:[ \t]*\n", text, flags=re.M)
        nxt = re.search(r"^\S", text[m.end():], flags=re.M)
        return m.start(), (m.end() + nxt.start()) if nxt else len(text)

    def set_in(text, name, line_re, new_line):
        a, b = block(text, name)
        blk, n = re.subn(line_re, new_line, text[a:b], count=1, flags=re.M)
        assert n == 1, (name, line_re)
        return text[:a] + blk + text[b:]

    def load(text):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "m.yaml"
            f.write_text(text, encoding="utf-8")
            return MpcConfig.load(f)

    t = set_in(src, "rl_pwm", r"^  heave_trim_n: .*$", "  heave_trim_n: 4.5")
    t = set_in(t, "rl_pwm_7d", r"^  heave_trim_n: .*$", "  heave_trim_n: -2.0")
    c = load(t)
    assert c.rl_pwm["heave_trim_n"] == 4.5 and c.rl_pwm_7d["heave_trim_n"] == -2.0
    for name in ("rl_pwm", "rl_pwm_7d"):                     # a misspelt key is refused at load, in both blocks
        with pytest.raises(ValueError, match="unknown key"):
            load(set_in(src, name, r"^  heave_trim_n: .*$", "  heave_trim: 4.5"))


# =============================================================================
# 7. meta and the build line
# =============================================================================
@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_meta_records_the_trim_and_one_build_line_says_it(low):
    blk, own = CAP1[low]
    with tempfile.TemporaryDirectory() as tmp:
        logs0, logs = [], []
        r0 = _build(low, blk, own, tmp, log=logs0.append)
        r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": 3.0}, log=logs.append)
        m0, m = r0.meta(), r.meta()
        json.dumps(m)
        assert m0["heave_trim_n"] == 0.0 and m0["heave_trim"]["applied"] is False
        assert m0["heave_trim"]["thrust_level_n"] == [0.0] * 8
        assert m["heave_trim_n"] == 3.0 and m["heave_trim"]["applied"] is True
        assert m["heave_trim"]["key"] == f"{low}.heave_trim_n" and m["heave_trim"]["max_abs_n"] == 10.0
        lv = np.array(m["heave_trim"]["thrust_level_n"])
        assert np.array_equal(lv, heave_trim_thrust(r.model, 3.0))
        assert np.allclose(m["heave_trim"]["wrench_ned_level"], [0, 0, -3.0, 0, 0, 0], atol=1e-9)
        assert np.allclose(np.array(m["heave_trim"]["thrust_level_curve"]) * r.model.voltage_scale, lv, atol=1e-12)
        assert m["heave_trim"]["counts"] == {"ticks_trimmed": 0, "ticks_scaled": 0, "ticks_scaled_by_trim": 0,
                                             "ticks_scale_tightened": 0, "ticks_edge": 0, "ticks_hold_untrimmed": 0}
        # (D-R1 / D-R2 / D-R3) the cap budget, the at-rest pattern, the delivered aggregates and the CSV boundary
        ht = m["heave_trim"]
        assert ht["cap_budget"] == heave_trim_budget(r.model, 3.0, r.pwm_cap) and ht["cap_budget"]["limit"] == 0.5
        assert ht["at_rest"] == heave_trim_at_rest(r.model, 3.0, r.pwm_cap, r.pwm_cap_mode)
        assert ht["at_rest"]["pulses_us"] == [1530, 1464, 1470, 1530]          # 6 clears its edge at 3 N, 5/7/8 do not
        assert ht["delivered"]["n_ticks"] == 0 and ht["delivered"]["mean_wrench_ned"] is None
        assert ht["rl_pwm_block"] is None and "policy + trim" in ht["csv_note"]
        assert "opposite sign" in ht["note"] and "deadband edge" in ht["note"]
        assert m0["heave_trim"]["cap_budget"]["max"] == 0.0 and m0["heave_trim"]["cap_budget"]["thruster"] is None
        assert m0["heave_trim"]["at_rest"]["pulses_us"] == [1500] * 4
        # one extra build line, naming the value and the LOW's own key; none at 0
        assert len(logs) == len(logs0) + 1 == 2
        assert logs[0] == logs0[0] and not any("heave trim" in x for x in logs0)
        assert logs[1].startswith(f"{low}: heave trim +3.00 N world-up ({low}.heave_trim_n")
        # ...which says the sign convention, the cap budget, the at-rest pattern and the all-edge band, and the CSV
        for part in ("unlike the NED hold_wrench_z_n", "2.6 % of thruster 6's capped range, limit 50 %",
                     "pulses 5..8 [1530, 1464, 1470, 1530] us = Z -3.82 N NED",
                     "every trim below 2.51 N up leaves as the one edge pattern [1530, 1470, 1470, 1530] us = Z -3.67 N",
                     "CSV pwm1..8 / uX..uN are policy + trim"):
            assert part in logs[1], (part, logs[1])
        # the counters are per engagement
        d = drive(r, scenario_ticks(3))
        n_hold = d["guard"].count(GUARD_HOLD)
        counts = r.meta()["heave_trim"]["counts"]
        assert counts["ticks_trimmed"] == N_TICKS - n_hold and counts["ticks_hold_untrimmed"] == n_hold
        assert r.meta()["heave_trim"]["last_thrust_n"] is not None
        dl = r.meta()["heave_trim"]["delivered"]
        assert dl["n_ticks"] == counts["ticks_trimmed"] and 0.0 < dl["scale_min"] <= dl["scale_mean"] <= 1.0
        # the pure heave asked for: -3 N x cos roll cos pitch (the scenario wanders around 0.2 rad of pitch)
        assert -3.0 <= dl["mean_requested_z_ned"] < -2.8
        json.dumps(r.meta())
        r.reset()
        assert r.meta()["heave_trim"]["counts"]["ticks_trimmed"] == 0
        assert r.meta()["heave_trim"]["delivered"]["n_ticks"] == 0


# =============================================================================
# 8. the deadband (review 2026-10-18 D-R1): where the zero-moment heave does NOT hold, pinned
# =============================================================================
#: the policy's verticals at REST (throttle 0 on 5..8), horizontals busy
A_REST = [0.12, -0.15, 0.10, -0.11, 0.0, 0.0, 0.0, 0.0]


def _edge_pattern_from_first_principles(model, direction: float):
    """(pulses 5..8, nominal NED wrench) of all four verticals AT their deadband edge, each with the sign of its share
    of a trim in ``direction`` (+1 up, -1 down) — built from the edge thrusts and B_flu alone, not through
    trim_and_cap."""
    ep, en = model.curve.edge_thrust()
    share = direction * heave_trim_thrust(model, 1.0)[4:8]
    F = np.zeros(8)
    F[4:8] = np.where(share > 0.0, ep, en)
    w = (model.B_flu @ (F * model.voltage_scale)) * FLU_TO_FRD
    db = model.curve.deadband
    pulses = [1500 + int(round(400 * db)) if v > 0.0 else 1500 - int(round(400 * db)) for v in share]
    return pulses, w


@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_with_the_verticals_at_rest_a_small_trim_leaves_as_one_edge_pattern(low):
    """Every 0 < |trim| < 2.51 N up (2.58 N down) leaves, with the policy's verticals at rest, as the SAME four edge
    pulses — ~3.67 N of heave and ~0.109 N*m of pitch (nose-down for an up trim) whatever the value [유도: the export's
    B_flu and t200; pwm10_s1 here, the same geometry and curve as pwm11att14R_s1]. Not a pure heave and not the value
    asked for: pinned so it is recorded (info trim_wrench_ned / trim_n_edge, meta counts.ticks_edge and at_rest, the
    build line), not hidden. Above 5.40 / 5.66 N every share clears its edge and the delivered wrench is exact again."""
    blk, own = CAP1[low]
    eta = np.array([0.1, -0.05, 0.7, 0.0, 0.0, 0.4])
    with tempfile.TemporaryDirectory() as tmp:
        r0 = _build(low, blk, own, tmp)
        m = r0.model
        p_up, w_up = _edge_pattern_from_first_principles(m, 1.0)
        p_dn, w_dn = _edge_pattern_from_first_principles(m, -1.0)
        assert p_up == [1530, 1470, 1470, 1530] and p_dn == [1470, 1530, 1530, 1470]
        # the numbers the review derived by hand [유도]: Z -3.665 N (up), K -0.014, M -0.109 N*m (FRD: nose-down)
        assert abs(w_up[2] + 3.665) < 2e-3 and abs(w_up[3] + 0.0142) < 5e-4 and abs(w_up[4] + 0.1093) < 5e-4
        assert abs(w_dn[2] - 3.665) < 2e-3 and abs(w_dn[4] - 0.1093) < 5e-4
        rest = heave_trim_at_rest(m, 1.0, 1.0)
        assert abs(rest["all_edge_below_n"]["up"] - 2.511) < 1e-3 and abs(rest["all_edge_below_n"]["down"] - 2.583) < 1e-3
        assert abs(rest["all_attainable_above_n"]["up"] - 5.397) < 1e-3
        assert abs(rest["all_attainable_above_n"]["down"] - 5.656) < 1e-3
        assert rest["edge_pattern"]["up"]["pulses_us"] == p_up and np.allclose(rest["edge_pattern"]["up"]["wrench_ned"],
                                                                               w_up, atol=1e-12)
        for trim, p_want, w_want in ((0.01, p_up, w_up), (1.0, p_up, w_up), (2.5, p_up, w_up),
                                     (-0.01, p_dn, w_dn), (-2.5, p_dn, w_dn)):
            r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": trim})
            for o in (r0, r):
                _pin_action(o, A_REST)
                o.reset()
                o.set_target_ned([0.2, 0.0, 0.7], 0.4)
                o.note_fix(5, 0.05)
            _, i0 = r0.step(eta, np.zeros(6), np.zeros(6), 0.0)
            _, i1 = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
            assert list(i0["pwm_us"][4:]) == [1500] * 4 and list(i1["pwm_us"][:4]) == list(i0["pwm_us"][:4])
            assert list(i1["pwm_us"][4:]) == p_want, (trim, i1["pwm_us"])
            assert i1["trim_n_edge"] == 4 and np.allclose(i1["trim_wrench_ned"], w_want, atol=1e-9), trim
            ht = r.meta()["heave_trim"]
            assert ht["counts"]["ticks_edge"] == 1 and ht["at_rest"]["pulses_us"] == p_want
            assert np.allclose(ht["at_rest"]["wrench_ned"], w_want, atol=1e-9) and all(ht["at_rest"]["at_edge"])
            assert np.allclose(ht["delivered"]["mean_wrench_ned"], w_want, atol=1e-9)
        # past the band every share clears its edge: the pure heave, no moment, no edge
        for trim in (5.5, -5.7):
            r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": trim})
            _pin_action(r, A_REST)
            r.set_target_ned([0.2, 0.0, 0.7], 0.4)
            r.note_fix(5, 0.05)
            _, i1 = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
            assert i1["trim_n_edge"] == 0 and np.allclose(i1["trim_wrench_ned"], [0, 0, -trim, 0, 0, 0], atol=1e-6)
            assert not any(r.meta()["heave_trim"]["at_rest"]["at_edge"])


#: The 2026-10-17 RL_PWM_7D tick the review computed by hand: data/20261017/1017_204524/mpc_204524.csv, file line 1500
#: (att14 at its trained cap 0.5; the vehicle's pwm1..8). Literal here: data/ is not in git.
POOL_TICK_PWM = (1516, 1461, 1483, 1538, 1441, 1546, 1441, 1540)


@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_on_a_recorded_pool_tick_a_3n_trim_delivers_2_45_n_and_a_roll_moment(low):
    """The policy's verticals near their edges with mixed signs (the flown regime): +3 N at level pushes thrusters 5
    and 6 into the deadband gap (-0.99 / +0.79 N curve), they leave AT the edge, and the trim delivers ~2.45 N of heave
    with K -0.083 / M -0.050 N*m — a roll moment opposite to the policy's own roll command that tick (~+0.09 N*m from
    these pulses) and of the same size [유도: nominal model]. Pinned, and the record says it (trim_n_edge 2)."""
    blk, own = CAP1[low]
    a = pwm_to_action(POOL_TICK_PWM, 1.0)
    eta = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
    with tempfile.TemporaryDirectory() as tmp:
        r0 = _build(low, blk, own, tmp)
        r = _build(low, blk, own, tmp, extra_own={"heave_trim_n": 3.0})
        for o in (r0, r):
            _pin_action(o, a / o.model.throttle_max)
            o.set_target_ned([0.0, 0.0, 0.7], 0.0)
            o.note_fix(5, 0.05)
        u0, i0 = r0.step(eta, np.zeros(6), np.zeros(6), 0.0)
        u1, i1 = r.step(eta, np.zeros(6), np.zeros(6), 0.0)
    assert tuple(i0["pwm_us"]) == POOL_TICK_PWM, "the untrimmed follower reproduces the recorded tick"
    assert list(i1["pwm_us"][:4]) == list(POOL_TICK_PWM[:4]) and i1["pwm_scale"] == 1.0
    tw = np.array(i1["trim_wrench_ned"])
    assert i1["trim_n_edge"] == 2
    assert abs(tw[2] + 2.448) < 2e-3 and abs(tw[3] + 0.0833) < 1e-3 and abs(tw[4] + 0.0498) < 1e-3, tw
    assert np.max(np.abs(tw[[0, 1, 5]])) < 1e-9
    k_pol = float(u0[3])
    assert abs(k_pol - 0.095) < 2e-3 and tw[3] < 0.0 < k_pol and 0.5 < abs(tw[3]) / k_pol < 1.5
    assert np.allclose(np.asarray(u1) - np.asarray(u0), tw, atol=1e-9), "policy part = the untrimmed command here"
    assert r.meta()["heave_trim"]["counts"]["ticks_edge"] == 1


# =============================================================================
# 9. the cap budget (review 2026-10-18 D-R2)
# =============================================================================
def test_heave_trim_budget_is_the_trims_share_of_each_capped_vertical():
    from rov_gui.control.rl_pwm import PwmModel

    m = PwmModel(json.loads((PWM10 / "obs_spec.json").read_text(encoding="utf-8"))["output"]["pwm_model"])
    assert HEAVE_TRIM_MAX_CAP_FRACTION == 0.5
    # the review's numbers [유도]: T200(-0.25) = -4.752 N curve, a 10 N trim puts -4.396 N curve on thruster 6
    assert abs(float(m.curve.thrust(-0.25)) + 4.752) < 1e-3 and abs(float(m.curve.thrust(-0.5)) + 14.855) < 1e-3
    for f, cap, want, k in ((10.0, 0.25, 0.925, 6), (10.0, 0.5, 0.296, 6), (4.5, 0.25, 0.416, 6),
                            (-10.0, 0.25, 0.900, 5), (-4.0, 0.25, 0.360, 5)):
        b = heave_trim_budget(m, f, cap)
        assert abs(b["max"] - want) < 1e-3 and b["thruster"] == k and b["cap"] == cap and b["limit"] == 0.5, (f, cap, b)
        vs = m.voltage_scale
        F = heave_trim_thrust(m, f)[4:8] / vs
        ref = [F[i] / (m.curve.thrust(cap) if F[i] > 0 else m.curve.thrust(-cap)) for i in range(4)]
        assert np.allclose(b["fraction"], ref, atol=1e-12)
        # level is the worst case: any tilt only shrinks every vertical's share
        for roll, pitch in ((0.2, 0.0), (0.0, -0.3), (0.15, 0.25)):
            assert np.all(np.abs(heave_trim_thrust(m, f, roll, pitch)[4:8]) <= np.abs(heave_trim_thrust(m, f)[4:8]) + 1e-12)
    b0 = heave_trim_budget(m, 0.0, 0.25)
    assert b0["max"] == 0.0 and b0["thruster"] is None
    with pytest.raises(ValueError):
        heave_trim_budget(m, 3.0, 0.0)


@pytest.mark.parametrize("low", ["rl_pwm", "rl_pwm_7d"])
def test_the_build_refuses_a_trim_taking_more_than_half_of_a_capped_vertical(low):
    """At the wire cap 0.25 a trim beyond ~+5.4 / -5.5 N would fill more than half of thruster 6's (5's) capped range
    — the cap acts on policy + trim, so the policy's whole command would be scaled down whenever it used that thruster
    the same way. Refused at build, naming the key, the thruster and the cap; at cap 0.5 the whole +-10 N passes."""
    if low == "rl_pwm":
        blk25 = ({"pwm_cap": 0.25, "allow_caps_below_trained": True, "require_thruster_probe": False}, None)
        blk50 = ({"pwm_cap": 0.5, "allow_caps_below_trained": True, "require_thruster_probe": False}, None)
    else:
        blk25 = ({"pwm_cap": 0.25, "require_thruster_probe": False}, {"allow_caps_below_trained": True})
        blk50 = ({"pwm_cap": 0.5, "require_thruster_probe": False}, {"allow_caps_below_trained": True})
    with tempfile.TemporaryDirectory() as tmp:
        for bad, k in ((5.45, 6), (6.0, 6), (-5.6, 5), (10.0, 6)):
            with pytest.raises(ValueError, match=rf"{low}\.heave_trim_n {re.escape(f'{bad:+.2f}')} N takes "
                                                 rf"5\d\.\d % of thruster {k}'s thrust range at the wire cap 0\.25 "
                                                 rf"\(limit 50 %|{low}\.heave_trim_n \+10\.00 N takes 92\.5 %"):
                _build(low, *blk25, tmp, extra_own={"heave_trim_n": bad})
        for ok in (5.4, -5.5, 4.5, -4.0):
            r = _build(low, *blk25, tmp, extra_own={"heave_trim_n": ok})
            assert r.meta()["heave_trim"]["cap_budget"]["max"] <= 0.5
        for ok in (10.0, -10.0):
            r = _build(low, *blk50, tmp, extra_own={"heave_trim_n": ok})
            assert abs(r.meta()["heave_trim"]["cap_budget"]["max"] - (0.296 if ok > 0 else 0.288)) < 1e-3


# =============================================================================
# 10. rl_pwm_7d says it does not fly the transport block's trim (review 2026-10-18 D-R5)
# =============================================================================
def test_rl_pwm_7d_notes_a_transport_block_trim_it_does_not_fly():
    from rov_gui.control.rl_policy import HwRlPwm

    base = {"pwm_cap": 1.0, "require_thruster_probe": False}
    with tempfile.TemporaryDirectory() as tmp:
        cases = (
            ({"heave_trim_n": 5.0}, {}, True, {"value": 5.0, "valid": True}),            # set there, not here
            ({"heave_trim_n": 5.0}, {"heave_trim_n": 2.0}, True, {"value": 5.0, "valid": True}),
            ({"heave_trim_n": 5.0}, {"heave_trim_n": 5.0}, False, {"value": 5.0, "valid": True}),  # the same: nothing to say
            ({"heave_trim_n": 0.0}, {"heave_trim_n": 2.0}, False, {"value": 0.0, "valid": True}),
            ({"heave_trim_n": "4"}, {}, True, {"value": None, "valid": False, "raw": "'4'"}),
            ({}, {"heave_trim_n": 2.0}, False, None),
        )
        for transport, own, noted, want in cases:
            logs = []
            r7 = _build("rl_pwm_7d", {**base, **transport}, own, tmp, log=logs.append)
            notes = [x for x in logs if "NOTE rl_pwm.heave_trim_n" in x]
            assert len(notes) == int(noted), (transport, own, logs)
            if noted:
                assert "is NOT applied here" in notes[0] and "rl_pwm_7d.heave_trim_n" in notes[0]
            mb = r7.meta()["heave_trim"]["rl_pwm_block"]
            if want is None:
                assert mb is None
            else:
                assert mb["applied_here"] is False and mb["noted"] is noted and mb["key"] == "rl_pwm.heave_trim_n"
                assert {k: mb[k] for k in want} == want, mb
            # the behaviour is unchanged: this LOW flies its own value only
            assert r7.heave_trim_n == float(own.get("heave_trim_n", 0.0))
        # rl_pwm itself never notes anything about it
        logs = []
        r = HwRlPwm(_cfg({**base, "heave_trim_n": 2.0}, {"heave_trim_n": 7.0}), log=logs.append)
        assert r.meta()["heave_trim"]["rl_pwm_block"] is None and not any("NOTE" in x for x in logs)


def main() -> int:
    if "--record-golden" in sys.argv:
        d = record_golden()
        print(f"wrote {GOLDEN} ({sum(len(s['pwm_us']) for s in d['scenarios'].values())} ticks) meta {d['meta']}")
        return 0
    return int(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


if __name__ == "__main__":
    sys.exit(main())
