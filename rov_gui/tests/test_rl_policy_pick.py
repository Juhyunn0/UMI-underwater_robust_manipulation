#!/usr/bin/env python3
"""test_rl_policy_pick.py — the RL POLICY picker's worker side (2026-11-01), offline.

Operator request 2026-11-01: pick the RL export per RL LOW on the trajectory panel's LOW row instead of editing
hw_mpc.yaml before launch. bus.cmd_rl_policy(low, path) -> MpcWorker.set_rl_policy rebuilds that LOW's follower with
the class's OWN constructor on a copy of the live config, commits only on success, and refuses everything else with a
reason (not READY, engaged, CSV open, not an export, no-fly, the build's own gates, files changed while loading, the
thruster sink's launch-time gate). What is pinned here, on the cheap StubCtrl harness (test_control._worker) with REAL
followers built from the station exports under rov_gui/control/rl_policies (copied into tmp_path for variants):

* every refusal, in the designed order, keeps the previous follower OBJECT, its *_error, cfg.<low>.policy_dir and the
  provenance record, sets `reason`, and the slot's status reads "refused" with the note for ~6 s, then the held
  state again;
* a success swaps that slot only, re-points + resets self.ctrl when the slot is the selected LOW, resets the stale
  per-follower pwm records, records content sha1s / id / source "panel", and makes a LOW whose launch build failed
  selectable;
* the sink pre-check (motor_direction / sigma / pwm_cap against a FRESH configured snapshot; nothing without one;
  a snapshot with no gate facts — the sink's tick-failure fallback — refused with the sink's own reason, review F2
  2026-11-02), the TOCTOU re-hash, the status per slot (+ "no_fly" in every slot; a banned launch export's note
  LEADS with the ban even below its trained cap, review F1 2026-11-02), the run meta (controller.rl_policy,
  sink_gate), the reworded _pwm_gate_now with its new sigma comparison, the sink snapshot's export identity, the
  launch provenance and the backends' wiring of bus.cmd_rl_policy (both backends built offline, the signal emitted,
  the worker answers).

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest \\
        rov_gui/tests/test_rl_policy_pick.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from rov_gui.control import rl_library as L  # noqa: E402
from rov_gui.control import rl_policy as RP  # noqa: E402
from rov_gui.control import workers as W  # noqa: E402
from rov_gui.logfilter import mission_shown, panel_text  # noqa: E402
from rov_gui.state import Conn, Telemetry, now  # noqa: E402

STATION = ROOT / "rov_gui" / "control" / "rl_policies"
#: station exports these tests fly (facts pinned by _spec_facts below, so a re-export fails loudly, not oddly):
ATT = "pwm11att9X_s0"     # pwm, attitude-trained, pwm_cap_trained 0.5 -> builds as rl_pwm AND rl_pwm_7d at cap 0.5
ATT2 = "pwm11att7_s0"     # another such export (a second, different pick)
LEVEL = "pwm11k0.2_s0"    # pwm, LEVEL-trained, cap_trained 0.5 -> rl_pwm only (rl_pwm_7d refuses it at build)
AXES = "traj12_s2"        # axes -> rl only
AXES2 = "traj10_s1"
BANNED = "pwm10_s1"       # rl_library.NO_FLY (cap_trained 1.0, level)
BANNED_7D = "att7pwm_s0"  # rl_library.NO_FLY (cap_trained 1.0, attitude-trained)

_HELD_STATE = {"rl": "ok", "rl_pwm": "error", "rl_7d": "error", "rl_pwm_7d": "error"}   # the harness at launch


def _spec(name: str) -> dict:
    return json.loads((STATION / name / "obs_spec.json").read_text(encoding="utf-8"))


def _pm(name: str) -> dict:
    return _spec(name)["output"]["pwm_model"]


def test_the_exports_these_tests_fly_are_what_the_tests_assume():
    for name in (ATT, ATT2):
        s = _spec(name)
        assert s["action_mode"] == "pwm" and s["attitude_reference"]["trained"] is True, name
        assert float(s["output"]["pwm_model"]["pwm_cap_trained"]) == 0.5, name
    s = _spec(LEVEL)
    assert s["action_mode"] == "pwm" and not (s.get("attitude_reference") or {}).get("trained")
    assert float(s["output"]["pwm_model"]["pwm_cap_trained"]) == 0.5
    for name in (AXES, AXES2):
        assert _spec(name).get("action_mode", "axes") == "axes", name
    assert BANNED in L.NO_FLY and BANNED_7D in L.NO_FLY
    # one motor_direction and one sigma across the pwm exports used (the sink tests flip them on purpose)
    assert _pm(ATT)["motor_direction"] == _pm(LEVEL)["motor_direction"] == _pm(ATT2)["motor_direction"]
    assert _pm(ATT)["sigma"] == _pm(LEVEL)["sigma"] == _pm(ATT2)["sigma"]


# ----------------------------------------------------------------------------- harness
def _mk(tmp_path, extra_yaml: str = ""):
    """test_control._worker, with the mission log and the mission events collected FROM SETUP ON (the launch
    provenance lines are said in setup) and an optional block appended to the test's own hw_mpc.yaml. The pwm
    transport block the picks build against is set AFTER setup: cap 0.5 (the 0.5-trained exports fly at their
    trained cap) and the sign-probe gate opted out — RECORDED, as on a vehicle; no artefact exists on a test box."""
    from rov_gui.bus import DataBus
    from rov_gui.tests.test_control import StubCtrl, _app, _test_opts

    _app()
    Opts = _test_opts(str(tmp_path))
    if extra_yaml:
        with open(Opts.mpc_config, "a", encoding="utf-8") as f:
            f.write(extra_yaml)
    bus = DataBus()
    logs, events = [], []
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    bus.mpc_event.connect(events.append)
    w = W.MpcWorker(bus, Opts(), controller_factory=StubCtrl)
    w.setup()
    w.cfg.log_dir = str(tmp_path)
    w.cfg.engage["warmup_s"] = 0.1
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.engage["mode_settle_s"] = 0.0
    if not extra_yaml:
        w.cfg.rl_pwm = {"pwm_cap": 0.5, "require_thruster_probe": False}
    return types.SimpleNamespace(w=w, bus=bus, logs=logs, events=events)


@pytest.fixture
def h(tmp_path):
    h = _mk(tmp_path)
    yield h
    h.w.teardown()


def _held(w) -> dict:
    """Everything a refused pick must leave alone, per slot: the follower OBJECT, its *_error, cfg.<low>.policy_dir,
    the provenance record (copied) and the object it describes; plus self.ctrl."""
    out = {"ctrl": w.ctrl}
    for low, (attr, err) in W.MpcWorker._RL_SLOTS.items():
        out[low] = (getattr(w, attr, None), getattr(w, err, None),
                    dict(getattr(w.cfg, low) or {}).get("policy_dir"),
                    json.dumps(w._rl_prov.get(low), sort_keys=True), w._rl_prov_obj.get(low))
    return out


def _same(a: dict, b: dict) -> bool:
    if a["ctrl"] is not b["ctrl"]:
        return False
    for low in L.RL_LOWS:
        (o1, e1, p1, r1, q1), (o2, e2, p2, r2, q2) = a[low], b[low]
        if o1 is not o2 or e1 != e2 or p1 != p2 or r1 != r2 or q1 is not q2:
            return False
    return True


def _assert_refused(h, monkeypatch, low, path, why_part):
    """One refused pick: nothing held changes, `reason` + the WARN line say it (the line reaches the mission-log
    panel), and the slot's status reads "refused" with the note for ~6 s, then the held state again."""
    w = h.w
    before = _held(w)
    real = W.now
    monkeypatch.setattr(W, "now", lambda: real() + 6.5)       # (an earlier refusal's note may still be showing)
    held_state = w._rl_policy_status().get(low, {}).get("state")
    monkeypatch.setattr(W, "now", real)
    n_logs = len(h.logs)
    w.set_rl_policy(low, path)
    label = L.LOW_LABELS.get(low, low)
    assert w.reason.startswith(f"RL policy refused: {label}: "), w.reason
    assert why_part in w.reason, (why_part, w.reason)
    assert _same(before, _held(w)), f"a refused pick changed what is held ({w.reason})"
    warns = [m for lvl, m in h.logs[n_logs:] if lvl == "warn" and "RL policy refused" in m]
    assert len(warns) == 1 and warns[0].endswith(w.reason), h.logs[n_logs:]
    assert panel_text("warn", warns[0]) == warns[0], "a refused pick must reach the mission-log panel"
    if low not in L.RL_LOWS:
        return
    st = w._rl_policy_status()[low]
    assert st["state"] == "refused" and why_part in st["note"], st
    monkeypatch.setattr(W, "now", lambda: real() + 6.5)
    try:
        assert w._rl_policy_status()[low]["state"] == held_state, "the refusal note outlived ~6 s"
    finally:
        monkeypatch.setattr(W, "now", real)


def _copy(name: str, dst: Path) -> Path:
    dst.mkdir(parents=True)
    for f in L.REQUIRED_FILES:
        shutil.copyfile(STATION / name / f, dst / f)
    return dst


def _gate(md=None, sigma="same", cap=0.5, age=0.0, configured=True, export="rov_gui/control/rl_policies/" + LEVEL,
          **kw):
    """A sink snapshot as hardware.MavlinkCommandSink._pwm_tick builds it (2026-11-01 keys included)."""
    g = {"configured": configured, "ok": True, "why": "ok", "stamp": now() - age, "active": False,
         "motor_direction": list(_pm(LEVEL)["motor_direction"] if md is None else md), "pwm_cap": cap,
         "export": export, "export_id": "0123456789ab", "cap_trained": 0.5}
    if sigma == "same":
        g["sigma"] = [float(v) for v in _pm(LEVEL)["sigma"]]
    elif sigma is not None:
        g["sigma"] = list(sigma)
    g.update(kw)
    return g


def _tel(w, gate):
    w.on_telemetry(Telemetry(armed=True, mode="MANUAL", conn=Conn.ONLINE, pwm_gate=gate))


def _resets(monkeypatch, cls):
    """Count reset() calls per instance of ``cls`` (construction calls it once through HwPid.__init__)."""
    calls = []
    orig = cls.reset

    def counting(self, *a, **k):
        calls.append(self)
        return orig(self, *a, **k)

    monkeypatch.setattr(cls, "reset", counting)
    return calls


def _n(calls, obj) -> int:
    return sum(1 for c in calls if c is obj)


# ----------------------------------------------------------------------------- launch
def test_launch_records_what_each_rl_low_holds_and_initialises_rl_error(h):
    w = h.w
    assert w._rl is not None and w._rl_error is None, "rl_error exists (None) after a good launch, like the others"
    d = os.path.join(RP.REPO_ROOT, RP.DEFAULT_POLICY_DIR)          # the harness yaml has no rl block: the class default
    obs, npz = L.export_hashes(d)
    p = w._rl_prov["rl"]
    assert p["name"] == "traj5_s2" and p["path"] == L.resolve_dir(d) and p["rel"] == L.rel_to_repo(d)
    assert (p["obs_sha1"], p["npz_sha1"], p["id"]) == (obs, npz, L.export_id(obs, npz))
    assert p["source"] == "hw_mpc.yaml" and p["at"] and w._rl_prov_obj["rl"] is w._rl
    # the other three failed their launch build in this harness: no record, their errors as before
    assert set(w._rl_prov) == {"rl"}
    assert "rl_pwm.pwm_cap 0.25 is below" in w._rl_pwm_error and "no policy_dir" in w._rl7d_error
    # ONE file-only line per good build, after the follower's own build line
    lines = [m for _, m in h.logs if "RL policy RL " in m]
    assert lines == [f"ctrl[dobmpc]: RL policy RL traj5_s2 (hw_mpc.yaml rl.policy_dir, id {p['id'][:8]})"], lines
    assert all(lvl == "info" for lvl, m in h.logs if "RL policy RL " in m)
    assert panel_text("info", lines[0]) is None, "configuration, not a panel line"
    # the existing failure lines are unchanged
    assert any(m.startswith("ctrl[dobmpc]: RL_PWM controller unavailable (ValueError: rl_pwm: the config cap")
               and m.endswith("— rl_pwm mode will refuse") for _, m in h.logs)


def test_launch_hashes_before_and_after_and_fails_a_build_whose_files_changed(tmp_path, monkeypatch):
    exp = _copy(AXES, tmp_path / "exports" / "mine")
    orig = RP.HwRl.__init__

    def writing_init(self, cfg, log=print):
        orig(self, cfg, log=log)
        if os.path.abspath(self.policy_dir) == str(exp):
            with open(exp / "obs_spec.json", "a", encoding="utf-8") as f:
                f.write(" ")                     # the export is being rewritten while the station loads it

    monkeypatch.setattr(RP.HwRl, "__init__", writing_init)
    h = _mk(tmp_path, f"rl:\n  policy_dir: {exp}\n")
    try:
        w = h.w
        assert w._rl is None and w._rl_error == "ValueError: export files changed while loading"
        assert "rl" not in w._rl_prov
        assert any(m == "ctrl[dobmpc]: RL controller unavailable (ValueError: export files changed while loading) "
                        "— rl mode will refuse" for _, m in h.logs), h.logs
        st = w._rl_policy_status()["rl"]
        assert st["state"] == "error" and "changed while loading" in st["note"]
    finally:
        h.w.teardown()


def test_a_launch_hash_failure_never_fails_a_good_build(tmp_path, monkeypatch):
    real = W.MpcWorker._rl_read_export
    seen = []

    def first_fails(d):
        seen.append(d)
        if len(seen) == 1:
            raise PermissionError(13, "Permission denied", d)
        return real(d)

    monkeypatch.setattr(W.MpcWorker, "_rl_read_export", staticmethod(first_fails))
    h = _mk(tmp_path)
    try:
        w = h.w
        assert w._rl is not None and w._rl_error is None
        p = w._rl_prov["rl"]
        assert p["id"] is None and p["obs_sha1"] is None and "PermissionError" in p["hash_error"]
        st = w._rl_policy_status()["rl"]
        assert st["state"] == "ok" and st["id"] == "" and "not hashed at launch" in st["note"]
        assert any("RL policy RL traj5_s2 (hw_mpc.yaml rl.policy_dir, id ? — not hashed: PermissionError" in m
                   for _, m in h.logs)
    finally:
        h.w.teardown()


def test_a_no_fly_export_named_at_launch_still_builds_but_says_so(tmp_path):
    """The flight ban is enforced at the PICK; the launch path builds what hw_mpc.yaml names as it always did — and
    the record, the WARN line (on the mission-log panel since 2026-11-02) and the LOW row's state say it is banned.
    Cap 1.0 = the trained cap: the ban alone (the below-cap overlap is the next test)."""
    h = _mk(tmp_path, f"rl_pwm:\n  policy_dir: rov_gui/control/rl_policies/{BANNED}\n  pwm_cap: 1.0\n"
                      f"  require_thruster_probe: false\n")
    try:
        w = h.w
        assert w._rl_pwm is not None and w._rl_pwm_error is None
        p = w._rl_prov["rl_pwm"]
        assert p["name"] == BANNED and p["no_fly"] == L.NO_FLY[BANNED]
        st = w._rl_policy_status()["rl_pwm"]
        assert st["state"] == "warn" and st["note"] == f"no-fly — {L.NO_FLY[BANNED]}"
        assert st["no_fly"] == L.NO_FLY[BANNED]
        line = [(lvl, m) for lvl, m in h.logs if "RL policy RL_PWM " in m]
        assert len(line) == 1 and line[0][0] == "warn" and "— WARN: no-fly — flight ban" in line[0][1]
        assert panel_text(*line[0]) == line[0][1], "the launch no-fly WARN reaches the mission-log panel"
    finally:
        h.w.teardown()


def test_a_no_fly_launch_export_below_its_trained_cap_leads_with_the_ban(tmp_path):
    """Review F1 (2026-11-02), the REALISTIC overlap the reviewers reproduced: config/hw_mpc.yaml flies rl_pwm.pwm_cap
    0.25 with rl_pwm_7d.allow_caps_below_trained true, and every banned export is trained at cap 1.0 — so att7pwm_s0
    (rl_pwm_7d's policy_dir on 2026-10-03, per that YAML's own comment) named there BUILDS below its trained cap.
    Until the fix the status tested the cap first: the row read the routine cap WARN, and the launch WARN was
    file-only. Now the note LEADS with the ban and keeps the cap fact, every slot carries "no_fly" (a refusal note
    never hides it), and the launch line reaches the mission-log panel. The ban is still enforced at the pick only
    (whether ENGAGE refuses a launch-named banned export is the operator's open decision — not pinned here)."""
    h = _mk(tmp_path, f"rl_pwm:\n  policy_dir: rov_gui/control/rl_policies/{LEVEL}\n  pwm_cap: 0.25\n"
                      f"  require_thruster_probe: false\n"
                      f"rl_pwm_7d:\n  policy_dir: rov_gui/control/rl_policies/{BANNED_7D}\n"
                      f"  allow_caps_below_trained: true\n")
    try:
        w = h.w
        assert w._rl_pwm7d is not None and w._rl_pwm7d_error is None, w._rl_pwm7d_error
        assert w._rl_pwm7d.caps_below_trained is True and w._rl_pwm7d.pwm_cap == 0.25
        assert float(_spec(BANNED_7D)["output"]["pwm_model"]["pwm_cap_trained"]) == 1.0
        assert w._rl_prov["rl_pwm_7d"]["no_fly"] == L.NO_FLY[BANNED_7D]
        st = w._rl_policy_status()
        s7 = st["rl_pwm_7d"]
        cap_note = "flies at cap 0.25 below trained 1 (allow_caps_below_trained)"
        assert s7["state"] == "warn" and s7["no_fly"] == L.NO_FLY[BANNED_7D], s7
        assert s7["note"].startswith("no-fly — ") and cap_note in s7["note"], s7["note"]
        assert s7["note"] == f"no-fly — {L.NO_FLY[BANNED_7D]}; {cap_note}"
        assert {k: v["no_fly"] for k, v in st.items() if k != "rl_pwm_7d"} == {"rl": "", "rl_pwm": "", "rl_7d": ""}
        assert st["rl_pwm"]["state"] == "error", "rl_pwm: 0.5-trained at cap 0.25 without its own opt-out (as shipped)"
        # the launch WARN reaches the mission-log panel (logfilter.PANEL_LOG); the ordinary launch line does not
        line = [(lvl, m) for lvl, m in h.logs if "RL policy RL_PWM_7D " in m]
        assert len(line) == 1 and line[0][0] == "warn" and "— WARN: no-fly — flight ban" in line[0][1], line
        assert panel_text(*line[0]) == line[0][1]
        ok_line = [(lvl, m) for lvl, m in h.logs if "RL policy RL " in m]
        assert len(ok_line) == 1 and panel_text(*ok_line[0]) is None, ok_line
        # a refused pick's note outranks the state for ~6 s — the ban stays in its own key meanwhile
        w.set_rl_policy("rl_pwm_7d", str(STATION / LEVEL))          # level-trained: the build refuses it
        s7 = w._rl_policy_status()["rl_pwm_7d"]
        assert s7["state"] == "refused" and "LEVEL goal" in s7["note"], s7
        assert s7["no_fly"] == L.NO_FLY[BANNED_7D] and s7["name"] == BANNED_7D, s7
    finally:
        h.w.teardown()


# ----------------------------------------------------------------------------- refusals
def test_every_refusal_keeps_what_is_held_and_says_why(h, tmp_path, monkeypatch):
    w = h.w
    # rl_pwm_7d holds a follower first, so "the previous follower" is a real object (rl_pwm keeps its launch error)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert w._rl_pwm7d is not None, w.reason
    held = {"rl": "ok", "rl_pwm": "error", "rl_7d": "error", "rl_pwm_7d": "ok"}
    assert {k: v["state"] for k, v in w._rl_policy_status().items()} == held
    good = str(STATION / ATT2)
    # 1 unknown LOW — never a silent return
    _assert_refused(h, monkeypatch, "pid", good, "unknown RL LOW 'pid'")
    _assert_refused(h, monkeypatch, "RL_PWM_7D", good, "unknown RL LOW 'RL_PWM_7D'")
    # 2 not READY
    w._ready = False
    _assert_refused(h, monkeypatch, "rl_pwm_7d", good, "controller still building — pick again when READY")
    w._ready = True
    # 3 engaged (a real engagement of the harness's stub dobmpc — ANY engagement refuses a pick of ANY slot; and the
    #   flag alone under LOW None = an observe engagement)
    from rov_gui.tests.test_control import _feed_good_state
    w.on_enable(True)
    _feed_good_state(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    _assert_refused(h, monkeypatch, "rl_pwm_7d", good, "not while engaged")
    w.disengage("test")
    w.set_mode("none")
    assert w.cfg.mode == "none"
    w.engaged = True
    _assert_refused(h, monkeypatch, "rl", str(STATION / AXES), "not while engaged")
    w.engaged = False
    w.set_mode("dobmpc")
    # 4 a controller CSV open (REC)
    w.set_sensor_log(True, str(tmp_path / "rec" / "x"))
    assert w._csv is not None
    _assert_refused(h, monkeypatch, "rl_pwm_7d", good,
                    "not while a controller CSV is open (REC) — stop the recording first")
    w.set_sensor_log(False, "")
    assert w._csv is None
    # 5 not an export: a missing dir, a dir with one of the two files, an empty path
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(tmp_path / "nope"),
                    f"not an RL export (needs obs_spec.json + policy_weights.npz): {tmp_path / 'nope'}")
    half = tmp_path / "half"
    half.mkdir()
    shutil.copyfile(STATION / ATT2 / "obs_spec.json", half / "obs_spec.json")
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(half), "not an RL export")
    _assert_refused(h, monkeypatch, "rl_pwm_7d", "", "not an RL export")
    # 6 no-fly: by name, by a RENAMED copy (the weights' sha1), and by the training run its checkpoint names
    _assert_refused(h, monkeypatch, "rl_pwm", str(STATION / BANNED), f"{BANNED} is no-fly — flight ban")
    renamed = _copy(BANNED_7D, tmp_path / "lib" / "my_great_policy")
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(renamed),
                    f"my_great_policy is no-fly — {L.NO_FLY[BANNED_7D]} — same weights as {BANNED_7D}")
    retrained = _copy(ATT2, tmp_path / "lib" / "innocent")
    spec = json.loads((retrained / "obs_spec.json").read_text(encoding="utf-8"))
    spec["checkpoint"] = "logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s2/model_1999.pt"
    (retrained / "obs_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(retrained), "innocent is no-fly — flight ban — the pwm")
    # 7 the follower's OWN build gates: an axes export and a LEVEL pwm export into rl_pwm_7d, a pwm one into rl
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(STATION / AXES), "ValueError: rl_pwm_7d: policy")
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(STATION / LEVEL), "trained with a LEVEL goal")
    _assert_refused(h, monkeypatch, "rl", str(STATION / ATT2), "is a 'pwm' policy")
    # ...and a config gate (the cap below trained without the LOW's opt-out) — the live cfg is what builds
    w.cfg.rl_pwm["pwm_cap"] = 0.25
    _assert_refused(h, monkeypatch, "rl_pwm_7d", good, "rl_pwm.pwm_cap 0.25 is below the policy's trained cap")
    w.cfg.rl_pwm["pwm_cap"] = 0.5
    # an error slot keeps its launch error through a refusal (rl_pwm: none was ever built)
    assert w._rl_pwm is None and "rl_pwm.pwm_cap 0.25 is below" in w._rl_pwm_error
    st = w._rl_policy_status()
    assert st["rl_pwm_7d"]["state"] == "refused" and st["rl_pwm_7d"]["name"] == ATT, "the HELD export, still named"
    real = W.now
    monkeypatch.setattr(W, "now", lambda: real() + 6.5)
    assert {k: v["state"] for k, v in w._rl_policy_status().items()} == held


def test_the_checks_run_in_the_designed_order(h, tmp_path, monkeypatch):
    """When several refusals apply, the earlier check wins (DESIGN §3c order)."""
    w = h.w
    good = str(STATION / ATT)
    w._ready = False
    w.set_rl_policy("bogus", good)
    assert "unknown RL LOW 'bogus'" in w.reason                         # 1 before 2
    w.engaged = True
    w.set_rl_policy("rl_pwm_7d", good)
    assert "controller still building" in w.reason                       # 2 before 3
    w._ready = True
    w.set_sensor_log(True, str(tmp_path / "rec" / "x"))
    w.set_rl_policy("rl_pwm_7d", str(tmp_path / "nope"))
    assert "not while engaged" in w.reason                               # 3 before 4
    w.engaged = False
    w.set_rl_policy("rl_pwm_7d", str(tmp_path / "nope"))
    assert "controller CSV is open" in w.reason                          # 4 before 5
    w.set_sensor_log(False, "")
    _tel(w, _gate(md=[1] * 8))                                           # a sink that would refuse every pwm export
    w.set_rl_policy("rl_pwm_7d", str(tmp_path / "nope"))
    assert "not an RL export" in w.reason                                # 5 before 6..9
    w.set_rl_policy("rl", str(STATION / BANNED))
    assert f"{BANNED} is no-fly" in w.reason                             # 6 before 7 (rl would refuse a pwm build)
    w.set_rl_policy("rl_pwm_7d", str(STATION / LEVEL))
    assert "trained with a LEVEL goal" in w.reason                       # 7 before 9
    exp = _copy(ATT, tmp_path / "x" / "moving")
    orig = RP.HwRlPwm7d.__init__

    def writing_init(self, cfg, log=print):
        orig(self, cfg, log=log)
        with open(exp / "policy_weights.npz", "ab") as f:
            f.write(b"\0")

    monkeypatch.setattr(RP.HwRlPwm7d, "__init__", writing_init)
    w.set_rl_policy("rl_pwm_7d", str(exp))
    assert "export files changed while loading — pick again" in w.reason  # 8 before 9
    monkeypatch.setattr(RP.HwRlPwm7d, "__init__", orig)
    w.set_rl_policy("rl_pwm_7d", good)
    assert "motor_direction" in w.reason and w._rl_pwm7d is None         # 9
    _tel(w, _gate())
    w.set_rl_policy("rl_pwm_7d", good)
    assert w._rl_pwm7d is not None and w.reason.startswith("RL policy RL_PWM_7D: "), w.reason


def test_files_that_change_during_the_build_are_refused(h, tmp_path, monkeypatch):
    """TOCTOU: the export is hashed before AND after the build; a byte that moved in between refuses the pick (the
    7d classes read obs_spec.json twice, the npz once — a sync or an RL_controller rewrite in progress must not fly,
    and the id recorded must be of the bytes built)."""
    w = h.w
    exp = _copy(ATT, tmp_path / "lib" / "moving")
    before = _held(w)
    orig = RP.HwRlPwm7d.__init__

    def writing_init(self, cfg, log=print):
        orig(self, cfg, log=log)
        with open(exp / "obs_spec.json", "a", encoding="utf-8") as f:
            f.write(" ")

    monkeypatch.setattr(RP.HwRlPwm7d, "__init__", writing_init)
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(exp), "export files changed while loading — pick again")
    assert _same(before, _held(w))
    # export_hashes (the library's own reader) disagreeing before / after is the same refusal
    monkeypatch.setattr(RP.HwRlPwm7d, "__init__", orig)
    real = L.file_sha1
    n = {"npz": 0}

    def drifting(p):
        if str(p).endswith("policy_weights.npz"):
            n["npz"] += 1
            if n["npz"] > 1:
                return "f" * 40
        return real(p)

    monkeypatch.setattr(L, "file_sha1", drifting)
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(STATION / ATT2), "export files changed while loading")


# ----------------------------------------------------------------------------- success
def test_a_pick_swaps_only_that_slot_and_makes_a_failed_launch_low_selectable(h, monkeypatch):
    w = h.w
    calls = _resets(monkeypatch, RP.HwRlPwm7d)
    stub = w.ctrl
    w.set_mode("rl_pwm_7d")
    assert w.cfg.mode == "dobmpc" and "mode rl_pwm_7d unavailable" in w.reason   # the launch build failed
    others = {k: getattr(w, a) for k, (a, _e) in W.MpcWorker._RL_SLOTS.items() if k != "rl_pwm_7d"}
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    new = w._rl_pwm7d
    assert isinstance(new, RP.HwRlPwm7d) and w._rl_pwm7d_error is None
    assert w.ctrl is stub, "a non-selected LOW's pick never touches the flying follower"
    assert _n(calls, new) == 1, "built (HwPid.__init__ resets once) — no extra reset for a non-selected slot"
    assert {k: getattr(w, a) for k, (a, _e) in W.MpcWorker._RL_SLOTS.items() if k != "rl_pwm_7d"} == others
    d = L.resolve_dir(STATION / ATT)
    assert w.cfg.rl_pwm_7d["policy_dir"] == d and new.policy_dir == d
    obs, npz = L.export_hashes(d)
    p = w._rl_prov["rl_pwm_7d"]
    assert p == {"name": ATT, "path": d, "rel": f"rov_gui/control/rl_policies/{ATT}", "obs_sha1": obs,
                 "npz_sha1": npz, "id": L.export_id(obs, npz), "source": "panel", "at": p["at"]}
    assert w._rl_prov_obj["rl_pwm_7d"] is new
    assert w.reason == f"RL policy RL_PWM_7D: {ATT} (id {p['id'][:8]})"
    ev = f"RL policy RL_PWM_7D ← {ATT} (panel pick, id {p['id'][:8]}, rov_gui/control/rl_policies/{ATT})"
    assert h.events[-1] == ev and not mission_shown(ev), "the LOW row shows the pick; the panel log does not"
    assert w._pending_events and w._pending_events[-1].endswith(ev), "file half waits for the next run folder"
    assert ("info", f"ctrl[dobmpc]: {w.reason}") in h.logs
    st = w._rl_policy_status()["rl_pwm_7d"]
    # (2026-11-02: a pwm slot also carries what its HELD follower flies / was trained at — the LOW row's cap ▾)
    assert st == {"name": ATT, "path": d, "rel": p["rel"], "id": p["id"], "source": "panel", "state": "ok",
                  "note": "", "no_fly": "", "cap": 0.5, "cap_trained": 0.5}
    # the slot whose launch build failed is now selectable
    w.set_mode("rl_pwm_7d")
    assert w.cfg.mode == "rl_pwm_7d" and w.ctrl is new, w.reason
    # a pick for the SELECTED LOW re-points self.ctrl and resets it (set_mode's step) — the old network never flies
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    new2 = w._rl_pwm7d
    assert new2 is not new and w.ctrl is new2 and _n(calls, new2) == 2, "built + reset once more as the selected LOW"
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT2 and w._rl_prov_obj["rl_pwm_7d"] is new2
    assert w._path_cursor is None and w._path_err == (None, None)


def test_a_pick_of_the_selected_axes_low_re_points_and_resets(h, monkeypatch):
    w = h.w
    calls = _resets(monkeypatch, RP.HwRl)
    w.set_mode("rl")
    assert w.cfg.mode == "rl" and w.ctrl is w._rl
    old = w._rl
    w.set_rl_policy("rl", str(STATION / AXES))
    assert w._rl is not old and w.ctrl is w._rl and _n(calls, w.ctrl) == 2, w.reason
    assert w._rl_error is None and w.cfg.rl["policy_dir"] == L.resolve_dir(STATION / AXES)
    assert w._rl_prov["rl"]["source"] == "panel" and w._rl_prov["rl"]["name"] == AXES
    # the swapped follower is the one that steps (no stale object anywhere in the worker)
    assert w._ctrl_for("rl") is w.ctrl


def test_a_pwm_pick_drops_the_previous_followers_pwm_records_and_only_those(h):
    w = h.w
    sentinel = object()

    def stale():
        w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since = sentinel, (1600,) * 8, 3, 12.0

    stale()
    w.set_rl_policy("rl", str(STATION / AXES))                     # not a pwm slot: untouched
    assert (w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since) == (sentinel, (1600,) * 8, 3, 12.0)
    w.set_rl_policy("rl_pwm_7d", str(STATION / LEVEL))             # a REFUSED pwm pick: untouched
    assert w._pwm_ilk is sentinel
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert (w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since) == (None, None, 0, None)
    stale()
    w.set_rl_policy("rl_pwm", str(STATION / LEVEL))
    assert w._rl_pwm is not None, w.reason
    assert (w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since) == (None, None, 0, None)
    # the meta no longer attaches the previous follower's interlock record
    w.set_mode("rl_pwm")
    assert w._controller_meta()["interlocks"]["armed"] is False


def test_the_build_lines_are_logged_only_when_the_pick_commits(h):
    w = h.w
    _tel(w, _gate(cap=0.25, cap_trained=0.25))                     # the sink refuses AFTER a good build
    n = len(h.logs)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert "pwm_cap 0.25 ≠ this export's 0.5" in w.reason
    assert not any(m.startswith("rl_pwm_7d: policy") for _, m in h.logs[n:]), "a refused pick logs no build line"
    _tel(w, _gate())
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    built = [i for i, (_, m) in enumerate(h.logs) if m.startswith("rl_pwm_7d: policy")]
    done = [i for i, (_, m) in enumerate(h.logs) if m.startswith("ctrl[dobmpc]: RL policy RL_PWM_7D: ")]
    assert built and done and built[-1] < done[-1], "the follower's own build line, then the pick's line"


# ----------------------------------------------------------------------------- the thruster sink's gate
def test_the_sink_pre_check_refuses_each_mismatch_on_a_fresh_snapshot(h, monkeypatch):
    w = h.w
    good = str(STATION / ATT)
    sink_exp = "rov_gui/control/rl_policies/" + LEVEL
    flipped = [float(v) for v in _pm(LEVEL)["sigma"]]
    flipped[7] = -flipped[7]
    for gate, field, part in (
            (_gate(md=[1] * 8), "motor_direction", "(motor_direction [1, 1, 1, 1, 1, 1, 1, 1] ≠ this export's [-1"),
            (_gate(sigma=flipped), "sigma", "(sigma [1, 1, -1, -1, 1, -1, -1, -1] ≠ this export's [1, 1, -1, -1, "),
            (_gate(cap=0.25, cap_trained=0.25), "pwm_cap", "(pwm_cap 0.25 ≠ this export's 0.5)")):
        _tel(w, gate)
        _assert_refused(h, monkeypatch, "rl_pwm_7d", good,
                        f"the thruster sink's gate was built at launch from rl_pwm.policy_dir {sink_exp} ")
        assert part in w.reason, (field, w.reason)
        # (2026-11-02: a pwm_cap mismatch names the runtime remedy first — the LOW row's cap ▾, at the smaller of the
        # two trained caps (here the sink export's 0.25) — then the restart)
        remedy = ("— pick an export with the same pwm_cap, or set the LOW row's cap to 0.25 (rl_pwm.pwm_cap for this "
                  "session), or set rl_pwm.policy_dir in hw_mpc.yaml and restart" if field == "pwm_cap" else
                  f"— pick an export with the same {field}, or set rl_pwm.policy_dir in hw_mpc.yaml and restart")
        assert remedy in w.reason and "restart the station" not in w.reason, w.reason
    # the rl_pwm slot is judged by the same gate
    _tel(w, _gate(cap=0.25, cap_trained=0.25))
    _assert_refused(h, monkeypatch, "rl_pwm", str(STATION / LEVEL), "pwm_cap 0.25 ≠ this export's 0.5")
    # review ui-3 (2026-11-02): a sink whose window is NOT min(the cap in force 0.5, its export's trained 0.5) has not
    # applied the cap in force — re-picking that cap is a no-op, so it is never named as the remedy
    _tel(w, _gate(cap=0.25))
    _assert_refused(h, monkeypatch, "rl_pwm_7d", good, "the thruster sink enforces pwm_cap 0.25 but this export flies "
                                                       "0.5 at the cap in force 0.5 — the sink has not applied that cap")
    assert "set the LOW row's cap to" not in w.reason, w.reason
    # no verdict at the pick without a fresh, configured snapshot — the ENGAGE check still applies
    for gate in (None, _gate(md=[1] * 8, age=5.0), _gate(md=[1] * 8, configured=False),
                 {"configured": False, "ok": False, "why": "no gate", "stamp": now()}):
        w._rl_pwm7d = None
        _tel(w, gate)
        w.set_rl_policy("rl_pwm_7d", good)
        assert w._rl_pwm7d is not None, (gate, w.reason)
    # sigma is compared only when BOTH sides know it (an older sink snapshot carries none)
    _tel(w, _gate(sigma=None))
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT2, w.reason
    # an axes slot never looks at the sink
    _tel(w, _gate(md=[1] * 8))
    w.set_rl_policy("rl", str(STATION / AXES2))
    assert w._rl_prov["rl"]["name"] == AXES2, w.reason


def test_pwm_gate_now_compares_sigma_and_names_the_sinks_export(h):
    """Engage time / every engaged tick: motor_direction and pwm_cap still refused, sigma now too (when both sides
    know it), and none of the three sentences says "restart the station" (a restart rebuilds the gate from the
    same file)."""
    w = h.w
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    w.set_mode("rl_pwm_7d")
    assert w.cfg.mode == "rl_pwm_7d", w.reason
    flipped = [float(v) for v in _pm(ATT)["sigma"]]
    flipped[0] = -flipped[0]
    ok = Telemetry(pwm_gate=_gate())
    assert w._pwm_gate_now(ok) == (True, "ok")
    assert w._pwm_gate_now(Telemetry(pwm_gate=_gate(sigma=None))) == (True, "ok"), "no sigma in the snapshot: skipped"
    for gate, field in ((_gate(md=[1] * 8), "motor_direction"), (_gate(sigma=flipped), "sigma"),
                        (_gate(cap=0.25, cap_trained=0.25), "pwm_cap"),
                        (_gate(export=None, cap=0.25, cap_trained=0.25), "pwm_cap"),
                        (_gate(cap=float("nan")), "pwm_cap"),             # NaN never passes (fail-closed)
                        (_gate(sigma=[1.0] * 7), "sigma")):               # nor a sigma of another length
        good, why = w._pwm_gate_now(Telemetry(pwm_gate=gate))
        assert good is False and f"({field} " in why and "restart the station" not in why, why
        assert "set rl_pwm.policy_dir in hw_mpc.yaml and restart" in why
        exp = gate.get("export")
        assert (f"rl_pwm.policy_dir {exp} (" in why) if exp else ("rl_pwm.policy_dir's export (" in why), why
    # a snapshot that cannot be read is a refusal, never an exception (a queued slot / every engaged pwm tick)
    bad = _gate()
    bad["motor_direction"] = 5
    good, why = w._pwm_gate_now(Telemetry(pwm_gate=bad))
    assert good is False and "cannot be compared with this export (TypeError" in why, why
    _tel(w, bad)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert "cannot be compared" in w.reason and w._rl_prov["rl_pwm_7d"]["name"] == ATT, w.reason
    # ...and ENGAGE says it through _pwm_refusal (same grace path as motor_direction while engaged)
    from rov_gui.tests.test_control import _fix, _imu
    w.on_enable(True)
    t = now()
    w.on_nav_fix(_fix([-2.0, 0.0, 0.8], 0.0, t))
    w.on_vehicle_imu(_imu(depth=0.8, t=t))
    _tel(w, _gate(sigma=flipped))
    w.set_engaged(True)
    assert not w.engaged and w.reason.startswith("engage refused: rl_pwm gate: the thruster sink's gate"), w.reason
    assert "(sigma " in w.reason


def _sink_cfg(tmp_path, policy_dir: str, cap: float = 0.25) -> str:
    p = tmp_path / "sink_mpc.yaml"
    p.write_text(f"mode: dobmpc\nrl_pwm:\n  policy_dir: {policy_dir}\n  pwm_cap: {cap}\n"
                 f"  require_thruster_probe: false\n", encoding="utf-8")
    return str(p)


def _sink(tmp_path, policy_dir: str, cap: float = 0.25):
    """test_rl_pwm._sink's pattern on this test's own config (independent of the operator's hw_mpc.yaml)."""
    from rov_gui.backends.hardware import MavlinkCommandSink
    from rov_gui.tests.test_control import _app
    from rov_gui.tests.test_rl_pwm import _Log, _Master

    _app()
    bus = types.SimpleNamespace(log=_Log(), payload=types.SimpleNamespace(emit=lambda *_: None))
    opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552",
                                 mpc_config=_sink_cfg(tmp_path, policy_dir, cap))
    s = MavlinkCommandSink(bus, opts)
    s.master = _Master(True)
    s.listening = True
    s._setup_pwm_gate()
    return s, bus.log


def test_the_sink_snapshot_names_the_export_its_gate_was_built_from(tmp_path):
    rel = f"rov_gui/control/rl_policies/{LEVEL}"
    s, log = _sink(tmp_path, rel)
    assert s._pwm_gate is not None, s._pwm_gate_error
    s._pwm_tick(time.monotonic())
    st = s.pwm_gate_state()
    obs, npz = L.export_hashes(rel)
    assert st["export"] == rel and st["export_id"] == L.export_id(obs, npz)
    assert st["sigma"] == [float(v) for v in _pm(LEVEL)["sigma"]] and st["cap_trained"] == 0.5
    assert st["pwm_cap"] == 0.25 and st["configured"] is True
    st2 = (s._pwm_tick(time.monotonic()), s.pwm_gate_state())[1]
    assert st2 is not st and st2["sigma"] is not st["sigma"], "a fresh list per snapshot (immutable once handed over)"
    lines = [(lvl, m) for lvl, m in log.lines if m.startswith("rl_pwm transport gate:")]
    assert lines == [("info", f"rl_pwm transport gate: export {rel} (id {st['export_id'][:8]}), cap 0.25")]
    assert panel_text(*lines[0]) is None, "file only"
    # outside the repo the export is ABSOLUTE (never a '../..' climb)
    out = _copy(LEVEL, tmp_path / "elsewhere" / "pwm_x")
    s2, _ = _sink(tmp_path, str(out))
    s2._pwm_tick(time.monotonic())
    assert s2.pwm_gate_state()["export"] == str(out)
    # no gate: the four keys are there, None
    s3, log3 = _sink(tmp_path, str(tmp_path / "missing"))
    assert s3._pwm_gate is None
    s3._pwm_tick(time.monotonic())
    st3 = s3.pwm_gate_state()
    assert st3["configured"] is False and all(st3[k] is None for k in ("export", "export_id", "sigma", "cap_trained"))
    assert not any(m.startswith("rl_pwm transport gate:") for _, m in log3.lines)


def test_a_real_sink_snapshot_refuses_a_pick_it_would_not_carry(h, tmp_path):
    """End to end: the snapshot the real sink builds (export pwm11k0.2_s0, rl_pwm.pwm_cap 0.25 -> cap 0.25) against a
    worker whose live rl_pwm.pwm_cap is 0.5 — the pick's cap 0.5 differs. Both trained caps are 0.5, so the window
    that would make them one IS the worker's cap in force: the sink has not applied it (review ui-3, 2026-11-02 —
    re-picking the cap in force is a no-op, so it is never named as the remedy)."""
    s, _ = _sink(tmp_path, f"rov_gui/control/rl_policies/{LEVEL}")
    s._pwm_tick(time.monotonic())
    w = h.w
    _tel(w, s.pwm_gate_state())
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert w._rl_pwm7d is None
    assert w.reason == ("RL policy refused: RL_PWM_7D: the thruster sink enforces pwm_cap 0.25 but this export "
                        "flies 0.5 at the cap in force 0.5 — the sink has not applied that cap (its 'transport cap "
                        "NOT changed' line says why if it refused): pick another cap and back, or restart the station")
    # the same config as the sink's: accepted, and the meta names both exports
    w.cfg.rl_pwm["pwm_cap"] = 0.25
    w.cfg.rl_pwm_7d = {"allow_caps_below_trained": True}
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert w._rl_pwm7d is not None, w.reason
    w.set_mode("rl_pwm_7d")
    rp = w._controller_meta()["rl_policy"]
    # (2026-11-02: + the window the sink said it enforces — the LOW row's cap ▾ changes it at runtime)
    assert rp["sink_gate"] == {"export": f"rov_gui/control/rl_policies/{LEVEL}",
                               "export_id": s.pwm_gate_state()["export_id"], "pwm_cap": 0.25}


def test_a_sink_snapshot_without_gate_facts_is_refused_with_the_sinks_own_reason(h, tmp_path, monkeypatch):
    """Review F2 (2026-11-02): when the command sink's rl_pwm bookkeeping raises, its tick publishes a fallback
    snapshot (hardware.py MavlinkCommandSink.tick, the except around _pwm_tick) — fresh and configured, but with NO
    gate facts. Compared, its empty motor_direction refused EVERY pwm export as "motor_direction [] ≠ this export's
    [...]" (the wrong cause; a remedy no export meets). The pick now refuses with the sink's own reason. Driven through
    the REAL sink (test_rl_pwm's tick-failure pattern), its snapshot handed to the worker as Telemetry carries it."""
    s, log = _sink(tmp_path, f"rov_gui/control/rl_policies/{LEVEL}", cap=0.5)
    assert s._pwm_gate is not None, s._pwm_gate_error

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    s._pwm_gate.state = boom
    s.tick()
    g = s.pwm_gate_state()
    assert g["configured"] is True and g["ok"] is False and g["latched"] is True and "motor_direction" not in g, g
    err = [m for lvl, m in log.lines if lvl == "error" and "sink tick failed" in m]
    assert len(err) == 1 and panel_text("error", err[0]) == err[0], "the sink's own error line is on the panel"
    w = h.w
    why = ("the thruster sink's gate cannot be judged right now (rl_pwm tick failed: RuntimeError: synthetic) — see "
           "the sink's error line")
    for low, path in (("rl_pwm_7d", str(STATION / ATT)), ("rl_pwm", str(STATION / LEVEL))):
        _tel(w, dict(g, stamp=now()))            # the sink re-stamps it every tick while its tick keeps failing
        _assert_refused(h, monkeypatch, low, path, why)
        assert w.reason == f"RL policy refused: {L.LOW_LABELS[low]}: {why}", w.reason
        assert "motor_direction" not in w.reason and "pick an export with the same" not in w.reason, w.reason
    # a fallback with no reason still names itself
    _tel(w, {"configured": True, "ok": False, "stamp": now(), "latched": True})
    _assert_refused(h, monkeypatch, "rl_pwm_7d", str(STATION / ATT), "(its snapshot carries no gate facts)")
    # keyed on the MISSING facts, not on ok / latched: a normal latched snapshot carries the gate's identity and is
    # compared as before — identical facts accepted (the engage check refuses a not-ok gate itself)
    _tel(w, _gate(ok=False, latched=True, why="no frame from the follower for 600 ms"))
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT, w.reason
    assert w._pwm_gate_now(Telemetry(pwm_gate=dict(g, stamp=now()))) == (False, g["why"]), \
        "engage names the same cause"
    # ...and a STALE fallback is no verdict at the pick, like any stale snapshot (the engage check refuses it)
    _tel(w, dict(g, stamp=now() - 5.0))
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT2, w.reason


# ----------------------------------------------------------------------------- status and meta
def test_the_status_per_slot_ok_warn_error_refused_and_a_fresh_dict_per_call(h, monkeypatch):
    w = h.w
    st = w._rl_policy_status()
    assert set(st) == set(L.RL_LOWS)
    assert {k: v["state"] for k, v in st.items()} == _HELD_STATE
    assert st["rl"]["name"] == "traj5_s2" and st["rl"]["source"] == "hw_mpc.yaml" and len(st["rl"]["id"]) == 12
    assert st["rl_7d"] == {"name": "", "path": "", "rel": "", "id": "", "source": "", "state": "error",
                           "note": w._rl7d_error, "no_fly": ""}
    assert st["rl_pwm"]["note"] == w._rl_pwm_error
    assert all(v["no_fly"] == "" for v in st.values()), "every slot carries the key (2026-11-02), empty unless banned"
    a, b = w._rl_policy_status(), w._rl_policy_status()
    assert a == b and a is not b and all(a[k] is not b[k] for k in a)
    a["rl"]["name"] = "tampered"
    assert w._rl_policy_status()["rl"]["name"] == "traj5_s2"
    # WARN: flies below its trained cap through the LOW's recorded opt-out
    w.cfg.rl_pwm["pwm_cap"] = 0.25
    w.cfg.rl_pwm_7d = {"allow_caps_below_trained": True}
    n = len(h.logs)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    note = "flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)"
    assert w.reason.endswith(f" — WARN: {note}") and w.reason.startswith(f"RL policy RL_PWM_7D: {ATT} (id ")
    assert [lvl for lvl, m in h.logs[n:] if "RL policy RL_PWM_7D: " in m] == ["warn"]
    s7 = w._rl_policy_status()["rl_pwm_7d"]
    assert s7["state"] == "warn" and s7["note"] == note and s7["no_fly"] == "", "not banned: the cap note alone"
    # the status rides MpcStatus on every tick — and a fault in it never costs the tick (the row says nothing)
    got = []
    h.bus.mpc_status.connect(got.append)
    w.tick()
    assert got and got[-1].rl_policy == w._rl_policy_status()

    def boom():
        raise RuntimeError("synthetic")

    monkeypatch.setattr(w, "_rl_policy_status", boom)
    w.tick()
    assert len(got) == 2 and got[-1].rl_policy == {}
    monkeypatch.undo()
    # a follower installed any other way (tests do) is shown by its dir, never with another object's hashes
    w._rl = RP.HwRl(w.cfg, log=lambda m: None)
    s_rl = w._rl_policy_status()["rl"]
    assert s_rl["name"] == "traj5_s2" and s_rl["id"] == "" and s_rl["source"] == "unknown"


def test_the_meta_names_the_flown_export_at_launch_and_after_a_pick(h):
    w = h.w
    w.set_mode("rl")
    cm = w._controller_meta()
    assert cm["rl_policy"] == {**w._rl_prov["rl"], "low": "rl"} and cm["rl_policy"]["source"] == "hw_mpc.yaml"
    own = w.ctrl.meta()
    rest = {k: v for k, v in cm.items() if k != "rl_policy"}
    assert json.dumps(rest, sort_keys=True, default=str) == json.dumps(own, sort_keys=True, default=str), \
        "the follower's own meta() is untouched"
    assert "rl_policy" not in own
    assert w._run_meta()["controller"]["rl_policy"]["id"] == w._rl_prov["rl"]["id"]
    # after a pick (pwm, selected): source panel, both sha1s, the id, and the sink gate's export from the snapshot
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    w.set_mode("rl_pwm_7d")
    rp = w._controller_meta()["rl_policy"]
    obs, npz = L.export_hashes(STATION / ATT)
    assert (rp["low"], rp["source"], rp["obs_sha1"], rp["npz_sha1"], rp["id"]) == \
        ("rl_pwm_7d", "panel", obs, npz, L.export_id(obs, npz))
    assert "sink_gate" not in rp, "no snapshot, no claim"
    _tel(w, _gate())
    rp = w._controller_meta()["rl_policy"]
    # (2026-11-02: + the window the sink said it enforces — the LOW row's cap ▾ changes it at runtime)
    assert rp["sink_gate"] == {"export": "rov_gui/control/rl_policies/" + LEVEL, "export_id": "0123456789ab",
                               "pwm_cap": 0.5}
    cm = w._controller_meta()
    own = w.ctrl.meta()
    assert all(json.dumps(cm[k], sort_keys=True, default=str) == json.dumps(own[k], sort_keys=True, default=str)
               for k in own if k != "type"), "the follower's own keys are kept"
    # (review D1, 2026-11-02: the runtime cap's origin is its OWN key; HwRlPwm7d's "pwm_cap_source" string stays)
    assert cm["pwm_cap_source"] == own["pwm_cap_source"] and isinstance(cm["pwm_cap_source"], str)
    assert cm["pwm_cap_origin"]["source"] == "hw_mpc.yaml"
    # a fault in the record says so — it never costs the meta (written at engage and by REC)
    real = w._rl_policy_meta
    w._rl_policy_meta = lambda low: (_ for _ in ()).throw(RuntimeError("synthetic"))
    assert w._controller_meta()["rl_policy"] == {"low": "rl_pwm_7d", "source": "unknown",
                                                 "error": "RuntimeError: synthetic"}
    w._rl_policy_meta = real
    # a non-RL follower gets no rl_policy block
    w.set_mode("dobmpc")
    assert "rl_policy" not in w._controller_meta()


# ----------------------------------------------------------------------------- wiring + log filter
def test_both_backends_wire_the_pick_to_the_worker(tmp_path):
    """A real check, not a source grep: each backend is BUILT offline (never started — no thread, no device, no
    socket) and bus.cmd_rl_policy is emitted; the worker answers it (it is not set up, so it refuses with the
    not-READY reason — which is exactly the proof that the signal reached set_rl_policy)."""
    from rov_gui.backends import make_backend
    from rov_gui.bus import DataBus, FrameMailbox
    from rov_gui.tests.test_control import _app, _test_opts

    app = _app()
    base = _test_opts(str(tmp_path))

    class Opts(base):
        fps = 15.0
        thrusters = 8
        rec_dir = str(tmp_path)
        policy = False
        pose = False
        fstereo = False
        allow_command = False
        land_dry_run = False
        panel2 = "none"
        ip = "127.0.0.1"

    for src in ("demo", "hw"):
        bus = DataBus()
        be = make_backend(src, bus, {k: FrameMailbox(k) for k in ("main", "second", "depth")}, Opts())
        assert be.mpc is not None and not be.mpc._ready, src
        bus.cmd_rl_policy.emit("rl_pwm_7d", str(STATION / ATT))
        app.processEvents()
        assert be.mpc.reason == ("RL policy refused: RL_PWM_7D: controller still building — pick again when "
                                 "READY"), (src, be.mpc.reason)


def test_the_pick_lines_through_the_mission_log_filter(h):
    w = h.w
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    ev = h.events[-1]
    assert ev.startswith("RL policy RL_PWM_7D ← ") and not mission_shown(ev)
    ok_line = [m for _, m in h.logs if m.startswith("ctrl[dobmpc]: RL policy RL_PWM_7D: ")][-1]
    assert panel_text("info", ok_line) is None
    w.set_rl_policy("rl_pwm_7d", str(STATION / BANNED_7D))
    bad = [m for lvl, m in h.logs if lvl == "warn" and "RL policy refused" in m][-1]
    assert panel_text("warn", bad) == bad
