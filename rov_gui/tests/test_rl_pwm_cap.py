#!/usr/bin/env python3
"""test_rl_pwm_cap.py — the pwm transport cap changed at runtime (2026-11-02): the worker and the command sink, offline.

Operator request 2026-11-02 ("./c3 gui --source hw --allow-command --mpc … --policy-dr 이렇게 gui를 열면 거기서 바꿀 수
있게 해달라는 소리였어. 그리고 캡도"): in the running station, change the RL policy AND the cap. The cap is
rl_pwm.pwm_cap — the per-thruster transport's ONE |throttle| ceiling, read by both pwm followers (HwRlPwm / HwRlPwm7d,
PWM_CAP_KEY) and enforced by the command sink. The LOW row's cap ▾ asks over bus.cmd_rl_pwm_cap;
MpcWorker.set_rl_pwm_cap rebuilds RL_PWM and RL_PWM_7D at the new cap with their own constructors, commits only when the
requested LOW (and the one flying) build, then tells the sink over bus.cmd_pwm_cap; MavlinkCommandSink.set_pwm_cap
re-windows to min(cap, its gate export's trained cap). Session only — hw_mpc.yaml is never written. The flow it is
for (config/hw_mpc.yaml's rl_pwm_7d comment): fly RL_PWM_7D at 0.25 first, raise to 0.5 (its trained cap) once an
empty-jaw hold passes. Contract: the cap SPEC (C1–C5, C9) of 2026-11-02.

What is pinned, on test_control's StubCtrl harness (as test_rl_policy_pick.py) with REAL followers built from the
station exports under rov_gui/control/rl_policies (tmp copies for the byte-changing variants), and on the real sink
(test_rl_pwm.py's offline _Master):

* every refusal of set_rl_pwm_cap (C3 1–9), in the designed order: one WARN whose head says "refused" (the mission-log
  panel shows it), the note on MpcStatus.rl_pwm_cap for ~6 s, and NOTHING else changed — both pwm followers (objects),
  their *_error, the records, the live cfg, the cap's source, self.ctrl, the pwm per-follower records, no
  bus.cmd_pwm_cap, no build line, no event;
* an equal cap is a silent no-op (no line, no rebuild, no emit);
* a commit: both slots rebuilt at the new cap (NEW objects), the live cfg.rl_pwm["pwm_cap"] mutated in place,
  `_pwm_cap_src` = ("panel", at), bus.cmd_pwm_cap emitted ONCE with the value, the records (launch choice kept +
  rebuilt_at / rebuilt_for_cap), the build lines held until the commit, one summary line, one pending event;
* the raise 0.25 -> 0.5 REVIVES an RL_PWM whose launch build failed (as shipped: rl_pwm.allow_caps_below_trained
  false), the lower 0.5 -> 0.25 DROPS a non-requested RL_PWM (exactly what a launch at 0.25 gives) while RL_PWM_7D
  (allow true) flies 0.25 — and raising again revives it; a requested LOW that cannot build refuses the whole change,
  so does the FLYING one; the selected follower is re-pointed and reset; an empty slot whose export is banned is never
  built (a held launch-named banned one is rebuilt and keeps its ban); export bytes that move during the rebuild fail
  that slot, and (review D4) a cap change NEVER swaps the network: bytes that moved since the load, or a held follower
  that was never hashed, refuse the change (requested / flying slot) or drop the other slot; set_rl_policy after a
  change builds at the new cap; every rebuilt slot is judged at the window the sink will enforce (review ui-2);
* the sink gate judged at the window it WILL enforce (never the old one), end to end with a real sink wired to
  bus.cmd_pwm_cap; the status (MpcStatus.rl_pwm_cap, the slot "cap" / "cap_trained") and the run meta
  (controller.pwm_cap_origin — review D1: its own key, HwRlPwm7d's "pwm_cap_source" string untouched —,
  rl_policy.sink_gate.pwm_cap); a snapshot older than the change is "not confirmed yet", a sink that did not apply
  the cap in force is never answered with that same cap (reviews hwsafe-2 / ui-3);
* MavlinkCommandSink.set_pwm_cap: min with the trained cap, refused (said, never raised) for a bad value / without a
  gate / while an rl_pwm engagement is open or latched (review hwsafe-3), the snapshot carries the window and who set
  it, a pulse beyond the NEW window is still refused and latched, the latch is never touched; NullCommandSink.
  set_pwm_cap drops it; both backends wire it.

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest -q -p no:cacheprovider \\
        rov_gui/tests/test_rl_pwm_cap.py
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from rov_gui.control import rl_library as L  # noqa: E402
from rov_gui.control import rl_policy as RP  # noqa: E402
from rov_gui.control import rl_pwm as RPW  # noqa: E402
from rov_gui.control import workers as W  # noqa: E402
from rov_gui.logfilter import mission_shown, panel_text  # noqa: E402
from rov_gui.state import Conn, MpcStatus, PwmCommand, Telemetry, now  # noqa: E402

STATION = ROOT / "rov_gui" / "control" / "rl_policies"
REL = "rov_gui/control/rl_policies/"
#: station exports these tests fly (facts pinned by the first test, so a re-export fails loudly, not oddly). All
#: git-tracked except SHIPPED_7D, which is checked only where it is on this machine (test_rl_library._skip_if_absent).
ATT = "pwm11att9X_s0"        # pwm, attitude-trained, trained cap 0.5: builds as rl_pwm AND rl_pwm_7d
ATT2 = "pwm11att7_s0"        # another such export
LEVEL = "pwm11k0.2_s0"       # config/hw_mpc.yaml's rl_pwm.policy_dir (= the sink gate's export); LEVEL-trained, 0.5
SHIPPED_7D = "pwm11att14Rw_s1"  # config/hw_mpc.yaml's rl_pwm_7d.policy_dir; not git-tracked (2026-11-02)
BANNED = "pwm10_s1"          # rl_library.NO_FLY, trained 1.0, level — also HwRlPwm.DEFAULT_DIR
BANNED_7D = "att7pwm_s0"     # rl_library.NO_FLY, trained 1.0, attitude-trained
_UNTRACKED = frozenset({SHIPPED_7D})

REFUSED = "RL_PWM cap refused: "
#: |throttle| <= 0.25 on every motor (1500 +- 100 us): flies inside the first-water window
PW_025 = (1600, 1400, 1550, 1450, 1520, 1480, 1500, 1590)
#: 180 us on two motors: inside the 0.5 window (+- 200 us), OUTSIDE the 0.25 one (+- 100 us)
PW_WIDE = (1680, 1320, 1500, 1500, 1500, 1500, 1500, 1500)


def _spec(name: str) -> dict:
    return json.loads((STATION / name / "obs_spec.json").read_text(encoding="utf-8"))


def _pm(name: str) -> dict:
    return _spec(name)["output"]["pwm_model"]


def _skip_if_absent(name: str) -> None:
    if (STATION / name / "obs_spec.json").is_file():
        return
    assert name in _UNTRACKED, f"station export {name} is git-tracked but missing ({STATION / name}) — restore it"
    pytest.skip(f"station export {name} is not on this machine ({STATION / name}/obs_spec.json absent — an export "
                f"the operator has not committed, on a clean checkout?)")


def test_the_exports_these_tests_fly_are_what_the_tests_assume():
    for name in (ATT, ATT2, LEVEL):
        assert _spec(name)["action_mode"] == "pwm" and float(_pm(name)["pwm_cap_trained"]) == 0.5, name
    for name in (ATT, ATT2):
        assert _spec(name)["attitude_reference"]["trained"] is True, name
    assert not (_spec(LEVEL).get("attitude_reference") or {}).get("trained"), "LEVEL-trained: rl_pwm only"
    for name in (BANNED, BANNED_7D):
        assert name in L.NO_FLY and float(_pm(name)["pwm_cap_trained"]) == 1.0, name
    assert _spec(BANNED_7D)["attitude_reference"]["trained"] is True
    # a yaml without rl_pwm.policy_dir builds the class default — a banned export (the empty-slot ban below)
    assert os.path.basename(RP.HwRlPwm.DEFAULT_DIR) == BANNED and RP.HwRlPwm7d.DEFAULT_DIR is None
    # one motor_direction / sigma across these exports (the sink snapshots below flip them on purpose; the real sink
    # is built from LEVEL, test_rl_pwm's _answer_all answers MOT)
    from rov_gui.tests.test_rl_pwm import MOT
    for name in (ATT, ATT2, BANNED_7D):
        assert _pm(name)["motor_direction"] == _pm(LEVEL)["motor_direction"] == MOT, name
        assert _pm(name)["sigma"] == _pm(LEVEL)["sigma"], name
    # C5's menu and parse_rl_pwm_block's default (the cap in force when the key is absent)
    assert W.RL_PWM_CAP_CHOICES == (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90,
                                    1.00)
    assert RPW.PWM_CAP_FIRST_WATER == 0.25


# ----------------------------------------------------------------------------- harness
def _d(x) -> str:
    return str(x) if os.path.isabs(str(x)) else REL + str(x)


def _yaml(cap, rl_pwm=LEVEL, allow=False, d7=ATT, allow7=True, trim7=None) -> str:
    """The two pwm blocks, shaped as config/hw_mpc.yaml ships them (rl_pwm: LEVEL, its allow_caps_below_trained
    false; rl_pwm_7d: allow_caps_below_trained true). The sign-probe gate is opted out — RECORDED, as on a vehicle; no
    artefact exists on a test box (test_rl_policy_pick's harness). ``rl_pwm=None``: no policy_dir (the class default,
    banned); ``d7=None``: no rl_pwm_7d block."""
    s = "rl_pwm:\n"
    if rl_pwm is not None:
        s += f"  policy_dir: {_d(rl_pwm)}\n"
    s += (f"  pwm_cap: {cap}\n  allow_caps_below_trained: {str(bool(allow)).lower()}\n"
          f"  require_thruster_probe: false\n")
    if d7 is not None:
        s += f"rl_pwm_7d:\n  policy_dir: {_d(d7)}\n  allow_caps_below_trained: {str(bool(allow7)).lower()}\n"
        if trim7 is not None:
            s += f"  heave_trim_n: {trim7}\n"
    return s


def _mk(tmp_path, extra_yaml: str = ""):
    """test_control._worker with the test's own hw_mpc.yaml blocks appended, the mission log / events / the caps the
    worker tells the sink (bus.cmd_pwm_cap) collected FROM SETUP ON."""
    from rov_gui.bus import DataBus
    from rov_gui.tests.test_control import StubCtrl, _app, _test_opts

    _app()
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    Opts = _test_opts(str(tmp_path))
    if extra_yaml:
        with open(Opts.mpc_config, "a", encoding="utf-8") as f:
            f.write(extra_yaml)
    bus = DataBus()
    logs, events, caps = [], [], []
    bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
    bus.mpc_event.connect(events.append)
    bus.cmd_pwm_cap.connect(caps.append)
    w = W.MpcWorker(bus, Opts(), controller_factory=StubCtrl)
    w.setup()
    w.cfg.log_dir = str(tmp_path)
    w.cfg.engage["warmup_s"] = 0.1
    w.cfg.engage["settle_s"] = 0.0
    w.cfg.engage["mode_settle_s"] = 0.0
    return types.SimpleNamespace(w=w, bus=bus, logs=logs, events=events, caps=caps)


@pytest.fixture
def shipped(tmp_path):
    """Launch at 0.25, as config/hw_mpc.yaml ships: RL_PWM (allow false) refused its 0.5-trained export at launch;
    RL_PWM_7D (allow true) flies ATT at 0.25, below its trained 0.5."""
    h = _mk(tmp_path, _yaml(0.25))
    yield h
    h.w.teardown()


@pytest.fixture
def both(tmp_path):
    """Launch at 0.5 (the trained cap): both pwm followers built."""
    h = _mk(tmp_path, _yaml(0.5))
    yield h
    h.w.teardown()


def _held(w) -> dict:
    """Everything a refused change must leave alone."""
    out = {"ctrl": w.ctrl, "mode": str(w.cfg.mode), "src": w._pwm_cap_src,
           "cfg": json.dumps({k: getattr(w.cfg, k, None) for k in ("rl_pwm", "rl_pwm_7d")}, sort_keys=True,
                             default=str),
           "pwm": (w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since)}
    for low, (attr, err) in W.MpcWorker._RL_SLOTS.items():
        out[low] = (getattr(w, attr, None), getattr(w, err, None), json.dumps(w._rl_prov.get(low), sort_keys=True),
                    w._rl_prov_obj.get(low))
    return out


def _same(a: dict, b: dict) -> bool:
    if a["ctrl"] is not b["ctrl"] or a["pwm"][0] is not b["pwm"][0]:
        return False
    if any(a[k] != b[k] for k in ("mode", "src", "cfg")) or a["pwm"][1:] != b["pwm"][1:]:
        return False
    for low in W.MpcWorker._RL_SLOTS:
        (o1, e1, r1, q1), (o2, e2, r2, q2) = a[low], b[low]
        if o1 is not o2 or e1 != e2 or r1 != r2 or q1 is not q2:
            return False
    return True


def _assert_refused(h, monkeypatch, low, cap, part: str = "", exact: str | None = None) -> str:
    """One refused change (C3): `reason` = "RL_PWM cap refused: <why>", ONE WARN line ending with it (and nothing else
    logged — no build line, no summary) that the mission-log panel shows, the note on the status for ~6 s, then "ok";
    nothing held changed, no bus.cmd_pwm_cap, no mission event. -> why."""
    w = h.w
    before = _held(w)
    n_logs, n_ev, n_caps = len(h.logs), len(h.events), len(h.caps)
    w.set_rl_pwm_cap(low, cap)
    assert w.reason.startswith(REFUSED), w.reason
    why = w.reason[len(REFUSED):]
    assert part in why, (part, w.reason)
    if exact is not None:
        assert why == exact, (exact, w.reason)
    assert _same(before, _held(w)), f"a refused change changed what is held ({w.reason})"
    new = h.logs[n_logs:]
    assert len(new) == 1, new
    lvl, msg = new[0]
    assert lvl == "warn" and msg.startswith("ctrl") and msg.endswith(f": {w.reason}"), new
    assert panel_text(lvl, msg) == msg, "a refused change must reach the mission-log panel"
    assert h.events[n_ev:] == [] and h.caps[n_caps:] == [], "nothing reaches the sink or the mission log"
    st = w._rl_pwm_cap_status()
    assert st["state"] == "refused" and st["note"] == why, st
    # the note lives in `_rl_pwm_cap_note` — NOT the SPEC's `_rl_cap_note`: that name is the staticmethod wording a
    # follower's below-trained WARN, and an instance attribute of that name would shadow it (the policy row's status
    # would raise on every publish and go blank)
    assert callable(w._rl_cap_note) and w._rl_pwm_cap_note is not None
    assert isinstance(w._rl_policy_status()["rl_pwm_7d"]["state"], str)
    real = W.now
    monkeypatch.setattr(W, "now", lambda: real() + 6.5)
    try:
        st = w._rl_pwm_cap_status()
        assert st["state"] == "ok" and st["note"] == "", "the refusal note outlived ~6 s"
    finally:
        monkeypatch.setattr(W, "now", real)
    return why


def _copy(name: str, dst: Path) -> Path:
    dst.mkdir(parents=True)
    for f in L.REQUIRED_FILES:
        shutil.copyfile(STATION / name / f, dst / f)
    return dst


def _gate(md=None, sigma="same", cap=0.5, age=0.0, configured=True, export=REL + LEVEL, cap_trained=0.5, **kw):
    """A sink snapshot as hardware.MavlinkCommandSink._pwm_tick builds it (2026-11-01/02 keys included)."""
    g = {"configured": configured, "ok": True, "why": "ok", "stamp": now() - age, "active": False,
         "motor_direction": list(_pm(LEVEL)["motor_direction"] if md is None else md), "pwm_cap": cap,
         "pwm_cap_source": "hw_mpc.yaml", "export": export, "export_id": "0123456789ab", "cap_trained": cap_trained}
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


def _summary(h, n: int):
    """The ONE summary line of a committed change among the lines logged since index ``n``."""
    got = [(lvl, m) for lvl, m in h.logs[n:] if " RL_PWM cap " in m and "(panel): " in m]
    assert len(got) == 1, h.logs[n:]
    return got[0]


# ----------------------------------------------------------------------------- the status before anything changes
def test_no_config_no_status_and_the_absent_key_is_the_first_water_cap(tmp_path):
    from rov_gui.bus import DataBus
    from rov_gui.tests.test_control import StubCtrl, _app, _test_opts

    _app()
    assert MpcStatus().rl_pwm_cap == {}, "the field exists, empty, on every status"
    bus = DataBus()
    w = W.MpcWorker(bus, _test_opts(str(tmp_path))(), controller_factory=StubCtrl)
    assert w.cfg is None and w._rl_pwm_cap_status() == {}, "no config yet: nothing to show (the combo hides)"
    assert w._pwm_cap_src == ("hw_mpc.yaml", None)
    caps = []
    bus.cmd_pwm_cap.connect(caps.append)
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)                    # before setup: the not-READY refusal, never an exception
    assert w.reason == REFUSED + "controller still building — pick again when READY" and caps == []
    w.setup()
    try:
        # the harness yaml has NO rl_pwm block: parse_rl_pwm_block's default is the cap in force
        assert w.cfg.rl_pwm == {}
        st = w._rl_pwm_cap_status()
        assert (st["cap"], st["source"], st["at"], st["ceiling"], st["sink_cap"]) == \
            (RPW.PWM_CAP_FIRST_WATER, "hw_mpc.yaml", None, None, None), st
        assert st["allow_below"] == {"rl_pwm": False, "rl_pwm_7d": False}
        assert st["state"] == "refused" and "controller still building" in st["note"], "the refusal of a moment ago"
    finally:
        w.teardown()


def test_the_status_says_the_cap_its_source_the_menu_and_each_lows_opt_out(shipped, monkeypatch):
    h, w = shipped, shipped.w
    st = w._rl_pwm_cap_status()
    assert st == {"cap": 0.25, "source": "hw_mpc.yaml", "at": None, "choices": list(W.RL_PWM_CAP_CHOICES),
                  "ceiling": None, "sink_cap": None, "allow_below": {"rl_pwm": False, "rl_pwm_7d": True},
                  "state": "ok", "note": ""}, st
    a, b = w._rl_pwm_cap_status(), w._rl_pwm_cap_status()
    assert a == b and a is not b and a["choices"] is not b["choices"] and a["allow_below"] is not b["allow_below"]
    a["choices"].append(9.0)
    a["allow_below"]["rl_pwm"] = True
    assert w._rl_pwm_cap_status() == st, "a fresh dict per call: a reader cannot tamper with the next one"
    # MpcStatus.rl_policy per pwm slot: what the HELD follower flies / was trained at — None without one; the two
    # axes slots carry no cap keys
    rp = w._rl_policy_status()
    assert (rp["rl_pwm"]["state"], rp["rl_pwm"]["cap"], rp["rl_pwm"]["cap_trained"]) == ("error", None, None)
    assert (rp["rl_pwm_7d"]["cap"], rp["rl_pwm_7d"]["cap_trained"]) == (0.25, 0.5)
    assert rp["rl_pwm_7d"]["state"] == "warn", "it flies below its trained cap (its recorded opt-out)"
    assert not ({"cap", "cap_trained"} & (set(rp["rl"]) | set(rp["rl_7d"])))
    # ...and the run meta of a pwm follower says who set the cap (hw_mpc.yaml at launch); another follower: no key
    assert "pwm_cap_origin" not in w._controller_meta(), "the stub dobmpc flies: no pwm transport"
    w.set_mode("rl_pwm_7d")
    assert w.cfg.mode == "rl_pwm_7d", w.reason
    assert w._controller_meta()["pwm_cap_origin"] == {"source": "hw_mpc.yaml", "at": None, "cap": 0.25}
    assert w._run_meta()["controller"]["pwm_cap_origin"]["cap"] == 0.25
    # review D1: the follower's own "pwm_cap_source" (which config key) reaches the meta untouched
    assert w._controller_meta()["pwm_cap_source"] == w.ctrl.meta()["pwm_cap_source"] == \
        "rl_pwm.pwm_cap (the transport's single cap)"
    # it rides MpcStatus on every tick — and a fault in it never costs the tick (the combo then hides)
    got = []
    h.bus.mpc_status.connect(got.append)
    w.tick()
    assert got and got[-1].rl_pwm_cap == w._rl_pwm_cap_status()

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(w, "_rl_pwm_cap_status_build", boom)
    assert w._rl_pwm_cap_status() == {}, "never raises (it runs in _publish): {} on any fault"
    monkeypatch.setattr(w, "_rl_pwm_cap_status", boom)
    w.tick()
    assert len(got) == 2 and got[-1].rl_pwm_cap == {}, "guarded in _publish too"


def test_the_menu_is_cut_at_the_sinks_ceiling_and_always_holds_the_cap_in_force(shipped, monkeypatch):
    w = shipped.w
    full = list(W.RL_PWM_CAP_CHOICES)
    upto = [v for v in full if v <= 0.5]
    _tel(w, _gate(cap=0.25))
    st = w._rl_pwm_cap_status()
    assert (st["ceiling"], st["sink_cap"], st["choices"]) == (0.5, 0.25, upto), st
    # no verdict without a FRESH, CONFIGURED snapshot carrying a finite trained cap > 0: the whole menu, no ceiling
    for g, sink_cap in ((_gate(cap=0.25, age=5.0), None), (_gate(cap=0.25, configured=False), None),
                        (_gate(cap=0.25, cap_trained=None), 0.25), (_gate(cap=0.25, cap_trained=float("nan")), 0.25),
                        (_gate(cap=0.25, cap_trained=0.0), 0.25), (None, None)):
        _tel(w, g)
        st = w._rl_pwm_cap_status()
        assert (st["ceiling"], st["sink_cap"], st["choices"]) == (None, sink_cap, full), (g, st)
    # a cap in force OFF the menu is offered exactly (the combo can always select the truth), sorted in
    w.set_rl_pwm_cap("rl_pwm_7d", 0.33)
    assert w.reason.startswith("RL_PWM cap 0.25 → 0.33 (panel; "), w.reason
    assert w._rl_pwm_cap_status()["choices"] == sorted(full + [0.33])
    # a value within 1e-9 of a menu value gives way to the cap in force (one entry, the exact truth)
    w.cfg.rl_pwm["pwm_cap"] = 0.3 + 1e-12
    ch = w._rl_pwm_cap_status()["choices"]
    assert len(ch) == len(full) and 0.3 + 1e-12 in ch and ch == sorted(ch)
    # a launch cap ABOVE the sink's ceiling (hw_mpc.yaml may say 1.0 for an export trained at 0.5) stays listed
    w.cfg.rl_pwm["pwm_cap"] = 1.0
    _tel(w, _gate(cap=0.5))
    assert w._rl_pwm_cap_status()["choices"] == upto + [1.0]


# ----------------------------------------------------------------------------- refusals (C3 1-9)
def test_every_refusal_keeps_what_is_held_and_says_why(both, tmp_path, monkeypatch):
    h, w = both, both.w
    assert isinstance(w._rl_pwm, RP.HwRlPwm) and isinstance(w._rl_pwm7d, RP.HwRlPwm7d)
    # 1 not an RL_PWM LOW — never a silent return
    for low in ("rl", "rl_7d", "pid", "dobmpc", "none", "RL_PWM_7D", ""):
        _assert_refused(h, monkeypatch, low, 0.25,
                        exact=f"the cap applies to RL_PWM / RL_PWM_7D only (asked for {low!r})")
    # 2 not READY
    w._ready = False
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.25, exact="controller still building — pick again when READY")
    w._ready = True
    # 3 engaged: a real engagement of the harness's stub dobmpc (ANY engagement refuses), and the flag alone
    from rov_gui.tests.test_control import _feed_good_state
    w.on_enable(True)
    _feed_good_state(w)
    w.set_engaged(True)
    assert w.engaged, w.reason
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.25, exact="not while engaged")
    w.disengage("test")
    w.engaged = True
    _assert_refused(h, monkeypatch, "rl_pwm", 0.25, exact="not while engaged")
    w.engaged = False
    # 4 a controller CSV open (REC): one record = one cap
    w.set_sensor_log(True, str(tmp_path / "rec" / "x"))
    assert w._csv is not None
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.25,
                    exact="not while a controller CSV is open (REC) — stop the recording first")
    w.set_sensor_log(False, "")
    assert w._csv is None
    # 5 not a finite number in (0, 1] (a bool / str / None is not a number)
    for bad in (0, 0.0, -0.25, 1.0000001, 1.5, float("nan"), float("inf"), -float("inf"), "0.5", True, False, None):
        _assert_refused(h, monkeypatch, "rl_pwm_7d", bad, exact=f"cap must be a number in (0, 1], got {bad!r}")
    # 7 above the trained cap of the export the sink's gate was built from (a fresh, configured snapshot says it)
    _tel(w, _gate(cap=0.5))
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.6,
                    exact=f"the thruster sink's export {REL}{LEVEL} is trained at cap 0.5: the transport cannot "
                          f"exceed it")
    _assert_refused(h, monkeypatch, "rl_pwm", 1.0, "is trained at cap 0.5: the transport cannot exceed it")
    _tel(w, _gate(cap=0.5, export=None))                     # an export-less snapshot still names the rule
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.6, "is trained at cap 0.5: the transport cannot exceed it")
    # 8 the REQUESTED LOW's own build at the new cap: RL_PWM refuses a cap below its trained one (allow false) —
    #   refused although RL_PWM_7D (allow true) would build there: nothing committed, RL_PWM not dropped
    _tel(w, None)
    why = _assert_refused(h, monkeypatch, "rl_pwm", 0.25,
                          f"RL_PWM cannot fly {LEVEL} at cap 0.25: ValueError: rl_pwm: the config cap rl_pwm.pwm_cap "
                          f"0.25 is below the policy's trained cap 0.50")
    assert why.startswith("RL_PWM cannot fly "), why
    # 9 the sink's gate, judged at the window it WILL enforce (min(0.4, its trained 0.5)): RL_PWM_7D builds at 0.4
    #   (its opt-out) — the gate's motor directions / sigma refuse it; and RL_PWM (it cannot fly 0.4) is NOT dropped
    _tel(w, _gate(md=[1] * 8, cap=0.5))
    why = _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.4,
                          "(motor_direction [1, 1, 1, 1, 1, 1, 1, 1] ≠ this export's [-1")
    assert why.startswith(f"the thruster sink's gate was built at launch from rl_pwm.policy_dir {REL}{LEVEL} "), why
    flipped = [float(v) for v in _pm(LEVEL)["sigma"]]
    flipped[3] = -flipped[3]
    _tel(w, _gate(sigma=flipped, cap=0.5))
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.4, "(sigma [1, 1, -1, 1, 1, -1, -1, 1] ≠ this export's ")
    # ...and a fresh, configured snapshot WITHOUT gate facts (the sink's tick-failure fallback): the sink's own reason
    _tel(w, {"configured": True, "ok": False, "why": "rl_pwm tick failed: RuntimeError: synthetic", "stamp": now(),
             "latched": True})
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.4,
                    exact="the thruster sink's gate cannot be judged right now (rl_pwm tick failed: RuntimeError: "
                          "synthetic) — see the sink's error line")
    _tel(w, {"configured": True, "ok": False, "stamp": now(), "latched": True})
    _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.4, "(its snapshot carries no gate facts)")
    # all of that: the launch state, whole
    assert (w._rl_pwm.pwm_cap, w._rl_pwm7d.pwm_cap, w.cfg.rl_pwm["pwm_cap"]) == (0.5, 0.5, 0.5)
    assert w._pwm_cap_src == ("hw_mpc.yaml", None) and h.caps == []


def test_the_checks_run_in_the_designed_order(both, tmp_path):
    w = both.w
    w._ready = False
    w.set_rl_pwm_cap("rl", float("nan"))
    assert "the cap applies to RL_PWM / RL_PWM_7D only" in w.reason            # 1 before 2
    w.engaged = True
    w.set_rl_pwm_cap("rl_pwm_7d", float("nan"))
    assert "controller still building" in w.reason                             # 2 before 3
    w._ready = True
    w.set_sensor_log(True, str(tmp_path / "rec" / "x"))
    w.set_rl_pwm_cap("rl_pwm_7d", float("nan"))
    assert "not while engaged" in w.reason                                     # 3 before 4
    w.engaged = False
    w.set_rl_pwm_cap("rl_pwm_7d", float("nan"))
    assert "controller CSV is open" in w.reason                                # 4 before 5
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert "controller CSV is open" in w.reason, "4 before 6: an equal cap is no excuse while a CSV is open"
    w.set_sensor_log(False, "")
    _tel(w, _gate(md=[1] * 8, cap=0.5))                     # a sink whose gate would refuse every pwm follower
    w.set_rl_pwm_cap("rl_pwm_7d", 1.5)
    assert "cap must be a number in (0, 1], got 1.5" in w.reason                # 5 before 6..9
    n, r = len(both.logs), w.reason
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)                      # 6: the cap in force — the silent no-op wins over 9
    assert w.reason == r and len(both.logs) == n
    w.set_rl_pwm_cap("rl", 0.5)
    assert "the cap applies to RL_PWM / RL_PWM_7D only" in w.reason, "1 before 6: an equal cap is not a pass"
    # 7 before 8: a sink export trained at 0.25 — 0.3 is above it, AND RL_PWM could not build at 0.3 (allow false)
    _tel(w, _gate(md=[1] * 8, cap=0.25, cap_trained=0.25))
    w.set_rl_pwm_cap("rl_pwm", 0.3)
    assert "is trained at cap 0.25: the transport cannot exceed it" in w.reason, w.reason
    _tel(w, _gate(md=[1] * 8, cap=0.5))
    w.set_rl_pwm_cap("rl_pwm", 0.25)
    assert f"RL_PWM cannot fly {LEVEL} at cap 0.25" in w.reason, w.reason       # 8 before 9
    w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
    assert "(motor_direction " in w.reason, w.reason                           # 9
    assert w._rl_pwm_cap_status()["state"] == "refused"
    _tel(w, _gate(cap=0.5))
    w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
    assert w.reason.startswith("RL_PWM cap 0.5 → 0.25 (panel; "), w.reason      # every check passed: committed
    st = w._rl_pwm_cap_status()
    assert (st["state"], st["note"]) == ("ok", ""), "a commit supersedes the refusal of a moment ago (C3: note None)"


def test_an_equal_cap_is_a_silent_no_op(both):
    h, w = both, both.w
    before = _held(w)
    n, n_ev, r = len(h.logs), len(h.events), w.reason
    for low in ("rl_pwm_7d", "rl_pwm"):
        for c in (0.5, 0.5 + 1e-10, 0.5 - 1e-10, 1 / 2):
            w.set_rl_pwm_cap(low, c)
    assert _same(before, _held(w)), "no rebuild"
    assert len(h.logs) == n and len(h.events) == n_ev and h.caps == [] and w.reason == r, "no line, no emit"
    st = w._rl_pwm_cap_status()
    assert (st["cap"], st["source"], st["at"], st["state"]) == (0.5, "hw_mpc.yaml", None, "ok")
    # ...but a value 1e-6 away is a change (the tolerance is 1e-9): both rebuilt
    old7 = w._rl_pwm7d
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5 + 1e-6)
    assert w._rl_pwm7d is not old7 and len(h.caps) == 1, w.reason


# ----------------------------------------------------------------------------- commits
def test_a_raise_to_the_trained_cap_revives_the_rl_pwm_slot_that_failed_at_launch(shipped, monkeypatch):
    h, w = shipped, shipped.w
    calls = _resets(monkeypatch, RP.HwRlPwm)
    calls7 = _resets(monkeypatch, RP.HwRlPwm7d)
    # launch as shipped: RL_PWM (allow false) refused its 0.5-trained export at 0.25; RL_PWM_7D flies below trained
    assert w._rl_pwm is None and "rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50" in w._rl_pwm_error
    assert "rl_pwm" not in w._rl_prov
    old7 = w._rl_pwm7d
    assert (old7.pwm_cap, old7.caps_below_trained) == (0.25, True)
    launch7 = dict(w._rl_prov["rl_pwm_7d"])
    assert launch7["source"] == "hw_mpc.yaml" and launch7["at"]
    w.set_mode("rl_pwm")
    assert w.cfg.mode == "dobmpc" and "unavailable" in w.reason, "not selectable before the raise"
    live = w.cfg.rl_pwm
    stub = w.ctrl
    n, n_ev = len(h.logs), len(h.events)
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    # both slots rebuilt at 0.5 — NEW objects, at their trained cap; the errors cleared
    new, new7 = w._rl_pwm, w._rl_pwm7d
    assert type(new) is RP.HwRlPwm and type(new7) is RP.HwRlPwm7d and new7 is not old7, w.reason
    assert (new.pwm_cap, new.caps_below_trained, new7.pwm_cap, new7.caps_below_trained) == (0.5, False, 0.5, False)
    assert w._rl_pwm_error is None and w._rl_pwm7d_error is None
    # the cap lives where every follower and the meta read it: the live block, mutated in place
    assert w.cfg.rl_pwm is live and live["pwm_cap"] == 0.5 and live["policy_dir"] == REL + LEVEL
    src, at = w._pwm_cap_src
    assert src == "panel" and isinstance(at, str) and at
    # the sink is told ONCE, the value as a float
    assert h.caps == [0.5] and type(h.caps[0]) is float
    # the records: content hashes / id; the launch choice (source, at) kept; when and for which cap it was rebuilt
    obs, npz = L.export_hashes(STATION / LEVEL)
    p = w._rl_prov["rl_pwm"]
    assert {k: v for k, v in p.items() if k != "at"} == {
        "name": LEVEL, "path": L.resolve_dir(STATION / LEVEL), "rel": REL + LEVEL, "obs_sha1": obs, "npz_sha1": npz,
        "id": L.export_id(obs, npz), "source": "hw_mpc.yaml", "rebuilt_at": at, "rebuilt_for_cap": 0.5}, p
    assert p["at"], "named by hw_mpc.yaml (its launch build failed, so no record existed)"
    assert w._rl_prov["rl_pwm_7d"] == {**launch7, "rebuilt_at": at, "rebuilt_for_cap": 0.5}
    assert w._rl_prov_obj["rl_pwm"] is new and w._rl_prov_obj["rl_pwm_7d"] is new7
    # neither is the selected follower: built (HwPid.__init__ resets once), no extra reset, self.ctrl untouched
    assert _n(calls, new) == 1 and _n(calls7, new7) == 1 and w.ctrl is stub
    # the words: reason = the event; the build lines held until the commit, then ONE summary (configuration: file only)
    assert w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5, RL_PWM_7D 0.5)"
    assert h.events[n_ev:] == [w.reason] and not mission_shown(w.reason), "the combo shows it, the panel log does not"
    assert w._pending_events and w._pending_events[-1].endswith(w.reason), "the file half waits for a run folder"
    new_logs = h.logs[n:]
    assert new_logs[-1] == ("info", f"ctrl[dobmpc]: RL_PWM cap 0.25 → 0.5 (panel): RL_PWM {LEVEL} flies 0.5 (trained "
                                    f"0.5); RL_PWM_7D {ATT} flies 0.5 (trained 0.5)"), new_logs
    heads = [m.split(":", 1)[0] for _, m in new_logs[:-1]]
    assert heads and set(heads) == {"rl_pwm", "rl_pwm_7d"} and heads == sorted(heads), new_logs
    assert all(lvl == "info" and panel_text(lvl, m) is None for lvl, m in new_logs), new_logs
    # the status: the new cap, from the panel, when; both rows ok (no more below-trained WARN)
    st = w._rl_pwm_cap_status()
    assert (st["cap"], st["source"], st["at"], st["state"], st["note"]) == (0.5, "panel", at, "ok", "")
    rp = w._rl_policy_status()
    assert (rp["rl_pwm"]["state"], rp["rl_pwm"]["cap"], rp["rl_pwm"]["cap_trained"]) == ("ok", 0.5, 0.5), rp["rl_pwm"]
    assert (rp["rl_pwm_7d"]["state"], rp["rl_pwm_7d"]["note"], rp["rl_pwm_7d"]["cap"]) == ("ok", "", 0.5)
    # RL_PWM is selectable now; its run meta says who set the cap, and its own meta flies it
    w.set_mode("rl_pwm")
    assert w.cfg.mode == "rl_pwm" and w.ctrl is new, w.reason
    cm = w._controller_meta()
    assert cm["pwm_cap_origin"] == {"source": "panel", "at": at, "cap": 0.5}
    assert cm["pwm_cap"] == 0.5 and cm["rl_policy"]["rebuilt_for_cap"] == 0.5
    assert cm["rl_policy"]["source"] == "hw_mpc.yaml" and cm["rl_policy"]["low"] == "rl_pwm"


def test_a_lower_drops_a_non_requested_rl_pwm_while_rl_pwm_7d_flies_below_trained(both, monkeypatch):
    h, w = both, both.w
    calls7 = _resets(monkeypatch, RP.HwRlPwm7d)
    old, old7 = w._rl_pwm, w._rl_pwm7d
    launch = dict(w._rl_prov["rl_pwm"])
    w.set_mode("rl_pwm_7d")
    assert w.ctrl is old7, w.reason
    sentinel = object()
    w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since = sentinel, (1600,) * 8, 3, 12.0
    w._path_cursor, w._path_err, w._path_depth = 7, (0.1, 0.2), 0.9
    n = len(h.logs)
    w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
    # RL_PWM_7D: rebuilt at 0.25 (below trained, its opt-out), RE-POINTED and reset as the selected follower
    new7 = w._rl_pwm7d
    assert new7 is not old7 and (new7.pwm_cap, new7.caps_below_trained) == (0.25, True), w.reason
    assert w.ctrl is new7 and _n(calls7, new7) == 2, "built + reset once more as the selected LOW (set_mode's step)"
    assert w._path_cursor is None and w._path_err == (None, None) and w._path_depth == 0.0
    assert (w._pwm_ilk, w._pwm_last_sent_us, w._pwm_guard, w._pwm_gate_bad_since) == (None, None, 0, None)
    # RL_PWM: DROPPED — exactly what a launch at 0.25 gives; the record stays (the row names it with its ERROR)
    assert w._rl_pwm is None
    err = w._rl_pwm_error
    assert err.startswith("at cap 0.25: ValueError: rl_pwm: the config cap rl_pwm.pwm_cap 0.25 is below the policy's "
                          "trained cap 0.50"), err
    why = err[len("at cap 0.25: "):]
    assert w._rl_prov["rl_pwm"] == launch and "rl_pwm" not in w._rl_prov_obj
    rp = w._rl_policy_status()
    # (review ui-4: the row's ERROR note names the running-GUI remedy — the cap ▾ on the row that can be reached)
    assert (rp["rl_pwm"]["state"], rp["rl_pwm"]["name"], rp["rl_pwm"]["note"], rp["rl_pwm"]["cap"]) == \
        ("error", LEVEL, err + " — the cap ▾ on the RL_PWM_7D row sets rl_pwm.pwm_cap for this session", None)
    assert (rp["rl_pwm_7d"]["state"], rp["rl_pwm_7d"]["note"], rp["rl_pwm_7d"]["cap"]) == \
        ("warn", "flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)", 0.25)
    # the lines: the drop is a file-only WARN (the row shows the ERROR); the summary a WARN (a slot dropped, one flies
    # below trained), file only; the event names both
    drop = ("warn", f"ctrl[rl_pwm_7d]: RL_PWM unavailable at cap 0.25 — {why}")
    assert drop in h.logs[n:] and panel_text(*drop) is None
    assert _summary(h, n) == ("warn", f"ctrl[rl_pwm_7d]: RL_PWM cap 0.5 → 0.25 (panel): RL_PWM → ERROR ({why}); "
                                      f"RL_PWM_7D {ATT} flies 0.25 (trained 0.5)")
    assert h.logs[-1] == _summary(h, n) and panel_text(*h.logs[-1]) is None
    assert w.reason == "RL_PWM cap 0.5 → 0.25 (panel; RL_PWM ERROR, RL_PWM_7D 0.25)" and h.events[-1] == w.reason
    assert h.caps == [0.25] and w.cfg.rl_pwm["pwm_cap"] == 0.25
    # RL_PWM cannot be selected now; raising again REVIVES it from its kept record (the launch choice kept)
    w.set_mode("rl_pwm")
    assert w.cfg.mode == "rl_pwm_7d" and "unavailable" in w.reason, w.reason
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert type(w._rl_pwm) is RP.HwRlPwm and w._rl_pwm is not old and w._rl_pwm_error is None, w.reason
    p = w._rl_prov["rl_pwm"]
    assert (p["source"], p["at"], p["id"], p["rebuilt_for_cap"]) == (launch["source"], launch["at"], launch["id"], 0.5)
    assert "prev_id" not in p and w._rl_prov_obj["rl_pwm"] is w._rl_pwm
    assert h.caps == [0.25, 0.5] and w.ctrl is w._rl_pwm7d


def test_a_requested_lows_failure_refuses_the_whole_change(both, tmp_path, monkeypatch):
    h, w = both, both.w
    # RL_PWM (allow false) cannot fly its 0.5-trained export at 0.25; RL_PWM_7D could — and is NOT changed either
    why = _assert_refused(h, monkeypatch, "rl_pwm", 0.25)
    assert why.startswith(f"RL_PWM cannot fly {LEVEL} at cap 0.25: ValueError: rl_pwm: the config cap "
                          f"rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50 "), why
    # the same through the HEAVE-TRIM cap budget (a build gate the cap moves): a trim that fits 0.5 does not fit 0.25
    h2 = _mk(tmp_path / "trim", _yaml(0.5, trim7=8.0))
    try:
        assert isinstance(h2.w._rl_pwm7d, RP.HwRlPwm7d), h2.w._rl_pwm7d_error
        why = _assert_refused(h2, monkeypatch, "rl_pwm_7d", 0.25)
        assert why.startswith(f"RL_PWM_7D cannot fly {ATT} at cap 0.25: ValueError: rl_pwm_7d.heave_trim_n +8.00 N "
                              f"takes "), why
        assert "at the wire cap 0.25" in why
    finally:
        h2.w.teardown()


def test_the_flying_follower_must_build_at_the_new_cap_too(both, monkeypatch):
    """must-succeed = the requested LOW and the one flying (self.ctrl): RL_PWM flies, RL_PWM_7D is asked for 0.25 —
    RL_PWM cannot fly 0.25 (allow false), so the change is refused rather than the flying follower dropped."""
    h, w = both, both.w
    w.set_mode("rl_pwm")
    assert w.ctrl is w._rl_pwm, w.reason
    why = _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.25)
    assert why.startswith(f"RL_PWM cannot fly {LEVEL} at cap 0.25: ValueError: rl_pwm: the config cap "), why
    assert w.cfg.mode == "rl_pwm" and w.ctrl is w._rl_pwm


def test_the_selected_follower_is_re_pointed_and_reset_and_a_reset_that_raises_refuses(tmp_path, monkeypatch):
    # both LOWs allow caps below trained: a change builds both; RL_PWM flies (selected), RL_PWM_7D is asked for
    h = _mk(tmp_path, _yaml(0.5, allow=True))
    try:
        w = h.w
        calls = _resets(monkeypatch, RP.HwRlPwm)
        calls7 = _resets(monkeypatch, RP.HwRlPwm7d)
        w.set_mode("rl_pwm")
        old = w._rl_pwm
        assert w.ctrl is old, w.reason
        w.set_rl_pwm_cap("rl_pwm_7d", 0.4)
        new, new7 = w._rl_pwm, w._rl_pwm7d
        assert new is not old and w.ctrl is new and new.pwm_cap == 0.4, w.reason
        assert _n(calls, new) == 2, "the FLYING slot: built + reset as the selected LOW, re-pointed"
        assert _n(calls7, new7) == 1, "the requested, not selected: built only"
        assert w._ctrl_for("rl_pwm") is w.ctrl, "no stale object anywhere in the worker"
        assert w._controller_meta()["pwm_cap_origin"]["cap"] == 0.4
        # a reset that raises on the selected follower refuses the change — nothing half-swapped
        orig = RP.HwRlPwm.reset
        seen: dict = {}

        def second_reset_raises(self, *a, **k):
            if self is not new:
                seen[id(self)] = seen.get(id(self), 0) + 1
                if seen[id(self)] >= 2:                      # construction resets once; the selection reset is next
                    raise RuntimeError("synthetic")
            return orig(self, *a, **k)

        monkeypatch.setattr(RP.HwRlPwm, "reset", second_reset_raises)
        _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.3,
                        exact=f"RL_PWM cannot fly {LEVEL} at cap 0.3: RuntimeError: synthetic")
        assert w.ctrl is new and w._rl_pwm is new and new.pwm_cap == 0.4
    finally:
        h.w.teardown()


def test_an_empty_slot_without_an_export_is_left_as_it_is(tmp_path, monkeypatch):
    h = _mk(tmp_path, _yaml(0.25, allow=True, d7=None))           # no rl_pwm_7d block: no policy_dir, no follower
    try:
        w = h.w
        err7 = w._rl_pwm7d_error
        assert w._rl_pwm7d is None and "no policy_dir" in err7
        # asked for, it is refused (nothing to build)...
        why = _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.5)
        assert why.startswith("RL_PWM_7D cannot fly at cap 0.5: ") and "no rl_pwm_7d.policy_dir" in why, why
        # ...as the OTHER slot it stays as it is: no follower, the same error
        n = len(h.logs)
        w.set_rl_pwm_cap("rl_pwm", 0.5)
        assert w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5)", w.reason
        assert w._rl_pwm7d is None and w._rl_pwm7d_error == err7 and "rl_pwm_7d" not in w._rl_prov
        assert w._rl_pwm.pwm_cap == 0.5 and h.caps == [0.5]
        assert _summary(h, n)[1].endswith("; RL_PWM_7D not built (no rl_pwm_7d.policy_dir)"), _summary(h, n)
    finally:
        h.w.teardown()


def test_an_absent_transport_block_is_created_by_the_change(tmp_path):
    """C3 COMMIT: the live cfg.rl_pwm["pwm_cap"] = cap — the dict created when the block is absent. A yaml without
    `rl_pwm:` loads as {}; a config object without the block at all (None) gets {"pwm_cap": cap}."""
    for blk in ({}, None):
        h = _mk(tmp_path / str(blk), f"rl_pwm_7d:\n  policy_dir: {REL}{ATT}\n  allow_caps_below_trained: true\n")
        try:
            w = h.w
            assert w.cfg.rl_pwm == {} and isinstance(w._rl_pwm7d, RP.HwRlPwm7d) and w._rl_pwm7d.pwm_cap == 0.25
            w.cfg.rl_pwm = blk
            assert w._rl_pwm_cap_status()["cap"] == RPW.PWM_CAP_FIRST_WATER
            w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
            assert w.cfg.rl_pwm == {"pwm_cap": 0.5}, (blk, w.reason, w.cfg.rl_pwm)
            assert w._rl_pwm7d.pwm_cap == 0.5 and h.caps == [0.5]
            # RL_PWM: no follower, its (class default) export is banned — never built by a cap change
            assert w._rl_pwm is None and w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM_7D 0.5)", w.reason
        finally:
            h.w.teardown()


# ----------------------------------------------------------------------------- the flight ban
def test_an_empty_slot_naming_a_banned_export_is_never_built(tmp_path, monkeypatch):
    """RL_PWM names pwm10_s1 (banned, trained 1.0): its launch build at 0.25 failed on the cap (below trained, no
    opt-out), so the slot is EMPTY. At 1.0 a launch WOULD build it — a cap change must not make a banned export
    newly available. The same for a renamed copy (recognised by its weights) and for the class default."""
    for d, name in ((BANNED, BANNED), ("copy", "innocent")):
        if d == "copy":
            d = str(_copy(BANNED, tmp_path / "lib" / "innocent"))
        h = _mk(tmp_path / name, _yaml(0.25, rl_pwm=d))
        try:
            w = h.w
            err = w._rl_pwm_error
            assert w._rl_pwm is None and "below the policy's trained cap 1.00" in err, err
            src = Path(d) if os.path.isabs(d) else STATION / d
            assert L.no_fly_reason(name, L.export_hashes(src)[1], None), "banned by name or by its weights"
            n = len(h.logs)
            w.set_rl_pwm_cap("rl_pwm_7d", 1.0)
            assert w.reason == "RL_PWM cap 0.25 → 1 (panel; RL_PWM_7D 0.5)", w.reason
            assert w._rl_pwm is None and w._rl_pwm_error == err and "rl_pwm" not in w._rl_prov, "not built"
            assert w._rl_pwm7d.pwm_cap == 0.5, "min(1.0, its trained 0.5)"
            line = _summary(h, n)[1]
            assert f"RL_PWM not built ({name} is no-fly — " in line, line
            # asked for directly: refused with the ban
            why = _assert_refused(h, monkeypatch, "rl_pwm", 0.9)
            assert why.startswith(f"RL_PWM cannot fly {name} at cap 0.9: {name} is no-fly — "), why
            assert why.endswith(" — a cap change never makes a banned export newly available"), why
        finally:
            h.w.teardown()
    # the class default (no rl_pwm.policy_dir): pwm10_s1 — the same
    h = _mk(tmp_path / "default", _yaml(0.25, rl_pwm=None))
    try:
        w = h.w
        assert w._rl_pwm is None
        why = _assert_refused(h, monkeypatch, "rl_pwm", 1.0)
        assert why.startswith(f"RL_PWM cannot fly {BANNED} at cap 1: {BANNED} is no-fly — {L.NO_FLY[BANNED]}"), why
    finally:
        h.w.teardown()


def test_a_held_launch_named_banned_follower_is_rebuilt_and_keeps_its_ban(tmp_path):
    """The ban is enforced at the PICK; a hw_mpc.yaml that names a banned export still builds it at launch (status
    quo, test_rl_policy_pick). A cap change rebuilds that HELD follower like any other and keeps its record's ban."""
    h = _mk(tmp_path, _yaml(0.25, d7=BANNED_7D))
    try:
        w = h.w
        old = w._rl_pwm7d
        assert old is not None and w._rl_prov["rl_pwm_7d"]["no_fly"] == L.NO_FLY[BANNED_7D]
        w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
        new = w._rl_pwm7d
        assert new is not old and (new.pwm_cap, new.caps_below_trained) == (0.5, True), w.reason
        assert w._rl_prov["rl_pwm_7d"]["no_fly"] == L.NO_FLY[BANNED_7D]
        s7 = w._rl_policy_status()["rl_pwm_7d"]
        assert s7["state"] == "warn" and s7["note"].startswith("no-fly — ") and s7["no_fly"] == L.NO_FLY[BANNED_7D]
        assert "flies at cap 0.5 below trained 1 (allow_caps_below_trained)" in s7["note"], s7
    finally:
        h.w.teardown()


def test_a_held_export_whose_weights_became_banned_is_never_rebuilt(tmp_path, monkeypatch):
    """Beyond the SPEC (the worker's own extension, kept): a HELD follower launched from an unbanned export whose
    policy_weights.npz was since REPLACED on disk by a banned network's. The follower in memory is the old network;
    a rebuild would load the banned one — so it is not rebuilt: asked for, the change is refused; as the other slot,
    the follower is dropped. (A pick of those bytes is refused by the same ban, recognised by the weights' sha1; only
    an export hw_mpc.yaml NAMED at launch, its record carrying "no_fly", is rebuilt — the status quo above.)"""
    exp = _copy(ATT, tmp_path / "lib" / "innocent")
    h = _mk(tmp_path / "w", _yaml(0.25, allow=True, d7=str(exp)))
    try:
        w = h.w
        assert isinstance(w._rl_pwm7d, RP.HwRlPwm7d) and not w._rl_prov["rl_pwm_7d"].get("no_fly")
        shutil.copyfile(STATION / BANNED_7D / "policy_weights.npz", exp / "policy_weights.npz")
        why = _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.5)
        assert why.startswith("RL_PWM_7D cannot fly innocent at cap 0.5: innocent is no-fly — "), why
        assert f"same weights as {BANNED_7D}" in why and why.endswith("(only a banned LAUNCH export is rebuilt)"), why
        n = len(h.logs)
        w.set_rl_pwm_cap("rl_pwm", 0.5)
        assert w._rl_pwm7d is None and w._rl_pwm7d_error.startswith("at cap 0.5: innocent is no-fly — "), w.reason
        assert any(m.startswith("ctrl[dobmpc]: RL_PWM_7D unavailable at cap 0.5 — innocent is no-fly — ")
                   for _, m in h.logs[n:]), h.logs[n:]
    finally:
        h.w.teardown()


# ----------------------------------------------------------------------------- the export's bytes
def test_export_files_that_change_during_the_rebuild_fail_that_slot(tmp_path, monkeypatch):
    exp = _copy(ATT, tmp_path / "lib" / "moving")
    h = _mk(tmp_path / "w", _yaml(0.25, allow=True, d7=str(exp)))   # both build below trained (both opt-outs)
    try:
        w = h.w
        assert isinstance(w._rl_pwm, RP.HwRlPwm) and isinstance(w._rl_pwm7d, RP.HwRlPwm7d)
        orig = RP.HwRlPwm7d.__init__

        def writing_init(self, cfg, log=print):
            orig(self, cfg, log=log)
            if os.path.abspath(self.policy_dir) == str(exp):
                with open(exp / "obs_spec.json", "a", encoding="utf-8") as f:
                    f.write(" ")                  # the export is being rewritten while the station loads it

        monkeypatch.setattr(RP.HwRlPwm7d, "__init__", writing_init)
        # the REQUESTED slot: refused, nothing committed
        _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.5,
                        exact="RL_PWM_7D cannot fly moving at cap 0.5: export files changed while loading")
        # the OTHER slot: it never flies bytes it did not hash — DROPPED, said. (The refused attempt above left its
        # write behind: put the loaded bytes back, so the record matches at the start and they move DURING the build.)
        with open(exp / "obs_spec.json", "r+", encoding="utf-8") as f:
            txt = f.read()
            f.seek(0)
            f.write(txt.rstrip(" "))
            f.truncate()
        n = len(h.logs)
        w.set_rl_pwm_cap("rl_pwm", 0.5)
        assert w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5, RL_PWM_7D ERROR)", w.reason
        assert w._rl_pwm7d is None and w._rl_pwm7d_error == "at cap 0.5: export files changed while loading"
        assert ("warn", "ctrl[dobmpc]: RL_PWM_7D unavailable at cap 0.5 — export files changed while loading") in \
            h.logs[n:]
        assert "rl_pwm_7d" not in w._rl_prov_obj and w._rl_prov["rl_pwm_7d"]["name"] == "moving"
    finally:
        h.w.teardown()


def test_a_cap_change_never_swaps_the_network_when_the_files_changed_since_the_load(tmp_path, monkeypatch):
    """Review D4 (2026-11-02): the rebuild reads the export again, so files rewritten in place since the slot loaded
    them (RL_controller re-exports in place) would fly a network nobody loaded or picked. Refused for the requested /
    flying slot, the other slot DROPPED; picking the export again (…) is how new files fly."""
    exp = _copy(ATT, tmp_path / "lib" / "rewritten")
    h = _mk(tmp_path / "w", _yaml(0.25, d7=str(exp)))
    try:
        w = h.w
        old_id = w._rl_prov["rl_pwm_7d"]["id"]
        with open(exp / "obs_spec.json", "a", encoding="utf-8") as f:
            f.write("\n")                         # rewritten in place after the launch (still a valid export)
        obs, npz = L.export_hashes(exp)
        new_id = L.export_id(obs, npz)
        assert new_id != old_id
        why = (f"rewritten's files changed since it was loaded (id {old_id[:8]} → {new_id[:8]}) — a cap change never "
               f"swaps the network; pick it again with … to fly the new files")
        # the REQUESTED slot: refused, nothing held changes
        _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.5, exact=f"RL_PWM_7D cannot fly rewritten at cap 0.5: {why}")
        # the OTHER slot (a change asked on RL_PWM, which builds at 0.5): DROPPED, its record keeps the loaded id
        old7 = w._rl_pwm7d
        w.set_rl_pwm_cap("rl_pwm", 0.5)
        assert w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5, RL_PWM_7D ERROR)", w.reason
        assert w._rl_pwm7d is None and old7 is not None and w._rl_pwm7d_error == f"at cap 0.5: {why}"
        assert w._rl_prov["rl_pwm_7d"]["id"] == old_id and "rl_pwm_7d" not in w._rl_prov_obj
        # picking it again flies the new files, recorded as the panel's pick
        w.set_rl_policy("rl_pwm_7d", str(exp))
        assert w._rl_pwm7d is not None and w._rl_prov["rl_pwm_7d"]["id"] == new_id, w.reason
        assert w._rl_prov["rl_pwm_7d"]["source"] == "panel"
    finally:
        h.w.teardown()


def test_a_held_follower_never_hashed_at_launch_is_not_rebuilt_by_a_cap_change(tmp_path, monkeypatch):
    """Review hwsafe-1 / worker-4 / tests-1: a launch build that succeeded while its export could not be hashed
    (record id None, hash_error) cannot be proven to rebuild the same network — refused / dropped like D4."""
    exp = _copy(ATT, tmp_path / "lib" / "unhashed")
    orig = W.MpcWorker._rl_read_export
    calls = {"n": 0}

    def flaky(d):
        if os.path.abspath(str(d)) == str(exp) and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("synthetic: being rewritten")
        return orig(d)

    monkeypatch.setattr(W.MpcWorker, "_rl_read_export", staticmethod(flaky))
    h = _mk(tmp_path / "w", _yaml(0.25, d7=str(exp)))
    try:
        w = h.w
        rec = w._rl_prov["rl_pwm_7d"]
        assert w._rl_pwm7d is not None and rec["id"] is None and "synthetic" in rec["hash_error"], rec
        why = (f"unhashed was not hashed when it was loaded ({rec['hash_error']}) — a cap change cannot prove it "
               f"rebuilds the same network; pick it again with … to fly it at this cap")
        _assert_refused(h, monkeypatch, "rl_pwm_7d", 0.5, exact=f"RL_PWM_7D cannot fly unhashed at cap 0.5: {why}")
        w.set_rl_pwm_cap("rl_pwm", 0.5)
        assert w._rl_pwm7d is None and w._rl_pwm7d_error == f"at cap 0.5: {why}", w.reason
    finally:
        h.w.teardown()


# ----------------------------------------------------------------------------- a pick after a change
def test_a_pick_after_a_cap_change_builds_at_the_new_cap(shipped):
    h, w = shipped, shipped.w
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert w._rl_pwm7d.pwm_cap == 0.5 and not w._rl_pwm7d.caps_below_trained, w.reason
    assert w.reason.startswith(f"RL policy RL_PWM_7D: {ATT2} (id ") and "WARN" not in w.reason
    # back to 0.25 (RL_PWM dropped again): a pick of RL_PWM's 0.5-trained export is refused by its build — and the
    # refusal names the runtime remedy, the LOW row's cap ▾ (C3 remedy texts)
    w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
    assert w._rl_pwm is None and w._rl_pwm7d.pwm_cap == 0.25
    w.set_rl_policy("rl_pwm", str(STATION / LEVEL))
    assert w.reason.startswith("RL policy refused: RL_PWM: ValueError: rl_pwm: the config cap rl_pwm.pwm_cap 0.25 is "
                               "below the policy's trained cap 0.50"), w.reason
    assert w.reason.endswith(" — the cap ▾ on the RL_PWM_7D row sets rl_pwm.pwm_cap for this session"), w.reason
    # ...and a pick for RL_PWM_7D builds at 0.25 with its below-trained WARN
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT))
    assert w._rl_pwm7d.pwm_cap == 0.25 and w.reason.endswith(
        " — WARN: flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)"), w.reason


# ----------------------------------------------------------------------------- end to end with a real sink
def _sink(tmp_path, policy_dir: str = REL + LEVEL, cap: float = 0.25):
    """test_rl_pwm._sink's pattern on this test's own config (independent of the operator's hw_mpc.yaml)."""
    from rov_gui.backends.hardware import MavlinkCommandSink
    from rov_gui.tests.test_control import _app
    from rov_gui.tests.test_rl_pwm import _Log, _Master

    _app()
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"sink_mpc_{cap:g}.yaml"
    p.write_text(f"mode: dobmpc\nrl_pwm:\n  policy_dir: {policy_dir}\n  pwm_cap: {cap}\n"
                 f"  require_thruster_probe: false\n", encoding="utf-8")
    bus = types.SimpleNamespace(log=_Log(), payload=types.SimpleNamespace(emit=lambda *_: None))
    opts = types.SimpleNamespace(cmd_sysid=255, mavlink_out="udpin:0.0.0.0:14552", mpc_config=str(p))
    s = MavlinkCommandSink(bus, opts)
    s.master = _Master(True)
    s.listening = True
    s._setup_pwm_gate()
    return s, bus.log


@pytest.mark.parametrize("d7", [ATT, SHIPPED_7D])
def test_end_to_end_the_change_moves_the_sinks_window_and_engage_agrees_again(tmp_path, d7):
    """The operator's flow on the shipped shape: transport 0.25, the sink's gate built from rl_pwm.policy_dir (LEVEL,
    trained 0.5) at window 0.25; RL_PWM_7D flies at 0.25; the empty-jaw hold passed -> raise to 0.5 on the LOW row.
    The worker judges the change against the window the sink WILL enforce (min(0.5, 0.5)) — never the 0.25 it
    enforces now, which would refuse every change — commits, tells the sink (wired as hardware.py wires it), and the
    sink's next snapshot agrees with the flying follower again."""
    _skip_if_absent(d7)
    from rov_gui.tests.test_rl_pwm import _answer_all

    h = _mk(tmp_path / "w", _yaml(0.25, d7=d7))
    try:
        w = h.w
        s, slog = _sink(tmp_path / "s", cap=0.25)
        assert s._pwm_gate is not None, s._pwm_gate_error
        _answer_all(s)                                            # the vehicle answered every gate parameter
        h.bus.cmd_pwm_cap.connect(s.set_pwm_cap)

        def tick():
            s._pwm_tick(time.monotonic())
            _tel(w, s.pwm_gate_state())
            return s.pwm_gate_state()

        g = tick()
        assert (g["ok"], g["pwm_cap"], g["pwm_cap_source"], g["cap_trained"]) == (True, 0.25, "hw_mpc.yaml", 0.5), g
        st = w._rl_pwm_cap_status()
        assert (st["ceiling"], st["sink_cap"]) == (0.5, 0.25)
        assert st["choices"] == [v for v in W.RL_PWM_CAP_CHOICES if v <= 0.5]
        w.set_mode("rl_pwm_7d")
        assert w.cfg.mode == "rl_pwm_7d", w.reason
        assert w._pwm_gate_now(Telemetry(pwm_gate=g)) == (True, "ok"), "at launch both fly 0.25"
        # above the sink export's trained cap: refused, the sink untouched
        w.set_rl_pwm_cap("rl_pwm_7d", 0.6)
        assert w.reason == (f"{REFUSED}the thruster sink's export {REL}{LEVEL} is trained at cap 0.5: the transport "
                            f"cannot exceed it") and s._pwm_cap == 0.25
        # 0.5: committed; the sink re-windows (its own line, file only), its next snapshot says so
        w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
        assert w.reason.startswith("RL_PWM cap 0.25 → 0.5 (panel; RL_PWM 0.5, RL_PWM_7D 0.5)"), w.reason
        assert w.ctrl is w._rl_pwm7d and w.ctrl.pwm_cap == 0.5 and h.caps == [0.5]
        assert (s._pwm_cap, s._pwm_cap_source) == (0.5, "panel")
        said = [(lvl, m) for lvl, m in slog.lines if m.startswith("rl_pwm transport cap ")]
        assert said == [("info", "rl_pwm transport cap 0.25 → 0.5 (panel 0.5; the gate's export trained 0.5)")]
        ok, why = w._pwm_gate_now(Telemetry(pwm_gate=g))
        # the OLD snapshot (window 0.25) no longer agrees — engage waits for the sink's next tick; review hwsafe-2:
        # said as such, never "set the cap to 0.5" (the cap already in force — re-picking it is a no-op)
        assert not ok and why == ("the thruster sink has not confirmed the new cap 0.5 yet (its last report predates "
                                  "the change) — try again in a moment"), why
        g = tick()
        assert (g["pwm_cap"], g["pwm_cap_source"]) == (0.5, "panel")
        assert w._pwm_gate_now(Telemetry(pwm_gate=g)) == (True, "ok"), "the engage check agrees again"
        st = w._rl_pwm_cap_status()
        assert (st["cap"], st["source"], st["sink_cap"], st["ceiling"]) == (0.5, "panel", 0.5, 0.5)
        sg = w._controller_meta()["rl_policy"]["sink_gate"]
        assert sg["export"] == REL + LEVEL and sg["pwm_cap"] == 0.5, sg
        # ...and back to 0.25 (RL_PWM, allow false, is dropped; the flying RL_PWM_7D follows): both sides again
        w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
        assert w._rl_pwm is None and w.ctrl is w._rl_pwm7d and w.ctrl.pwm_cap == 0.25, w.reason
        g = tick()
        assert (s._pwm_cap, g["pwm_cap"]) == (0.25, 0.25) and h.caps == [0.5, 0.25]
        assert w._pwm_gate_now(Telemetry(pwm_gate=g)) == (True, "ok")
        # a sink that REFUSED the change (here: an engagement it still thinks is open) — its report is newer than the
        # commit and shows the old window: the sink has not applied the cap in force; never "set the cap to 0.4"
        s._pwm_open = True
        w.set_rl_pwm_cap("rl_pwm_7d", 0.4)
        assert w.ctrl.pwm_cap == 0.4 and s._pwm_cap == 0.25, w.reason
        g = tick()
        ok, why = w._pwm_gate_now(Telemetry(pwm_gate=g))
        assert not ok and why.startswith("the thruster sink enforces pwm_cap 0.25 but this export flies 0.4 at the cap "
                                         "in force 0.4 — the sink has not applied that cap"), why
        assert "set the LOW row's cap to" not in why
    finally:
        h.w.teardown()


# ----------------------------------------------------------------------------- the log filter, real emitters
def test_the_cap_lines_through_the_mission_log_filter(both, monkeypatch):
    h, w = both, both.w
    n = len(h.logs)
    w.set_rl_pwm_cap("rl", 0.25)
    w.set_rl_pwm_cap("rl_pwm", 0.25)
    refusals = [(lvl, m) for lvl, m in h.logs[n:]]
    assert [m.split(": ", 1)[1].split(": ", 1)[0] for _, m in refusals] == ["RL_PWM cap refused"] * 2, refusals
    assert all(panel_text(lvl, m) == m for lvl, m in refusals), "a refused change is shown"
    n = len(h.logs)
    w.set_rl_pwm_cap("rl_pwm_7d", 0.25)                        # RL_PWM dropped, RL_PWM_7D below trained
    assert any("RL_PWM unavailable at cap 0.25 — " in m for _, m in h.logs[n:])
    assert all(panel_text(lvl, m) is None for lvl, m in h.logs[n:]), \
        "a committed change, its build lines and a pwm LOW it dropped are file only (the LOW row shows them)"
    assert h.events[-1].startswith("RL_PWM cap 0.5 → 0.25 (panel; ") and not mission_shown(h.events[-1])
    assert mission_shown("RL_PWM gate lost — disengaged"), "NOT 'RL_PWM ' alone: the transport's run events stay"


# ----------------------------------------------------------------------------- the command sink (C4)
def test_the_sink_re_windows_to_the_min_with_its_gate_exports_trained_cap(tmp_path):
    s, log = _sink(tmp_path, cap=0.25)
    assert s._pwm_gate is not None, s._pwm_gate_error
    gate = s._pwm_gate
    assert (s._pwm_cap, s._pwm_cap_trained, s._pwm_cap_source) == (0.25, 0.5, "hw_mpc.yaml")
    for c, want in ((0.5, 0.5), (0.3, 0.3), (0.9, 0.5), (1.0, 0.5), (0.1, 0.1), (0.25, 0.25), (1, 0.5)):
        prev = s._pwm_cap
        s.set_pwm_cap(c)
        assert (s._pwm_cap, s._pwm_cap_source) == (want, "panel"), c
        assert log.lines[-1] == ("info", f"rl_pwm transport cap {prev:g} → {want:g} (panel {float(c):g}; the gate's "
                                         f"export trained 0.5)"), log.lines[-1]
    assert s._pwm_gate is gate and s._pwm_cap_trained == 0.5, "the window moves; the gate is never rebuilt"
    # a gate built at a launch cap ABOVE the export's trained cap enforces the trained one (as _setup_pwm_gate did)
    s2, _ = _sink(tmp_path / "hi", cap=1.0)
    assert (s2._pwm_cap, s2._pwm_cap_trained) == (0.5, 0.5)


def test_the_sink_refuses_a_bad_value_no_gate_and_flowing_pulses_and_says_so(tmp_path):
    from rov_gui.tests.test_rl_pwm import _answer_all

    s, log = _sink(tmp_path, cap=0.25)
    for bad in (0, 0.0, -0.1, 1.5, float("nan"), float("inf"), True, "0.5", None):
        n = len(log.lines)
        s.set_pwm_cap(bad)
        assert (s._pwm_cap, s._pwm_cap_source) == (0.25, "hw_mpc.yaml"), bad
        assert log.lines[n:] == [("error", f"rl_pwm: transport cap NOT changed — {bad!r} is not a number in (0, 1]")]
        assert panel_text(*log.lines[-1]) == log.lines[-1][1], "an error is always on the panel"
    # while an rl_pwm ENGAGEMENT is open (defence in depth: the worker refuses a change while engaged)
    _answer_all(s)
    s.enabled = True
    open_txt = ("error", "rl_pwm: transport cap NOT changed while an rl_pwm engagement is open")
    s.set_pwm(PwmCommand(pulses=PW_025, source="mpc"))
    assert s._pwm_active and s._pwm_open, log.lines
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.25 and log.lines[-1] == open_txt
    # review hwsafe-3: an engaged follower's own neutrals (a coast, the gate grace) pause the pulses — _pwm_active
    # drops — but the engagement goes on: still refused
    s.set_pwm(PwmCommand(source="mpc"))
    assert not s._pwm_active and s._pwm_open
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.25 and log.lines[-1] == open_txt
    # ...and a latched fault (only the follower's stop / idle clears it) is refused too, naming the latch, not an
    # engagement (final review 2026-11-02)
    s._pwm_latch = "a test latch"
    s._pwm_open = False
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.25 and log.lines[-1] == (
        "error", "rl_pwm: transport cap NOT changed while a gate fault is latched (a test latch) — the follower's "
                 "stop / idle clears it")
    assert panel_text(*log.lines[-1]) == log.lines[-1][1], "an error is always on the panel"
    s._pwm_open = True
    s.set_pwm(PwmCommand(source="stop"))                       # the follower's disengage: ends both
    assert not s._pwm_active and not s._pwm_open and not s._pwm_latch
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.5
    # no gate (its export cannot be read): said, nothing changes
    s3, log3 = _sink(tmp_path / "nogate", policy_dir=str(tmp_path / "missing"))
    assert s3._pwm_gate is None and (s3._pwm_cap, s3._pwm_cap_trained) == (0.0, 0.0)
    s3.set_pwm_cap(0.5)
    assert (s3._pwm_cap, s3._pwm_cap_source) == (0.0, "hw_mpc.yaml")
    lvl, msg = log3.lines[-1]
    assert lvl == "warn" and msg == f"rl_pwm: transport cap NOT changed — no rl_pwm gate ({s3._pwm_gate_error})", msg
    s3.set_pwm_cap(float("nan"))                               # the value is judged first (C4's order)
    assert log3.lines[-1] == ("error", "rl_pwm: transport cap NOT changed — nan is not a number in (0, 1]")


def test_the_sink_snapshot_carries_the_window_and_who_set_it(tmp_path):
    s, _ = _sink(tmp_path, cap=0.25)
    s._pwm_tick(time.monotonic())
    st = s.pwm_gate_state()
    assert (st["pwm_cap"], st["pwm_cap_source"], st["cap_trained"], st["configured"]) == \
        (0.25, "hw_mpc.yaml", 0.5, True)
    s.set_pwm_cap(0.5)
    assert s.pwm_gate_state() is st and st["pwm_cap"] == 0.25, "published by the next tick, never by the slot"
    s._pwm_tick(time.monotonic())
    st2 = s.pwm_gate_state()
    assert (st2["pwm_cap"], st2["pwm_cap_source"]) == (0.5, "panel") and st["pwm_cap"] == 0.25


def test_a_pulse_beyond_the_new_window_is_still_refused_and_latched(tmp_path):
    from rov_gui.tests.test_rl_pwm import _answer_all, _motors

    s, log = _sink(tmp_path, cap=0.5)                           # 1500 +- 200 us
    _answer_all(s)
    s.enabled = True
    s.set_pwm(PwmCommand(pulses=PW_WIDE, source="mpc"))
    assert s._pwm_active and _motors(s.master.mav.rc[-1]) == PW_WIDE, log.lines
    s.set_pwm(PwmCommand(source="stop"))
    s.set_pwm_cap(0.25)                                         # 1500 +- 100 us from here on
    assert s._pwm_cap == 0.25
    n = len(s.master.mav.rc)
    s.set_pwm(PwmCommand(pulses=PW_WIDE, source="mpc"))
    assert not s._pwm_active and "outside the cap window 1500 +- 100 us" in s._pwm_latch, s._pwm_latch
    assert all(_motors(f) == RPW.NEUTRAL_PULSES for f in s.master.mav.rc[n:]), "refused and neutralised, not clamped"
    assert any(lvl == "error" and "refused, not clamped" in m for lvl, m in log.lines)
    # set_pwm_cap never touches the latch — and (review hwsafe-3) is refused while one holds: the flow stays shut
    # until the follower stops, at the window it had
    latch = s._pwm_latch
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.25 and s._pwm_latch == latch
    n = len(s.master.mav.rc)
    s.set_pwm(PwmCommand(pulses=PW_WIDE, source="mpc"))
    assert not s._pwm_active and all(_motors(f) == RPW.NEUTRAL_PULSES for f in s.master.mav.rc[n:])
    s.set_pwm(PwmCommand(source="stop"))                        # the follower's disengage clears it
    assert s._pwm_latch == ""
    s.set_pwm_cap(0.5)
    assert s._pwm_cap == 0.5
    s.set_pwm(PwmCommand(pulses=PW_WIDE, source="mpc"))
    assert s._pwm_active and _motors(s.master.mav.rc[-1]) == PW_WIDE


def test_the_null_sink_drops_a_cap_change():
    from rov_gui.backends.hardware import NullCommandSink
    from rov_gui.tests.test_control import _app
    from rov_gui.tests.test_rl_pwm import _Log

    _app()
    log = _Log()
    s = NullCommandSink(types.SimpleNamespace(log=log, payload=types.SimpleNamespace(emit=lambda *_: None)),
                        types.SimpleNamespace())
    before = dict(vars(s))
    for v in (0.5, 0.25, 1.0, float("nan"), True, None):
        assert s.set_pwm_cap(v) is None
    assert log.lines == [] and vars(s) == before


def test_both_backends_wire_the_cap(tmp_path, monkeypatch):
    """A real check, not a source grep: each backend BUILT offline (never started — no thread, no device, no socket).
    bus.cmd_rl_pwm_cap reaches the worker (not set up: it refuses with the not-READY reason — the proof the signal
    reached set_rl_pwm_cap); on the hardware backend bus.cmd_pwm_cap reaches the command sink — the real
    MavlinkCommandSink with --allow-command (no gate before its setup: it says so), the NullCommandSink without; the
    demo has no sink and connects nothing to it."""
    from rov_gui.backends import make_backend
    from rov_gui.backends.hardware import NullCommandSink
    from rov_gui.bus import DataBus, FrameMailbox
    from rov_gui.tests.test_control import _app, _test_opts

    app = _app()
    base = _test_opts(str(tmp_path))
    got = []
    monkeypatch.setattr(NullCommandSink, "set_pwm_cap", lambda self, cap: got.append(cap))
    for src, allow in (("demo", False), ("hw", False), ("hw", True)):
        class Opts(base):
            fps = 15.0
            thrusters = 8
            rec_dir = str(tmp_path)
            policy = False
            pose = False
            fstereo = False
            allow_command = allow
            land_dry_run = False
            panel2 = "none"
            ip = "127.0.0.1"

        bus = DataBus()
        logs = []
        bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
        be = make_backend(src, bus, {k: FrameMailbox(k) for k in ("main", "second", "depth")}, Opts())
        assert be.mpc is not None and not be.mpc._ready, src
        bus.cmd_rl_pwm_cap.emit("rl_pwm_7d", 0.5)
        app.processEvents()
        assert be.mpc.reason == REFUSED + "controller still building — pick again when READY", (src, be.mpc.reason)
        n = len(got)
        bus.cmd_pwm_cap.emit(0.5)
        app.processEvents()
        if src == "demo":
            assert getattr(be, "sink", None) is None and len(got) == n
        elif not allow:
            assert isinstance(be.sink, NullCommandSink) and got[n:] == [0.5]
        else:
            assert type(be.sink).__name__ == "MavlinkCommandSink" and len(got) == n
            assert ("warn", "rl_pwm: transport cap NOT changed — no rl_pwm gate (rl_pwm is not configured)") in logs


# ----------------------------------------------------------------------------- the review fixes (2026-11-02)
def _trained_025(d: Path) -> Path:
    """A git-tracked station export (pwm16N_s3) whose obs_spec claims TRAINED cap 0.25 — test_rl_policy_dialog's
    _cap_copy (one exact text edit, asserted present once)."""
    from rov_gui.tests.test_rl_policy_dialog import _cap_copy

    return _cap_copy(d, "0.25")


def test_the_cap_hint_names_a_cap_row_only_when_one_can_be_reached(shipped, tmp_path):
    """Review D3 / worker-1 / tests-5 / worker-2: set_mode's refusal of a pwm LOW that a higher cap would cure names
    the cap ▾ on a row the operator can SELECT (a pwm LOW holding a follower) — never one that cannot appear, never
    for a trained cap above the sink's ceiling; the heave-trim budget refusal (also a cap) gets it too."""
    h, w = shipped, shipped.w
    hint = " — the cap ▾ on the RL_PWM_7D row sets rl_pwm.pwm_cap for this session"
    n = len(h.logs)
    w.set_mode("rl_pwm")
    assert w.cfg.mode == "dobmpc" and w.reason.startswith("mode rl_pwm unavailable (ValueError: rl_pwm: the config "
                                                          "cap rl_pwm.pwm_cap 0.25 is below the policy's trained cap "
                                                          "0.50"), w.reason
    assert w.reason.endswith(hint), w.reason
    assert h.logs[n:][-1] == ("error", f"ctrl[dobmpc]: {w.reason} — staying on dobmpc"), h.logs[n:]
    assert w._rl_policy_status()["rl_pwm"]["note"].endswith(hint), "the row / the picker's held line say it too"
    # the cap it needs (0.50) above the sink's ceiling: no LOW-row cap reaches it — no hint
    _tel(w, _gate(cap=0.25, cap_trained=0.25))
    w.set_mode("rl_pwm")
    assert "unavailable" in w.reason and "cap ▾" not in w.reason, w.reason
    # neither pwm LOW holds a follower: the combo can never show — no hint (the follower's text names hw_mpc.yaml)
    h2 = _mk(tmp_path / "none", _yaml(0.25, allow7=False))
    try:
        w2 = h2.w
        assert w2._rl_pwm is None and w2._rl_pwm7d is None
        for low in ("rl_pwm", "rl_pwm_7d"):
            w2.set_mode(low)
            assert "unavailable" in w2.reason and "cap ▾" not in w2.reason, w2.reason
            assert "cap ▾" not in w2._rl_policy_status()[low]["note"]
    finally:
        h2.w.teardown()
    # the heave-trim cap budget (rl_policy: "... or fly a higher rl_pwm.pwm_cap"): cured by a cap too
    h3 = _mk(tmp_path / "trim", _yaml(0.25, allow=True, trim7=8.0))
    try:
        w3 = h3.w
        assert w3._rl_pwm is not None and w3._rl_pwm7d is None, w3._rl_pwm7d_error
        assert "fly a higher rl_pwm.pwm_cap" in w3._rl_pwm7d_error, w3._rl_pwm7d_error
        w3.set_mode("rl_pwm_7d")
        assert w3.reason.endswith(" — the cap ▾ on the RL_PWM row sets rl_pwm.pwm_cap for this session"), w3.reason
    finally:
        h3.w.teardown()
    # any other build failure: no hint
    assert w._rl_cap_hint("rl_pwm", "ValueError: velocity_source mismatch") == ""
    assert w._rl_cap_hint("rl", "below the policy's trained cap 0.50") == "", "pwm LOWs only"


def test_a_committed_change_clears_refused_pick_notes_and_a_committed_pick_the_cap_note(shipped):
    """Review D2 / worker-5: a refusal note (~6 s) must never read as the outcome of the NEXT, committed change."""
    h, w = shipped, shipped.w
    w.set_rl_policy("rl_pwm_7d", str(STATION / BANNED_7D))                  # refused: no-fly
    assert w._rl_policy_status()["rl_pwm_7d"]["state"] == "refused"
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert w.cfg.rl_pwm["pwm_cap"] == 0.5, w.reason
    assert w._rl_policy_status()["rl_pwm_7d"]["state"] != "refused" and not w._rl_pick_note
    w.set_rl_pwm_cap("rl_pwm_7d", 1.5)                                       # refused: not in (0, 1]
    assert w._rl_pwm_cap_status()["state"] == "refused"
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))                       # a committed pwm pick
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT2, w.reason
    assert w._rl_pwm_cap_status()["state"] == "ok" and w._rl_pwm_cap_note is None


def test_an_empty_slot_that_fails_again_gets_its_error_for_the_new_cap(shipped):
    """Review tests-3: RL_PWM (empty since launch at 0.25) still cannot fly at 0.33 — its row then says 0.33's
    error, never the launch's 0.25 one."""
    h, w = shipped, shipped.w
    w.set_rl_pwm_cap("rl_pwm_7d", 0.33)
    assert w.cfg.rl_pwm["pwm_cap"] == 0.33 and w._rl_pwm is None, w.reason
    err = w._rl_pwm_error
    assert err.startswith("at cap 0.33: ValueError: rl_pwm: the config cap rl_pwm.pwm_cap 0.33 is below the "
                          "policy's trained cap 0.50"), err
    assert w._rl_policy_status()["rl_pwm"]["note"] == \
        err + " — the cap ▾ on the RL_PWM_7D row sets rl_pwm.pwm_cap for this session"


def test_every_rebuilt_slot_is_judged_at_the_window_the_sink_will_enforce(tmp_path, monkeypatch):
    """Review ui-2 / ui-3 / tests-2: a follower trained BELOW the new window would fly its trained cap while the
    sink enforces the window — engage would refuse it. The other slot is DROPPED (its row never says OK for it), the
    requested / flying one REFUSES the change, and the words never name the cap already in force."""
    low025 = _trained_025(tmp_path / "lib" / "q025")
    h = _mk(tmp_path / "w", _yaml(0.25, rl_pwm=str(low025)))
    try:
        w = h.w
        assert w._rl_pwm is not None and w._rl_pwm.pwm_cap == 0.25, w._rl_pwm_error
        _tel(w, _gate(cap=0.25))                                  # the sink: window 0.25, its export trained 0.5
        text = ("RL_PWM holds q025, trained at cap 0.25: at cap 0.5 it would fly 0.25 while the thruster sink "
                "enforces 0.5 — keep the cap at 0.25 or below, or first pick (…) an export trained at 0.5 or above "
                "for RL_PWM")
        # requested on RL_PWM: refused, nothing changes
        _assert_refused(h, monkeypatch, "rl_pwm", 0.5, exact=text)
        assert "set the LOW row's cap to" not in w.reason
        # requested on RL_PWM_7D: committed; RL_PWM, the OTHER slot, dropped with the same words
        old = w._rl_pwm
        w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
        assert w.reason == "RL_PWM cap 0.25 → 0.5 (panel; RL_PWM ERROR, RL_PWM_7D 0.5)", w.reason
        assert w._rl_pwm is None and old is not None and w._rl_pwm_error == f"at cap 0.5: {text}"
        assert w._rl_policy_status()["rl_pwm"]["state"] == "error" and h.caps == [0.5]
        # back to 0.25 (the sink re-windowed to 0.5): RL_PWM revives from its kept record, the same bytes
        _tel(w, _gate(cap=0.5))
        w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
        assert w._rl_pwm is not None and w._rl_pwm.pwm_cap == 0.25 and w._rl_pwm_error is None, w.reason
        assert w._rl_prov["rl_pwm"]["name"] == "q025"
    finally:
        h.w.teardown()


def test_a_meta_written_before_the_sink_confirms_marks_its_window_stale(shipped):
    """Review worker-3: a REC re-dump right after a change reads the snapshot the sink published BEFORE it — the
    record says the window is stale instead of implying a follower-vs-sink mismatch that never flew."""
    h, w = shipped, shipped.w
    _tel(w, _gate(cap=0.25))
    w.set_mode("rl_pwm_7d")
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert w.cfg.rl_pwm["pwm_cap"] == 0.5, w.reason
    sg = w._controller_meta()["rl_policy"]["sink_gate"]
    assert (sg["pwm_cap"], sg.get("pwm_cap_stale")) == (0.25, True), sg
    time.sleep(0.001)
    _tel(w, _gate(cap=0.5))                                       # the sink's next report
    sg = w._controller_meta()["rl_policy"]["sink_gate"]
    assert sg["pwm_cap"] == 0.5 and "pwm_cap_stale" not in sg, sg


def test_a_refused_pwm_pick_retires_an_older_refused_cap_note(both):
    """Review worker-5, part (b) (the workflow's backend fixer, 2026-11-02): the panel's cap face outranks every face
    on BOTH pwm rows, so a newer refused pick on a pwm slot must retire the older cap refusal — else its row keeps
    "cap refused: …" for up to RL_PICK_NOTE_S and hides the pick's own refusal. A refused pick on another RL LOW
    (whose row shows no cap face) leaves it."""
    h, w = both, both.w
    w.set_rl_pwm_cap("rl_pwm", 0.25)               # refused: RL_PWM refuses caps below trained (allow false)
    assert w.reason.startswith(REFUSED) and w._rl_pwm_cap_status()["state"] == "refused", w.reason
    w.set_rl_policy("rl", str(STATION / BANNED))                            # refused on RL: the cap note stays
    assert w.reason.startswith("RL policy refused: RL: ") and w._rl_pwm_cap_status()["state"] == "refused", w.reason
    w.set_rl_policy("rl_pwm", str(STATION / BANNED))                        # refused on RL_PWM: no-fly
    assert w.reason.startswith(f"RL policy refused: RL_PWM: {BANNED} is no-fly"), w.reason
    st = w._rl_pwm_cap_status()
    assert (st["state"], st["note"], w._rl_pwm_cap_note) == ("ok", "", None), st
    rp = w._rl_policy_status()["rl_pwm"]
    assert rp["state"] == "refused" and rp["note"].startswith(f"{BANNED} is no-fly"), rp


def test_a_pwm_low_refused_for_a_reason_no_cap_cures_gets_no_cap_hint(tmp_path):
    """Review D3 (the workflow's backend fixer, 2026-11-02): the hint rides only a build error a higher cap cures. A
    cap ▾ row IS reachable here (RL_PWM holds a follower — its opt-out admits 0.25) while RL_PWM_7D is refused for
    another reason (a LEVEL-trained export): set_mode's refusal and the row's ERROR note carry no hint."""
    h = _mk(tmp_path, _yaml(0.25, allow=True, d7=LEVEL))
    try:
        w = h.w
        assert w._rl_pwm is not None and w._rl_pwm7d is None
        err = w._rl_pwm7d_error
        assert "trained with a LEVEL goal" in err and not any(k in err for k in W.MpcWorker._RL_CAP_CURABLE), err
        n = len(h.logs)
        w.set_mode("rl_pwm_7d")
        assert w.cfg.mode == "dobmpc" and w.reason == f"mode rl_pwm_7d unavailable ({err})", w.reason
        assert h.logs[n:][-1] == ("error", f"ctrl[dobmpc]: {w.reason} — staying on dobmpc"), h.logs[n:]
        assert w._rl_policy_status()["rl_pwm_7d"]["note"] == err
        # ...while a failure a higher cap cures would name the reachable row (so the checks above are not vacuous)
        assert w._rl_cap_hint("rl_pwm_7d", "below the policy's trained cap 0.50") == \
            " — the cap ▾ on the RL_PWM row sets rl_pwm.pwm_cap for this session"
    finally:
        h.w.teardown()


def test_a_cap_change_never_swaps_the_flying_network_either(tmp_path, monkeypatch):
    """Review D4, the FLYING slot (the workflow's backend fixer, 2026-11-02): it must succeed like the requested one.
    RL_PWM_7D is selected and its files were rewritten in place since the load — a change asked on the RL_PWM row is
    refused whole (RL_PWM is not revived either), never flown on the new bytes."""
    exp = _copy(ATT, tmp_path / "lib" / "flying")
    h = _mk(tmp_path / "w", _yaml(0.25, d7=str(exp)))
    try:
        w = h.w
        w.set_mode("rl_pwm_7d")
        assert w.cfg.mode == "rl_pwm_7d" and w.ctrl is w._rl_pwm7d and w._rl_pwm is None, w.reason
        old_id = w._rl_prov["rl_pwm_7d"]["id"]
        with open(exp / "obs_spec.json", "a", encoding="utf-8") as f:
            f.write("\n")                         # rewritten in place after the launch (still a valid export)
        new_id = L.export_id(*L.export_hashes(exp))
        assert new_id != old_id
        _assert_refused(h, monkeypatch, "rl_pwm", 0.5,
                        exact=f"RL_PWM_7D cannot fly flying at cap 0.5: flying's files changed since it was loaded "
                              f"(id {old_id[:8]} → {new_id[:8]}) — a cap change never swaps the network; pick it "
                              f"again with … to fly the new files")
    finally:
        h.w.teardown()


def test_no_cap_hint_for_an_empty_slot_whose_export_a_cap_change_never_builds(tmp_path):
    """Review worker-1 (the workflow's backend fixer, 2026-11-02): RL_PWM names a banned export (by name, or the class
    default when rl_pwm.policy_dir is absent), trained 1.0 — its launch build at 0.25 failed on the cap, a failure a
    higher cap cures in words. But a cap change never builds an EMPTY slot naming a banned export, so the cap ▾ on the
    RL_PWM_7D row (reachable here) would never revive it: no hint."""
    for d in (BANNED, None):
        h = _mk(tmp_path / str(d), _yaml(0.25, rl_pwm=d))
        try:
            w = h.w
            err = w._rl_pwm_error
            assert w._rl_pwm is None and w._rl_pwm7d is not None, err
            assert "below the policy's trained cap 1.00" in err, err
            w.set_mode("rl_pwm")
            assert w.cfg.mode == "dobmpc" and w.reason == f"mode rl_pwm unavailable ({err})", w.reason
            assert w._rl_policy_status()["rl_pwm"]["note"] == err
        finally:
            h.w.teardown()


def test_a_pick_right_after_a_change_waits_for_the_sink_to_confirm_it(shipped):
    """Review hwsafe-2, the PICK half (the workflow's backend fixer, 2026-11-02): a pwm pick judged on the snapshot the
    sink published BEFORE the change says the sink has not confirmed the new cap yet — never "set the LOW row's cap to
    0.5" (the cap already in force: re-picking it is a no-op) nor "pick another cap and back"; the sink's next report
    lets the same pick through."""
    h, w = shipped, shipped.w
    _tel(w, _gate(cap=0.25))                                      # the sink: window 0.25, its export trained 0.5
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert w.cfg.rl_pwm["pwm_cap"] == 0.5 and h.caps == [0.5], w.reason
    held = w._rl_pwm7d
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert w._rl_pwm7d is held, "refused: nothing swapped"
    assert w.reason == ("RL policy refused: RL_PWM_7D: the thruster sink has not confirmed the new cap 0.5 yet (its "
                        "last report predates the change) — try again in a moment"), w.reason
    time.sleep(0.001)
    _tel(w, _gate(cap=0.5))                                       # the sink's next report
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert w._rl_prov["rl_pwm_7d"]["name"] == ATT2 and w._rl_pwm7d.pwm_cap == 0.5, w.reason


def test_the_sink_is_said_to_lag_only_when_the_new_window_fits_the_export(shipped, tmp_path):
    """Review hwsafe-2's condition (the workflow's backend fixer, 2026-11-02): "the sink has not confirmed / applied
    the cap" is said only when the follower flies the window the sink WILL enforce at the cap in force — min(cap, its
    gate export's trained cap). An export trained BETWEEN the old and the new window (0.3: flies 0.3 at 0.5, the sink
    will enforce 0.5) disagrees with the new window too: waiting, or another cap and back, would not help it — the
    remedy sentence names the cap that fits."""
    from rov_gui.tests.test_rl_policy_dialog import _cap_copy

    h, w = shipped, shipped.w
    q03 = _cap_copy(tmp_path / "lib" / "q03", "0.3")
    _tel(w, _gate(cap=0.25))                                      # the sink: window 0.25, its export trained 0.5
    w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
    assert w.cfg.rl_pwm["pwm_cap"] == 0.5 and h.caps == [0.5], w.reason
    remedy = "set the LOW row's cap to 0.3 (rl_pwm.pwm_cap for this session)"
    # the sink's last report predates the change
    w.set_rl_policy("rl_pwm_7d", str(q03))
    assert w.reason.startswith("RL policy refused: RL_PWM_7D: ") and remedy in w.reason, w.reason
    assert "not confirmed" not in w.reason and "has not applied" not in w.reason, w.reason
    # a NEWER report still at the old window (the sink did not apply the change): the same remedy...
    time.sleep(0.001)
    _tel(w, _gate(cap=0.25))
    w.set_rl_policy("rl_pwm_7d", str(q03))
    assert remedy in w.reason and "has not applied" not in w.reason, w.reason
    # ...while an export that fits the new window is told the sink is behind (never a cap to set)
    w.set_rl_policy("rl_pwm_7d", str(STATION / ATT2))
    assert "the sink has not applied that cap" in w.reason and "set the LOW row's cap to" not in w.reason, w.reason


def test_a_dropped_slot_whose_files_changed_is_not_revived_on_new_bytes(tmp_path):
    """Review (final regress, 2026-11-02): D4 for an EMPTY slot with a KEPT record — dropped by a lower, its files
    rewritten in place, then a raise. Reviving it would fly bytes nobody loaded under the old source / at; it stays
    ERROR with D4's words and its record keeps the id that was loaded. Picking it again (…) flies the new files."""
    lvl = _copy(LEVEL, tmp_path / "lib" / "lvl_copy")
    h = _mk(tmp_path / "w", _yaml(0.25, rl_pwm=str(lvl)))
    try:
        w = h.w
        assert w._rl_pwm is None, "RL_PWM (allow false) refused its 0.5-trained export at 0.25"
        w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
        assert w._rl_pwm is not None, w.reason
        loaded = w._rl_prov["rl_pwm"]["id"]
        w.set_rl_pwm_cap("rl_pwm_7d", 0.25)
        assert w._rl_pwm is None and w._rl_prov["rl_pwm"]["id"] == loaded, "dropped; the record kept"
        with open(lvl / "obs_spec.json", "a", encoding="utf-8") as f:
            f.write("\n")                                      # rewritten in place while the slot was empty
        obs, npz = L.export_hashes(lvl)
        new_id = L.export_id(obs, npz)
        assert new_id != loaded
        w.set_rl_pwm_cap("rl_pwm_7d", 0.5)
        assert w.cfg.rl_pwm["pwm_cap"] == 0.5, w.reason
        assert w._rl_pwm is None and w._rl_pwm_error == (
            f"at cap 0.5: lvl_copy's files changed since it was loaded (id {loaded[:8]} → {new_id[:8]}) — a cap "
            f"change never swaps the network; pick it again with … to fly the new files"), w._rl_pwm_error
        assert w._rl_prov["rl_pwm"]["id"] == loaded
        w.set_rl_policy("rl_pwm", str(lvl))
        assert w._rl_pwm is not None and w._rl_prov["rl_pwm"]["id"] == new_id, w.reason
    finally:
        h.w.teardown()
