#!/usr/bin/env python3
"""test_rl_pwm_cap_panel.py — the RL_PWM transport CAP on the trajectory panel's LOW row (2026-11-02, operator request:
"./c3 gui --source hw … --dvl_pose --dvl-trust --policy-dr 이렇게 gui를 열면 거기서 바꿀 수 있게 해달라는 소리였어. 그리고
캡도"): the `cap` label + combo after the policy `…` (widgets/trajectory.py, SPEC C6) and the window that sends a pick
(window.py _request_rl_pwm_cap, C7). The worker side is driven only in the last test (the contract); everywhere else
the panel is fed MpcStatus(rl_pwm_cap={...}, rl_policy={...}) exactly as MpcWorker._rl_pwm_cap_status /
_rl_policy_status publish them (state.py MpcStatus.rl_pwm_cap, C2).

What is pinned, in the order a failure would hurt:
  * the combo never commands by itself: a status sync moves it with its signals blocked (no request, no
    `activated`), a wheel notch over it is swallowed (Qt 5's wheel handler EMITS `activated` — asserted, so the
    guard is shown to be what holds it), Esc in its open list is ONE E-STOP and picks nothing;
  * a pick is a REQUEST (low, cap) sent once, shown until the worker answers — accepted, refused with a NEW note,
    or 1.5 s of silence — then the worker's truth again; the cap in force re-picked sends nothing;
  * it shows for RL_PWM / RL_PWM_7D only, and only once the worker reports a cap; it lists the worker's "choices"
    as "0.25" text / float data, rebuilt only when they change, the current row = the cap in force;
  * frozen while engaged or with a controller CSV open (btn_rlpol's rule);
  * a refused change takes the policy row's state label — "cap refused: <note>" in FAIL red, the full note in the
    tooltip — over every face of the slot, on the two pwm LOWs only; without one every face is byte-identical to a
    panel that never saw a cap;
  * the tooltip: the spec's text verbatim + the live line (now · sink window · what this LOW flies);
  * the window sends the pick on bus.cmd_rl_pwm_cap, THEN re-writes each running recording's controller.json;
  * the cap stays readable at the station's sizes while a long refusal shows, and moves no station minimum
    (2026-11-02: as first written — minimum width 0 — the combo was squeezed below its digits exactly while a
    "cap refused: …" face showed, which this test fails on; ElidingCombo now keeps its content width);
  * `…` and the cap never move with the policy name / state texts — the controls come first on the LOW row, the
    labels last (2026-11-02 review: a refusal face pushed them right for its few seconds, and START took their
    spot when it expired); only START's left edge moves, beside the labels;
  * the panel reads the REAL worker's status and its request is what MpcWorker.set_rl_pwm_cap takes.

Offscreen, a real click in the open list does not reach `activated` (not even on a plain QComboBox — the UI agent's
note, 2026-11-02), so the operator's pick is driven as Qt delivers it: setCurrentIndex(i) then activated.emit(i).

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest -q -p no:cacheprovider \\
        rov_gui/tests/test_rl_pwm_cap_panel.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui import theme                                         # noqa: E402
from rov_gui.qt import QT_API, QtCore, QtGui, QtWidgets, Qt       # noqa: E402
from rov_gui.state import MpcStatus                               # noqa: E402
from rov_gui.tests.test_offline import Opts, _app, _pump          # noqa: E402

QTest = importlib.import_module(f"{QT_API}.QtTest").QTest
PWM_LOWS = ("rl_pwm", "rl_pwm_7d")
#: The worker's menu for the SHIPPED station (SPEC C5): RL_PWM_CAP_CHOICES up to the thruster sink's ceiling — the
#: trained cap of rl_pwm.policy_dir's export, 0.5 for pwm11k0.2_s0 — with the cap in force (0.25) among them.
SHIPPED_CHOICES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
#: ...and with no sink ceiling known (demo / stale snapshot): the whole menu.
ALL_CHOICES = SHIPPED_CHOICES + [0.60, 0.70, 0.80, 0.90, 1.00]
#: SPEC C6, verbatim: the combo's tooltip opens with this.
STATIC_TIP = ("transport cap rl_pwm.pwm_cap — the |throttle| ceiling RL_PWM and RL_PWM_7D share (session only; "
              "hw_mpc.yaml stays the launch default). A follower flies min(cap, its trained cap); the thruster sink "
              "enforces the same window. Refused while engaged or while a REC controller CSV is open.")
CAP_NOTE = "flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)"        # workers.py _rl_cap_note's text
#: what MpcWorker.set_rl_pwm_cap says when the build refuses the requested LOW (step 8) — the shape of a real one
BUILD_REFUSAL = ("RL_PWM_7D cannot fly pwm11att9X_s0 at cap 0.25: ValueError: rl_pwm_7d: the config cap "
                 "rl_pwm.pwm_cap 0.25 is below the policy's trained cap 0.50 "
                 "(rov_gui/control/rl_policies/pwm11att9X_s0) — raise rl_pwm.pwm_cap to >= 0.50, or set "
                 "rl_pwm_7d.allow_caps_below_trained: true (recorded in meta)")


# ----------------------------------------------------------------------------- helpers
def _cap(**kw) -> dict:
    """One MpcStatus.rl_pwm_cap as the worker publishes it (C2) — the shipped station by default: cap 0.25 from
    hw_mpc.yaml, a fresh sink snapshot (window 0.25, its export trained at 0.5), RL_PWM refusing caps below trained
    and RL_PWM_7D allowing them."""
    d = {"cap": 0.25, "source": "hw_mpc.yaml", "at": None, "choices": list(SHIPPED_CHOICES), "ceiling": 0.5,
         "sink_cap": 0.25, "allow_below": {"rl_pwm": False, "rl_pwm_7d": True}, "state": "ok", "note": ""}
    d.update(kw)
    return d


def _slot(name: str = "", state: str = "ok", note: str = "", cap=None, cap_trained=None, no_fly: str = "") -> dict:
    """One MpcStatus.rl_policy slot — every key the worker publishes, the pwm slots' "cap" / "cap_trained" too."""
    return {"name": name, "path": f"/x/{name}" if name else "", "rel": f"rov_gui/control/rl_policies/{name}"
            if name else "", "id": "777761e2ab" if name else "", "source": "hw_mpc.yaml" if name else "",
            "state": state, "note": note, "no_fly": no_fly, "cap": cap, "cap_trained": cap_trained}


def _policy(**over) -> dict:
    """MpcStatus.rl_policy at the shipped launch: RL_PWM_7D flies pwm11att9X_s0 at 0.25 below its trained 0.5 (it
    allows that: WARN); RL_PWM refuses caps below trained, so its launch build failed (no follower); RL holds an
    axes export; RL_7D names none. No file is read: the names are strings here."""
    d = {"rl": _slot("traj12_s2"),
         "rl_7d": _slot(state="error", note="ValueError: rl_7d: no policy_dir"),
         "rl_pwm": _slot(state="error", note="ValueError: rl_pwm: the config cap rl_pwm.pwm_cap 0.25 is below the "
                                              "policy's trained cap 0.50"),
         "rl_pwm_7d": _slot("pwm11att9X_s0", "warn", CAP_NOTE, cap=0.25, cap_trained=0.5)}
    d.update(over)
    return d


def _st(mode: str = "rl_pwm_7d", cap=None, policy=None, **kw) -> MpcStatus:
    return MpcStatus(mode=mode, rl_pwm_cap=_cap() if cap is None else cap,
                     rl_policy=_policy() if policy is None else policy, **kw)


def _items(box) -> list:
    return [(box.itemText(i), box.itemData(i)) for i in range(box.count())]


def _row(box, v: float) -> int:
    """The combo row whose data is ``v`` (to 1e-9) — asserted present."""
    for i in range(box.count()):
        if abs(float(box.itemData(i)) - v) <= 1e-9:
            return i
    raise AssertionError(f"{v} is not in the combo: {_items(box)}")


def _pick(panel, v: float) -> None:
    """The operator's pick in the open list, as Qt delivers it: the row becomes current, then `activated`."""
    i = _row(panel.rlcap_box, v)
    panel.rlcap_box.setCurrentIndex(i)
    panel.rlcap_box.activated.emit(i)


def _cut(text: str) -> str:
    """The policy row's state label cuts every face to 48 chars (widgets/trajectory.py _render_rlpol)."""
    return text if len(text) <= 48 else text[:47] + "…"


def _face(p) -> tuple:
    """Everything the policy row shows: the name and state labels' text, tooltip and sheet."""
    return (p.rlpol_name.text(), p.rlpol_name.toolTip(), p.rlpol_state.text(), p.rlpol_state.toolTip(),
            p.rlpol_state.styleSheet())


def _esc() -> QtGui.QKeyEvent:
    return QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, int(Qt.Key.Key_Escape), Qt.KeyboardModifier.NoModifier)


def _wheel(box, dy: int) -> QtGui.QWheelEvent:
    pos = QtCore.QPointF(box.rect().center())
    return QtGui.QWheelEvent(pos, QtCore.QPointF(box.mapToGlobal(box.rect().center())), QtCore.QPoint(0, 0),
                             QtCore.QPoint(0, dy), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                             Qt.ScrollPhase.NoScrollPhase, False)


def _live(p) -> str:
    """The combo tooltip's live line (the one after the spec's static text)."""
    lines = p.rlcap_box.toolTip().split("\n")
    assert lines[0] == STATIC_TIP, p.rlcap_box.toolTip()
    return lines[1]


@pytest.fixture
def panel():
    from rov_gui.widgets.trajectory import TrajectoryWindow

    app = _app()
    p = TrajectoryWindow()
    p.resize(900, 420)
    p.show()
    _pump(app, 30)
    try:
        yield p
    finally:
        p.close()
        p.deleteLater()
        _pump(app, 30)


class MpcOpts(Opts):
    mpc = True
    mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
    nav_config = str(ROOT / "config" / "hw_nav.yaml")
    nav_geometry = None


# ----------------------------------------------------------------------------- where and when it shows
def test_the_cap_combo_shows_for_the_two_pwm_lows_once_the_worker_reports_a_cap(panel):
    from rov_gui.widgets.trajectory import MODE_LABELS

    p = panel
    # what it is (C6): a "cap" label and a combo that never takes the pilot's keyboard
    assert p.rlcap_widgets == (p.lbl_rlcap, p.rlcap_box)
    assert isinstance(p.lbl_rlcap, QtWidgets.QLabel) and p.lbl_rlcap.text() == "cap"
    assert isinstance(p.rlcap_box, QtWidgets.QComboBox)
    assert p.rlcap_box.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert p.rl_pwm_cap_status() == {}

    # no status yet: hidden on every LOW
    for low in MODE_LABELS:
        p.set_mode_default(low)
        assert not any(x.isVisible() for x in p.rlcap_widgets), low
    # a worker that reports no cap ({} = no config yet), or no usable number under "cap": hidden — never a number
    # the worker did not give
    p.set_mode_default("rl_pwm_7d")
    for bad in ({}, {"source": "hw_mpc.yaml"}, _cap(cap=None), _cap(cap=float("nan")), _cap(cap=float("inf"))):
        p.add_status(_st(cap=bad))
        assert not any(x.isVisible() for x in p.rlcap_widgets), bad
    p.add_status(MpcStatus(mode="rl_pwm_7d"))                       # a status without the field at all
    assert not any(x.isVisible() for x in p.rlcap_widgets)

    # a cap reported: shown for RL_PWM / RL_PWM_7D, hidden for every other LOW
    p.add_status(_st())
    for low in MODE_LABELS:
        p.set_mode_default(low)
        want = low in PWM_LOWS
        assert all(x.isVisible() == want for x in p.rlcap_widgets), low
    # the operator's path (the LOW combo) and the worker's (a status snapping the LOW back) do the same
    p.set_mode_default("dobmpc")
    p.mode_box.setCurrentIndex(p.mode_box.findData("rl_pwm"))
    assert all(x.isVisible() for x in p.rlcap_widgets)
    p.mode_box.setCurrentIndex(p.mode_box.findData("rl"))
    assert not any(x.isVisible() for x in p.rlcap_widgets)
    p.add_status(_st("rl_pwm_7d"))                                  # the worker says RL_PWM_7D: the row follows
    assert p.current_mode() == "rl_pwm_7d" and all(x.isVisible() for x in p.rlcap_widgets)
    # ...and the worker stops reporting one: hidden again
    p.add_status(_st(cap={}))
    assert not any(x.isVisible() for x in p.rlcap_widgets)


def test_the_items_are_the_workers_choices_and_the_current_row_is_the_cap_in_force(panel):
    p, box = panel, panel.rlcap_box
    changed, acts, sent = [], [], []
    box.currentIndexChanged.connect(lambda i: changed.append(i))
    box.activated.connect(lambda i: acts.append(i))
    p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())
    # text "0.25", data the float (C6)
    assert _items(box) == [(f"{v:.2f}", v) for v in SHIPPED_CHOICES]
    assert all(type(box.itemData(i)) is float for i in range(box.count()))
    assert box.currentText() == "0.25" and box.currentData() == 0.25

    # the TRUTH moves the row — a status, never a request: the combo's signals are blocked while it syncs
    p.add_status(_st(cap=_cap(cap=0.5, source="panel", at="2026-11-02T09:30:00-0800", sink_cap=0.5)))
    assert box.currentData() == 0.5
    p.add_status(_st())
    assert box.currentData() == 0.25
    assert changed == [] and acts == [] and sent == [], (changed, acts, sent)

    # rebuilt only when the worker's choices change: a mark on a row survives a status with the same menu...
    mark = int(Qt.ItemDataRole.UserRole) + 7
    box.setItemData(0, "kept", mark)
    p.add_status(_st(cap=_cap(cap=0.4)))                            # same menu, another cap in it
    assert box.itemData(0, mark) == "kept" and box.currentData() == 0.4
    # ...and not one that changes it (no sink ceiling known: the whole menu)
    p.add_status(_st(cap=_cap(choices=list(ALL_CHOICES), ceiling=None, sink_cap=None)))
    assert box.itemData(0, mark) is None
    assert _items(box) == [(f"{v:.2f}", v) for v in ALL_CHOICES] and box.currentData() == 0.25
    assert changed == [] and acts == [] and sent == []

    # the honesty rule: a cap in force the menu lacks is still SHOWN as itself, never as a neighbouring row
    p.add_status(_st(cap=_cap(cap=0.33, choices=list(SHIPPED_CHOICES))))
    assert box.currentText() == "0.33" and box.currentData() == 0.33
    assert sent == []


# ----------------------------------------------------------------------------- the pick
def test_a_pick_is_a_request_sent_once_until_the_worker_answers(panel):
    p, box = panel, panel.rlcap_box
    sent = []
    p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())

    # the operator picks 0.5 in the open list: ONE request (the LOW the row shows, the float cap)
    _pick(p, 0.5)
    assert sent == [("rl_pwm_7d", 0.5)] and type(sent[0][1]) is float
    assert box.currentData() == 0.5
    # not answered yet (the worker still says 0.25): the request stands, nothing re-sent
    p.add_status(_st())
    assert box.currentData() == 0.5 and sent == [("rl_pwm_7d", 0.5)]
    # accepted: the cap in force IS it
    p.add_status(_st(cap=_cap(cap=0.5, source="panel", at="2026-11-02T09:30:00-0800", sink_cap=0.5)))
    assert p._rlcap_req is None and box.currentData() == 0.5
    # ...and from then on the worker's truth moves it at once (no request left to stand)
    p.add_status(_st())
    assert box.currentData() == 0.25
    # the cap in force picked again: nothing to ask
    _pick(p, 0.25)
    p._apply_rl_pwm_cap_choice(0.25)
    assert sent == [("rl_pwm_7d", 0.5)] and p._rlcap_req is None

    # refused with a NEW note (none was showing): answered at once, the row back on the cap in force
    _pick(p, 0.45)
    assert sent[-1] == ("rl_pwm_7d", 0.45) and box.currentData() == 0.45
    refused = _cap(state="refused", note=BUILD_REFUSAL)
    p.add_status(_st(cap=refused))
    assert p._rlcap_req is None and box.currentData() == 0.25
    # a second pick while that SAME refusal still shows (the worker's ~6 s): it is not this pick's answer — the
    # request stands until a refusal with another note, an acceptance or 1.5 s
    _pick(p, 0.45)
    assert sent[-1] == ("rl_pwm_7d", 0.45) and len(sent) == 3
    p.add_status(_st(cap=refused))
    assert p._rlcap_req is not None and box.currentData() == 0.45
    low, cap, t, was = p._rlcap_req
    assert (low, cap, was) == ("rl_pwm_7d", 0.45, BUILD_REFUSAL)
    p._rlcap_req = (low, cap, t - 2.0, was)                       # 1.5 s of silence
    p.add_status(_st(cap=refused))
    assert p._rlcap_req is None and box.currentData() == 0.25
    # ...while a refusal with ANOTHER note answers at once
    _pick(p, 0.45)
    p.add_status(_st(cap=_cap(state="refused", note="not while engaged")))
    assert p._rlcap_req is None and box.currentData() == 0.25
    # no answer at all (the worker silent, its cap unchanged): 1.5 s, then the truth
    p.add_status(_st())
    _pick(p, 0.3)
    p.add_status(_st())
    assert box.currentData() == 0.3
    low, cap, t, was = p._rlcap_req
    assert was is None
    p._rlcap_req = (low, cap, t - 2.0, was)
    p.add_status(_st())
    assert p._rlcap_req is None and box.currentData() == 0.25

    # the plain-method path (what `activated` calls) uses the LOW the row shows: RL_PWM here
    n = len(sent)
    p.set_mode_default("rl_pwm")
    p.add_status(_st("rl_pwm"))
    p._apply_rl_pwm_cap_choice(0.4)
    assert sent[n:] == [("rl_pwm", 0.4)]
    p._rlcap_req = None
    # a status sync never sends anything, whatever it says
    n = len(sent)
    for c in (_cap(cap=0.5), _cap(cap=0.1), _cap(state="refused", note="x"), _cap()):
        p.add_status(_st("rl_pwm", cap=c))
    assert len(sent) == n


def test_a_pick_made_in_flight_is_what_the_worker_gets_last(panel):
    """UI deviation from C6's literal "differs from status cap" (ui_notes.md, 2026-11-02): while a pick is in
    flight, the OPERATOR'S LAST pick is what goes out — re-picking the cap in force takes the pending pick back (it
    is sent: both requests reach the one MpcWorker in order, so it ends on the last), and re-picking the pending
    value is not sent twice. With the spec's literal rule the take-back would be dropped and the vehicle would end
    on a cap the operator had just un-chosen."""
    p, box = panel, panel.rlcap_box
    sent = []
    p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())
    _pick(p, 0.5)
    _pick(p, 0.5)                                                   # the pending value again: not re-sent
    assert sent == [("rl_pwm_7d", 0.5)]
    _pick(p, 0.25)                                                  # taken back: sent
    assert sent == [("rl_pwm_7d", 0.5), ("rl_pwm_7d", 0.25)]
    assert box.currentData() == 0.25
    # the worker commits 0.5 first: not this request's answer — it stands on 0.25 until the worker gets there
    p.add_status(_st(cap=_cap(cap=0.5, source="panel", at="2026-11-02T09:30:00-0800")))
    assert p._rlcap_req is not None and box.currentData() == 0.25
    p.add_status(_st(cap=_cap(source="panel", at="2026-11-02T09:30:01-0800")))
    assert p._rlcap_req is None and box.currentData() == 0.25


def test_the_combo_is_frozen_while_engaged_or_with_a_controller_csv_open(panel):
    p = panel
    p.set_mode_default("rl_pwm_7d")
    for engaged, csv_open in ((False, False), (True, False), (False, True), (True, True), (False, False)):
        p.add_status(_st(engaged=engaged, csv_open=csv_open))
        assert p.rlcap_box.isEnabled() == (not engaged and not csv_open), (engaged, csv_open)
        # the policy `…`'s rule — the worker refuses both then too
        assert p.rlcap_box.isEnabled() == p.btn_rlpol.isEnabled(), (engaged, csv_open)


def test_a_wheel_notch_over_the_combo_neither_moves_it_nor_asks(panel):
    app = _app()
    p, box = panel, panel.rlcap_box
    sent, acts = [], []
    p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))
    box.activated.connect(lambda i: acts.append(i))
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())
    assert box.isVisible() and box.isEnabled()
    # both directions: 0.25 sits mid-menu, so either notch could move it
    for dy in (120, -120):
        QtWidgets.QApplication.sendEvent(box, _wheel(box, dy))
        _pump(app, 20)
        assert box.currentData() == 0.25 and sent == [] and acts == [], dy
    # the guard is what holds it: without the panel's filter the SAME notch moves the combo AND fires `activated`
    # (Qt 5's QComboBox::wheelEvent) — the one signal that sends a request to rebuild both pwm followers
    box.removeEventFilter(p)
    try:
        QtWidgets.QApplication.sendEvent(box, _wheel(box, -120))
        _pump(app, 20)
        assert box.currentData() != 0.25, "the wheel never reached the combo — this test proves nothing"
        assert acts and sent and sent[-1][0] == "rl_pwm_7d", (acts, sent)
    finally:
        box.installEventFilter(p)
        p._rlcap_req = None
        p.add_status(_st())
    assert box.currentData() == 0.25


def test_esc_in_the_open_cap_list_is_one_estop_and_picks_nothing(panel):
    app = _app()
    p, box = panel, panel.rlcap_box
    stops, sent = [], []
    p.estop_requested.connect(lambda: stops.append(1))
    p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())

    # 1. the open list grabs the keyboard and would spend Esc on closing itself: the panel takes it as E-STOP
    box.showPopup()
    _pump(app, 30)
    assert box.view().isVisible()
    QtWidgets.QApplication.sendEvent(box.view(), _esc())
    _pump(app, 30)
    assert stops == [1] and not box.view().isVisible()
    # 2. a key click (ShortcutOverride, then the KeyPress) — once again, once
    box.showPopup()
    _pump(app, 30)
    QTest.keyClick(box.view(), Qt.Key.Key_Escape)
    _pump(app, 30)
    assert stops == [1, 1] and not box.view().isVisible()
    # 3. the handler with the list still OPEN (offscreen, the ShortcutOverride above may already hide it): the
    #    list is closed by the panel, then the E-STOP
    box.showPopup()
    _pump(app, 30)
    pop = app.activePopupWidget()
    assert pop is not None and box.view().isVisible()
    assert p.eventFilter(box.view(), _esc()) is True
    _pump(app, 30)
    assert stops == [1, 1, 1] and not box.view().isVisible() and not pop.isVisible()
    # Esc chose nothing
    assert sent == [] and box.currentData() == 0.25

    # while the list is open the operator is choosing: a status neither re-lists nor moves it under the pointer;
    # the first one after it closes does
    box.showPopup()
    _pump(app, 30)
    p.add_status(_st(cap=_cap(cap=0.5, choices=[0.25, 0.5])))
    assert box.count() == len(SHIPPED_CHOICES) and box.currentData() == 0.25
    box.hidePopup()
    _pump(app, 30)
    p.add_status(_st(cap=_cap(cap=0.5, choices=[0.25, 0.5])))
    assert _items(box) == [("0.25", 0.25), ("0.50", 0.5)] and box.currentData() == 0.5
    assert sent == []


# ----------------------------------------------------------------------------- what it says
def test_a_refused_cap_change_takes_the_policy_state_label_in_fail_red(panel):
    from rov_gui.widgets.trajectory import TrajectoryWindow

    app = _app()
    p = panel
    ref = TrajectoryWindow()                    # a panel that never sees a cap: the faces as they were
    ref.show()
    _pump(app, 30)
    try:
        ban = "flight ban — test reason"
        faces = {
            "ok": _slot("pwm11att9X_s0", "ok", cap=0.5, cap_trained=0.5),
            "warn": _slot("pwm11att9X_s0", "warn", CAP_NOTE, cap=0.25, cap_trained=0.5),
            "error": _slot(state="error", note="at cap 0.25: ValueError: below the policy's trained cap"),
            "refused": _slot("pwm11att9X_s0", "refused", "RL policy refused: RL_PWM_7D: not while engaged",
                             cap=0.25, cap_trained=0.5),
            "no_fly": _slot("att7pwm_s0", "warn", f"no-fly — {ban}", cap=0.25, cap_trained=1.0, no_fly=ban),
        }
        short = "not while engaged"
        for low in PWM_LOWS:
            for name, d in faces.items():
                pol = _policy(**{low: d})
                p.set_mode_default(low)
                ref.set_mode_default(low)
                # a refused change: the state label says it, in FAIL red, over the slot's own face; the name label
                # still names the HELD export
                p.add_status(_st(low, cap=_cap(state="refused", note=short), policy=pol))
                ref.add_status(MpcStatus(mode=low, rl_policy=pol))
                assert p.rlpol_state.text() == f"cap refused: {short}", (low, name, p.rlpol_state.text())
                css = p.rlpol_state.styleSheet()
                assert theme.FAIL in css, (low, name, css)
                assert short in p.rlpol_state.toolTip(), (low, name)
                assert (p.rlpol_name.text(), p.rlpol_name.toolTip()) == (ref.rlpol_name.text(),
                                                                         ref.rlpol_name.toolTip()), (low, name)
                # no refusal (state ok): EVERY face byte-identical to the panel that never saw a cap
                p.add_status(_st(low, cap=_cap(), policy=pol))
                assert _face(p) == _face(ref), (low, name)
        # the worker's long build refusal: cut on the row like every face, whole in the tooltip
        p.set_mode_default("rl_pwm_7d")
        p.add_status(_st(cap=_cap(state="refused", note=BUILD_REFUSAL)))
        full = f"cap refused: {BUILD_REFUSAL}"
        assert len(full) > 48
        assert p.rlpol_state.text() == _cut(full) and p.rlpol_state.text().endswith("…")
        assert BUILD_REFUSAL in p.rlpol_state.toolTip()
        assert theme.FAIL in p.rlpol_state.styleSheet()
        # not on RL / RL_7D: the cap is the pwm transport's — their faces are their own
        for low in ("rl", "rl_7d"):
            p.set_mode_default(low)
            ref.set_mode_default(low)
            ref.add_status(MpcStatus(mode=low, rl_policy=_policy()))
            p.add_status(_st(low, cap=_cap(state="refused", note=BUILD_REFUSAL)))
            assert "cap refused" not in p.rlpol_state.text(), low
            assert _face(p) == _face(ref), low
        # the refusal expires (the worker's ~6 s): RL_PWM_7D's own WARN face again
        p.set_mode_default("rl_pwm_7d")
        p.add_status(_st())
        assert p.rlpol_state.text() == "WARN" and theme.WARN in p.rlpol_state.styleSheet()
    finally:
        ref.close()
        ref.deleteLater()
        _pump(app, 30)


def test_the_combo_tooltip_carries_the_spec_text_and_the_live_line(panel):
    p = panel
    p.set_mode_default("rl_pwm_7d")
    p.add_status(_st())
    assert p.rlcap_box.toolTip().startswith(STATIC_TIP)
    # the shipped launch: the transport at 0.25, the sink's window 0.25, RL_PWM_7D flying 0.25 of its trained 0.5
    assert _live(p) == "now 0.25 (hw_mpc.yaml) · sink window 0.25 · this LOW flies 0.25 (trained 0.5)"
    # after a panel change: where it came from and when
    at = "2026-11-02T09:30:00-0800"
    pol = _policy(rl_pwm_7d=_slot("pwm11att9X_s0", cap=0.5, cap_trained=0.5),
                  rl_pwm=_slot("pwm11k0.2_s0", cap=0.5, cap_trained=0.5))
    p.add_status(_st(cap=_cap(cap=0.5, source="panel", at=at, sink_cap=0.5), policy=pol))
    assert _live(p) == f"now 0.50 (panel, {at}) · sink window 0.5 · this LOW flies 0.5 (trained 0.5)"
    # no sink snapshot (demo / stale) and a LOW with no follower: dashes, never a guess
    p.set_mode_default("rl_pwm")
    p.add_status(_st("rl_pwm", cap=_cap(ceiling=None, sink_cap=None, choices=list(ALL_CHOICES))))
    assert _live(p) == "now 0.25 (hw_mpc.yaml) · sink window — · this LOW flies — (trained —)"
    # it follows the LOW the row shows
    p.add_status(_st("rl_pwm", cap=_cap(cap=0.5, source="panel", at=at, sink_cap=0.5), policy=pol))
    assert _live(p) == f"now 0.50 (panel, {at}) · sink window 0.5 · this LOW flies 0.5 (trained 0.5)"


def test_rl_pwm_cap_status_is_the_workers_dict_as_a_copy(panel):
    """What the window hands the RL policy dialog (cap_status=…): the worker's dict, a DEEP copy — the dialog must
    not be able to edit the panel's truth through the nested choices / allow_below."""
    p = panel
    assert p.rl_pwm_cap_status() == {}
    cs = _cap(state="refused", note="x")
    p.add_status(_st(cap=cs))
    got = p.rl_pwm_cap_status()
    assert got == cs and got is not cs
    got["choices"].append(9.0)
    got["allow_below"]["rl_pwm"] = True
    got["cap"] = 1.0
    assert p.rl_pwm_cap_status() == cs
    cs["choices"].append(7.0)                   # nor the status object it came from
    assert p.rl_pwm_cap_status()["choices"] == SHIPPED_CHOICES


# ----------------------------------------------------------------------------- the window
def test_the_window_sends_the_cap_then_rewrites_the_running_recordings_controller_json(tmp_path):
    """C7: rl_pwm_cap_requested -> MainWindow._request_rl_pwm_cap -> bus.cmd_rl_pwm_cap (low, cap), THEN — like an
    accepted policy pick (review F3) — cmd_mpc_dump_meta once per running recording, in the folders the REC buttons
    used (REC NAV: its nav_<hhmmss> dir; REC UI: the UI video's folder). Both go queued to the one MpcWorker, so
    the dump runs after the change. Nothing without a recording; nothing at all for the cap already in force."""
    from rov_gui.recorder import RecStats
    from rov_gui.window import MainWindow

    app = _app()
    win = MainWindow(MpcOpts())                 # never attached: no backend answers, the bus is the test's
    stats0 = win.recorder.stats
    try:
        panel = win._traj_panel
        sent = []
        win.bus.cmd_rl_pwm_cap.connect(lambda low, cap: sent.append(("cap", low, cap)))
        win.bus.cmd_mpc_dump_meta.connect(lambda d: sent.append(("dump", d)))
        panel.set_mode_default("rl_pwm_7d")
        panel.add_status(_st())
        run = tmp_path / "20261102" / "1102_120000"
        nav_dir, ui_path = run / "nav_120000", run / "ui_20261102_120000.mp4"

        # no recording running: the request alone — the operator's path, through the combo
        assert win._nav_rec is None and not win.recorder.stats.recording
        _pick(panel, 0.5)
        _pump(app, 30)
        assert sent == [("cap", "rl_pwm_7d", 0.5)], sent
        assert type(sent[0][2]) is float
        panel.add_status(_st(cap=_cap(cap=0.5, source="panel", at="2026-11-02T12:00:01-0800")))   # accepted

        # REC NAV and REC UI both running: the request, THEN one dump per recording
        win._nav_rec = {"dir": nav_dir}         # the key _toggle_nav_record keeps (the rest are file handles)
        win.recorder.stats = RecStats(recording=True, path=ui_path)
        sent.clear()
        _pick(panel, 0.25)
        _pump(app, 30)
        assert sent == [("cap", "rl_pwm_7d", 0.25), ("dump", str(nav_dir)), ("dump", str(run))], sent
        panel.add_status(_st())                 # accepted (back on 0.25)

        # the cap in force picked again: neither a request nor a dump
        sent.clear()
        _pick(panel, 0.25)
        _pump(app, 30)
        assert sent == [], sent

        # one recording each: REC UI only, then REC NAV only (RL_PWM's row this time)
        panel.set_mode_default("rl_pwm")
        panel.add_status(_st("rl_pwm"))
        win._nav_rec = None
        sent.clear()
        _pick(panel, 0.3)
        assert sent == [("cap", "rl_pwm", 0.3), ("dump", str(run))], sent
        panel.add_status(_st("rl_pwm"))
        panel._rlcap_req = None
        win._nav_rec = {"dir": nav_dir}
        win.recorder.stats = stats0
        sent.clear()
        _pick(panel, 0.4)
        assert sent == [("cap", "rl_pwm", 0.4), ("dump", str(nav_dir))], sent
    finally:
        # the fakes go before shutdown: it would close REC NAV's (absent) files and stop a recording never started
        win._nav_rec = None
        win.recorder.stats = stats0
        win.shutdown()
        win.close()
        win.deleteLater()
        _pump(app, 30)


# ----------------------------------------------------------------------------- the layout
def test_the_cap_stays_readable_at_the_station_and_moves_no_minimum():
    """The combo sits ON the LOW row, after the policy `…` and before START (C6), and adds no height. It keeps its
    CONTENT width where the station is laid out — the operator's 1920x1080 screen (test_control
    test_mpc_panel_lives_in_the_grid_and_fits_the_operator_screen) and the 1600x1000 the station tests use — also
    while a long "cap refused: …" face fills the state label beside it, which is exactly when the operator reads the
    cap: as first written (minimum width 0, 2026-11-02) it was squeezed below its digits here — this test fails on
    that combo — because a box layout takes a deficit equally from every shrinkable item. And the station's minimum
    size does not move when the cap shows."""
    from rov_gui.widgets.trajectory import TrajectoryWindow
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    long_refusal = _cap(state="refused", note=BUILD_REFUSAL + " " + "z" * 120)
    pol = _policy(rl_pwm_7d=_slot("pwm11att14Rw_s1_" + "x" * 20, "warn", CAP_NOTE, cap=0.25, cap_trained=0.5))

    p = TrajectoryWindow()
    p.resize(900, 420)
    p.show()
    _pump(app, 50)
    try:
        p.set_mode_default("dobmpc")
        _pump(app, 30)
        h0 = p.minimumSizeHint().height()
        p.set_mode_default("rl_pwm_7d")
        p.add_status(_st(policy=pol))
        _pump(app, 30)
        assert all(x.isVisible() for x in p.rlcap_widgets)
        assert p.minimumSizeHint().height() == h0, "the cap must add no row"
        cy = p.mode_box.geometry().center().y()
        for x in p.rlcap_widgets:
            assert abs(x.geometry().center().y() - cy) <= 1, x
        assert p.btn_rlpol.x() < p.lbl_rlcap.x() < p.rlcap_box.x() < p.btn_start.x()
    finally:
        p.close()
        p.deleteLater()
        _pump(app, 30)

    win = MainWindow(MpcOpts())
    try:
        panel = win._traj_panel
        win.resize(1600, 1000)
        win.show()
        _pump(app, 250)
        panel.set_mode_default("dobmpc")
        _pump(app, 50)
        before = win.minimumSizeHint()
        for w, h in ((1920, 1080), (1600, 1000)):
            win.resize(w, h)
            _pump(app, 120)
            for name, cs in (("ok", _cap()), ("refused", long_refusal)):
                panel.add_status(_st(cap=cs, policy=pol))
                _pump(app, 60)
                # the long face IS beside it while it is measured
                assert panel.rlpol_state.text().startswith("cap refused: ") == (name == "refused"), (w, name)
                box = panel.rlcap_box
                assert box.isVisible(), (w, name)
                assert box.width() >= box.sizeHint().width(), (w, name, box.width(), box.sizeHint().width())
                assert box.currentText() == "0.25"
                hint = win.minimumSizeHint()
                assert (hint.width(), hint.height()) == (before.width(), before.height()), (w, name, hint)
    finally:
        win.shutdown()
        # deleted HERE (test_offline test_status_text_never_moves_the_station_grid: a window collected later
        # destroys its combos' popup views under a later test's app-wide filter)
        win.close()
        win.deleteLater()
        _pump(app, 30)


def test_the_low_row_controls_never_move_with_the_policy_texts():
    """2026-11-02 (review): the policy name and state labels prefer their TEXT's width, and they used to sit BEFORE
    the `…` and the cap. A refused pick or cap change shows a long face for the worker's ~6 s (RL_PICK_NOTE_S) and
    sends the operator to exactly those two controls: the face pushed them right, and when it expired they jumped
    back and the spot under the operator's pointer became START (in a Diffusion Policy session START needs no hold:
    one click engages); another export name moved them the same way. The controls now come first and the labels
    last. At the operator's 1920x1080 station, the 1600x1000 the station tests use and the station's minimum size,
    `…`, the cap label and the cap combo keep ONE geometry for every face and export name on both pwm LOWs (`…` on
    RL_7D too), START never covers them, and once a refusal expires the spots where they were are still theirs —
    only START's left edge moves, beside the labels. The fix is the ORDER, not fixed label widths: at 1920 the
    operator still reads a refusal's whole 48-char face and a library export's whole name (its timestamp is what
    tells two imports apart), both of which a fixed width elides."""
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    station, library = "pwm11att9X_s0", "pwm11att13Q_s1@20261020-101010"    # a station copy; a library import
    picked = "/x/logs/rsl_rl/2026-10-30_19-40-11_pwm11att14Rcfd_s1/exported"

    def faces(low: str, name: str) -> dict:
        s = lambda st, note="", **kw: _slot(name, st, note, cap=0.25, cap_trained=0.5, **kw)     # noqa: E731
        lab = "RL_PWM_7D" if low == "rl_pwm_7d" else "RL_PWM"
        return {
            "ok": (_cap(), s("ok")),
            "warn": (_cap(), s("warn", CAP_NOTE)),
            "pick refused": (_cap(), s("refused", f"RL policy refused: {lab}: " + "y" * 60)),
            "cap refused": (_cap(state="refused", note=BUILD_REFUSAL), s("warn", CAP_NOTE)),
            "error": (_cap(), _slot(state="error", note="ValueError: " + "e" * 80)),
            "no-fly": (_cap(), s("warn", "no-fly — x", no_fly="x")),
        }

    def rect(wd) -> tuple:
        g = wd.geometry()
        return (g.x(), g.y(), g.width(), g.height())

    def whole(lab) -> bool:
        """The label PAINTS its whole text (ElidingLabel elides in paint only)."""
        fm = QtGui.QFontMetrics(lab.font())
        return fm.elidedText(lab.text(), Qt.TextElideMode.ElideRight, lab.contentsRect().width()) == lab.text()

    win = MainWindow(MpcOpts())
    try:
        p = win._traj_panel
        win.resize(1600, 1000)
        win.show()
        _pump(app, 250)
        p.set_mode_default("dobmpc")
        _pump(app, 50)
        mn = win.minimumSizeHint()
        for w, h in ((1920, 1080), (1600, 1000), (mn.width(), mn.height())):
            win.resize(w, h)
            _pump(app, 120)
            for low in PWM_LOWS:
                p.set_mode_default(low)
                seen = {}

                def measure(key) -> None:
                    assert all(x.isVisible() for x in p.rlpol_widgets + p.rlcap_widgets), (w, low, key)
                    ctl = (p.btn_rlpol, p.lbl_rlcap, p.rlcap_box)
                    for wd in ctl:
                        assert not wd.geometry().intersects(p.btn_start.geometry()), (w, low, key, wd)
                    seen[key] = tuple(rect(wd) for wd in ctl)

                for name in (station, library):
                    for face, (cs, slot) in faces(low, name).items():
                        p.add_status(_st(low, cap=cs, policy=_policy(**{low: slot})))
                        _pump(app, 40)
                        # the face IS what is measured: long ones long, the name in text() and in the tooltip
                        st = p.rlpol_state.text()
                        assert (st.startswith("cap refused: ") if face == "cap refused"
                                else st.startswith("RL policy refused: ") if face == "pick refused"
                                else st.startswith("ERROR: ") if face == "error"
                                else st == {"ok": "OK", "warn": "WARN", "no-fly": "NO-FLY"}[face]), (face, st)
                        if face != "error":                             # an error slot holds no export: "—"
                            assert p.rlpol_name.text() == name and name in p.rlpol_name.toolTip(), (face, name)
                        measure((name, face))
                        if (w, h) == (1920, 1080):
                            # ...and the operator reads it there: the refusal beside the cap, the import's stamp
                            if (name, face) == (station, "cap refused"):
                                assert len(st) == 48 and whole(p.rlpol_state), (low, st)
                            if name == library and face in ("ok", "warn"):
                                assert whole(p.rlpol_name), (low, face)
                # ...and a pick in flight ("→ name")
                p.add_status(_st(low, policy=_policy(**{low: _slot(station, "ok")})))
                p._apply_rl_policy_choice(low, picked)
                _pump(app, 40)
                assert p.rlpol_name.text() == "→ pwm11att14Rcfd_s1"
                measure(("request", "ok"))
                p._rlpol_req = None
                assert len(set(seen.values())) == 1, (w, low, seen)
                assert (p.mode_box.x() < p.lbl_rlpol.x() < p.btn_rlpol.x() < p.lbl_rlcap.x() < p.rlcap_box.x()
                        < p.rlpol_name.x() < p.rlpol_state.x() < p.btn_start.x()), (w, low)

                # the refusal expires (the worker's next status): what was under the operator's pointer still is
                for face in ("cap refused", "pick refused"):
                    cs, slot = faces(low, library)[face]
                    p.add_status(_st(low, cap=cs, policy=_policy(**{low: slot})))
                    _pump(app, 40)
                    at = {wd: wd.geometry().center() for wd in (p.btn_rlpol, p.rlcap_box)}
                    p.add_status(_st(low, policy=_policy(**{low: faces(low, station)["warn"][1]})))
                    _pump(app, 40)
                    assert p.rlpol_state.text() == "WARN", p.rlpol_state.text()
                    for wd, c in at.items():
                        assert p.childAt(c) is wd, (w, low, face, wd, p.childAt(c))

            # RL_7D (no cap): its `…` the same way — the 2026-11-01 pick refusal moved it too
            p.set_mode_default("rl_7d")
            seen7 = set()
            for name in (station, library):
                for st, note in (("ok", ""), ("refused", "RL policy refused: RL_7D: " + "y" * 60),
                                 ("error", "ValueError: " + "e" * 80)):
                    d = _slot(name if st != "error" else "", st, note)
                    p.add_status(MpcStatus(mode="rl_7d", rl_policy=_policy(rl_7d=d)))
                    _pump(app, 40)
                    assert p.btn_rlpol.isVisible() and not p.btn_rlpol.geometry().intersects(p.btn_start.geometry())
                    seen7.add(rect(p.btn_rlpol))
            assert len(seen7) == 1, (w, seen7)
        assert p.rlpol_name.minimumSizeHint().width() == 0 and p.rlpol_state.minimumSizeHint().width() == 0
    finally:
        win.shutdown()
        # deleted HERE (test_offline test_status_text_never_moves_the_station_grid: a window collected later
        # destroys its combos' popup views under a later test's app-wide filter)
        win.close()
        win.deleteLater()
        _pump(app, 30)


# ----------------------------------------------------------------------------- the contract with the worker
def test_the_panel_reads_the_real_workers_cap_and_its_request_is_what_the_worker_takes(tmp_path):
    """Both halves together: the panel is fed MpcWorker._rl_pwm_cap_status() / _rl_policy_status() exactly as
    _publish puts them in MpcStatus, and the (low, cap) it emits goes to MpcWorker.set_rl_pwm_cap as
    bus.cmd_rl_pwm_cap delivers it (called directly here: one thread, no event loop). The worker harness is
    test_rl_policy_pick's (_mk: the test's own hw_mpc.yaml, the pwm transport at cap 0.5, no thruster sink — so no
    ceiling and no sink window); RL_PWM_7D is given the git-tracked pwm11att9X_s0 (trained 0.5)."""
    from rov_gui.widgets.trajectory import TrajectoryWindow

    pick = importlib.import_module("rov_gui.tests.test_rl_policy_pick")
    if not (pick.STATION / pick.ATT).is_dir():
        pytest.skip(f"station export {pick.ATT} not in this checkout")
    app = _app()
    h = pick._mk(tmp_path)
    w = h.w
    p = None
    try:
        w.set_rl_policy("rl_pwm_7d", str(pick.STATION / pick.ATT))
        assert w._rl_policy_status()["rl_pwm_7d"]["name"] == pick.ATT, w.reason
        # RL_PWM_7D refuses caps below trained HERE, explicitly — the refusal below must not lean on the test yaml
        w.cfg.rl_pwm_7d = dict(w.cfg.rl_pwm_7d or {}, allow_caps_below_trained=False)
        p = TrajectoryWindow()
        p.show()
        _pump(app, 30)
        sent = []
        p.rl_pwm_cap_requested.connect(lambda low, cap: sent.append((low, cap)))

        def publish():
            cs = w._rl_pwm_cap_status()
            p.add_status(MpcStatus(mode="rl_pwm_7d", rl_pwm_cap=cs, rl_policy=w._rl_policy_status()))
            return cs

        p.set_mode_default("rl_pwm_7d")
        cs = publish()
        assert cs["cap"] == 0.5 and cs["source"] == "hw_mpc.yaml" and cs["ceiling"] is None
        assert p.rlcap_box.isVisible() and p.rlcap_box.currentText() == "0.50"
        assert _items(p.rlcap_box) == [(f"{v:.2f}", v) for v in cs["choices"]]
        assert _live(p) == "now 0.50 (hw_mpc.yaml) · sink window — · this LOW flies 0.5 (trained 0.5)"

        # raise to 0.6: the worker rebuilds and commits (RL_PWM_7D flies min(0.6, 0.5)); the panel's request ends
        _pick(p, 0.6)
        assert sent == [("rl_pwm_7d", 0.6)]
        w.set_rl_pwm_cap(*sent[-1])
        cs = publish()
        assert cs["cap"] == 0.6 and cs["source"] == "panel" and cs["state"] == "ok", (cs, w.reason)
        assert p._rlcap_req is None and p.rlcap_box.currentText() == "0.60"
        assert _live(p) == f"now 0.60 (panel, {cs['at']}) · sink window — · this LOW flies 0.5 (trained 0.5)"

        # lower to 0.25: RL_PWM_7D's build refuses a cap below its trained 0.5 — the worker says why, the panel
        # snaps back to the cap in force and puts the worker's note on the policy row in FAIL red
        _pick(p, 0.25)
        assert sent[-1] == ("rl_pwm_7d", 0.25)
        w.set_rl_pwm_cap(*sent[-1])
        cs = publish()
        assert cs["state"] == "refused" and cs["cap"] == 0.6, cs
        assert "RL_PWM_7D" in cs["note"] and "0.25" in cs["note"], cs["note"]
        assert p._rlcap_req is None and p.rlcap_box.currentText() == "0.60"
        assert p.rlpol_state.text() == _cut(f"cap refused: {cs['note']}")
        assert cs["note"] in p.rlpol_state.toolTip() and theme.FAIL in p.rlpol_state.styleSheet()
    finally:
        if p is not None:
            p.close()
            p.deleteLater()
            _pump(app, 30)
        w.teardown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
