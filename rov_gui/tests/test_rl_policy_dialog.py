#!/usr/bin/env python3
"""test_rl_policy_dialog.py — the RL POLICY picker on the trajectory panel's LOW row (2026-11-01, DESIGN §4): the
panel's policy widgets (widgets/trajectory.py), the table dialog (widgets/rl_policy_dialog.py) and the window that
owns it (window.py _pick_rl_policy). The worker side (MpcWorker.set_rl_policy) is NOT needed here: the panel is
driven with MpcStatus(rl_policy={...}) exactly as the worker publishes it (state.py MpcStatus.rl_policy).

What is pinned, in the order a failure would hurt:
  * Esc ANYWHERE in the dialog — sent to the dialog, the table after a click, the filter box, the slot combo with
    its list OPEN, with the app filter gone (the keyPressEvent fallback) — is ONE E-STOP + reject; an open popup
    does not outlive it; no application-level event filter survives it; Esc elsewhere afterwards is not its
    business. The window hands the keyboard back (teleop.all_stop) BEFORE the dialog runs, says so in the log,
    and wires the dialog's E-STOP to its own;
  * a refused / no-fly / smoke-length / unusable export is hidden by default and, when shown, greyed (red for a ban
    or an unreadable export) and neither selectable nor acceptable (button, Enter, double-click) — by the dialog's
    refusal() = rl_library.refusal_reason, the ONE rule (review F10, 2026-11-02); the slot the pick is for travels
    WITH it (the dialog's slot at accept, not the panel's LOW);
  * a BANNED held export (a launch config can still name one) reads NO-FLY in FAIL red on the row and on the
    dialog's held line — never the amber cap WARN — with the ban first in the tooltip (review F1, 2026-11-02);
  * the row shows the WORKER's truth only: "→ name" until accepted / refused / 1.5 s, OK / WARN / NO-FLY / ERROR /
    the refusal, frozen while engaged or with a controller CSV open, visible for the four RL LOWs only;
  * a pick accepted while REC NAV / REC UI records re-writes that recording's controller.json, after the pick
    (review F3, 2026-11-02);
  * the pwm slots judged at the transport cap IN FORCE (2026-11-02, SPEC C8: the window hands in the panel's
    MpcStatus.rl_pwm_cap): (a) trained above it on a LOW that refuses caps below trained, (b) a window other than
    the thruster sink's — greyed with the reason naming the LOW row's cap, never pickable; the count line ends
    " · cap 0.25" for RL_PWM / RL_PWM_7D; no cap status = exactly the dialog of before (the LOW row's cap combo
    itself: test_rl_pwm_cap_panel.py);
  * the row adds no height to the panel and never drives its (or the station's) minimum width;
  * LOW_LABELS = the panel's MODE_LABELS for the RL keys; the DP forecast's "<sibling> tracks" hint follows the
    worker's per-slot state once it reports (the launch seed until then).

The fixtures read git-TRACKED station exports only (FIXTURE_EXPORTS; review F8, 2026-11-02: they read the operator's
untracked pwm11att14Rw_s1 until then, so a checkout without that folder failed most of this file).

    cd <repo> && QT_QPA_PLATFORM=offscreen ~/miniforge3/envs/robust/bin/python -m pytest -q -p no:cacheprovider \\
        rov_gui/tests/test_rl_policy_dialog.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rov_gui import theme                                         # noqa: E402
from rov_gui.control import rl_library as L                       # noqa: E402
from rov_gui.qt import QT_API, QtCore, QtGui, QtWidgets, Qt       # noqa: E402
from rov_gui.state import MpcStatus, PolicyStatus                 # noqa: E402
from rov_gui.tests.test_offline import Opts, _app, _pump          # noqa: E402

QTest = importlib.import_module(f"{QT_API}.QtTest").QTest
STATION = ROOT / "rov_gui" / "control" / "rl_policies"
#: The ONLY station exports the fixtures below read — each one git-TRACKED (pinned by
#: test_the_fixtures_read_only_git_tracked_station_exports; _copy and _station_entries refuse any other name). Review
#: F8 (2026-11-02): the fixtures read the operator's UNTRACKED pwm11att14Rw_s1 until then. pwm11att9X_s0 stands in for
#: it: attitude-trained pwm, fits RL_PWM and RL_PWM_7D; att7pwm_s0 is the banned 7D export (flight ban, cap 1.0).
FIXTURE_EXPORTS = ("pwm11att9X_s0", "pwm16N_s3", "pwm11k0.2_s0", "traj12_s2", "pwm10_s1", "att7pwm_s0")


# ----------------------------------------------------------------------------- helpers
def _copy(name: str, d: Path) -> Path:
    assert name in FIXTURE_EXPORTS, f"{name}: fixtures read FIXTURE_EXPORTS only (git-tracked; review F8)"
    d.mkdir(parents=True, exist_ok=True)
    for f in L.REQUIRED_FILES:
        shutil.copyfile(STATION / name / f, d / f)
    return d


def _station_entries() -> dict:
    """{name: Entry} of the REAL station folder (read, never written), FIXTURE_EXPORTS only — so a test reaching for
    another export fails with a KeyError on EVERY machine, not only on a checkout that lacks its folder (review F8)."""
    return {e.name: e for e in L.scan([("station", STATION)]) if e.name in FIXTURE_EXPORTS}


def _library(tmp_path: Path) -> list:
    """A two-root library from real station exports (read, never written): two attitude-trained pwm exports
    (RL_PWM + RL_PWM_7D; pwm11att9X_s0 in the station root, trained 2026-10-15, OLDER than pwm16N_s3's 2026-10-19 —
    their checkpoints' run stamps), a level-trained pwm one (RL_PWM only), an axes one (RL only), the BANNED pwm10_s1
    weights under an innocent name (no-fly by sha1) and an unreadable export."""
    st, lib = tmp_path / "station", tmp_path / "library"
    _copy("pwm11att9X_s0", st / "pwm11att9X_s0")
    _copy("pwm16N_s3", lib / "pwm16N_s3")
    _copy("pwm11k0.2_s0", lib / "pwm11k0.2_s0")
    _copy("traj12_s2", lib / "traj12_s2")
    _copy("pwm10_s1", lib / "innocent_s0")
    (lib / "broken").mkdir()
    (lib / "broken" / "obs_spec.json").write_text("{", encoding="utf-8")
    shutil.copyfile(STATION / "traj12_s2" / "policy_weights.npz", lib / "broken" / "policy_weights.npz")
    return L.dedupe(L.scan([("station", st), ("library", lib)]))


def _smoke_copy(d: Path) -> Path:
    """A hand-copied SMOKE export: pwm11att9X_s0's files, its obs_spec claiming a 2-iteration run of a newer
    'smoke_pwm11att' training run (the shape of RL_controller's smoke_* exports — complete obs_spec, so slot_fit
    and the follower build pass it; rl_library.MIN_ITERATION is what refuses it). Edited as TEXT: two exact
    substrings, each asserted present once, so a changed station export fails here instead of testing nothing."""
    src = STATION / "pwm11att9X_s0"
    assert "pwm11att9X_s0" in FIXTURE_EXPORTS
    txt = (src / "obs_spec.json").read_text(encoding="utf-8")
    it_old, ck_old = '"rsl_rl_iteration": 1799', "2026-10-15_10-33-22_pwm11att9X_s0/model_1799.pt"
    assert txt.count(it_old) == 1 and txt.count(ck_old) == 1, "pwm11att9X_s0's obs_spec changed — re-derive"
    d.mkdir(parents=True, exist_ok=True)
    (d / "obs_spec.json").write_text(txt.replace(it_old, '"rsl_rl_iteration": 2')
                                     .replace(ck_old, "2026-11-01_09-00-00_smoke_pwm11att/model_2.pt"),
                                     encoding="utf-8")
    shutil.copyfile(src / "policy_weights.npz", d / "policy_weights.npz")
    return d


def _cap_copy(d: Path, trained: str = "0.25") -> Path:
    """pwm16N_s3's files with its obs_spec claiming another TRAINED cap (output.pwm_model.pwm_cap_trained, what
    Entry.cap_trained reads) — an export trained at the station's first-water cap, for the cap rule (b) (2026-11-02).
    Edited as TEXT: one exact substring, asserted present once, so a changed station export fails here instead of
    testing nothing. Its weights are pwm16N_s3's (not banned), so only the cap tells the two apart."""
    src = STATION / "pwm16N_s3"
    assert "pwm16N_s3" in FIXTURE_EXPORTS
    txt = (src / "obs_spec.json").read_text(encoding="utf-8")
    old = '"pwm_cap_trained": 0.5'
    assert txt.count(old) == 1, "pwm16N_s3's obs_spec changed — re-derive"
    d.mkdir(parents=True, exist_ok=True)
    (d / "obs_spec.json").write_text(txt.replace(old, f'"pwm_cap_trained": {trained}'), encoding="utf-8")
    shutil.copyfile(src / "policy_weights.npz", d / "policy_weights.npz")
    return d


def _cap_status(**kw) -> dict:
    """MpcStatus.rl_pwm_cap as the worker publishes it (state.py; SPEC C2) — the SHIPPED station by default: the
    transport at 0.25 from hw_mpc.yaml, a fresh sink snapshot whose export (pwm11k0.2_s0) is trained at 0.5, RL_PWM
    refusing caps below trained and RL_PWM_7D allowing them (config/hw_mpc.yaml allow_caps_below_trained)."""
    d = {"cap": 0.25, "source": "hw_mpc.yaml", "at": None,
         "choices": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], "ceiling": 0.5, "sink_cap": 0.25,
         "allow_below": {"rl_pwm": False, "rl_pwm_7d": True}, "state": "ok", "note": ""}
    d.update(kw)
    return d


#: SPEC C8 (a): a 0.5-trained export for RL_PWM (it refuses caps below trained) at the shipped transport cap 0.25
CAP_A_RL_PWM = ("trained at cap 0.5 > the transport cap 0.25 and RL_PWM refuses caps below trained — set the LOW "
                "row's cap to 0.5")
#: SPEC C8 (b): a 0.25-trained export at transport cap 0.5 while the sink's export is trained at 0.5
CAP_B = "flies cap 0.25 but the thruster sink's window is 0.5 — set the LOW row's cap to 0.25"


def _held(e, state: str = "ok", note: str = "", source: str = "hw_mpc.yaml", no_fly: str = "") -> dict:
    """One MpcStatus.rl_policy slot holding Entry ``e`` — every key the worker publishes, "no_fly" included ("" or
    the held export's ban, in every state: workers.py _rl_policy_status, 2026-11-02)."""
    return {"name": e.name, "path": e.path, "rel": e.rel, "id": e.id, "source": source, "state": state,
            "note": note, "no_fly": no_fly}


def _status_from_station() -> dict:
    """A launch as the worker reports it (rl_7d: policy_dir null -> no follower). RL_PWM_7D holds pwm11att9X_s0 —
    its launch default 2026-10-15 .. 10-17 per the config/hw_mpc.yaml rl_pwm_7d.policy_dir comment, and git-tracked;
    today's default there, pwm11att14Rw_s1, is not (review F8)."""
    by = _station_entries()
    return {"rl": _held(by["traj12_s2"]),
            "rl_pwm": _held(by["pwm11k0.2_s0"], "warn", "flies at cap 0.25 below trained 0.5 "
                                                       "(allow_caps_below_trained)"),
            "rl_pwm_7d": _held(by["pwm11att9X_s0"]),
            "rl_7d": {"name": "", "path": "", "rel": "", "id": "", "source": "", "state": "error",
                      "note": "ValueError: rl_7d: no policy_dir", "no_fly": ""}}


def _names(dlg) -> list:
    return [dlg.table.item(r, 0).text() for r in range(dlg.table.rowCount())]


def _row_of(dlg, name: str) -> int:
    return next(r for r, row in enumerate(dlg._rows) if row is not None and row[0].name == name)


def _esc() -> QtGui.QKeyEvent:
    return QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, int(Qt.Key.Key_Escape), Qt.KeyboardModifier.NoModifier)


def _double_click(tv, row: int, col: int = 1) -> None:
    """A REAL double-click: press + release, then the double-click. QTest.mouseDClick on a widget sends only the
    DblClick event, which QAbstractItemView treats as a plain press when no press on that index came first
    (mouseDoubleClickEvent's pressedIndex check) — so it alone would never reach cellDoubleClicked."""
    pos = tv.visualRect(tv.model().index(row, col)).center()
    QTest.mouseClick(tv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
    QTest.mouseDClick(tv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


# ----------------------------------------------------------------------------- shared names
def test_low_labels_are_the_panel_mode_labels():
    """The dialog's slot combo and the worker's texts say LOW_LABELS; the panel's LOW combo says MODE_LABELS —
    one name per LOW, in the panel's order."""
    from rov_gui.widgets.trajectory import MODE_LABELS

    assert {k: MODE_LABELS[k] for k in L.RL_LOWS} == L.LOW_LABELS
    assert set(L.RL_LOWS) <= set(MODE_LABELS)


def test_the_fixtures_read_only_git_tracked_station_exports():
    """Review F8 (2026-11-02): every export the fixtures read is in git, so a clean checkout runs this file. Both
    files of each FIXTURE_EXPORTS folder must be tracked (git ls-files, read-only); outside a git checkout there is
    nothing to compare against — skipped."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout — nothing to compare the fixtures against")
    try:
        out = subprocess.run([git, "-C", str(ROOT), "ls-files", "--", "rov_gui/control/rl_policies"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"git ls-files did not run: {e}")
    if out.returncode != 0:
        pytest.skip(f"git ls-files failed: {out.stderr.strip()[:200]}")
    tracked = set(out.stdout.split())
    missing = [f"{n}/{f}" for n in FIXTURE_EXPORTS for f in L.REQUIRED_FILES
               if f"rov_gui/control/rl_policies/{n}/{f}" not in tracked]
    assert not missing, f"fixture exports not in git (a clean checkout would lack them): {missing}"


# ----------------------------------------------------------------------------- the panel's LOW row
def test_the_policy_row_shows_for_the_rl_lows_only_and_reflects_the_worker():
    from rov_gui.widgets.trajectory import MODE_LABELS, TrajectoryWindow

    _app()
    w = TrajectoryWindow()
    w.show()
    try:
        for low in MODE_LABELS:
            w.set_mode_default(low)
            assert all(x.isVisible() == (low in L.RL_LOWS) for x in w.rlpol_widgets), low
            # the tune widgets share the slot after the LOW combo: never both
            assert not (any(x.isVisible() for x in w.rlpol_widgets)
                        and any(x.isVisible() for x in w.tune_widgets)), low
        # a pick from the LOW combo shows the row too (the operator's path, not only the status')
        w.set_mode_default("none")
        w.mode_box.setCurrentIndex(w.mode_box.findData("rl_pwm"))
        assert all(x.isVisible() for x in w.rlpol_widgets)

        w.set_mode_default("rl_pwm_7d")
        assert w.rl_policy_status() == {}
        w.add_status(MpcStatus(mode="rl_pwm_7d"))               # a worker that reports nothing (yet)
        assert w.rlpol_name.text() == "—" and w.rlpol_state.text() == ""
        assert "not reported" in w.rlpol_name.toolTip()

        held = _status_from_station()
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))
        assert w.rlpol_name.text() == "pwm11att9X_s0"
        assert w.rlpol_state.text() == "OK" and theme.OK in w.rlpol_state.styleSheet()
        tip = w.rlpol_name.toolTip()                            # set AFTER setText (ElidingLabel writes its own)
        assert held["rl_pwm_7d"]["rel"] in tip and f"id {held['rl_pwm_7d']['id']}" in tip
        assert "source hw_mpc.yaml" in tip
        # rl_policy_status is the worker's dict, as a copy the picker cannot edit through
        st = w.rl_policy_status()
        assert st == held
        st["rl_pwm_7d"]["name"] = "x"
        assert w.rl_policy_status()["rl_pwm_7d"]["name"] == "pwm11att9X_s0"

        # the row is the CURRENT LOW's slot — re-rendered on the LOW switch, not one status later
        w.set_mode_default("rl_pwm")
        assert w.rlpol_name.text() == "pwm11k0.2_s0" and w.rlpol_state.text() == "WARN"
        assert theme.WARN in w.rlpol_state.styleSheet()
        assert "allow_caps_below_trained" in w.rlpol_state.toolTip()
        w.set_mode_default("rl_7d")
        assert w.rlpol_name.text() == "—"
        assert w.rlpol_state.text() == "ERROR: ValueError: rl_7d: no policy_dir"
        assert theme.FAIL in w.rlpol_state.styleSheet()
        w.set_mode_default("rl")
        assert w.rlpol_name.text() == "traj12_s2" and w.rlpol_state.text() == "OK"

        # a pick shows as a REQUEST until the worker answers
        w.set_mode_default("rl_pwm_7d")
        sent: list = []
        w.rl_policy_requested.connect(lambda low, p: sent.append((low, p)))
        by = _station_entries()
        new = by["pwm16N_s3"].path                      # what the dialog hands over: absolute, as scanned
        w._apply_rl_policy_choice("rl_pwm_7d", new)
        assert sent == [("rl_pwm_7d", new)]
        assert w.rlpol_name.text() == "→ pwm16N_s3" and new in w.rlpol_name.toolTip()
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))         # not answered yet
        assert w.rlpol_name.text() == "→ pwm16N_s3"
        accepted = dict(held, rl_pwm_7d=_held(by["pwm16N_s3"], source="panel"))
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=accepted))
        assert w.rlpol_name.text() == "pwm16N_s3" and w.rlpol_state.text() == "OK"
        assert "source panel" in w.rlpol_name.toolTip()
        # a raw RL_controller export is named by its run tag, never "exported" (a synthetic path: nothing is read
        # from it, so it needs no folder — review F8)
        raw = "/x/logs/rsl_rl/rov_posehold/2026-10-30_19-40-11_pwm11att14Rw_s1/exported"
        w._apply_rl_policy_choice("rl_pwm_7d", raw)
        assert w.rlpol_name.text() == "→ pwm11att14Rw_s1"

        # refused: the refusal in FAIL, the HELD name unchanged
        refused = dict(accepted, rl_pwm_7d=dict(accepted["rl_pwm_7d"], state="refused",
                                                note="RL policy refused: RL_PWM_7D: not while engaged"))
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=refused))
        assert w.rlpol_name.text() == "pwm16N_s3"
        assert w.rlpol_state.text() == "RL policy refused: RL_PWM_7D: not while engaged"
        assert theme.FAIL in w.rlpol_state.styleSheet()
        # a second pick while that SAME refusal still shows (the worker's ~6 s): "→" stands until a new answer
        # or 1.5 s — the old refusal is not this pick's answer
        w._apply_rl_policy_choice("rl_pwm_7d", raw)
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=refused))
        assert w.rlpol_name.text() == "→ pwm11att14Rw_s1"
        low, path, t, was = w._rlpol_req
        w._rlpol_req = (low, path, t - 2.0, was)                  # 1.5 s of silence
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=refused))
        assert w.rlpol_name.text() == "pwm16N_s3" and w._rlpol_req is None
        # ...while a refusal with ANOTHER note answers at once
        w._apply_rl_policy_choice("rl_pwm_7d", raw)
        other = dict(refused, rl_pwm_7d=dict(refused["rl_pwm_7d"], note="RL policy refused: RL_PWM_7D: no-fly"))
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=other))
        assert w.rlpol_name.text() == "pwm16N_s3" and w.rlpol_state.text().endswith("no-fly")
        # a long refusal is cut to 48 chars on the row, whole in the tooltip
        long_note = "RL policy refused: RL_PWM_7D: " + "y" * 80
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=dict(
            refused, rl_pwm_7d=dict(refused["rl_pwm_7d"], note=long_note))))
        assert len(w.rlpol_state.text()) == 48 and w.rlpol_state.text().endswith("…")
        assert long_note in w.rlpol_state.toolTip()

        # a pick for ANOTHER slot (the dialog's slot combo) is sent with its slot and leaves this row alone
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=accepted))
        w._apply_rl_policy_choice("rl_pwm", new)
        assert sent[-1] == ("rl_pwm", new) and w.rlpol_name.text() == "pwm16N_s3"
        # ...and shows as the request when that LOW is the one shown
        w.set_mode_default("rl_pwm")
        assert w.rlpol_name.text() == "→ pwm16N_s3"
        w.set_mode_default("rl_pwm_7d")
        # an empty pick (cancelled) sends nothing
        n = len(sent)
        w._apply_rl_policy_choice("rl_pwm_7d", "")
        w._apply_rl_policy_choice("", new)
        assert len(sent) == n

        # frozen while engaged / CSV open (the worker refuses then too)
        w.add_status(MpcStatus(mode="rl_pwm_7d", engaged=True, rl_policy=accepted))
        assert not w.btn_rlpol.isEnabled()
        w.add_status(MpcStatus(mode="rl_pwm_7d", csv_open=True, rl_policy=accepted))
        assert not w.btn_rlpol.isEnabled()
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=accepted))
        assert w.btn_rlpol.isEnabled()
        # `…` asks the WINDOW for the picker; the button never takes the keyboard
        asked: list = []
        w.rl_policy_pick_requested.connect(lambda: asked.append(1))
        w.btn_rlpol.click()
        assert asked == [1]
        assert w.btn_rlpol.focusPolicy() == Qt.FocusPolicy.NoFocus
        assert "Esc closes it AND is E-STOP" in w.btn_rlpol.toolTip()
        assert "rl-policies sync" in w.btn_rlpol.toolTip()
    finally:
        w.close()
        w.deleteLater()


def test_a_banned_held_export_reads_no_fly_on_the_row_never_the_cap_warn():
    """Review F1 (2026-11-02). A banned export can still be HELD — a pick of one is refused, but hw_mpc.yaml may name
    one at launch and it builds. The worker then reports state "warn", a note that LEADS with the ban ("no-fly —
    <reason>", + "; <cap note>" below its trained cap), and the ban itself in "no_fly", in every state. The row:
    NO-FLY in FAIL red — never the amber WARN of a cap below the trained one, which a banned station copy shows at
    the station's cap (att7pwm_s0 and pwm10_s1 are trained at cap 1.0, asserted below for att7pwm_s0) — with the ban
    leading the tooltip and the "below the export's trained cap" sentence only when the note carries the cap note.
    A refusal keeps its face for its few seconds; the ban rides in the tooltips meanwhile. The ordinary cap WARN is
    unchanged, byte for byte."""
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    by = _station_entries()
    ban = L.NO_FLY["att7pwm_s0"]
    assert by["att7pwm_s0"].no_fly == ban and by["att7pwm_s0"].cap_trained == 1.0     # the fixture is what it says
    cap_note = "flies at cap 0.25 below trained 1 (allow_caps_below_trained)"        # workers.py _rl_cap_note
    banned = _held(by["att7pwm_s0"], "warn", f"no-fly — {ban}; {cap_note}", no_fly=ban)
    held = dict(_status_from_station(), rl_pwm_7d=banned)
    w = TrajectoryWindow()
    w.show()
    try:
        w.set_mode_default("rl_pwm_7d")
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))
        assert w.rlpol_name.text() == "att7pwm_s0"
        assert w.rlpol_state.text() == "NO-FLY"
        css = w.rlpol_state.styleSheet()
        assert theme.FAIL in css and theme.WARN not in css, css
        tip = w.rlpol_state.toolTip()
        assert tip.startswith(f"NO-FLY: {ban}\n"), tip                        # the ban LEADS
        assert f"WARN: {cap_note}" in tip and "below the export's trained cap" in tip, tip
        assert tip.index(ban) < tip.index(cap_note) and tip.count(ban) == 1, tip
        assert banned["rel"] in tip and f"id {banned['id']}" in tip
        assert f"no-fly — {ban}" in w.rlpol_name.toolTip()                   # the worker's note, ban first

        # at its trained cap the note is the ban alone: no cap sentence, no WARN line
        at_cap = dict(banned, note=f"no-fly — {ban}")
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=dict(held, rl_pwm_7d=at_cap)))
        assert w.rlpol_state.text() == "NO-FLY" and theme.FAIL in w.rlpol_state.styleSheet()
        tip = w.rlpol_state.toolTip()
        assert tip.startswith(f"NO-FLY: {ban}\n") and "trained cap" not in tip and "WARN" not in tip, tip

        # a refused pick keeps its face (it answers the pick just made); the held export's ban rides in BOTH
        # tooltips, and the face is NO-FLY again once the worker's refusal note expires
        why = "not while a controller CSV is open (REC) — stop the recording first"
        refused = dict(banned, state="refused", note=why)
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=dict(held, rl_pwm_7d=refused)))
        assert w.rlpol_state.text().startswith("not while a controller CSV is open")
        assert theme.FAIL in w.rlpol_state.styleSheet()
        assert f"pick REFUSED: {why}" in w.rlpol_state.toolTip() and ban in w.rlpol_state.toolTip()
        assert f"NO-FLY: {ban}" in w.rlpol_name.toolTip() and w.rlpol_name.text() == "att7pwm_s0"
        w.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))
        assert w.rlpol_state.text() == "NO-FLY"
        # a pick in flight for the slot shows "→ name"; the face stays the HELD export's ban
        w._apply_rl_policy_choice("rl_pwm_7d", by["pwm11att9X_s0"].path)
        assert w.rlpol_name.text() == "→ pwm11att9X_s0" and w.rlpol_state.text() == "NO-FLY"
        w._rlpol_req = None

        # the ordinary cap WARN (not banned): unchanged — amber WARN and the cap sentence, no NO-FLY anywhere
        w.set_mode_default("rl_pwm")
        d = held["rl_pwm"]
        assert w.rlpol_state.text() == "WARN" and theme.WARN in w.rlpol_state.styleSheet()
        assert w.rlpol_state.toolTip() == (
            f"WARN: {d['note']}\n\nWARN = it flies, but below the export's trained cap (allow_caps_below_trained)"
            f" — the run meta records the export\n{d['rel']}\nid {d['id']} · source hw_mpc.yaml")
        assert "NO-FLY" not in w.rlpol_name.toolTip()
        # ...and a WARN whose note is NOT the cap note is not explained as one
        w.add_status(MpcStatus(mode="rl_pwm", rl_policy=dict(held, rl_pwm=dict(d, note="something else"))))
        assert w.rlpol_state.text() == "WARN" and "trained cap" not in w.rlpol_state.toolTip()
        # an unbanned export's OK face is untouched by the key ("" = not banned)
        w.set_mode_default("rl")
        assert w.rlpol_state.text() == "OK" and "NO-FLY" not in w.rlpol_state.toolTip()
    finally:
        w.close()
        w.deleteLater()


def test_the_dp_forecast_sibling_hint_follows_the_workers_slot_state():
    """The launch seed (window.py: the YAML names a policy_dir) says which RL *_7d sibling the forecast may name;
    once the worker reports a slot, its state replaces the seed for that slot (error = no follower; a held no-fly
    export is not named either)."""
    from rov_gui.widgets.trajectory import TrajectoryWindow

    _app()
    p = TrajectoryWindow()
    try:
        p.set_mode_default("rl_pwm")
        p.set_shape("policy")
        p.set_policy_status(PolicyStatus(ready=True, action_repr="pos_rpy_width"))
        p.set_policy_config_default("auto", True, False, unavailable=("rl_7d",))     # the launch seed
        assert "(RL_PWM is 4-DoF; RL_PWM_7D tracks)" in p.mission_lbl.text(), p.mission_lbl.text()
        p.add_status(MpcStatus(mode="rl_pwm"))                    # nothing reported: the seed stands
        assert p._pol_unavailable == ("rl_7d",)
        # the worker: RL_PWM_7D has no follower -> no hint
        p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_pwm_7d": {"state": "error", "note": "x"}}))
        assert p._pol_unavailable == ("rl_7d", "rl_pwm_7d")
        assert "(RL_PWM is 4-DoF)" in p.mission_lbl.text(), p.mission_lbl.text()
        # a refused pick on a slot that holds NOTHING is still nothing
        p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_pwm_7d": {"state": "refused", "path": "",
                                                                         "note": "x"}}))
        assert "(RL_PWM is 4-DoF)" in p.mission_lbl.text()
        # a pick gave it one -> the hint is back; a refused pick on a slot that HOLDS one keeps it
        for st in ("ok", "warn", "refused"):
            p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_pwm_7d": {"state": st, "path": "/x/a",
                                                                             "note": "n"}}))
            assert "(RL_PWM is 4-DoF; RL_PWM_7D tracks)" in p.mission_lbl.text(), st
            assert p._pol_unavailable == ("rl_7d",), st
        # a slot holding a flight-banned export is not named, whatever its state; a good pick brings the hint back
        for st in ("ok", "warn", "refused"):
            p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_pwm_7d": {"state": st, "path": "/x/att7pwm_s0",
                                                                             "note": "n", "no_fly": "flight ban"}}))
            assert "(RL_PWM is 4-DoF)" in p.mission_lbl.text(), st
            assert p._pol_unavailable == ("rl_7d", "rl_pwm_7d"), st
        p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_pwm_7d": {"state": "ok", "path": "/x/a", "no_fly": ""}}))
        assert "(RL_PWM is 4-DoF; RL_PWM_7D tracks)" in p.mission_lbl.text()
        # the worker's rl_7d truth replaces the seed's too
        p.add_status(MpcStatus(mode="rl_pwm", rl_policy={"rl_7d": {"state": "ok", "path": "/x/b"},
                                                          "rl_pwm_7d": {"state": "ok", "path": "/x/a"}}))
        assert p._pol_unavailable == ()
    finally:
        p.close()
        p.deleteLater()


def test_the_policy_row_adds_no_height_and_never_drives_the_width():
    """The RL policy widgets sit ON the LOW row (the window's height budget has no room for a row:
    test_control test_mpc_panel_lives_in_the_grid_and_fits_the_operator_screen), right after the LOW combo; with
    the theme applied (as the station runs) the panel is exactly as tall and no wider with them shown, whatever
    the status says; the station's minimum stays inside the operator's screen."""
    from rov_gui.widgets.trajectory import ElidingLabel, TrajectoryWindow
    from rov_gui.window import MainWindow

    app = _app()
    theme.apply(app)
    p = TrajectoryWindow()
    p.resize(800, 400)
    p.show()
    _pump(app, 50)
    p.set_mode_default("dobmpc")
    _pump(app, 30)
    h0, w0 = p.minimumSizeHint().height(), p.minimumSizeHint().width()
    held = _status_from_station()
    long_note = "RL policy refused: RL_PWM_7D: " + "z" * 120
    held["rl_pwm"] = dict(held["rl_pwm"], name="x" * 70, state="refused", note=long_note)
    for low in L.RL_LOWS:
        p.set_mode_default(low)
        p.add_status(MpcStatus(mode=low, rl_policy=held))
        _pump(app, 30)
        assert all(x.isVisible() for x in p.rlpol_widgets), low
        assert p.minimumSizeHint().height() == h0, f"{low}: the policy widgets must add no row"
        assert p.minimumSizeHint().width() <= w0, f"{low}: the policy widgets must not widen the panel"
        # on the LOW row, between the LOW combo and START — the `…` before the name / state labels, so their text
        # never moves it (2026-11-02 review; test_rl_pwm_cap_panel.py
        # test_the_low_row_controls_never_move_with_the_policy_texts)
        cy = p.mode_box.geometry().center().y()
        for x in p.rlpol_widgets:
            assert abs(x.geometry().center().y() - cy) <= 1, (low, x)
        assert (p.mode_box.x() < p.lbl_rlpol.x() < p.btn_rlpol.x() < p.rlpol_name.x() < p.rlpol_state.x()
                < p.btn_start.x()), low
    assert isinstance(p.rlpol_name, ElidingLabel) and isinstance(p.rlpol_state, ElidingLabel)
    assert p.rlpol_name.minimumSizeHint().width() == 0 and p.rlpol_state.minimumSizeHint().width() == 0
    # 11 px, in the state's colour through the polished palette (rl_pwm: the refusal)
    p.set_mode_default("rl_pwm")
    p.add_status(MpcStatus(mode="rl_pwm", rl_policy=held))
    _pump(app, 30)
    assert p.rlpol_name.font().pixelSize() == 11 and p.rlpol_state.font().pixelSize() == 11
    fg = p.rlpol_state.palette().color(p.rlpol_state.foregroundRole()).name()
    assert fg.lower() == theme.FAIL.lower(), (fg, theme.FAIL)
    p.close()
    p.deleteLater()
    _pump(app, 30)

    # the station: an RL LOW with a long refusal showing moves nothing and fits the screen
    class MpcOpts(Opts):
        mpc = True
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    win = MainWindow(MpcOpts())
    try:
        panel = win._traj_panel
        win.resize(1600, 1000)
        win.show()
        _pump(app, 250)
        panel.set_mode_default("dobmpc")
        _pump(app, 50)
        before = win.minimumSizeHint()
        panel.set_mode_default("rl_pwm")
        panel.add_status(MpcStatus(mode="rl_pwm", rl_policy=held))
        _pump(app, 60)
        hint = win.minimumSizeHint()
        assert (hint.width(), hint.height()) == (before.width(), before.height())
        assert hint.width() <= 1366 and hint.height() <= 1000, hint
    finally:
        win.shutdown()
        # deleted HERE (test_offline test_status_text_never_moves_the_station_grid: a window collected later
        # destroys its combos' popup views under a later test's app-wide filter)
        win.close()
        win.deleteLater()
        _pump(app, 30)


# ----------------------------------------------------------------------------- the dialog
def test_the_dialog_lists_the_slots_exports_and_marks_the_held_one(tmp_path):
    from rov_gui.widgets.rl_policy_dialog import COLUMNS, HELD_MARK, RlPolicyDialog, empty_text

    _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    assert by["innocent_s0"].no_fly and by["broken"].error            # the fixture is what it says
    held = {"rl_pwm_7d": _held(by["pwm11att9X_s0"])}
    d = RlPolicyDialog(None, ents, "rl_pwm_7d", held)
    try:
        assert d.current_slot() == "rl_pwm_7d" and d.btn_use.text() == "Use for RL_PWM_7D"
        assert [d.table.horizontalHeaderItem(c).text() for c in range(d.table.columnCount())] == list(COLUMNS)
        # only what fits the slot, newest first (pwm16N_s3 2026-10-19 before the held pwm11att9X_s0 2026-10-15:
        # their checkpoints' run stamps); the held one marked and bold
        assert _names(d) == ["pwm16N_s3", HELD_MARK + "pwm11att9X_s0"]
        assert d.table.item(1, 0).font().bold() and not d.table.item(0, 0).font().bold()
        assert "pwm11att9X_s0" in d.held_lbl.text() and "hw_mpc.yaml" in d.held_lbl.text()
        e, h = by["pwm16N_s3"], by["pwm11att9X_s0"]
        assert [d.table.item(0, c).text() for c in range(len(COLUMNS))] == [
            "pwm16N_s3", e.trained, e.dr_preset, "0.5", "2399", "DVL blend", "library", e.id]
        assert [d.table.item(1, c).text() for c in range(len(COLUMNS))] == [
            HELD_MARK + "pwm11att9X_s0", h.trained, "pwm11att7", "0.5", "1799", "tag fd", "station", h.id]
        tip = d.table.item(0, 3).toolTip()                               # every cell carries the row tooltip
        for part in (e.path, e.checkpoint, e.id, "fits RL_PWM_7D", "fits: RL_PWM / RL_PWM_7D"):
            assert part in tip, part
        assert not d.btn_use.isEnabled() and d.chosen() is None            # nothing selected yet
        assert "2 fit RL_PWM_7D" in d.count_lbl.text() and "4 refused hidden" in d.count_lbl.text()

        # per slot
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm"))
        assert d.btn_use.text() == "Use for RL_PWM"
        assert set(_names(d)) == {"pwm11att9X_s0", "pwm16N_s3", "pwm11k0.2_s0"}        # no held marker here
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl"))
        assert _names(d) == ["traj12_s2"]
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_7d"))
        assert d.table.rowCount() == 1 and _names(d)[0].startswith(empty_text("RL_7D"))
        assert int(d.table.item(0, 0).flags()) == 0, "the empty line is a sentence, not a row"
        d.table.selectRow(0)
        assert not d.btn_use.isEnabled()

        # show refused / no-fly: everything, the refused rows greyed (red for a ban / unusable) and unselectable
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm_7d"))
        d.chk_all.setChecked(True)
        assert {n.replace(HELD_MARK, "") for n in _names(d)} == set(by)
        for r, (ent, why, _h) in enumerate(d._rows):
            it = d.table.item(r, 0)
            selectable = bool(it.flags() & Qt.ItemFlag.ItemIsSelectable)
            assert selectable == (why is None), ent.name
            if why is not None:
                assert it.font().italic(), ent.name
                want = theme.FAIL if (ent.error or ent.no_fly) else theme.TEXT_DIM
                assert it.foreground().color().name().lower() == want.lower(), ent.name
                assert f"REFUSED for RL_PWM_7D: {why}" in it.toolTip(), ent.name
            else:
                # a pickable row carries no colour of its own: the sheet's selection colours must win
                assert it.data(Qt.ItemDataRole.ForegroundRole) is None, ent.name
        assert "level-trained" in d._rows[_row_of(d, "pwm11k0.2_s0")][1]
        assert "flight ban" in d._rows[_row_of(d, "innocent_s0")][1]
        assert "unusable export" in d._rows[_row_of(d, "broken")][1]
        for name in ("pwm11k0.2_s0", "innocent_s0", "broken", "traj12_s2"):
            d.table.selectRow(_row_of(d, name))
            assert d._selected_entry() is None and not d.btn_use.isEnabled(), name
            d.btn_use.click()
            assert d.chosen() is None, name
        # the SECOND guard: even a refused row that Qt WOULD select (its flags mis-set) is never accepted —
        # _selected_entry re-checks the row's refusal itself, and the button, Enter and a double-click all ask it
        r = _row_of(d, "innocent_s0")
        for c in range(d.table.columnCount()):
            d.table.item(r, c).setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        d.table.selectRow(r)
        assert d.table.selectionModel().selectedIndexes(), "the mis-flagged row should be selected by Qt"
        assert d._selected_entry() is None and not d.btn_use.isEnabled()
        d._use()
        d._double_clicked(r, 0)
        assert d.chosen() is None and int(d.result()) != int(QtWidgets.QDialog.DialogCode.Accepted)
        d.chk_all.setChecked(False)
        d.chk_all.setChecked(True)                                         # rebuilt: the flags are right again

        # the filter: a case-insensitive substring of name / preset / where
        d.chk_all.setChecked(False)
        d.filter_edit.setText("PWM16")
        assert _names(d) == ["pwm16N_s3"]
        d.filter_edit.setText("station")                                    # where
        assert _names(d) == [HELD_MARK + "pwm11att9X_s0"]
        d.filter_edit.setText("PWM11ATT7")                                  # the preset ONLY (not in the name)
        assert _names(d) == [HELD_MARK + "pwm11att9X_s0"]
        d.filter_edit.setText("nothing-like-this")
        assert d.table.rowCount() == 1 and "matches 'nothing-like-this'" in _names(d)[0]
        d.filter_edit.setText("")

        # accept: select a pickable row, Use -> (slot, absolute export dir)
        d.table.selectRow(_row_of(d, "pwm16N_s3"))
        assert d.btn_use.isEnabled()
        # the selection survives a filter change that keeps the row
        d.filter_edit.setText("pwm1")
        assert d._selected_entry() is not None and d._selected_entry().name == "pwm16N_s3"
        d.btn_use.click()
        assert d.chosen() == ("rl_pwm_7d", by["pwm16N_s3"].path)
        assert int(d.result()) == int(QtWidgets.QDialog.DialogCode.Accepted)
        assert os.path.isabs(d.chosen()[1])
    finally:
        d.deleteLater()


def test_the_held_line_names_a_banned_held_export_no_fly_in_red(tmp_path):
    """Review F1 (2026-11-02): the line above the table — what the worker HOLDS — says NO-FLY with the ban's reason,
    in FAIL red, BEFORE the cap fact (never the amber "WARN: flies at cap …" alone); the banned held row is listed
    although refused (it is what flies), marked, red and never pickable. A refusal shows beside the ban; the
    ordinary cap WARN line is unchanged."""
    from rov_gui.widgets.rl_policy_dialog import HELD_MARK, RlPolicyDialog

    _app()
    by = _station_entries()
    ents = _library(tmp_path) + [by["att7pwm_s0"]]
    ban = L.NO_FLY["att7pwm_s0"]
    cap_note = "flies at cap 0.25 below trained 1 (allow_caps_below_trained)"
    banned = _held(by["att7pwm_s0"], "warn", f"no-fly — {ban}; {cap_note}", no_fly=ban)
    plain = _held(by["pwm11k0.2_s0"], "warn", "flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)")
    head = f"RL_PWM_7D holds att7pwm_s0 (id {banned['id'][:8]}, from hw_mpc.yaml)"
    dialogs = []

    def _open(held_7d, slot="rl_pwm_7d"):
        dlg = RlPolicyDialog(None, ents, slot, {"rl_pwm_7d": held_7d, "rl_pwm": plain})
        dialogs.append(dlg)
        return dlg

    try:
        d = _open(banned)
        assert d.held_lbl.text() == f"{head} — NO-FLY: {ban} — WARN: {cap_note}", d.held_lbl.text()
        sheet = d.held_lbl.styleSheet()
        assert theme.FAIL in sheet and theme.WARN not in sheet, sheet
        r = _row_of(d, "att7pwm_s0")
        it = d.table.item(r, 0)
        assert it.text() == HELD_MARK + "att7pwm_s0" and it.font().bold() and it.font().italic()
        assert it.foreground().color().name().lower() == theme.FAIL.lower()
        assert not (it.flags() & Qt.ItemFlag.ItemIsSelectable)
        assert d._rows[r][1] == ban                                          # refusal_reason: the ban speaks
        assert ban in d.detail_lbl.text() and theme.FAIL in d.detail_lbl.styleSheet()     # the held row is current
        # at its trained cap: the ban alone
        assert _open(dict(banned, note=f"no-fly — {ban}")).held_lbl.text() == f"{head} — NO-FLY: {ban}"
        # a refusal while it is held: both, the ban first, red
        d3 = _open(dict(banned, state="refused", note="not while engaged"))
        assert d3.held_lbl.text() == f"{head} — NO-FLY: {ban} — the last pick was refused: not while engaged"
        assert theme.FAIL in d3.held_lbl.styleSheet()
        # the ordinary cap WARN line: unchanged, amber, no NO-FLY
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm"))
        assert d.held_lbl.text() == (f"RL_PWM holds pwm11k0.2_s0 (id {plain['id'][:8]}, from hw_mpc.yaml) — WARN: "
                                     f"flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)")
        assert theme.WARN in d.held_lbl.styleSheet()
    finally:
        for dlg in dialogs:
            dlg.deleteLater()


def test_refusal_is_the_librarys_one_rule_and_a_smoke_export_is_never_offered(tmp_path, monkeypatch):
    """Review F10 (2026-11-02): the dialog's refusal() IS rl_library.refusal_reason — the rule candidates() and
    `./c3 rl-policies list` apply — delegated, not a copy. Its iteration floor (MIN_ITERATION) is what keeps a
    hand-copied SMOKE export out of the picker (a complete obs_spec passes slot_fit and the follower build): not
    listed by default although it is the newest, counted as refused; with "show refused / no-fly" ticked it is
    greyed — a misfit's grey, not a ban's red — with the reason, and neither selectable nor acceptable."""
    from rov_gui.widgets import rl_policy_dialog as RPD

    app = _app()
    _library(tmp_path)
    _smoke_copy(tmp_path / "library" / "smoke_pwm11att")
    ents = L.dedupe(L.scan([("station", tmp_path / "station"), ("library", tmp_path / "library")]))
    by = {e.name: e for e in ents}
    smoke = by["smoke_pwm11att"]
    floor = f"smoke-length export (2 iterations < {L.MIN_ITERATION})"
    # the fixture is what it says: readable, not banned, FITS RL_PWM / RL_PWM_7D by slot_fit, the newest stamp
    assert smoke.iteration == 2 and not smoke.error and smoke.no_fly is None
    assert smoke.fit["rl_pwm_7d"] is None and smoke.fit["rl_pwm"] is None
    assert smoke.trained == max(e.trained for e in ents)
    # ONE rule: the same verdict as rl_library on every entry and slot — and delegated (a changed rule changes
    # the dialog: a copy would not follow)
    for e in ents:
        for low in L.RL_LOWS:
            assert RPD.refusal(e, low) == L.refusal_reason(e, low), (e.name, low)
    assert {low: RPD.refusal(smoke, low) for low in L.RL_LOWS} == {low: floor for low in L.RL_LOWS}
    with monkeypatch.context() as m:
        m.setattr(L, "refusal_reason", lambda entry, low: f"sentinel {entry.name} {low}")
        assert RPD.refusal(smoke, "rl_pwm") == "sentinel smoke_pwm11att rl_pwm"
    assert smoke not in L.candidates(ents, "rl_pwm_7d")

    d = RPD.RlPolicyDialog(None, ents, "rl_pwm_7d", {})
    d.show()
    _pump(app, 30)
    try:
        # never offered: not listed by default — it would head the list (the newest) — and counted as refused
        assert _names(d) == ["pwm16N_s3", "pwm11att9X_s0"]
        assert "2 fit RL_PWM_7D" in d.count_lbl.text() and "5 refused hidden" in d.count_lbl.text()
        # shown with the box ticked: greyed (TEXT_DIM, italic), the reason in the row tooltip and the detail line
        d.chk_all.setChecked(True)
        r = _row_of(d, "smoke_pwm11att")
        assert d._rows[r][1] == floor
        it = d.table.item(r, 0)
        assert it.font().italic() and it.foreground().color().name().lower() == theme.TEXT_DIM.lower()
        assert not (it.flags() & Qt.ItemFlag.ItemIsSelectable)
        assert f"REFUSED for RL_PWM_7D: {floor}" in it.toolTip() and "fits: no RL LOW" in it.toolTip()
        d.table.setCurrentCell(r, 0)
        assert floor in d.detail_lbl.text() and theme.TEXT_DIM in d.detail_lbl.styleSheet()
        # never accepted: Use, a real double-click, Enter
        d.table.selectRow(r)
        assert d._selected_entry() is None and not d.btn_use.isEnabled()
        d.btn_use.click()
        _double_click(d.table, r)
        QTest.keyClick(d.table, Qt.Key.Key_Return)
        _pump(app, 30)
        assert d.chosen() is None and d.isVisible()
        # the same on the other slot it would fit by slot_fit
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm"))
        assert d._rows[_row_of(d, "smoke_pwm11att")][1] == floor
    finally:
        d.reject()
        d.deleteLater()
        _pump(app, 30)


def test_the_dialog_judges_the_pwm_slots_at_the_cap_in_force(tmp_path):
    """SPEC C8 (2026-11-02, the LOW row's cap ▾): handed the panel's MpcStatus.rl_pwm_cap (cap_status=…), the
    dialog judges RL_PWM / RL_PWM_7D at the transport cap IN FORCE — refusal() passes it to
    rl_library.refusal_reason as cap_ctx. At the shipped station (cap 0.25, sink export trained 0.5) RL_PWM refuses
    caps below trained, so (a) every 0.5-trained pwm export is refused for it — hidden by default, greyed (a misfit's
    grey, not a ban's red) with the reason naming the remedy when shown, never selectable or acceptable — while
    RL_PWM_7D (it allows them) lists exactly what it lists without a cap. The count line says the cap it judged at
    (" · cap 0.25"), for the pwm slots only."""
    from rov_gui.widgets import rl_policy_dialog as RPD
    from rov_gui.widgets.rl_policy_dialog import HELD_MARK, RlPolicyDialog, empty_text

    _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    pwm05 = ("pwm11att9X_s0", "pwm16N_s3", "pwm11k0.2_s0")
    assert all(by[n].cap_trained == 0.5 for n in pwm05)                     # the fixture is what it says
    cs = _cap_status()
    cap_note = "flies at cap 0.25 below trained 0.5 (allow_caps_below_trained)"
    held = {"rl_pwm_7d": _held(by["pwm11att9X_s0"], "warn", cap_note),
            "rl_pwm": {"name": "", "path": "", "rel": "", "id": "", "source": "", "state": "error",
                       "note": "ValueError: rl_pwm: below the policy's trained cap", "no_fly": ""}}
    # ONE rule: the dialog's verdict IS the library's with the same cap_ctx, on every entry and slot
    for e in ents:
        for low in L.RL_LOWS:
            assert RPD.refusal(e, low, cs) == L.refusal_reason(e, low, cs), (e.name, low)
    d = RlPolicyDialog(None, ents, "rl_pwm", held, cap_status=cs)
    try:
        # (a): nothing fits RL_PWM at 0.25 — hidden by default, counted, the cap said
        assert d.table.rowCount() == 1 and _names(d)[0].startswith(empty_text("RL_PWM"))
        assert "6 refused hidden" in _names(d)[0]
        assert d.count_lbl.text() == "0 fit RL_PWM · 6 refused hidden · cap 0.25", d.count_lbl.text()
        d.chk_all.setChecked(True)
        assert d.count_lbl.text() == "0 fit RL_PWM · 6 refused shown · cap 0.25", d.count_lbl.text()
        for name in pwm05:
            r = _row_of(d, name)
            e, why, _h = d._rows[r]
            assert why == CAP_A_RL_PWM, (name, why)
            it = d.table.item(r, 0)
            assert it.font().italic() and it.foreground().color().name().lower() == theme.TEXT_DIM.lower(), name
            assert not (it.flags() & Qt.ItemFlag.ItemIsSelectable), name
            assert f"REFUSED for RL_PWM: {CAP_A_RL_PWM}" in it.toolTip(), name
            d.table.setCurrentCell(r, 0)
            assert CAP_A_RL_PWM in d.detail_lbl.text(), name
            # never accepted: Use, a double-click, the button with the row selected
            d.table.selectRow(r)
            assert d._selected_entry() is None and not d.btn_use.isEnabled(), name
            d.btn_use.click()
            d._double_clicked(r, 0)
            assert d.chosen() is None, name
        # RL_PWM_7D allows caps below trained: what it lists without a cap, the cap said
        d.chk_all.setChecked(False)
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm_7d"))
        assert _names(d) == ["pwm16N_s3", HELD_MARK + "pwm11att9X_s0"]
        assert d.count_lbl.text() == "2 fit RL_PWM_7D · 4 refused hidden · cap 0.25", d.count_lbl.text()
        # the row tooltip's "fits:" line is judged at the cap in force too (_row_tip, review tests-6): at 0.25 the
        # 0.5-trained exports fit RL_PWM_7D only — never "RL_PWM / RL_PWM_7D", which RL_PWM would refuse
        for name in ("pwm11att9X_s0", "pwm16N_s3"):
            tip = d.table.item(_row_of(d, name), 0).toolTip().split("\n")
            assert "fits: RL_PWM_7D" in tip and "fits: RL_PWM / RL_PWM_7D" not in tip, (name, tip)
        # ...and it picks as before
        d.table.selectRow(_row_of(d, "pwm16N_s3"))
        assert d.btn_use.isEnabled()
        # the RL slots have no transport cap: no cap on the count line, the rule without it
        for low in ("rl", "rl_7d"):
            d.slot_box.setCurrentIndex(d.slot_box.findData(low))
            assert " · cap" not in d.count_lbl.text(), (low, d.count_lbl.text())
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm_7d"))
        d.table.selectRow(_row_of(d, "pwm16N_s3"))
        d.btn_use.click()
        assert d.chosen() == ("rl_pwm_7d", by["pwm16N_s3"].path)
    finally:
        d.deleteLater()

    # the cap raised to its trained 0.5 (the panel's remedy): RL_PWM lists all three, " · cap 0.50"
    d = RlPolicyDialog(None, ents, "rl_pwm", held, cap_status=_cap_status(cap=0.5, source="panel", sink_cap=0.5))
    try:
        assert set(_names(d)) == set(pwm05)
        assert d.count_lbl.text() == "3 fit RL_PWM · 3 refused hidden · cap 0.50", d.count_lbl.text()
        # ...and the same export's "fits:" line now names both pwm slots — on either slot's view
        for low in ("rl_pwm", "rl_pwm_7d"):
            d.slot_box.setCurrentIndex(d.slot_box.findData(low))
            tip = d.table.item(_row_of(d, "pwm11att9X_s0"), 0).toolTip().split("\n")
            assert "fits: RL_PWM / RL_PWM_7D" in tip, (low, tip)
    finally:
        d.deleteLater()


def test_the_dialog_refuses_an_export_the_sink_would_window_differently(tmp_path):
    """SPEC C8 (b): an export trained BELOW the sink export's cap flies min(cap, its trained cap) while the thruster
    sink enforces min(cap, ceiling) — at the transport cap 0.5 a 0.25-trained export would fly another window than
    the sink's, and workers._sink_gate_compare refuses that pick; the dialog says so first, for BOTH pwm slots (RL_PWM_7D
    allows caps below trained — (b) is not about that). At the shipped 0.25 the same export fits both, and it is the
    one RL_PWM can fly there."""
    from rov_gui.widgets.rl_policy_dialog import RlPolicyDialog

    _app()
    _library(tmp_path)
    _cap_copy(tmp_path / "library" / "pwm16N_t025")
    ents = L.dedupe(L.scan([("station", tmp_path / "station"), ("library", tmp_path / "library")]))
    by = {e.name: e for e in ents}
    low_cap = by["pwm16N_t025"]
    # the fixture is what it says: trained at 0.25, fits both pwm LOWs by slot_fit, readable, not banned
    assert low_cap.cap_trained == 0.25 and not low_cap.error and low_cap.no_fly is None
    assert low_cap.fit["rl_pwm"] is None and low_cap.fit["rl_pwm_7d"] is None
    raised = _cap_status(cap=0.5, source="panel", at="2026-11-02T09:30:00-0800", sink_cap=0.5)
    for slot in ("rl_pwm_7d", "rl_pwm"):
        label = L.LOW_LABELS[slot]
        d = RlPolicyDialog(None, ents, slot, {}, cap_status=raised)
        try:
            assert "pwm16N_t025" not in _names(d), slot                        # hidden by default
            assert d.count_lbl.text().endswith(" · cap 0.50"), d.count_lbl.text()
            d.chk_all.setChecked(True)
            r = _row_of(d, "pwm16N_t025")
            assert d._rows[r][1] == CAP_B, (slot, d._rows[r][1])
            it = d.table.item(r, 0)
            assert it.font().italic() and it.foreground().color().name().lower() == theme.TEXT_DIM.lower()
            assert not (it.flags() & Qt.ItemFlag.ItemIsSelectable)
            assert f"REFUSED for {label}: {CAP_B}" in it.toolTip()
            d.table.selectRow(r)
            assert d._selected_entry() is None and not d.btn_use.isEnabled()
            d._use()
            assert d.chosen() is None
            # the 0.5-trained ones fit at 0.5 — on both slots
            assert d._rows[_row_of(d, "pwm16N_s3")][1] is None, slot
        finally:
            d.deleteLater()
    # at the shipped 0.25 the 0.25-trained export fits both — the only pwm export RL_PWM flies there
    d = RlPolicyDialog(None, ents, "rl_pwm", {}, cap_status=_cap_status())
    try:
        assert _names(d) == ["pwm16N_t025"], _names(d)
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm_7d"))
        assert "pwm16N_t025" in _names(d)
    finally:
        d.deleteLater()


def test_without_a_cap_status_the_dialog_is_exactly_as_before(tmp_path):
    """SPEC C8: cap_status None / {} (a worker that reports no cap) / not passed = today's dialog — the 2-argument
    rule, the same rows, reasons and count line, no " · cap"; and for RL / RL_7D even a cap_status changes nothing."""
    from rov_gui.widgets import rl_policy_dialog as RPD
    from rov_gui.widgets.rl_policy_dialog import RlPolicyDialog

    _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    held = {"rl_pwm_7d": _held(by["pwm11att9X_s0"])}
    for e in ents:
        for low in L.RL_LOWS:
            plain = L.refusal_reason(e, low)
            assert RPD.refusal(e, low) == plain and RPD.refusal(e, low, None) == plain, (e.name, low)
            assert RPD.refusal(e, low, {}) == plain, (e.name, low)

    def view(**kw):
        out = []
        dlg = RlPolicyDialog(None, ents, "rl_pwm_7d", held, **kw)
        try:
            for low in L.RL_LOWS:
                dlg.slot_box.setCurrentIndex(dlg.slot_box.findData(low))
                for show_all in (False, True):
                    dlg.chk_all.setChecked(show_all)
                    out.append((low, show_all, _names(dlg), dlg.count_lbl.text(),
                                [None if r is None else (r[0].name, r[1], r[2]) for r in dlg._rows]))
        finally:
            dlg.deleteLater()
        return out

    base = view()
    assert view(cap_status=None) == base and view(cap_status={}) == base
    assert not any(" · cap" in v[3] for v in base)
    # RL / RL_7D with a cap: identical (the cap is the pwm transport's)
    with_cap = {(v[0], v[1]): v for v in view(cap_status=_cap_status())}
    for v in base:
        if v[0] in ("rl", "rl_7d"):
            assert with_cap[(v[0], v[1])] == v, v[0]


def test_the_slot_travels_with_the_pick_and_only_pickable_rows_accept(tmp_path):
    from rov_gui.widgets.rl_policy_dialog import RlPolicyDialog

    app = _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    # opened on the panel's LOW (rl_pwm_7d), switched to RL_PWM: the pick is FOR RL_PWM
    d = RlPolicyDialog(None, ents, "rl_pwm_7d", {})
    d.show()
    _pump(app, 30)
    try:
        assert "has not reported" in d.held_lbl.text()
        d.slot_box.setCurrentIndex(d.slot_box.findData("rl_pwm"))
        d.chk_all.setChecked(True)
        tv = d.table
        # a double-click on a refused row does nothing — and the click before it selects nothing
        dclicks: list = []
        tv.cellDoubleClicked.connect(lambda r, c: dclicks.append(r))
        r_bad = _row_of(d, "innocent_s0")
        _double_click(tv, r_bad)
        _pump(app, 30)
        assert dclicks == [r_bad], "the double-click did not arrive — the test proves nothing"
        assert d.isVisible() and d.chosen() is None and not d.btn_use.isEnabled()
        assert not tv.selectionModel().selectedIndexes()
        # Enter with nothing pickable selected does nothing either
        QTest.keyClick(tv, Qt.Key.Key_Return)
        _pump(app, 30)
        assert d.isVisible() and d.chosen() is None
        # a click on a pickable row selects it; Enter (the default button) uses it
        r_ok = _row_of(d, "pwm11k0.2_s0")
        rect = tv.visualRect(tv.model().index(r_ok, 1))
        QTest.mouseClick(tv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, rect.center())
        _pump(app, 30)
        assert d.btn_use.isEnabled()
        QTest.keyClick(tv, Qt.Key.Key_Return)
        _pump(app, 30)
        assert d.chosen() == ("rl_pwm", by["pwm11k0.2_s0"].path)
        assert not d.isVisible() and not d._filtering
    finally:
        d.deleteLater()
        _pump(app, 30)
    # a double-click on a pickable row accepts it
    d = RlPolicyDialog(None, ents, "rl", {})
    d.show()
    _pump(app, 30)
    try:
        tv = d.table
        _double_click(tv, _row_of(d, "traj12_s2"), 0)
        _pump(app, 30)
        assert d.chosen() == ("rl", by["traj12_s2"].path) and not d.isVisible()
    finally:
        d.deleteLater()
        _pump(app, 30)


def test_the_dialog_keeps_the_space_bar_and_the_wheel_away_from_its_controls(tmp_path):
    """Space is ALL STOP at the station: no button / checkbox / combo here takes the keyboard, so space can
    never press "Use for …". A wheel notch over the NoFocus slot combo must not change the slot."""
    from rov_gui.widgets.rl_policy_dialog import RlPolicyDialog

    app = _app()
    d = RlPolicyDialog(None, _library(tmp_path), "rl_pwm_7d", {})
    d.show()
    _pump(app, 30)
    try:
        for w in (d.btn_use, d.btn_cancel, d.chk_all, d.slot_box):
            assert w.focusPolicy() == Qt.FocusPolicy.NoFocus, w
        assert app.focusWidget() is d.table, app.focusWidget()
        pos = QtCore.QPointF(5.0, 5.0)
        # both directions: RL_PWM_7D is the LAST item, so a notch down alone could not move it anyway
        for dy in (120, -120):
            wheel = QtGui.QWheelEvent(pos, pos, QtCore.QPoint(0, 0), QtCore.QPoint(0, dy), Qt.MouseButton.NoButton,
                                      Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            QtWidgets.QApplication.sendEvent(d.slot_box, wheel)
            _pump(app, 20)
            assert d.current_slot() == "rl_pwm_7d", f"the wheel ({dy}) changed the slot"
        d.table.selectRow(_row_of(d, "pwm16N_s3"))
        QTest.keyClick(d.table, Qt.Key.Key_Space)
        _pump(app, 20)
        assert d.isVisible() and d.chosen() is None, "space accepted a pick"
    finally:
        d.reject()
        d.deleteLater()
        _pump(app, 30)


def test_esc_anywhere_in_the_dialog_is_estop_and_closes(tmp_path):
    """The RL picker's Esc contract — CkptDialog's (test_offline test_ckpt_dialog_escape_is_estop_and_closes),
    reproduced on a table dialog: every case fires the E-STOP from ONE Esc and rejects, a popup does not outlive
    the dialog, and the app-level filter is gone once the dialog is."""
    from rov_gui.widgets.rl_policy_dialog import RlPolicyDialog

    app = _app()
    ents = _library(tmp_path)

    def _open():
        dlg = RlPolicyDialog(None, ents, "rl_pwm_7d", {})
        fired: list = []
        dlg.estop_requested.connect(lambda: fired.append(1))
        rejected: list = []
        dlg.rejected.connect(lambda: rejected.append(1))
        dlg.show()
        _pump(app, 30)
        assert dlg.isVisible() and dlg._filtering
        return dlg, fired, rejected

    def _closed(dlg, fired, rejected):
        assert fired == [1], "Esc in the picker must be E-STOP, once"
        assert rejected == [1] and not dlg.isVisible()
        assert int(dlg.result()) == int(QtWidgets.QDialog.DialogCode.Rejected)
        assert dlg.chosen() is None, "Esc must never pick"
        assert not dlg._filtering, "the app filter must not outlive the dialog"
        assert app.activePopupWidget() is None, "a popup outlived the dialog"
        dlg.deleteLater()
        _pump(app, 30)

    # 1. a key event sent to the dialog itself
    dlg, fired, rejected = _open()
    QtWidgets.QApplication.sendEvent(dlg, _esc())
    _pump(app, 30)
    _closed(dlg, fired, rejected)

    # 2. the operator clicked a row (focus = the table, a pickable row selected), then Esc
    dlg, fired, rejected = _open()
    tv = dlg.table
    rect = tv.visualRect(tv.model().index(_row_of(dlg, "pwm16N_s3"), 1))
    QTest.mouseClick(tv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, rect.center())
    _pump(app, 30)
    assert app.focusWidget() is tv and dlg.btn_use.isEnabled()
    QTest.keyClick(tv, Qt.Key.Key_Escape)
    _pump(app, 30)
    _closed(dlg, fired, rejected)

    # 3. typing in the filter box (focus = the line edit), then Esc
    dlg, fired, rejected = _open()
    dlg.filter_edit.setFocus()
    QTest.keyClicks(dlg.filter_edit, "pwm")
    _pump(app, 30)
    assert app.focusWidget() is dlg.filter_edit and dlg.filter_edit.text() == "pwm"
    QTest.keyClick(dlg.filter_edit, Qt.Key.Key_Escape)
    _pump(app, 30)
    _closed(dlg, fired, rejected)

    # 4. the slot combo's list is OPEN (it grabs the keyboard and would spend Esc on closing itself): ONE Esc,
    #    and the list is closed BEFORE the E-STOP goes out (close popups -> E-STOP -> reject)
    dlg, fired, rejected = _open()
    popups_at_estop: list = []
    dlg.estop_requested.connect(lambda: popups_at_estop.append(app.activePopupWidget()))
    dlg.slot_box.showPopup()
    _pump(app, 30)
    pop = app.activePopupWidget()
    assert pop is not None and dlg.slot_box.view().isVisible()
    QTest.keyClick(dlg.slot_box.view(), Qt.Key.Key_Escape)
    _pump(app, 30)
    assert popups_at_estop == [None], "a popup was still open when the E-STOP went out"
    assert not dlg.slot_box.view().isVisible() and not pop.isVisible()
    assert dlg.current_slot() == "rl_pwm_7d"
    _closed(dlg, fired, rejected)

    # 4b. ...where nothing closed that list before the KeyPress reached the filter. On the offscreen platform
    #     the Esc's ShortcutOverride already hides a combo list (traced 2026-11-01: ShortcutOverride -> Hide of
    #     QComboBoxPrivateContainer -> KeyPress), so case 4 alone cannot see whether _escape closes it; the
    #     handler the filter calls is driven here with the list still open: closed FIRST, then the E-STOP
    dlg, fired, rejected = _open()
    popups_at_estop = []
    dlg.estop_requested.connect(lambda: popups_at_estop.append(app.activePopupWidget()))
    dlg.slot_box.showPopup()
    _pump(app, 30)
    pop = app.activePopupWidget()
    assert pop is not None
    dlg._escape()
    _pump(app, 30)
    assert popups_at_estop == [None], "a popup was still open when the E-STOP went out"
    assert not pop.isVisible()
    _closed(dlg, fired, rejected)

    # 5. the fallback: with the app filter gone, an Esc that reaches the dialog is still E-STOP (never QDialog's
    #    bare reject)
    dlg, fired, rejected = _open()
    dlg._drop_filter()
    QtWidgets.QApplication.sendEvent(dlg, _esc())
    _pump(app, 30)
    _closed(dlg, fired, rejected)

    # 6. Cancel is a plain reject — no E-STOP — and drops the filter too
    dlg, fired, rejected = _open()
    dlg.btn_cancel.click()
    _pump(app, 30)
    assert fired == [] and rejected == [1] and not dlg._filtering and dlg.chosen() is None
    dlg.deleteLater()
    _pump(app, 30)

    # 7. with the dialogs gone, Esc elsewhere is no longer their business
    other = QtWidgets.QWidget()
    other.show()
    _pump(app, 30)
    seen: list = []
    probe = RlPolicyDialog(None, ents, "rl", {})                # built, never shown: no filter
    probe.estop_requested.connect(lambda: seen.append(1))
    QTest.keyClick(other, Qt.Key.Key_Escape)
    _pump(app, 30)
    assert seen == [] and not probe._filtering
    probe.deleteLater()
    other.close()
    other.deleteLater()
    _pump(app, 30)


# ----------------------------------------------------------------------------- the window
def test_the_window_owns_the_picker_hands_the_keys_back_and_routes_the_pick(tmp_path, monkeypatch):
    """--mpc: `…` on the policy row -> MainWindow._pick_rl_policy: keys handed back (teleop.all_stop) BEFORE the
    dialog runs and the log says so; the dialog is built on the panel's LOW with the worker's rl_policy as held;
    its E-STOP is the window's; an accept goes panel._apply_rl_policy_choice -> bus.cmd_rl_policy (slot, dir); a
    cancel sends nothing; the dialog is deleted either way."""
    from rov_gui.widgets import rl_policy_dialog as RPD
    from rov_gui.window import MainWindow

    class MpcOpts(Opts):
        mpc = True
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    app = _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    monkeypatch.setattr(L, "scan", lambda roots=None: list(ents))      # hermetic: never the real folders
    win = MainWindow(MpcOpts())
    try:
        panel = win._traj_panel
        order: list = []
        monkeypatch.setattr(win.teleop, "all_stop", lambda: order.append("all_stop"))
        stops: list = []
        monkeypatch.setattr(win, "estop", lambda: stops.append(1))     # read when the dialog is wired
        logs: list = []
        win.bus.log.connect(lambda lvl, msg: logs.append((lvl, msg)))
        got: list = []
        win.bus.cmd_rl_policy.connect(lambda low, p: got.append((low, p)))
        dead: list = []
        script = {"pick": "pwm16N_s3"}
        held = {"rl_pwm_7d": _held(by["pwm11att9X_s0"])}

        def fake_exec(dlg):
            order.append("exec")
            dlg.destroyed.connect(lambda *_: dead.append(1))
            script["slot"] = dlg.current_slot()
            script["held"] = dict(dlg._held)
            dlg.estop_requested.emit()                  # Esc's signal reaches the WINDOW's E-STOP
            if script["pick"] is None:
                dlg.reject()
            else:
                dlg.table.selectRow(_row_of(dlg, script["pick"]))
                dlg.btn_use.click()
            return int(dlg.result())

        monkeypatch.setattr(RPD.RlPolicyDialog, "exec_", fake_exec, raising=False)
        monkeypatch.setattr(RPD.RlPolicyDialog, "exec", fake_exec, raising=False)

        panel.set_mode_default("rl_pwm_7d")
        panel.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))
        panel.btn_rlpol.click()                         # the button -> rl_policy_pick_requested -> the window
        _pump(app, 30)
        assert "exec" in order and "all_stop" in order[:order.index("exec")], order
        assert any(lvl == "warn" and "choosing an RL policy" in m and "keyboard is NOT flying" in m
                   and "E-STOP" in m for lvl, m in logs), logs
        assert script["slot"] == "rl_pwm_7d" and script["held"] == held
        assert stops == [1]
        assert got == [("rl_pwm_7d", by["pwm16N_s3"].path)], got
        assert panel.rlpol_name.text() == "→ pwm16N_s3"
        assert dead == [1], "the dialog must be deleted"

        # cancelled: nothing is sent, the dialog is deleted all the same
        script["pick"] = None
        panel.btn_rlpol.click()
        _pump(app, 30)
        assert got == [("rl_pwm_7d", by["pwm16N_s3"].path)] and dead == [1, 1]

        # a LOW that is not RL (the button is hidden then; called directly): the dialog opens on RL_PWM_7D
        script["pick"] = "pwm11att9X_s0"
        panel.set_mode_default("dobmpc")
        win._pick_rl_policy()
        _pump(app, 30)
        assert script["slot"] == "rl_pwm_7d" and got[-1] == ("rl_pwm_7d", by["pwm11att9X_s0"].path)

        # a library that cannot be listed: logged, no dialog
        n = order.count("exec")

        def boom(roots=None):
            raise OSError("disk gone")

        monkeypatch.setattr(L, "scan", boom)
        win._pick_rl_policy()
        assert order.count("exec") == n
        assert any(lvl == "error" and "could not be listed" in m and "disk gone" in m for lvl, m in logs), logs
    finally:
        win.shutdown()
        win.close()
        win.deleteLater()
        _pump(app, 30)


def test_an_accepted_pick_rewrites_the_running_recordings_controller_json(tmp_path, monkeypatch):
    """Review F3 (2026-11-02): REC NAV / REC UI write controller.json ONCE, at the press (MpcWorker.dump_run_meta,
    via bus.cmd_mpc_dump_meta) — so a pick accepted while either still records emits cmd_mpc_dump_meta again for
    each running recording, with the folders the REC buttons used (REC NAV: its nav_<hhmmss> dir; REC UI: the UI
    video's folder), and only AFTER bus.cmd_rl_policy: both go queued to the one MpcWorker, so the dump runs after
    the pick. Not on a cancel, not without a recording."""
    from rov_gui.recorder import RecStats
    from rov_gui.widgets import rl_policy_dialog as RPD
    from rov_gui.window import MainWindow

    class MpcOpts(Opts):
        mpc = True
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    app = _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    monkeypatch.setattr(L, "scan", lambda roots=None: list(ents))      # hermetic: never the real folders
    win = MainWindow(MpcOpts())
    stats0 = win.recorder.stats
    try:
        panel = win._traj_panel
        monkeypatch.setattr(win.teleop, "all_stop", lambda: None)
        sent: list = []
        win.bus.cmd_rl_policy.connect(lambda low, p: sent.append(("pick", low, p)))
        win.bus.cmd_mpc_dump_meta.connect(lambda d: sent.append(("dump", d)))
        script = {"pick": "pwm16N_s3"}

        def fake_exec(dlg):
            if script["pick"] is None:
                dlg.reject()
            else:
                dlg.table.selectRow(_row_of(dlg, script["pick"]))
                dlg.btn_use.click()
            return int(dlg.result())

        monkeypatch.setattr(RPD.RlPolicyDialog, "exec_", fake_exec, raising=False)
        monkeypatch.setattr(RPD.RlPolicyDialog, "exec", fake_exec, raising=False)
        panel.set_mode_default("rl_pwm_7d")
        panel.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy={"rl_pwm_7d": _held(by["pwm11att9X_s0"])}))
        pick = ("pick", "rl_pwm_7d", by["pwm16N_s3"].path)
        run = tmp_path / "20261102" / "1102_120000"
        nav_dir, ui_path = run / "nav_120000", run / "ui_20261102_120000.mp4"

        # no recording running: the pick alone
        assert win._nav_rec is None and not win.recorder.stats.recording
        win._pick_rl_policy()
        _pump(app, 30)
        assert sent == [pick], sent

        # REC NAV and REC UI both running: the pick, THEN one dump per recording, in the REC buttons' folders
        win._nav_rec = {"dir": nav_dir}             # the key _toggle_nav_record keeps (the rest are file handles)
        win.recorder.stats = RecStats(recording=True, path=ui_path)
        sent.clear()
        win._pick_rl_policy()
        _pump(app, 30)
        assert sent == [pick, ("dump", str(nav_dir)), ("dump", str(run))], sent

        # cancelled: neither the pick nor a dump
        script["pick"] = None
        sent.clear()
        win._pick_rl_policy()
        _pump(app, 30)
        assert sent == [], sent

        # one recording each: REC UI only, then REC NAV only
        script["pick"] = "pwm16N_s3"
        win._nav_rec = None
        sent.clear()
        win._pick_rl_policy()
        assert sent == [pick, ("dump", str(run))], sent
        win._nav_rec = {"dir": nav_dir}
        win.recorder.stats = stats0
        sent.clear()
        win._pick_rl_policy()
        assert sent == [pick, ("dump", str(nav_dir))], sent
    finally:
        # the fakes go before shutdown: it would close REC NAV's (absent) files and stop a recording never started
        win._nav_rec = None
        win.recorder.stats = stats0
        win.shutdown()
        win.close()
        win.deleteLater()
        _pump(app, 30)


def test_the_window_hands_the_dialog_the_cap_the_panel_last_saw(tmp_path, monkeypatch):
    """SPEC C7/C8 (2026-11-02): MainWindow._pick_rl_policy builds the dialog with cap_status = the panel's
    rl_pwm_cap_status() — the worker's cap in force as last reported — so the picker judges RL_PWM at the cap the
    LOW row shows; a worker that reports no cap ({}) leaves the dialog judging without one."""
    from rov_gui.widgets import rl_policy_dialog as RPD
    from rov_gui.window import MainWindow

    class MpcOpts(Opts):
        mpc = True
        mpc_config = str(ROOT / "config" / "hw_mpc.yaml")
        nav_config = str(ROOT / "config" / "hw_nav.yaml")
        nav_geometry = None

    app = _app()
    ents = _library(tmp_path)
    by = {e.name: e for e in ents}
    monkeypatch.setattr(L, "scan", lambda roots=None: list(ents))      # hermetic: never the real folders
    win = MainWindow(MpcOpts())
    try:
        panel = win._traj_panel
        monkeypatch.setattr(win.teleop, "all_stop", lambda: None)
        seen: dict = {}

        def fake_exec(dlg):
            seen["count"] = dlg.count_lbl.text()                # opened on the panel's LOW (RL_PWM_7D)
            dlg.slot_box.setCurrentIndex(dlg.slot_box.findData("rl_pwm"))
            dlg.chk_all.setChecked(True)
            seen["rl_pwm"] = {r[0].name: r[1] for r in dlg._rows if r is not None}
            dlg.reject()
            return int(dlg.result())

        monkeypatch.setattr(RPD.RlPolicyDialog, "exec_", fake_exec, raising=False)
        monkeypatch.setattr(RPD.RlPolicyDialog, "exec", fake_exec, raising=False)
        held = {"rl_pwm_7d": _held(by["pwm11att9X_s0"])}
        panel.set_mode_default("rl_pwm_7d")
        panel.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held, rl_pwm_cap=_cap_status()))
        win._pick_rl_policy()
        _pump(app, 30)
        assert seen["count"] == "2 fit RL_PWM_7D · 4 refused hidden · cap 0.25", seen
        assert seen["rl_pwm"]["pwm16N_s3"] == CAP_A_RL_PWM and seen["rl_pwm"]["pwm11k0.2_s0"] == CAP_A_RL_PWM

        # the worker reports no cap: judged without one
        panel.add_status(MpcStatus(mode="rl_pwm_7d", rl_policy=held))
        assert panel.rl_pwm_cap_status() == {}
        win._pick_rl_policy()
        _pump(app, 30)
        assert seen["count"] == "2 fit RL_PWM_7D · 4 refused hidden", seen
        assert seen["rl_pwm"]["pwm16N_s3"] is None and seen["rl_pwm"]["pwm11k0.2_s0"] is None
    finally:
        win.shutdown()
        win.close()
        win.deleteLater()
        _pump(app, 30)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
