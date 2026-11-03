#!/usr/bin/env python3
"""rl_policy_dialog.py — the RL POLICY picker: which export an RL LOW flies (2026-11-01, control/rl_library.py;
operator request: "rl_7d_pwm 여러 개를 훈련해 왔다. GUI 켜기 전에 hw_mpc.yaml 에서 policy_dir 을 정하는 대신 GUI 에서 바로
바꾸고 싶다. 그리고 RL 훈련한 것들을 어떤 폴더에 다 모아두고.").

A table of the RL exports the station can fly — rl_library.scan() over the station folder rov_gui/control/rl_policies/
and the library data/rl_policies/, deduped by content id (the window does the scan) — for ONE RL LOW slot: the
"Slot" combo, which opens on the LOW the panel shows and may be switched to any of the four (a slot whose launch build
failed cannot be SELECTED as the LOW — the worker refuses it — so its export can only be fixed from here). The export
the worker HOLDS for the slot is marked "● " (MpcStatus.rl_policy, handed in by the window: the honesty rule — this
dialog never guesses what is flying), and the line above the table says what it is — in FAIL red, "NO-FLY: <reason>"
first, when that export is banned from flight (MpcStatus.rl_policy[slot]["no_fly"]: a pick of a banned export is
refused, but a launch config can still name one; 2026-11-02, review F1).

OWNED BY THE WINDOW (MainWindow._pick_rl_policy), never the panel — CkptDialog's contract (widgets/trajectory.py):
the window hands the keyboard back from the vehicle first (teleop.all_stop) and says so in the log, connects
``estop_requested`` to its E-STOP, runs this modal and sends the pick through the panel (_apply_rl_policy_choice ->
bus.cmd_rl_policy -> MpcWorker.set_rl_policy, which rebuilds that LOW's follower with every build gate, or refuses and
says why). The JOYSTICK keeps flying: its reader is a timer, and a modal exec keeps timers alive. Nothing here writes
anything: hw_mpc.yaml <low>.policy_dir stays the launch default (a pick is session only).

What the table offers is a PRE-FILTER only: rl_library.refusal_reason, the ONE rule candidates(), this dialog and
`./c3 rl-policies list` share (unreadable, the flight ban, a smoke-length export below MIN_ITERATION training
iterations, then slot_fit — refusal() delegates to it, 2026-11-02, review F10). A refused row is hidden unless "show
refused / no-fly" is ticked; then it is shown greyed (red for a flight ban or an unreadable export) with the reason in
its tooltip and on the line under the table, and it can be neither selected nor accepted (no ItemIsSelectable, and
``_use`` re-checks the row itself). The held row is shown even when refused — it is what flies, and hiding it would
hide that. The follower's build stays authoritative.

THE CAP (2026-11-02, operator: "… 이렇게 gui를 열면 거기서 바꿀 수 있게 해달라는 소리였어. 그리고 캡도" — the transport
cap rl_pwm.pwm_cap is now set on the LOW row's cap combo, session only): the window also hands in the panel's
MpcStatus.rl_pwm_cap (``cap_status``: the cap the worker holds, the thruster sink's window, each pwm LOW's
allow_caps_below_trained), and refusal() passes it to rl_library.refusal_reason as its ``cap_ctx`` — so for RL_PWM /
RL_PWM_7D an export the cap IN FORCE would get refused (trained above it on a LOW that refuses caps below trained, or
flying another window than the sink's) is greyed like a misfit, its reason naming the remedy ("set the LOW row's cap
to …"). The count line says the cap it judged at (" · cap 0.25"). No cap_status, or an empty one (a worker that
reports no cap) = the rule without the cap, exactly as before (the 2-argument call).

KEYBOARD (window.py "Keyboard"): Esc ANYWHERE in the dialog — the table, the filter box, the slot combo with its list
open, a context menu — closes the open popups, emits ``estop_requested`` and rejects: ONE Esc, the E-STOP it is
everywhere else in the station. It is caught by an APPLICATION-level event filter installed on show and removed on
hide — CkptDialog's mechanism (review 2026-09-11), reproduced here rather than shared: a popup eats Esc to close
itself, and a plain QDialog closes on Esc (QDialog::keyPressEvent -> reject) with no E-STOP at all. The buttons and
the checkbox are NoFocus, so the space bar (ALL STOP at the station) cannot press "Use for …"; Enter still uses the
selected row (the default button). The slot combo ignores the mouse wheel (the panel's yaw-box rule, review
2026-10-14: a notch over a NoFocus combo changes it without a click).

STYLE: theme.py has no QDialog / QTableView / QHeaderView rule and its '*' rule forces light text, so a bare table
would draw light text on Qt's light base. The dialog carries its own sheet built from the theme constants: a dark
well, a readable header, the ACCENT selection written in TEXT_ON_FILL (8.9:1), refused rows in TEXT_DIM (5.7:1 on BG,
5.3:1 on PANEL — the rows alternate the two) and banned / unusable rows in FAIL (5.1:1 / 4.7:1); nothing a reader
must see is in TEXT_FAINT (2.7:1 on PANEL) [유도: WCAG contrast of the theme.py hex constants, 2026-11-01].
"""

from __future__ import annotations

import copy
import math

from .. import theme
from ..control import rl_library
from ..qt import QtCore, QtGui, QtWidgets, Qt, Signal

#: The table's columns, in order (DESIGN 2026-11-01 §4b).
COLUMNS = ("name", "trained", "preset", "cap", "iter", "vel", "where", "id")
C_NAME = 0
#: The held export's marker on its name cell (MpcStatus.rl_policy[slot]["id"] == the row's id).
HELD_MARK = "● "
#: The slot a dialog opened for an unknown LOW starts on (= MainWindow._pick_rl_policy's fallback).
DEFAULT_SLOT = "rl_pwm_7d"
#: Both texts of the worker's below-the-trained-cap note end with this (control/workers.py MpcWorker._rl_cap_note):
#: how the LOW row and the held line know a status note CARRIES the cap fact (2026-11-02, review F1 — the row used
#: to explain every WARN as the cap case, a banned export's included).
CAP_NOTE_MARK = "(allow_caps_below_trained)"
#: The RL LOWs that fly the pwm transport and so SHARE its cap rl_pwm.pwm_cap (MpcWorker._RL_PWM_SLOTS; the cap
#: lives in the transport block — geometry refuses a pwm_cap in rl_pwm_7d:). The dialog's count line and the LOW
#: row's cap combo (widgets/trajectory.py) speak of the cap for these two only (2026-11-02).
PWM_SLOTS = ("rl_pwm", "rl_pwm_7d")


def empty_text(label: str) -> str:
    """The one disabled line of a slot with nothing pickable."""
    return f"no exports for {label} — run ./c3 rl-policies sync"


def refusal(entry, slot: str, cap_ctx=None):
    """Why ``entry`` cannot be picked for ``slot``, or None — rl_library.refusal_reason, THE rule (unreadable, the
    flight ban, a smoke-length export, then the slot pre-filter; a missing verdict is a refusal: fail-closed).
    Delegated, never worded here again (2026-11-02, review F10): this was a copy of candidates' rule, and the
    iteration floor added there (MIN_ITERATION) would have left a smoke export pickable HERE while candidates()
    refused it.

    ``cap_ctx`` (2026-11-02): MpcStatus.rl_pwm_cap as the panel last saw it. Passed on — refusal_reason judges the
    pwm slots at the cap in force too — only when it is a NON-EMPTY dict; None / {} (a worker that reports no cap)
    keeps the 2-argument call, i.e. exactly the rule without the cap."""
    if isinstance(cap_ctx, dict) and cap_ctx:
        return rl_library.refusal_reason(entry, slot, cap_ctx)
    return rl_library.refusal_reason(entry, slot)


def cap_in_force(cap_status):
    """The transport cap MpcStatus.rl_pwm_cap says is in force (its "cap"), or None — absent, not a number, not
    finite: the dialog then says nothing about a cap (no number the worker did not give)."""
    if not isinstance(cap_status, dict):
        return None
    try:
        v = float(cap_status.get("cap"))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _red(entry) -> bool:
    """A refused row drawn in FAIL red rather than grey: unreadable, or banned — ``no_fly is not None``,
    refusal_reason's own test (not truthiness), so a row whose reason reads "flight ban" is never drawn as a mere
    misfit."""
    return bool(entry.error) or entry.no_fly is not None


def nofly_note_rest(note, no_fly) -> str:
    """What a held slot's status note says BESIDES its flight ban (2026-11-02, review F1). MpcWorker._rl_policy_status
    writes a banned held export's note as "no-fly — <reason>", + "; <cap note>" when it also flies below its trained
    cap; split on that known head, never on "; " — the ban texts contain "; " themselves (rl_library._BAN_PWM10). A
    note without the head (a refusal, an unbanned export's) comes back whole; "" when the note is the ban alone. One
    parser for the LOW row (widgets/trajectory.py _render_rlpol) and the held line here."""
    note, no_fly = str(note or ""), str(no_fly or "")
    head = f"no-fly — {no_fly}"
    if not no_fly or not note.startswith(head):
        return note
    rest = note[len(head):]
    return rest[2:] if rest.startswith("; ") else rest.strip()


def _sheet() -> str:
    """The dialog's own rules (see the module docstring), from the theme constants. Only what theme.STYLESHEET lacks
    — buttons, the combo, the line edit, the checkbox indicator and the scroll bars keep the station's language."""
    t = theme
    return f"""
QDialog {{ background: {t.PANEL}; }}
QLabel, QCheckBox {{ color: {t.TEXT}; }}
QTableView {{
    background: {t.BG};
    alternate-background-color: {t.PANEL};
    color: {t.TEXT};
    gridline-color: {t.BORDER};
    border: 1px solid {t.BORDER};
    border-radius: 4px;
    selection-background-color: {t.ACCENT};
    selection-color: {t.TEXT_ON_FILL};
    outline: none;
}}
QTableView::item {{ padding: 0px 6px; border: none; }}
QTableView::item:selected {{ background: {t.ACCENT}; color: {t.TEXT_ON_FILL}; }}
QTableView::item:focus {{ border: 1px solid {t.BORDER_HI}; }}
QTableView::item:selected:focus {{ border: 1px solid {t.ACCENT}; }}
QHeaderView {{ background: {t.PANEL_HI}; border: none; }}
QHeaderView::section {{
    background: {t.PANEL_HI};
    color: {t.TEXT_DIM};
    border: none;
    border-right: 1px solid {t.BORDER};
    border-bottom: 1px solid {t.BORDER_HI};
    padding: 3px 6px;
    font-size: 11px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {t.PANEL_HI}; border: none; }}
"""


class RlPolicyDialog(QtWidgets.QDialog):
    """Pick an RL export for one RL LOW slot. ``chosen()`` = (slot, absolute export dir) after an accept, else None.

    ``entries``: rl_library.Entry list (the window passes dedupe(scan())); ``slot``: the RL LOW to open on;
    ``held``: MpcStatus.rl_policy ({slot: {name, path, rel, id, source, state, note, no_fly}}) — what the worker
    holds; ``cap_status`` (2026-11-02): MpcStatus.rl_pwm_cap as the panel last saw it (TrajectoryWindow
    .rl_pwm_cap_status()) — the pwm slots are judged at the cap in force (refusal()'s ``cap_ctx``); None / {} = the
    rule without the cap."""

    estop_requested = Signal()

    def __init__(self, parent=None, entries=(), slot: str = DEFAULT_SLOT, held=None, cap_status=None):
        super().__init__(parent)
        self.setWindowTitle("RL policy")
        self.setModal(True)
        self.setStyleSheet(_sheet())
        self._entries = list(entries or ())
        self._held = {str(k): dict(v) for k, v in (held or {}).items() if isinstance(v, dict)}
        # a COPY (nested lists / dicts too): the dialog judges at the cap the worker reported when it opened, and
        # nothing here may edit the panel's truth
        self._cap = copy.deepcopy(cap_status) if isinstance(cap_status, dict) else {}
        self._chosen = None
        self._filtering = False          # the app-level Esc filter is installed
        self._rows: list = []            # per table row: (Entry, refusal or None, held) | None (the empty line)

        # ---- top: slot · filter · show refused ---------------------------------------------------------------
        self.slot_box = QtWidgets.QComboBox()
        for low in rl_library.RL_LOWS:
            self.slot_box.addItem(rl_library.LOW_LABELS[low], low)
        i = self.slot_box.findData(str(slot))
        self.slot_box.setCurrentIndex(i if i >= 0 else self.slot_box.findData(DEFAULT_SLOT))
        self.slot_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slot_box.setToolTip(
            "which RL LOW the pick is for — opens on the LOW the panel shows. A LOW whose launch build failed "
            "cannot be selected on the panel (the worker refuses it), so its export is chosen here.")
        self.slot_box.installEventFilter(self)          # the wheel guard (eventFilter)
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("filter: name / preset / where")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip("case-insensitive substring of the name, the DR preset or where it lives")
        self.chk_all = QtWidgets.QCheckBox("show refused / no-fly")
        self.chk_all.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_all.setToolTip(
            "also list the exports this slot cannot fly — the wrong action mode / output / no roll-pitch "
            "reference, banned from flight (RL_controller README §13.8), smoke-length (fewer than "
            f"{rl_library.MIN_ITERATION} training iterations), or unreadable — greyed, with the reason; "
            "they cannot be picked")
        self.count_lbl = QtWidgets.QLabel("")
        self.count_lbl.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        lbl_slot = QtWidgets.QLabel("Slot")
        lbl_slot.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(lbl_slot)
        top.addWidget(self.slot_box)
        top.addWidget(self.filter_edit, 1)
        top.addWidget(self.chk_all)
        top.addWidget(self.count_lbl)

        # ---- what the worker holds for the slot ------------------------------------------------------------
        self.held_lbl = QtWidgets.QLabel("")
        self.held_lbl.setWordWrap(True)

        # ---- the table --------------------------------------------------------------------------------------
        tv = self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        tv.setHorizontalHeaderLabels(list(COLUMNS))
        tv.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tv.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        tv.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tv.verticalHeader().setVisible(False)
        # Fixed one-line rows, compact: a synced library is on the order of the RL_controller exports
        # (logs/rsl_rl/rov_posehold/*/exported — 182 dirs listed 2026-11-01), most of them pwm candidates, and
        # the list is read by scrolling; Qt's default row is about a third taller
        tv.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        tv.verticalHeader().setDefaultSectionSize(24)
        tv.setShowGrid(False)
        tv.setAlternatingRowColors(True)
        tv.setWordWrap(False)
        tv.setTabKeyNavigation(False)                   # Tab leaves the table instead of walking its cells
        # middle elision: '<tag>@<YYYYMMDD-HHMMSS>' library names differ at the END, run tags at the start
        tv.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        hh = tv.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(C_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setHighlightSections(False)
        hh.setSectionsClickable(False)                  # one order: newest first (rl_library.candidates)
        tv.itemSelectionChanged.connect(self._sync)
        tv.currentCellChanged.connect(lambda r, _c, _pr, _pc: self._show_detail(r))
        tv.cellDoubleClicked.connect(self._double_clicked)

        # ---- the current row, spelled out (the reason a greyed row is greyed, without hovering) -------------
        self.detail_lbl = QtWidgets.QLabel("")
        self.detail_lbl.setWordWrap(True)
        self.detail_lbl.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")

        # ---- buttons ------------------------------------------------------------------------------------------
        hint = QtWidgets.QLabel("Esc closes this AND is E-STOP · the keyboard is not flying · the joystick is · "
                                "session only (hw_mpc.yaml is the launch default)")
        hint.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px;")
        # wraps instead of demanding its one-line width: unwrapped it set the dialog's MINIMUM width to 919 px
        # (offscreen, theme applied, 2026-11-01), wider than the 900 px it opens at
        hint.setWordWrap(True)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_cancel.setAutoDefault(False)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_use = QtWidgets.QPushButton("")
        self.btn_use.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_use.setAutoDefault(False)
        self.btn_use.setDefault(True)                   # Enter = use the selected row (when one is pickable)
        self.btn_use.clicked.connect(self._use)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(hint, 1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_use)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        lay.addLayout(top)
        lay.addWidget(self.held_lbl)
        lay.addWidget(tv, 1)
        lay.addWidget(self.detail_lbl)
        lay.addLayout(bottom)
        self.setSizeGripEnabled(True)
        self.resize(900, 520)

        self.slot_box.currentIndexChanged.connect(lambda _i: self._slot_changed())
        self.filter_edit.textChanged.connect(lambda _t: self._rebuild())
        self.chk_all.toggled.connect(lambda _on: self._rebuild())
        self._slot_changed(initial=True)

    # ---------------------------------------------------------------------------------------------- the answer
    def chosen(self):
        """(slot, absolute export dir) once accepted with a pickable row; None otherwise (cancelled, Esc)."""
        return self._chosen

    def current_slot(self) -> str:
        """The RL LOW key the dialog is picking for (never the label)."""
        return str(self.slot_box.currentData() or DEFAULT_SLOT)

    def _selected_entry(self):
        """The selected row's Entry when it is PICKABLE, else None — the button, Enter and a double-click all ask
        this, so a row that is refused can never be accepted even if Qt let it be selected."""
        sm = self.table.selectionModel()
        rows = sorted({ix.row() for ix in sm.selectedIndexes()}) if sm is not None else []
        if len(rows) != 1 or not 0 <= rows[0] < len(self._rows) or self._rows[rows[0]] is None:
            return None
        entry, why, _held = self._rows[rows[0]]
        return entry if why is None else None

    def _use(self) -> None:
        e = self._selected_entry()
        if e is None:
            return
        self._chosen = (self.current_slot(), str(e.path))
        self.accept()

    def _double_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._rows) and self._rows[row] is not None and self._rows[row][1] is None:
            self.table.selectRow(row)
            self._use()

    def _sync(self) -> None:
        self.btn_use.setEnabled(self._selected_entry() is not None)

    # ---------------------------------------------------------------------------------------------- the table
    def _slot_changed(self, initial: bool = False) -> None:
        label = rl_library.LOW_LABELS.get(self.current_slot(), self.current_slot())
        self.btn_use.setText(f"Use for {label}")
        self._show_held()
        self._rebuild(scroll_to_held=True)

    def _show_held(self) -> None:
        """The line above the table: what the WORKER holds for the slot (MpcStatus.rl_policy at open)."""
        slot = self.current_slot()
        label = rl_library.LOW_LABELS.get(slot, slot)
        d = self._held.get(slot)
        if not d:
            text, colour = f"{label}: the worker has not reported what it holds", theme.TEXT_DIM
            tip = ""
        else:
            name, st, note = str(d.get("name") or ""), str(d.get("state") or ""), str(d.get("note") or "")
            nf = str(d.get("no_fly") or "")             # "" or the held export's ban, in every state
            pid = str(d.get("id") or "")
            text = (f"{label} holds {name} (id {pid[:8] or '—'}, from {d.get('source') or '—'})" if name
                    else f"{label} holds no export")
            colour = {"ok": theme.OK, "warn": theme.WARN, "error": theme.FAIL,
                      "refused": theme.FAIL}.get(st, theme.TEXT_DIM)
            if nf:
                # 2026-11-02 (review F1): a banned held export says so FIRST and in FAIL red, whatever the state —
                # never the amber WARN its cap note alone would give it (the worker's warn note leads with the ban;
                # nofly_note_rest leaves the cap fact, shown after it as before)
                text += f" — NO-FLY: {nf}"
                colour = theme.FAIL
            if st == "error":
                text += f" — ERROR: {note or 'not built'}"
            elif st == "refused":
                text += f" — the last pick was refused: {note}"
            elif st == "warn":
                rest = nofly_note_rest(note, nf)
                if rest or not nf:
                    text += f" — WARN: {rest}"
            tip = str(d.get("path") or d.get("rel") or "")
        self.held_lbl.setText(text)
        self.held_lbl.setToolTip(tip)
        self.held_lbl.setStyleSheet(f"color:{colour}; font-size:11px;")

    @staticmethod
    def _cells(e) -> tuple:
        bad = bool(e.error)
        return (e.name,
                e.trained or "—",
                e.dr_preset or "—",
                "—" if e.cap_trained is None else f"{e.cap_trained:g}",
                "—" if e.iteration is None else str(e.iteration),
                "—" if bad else ("DVL blend" if e.vel_trained else "tag fd"),
                e.where or "—",
                e.id or "—")

    @staticmethod
    def _row_tip(e, why, label: str, held: bool, cap_ctx=None) -> str:
        # "fits:" judged by the same rule as the row (the cap in force included for the pwm slots)
        fits = [rl_library.LOW_LABELS[k] for k in rl_library.RL_LOWS if refusal(e, k, cap_ctx) is None]
        lines = [str(e.path),
                 f"checkpoint {e.checkpoint or '—'}",
                 f"source {e.source or '—'}",
                 f"id {e.id or '—'}  (obs_spec.json {e.obs_sha1[:12] or '—'} · "
                 f"policy_weights.npz {e.npz_sha1[:12] or '—'})"]
        if held:
            lines.append(f"● the worker holds this export for {label}")
        lines.append(f"fits {label}" if why is None else f"REFUSED for {label}: {why}")
        lines.append("fits: " + (" / ".join(fits) if fits else "no RL LOW"))
        return "\n".join(lines)

    def _rebuild(self, scroll_to_held: bool = False) -> None:
        slot = self.current_slot()
        label = rl_library.LOW_LABELS.get(slot, slot)
        keep = self._selected_entry()
        keep_path = None if keep is None else keep.path
        show_all = self.chk_all.isChecked()
        flt = self.filter_edit.text().strip().lower()
        held_id = str((self._held.get(slot) or {}).get("id") or "")
        rows, n_fit, n_hidden = [], 0, 0
        for e in rl_library.candidates(self._entries, slot, include_refused=True):
            why = refusal(e, slot, self._cap)
            held = bool(held_id) and e.id == held_id
            if why is None:
                n_fit += 1
            elif not show_all and not held:
                n_hidden += 1
                continue
            if flt and flt not in f"{e.name}\n{e.dr_preset}\n{e.where}".lower():
                continue
            rows.append((e, why, held))
        tv = self.table
        tv.setUpdatesEnabled(False)
        tv.blockSignals(True)
        try:
            tv.clearSpans()
            tv.setRowCount(0)
            self._rows = []
            if not rows:
                msg = empty_text(label) if not flt else f"no export of {label} matches '{flt}'"
                if not flt and n_hidden:
                    msg += f" ({n_hidden} refused hidden — tick 'show refused / no-fly' to see why)"
                tv.setRowCount(1)
                it = QtWidgets.QTableWidgetItem(msg)
                it.setFlags(Qt.ItemFlag.NoItemFlags)    # disabled: not a row, a sentence
                it.setForeground(QtGui.QBrush(QtGui.QColor(theme.TEXT_DIM)))
                tv.setItem(0, C_NAME, it)
                tv.setSpan(0, 0, 1, len(COLUMNS))
                self._rows = [None]
            else:
                tv.setRowCount(len(rows))
                dim = QtGui.QBrush(QtGui.QColor(theme.TEXT_DIM))
                red = QtGui.QBrush(QtGui.QColor(theme.FAIL))
                for r, (e, why, held) in enumerate(rows):
                    tip = self._row_tip(e, why, label, held, self._cap)
                    for c, text in enumerate(self._cells(e)):
                        it = QtWidgets.QTableWidgetItem((HELD_MARK + text) if (c == C_NAME and held) else text)
                        flags = Qt.ItemFlag.ItemIsEnabled
                        if why is None:
                            flags |= Qt.ItemFlag.ItemIsSelectable
                        it.setFlags(flags)
                        it.setToolTip(tip)
                        if held or why is not None:
                            f = it.font()
                            f.setBold(held)
                            f.setItalic(why is not None)
                            it.setFont(f)
                        if why is not None:
                            # NO foreground on a pickable row: the ACCENT selection is written in the sheet's
                            # TEXT_ON_FILL only where the item carries no colour of its own
                            it.setForeground(red if _red(e) else dim)
                        tv.setItem(r, c, it)
                    self._rows.append((e, why, held))
        finally:
            tv.blockSignals(False)
            tv.setUpdatesEnabled(True)
        # the selection survives a filter / checkbox change while its row is still listed and pickable
        sel = held_row = None
        for r, row in enumerate(self._rows):
            if row is None:
                continue
            if keep_path is not None and row[0].path == keep_path and row[1] is None:
                sel = r
            if row[2]:
                held_row = r
        if sel is not None:
            tv.selectRow(sel)
        if held_row is not None and scroll_to_held:
            # the held export is where the eye starts: scrolled to and CURRENT (arrow keys start there) — not
            # selected, so Enter does not re-pick what already flies
            tv.scrollToItem(tv.item(held_row, C_NAME), QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
            if sel is None:
                tv.selectionModel().setCurrentIndex(
                    tv.model().index(held_row, C_NAME), QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
        n_ref = sum(1 for row in self._rows if row is not None and row[1] is not None)
        # the cap the pwm slots were judged at (2026-11-02), last: the counts above depend on it — and nothing when
        # the worker reported no cap or the slot is not a pwm LOW (its rule has no cap)
        cap = cap_in_force(self._cap) if slot in PWM_SLOTS else None
        self.count_lbl.setText(f"{n_fit} fit {label}" + (f" · {n_hidden} refused hidden" if n_hidden
                                                         else (f" · {n_ref} refused shown" if n_ref else ""))
                               + ("" if cap is None else f" · cap {cap:.2f}"))
        self.count_lbl.setToolTip(
            "" if cap is None else
            f"judged at the transport cap in force, rl_pwm.pwm_cap {cap:.2f} (the LOW row's cap combo; session "
            f"only): an export trained above it is refused for a LOW that refuses caps below trained "
            f"(allow_caps_below_trained), and one that would fly another window than the thruster sink's is "
            f"refused for both pwm LOWs")
        self._fit_width()
        self._show_detail(tv.currentRow())
        self._sync()

    def _fit_width(self) -> None:
        """The table is never narrower than its columns' contents. The NAME stretches into any spare width, but a
        Stretch section is also the one that gives way first: at 640 px it was cut to "p…1" while every other
        column kept its content (offscreen render, 2026-11-01) — and the name is what the operator picks by.
        The empty-slot sentence spans the row and does not count (it elides)."""
        tv, hh = self.table, self.table.horizontalHeader()
        need = 0
        for c in range(tv.columnCount()):
            w = hh.sectionSizeHint(c)
            if self._rows and self._rows[0] is not None:
                w = max(w, tv.sizeHintForColumn(c))
            need += w
        tv.setMinimumWidth(need + 2 * tv.frameWidth() + tv.verticalScrollBar().sizeHint().width() + 2)

    def _show_detail(self, row: int) -> None:
        slot = self.current_slot()
        label = rl_library.LOW_LABELS.get(slot, slot)
        if not 0 <= row < len(self._rows) or self._rows[row] is None:
            self.detail_lbl.setText("")
            return
        e, why, held = self._rows[row]
        head = (HELD_MARK if held else "") + e.name
        text = f"{head} — {why}" if why is not None else f"{head} — fits {label}"
        self.detail_lbl.setText(f"{text} · {e.rel}")
        colour = theme.FAIL if _red(e) else theme.TEXT_DIM
        self.detail_lbl.setStyleSheet(f"color:{colour}; font-size:11px;")

    # ------------------------------------------------------------------------- the Esc contract (CkptDialog's)
    def showEvent(self, ev) -> None:                    # noqa: N802 - Qt naming
        super().showEvent(ev)
        app = QtWidgets.QApplication.instance()
        if app is not None and not self._filtering:
            app.installEventFilter(self)
            self._filtering = True
        self.table.setFocus()

    def hideEvent(self, ev) -> None:                    # noqa: N802 - Qt naming
        self._drop_filter()
        super().hideEvent(ev)

    def _drop_filter(self) -> None:
        # Safe from inside eventFilter: QObject.removeEventFilter nulls the entry rather than reshaping the list
        # being walked. A dialog destroyed while shown needs nothing: Qt keeps event filters as QPointers.
        app = QtWidgets.QApplication.instance()
        if app is not None and self._filtering:
            app.removeEventFilter(self)
        self._filtering = False

    def eventFilter(self, obj, ev) -> bool:             # noqa: N802 - Qt naming
        # Installed on the APPLICATION while visible (every event of the GUI thread passes here, so: type first,
        # nothing else touched for the rest) and on the slot combo (the wheel guard). Never super(): the base
        # filter answers False anyway, and a dying widget (a deleted window's popup list, test_offline's
        # 2026-10-03 abort) must not be handed back to it.
        kind = ev.type()
        if kind == QtCore.QEvent.Type.KeyPress:
            if self._filtering and self.isVisible() and ev.key() == Qt.Key.Key_Escape:
                self._escape()
                return True                     # the table / line edit / popup never sees it
            return False
        if kind == QtCore.QEvent.Type.Wheel and obj is self.slot_box:
            return True
        return False

    def _escape(self) -> None:
        # A popup open over the dialog (the slot combo's list, the line edit's context menu) would have eaten
        # this Esc to close itself; the filter took it instead, so close it here or it outlives the dialog as an
        # orphan window. Bounded: a popup that refuses to close must not hang the E-STOP — and whatever happens
        # while closing them, the E-STOP and the reject still run.
        app = QtWidgets.QApplication.instance()
        try:
            if self.slot_box.view().isVisible():
                self.slot_box.hidePopup()
            for _ in range(4):
                pw = app.activePopupWidget() if app is not None else None
                if pw is None:
                    break
                pw.close()
        finally:
            try:
                self.estop_requested.emit()
            finally:
                self.reject()                   # hides -> hideEvent drops the filter

    def keyPressEvent(self, ev) -> None:                # noqa: N802 - Qt naming
        # The filter answers first while it is installed; this is the same contract for a key that reaches the
        # dialog itself (QDialog's own Esc is a bare reject — never let that run).
        if ev.key() == Qt.Key.Key_Escape:
            self._escape()
            return
        super().keyPressEvent(ev)
