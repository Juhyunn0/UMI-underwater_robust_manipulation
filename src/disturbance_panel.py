#!/usr/bin/env python3
"""
disturbance_panel.py — ONE manual panel for the pool DISTURBANCE RIG:
FMC4030 gantry + Damiao rotation motor + external T200 thruster.

    ./c3 disturb                       # real rig (whisker conda env, preflight first)
    ./c3 disturb --sim                 # simulated devices, no hardware
    ./c3 disturb --preflight-only      # just the readiness report
    ./c3 disturb --thruster-max 30

The jet is an external disturbance SOURCE for BlueROV2 control experiments.
(rov_gui's "disturbance" is the DOB/EAOB estimated wrench — a different thing.)

What this file is and is not
----------------------------
It is a thin Qt skin over whisker_vision's device-owning links
(``~/Desktop/whisker_vision/motion_control.py``: GantryLink, MotorLink,
ThrusterLink), imported BY PATH through ``disturbance_core`` and never modified.
Every rule that protects the hardware lives in those links: soft limits on
absolute moves and relative endpoints, one axis at a time, POS_VEL verification
before the motor moves, the thruster staircase, the latched STOP. The gantry
link is whisker's GantryLink plus four rig methods (disturbance_core
.RigGantryLink): per-axis limits set while connected, one long relative move per
hold-jog, the vendor's NON-latching jog stop on release, and a STOP sequence
that leads with that same stop (whisker's own did not stop a jog on this
controller — data/20261026/1026_131512_disturb/). This panel adds only what the
links cannot know: the enable matrix, the operator-side interlocks (thrust needs
a holding motor), the taught homes, the motor's zero-at-connect sequence, the
gantry patterns (Y-Z circle / Y back-and-forth, run by the rig link), the
watchdogs for a stop that does not take or a link that stays busy, and the
run-folder log.

The GUI thread never touches a device directly. It calls the links' queueing
methods and their cached getters; the device actions it starts itself are the
stops (``stop_all()``, ``jog_stop()``), which run in their own threads.

Frame: the FMC4030 counter in mm — 0 is wherever the axes stood when the
controller was powered up. There is no switch homing: "−X home" / "+X home"
make the CURRENT position that side's soft limit (teach-in), so jog to each end
and press its button. They hold until the controller is powered off (saved to
data/disturbance_rig_limits.json; 'Restore saved' only if it was not). NOT the
user frame of ./c3 gantry — numbers are not interchangeable.

Exclusive: libFMC4030 allows one connection per process and the controller one
TCP client. Never run this beside ./c3 gantry, whisker's motion_panel.py or
gantry_soft_limits.py.

Motor: Connect = Disable, Zero here, Enable — the shaft angle at connect is
0° and the motor holds it (thrust needs it holding). The angle therefore has no
meaning across sessions; the jet direction is whatever the mount points at.

Thruster: Set thrust / Turn off always ramp (thruster.ramp_step_pct every
ramp_interval_s of the rig YAML, 10 % / 3 s); there is no UI for the ramp.

STOP ALL (Esc): thruster -> 0 immediately, motor holds its angle, gantry stops
(a pattern too); everything stays latched until Reset. Turn off is not STOP: it
ramps down.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import disturbance_core as core  # noqa: E402
import disturbance_preflight as PF  # noqa: E402

try:
    WHISKER_ROOT = core.add_whisker_path(core.whisker_dir_from_argv())
    WHISKER_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 — reported by main(); tests skip
    WHISKER_ROOT = None
    WHISKER_IMPORT_ERROR = str(_exc)
    if str(core.REPO_ROOT) not in sys.path:
        sys.path.insert(1, str(core.REPO_ROOT))

from rov_gui import qt, theme  # noqa: E402
from rov_gui.qt import Qt, QTimer, QtCore, QtGui, QtWidgets  # noqa: E402
from rov_gui.state import Conn  # noqa: E402
from rov_gui.widgets.indicators import HoldButton, Panel, StatusPill  # noqa: E402

if WHISKER_IMPORT_ERROR is None:
    from motion.fmc4030 import Axis  # noqa: E402
    from motion_config import ANGLE_LIMIT_DEG, MAX_MOTOR_VEL_DEG_S  # noqa: E402
    from motion_control import AXES, pose_snapshot  # noqa: E402
else:  # pragma: no cover — only on a machine without whisker_vision
    Axis = None
    AXES = ()
    ANGLE_LIMIT_DEG = 360.0
    MAX_MOTOR_VEL_DEG_S = 180.0
    pose_snapshot = None

READOUT_MS = 100          # == the links' poll period (motion_control.py:45)
STATUS_MS = 250
STICKY_S = 3.0
#: hold-to-jog: closer than this to the limit end, a jog does not start
JOG_MIN_TRAVEL_MM = 0.5
#: a jog toward a taught home stops this short of it: the worker re-reads the
#: position under the SDK lock and refuses an endpoint 0.001 mm outside
LIMIT_EDGE_MM = 0.1
#: one jog toward a taught home is a single move; longer than any axis's travel,
#: so this only catches a nonsense distance
JOG_MAX_MOVE_MM = 2000.0
#: the two homes of an axis must be at least this far apart
MIN_SPAN_MM = 1.0
#: "at rest" = every axis moved less than REST_TOL_MM for REST_S, on positions no
#: older than REST_FRESH_S and with no link error (a failing poll freezes the
#: cache — it also sets the link error). The cache updates at 10 Hz.
REST_TOL_MM = 0.1
REST_S = 0.3
REST_FRESH_S = 0.5
#: a reconnect / Restore must find every axis within this of where the panel
#: last saw it, or the homes are not applied (a power cycle restarts the counter)
FRAME_TOL_MM = 1.0
#: one press moves at most this far in total (backstop for a lost release)
JOG_PRESS_MAX_MM = 2000.0
#: slowest jog/Send: below ~0.4 mm/s a run-on looks like rest to the watches
MIN_SPEED_MM_S = 1.0
#: a released axis must be at rest by RELEASE_GRACE_S + 2 x its deceleration
#: time, and within RELEASE_SLACK_MM + 2 x (deceleration path + 0.3 s of travel);
#: otherwise the jog stop did not take and STOP ALL follows
RELEASE_GRACE_S = 1.5
RELEASE_SLACK_MM = 5.0
#: ...and it must be SLOWING DOWN: from RELEASE_LAG_S + 0.4·v/a after the stop on,
#: its speed over the last SPEED_WINDOW_S must stay under the deceleration it
#: should be doing (+ 20 % of v + 2 mm/s). At 100 mm/s the slack alone (265 mm)
#: is longer than what is left of a piece, so an IGNORED stop (10-26) would end
#: the piece inside it and pass; still at cruise speed 1.2 s after the stop is
#: caught at ~120 mm [유도]. Lag = command + 10 Hz cache.
RELEASE_LAG_S = 0.4
SPEED_WINDOW_S = 0.3
#: STOP ALL (mode 1, then mode 2 = immediate) on an axis moving faster than
#: this: its taught homes become SUSPECT — an abrupt stop above anything this rig
#: has run may lose steps, and the counter has no encoder behind it
SUSPECT_ABOVE_MM_S = 50.0
#: after a release has settled the axis must stay put; watched this long for drift
JOG_WATCH_S = 5.0
TARGET_TOL_MM = 0.05      # an axis this close to its target is not re-commanded
STALL_TOL_MM = 1.0        # final position further than this from the target = stalled
MOTOR_SETTLE_TIMEOUT_S = 25.0     # whisker AIM_TIMEOUT_S
GANTRY_IDLE_BUSY_TIMEOUT_S = 60.0  # whisker SWEEP_BUSY_WAIT_S
#: STOP ALL re-sends the thruster zero: the worker checks the latch outside the
#: SDK lock and sends inside it (motion_control.py:1119-1129), so one non-zero
#: step can physically follow the first zero. 0.3 s covers the normal case,
#: 1.5 s the 1 s serial echo timeout.
STOP_RESEND_MS = (300, 1500)
GANTRY_STOP_SETTLE_TIMEOUT_S = 5.0   # a healthy stop settles in < 1 s; the link waits 3 s
INTERLOCK_GRACE_S = 2.0              # thrust on while the motor is no longer holding
POST_STOP_MOVE_TOL_MM = 2.0          # cumulative change from the settled reference = moving
FAULT_LOG_PERIOD_S = 1.0
#: motor zero-at-connect: each step (Disable, Zero, Enable) must be confirmed by
#: the link within this; the first waits for the link to come up
MOTOR_AUTO_STEP_S = 5.0
MOTOR_AUTO_READY_S = 20.0
MOTOR_AUTO_ZERO_TOL_DEG = 1.0
#: dwell after each Disable (no ack on CAN: the cache flips when the frame is SENT)
#: and after the Enable, during which the angle must stay within the tolerance
MOTOR_AUTO_DWELL_S = 0.5
#: pattern: no new vertex for max(this, 3 x the slowest segment + 5 s) = stalled
PATTERN_STALL_MIN_S = 10.0
#: a live pattern-axis reading this far outside the pattern box = STOP ALL
PATTERN_ENVELOPE_SLACK_MM = 5.0
#: Send: a commanded axis reading this far outside start..target = STOP ALL (an
#: absolute move read as relative / in another frame runs on otherwise)
SEND_BOX_SLACK_MM = 5.0
_CONNECT_ERROR_PREFIXES = ("gantry connect:", "motor connect:", "thruster connect:")

#: Added to rov_gui.theme.STYLESHEET (the station's dark palette and button
#: language), only for what the station has no rule for.
EXTRA_QSS = f"""
QLabel#HeaderTitle {{
    color: {theme.TEXT}; font-size: 15px; font-weight: 700; letter-spacing: 2px;
}}
QLabel#AxisName {{
    font-family: {theme.MONO}; font-size: 15px; font-weight: 700; color: {theme.ACCENT};
}}
QLabel#StatusLine {{
    font-family: {theme.MONO}; font-size: 11px; color: {theme.TEXT_DIM};
    background: {theme.PANEL_HI}; border: 1px solid {theme.BORDER};
    border-radius: 4px; padding: 4px 8px;
}}
QLabel#StatusLine[level="warning"] {{ color: {theme.WARN}; border-color: {theme.WARN}; }}
QLabel#StatusLine[level="error"] {{
    color: {theme.FAIL}; border-color: {theme.FAIL}; font-weight: 700;
}}
QLabel#Caption[level="warning"] {{ color: {theme.WARN}; font-weight: 700; }}
QLabel#Limit {{ color: {theme.TEXT_FAINT}; font-family: {theme.MONO}; font-weight: 700; }}
QLabel#Limit[ok="true"] {{ color: {theme.OK}; }}
QPushButton#Jog {{
    font-family: {theme.MONO}; font-size: 13px; font-weight: 700;
    min-width: 64px; min-height: 30px;
}}
QPushButton#Danger {{ font-size: 14px; }}
"""


def _fmt_mm(v) -> str:
    return "--" if v is None else f"{v:.1f} mm"


def _repolish(widget) -> None:
    """Qt re-evaluates [property] selectors only after an explicit re-polish."""
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)


def _sign_txt(sign: int) -> str:
    return "+" if sign > 0 else "−"


class DisturbanceWindow(QtWidgets.QMainWindow):
    """The panel. ``builders`` lets tests inject link factories."""

    def __init__(self, settings, options: core.PanelOptions,
                 whisker_root: "Path | None" = None, preflight: "dict | None" = None,
                 builders: "dict | None" = None) -> None:
        super().__init__()
        self._settings = settings
        self._options = options
        self._whisker_root = Path(whisker_root or WHISKER_ROOT or ".")
        self._preflight = preflight
        self._build = {"gantry": core.make_gantry_link, "motor": core.make_motor_link,
                       "thruster": core.make_thruster_link}
        self._build.update(builders or {})        # tests override one link at a time
        self._gantry = self._motor = self._thruster = None
        self._was_ready = {"gantry": False, "motor": False, "thruster": False}
        self._latched = False
        self._cut_pending = False
        self._sticky_until = 0.0
        self._fault_count = 0
        self._last_fault_log = float("-inf")
        self._interlock_lost_since: "float | None" = None
        self._gantry_stop_deadline: "float | None" = None
        self._gantry_recover_pending = False
        self._recover_deadline = float("inf")
        self._motor_disconnect_after_disable = False
        # provenance ONCE, before any device exists (six git subprocesses)
        self._provenance = core.provenance(self._whisker_root)
        self._target_dirty: set = set()      # axes whose target field the operator edited
        self._seeding_targets = False
        self._post_stop_ref: "tuple | None" = None   # (monotonic, positions) after STOP
        self._post_stop_warned = False
        self._watch_ref_at: "float | None" = None     # when to take the drift reference
        # hold-to-jog
        self._jog: "dict | None" = None      # {axis, sign, btn, t0, speed, chunk_from, end}
        self._jog_release: "dict | None" = None   # a release stop settling
        self._jog_watch_until = 0.0          # drift watch after a release settled
        # taught homes = soft limits, counter mm: {"X": {"lo", "hi", "*_epoch_s"}}.
        # Kept across a reconnect (the counter survives it: the 10-26 run read
        # X −389.978 before and after one); lost with the controller's power.
        self._taught: dict = {}
        self._saved = core.load_taught_limits(options.limits_file)
        self._link_limits0: "tuple | None" = None   # the link's own limits at connect
        self._rest = {"pos": None, "since": None}    # per-axis physical rest tracker
        #: last fresh position, kept across a reconnect: the frame check for the homes
        self._last_fresh_pos: "tuple | None" = None
        self._suspect_axes: set = set()     # homes after a stop that did not take
        self._commanded_axes: set = set()   # moving on purpose: not drift-watched
        # gantry state
        self._limits_applied = False
        self._gantry_targets_initialized: set = set()
        self._move_targets: dict = {}
        self._move_deadline: "float | None" = None
        self._gantry_was_busy = False
        self._gantry_busy_since: "float | None" = None
        self._gantry_cmd_in_flight = False
        self._gantry_stuck = False
        self._pattern: "dict | None" = None   # running gantry pattern
        # motor state
        self._motor_auto: "dict | None" = None   # zero-at-connect sequence
        self._motor_zeroed = False                # 0° = this connection's start angle
        #: the last thruster link closed with an echoed 0 (see _jet_off_known)
        self._jet_off_at_close = False
        self._move_box: dict = {}                 # Send: axis -> (lo, hi) it must stay in
        self._move_speed = 0.0                    # Send speed actually commanded
        self._stop_fail_text = ""                 # a failed STOP ALL call, kept on screen
        self._stop_fail_until = 0.0
        self._motor_target_initialized = False
        self._disabled_this_session = False
        self._motor_goto_deadline: "float | None" = None
        self._motor_stuck = False
        # log
        self._log: "core.DisturbLog | None" = None
        self._thruster_rows: list = []
        self._last_link_errors: dict = {}
        self._build_ui()
        self._readout_timer = QTimer(self)
        self._readout_timer.timeout.connect(self._update_readout)
        self._readout_timer.start(READOUT_MS)     # also what lets Ctrl-C through
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(STATUS_MS)
        self._update_buttons()

    # ================================================================== UI
    def _build_ui(self) -> None:
        """Station look (rov_gui.theme + its Panel / StatusPill / HoldButton):
        header with device pills and STOP ALL, gantry on the left, motor /
        thruster / log on the right, one status line at the bottom."""
        sim = bool(self._settings.simulator)
        self.setWindowTitle("Disturbance rig — gantry + motor + T200" + ("  [SIM]" if sim else ""))
        self.setMinimumSize(1120, 660)
        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        # ---- header ----
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(10)
        title = QtWidgets.QLabel("DISTURBANCE RIG")
        title.setObjectName("HeaderTitle")
        head.addWidget(title)
        head.addSpacing(8)
        self.gantry_pill = StatusPill("GANTRY")
        self.motor_pill = StatusPill("MOTOR")
        # the thruster link is the Arduino on USB: it answers with the ESC battery
        # unplugged, so the pill must not read as "thruster powered"
        self.thruster_pill = StatusPill("ARDUINO")
        self.log_pill = StatusPill("LOG")
        for pill in (self.gantry_pill, self.motor_pill, self.thruster_pill, self.log_pill):
            head.addWidget(pill)
        head.addStretch(1)
        self.sim_badge = QtWidgets.QLabel("SIMULATED DEVICES" if sim else "")
        self.sim_badge.setObjectName("Banner")
        self.sim_badge.setVisible(sim)
        head.addWidget(self.sim_badge)
        self.stop_btn = QtWidgets.QPushButton("STOP ALL  (Esc)")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setMinimumSize(210, 44)
        # ONE binding only: an extra application-level QShortcut on the same key
        # makes Esc ambiguous and Qt then fires neither (judge probe 2026-10-22).
        self.stop_btn.setShortcut(Qt.Key_Escape)
        self.stop_btn.clicked.connect(self._stop_all)
        head.addWidget(self.stop_btn)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setMinimumSize(80, 44)
        self.reset_btn.clicked.connect(self._reset_latch)
        head.addWidget(self.reset_btn)
        outer.addLayout(head)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(left, 3)
        body.addLayout(right, 2)
        outer.addLayout(body, 1)

        def caption(text):
            lab = QtWidgets.QLabel(text)
            lab.setObjectName("Caption")
            return lab

        # ---- GANTRY ----
        gp = Panel("Gantry  ·  FMC4030  ·  controller counter, mm")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("IP"))
        self.ip_edit = QtWidgets.QLineEdit(self._settings.gantry_ip)
        self.ip_edit.setFixedWidth(130)
        row.addWidget(self.ip_edit)
        self.gantry_btn = QtWidgets.QPushButton("Connect gantry")
        self.gantry_btn.clicked.connect(self._toggle_gantry)
        row.addWidget(self.gantry_btn)
        row.addStretch(1)
        gp.body.addLayout(row)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for col, text in enumerate(("", "position", "jog — hold", "", "target")):
            grid.addWidget(caption(text.upper()), 0, col)
        self._live_labels: dict = {}
        self._target_spins: dict = {}
        self._jog_buttons: dict = {}         # (axis, +1|-1) -> HoldButton
        for r, axis in enumerate(AXES, start=1):
            name = QtWidgets.QLabel(axis.name)
            name.setObjectName("AxisName")
            grid.addWidget(name, r, 0)
            live = QtWidgets.QLabel("--")
            live.setObjectName("BigValue")
            live.setMinimumWidth(130)
            live.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(live, r, 1)
            self._live_labels[axis] = live
            for col, sign in ((2, -1), (3, +1)):
                btn = HoldButton(f"{_sign_txt(sign)}{axis.name}",
                                 f"Hold to move {axis.name} toward {_sign_txt(sign)} "
                                 "continuously; release to stop. With a "
                                 f"{_sign_txt(sign)}{axis.name} home set it stops there; "
                                 "without one there is NO software limit on that side — "
                                 "watch the axis.")
                btn.setObjectName("Jog")
                btn.held.connect(lambda down, a=axis, sg=sign: self._on_jog_held(a, sg, down))
                grid.addWidget(btn, r, col)
                self._jog_buttons[(axis, sign)] = btn
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)    # narrowed to the soft limits
            spin.setDecimals(1)
            spin.setSuffix(" mm")
            spin.setToolTip("Send targets moves ONLY the axes you edited since the gantry "
                            "last settled; the others follow the live position.")
            spin.valueChanged.connect(lambda _v, a=axis: self._on_target_edited(a))
            grid.addWidget(spin, r, 4)
            self._target_spins[axis] = spin
        gp.body.addLayout(grid)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("SPEED"))
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(MIN_SPEED_MM_S, core.RIG_MAX_SPEED_MM_S)
        self.speed_spin.setValue(self._settings.speed_mm_s)
        self.speed_spin.setSuffix(" mm/s")
        # run-on after a release: 5.5-27.2 mm at 50 mm/s
        # (data/20261026/1026_140909_disturb/events.csv, jog_stopped); ~4x that
        # at 100 mm/s [유도: v²/2a]
        self.speed_spin.setToolTip(f"Jog and Send speed, 1..{core.RIG_MAX_SPEED_MM_S:g} mm/s. A "
                                   "release decelerates: up to ~3 cm of run-on at 50 mm/s, "
                                   "~10 cm at 100 mm/s — lower it to position.")
        row.addWidget(self.speed_spin)
        row.addStretch(1)
        self.send_btn = QtWidgets.QPushButton("Send targets")
        self.send_btn.setToolTip("Absolute move of the axes whose target you EDITED since the "
                                 "last settle, one axis at a time (X->Y->Z). Needs both homes "
                                 "of each such axis: targets stay inside them.")
        self.send_btn.clicked.connect(self._send_targets)
        row.addWidget(self.send_btn)
        self.recover_gantry_btn = QtWidgets.QPushButton("Recover gantry")
        self.recover_gantry_btn.setToolTip("The link stayed busy (stalled move): stop the axes, "
                                           "clear the link's latch")
        self.recover_gantry_btn.clicked.connect(self._recover_gantry)
        row.addWidget(self.recover_gantry_btn)
        gp.body.addLayout(row)

        # ---- home = soft limit (teach-in) ----
        gp.body.addSpacing(4)
        gp.body.addWidget(caption("HOME = SOFT LIMIT  ·  jog there, then press"))
        tgrid = QtWidgets.QGridLayout()
        tgrid.setHorizontalSpacing(8)
        tgrid.setVerticalSpacing(6)
        for col, text in enumerate(("", "− home", "", "", "+ home", "")):
            tgrid.addWidget(caption(text.upper()), 0, col)
        self._limit_labels: dict = {}        # (axis, "lo"|"hi") -> QLabel
        self._teach_buttons: dict = {}       # (axis, "lo"|"hi") -> QPushButton
        self._clear_buttons: dict = {}       # axis -> QPushButton
        for r, axis in enumerate(AXES, start=1):
            name = QtWidgets.QLabel(axis.name)
            name.setObjectName("AxisName")
            tgrid.addWidget(name, r, 0)
            for side, lab_col, btn_col in (("lo", 1, 2), ("hi", 4, 3)):
                sign_t = "−" if side == "lo" else "+"
                lab = QtWidgets.QLabel("--")
                lab.setObjectName("Limit")
                lab.setMinimumWidth(110)
                lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter if side == "lo"
                                 else Qt.AlignLeft | Qt.AlignVCenter)
                tgrid.addWidget(lab, r, lab_col)
                self._limit_labels[(axis, side)] = lab
                btn = QtWidgets.QPushButton(f"{sign_t}{axis.name} home")
                btn.setToolTip(f"Make the CURRENT {axis.name} position the {sign_t} soft limit: "
                               f"{axis.name} can then never be jogged or sent "
                               f"{'below' if side == 'lo' else 'above'} it. Pressing again "
                               "moves it to the new spot.")
                btn.clicked.connect(lambda _=False, a=axis, sd=side: self._teach(a, sd))
                tgrid.addWidget(btn, r, btn_col)
                self._teach_buttons[(axis, side)] = btn
            clear = QtWidgets.QPushButton("Clear")
            clear.setToolTip(f"Forget both {axis.name} homes: no software limit on {axis.name} "
                             "until they are set again")
            clear.clicked.connect(lambda _=False, a=axis: self._clear_limits(a))
            tgrid.addWidget(clear, r, 5)
            self._clear_buttons[axis] = clear
        gp.body.addLayout(tgrid)
        row = QtWidgets.QHBoxLayout()
        self.restore_btn = QtWidgets.QPushButton("Restore saved")
        self.restore_btn.setToolTip("Re-use the homes saved by an earlier session — ONLY if the "
                                    "gantry controller has not been powered off since (its "
                                    "counter restarts at 0 on power-up).")
        self.restore_btn.clicked.connect(self._restore_saved)
        row.addWidget(self.restore_btn)
        self.saved_label = caption(self._saved_text())
        self.saved_label.setWordWrap(True)
        row.addWidget(self.saved_label, 1)
        gp.body.addLayout(row)
        self.limits_label = caption("soft limits: --")
        self.limits_label.setWordWrap(True)
        gp.body.addWidget(self.limits_label)
        gp.body.addStretch(1)
        left.addWidget(gp, 1)

        # ---- PATTERN ----
        pp = Panel("Pattern  ·  X stays  ·  centred where the gantry stands")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("MODE"))
        self.pattern_combo = QtWidgets.QComboBox()
        self.pattern_combo.addItem("Circle  Y-Z", "circle")
        self.pattern_combo.addItem("Back-and-forth  Y", "line_y")
        self.pattern_combo.currentIndexChanged.connect(self._on_pattern_mode)
        row.addWidget(self.pattern_combo)
        self.pattern_size_caption = caption("RADIUS")
        row.addWidget(self.pattern_size_caption)
        self.pattern_size_spin = QtWidgets.QDoubleSpinBox()
        self.pattern_size_spin.setDecimals(0)
        self.pattern_size_spin.setSuffix(" mm")
        row.addWidget(self.pattern_size_spin)
        row.addWidget(caption("SPEED"))
        self.pattern_speed_spin = QtWidgets.QDoubleSpinBox()
        self.pattern_speed_spin.setRange(*core.PATTERN_SPEED_RANGE)
        self.pattern_speed_spin.setDecimals(1)
        self.pattern_speed_spin.setValue(self._options.pattern_speed_mm_s)
        self.pattern_speed_spin.setSuffix(" mm/s")
        row.addWidget(self.pattern_speed_spin)
        row.addStretch(1)
        self.pattern_start_btn = QtWidgets.QPushButton("Start")
        self.pattern_start_btn.setToolTip("Runs lap after lap until Stop / STOP ALL. Every "
                                          "target stays inside the box start ± size.")
        self.pattern_start_btn.clicked.connect(self._start_pattern)
        row.addWidget(self.pattern_start_btn)
        self.pattern_stop_btn = QtWidgets.QPushButton("Stop")
        self.pattern_stop_btn.clicked.connect(lambda: self._end_pattern("stopped"))
        row.addWidget(self.pattern_stop_btn)
        pp.body.addLayout(row)
        self.pattern_info = caption("")
        self.pattern_info.setWordWrap(True)
        self.pattern_info.setVisible(False)
        pp.body.addWidget(self.pattern_info)
        left.addWidget(pp)
        self._on_pattern_mode()

        # ---- MOTOR ----
        mp = Panel("Rotation motor  ·  Damiao  ·  CAN")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("CAN"))
        self.can_edit = QtWidgets.QLineEdit(self._settings.can_channel)
        self.can_edit.setFixedWidth(70)
        row.addWidget(self.can_edit)
        self.motor_btn = QtWidgets.QPushButton("Connect motor")
        self.motor_btn.clicked.connect(self._toggle_motor)
        row.addWidget(self.motor_btn)
        row.addStretch(1)
        self.motor_live = QtWidgets.QLabel("--")
        self.motor_live.setObjectName("BigValue")
        row.addWidget(self.motor_live)
        mp.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("ANGLE"))
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSuffix(" °")
        row.addWidget(self.angle_spin)
        row.addWidget(caption("SPEED"))
        self.vel_spin = QtWidgets.QDoubleSpinBox()
        self.vel_spin.setRange(0.5, min(MAX_MOTOR_VEL_DEG_S,
                                        float(self._options.motor_vel_max_deg_s)))
        self.vel_spin.setValue(self._settings.motor_vel_deg_s)
        self.vel_spin.setSuffix(" °/s")
        self.vel_spin.setToolTip("STOP ALL holds at the CACHED angle (0.1-0.3 s old): "
                                 "a fast move stopped mid-way swings back by speed x lag")
        row.addWidget(self.vel_spin)
        self.go_theta_btn = QtWidgets.QPushButton("Go θ")
        self.go_theta_btn.clicked.connect(self._go_theta)
        row.addWidget(self.go_theta_btn)
        row.addStretch(1)
        mp.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.enable_btn = QtWidgets.QPushButton("Enable")
        self.enable_btn.setToolTip("Connect already zeroes the motor where it stands and "
                                   "enables it. Enable/Disable by hand only at thrust 0 "
                                   "(a limp mount swings under the jet).")
        self.enable_btn.clicked.connect(self._toggle_enable)
        row.addWidget(self.enable_btn)
        self.recover_motor_btn = QtWidgets.QPushButton("Recover motor")
        self.recover_motor_btn.setToolTip("Go θ never settled (busy past the timeout): hold "
                                          "here and clear the link's latch")
        self.recover_motor_btn.clicked.connect(self._recover_motor)
        row.addWidget(self.recover_motor_btn)
        row.addStretch(1)
        mp.body.addLayout(row)
        self.motor_info = caption("")
        self.motor_info.setWordWrap(True)
        mp.body.addWidget(self.motor_info)
        right.addWidget(mp)

        # ---- THRUSTER ----
        tp = Panel("Thruster  ·  T200")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(caption("PORT"))
        self.thruster_edit = QtWidgets.QLineEdit(self._settings.thruster_port)
        self.thruster_edit.setMinimumWidth(160)
        row.addWidget(self.thruster_edit, 1)
        self.thruster_btn = QtWidgets.QPushButton("Connect thruster")
        self.thruster_btn.setToolTip("Opens the Arduino's USB serial port (the board resets and "
                                     "outputs 1500 µs = stop). It does not need, and cannot see, "
                                     "the ESC battery.")
        self.thruster_btn.clicked.connect(self._toggle_thruster)
        row.addWidget(self.thruster_btn)
        tp.body.addLayout(row)
        self.thruster_live = QtWidgets.QLabel("--")
        self.thruster_live.setObjectName("BigValue")
        self.thruster_live.setWordWrap(True)
        tp.body.addWidget(self.thruster_live)
        row = QtWidgets.QHBoxLayout()
        self.thruster_spin = QtWidgets.QSpinBox()
        self.thruster_spin.setRange(-self._settings.thruster_max_pct,
                                    self._settings.thruster_max_pct)
        self.thruster_spin.setSingleStep(5)
        self.thruster_spin.setSuffix(" %")
        row.addWidget(self.thruster_spin)
        self.thruster_set_btn = QtWidgets.QPushButton("Set thrust")
        self.thruster_set_btn.setToolTip(self._ramp_text())
        self.thruster_set_btn.clicked.connect(self._set_thruster)
        row.addWidget(self.thruster_set_btn)
        self.thruster_zero_btn = QtWidgets.QPushButton("Turn off")
        self.thruster_zero_btn.setToolTip("Ramps down to 0 (" + self._ramp_text() + "). "
                                          "Immediate 0: STOP ALL (Esc).")
        self.thruster_zero_btn.clicked.connect(self._zero_thruster)
        row.addWidget(self.thruster_zero_btn)
        row.addStretch(1)
        tp.body.addLayout(row)
        self.thruster_info = caption("")
        self.thruster_info.setWordWrap(True)
        self.thruster_info.setVisible(False)
        tp.body.addWidget(self.thruster_info)
        right.addWidget(tp)

        # ---- LOG ----
        lp = Panel("Log")
        row = QtWidgets.QHBoxLayout()
        self.log_label = QtWidgets.QLabel("not logging")
        self.log_label.setObjectName("Value")
        self.log_label.setWordWrap(True)
        row.addWidget(self.log_label, 1)
        self.log_new_btn = QtWidgets.QPushButton("● New log")
        self.log_new_btn.clicked.connect(self._new_log)
        row.addWidget(self.log_new_btn)
        self.log_stop_btn = QtWidgets.QPushButton("■ Stop log")
        self.log_stop_btn.clicked.connect(self._stop_log)
        row.addWidget(self.log_stop_btn)
        lp.body.addLayout(row)
        right.addWidget(lp)
        right.addStretch(1)

        # ---- status line ----
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("StatusLine")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("level", "info")
        outer.addWidget(self.status_label)
        self.setCentralWidget(root)

    # ========================================================= helpers
    def _say(self, text: str, severity: str = "info", sticky_s: "float | None" = None) -> None:
        if (severity == "error" and time.monotonic() < self._stop_fail_until
                and "a stop call FAILED" not in text):
            # a caller's reason after STOP ALL must not hide that a stop call failed
            text += f"  |  STOP ALL: a stop call FAILED — {self._stop_fail_text} — cut that " \
                    "device's power"
        if sticky_s is not None:
            self._sticky_until = time.monotonic() + float(sticky_s)
        elif severity in ("warning", "error"):
            self._sticky_until = time.monotonic() + STICKY_S
        else:
            self._sticky_until = time.monotonic() + 1.5    # confirmations stay readable
        self._render_status(text, severity)
        if severity in ("warning", "error"):
            self._log_event("status", f"{severity}: {text}")

    def _render_status(self, text: str, severity: str) -> None:
        if severity not in ("info", "warning", "error"):
            raise KeyError(severity)
        self.status_label.setText(text)
        if self.status_label.property("level") != severity:
            self.status_label.setProperty("level", severity)
            _repolish(self.status_label)

    def _ramp_text(self) -> str:
        return (f"ramps {self._settings.thruster_ramp_step_pct} % every "
                f"{self._settings.thruster_ramp_interval_s:g} s")

    def _thrust_is_zero(self) -> bool:
        """Fail CLOSED: a thruster link with an error has an unknown physical
        throttle (the firmware holds the last value it received), so it is
        never 'zero'. Only an absent link or a cached 0 with no ramp counts."""
        t = self._thruster
        if t is None:
            return True
        if t.error is not None:
            return False
        # a queued ramp (pending, not yet dequeued) counts as a ramp
        return t.throttle_pct() == 0 and t.ramp_target() is None and not t.busy

    def _jet_off_known(self) -> bool:
        """The JET is known to be off: a ready thruster link at 0 (opening the
        port reset the Arduino to 1500 µs), or no link now but the last one
        closed with an echoed 0. Otherwise a panel that died may have left it
        running (the firmware has no host-loss failsafe) — the mount must not
        be made limp then, by the zero sequence or by hand."""
        t = self._thruster
        if t is not None:
            return t.is_ready and self._thrust_is_zero()
        return self._jet_off_at_close

    def _motor_holding(self) -> bool:
        """Enabled AND still talking: a drive whose supply dropped keeps the
        cached enabled flag (the link never re-reads it) but stops answering."""
        m = self._motor
        return (m is not None and m.is_ready and m.is_enabled and m.telemetry_fresh())

    def _taught_pair(self, axis) -> "tuple[float | None, float | None]":
        t = self._taught.get(axis.name) or {}
        return t.get("lo"), t.get("hi")

    def _effective_limits(self, axis) -> "tuple[float | None, float | None]":
        """The taught homes intersected with whatever limits the link had at
        connect (valid firmware limits — the sim has them; this rig's
        controller does not). Either side may be None = no software limit."""
        lo, hi = self._taught_pair(axis)
        if self._link_limits0 is not None:
            l_lo = self._link_limits0[0][int(axis)]
            l_hi = self._link_limits0[1][int(axis)]
            if l_lo is not None:
                lo = l_lo if lo is None else max(lo, l_lo)
            if l_hi is not None:
                hi = l_hi if hi is None else min(hi, l_hi)
        return lo, hi

    def _endpoint_refused(self, axis, endpoint_mm: float) -> "str | None":
        """Why a target/endpoint must not be sent: a home or a firmware stop."""
        lo, hi = self._effective_limits(axis)
        if lo is not None and endpoint_mm < lo - 1e-6:
            return (f"{axis.name} {endpoint_mm:.1f} mm is below its − home {lo:.1f} mm "
                    "(soft limit)")
        if hi is not None and endpoint_mm > hi + 1e-6:
            return (f"{axis.name} {endpoint_mm:.1f} mm is above its + home {hi:.1f} mm "
                    "(soft limit)")
        gantry = self._gantry
        stop = gantry.firmware_stops_mm().get(axis.name) if gantry is not None else None
        if stop is not None and endpoint_mm > stop + 1e-6:
            return (f"{axis.name} {endpoint_mm:.1f} mm is beyond the firmware stop at "
                    f"{stop:.0f} mm — the axis would stall there silently; rewrite the limit "
                    "first (whisker_vision/gantry_soft_limits.py)")
        return None

    def _thrust_allowed(self) -> bool:
        return self._thrust_block_reason() is None

    def _thrust_block_reason(self) -> "str | None":
        """Why Set thrust is locked (None = allowed), worded for the operator."""
        t = self._thruster
        if t is None or not t.is_ready:
            return "Set thrust locked: thruster (Arduino) not connected"
        if self._latched:
            return "Set thrust locked: STOP ALL is latched — press Reset"
        if self._cut_pending:
            return "Set thrust locked: Cut in progress"
        if self._motor_auto is not None and self._motor is not None:
            return "Set thrust locked: the rotation motor is still zeroing"
        if not self._options.motor_interlock or self._motor_holding():
            return None
        m = self._motor
        if m is None:
            return ("Set thrust locked: connect the ROTATION MOTOR, then press Enable "
                    "(the mount must hold against the jet)")
        if not m.is_ready:
            return "Set thrust locked: rotation motor still connecting"
        if not m.is_enabled:
            return ("Set thrust locked: press Enable on the ROTATION MOTOR "
                    "(connected is not enough — it must hold the angle)")
        return "Set thrust locked: motor telemetry stale — check the motor supply / CAN"

    def _axis_absolute_ok(self, axis) -> bool:
        """Absolute moves (Send) need both ends of the axis bounded."""
        lo, hi = self._effective_limits(axis)
        return self._limits_applied and lo is not None and hi is not None

    def _absolute_gantry_ready(self) -> bool:
        return (self._gantry is not None and self._gantry.is_ready and self._limits_applied
                and len(self._gantry_targets_initialized) == len(AXES))

    def _gantry_idle(self) -> bool:
        """Nothing commanded or settling: no jog, no release, no move in flight."""
        g = self._gantry
        return (g is not None and not g.busy and self._jog is None and self._pattern is None
                and self._jog_release is None and not self._gantry_cmd_in_flight)

    def _all_at_rest(self, now: "float | None" = None) -> bool:
        """PHYSICALLY still: every axis within REST_TOL_MM for REST_S on fresh,
        error-free telemetry. Every new gantry command and Reset need it — a
        stop that did not take leaves a move running that 'busy' never shows."""
        return self._axes_at_rest(AXES, now)

    def _axes_at_rest(self, axes, now: "float | None" = None) -> bool:
        """Those axes physically still (same rule, per axis): a jog on one axis
        may start while another, just released, is still decelerating — its
        own release watch covers it."""
        r = self._rest
        pos = self._fresh_pos()
        if pos is None or r["pos"] is None:
            return False
        now = time.monotonic() if now is None else now
        for axis in axes:
            i = int(axis)
            if abs(pos[i] - r["pos"][i]) > REST_TOL_MM:
                return False             # the cache moved since the last tick saw it
            if now - r["since"][i] < REST_S:
                return False
        return True

    def _fresh_pos(self) -> "tuple | None":
        """The cached position, only if fresh and the link reports no error."""
        g = self._gantry
        if g is None or g.error is not None or not g.telemetry_fresh(REST_FRESH_S):
            return None
        pos = g.positions_mm()
        return tuple(pos) if pos is not None else None

    def _track_rest(self, pos, now: float) -> None:
        r = self._rest
        if pos is None:
            r["pos"] = r["since"] = None
            return
        if r["pos"] is None:
            r["pos"], r["since"] = list(pos), [now] * 3
        for i in range(3):
            if abs(pos[i] - r["pos"][i]) > REST_TOL_MM:
                r["pos"][i], r["since"][i] = pos[i], now
        if self._limits_applied:
            # only AFTER this connection's frame check: before it, a new reading
            # (counter restarted by a power cycle) would overwrite the very
            # reference the check compares against
            self._last_fresh_pos = tuple(pos)

    @staticmethod
    def _frame_mismatch(pos, ref) -> "str | None":
        """Why a home set recorded at ``ref`` must not be trusted at ``pos``."""
        if ref is None:
            return "no position was recorded with these homes"
        if pos is None:
            return "no live position"
        moved = [(name, ref[i], pos[i]) for i, name in enumerate(core.AXIS_NAMES)
                 if abs(pos[i] - ref[i]) > FRAME_TOL_MM]
        if moved:
            return "the gantry reads " + ", ".join(
                f"{n} {b:.1f} mm (was {a:.1f})" for n, a, b in moved)
        return None

    def _restart_drift_watch(self, now: float) -> None:
        """Everything must stay where it is now for JOG_WATCH_S (axes moving on
        purpose excluded); the reference is taken on the next tick."""
        self._jog_watch_until = now + JOG_WATCH_S
        self._watch_ref_at = now
        self._post_stop_ref = None
        self._post_stop_warned = False

    def _watch_settle_s(self) -> float:
        """After STOP ALL, wait this long before taking the drift reference:
        0.5 s + the deceleration from the rig's top speed (2.0 s at 100 mm/s
        and 50 mm/s² [유도: v/a]) — earlier, a still-decelerating axis reads
        as a stop that did not take."""
        a = max(float(getattr(self._settings, "acc_mm_s2", 50.0)), 1.0)
        return 0.5 + core.RIG_MAX_SPEED_MM_S / a

    def _motion_allowed(self) -> bool:
        return not self._latched

    # ------------------------------------------------------ target fields
    def _on_target_edited(self, axis) -> None:
        if not self._seeding_targets:
            self._target_dirty.add(axis)

    def _seed_targets(self, positions, only_clean: bool = True) -> None:
        """Target fields follow the live position; an operator edit sticks
        until the gantry settles after a Send."""
        if positions is None:
            return
        self._seeding_targets = True
        try:
            for axis in AXES:
                if only_clean and axis in self._target_dirty:
                    continue
                self._target_spins[axis].setValue(positions[int(axis)])
        finally:
            self._seeding_targets = False

    # ======================================================= connections
    def _toggle_gantry(self) -> None:
        if self._gantry is not None:
            if not self._latched and (self._pattern is not None or self._jog is not None
                                      or self._jog_release is not None or self._gantry.busy):
                self._say("Stop the gantry first (pattern / jog / move) — a disconnect would "
                          "drop the watch on a stop it cannot verify.", "warning")
                return
            if not self._thrust_is_zero():
                self._say("Thrust 0 before disconnecting the gantry (the disconnect joins "
                          "for several seconds — longer on a dead link — during which Esc "
                          "is not processed).", "warning")
                return
            self._log_event("disconnect", "gantry")
            self._save_homes()                   # with the position it was last seen at
            self._gantry.stop()
            self._gantry = None
            self._reset_gantry_target_state()
            self._was_ready["gantry"] = False
            self.gantry_btn.setText("Connect gantry")
        else:
            self._settings.gantry_ip = self.ip_edit.text().strip()
            if not self._settings.simulator:
                held, note = core.gantry_link_held(self._settings.gantry_ip,
                                                   self._settings.gantry_port)
                if held:
                    self._say("Gantry link already held by another process — close "
                              "./c3 gantry / whisker motion_panel first.  " + note, "error")
                    self._log_event("connect_refused", f"gantry: {note}")
                    return
            self._reset_gantry_target_state()
            # no client limits: the link connects relative-only (this controller's
            # firmware pair is invalid) and the taught homes are installed per
            # axis once the link is ready (_apply_soft_limits)
            self._settings.client_soft_limits_mm = {}
            self._gantry = self._build["gantry"](self._settings)
            self._gantry.start()
            self.gantry_btn.setText("Disconnect gantry")
            self._say("Connecting to gantry…")
            self._log_event("connect", f"gantry {self._settings.gantry_ip}")
        self._update_buttons()

    def _toggle_motor(self) -> None:
        if self._motor is not None:
            if not self._thrust_is_zero():
                self._say("Thrust 0 before disconnecting the motor: the mount must keep "
                          "holding while the jet runs.", "warning")
                return
            if self._gantry is not None and self._gantry.busy:
                self._say("Wait for the gantry to stop before disconnecting the motor (the "
                          "confirmation dialog hides STOP ALL).", "warning")
                return
            if self._motor.is_enabled:
                box = QtWidgets.QMessageBox(self)
                box.setIcon(QtWidgets.QMessageBox.Question)
                box.setWindowTitle("Disconnect motor")
                may_limp = self._jet_off_known() or not self._options.motor_interlock
                box.setText("The motor is ENABLED. Disconnecting keeps it energised and "
                            "holding its angle (whisker contract).\n"
                            + ("Disable first?" if may_limp else
                               "Disable is not offered: connect the thruster first (its reset "
                               "makes the jet 0) — the mount must not go limp under a jet."))
                disable = (box.addButton("Disable first", QtWidgets.QMessageBox.AcceptRole)
                           if may_limp else None)
                box.addButton("Keep holding", QtWidgets.QMessageBox.DestructiveRole)
                cancel = box.addButton(QtWidgets.QMessageBox.Cancel)
                box.setDefaultButton(cancel)
                box.exec_()
                if box.clickedButton() is cancel:
                    return
                if disable is not None and box.clickedButton() is disable:
                    try:
                        self._motor.set_enabled(False)
                    except (ValueError, RuntimeError) as exc:
                        self._say(str(exc), "warning")
                        return
                    self._disabled_this_session = True
                    self._log_event("disable", "motor (before disconnect)")
                    # finish on the readout tick once the drive reports disabled —
                    # no nested event loop (a processEvents() wait re-enters slots)
                    self._motor_disconnect_after_disable = True
                    self._say("Disabling… the motor disconnects when the link confirms.")
                    self._update_buttons()
                    return
            self._disconnect_motor_now()
        else:
            self._settings.can_channel = self.can_edit.text().strip()
            self._motor_target_initialized = False
            self._disabled_this_session = False
            self._motor = self._build["motor"](self._settings)
            self._motor.start()
            self._motor_auto = {"stage": "ready", "t": time.monotonic()}
            self._motor_zeroed = False
            self.motor_btn.setText("Disconnect motor")
            self._say("Connecting to motor… once the thruster is connected too, it is zeroed "
                      "where it stands and holds 0°.")
            self._log_event("connect", f"motor {self._settings.can_channel}")
        self._update_buttons()

    def _disconnect_motor_now(self) -> None:
        if self._motor is None:
            return
        self._motor_disconnect_after_disable = False
        self._motor_auto = None
        self._motor_zeroed = False
        self._log_event("disconnect", "motor")
        self._motor.stop()
        self._motor = None
        self._motor_target_initialized = False
        self._motor_goto_deadline = None
        self._motor_stuck = False
        self._interlock_lost_since = None
        self._was_ready["motor"] = False
        self.motor_btn.setText("Connect motor")

    def _toggle_thruster(self) -> None:
        if self._thruster is not None:
            # zero FIRST while the link still records, so the 0 lands in
            # thruster.csv; stop() would send it after stop_recording()
            self._thruster.stop_all()
            self._jet_off_at_close = self._note_zero_outcome(self._thruster, "disconnect") is None
            self._log_event("disconnect", "thruster")
            if self._log is not None:
                try:
                    self._thruster_rows.extend(self._thruster.stop_recording())
                except Exception:  # noqa: BLE001
                    pass
            self._thruster.stop()        # zeroes again before closing the port
            self._thruster = None
            self._cut_pending = False
            self._was_ready["thruster"] = False
            self.thruster_btn.setText("Connect thruster")
        else:
            self._settings.thruster_port = self.thruster_edit.text().strip()
            self._thruster = self._build["thruster"](self._settings)
            self._thruster.start()
            self.thruster_btn.setText("Disconnect thruster")
            self._say("Connecting to thruster (Arduino resets, ESC arms ~3 s, starts at 0 %)…")
            self._log_event("connect", f"thruster {self._settings.thruster_port}")
        self._update_buttons()

    def _reap_failed_connects(self) -> None:
        """A link whose worker died at connect is stopped and cleared, so the
        operator sees 'Connect …' again instead of a dead 'Disconnect'."""
        for name in ("gantry", "motor", "thruster"):
            link = getattr(self, f"_{name}")
            if link is None:
                continue
            err = link.error or ""
            if not err.startswith(_CONNECT_ERROR_PREFIXES):
                continue
            try:
                link.stop()
            except Exception:  # noqa: BLE001
                pass
            setattr(self, f"_{name}", None)
            self._was_ready[name] = False
            getattr(self, f"{name}_btn").setText(f"Connect {name}")
            if name == "gantry":
                self._reset_gantry_target_state()
            if name == "thruster":
                self._cut_pending = False
            self._say(f"{name}: {err}", "error", sticky_s=10.0)
            self._log_event("connect_failed", f"{name}: {err}")
            self._update_buttons()

    # ============================================================ gantry
    def _send_targets(self) -> None:
        if not self._motion_allowed():
            return
        gantry = self._gantry
        if gantry is None or not gantry.is_ready:
            self._say("Connect the gantry first.", "warning")
            return
        if not self._gantry_idle() or not self._all_at_rest():
            self._say("Gantry not at rest yet (or no fresh position) — Send waits for it.",
                      "warning")
            return
        open_axes = [a.name for a in self._target_dirty if not self._axis_absolute_ok(a)]
        if open_axes:
            self._say(f"{', '.join(sorted(open_axes))}: set both homes (− and +) first — Send "
                      "moves only inside them.", "warning")
            return
        if not self._absolute_gantry_ready():
            self._say("Absolute move unavailable: wait for a position readout. Hold-jog "
                      "remains available.", "warning")
            return
        positions = gantry.positions_mm() or (None,) * 3
        targets = {}
        if not self._target_dirty:
            self._say("Edit a target field first — Send moves only the axes you changed "
                      "since the gantry last settled.", "warning")
            return
        for axis in AXES:
            if axis not in self._target_dirty:
                continue
            target = self._target_spins[axis].value()
            why = self._endpoint_refused(axis, target)
            if why:
                self._say(why, "warning")
                return
            current = positions[int(axis)]
            if current is None or abs(target - current) > TARGET_TOL_MM:
                targets[axis] = target
        if not targets:
            self._target_dirty.clear()
            self._seed_targets(positions, only_clean=False)
            self._say("Gantry already at the edited targets.")
            return
        # Send stays at <= 50 mm/s until one has run on this controller (the
        # link clamps too; this keeps the watches and the log on the real value)
        speed = min(float(self.speed_spin.value()), core.SEND_MAX_SPEED_MM_S,
                    min(core.axis_max_speed(a) for a in targets))
        try:
            gantry.request_move(targets, speed)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            self._log_event("command_rejected", f"move: {exc}")
            return
        self._move_targets = dict(targets)
        self._move_speed = speed
        self._move_box = {a: (min(positions[int(a)], t) - SEND_BOX_SLACK_MM,
                              max(positions[int(a)], t) + SEND_BOX_SLACK_MM)
                          for a, t in targets.items() if positions[int(a)] is not None}
        self._target_dirty.clear()
        self._arm_gantry_deadline({a: abs(t - (positions[int(a)] or 0.0))
                                   for a, t in targets.items()}, speed=speed)
        text = ", ".join(f"{a.name} {v:.1f}" for a, v in targets.items())
        capped = speed < float(self.speed_spin.value()) - 1e-9
        self._say(f"Gantry → {text} mm at {speed:g} mm/s"
                  + (" (Send ≤ 50 until verified on the rig)" if capped else ""), sticky_s=2.0)
        self._log_event("move", f"{text} mm @ {speed:g} mm/s")

    def _watch_send_box(self, positions) -> None:
        """A Send axis outside start..target (± SEND_BOX_SLACK_MM) = STOP ALL:
        the first absolute moves on this controller are the test of whether
        mode 2 really is absolute in this counter frame (KNOWN_ISSUES 10-26)."""
        for axis, (lo, hi) in self._move_box.items():
            v = positions[int(axis)]
            if lo <= v <= hi:
                continue
            self._move_box = {}
            self._log_event("stuck", f"send {axis.name} {v:.1f} mm outside {lo:.1f}..{hi:.1f}")
            self._stop_all()                     # first: its own message must not win
            self._say(f"Send: {axis.name} at {v:.1f} mm left its start..target range "
                      f"{lo:.1f}..{hi:.1f} mm — STOP ALL pressed (absolute move misread?). If it "
                      "keeps moving, CUT THE GANTRY POWER.", "error", sticky_s=20.0)
            return

    def _arm_gantry_deadline(self, distances: dict, speed: "float | None" = None) -> None:
        v = max(float(speed if speed is not None else self.speed_spin.value()), 0.1)
        total = sum(float(d) for d in distances.values())
        self._move_deadline = time.monotonic() + 3.0 * total / v + 10.0
        self._gantry_cmd_in_flight = True
        self._gantry_stuck = False
        # the drift watch goes on for the OTHER axes (a press must not blind it)
        self._commanded_axes |= set(distances)

    def _on_gantry_settled(self) -> None:
        """busy -> idle: close the in-flight command (stall check)."""
        gantry = self._gantry
        if gantry is None:
            return
        err = gantry.last_command_error
        self._seed_targets(gantry.positions_mm())      # clean fields follow the rig
        if self._move_targets:
            pos = gantry.positions_mm()
            if pos is not None and not err:
                short = [f"{a.name} {abs(pos[int(a)] - t):.1f} mm short"
                         for a, t in self._move_targets.items()
                         if abs(pos[int(a)] - t) > STALL_TOL_MM]
                if short:
                    self._say("Move ended " + ", ".join(short) + " — firmware stop or "
                              "stall? (no error from the link)", "warning")
                    self._log_event("stall", ", ".join(short))
            self._commanded_axes -= set(self._move_targets)
            self._move_targets = {}
            self._move_box = {}
            self._restart_drift_watch(time.monotonic())
            self._save_homes()
        self._move_deadline = None
        self._gantry_cmd_in_flight = False
        self._gantry_stuck = False

    def _recover_gantry(self) -> None:
        """Stop the axes (off the GUI thread) and reset the link once the stop
        sequence has finished — reset_stop() raises while it is still running,
        so the readout tick retries it."""
        gantry = self._gantry
        if gantry is None:
            return
        gantry.stop_all(wait=False)
        self._gantry_recover_pending = True
        self._recover_deadline = time.monotonic() + GANTRY_STOP_SETTLE_TIMEOUT_S
        self._move_targets = {}
        self._move_box = {}
        self._move_deadline = None
        self._gantry_cmd_in_flight = False
        self._gantry_busy_since = None
        self._say("Gantry stop requested; the link resets when the stop sequence ends.",
                  "warning")
        self._log_event("recover", "gantry")

    def _finish_gantry_recover(self) -> None:
        gantry = self._gantry
        if gantry is None:
            self._gantry_recover_pending = False
            return
        if self._latched:
            return                   # STOP ALL owns the latches now; Reset clears them
        if not self._all_at_rest():
            # not before the axes are physically still — but not silently forever:
            # a Recover whose stop did not take is the 10-26 failure again
            if time.monotonic() > self._recover_deadline:
                self._gantry_recover_pending = False
                self._suspect_axes |= set(self._taught)
                self._stop_all()
                self._say("Recover: the gantry is still not at rest 5 s after its stop — STOP "
                          "ALL pressed. If an axis keeps moving, CUT THE GANTRY POWER.",
                          "error", sticky_s=20.0)
                self._log_event("stuck", "gantry not at rest after Recover")
            return
        try:
            gantry.reset_stop()
        except RuntimeError:
            return                              # stop sequence still running: retry
        self._gantry_recover_pending = False
        self._gantry_stuck = False
        self._commanded_axes = set()
        self._restart_drift_watch(time.monotonic())
        self._say("Gantry link reset after a stalled move.", "warning")
        self._log_event("recover_done", "gantry")

    def _apply_soft_limits(self, gantry, first_pos=None) -> None:
        """First look after connect (``first_pos`` = the fresh reading the
        caller already checked): remember the link's own limits, install the
        taught homes if the gantry is where it was last seen; afterwards
        (teach / clear / restore) re-derive the labels, the target ranges and
        the limits line."""
        if self._link_limits0 is None:
            lo0, hi0 = gantry.soft_limits_mm()
            self._link_limits0 = (list(lo0), list(hi0))
            if self._taught:
                # homes from before a disconnect: only if the gantry is exactly where
                # it was last seen. A controller power cycle (the panel itself asks
                # for one when a stop does not take) restarts the counter at 0.
                why = self._frame_mismatch(first_pos if first_pos is not None
                                           else gantry.positions_mm(), self._last_fresh_pos)
                if why:
                    dropped, self._taught = self._taught, {}
                    self._suspect_axes = set()
                    self._say(f"Homes NOT re-applied: {why} — controller power-cycled? "
                              "Re-teach them.", "error", sticky_s=15.0)
                    self._log_event("homes_dropped", f"{why}; were {dropped}")
                else:
                    for axis in AXES:
                        if axis.name in self._taught:
                            self._install_limits(axis)
        stops = gantry.firmware_stops_mm()
        self._limits_applied = True
        self._seeding_targets = True      # a clamped value is not an operator edit
        try:
            for axis in AXES:
                e_lo, e_hi = self._effective_limits(axis)
                t_lo, t_hi = self._taught_pair(axis)
                for side, val, taught in (("lo", e_lo, t_lo), ("hi", e_hi, t_hi)):
                    lab = self._limit_labels[(axis, side)]
                    if val is None:
                        lab.setText("--  (none)")
                    else:
                        lab.setText(f"{val:.1f} mm" + ("" if taught is not None
                                                       and taught == val else "  (fw)"))
                    ok = "true" if val is not None else "false"
                    if lab.property("ok") != ok:
                        lab.setProperty("ok", ok)
                        _repolish(lab)
                spin = self._target_spins[axis]
                if e_lo is not None and e_hi is not None:
                    # the spin keeps 0.1 mm: round INSIDE the homes, never out
                    spin.setRange(math.ceil(e_lo * 10.0 - 1e-9) / 10.0,
                                  math.floor(e_hi * 10.0 + 1e-9) / 10.0)
                    spin.setToolTip("Send targets moves ONLY the axes you edited since the "
                                    "gantry last settled; inside the homes.")
                else:
                    spin.setRange(-10000.0, 10000.0)
                    spin.setToolTip(f"Set both {axis.name} homes (− and +) first: Send moves "
                                    "only inside them.")
        finally:
            self._seeding_targets = False
        parts = []
        for axis in AXES:
            e_lo, e_hi = self._effective_limits(axis)
            if e_lo is None and e_hi is None:
                parts.append(f"{axis.name} –")
            else:
                parts.append(f"{axis.name} {'–' if e_lo is None else f'{e_lo:.1f}'}.."
                             f"{'–' if e_hi is None else f'{e_hi:.1f}'}")
        text = "limits mm: " + ",  ".join(parts)
        if stops:
            text += "   ·   fw stop +: " + ", ".join(f"{k} {v:.0f}" for k, v in stops.items())
        self.limits_label.setText(text)
        self.saved_label.setText(self._saved_text())

    def _reset_gantry_target_state(self) -> None:
        """Connect / disconnect. The taught homes are KEPT (same counter frame
        across a reconnect) and re-installed into the next link."""
        self._limits_applied = False
        self._link_limits0 = None
        self._gantry_targets_initialized.clear()
        self._target_dirty.clear()
        self._post_stop_ref = None
        self._gantry_recover_pending = False
        self._move_targets = {}
        self._move_box = {}
        self._move_deadline = None
        self._gantry_was_busy = False
        self._gantry_busy_since = None
        self._gantry_cmd_in_flight = False
        self._gantry_stuck = False
        self._jog = None
        self._pattern = None
        self._jog_release = None
        self._jog_watch_until = 0.0
        self._watch_ref_at = None
        self._rest = {"pos": None, "since": None}
        self._commanded_axes = set()
        for spin in self._target_spins.values():
            spin.setRange(-10000.0, 10000.0)
            spin.setToolTip("Waiting for the gantry.")
        self.limits_label.setText("soft limits: --")

    # ------------------------------------------------------ taught homes
    def _saved_text(self) -> str:
        saved = self._saved
        if saved.get("error"):
            return f"saved homes unreadable: {saved['error']}"
        axes = saved.get("axes") or {}
        if not axes:
            return "no saved homes"
        when = saved.get("saved_epoch_s")
        stamp = time.strftime("%m-%d %H:%M", time.localtime(when)) if when else "?"
        bits = []
        for name in core.AXIS_NAMES:
            e = axes.get(name)
            if e:
                lo, hi = e.get("lo"), e.get("hi")
                bits.append(f"{name} {'--' if lo is None else f'{lo:.1f}'}.."
                            f"{'--' if hi is None else f'{hi:.1f}'}")
        return f"saved {stamp}: " + ", ".join(bits)

    def _install_limits(self, axis) -> None:
        """Hand the axis's limits to the link: the pair when both ends are
        known (Send, and the worker's endpoint re-check), else none — a lone
        home is enforced by the panel (jog end, Send refused)."""
        lo, hi = self._effective_limits(axis)
        if lo is None or hi is None:
            lo = hi = None
        setter = getattr(self._gantry, "set_axis_limits", None)
        if setter is None:          # a plain GantryLink: limits stay as connected
            self._say("gantry link cannot take per-axis limits — reconnect with the rig "
                      "link (disturbance_core.make_gantry_link)", "error")
            return
        setter(axis, lo, hi)

    def _save_homes(self, force: bool = False) -> None:
        """Persist the homes WITH the position the gantry stands at now — a
        later Restore must find it exactly there. An empty set is written only
        on an explicit Clear, so jogging never wipes a previous session's homes."""
        if not self._taught and not force:
            return
        pos = self._fresh_pos()
        if pos is not None and self._limits_applied:     # see _track_rest
            self._last_fresh_pos = pos
        err = core.save_taught_limits(self._options.limits_file, self._taught,
                                      self._last_fresh_pos)
        if err:
            self._say(f"homes not saved: {err}", "warning")
        self._saved = core.load_taught_limits(self._options.limits_file)
        self.saved_label.setText(self._saved_text())

    def _limits_changed(self, axis) -> None:
        if self._gantry is not None and self._limits_applied:
            self._install_limits(axis)
            self._apply_soft_limits(self._gantry)
        self._save_homes(force=True)
        self._update_buttons()

    def _teach(self, axis, side: str) -> None:
        """"−X home" / "+X home": the CURRENT position becomes that soft limit."""
        gantry = self._gantry
        if gantry is None or not gantry.is_ready or side not in ("lo", "hi"):
            return
        if self._latched:
            self._say("Reset the STOP latch before setting a home.", "warning")
            return
        if not (self._gantry_idle() and self._all_at_rest()):
            self._say("Set a home with the gantry at rest (release the jog, let a move end).",
                      "warning")
            return
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            self._say("Setting a home needs a live gantry position.", "warning")
            return
        value = float(pos[int(axis)])
        label = f"{'−' if side == 'lo' else '+'}{axis.name} home"
        e_lo, e_hi = self._effective_limits(axis)
        if side == "lo" and e_hi is not None and value > e_hi - MIN_SPAN_MM:
            self._say(f"{label} at {value:.1f} mm must be below the +{axis.name} limit "
                      f"{e_hi:.1f} mm — move to the − end first, or Clear {axis.name}.", "warning")
            return
        if side == "hi" and e_lo is not None and value < e_lo + MIN_SPAN_MM:
            self._say(f"{label} at {value:.1f} mm must be above the −{axis.name} limit "
                      f"{e_lo:.1f} mm — move to the + end first, or Clear {axis.name}.", "warning")
            return
        entry = self._taught.setdefault(axis.name, {})
        old = entry.get(side)
        self._suspect_axes.discard(axis.name)
        entry[side] = value
        entry[f"{side}_epoch_s"] = time.time()
        self._limits_changed(axis)
        lo, hi = self._effective_limits(axis)
        rng = (f"{'--' if lo is None else f'{lo:.1f}'} .. {'--' if hi is None else f'{hi:.1f}'}")
        self._say(f"{label} set at {value:.1f} mm — {axis.name} soft limits now {rng} mm.",
                  sticky_s=3.0)
        self._log_event("teach", f"{label} = {value:.3f} mm"
                        + ("" if old is None else f" (was {old:.3f})") + f"; {axis.name} {rng}")

    def _clear_limits(self, axis) -> None:
        if axis.name not in self._taught:
            return
        if not (self._gantry_idle() and self._limits_applied):
            self._say("Clear a home with the gantry connected and at rest.", "warning")
            return
        old = self._taught.pop(axis.name)
        self._suspect_axes.discard(axis.name)
        self._limits_changed(axis)
        self._say(f"{axis.name} homes cleared — NO software limit on {axis.name} until both "
                  "are set again.", "warning")
        self._log_event("teach_clear", f"{axis.name} (was {old.get('lo')}..{old.get('hi')})")

    def _restore_saved(self) -> None:
        """Re-use the saved homes only where they were saved: the gantry must
        read exactly the position stored with them (FRAME_TOL_MM per axis),
        which a controller power cycle (counter back to 0) or any move since
        breaks. Nothing is installed on a mismatch."""
        axes = core.clean_taught_limits(self._saved.get("axes"))
        gantry = self._gantry
        if not axes or self._taught or gantry is None or not self._limits_applied:
            return
        if self._latched or not (self._gantry_idle() and self._all_at_rest()):
            self._say("Restore with the gantry at rest and the STOP latch reset.", "warning")
            return
        why = self._frame_mismatch(gantry.positions_mm(), self._saved.get("last_pos_mm"))
        if why:
            self._say(f"Saved homes NOT restored: {why} — the controller was powered off or "
                      "the gantry moved since they were saved. Re-teach them.", "error",
                      sticky_s=12.0)
            self._log_event("teach_restore_refused", f"{why}; {self._saved_text()}")
            return
        self._taught = {k: dict(v) for k, v in axes.items()}
        self._suspect_axes = set()
        for axis in AXES:
            self._install_limits(axis)
        self._apply_soft_limits(gantry)
        self._log_event("teach_restore", self._saved_text())
        self._say("Saved homes restored — the gantry reads exactly where they were saved. "
                  "If anyone moved it by hand while it was off, Clear and re-teach.", "warning",
                  sticky_s=8.0)
        self._update_buttons()

    # ------------------------------------------------------------- hold-jog
    def _on_jog_held(self, axis, sign: int, down: bool) -> None:
        if down:
            self._start_jog(axis, sign)
        elif self._jog is not None and self._jog["axis"] == axis and self._jog["sign"] == sign:
            self._end_jog("release")

    def _jog_bounds(self, axis, sign: int) -> "tuple[float | None, float | None]":
        """(home, firmware stop) on the pressed side. Only a HOME (taught, or a
        valid link limit) gets the one-move-to-the-end jog. A firmware stop
        (+ side, counter frame — no fixed physical place) only caps a piece."""
        lo, hi = self._effective_limits(axis)
        if sign < 0:
            return lo, None
        gantry = self._gantry
        stop = gantry.firmware_stops_mm().get(axis.name) if gantry is not None else None
        return hi, stop

    def _jog_end_mm(self, axis, sign: int) -> "float | None":
        """The furthest a jog toward ``sign`` may ever end (home ∩ firmware stop)."""
        home, stop = self._jog_bounds(axis, sign)
        ends = [v for v in (home, stop) if v is not None]
        return (min(ends) if sign > 0 else max(ends)) if ends else None

    def _start_jog(self, axis, sign: int) -> None:
        """Press: ONE continuous controller-relative move toward the pressed
        side, stopped on release. Toward a home it ends AT the home; with no
        home on that side it is at most jog_piece_mm and the next follows
        while the button is held."""
        gantry = self._gantry
        if (gantry is None or not gantry.is_ready or not self._motion_allowed()
                or self._jog is not None or self._pattern is not None or gantry.busy):
            return
        if axis in self._releasing_axes():
            return                  # still stopping: the controller ignores a new jog
        if not self._axes_at_rest(self._settled_axes()):
            self._say("Gantry not at rest yet (or no fresh position) — jog waits for it.",
                      "warning")
            return
        if not (hasattr(gantry, "jog_stop") and hasattr(gantry, "request_jog_relative")):
            self._say("Hold-jog needs the rig gantry link (disturbance_core.make_gantry_link).",
                      "error")
            return
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            self._say("Jog needs a live gantry position.", "warning")
            return
        if time.monotonic() >= self._jog_watch_until:
            self._restart_drift_watch(time.monotonic())   # the OTHER axes, from now
        self._jog = {"axis": axis, "sign": sign, "btn": self._jog_buttons[(axis, sign)],
                     "t0": time.monotonic(),
                     "speed": min(float(self.speed_spin.value()), core.axis_max_speed(axis)),
                     "sent_mm": 0.0}
        why = self._jog_segment()
        if why:
            self._jog = None
            self._say(why, "warning")
            return
        label = f"{_sign_txt(sign)}{axis.name}"
        end = self._jog["end"]
        speed = self._jog["speed"]
        if end is not None:
            self._say(f"JOG {label} (hold) at {speed:g} mm/s — stops at {end:.1f} mm at most")
        else:
            self._say(f"JOG {label} (hold) at {speed:g} mm/s — no {label} home: NO software "
                      "limit on this side, watch the axis", "warning")
        self._log_event("jog_start", f"{label} {'to home' if end is not None else 'free'} "
                                     f"@ {speed:g} mm/s from {pos[int(axis)]:.1f} mm")
        self._update_buttons()

    def _jog_segment(self) -> "str | None":
        """Send the next move of the current jog; None, or why it cannot go."""
        j, gantry = self._jog, self._gantry
        if j is None or gantry is None:
            return "no jog"
        axis, sign = j["axis"], j["sign"]
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            return "no live gantry position"
        cur = pos[int(axis)]
        home, stop = self._jog_bounds(axis, sign)
        end = self._jog_end_mm(axis, sign)
        side = f"{_sign_txt(sign)}{axis.name}"
        if end is not None:
            room = sign * (end - cur)
            what = "limit" if home is not None and end == home else "firmware stop"
            if room < 0:
                return (f"{axis.name} is beyond its {side} {what} ({end:.1f} mm) — jog the other "
                        f"way, or Clear {axis.name}")
            if room - LIMIT_EDGE_MM < JOG_MIN_TRAVEL_MM:
                return f"{axis.name} is at its {side} {what} ({end:.1f} mm)"
        if home is not None:
            # toward a home: ONE move that ends at it (or at a nearer firmware stop)
            delta, cap = sign * (sign * (end - cur) - LIMIT_EDGE_MM), JOG_MAX_MOVE_MM
        else:
            piece = float(self._options.jog_piece_mm)
            if end is not None:             # a firmware stop only shortens the piece
                piece = min(piece, sign * (end - cur) - LIMIT_EDGE_MM)
            delta, cap = sign * piece, float(self._options.jog_piece_mm)
        if j["sent_mm"] + abs(delta) > JOG_PRESS_MAX_MM + 1e-6:
            return (f"one press moves at most {JOG_PRESS_MAX_MM:.0f} mm — release and press "
                    "again to go on")
        why = self._endpoint_refused(axis, cur + delta)
        if why:
            return why
        try:
            gantry.request_jog_relative(axis, delta, j["speed"], cap)
        except (ValueError, RuntimeError) as exc:
            self._log_event("command_rejected", f"jog {axis.name} {delta:+.1f}: {exc}")
            return str(exc)
        self._move_targets = {}
        self._arm_gantry_deadline({axis: abs(delta)}, speed=j["speed"])
        j["chunk_from"] = cur
        j["end"] = home if home is None else end      # None = no home on this side
        j["sent_mm"] += abs(delta)
        j["rest_pos"] = j["rest_since"] = None
        return None

    @staticmethod
    def _at_rest(track: dict, value: float, now: float) -> bool:
        """True once ``value`` has stayed within REST_TOL_MM for REST_S."""
        if track.get("rest_pos") is None or abs(value - track["rest_pos"]) > REST_TOL_MM:
            track["rest_pos"], track["rest_since"] = value, now
            return False
        return now - track["rest_since"] >= REST_S

    def _continue_jog(self, gantry, now: float) -> None:
        """Readout tick while a jog is held: catch a lost release; send the
        next piece only after the last one has ended cleanly AND the axis is
        physically at rest (a swallowed SDK error or an early 'stopped' must
        not stack a second relative move on a running one)."""
        j = self._jog
        if not j["btn"].isDown():
            self._end_jog("button no longer held")
            return
        if gantry.busy or self._latched:
            return
        err = gantry.last_command_error
        if err:
            self._end_jog(f"link error — {err}")
            return
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            self._end_jog("gantry position stale")
            return
        cur = pos[int(j["axis"])]
        if not (self._at_rest(j, cur, now) and self._axes_at_rest(self._settled_axes(), now)):
            return
        frm = j.get("chunk_from")
        if frm is not None and abs(cur - frm) < JOG_MIN_TRAVEL_MM:
            # the last move went nowhere: a limit switch, a firmware stop or a
            # fault — do not keep hammering it
            self._end_jog(f"{j['axis'].name} did not move (limit switch, firmware stop or "
                          "fault?)")
            return
        why = self._jog_segment()
        if why:
            self._end_jog(why)

    def _end_jog(self, reason: str) -> None:
        """Release (or a lost release): ALWAYS stop — 'busy' means "the worker
        is still waiting on the move", not "the axis is still", so it is not
        asked. The vendor's decelerating per-axis stop, NOT latching: the
        panel then waits for the axis to come to rest (_watch_release) and
        escalates to STOP ALL if it does not."""
        j = self._jog
        if j is None:
            return
        self._jog = None
        gantry = self._gantry
        now = time.monotonic()
        label = f"{_sign_txt(j['sign'])}{j['axis'].name}"
        if gantry is not None:
            try:
                sent = gantry.jog_stop(j["axis"])
            except Exception as exc:  # noqa: BLE001 — a release must never fail silently
                sent = False
                self._log_event("stuck", f"jog stop raised: {exc}")
            if not sent and not gantry.jog_stop_running:
                self._stop_all()
                self._say(f"Jog release stop could not be sent — STOP ALL pressed. Watch {label}.",
                          "error", sticky_s=15.0)
                return
            self._add_release([j["axis"]], label, j["speed"], now)
        self._move_deadline = None
        self._gantry_cmd_in_flight = False
        self._gantry_busy_since = now           # the worker exits on its next poll
        # the drift watch goes on through the release (the jog axis stays in
        # _commanded_axes until it is at rest)
        self._log_event("jog_stop", f"{label} {reason}")
        if reason != "release":
            self._say(f"Jog ended: {reason}.", "warning")
        self._update_buttons()

    def _releasing_axes(self) -> set:
        r = self._jog_release
        return set(r["axes"]) if r is not None else set()

    def _settled_axes(self) -> list:
        """Every axis that is not still stopping from a release."""
        rel = self._releasing_axes()
        return [a for a in AXES if a not in rel]

    def _add_release(self, axes, label: str, speed: float, now: float,
                     kind: str = "jog") -> None:
        """What a non-latching stop of ``axes`` moving at ``speed`` must achieve,
        per axis: at rest by its deadline, within its slack of where it was.
        Joins a release still being watched (another axis may be jogged and
        released before the first one has settled)."""
        pos = self._gantry.positions_mm() if self._gantry is not None else None
        v = max(float(speed), 0.1)
        a = max(float(getattr(self._settings, "acc_mm_s2", 50.0)), 1.0)
        entries = {ax: {"label": label, "kind": kind, "t0": now, "v0": v, "a": a,
                        "from": pos[int(ax)] if pos is not None else None,
                        "deadline": now + RELEASE_GRACE_S + 2.0 * v / a,
                        "slack": RELEASE_SLACK_MM + 2.0 * (v * v / (2.0 * a) + 0.3 * v),
                        "hist": [], "restops": 0}
                   for ax in axes}
        r = self._jog_release
        if r is None:
            self._jog_release = {"label": label, "axes": entries}
        else:
            # this stop (mode 1 on every axis) also reaches the axes still
            # stopping: say so in their record — a first stop the controller
            # ignored would otherwise look routine once this one lands
            for e in r["axes"].values():
                e["restops"] += 1
            r["axes"].update(entries)
            r["label"] = ", ".join(sorted({e["label"] for e in r["axes"].values()}))

    @staticmethod
    def _release_speed(e: dict, cur: float, now: float) -> "float | None":
        """|speed| of a released axis over the last SPEED_WINDOW_S of ticks
        (None until the window is filled)."""
        hist = e["hist"]
        hist.append((now, cur))
        while len(hist) > 2 and now - hist[1][0] >= SPEED_WINDOW_S:
            hist.pop(0)
        t_old, p_old = hist[0]
        if now - t_old < SPEED_WINDOW_S:
            return None
        return abs(cur - p_old) / (now - t_old)

    @staticmethod
    def _release_speed_limit(e: dict, now: float) -> float:
        """The fastest a released axis may still be going ``now``: what a
        deceleration at ``a`` after RELEASE_LAG_S leaves of v0 at the middle
        of the speed window, + 20 % of v0 + 2 mm/s; unlimited before
        RELEASE_LAG_S + 0.4·v0/a."""
        v0, a, t = e["v0"], e["a"], now - e["t0"]
        if t < RELEASE_LAG_S + 0.4 * v0 / a:
            return float("inf")
        expected = max(0.0, v0 - a * (t - SPEED_WINDOW_S / 2.0 - RELEASE_LAG_S))
        return expected + 0.2 * v0 + 2.0

    def _watch_release(self, gantry, now: float) -> None:
        """After a release: each axis at rest within its deadline and slack →
        its buttons back (no latch). Otherwise the controller did not take the
        stop → STOP ALL."""
        r = self._jog_release
        if self._latched:
            self._jog_release = None             # STOP ALL's own watch takes over
            return
        if gantry.error:
            # the cached position no longer updates: a stopping axis can never be
            # seen at rest, and its record must not say "not taken" for that
            self._jog_release = None
            self._stop_all()
            self._say(f"Gantry link error during the {r['label']} stop — stop delivery unknown "
                      f"({gantry.error}). STOP ALL pressed; watch the axes.", "error",
                      sticky_s=15.0)
            self._log_event("stuck", f"link error during release ({r['label']}): {gantry.error}")
            return
        err = getattr(gantry, "jog_stop_error", None)
        if err:
            self._jog_release = None
            self._stop_all()                     # first: its own message must not win
            self._say(f"Jog release stop FAILED — {err} — STOP ALL pressed. If {r['label']} keeps "
                      "moving, CUT THE GANTRY POWER.", "error", sticky_s=15.0)
            self._log_event("stuck", f"jog release stop failed: {err}")
            return
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            self._jog_release = None
            self._stop_all()
            self._say(f"Gantry telemetry STALE after the {r['label']} release — stop delivery "
                      "unknown (link lost?). A controller move runs to its end on its own.",
                      "error", sticky_s=15.0)
            self._log_event("stuck", "gantry telemetry stale after jog release")
            return
        done = []
        for axis, e in r["axes"].items():
            moved = abs(pos[int(axis)] - e["from"]) if e["from"] is not None else 0.0
            speed = self._release_speed(e, pos[int(axis)], now)
            if speed is not None and speed > self._release_speed_limit(e, now):
                self._jog_release = None
                self._suspect_axes |= {ax.name for ax in r["axes"] if ax.name in self._taught}
                self._stop_all()
                self._say(f"{e['label']} still moving: {axis.name} at {speed:.0f} mm/s "
                          f"{now - e['t0']:.1f} s after the stop — NOT decelerating, the controller "
                          "did not take it. STOP ALL pressed; if it keeps moving, CUT THE GANTRY "
                          "POWER.", "error", sticky_s=20.0)
                self._log_event("stuck", f"{e['label']} {axis.name} at {speed:.1f} mm/s "
                                         f"{now - e['t0']:.2f} s after its stop, {moved:.1f} mm, "
                                         "not decelerating (not taken)")
                return
            if (self._axes_at_rest([axis], now) and not gantry.jog_stop_running
                    and moved <= e["slack"]):
                done.append((axis, e, moved))
            elif moved > e["slack"] or now > e["deadline"]:
                self._jog_release = None
                self._suspect_axes |= {ax.name for ax in r["axes"] if ax.name in self._taught}
                self._stop_all()
                self._say(f"{e['label']} still moving {moved:.0f} mm / {now - e['t0']:.1f} s "
                          "after the stop — the controller did not take it. STOP ALL pressed; "
                          "if it keeps moving, CUT THE GANTRY POWER.", "error", sticky_s=20.0)
                self._log_event("stuck", f"{e['label']} {axis.name} moved {moved:.1f} mm in "
                                         f"{now - e['t0']:.2f} s after its stop (not taken)")
                return
        for axis, e, moved in done:
            del r["axes"][axis]
            self._commanded_axes.discard(axis)
            if self._post_stop_ref is not None:
                # watched again from where it stopped, not from before it moved
                t_ref, ref = self._post_stop_ref
                ref = list(ref)
                ref[int(axis)] = pos[int(axis)]
                self._post_stop_ref = (t_ref, tuple(ref))
            self._log_event(f"{e['kind']}_stopped", f"{e['label']} {axis.name} at rest "
                                                    f"{moved:.1f} mm and {now - e['t0']:.2f} s "
                                                    "after its stop"
                            + (f" (stopped again by {e['restops']} later release(s))"
                               if e["restops"] else ""))
        if done and not r["axes"]:
            self._jog_release = None
            # drift watch from here: everything must now stay where it stopped
            self._restart_drift_watch(now)
            if self._jog is None:               # never record a position mid-jog
                self._save_homes()
        if done:
            self._update_buttons()

    # ------------------------------------------------------------- patterns
    def _on_pattern_mode(self, *_args) -> None:
        circle = self.pattern_combo.currentData() == "circle"
        self.pattern_size_caption.setText("RADIUS" if circle else "RANGE")
        spin = self.pattern_size_spin
        spin.blockSignals(True)
        if circle:
            spin.setRange(*core.CIRCLE_RADIUS_RANGE)
            spin.setValue(self._options.circle_radius_mm)
            spin.setToolTip("Circle radius in the Y-Z plane, around the current position")
        else:
            spin.setRange(*core.LINE_RANGE_RANGE)
            spin.setValue(self._options.line_range_mm)
            spin.setToolTip("Y travel peak to peak (100 = ±50 mm around the current position)")
        spin.blockSignals(False)

    def _pattern_room_refused(self, axis, lo: float, hi: float, cur: float) -> "str | None":
        """Why the pattern box lo..hi on ``axis`` does not fit: a home or the
        + firmware stop cuts into it. Worded as how far to move first."""
        e_lo, e_hi = self._effective_limits(axis)
        gantry = self._gantry
        stop = gantry.firmware_stops_mm().get(axis.name) if gantry is not None else None
        top = min([v for v in (e_hi, stop) if v is not None], default=None)
        if top is not None and hi > top + 1e-6:
            what = "firmware stop" if top == stop and (e_hi is None or stop < e_hi) else "+ home"
            return (f"{axis.name} needs room up to {hi:.1f} mm but its {what} is at {top:.1f} mm "
                    f"— move {axis.name} down by {hi - top:.1f} mm first, or make it smaller")
        if e_lo is not None and lo < e_lo - 1e-6:
            return (f"{axis.name} needs room down to {lo:.1f} mm but its − home is at "
                    f"{e_lo:.1f} mm — move {axis.name} up by {e_lo - lo:.1f} mm first, or make "
                    "it smaller")
        return None

    def _start_pattern(self) -> None:
        """Hand the rig link ONE path command (a lap of absolute vertices,
        repeated until Stop) centred on where the gantry stands now. X is not
        part of it and stays drift-watched."""
        gantry = self._gantry
        if gantry is None or not gantry.is_ready or not self._motion_allowed():
            return
        if not hasattr(gantry, "request_path") or not hasattr(gantry, "jog_stop"):
            self._say("Patterns need the rig gantry link (disturbance_core.make_gantry_link).",
                      "error")
            return
        if not (self._gantry_idle() and self._all_at_rest()):
            self._say("Start a pattern with the gantry at rest.", "warning")
            return
        pos = gantry.positions_mm()
        if pos is None or not gantry.telemetry_fresh():
            self._say("A pattern needs a live gantry position.", "warning")
            return
        mode = self.pattern_combo.currentData()
        size = float(self.pattern_size_spin.value())
        speed = float(self.pattern_speed_spin.value())
        y0, z0 = pos[int(Axis.Y)], pos[int(Axis.Z)]
        try:
            if mode == "circle":
                axes = [Axis.Y, Axis.Z]
                points = core.circle_points((y0, z0), size, self._options.path_segment_mm)
                start = (y0, z0)
                label = f"circle r {size:g} mm"
            else:
                axes = [Axis.Y]
                points = core.line_points(y0, size)
                start = (y0,)
                label = f"Y back-and-forth {size:g} mm"
            env = core.path_envelope(start, points)
        except ValueError as exc:
            self._say(str(exc), "warning")
            return
        for k, axis in enumerate(axes):
            why = self._pattern_room_refused(axis, env[k][0], env[k][1], start[k])
            if why:
                self._say(why, "warning", sticky_s=8.0)
                self._log_event("command_rejected", f"pattern {label}: {why}")
                return
        try:
            gantry.request_path(axes, points, speed, env)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            self._log_event("command_rejected", f"pattern {label}: {exc}")
            return
        now = time.monotonic()
        prev, longest = start, 0.0
        for q in list(points) + [points[0]]:
            longest = max(longest, math.dist(prev, q))
            prev = q
        acc = max(float(getattr(self._settings, "acc_mm_s2", 50.0)), 1.0)
        self._pattern = {
            "mode": mode, "axes": axes, "label": label, "speed": speed, "env": env,
            "center": start, "t0": now, "progress": None, "progress_t": now, "lap": 0,
            "stall_s": max(PATTERN_STALL_MIN_S, 3.0 * (longest / speed + speed / acc) + 5.0)}
        self._move_targets = {}
        self._move_deadline = None
        self._gantry_cmd_in_flight = True
        self._gantry_stuck = False
        self._commanded_axes |= set(axes)
        self._restart_drift_watch(now)          # X (and Z for the line) must stay put
        open_sides = [f"{_sign_txt(sg)}{a.name}" for a in axes for sg in (-1, 1)
                      if self._effective_limits(a)[0 if sg < 0 else 1] is None]
        centre = ", ".join(f"{a.name} {c:.1f}" for a, c in zip(axes, start))
        self._say(f"Pattern {label} at {speed:g} mm/s around {centre} mm"
                  + (f" — no home on {' '.join(open_sides)}: only the box limits it"
                     if open_sides else ""), "warning" if open_sides else "info", sticky_s=4.0)
        self._log_event("pattern_start", f"{label} @ {speed:g} mm/s, centre {centre} mm, "
                                         f"{len(points)} vertices/lap, box "
                        + ", ".join(f"{a.name} {lo:.1f}..{hi:.1f}"
                                    for a, (lo, hi) in zip(axes, env)))
        self._update_buttons()

    def _end_pattern(self, reason: str) -> None:
        """Stop button (or the pattern ended / stalled): the same non-latching
        decelerating stop as a jog release, on every axis, then the release
        watch — at rest in time or STOP ALL."""
        pat = self._pattern
        if pat is None:
            return
        self._pattern = None
        gantry = self._gantry
        now = time.monotonic()
        if gantry is not None:
            try:
                sent = gantry.jog_stop(*pat["axes"])       # mode 1, moving axes first
            except Exception as exc:  # noqa: BLE001 — a stop must never fail silently
                sent = False
                self._log_event("stuck", f"pattern stop raised: {exc}")
            if not sent and not gantry.jog_stop_running:
                self._stop_all()
                self._say("Pattern stop could not be sent — STOP ALL pressed.", "error",
                          sticky_s=15.0)
                return
            self._add_release(pat["axes"], "pattern", pat["speed"], now, kind="pattern")
        self._move_deadline = None
        self._gantry_cmd_in_flight = False
        self._gantry_busy_since = now           # the worker exits on its next poll
        laps = pat["lap"]
        self._log_event("pattern_stop", f"{pat['label']}: {reason} after {laps} lap(s), "
                                        f"{now - pat['t0']:.1f} s")
        if reason != "stopped":
            self._say(f"Pattern ended: {reason}.", "warning", sticky_s=8.0)
        else:
            self._say(f"Pattern stopping ({laps} lap(s) done)…")
        self._update_buttons()

    def _watch_pattern(self, gantry, now: float) -> None:
        """Readout tick while a pattern runs: lap count, the box, progress."""
        pat = self._pattern
        if not gantry.busy:
            err = gantry.last_command_error or gantry.error
            self._abort_pattern(f"link error — {err}" if err else "the link stopped running it")
            return
        pos = gantry.positions_mm()
        if pos is not None and gantry.telemetry_fresh():
            for k, axis in enumerate(pat["axes"]):
                lo, hi = pat["env"][k]
                v = pos[int(axis)]
                if v < lo - PATTERN_ENVELOPE_SLACK_MM or v > hi + PATTERN_ENVELOPE_SLACK_MM:
                    self._log_event("stuck", f"pattern {axis.name} {v:.1f} mm outside "
                                             f"{lo:.1f}..{hi:.1f}")
                    self._abort_pattern(f"{axis.name} at {v:.1f} mm left its box "
                                        f"{lo:.1f}..{hi:.1f} mm")
                    return
        prog = gantry.path_progress
        if prog != pat["progress"]:
            pat["progress"], pat["progress_t"] = prog, now
            if prog is not None and prog[0] > pat["lap"]:
                pat["lap"] = prog[0]
                self._log_event("pattern_lap", f"{pat['label']}: lap {pat['lap']} done")
        elif now - pat["progress_t"] > pat["stall_s"]:
            self._abort_pattern(f"no progress for {pat['stall_s']:.0f} s (stalled?)")

    def _abort_pattern(self, reason: str) -> None:
        """Every pattern ending the operator did not ask for latches: the
        frame or the controller may be wrong, so nothing moves before Reset."""
        pat, self._pattern = self._pattern, None
        self._stop_all()                         # first: its own message must not win
        self._say(f"Pattern {pat['label']} ended: {reason} — STOP ALL pressed. Check the "
                  "gantry, then Reset.", "error", sticky_s=15.0)
        self._log_event("pattern_stop", f"{pat['label']}: {reason} after {pat['lap']} lap(s); "
                                        "STOP ALL")

    # ============================================================= motor
    def _toggle_enable(self) -> None:
        motor = self._motor
        if motor is None:
            return
        if self._motor_auto is not None:
            return                          # the zero-at-connect sequence owns it
        # unknown state (the sequence stopped before its Disable): the button
        # sends Disable — the only way left to de-energise a drive held over
        enabling = not motor.is_enabled and self._disabled_this_session
        if enabling and (not motor.is_ready or not self._motion_allowed()):
            return                          # enabling is motion-adjacent; disabling always OK
        if not enabling and not motor.is_connected:
            return
        if enabling and not self._thrust_is_zero():
            self._say("Turn the thrust off first: the mount must hold while the jet runs "
                      "(Enable/Disable only at thrust 0).", "warning")
            return
        if not enabling and self._options.motor_interlock and not self._jet_off_known():
            self._say("Disable needs the jet known OFF: connect the thruster (its reset makes it "
                      "0) or turn the thrust off — the mount must not go limp under a jet.",
                      "warning")
            return
        try:
            motor.set_enabled(enabling)
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            self._log_event("command_rejected", f"{'enable' if enabling else 'disable'}: {exc}")
            return
        if not enabling:
            self._disabled_this_session = True
        self._log_event("enable" if enabling else "disable", "motor")

    def _advance_motor_auto(self, motor) -> None:
        """Zero-at-connect, one step per tick: Disable, Disable again, Zero
        here, Enable, then watch the angle — each step confirmed by the link
        (and dwelt on) before the next. The drive's enable state is unknown at
        connect (a previous session may have left it holding); Zero needs it
        limp (whisker zero_here); Enable holds where it stands — 0 after the
        zero (hardware, by hand, n = 1: Disable, Zero at 110°, Enable stayed at
        -0.01°, data/20261026/1026_171856_disturb/samples.csv).

        It starts only with the THRUSTER connected: opening its port resets the
        Arduino to 1500 µs, so the mount is never made limp under a jet left
        running by a panel that died (the firmware has no host-loss failsafe).
        Any doubt stops the sequence; once the zero was sent, a stop also
        sends Disable (an energised drive chasing an old target must not)."""
        a = self._motor_auto
        if a is None:
            return
        now = time.monotonic()
        stage = a["stage"]

        def fail(why: str, quiet: bool = False, limp: "bool | None" = None) -> None:
            self._motor_auto = None
            if limp is None:
                limp = stage in ("zero", "enable", "settle")
            if limp:
                try:
                    motor.set_enabled(False)
                    self._disabled_this_session = True
                except (ValueError, RuntimeError):
                    pass
            if not quiet:
                self._say(f"Motor zero-at-connect stopped ({stage}): {why}."
                          + (" Disable sent." if limp else "") + " Use Enable / Disable by hand.",
                          "warning", sticky_s=8.0)
            self._log_event("motor_auto_failed", f"{stage}: {why}"
                            + ("; Disable sent" if limp else ""))

        if self._latched:
            # quiet: whatever pressed STOP ALL owns the status line (a gantry
            # "CUT THE GANTRY POWER" must not be replaced by this)
            fail("STOP ALL latched", quiet=True, limp=False)   # STOP ALL holds the motor
            return
        err = motor.error or motor.last_command_error
        if err and motor.is_ready:
            fail(err)
            return
        if stage == "ready" and motor.is_ready and not self._jet_off_known():
            a["t"] = now                        # waiting for the thruster is not a timeout
            return
        limit = MOTOR_AUTO_READY_S if stage == "ready" else MOTOR_AUTO_STEP_S
        if now - a["t"] > limit:
            fail(f"no confirmation within {limit:g} s")
            return
        angle = motor.angle_deg()
        fresh = motor.telemetry_fresh()
        try:
            if stage == "ready":
                if not motor.is_ready:
                    return
                if not motor.mode_verified:
                    fail("POS_VEL mode not verified")
                    return
                if not self._thrust_is_zero():
                    fail("thrust is not 0")
                    return
                motor.set_enabled(False)
                self._disabled_this_session = True
                a.update(stage="disable", t=now, confirmed=False)
            elif stage in ("disable", "disable2"):
                if motor.busy or motor.is_enabled:
                    return
                if not a["confirmed"]:
                    a.update(confirmed=True, t=now)   # dwell from when the link sent it
                    return
                if now - a["t"] < MOTOR_AUTO_DWELL_S:
                    return
                if stage == "disable":
                    motor.set_enabled(False)        # again: a lost CAN frame has no ack
                    a.update(stage="disable2", t=now, confirmed=False)
                    return
                if angle is None or not fresh:
                    return
                motor.zero_here()
                a.update(stage="zero", t=now, zero_angle=None)
            elif stage == "zero":
                if motor.busy or angle is None or not fresh:
                    return
                if abs(angle) > MOTOR_AUTO_ZERO_TOL_DEG:
                    # the poll may still show the old angle once; a drive that keeps
                    # turning is still energised (a lost Disable): stop it now
                    first = a["zero_angle"]
                    if first is None:
                        a["zero_angle"] = angle
                    elif abs(angle - first) > MOTOR_AUTO_ZERO_TOL_DEG:
                        fail(f"the shaft is turning after Zero ({first:.1f}° → {angle:.1f}°)")
                    return
                if not self._thrust_is_zero():
                    fail("thrust is not 0")
                    return
                motor.set_enabled(True)
                a.update(stage="enable", t=now)
            elif stage == "enable":
                if motor.busy or not motor.is_enabled or not fresh:
                    return
                a.update(stage="settle", t=now)
            elif stage == "settle":
                if angle is None or not fresh:
                    return
                if abs(angle) > MOTOR_AUTO_ZERO_TOL_DEG:
                    fail(f"the shaft moved to {angle:.1f}° after Enable")
                    return
                if now - a["t"] < MOTOR_AUTO_DWELL_S:
                    return
                self._motor_auto = None
                self._motor_zeroed = True
                self.angle_spin.setValue(0.0)
                self._motor_target_initialized = True
                self._say("Motor zeroed where it stood and holding at 0°.")
                self._log_event("motor_auto_zero", f"disable x2, zero, enable; holding at "
                                                   f"{angle:.2f} deg")
        except (ValueError, RuntimeError) as exc:
            fail(str(exc))

    def _go_theta(self) -> None:
        motor = self._motor
        if motor is None or not motor.is_ready or not self._motion_allowed():
            return
        target = float(self.angle_spin.value())
        measured = motor.angle_deg()
        if measured is not None:
            # never the long way round: a +-180 seam crossing sweeps the jet and
            # winds the cable (whisker shortest_aim_command, motion_config.py:428)
            from motion_config import shortest_aim_command  # noqa: PLC0415
            target = shortest_aim_command(target, measured, ANGLE_LIMIT_DEG)
        try:
            motor.go_to(target, self.vel_spin.value())
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            self._log_event("command_rejected", f"goto: {exc}")
            return
        self._motor_goto_deadline = time.monotonic() + MOTOR_SETTLE_TIMEOUT_S
        self._motor_stuck = False
        self._say(f"Motor → {target:.1f}° at {self.vel_spin.value():g} °/s")
        self._log_event("goto", f"{target:.1f} deg (spin {self.angle_spin.value():.1f}) @ "
                                f"{self.vel_spin.value():g} deg/s")

    def _recover_motor(self) -> None:
        motor = self._motor
        if motor is None:
            return
        try:
            motor.stop_all()
            motor.reset_stop()
        except (ValueError, RuntimeError) as exc:
            self._say(f"motor recover: {exc}", "warning")
            return
        self._motor_goto_deadline = None
        self._motor_stuck = False
        self._say("Motor held here and its link reset.", "warning")
        self._log_event("recover", "motor")

    # ============================================================ thruster
    def _set_thruster(self) -> None:
        thruster = self._thruster
        if thruster is None or not thruster.is_ready or self._latched or self._cut_pending:
            return
        if self._options.motor_interlock and not self._motor_holding():
            self._say("Thrust needs the motor enabled and holding (a limp mount swings under "
                      "the jet). Bench-only override: --no-motor-interlock.", "warning")
            return
        try:
            thruster.set_throttle(self.thruster_spin.value())
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            self._log_event("command_rejected", f"thrust: {exc}")
            return
        self._say(f"Thruster ramping → {self.thruster_spin.value():+d} % ({self._ramp_text()})")
        self._log_event("thrust", f"{self.thruster_spin.value():+d} %")

    def _zero_thruster(self) -> None:
        thruster = self._thruster
        if thruster is None or not thruster.is_ready or self._latched or self._cut_pending:
            return
        try:
            thruster.stop_thruster()
        except (ValueError, RuntimeError) as exc:
            self._say(str(exc), "warning")
            return
        self.thruster_spin.setValue(0)
        self._say("Thruster ramping → 0 %")
        self._log_event("thrust0", "ramp down")

    def _cut_thrust(self) -> None:
        """Immediate zero for the thruster only. stop_all() latches the link;
        reset_stop() follows automatically once the worker's ramp loop has
        exited (busy False, ramp_target None). stop_all(latch=False) is NOT
        used: it leaves a running ramp alive, which climbs back from 0."""
        thruster = self._thruster
        if thruster is None:
            return
        thruster.stop_all()             # works on the driver alone, error or not
        self._cut_pending = True
        self.thruster_spin.setValue(0)
        self._say("Thrust cut to 0 (thruster link latched until its ramp loop exits).",
                  "warning")
        self._log_event("cut", "thruster stop_all (immediate 0)")
        self._update_buttons()

    def _finish_cut(self) -> None:
        thruster = self._thruster
        if thruster is None:
            self._cut_pending = False
            return
        if thruster.busy or thruster.ramp_target() is not None:
            return
        # the ramp loop has exited: one more zero covers the latch-check/send
        # race (a step can follow the first zero, motion_control.py:1119-1129)
        thruster.stop_all()
        if thruster.throttle_pct() != 0:
            return                           # re-check on the next tick
        try:
            thruster.reset_stop()
        except RuntimeError:
            return
        self._cut_pending = False
        self._log_event("cut_done", "thruster link reset")

    # ========================================================== STOP / Reset
    def _fast_axes(self) -> set:
        """Axes being driven faster than SUSPECT_ABOVE_MM_S right now (a jog, a
        Send, a pattern, or a release still decelerating from such a speed)."""
        fast = set()
        if self._jog is not None and self._jog["speed"] > SUSPECT_ABOVE_MM_S:
            fast.add(self._jog["axis"])
        if self._move_targets and self._move_speed > SUSPECT_ABOVE_MM_S:
            fast |= set(self._move_targets)
        if self._pattern is not None and self._pattern["speed"] > SUSPECT_ABOVE_MM_S:
            fast |= set(self._pattern["axes"])
        if self._jog_release is not None:
            fast |= {ax for ax, e in self._jog_release["axes"].items()
                     if e["v0"] > SUSPECT_ABOVE_MM_S}
        return fast

    def _stop_all(self) -> None:
        fast = {ax.name for ax in self._fast_axes()} & set(self._taught)
        if fast:
            self._suspect_axes |= fast
            self._log_event("homes_suspect", f"STOP ALL above {SUSPECT_ABOVE_MM_S:g} mm/s on "
                                             f"{','.join(sorted(fast))}: steps may be lost")
        self._latched = True
        self._cut_pending = False
        # Order: gantry FIRST because stop_all(wait=False) only sets flags and
        # spawns the stop thread (microseconds), then the thruster zero (a
        # serial write under the thruster lock, <= ~2 s worst case), then the
        # motor hold (under the CAN lock, which the connect handshake can hold
        # for seconds). Any other order would delay the gantry stop behind I/O.
        failed = []
        for name, link, call in (("gantry", self._gantry, lambda g: g.stop_all(wait=False)),
                                 ("thruster", self._thruster, lambda t: t.stop_all()),
                                 ("motor", self._motor, lambda m: m.stop_all())):
            if link is None:
                continue
            try:
                call(link)
            except Exception as exc:  # noqa: BLE001 — never skip the next device's stop
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
        self._move_targets = {}
        self._move_box = {}
        self._move_deadline = None
        self._gantry_cmd_in_flight = False
        self._gantry_recover_pending = False          # Reset is the only latch-clearer now
        self._jog = None                              # the stop above also ends a jog
        self._pattern = None                          # ... and a pattern
        self._jog_release = None
        self._commanded_axes = set()                  # the latched watch covers every axis
        self._jog_watch_until = 0.0                   # the latched watch takes over
        self._gantry_stop_deadline = (time.monotonic() + GANTRY_STOP_SETTLE_TIMEOUT_S
                                      if self._gantry is not None else None)
        self._post_stop_ref = None
        self._post_stop_warned = False
        self._watch_ref_at = (time.monotonic() + self._watch_settle_s()
                              if self._gantry is not None else None)
        self._motor_goto_deadline = None
        self._interlock_lost_since = None
        cancelled = self._motor_disconnect_after_disable
        self._motor_disconnect_after_disable = False  # STOP dropped the queued Disable
        for ms in STOP_RESEND_MS:
            QTimer.singleShot(ms, self._stop_thruster_again)
        self._update_buttons()          # before anything that can touch the disk
        self._say("STOP requested — motion latched off until Reset."
                  + (" Pending motor disconnect cancelled (motor holds)." if cancelled else ""),
                  "error")
        self._log_event("stop_all", "gantry stop, thruster 0, motor hold")
        if cancelled:
            self._log_event("disconnect_cancelled", "motor: STOP dropped the queued Disable")
        if failed:                      # last, so this message is the one on screen
            self._stop_fail_text = "; ".join(failed)
            self._stop_fail_until = time.monotonic() + 20.0
            self._say("STOP ALL: a stop call FAILED — " + "; ".join(failed)
                      + " — cut that device's power.", "error", sticky_s=20.0)
            self._log_event("stop_failed", "; ".join(failed))

    def _stop_thruster_again(self) -> None:
        if not self._latched or self._thruster is None:
            return
        self._thruster.stop_all()
        self._log_event("stop_all_resend", "thruster 0 again")

    def _reset_latch(self) -> None:
        t = self._thruster
        if t is not None:
            if t.busy or t.ramp_target() is not None or t.throttle_pct() != 0:
                # the ramp worker is still inside its loop or the cache is not 0:
                # clearing the latch now would let the staircase resume
                t.stop_all()
                self._say("Thruster not idle at 0 yet — zero re-sent; press Reset again.",
                          "warning")
                return
            if t.error is not None:
                t.stop_all()        # best effort on a dead link; physical state unknown
                self._say("Thruster link has an error — physical thrust UNKNOWN. Disconnect/"
                          "Connect the thruster before any thrust.", "warning", sticky_s=6.0)
        if self._gantry is not None and not self._all_at_rest():
            self._say("Gantry not at rest (still moving, or no fresh position) — Reset refused. "
                      "If an axis keeps moving, CUT THE GANTRY POWER; a dead link: Disconnect "
                      "gantry.", "error", sticky_s=8.0)
            self._log_event("reset_refused", "gantry not at rest")
            return
        failures = []
        for name, link in (("thruster", self._thruster), ("motor", self._motor),
                           ("gantry", self._gantry)):
            if link is None:
                continue
            try:
                link.reset_stop()
            except RuntimeError as exc:
                failures.append(f"{name}: {exc}")
        if failures:
            self._say("Reset incomplete — " + "; ".join(failures) + " — press Reset again.",
                      "warning")
            return
        self._latched = False
        self._cut_pending = False
        self._gantry_stop_deadline = None
        self._post_stop_ref = None
        self._watch_ref_at = None
        if self._gantry is not None:
            self._restart_drift_watch(time.monotonic())   # still watched after Reset
        self._gantry_recover_pending = False
        self._gantry_stuck = False
        self._say("Stop latch cleared. Nothing is re-commanded.")
        self._log_event("reset", "latch cleared")
        self._update_buttons()

    def _on_ui_fault(self, level: str, message: str) -> None:
        """qt.install_slot_guard callback: a slot raised. Stop first (once),
        then report — the report path touches the disk and must not be able to
        prevent the stop; a fault that repeats every tick must not become a
        10 Hz STOP storm."""
        self._fault_count += 1
        try:
            if not self._latched:
                self._stop_all()
        except Exception:  # noqa: BLE001 — the guard must never raise
            pass
        try:
            now = time.monotonic()
            if now - self._last_fault_log >= FAULT_LOG_PERIOD_S:
                self._last_fault_log = now
                self._say(f"UI FAULT #{self._fault_count} — {message}", "error", sticky_s=30.0)
                self._log_event("fault", f"#{self._fault_count} {message}")
        except Exception:  # noqa: BLE001
            pass

    # ================================================================ log
    def _note_zero_outcome(self, thruster, where: str) -> "str | None":
        """After a final stop_all(): did the zero demonstrably reach the board?
        stop_all caches 0 even when the write raised, and pwm_us is None when
        the echo never came — both mean the physical thrust is UNKNOWN."""
        if thruster is None:
            return None
        problem = thruster.error or thruster.last_command_error
        if problem is None and thruster.pwm_us() is None:
            problem = "zero not echoed by the board"
        if problem is None:
            return None
        text = f"thruster zero UNCONFIRMED at {where} — {problem} — physical thrust UNKNOWN"
        self._log_event("zero_unconfirmed", text)
        print(f"\n*** {text}: cut the ESC supply ***\n", file=sys.stderr, flush=True)
        return text

    def _links_at_stop(self) -> dict:
        out = {}
        for name, link in (("gantry", self._gantry), ("motor", self._motor),
                           ("thruster", self._thruster)):
            if link is not None:
                out[name] = {"error": link.error, "last_command_error": link.last_command_error}
        return out

    def _log_event(self, kind: str, detail: str = "") -> None:
        if self._log is not None and not self._log.closed:
            self._log.event(kind, detail)

    def _ensure_log(self) -> None:
        if self._log is not None or not self._options.autolog:
            return
        self._new_log()

    def _new_log(self) -> None:
        if self._log is not None:
            self._stop_log()
        try:
            self._log = core.DisturbLog(
                self._options.data_root,
                core.base_meta(self._settings, self._options, self._whisker_root,
                               preflight=self._preflight, prov=self._provenance))
        except OSError as exc:
            self._log = None
            self._say(f"log folder could not be created: {exc}", "error")
            return
        self._thruster_rows = []
        if self._thruster is not None:            # lock-only; harmless before ready
            self._thruster.start_recording()
        self.log_label.setText(str(self._log.dir))
        self._log.event("log", "started")
        self._update_buttons()

    def _stop_log(self) -> None:
        log = self._log
        if log is None:
            return
        rows = list(self._thruster_rows)
        if self._thruster is not None:            # also on an errored link: lock-only
            try:
                rows.extend(self._thruster.stop_recording())
            except Exception:  # noqa: BLE001
                pass
        extra = {"gantry": {"taught_homes_mm": {k: dict(v) for k, v in self._taught.items()},
                            "suspect_axes": sorted(self._suspect_axes),
                            "last_pos_mm": self._last_fresh_pos,
                            "limits_file": str(self._options.limits_file),
                            "link_limits_at_connect_mm": (list(self._link_limits0)
                                                          if self._link_limits0 else None),
                            "soft_limits_mm": (list(self._gantry.soft_limits_mm())
                                               if self._gantry is not None else None),
                            "firmware_stops_mm": (self._gantry.firmware_stops_mm()
                                                  if self._gantry is not None else None)},
                 "motor": {"gear_ratio": getattr(self._motor, "gear_ratio", None),
                           "limits": getattr(self._motor, "limits", None),
                           "limits_verified": getattr(self._motor, "limits_verified", None),
                           "mode_verified": getattr(self._motor, "mode_verified", None)},
                 "pose_stop": pose_snapshot(self._gantry, self._motor) if pose_snapshot else None,
                 "links_at_stop": self._links_at_stop(),
                 "settings_at_stop": core.settings_summary(self._settings, self._options)}
        log.close(thruster_rows=rows, extra=extra)
        self._log = None
        self._thruster_rows = []
        self.log_label.setText(f"stopped: {log.dir.name}")
        self._update_buttons()

    def _log_sample(self) -> None:
        if self._log is None or self._log.closed:
            return
        g, m, t = self._gantry, self._motor, self._thruster
        pos = g.positions_mm() if g is not None else None
        self._log.sample({
            "host_epoch_s": time.time(),
            "t_monotonic_s": time.monotonic(),
            "x_mm": pos[0] if pos else None,
            "y_mm": pos[1] if pos else None,
            "z_mm": pos[2] if pos else None,
            "gantry_busy": (g.busy if g is not None else None),
            "motor_deg": (m.angle_deg() if m is not None else None),
            "torque_nm": (m.torque_nm() if m is not None else None),
            "motor_enabled": (m.is_enabled if m is not None else None),
            "motor_busy": (m.busy if m is not None else None),
            "thr_pct_cmd": (t.throttle_pct() if t is not None else None),
            "pwm_us": (t.pwm_us() if t is not None else None),
            "ramp_target_pct": (t.ramp_target() if t is not None else None),
            "latched": self._latched,
        })
        if self._log.failed and not getattr(self, "_log_failure_said", False):
            self._log_failure_said = True
            self._say(f"LOG WRITE FAILED — {self._log.failed}; the panel keeps running "
                      "but this run is not being recorded", "error", sticky_s=30.0)

    # ============================================================ readout
    def _update_readout(self) -> None:
        gantry, motor, thruster = self._gantry, self._motor, self._thruster
        # ready transitions (autolog, thruster recording, events)
        for name, link in (("gantry", gantry), ("motor", motor), ("thruster", thruster)):
            ready = link is not None and link.is_ready
            if ready and not self._was_ready[name]:
                self._was_ready[name] = True
                self._log_event("ready", name)
                self._ensure_log()
                if name == "thruster" and self._log is not None:
                    link.start_recording()
            elif not ready and link is None:
                self._was_ready[name] = False
        # gantry
        if gantry is not None:
            positions = gantry.positions_mm()
            if positions is not None:
                for axis in AXES:
                    self._live_labels[axis].setText(_fmt_mm(positions[int(axis)]))
                if len(self._gantry_targets_initialized) < len(AXES):
                    self._target_dirty.clear()
                    self._seed_targets(positions, only_clean=False)
                    self._gantry_targets_initialized.update(AXES)
            now = time.monotonic()
            fresh = (positions is not None and gantry.error is None
                     and gantry.telemetry_fresh(REST_FRESH_S))
            if gantry.is_ready and not self._limits_applied and fresh:
                # frame check against the last position, on THIS tick's reading
                self._apply_soft_limits(gantry, first_pos=positions)
            self._track_rest(positions if fresh else None, now)
            if self._move_box and fresh:
                self._watch_send_box(positions)
            busy = gantry.busy
            if busy and not self._gantry_was_busy:
                self._gantry_busy_since = now
            # a short command (Home on an axis already at its switch, a 1 mm
            # jog) can start and finish between two 100 ms ticks, so "busy
            # went away" alone would miss it: a command we sent counts as
            # settled as soon as the link is idle.
            if not busy and (self._gantry_was_busy or self._gantry_cmd_in_flight):
                self._on_gantry_settled()
                self._gantry_busy_since = None
            self._gantry_was_busy = busy
            if busy:
                if self._gantry_cmd_in_flight and self._move_deadline is not None \
                        and now > self._move_deadline and not self._gantry_stuck:
                    self._gantry_stuck = True
                    self._say("Gantry move timed out (still busy) — Recover gantry.",
                              "warning")
                    self._log_event("stuck", "gantry move deadline")
                elif (self._gantry_stop_deadline is not None and now > self._gantry_stop_deadline
                      and not self._gantry_stuck):
                    self._gantry_stuck = True
                    self._say("Gantry still busy 5 s after STOP — the stop may not have "
                              "reached the controller (link?). Disconnect gantry or close the "
                              "panel; if an axis moves, CUT THE GANTRY POWER.",
                              "error")
                    self._log_event("stuck", "gantry busy after STOP")
                elif (not self._gantry_cmd_in_flight and self._gantry_busy_since is not None
                      and now - self._gantry_busy_since > GANTRY_IDLE_BUSY_TIMEOUT_S
                      and not self._gantry_stuck):
                    self._gantry_stuck = True
                    self._say("Gantry link busy with nothing sent for 60 s — Recover gantry.",
                              "warning")
                    self._log_event("stuck", "gantry busy, no command in flight")
            else:
                self._gantry_stop_deadline = None
            # hold-jog: the next piece while held (and a release that never
            # arrived: focus lost, button disabled without a signal)
            if self._jog is not None:
                self._continue_jog(gantry, now)
            if self._pattern is not None:
                self._watch_pattern(gantry, now)
            if self._jog_release is not None:
                self._watch_release(gantry, now)
            # drift watch after STOP ALL, and after a release has settled. busy
            # clears within 50 ms of a stop whatever the axis does; what we CAN
            # see is the cached position moving away from a reference taken after
            # the deceleration margin (cumulative, no sliding window: slow creep
            # adds up), or the position poll going stale
            after_stop = "STOP ALL" if self._latched else "jog release"
            watching = (self._latched or now < self._jog_watch_until or self._jog is not None
                        or self._pattern is not None or self._jog_release is not None)
            if watching and not self._post_stop_warned:
                pos_now = gantry.positions_mm()
                if not gantry.telemetry_fresh():
                    if not self._latched:
                        self._stop_all()       # first: its own message must not win
                    self._post_stop_warned = True
                    self._say(f"Gantry telemetry STALE after {after_stop} — stop delivery "
                              "unknown (link lost?). A controller move runs to its end on "
                              "its own.", "error", sticky_s=15.0)
                    self._log_event("stuck", f"gantry telemetry stale after {after_stop}")
                elif (self._watch_ref_at is not None and now >= self._watch_ref_at
                      and pos_now is not None):
                    if self._post_stop_ref is None:
                        self._post_stop_ref = (now, pos_now)
                    else:
                        p_ref = self._post_stop_ref[1]
                        watched = [a for a in AXES if a not in self._commanded_axes]
                        drift = {a: abs(pos_now[int(a)] - p_ref[int(a)]) for a in watched}
                        moved = max(drift.values(), default=0.0)
                        if moved > POST_STOP_MOVE_TOL_MM:
                            self._suspect_axes |= {a.name for a, d in drift.items()
                                                   if d > POST_STOP_MOVE_TOL_MM
                                                   and a.name in self._taught}
                            if not self._latched:
                                self._stop_all()   # first: its own message must not win
                            self._post_stop_warned = True
                            self._say(f"Gantry still moving {moved:.0f} mm after {after_stop} "
                                      "— the stop did not take. CUT THE GANTRY POWER if it "
                                      "keeps going.", "error", sticky_s=15.0)
                            self._log_event("stuck", f"gantry moved {moved:.1f} mm after "
                                                     f"{after_stop}")
            if self._gantry_recover_pending:
                self._finish_gantry_recover()
        else:
            for axis in AXES:
                self._live_labels[axis].setText("--")
        # motor
        if motor is not None:
            if self._motor_disconnect_after_disable and not motor.is_enabled:
                self._disconnect_motor_now()
                self._update_buttons()
                return
            self._advance_motor_auto(motor)
            angle = motor.angle_deg()
            fresh = motor.telemetry_fresh()
            if angle is not None:
                torque = motor.torque_nm()
                self.motor_live.setText(
                    f"{angle:.1f} °" + (f"   {torque:.2f} N·m" if torque is not None else "")
                    + ("" if fresh else "   [STALE]"))
                if not self._motor_target_initialized and self._motor_auto is None:
                    self.angle_spin.setValue(angle)
                    self._motor_target_initialized = True
            self.enable_btn.setText("Disable" if (motor.is_enabled
                                                  or not self._disabled_this_session)
                                    else "Enable")
            if self._motor_auto is not None:
                waiting = not self._jet_off_known()
                state = ("connecting…" if not motor.is_ready else
                         "connect the thruster to zero + hold (its reset = 0 thrust)"
                         if waiting and self._motor_auto["stage"] == "ready" else "zeroing here…")
            elif motor.is_enabled:
                state = (("holding  ·  0° = where it stood at connect" if self._motor_zeroed
                          else "holding  ·  NOT zeroed this connection") if fresh else
                         "enabled but telemetry STALE — supply/CAN? hold unknown")
            elif self._disabled_this_session:
                state = "disabled (limp)"
            else:
                state = ("state unknown (may be holding) — "
                         + ("Disable" if self._jet_off_known() or not self._options.motor_interlock
                            else "connect the thruster, then Disable"))
            info = [state]
            if motor.is_ready and not motor.mode_verified:
                info.append("POS_VEL UNVERIFIED")
            if motor.is_ready and not getattr(motor, "limits_verified", True):
                info.append("limits fallback")
            self.motor_info.setText("  |  ".join(info))
            # interlock watchdog: thrust on while the mount is no longer holding
            if (self._options.motor_interlock and thruster is not None
                    and not self._latched and not self._cut_pending
                    and (thruster.throttle_pct() != 0 or thruster.ramp_target() is not None)
                    and not self._motor_holding()):
                now_i = time.monotonic()
                if self._interlock_lost_since is None:
                    self._interlock_lost_since = now_i
                elif now_i - self._interlock_lost_since > INTERLOCK_GRACE_S:
                    self._interlock_lost_since = None
                    self._say("Motor no longer holding while the jet runs (telemetry stale / "
                              "disabled) — thrust CUT automatically.", "error", sticky_s=10.0)
                    self._log_event("interlock_lost", "motor not holding under thrust; auto cut")
                    self._cut_thrust()
            else:
                self._interlock_lost_since = None
            if (self._motor_goto_deadline is not None and motor.busy
                    and time.monotonic() > self._motor_goto_deadline and not self._motor_stuck):
                self._motor_stuck = True
                self._say("Motor never settled (busy past 25 s) — Recover motor.", "warning")
                self._log_event("stuck", "motor settle timeout")
            if not motor.busy:
                self._motor_goto_deadline = None
        else:
            self.motor_live.setText("--")
            self.motor_info.setText("")
            self.enable_btn.setText("Enable")
            if (self._options.motor_interlock and thruster is not None
                    and not self._latched and not self._cut_pending
                    and (thruster.throttle_pct() != 0 or thruster.ramp_target() is not None)):
                now_i = time.monotonic()
                if self._interlock_lost_since is None:
                    self._interlock_lost_since = now_i
                elif now_i - self._interlock_lost_since > INTERLOCK_GRACE_S:
                    self._interlock_lost_since = None
                    self._say("Jet running with no motor link — thrust CUT automatically.",
                              "error", sticky_s=10.0)
                    self._log_event("interlock_lost", "no motor link under thrust; auto cut")
                    self._cut_thrust()
        # thruster — the cached values stay visible on a link error: the
        # physical throttle is then UNKNOWN, which is the one thing to show
        if thruster is not None and (thruster.is_ready or thruster.error):
            pwm = thruster.pwm_us()
            target = thruster.ramp_target()
            text = f"{thruster.throttle_pct():+d} %"
            if target is not None and target != thruster.throttle_pct():
                text += f"  → {target:+d} %"
            if pwm is not None:
                text += f"   ·  {pwm} µs"
            if self._cut_pending:
                text += "  [cut]"
            if thruster.error:
                text += "  LINK ERROR — physical thrust unknown: STOP ALL, then reconnect"
            self.thruster_live.setText(text)
            if self._cut_pending:
                self._finish_cut()
        else:
            self.thruster_live.setText("--")
        self._log_sample()
        self._update_buttons()

    def _update_buttons(self) -> None:
        gantry_ok = self._gantry is not None and self._gantry.is_ready
        motor_ok = self._motor is not None and self._motor.is_ready
        motor_connected = self._motor is not None and self._motor.is_connected
        thruster_ok = self._thruster is not None and self._thruster.is_ready
        allowed = self._motion_allowed()
        busy = gantry_ok and self._gantry.busy
        thrust_zero = self._thrust_is_zero()
        jog = self._jog
        releasing = self._jog_release is not None
        jogging = jog is not None or releasing
        still = gantry_ok and self._all_at_rest()
        pattern = self._pattern is not None
        stopping = self._releasing_axes()
        settled_still = gantry_ok and self._axes_at_rest(self._settled_axes())
        for (axis, sign), btn in self._jog_buttons.items():
            # the held button must stay enabled for the whole jog: disabling a
            # pressed button would itself end the jog. While a released axis is
            # still stopping only ITS buttons wait (the controller ignores a
            # new jog on an axis that has not finished, and its watch runs on)
            mine = jog is not None and jog["btn"] is btn
            btn.setEnabled(gantry_ok and allowed and not pattern
                           and (mine or (jog is None and not busy and axis not in stopping
                                         and not self._gantry.jog_stop_running
                                         and settled_still)))
        idle = gantry_ok and self._limits_applied and self._gantry_idle()
        for btn in self._teach_buttons.values():
            btn.setEnabled(idle and still and allowed)   # records a position; moves nothing
        for axis, btn in self._clear_buttons.items():
            btn.setEnabled(axis.name in self._taught and self._gantry_idle()
                           and self._limits_applied)
        self.restore_btn.setEnabled(idle and still and allowed and not self._taught
                                    and bool(self._saved.get("axes")))
        for axis, spin in self._target_spins.items():
            spin.setEnabled(gantry_ok and allowed and not busy and self._axis_absolute_ok(axis))
        self.send_btn.setEnabled(gantry_ok and allowed and not busy and not jogging and still
                                 and self._absolute_gantry_ready() and bool(self._target_dirty)
                                 and all(self._axis_absolute_ok(a) for a in self._target_dirty))
        self.recover_gantry_btn.setEnabled(self._gantry is not None and self._gantry_stuck
                                           and not self._gantry_recover_pending
                                           and not self._latched)
        enabling = (motor_ok and not self._motor.is_enabled and self._motor.mode_verified
                    and self._disabled_this_session)
        disabling = motor_connected and (self._motor.is_enabled
                                         or not self._disabled_this_session)
        # symmetric interlock: a limp mount under the jet is the hazard in BOTH
        # directions; with --no-motor-interlock Disable stays free (bench)
        disable_ok = disabling and (self._jet_off_known() or not self._options.motor_interlock)
        self.enable_btn.setEnabled(self._motor_auto is None and (
            disable_ok or (enabling and allowed and thrust_zero)))
        self.go_theta_btn.setEnabled(motor_ok and allowed and self._motor.is_enabled
                                     and not self._motor.busy and self._motor_auto is None)
        self.recover_motor_btn.setEnabled(motor_ok and self._motor_stuck and not self._latched)
        self.angle_spin.setEnabled(motor_connected)
        thrust_ok = self._thrust_allowed()
        self.thruster_set_btn.setEnabled(thrust_ok)
        self.thruster_zero_btn.setEnabled(thruster_ok and not self._latched
                                          and not self._cut_pending)
        self.thruster_spin.setEnabled(thruster_ok and not self._latched)
        blocked = self._thrust_block_reason() if self._thruster is not None else None
        if blocked:
            info = blocked
        elif not self._options.motor_interlock:
            info = "interlock OFF (--no-motor-interlock)"
        else:
            info = ""
        if self.thruster_info.text() != info:
            self.thruster_info.setText(info)
            self.thruster_info.setVisible(bool(info))
        self.thruster_set_btn.setToolTip(blocked or self._ramp_text())
        level = "warning" if blocked else "info"
        if self.thruster_info.property("level") != level:
            self.thruster_info.setProperty("level", level)
            _repolish(self.thruster_info)
        idle_now = gantry_ok and self._gantry_idle()
        self.pattern_start_btn.setEnabled(idle_now and allowed and still and not pattern)
        self.pattern_stop_btn.setEnabled(pattern)
        for wdg in (self.pattern_combo, self.pattern_size_spin, self.pattern_speed_spin):
            wdg.setEnabled(not pattern)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(self._latched)
        any_ready = gantry_ok or motor_ok or thruster_ok
        self.log_new_btn.setEnabled(any_ready)
        self.log_stop_btn.setEnabled(self._log is not None)
        self.gantry_btn.setEnabled(True)
        self.motor_btn.setEnabled(True)
        self.thruster_btn.setEnabled(True)
        self.ip_edit.setEnabled(self._gantry is None)
        self.can_edit.setEnabled(self._motor is None)
        self.thruster_edit.setEnabled(self._thruster is None)

    def _update_status(self) -> None:
        self._reap_failed_connects()
        # worker-side errors never pass through _say(): log them on change
        for name, link in (("gantry", self._gantry), ("motor", self._motor),
                           ("thruster", self._thruster)):
            err = (link.error or link.last_command_error) if link is not None else None
            if err != self._last_link_errors.get(name):
                self._last_link_errors[name] = err
                if err:
                    self._log_event("link_error", f"{name}: {err}")
        self._update_pills()
        pat = self._pattern
        info = "" if pat is None else (
            f"running {pat['label']}  ·  lap {pat['lap'] + 1}  ·  centre "
            + ", ".join(f"{a.name} {c:.1f}" for a, c in zip(pat["axes"], pat["center"])) + " mm")
        if self.pattern_info.text() != info:
            self.pattern_info.setText(info)
            self.pattern_info.setVisible(bool(info))
        if time.monotonic() < self._sticky_until:
            return
        parts = []
        severity = "info"
        if self._jog is not None:
            parts.append(f"JOG {_sign_txt(self._jog['sign'])}{self._jog['axis'].name}")
        elif self._pattern is not None:
            parts.append(f"PATTERN {self._pattern['label']}, lap {self._pattern['lap'] + 1}")
        if self._jog_release is not None:
            parts.append(f"stopping {self._jog_release['label']}…")
        for name, link in (("gantry", self._gantry), ("motor", self._motor),
                           ("thruster", self._thruster)):
            if link is None:
                parts.append(f"{name}: not connected")
                continue
            err = link.error or link.last_command_error
            if err:
                parts.append(f"{name}: {err}")
                severity = "error"
            elif not link.is_ready:
                parts.append(f"{name}: connecting…")
            else:
                parts.append(f"{name}: ok")
        gantry = self._gantry
        if gantry is not None and gantry.is_ready and self._limits_applied:
            open_sides = []
            for a in AXES:
                lo, hi = self._effective_limits(a)
                open_sides += ([f"−{a.name}"] if lo is None else []) + (
                    [f"+{a.name}"] if hi is None else [])
            if open_sides:
                parts.append(f"no home: {' '.join(open_sides)}")
                severity = "warning" if severity == "info" else severity
        if self._suspect_axes:
            parts.append(f"homes SUSPECT on {','.join(sorted(self._suspect_axes))} (a stop did "
                         "not take — Clear and re-teach)")
            severity = "error"
        if self._settings.simulator:
            parts.append("SIM")
        if self._log is not None:
            parts.append("LOG " + self._log.dir.name)
        elif any(l is not None for l in (self._gantry, self._motor, self._thruster)):
            parts.append("NOT LOGGING")
            severity = "warning" if severity == "info" else severity
        if self._cut_pending:
            parts.append("thrust cut")
        if self._latched:
            parts.append("STOP latched (press Reset)")
            severity = "error"
        self._render_status("  |  ".join(parts), severity)

    def _update_pills(self) -> None:
        def link_state(link, fresh=True):
            if link is None:
                return Conn.OFFLINE, ""
            if link.error:
                return Conn.FAULT, ""
            if not link.is_ready:
                return Conn.CONNECTING, ""
            if not fresh:
                return Conn.STALE, ""
            return Conn.ONLINE, ""
        g, m, t = self._gantry, self._motor, self._thruster
        self.gantry_pill.set_state(*link_state(
            g, fresh=(g is None or not g.is_ready or g.telemetry_fresh())))
        self.motor_pill.set_state(*link_state(
            m, fresh=(m is None or not m.is_ready or m.telemetry_fresh())))
        self.thruster_pill.set_state(*link_state(t))
        if self._log is None:
            self.log_pill.set_state(Conn.OFFLINE)
        elif self._log.failed:
            self.log_pill.set_state(Conn.FAULT)
        else:
            self.log_pill.set_state(Conn.ONLINE)

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        # a window that loses focus may never see the mouse release
        if (event.type() == QtCore.QEvent.ActivationChange and not self.isActiveWindow()
                and getattr(self, "_jog", None) is not None):
            self._end_jog("window lost focus")
        super().changeEvent(event)

    # ============================================================= closing
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        # the gantry stop is non-blocking (its own thread): send it before the
        # seconds of thruster I/O and joins below, and end any jog
        moving = (self._jog is not None or self._jog_release is not None
                  or self._pattern is not None or self._gantry_cmd_in_flight
                  or (self._gantry is not None and (
                      self._gantry.busy or (self._latched and not self._all_at_rest())
                      or (self._rest["pos"] is not None and not self._all_at_rest()))))
        self._jog = None
        self._pattern = None
        if self._gantry is not None:
            try:
                self._gantry.stop_all(wait=False)
            except Exception:  # noqa: BLE001
                pass
            if moving:
                print("\n*** gantry stop sent at close, NOT verified (the window is gone) — "
                      "watch the axes; cut the gantry power if one keeps moving ***\n",
                      file=sys.stderr, flush=True)
            try:
                self._save_homes()
            except Exception:  # noqa: BLE001
                pass
        for timer in (self._readout_timer, self._status_timer):
            timer.stop()
        # zero the thruster FIRST, while the link still records, so the 0 is
        # the last row of thruster.csv and samples.csv (stop() alone would send
        # it after stop_recording and the log would end on the running value)
        if self._thruster is not None:
            try:
                self._thruster.stop_all()
                self._note_zero_outcome(self._thruster, "close")
            except Exception:  # noqa: BLE001
                pass
        if self._log is not None:
            try:
                self._latched = True
                self._log_sample()
                self._log_event("close", "window")
                self._stop_log()
            except Exception:  # noqa: BLE001
                pass
        # then close the thruster port (zero again), the gantry, and the motor
        # (keeps holding — the mount must not swing when the window closes)
        if self._thruster is not None:
            self._thruster.stop()
            self._thruster = None
        if self._gantry is not None:
            self._gantry.stop()
            self._gantry = None
        if self._motor is not None:
            self._motor.stop()
            self._motor = None
        super().closeEvent(event)


# ================================================================== CLI
def build_arg_parser(cfg: "core.RigConfig | None" = None) -> argparse.ArgumentParser:
    ap = core.build_arg_parser(cfg)
    PF.add_preflight_args(ap)
    return ap


def main(argv: "list[str] | None" = None) -> int:
    if WHISKER_IMPORT_ERROR:
        print(f"disturbance_panel: {WHISKER_IMPORT_ERROR}", file=sys.stderr)
        return 2
    cfg = core.load_rig_config(core.rig_config_path_from_argv(argv))
    args = build_arg_parser(cfg).parse_args(argv)
    settings = core.settings_from_args(args, cfg)
    options = core.options_from_args(args, cfg)
    rc = PF.gate(args, title="disturbance rig", specs=PF.disturb_checks(
        sim=settings.simulator, whisker_root=WHISKER_ROOT, gantry_ip=settings.gantry_ip,
        gantry_port=settings.gantry_port, can_channel=settings.can_channel,
        thruster_port=settings.thruster_port, data_root=options.data_root))
    if rc is not None:
        return rc
    report = getattr(args, "preflight_report", None)
    preflight = report.to_dict() if report is not None else None
    qt.sanitize_plugin_path()
    qt.preload_platform_libs()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app)                                    # the ROV station's look
    app.setStyleSheet(app.styleSheet() + EXTRA_QSS)
    window = DisturbanceWindow(settings, options, WHISKER_ROOT, preflight)
    window.show()
    qt.install_slot_guard(window._on_ui_fault)

    def request_shutdown(_signum, _frame) -> None:
        app.quit()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGHUP"):        # the launching terminal / ssh session went away
        signal.signal(signal.SIGHUP, request_shutdown)
    try:
        return qt.run_app(app)
    finally:
        window.close()


if __name__ == "__main__":
    raise SystemExit(main())
