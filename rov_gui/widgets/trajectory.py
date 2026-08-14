#!/usr/bin/env python3
"""trajectory.py — the MPC panel: engage controls + reference-vs-actual plot.

Embedded in the main grid (window.py) — since 2026-08-14 in the BOTTOM ROW
spanning columns 2-3 (the old PROPULSION/SENSORS slots, which moved to column
3 under SYSTEM HEALTH), per operator request: the plot is the thing watched
during an experiment, so it gets the wide slot.

The plot is a top-down map of the NED tag-world (screen up = +x_ned, screen
right = +y_ned — the frame the pool is measured in), drawn with the house
QPainter pattern (no pyqtgraph/matplotlib, deliberately — indicators.py:5).
It shows the geofence box, the placed reference square, the reference and
actual trails, and the vehicle's heading. Everything numeric it prints comes
FROM MpcStatus/NavFix — this window computes nothing, so the plot and the CSV
can never disagree.

Engage discipline mirrors the ARM button: ENGAGE is a hold-to-confirm (the
vehicle starts holding position the moment it fires), DISENGAGE and STOP are
single clicks — a stop must never be gated (teleop.py's rule).
"""

from __future__ import annotations

import math
import time
from collections import deque

from .. import theme
from ..qt import QColor, QPainter, Qt, QtCore, QtGui, QtWidgets, Signal
from ..state import MpcStatus, NavFix
from .indicators import HoldToConfirmButton

TRAIL_MAX = 4800          # hard cap on stored points (memory bound)
TRAIL_AGE_S = 90.0        # points older than this fade out and are dropped


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


# Viridis anchor points (matplotlib's default perceptual ramp), so the GUI
# trail and the offline plot (rov_gui/tools/plot_nav_run.py, real viridis)
# read the same: dark purple = oldest, bright yellow = newest.
_VIRIDIS = ((68, 1, 84), (59, 82, 139), (33, 145, 140),
            (94, 201, 98), (253, 231, 37))


def _age_color(u: float, alpha: int = 255) -> QColor:
    """u in [0, 1]: 0 = oldest (about to disappear), 1 = newest."""
    u = max(0.0, min(1.0, u))
    seg = u * (len(_VIRIDIS) - 1)
    i = min(int(seg), len(_VIRIDIS) - 2)
    f = seg - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    col = QColor(int(a[0] + f * (b[0] - a[0])),
                 int(a[1] + f * (b[1] - a[1])),
                 int(a[2] + f * (b[2] - a[2])))
    # old points also fade toward transparent so the tail vanishes smoothly
    col.setAlpha(int(alpha * (0.25 + 0.75 * u)))
    return col


class TrajectoryView(QtWidgets.QWidget):
    """Top-down NED map. Fixed world, movable eye (pan/zoom), like pose3d."""

    def __init__(self):
        super().__init__()
        self.geofence: dict | None = None       # {"x":[..],"y":[..],...} NED
        self.map_tags: list = []                # [(x, y, yaw, id, inst), ...]
        self.tag_size_m = 0.170                 # drawn edge; window sets from cfg
        self.pool: list | None = None           # 4 corners [(x,y),...] NED
        self.square_ned: list | None = None     # 4 corners [(x,y),...]
        self.trail_act: deque = deque(maxlen=TRAIL_MAX)   # (t, x, y) NED
        self.trail_ref: deque = deque(maxlen=TRAIL_MAX)
        self.p_act = None                       # (x, y, z) NED
        self.p_ref = None
        # (id, instance) of every tag in the LATEST accepted fix. The INSTANCE
        # matters: a duplicated id has two squares on the floor and only one of
        # them carried the fix — lighting both would be a lie about the very
        # thing the operator is watching for.
        self.used_ids: frozenset = frozenset()
        self._used_t = 0.0
        self.yaw_ned = None
        self.fix_hz = None
        self.fix_det_ms = 0.0
        self.fix_src = ""                       # which feed localizes
        self.status: MpcStatus | None = None
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)     # screen px
        self._drag_from = None
        # Tiny minimum, deliberately: this view lives IN the main grid
        # (column 3, under SYSTEM HEALTH) and its minimum is a floor for
        # extreme shrink only — at the operator's real screen size it takes
        # the column's stretch space. A useful-looking minimum here would
        # push the whole window's minimum past a laptop screen.
        self.setMinimumSize(200, 110)
        self.setToolTip(
            "NED top-down: screen up = +x, screen right = +y\n"
            "drag to pan · mouse wheel to zoom · double-click to reset the view")

    # ------------------------------------------------------------- data in
    def set_geofence(self, box: dict) -> None:
        self.geofence = box
        self.update()

    def set_pool(self, corners: list | None) -> None:
        """The pool boundary's 4 corners (x, y) NED, ALREADY in the plot's
        frame (the window applies the engage datum). Also sets the view scale:
        _fit prefers the pool over the geofence when it exists."""
        self.pool = list(corners) if corners else None
        self.update()

    def set_map_tags(self, pts: list) -> None:
        """The tag map's (x, y, yaw, id, instance) entries, ALREADY in the
        frame the rest of the plot uses (the window applies the engage datum).
        yaw is the tag's in-plane rotation, so the square is drawn the way the
        tag actually lies on the floor — the old tagslam visualization. A
        duplicated id contributes one entry PER physical copy."""
        self.map_tags = list(pts)
        self.update()

    @staticmethod
    def _trail_push(trail: deque, x: float, y: float, eps: float) -> None:
        if not trail or (abs(trail[-1][1] - x) > eps
                         or abs(trail[-1][2] - y) > eps):
            trail.append((time.monotonic(), x, y))

    def _prune_trails(self) -> None:
        cut = time.monotonic() - TRAIL_AGE_S
        for trail in (self.trail_act, self.trail_ref):
            while trail and trail[0][0] < cut:
                trail.popleft()

    def add_fix(self, f: NavFix) -> None:
        """Localizer health readout + WHICH tags carried the fix, and — while
        the MPC worker's state stream is NOT driving the plot (not engaged,
        e.g. bench runs with no ArduSub telemetry) — the marker itself. Once
        engaged, MpcStatus (20 Hz, velocity-bridged, same datum frame: the
        window transformed this fix already) takes over so the marker moves
        at the control rate."""
        if not f.ok:
            return
        self.fix_hz = f.hz
        self.fix_det_ms = f.detect_ms
        self.fix_src = f.source or self.fix_src
        insts = f.tag_insts or ((0,) * len(f.tag_ids))
        self.used_ids = frozenset(
            (int(i), int(k)) for i, k in zip(f.tag_ids, insts))
        self._used_t = time.monotonic()
        engaged = self.status is not None and self.status.engaged
        if not engaged:
            x, y = float(f.p_ned[0]), float(f.p_ned[1])
            self._trail_push(self.trail_act, x, y, 1e-3)
            self.p_act = tuple(float(v) for v in f.p_ned)
            if f.yaw_ned is not None:
                self.yaw_ned = float(f.yaw_ned)
        self._prune_trails()
        self.update()

    def add_status(self, s: MpcStatus) -> None:
        self.status = s
        if s.p_flu is not None:
            # FLU -> NED mirror (the map frame the pool is measured in)
            x, y, z = s.p_flu[0], -s.p_flu[1], -s.p_flu[2]
            self._trail_push(self.trail_act, x, y, 1e-3)
            self.p_act = (x, y, z)
            if s.yaw_flu_deg is not None:
                self.yaw_ned = -math.radians(s.yaw_flu_deg)
        if s.ref_flu is not None:
            # FLU -> NED mirror for display (the map frame the pool is
            # measured in): (x, -y, -z).
            r = (s.ref_flu[0], -s.ref_flu[1], -s.ref_flu[2])
            self.p_ref = r
            if s.engaged:
                self._trail_push(self.trail_ref, r[0], r[1], 1e-4)
        if s.scenario and s.scenario.get("kind") == "square":
            self.square_ned = self._square_corners(s.scenario)
        elif not s.traj_on:
            self.square_ned = self.square_ned if s.engaged else None
        self._prune_trails()
        self.update()

    @staticmethod
    def _square_corners(sc: dict) -> list:
        """The placed square's corners in NED (mirror of the FLU generator)."""
        from ..control.reference import square_corners_world

        ox, oy = sc["origin_ned"]
        rot_flu = -math.radians(sc.get("rot_deg", 0.0))
        flu = square_corners_world(sc["size"], (ox, -oy), rot_flu)
        return [(x, -y) for x, y in flu]

    def clear(self) -> None:
        self.trail_act.clear()
        self.trail_ref.clear()
        self.update()

    # ---------------------------------------------------------- projection
    def _fit(self) -> tuple[float, float, float]:
        """(px_per_m, cx_ned, cy_ned). The POOL sets the scale when it is
        known (operator request 2026-08-14: the boundary is the pool and the
        axes are scaled to it); otherwise the geofence, otherwise a 4 m box."""
        if self.pool:
            xs = [c[0] for c in self.pool]
            ys = [c[1] for c in self.pool]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        elif self.geofence:
            x0, x1 = self.geofence["x"]
            y0, y1 = self.geofence["y"]
        else:
            x0, x1, y0, y1 = -2.0, 2.0, -2.0, 2.0
        w = max(0.5, y1 - y0)
        h = max(0.5, x1 - x0)
        s = 0.85 * min(self.width() / w, self.height() / h) * self.zoom
        return s, (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def _px(self, x_ned: float, y_ned: float) -> QtCore.QPointF:
        s, cx, cy = self._fit()
        return QtCore.QPointF(
            self.width() / 2.0 + (y_ned - cy) * s + self.pan.x(),
            self.height() / 2.0 - (x_ned - cx) * s + self.pan.y())

    def _world_bounds(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the NED world visible right now —
        the exact inverse of _px, so grid lines span the viewport at any
        pan/zoom instead of only the geofence."""
        s, cx, cy = self._fit()
        s = max(1e-6, s)
        w, h = self.width(), self.height()
        y_min = (0 - w / 2.0 - self.pan.x()) / s + cy
        y_max = (w - w / 2.0 - self.pan.x()) / s + cy
        x_min = (h / 2.0 + self.pan.y() - h) / s + cx
        x_max = (h / 2.0 + self.pan.y() - 0) / s + cx
        return x_min, x_max, y_min, y_max

    # --------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        self._drag_from = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._drag_from is None:
            return
        d = ev.pos() - self._drag_from
        self._drag_from = ev.pos()
        self.pan += QtCore.QPointF(d.x(), d.y())
        self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag_from = None

    def wheelEvent(self, ev):
        step = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.3, min(8.0, self.zoom * step))
        self.update()

    def mouseDoubleClickEvent(self, _ev):
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0.0, 0.0)
        self.update()

    # --------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _c(theme.VIDEO_BG))

        def poly(points, pen, close=False):
            if len(points) < 2:
                return
            path = QtGui.QPainterPath()
            path.moveTo(self._px(*points[0]))
            for q in points[1:]:
                path.lineTo(self._px(*q))
            if close:
                path.closeSubpath()
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # Adaptive metre grid across the whole viewport, with tick labels:
        # values of x_ned up the LEFT edge (horizontal lines are constant-x),
        # values of y_ned along the BOTTOM edge. The step is chosen so ticks
        # never crowd (>= 48 px apart) at any zoom.
        s, _cx, _cy = self._fit()
        xw0, xw1, yw0, yw1 = self._world_bounds()
        step = next((c for c in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
                     if c * s >= 48.0), 10.0)
        grid_pen = QtGui.QPen(_c(theme.TEXT_FAINT, 34), 1)
        p.setFont(QtGui.QFont("monospace", 8))
        gx = math.floor(xw0 / step) * step
        while gx <= xw1 + 1e-9:
            poly([(gx, yw0), (gx, yw1)], grid_pen)
            q = self._px(gx, yw0)
            p.setPen(_c(theme.TEXT_DIM, 170))
            p.drawText(4, int(q.y()) + 3, f"{gx:g}")
            gx += step
        gy = math.floor(yw0 / step) * step
        while gy <= yw1 + 1e-9:
            poly([(xw0, gy), (xw1, gy)], grid_pen)
            q = self._px(xw0, gy)
            p.setPen(_c(theme.TEXT_DIM, 170))
            p.drawText(int(q.x()) + 2, self.height() - 18, f"{gy:g}")
            gy += step
        # Axis captions, in the corners the ticks accumulate toward.
        p.setPen(_c(theme.TEXT_DIM))
        p.drawText(4, 12, "x [m] ↑N")
        p.drawText(self.width() - 52, self.height() - 18, "y [m] →")

        # Pool boundary: the physical wall, drawn solid — the one line on this
        # plot the vehicle must never cross. Placement is config (hw_nav.yaml
        # pool_ned, [예측] until the map->wall offsets are taped).
        if self.pool:
            poly(self.pool, QtGui.QPen(_c(theme.TEXT_DIM, 220), 2), close=True)
            q = self._px(*self.pool[0])
            p.setPen(_c(theme.TEXT_FAINT, 180))
            p.drawText(int(q.x()) + 4, int(q.y()) - 4, "POOL")

        # Geofence box — ONLY while engaged, and that is a correctness point
        # rather than a decluttering one: the fence is DATUM-relative (it is
        # "±1.2 m around wherever START was pressed"), so before an engagement
        # there is no datum and drawing it around the map origin would put the
        # box somewhere the fence will never be.
        if self.geofence and self.status is not None and self.status.engaged:
            x0, x1 = self.geofence["x"]
            y0, y1 = self.geofence["y"]
            poly([(x0, y0), (x0, y1), (x1, y1), (x1, y0)],
                 QtGui.QPen(_c(theme.WARN, 120), 1, Qt.PenStyle.DashLine),
                 close=True)

        # The surveyed tag map, tagslam-viz style: each tag an oriented square
        # at its true printed size; the tags carrying the CURRENT fix fill
        # green, the rest stay faint. Highlight decays with the fix (1 s), so
        # a dead localizer cannot keep tags lit. Ids once zoomed in enough for
        # the text to be legible rather than confetti.
        if self.map_tags:
            show_ids = s * self.tag_size_m > 12.0   # tag edge spans >= 12 px
            fresh = (time.monotonic() - self._used_t) < 1.0
            half = self.tag_size_m / 2.0
            p.setFont(QtGui.QFont("monospace", 7))
            for tx, ty, tyaw, tid, tinst in self.map_tags:
                c, sn = math.cos(tyaw), math.sin(tyaw)
                corners = [(tx + c * dx - sn * dy, ty + sn * dx + c * dy)
                           for dx, dy in ((-half, -half), (-half, half),
                                          (half, half), (half, -half))]
                path = QtGui.QPainterPath()
                path.moveTo(self._px(*corners[0]))
                for q2 in corners[1:]:
                    path.lineTo(self._px(*q2))
                path.closeSubpath()
                used = fresh and (int(tid), int(tinst)) in self.used_ids
                if used:
                    p.setPen(QtGui.QPen(_c(theme.OK), 1))
                    p.setBrush(_c(theme.OK, 140))
                else:
                    p.setPen(QtGui.QPen(_c(theme.TEXT_FAINT, 120), 1))
                    p.setBrush(_c(theme.TEXT_DIM, 45))
                p.drawPath(path)
                if show_ids or used:
                    q = self._px(tx, ty)
                    p.setPen(_c(theme.TEXT, 220) if used
                             else _c(theme.TEXT_FAINT, 160))
                    p.drawText(int(q.x()) + 4, int(q.y()) + 3, str(tid))

        # The placed reference square.
        if self.square_ned:
            poly(self.square_ned, QtGui.QPen(_c(theme.TEXT_DIM, 160), 1,
                                             Qt.PenStyle.DotLine), close=True)

        # Trails. Reference: plain blue. Actual: colored by AGE (viridis,
        # dark purple = oldest, yellow = newest) so time order is readable on
        # a path that crosses itself; points older than TRAIL_AGE_S were
        # pruned on the way in. Drawn in chunks — one pen per ~METERED span —
        # so the cost stays one path per color, not one line per sample.
        self._prune_trails()
        poly([(x, y) for _t, x, y in self.trail_ref],
             QtGui.QPen(_c("#4d9dff", 150), 1))
        act = list(self.trail_act)
        if len(act) >= 2:
            t_now = time.monotonic()
            chunk = max(2, len(act) // 48 + 1)
            for i0 in range(0, len(act) - 1, chunk):
                seg = act[i0:i0 + chunk + 1]     # overlap 1 pt = no gaps
                if len(seg) < 2:
                    continue
                age = t_now - seg[len(seg) // 2][0]
                u = 1.0 - min(1.0, age / TRAIL_AGE_S)
                poly([(x, y) for _t, x, y in seg],
                     QtGui.QPen(_age_color(u), 2))

        # Current reference: a cross.
        if self.p_ref is not None:
            q = self._px(self.p_ref[0], self.p_ref[1])
            p.setPen(QtGui.QPen(_c("#4d9dff"), 2))
            p.drawLine(q + QtCore.QPointF(-6, 0), q + QtCore.QPointF(6, 0))
            p.drawLine(q + QtCore.QPointF(0, -6), q + QtCore.QPointF(0, 6))

        # The vehicle: dot + heading arrow (NED yaw 0 = +x = screen up).
        if self.p_act is not None:
            q = self._px(self.p_act[0], self.p_act[1])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_c(theme.OK))
            p.drawEllipse(q, 5, 5)
            if self.yaw_ned is not None:
                a = float(self.yaw_ned)
                tip = self._px(self.p_act[0] + 0.35 * math.cos(a),
                               self.p_act[1] + 0.35 * math.sin(a))
                p.setPen(QtGui.QPen(_c(theme.OK), 2))
                p.drawLine(q, tip)

        # Error line between them.
        if self.p_act is not None and self.p_ref is not None:
            p.setPen(QtGui.QPen(_c(theme.FAIL, 130), 1, Qt.PenStyle.DashLine))
            p.drawLine(self._px(self.p_act[0], self.p_act[1]),
                       self._px(self.p_ref[0], self.p_ref[1]))

        # Corner readout — numbers from MpcStatus only (module docstring).
        # Starts below the axis caption so the two never overprint.
        p.setFont(QtGui.QFont("monospace", 8))
        p.setPen(_c(theme.TEXT_DIM))
        s = self.status
        lines = []
        if self.p_act is not None:
            lines.append(f"pos NED ({self.p_act[0]:+.2f}, {self.p_act[1]:+.2f}, "
                         f"{self.p_act[2]:+.2f}) m")
            hz = "--" if self.fix_hz is None else f"{self.fix_hz:.0f}"
            src = {"main": "C3 RGB", "second": "DEFAULT RGB"}.get(
                self.fix_src, self.fix_src or "--")
            lines.append(f"loc {src} · fix {hz} Hz · detect "
                         f"{self.fix_det_ms:.0f} ms")
        if s is not None:
            if s.err_xy is not None:
                lines.append(f"err_xy {s.err_xy * 100:5.1f} cm   lap {s.lap}")
            solve = ("--" if s.solve_ms is None else f"{s.solve_ms:.1f}")
            lines.append(f"{s.mode} {s.solver} solve {solve} ms "
                         f"fail {s.n_fail}")
            lines.append(f"tags {s.n_tags}  age "
                         + ("--" if s.tag_age_s is None else f"{s.tag_age_s:.2f}s")
                         + ("  GEOFENCE!" if not s.geofence_ok else ""))
            if len(s.w_hat) == 6:
                w = s.w_hat
                lines.append(f"w_hat [{w[0]:+.1f} {w[1]:+.1f} {w[2]:+.1f}] N "
                             f"[{w[3]:+.1f} {w[4]:+.1f} {w[5]:+.1f}] Nm")
            if s.reason:
                lines.append(s.reason)
        for i, ln in enumerate(lines):
            p.drawText(8, 26 + 12 * i, ln)
        # (No frame/mouse caption: the axes are labelled at their own ends and
        # the mouse gestures are on the widget's tooltip, not burned into the
        # picture — operator request 2026-08-14.)

        # Trail-age legend, bottom-right: the color ramp with its time span.
        # Laid out from the RIGHT EDGE inwards so the "now" label cannot fall
        # off the widget at small widths.
        fm = p.fontMetrics()
        old_lbl, new_lbl = f"-{TRAIL_AGE_S:.0f}s", "now"
        bar_w, bar_h = 56, 5
        bx = (self.width() - 6 - fm.horizontalAdvance(new_lbl) - 3 - bar_w)
        by = self.height() - 14
        for i in range(bar_w):
            p.setPen(_age_color(i / (bar_w - 1)))
            p.drawLine(bx + i, by, bx + i, by + bar_h)
        p.setPen(_c(theme.TEXT_FAINT, 190))
        p.drawText(bx - 4 - fm.horizontalAdvance(old_lbl), by + bar_h, old_lbl)
        p.drawText(bx + bar_w + 3, by + bar_h, new_lbl)


class TrajectoryWindow(QtWidgets.QWidget):
    """Controls + the view, now DOCKED in the main window (window.py builds a
    QDockWidget around it). Emits requests; never flips its own state — every
    label reflects what MpcStatus says actually happened, the same honesty
    rule the ARM button follows.

    Two ways to begin, deliberately:
      START (hold)  the one-button mission — engage, warm up, fly the square,
                    with the CSV recording open from the engage. What the
                    operator asked for: press once, it goes and it logs.
      HOLD  (hold)  engage only (DP hold) — the calibration / station-keeping
                    mode, and the safe place to stage before a manual START.
    STOP / DISENGAGE / E-STOP are single clicks: a stop is never gated."""

    engage_requested = Signal(bool)
    traj_requested = Signal(bool)
    mission_requested = Signal()
    mode_requested = Signal(str)
    estop_requested = Signal()
    # Raw localization recording (map.json + fixes.csv), independent of the
    # engage-gated MPC CSV: the operator can log a hand-flown survey pass.
    record_requested = Signal(bool)

    _BTN_CSS = "QPushButton{font-size:11px; padding:3px 8px;}"

    def __init__(self, geofence: dict | None = None):
        super().__init__()
        self.setWindowTitle("MPC — reference vs actual (NED tag world)")
        self.view = TrajectoryView()
        if geofence:
            self.view.set_geofence(geofence)

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["dobmpc", "mpc", "pid"])
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_box.currentTextChanged.connect(self.mode_requested)

        self.btn_start = HoldToConfirmButton(
            "START", hold_ms=1000,
            tooltip="hold: engage, warm up, fly the square — recording (CSV) "
                    "starts at engage")
        self.btn_start.setStyleSheet(self._BTN_CSS)
        self.btn_start.confirmed.connect(self.mission_requested)
        self.btn_hold = HoldToConfirmButton(
            "HOLD", hold_ms=1000,
            tooltip="hold: engage only — MPC holds the current pose (DP)")
        self.btn_hold.setStyleSheet(self._BTN_CSS)
        self.btn_hold.confirmed.connect(
            lambda: self.engage_requested.emit(True))
        self.btn_release = QtWidgets.QPushButton("DISENG")
        self.btn_release.setObjectName("Danger")
        self.btn_release.setStyleSheet(self._BTN_CSS)
        self.btn_release.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_release.setToolTip("instant: back to pilot control")
        self.btn_release.clicked.connect(
            lambda: self.engage_requested.emit(False))
        self.btn_stop = QtWidgets.QPushButton("STOP TRAJ")
        self.btn_stop.setStyleSheet(self._BTN_CSS)
        self.btn_stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_stop.setToolTip("instant: abandon the square, hold here (DP)")
        self.btn_stop.clicked.connect(lambda: self.traj_requested.emit(False))
        self.btn_estop = QtWidgets.QPushButton("E-STOP")
        self.btn_estop.setObjectName("Danger")
        self.btn_estop.setStyleSheet(self._BTN_CSS)
        self.btn_estop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_estop.setToolTip("zero all axes, disable transmission — the "
                                  "same E-STOP as the header (Esc)")
        self.btn_estop.clicked.connect(self.estop_requested)
        clear = QtWidgets.QPushButton("CLR")
        clear.setStyleSheet(self._BTN_CSS)
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.setToolTip("clear the trails")
        clear.clicked.connect(self.view.clear)
        self.btn_rec = QtWidgets.QPushButton("REC NAV")
        self.btn_rec.setObjectName("Rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setStyleSheet(self._BTN_CSS)
        self.btn_rec.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_rec.setToolTip("record raw localization: tag map + every fix "
                                "to sessions/nav_runs/<stamp>/ — replot later "
                                "with  python -m rov_gui.tools.plot_nav_run")
        self.btn_rec.toggled.connect(self.record_requested)

        self.chip = QtWidgets.QLabel("not engaged")
        self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(self.mode_box)
        row1.addWidget(self.btn_start, 1)
        row1.addWidget(self.btn_hold)
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)
        row2.addWidget(self.btn_stop)
        row2.addWidget(self.btn_release)
        row2.addWidget(self.btn_estop)
        row2.addWidget(clear)
        row2.addWidget(self.btn_rec)
        row2.addStretch(1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.chip)
        bottom.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.view, 1)
        lay.addLayout(bottom)
        self._traj_on = False
        self._engaged = False

    def set_recording(self, on: bool) -> None:
        """Reflect what the window's recorder ACTUALLY did (honesty rule: the
        button never shows a recording that failed to open)."""
        self.btn_rec.blockSignals(True)
        self.btn_rec.setChecked(on)
        self.btn_rec.setText("REC ●" if on else "REC NAV")
        self.btn_rec.blockSignals(False)

    # ------------------------------------------------------------- data in
    def add_fix(self, f: NavFix) -> None:
        self.view.add_fix(f)
        # A localizer that is FAILING explains itself here (full reason — the
        # sensor row elides): outside an engagement the chip has nothing more
        # important to say.
        if not f.ok and f.note and not self._engaged:
            self.chip.setText(f"nav: {f.note}")
            self.chip.setStyleSheet(
                f"color: {theme.WARN}; font-size: 11px;")

    def add_status(self, s: MpcStatus) -> None:
        self.view.add_status(s)
        self._engaged = s.engaged
        self._traj_on = s.traj_on
        self.btn_start.setEnabled(not s.traj_on)
        self.btn_hold.setEnabled(not s.engaged)
        self.btn_stop.setEnabled(s.traj_on)
        self.mode_box.setEnabled(not s.engaged)
        if s.engaged:
            warm = (f" · warm-up {s.warmup_left_s:.1f}s"
                    if s.warmup_left_s > 0 else "")
            what = "SQUARE" if s.traj_on else "DP HOLD"
            err = "" if s.err_xy is None else f" · err {s.err_xy * 100:.0f} cm"
            self.chip.setText(f"ENGAGED [{s.mode}] {what}{err}{warm}")
            self.chip.setStyleSheet(
                f"color: {theme.WARN}; font-size: 11px; font-weight: 600;")
        else:
            self.chip.setText(s.reason or "not engaged")
            self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
