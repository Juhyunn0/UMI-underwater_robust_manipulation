#!/usr/bin/env python3
"""
pose3d.py — a 3-D view of where the tracked object is, in the camera's frame.

A separate top-level window, not a panel. The main window has about five
pixels of vertical headroom against its one-screen promise (window.py), so a
new panel is impossible — and a 3-D plot is something the operator glances at
occasionally, not an instrument that must always be visible. It opens by
itself when FoundationPose comes up, and the MAP button on the C3 RGB panel
brings it back if it was closed.

What it draws is deliberately literal: the CAMERA sits at the origin looking
along +Z, with OpenCV's optical axes (+X right, +Y DOWN, +Z forward), because
that is exactly the frame ``T_cam_obj`` is expressed in — the plot and the
POSE log rows must never disagree about what a coordinate means. The plotted
volume is FIXED at x,y ∈ [−1, +1] m and z ∈ [0, 2] m: pose only works out to
about 2 m (past it the stereo noise exceeds what registration tolerates), so
an auto-scaling view would spend its time re-zooming on noise while making
equal distances look different from one glance to the next. The view — not
the volume — is what the mouse changes: drag orbits, wheel zooms, and a
double click puts the default view back.

No OpenGL, no extra dependency: at a few thousand points a QPainter polyline
is cheap, and this repaints only when a new pose arrives (10 Hz), not on the
UI clock.
"""

from __future__ import annotations

import math
from collections import deque

from .. import theme
from ..qt import QColor, QPainter, QRectF, Qt, QtCore, QtGui, QtWidgets
from ..state import PoseTrack

# The fixed plotted volume, metres, camera-optical axes.
RANGE_XY = 1.0            # x, y ∈ [-RANGE_XY, +RANGE_XY]
RANGE_Z = 2.0             # z ∈ [0, RANGE_Z] — pose does not exist past ~2 m
TRAIL_MAX = 3000          # ~5 min of 10 Hz poses
ZOOM_MIN, ZOOM_MAX = 0.3, 8.0


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


class Pose3DView(QtWidgets.QWidget):
    """The canvas: fixed world, movable eye.

    The projection is a plain orbit: yaw about the world's vertical, then
    pitch, then orthographic drop of the depth axis. Orthographic rather than
    perspective on purpose — with a fixed 2 m volume there is nothing for
    foreshortening to add except the inability to compare two distances by
    eye.
    """

    def __init__(self):
        super().__init__()
        # Default eye: slightly above and to the side, the classic 3/4 view.
        self.yaw = math.radians(-35.0)
        self.pitch = math.radians(-25.0)
        self.zoom = 1.0
        self.trail: deque = deque(maxlen=TRAIL_MAX)   # (x, y, z) metres
        self.pose_T: tuple | None = None              # latest 16 floats, or None
        self.state = "off"
        self._drag_from = None
        self.setMinimumSize(320, 300)
        self.setMouseTracking(False)

    # ------------------------------------------------------------- data in
    def add(self, t: PoseTrack) -> None:
        """One published track. Appends only when there is a NEW pose."""
        self.state = t.state
        if t.T_cam_obj is None:
            self.pose_T = None
            self.update()
            return
        T = t.T_cam_obj
        p = (T[3], T[7], T[11])
        # 10 Hz republishes the same pose while the estimator is between
        # frames; a trail of duplicates is length without information.
        if not self.trail or self.trail[-1] != p:
            self.trail.append(p)
        self.pose_T = T
        self.update()

    def clear(self) -> None:
        self.trail.clear()
        self.pose_T = None
        self.update()

    # ---------------------------------------------------------- projection
    def _project(self, x: float, y: float, z: float):
        """World metres (camera-optical) -> widget px, via the orbit eye.

        The world's OWN vertical is -Y (OpenCV +Y is down), so the eye yaws
        about Y and the screen's up is world -Y at zero orbit — the mesh's
        "up" on the video and "up" in this plot agree by construction.
        """
        cz = z - RANGE_Z / 2.0        # orbit about the volume centre, not the camera
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        # yaw about world Y
        x1 = cy * x + sy * cz
        z1 = -sy * x + cy * cz
        # pitch about world X
        y2 = cp * y - sp * z1
        # orthographic: drop the rotated depth
        s = self._px_per_m()
        w, h = self.width(), self.height()
        return (w / 2.0 + x1 * s, h / 2.0 + y2 * s)

    def _px_per_m(self) -> float:
        # The whole fixed volume fits at zoom 1, with a margin.
        return self.zoom * min(self.width(), self.height()) / (2.0 * RANGE_Z * 0.62)

    # --------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        self._drag_from = ev.pos()

    def mouseMoveEvent(self, ev):
        if self._drag_from is None:
            return
        d = ev.pos() - self._drag_from
        self._drag_from = ev.pos()
        self.yaw += d.x() * 0.008
        self.pitch = max(-1.5, min(1.5, self.pitch + d.y() * 0.008))
        self.update()

    def mouseReleaseEvent(self, _ev):
        self._drag_from = None

    def wheelEvent(self, ev):
        step = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * step))
        self.update()

    def mouseDoubleClickEvent(self, _ev):
        self.yaw = math.radians(-35.0)
        self.pitch = math.radians(-25.0)
        self.zoom = 1.0
        self.update()

    # --------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _c(theme.VIDEO_BG))

        def line3(a, b, pen):
            p.setPen(pen)
            ax, ay = self._project(*a)
            bx, by = self._project(*b)
            p.drawLine(QtCore.QPointF(ax, ay), QtCore.QPointF(bx, by))

        # The fixed volume: a wireframe box x,y ±1 m, z 0..2 m, faint.
        box_pen = QtGui.QPen(_c(theme.TEXT_FAINT, 70), 1)
        R, Z = RANGE_XY, RANGE_Z
        corners = [(sx * R, sy * R, z) for z in (0.0, Z)
                   for sy in (-1, 1) for sx in (-1, 1)]
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7),
                     (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            line3(corners[a], corners[b], box_pen)
        # Depth rings on the floor of the volume every 0.5 m, so "how far out
        # is it" can be read without any mental arithmetic.
        ring_pen = QtGui.QPen(_c(theme.TEXT_FAINT, 45), 1)
        for z in (0.5, 1.0, 1.5):
            for a, b in (((-R, R, z), (R, R, z)),):
                line3(a, b, ring_pen)

        # Camera axes at the origin: the frame every number in the POSE log is
        # expressed in. Same colours as the video overlay (X red, Y green DOWN,
        # Z blue forward), so the two views cannot be read differently.
        for axis, col in (((0.5, 0, 0), "#ff4d4d"), ((0, 0.5, 0), "#4dff4d"),
                          ((0, 0, 2.0), "#4d9dff")):
            line3((0, 0, 0), axis, QtGui.QPen(QColor(col), 2))
        p.setFont(QtGui.QFont("monospace", 8))
        for pos, label in (((0.55, 0, 0), "X 0.5m"), ((0, 0.58, 0), "Y 0.5m"),
                           ((0, 0, 2.05), "Z 2m")):
            x, y = self._project(*pos)
            p.setPen(_c(theme.TEXT_DIM))
            p.drawText(int(x), int(y), label)

        # Trail: everywhere the object has been this session.
        if len(self.trail) >= 2:
            path = QtGui.QPainterPath()
            pts = list(self.trail)
            path.moveTo(*self._project(*pts[0]))
            for q in pts[1:]:
                path.lineTo(*self._project(*q))
            p.setPen(QtGui.QPen(_c(theme.ACCENT, 150), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # The object now: its own axes triad (10 cm), from the pose rotation,
        # so orientation is visible, not just position.
        if self.pose_T is not None:
            T = self.pose_T
            o = (T[3], T[7], T[11])
            ox, oy = self._project(*o)
            for i, col in ((0, "#ff4d4d"), (1, "#4dff4d"), (2, "#4d9dff")):
                tip = (o[0] + 0.1 * T[0 + i], o[1] + 0.1 * T[4 + i],
                       o[2] + 0.1 * T[8 + i])
                line3(o, tip, QtGui.QPen(QColor(col), 2))
            p.setPen(_c(theme.TEXT))
            d = math.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
            p.drawText(int(ox) + 8, int(oy) - 8,
                       f"{o[0]:+.2f} {o[1]:+.2f} {o[2]:+.2f} m   d {d:.2f}")

        # Corner status: what the numbers mean, and that the view is fixed.
        p.setFont(QtGui.QFont("monospace", 8))
        p.setPen(_c(theme.TEXT_DIM))
        p.drawText(8, self.height() - 22,
                   f"camera frame, +Z forward, +Y down · fixed volume "
                   f"x,y ±{RANGE_XY:.0f} m, z 0-{RANGE_Z:.0f} m · "
                   f"{len(self.trail)} pts")
        p.drawText(8, self.height() - 8,
                   "drag orbit · wheel zoom · double-click reset")


class Pose3DWindow(QtWidgets.QWidget):
    """The floating window around the view: title, CLEAR, and the state chip."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object pose — camera frame")
        self.view = Pose3DView()
        top = QtWidgets.QHBoxLayout()
        self.chip = QtWidgets.QLabel("no pose yet")
        self.chip.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        clear = QtWidgets.QPushButton("CLEAR")
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        clear.clicked.connect(self.view.clear)
        top.addWidget(self.chip)
        top.addStretch(1)
        top.addWidget(clear)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        self.resize(420, 400)

    def add(self, t: PoseTrack) -> None:
        self.view.add(t)
        if t.T_cam_obj is not None:
            T = t.T_cam_obj
            d = math.sqrt(T[3] ** 2 + T[7] ** 2 + T[11] ** 2)
            self.chip.setText(f"d {d:.2f} m · {t.pose_hz:.0f} Hz"
                              + (f" · reg {t.n_register}" if t.n_register else ""))
        else:
            self.chip.setText({"pose_loading": "loading FoundationPose...",
                               "registering": "registering...",
                               "building": "reconstructing the mesh...",
                               }.get(t.state, "no pose yet"))
