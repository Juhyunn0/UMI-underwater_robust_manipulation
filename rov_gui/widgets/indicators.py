#!/usr/bin/env python3
"""
indicators.py — the small painted widgets everything else is built from.

Custom ``paintEvent`` rather than composed QLabels for anything that updates at
UI rate. Three reasons, in order of how much they bite:

* A painted widget has a size hint that depends only on its font, so it cannot
  push the layout around when its value changes from "7" to "1013". Stacking
  QLabels inside QHBoxLayouts gives you a dashboard that twitches horizontally
  every time a number gains a digit.
* Updating a value is ``self._v = x; self.update()`` — one repaint of one small
  rect, no layout invalidation, no stylesheet re-resolution. Setting a QLabel's
  text triggers a relayout of its parent chain, which at 10 Hz across ~60
  readouts is real GUI-thread time.
* Colour by state is one line in the painter instead of a per-widget stylesheet
  string, and stylesheet churn is surprisingly expensive in Qt.

The exception is :class:`Panel`, which is a plain QFrame styled by QSS: it is
static furniture, so none of the above applies.
"""

from __future__ import annotations

from collections import deque

from .. import theme
from ..qt import (QColor, QFont, QPainter, QRectF, QSize, Qt, QtCore, QtGui,
                  QtWidgets, Signal)
from ..state import Conn


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


class Panel(QtWidgets.QFrame):
    """A titled box. ``panel.body`` is the layout children go into.

    ``right`` is an optional widget pinned to the title row — in practice always
    a :class:`StatusPill`, so that every module's connection state sits in the
    same place relative to its name.
    """

    def __init__(self, title: str, right: QtWidgets.QWidget | None = None,
                 margin: int = 8, spacing: int = 6):
        super().__init__()
        self.setObjectName("Panel")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(margin, margin - 2, margin, margin)
        outer.setSpacing(spacing)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self.title_label = QtWidgets.QLabel(title.upper())
        self.title_label.setObjectName("PanelTitle")
        head.addWidget(self.title_label)
        head.addStretch(1)
        self.right = right
        if right is not None:
            head.addWidget(right)
        outer.addLayout(head)

        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(spacing)
        outer.addLayout(self.body, 1)

    def set_alarm(self, on: bool) -> None:
        """Red border on the whole panel. Reserved for conditions that need
        the pilot's eyes now (leak, battery critical, e-stop)."""
        if self.property("alarm") == ("true" if on else "false"):
            return
        self.setProperty("alarm", "true" if on else "false")
        # Qt only re-evaluates property selectors after an explicit re-polish.
        self.style().unpolish(self)
        self.style().polish(self)


class StatusDot(QtWidgets.QWidget):
    """A filled circle in the state colour, with a dimmer halo when healthy."""

    def __init__(self, state: Conn = Conn.OFFLINE, size: int = 10):
        super().__init__()
        self._state = state
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def set_state(self, state: Conn) -> None:
        if state is not self._state:
            self._state = state
            self.setToolTip(theme.state_text(state))
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = _c(theme.state_color(self._state))
        r = QRectF(2, 2, self._size, self._size)
        if self._state is Conn.ONLINE:
            p.setBrush(_c(theme.state_color(self._state), 60))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(r.adjusted(-2, -2, 2, 2))
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(r)


class StatusPill(QtWidgets.QWidget):
    """Dot + label + optional detail, e.g.  ``* MAIN CAM  ONLINE``."""

    def __init__(self, label: str = "", state: Conn = Conn.OFFLINE,
                 show_state_text: bool = True, font_px: int = 10):
        super().__init__()
        self._label = label
        self._state = state
        self._detail = ""
        self._show_state = show_state_text
        self._font = QFont(theme.MONO.split(",")[0], -1)
        self._font.setPixelSize(font_px)
        self.setMinimumHeight(font_px + 8)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        # Reserve room for the WIDEST state word this pill can ever show. A
        # pill that fits "ONLINE" and clips "DEGRADED" to "DEGR/" is worse than
        # useless: the two states it most needs to distinguish are exactly the
        # ones it truncates.
        widest = max(theme.STATE_TEXT.values(), key=len) if show_state_text else ""
        fm = QtGui.QFontMetrics(self._font)
        self.setMinimumWidth(
            fm.horizontalAdvance(f"{label}  {widest}".strip()) + 16)

    def set_state(self, state: Conn, detail: str = "") -> None:
        if state is not self._state or detail != self._detail:
            self._state = state
            self._detail = detail
            self.updateGeometry()
            self.update()

    def text(self) -> str:
        parts = [self._label] if self._label else []
        if self._show_state:
            parts.append(theme.state_text(self._state))
        if self._detail:
            parts.append(self._detail)
        return "  ".join(parts)

    def sizeHint(self):
        fm = QtGui.QFontMetrics(self._font)
        # Measured from the CURRENT text, and the pill sits after a stretch in
        # the panel's title row, so it grows leftwards into slack space and can
        # never widen the panel beyond its share of the grid.
        return QSize(fm.horizontalAdvance(self.text()) + 16,
                     max(self.minimumHeight(), fm.height() + 4))

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = _c(theme.state_color(self._state))
        h = self.height()
        d = 7
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(1, (h - d) / 2, d, d))
        p.setFont(self._font)
        p.setPen(_c(theme.TEXT if self._state.ok else theme.state_color(self._state)))
        p.drawText(QRectF(d + 6, 0, self.width() - d - 6, h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self.text())


class HoldToConfirmButton(QtWidgets.QPushButton):
    """A button that only fires after being held, with the wait drawn on it.

    For actions that are safe to *do* and expensive to do *by accident* — arming
    a vehicle is the case this exists for. A modal "are you sure?" dialog is the
    usual answer and is worse here: it steals keyboard focus from the window
    that is flying the ROV, and pilots learn to dismiss it without reading.
    A hold cannot be triggered by one stray click, needs no focus, and shows its
    own progress, so the operator can abort by simply letting go.

    Releasing early cancels and resets — there is no partial credit.
    """

    confirmed = Signal()

    def __init__(self, text: str, hold_ms: int = 1000, tooltip: str = ""):
        super().__init__(text)
        self._label = text
        self._hold_ms = max(200, int(hold_ms))
        self._elapsed = 0
        self.setToolTip(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._step)
        self.pressed.connect(self._start)
        self.released.connect(self._cancel)

    def _start(self) -> None:
        self._elapsed = 0
        self._timer.start()

    def _cancel(self) -> None:
        self._timer.stop()
        self._elapsed = 0
        self.update()

    def _step(self) -> None:
        self._elapsed += self._timer.interval()
        if self._elapsed >= self._hold_ms:
            self._cancel()
            self.setDown(False)
            self.confirmed.emit()
        else:
            self.update()

    def leaveEvent(self, ev):
        if self.isDown():
            self.setDown(False)
        self._cancel()
        super().leaveEvent(ev)

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        if not self._elapsed:
            return
        frac = min(1.0, self._elapsed / self._hold_ms)
        p = QPainter(self)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_c(theme.WARN, 110))
        p.drawRect(QRectF(0, 0, self.width() * frac, self.height()))
        p.setPen(_c(theme.TEXT))
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter),
                   f"HOLD {(self._hold_ms - self._elapsed) / 1000:.1f}s")


class ElidedLabel(QtWidgets.QLabel):
    """A label that elides its text instead of demanding width for it.

    A plain QLabel's size hint is its full text width, so one long status
    string ("no answer from 192.168.2.2:80") silently widens its column and
    squeezes a neighbour — or, in a fixed grid, gets clipped mid-word with no
    ellipsis to show that anything is missing. This paints ``…`` and reports a
    minimum width of zero, so it always yields space rather than taking it.
    """

    def __init__(self, text: str = "", px: int = 11, colour: str = theme.TEXT_DIM,
                 mono: bool = True):
        super().__init__(text)
        self._full = text
        self._colour = colour
        self._font = QFont((theme.MONO if mono else theme.SANS).split(",")[0])
        self._font.setPixelSize(px)
        self.setFont(self._font)
        self.setFixedHeight(px + 4)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:          # noqa: N802 - Qt naming
        self._full = text
        self.setToolTip(text)
        self.update()

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self.update()

    def minimumSizeHint(self):                      # noqa: N802 - Qt naming
        return QSize(0, self.height())

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setFont(self._font)
        p.setPen(_c(self._colour))
        fm = QtGui.QFontMetrics(self._font)
        text = fm.elidedText(self._full, Qt.TextElideMode.ElideRight,
                             max(0, self.width()))
        p.drawText(self.rect(),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   text)


class MeterBar(QtWidgets.QWidget):
    """Label + horizontal bar + value, in one 16 px row.

    Thresholds are given as ``(warn, crit)`` on the *normalised* value, and
    ``invert=True`` means low is bad (battery) rather than high is bad (load).
    """

    def __init__(self, label: str = "", unit: str = "", spec: str = ".1f",
                 warn: float | None = None, crit: float | None = None,
                 invert: bool = False, label_px: int = 46, value_px: int = 58,
                 height: int = 16):
        super().__init__()
        self._label = label
        self._unit = unit
        self._spec = spec
        self._warn, self._crit, self._invert = warn, crit, invert
        self._label_w, self._value_w = label_px, value_px
        self._value: float | None = None
        self._frac = 0.0
        self._conn = Conn.OFFLINE
        self._font = QFont(theme.MONO.split(",")[0])
        self._font.setPixelSize(10)
        self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def set_value(self, value: float | None, frac: float | None = None,
                  conn: Conn | None = None) -> None:
        self._value = value
        if frac is not None:
            self._frac = max(0.0, min(1.0, float(frac)))
        elif value is not None:
            self._frac = max(0.0, min(1.0, float(value)))
        if conn is not None:
            self._conn = conn
        self.update()

    def _bar_color(self) -> QColor:
        if self._value is None or self._conn.bad:
            return _c(theme.IDLE)
        v = self._frac
        if self._warn is None:
            return _c(theme.OK)
        bad = (v <= (self._crit if self._crit is not None else -1)) if self._invert \
            else (v >= (self._crit if self._crit is not None else 2))
        warn = (v <= self._warn) if self._invert else (v >= self._warn)
        if bad:
            return _c(theme.FAIL)
        return _c(theme.WARN) if warn else _c(theme.OK)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        h = self.height()
        # label
        p.setPen(_c(theme.TEXT_DIM))
        p.drawText(QRectF(0, 0, self._label_w, h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self._label)
        # track
        x0 = self._label_w + 4
        x1 = self.width() - self._value_w - 4
        track = QRectF(x0, h / 2 - 3, max(2.0, x1 - x0), 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_c(theme.BG))
        p.drawRoundedRect(track, 3, 3)
        if self._value is not None:
            fill = QRectF(track)
            fill.setWidth(max(1.0, track.width() * self._frac))
            p.setBrush(self._bar_color())
            p.drawRoundedRect(fill, 3, 3)
        # value
        p.setPen(_c(theme.TEXT if self._value is not None else theme.TEXT_FAINT))
        p.drawText(QRectF(self.width() - self._value_w, 0, self._value_w, h),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                   theme.fmt(self._value, self._spec, self._unit))


class BiBar(QtWidgets.QWidget):
    """Signed −1..+1 bar that grows out from the centre.

    For thrusters. A unipolar bar cannot distinguish "stopped" from "full
    astern", and on a vectored ROV holding station against a current, half the
    thrusters are astern all the time.
    """

    def __init__(self, label: str = "", height: int = 18, label_px: int = 26,
                 value_px: int = 54):
        super().__init__()
        self._label = label
        self._v = 0.0
        self._text = "--"
        self._conn = Conn.OFFLINE
        self._label_w, self._value_w = label_px, value_px
        self._font = QFont(theme.MONO.split(",")[0])
        self._font.setPixelSize(10)
        self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def set_value(self, v: float, text: str = "", conn: Conn = Conn.ONLINE) -> None:
        self._v = max(-1.0, min(1.0, float(v)))
        self._text = text or f"{self._v:+.2f}"
        self._conn = conn
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        h = self.height()
        p.setPen(_c(theme.TEXT_DIM))
        p.drawText(QRectF(0, 0, self._label_w, h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self._label)

        x0 = self._label_w + 2
        x1 = self.width() - self._value_w - 4
        track = QRectF(x0, h / 2 - 4, max(4.0, x1 - x0), 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_c(theme.BG))
        p.drawRoundedRect(track, 2, 2)

        mid = track.center().x()
        if not self._conn.bad:
            col = _c(theme.OK if abs(self._v) < 0.85 else theme.WARN)
            w = abs(self._v) * track.width() / 2
            bar = QRectF(mid, track.top() + 1, max(1.0, w), track.height() - 2) \
                if self._v >= 0 else \
                QRectF(mid - max(1.0, w), track.top() + 1, max(1.0, w), track.height() - 2)
            p.setBrush(col)
            p.drawRect(bar)
        # centre tick, drawn last so it stays visible through the fill
        p.setPen(_c(theme.BORDER_HI))
        p.drawLine(int(mid), int(track.top() - 2), int(mid), int(track.bottom() + 2))

        p.setPen(_c(theme.TEXT if not self._conn.bad else theme.TEXT_FAINT))
        p.drawText(QRectF(self.width() - self._value_w, 0, self._value_w, h),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                   self._text if not self._conn.bad else "--")


class Sparkline(QtWidgets.QWidget):
    """A short rolling history. Trend is the point, not the value.

    Bandwidth is the case that motivates it: 40 Mbit/s means nothing on its own,
    but 40 after a flat 12 means something started, and 12 with a sawtooth means
    something is stalling and bursting.
    """

    def __init__(self, n: int = 120, vmax: float | None = None,
                 color: str = theme.ACCENT, height: int = 26):
        super().__init__()
        self._data: deque[float] = deque(maxlen=n)
        self._vmax = vmax
        self._color = color
        self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def push(self, v: float | None) -> None:
        self._data.append(0.0 if v is None else float(v))
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _c(theme.BG))
        if len(self._data) < 2:
            return
        vmax = self._vmax if self._vmax else max(max(self._data), 1e-6)
        w, h = self.width(), self.height()
        step = w / (len(self._data) - 1)
        path = QtGui.QPainterPath()
        for i, v in enumerate(self._data):
            y = h - 1 - (min(v, vmax) / vmax) * (h - 2)
            pt = (i * step, y)
            path.moveTo(*pt) if i == 0 else path.lineTo(*pt)
        p.setPen(QtGui.QPen(_c(self._color), 1.2))
        p.drawPath(path)
        # fill under the curve, very faint
        path.lineTo(w, h)
        path.lineTo(0, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_c(self._color, 34))
        p.drawPath(path)


class KeyChip(QtWidgets.QWidget):
    """One key cap in the input visualiser. Lights while its input is held."""

    def __init__(self, cap: str, hint: str = "", w: int = 30, h: int = 26):
        super().__init__()
        self._cap = cap
        self._hint = hint
        self._active = False
        self.setFixedSize(w, h)
        self.setToolTip(hint)

    def set_active(self, on: bool) -> None:
        if on != self._active:
            self._active = on
            self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        if self._active:
            p.setBrush(_c(theme.ACCENT))
            p.setPen(QtGui.QPen(_c(theme.ACCENT), 1))
        else:
            p.setBrush(_c(theme.PANEL_HI))
            p.setPen(QtGui.QPen(_c(theme.BORDER_HI), 1))
        p.drawRoundedRect(r, 4, 4)
        f = QFont(theme.MONO.split(",")[0])
        f.setPixelSize(11)
        f.setBold(self._active)
        p.setFont(f)
        p.setPen(_c("#04121b" if self._active else theme.TEXT_DIM))
        p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), self._cap)


class Tile(QtWidgets.QWidget):
    """Caption over a big monospace number. For the handful of readings the
    pilot looks at without reading — depth, heading, battery."""

    def __init__(self, caption: str, unit: str = "", spec: str = ".1f",
                 value_px: int = 20, width: int = 84):
        super().__init__()
        self._caption = caption
        self._unit = unit
        self._spec = spec
        self._value: float | None = None
        self._color = theme.TEXT
        self._value_px = value_px
        self.setMinimumWidth(width)
        self.setFixedHeight(value_px + 16)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def set_value(self, value: float | None, color: str | None = None) -> None:
        self._value = value
        self._color = color or theme.TEXT
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        f = QFont(theme.SANS.split(",")[0])
        f.setPixelSize(9)
        p.setFont(f)
        p.setPen(_c(theme.TEXT_FAINT))
        p.drawText(QRectF(0, 0, self.width(), 12),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self._caption.upper())
        f = QFont(theme.MONO.split(",")[0])
        f.setPixelSize(self._value_px)
        f.setBold(True)
        p.setFont(f)
        p.setPen(_c(self._color if self._value is not None else theme.TEXT_FAINT))
        p.drawText(QRectF(0, 12, self.width(), self.height() - 12),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   theme.fmt(self._value, self._spec, self._unit))


class HoldButton(QtWidgets.QPushButton):
    """A button whose *hold* is the command, not its click.

    Emits ``held(True)`` on press and ``held(False)`` on release, including the
    release you get when the pointer leaves the button while down — without that
    last case a dragged-off mouse leaves a thruster running, which is exactly
    the class of bug this repo's safety reviewer looks for.
    """

    held = Signal(bool)

    def __init__(self, text: str, tooltip: str = ""):
        super().__init__(text)
        self.setToolTip(tooltip)
        # Never take focus: if a motion button has focus, the space bar and the
        # arrow keys go to the button instead of to keyPressEvent, and the pilot
        # silently loses the keyboard.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pressed.connect(lambda: self.held.emit(True))
        self.released.connect(lambda: self.held.emit(False))

    def leaveEvent(self, ev):
        if self.isDown():
            self.setDown(False)
            self.held.emit(False)
        super().leaveEvent(ev)
