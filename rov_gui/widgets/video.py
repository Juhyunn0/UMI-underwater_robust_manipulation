#!/usr/bin/env python3
"""
video.py — one video feed: the picture, its overlay, and its liveness.

The picture
-----------
:class:`VideoCanvas` is a ``QLabel`` subclass that never calls ``setPixmap``.
That sounds perverse and is the single most important line in this file. A
QLabel with a pixmap reports ``sizeHint == pixmap.size()``, so a 1920x1080
frame asks the layout for 1920x1080 pixels; the layout obliges, the window
grows past the screen, and the "fits on one screen" requirement is gone. The
canvas instead keeps a ``QImage`` and paints it in ``paintEvent``, with
``QSizePolicy.Ignored`` in both directions, so its size is decided entirely by
the grid and never by its content.

The frame never arrives through a signal either — the panel pulls it from a
:class:`~rov_gui.bus.FrameMailbox` on the shared UI timer. See bus.py for why
(conflation, bounded latency). The panel publishes its own pixel size back into
the mailbox on every resize, so the *worker* does the expensive rescale.

Freeze is the failure mode that kills people
--------------------------------------------
A dead video feed does not go black. It shows the last good frame, forever, and
it looks exactly like a perfectly good picture of a stationary scene — which is
what an ROV holding station in front of a structure legitimately looks like. So
staleness is drawn destructively: the frame is dimmed, hatched, and stamped with
how long ago it arrived. The pilot has to be unable to mistake it for live.
"""

from __future__ import annotations

from .. import theme
from ..bus import FrameMailbox, Freshness
from ..qt import (QColor, QFont, QImage, QPainter, QRectF, Qt, QtGui, QtWidgets,
                  Signal)
from ..state import Conn, VideoStat


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


class VideoCanvas(QtWidgets.QLabel):
    """Draws the latest frame plus its overlay. Owns no data source."""

    double_clicked = Signal()

    def __init__(self, title: str, mailbox: FrameMailbox,
                 legend: tuple[str, str] | None = None):
        super().__init__()
        self.title = title
        self.mailbox = mailbox
        self.legend = legend            # depth colourbar end labels, if any
        self._image: QImage | None = None
        self._stat: VideoStat | None = None
        self._conn = Conn.OFFLINE
        self._age_s: float | None = None
        self._note = ""
        self._mbps_est: float | None = None

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(160, 90)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background:{theme.VIDEO_BG}; border-radius:4px;")
        self._hud_font = QFont(theme.MONO.split(",")[0])
        self._hud_font.setPixelSize(10)
        self._big_font = QFont(theme.SANS.split(",")[0])
        self._big_font.setPixelSize(15)
        self._big_font.setBold(True)

    # ------------------------------------------------------------------ data
    def resizeEvent(self, ev):
        # Tell the producer how big to make the next frame. Scaling 960x540 to
        # a 430-wide panel is ~0.5 MB of resampling per frame; on the GUI thread
        # that is dropped input events, on the worker thread it is free.
        self.mailbox.set_target_size(self.width(), self.height())
        super().resizeEvent(ev)

    def pull(self) -> bool:
        """Take a pending frame if there is one. Returns True if it changed."""
        img, stat = self.mailbox.take()
        if img is None:
            return False
        self._image, self._stat = img, stat
        return True

    def set_mbps_estimate(self, mbps: float | None) -> None:
        """A derived bandwidth for a stream that cannot count its own bytes.

        Kept on the canvas rather than written into the VideoStat, because a
        stat object is replaced by every new frame — an estimate stored there
        would be overwritten 30 times a second and never drawn.
        """
        self._mbps_est = mbps

    def set_note(self, note: str) -> None:
        if note != self._note:
            self._note = note
            self.update()

    def set_conn(self, conn: Conn, age_s: float | None) -> None:
        if conn is not self._conn or age_s != self._age_s:
            self._conn, self._age_s = conn, age_s
            self.update()

    # ----------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()
        p.fillRect(rect, _c(theme.VIDEO_BG))

        if self._image is not None and not self._image.isNull():
            dst = self._fit(self._image.width(), self._image.height())
            p.drawImage(dst, self._image)
        else:
            p.setFont(self._big_font)
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "NO SIGNAL")
            if self._note:
                # A retrying camera and a dead one look identical without this.
                p.setFont(self._hud_font)
                p.setPen(_c(theme.WARN if self._conn is Conn.FAULT
                            else theme.ACCENT))
                p.drawText(rect.adjusted(0, 26, 0, 0),
                           int(Qt.AlignmentFlag.AlignCenter), self._note)

        stale = self._conn in (Conn.STALE, Conn.OFFLINE, Conn.FAULT)
        if stale and self._image is not None:
            self._paint_stale(p, rect)
        self._paint_hud(p, rect)
        if self.legend:
            self._paint_legend(p, rect)
        # Border last, in the state colour, so the panel edge is readable from
        # across the cabin without reading any text.
        pen = QtGui.QPen(_c(theme.state_color(self._conn)), 2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), 4, 4)

    def _fit(self, iw: int, ih: int) -> QRectF:
        """Letterbox: preserve aspect, centre, never crop."""
        if iw <= 0 or ih <= 0:
            return QRectF(self.rect())
        w, h = self.width(), self.height()
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        return QRectF((w - dw) / 2, (h - dh) / 2, dw, dh)

    def _paint_stale(self, p: QPainter, rect) -> None:
        """Make a frozen frame impossible to read as a live one."""
        p.fillRect(rect, _c("#000000", 150))
        p.setPen(QtGui.QPen(_c(theme.FAIL, 90), 2, Qt.PenStyle.DashLine))
        step = 28
        for x in range(-rect.height(), rect.width(), step):
            p.drawLine(x, rect.height(), x + rect.height(), 0)
        p.setFont(self._big_font)
        p.setPen(_c(theme.FAIL))
        age = f"  {self._age_s:.1f} s ago" if self._age_s is not None else ""
        p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter),
                   f"{theme.state_text(self._conn)}{age}")

    def _paint_hud(self, p: QPainter, rect) -> None:
        s = self._stat
        p.setFont(self._hud_font)
        fm = QtGui.QFontMetrics(self._hud_font)
        bar_h = fm.height() + 6

        # top strip: name + state
        p.fillRect(0, 0, rect.width(), bar_h, _c("#000000", 130))
        p.setPen(_c(theme.TEXT))
        p.drawText(QRectF(8, 0, rect.width() - 16, bar_h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self.title.upper())
        p.setPen(_c(theme.state_color(self._conn)))
        p.drawText(QRectF(8, 0, rect.width() - 16, bar_h),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                   f"● {theme.state_text(self._conn)}")

        # bottom strip: the numbers that say whether the feed is healthy
        y = rect.height() - bar_h
        p.fillRect(0, y, rect.width(), bar_h, _c("#000000", 130))
        if s is None:
            left, right = "-- x --", "-- fps"
        else:
            left = f"{s.resolution}  {s.encoding}" if s.encoding else s.resolution
            bits = [f"{s.fps:4.1f} fps"]
            if s.latency_ms is not None:
                bits.append(f"{s.latency_ms:5.1f} ms")
            if s.mbps is not None:
                bits.append(f"{'~' if s.mbps_estimated else ''}{s.mbps:4.1f} Mb/s")
            elif self._mbps_est is not None:
                bits.append(f"~{self._mbps_est:4.1f} Mb/s")
            if s.drop_rate > 0.005:
                bits.append(f"drop {s.drop_rate * 100:.1f}%")
            right = "  ".join(bits)
        p.setPen(_c(theme.TEXT_DIM))
        p.drawText(QRectF(8, y, rect.width() - 16, bar_h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), left)
        drop_bad = s is not None and s.drop_rate > 0.02
        p.setPen(_c(theme.WARN if drop_bad else theme.TEXT_DIM))
        p.drawText(QRectF(8, y, rect.width() - 16, bar_h),
                   int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), right)

    def _paint_legend(self, p: QPainter, rect) -> None:
        """Colourbar for the depth view.

        Depth is colourised over a FIXED millimetre range upstream
        (``c3_camera.viz.colorize_depth``), not auto-scaled per frame, so a
        colour means the same distance in every frame — which is the only way
        this legend can be honest.
        """
        lo, hi = self.legend
        w, h = 10, min(90, int(rect.height() * 0.35))
        x = rect.width() - w - 10
        y = int(rect.height() / 2 - h / 2)
        grad = QtGui.QLinearGradient(0, y + h, 0, y)
        for pos, col in ((0.0, "#30123b"), (0.25, "#4686fb"), (0.5, "#1ae4b6"),
                         (0.75, "#fabc39"), (1.0, "#7a0403")):
            grad.setColorAt(pos, QColor(col))          # matches cv2 TURBO
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QBrush(grad))
        p.drawRect(x, y, w, h)
        p.setFont(self._hud_font)
        p.setPen(_c(theme.TEXT_DIM))
        # Right-aligned to just left of the bar, not overlapping it.
        p.drawText(QRectF(x - 66, y - 13, 60, 12),
                   int(Qt.AlignmentFlag.AlignRight), hi)
        p.drawText(QRectF(x - 66, y + h + 1, 60, 12),
                   int(Qt.AlignmentFlag.AlignRight), lo)

    def mouseDoubleClickEvent(self, _ev):
        self.double_clicked.emit()


class VideoPanel(QtWidgets.QFrame):
    """A canvas plus the freshness watchdog that decides its state.

    ``warn_s``/``fail_s`` default to roughly 3 and 10 missed frames at the
    configured rate. Anything tighter flickers on a link that is merely busy;
    anything looser lets a frozen picture look live for long enough to matter.
    """

    focus_requested = Signal(str)
    record_toggled = Signal(str)

    def __init__(self, name: str, title: str, mailbox: FrameMailbox,
                 expected_fps: float = 15.0,
                 legend: tuple[str, str] | None = None):
        super().__init__()
        self.name = name
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(0)
        self.canvas = VideoCanvas(title, mailbox, legend=legend)
        self.canvas.double_clicked.connect(lambda: self.focus_requested.emit(self.name))
        lay.addWidget(self.canvas, 1)

        # The record toggle lives ON the feed, not in the header. Four buttons
        # in the header had to be 42 px wide to fit, which elided "ROV" to
        # nonsense; here there is room for the feed's real name, and the control
        # is next to the thing it controls.
        self.rec_btn = QtWidgets.QPushButton("REC", self)
        self.rec_btn.setObjectName("Rec")
        self.rec_btn.setCheckable(True)
        # Its own compact style: the panel-wide QSS pads buttons for fingers,
        # which at 16 px clipped the label to nonsense — the same failure the
        # header buttons had, for the same reason.
        self.rec_btn.setStyleSheet(
            "QPushButton{font-size:10px; padding:1px 7px; border-radius:3px;}")
        self.rec_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rec_btn.setToolTip(f"record the {title} feed on its own, as displayed")
        self.rec_btn.clicked.connect(lambda: self.record_toggled.emit(self.name))
        self.rec_btn.raise_()

        period = 1.0 / max(1.0, expected_fps)
        self.fresh = Freshness(warn_s=max(0.35, 3 * period),
                               fail_s=max(1.5, 10 * period))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 110)

    def _place_rec(self) -> None:
        # Slack after adjustSize(): the checked state turns the label bold via
        # the stylesheet AFTER the size was computed, and the last glyph gets
        # clipped ("C3 RGB" -> "C3 RGE").
        self.rec_btn.adjustSize()
        self.rec_btn.resize(self.rec_btn.width() + 10, self.rec_btn.height())
        self.rec_btn.move(self.width() - self.rec_btn.width() - 10, 26)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_rec()

    def set_recording(self, on: bool, label: str) -> None:
        self.rec_btn.setChecked(on)
        self.rec_btn.setText(f"● REC {label}" if on else "REC")
        self._place_rec()

    def tick(self):
        """Called from the window's UI timer. Pull, age, repaint.

        Returns the NEW frame if one arrived, so the window can hand it to a
        recorder — the panel is the only place that knows a frame is new, and
        re-reading the mailbox elsewhere would steal frames from the display.
        """
        fresh_image = None
        if self.canvas.pull():
            stat = self.canvas._stat
            self.fresh.mark(reported=stat.conn if stat else None)
            fresh_image = self.canvas._image
        conn = self.fresh.state()
        self.canvas.set_conn(conn, self.fresh.age)
        self.canvas.update()
        return fresh_image

    def set_mbps_estimate(self, mbps: float | None) -> None:
        self.canvas.set_mbps_estimate(mbps)

    def set_status(self, stat: VideoStat) -> None:
        """A status-only update from the source (connecting, failed, retrying).

        Carries no image, so it goes over the bus rather than the mailbox, and
        it sets the *reported* half of the watchdog — which is what lets the
        panel say CONNECTING before it has ever received a frame.
        """
        self.fresh.reported = stat.conn
        self.canvas.set_note(stat.note)

    def state(self) -> Conn:
        return self.fresh.state()

    def stat(self) -> VideoStat | None:
        return self.canvas._stat
