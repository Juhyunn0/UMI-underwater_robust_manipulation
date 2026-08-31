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
from ..qt import (QColor, QFont, QImage, QPainter, QRectF, Qt, QtCore, QtGui,
                  QtWidgets, QTimer, Signal)
from ..state import Conn, PoseTrack, VideoStat


def _c(hex_str: str, alpha: int = 255) -> QColor:
    col = QColor(hex_str)
    col.setAlpha(alpha)
    return col


class VideoCanvas(QtWidgets.QLabel):
    """Draws the latest frame plus its overlay. Owns no data source."""

    double_clicked = Signal()
    prompt_clicked = Signal(float, float)     # SOURCE image pixels

    def __init__(self, title: str, mailbox: FrameMailbox,
                 legend: tuple[str, str] | None = None, pose: bool = False,
                 box3d: bool = False):
        super().__init__()
        self.title = title
        self.mailbox = mailbox
        self.legend = legend            # depth colourbar end labels, if any
        self.box3d = box3d              # 12-edge oriented box vs axes only
        # Optional overlay pass, exactly like `legend`: None/False means the
        # paintEvent is byte-for-byte what it was, and the feature costs nothing.
        self.pose = pose
        self.pose_armed = False         # TRACK on: a click is a prompt
        self._track: PoseTrack | None = None
        # AprilTag detections on THIS feed (state.TagOverlay), same optional-
        # overlay contract as `pose`: None costs nothing in paintEvent.
        self._tags = None
        self._image: QImage | None = None
        self._stat: VideoStat | None = None
        # The raw data behind the picture, when the producer sends it (the
        # depth panel's uint16 millimetre map). Travels with its frame through
        # the mailbox, so a cursor readout can never quote a different frame
        # than the one on screen.
        self._aux = None
        self._hover = None              # cursor pos while over the canvas
        self._conn = Conn.OFFLINE
        self._age_s: float | None = None
        self._note = ""
        self._mbps_est: float | None = None
        # Where the image was actually drawn last paint. Needed to map a click
        # back to a source pixel; _fit() recomputes it every paint and keeps
        # nothing, so it is captured here.
        self._fit_rect: QRectF | None = None
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._emit_prompt)
        self._pending_click = None
        self._swallow_release = False   # eat the release AFTER a double click

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(160, 90)
        # For the depth probe: we need moves without a button held. Costs one
        # attribute store per move on canvases with no aux (see mouseMoveEvent).
        self.setMouseTracking(True)
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
        img, stat, aux = self.mailbox.take()
        if img is None:
            return False
        self._image, self._stat, self._aux = img, stat, aux
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
            self._fit_rect = dst
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
        if self._aux is not None:
            self._paint_probe(p, rect)
        if self.pose:
            self._paint_pose(p, rect)
        if self._tags is not None:
            self._paint_tags(p, rect)
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

    # ------------------------------------------------------------ geometry
    def to_source(self, pos) -> tuple[float, float] | None:
        """Canvas point -> SOURCE image pixel, or None if it is not on the image.

        THREE coordinate systems, and the middle one is the trap:

            canvas px  --(letterbox)-->  displayed image px  --(scale)-->  source px

        ``self._image`` is what the worker put in the mailbox, which it already
        shrank to fit the panel; ``self._stat.width/height`` is the size the
        sensor actually produced. Those differ by a ratio that is NOT constant:
        ``imaging.scale_to_fit`` returns the frame unshrunk whenever the
        reduction would be less than 15% (NO_RESIZE_ABOVE), so the ratio flips
        between 1.0 and something else as the pilot resizes the window. Assuming
        either one is silently wrong at some window sizes.

        Returns None on the letterbox bars, so a click into the black does not
        become a prompt somewhere on the edge of the image.
        """
        r, img, st = self._fit_rect, self._image, self._stat
        if r is None or img is None or img.isNull() or not r.contains(pos):
            return None
        if r.width() <= 0 or r.height() <= 0:
            return None
        u = (pos.x() - r.x()) / r.width()          # 0..1 across the drawn image
        v = (pos.y() - r.y()) / r.height()
        sw = st.width if (st and st.width) else img.width()
        sh = st.height if (st and st.height) else img.height()
        return (u * sw, v * sh)

    def source_size(self) -> tuple[int, int]:
        st, img = self._stat, self._image
        if st and st.width:
            return int(st.width), int(st.height)
        if img is not None and not img.isNull():
            return img.width(), img.height()
        return 0, 0

    # --------------------------------------------------------------- input
    def mouseReleaseEvent(self, ev):
        """A click becomes a prompt only after the double-click window passes.

        Qt sends press+release BEFORE it decides a gesture was a double click,
        so a naive handler fires a prompt on the first half of every
        double-click-to-promote. Deferring by the system's own
        ``doubleClickInterval`` and cancelling in ``mouseDoubleClickEvent`` is
        the only way to keep the left button meaning both things.

        The 400 ms default is invisible here: selecting an object is not an
        instantaneous operation on the far side either.
        """
        if self._swallow_release:
            # The second half of a double click: Qt sends Press, Release,
            # DblClick, then a SECOND Release. mouseDoubleClickEvent cancelled
            # the pending prompt, but without eating this trailing release the
            # handler below would immediately re-arm the timer — and every
            # double-click-to-promote would still fire a prompt 400 ms later
            # on whatever pixel was under the cursor.
            self._swallow_release = False
            super().mouseReleaseEvent(ev)
            return
        if (self.pose and self.pose_armed
                and ev.button() == Qt.MouseButton.LeftButton):
            self._pending_click = self.to_source(ev.pos())
            if self._pending_click is not None:
                self._click_timer.start(
                    QtWidgets.QApplication.doubleClickInterval())
        super().mouseReleaseEvent(ev)

    def _emit_prompt(self) -> None:
        if self._pending_click is not None:
            x, y = self._pending_click
            self._pending_click = None
            self.prompt_clicked.emit(x, y)

    def set_track(self, track: PoseTrack | None) -> None:
        self._track = track
        if self.pose:
            self.update()

    def set_tags(self, overlay) -> None:
        """The latest AprilTag detections for this feed (state.TagOverlay).
        ``enabled=False`` is the detector's explicit CLEAR on toggle-off, so a
        stale outline can never outlive the feature that drew it."""
        self._tags = None if (overlay is None or not overlay.enabled) else overlay
        self.update()

    # ------------------------------------------------------- AprilTag overlay
    def _paint_tags(self, p: QPainter, rect) -> None:
        """Detected tag outlines + ids, in the same fit space as the pose
        overlay. Mapped tags (the ones the localizer can use) draw solid;
        unmapped ones draw faint — seeing a tag is not the same as being
        localized by it, and the colour says which is happening."""
        t = self._tags
        r = self._fit_rect
        if t is None or r is None or not t.quads or t.src_w <= 0:
            return
        sx, sy = r.width() / t.src_w, r.height() / t.src_h
        p.setFont(self._hud_font)
        fm = p.fontMetrics()
        for i, quad in enumerate(t.quads):
            mapped = bool(t.mapped[i]) if i < len(t.mapped) else False
            col = _c(theme.OK if (mapped and t.localizes) else theme.ACCENT,
                     255 if mapped else 120)
            path = QtGui.QPainterPath()
            path.moveTo(r.x() + quad[0][0] * sx, r.y() + quad[0][1] * sy)
            for (qx, qy) in quad[1:]:
                path.lineTo(r.x() + qx * sx, r.y() + qy * sy)
            path.closeSubpath()
            p.setPen(QtGui.QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            if i < len(t.ids):
                cx = r.x() + sum(q[0] for q in quad) / 4.0 * sx
                cy = r.y() + sum(q[1] for q in quad) / 4.0 * sy
                label = str(t.ids[i])
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(0, 0, 0, 150))
                w = fm.horizontalAdvance(label) + 6
                p.drawRoundedRect(QRectF(cx - w / 2, cy - fm.height() / 2 - 1,
                                         w, fm.height() + 2), 2, 2)
                p.setPen(col)
                p.drawText(QRectF(cx - w / 2, cy - fm.height() / 2 - 1,
                                  w, fm.height() + 2),
                           int(Qt.AlignmentFlag.AlignCenter), label)

    # ---------------------------------------------------------- pose overlay
    def _paint_pose(self, p: QPainter, rect, fit=None) -> None:
        """Mask outline + a state chip. Draws POINTS, never pixels.

        The mask arrives as simplified polylines in source pixels (see
        ``perception/session.py``), so this is a path fill and a stroke — no
        per-pixel work on the GUI thread, which is the constraint the whole
        widget is built around.

        ``fit`` is the rectangle the SOURCE image occupies, defaulting to where
        it sits on this widget. It is a parameter so the same code can draw the
        overlay into a recorded frame, where the image fills the whole thing and
        there is no letterbox — one implementation, so the video cannot end up
        showing something the pilot never saw.
        """
        t = self._track
        if t is None or t.state == "off":
            return
        r = self._fit_rect if fit is None else fit
        sw, sh = (t.src_w, t.src_h)
        if r is not None and t.contours and sw > 0 and sh > 0:
            sx, sy = r.width() / sw, r.height() / sh
            path = QtGui.QPainterPath()
            for poly in t.contours:
                if len(poly) < 3:
                    continue
                path.moveTo(r.x() + poly[0][0] * sx, r.y() + poly[0][1] * sy)
                for (px, py) in poly[1:]:
                    path.lineTo(r.x() + px * sx, r.y() + py * sy)
                path.closeSubpath()
            col = _c(theme.state_color(t.conn))
            fill = QtGui.QColor(col)
            fill.setAlpha(56)
            p.fillPath(path, fill)
            p.setPen(QtGui.QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # 3-D: axes always, the full oriented box only on request. Both arrive
        # already projected to source pixels — the GUI thread does no geometry.
        if r is not None and sw > 0 and sh > 0:
            sx, sy = r.width() / sw, r.height() / sh

            def _pt(q):
                return QtCore.QPointF(r.x() + q[0] * sx, r.y() + q[1] * sy)

            if self.box3d and len(t.box_px) == 8:
                # Corner order is (x, y, z) nested loops upstream, so these are
                # the 12 edges of that ordering.
                edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
                p.setPen(QtGui.QPen(_c(theme.ACCENT), 1))
                for a, b in edges:
                    p.drawLine(_pt(t.box_px[a]), _pt(t.box_px[b]))
            if len(t.axes_px) == 4:
                origin = _pt(t.axes_px[0])
                for i, col in enumerate(("#ff4d4d", "#4dff4d", "#4d9dff")):
                    p.setPen(QtGui.QPen(QtGui.QColor(col), 2))
                    p.drawLine(origin, _pt(t.axes_px[i + 1]))

        # NO POSE NUMBERS HERE (removed 2026-08-23, operator request). The
        # x/y/z/d block that used to sit above the HUD strip was the object in
        # the CAMERA frame — a frame nothing else on this station works in, so
        # it could not be compared with anything and could not be flown on.
        # The object's position now appears in ONE place, the trajectory plot's
        # readout, in the MAP frame the pool and the vehicle are drawn in
        # (widgets/trajectory.py, "THE OBJECT READOUT"). The rate and the
        # registration count stay on this panel's own state chip below, so
        # nothing that was a HEALTH number was lost with them.

        # Status block, under the title strip so it never covers the HUD.
        #
        # TWO lines, always both drawn when there is anything to say: the stage
        # and the reason. Splitting them is not decoration — the stage answers
        # "what is it doing" and the reason answers "why is it not moving", and
        # the pilot needs both at once during a capture. A version of this that
        # showed the reason only in some states shipped, and a capture that had
        # given up looked identical to one running normally.
        stage, reason = self._pose_status(t)
        p.setFont(self._hud_font)
        fm = p.fontMetrics()
        bar_h = fm.height() + 6
        bg = QtGui.QColor(0, 0, 0, 150)
        y = rect.top() + bar_h + 6
        for text, col in ((stage, _c(theme.state_color(t.conn))),
                          (reason, _c(theme.TEXT_DIM))):
            if not text:
                continue
            box = QRectF(rect.left() + 8, y,
                         fm.horizontalAdvance(text) + 12, bar_h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(box, 3, 3)
            p.setPen(col)
            p.drawText(box.adjusted(6, 0, 0, 0),
                       int(Qt.AlignmentFlag.AlignVCenter), text)
            y += bar_h + 4

    @staticmethod
    def _pose_status(t) -> tuple[str, str]:
        """(stage, reason) for the overlay. English, and never silent.

        Every stage the pipeline can be in gets a line here, including the ones
        that are only reached by something going wrong. A stage with no entry
        would fall through to a bare uppercased state name, which is how a
        pilot ends up watching a green outline with no idea whether the station
        is collecting views, reconstructing, or has quietly given up.
        """
        stage = {
            "off": "TRACKING OFF",
            "loading": f"LOADING SAM2 — {t.load_s:.0f}s",
            "idle": "READY — CLICK AN OBJECT",
            "live": "READY — CLICK AN OBJECT",
            # Distance is in the chip because it is the constraint that decides
            # whether this can work at all: outside 0.3-0.8 m the stereo depth
            # is too noisy to register and every frame will be rejected, and
            # from the picture alone the pilot cannot tell how far away it is.
            # SMEAR rides beside the range because it is the thing the range is
            # only a proxy for, and it is the number that decides whether the
            # mesh comes out right. Live, so it can be fixed by moving closer
            # mid-orbit instead of being read in the post-mortem.
            "capturing": (f"COLLECTING VIEWS  {t.n_views}/{t.max_views}"
                          f"    ORBIT {t.arc_deg:.0f}/{t.max_arc:.0f}°"
                          + (f"    {t.distance_m:.2f} m"
                             if t.distance_m else "")
                          + (f"    SMEAR {t.smear_ratio:.1f}x"
                             + ("!" if (t.smear_max
                                        and t.smear_ratio > t.smear_max)
                                else "")
                             if t.smear_ratio is not None else "")),
            "building": f"RECONSTRUCTING MESH — {t.build_s:.0f}s",
            "pose_loading": f"LOADING FOUNDATIONPOSE — {t.fp_load_s:.0f}s",
            "registering": "REGISTERING POSE...",
            "lost": "LOST — CLICK THE OBJECT AGAIN",
            "failed": "STOPPED",
            "fault": "FAULT",
        }.get(t.state)
        if stage is None:                                  # i.e. "tracking"
            # THREE NUMBERS, and they answer three different questions.
            #   n Hz        how often a new pose ARRIVES — the rate anything
            #               downstream actually gets (measured; see
            #               perception/session.py _RateMeter)
            #   solve n ms  how long the GPU takes on one frame. This is the
            #               upstream tracker's own `hz` put back into the
            #               milliseconds it always was: until 2026-08-23 its
            #               reciprocal was printed here AS the rate, which is
            #               why the chip read "5 Hz" and then "9 Hz" while the
            #               object sat still — one 0.7 s re-registration in the
            #               30-sample window and thirty jobs to climb back out.
            #   reg n       how many times the estimator had to re-register,
            #               i.e. how often the pose slid off the object.
            # Together they say WHERE a slow rate comes from: solve >> 1/rate
            # means the GPU is the bottleneck, solve << 1/rate means it is the
            # camera or the mask.
            reg = f"    reg {t.n_register}" if t.n_register else ""
            solve = (f"    solve {t.pose_solve_ms:.0f} ms"
                     if t.pose_solve_ms else "")
            sam_solve = (f"    solve {t.sam_solve_ms:.0f} ms"
                         if t.sam_solve_ms else "")
            stage = (f"POSE TRACKING  {t.pose_hz:.0f} Hz{solve}{reg}"
                     if t.T_cam_obj is not None else
                     (f"TRACKING  {t.sam_hz:.0f} Hz{sam_solve} — no pose yet"
                      if t.pose_expected else
                      f"TRACKING  {t.sam_hz:.0f} Hz{sam_solve} — mask only"))
        reason = t.note.strip()
        if not reason:
            reason = {
                "capturing": "turn the object slowly; keep it 30-80 cm away",
                "building": "BundleSDF is running in a child process",
                "registering": "matching the mesh to the object (~0.7 s)",
                "pose_loading": "first run also compiles CUDA kernels",
            }.get(t.state, "")
        return stage, reason[:78]

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
        from ..imaging import DEPTH_MAX_MM, DEPTH_MIN_MM

        lo, hi = self.legend
        w = 16
        h = min(220, max(90, int(rect.height() * 0.5)))
        x = rect.width() - w - 12
        y = int(rect.height() / 2 - h / 2)
        grad = QtGui.QLinearGradient(0, y + h, 0, y)
        for pos, col in ((0.0, "#30123b"), (0.25, "#4686fb"), (0.5, "#1ae4b6"),
                         (0.75, "#fabc39"), (1.0, "#7a0403")):
            grad.setColorAt(pos, QColor(col))          # matches cv2 TURBO
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 120))         # keep it readable on any scene
        p.drawRoundedRect(QRectF(x - 4, y - 18, w + 8, h + 36), 3, 3)
        p.setBrush(QtGui.QBrush(grad))
        p.drawRect(x, y, w, h)
        p.setFont(self._hud_font)
        p.setPen(_c(theme.TEXT))
        p.drawText(QRectF(x - 60, y - 16, w + 60, 12),
                   int(Qt.AlignmentFlag.AlignRight), hi)
        p.drawText(QRectF(x - 60, y + h + 3, w + 60, 12),
                   int(Qt.AlignmentFlag.AlignRight), lo)

        # Intermediate ticks at whole metres, placed by the SAME linear
        # mapping colorize uses (imaging.depth_to_bgr) — the legend and the
        # picture must share one formula or the legend lies.
        span = DEPTH_MAX_MM - DEPTH_MIN_MM

        def bar_y(mm: float) -> float:
            f = (mm - DEPTH_MIN_MM) / span
            return y + h - f * h

        p.setPen(_c(theme.TEXT_DIM))
        for metres in (1.0, 2.0, 4.0):
            ty = bar_y(metres * 1000.0)
            p.drawLine(int(x - 3), int(ty), int(x + w), int(ty))
            p.drawText(QRectF(x - 40, ty - 6, 34, 12),
                       int(Qt.AlignmentFlag.AlignRight), f"{metres:.0f}")

        # And the cursor's own depth as a marker on the bar, when it has one:
        # the reading and where that reading sits in the colour scale, at once.
        if self._hover is not None:
            mm = self.depth_at(self._hover)
            if mm is not None:
                ty = bar_y(min(max(mm, DEPTH_MIN_MM), DEPTH_MAX_MM))
                p.setPen(QtGui.QPen(_c(theme.TEXT), 2))
                p.drawLine(int(x - 4), int(ty), int(x + w + 4), int(ty))

    def mouseMoveEvent(self, ev):
        # Only remembered, never acted on here: the paint pass reads it. No
        # update() call either — the canvas repaints at the UI rate anyway, and
        # a repaint per mouse move would double the paint load for nothing.
        self._hover = ev.pos()
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev):
        self._hover = None
        super().leaveEvent(ev)

    def depth_map(self):
        """The raw uint16 millimetre map behind this frame, or None.

        Public because the DEPTH stream is the one quantity on this station
        with no independent witness of its own — the tag solution never touches
        it — so `window._check_depth_scale` has to reach in and compare the two
        (see the "depth-vs-TAG" line on the trajectory readout)."""
        return self._aux

    def depth_at(self, pos) -> float | None:
        """Millimetres under a canvas point, or None.

        None covers three different situations on purpose — no raw data with
        this frame, cursor off the image, and a hole in the depth map (0 is
        DepthAI's "no measurement", and showing it as 0.00 m would put the
        object on the lens).
        """
        aux = self._aux
        if aux is None:
            return None
        src = self.to_source(pos)
        if src is None:
            return None
        x, y = int(src[0]), int(src[1])
        if not (0 <= y < aux.shape[0] and 0 <= x < aux.shape[1]):
            return None
        mm = float(aux[y, x])
        return mm if mm > 0 else None

    def _paint_probe(self, p: QPainter, rect) -> None:
        """The distance under the cursor, next to the cursor."""
        pos = self._hover
        if pos is None or self._aux is None:
            return
        mm = self.depth_at(pos)
        label = f"{mm / 1000:.2f} m" if mm is not None else "no data"
        p.setFont(self._hud_font)
        fm = p.fontMetrics()
        # Crosshair at the probe point, then the value beside it — flipped to
        # the other side near the right edge so it never runs off the panel.
        p.setPen(QtGui.QPen(_c(theme.TEXT), 1))
        p.drawLine(pos.x() - 6, pos.y(), pos.x() + 6, pos.y())
        p.drawLine(pos.x(), pos.y() - 6, pos.x(), pos.y() + 6)
        w = fm.horizontalAdvance(label) + 10
        h = fm.height() + 4
        x = pos.x() + 10
        if x + w > rect.width() - 8:
            x = pos.x() - 10 - w
        y = min(max(pos.y() - h // 2, 4), rect.height() - h - 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 170))
        p.drawRoundedRect(QRectF(x, y, w, h), 3, 3)
        p.setPen(_c(theme.TEXT) if mm is not None else _c(theme.TEXT_DIM))
        p.drawText(QRectF(x + 5, y, w, h),
                   int(Qt.AlignmentFlag.AlignVCenter), label)

    def mouseDoubleClickEvent(self, _ev):
        # Cancel the pending prompt: a promote gesture is not a selection. Qt
        # already delivered press+release for the first click, so without this
        # every promote also drops a prompt on whatever was under the cursor.
        # And Qt is not done: a SECOND release follows this event, which
        # mouseReleaseEvent must swallow or it re-arms the prompt it just
        # cancelled.
        self._click_timer.stop()
        self._pending_click = None
        self._swallow_release = True
        self.double_clicked.emit()


class VideoPanel(QtWidgets.QFrame):
    """A canvas plus the freshness watchdog that decides its state.

    ``warn_s``/``fail_s`` default to roughly 3 and 10 missed frames at the
    configured rate. Anything tighter flickers on a link that is merely busy;
    anything looser lets a frozen picture look live for long enough to matter.
    """

    focus_requested = Signal(str)
    record_toggled = Signal(str)
    track_toggled = Signal(bool)
    tag_toggled = Signal(str, bool)          # (panel name, on)
    prompt_clicked = Signal(float, float)

    def __init__(self, name: str, title: str, mailbox: FrameMailbox,
                 expected_fps: float = 15.0,
                 legend: tuple[str, str] | None = None, pose: bool = False,
                 box3d: bool = False, tag: bool = False):
        super().__init__()
        self.name = name
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(0)
        self.canvas = VideoCanvas(title, mailbox, legend=legend, pose=pose,
                                  box3d=box3d)
        self.canvas.double_clicked.connect(lambda: self.focus_requested.emit(self.name))
        self.canvas.prompt_clicked.connect(self.prompt_clicked)
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

        # TRACK sits beside REC and for the same reason: an absolutely
        # positioned child of the panel costs ZERO layout height, and the
        # station has about 5 px of vertical headroom against its one-screen
        # promise. Only built when this panel can actually be tracked on.
        self.track_btn = None
        # (A MAP button used to sit here, opening a floating 3-D window that
        # drew the object in the CAMERA frame. Removed 2026-08-21 at the
        # operator's request — the trajectory panel's own 3D button shows the
        # same object in the POOL frame, beside the vehicle it is relative to,
        # which is the picture anyone actually wanted.)
        if pose:
            self.track_btn = QtWidgets.QPushButton("TRACK", self)
            self.track_btn.setObjectName("Rec")
            self.track_btn.setCheckable(True)
            self.track_btn.setStyleSheet(
                "QPushButton{font-size:10px; padding:1px 7px; border-radius:3px;}")
            self.track_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.track_btn.setToolTip(
                "object tracking (SAM2).\n"
                "When on, a single click on this feed selects an object;\n"
                "double click still promotes the feed to the big slot.")
            self.track_btn.toggled.connect(self._track_toggled)
            self.track_btn.raise_()

        # TAG: per-feed AprilTag detection toggle, same absolute-positioning
        # trick as REC/TRACK (zero layout height). Only built for feeds a
        # detector actually watches (--mpc).
        self.tag_btn = None
        if tag:
            self.tag_btn = QtWidgets.QPushButton("TAG", self)
            self.tag_btn.setObjectName("Rec")
            self.tag_btn.setCheckable(True)
            self.tag_btn.setStyleSheet(
                "QPushButton{font-size:10px; padding:1px 7px; border-radius:3px;}")
            self.tag_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.tag_btn.setToolTip(
                "AprilTag detection on this feed.\n"
                "Solid green outlines = tags the localizer USES (calibrated "
                "feed);\nfaint outlines = seen but not used (uncalibrated "
                "feed / unmapped id).\nTurning the C3 feed off starves the "
                "MPC and it will disengage.")
            self.tag_btn.toggled.connect(self._tag_toggled)
            self.tag_btn.raise_()

        period = 1.0 / max(1.0, expected_fps)
        self.fresh = Freshness(warn_s=max(0.35, 3 * period),
                               fail_s=max(1.5, 10 * period))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 110)

    def _track_toggled(self, on: bool) -> None:
        self.canvas.pose_armed = bool(on)
        self.track_btn.setText("● TRACK" if on else "TRACK")
        self._place_rec()
        self.track_toggled.emit(bool(on))

    def _tag_toggled(self, on: bool) -> None:
        self.tag_btn.setText("● TAG" if on else "TAG")
        if not on:
            self.canvas.set_tags(None)     # clear instantly; worker also clears
        self._place_rec()
        self.tag_toggled.emit(self.name, bool(on))

    def set_tag_checked(self, on: bool) -> None:
        """Reflect the worker's default state WITHOUT re-emitting the command
        (the worker already holds that state; an echo would race its setup)."""
        if self.tag_btn is None:
            return
        self.tag_btn.blockSignals(True)
        self.tag_btn.setChecked(on)
        self.tag_btn.setText("● TAG" if on else "TAG")
        self.tag_btn.blockSignals(False)
        self._place_rec()

    def _place_rec(self) -> None:
        # Slack after adjustSize(): the checked state turns the label bold via
        # the stylesheet AFTER the size was computed, and the last glyph gets
        # clipped ("C3 RGB" -> "C3 RGE").
        self.rec_btn.adjustSize()
        self.rec_btn.resize(self.rec_btn.width() + 10, self.rec_btn.height())
        self.rec_btn.move(self.width() - self.rec_btn.width() - 10, 26)
        if self.track_btn is not None:
            self.track_btn.adjustSize()
            self.track_btn.resize(self.track_btn.width() + 10,
                                  self.track_btn.height())
            self.track_btn.move(
                self.rec_btn.x() - self.track_btn.width() - 6, 26)
        if self.tag_btn is not None:
            self.tag_btn.adjustSize()
            self.tag_btn.resize(self.tag_btn.width() + 10,
                                self.tag_btn.height())
            left = self.track_btn or self.rec_btn
            self.tag_btn.move(left.x() - self.tag_btn.width() - 6, 26)

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

    def burn_overlay(self, image):
        """Return the frame with the pose overlay drawn into it, for recording.

        A recording of a tracking session that does not show the tracking is a
        recording of a floor. The mask, the axes and the stage lines are the
        result — they belong in the file, not only on the glass.

        Deliberately a COPY and deliberately only when there is something to
        draw: the canvas still holds the original and paints its own overlay on
        top of it live, so drawing into that same QImage would double every
        stroke. When tracking is off this returns the frame untouched, so the
        recording path costs exactly what it did before.
        """
        t = self.canvas._track
        if not self.canvas.pose or t is None or t.state == "off":
            return image
        out = image.copy()
        p = QPainter(out)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            box = QtCore.QRect(0, 0, out.width(), out.height())
            # The recorded frame IS the image: no letterbox, so the fit
            # rectangle is the whole thing.
            self.canvas._paint_pose(p, box, fit=QRectF(box))
        finally:
            p.end()
        return out

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
