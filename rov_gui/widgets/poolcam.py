#!/usr/bin/env python3
"""
poolcam.py — the POOL CAMS tab: one tile per pool-corner camera.

Watching and recording only. The page talks to :class:`rov_gui.poolcam.
PoolCamClient` and to nothing else — no DataBus, no mailbox — so nothing shown
here can become an input to the controller or the policy (see that module for
why the cameras live in a process of their own).

Per tile
--------
    header   name · size · fps · USB port            [AUTO REC ON / OFF]
    picture  the camera's newest frame, scaled in the CAMERA's process to the
             tile's size (the GUI thread only blits it); red frame + time and
             size while that camera is writing a file

AUTO REC is whether the camera records when the station's REC UI does —
"off" leaves it out (it keeps streaming to this tab). Switching it during a
recording stops that camera's file, or starts a new one at its next frame.

Double-click a picture to show that camera alone; again to go back to all.

Cost: previews are only asked for while this page is on screen, and at the
size each tile is actually drawn — a hidden tab costs the station nothing but
the status line twice a second.
"""

from __future__ import annotations

import math

from .. import theme
from ..qt import QColor, QFont, QImage, QPainter, QPen, QRectF, Qt, QtWidgets, Signal
from .indicators import ElidedLabel

#: Status arrives twice a second; this long without one means the camera
#: process is wedged (alive, not talking).
STALE_S = 2.0

#: Preview sizes are capped here: past this a tile gains nothing visible and
#: the camera process pays to scale and ship the extra pixels.
MAX_PREVIEW = (1280, 720)


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


class PoolCamCanvas(QtWidgets.QWidget):
    """Draws one camera's latest preview plus its recording state."""

    double_clicked = Signal()

    def __init__(self, label: str):
        super().__init__()
        self.label = label
        self._image: QImage | None = None
        self._seq = -1
        self._st: dict = {}
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(96, 54)
        self._hud = QFont(theme.MONO.split(",")[0])
        self._hud.setPixelSize(11)
        self._big = QFont(theme.SANS.split(",")[0])
        self._big.setPixelSize(15)
        self._big.setBold(True)

    def set_frame(self, seq: int, image: QImage) -> None:
        self._seq, self._image = seq, image
        self.update()

    def set_status(self, st: dict) -> None:
        if st != self._st:
            self._st = dict(st)
            self.update()

    @property
    def seq(self) -> int:
        return self._seq

    def mouseDoubleClickEvent(self, ev):
        self.double_clicked.emit()
        ev.accept()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()
        p.fillRect(rect, QColor(theme.VIDEO_BG))
        st = self._st
        live = bool(st.get("connected"))
        if live and self._image is not None and not self._image.isNull():
            iw, ih = self._image.width(), self._image.height()
            s = min(rect.width() / iw, rect.height() / ih)
            w, h = iw * s, ih * s
            p.drawImage(QRectF((rect.width() - w) / 2, (rect.height() - h) / 2, w, h),
                        self._image)
        elif st.get("empty"):
            # Not a fault: a slot waiting for its camera. Faint, no warning colour.
            p.setFont(self._big)
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "EMPTY SLOT")
            p.setFont(self._hud)
            p.drawText(rect.adjusted(0, 30, 0, 0), int(Qt.AlignmentFlag.AlignCenter),
                       "plug a pool camera into a free USB port — it appears here")
        else:
            p.setFont(self._big)
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "NO SIGNAL")
            p.setFont(self._hud)
            p.setPen(QColor(theme.WARN))
            p.drawText(rect.adjusted(0, 30, 0, 0), int(Qt.AlignmentFlag.AlignCenter),
                       str(st.get("status") or "waiting for the camera process"))

        p.setFont(self._hud)
        if st.get("recording"):
            pen = QPen(QColor(theme.FAIL))
            pen.setWidth(3)
            p.setPen(pen)
            p.drawRect(rect.adjusted(1, 1, -2, -2))
            text = f"● REC {_hms(st.get('rec_s', 0))}  {st.get('rec_mb', 0):.0f} MB"
            self._chip(p, text, theme.FAIL, right=True)
        elif not st.get("armed", True):
            self._chip(p, "left out of REC", theme.TEXT_DIM, right=True)
        if st.get("rec_error"):
            p.setPen(QColor(theme.FAIL))
            p.drawText(rect.adjusted(8, 0, -8, -6),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom),
                       f"recording failed: {st['rec_error']}"[:140])
        p.end()

    def _chip(self, p: QPainter, text: str, colour: str, right: bool) -> None:
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 12, fm.height() + 6
        x = self.width() - w - 8 if right else 8
        box = QRectF(x, 8, w, h)
        p.fillRect(box, QColor(0, 0, 0, 170))
        p.setPen(QColor(colour))
        p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)


class PoolCamTile(QtWidgets.QFrame):
    """Header row (name, numbers, AUTO REC switch) over the picture."""

    arm_toggled = Signal(str, bool)

    def __init__(self, label: str, armed: bool = True, rec_label: str = "REC UI"):
        super().__init__()
        self.label = label
        self.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(4, 0, 0, 0)
        head.setSpacing(8)
        name = QtWidgets.QLabel(label)
        name.setObjectName("PanelTitle")
        head.addWidget(name)
        self.info = ElidedLabel("", px=11, colour=theme.TEXT_DIM, mono=True)
        head.addWidget(self.info, 1)
        self.arm_btn = QtWidgets.QPushButton()
        self.arm_btn.setCheckable(True)
        self.arm_btn.setChecked(armed)
        self.arm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # min-width: the checked state is bold (app stylesheet) and Qt sized
        # the button for the regular weight — the clipped-glyph trap
        # widgets/video.py documents for its REC button.
        self.arm_btn.setStyleSheet(
            "QPushButton{font-size:10px; padding:1px 7px; border-radius:3px;"
            " min-width:84px;}")
        self.arm_btn.setToolTip(
            f"ON: this camera records whenever {rec_label} does.\n"
            "OFF: leave it out (it keeps streaming here).\n"
            "Switching it during a recording stops this camera's file, or\n"
            "starts a new one at its next frame.")
        self.arm_btn.toggled.connect(self._toggled)
        self._label_arm()
        head.addWidget(self.arm_btn)
        lay.addLayout(head)
        self.canvas = PoolCamCanvas(label)
        lay.addWidget(self.canvas, 1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

    def _label_arm(self) -> None:
        self.arm_btn.setText("AUTO REC ON" if self.arm_btn.isChecked()
                             else "AUTO REC OFF")

    def _toggled(self, on: bool) -> None:
        self._label_arm()
        self.arm_toggled.emit(self.label, bool(on))

    def set_status(self, st: dict) -> None:
        self.canvas.set_status(st)
        size = st.get("size")
        parts = []
        if st.get("empty"):
            parts.append("empty slot")
        elif st.get("connected") and size:
            parts.append(f"{size[0]}x{size[1]}  {st.get('fps', 0):.1f} fps")
        elif st.get("status"):
            parts.append(str(st["status"]))
        if st.get("usb") and st.get("usb") != "?":
            parts.append(str(st["usb"]))
        self.info.setText("   ".join(parts))
        armed = bool(st.get("armed", True))
        if armed != self.arm_btn.isChecked():       # the camera process's word wins
            self.arm_btn.blockSignals(True)
            self.arm_btn.setChecked(armed)
            self.arm_btn.blockSignals(False)
            self._label_arm()


class PoolCamPanel(QtWidgets.QWidget):
    """The tab page. Built empty; tiles appear when the camera process says
    which cameras it has (its hello), and are rebuilt if a restart changes that."""

    def __init__(self, client, rec_label: str = "REC UI"):
        super().__init__()
        self.client = client
        # What the record button that drives these cameras is called: the
        # station's REC UI, or the CCTV window's own REC (rov_gui/cctv.py).
        self.rec_label = rec_label
        self.tiles: dict[str, PoolCamTile] = {}
        self._solo: str | None = None
        self._hello_pid = None
        self._status_seq = -1
        self._summary_text = ""

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QtWidgets.QFrame()
        bar.setObjectName("Panel")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(10)
        title = QtWidgets.QLabel("POOL CAMERAS")
        title.setObjectName("PanelTitle")
        row.addWidget(title)
        note = QtWidgets.QLabel("monitoring + recording only — never an input "
                                "to control or the policy")
        note.setObjectName("Caption")
        row.addWidget(note)
        self.summary = ElidedLabel("starting the camera process …", px=11,
                                   colour=theme.TEXT, mono=False)
        row.addWidget(self.summary, 1)
        self.restart_btn = QtWidgets.QPushButton("RESTART")
        self.restart_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.restart_btn.setToolTip("start the camera process again")
        self.restart_btn.setVisible(False)
        self.restart_btn.clicked.connect(self._restart)
        row.addWidget(self.restart_btn)
        lay.addWidget(bar)

        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        lay.addWidget(self.grid_host, 1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Ignored)

    # -------------------------------------------------------------- building
    def _build_tiles(self, labels: list[str]) -> None:
        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
            tile.deleteLater()
        self.tiles = {}
        self._solo = None
        for label in labels:
            tile = PoolCamTile(label, armed=self.client.armed(label),
                               rec_label=self.rec_label)
            tile.arm_toggled.connect(self.client.arm)
            tile.canvas.double_clicked.connect(
                lambda lbl=label: self._toggle_solo(lbl))
            self.tiles[label] = tile
        self._place()

    def _place(self) -> None:
        shown = [self._solo] if self._solo else list(self.tiles)
        cols = max(1, math.ceil(math.sqrt(len(shown))))
        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
            tile.setVisible(False)
        for i, label in enumerate(shown):
            r, c = divmod(i, cols)
            self.grid.addWidget(self.tiles[label], r, c)
            self.tiles[label].setVisible(True)

    def _toggle_solo(self, label: str) -> None:
        self._solo = None if self._solo == label else label
        self._place()

    def _restart(self) -> None:
        self.restart_btn.setVisible(False)
        self._say("restarting the camera process …")
        self.client.start()

    # ------------------------------------------------------------------ tick
    def tick(self) -> None:
        """From the window's UI timer. Cheap when nothing changed."""
        hello, status, seq, exit_note = self.client.snapshot()
        if hello is not None and hello.get("pid") != self._hello_pid:
            self._hello_pid = hello.get("pid")
            self._build_tiles([c["label"] for c in hello.get("cams", [])])
        visible = self.isVisible()
        self.client.set_preview(self._wanted_sizes() if visible else {})
        if visible:
            for label, tile in self.tiles.items():
                got = self.client.take_frame(label, tile.canvas.seq)
                if got is not None:
                    tile.canvas.set_frame(*got)
        if seq != self._status_seq:
            self._status_seq = seq
            for st in (status or {}).get("cams", []):
                tile = self.tiles.get(st.get("label"))
                if tile is not None:
                    tile.set_status(st)
        self._summarise(hello, status, exit_note)

    def _wanted_sizes(self) -> dict[str, tuple[int, int]]:
        out = {}
        for label, tile in self.tiles.items():
            if not tile.isVisible():
                continue
            w, h = tile.canvas.width(), tile.canvas.height()
            if w >= 16 and h >= 16:
                # Rounded to 16 px so a window resize does not resend on
                # every pixel of the drag.
                out[label] = (min(MAX_PREVIEW[0], (w + 15) // 16 * 16),
                              min(MAX_PREVIEW[1], (h + 15) // 16 * 16))
        return out

    def _say(self, text: str) -> None:
        if text != self._summary_text:
            self._summary_text = text
            self.summary.setText(text)

    def _summarise(self, hello, status, exit_note: str) -> None:
        dead = not self.client.alive
        if dead == self.restart_btn.isHidden():     # (isVisible: false off-page)
            self.restart_btn.setVisible(dead)
        if dead:
            self._say((f"{self.rec_label} is on but the pool cameras are NOT "
                       "recording — " if self.client.recording else "")
                      + (exit_note or "camera process not running"))
            return
        if hello is None:
            return
        age = self.client.status_age()
        if age is not None and age > STALE_S:
            # Alive but silent: every tile below is a frozen picture, and a
            # "● REC" on one of them is the last thing it said, not the truth.
            self._say(f"camera process not answering for {age:.0f} s — the tiles "
                      "are frozen; nothing here is current")
            return
        cams = (status or {}).get("cams", [])
        n = len(hello.get("cams", []))
        empty = sum(1 for c in cams if c.get("empty"))
        live = sum(1 for c in cams if c.get("connected"))
        armed = sum(1 for c in cams if c.get("armed", True) and not c.get("empty"))
        writing = sum(1 for c in cams if c.get("recording"))
        if (status or {}).get("session"):
            text = (f"● REC  {writing}/{armed} writing  →  "
                    f"{(status or {}).get('outdir')}")
        else:
            text = (f"{live}/{n - empty} streaming"
                    + (f"  ({empty} empty slot{'s' if empty != 1 else ''})" if empty else "")
                    + f"   ·   {self.rec_label} will record {armed}   ·   "
                    f"{hello.get('codec', '')}")
        self._say(text)

    # ------------------------------------------------------------ visibility
    def hideEvent(self, ev):
        # Previews stop the moment the page is covered, not on the next tick:
        # the tick only runs the visible branch while the page is shown.
        self.client.set_preview({})
        super().hideEvent(ev)
