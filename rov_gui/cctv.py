#!/usr/bin/env python3
"""
cctv.py — the pool-corner cameras ALONE: no ROV, no C3, no tether.

    ./c3 cctv                       # every pool camera, 4 slots, REC button
    ./c3 cctv --record              # start recording at once
    ./c3 cctv --device NE=/dev/v4l/by-path/...   (repeat; replaces the config list)
    ./c3 cctv --demo                # synthetic cameras, no hardware at all

The station's POOL CAMS tab (``./c3 gui --pool-cams``) in a window of its own,
for when the vehicle is not connected: the same camera process
(``pool_cam/pool_cam.py --serve``), the same tiles, the same 4 slots that a
camera plugged in later fills, the same AUTO REC switch per camera — and a REC
button of its own in place of the station's REC UI.

Recordings: each REC press is one folder,
``data/YYYYMMDD/MMDD_HHMMSS_poolcam/`` — the ``_poolcam`` kind keeps them out
of every statistic over water runs (rov_gui/runstore.py) — holding one
``<cam>_<stamp>.mp4`` + ``.json`` per camera, ``session.json`` (who recorded,
who was left out, which slots were empty, the shared t0) and ``log.txt``.

Not at the same time as ``./c3 gui --pool-cams``: a camera can only be open in
one process, so whichever starts second shows its cameras as busy.

Keys: Ctrl+R record / stop, F11 fullscreen, Ctrl+Q quit (a recording in
progress is finished and kept).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rov_gui import runstore, theme                       # noqa: E402
from rov_gui.poolcam import DEFAULT_CONFIG, client_from_opts  # noqa: E402
from rov_gui.qt import (QObject, QShortcut, Qt, QTimer, QtGui,  # noqa: E402
                        QtWidgets, Signal, install_slot_guard,
                        preload_platform_libs, run_app, sanitize_plugin_path)
from rov_gui.widgets.poolcam import PoolCamPanel           # noqa: E402

KIND = "poolcam"


class _LogBridge(QObject):
    """The client logs from its reader thread; this carries a line to the GUI
    thread (a queued signal) so the window can show it."""
    line = Signal(str, str)


class CctvWindow(QtWidgets.QMainWindow):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.setWindowTitle("Pool cameras — CCTV")
        self.setMinimumSize(800, 500)
        self._lines: list[str] = []           # this session's log, for log.txt
        self._rec_dir: Path | None = None
        self._rec_t0: float | None = None
        self._closed = False

        self._bridge = _LogBridge()
        self._bridge.line.connect(self._on_log)
        self.client = client_from_opts(opts, self._bridge.line.emit)

        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QtWidgets.QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(8)

        bar = QtWidgets.QFrame()
        bar.setObjectName("Panel")
        bar.setFixedHeight(46)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(10)
        title = QtWidgets.QLabel("POOL CAMERAS  CCTV")
        title.setStyleSheet(f"font-size:14px; font-weight:700; color:{theme.TEXT};"
                            "letter-spacing:1px;")
        row.addWidget(title)
        self.where = QtWidgets.QLabel("")
        self.where.setObjectName("Caption")
        row.addWidget(self.where, 1)
        if getattr(opts, "source", "hw") == "demo":
            banner = QtWidgets.QLabel("SYNTHETIC CAMERAS")
            banner.setObjectName("Banner")
            row.addWidget(banner)
        self.clock = QtWidgets.QLabel("--:--:--")
        self.clock.setObjectName("Value")
        row.addWidget(self.clock)
        self.rec_btn = QtWidgets.QPushButton("REC")
        self.rec_btn.setObjectName("Rec")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Wide enough for "● REC 0:00:00" in the checked (bold) face: Qt sizes
        # a button for its regular weight (widgets/video.py, clipped glyphs).
        self.rec_btn.setMinimumWidth(150)
        self.rec_btn.setToolTip("record every camera whose AUTO REC is on  (Ctrl+R)")
        self.rec_btn.clicked.connect(self.toggle_record)
        row.addWidget(self.rec_btn)
        lay.addWidget(bar)

        self.panel = PoolCamPanel(self.client, rec_label="REC")
        lay.addWidget(self.panel, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        for keys, slot in (("Ctrl+R", self.toggle_record),
                           ("F11", self._toggle_fullscreen),
                           ("Ctrl+Q", self.close)):
            QShortcut(QtGui.QKeySequence(keys), self).activated.connect(slot)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.client.start()
        self.timer.start()
        if getattr(self.opts, "record", False):
            # Once the cameras are streaming (or after 5 s regardless), so the
            # files share one t0 and do not open on a frozen first frame.
            self._record_when_ready(time.monotonic() + 5.0)

    def _record_when_ready(self, give_up: float) -> None:
        status = self.client.snapshot()[1] or {}
        cams = [c for c in status.get("cams", []) if not c.get("empty")]
        if (cams and all(c.get("connected") for c in cams)) or time.monotonic() > give_up:
            if not self.client.recording:
                self.toggle_record()
        else:
            QTimer.singleShot(200, lambda: self._record_when_ready(give_up))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        if self.client.recording:
            self._stop_recording()
        self.client.stop()                 # files finished (bounded wait)
        self._write_log()

    def closeEvent(self, ev) -> None:
        self.shutdown()
        super().closeEvent(ev)

    # ------------------------------------------------------------ recording
    def toggle_record(self) -> None:
        if self.client.recording:
            self._stop_recording()
        else:
            t0 = time.monotonic()
            outdir = runstore.run_dir(getattr(self.opts, "rec_dir", REPO / "data"),
                                      kind=KIND, join=False)
            if self.client.start_recording(t0, outdir):
                self._rec_dir, self._rec_t0 = outdir, t0
                self._on_log("info", f"recording to {outdir}")
            else:
                try:
                    outdir.rmdir()         # empty: a refused REC leaves no folder
                except OSError:
                    pass
        self._show_rec()

    def _stop_recording(self) -> None:
        self.client.stop_recording()
        self._on_log("info", f"recording stopped (files are finishing): {self._rec_dir}")
        self._write_log()
        self._show_rec()

    def _show_rec(self) -> None:
        on = self.client.recording
        self.rec_btn.setChecked(on)
        if not on:
            self.rec_btn.setText("REC")
        self.where.setText(f"→ {self._rec_dir}" if on and self._rec_dir else
                           (f"last: {self._rec_dir}" if self._rec_dir else
                            "recordings → data/YYYYMMDD/MMDD_HHMMSS_poolcam/"))

    # ---------------------------------------------------------------- ticks
    def _tick(self) -> None:
        try:
            self.panel.tick()
        except Exception as e:                                   # noqa: BLE001
            self.status.showMessage(f"[error] tab: {type(e).__name__}: {e}")
        self.clock.setText(time.strftime("%H:%M:%S"))
        if self.client.recording and self._rec_t0 is not None:
            s = int(time.monotonic() - self._rec_t0)
            self.rec_btn.setText(f"● REC {s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}")
        elif self.rec_btn.isChecked():
            self._show_rec()               # the camera process died mid-recording

    def _on_log(self, level: str, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [{level}] {msg}"
        print(line, flush=True)
        self._lines.append(line)
        self.status.showMessage(line, 8000 if level == "info" else 0)

    def _write_log(self) -> None:
        """The session's log beside the files it explains (rewritten whole on
        every stop, like the station's mission_log.txt)."""
        if self._rec_dir is None:
            return
        try:
            self._rec_dir.mkdir(parents=True, exist_ok=True)
            (self._rec_dir / "log.txt").write_text(
                "# rov_gui.cctv log — wall-clock stamps\n"
                + "\n".join(self._lines) + "\n", encoding="utf-8")
        except OSError as e:
            print(f"[error] log.txt not written: {e}", flush=True)

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="./c3 cctv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", dest="pool_cam_config", default=DEFAULT_CONFIG,
                   metavar="YAML", help="camera names/ports, slots, size, fps, "
                                        "encoder (default: %(default)s)")
    p.add_argument("--device", dest="pool_cam", action="append", default=None,
                   metavar="NAME=DEV", help="one camera by name (repeat); replaces "
                                            "the config file's list")
    p.add_argument("--rec-dir", default=str(REPO / "data"),
                   help="root of the dated data tree (default: this repo's data/)")
    p.add_argument("--record", action="store_true",
                   help="start recording once the cameras stream")
    p.add_argument("--demo", action="store_true",
                   help="synthetic cameras instead of hardware")
    p.add_argument("--fullscreen", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    opts.source = "demo" if opts.demo else "hw"
    # Before the QApplication exists: Qt reads the platform plugin path once
    # (rov_gui/qt.py, "cv2 hijacks QT_QPA_PLATFORM_PLUGIN_PATH").
    sanitize_plugin_path()
    preload_platform_libs()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app)
    win = CctvWindow(opts)
    install_slot_guard(lambda level, msg: win._on_log(level, msg))
    app.aboutToQuit.connect(win.shutdown)
    win.show()
    if opts.fullscreen:
        win.showFullScreen()
    else:
        win.resize(1600, 950)
    win.start()
    return run_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
