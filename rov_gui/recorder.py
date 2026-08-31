#!/usr/bin/env python3
"""
recorder.py — record the control station's own screen to an mp4.

What it records is the *widget tree*, via ``QWidget.grab()``, not the X11 root
window. That choice matters:

* it captures exactly the UI, at its exact pixel size, with no other window ever
  appearing over it and no compositor/Wayland screen-capture permission dance;
* it works headless (offscreen platform), which is what makes it testable;
* but it renders the widgets a second time, on the GUI thread, so it is not
  free. 12 fps is the default for that reason — a UI recording is documentation
  of what the pilot saw and what they did, not a video feed, and the video feeds
  are already recorded at full rate by ``c3_camera.c3_collect``.

If you need a zero-GUI-cost recording of the whole desktop instead, the right
tool is not this class:

    ffmpeg -f x11grab -framerate 15 -i :0.0 -c:v libx264 -preset veryfast out.mp4

Threading
---------
``grab()`` must happen on the GUI thread (it renders widgets). Encoding must
not: ``cv2.VideoWriter.write`` is a blocking call into libavcodec that routinely
takes tens of milliseconds. So the timer grabs and hands a QImage to a bounded
queue, and a plain worker thread encodes. The queue is bounded and *drops* when
full: a recorder that blocks the GUI thread to keep every frame has turned a
convenience feature into an outage. Drops are counted and written into the
sidecar JSON, so a recording never quietly claims a frame rate it did not have.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import runstore
from .qt import QImage, QObject, QTimer, Signal, import_cv2

# The BASE of the dated tree, not the folder written to: both recorders ask
# runstore.run_dir() at start() so their files land in the same minute folder
# as the nav recording and the controller CSV of the same run.
DEFAULT_DIR = runstore.DEFAULT_BASE


@dataclass
class RecStats:
    recording: bool = False
    path: Path | None = None
    frames: int = 0
    dropped: int = 0
    started_at: float | None = None
    size: tuple[int, int] = (0, 0)
    fps: float = 0.0
    error: str = ""
    name: str = ""            # the operator's label for this recording, if any
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def label(self) -> str:
        if not self.recording:
            return "REC"
        m, s = divmod(int(self.elapsed_s), 60)
        drop = f"  -{self.dropped}" if self.dropped else ""
        return f"REC {m:02d}:{s:02d}{drop}"


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    """QImage -> HxWx3 BGR uint8, honouring the scanline stride.

    ``bytesPerLine`` is NOT ``width * 3``: Qt pads each scanline to a 4-byte
    boundary, so a 1366-wide RGB888 image has 4098-byte rows. Reshaping by
    width silently shears the picture by one pixel per row — a bug that looks
    like a corrupt codec.
    """
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    ptr = img.constBits()
    if hasattr(ptr, "setsize"):        # PyQt's sip.voidptr has no length of its own
        ptr.setsize(img.sizeInBytes())
    buf = np.frombuffer(ptr, dtype=np.uint8)[: h * bpl]
    rgb = buf.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])          # RGB -> BGR


class StreamRecorder(QObject):
    """Record ONE video feed, exactly as the panel received it.

    Separate from :class:`ScreenRecorder` because the source is different, not
    because the encoding is: this one is *fed* QImages that a worker already
    decoded and scaled, so there is no grab and no GUI-thread render cost — the
    only extra work per frame is the memcpy into the encoder queue.

    What it records is therefore the DISPLAYED frame, at the size the panel was
    receiving. That is the honest description and it is a real limitation: it is
    a record of what the operator saw, not a capture of the sensor. For a
    dataset at source resolution with proper timestamps, use
    ``c3_camera/c3_collect.py`` — that is what it is for.
    """

    state_changed = Signal(bool)

    def __init__(self, name: str, out_dir: Path | str = DEFAULT_DIR,
                 queue_depth: int = 8):
        super().__init__()
        self.name = name
        self.out_dir = Path(out_dir)
        self.stats = RecStats()
        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._thread: threading.Thread | None = None

    def toggle(self, fps: float = 15.0) -> bool:
        self.stop() if self.stats.recording else self.start(fps=fps)
        return self.stats.recording

    def start(self, fps: float = 15.0, path: Path | None = None) -> Path | None:
        if self.stats.recording:
            return self.stats.path
        # A VideoWriter needs a constant rate, but frames arrive at whatever
        # rate the link allows. Open at the rate the feed is running NOW and
        # write what actually happened into the sidecar, so anything timed off
        # the file can be corrected rather than silently wrong.
        fps = float(max(5.0, min(60.0, fps or 15.0)))
        path = (Path(path) if path else
                runstore.run_dir(self.out_dir)
                / f"{self.name}_{runstore.stamp()}.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = RecStats(recording=True, path=path,
                              started_at=time.monotonic(), fps=fps)
        self._q = queue.Queue(maxsize=self._q.maxsize)
        self._thread = threading.Thread(target=self._encode, name=f"rec-{self.name}",
                                        daemon=True)
        self._thread.start()
        self.state_changed.emit(True)
        return path

    def feed(self, image) -> None:
        """Hand over a frame. GUI thread, never blocks."""
        if not self.stats.recording:
            return
        try:
            self._q.put_nowait(image)
        except queue.Full:
            with self.stats._lock:
                self.stats.dropped += 1

    def stop(self) -> Path | None:
        if not self.stats.recording:
            return None
        self.stats.recording = False
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        path = self.stats.path
        _write_sidecar(self.stats, source=f"rov_gui panel '{self.name}', as displayed")
        self.state_changed.emit(False)
        return path

    def _encode(self) -> None:
        cv2 = import_cv2()
        writer = None
        try:
            while True:
                img = self._q.get()
                if img is None:
                    break
                frame = _qimage_to_bgr(img)
                h, w = frame.shape[:2]
                if writer is None:
                    w, h = w - (w % 2), h - (h % 2)
                    self.stats.size = (w, h)
                    writer = cv2.VideoWriter(str(self.stats.path),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             self.stats.fps, (w, h))
                    if not writer.isOpened():
                        self.stats.error = f"could not open {self.stats.path}"
                        return
                tw, th = self.stats.size
                if (frame.shape[1], frame.shape[0]) != (tw, th):
                    # The panel was resized, or the pipeline relaunched at a new
                    # size. VideoWriter's geometry is fixed at open.
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                with self.stats._lock:
                    self.stats.frames += 1
        except Exception as e:                                   # noqa: BLE001
            self.stats.error = f"{type(e).__name__}: {e}"
        finally:
            if writer is not None:
                writer.release()


class ScreenRecorder(QObject):
    """Grab a widget on a timer, encode on a worker thread."""

    state_changed = Signal(bool)          # recording on/off

    def __init__(self, widget, out_dir: Path | str = DEFAULT_DIR,
                 fps: float = 12.0, queue_depth: int = 6):
        super().__init__()
        self.widget = widget
        self.out_dir = Path(out_dir)
        self.fps = float(fps)
        self.stats = RecStats(fps=self.fps)
        self._q: queue.Queue[QImage | None] = queue.Queue(maxsize=queue_depth)
        self._thread: threading.Thread | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab)

    # ------------------------------------------------------------- lifecycle
    def toggle(self, name: str = "") -> bool:
        self.stop() if self.stats.recording else self.start(name=name)
        return self.stats.recording

    def start(self, path: Path | None = None, name: str = "") -> Path | None:
        """Open a recording. ``name`` is the operator's own label for this run
        (header field in window.py): it becomes ``ui_<stamp>_<name>.mp4``, and
        an empty one leaves the plain ``ui_<stamp>.mp4`` naming untouched.

        Sanitised through :func:`runstore.slug`, because this string is typed
        during a session and arrives with spaces, slashes and dots in it — and
        a dot would make ``with_suffix('.json')`` overwrite part of the name
        instead of adding the sidecar.
        """
        if self.stats.recording:
            return self.stats.path
        cv2 = import_cv2()

        w, h = self.widget.width(), self.widget.height()
        # H.264/mpeg4 encoders reject odd dimensions; round down rather than
        # letting VideoWriter fail silently (it returns a closed writer and
        # every write() is a no-op, producing a 0-byte file).
        w, h = w - (w % 2), h - (h % 2)
        if w < 16 or h < 16:
            self.stats.error = f"widget is {w}x{h}, too small to record"
            return None

        if path is None:
            tag = runstore.slug(name)
            path = (runstore.run_dir(self.out_dir)
                    / f"ui_{runstore.stamp()}{'_' + tag if tag else ''}.mp4")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (w, h))
        if not writer.isOpened():
            self.stats.error = f"could not open {path} for writing"
            return None

        self.stats = RecStats(recording=True, path=path, started_at=time.monotonic(),
                              size=(w, h), fps=self.fps,
                              name=str(name or "").strip())
        self._q = queue.Queue(maxsize=self._q.maxsize)
        self._thread = threading.Thread(target=self._encode, args=(writer, cv2, w, h),
                                        name="ui-encoder", daemon=True)
        self._thread.start()
        self._timer.start(int(round(1000.0 / self.fps)))
        self.state_changed.emit(True)
        return path

    def stop(self) -> Path | None:
        if not self.stats.recording:
            return None
        self._timer.stop()
        self.stats.recording = False
        self._q.put(None)                      # sentinel: flush and close
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        path = self.stats.path
        self._write_sidecar()
        self.state_changed.emit(False)
        return path

    # ---------------------------------------------------------------- frames
    def _grab(self) -> None:
        """GUI thread: render the widget, hand the image over, never block."""
        img = self.widget.grab().toImage()
        try:
            self._q.put_nowait(img)
        except queue.Full:
            # The encoder is behind. Drop this frame and count it. The
            # alternative — blocking until the encoder catches up — freezes the
            # GUI thread, i.e. freezes the video feeds and the controls, to
            # protect a recording. Never the right trade on a control station.
            with self.stats._lock:
                self.stats.dropped += 1

    def _encode(self, writer, cv2, w: int, h: int) -> None:
        """Worker thread: convert and write. Owns the writer for its lifetime."""
        try:
            while True:
                img = self._q.get()
                if img is None:
                    break
                frame = _qimage_to_bgr(img)
                if frame.shape[1] != w or frame.shape[0] != h:
                    # The window was resized mid-recording. VideoWriter's frame
                    # size is fixed at open, so scale rather than write a frame
                    # it will silently discard.
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                with self.stats._lock:
                    self.stats.frames += 1
        except Exception as e:                                   # noqa: BLE001
            self.stats.error = f"{type(e).__name__}: {e}"
        finally:
            writer.release()

    def _write_sidecar(self) -> None:
        _write_sidecar(self.stats,
                       source="QWidget.grab() of the rov_gui main window")


def _write_sidecar(stats: RecStats, source: str) -> None:
    """Provenance beside the file: what was recorded, and what was lost.

    The dropped count is the honest part. Without it a 12 fps recording that
    actually captured 7 fps still says "12 fps" in its container metadata, and
    anything timed off the video is wrong by 40%.
    """
    if stats.path is None:
        return
    meta = {
        "file": stats.path.name,
        # What the operator called this run. In the sidecar as well as the
        # filename because a slug loses characters (spaces, punctuation) and
        # the JSON can keep them — but only the SLUG went into the name, so
        # the two are recorded separately rather than one standing for both.
        "name": stats.name or None,
        "requested_fps": stats.fps,
        "frames_written": stats.frames,
        "frames_dropped": stats.dropped,
        "duration_s": round(stats.elapsed_s, 2),
        "effective_fps": (round(stats.frames / stats.elapsed_s, 2)
                          if stats.elapsed_s > 0 else None),
        "size": list(stats.size),
        "source": source,
        "error": stats.error or None,
    }
    stats.path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
