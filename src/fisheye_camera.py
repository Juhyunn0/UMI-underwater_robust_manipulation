#!/usr/bin/env python3
"""
fisheye_camera.py — Persistent fisheye camera session for the UMI gantry panel.

FisheyeCameraSession owns one background grab thread that continuously reads
frames at the configured FPS and emits frame_ready(ndarray, t_monotonic).
It can also share that frame stream with an experiment worker via
attach_worker_queue() / detach_worker_queue() — the grab thread fills a
Queue(maxsize=2) with drop-oldest semantics so the SLAM worker never blocks
the grab loop, and the grab loop never waits on the SLAM worker.

State machine:
    disconnected → connecting → connected   → disconnected
                             → connected_mock → disconnected
                             → error          → disconnected

Thread safety:
    open() / close() must be called from the GUI thread.
    frame_ready is emitted from the grab thread; PyQt5 AutoConnection
    automatically uses QueuedConnection for cross-thread slots, so GUI
    widgets can connect directly.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any

from PyQt5.QtCore import QObject, QThread, pyqtSignal


# =============================================================================
# Grab threads
# =============================================================================
class _FisheyeGrabThread(QThread):
    """Blocking cv2.VideoCapture grab loop.

    Emits:
      connected()            — after cap.isOpened() returns True (first grab OK)
      frame_ready(frame, t)  — at most at the configured FPS
      stats_ready(fps, ms)   — once per second
      error_occurred(str)    — on open failure or read failure
    """

    connected      = pyqtSignal()
    frame_ready    = pyqtSignal(object, float)  # (np.ndarray BGR, t_monotonic)
    stats_ready    = pyqtSignal(int, float)     # (fps_int, last_grab_ms)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        device: int | str,
        width: int,
        height: int,
        fps: int,
        stop_event: threading.Event,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._device = device
        self._width = width
        self._height = height
        self._fps = max(1, fps)
        self._stop = stop_event
        # Mutable holder so the GUI thread can swap the queue without locking.
        self._worker_queue_holder: list[Queue | None] = [None]

    def set_worker_queue(self, q: Queue | None) -> None:
        self._worker_queue_holder[0] = q

    def run(self) -> None:  # type: ignore[override]
        try:
            import cv2
        except ImportError:
            self.error_occurred.emit("OpenCV (cv2) is not installed")
            return

        try:
            idx = int(self._device)
            cap = cv2.VideoCapture(idx)
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(str(self._device))

        if not cap.isOpened():
            self.error_occurred.emit(
                f"Cannot open camera device {self._device!r}. "
                "Check the index and that no other app holds the device."
            )
            return

        if self._width and self._height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        if self._fps:
            cap.set(cv2.CAP_PROP_FPS, float(self._fps))

        self.connected.emit()

        fps_window = 0
        fps_t0 = time.monotonic()
        last_grab_ms = 0.0

        try:
            while not self._stop.is_set():
                t_before = time.monotonic()
                ok, frame = cap.read()
                t_after = time.monotonic()

                if not ok:
                    self.error_occurred.emit(
                        "cap.read() returned False — camera disconnected?"
                    )
                    break

                last_grab_ms = (t_after - t_before) * 1000.0
                fps_window += 1

                self.frame_ready.emit(frame, t_after)

                # Feed worker queue with drop-oldest so SLAM worker never blocks us.
                q = self._worker_queue_holder[0]
                if q is not None:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    try:
                        q.put_nowait((frame, t_after))
                    except Exception:
                        pass

                # Emit stats ~ once per second.
                now = time.monotonic()
                if now - fps_t0 >= 1.0:
                    self.stats_ready.emit(fps_window, last_grab_ms)
                    fps_window = 0
                    fps_t0 = now
        finally:
            cap.release()


class _MockGrabThread(QThread):
    """Synthetic frame generator — no real camera required.

    Produces blank BGR frames at the configured FPS using time.sleep to pace
    the loop. Same signal interface as _FisheyeGrabThread.
    """

    connected      = pyqtSignal()
    frame_ready    = pyqtSignal(object, float)
    stats_ready    = pyqtSignal(int, float)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        stop_event: threading.Event,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._w = max(1, width or 1280)
        self._h = max(1, height or 720)
        self._fps = max(1, fps or 30)
        self._stop = stop_event
        self._worker_queue_holder: list[Queue | None] = [None]

    def set_worker_queue(self, q: Queue | None) -> None:
        self._worker_queue_holder[0] = q

    def run(self) -> None:  # type: ignore[override]
        try:
            import numpy as np
        except ImportError:
            self.error_occurred.emit("numpy not installed — cannot generate mock frames")
            return

        interval = 1.0 / self._fps
        self.connected.emit()

        fps_window = 0
        fps_t0 = time.monotonic()

        while not self._stop.is_set():
            t0 = time.monotonic()

            frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
            t1 = time.monotonic()
            fps_window += 1

            self.frame_ready.emit(frame, t1)

            q = self._worker_queue_holder[0]
            if q is not None:
                if q.full():
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                try:
                    q.put_nowait((frame, t1))
                except Exception:
                    pass

            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                self.stats_ready.emit(fps_window, 0.0)
                fps_window = 0
                fps_t0 = now

            elapsed = time.monotonic() - t0
            rem = interval - elapsed
            if rem > 0.002:
                time.sleep(rem)


# =============================================================================
# Session (owns one grab thread)
# =============================================================================
class FisheyeCameraSession(QObject):
    """Long-lived camera session shared by the panel and the experiment runner.

    State values:
        "disconnected"   — no grab thread running
        "connecting"     — thread started, waiting for first-frame confirmation
        "connected"      — real camera open and grabbing
        "connected_mock" — mock generator running
        "error"          — grab failed; call close() then open() to retry

    Usage::
        session = FisheyeCameraSession(parent=panel)
        session.state_changed.connect(slot_on_gui_thread)
        session.frame_ready.connect(preview_widget.on_frame)
        session.open(device=0, width=1280, height=720, fps=30,
                     calib_path="calib.yaml")

    Worker queue (for experiment runner)::
        q = Queue(maxsize=2)
        session.attach_worker_queue(q)
        # experiment worker reads from q with get(timeout=0.2)
        session.detach_worker_queue()
    """

    state_changed = pyqtSignal(str)           # "disconnected"|"connecting"|…
    frame_ready   = pyqtSignal(object, float) # (np.ndarray BGR, t_monotonic)
    stats         = pyqtSignal(int, float)    # (fps_int, last_grab_ms)
    error         = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state: str = "disconnected"
        self._grab_thread: _FisheyeGrabThread | _MockGrabThread | None = None
        self._stop_event = threading.Event()
        self._calib_path: Path | None = None
        self._config: dict[str, Any] = {}

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._state in ("connected", "connected_mock")

    @property
    def calib_path(self) -> Path | None:
        return self._calib_path

    @property
    def device_config(self) -> dict:
        return dict(self._config)

    def open(
        self,
        device: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_path: Path | str | None = None,
        *,
        mock: bool = False,
    ) -> None:
        """Open the camera (or mock generator).

        Emits state_changed('connecting') immediately, then 'connected'
        (or 'connected_mock') once the grab thread confirms the device is open,
        or 'error' if open fails.
        """
        if self._grab_thread is not None:
            self.close()

        self._calib_path = Path(calib_path) if calib_path else None
        self._config = {
            "device": device, "width": width, "height": height,
            "fps": fps, "mock": mock,
        }
        self._stop_event.clear()
        self._set_state("connecting")

        if mock:
            t: _FisheyeGrabThread | _MockGrabThread = _MockGrabThread(
                width, height, fps, self._stop_event, self
            )
            connected_state = "connected_mock"
        else:
            t = _FisheyeGrabThread(
                device, width, height, fps, self._stop_event, self
            )
            connected_state = "connected"

        t.connected.connect(lambda: self._set_state(connected_state))
        t.frame_ready.connect(self._relay_frame)
        t.stats_ready.connect(self._relay_stats)
        t.error_occurred.connect(self._on_grab_error)
        t.finished.connect(self._on_grab_finished)
        self._grab_thread = t
        t.start()

    def close(self) -> None:
        """Stop the grab thread and transition to 'disconnected'."""
        self._stop_event.set()
        t = self._grab_thread
        if t is not None:
            if t.isRunning():
                t.wait(2000)
            try:
                t.deleteLater()
            except RuntimeError:
                pass
            self._grab_thread = None
        self._set_state("disconnected")

    def attach_worker_queue(self, q: Queue) -> None:
        """Register a Queue(maxsize=2) for the experiment worker to read frames."""
        if self._grab_thread is not None:
            self._grab_thread.set_worker_queue(q)

    def detach_worker_queue(self) -> None:
        """Unregister the worker queue."""
        if self._grab_thread is not None:
            self._grab_thread.set_worker_queue(None)

    # ── private ──────────────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def _relay_frame(self, frame: Any, t_mono: float) -> None:
        self.frame_ready.emit(frame, t_mono)

    def _relay_stats(self, fps: int, grab_ms: float) -> None:
        self.stats.emit(fps, grab_ms)

    def _on_grab_error(self, msg: str) -> None:
        print(f"[fisheye-cam] {msg}", file=sys.stderr)
        self._set_state("error")
        self.error.emit(msg)

    def _on_grab_finished(self) -> None:
        if self._state not in ("error", "disconnected"):
            self._set_state("disconnected")
        t = self._grab_thread
        if t is not None:
            try:
                t.deleteLater()
            except RuntimeError:
                pass
            self._grab_thread = None
