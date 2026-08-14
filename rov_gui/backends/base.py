#!/usr/bin/env python3
"""
base.py — the two worker shapes, and the rule that decides between them.

Every source of data in this application runs in its own thread. There are
exactly two ways to write one, and picking the wrong one produces a bug that is
hard to see and easy to ship: **slots that never run**.

TimerWorker — for anything that must also RECEIVE
-------------------------------------------------
Lives on its thread's event loop. A ``QTimer`` created *inside* the thread ticks
it; queued signal connections from the GUI (commands) are delivered between
ticks. Use this whenever the worker has to accept commands, and for any source
that is polled rather than blocking (draining a queue someone else fills).

LoopWorker — for anything that BLOCKS
-------------------------------------
Runs a plain ``while not stopped`` loop and blocks inside it (``getQueueEvents``,
``recv_match``, ``sock.recv``). Blocking is the right thing for a video source:
waking on arrival costs nothing and adds no latency, whereas polling burns a
core to be *later*. The cost is that the thread's event loop never runs, so this
worker **cannot have slots** — a queued invocation would sit in a queue nobody
services. Stop is therefore a ``threading.Event``, not a signal.

The trap, spelled out
---------------------
``QTimer`` belongs to the thread it was *created* in, not the thread of the
object that owns it. Create the timer in ``__init__`` (which runs on the GUI
thread) and the "worker" timer fires on the GUI thread — the code looks
threaded, the profiler says otherwise, and the UI stutters exactly as if there
were no worker at all. So every timer here is created in :meth:`on_started`,
which runs after ``moveToThread``.
"""

from __future__ import annotations

import threading
import time

from ..qt import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

# Threads that missed their stop window (a worker blocked in setup()/run()
# past the wait budget — the MPC worker's first-run acados code generation is
# the concrete case). They are PARKED here instead of dropped: releasing the
# last Python reference to a running, parentless QThread destroys the C++
# object under it, and Qt 5's ~QThread qFatal()s the whole process for that
# ("QThread: Destroyed while thread is still running"). A bounded leak beats
# an abort mid-shutdown with a claimed camera. wait_for_orphans() gives them
# one last, generous chance to finish before the interpreter tears down.
_ORPHANS: list = []


def wait_for_orphans(total_ms: int = 60000) -> int:
    """Block up to ``total_ms`` for parked threads to finish. Returns how many
    are STILL running (0 is the normal case — an acados build finishes in
    seconds; it just does not finish inside one worker-stop budget)."""
    deadline = time.monotonic() + total_ms / 1000.0
    for thread, _worker in list(_ORPHANS):
        remaining_ms = int(max(1.0, (deadline - time.monotonic()) * 1000.0))
        if thread.isRunning():
            thread.wait(remaining_ms)
    still = [(t, w) for (t, w) in _ORPHANS if t.isRunning()]
    _ORPHANS[:] = still
    return len(still)


class _WorkerBase(QObject):
    """Shared lifecycle: move to a thread, start, stop, report."""

    failed = Signal(str)
    started_ok = Signal()

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._thread: QThread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = QThread()
        self._thread.setObjectName(self.name)
        self.moveToThread(self._thread)
        self._thread.started.connect(self.on_started)
        self._thread.start()

    def stop(self, timeout_ms: int = 4000) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self.request_exit()
        if not self._thread.wait(timeout_ms):
            # Never terminate(): a killed thread leaves a half-closed device
            # handle behind, and on DepthAI that means the camera stays claimed
            # until the process exits. Report it and move on — but PARK the
            # thread rather than dropping the reference: destroying a running
            # QThread is a process abort, not a warning (see _ORPHANS above;
            # found 2026-08-12 with the MPC worker's first-run solver build).
            self.failed.emit(f"{self.name}: thread did not exit in {timeout_ms} ms")
            orphan = self._thread
            _ORPHANS.append((orphan, self))
            orphan.finished.connect(orphan.deleteLater)
        self._thread = None

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # ----------------------------------------------------------- for override
    @Slot()
    def on_started(self) -> None:
        """Runs ON the worker thread. Create timers and open devices here."""

    def request_exit(self) -> None:
        """Ask the worker's thread to finish. Runs on the CALLER's thread.

        Only for asking. Anything that touches the device must happen on the
        worker thread — closing a DepthAI device from another thread while its
        reader is mid-call gives a hang instead of an exit.
        """
        self._thread.quit()


class TimerWorker(_WorkerBase):
    """Event-loop worker ticked by a QTimer. Can receive queued commands."""

    def __init__(self, name: str, interval_ms: int = 50):
        super().__init__(name)
        self.interval_ms = interval_ms
        self._timer: QTimer | None = None

    @Slot()
    def on_started(self) -> None:
        self._timer = QTimer()             # created here => owned by THIS thread
        self._timer.setInterval(self.interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        try:
            self.setup()
        except Exception as e:                                   # noqa: BLE001
            self.failed.emit(f"{self.name}: {type(e).__name__}: {e}")
            return
        self.started_ok.emit()

    def request_exit(self) -> None:
        """Post a shutdown request; do NOT call quit() from here.

        A QTimer may only be started or stopped by the thread that owns it.
        ``QThread::quit()`` takes effect immediately — it is not itself a queued
        event — so calling it here would tear the event loop down *before* a
        queued shutdown ran, leaving the timer alive on a dead thread. Python
        then garbage-collects the worker on the main thread and Qt prints
        "QObject::killTimer: Timers cannot be stopped from another thread",
        which is a warning today and a crash on some builds.

        So the request is queued and :meth:`_shutdown` calls ``quit()`` itself,
        from inside the worker thread, once the timer is really stopped.
        """
        QMetaObject.invokeMethod(self, "_shutdown",
                                 Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _shutdown(self) -> None:
        """Runs on the worker thread: stop the timer, close, then exit the loop."""
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        try:
            self.teardown()
        except Exception as e:                                   # noqa: BLE001
            self.failed.emit(f"{self.name} teardown: {type(e).__name__}: {e}")
        thread = QThread.currentThread()
        if thread is not None:
            thread.quit()

    @Slot()
    def _tick(self) -> None:
        if self.stopping:
            if self._timer is not None:
                self._timer.stop()
            return
        try:
            self.tick()
        except Exception as e:                                   # noqa: BLE001
            # One bad tick must not kill the worker: a telemetry source that
            # dies on a single malformed packet takes the whole dashboard's
            # vehicle half with it, and the pilot sees "STALE" with no reason.
            self.failed.emit(f"{self.name}: {type(e).__name__}: {e}")

    # ----------------------------------------------------------- for override
    def setup(self) -> None:
        """Open resources. Runs on the worker thread, once."""

    def tick(self) -> None:
        """Called every ``interval_ms``. Must not block for long."""

    def teardown(self) -> None:
        """Close resources. Runs on the worker thread."""


class LoopWorker(_WorkerBase):
    """Blocking worker. No slots — see the module docstring."""

    @Slot()
    def on_started(self) -> None:
        try:
            self.setup()
            self.started_ok.emit()
            self.run()
        except Exception as e:                                   # noqa: BLE001
            self.failed.emit(f"{self.name}: {type(e).__name__}: {e}")
        finally:
            try:
                self.teardown()
            except Exception as e:                               # noqa: BLE001
                self.failed.emit(f"{self.name} teardown: {type(e).__name__}: {e}")

    def setup(self) -> None:
        """Open the device. Runs on the worker thread."""

    def run(self) -> None:
        """Block here until :attr:`stopping`."""

    def teardown(self) -> None:
        """Close the device. Runs on the worker thread."""


class Backend:
    """A named set of workers plus a command sink.

    The window only ever calls :meth:`start`, :meth:`stop`, :meth:`describe`,
    and connects the bus's command signals to :attr:`sink`.
    """

    name = "backend"
    simulated = False          # True => the UI shows a SIMULATED DATA banner

    def __init__(self, bus, mailboxes, opts):
        self.bus = bus
        self.mailboxes = mailboxes
        self.opts = opts
        self.workers: list[_WorkerBase] = []
        self.sink = None       # object with send(PilotInput) / gripper() / lights()

    def start(self) -> None:
        for w in self.workers:
            w.failed.connect(lambda m: self.bus.log.emit("error", m))
            w.start()

    def stop(self) -> None:
        for w in reversed(self.workers):
            w.stop()
        # Anything that missed its individual stop budget gets one shared,
        # generous drain here, so a slow-but-finite worker (first-run acados
        # code generation) completes before Python teardown would destroy its
        # still-running QThread.
        left = wait_for_orphans(60000)
        if left:
            self.bus.log.emit("error", f"{left} worker thread(s) still "
                                       f"running at shutdown — leaking them "
                                       f"to avoid a Qt abort")

    def describe(self) -> str:
        return self.name
