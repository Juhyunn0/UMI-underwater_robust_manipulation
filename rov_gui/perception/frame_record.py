#!/usr/bin/env python3
"""
frame_record.py — the recorder plumbing, and the five rules it exists under.

Extracted from ``depth_record.py`` on 2026-09-06, unchanged, when a SECOND
recorder appeared (``stereo_record.py``, the raw mono pairs). The plumbing is
shared rather than copied for one reason: the rules below are not conveniences,
they are the findings of the 2026-09-06 safety audit, and a copy of them is a
copy that will drift from the audit that produced it. ``test_depth_record.py``
therefore covers both recorders' machinery.

Safety: neither recorder may ever be able to hurt a dive
--------------------------------------------------------
A recorder runs inside a worker that shares a tick with GPU inference on a
budget with no slack, while the vehicle is armed. Five rules follow, and each
one is a defect this code had in review before it had the rule:

1. **No filesystem call ever runs on the producer thread.** Not the mkdir, not
   the ``open``, not the encode. The writer thread resolves the run folder and
   creates it when the FIRST frame arrives; everything the producer does is one
   array copy and one non-blocking queue put. An earlier draft did the mkdir in
   ``setup()``, which both stalled the tick and — because ``runstore.run_dir``
   JOINS a folder only within 90 s — could file the recording under a different
   run than the CSVs it is meant to be compared against.

2. **Bounded queue, drop rather than block.** When the disk cannot keep up the
   frame is dropped and counted. A recorder that applies backpressure to a
   control loop is a recorder that can hurt you.

3. **A failure here never ends the dive.** Every path is caught; the recorder
   disables itself and says why. The caller's construction is wrapped too, so a
   read-only mount cannot stop the worker from starting.

4. **A failed write is never counted as a success.** ``cv2.imwrite`` RETURNS
   False on a full disk rather than raising, so its return value is checked. The
   drop and error counts reach ``meta.json`` and the run record, so a gappy
   recording is always visible as a gap and never mistaken for a gap in the
   data itself.

5. **Shutdown is bounded.** ``close`` waits ~1 s, not ~6, because overrunning
   the worker-stop budget parks the thread and freezes the station's shutdown
   with the vehicle in the water.

Arrays are copied on submit. That is defensive rather than load-bearing — the
mailbox contract already hands over private arrays (``bus.py``) — but the copy
is cheap and makes a recorder independent of that staying true.

What a subclass supplies
------------------------
The class attributes name the folder and the log wording; three methods say
what a frame IS: ``submit`` (public, producer side, builds the item tuple and
hands it to :meth:`_enqueue`), ``_write_item`` (writer side, one item to files
plus its CSV row), and ``_meta_payload`` (what meta.json says beyond the
counters). Nothing else should need overriding.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

#: Queue depth. Four frames is ~0.4 s at the measured 11 Hz depth rate
#: [측정: data/20260903/0903_183555/controller.json,
#: policy.worker.depth_hz = 11.09] — long enough to ride out one slow write,
#: short enough that a stalled disk drops frames immediately instead of
#: silently buffering a second of them.
QUEUE_DEPTH = 4

#: How long ``close`` will wait for the writer to drain. Kept well under the
#: 4 s worker-stop budget in ``backends/base.py``: overrunning it logs a false
#: "thread did not exit" error, parks the thread, and stalls the station's
#: shutdown while the vehicle is still in the water. Losing the last frame or
#: two of a diagnostic is the cheaper failure by a wide margin.
CLOSE_TIMEOUT_S = 1.0


class FrameRecorder:
    """Write frames to a run folder from a worker thread, without blocking it.

    Every method is safe to call when the recorder is disabled, failed or
    already closed; the caller never has to guard.

    ``run_dir_fn`` is a CALLABLE, not a path: it is invoked on the writer
    thread when the first frame arrives, so the run folder is both resolved
    and created at a moment when the vehicle is demonstrably flying. Resolving
    it earlier creates a folder for a launch that never became a run, and can
    name a different folder than the one the run's other writers joined.
    """

    #: Log prefix, e.g. "depth recorder".
    LABEL = "recorder"
    #: Folder under the run dir, and the subfolders created inside it.
    SUBDIR = "frames"
    SUBFOLDERS: tuple = ()
    CSV_HEADER: tuple = ("seq", "t_capture")
    THREAD_NAME = "frame-rec"
    #: The flag that raises the budget, named in the message that reports it.
    BUDGET_FLAG = "--record-max"
    META_SCHEMA = "rov_gui/frame_recording/1"
    #: Why nothing was recorded, before anything has been offered. Never empty
    #: while the recorder is not writing: "nothing was recorded" with no
    #: sentence saying why is the same silent failure the flag exists to end,
    #: and `counters` alone cannot tell "the worker never ran" from "the
    #: recorder blew up on arming".
    WHY_UNFED = "armed but never fed"
    #: PNG compression. 1 is fast; these are diagnostics, not archives, and the
    #: producer's thread budget matters more than the last few percent of size
    #: [측정 2026-09-06, a real 640x400 mono pair from
    #: "~/Desktop/data collection/videos/0/{left,right}.mp4" frame 150:
    #: level 1 = 266 KiB, level 3 = 257 KiB, level 6 = 243 KiB per pair].
    PNG_LEVEL = 1

    def __init__(self, run_dir_fn: Callable[[], Path], *,
                 max_frames: int, on_log=None):
        self._run_dir_fn = run_dir_fn
        self.max_frames = max(1, int(max_frames))
        self._log = on_log or (lambda level, msg: None)

        #: Set by the writer thread once the folder exists. None until then,
        #: and None forever if opening it failed — ``meta()`` distinguishes
        #: the two through ``started`` / ``why``.
        self.dir: Optional[Path] = None
        self.started = False
        self.why = self.WHY_UNFED

        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(QUEUE_DEPTH)
        self._thread: Optional[threading.Thread] = None
        self._csv = None
        self._csv_fh = None
        self._spun = False
        self._closed = False
        self._closed_done = False
        self._seq = 0
        self.counters = {"offered": 0, "written": 0, "dropped_full": 0,
                         "dropped_budget": 0, "dropped_no_writer": 0,
                         "dropped_disabled": 0,
                         "write_errors": 0, "write_errors_writer": 0}
        # A subclass may add its own counters HERE, before the writer thread
        # exists; the key set must never change afterwards (describe() is read
        # from another thread and relies on dict() being tear-free).
        self._meta_extra: dict = {}
        self._t0 = 0.0
        self._cv2 = None

    # ------------------------------------------------------------------ start
    def _start(self, **meta_extra) -> bool:
        """Spin the writer thread. NO filesystem call happens here.

        Returns False (and disables the recorder) on any failure; it never
        raises, because losing a diagnostic must not end a dive.
        """
        if self._spun or self._closed:
            return self._spun and not self._closed
        self._spun = True
        try:
            from ..qt import import_cv2
            self._cv2 = import_cv2()
        except Exception as e:                                   # noqa: BLE001
            # Imported on the worker thread, once, rather than on the writer:
            # import_cv2 mutates a process-global Qt plugin path non-atomically
            # (rov_gui/qt.py), and doing that concurrently with another
            # worker's import is a documented crash mode.
            self.why = f"no cv2: {type(e).__name__}: {e}"
            self._closed = True
            self._log("warn", f"{self.LABEL} disabled — {self.why}")
            return False
        self._meta_extra = dict(meta_extra)
        self._t0 = time.time()
        try:
            self._thread = threading.Thread(target=self._run,
                                            name=self.THREAD_NAME, daemon=True)
            self._thread.start()
        except Exception as e:                                   # noqa: BLE001
            self.why = f"writer thread: {type(e).__name__}: {e}"
            self._closed = True
            self._log("warn", f"{self.LABEL} disabled — {self.why}")
            return False
        self.started = True
        self._log("info", self._arm_note())
        return True

    def _arm_note(self) -> str:
        return (f"{self.LABEL}: armed (max {self.max_frames} frames). The run "
                f"folder is created by the writer thread on the first frame. "
                f"Frames are DROPPED, never queued indefinitely, if the disk "
                f"cannot keep up; the drop and error counts are in meta.json "
                f"and in the run record.")

    # ----------------------------------------------------------------- submit
    def _enqueue(self, make_item: Callable[[int], tuple]) -> None:
        """Offer one frame. Returns immediately; never raises, never blocks.

        Called from a worker tick while the vehicle is armed, so this does
        exactly two things: copy (inside ``make_item``), and try to enqueue. No
        lock is taken and no filesystem is touched.
        """
        # Counted BEFORE the guard: after the writer disables itself, a
        # 20-minute dive would otherwise read `offered: 1`, hiding how much was
        # lost behind a single number that looks like a short run.
        self.counters["offered"] += 1
        if not self.started or self._closed:
            self.counters["dropped_disabled"] += 1
            return
        if self._thread is not None and not self._thread.is_alive():
            self.counters["dropped_no_writer"] += 1
            return
        if self._seq >= self.max_frames:
            self.counters["dropped_budget"] += 1
            if self.counters["dropped_budget"] == 1:
                self._log("warn",
                          f"{self.LABEL}: reached the {self.max_frames}-frame "
                          f"budget; no further frames are written (raise it "
                          f"with {self.BUDGET_FLAG}). The run is unaffected.")
            return
        try:
            item = make_item(self._seq)
        except Exception:                                        # noqa: BLE001
            self.counters["write_errors"] += 1
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.counters["dropped_full"] += 1
            return
        self._seq += 1

    # ------------------------------------------------------------------ close
    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> dict:
        """Drain briefly, then write meta.json. Idempotent, bounded, no raise.

        Bounded on purpose: see CLOSE_TIMEOUT_S. The writer is a daemon, so a
        thread that has not finished is abandoned rather than waited on, and
        meta.json records that its counts may be short.
        """
        if self._closed_done or not self._spun:
            self._closed = True
            self._closed_done = True
            return dict(self.counters)
        self._closed = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass                       # the writer sees _closed and stops
        t = self._thread
        drained = True
        if t is not None:
            t.join(timeout=max(0.1, float(timeout)))
            drained = not t.is_alive()
        try:
            if self._csv_fh is not None and drained:
                self._csv_fh.close()
        except OSError:
            pass
        self._write_meta(drained)
        self._closed_done = True
        c = self.counters
        where = str(self.dir) if self.dir is not None else "(never opened)"
        self._log("info",
                  f"{self.LABEL}: {c['written']} frames written to {where} "
                  f"({c['dropped_full']} dropped for a full queue, "
                  f"{c['dropped_budget']} past the budget, "
                  f"{c['write_errors'] + c['write_errors_writer']} write errors"
                  f"{'' if drained else ', writer still draining at shutdown'})")
        return dict(c)

    # ------------------------------------------------------------------- meta
    def describe(self) -> dict:
        """What the run record says about this recording. Safe from any thread.

        The counters dict never gains or loses keys, so ``dict()`` on it cannot
        raise or tear; the counts can be mutually stale by one frame, which is
        immaterial and cheaper than a lock on the inference path.
        """
        d = {"enabled": True,
             "started": bool(self.started),
             "dir": str(self.dir) if self.dir is not None else None,
             "why": self.why,
             "max_frames": self.max_frames,
             "counters": dict(self.counters)}
        d.update(self._describe_extra())
        return d

    def _describe_extra(self) -> dict:
        return {}

    def _meta_payload(self, drained: bool = True) -> dict:
        return {}

    def _write_meta(self, drained: bool = True) -> None:
        if self.dir is None:
            return                     # nothing was ever opened; nowhere to write
        meta = {
            "schema": self.META_SCHEMA,
            "t_start_wall": self._t0,
            "duration_s": max(0.0, time.time() - self._t0),
            "counters": dict(self.counters),
            "counts_complete": bool(drained),
            "max_frames": self.max_frames,
        }
        meta.update(self._meta_payload(drained))
        try:
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2,
                                                           default=str))
        except OSError as e:
            self._log("warn", f"{self.LABEL}: could not write meta.json: {e}")

    # ----------------------------------------------------------------- writer
    def _open(self) -> bool:
        """Resolve and create the run folder. WRITER THREAD ONLY."""
        try:
            d = Path(self._run_dir_fn()) / self.SUBDIR
            d.mkdir(parents=True, exist_ok=True)
            for sub in self.SUBFOLDERS:
                (d / sub).mkdir(parents=True, exist_ok=True)
            fh = (d / "index.csv").open("w", newline="")
        except Exception as e:                                   # noqa: BLE001
            self.why = f"could not open the run folder: {type(e).__name__}: {e}"
            self._log("warn", f"{self.LABEL} disabled — {self.why}")
            return False
        try:
            w = csv.writer(fh)
            w.writerow(list(self.CSV_HEADER))
            fh.flush()
        except Exception as e:                                   # noqa: BLE001
            # Close the handle we just opened: this is the disk-full path the
            # recorder is meant to survive, and leaking an fd through the whole
            # dive is not surviving it.
            try:
                fh.close()
            except OSError:
                pass
            self.why = f"could not write index.csv: {type(e).__name__}: {e}"
            self._log("warn", f"{self.LABEL} disabled — {self.why}")
            return False
        self._csv_fh, self._csv = fh, w
        self.why = ""                  # writing; nothing left to explain
        self.dir = d                   # published last: a non-None dir is usable
        self._log("info", f"{self.LABEL}: writing to {d}")
        return True

    def _write_item(self, cv2, png, item) -> tuple:
        """One queued item -> files on disk. Returns (ok, csv_row).

        WRITER THREAD ONLY. Must not raise: a failure is ``(False, ...)``, and
        rule 4 says a False here is never counted as written.
        """
        raise NotImplementedError

    def _run(self) -> None:
        cv2 = self._cv2
        png = [cv2.IMWRITE_PNG_COMPRESSION, self.PNG_LEVEL]
        opened = False
        while True:
            item = self._q.get()
            # `_closed` as well as the sentinel: close() puts the sentinel with
            # put_nowait, which SILENTLY FAILS on a queue that was already full
            # when the disk stalled. Breaking on the flag too means a lost
            # sentinel costs at most the frames still queued, instead of
            # leaving this thread blocked in get() for the life of the process
            # holding the csv handle open and appending rows after meta.json
            # was written.
            if item is None or self._closed:
                break
            if not opened:
                if not self._open():
                    self._closed = True             # stop the producer copying
                    return
                opened = True
            try:
                ok, row = self._write_item(cv2, png, item)
            except Exception:                                    # noqa: BLE001
                ok, row = False, None
            if not ok:
                self.counters["write_errors_writer"] += 1
                continue
            try:
                self._csv.writerow(row)
                self._csv_fh.flush()
            except (OSError, ValueError):
                self.counters["write_errors_writer"] += 1
                continue
            self.counters["written"] += 1


__all__ = ["FrameRecorder", "QUEUE_DEPTH", "CLOSE_TIMEOUT_S"]
