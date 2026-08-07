#!/usr/bin/env python3
"""
sensorlog.py — the sensor stream that accompanies a depth recording.

A depth video on its own is not a dataset. What makes it one is knowing where
the camera was and how it was moving while each frame was captured, and that
lives in the vehicle's IMU, barometer and compass — at rates far above the
video's. So when the C3 Depth feed is being recorded, every sensor record that
arrives is written beside it.

Format: JSONL, one record per line, exactly as the source produced it.

    {"t": 1234.567, "t_unix": 1786079630.903, "src": "rov", "msg": "RAW_IMU",
     "xacc": 12, "yacc": -3, ...}

Not CSV, and the reason is the failure mode CSV has here: these sources emit
half a dozen message types with disjoint fields at different rates, so a CSV
needs either a union schema full of blanks or one file per type — and the union
schema silently drops any field a firmware version adds. JSONL keeps whatever
arrived. ``pandas.read_json(path, lines=True)`` loads it in one call.

Clocks
------
``t`` is ``time.monotonic()`` on the topside host — the same clock the video
recorder and every other timestamp in this package use, so it is what lines
records up against frames. ``t_unix`` is wall time for cross-referencing other
systems. Neither is the sensor's own timestamp: MAVLink records carry the
vehicle's ``time_boot_ms`` and the C3's IMU carries device time, both of which
are preserved in the record itself when the source provides them. Those are the
ones to use for anything time-critical — the host timestamp includes link
latency and this file does not pretend otherwise.

Writing happens on the worker thread that owns the source, which is deliberate:
these are small appends to a buffered file, far cheaper than the frame the same
thread just decoded, and routing them through a queue would add a second place
for the data to be lost.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path


class SensorLog:
    """One JSONL sink. Thread-safe, append-only, honest about what it wrote.

    Never raises at the caller: a recording that fails must not take the video
    (or the control station) down with it. The first error is kept and reported
    in the sidecar, and writing stops there rather than filling a disk with the
    same message.
    """

    def __init__(self, path: Path | str, source: str):
        self.path = Path(path)
        self.source = source
        self.counts: Counter[str] = Counter()
        self.error: str = ""
        self.started_at = time.monotonic()
        self.started_unix = time.time()
        self._lock = threading.Lock()
        self._fh = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", buffering=1 << 16)
        except OSError as e:
            self.error = f"{type(e).__name__}: {e}"

    @property
    def open(self) -> bool:
        return self._fh is not None

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def write(self, msg: str, fields: dict, t: float | None = None) -> None:
        """Append one record. Any thread; never raises."""
        if self._fh is None:
            return
        rec = {"t": time.monotonic() if t is None else t,
               "t_unix": time.time(), "src": self.source, "msg": msg}
        # Source fields last so a source can carry its own device timestamp
        # without us renaming it, but never let it overwrite our clock.
        for k, v in fields.items():
            if k not in ("t", "t_unix", "src", "msg"):
                rec[k] = v
        try:
            self._fh.write(json.dumps(rec, default=_jsonable) + "\n")
            self.counts[msg] += 1
        except (OSError, TypeError, ValueError) as e:
            if not self.error:
                self.error = f"{type(e).__name__}: {e}"
            self.close()

    def close(self) -> dict:
        """Close and return the manifest (also written as a .json sidecar)."""
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError as e:
                if not self.error:
                    self.error = f"{type(e).__name__}: {e}"
        dur = time.monotonic() - self.started_at
        meta = {
            "file": self.path.name,
            "source": self.source,
            "duration_s": round(dur, 2),
            "records": self.total,
            "by_message": dict(self.counts),
            # The rate we actually captured, per message type. This is the
            # number to check before trusting the log: a 200 Hz IMU that shows
            # up here at 47 Hz means the link, not the sensor, set the rate.
            "hz_measured": {k: round(v / dur, 1) for k, v in self.counts.items()
                            if dur > 0},
            "clock": "t = topside time.monotonic(); t_unix = topside wall clock; "
                     "sensor-side timestamps are preserved in each record",
            "error": self.error or None,
        }
        try:
            self.path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        except OSError:
            pass
        return meta


def _jsonable(v):
    """Last-resort encoder: bytes and odd MAVLink field types become strings."""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)
