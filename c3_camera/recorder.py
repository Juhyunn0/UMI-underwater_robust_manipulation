#!/usr/bin/env python3
"""
recorder.py — write an RGB-D sequence to disk without disturbing the stream.

Design constraints, in priority order:

1. **Never stall the stream loop.** Encoding a 16-bit PNG takes a few ms, which
   at 20 fps is a meaningful slice of the frame interval; doing it inline would
   add latency and cause the device to drop frames — undoing the whole point of
   this client. So frames go onto a bounded queue and a writer thread does the
   I/O. If the queue fills, frames are dropped and *counted*, because a silent
   gap in a research recording is far worse than a reported one.
2. **Depth must stay lossless.** Depth is uint16 millimetres; a lossy codec
   would quietly corrupt the measurement. 16-bit PNG (default) or raw .npy.
3. **Don't re-encode what is already encoded.** With `--color-encode mjpeg` the
   colour frames arrive as JPEG bitstreams, so recording writes those bytes
   straight through: no decode, no re-encode, no generation loss beyond the one
   the link already imposed. This is the cheapest possible colour path.
4. **Timestamps travel with the pixels.** `t_device` (capture time) is what makes
   a recording replayable and fusable later, so every frame's device timestamp,
   sequence number and measured latency go into the index.

Layout
------
    <out_dir>/
        meta.json      config, device, intrinsics, calibration, clock offsets
        frames.csv     one row per recorded frame -> file names + timestamps
        color/000001.jpg   (or .png when colour is raw)
        depth/000001.png   16-bit PNG, millimetres  (or .npy)

That is a plain-files layout on purpose: it replays with `c3_replay.py`, loads
with cv2/numpy from anything, and survives a killed process — a container format
truncated mid-write usually does not.
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FRAME_FIELDS = (
    "idx", "color_file", "depth_file",
    "color_seq", "depth_seq",
    "color_t_device", "depth_t_device", "skew_ms",
    "color_latency_ms", "depth_latency_ms", "exposure_ms",
    "t_wall",
)


@dataclass
class _Item:
    """One frame pair, already detached from depthai's buffers."""

    idx: int
    color: np.ndarray | None
    color_raw: bytes | None
    depth: np.ndarray | None
    row: dict


@dataclass
class RecorderStats:
    written: int = 0
    dropped: int = 0
    waited: int = 0          # frames kept only because submit() waited for space
    bytes_written: int = 0
    write_ms_total: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def queue_drop_rate(self) -> float:
        total = self.written + self.dropped
        return self.dropped / total if total else 0.0

    @property
    def mean_write_ms(self) -> float:
        return self.write_ms_total / self.written if self.written else 0.0

    @property
    def mb_per_s(self) -> float:
        el = time.monotonic() - self.started_at
        return (self.bytes_written / 1e6 / el) if el > 0 else 0.0


class Recorder:
    """Background RGB-D writer. Construct, `submit()` bundles, then `close()`."""

    def __init__(self, out_dir: Path,
                 depth_format: str = "png",
                 color_quality: int = 95,
                 queue_size: int = 16,
                 block_ms: float = 50.0,
                 verbose: bool = True):
        if depth_format not in ("png", "npy"):
            raise ValueError(f"depth_format must be png or npy, got {depth_format!r}")

        self.dir = Path(out_dir)
        self.depth_format = depth_format
        self.color_quality = color_quality
        # How long submit() may wait for queue space before giving up on a frame.
        # Dropping instantly is too eager: writing a frame costs well under a
        # millisecond, so a full queue is nearly always a brief burst that clears
        # in a few ms, and a short bounded wait keeps the data instead of losing
        # it. Bounded, so a genuinely too-slow disk still degrades into dropping
        # rather than throttling the camera.
        self.block_s = max(0.0, block_ms / 1000.0)
        self.verbose = verbose
        self.stats = RecorderStats()

        self.dir.mkdir(parents=True, exist_ok=True)
        self.color_dir = self.dir / "color"
        self.depth_dir = self.dir / "depth"

        self._idx = 0
        self._last_color_seq: int | None = None
        self._last_depth_seq: int | None = None
        self._closed = False

        self._q: queue.Queue[_Item | None] = queue.Queue(maxsize=queue_size)
        self._csv_fh = (self.dir / "frames.csv").open("w", newline="")
        self._csv = csv.DictWriter(self._csv_fh, fieldnames=FRAME_FIELDS)
        self._csv.writeheader()

        self._thread = threading.Thread(target=self._run, name="c3-recorder",
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ meta
    def write_meta(self, source_summary: dict, extra: dict | None = None) -> None:
        """Everything needed to interpret the recording later.

        Includes the wall-clock/dai-clock offset, because `t_device` is on a
        monotonic clock that means nothing after the process exits unless you can
        map it back to absolute time.
        """
        import depthai as dai

        meta = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "clock": {
                "dai_now_s": dai.Clock.now().total_seconds(),
                "time_time_s": time.time(),
                "note": "t_device is in the dai.Clock (steady) domain; "
                        "absolute = t_device - dai_now_s + time_time_s",
            },
            "depth_units": "millimetre, uint16, 0 = no measurement",
            "depth_format": self.depth_format,
            "layout": {
                "color": "color/<idx>.jpg|png",
                "depth": f"depth/<idx>.{self.depth_format}",
                "index": "frames.csv",
            },
        }
        meta.update(source_summary or {})
        if extra:
            meta.update(extra)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---------------------------------------------------------------- submit
    def submit(self, bundle) -> bool:
        """Queue one bundle. Returns False if it had to be dropped.

        Only *fresh* frames are worth recording: with pair_mode="latest" the same
        depth frame can appear in several consecutive bundles, and writing it
        repeatedly would inflate the recording with duplicates. Sequence numbers
        are the reliable test for that.
        """
        if self._closed:
            return False

        c, d = bundle.color, bundle.depth
        c_new = c is not None and c.seq != self._last_color_seq
        d_new = d is not None and d.seq != self._last_depth_seq
        if not (c_new or d_new):
            return True   # nothing new; not a drop

        self._idx += 1
        idx = self._idx
        if c is not None:
            self._last_color_seq = c.seq
        if d is not None:
            self._last_depth_seq = d.seq

        color_ext = "jpg" if (c is not None and c.raw is not None) else "png"
        row = {
            "idx": idx,
            "color_file": f"color/{idx:06d}.{color_ext}" if c is not None else "",
            "depth_file": f"depth/{idx:06d}.{self.depth_format}" if d is not None else "",
            "color_seq": c.seq if c else "",
            "depth_seq": d.seq if d else "",
            "color_t_device": f"{c.t_device:.6f}" if c else "",
            "depth_t_device": f"{d.t_device:.6f}" if d else "",
            "skew_ms": (f"{bundle.skew_ms:.2f}"
                        if bundle.skew_ms == bundle.skew_ms else ""),
            "color_latency_ms": f"{c.latency_ms:.2f}" if c else "",
            "depth_latency_ms": f"{d.latency_ms:.2f}" if d else "",
            "exposure_ms": (f"{c.exposure_ms:.3f}"
                            if c and c.exposure_ms == c.exposure_ms else ""),
            "t_wall": f"{time.time():.6f}",
        }

        item = _Item(
            idx=idx,
            # c.image is already an independently owned array (imdecode/getCvFrame
            # allocate); depth is copied at the source. Nothing to clone here.
            color=c.image if (c is not None and c.raw is None) else None,
            color_raw=c.raw if c is not None else None,
            depth=d.image if d is not None else None,
            row=row,
        )
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            pass
        # Queue momentarily full: wait a bounded time for the writer to catch up.
        try:
            if self.block_s > 0:
                self._q.put(item, timeout=self.block_s)
                self.stats.waited += 1
                return True
        except queue.Full:
            pass
        self.stats.dropped += 1
        self._idx -= 1              # this index was never written
        return False

    # ---------------------------------------------------------------- writer
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            t0 = time.monotonic()
            try:
                self._write(item)
                self.stats.written += 1
                self._csv.writerow(item.row)
            except Exception as e:  # noqa: BLE001 - a bad frame must not kill the thread
                if self.verbose:
                    print(f"  recorder: failed to write frame {item.idx}: "
                          f"{type(e).__name__}: {e}")
            finally:
                self.stats.write_ms_total += (time.monotonic() - t0) * 1000.0
                self._q.task_done()

    def _write(self, item: _Item) -> None:
        if item.color_raw is not None:
            self.color_dir.mkdir(parents=True, exist_ok=True)
            path = self.color_dir / f"{item.idx:06d}.jpg"
            path.write_bytes(item.color_raw)          # straight through, no re-encode
            self.stats.bytes_written += len(item.color_raw)
        elif item.color is not None:
            self.color_dir.mkdir(parents=True, exist_ok=True)
            path = self.color_dir / f"{item.idx:06d}.png"
            cv2.imwrite(str(path), item.color)
            self.stats.bytes_written += path.stat().st_size

        if item.depth is not None:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            if self.depth_format == "npy":
                path = self.depth_dir / f"{item.idx:06d}.npy"
                np.save(path, item.depth)
            else:
                path = self.depth_dir / f"{item.idx:06d}.png"
                # 16-bit PNG keeps every millimetre; cv2 writes uint16 losslessly.
                cv2.imwrite(str(path), item.depth)
            self.stats.bytes_written += path.stat().st_size

    # ----------------------------------------------------------------- close
    def close(self, timeout_s: float = 30.0) -> RecorderStats:
        """Flush the queue and stop the writer. Safe to call twice."""
        if self._closed:
            return self.stats
        self._closed = True
        try:
            self._q.put(None, timeout=5.0)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive() and self.verbose:
            print(f"  recorder: writer still busy after {timeout_s:g}s; "
                  f"{self._q.qsize()} frames may be missing")
        self._csv_fh.close()
        return self.stats

    # ------------------------------------------------------------------- HUD
    def hud_line(self) -> str:
        s = self.stats
        el = time.monotonic() - s.started_at
        return (f"REC {s.written:5d} frames  {el:5.1f}s  "
                f"{s.mb_per_s:5.1f} MB/s  {s.bytes_written/1e6:6.1f} MB  "
                f"q{self._q.qsize()}  dropped {s.dropped}")

    def summary(self) -> dict:
        s = self.stats
        return {
            "dir": str(self.dir),
            "frames": s.written,
            "dropped": s.dropped,
            "waited_for_disk": s.waited,
            "queue_drop_rate": round(s.queue_drop_rate, 4),
            "megabytes": round(s.bytes_written / 1e6, 1),
            "mb_per_s": round(s.mb_per_s, 2),
            "mean_write_ms": round(s.mean_write_ms, 2),
            "seconds": round(time.monotonic() - s.started_at, 1),
        }


def estimate_disk_mb_per_min(color_kb: float, depth_wh: tuple[int, int],
                             fps: float, depth_format: str = "png") -> float:
    """Rough recording rate, for deciding whether the disk will keep up.

    Depth PNG compresses well on real scenes (large invalid regions collapse), so
    ~45% of raw is a reasonable planning figure; .npy is exactly raw.
    """
    depth_raw_kb = depth_wh[0] * depth_wh[1] * 2 / 1024
    depth_kb = depth_raw_kb if depth_format == "npy" else depth_raw_kb * 0.45
    return (color_kb + depth_kb) * fps * 60 / 1024
