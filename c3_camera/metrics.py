#!/usr/bin/env python3
"""
metrics.py — fps, latency and drop accounting per stream.

Latency
-------
DepthAI stamps every ImgFrame on the device and runs a timesync service that
keeps the device clock in the same domain as `dai.Clock.now()` on the host.  So

    latency = dai.Clock.now() - frame.getTimestamp()

is a genuine one-way sensor-to-host interval, not a round trip, and it does not
require the host and device clocks to be manually offset.

What it includes: sensor readout, ISP, (stereo compute for depth), device-side
queueing, XLink/TCP transfer over PoE, and host queue wait until we popped it.
What it excludes: our own colour conversion, colourisation, and `cv2.imshow`.
`Frame.age_ms()` covers that remainder, and the HUD shows both.

`getTimestamp()` refers to the middle of exposure by default; at long exposures
in dark water that shifts the reported latency by up to half the exposure time,
which is why the HUD also reports exposure. `getTimestamp(START)` / `(END)` are
available if you need a different reference point.

Drops
-----
Sequence numbers are assigned on the device before transmission, so a gap means
a frame was produced and then discarded — either by our non-blocking device-side
queue (the link could not keep up) or by the host queue. `requested_fps` minus
`fps` tells you the same story from the other side.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import depthai as dai


def now_s() -> float:
    """Host time in the same domain as ImgFrame timestamps, in seconds."""
    return dai.Clock.now().total_seconds()


def frame_latency_ms(frame, offset: dai.CameraExposureOffset | None = None) -> float:
    """Sensor-to-host latency for one DepthAI message, in milliseconds."""
    ts = frame.getTimestamp() if offset is None else frame.getTimestamp(offset)
    return (dai.Clock.now() - ts).total_seconds() * 1000.0


@dataclass
class StreamStats:
    """Rolling statistics for a single stream."""

    name: str
    window: int = 120

    _arrivals: deque[float] = field(default_factory=lambda: deque(maxlen=600), repr=False)
    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=600), repr=False)
    _sizes: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=600), repr=False)
    count: int = 0
    dropped: int = 0
    total_bytes: int = 0
    last_seq: int | None = None
    last_latency_ms: float = float("nan")
    last_exposure_ms: float = float("nan")
    first_arrival: float | None = None
    last_arrival: float | None = None

    def update(self, frame, wire_bytes: int = 0) -> float:
        """Record one received frame; returns its latency in ms.

        `wire_bytes` is what actually crossed the link for this frame, so the
        reported throughput is measured rather than assumed. That matters most
        for MJPEG, where the compression ratio depends on the scene.
        """
        t = time.monotonic()
        lat = frame_latency_ms(frame)

        self.count += 1
        self._arrivals.append(t)
        self._latencies.append(lat)
        self.last_latency_ms = lat
        if wire_bytes:
            self.total_bytes += wire_bytes
            self._sizes.append((t, wire_bytes))
        if self.first_arrival is None:
            self.first_arrival = t
        self.last_arrival = t

        try:
            self.last_exposure_ms = frame.getExposureTime().total_seconds() * 1000.0
        except Exception:  # noqa: BLE001 - depth frames carry no exposure
            pass

        seq = frame.getSequenceNum()
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.dropped += seq - self.last_seq - 1
        self.last_seq = seq
        return lat

    # -------------------------------------------------------------- readouts
    @property
    def fps(self) -> float:
        """Received fps over the recent window (not a lifetime average)."""
        if len(self._arrivals) < 2:
            return 0.0
        recent = list(self._arrivals)[-self.window:]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    @property
    def fps_lifetime(self) -> float:
        if self.first_arrival is None or self.last_arrival is None or self.count < 2:
            return 0.0
        span = self.last_arrival - self.first_arrival
        return (self.count - 1) / span if span > 0 else 0.0

    def _lat_window(self) -> list[float]:
        return list(self._latencies)[-self.window:]

    @property
    def latency_mean_ms(self) -> float:
        w = self._lat_window()
        return sum(w) / len(w) if w else float("nan")

    def latency_pct(self, pct: float) -> float:
        w = sorted(self._lat_window())
        if not w:
            return float("nan")
        i = min(len(w) - 1, max(0, int(round((pct / 100.0) * (len(w) - 1)))))
        return w[i]

    @property
    def latency_p50_ms(self) -> float:
        return self.latency_pct(50)

    @property
    def latency_p95_ms(self) -> float:
        return self.latency_pct(95)

    @property
    def latency_max_ms(self) -> float:
        w = self._lat_window()
        return max(w) if w else float("nan")

    @property
    def drop_rate(self) -> float:
        produced = self.count + self.dropped
        return self.dropped / produced if produced else 0.0

    @property
    def mbps(self) -> float:
        """Measured link throughput for this stream over the recent window."""
        recent = list(self._sizes)[-self.window:]
        if len(recent) < 2:
            return 0.0
        span = recent[-1][0] - recent[0][0]
        if span <= 0:
            return 0.0
        # Skip the first sample's bytes: they arrived before the window opened.
        total = sum(b for _, b in recent[1:])
        return total * 8 / span / 1e6

    @property
    def mean_frame_kb(self) -> float:
        recent = list(self._sizes)[-self.window:]
        return (sum(b for _, b in recent) / len(recent) / 1024) if recent else 0.0

    def hud_line(self) -> str:
        return (f"{self.name:<6} {self.fps:5.1f} fps  "
                f"lat {self.last_latency_ms:6.1f} / p50 {self.latency_p50_ms:6.1f} / "
                f"p95 {self.latency_p95_ms:6.1f} ms  "
                f"drop {self.dropped:4d} ({self.drop_rate*100:4.1f}%)  "
                f"{self.mbps:5.1f} Mbit/s")

    def summary(self) -> dict:
        return {
            "stream": self.name,
            "frames": self.count,
            "dropped": self.dropped,
            "drop_rate": round(self.drop_rate, 4),
            "fps_window": round(self.fps, 2),
            "fps_lifetime": round(self.fps_lifetime, 2),
            "mbps_measured": round(self.mbps, 1),
            "mean_frame_kb": round(self.mean_frame_kb, 1),
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_max_ms": round(self.latency_max_ms, 1),
            "exposure_ms": (None if self.last_exposure_ms != self.last_exposure_ms
                            else round(self.last_exposure_ms, 2)),
        }


class Metrics:
    """All stream statistics, plus the display-loop rate."""

    def __init__(self, streams, window: int = 120):
        self.streams = {s: StreamStats(s, window=window) for s in streams}
        self.loop = StreamStats("loop", window=window)
        self._loop_marks: deque[float] = deque(maxlen=600)

    def update(self, name: str, frame, wire_bytes: int = 0) -> float:
        if name not in self.streams:
            self.streams[name] = StreamStats(name)
        return self.streams[name].update(frame, wire_bytes)

    @property
    def mbps_total(self) -> float:
        """Measured throughput across every stream — the number to compare
        against the PoE ceiling."""
        return sum(st.mbps for st in self.streams.values())

    def mark_loop(self) -> None:
        self._loop_marks.append(time.monotonic())

    @property
    def loop_hz(self) -> float:
        if len(self._loop_marks) < 2:
            return 0.0
        recent = list(self._loop_marks)[-120:]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def hud_lines(self) -> list[str]:
        lines = [st.hud_line() for st in self.streams.values() if st.count]
        lines.append(f"{'loop':<6} {self.loop_hz:5.1f} Hz")
        return lines

    def summary(self) -> dict:
        return {
            "streams": [st.summary() for st in self.streams.values()],
            "loop_hz": round(self.loop_hz, 2),
            "mbps_measured_total": round(self.mbps_total, 1),
        }
