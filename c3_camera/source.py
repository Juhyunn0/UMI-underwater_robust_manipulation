#!/usr/bin/env python3
"""
source.py — C3Source: the programmatic RGB-D interface.

This is the module downstream code should use; the scripts are thin shells over
it.  Typical use:

    from c3_camera.config import StreamConfig
    from c3_camera.source import C3Source

    with C3Source(StreamConfig(fps=15)) as src:
        for bundle in src.stream():
            colour = bundle.color.image          # HxWx3 uint8 BGR
            depth  = bundle.depth.image          # HxW  uint16 millimetres
            xyz, rgb = bundle.pointcloud(stride=4)

ROS integration seam
--------------------
`Frame` carries exactly what a ROS 2 publisher needs and nothing ROS-specific:

    Frame.image      -> sensor_msgs/Image  (bgr8, or 16UC1 for depth)
    Frame.t_device   -> header.stamp       (device clock, timesynced to the host)
    Frame.seq        -> device sequence number, for gap detection
    src.intrinsics   -> sensor_msgs/CameraInfo (K, D, at the streamed size)

Use `t_device`, not host arrival time, as the message stamp: it is the capture
instant, so it stays correct even when the link hiccups. That is the difference
between a TF tree that holds up under load and one that drifts.

Waiting without spinning
------------------------
The loop blocks in `Device.getQueueEvents(...)` until some queue has data,
instead of polling in a spin loop. Waking on arrival adds no latency and leaves
the CPU alone.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import cv2
import depthai as dai
import numpy as np

from . import device as D
from .config import (
    ResolvedConfig,
    SOCKET_COLOR,
    SOCKET_LEFT,
    SOCKET_RIGHT,
    STREAM_COLOR,
    STREAM_DEPTH,
    STREAM_LEFT,
    STREAM_RIGHT,
    StreamConfig,
)
from .geometry import Intrinsics, intrinsics_from_calibration, rgbd_to_pointcloud
from .imu import ImuStats, read_imu_extrinsics
from .metrics import Metrics, frame_latency_ms
from .pipeline import build_pipeline


# Wire formats whose payload size is not w*h*bpp, so the real packet length has
# to be read instead of computed. "mjpeg_gray" is the encoded mono pair: same
# JPEG bitstream as colour, but it decodes back to one channel (see _decode).
_ENCODED_FORMATS = ("mjpeg", "mjpeg_gray", "h264", "h265")


def _annexb_nal_spans(data: bytes) -> list[tuple[int, int]]:
    """Annex-B start-code offsets as (start, prefix_length) pairs.

    `bytes.find` rather than a per-byte loop: this runs on the capture thread
    for every encoded packet, and a 4 Mbit/s H.265 stream at 20 fps is ~25 kB a
    frame. The byte-at-a-time version built two throwaway slices per byte, which
    turned a host-CPU cost into what looks like a link limit in exactly the
    benchmark that reads these numbers.
    """
    starts: list[tuple[int, int]] = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            return starts
        # A 4-byte start code is a 3-byte one with an extra leading zero.
        if j > 0 and data[j - 1] == 0:
            starts.append((j - 1, 4))
        else:
            starts.append((j, 3))
        i = j + 3


def _annexb_nal_types(data: bytes, codec: str, spans=None):
    """Yield (nal_type, start, end) for each NAL unit, payload-bearing only."""
    starts = _annexb_nal_spans(data) if spans is None else spans
    for n, (start, prefix) in enumerate(starts):
        payload = start + prefix
        end = starts[n + 1][0] if n + 1 < len(starts) else len(data)
        if payload >= end:
            continue
        b0 = data[payload]
        yield ((b0 & 0x1F) if codec == "h264" else ((b0 >> 1) & 0x3F)), start, end


def _annexb_parameter_sets(data: bytes, codec: str, spans=None) -> dict[int, bytes]:
    """Return VPS/SPS/PPS NAL units, retaining each Annex-B start code."""
    wanted = {7, 8} if codec == "h264" else {32, 33, 34}
    return {t: data[s:e] for t, s, e in _annexb_nal_types(data, codec, spans)
            if t in wanted}


def _annexb_frame_type(data: bytes, codec: str, spans=None) -> str:
    """'I' if the packet carries a keyframe NAL, else 'P'.

    `ImgFrame.getFrameType()` is the obvious source and does NOT exist on
    depthai 2.32 (the version pinned here) — `hasattr` is False, and the call
    below it raises AttributeError into a bare except. That left frame_type ""
    on every packet, which is not a cosmetic loss: DatasetWriter opens its
    encoded timeline on the first `frame_type == "I"`, so with a permanently
    empty value the gate never opens and an H.264/H.265 dataset records ZERO
    frames while reporting no error.

    The bytes already in hand answer the question. H.264 nal_unit_type 5 is an
    IDR slice; H.265 types 16-23 are the IRAP class (BLA/IDR/CRA), any of which
    is a valid random-access point. Empty input is treated as 'P', because a
    packet we cannot parse must not be trusted to start a decodable stream.
    """
    for nal, _s, _e in _annexb_nal_types(data or b"", codec, spans):
        if codec == "h264" and nal == 5:
            return "I"
        if codec == "h265" and 16 <= nal <= 23:
            return "I"
    return "P"


@dataclass
class Frame:
    """One decoded image or encoded packet plus its timing provenance."""

    name: str
    # H.264/H.265 stay encoded during collection. For those formats image is
    # None and `raw` holds the exact Annex-B packet bytes.
    image: np.ndarray | None
    seq: int
    t_device: float      # capture time, seconds in the dai.Clock domain
    t_recv: float        # dai.Clock.now() when the host popped it
    latency_ms: float    # (t_recv - t_device) * 1000
    exposure_ms: float = float("nan")
    # Compressed bytes exactly as they came off the wire, kept only when the
    # source was opened with keep_raw=True (see C3Source). Lets the recorder
    # store MJPEG colour without re-encoding it.
    raw: bytes | None = None
    encoding: str = ""
    frame_type: str = ""
    codec_header: bytes = b""

    def age_ms(self) -> float:
        """Total staleness now, including whatever the host has since spent."""
        return (dai.Clock.now().total_seconds() - self.t_device) * 1000.0

    @property
    def shape(self) -> tuple[int, ...]:
        if self.image is None:
            raise RuntimeError(f"{self.name} is {self.encoding}, not decoded")
        return self.image.shape


@dataclass
class Bundle:
    """The latest frame of each stream, with a note of which just arrived."""

    frames: dict[str, Frame] = field(default_factory=dict)
    fresh: frozenset[str] = frozenset()
    intrinsics: dict[str, Intrinsics] = field(default_factory=dict)

    @property
    def color(self) -> Frame | None:
        return self.frames.get(STREAM_COLOR)

    @property
    def depth(self) -> Frame | None:
        return self.frames.get(STREAM_DEPTH)

    @property
    def left(self) -> Frame | None:
        return self.frames.get(STREAM_LEFT)

    @property
    def right(self) -> Frame | None:
        return self.frames.get(STREAM_RIGHT)

    @property
    def skew_ms(self) -> float:
        """Capture-time difference between colour and depth."""
        c, d = self.color, self.depth
        if c is None or d is None:
            return float("nan")
        return abs(c.t_device - d.t_device) * 1000.0

    def pointcloud(self, stride: int = 2, min_mm: float = 100.0,
                   max_mm: float = 20000.0):
        """Coloured cloud from this bundle, in metres, camera optical frame."""
        if self.depth is None:
            raise RuntimeError("no depth stream in this bundle")
        intr = self.intrinsics.get(STREAM_DEPTH)
        if intr is None:
            raise RuntimeError("no depth intrinsics available")
        if self.color is None:
            from .geometry import depth_to_xyz
            return depth_to_xyz(self.depth.image, intr, stride, min_mm, max_mm), None
        if self.color.image is None:
            raise RuntimeError(
                f"colour is {self.color.encoding}; extract/decode it before "
                "building a point cloud")
        return rgbd_to_pointcloud(self.color.image, self.depth.image, intr,
                                  stride, min_mm, max_mm)


class C3Source:
    """Owns the device connection, the queues, and the decoding."""

    def __init__(self, cfg: StreamConfig, verbose: bool = True,
                 retries: int = 3, strict: bool = False,
                 keep_raw: bool = False):
        self.cfg = cfg
        self.rc: ResolvedConfig = cfg.resolve()
        self.verbose = verbose
        self.retries = retries
        self.strict = strict
        # keep_raw makes each MJPEG colour Frame carry its original compressed
        # bytes, so a recorder can write the JPEG straight to disk instead of
        # re-encoding a decoded image. Costs one copy per frame; off by default.
        self.keep_raw = keep_raw

        self.device: dai.Device | None = None
        self.queues: dict[str, dai.DataOutputQueue] = {}
        self.imu_queue: dai.DataOutputQueue | None = None
        self.formats: dict[str, str] = {}
        self.intrinsics: dict[str, Intrinsics] = {}
        self.metrics = Metrics(cfg.streams)
        self.imu_stats = ImuStats()
        self.imu_extrinsics: dict = {}
        self.device_report: dict = {}

        self._latest: dict[str, Frame] = {}
        self._encoded_pending: list[Frame] = []
        self._codec_parameter_sets: dict[str, dict[int, bytes]] = {}

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> "C3Source":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> "C3Source":
        rc = self.rc
        if self.verbose:
            for w in rc.warnings:
                print(f"  warning: {w}")
            print(f"  config : {rc.describe()}")
            print(f"  budget : {rc.budget_note()}")

        pipeline, self.formats = build_pipeline(rc)

        if self.verbose:
            print(f"  opening {self.cfg.mxid or self.cfg.ip} "
                  f"(~{D.TYPICAL_POE_BOOT_S:.0f} s: firmware upload + boot over PoE) ...")
        self.device = D.open_device(pipeline, ip=self.cfg.ip, mxid=self.cfg.mxid,
                                    retries=self.retries, verbose=self.verbose)

        problems = rc.validate_against_device(self.device)
        if problems:
            for p in problems:
                print(f"  device mismatch: {p}")
            if self.strict:
                self.close()
                raise RuntimeError("requested config is not supported by this device")

        self.device_report = D.describe_device(self.device)
        if self.verbose:
            r = self.device_report
            print(f"  device : {r['product']} mxid={r['mxid']} "
                  f"bootloader={r['bootloader']}")

        # Timesync keeps device timestamps in the host's dai.Clock domain, which
        # is what makes the latency numbers meaningful. It is on by default; this
        # tightens it (500 ms period, 10 samples) so the estimate settles fast.
        try:
            self.device.setTimesync(datetime.timedelta(milliseconds=500), 10, True)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"  note: could not reconfigure timesync ({e}); using defaults")

        for name in self.cfg.streams:
            # Inter-frame codecs must not lose a queued P frame merely because
            # the host was briefly busy. Other image streams deliberately keep
            # a tiny low-latency queue; encoded RGB gets 1.5 seconds of headroom.
            queue_size = max(1, self.cfg.queue_size)
            if self.formats.get(name) in ("h264", "h265"):
                queue_size = max(queue_size, int(round(self.cfg.fps * 1.5)))
            blocking = (self.cfg.queue_blocking or
                        self.formats.get(name) in ("h264", "h265"))
            self.queues[name] = self.device.getOutputQueue(
                name, maxSize=queue_size,
                blocking=blocking)

        # The IMU gets its own deeper queue and is NOT part of cfg.streams: it is
        # not an image stream, arrives an order of magnitude more often, and for a
        # dataset we would rather buffer a burst than drop samples.
        if getattr(self.cfg, "imu_enable", False):
            self.imu_queue = self.device.getOutputQueue("imu", maxSize=50,
                                                        blocking=False)

        self._load_intrinsics()
        return self

    # -------------------------------------------------------------------- IMU
    def drain_imu(self) -> list:
        """Pop every IMU sample that has arrived. Returns [] when the IMU is off.

        Kept separate from poll() so the caller can drain the IMU on its own
        cadence — image frames arrive at ~10-20 Hz while the IMU runs at 200-500,
        and tying the two together would throttle the IMU to the frame rate.
        """
        if self.imu_queue is None:
            return []
        from .imu import packet_to_sample

        out = []
        t_host = dai.Clock.now().total_seconds()
        for data in self.imu_queue.tryGetAll():
            for pkt in data.packets:
                s = packet_to_sample(pkt, t_host)
                if s is not None:
                    out.append(s)
                    self.imu_stats.update(s)
        return out

    def drain_encoded_rgb(self) -> list[Frame]:
        """Return every H.264/H.265 packet received since the last drain.

        This is separate from RGB-D display pairing: an encoded P frame must be
        preserved even if its matching depth frame has not arrived yet.
        """
        out, self._encoded_pending = self._encoded_pending, []
        return out

    def close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            finally:
                self.device = None
        self.queues.clear()

    # --------------------------------------------------------- calibration
    def _load_intrinsics(self) -> None:
        """Intrinsics at the sizes actually being streamed."""
        if self.device is None:
            return
        try:
            calib = self.device.readCalibration()
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"  note: no usable calibration on the device ({e}); "
                      "point clouds will be unavailable")
            return

        cw, ch = self.rc.color_out_size
        dw, dh = self.rc.depth_out_size
        mw, mh = self.rc.mono_size

        want: list[tuple[str, object, int, int]] = []
        if STREAM_COLOR in self.cfg.streams:
            want.append((STREAM_COLOR, SOCKET_COLOR, cw, ch))
        if STREAM_DEPTH in self.cfg.streams:
            # Depth carries the intrinsics of whatever camera it was aligned to.
            sock = SOCKET_COLOR if self.cfg.depth_align == "color" else (
                SOCKET_LEFT if self.cfg.depth_align == "left" else SOCKET_RIGHT)
            want.append((STREAM_DEPTH, sock, dw, dh))
        if STREAM_LEFT in self.cfg.streams:
            want.append((STREAM_LEFT, SOCKET_LEFT, mw, mh))
        if STREAM_RIGHT in self.cfg.streams:
            want.append((STREAM_RIGHT, SOCKET_RIGHT, mw, mh))

        for name, sock, w, h in want:
            try:
                self.intrinsics[name] = intrinsics_from_calibration(calib, sock, w, h)
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"  note: no intrinsics for {name} ({e})")

        if getattr(self.cfg, "imu_enable", False):
            self.imu_extrinsics = read_imu_extrinsics(calib, SOCKET_COLOR)
            if self.verbose and not self.imu_extrinsics.get("available"):
                print("  note: the device has NO IMU-to-camera extrinsics; "
                      "T_imu_cam is unknown and must be calibrated separately "
                      "(recorded in the dataset metadata)")

    # ---------------------------------------------------------------- decode
    def _nominal_bytes(self, name: str, msg) -> int:
        """Wire size of a message we are NOT going to decode.

        Used for frames discarded as stale: they still crossed the link, so they
        belong in the throughput total. Encoded formats have variable packet
        sizes, so those are read from the buffer.
        """
        fmt = self.formats.get(name, "nv12")
        w, h = msg.getWidth(), msg.getHeight()
        if fmt in _ENCODED_FORMATS:
            return int(len(msg.getData()))
        if fmt == "depth16":
            return w * h * 2
        if fmt == "gray8":
            return w * h
        return (w * h * 3) // 2

    def _decode(self, name: str, msg) -> tuple[np.ndarray | None, int, bytes | None]:
        """Decode one message; report wire bytes and (optionally) the raw payload.

        Raw formats are computed from the frame dimensions rather than by
        touching the buffer, so accounting for a 460 kB depth frame costs
        nothing. Only MJPEG needs its real length, since that is the whole point
        of measuring instead of estimating.

        Ownership matters here. `getData()` hands back a *non-owning reference to
        depthai's internal buffer* and `getFrame()` defaults to copy=False, so
        both can be recycled once we move on to the next message. Anything that
        outlives this call — a recorder's writer thread, a queued frame — must
        own its memory, hence the explicit copies.
        """
        fmt = self.formats.get(name, "nv12")
        w, h = msg.getWidth(), msg.getHeight()

        if fmt in ("mjpeg", "mjpeg_gray"):
            data = msg.getData()
            buf = np.frombuffer(data, dtype=np.uint8)
            # One decode path for both, differing only in the destination channel
            # count. The mono pair MUST come back HxW single-channel: that is the
            # contract every existing consumer of "left"/"right" was written
            # against, and IMREAD_COLOR would hand back a 3-channel image that
            # looks fine, raises nothing, and is wrong everywhere downstream.
            flag = cv2.IMREAD_GRAYSCALE if fmt == "mjpeg_gray" else cv2.IMREAD_COLOR
            img = cv2.imdecode(buf, flag)   # imdecode allocates anew
            if img is None and self.verbose:
                print(f"  warning: MJPEG decode failed on {name} "
                      f"(seq {msg.getSequenceNum()})")
            raw = bytes(data) if self.keep_raw else None   # bytes() copies
            return img, int(buf.nbytes), raw
        if fmt in ("h264", "h265"):
            # Inter-frame video cannot be decoded one packet at a time. Keep the
            # exact stream bytes and decode them after capture, once all
            # reference frames are available.
            data = bytes(msg.getData())
            return None, len(data), data
        if fmt == "depth16":
            # copy=True: a view into a buffer depthai may reuse is not safe to
            # keep, and depth is what a recorder most needs to be intact.
            return msg.getFrame(copy=True), w * h * 2, None
        if fmt == "gray8":
            return msg.getCvFrame(), w * h, None
        return msg.getCvFrame(), (w * h * 3) // 2, None   # NV12 -> BGR on host

    def _frame_from_msg(self, name: str, msg) -> Frame | None:
        img, nbytes, raw = self._decode(name, msg)
        lat = self.metrics.update(name, msg, nbytes)
        if img is None and raw is None:
            return None
        try:
            exposure = msg.getExposureTime().total_seconds() * 1000.0
        except Exception:  # noqa: BLE001
            exposure = float("nan")

        frame_type = ""
        encoding = self.formats.get(name, "")
        codec_header = b""
        if encoding in ("h264", "h265"):
            try:
                frame_type = str(msg.getFrameType()).rsplit(".", 1)[-1]
            except Exception:  # noqa: BLE001
                frame_type = ""
            # One scan, shared by both readers below: this is the capture
            # thread, and walking every packet twice showed up as throughput.
            spans = _annexb_nal_spans(raw or b"")
            if frame_type not in ("I", "P", "B"):
                # depthai 2.32 has no getFrameType(), so this is the normal path,
                # not a fallback for exotic builds. Read the NAL type instead —
                # see _annexb_frame_type for why an empty value is not harmless.
                frame_type = _annexb_frame_type(raw or b"", encoding, spans)
            params = self._codec_parameter_sets.setdefault(encoding, {})
            params.update(_annexb_parameter_sets(raw or b"", encoding, spans))
            order = (7, 8) if encoding == "h264" else (32, 33, 34)
            codec_header = b"".join(params[t] for t in order if t in params)

        return Frame(
            name=name,
            image=img,
            seq=msg.getSequenceNum(),
            t_device=msg.getTimestamp().total_seconds(),
            t_recv=dai.Clock.now().total_seconds(),
            latency_ms=lat,
            exposure_ms=exposure,
            raw=raw,
            encoding=encoding,
            frame_type=frame_type,
            codec_header=codec_header,
        )

    # ------------------------------------------------------------------ read
    def poll(self, timeout_ms: float = 1000.0) -> frozenset[str]:
        """Block until at least one stream delivers, then drain all queues.

        Returns the set of streams that produced a new frame. Raw/MJPEG display
        streams keep only the newest queued image. H.264/H.265 is different:
        every packet is preserved because later P frames depend on earlier ones.
        """
        if self.device is None:
            raise RuntimeError("source is not open")

        names = list(self.queues)
        try:
            self.device.getQueueEvents(
                names, maxNumEvents=len(names) * 2,
                timeout=datetime.timedelta(milliseconds=timeout_ms))
        except Exception:  # noqa: BLE001 - fall through to a plain drain
            pass

        fresh: set[str] = set()
        for name, q in self.queues.items():
            msgs = q.tryGetAll()
            if not msgs:
                continue
            if self.formats.get(name) in ("h264", "h265"):
                for msg in msgs:
                    frame = self._frame_from_msg(name, msg)
                    if frame is not None:
                        self._latest[name] = frame
                        self._encoded_pending.append(frame)
                        fresh.add(name)
                continue

            # Account for every message so drop detection stays honest, but only
            # decode the newest one. Discarded messages still crossed the link,
            # so they count toward throughput.
            for m in msgs[:-1]:
                self.metrics.update(name, m, self._nominal_bytes(name, m))
            msg = msgs[-1]

            frame = self._frame_from_msg(name, msg)
            if frame is None:
                continue
            self._latest[name] = frame
            fresh.add(name)
        return frozenset(fresh)

    def read(self, timeout_ms: float = 1000.0) -> Bundle | None:
        """One bundle, or None if nothing usable arrived in `timeout_ms`.

        pair_mode="latest"    — return as soon as anything is fresh. Lowest
                                latency; colour and depth may be one frame apart.
        pair_mode="timestamp" — only return once colour and depth capture times
                                agree within the tolerance. Costs up to one frame
                                interval, gives properly registered RGB-D.
        """
        fresh = self.poll(timeout_ms)
        if not fresh:
            return None

        cfg = self.cfg
        if cfg.pair_mode == "timestamp" and \
                {STREAM_COLOR, STREAM_DEPTH} <= set(cfg.streams):
            c, d = self._latest.get(STREAM_COLOR), self._latest.get(STREAM_DEPTH)
            if c is None or d is None:
                return None
            if abs(c.t_device - d.t_device) > self.rc.pair_tolerance_s():
                return None

        return Bundle(frames=dict(self._latest), fresh=fresh,
                      intrinsics=dict(self.intrinsics))

    def stream(self, timeout_ms: float = 1000.0):
        """Generator over bundles. Skips timeouts rather than ending the stream."""
        while True:
            b = self.read(timeout_ms)
            if b is not None:
                yield b

    # -------------------------------------------------------------- summary
    def summary(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "resolved": {
                "color_out": list(self.rc.color_out_size),
                "depth_out": list(self.rc.depth_out_size),
                "mono": list(self.rc.mono_size),
                "mono_fps": self.rc.mono_fps,
                "bandwidth_mbps": {k: round(v, 1)
                                   for k, v in self.rc.bandwidth_mbps().items()},
            },
            "device": self.device_report,
            "intrinsics": {k: v.to_dict() for k, v in self.intrinsics.items()},
            "metrics": self.metrics.summary(),
        }
