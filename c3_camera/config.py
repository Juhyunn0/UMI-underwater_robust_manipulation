#!/usr/bin/env python3
"""
config.py — stream configuration + the C3's *probed* hardware capability tables.

The resolution tables below are not copied from documentation; they were read off
this exact camera with `Device.getConnectedCameraFeatures()` (see
`discover_c3.py --probe`), so an entry here means the sensor really accepts it:

    CAM_A  IMX378  colour, 4056x3040 native, FIXED FOCUS (hasAutofocus=False)
    CAM_B  OV9282  mono LEFT,  1280x800 native
    CAM_C  OV9282  mono RIGHT, 1280x800 native
    stereo baseline 7.5 cm, factory calibration present on EEPROM

`validate_against_device()` re-checks a config against whatever device is
actually connected, so this stays honest if the hardware is ever swapped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, field

import depthai as dai

# =============================================================================
# Probed capability tables
# =============================================================================

# name -> (enum, width, height, max_fps_reported_by_sensor)
COLOR_RESOLUTIONS: dict[str, tuple[dai.ColorCameraProperties.SensorResolution, int, int, float]] = {
    "1352x1012": (dai.ColorCameraProperties.SensorResolution.THE_1352X1012, 1352, 1012, 52.0),
    "1080p":     (dai.ColorCameraProperties.SensorResolution.THE_1080_P,     1920, 1080, 60.0),
    "2024x1520": (dai.ColorCameraProperties.SensorResolution.THE_2024X1520,  2024, 1520, 85.0),
    "4k":        (dai.ColorCameraProperties.SensorResolution.THE_4_K,        3840, 2160, 42.0),
    "12mp":      (dai.ColorCameraProperties.SensorResolution.THE_12_MP,      4056, 3040, 30.0),
}

MONO_RESOLUTIONS: dict[str, tuple[dai.MonoCameraProperties.SensorResolution, int, int, float]] = {
    "400p": (dai.MonoCameraProperties.SensorResolution.THE_400_P,  640,  400, 255.7),
    "720p": (dai.MonoCameraProperties.SensorResolution.THE_720_P, 1280,  720, 143.1),
    "800p": (dai.MonoCameraProperties.SensorResolution.THE_800_P, 1280,  800, 129.6),
}

DEPTH_PRESETS: dict[str, dai.node.StereoDepth.PresetMode] = {
    "default":       dai.node.StereoDepth.PresetMode.DEFAULT,
    "high_accuracy": dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
    "high_density":  dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
    "high_detail":   dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    "robotics":      dai.node.StereoDepth.PresetMode.ROBOTICS,
    "face":          dai.node.StereoDepth.PresetMode.FACE,
}

MEDIAN_FILTERS: dict[str, dai.StereoDepthConfig.MedianFilter] = {
    "off": dai.StereoDepthConfig.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthConfig.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthConfig.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthConfig.MedianFilter.KERNEL_7x7,
}

# Sockets, fixed by the C3's EEPROM calibration (stereoLeft=CAM_B, stereoRight=CAM_C).
SOCKET_COLOR = dai.CameraBoardSocket.CAM_A
SOCKET_LEFT  = dai.CameraBoardSocket.CAM_B
SOCKET_RIGHT = dai.CameraBoardSocket.CAM_C

DEFAULT_IP   = "192.168.2.191"
DEFAULT_MXID = "19443010315B0B2F00"

# Stream names on the XLink boundary.
STREAM_COLOR  = "color"
STREAM_DEPTH  = "depth"
STREAM_LEFT   = "left"
STREAM_RIGHT  = "right"
ALL_STREAMS   = (STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT, STREAM_RIGHT)

# Bytes per pixel on the XLink wire, per stream flavour.
BPP = {
    "nv12":   1.5,   # ColorCamera.video  — YUV420, what --color-encode none ships
    "bgr":    3.0,   # ColorCamera.preview — 2x the bandwidth of NV12 for no gain
    "mono":   1.0,   # MonoCamera.out     — GRAY8
    "depth": 2.0,    # StereoDepth.depth  — uint16
}

# H.264/H.265 are rate-controlled, so their link cost is configured directly
# instead of estimated from a scene-dependent compression ratio.
DEFAULT_VIDEO_BITRATE_KBPS = 4000

# MJPEG compression ratio at quality 90, MEASURED on a normal lit lab scene:
# 960x540 came out at 112-139 kB/frame against 778 kB raw, i.e. 5.6-6.9x. An
# initial 10x guess underestimated the bandwidth by nearly half, which is exactly
# the kind of error that turns a "should fit" config into a dropping one.
#
# This is scene-dependent, and that dependence is real rather than noise: murky
# or dim water has less high-frequency detail and will compress better, clear
# water over textured seabed worse. Treat the estimate as a planning figure and
# trust the measured Mbit/s in the HUD.
MJPEG_RATIO = 6.0

# Practical XLink-over-PoE throughput ceiling, MEASURED on this camera over this
# network with c3_bench.py — not a datasheet figure. The PHY is gigabit, but the
# XLink/TCP path through the Myriad X tops out far below that, and the ceiling is
# what actually decides which resolution/fps combinations survive:
#
#   colour only, 960x540 NV12   13.8 fps x 778 kB  = 86 Mbit/s
#   colour + depth 640x360      9.25 fps x 1.24 MB = 92 Mbit/s
#
# Two very different pipelines converging on the same number is what makes this a
# link ceiling rather than a compute limit. Re-measure with c3_bench.py if the
# cabling, the switch, or the fibre run changes.
POE_BUDGET_MBPS = 90.0


# =============================================================================
# Stream configuration
# =============================================================================
@dataclass
class StreamConfig:
    """Everything that determines the on-device pipeline.

    A pipeline is fixed at boot: changing any of these fields requires closing
    the device and reconnecting (that is exactly what c3_bench.py does).
    """

    # --- connection -------------------------------------------------------
    ip: str = DEFAULT_IP
    mxid: str | None = None

    # --- which streams to ship over the link ------------------------------
    streams: tuple[str, ...] = (STREAM_COLOR, STREAM_DEPTH)

    # --- colour path ------------------------------------------------------
    # These defaults are the outcome of the c3_bench.py sweep, not a guess. At
    # 960x540 over this PoE link, raw NV12 delivered 9.1 of 20 requested fps at
    # 250 ms; the same config with device-side MJPEG delivered 19.1 fps at
    # 123 ms. Encoding on the device is the single largest win available, so it
    # is on by default. Use --color-encode none if you need untouched pixels
    # (photometric work, calibration) and can live with ~9 fps.
    color_res: str = "1080p"
    isp_scale: tuple[int, int] | None = (1, 2)   # 1080p * 1/2 -> 960x540
    color_encode: str = "mjpeg"                  # none | mjpeg | h264 | h265
    mjpeg_quality: int = 90
    video_bitrate_kbps: int = DEFAULT_VIDEO_BITRATE_KBPS
    # 0 means one keyframe per second (derived from fps).
    video_keyframe_frequency: int = 0

    # --- stereo path ------------------------------------------------------
    mono_res: str = "400p"
    depth_preset: str = "robotics"
    depth_align: str = "color"                   # color | right | left | center | none
    depth_size: tuple[int, int] | None = None    # None -> derived, see resolve()
    # Force the depth output onto exactly the colour grid. RGB-D SLAM needs
    # rgb[u,v] and depth[u,v] to be the same ray, which means identical
    # resolution as well as alignment — so dataset collection sets this. The
    # live-preview path leaves it off, where a smaller depth map costs less
    # bandwidth and nothing depends on 1:1 correspondence.
    depth_match_color: bool = False
    lr_check: bool = True
    subpixel: bool = False
    extended: bool = False
    median: str = "5x5"
    confidence: int | None = None                # None -> leave preset default

    # --- rates ------------------------------------------------------------
    fps: float = 15.0
    mono_fps: float | None = None                # None -> same as fps

    # --- dataset-collection options ---------------------------------------
    # Wire format for colour when not encoding. "nv12" is 1.5 B/px but 4:2:0
    # chroma-subsampled; "bgr" is 3.0 B/px with full chroma. Luma is full
    # resolution either way, so for feature-based SLAM nv12 loses nothing that
    # matters — but a dataset advertised as "lossless" should say which it is.
    color_wire: str = "nv12"                     # nv12 | bgr
    # Raw sensor images, or the rectified pair StereoDepth produces. Raw + the
    # factory calibration lets a consumer do either; rectified is what stereo
    # SLAM usually wants fed to it directly.
    mono_source: str = "raw"                     # raw | rectified
    # On-board BNO086. accel+gyro alone reach ~486 Hz; adding the fused
    # ROTATION_VECTOR or the magnetometer collapses every stream to ~40 Hz
    # (measured), so those are opt-in.
    imu_enable: bool = False
    imu_rate: int = 200
    imu_rotation_vector: bool = False
    imu_magnetometer: bool = False
    imu_calibrated: bool = False                 # ACCELEROMETER vs ACCELEROMETER_RAW
    imu_batch_threshold: int = 1                 # 1 = send immediately, lowest latency
    imu_max_batch_reports: int = 20

    # --- latency knobs ----------------------------------------------------
    xlink_chunk: int = 0          # 0 disables XLink chunking (lowest latency)
    queue_size: int = 1           # host-side output queue depth
    queue_blocking: bool = False  # False = drop-oldest, never stall the device
    device_queue_size: int = 1    # XLinkOut.input queue depth (device side)
    small_pools: bool = False     # shrink node frame pools (see pipeline.py)

    # --- host-side pairing ------------------------------------------------
    pair_mode: str = "latest"     # latest | timestamp
    pair_tolerance_ms: float | None = None  # None -> 0.75 * frame interval

    def resolve(self) -> "ResolvedConfig":
        """Expand into concrete pixel sizes and validate the combination."""
        return ResolvedConfig.from_config(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["streams"] = list(self.streams)
        return d


@dataclass
class ResolvedConfig:
    """A StreamConfig with every derived size computed and checked."""

    cfg: StreamConfig
    color_sensor_size: tuple[int, int]
    color_out_size: tuple[int, int]
    mono_size: tuple[int, int]
    depth_out_size: tuple[int, int]
    mono_fps: float
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: StreamConfig) -> "ResolvedConfig":
        warns: list[str] = []

        if cfg.color_res not in COLOR_RESOLUTIONS:
            raise ValueError(
                f"unknown --color-res {cfg.color_res!r}; "
                f"choose from {sorted(COLOR_RESOLUTIONS)}"
            )
        if cfg.mono_res not in MONO_RESOLUTIONS:
            raise ValueError(
                f"unknown --mono-res {cfg.mono_res!r}; "
                f"choose from {sorted(MONO_RESOLUTIONS)}"
            )
        if cfg.depth_preset not in DEPTH_PRESETS:
            raise ValueError(f"unknown --depth-preset {cfg.depth_preset!r}")
        if cfg.median not in MEDIAN_FILTERS:
            raise ValueError(f"unknown --median {cfg.median!r}")
        if cfg.color_encode not in ("none", "mjpeg", "h264", "h265"):
            raise ValueError(f"unknown --color-encode {cfg.color_encode!r}")
        if cfg.video_bitrate_kbps <= 0:
            raise ValueError("--video-bitrate-kbps must be positive")
        if cfg.video_keyframe_frequency < 0:
            raise ValueError("--video-keyframe-frequency must be >= 0")
        if cfg.color_wire not in ("nv12", "bgr"):
            raise ValueError(f"unknown --color-wire {cfg.color_wire!r}")
        if cfg.mono_source not in ("raw", "rectified"):
            raise ValueError(f"unknown --mono-source {cfg.mono_source!r}")
        if not cfg.streams:
            raise ValueError("at least one stream must be requested")
        for s in cfg.streams:
            if s not in ALL_STREAMS:
                raise ValueError(f"unknown stream {s!r}; choose from {list(ALL_STREAMS)}")

        _, cw, ch, cmax = COLOR_RESOLUTIONS[cfg.color_res]
        _, mw, mh, mmax = MONO_RESOLUTIONS[cfg.mono_res]
        mono_fps = cfg.mono_fps if cfg.mono_fps is not None else cfg.fps

        if cfg.fps > cmax:
            warns.append(
                f"fps {cfg.fps:g} exceeds the IMX378 maximum {cmax:g} at "
                f"{cfg.color_res}; the sensor will clamp it"
            )
        if mono_fps > mmax:
            warns.append(
                f"mono fps {mono_fps:g} exceeds the OV9282 maximum {mmax:g} at "
                f"{cfg.mono_res}"
            )

        # Colour output size after the ISP scaler.
        if cfg.isp_scale is not None:
            n, d = cfg.isp_scale
            if not (1 <= n <= 16 and 1 <= d <= 63):
                raise ValueError(
                    f"--isp-scale {n}/{d} violates the hardware scaler limits "
                    "(numerator <= 16, denominator <= 63)"
                )
            ow, oh = (cw * n) // d, (ch * n) // d
            # The ISP scaler emits even dimensions; NV12 needs them too.
            ow -= ow % 2
            oh -= oh % 2
        else:
            ow, oh = cw, ch

        # The video encoder wants its input width 32-aligned. Height alignment is
        # NOT required: 960x540 and 480x270 both encode fine on this device
        # (verified by c3_bench.py), so do not warn about odd heights.
        if cfg.color_encode != "none" and ow % 32:
            warns.append(
                f"colour output width {ow} is not a multiple of 32; the video "
                f"encoder may reject it — pick an --isp-scale that lands on one "
                f"(1/2 -> 960 and 1/4 -> 480 are both verified working)"
            )

        # Depth output size.
        if cfg.depth_size is not None:
            dw, dh = cfg.depth_size
        elif cfg.depth_match_color and cfg.depth_align == "color":
            # Exactly the colour grid: same rays, so the colour intrinsics apply
            # to the depth image unchanged. This is what RGB-D SLAM requires.
            dw, dh = ow, oh
        elif cfg.depth_align == "color":
            # Match the colour aspect ratio at mono-ish resolution: aligning to
            # colour warps depth into CAM_A's perspective, and emitting it at
            # full colour resolution would cost 2 bytes/px for no extra
            # information (depth detail is bounded by the mono pair).
            dw = mw
            dh = int(round(mw * oh / ow))
            dh -= dh % 2
        else:
            dw, dh = mw, mh

        if cfg.depth_align == "color" and abs(ow / oh - mw / mh) > 0.02:
            warns.append(
                f"colour {ow}x{oh} ({ow/oh:.3f}) and mono {mw}x{mh} ({mw/mh:.3f}) "
                "have different aspect ratios; depth aligned to colour will be "
                "cropped/letterboxed at the edges"
            )
        if cfg.subpixel and cfg.depth_align == "color":
            warns.append(
                "subpixel + depth-align=color both consume stereo post-processing "
                "hardware; if the pipeline fails to start, drop one"
            )

        return cls(
            cfg=cfg,
            color_sensor_size=(cw, ch),
            color_out_size=(ow, oh),
            mono_size=(mw, mh),
            depth_out_size=(dw, dh),
            mono_fps=mono_fps,
            warnings=warns,
        )

    # ---------------------------------------------------------------- budget
    def bandwidth_mbps(self) -> dict[str, float]:
        """Estimated XLink payload per stream, in Mbit/s.

        Raw formats are exact. MJPEG uses MJPEG_RATIO, a measured average that
        varies with the scene — `Metrics.mbps_total` reports what really crossed
        the link, and that is the number to trust when the two disagree.
        """
        cfg = self.cfg
        out: dict[str, float] = {}
        ow, oh = self.color_out_size
        dw, dh = self.depth_out_size
        mw, mh = self.mono_size

        if STREAM_COLOR in cfg.streams:
            raw = ow * oh * BPP["bgr" if cfg.color_wire == "bgr" else "nv12"]
            if cfg.color_encode == "mjpeg":
                raw = ow * oh * BPP["nv12"] / MJPEG_RATIO
            if cfg.color_encode in ("h264", "h265"):
                out[STREAM_COLOR] = cfg.video_bitrate_kbps / 1000.0
            else:
                out[STREAM_COLOR] = raw * cfg.fps * 8 / 1e6
        if STREAM_DEPTH in cfg.streams:
            out[STREAM_DEPTH] = dw * dh * BPP["depth"] * self.mono_fps * 8 / 1e6
        for s in (STREAM_LEFT, STREAM_RIGHT):
            if s in cfg.streams:
                out[s] = mw * mh * BPP["mono"] * self.mono_fps * 8 / 1e6
        out["total"] = sum(v for k, v in out.items() if k != "total")
        return out

    def budget_note(self) -> str:
        bw = self.bandwidth_mbps()
        frac = bw["total"] / POE_BUDGET_MBPS
        verdict = "ok" if frac < 0.6 else ("tight" if frac < 1.0 else "OVER BUDGET")
        parts = ", ".join(f"{k}={v:.0f}" for k, v in bw.items() if k != "total")
        return (
            f"{bw['total']:.0f} Mbit/s ({parts}) "
            f"= {frac*100:.0f}% of the ~{POE_BUDGET_MBPS:.0f} Mbit/s PoE budget [{verdict}]"
        )

    def pair_tolerance_s(self) -> float:
        if self.cfg.pair_tolerance_ms is not None:
            return self.cfg.pair_tolerance_ms / 1000.0
        return 0.75 / max(self.cfg.fps, 1.0)

    def describe(self) -> str:
        cfg = self.cfg
        ow, oh = self.color_out_size
        dw, dh = self.depth_out_size
        bits = [f"streams={'+'.join(cfg.streams)}"]
        if STREAM_COLOR in cfg.streams:
            enc = "" if cfg.color_encode == "none" else f" {cfg.color_encode}"
            isp = "" if cfg.isp_scale is None else f" isp {cfg.isp_scale[0]}/{cfg.isp_scale[1]}"
            bits.append(f"color {cfg.color_res}->{ow}x{oh}{isp}{enc}")
        if STREAM_DEPTH in cfg.streams:
            bits.append(
                f"depth {cfg.mono_res}->{dw}x{dh} align={cfg.depth_align} "
                f"preset={cfg.depth_preset}"
                f"{' lr' if cfg.lr_check else ''}"
                f"{' subpx' if cfg.subpixel else ''}"
                f"{' ext' if cfg.extended else ''}"
            )
        bits.append(f"{cfg.fps:g} fps")
        return " | ".join(bits)

    # ------------------------------------------------------------- validate
    def validate_against_device(self, device: dai.Device) -> list[str]:
        """Cross-check the requested config against the connected hardware."""
        problems: list[str] = []
        features = {f.socket: f for f in device.getConnectedCameraFeatures()}

        needed = {SOCKET_COLOR} if STREAM_COLOR in self.cfg.streams else set()
        if {STREAM_DEPTH, STREAM_LEFT, STREAM_RIGHT} & set(self.cfg.streams):
            needed |= {SOCKET_LEFT, SOCKET_RIGHT}
        for sock in needed:
            if sock not in features:
                problems.append(
                    f"{sock.name} is not present on this device "
                    f"(has {[s.name for s in features]})"
                )

        def _check(sock, want_w, want_h, want_fps, label):
            feat = features.get(sock)
            if feat is None:
                return
            exact = [c for c in feat.configs if (c.width, c.height) == (want_w, want_h)]
            if not exact:
                sizes = sorted({(c.width, c.height) for c in feat.configs})
                problems.append(
                    f"{label}: {feat.sensorName} on {sock.name} does not list "
                    f"{want_w}x{want_h}; it supports {sizes}"
                )
                return
            best = max(c.maxFps for c in exact)
            if want_fps > best + 0.01:
                problems.append(
                    f"{label}: {feat.sensorName} caps {want_w}x{want_h} at "
                    f"{best:.1f} fps, {want_fps:g} requested"
                )

        if STREAM_COLOR in self.cfg.streams:
            cw, ch = self.color_sensor_size
            _check(SOCKET_COLOR, cw, ch, self.cfg.fps, "color")
        if {STREAM_DEPTH, STREAM_LEFT, STREAM_RIGHT} & set(self.cfg.streams):
            mw, mh = self.mono_size
            _check(SOCKET_LEFT, mw, mh, self.mono_fps, "mono left")
            _check(SOCKET_RIGHT, mw, mh, self.mono_fps, "mono right")
        return problems


# =============================================================================
# CLI plumbing
# =============================================================================
def _parse_fraction(text: str) -> tuple[int, int] | None:
    if text.lower() in ("none", "off", "1", "1/1"):
        return None
    if "/" not in text:
        raise argparse.ArgumentTypeError(f"--isp-scale wants N/D, got {text!r}")
    n, d = text.split("/", 1)
    return int(n), int(d)


def _parse_size(text: str) -> tuple[int, int]:
    sep = "x" if "x" in text else ("," if "," in text else None)
    if sep is None:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}")
    w, h = text.split(sep, 1)
    return int(w), int(h)


def _parse_streams(text: str) -> tuple[str, ...]:
    items = tuple(s.strip() for s in text.split(",") if s.strip())
    for s in items:
        if s not in ALL_STREAMS:
            raise argparse.ArgumentTypeError(
                f"unknown stream {s!r}; choose from {','.join(ALL_STREAMS)}"
            )
    return items


def add_connection_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("connection")
    g.add_argument("--ip", default=DEFAULT_IP,
                   help="camera IP address (default: %(default)s)")
    g.add_argument("--mxid", default=None,
                   help="connect by mx_id instead of IP (needs UDP discovery)")


def add_stream_args(p: argparse.ArgumentParser) -> None:
    """Every knob that shapes the pipeline, so tests can sweep them."""
    d = StreamConfig()

    g = p.add_argument_group("streams")
    g.add_argument("--streams", type=_parse_streams, default=d.streams,
                   help="comma list of %s (default: color,depth)" % ",".join(ALL_STREAMS))

    g = p.add_argument_group("colour path")
    g.add_argument("--color-res", default=d.color_res, choices=sorted(COLOR_RESOLUTIONS),
                   help="IMX378 sensor mode (default: %(default)s)")
    g.add_argument("--isp-scale", type=_parse_fraction, default=d.isp_scale, metavar="N/D",
                   help="downscale colour on the ISP, e.g. 1/2 or 2/3, or 'none' "
                        "(default: 1/2). This is the cheapest bandwidth knob.")
    g.add_argument("--color-encode", default=d.color_encode,
                   choices=("none", "mjpeg"),
                   help="'none' ships raw NV12; 'mjpeg' uses the on-device "
                        "hardware encoder (default: %(default)s). H.264/H.265 "
                        "dataset capture is exposed by c3_collect.py.")
    g.add_argument("--mjpeg-quality", type=int, default=d.mjpeg_quality,
                   help="MJPEG quality 0-100 (default: %(default)s)")
    g.add_argument("--video-bitrate-kbps", type=int,
                   default=d.video_bitrate_kbps,
                   help="target H.264/H.265 bitrate in kbit/s "
                        "(default: %(default)s)")
    g.add_argument("--video-keyframe-frequency", type=int,
                   default=d.video_keyframe_frequency,
                   help="H.264/H.265 keyframe interval in frames; 0 means once "
                        "per second (default: %(default)s)")

    g = p.add_argument_group("stereo path")
    g.add_argument("--mono-res", default=d.mono_res, choices=sorted(MONO_RESOLUTIONS),
                   help="OV9282 sensor mode (default: %(default)s)")
    g.add_argument("--depth-preset", default=d.depth_preset, choices=sorted(DEPTH_PRESETS),
                   help="StereoDepth profile preset (default: %(default)s)")
    g.add_argument("--depth-align", default=d.depth_align,
                   choices=("color", "right", "left", "center", "none"),
                   help="warp depth into this camera's perspective "
                        "(default: %(default)s — needed for pixel-aligned RGB-D)")
    g.add_argument("--depth-size", type=_parse_size, default=d.depth_size, metavar="WxH",
                   help="override the depth output size (default: derived)")
    g.add_argument("--median", default=d.median, choices=sorted(MEDIAN_FILTERS),
                   help="depth median filter (default: %(default)s)")
    g.add_argument("--confidence", type=int, default=d.confidence, metavar="0-255",
                   help="stereo confidence threshold; lower = stricter "
                        "(default: preset value)")
    g.add_argument("--subpixel", action="store_true", default=d.subpixel,
                   help="subpixel disparity: better far-range resolution, more compute")
    g.add_argument("--extended", action="store_true", default=d.extended,
                   help="extended disparity: closer minimum range, more compute")
    g.add_argument("--no-lr-check", dest="lr_check", action="store_false", default=d.lr_check,
                   help="disable left-right consistency check (faster, more outliers)")

    g = p.add_argument_group("rates")
    g.add_argument("--fps", type=float, default=d.fps,
                   help="colour fps (default: %(default)s)")
    g.add_argument("--mono-fps", type=float, default=d.mono_fps,
                   help="stereo fps (default: same as --fps)")

    g = p.add_argument_group("latency")
    g.add_argument("--xlink-chunk", type=int, default=d.xlink_chunk, metavar="BYTES",
                   help="XLink chunk size; 0 disables chunking (default: %(default)s)")
    g.add_argument("--queue-size", type=int, default=d.queue_size,
                   help="host output queue depth (default: %(default)s)")
    g.add_argument("--blocking-queues", dest="queue_blocking", action="store_true",
                   default=d.queue_blocking,
                   help="block instead of dropping old frames (raises latency; "
                        "only for lossless capture)")
    g.add_argument("--device-queue-size", type=int, default=d.device_queue_size,
                   help="XLinkOut input queue depth on the device (default: %(default)s)")
    g.add_argument("--small-pools", action="store_true", default=d.small_pools,
                   help="shrink node frame pools; saves a little latency, can stall "
                        "the ISP if too small")

    g = p.add_argument_group("host-side pairing")
    g.add_argument("--pair-mode", default=d.pair_mode, choices=("latest", "timestamp"),
                   help="'latest' never waits (lowest latency); 'timestamp' only "
                        "emits colour+depth within the tolerance (default: %(default)s)")
    g.add_argument("--pair-tolerance-ms", type=float, default=d.pair_tolerance_ms,
                   help="pairing tolerance for --pair-mode timestamp "
                        "(default: 0.75 frame intervals)")


def config_from_args(a: argparse.Namespace) -> StreamConfig:
    return StreamConfig(
        ip=a.ip,
        mxid=a.mxid,
        streams=tuple(a.streams),
        color_res=a.color_res,
        isp_scale=a.isp_scale,
        color_encode=a.color_encode,
        mjpeg_quality=a.mjpeg_quality,
        video_bitrate_kbps=a.video_bitrate_kbps,
        video_keyframe_frequency=a.video_keyframe_frequency,
        mono_res=a.mono_res,
        depth_preset=a.depth_preset,
        depth_align=a.depth_align,
        depth_size=a.depth_size,
        lr_check=a.lr_check,
        subpixel=a.subpixel,
        extended=a.extended,
        median=a.median,
        confidence=a.confidence,
        fps=a.fps,
        mono_fps=a.mono_fps,
        xlink_chunk=a.xlink_chunk,
        queue_size=a.queue_size,
        queue_blocking=a.queue_blocking,
        device_queue_size=a.device_queue_size,
        small_pools=a.small_pools,
        pair_mode=a.pair_mode,
        pair_tolerance_ms=a.pair_tolerance_ms,
    )
