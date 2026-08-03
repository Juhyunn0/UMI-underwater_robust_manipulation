#!/usr/bin/env python3
"""
pipeline.py — build the on-device DepthAI pipeline from a StreamConfig.

Graph (nodes present only if their stream was requested):

    ColorCamera CAM_A ──isp scale──> video (NV12) ──┬─> XLinkOut "color"
                                                    └─> VideoEncoder(MJPEG/H26x)
                                                        -> XLinkOut "color"
    MonoCamera CAM_B ─┐
                      ├─> StereoDepth ──depth (uint16 mm)──> XLinkOut "depth"
    MonoCamera CAM_C ─┘        └─ rectified/raw mono ───────> XLinkOut "left"/"right"

Latency decisions, and why:

* Colour leaves as **NV12** from the `video` output, not BGR from `preview`.
  NV12 is 1.5 bytes/px against 3.0 for BGR — half the wire time for identical
  pixels, because the BGR conversion happens on the host either way. This is the
  single biggest difference from the usual `preview` example code.
* `setIspScale` downscales in the ISP, so the sensor still runs its native mode
  (keeping its full field of view) while the link carries fewer bytes. Cropping
  to a smaller sensor mode would narrow the FOV instead.
* `pipeline.setXLinkChunkSize(0)` disables chunking, so a frame is handed to the
  link as one transfer instead of being split.
* Each `XLinkOut.input` gets queue size 1 and non-blocking, so if the link is
  momentarily behind, the device discards the stale frame instead of queueing
  it. Latency stays bounded; you lose frames rather than freshness. That is the
  right trade for teleoperation and for closing a visual loop.
* Depth units are pinned to millimetres explicitly rather than relying on the
  default, so the contract with downstream code is stated in code.
"""

from __future__ import annotations

import depthai as dai

from .config import (
    MEDIAN_FILTERS,
    COLOR_RESOLUTIONS,
    DEPTH_PRESETS,
    MONO_RESOLUTIONS,
    ResolvedConfig,
    SOCKET_COLOR,
    SOCKET_LEFT,
    SOCKET_RIGHT,
    STREAM_COLOR,
    STREAM_DEPTH,
    STREAM_LEFT,
    STREAM_RIGHT,
)

_ALIGN_SOCKET = {
    "color": SOCKET_COLOR,
    "left": SOCKET_LEFT,
    "right": SOCKET_RIGHT,
}

STREAM_IMU = "imu"


def _add_imu(pipeline: dai.Pipeline, cfg) -> None:
    """Attach the on-board BNO086.

    Measured on this camera (OAK-D-W-POE, BNO086 firmware 3.9.9):

      ACCELEROMETER_RAW + GYROSCOPE_RAW @500 requested  ->  486 Hz achieved
      the same @200                                     ->  194 Hz
      ACCELEROMETER + GYROSCOPE_CALIBRATED @200         ->  194 Hz
      accel+gyro @200 + ROTATION_VECTOR @100            ->   97 Hz on ALL of them
      accel+gyro @200 + MAGNETOMETER_RAW @100           ->   97 Hz on ALL of them
      accel+gyro @200 + ROTATION_VECTOR @200            ->  194 Hz  <- no penalty

    The rule those measurements imply: **the rate achieved by every enabled
    sensor tracks the slowest one requested.** A slower extra sensor drags the
    accelerometer and gyroscope down with it, which for visual-inertial work is a
    bad trade. So request the same rate for every sensor we enable, and treat the
    magnetometer (which tops out around 100 Hz on this part) as something that
    caps the whole IMU.

    Batching matters just as much, and separately. While four lossless image
    streams were also running:

      setBatchReportThreshold(1)   @200 Hz  ->  130 Hz received, 35% LOST
      setBatchReportThreshold(10)  @200 Hz  ->  199.5 Hz received, 0 lost

    So the loss was per-message XLink overhead, not bandwidth — one message per
    report cannot keep up when images are competing. Threshold 1 minimises latency
    and is right for a live view; a dataset wants completeness, because gaps in the
    IMU stream break VIO preintegration. c3_collect.py therefore defaults to 10.
    """
    imu = pipeline.create(dai.node.IMU)
    rate = int(cfg.imu_rate)
    accel = (dai.IMUSensor.ACCELEROMETER if cfg.imu_calibrated
             else dai.IMUSensor.ACCELEROMETER_RAW)
    gyro = (dai.IMUSensor.GYROSCOPE_CALIBRATED if cfg.imu_calibrated
            else dai.IMUSensor.GYROSCOPE_RAW)
    imu.enableIMUSensor([accel, gyro], rate)
    if cfg.imu_rotation_vector:
        # Same rate on purpose — see the table above.
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, rate)
    if cfg.imu_magnetometer:
        imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_RAW, min(rate, 100))
    imu.setBatchReportThreshold(max(1, int(cfg.imu_batch_threshold)))
    imu.setMaxBatchReports(max(1, int(cfg.imu_max_batch_reports)))

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(STREAM_IMU)
    # A deeper queue than the image streams: IMU packets are tiny and arrive far
    # more often, and for a dataset we would rather buffer them than drop them.
    xout.input.setQueueSize(20)
    xout.input.setBlocking(False)
    imu.out.link(xout.input)


def _wire_xout(pipeline: dai.Pipeline, name: str, source: dai.Node.Output,
               rc: ResolvedConfig) -> dai.node.XLinkOut:
    """One XLinkOut per stream, configured for freshness over completeness."""
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(name)
    interframe = (name == STREAM_COLOR and
                  rc.cfg.color_encode in ("h264", "h265"))
    if interframe:
        # Dropping one queued P frame damages every dependent frame until the
        # next keyframe. Compressed packets are small enough to buffer 1.5 s.
        xout.input.setQueueSize(
            max(rc.cfg.device_queue_size, int(round(rc.cfg.fps * 1.5))))
        xout.input.setBlocking(True)
    else:
        xout.input.setQueueSize(max(1, rc.cfg.device_queue_size))
        xout.input.setBlocking(False)
    source.link(xout.input)
    return xout


def build_pipeline(rc: ResolvedConfig) -> tuple[dai.Pipeline, dict[str, str]]:
    """Return (pipeline, {stream: wire_format}).

    The format map tells the host how to handle each stream: "nv12", "mjpeg",
    "h264", "h265", "depth16" or "gray8".
    """
    cfg = rc.cfg
    pipeline = dai.Pipeline()
    pipeline.setXLinkChunkSize(cfg.xlink_chunk)

    formats: dict[str, str] = {}
    want_color = STREAM_COLOR in cfg.streams
    want_depth = STREAM_DEPTH in cfg.streams
    want_mono = bool({STREAM_LEFT, STREAM_RIGHT} & set(cfg.streams))

    # ---------------------------------------------------------------- colour
    if want_color:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(SOCKET_COLOR)
        cam.setResolution(COLOR_RESOLUTIONS[cfg.color_res][0])
        cam.setFps(cfg.fps)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setInterleaved(False)
        if cfg.isp_scale is not None:
            cam.setIspScale(*cfg.isp_scale)
        # ColorCamera validates its 'preview' output even when nothing is linked
        # to it, and its default is 300x300 — so any ISP output smaller than that
        # (e.g. --isp-scale 1/4 -> 480x270) fails at pipeline start with
        # "'preview' width or height (300, 300) bigger than sensor resolution".
        # Pin it small by default; the --color-wire bgr branch below overrides
        # this with the real size, since that is the one case where we read it.
        cam.setPreviewSize(32, 32)
        # The IMX378 on this unit is fixed focus (hasAutofocus=False), so there
        # is no lens to drive; exposure/white balance stay on auto.
        if cfg.small_pools:
            # (raw, isp, preview, video, still) — preview/still are unused here.
            cam.setNumFramesPool(2, 2, 1, 2, 1)

        if cfg.color_encode in ("mjpeg", "h264", "h265"):
            enc = pipeline.create(dai.node.VideoEncoder)
            profiles = {
                "mjpeg": dai.VideoEncoderProperties.Profile.MJPEG,
                # Baseline avoids B-frame reordering, so encoded packets and
                # capture timestamps remain in the same order for extraction.
                "h264": dai.VideoEncoderProperties.Profile.H264_BASELINE,
                "h265": dai.VideoEncoderProperties.Profile.H265_MAIN,
            }
            enc.setDefaultProfilePreset(cfg.fps, profiles[cfg.color_encode])
            if cfg.color_encode == "mjpeg":
                enc.setQuality(cfg.mjpeg_quality)
            else:
                enc.setBitrateKbps(cfg.video_bitrate_kbps)
                # Keep output/capture order identical. It also minimizes latency
                # and makes one sidecar timestamp correspond to one decoded frame.
                enc.setNumBFrames(0)
                keyframes = (cfg.video_keyframe_frequency
                             or max(1, int(round(cfg.fps))))
                enc.setKeyframeFrequency(keyframes)
            if cfg.small_pools:
                enc.setNumFramesPool(2)
            cam.video.link(enc.input)
            if cfg.color_encode in ("h264", "h265"):
                enc.input.setQueueSize(max(4, cfg.device_queue_size))
                enc.input.setBlocking(True)
            else:
                enc.input.setQueueSize(max(1, cfg.device_queue_size))
                enc.input.setBlocking(False)
            _wire_xout(pipeline, STREAM_COLOR, enc.bitstream, rc)
            formats[STREAM_COLOR] = cfg.color_encode
        elif cfg.color_wire == "bgr":
            # Full-chroma BGR888 straight off the ISP: 3.0 B/px against NV12's
            # 1.5, i.e. twice the wire time, in exchange for chroma that is not
            # 4:2:0 subsampled. Only worth it if a consumer needs colour
            # fidelity; feature-based SLAM works on luma, which NV12 keeps at
            # full resolution.
            cam.setPreviewSize(*rc.color_out_size)
            cam.setPreviewKeepAspectRatio(True)
            _wire_xout(pipeline, STREAM_COLOR, cam.preview, rc)
            formats[STREAM_COLOR] = "bgr"
        else:
            _wire_xout(pipeline, STREAM_COLOR, cam.video, rc)
            formats[STREAM_COLOR] = "nv12"

    # ---------------------------------------------------------------- stereo
    if want_depth or want_mono:
        mono_enum = MONO_RESOLUTIONS[cfg.mono_res][0]
        left = pipeline.create(dai.node.MonoCamera)
        left.setBoardSocket(SOCKET_LEFT)
        left.setResolution(mono_enum)
        left.setFps(rc.mono_fps)

        right = pipeline.create(dai.node.MonoCamera)
        right.setBoardSocket(SOCKET_RIGHT)
        right.setResolution(mono_enum)
        right.setFps(rc.mono_fps)

        if cfg.small_pools:
            left.setNumFramesPool(2)
            right.setNumFramesPool(2)

        if want_depth:
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(DEPTH_PRESETS[cfg.depth_preset])
            stereo.setLeftRightCheck(cfg.lr_check)
            stereo.setSubpixel(cfg.subpixel)
            stereo.setExtendedDisparity(cfg.extended)
            stereo.initialConfig.setMedianFilter(MEDIAN_FILTERS[cfg.median])
            # Millimetres, stated explicitly — downstream code depends on it.
            stereo.initialConfig.setDepthUnit(
                dai.RawStereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER
            )
            if cfg.confidence is not None:
                stereo.initialConfig.setConfidenceThreshold(cfg.confidence)

            if cfg.depth_align in _ALIGN_SOCKET:
                stereo.setDepthAlign(_ALIGN_SOCKET[cfg.depth_align])
            elif cfg.depth_align == "center":
                stereo.setDepthAlign(
                    dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
                )
            # "none" leaves the preset default (rectified right).

            stereo.setOutputSize(*rc.depth_out_size)
            if cfg.small_pools:
                stereo.setNumFramesPool(2)

            left.out.link(stereo.left)
            right.out.link(stereo.right)
            _wire_xout(pipeline, STREAM_DEPTH, stereo.depth, rc)
            formats[STREAM_DEPTH] = "depth16"

            # raw = straight off the sensors, rectified = StereoDepth's undistorted
            # pair. Raw plus the factory calibration lets a consumer do either,
            # which is why it is the default for a dataset; rectified is what a
            # stereo SLAM front end usually wants handed to it directly.
            l_src = left.out if cfg.mono_source == "raw" else stereo.rectifiedLeft
            r_src = right.out if cfg.mono_source == "raw" else stereo.rectifiedRight
            if STREAM_LEFT in cfg.streams:
                _wire_xout(pipeline, STREAM_LEFT, l_src, rc)
                formats[STREAM_LEFT] = "gray8"
            if STREAM_RIGHT in cfg.streams:
                _wire_xout(pipeline, STREAM_RIGHT, r_src, rc)
                formats[STREAM_RIGHT] = "gray8"
        else:
            # Mono without depth: ship the raw sensor images.
            if STREAM_LEFT in cfg.streams:
                _wire_xout(pipeline, STREAM_LEFT, left.out, rc)
                formats[STREAM_LEFT] = "gray8"
            if STREAM_RIGHT in cfg.streams:
                _wire_xout(pipeline, STREAM_RIGHT, right.out, rc)
                formats[STREAM_RIGHT] = "gray8"

    # ------------------------------------------------------------------- IMU
    if getattr(cfg, "imu_enable", False):
        _add_imu(pipeline, cfg)
        formats[STREAM_IMU] = "imu"

    return pipeline, formats
