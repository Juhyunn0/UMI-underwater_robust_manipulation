#!/usr/bin/env python3
"""
c3_collect.py — record an underwater RGB-D / visual-SLAM dataset from the C3 + ArduSub.

Camera over DepthAI (Madrona bypassed), telemetry over MAVLink, everything on one
timeline, written in TUM RGB-D layout.

    # look first: is the camera free, does it have an IMU, what can it do?
    python c3_camera/discover_c3.py --probe

    # long exploratory dive: mp4 + telemetry only
    python c3_camera/c3_collect.py --mode review

    # SLAM dataset: lossless rgb/depth/left/right + IMU + telemetry
    python c3_camera/c3_collect.py --mode research

    # plan a configuration without touching the camera
    python c3_camera/c3_collect.py --mode research --dry-run

Press SPACE (or pass --record-now) to start and stop recording, so you can dive
first and capture only the segment that matters. Ctrl-C shuts down cleanly:
writers flush, the mp4 is finalised, metadata is written.

This program NEVER commands the vehicle. Pilot with QGroundControl; the pilot's
inputs are logged from MAVLink. See control.py for the seam where a joystick or a
vision policy would later plug in — ArduSub stays the flight controller in every
case.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from c3_camera import config as C
from c3_camera import device as D
from c3_camera import viz
from c3_camera.config import (STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT,
                              STREAM_RIGHT, StreamConfig)
from c3_camera.control import NullControlSource
from c3_camera.dataset import DEPTH_SCALE, DatasetWriter, estimate_rate
from c3_camera.imu import IMU_FIELDS
from c3_camera.mavlink_log import MavlinkLogger
from c3_camera.source import C3Source

# Research mode defaults. Chosen from measurement, not taste: four lossless
# streams at 480x270 (+640x400 mono) cost ~62 Mbit/s at 8 fps, i.e. ~68% of the
# measured 91.5 Mbit/s PoE ceiling, which is the band where nothing drops.
RESEARCH_DEFAULTS = dict(
    color_res="1080p", isp_scale=(1, 4), color_encode="none", color_wire="nv12",
    mono_res="400p", depth_align="color", mono_source="raw",
    # depth_match_color is the non-negotiable one for RGB-D SLAM: depth is warped
    # into the colour camera's frame AND emitted on the colour grid, so rgb[u,v]
    # and depth[u,v] are the same ray and the colour intrinsics describe both.
    depth_match_color=True,
    # "timestamp" rather than "latest": a dataset wants colour and depth that were
    # actually captured together, and can afford to wait up to a frame for it.
    # "latest" never waits, which is right for a live view and wrong here.
    pair_mode="timestamp",
    fps=8.0, streams=(STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT, STREAM_RIGHT),
)
REVIEW_DEFAULTS = dict(
    color_res="1080p", isp_scale=(1, 2), color_encode="mjpeg",
    mono_res="400p", depth_align="color", depth_match_color=False,
    fps=15.0, streams=(STREAM_COLOR, STREAM_DEPTH),
)


def build_config(a: argparse.Namespace) -> StreamConfig:
    """Mode defaults, overridden by anything given explicitly on the command line.

    Every camera option defaults to None so that "not given" is distinguishable
    from "given the same value the mode default happens to use" — otherwise
    switching --mode would silently ignore an explicit flag.
    """
    base = dict(RESEARCH_DEFAULTS if a.mode == "research" else REVIEW_DEFAULTS)
    for key in ("color_res", "isp_scale", "mono_res", "depth_size", "fps",
                "color_encode", "video_bitrate_kbps",
                "video_keyframe_frequency", "color_wire", "mono_source"):
        val = getattr(a, key, None)
        if val is not None:
            base[key] = val
    if a.streams is not None:
        base["streams"] = tuple(a.streams)

    return StreamConfig(
        ip=a.ip, mxid=a.mxid,
        imu_enable=not a.no_camera_imu,
        imu_rate=a.imu_rate,
        imu_rotation_vector=a.imu_rotation_vector,
        imu_magnetometer=a.imu_magnetometer,
        imu_calibrated=a.imu_calibrated,
        imu_batch_threshold=a.imu_batch,
        imu_max_batch_reports=max(20, a.imu_batch * 4),
        queue_size=a.queue_size,
        **base,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("mode")
    g.add_argument("--mode", default="research", choices=("research", "review"),
                   help="research = lossless TUM RGB-D dataset; review = mp4 + "
                        "telemetry only, for long exploratory dives (default: %(default)s)")
    g.add_argument("--out", type=Path, default=None,
                   help="output directory (default: "
                        "c3_camera/datasets/dataset_<YYYYMMDD_HHMMSS>)")
    g.add_argument("--notes", default="",
                   help="free-text session notes, copied into metadata.txt")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve the config, print the wire/disk budget, and exit "
                        "without connecting to anything")

    C.add_connection_args(p)

    g = p.add_argument_group("camera (defaults depend on --mode)")
    g.add_argument("--color-res", default=None, choices=sorted(C.COLOR_RESOLUTIONS))
    g.add_argument("--isp-scale", type=C._parse_fraction, default=None, metavar="N/D")
    g.add_argument("--mono-res", default=None, choices=sorted(C.MONO_RESOLUTIONS))
    g.add_argument("--depth-size", type=C._parse_size, default=None, metavar="WxH")
    g.add_argument("--fps", type=float, default=None)
    g.add_argument("--streams", type=C._parse_streams, default=None,
                   help="default research: color,depth,left,right")
    g.add_argument("--color-encode", default=None,
                   choices=("none", "mjpeg", "h264", "h265"),
                   help="RGB wire format: raw NV12/BGR, per-frame MJPEG, or "
                        "hardware H.264/H.265 (research default: none)")
    g.add_argument("--video-bitrate-kbps", type=int, default=None,
                   help="target H.264/H.265 bitrate in kbit/s (default: 4000)")
    g.add_argument("--video-keyframe-frequency", type=int, default=None,
                   help="H.264/H.265 keyframe interval in frames; 0 means once "
                        "per second (default: 0)")
    g.add_argument("--color-wire", default=None, choices=("nv12", "bgr"),
                   help="nv12 (1.5 B/px, 4:2:0 chroma; luma full res) or bgr "
                        "(3.0 B/px, full chroma at twice the bandwidth)")
    g.add_argument("--mono-source", default=None, choices=("raw", "rectified"),
                   help="raw sensor images (default; most general) or the "
                        "rectified pair StereoDepth produces")
    g.add_argument("--queue-size", type=int, default=2,
                   help="host output queue depth (default: %(default)s)")

    g = p.add_argument_group("camera IMU (BNO086, on the camera board)")
    g.add_argument("--imu-rate", type=int, default=200,
                   help="Hz to request for accel+gyro; ~486 Hz is achievable "
                        "(default: %(default)s)")
    g.add_argument("--imu-rotation-vector", action="store_true",
                   help="also log the fused orientation quaternion (requested at "
                        "the same rate, so it costs nothing)")
    g.add_argument("--imu-magnetometer", action="store_true",
                   help="also log the magnetometer. WARNING: it tops out near "
                        "100 Hz and the slowest sensor sets the rate for ALL of "
                        "them, so this caps accel/gyro at ~100 Hz too")
    g.add_argument("--imu-calibrated", action="store_true",
                   help="use ACCELEROMETER/GYROSCOPE_CALIBRATED instead of the RAW "
                        "reports")
    g.add_argument("--imu-batch", type=int, default=10,
                   help="IMU reports batched into one XLink message. 1 gives the "
                        "lowest latency but the most per-message overhead, which "
                        "loses samples when the image streams are busy; a dataset "
                        "wants completeness instead (default: %(default)s)")
    g.add_argument("--no-camera-imu", action="store_true",
                   help="do not record the camera IMU at all")

    g = p.add_argument_group("MAVLink telemetry")
    g.add_argument("--mavlink", default="udpin:0.0.0.0:14551",
                   help="pymavlink connection string (default: %(default)s). "
                        "QGroundControl holds 14550, so a second BlueOS endpoint "
                        "is needed — see --setup-endpoint")
    g.add_argument("--mavlink-transport", default="udp", choices=("udp", "rest", "none"),
                   help="udp = full stream on its own port; rest = poll "
                        "MAVLink2Rest (no vehicle config needed, samples rather "
                        "than captures); none = no telemetry (default: %(default)s)")
    g.add_argument("--mavlink-rest-url", default="http://192.168.2.2:6040")
    g.add_argument("--mavlink-rate", type=int, default=50,
                   help="Hz to request for IMU/ATTITUDE when --mavlink-set-rates "
                        "is given (default: %(default)s)")
    g.add_argument("--mavlink-set-rates", action="store_true",
                   help="raise ArduSub message rates with SET_MESSAGE_INTERVAL. "
                        "This TRANSMITS to the autopilot; off by default so the "
                        "recorder is purely passive while you pilot")
    g.add_argument("--mavlink-wait", type=float, default=6.0,
                   help="seconds to wait for first telemetry before warning")
    g.add_argument("--setup-endpoint", action="store_true",
                   help="add the BlueOS MAVLink endpoint for --mavlink's port if "
                        "missing (changes vehicle config; may blip QGC's link)")

    g = p.add_argument_group("recording")
    g.add_argument("--record-now", action="store_true",
                   help="start recording immediately instead of waiting for SPACE")
    g.add_argument("--writer-threads", type=int, default=3,
                   help="PNG encoder threads (default: %(default)s)")
    g.add_argument("--writer-queue", type=int, default=12,
                   help="frames buffered for the writers (default: %(default)s)")
    g.add_argument("--png-compression", type=int, default=1,
                   help="0-9, lossless at every level; 1 is fast and nearly as "
                        "small (default: %(default)s)")
    g.add_argument("--no-mp4", action="store_true",
                   help="skip the rgb.mp4 preview in research mode")
    g.add_argument("--extract-encoded-rgb", action="store_true",
                   help="after recording, decode rgb.h264/rgb.h265 into timestamped "
                        "RGB PNGs referenced by the TUM lists")
    g.add_argument("--min-free-gb", type=float, default=10.0,
                   help="stop recording if free disk falls below this (default: %(default)s)")

    g = p.add_argument_group("display")
    g.add_argument("--no-display", action="store_true",
                   help="headless: stats on the console only (start recording with "
                        "--record-now, stop with Ctrl-C)")
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--min-mm", type=float, default=300.0)
    g.add_argument("--max-mm", type=float, default=6000.0)
    g.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds of streaming")
    g.add_argument("--print-every", type=float, default=3.0)
    return p.parse_args(argv)


# =============================================================================
# helpers
# =============================================================================
def calibration_payload(src: C3Source) -> dict:
    """Everything a consumer needs to use these images metrically."""
    out: dict = {
        "source": "on-device factory calibration (OAK-D-W-POE EEPROM)",
        "note": ("IN-AIR calibration. Underwater the port changes the effective "
                 "focal length and adds radial distortion; re-calibrate in water "
                 "for metric work."),
        "streamed_intrinsics": {k: v.to_dict() for k, v in src.intrinsics.items()},
    }
    if src.device is not None:
        try:
            out["device"] = D.describe_calibration(src.device)
        except Exception as e:  # noqa: BLE001
            out["device_error"] = str(e)
    out["imu_extrinsics"] = src.imu_extrinsics or {
        "available": False,
        "note": "IMU not enabled this session",
    }
    return out


def blueos_info(host: str = "192.168.2.2") -> dict:
    """ArduSub version and frame type, for the metadata. Never fatal."""
    import urllib.request
    out: dict = {}
    for key, path in (("ardusub_version", "/v1.0/firmware_info"),
                      ("vehicle_type", "/v1.0/vehicle_type")):
        try:
            with urllib.request.urlopen(f"http://{host}:8000{path}", timeout=4) as r:
                out[key] = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            out[key] = None
    if isinstance(out.get("ardusub_version"), dict):
        v = out["ardusub_version"]
        out["ardusub_version"] = f"{v.get('version')} ({v.get('type')})"
    return out


def compose_view(bundle, mode: str, min_mm: float, max_mm: float, scale: float):
    panes = []
    if bundle.color is not None and bundle.color.image is not None:
        panes.append(viz.label(bundle.color.image.copy(), "rgb"))
    if bundle.depth is not None and bundle.depth.image is not None:
        panes.append(viz.label(viz.colorize_depth(bundle.depth.image, min_mm, max_mm),
                               "depth mm"))
    if bundle.left is not None and bundle.left.image is not None:
        panes.append(viz.label(bundle.left.image.copy(), "left"))
    if bundle.right is not None and bundle.right.image is not None:
        panes.append(viz.label(bundle.right.image.copy(), "right"))
    if not panes:
        return None
    canvas = viz.hstack_pad(panes)
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    return canvas


def _extract_if_requested(a, cfg, out_dir: Path) -> None:
    if not a.extract_encoded_rgb or cfg.color_encode not in ("h264", "h265"):
        return
    from c3_camera.c3_extract_rgb_video import extract_rgb_video

    print(f"  extracting {cfg.color_encode.upper()} RGB into timestamped PNGs ...")
    try:
        summary = extract_rgb_video(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  RGB extraction failed: {e}")
        print(f"  raw video is intact; retry with: python "
              f"c3_camera/c3_extract_rgb_video.py {out_dir}")
        return
    print(f"  extracted {summary['frames_decoded']} RGB frames")


# =============================================================================
# main
# =============================================================================
def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = build_config(a)
    rc = cfg.resolve()

    print("=" * 78)
    print(f"C3 dataset collection — {a.mode} mode")
    print("=" * 78)
    for w in rc.warnings:
        print(f"  warning: {w}")
    print(f"  camera : {rc.describe()}")

    est = estimate_rate(rc.color_out_size, rc.depth_out_size, rc.mono_size,
                        cfg.fps, cfg.streams, cfg.color_wire, cfg.color_encode,
                        cfg.video_bitrate_kbps)
    print(f"  wire   : {est['wire_mbit_s']:.0f} Mbit/s "
          f"({est['wire_mbit_s']/C.POE_BUDGET_MBPS*100:.0f}% of the measured "
          f"~{C.POE_BUDGET_MBPS:.0f} Mbit/s PoE ceiling)")
    if a.mode == "research":
        print(f"  disk   : ~{est['disk_mb_min']:.0f} MB/min "
              f"(~{est['disk_gb_hour']:.1f} GB/hour)")
        if cfg.color_encode in ("h264", "h265") and a.extract_encoded_rgb:
            print("           encoded-capture footprint; extracted RGB PNGs add "
                  "more disk use after recording")
    if est["wire_mbit_s"] > C.POE_BUDGET_MBPS * 0.85:
        print(f"  NOTE: this is close to or above the link ceiling. Expect the "
              f"achieved rate to settle near {est['link_limited_fps']:.1f} fps "
              f"with the remainder dropped. Lower --fps or --isp-scale to fix it.")
        if cfg.color_encode in ("h264", "h265") and \
                {"left", "right"} & set(cfg.streams):
            print("  NOTE: H.264/H.265 cannot fix this while left/right are sent: "
                  "those mono streams dominate the link. For 20 fps RGB-D use "
                  "--streams color,depth (StereoDepth still works internally).")
    if cfg.imu_enable and cfg.imu_magnetometer:
        print("  NOTE: --imu-magnetometer caps the WHOLE IMU near 100 Hz "
              "(slowest enabled sensor sets the rate for all of them).")

    if a.dry_run:
        print("\ndry run — nothing connected.")
        return 0

    out_dir = a.out or (Path("c3_camera/datasets") /
                        f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # ------------------------------------------------------------- MAVLink
    mav = None
    if a.mavlink_transport != "none":
        if a.setup_endpoint and a.mavlink_transport == "udp":
            try:
                from c3_camera import blueos_endpoint as BE
                port = int(a.mavlink.rsplit(":", 1)[-1])
                if BE.find_recorder_endpoint(port) is None:
                    print(f"  adding BlueOS endpoint udpout -> 192.168.2.1:{port} ...")
                    BE.add_endpoint(port)
                    print("    added (QGC's link may have blipped); waiting 3 s")
                    time.sleep(3.0)
                else:
                    print(f"  BlueOS already pushes to port {port}")
            except Exception as e:  # noqa: BLE001
                print(f"  could not configure the BlueOS endpoint: {e}")

        mav = MavlinkLogger(connection=a.mavlink, transport=a.mavlink_transport,
                            rest_url=a.mavlink_rest_url,
                            set_rates=a.mavlink_set_rates,
                            default_rate_hz=a.mavlink_rate)
        mav.start(wait_s=a.mavlink_wait)

    # -------------------------------------------------------------- camera
    writer: DatasetWriter | None = None
    control = NullControlSource()   # inert; see control.py
    stop = {"flag": False}

    def _sigint(_sig, _frm):
        stop["flag"] = True
        print("\n  Ctrl-C — finishing the current frames and writing metadata ...")

    signal.signal(signal.SIGINT, _sigint)

    t_connect = time.monotonic()
    rc_code = 0
    # The finally block has to be able to finalise metadata after the `with` has
    # exited (Ctrl-C, an exception, a --duration timeout), so the pieces it needs
    # live here rather than only in the loop's scope.
    session: dict = {"src": None, "t_start": t_connect}
    try:
        with C3Source(cfg, strict=False,
                      keep_raw=cfg.color_encode in ("mjpeg", "h264", "h265")) as src:
            session["src"] = src
            t_start = time.monotonic()
            session["t_start"] = t_start
            print(f"  connected in {t_start - t_connect:.1f} s")
            print(f"  streams: {', '.join(f'{v}:{k}' for k, v in src.formats.items())}")
            if cfg.color_encode in ("h264", "h265") and not a.no_display:
                print("  note: inter-frame RGB stays encoded during capture; "
                      "the live window shows depth/stereo only")
            if cfg.imu_enable:
                print(f"  camera IMU: {src.device_report.get('mxid') and ''}"
                      f"{src.device.getConnectedIMU()} requested at {cfg.imu_rate} Hz")
            print()
            print("  SPACE start/stop recording   q quit" if not a.no_display
                  else "  recording controlled by --record-now / Ctrl-C")
            print()

            window = "C3 dataset collection"
            if not a.no_display:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            last_print = 0.0
            frames_seen = 0

            def start_rec() -> DatasetWriter:
                w = DatasetWriter(out_dir, mode=a.mode,
                                  threads=a.writer_threads,
                                  queue_size=a.writer_queue,
                                  png_compression=a.png_compression,
                                  mp4_fps=cfg.fps,
                                  write_mp4=((not a.no_mp4) or a.mode == "review")
                                            and cfg.color_encode not in
                                            ("h264", "h265"),
                                  color_encode=cfg.color_encode)
                w.write_calibration(calibration_payload(src))
                intr = src.intrinsics.get(STREAM_COLOR)
                if intr is not None and a.mode == "research":
                    try:
                        # getBaselineDistance() is centimetres; ORB-SLAM3 wants mm here.
                        baseline_mm = src.device.readCalibration().getBaselineDistance() * 10.0
                    except Exception:  # noqa: BLE001
                        baseline_mm = None
                    w.write_orbslam3_yaml(intr, baseline_mm, cfg.fps)
                print(f"  RECORDING -> {out_dir}")
                if a.mode == "research":
                    if cfg.color_encode in ("h264", "h265"):
                        print(f"    RGB hardware {cfg.color_encode.upper()} "
                              f"@ {cfg.video_bitrate_kbps} kbit/s; depth remains "
                              f"lossless 16-bit PNG")
                        print("    recording begins on the next RGB keyframe "
                              "(at most about one second)")
                    else:
                        print(f"    {'lossy JPEG RGB + ' if cfg.color_encode == 'mjpeg' else ''}"
                              f"lossless PNG streams, depth 16-bit mm, "
                              f"depth_scale={DEPTH_SCALE:.0f}")
                return w

            if a.record_now:
                writer = start_rec()

            while not stop["flag"]:
                bundle = src.read(timeout_ms=2000)

                encoded_rgb = src.drain_encoded_rgb()
                if writer is not None and encoded_rgb:
                    writer.encoded_rgb_frames(encoded_rgb)
                imu_samples = src.drain_imu()
                if writer is not None and imu_samples:
                    writer.imu_camera_rows(imu_samples, IMU_FIELDS)
                if mav is not None:
                    recs = mav.drain()
                    if writer is not None and recs:
                        writer.mavlink_rows(recs)

                if bundle is None:
                    if a.duration and time.monotonic() - t_start > a.duration:
                        break
                    continue
                frames_seen += 1
                src.metrics.mark_loop()

                if writer is not None:
                    writer.submit(bundle)
                    if writer.stats.written % 20 == 0:
                        import shutil as _sh
                        if _sh.disk_usage(out_dir).free / 1e9 < a.min_free_gb:
                            print(f"  free disk below {a.min_free_gb:g} GB — "
                                  f"stopping recording")
                            writer.close()
                            _finalize_metadata(writer, src, cfg, rc, mav, a,
                                               time.monotonic() - t_start)
                            _extract_if_requested(a, cfg, out_dir)
                            print(json.dumps(writer.summary(), indent=2))
                            writer = None

                now = time.monotonic()
                if a.print_every and now - last_print >= a.print_every:
                    last_print = now
                    head = f"[{now - t_start:6.1f}s]"
                    lines = src.metrics.hud_lines()
                    if cfg.imu_enable:
                        lines.append(src.imu_stats.hud_line())
                    if mav is not None:
                        lines += mav.hud_lines()
                    if writer is not None:
                        lines.append(writer.hud_line())
                    else:
                        lines.append("REC    (not recording — press SPACE)")
                    for line in lines:
                        print(f"  {head} {line}")
                        head = " " * len(head)

                if not a.no_display:
                    canvas = compose_view(bundle, a.mode, a.min_mm, a.max_mm, a.scale)
                    if canvas is not None:
                        hud = [f"{a.mode} | {rc.describe()}"]
                        hud += src.metrics.hud_lines()
                        if cfg.imu_enable:
                            hud.append(src.imu_stats.hud_line())
                        if mav is not None:
                            hud += mav.hud_lines()
                        hud.append(writer.hud_line() if writer is not None
                                   else "REC    stopped — SPACE to start")
                        hud.append("SPACE record   q quit")
                        viz.draw_hud(canvas, hud)
                        cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        if writer is None:
                            writer = start_rec()
                        else:
                            writer.close()
                            _finalize_metadata(writer, src, cfg, rc, mav, a,
                                               time.monotonic() - t_start)
                            _extract_if_requested(a, cfg, out_dir)
                            print("  stopped recording:")
                            print(json.dumps(writer.summary(), indent=2))
                            writer = None

                if a.duration and time.monotonic() - t_start > a.duration:
                    break

    except D.PipelineConfigError as e:
        print(f"\nPIPELINE REJECTED: {e}")
        return 1
    except D.DeviceBusyError as e:
        print(f"\nCAMERA UNAVAILABLE\n{e}")
        print("\n  If the message above mentions another pipeline, the BlueOS")
        print("  Madrona extension is holding the camera. An OAK device accepts")
        print("  only one pipeline at a time. Free it with:")
        print("      python -m c3_camera.madrona stop --yes")
        print("  then re-run. Restore it afterwards with 'madrona start'.")
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}")
        rc_code = 1
    finally:
        if writer is not None:
            # Order matters: flush the writers first so frames.csv and the TUM
            # lists are complete, THEN write metadata describing what was written.
            writer.close()
            src_obj = session.get("src")
            if src_obj is not None:
                try:
                    _finalize_metadata(writer, src_obj, cfg, rc, mav, a,
                                       time.monotonic() - session["t_start"])
                    _extract_if_requested(a, cfg, out_dir)
                except Exception as e:  # noqa: BLE001
                    print(f"  could not write metadata: {type(e).__name__}: {e}")
            print("\n  recording:")
            print("  " + json.dumps(writer.summary(), indent=2).replace("\n", "\n  "))
            if writer.stats.dropped:
                print(f"  WARNING: {writer.stats.dropped} frames "
                      f"({writer.stats.drop_rate*100:.1f}%) never reached disk. "
                      f"Timestamps are still correct — a drop is a gap, not a "
                      f"shift — but lower --fps or --isp-scale next time.")
            print(f"  dataset: {out_dir}")
        if mav is not None:
            mav.close()
        if not a.no_display:
            cv2.destroyAllWindows()
        control.close()

    return rc_code


def _finalize_metadata(writer: DatasetWriter, src: C3Source, cfg, rc,
                       mav, a, elapsed: float) -> None:
    """Write metadata.txt/json. Called on stop and on shutdown."""
    cs = src.metrics.streams.get(STREAM_COLOR)
    meta = {
        "camera": src.device_report,
        "resolved": {
            "color_out": list(rc.color_out_size),
            "depth_out": list(rc.depth_out_size),
            "mono": list(rc.mono_size),
        },
        "fps_requested": cfg.fps,
        "fps_achieved": round(cs.fps_lifetime, 2) if cs else None,
        "color_encode": cfg.color_encode,
        "color_wire": cfg.color_wire,
        "video_bitrate_kbps": (cfg.video_bitrate_kbps
                               if cfg.color_encode in ("h264", "h265") else None),
        "video_keyframe_frequency": (
            cfg.video_keyframe_frequency or max(1, int(round(cfg.fps)))
            if cfg.color_encode in ("h264", "h265") else None),
        "mono_source": cfg.mono_source,
        "depth_align": cfg.depth_align,
        "depth_scale": DEPTH_SCALE,
        "clock": {
            "offset": f"{writer._t_offset:.6f}",
            "captured_at": f"{time.time():.6f}",
            "note": "t_unix = t_monotonic + offset; dai.Clock.now() == "
                    "time.monotonic() (CLOCK_MONOTONIC) on this host",
        },
        "imu": {
            "present": bool(cfg.imu_enable),
            "model": (src.device_report or {}).get("imu")
                     or (src.device.getConnectedIMU() if src.device else None),
            "rate_hz": round(src.imu_stats.hz, 1),
            "requested_rate_hz": cfg.imu_rate,
            "static_accel_magnitude": (round(src.imu_stats.accel_magnitude, 3)
                                       if src.imu_stats.accel_magnitude ==
                                       src.imu_stats.accel_magnitude else None),
            "extrinsics_note": (src.imu_extrinsics or {}).get(
                "note", "not queried"),
            "samples": src.imu_stats.count,
        },
        "mavlink": mav.summary() if mav is not None else {"transport": "none"},
        "duration_s": round(elapsed, 1),
        "frames_written": writer.stats.written,
        "frames_dropped": writer.stats.dropped,
        "writer": writer.summary(),
        "camera_metrics": src.metrics.summary(),
    }
    meta.update(blueos_info())
    writer.write_metadata(meta, notes=a.notes)


if __name__ == "__main__":
    raise SystemExit(main())
