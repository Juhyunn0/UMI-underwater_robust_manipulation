#!/usr/bin/env python3
"""
c3_stream.py — live RGB + depth from the C3 over a direct DepthAI connection.

    python c3_camera/c3_stream.py                        # sensible defaults
    python c3_camera/c3_stream.py --color-res 1080p --isp-scale 1/2 --fps 20
    python c3_camera/c3_stream.py --color-encode mjpeg --fps 30
    python c3_camera/c3_stream.py --no-display --duration 20 --csv run.csv

The HUD reports, per stream, the received fps and the sensor-to-host latency
(`dai.Clock.now() - frame.getTimestamp()`), so different resolution/fps settings
can be compared directly. `c3_bench.py` automates that sweep.

Keys
----
    q / ESC  quit                     v  start / stop recording RGB-D to disk
    s        snapshot (PNG + NPZ)     c  save a point cloud (PLY)
    o        cycle view               p  print stats
    r        reset statistics         [ ]  depth colour range down / up
    h        toggle the HUD

Recording
---------
    python c3_camera/c3_stream.py --record            # start immediately
    python c3_camera/c3_stream.py                     # then press 'v'

Writes colour + uint16-millimetre depth + a per-frame index to
`c3_camera/recordings/<timestamp>/`; see recorder.py for the layout and
c3_replay.py to verify or play it back. Disk I/O happens on a writer thread so
recording does not add latency to the stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from c3_camera import config as C
from c3_camera import preflight as PF
from c3_camera import viz
from c3_camera.config import STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT, STREAM_RIGHT
from c3_camera.recorder import Recorder, estimate_disk_mb_per_min
from c3_camera.source import Bundle, C3Source

VIEW_MODES = ("side", "overlay", "color", "depth")


# =============================================================================
# Output helpers
# =============================================================================
def save_snapshot(bundle: Bundle, out_dir: Path, min_mm: float, max_mm: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = out_dir / f"c3_{stamp}"

    payload: dict = {}
    if bundle.color is not None:
        cv2.imwrite(str(base.with_suffix(".color.png")), bundle.color.image)
        payload["color_bgr"] = bundle.color.image
        payload["color_t_device"] = bundle.color.t_device
        payload["color_seq"] = bundle.color.seq
    if bundle.depth is not None:
        # 16-bit PNG keeps the millimetres intact; the colourised version is for
        # eyeballs only.
        cv2.imwrite(str(base.with_suffix(".depth_mm.png")), bundle.depth.image)
        cv2.imwrite(str(base.with_suffix(".depth_vis.png")),
                    viz.colorize_depth(bundle.depth.image, min_mm, max_mm))
        payload["depth_mm"] = bundle.depth.image
        payload["depth_t_device"] = bundle.depth.t_device
        payload["depth_seq"] = bundle.depth.seq
    for name, intr in bundle.intrinsics.items():
        payload[f"intrinsics_{name}"] = json.dumps(intr.to_dict())

    npz = base.with_suffix(".npz")
    np.savez_compressed(npz, **payload)
    return npz


def save_pointcloud(bundle: Bundle, out_dir: Path, stride: int = 2) -> Path | None:
    """Minimal binary-free PLY, so it opens anywhere without a dependency."""
    try:
        xyz, rgb = bundle.pointcloud(stride=stride)
    except RuntimeError as e:
        print(f"  point cloud unavailable: {e}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"c3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    return path


class CsvLog:
    """Per-frame log, for offline latency/fps analysis."""

    FIELDS = ("wall_time", "t_mono", "loop_hz",
              "color_seq", "color_lat_ms", "color_fps", "color_drop",
              "depth_seq", "depth_lat_ms", "depth_fps", "depth_drop",
              "skew_ms", "depth_valid_pct", "exposure_ms")

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._w.writeheader()
        self.path = path

    def write(self, src: C3Source, bundle: Bundle, t0: float,
              depth_valid: float | None) -> None:
        cs = src.metrics.streams.get(STREAM_COLOR)
        ds = src.metrics.streams.get(STREAM_DEPTH)
        row = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "t_mono": round(time.monotonic() - t0, 4),
            "loop_hz": round(src.metrics.loop_hz, 2),
            "skew_ms": round(bundle.skew_ms, 2) if bundle.skew_ms == bundle.skew_ms else "",
            "depth_valid_pct": "" if depth_valid is None else round(depth_valid, 2),
            "exposure_ms": (round(cs.last_exposure_ms, 2)
                            if cs and cs.last_exposure_ms == cs.last_exposure_ms else ""),
        }
        for tag, st, fr in (("color", cs, bundle.color), ("depth", ds, bundle.depth)):
            row[f"{tag}_seq"] = fr.seq if fr else ""
            row[f"{tag}_lat_ms"] = round(fr.latency_ms, 2) if fr else ""
            row[f"{tag}_fps"] = round(st.fps, 2) if st else ""
            row[f"{tag}_drop"] = st.dropped if st else ""
        self._w.writerow(row)

    def close(self) -> None:
        self._fh.close()


# =============================================================================
# Rendering
# =============================================================================
def compose(bundle: Bundle, mode: str, min_mm: float, max_mm: float,
            scale: float) -> np.ndarray | None:
    colour = bundle.color.image if bundle.color else None
    depth = bundle.depth.image if bundle.depth else None
    mono = [f.image for f in (bundle.left, bundle.right) if f is not None]

    if mode == "color" and colour is not None:
        canvas = colour.copy()
    elif mode == "depth" and depth is not None:
        canvas = viz.colorize_depth(depth, min_mm, max_mm)
    elif mode == "overlay" and colour is not None and depth is not None:
        canvas = viz.overlay_depth_on_color(colour, depth, 0.45, min_mm, max_mm)
    else:
        panes = []
        if colour is not None:
            panes.append(viz.label(colour.copy(), f"color {colour.shape[1]}x{colour.shape[0]}"))
        if depth is not None:
            dv = viz.colorize_depth(depth, min_mm, max_mm)
            panes.append(viz.label(dv, f"depth {depth.shape[1]}x{depth.shape[0]} mm"))
        panes.extend(viz.label(m.copy(), "mono") for m in mono)
        if not panes:
            return None
        canvas = viz.hstack_pad(panes)

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return canvas


def hud_lines(src: C3Source, bundle: Bundle, mode: str,
              min_mm: float, max_mm: float, dstats: dict | None,
              rec: Recorder | None = None) -> list[str]:
    rc = src.rc
    lines = [f"C3 direct  |  {rc.describe()}"]
    lines += src.metrics.hud_lines()

    ages = "  ".join(
        f"{n} {f.age_ms():.0f}ms" for n, f in bundle.frames.items() if f is not None
    )
    skew = bundle.skew_ms
    tail = f"age {ages}"
    if skew == skew:
        tail += f"  |  color/depth skew {skew:.1f} ms"
    lines.append(tail)

    if dstats:
        lines.append(f"depth valid {dstats['valid_pct']:.1f}%  "
                     f"range {dstats['min_mm']}-{dstats['max_mm']} mm  "
                     f"median {dstats['median_mm']} mm")
    measured = src.metrics.mbps_total
    est = rc.bandwidth_mbps()["total"]
    lines.append(f"link {measured:.0f} Mbit/s measured "
                 f"({measured/C.POE_BUDGET_MBPS*100:.0f}% of the ~{C.POE_BUDGET_MBPS:.0f} "
                 f"Mbit/s ceiling, est. {est:.0f})  |  "
                 f"view {mode}  depth {min_mm:.0f}-{max_mm:.0f} mm")
    if rec is not None:
        lines.append(rec.hud_line())
    lines.append("q quit  v REC  s snap  c cloud  o view  p stats  r reset  "
                 "[ ] range  h hud")
    return lines


# =============================================================================
# Main
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    C.add_connection_args(p)
    C.add_stream_args(p)

    g = p.add_argument_group("display")
    g.add_argument("--view", default="side", choices=VIEW_MODES,
                   help="initial view (default: %(default)s)")
    g.add_argument("--min-mm", type=float, default=300.0,
                   help="depth colour-scale minimum (default: %(default)s)")
    g.add_argument("--max-mm", type=float, default=6000.0,
                   help="depth colour-scale maximum (default: %(default)s)")
    g.add_argument("--scale", type=float, default=1.0,
                   help="scale the window (default: %(default)s)")
    g.add_argument("--no-display", action="store_true",
                   help="headless: no window, just statistics (for benchmarking "
                        "and for hosts where cv2 GUI misbehaves)")
    g.add_argument("--no-hud", action="store_true", help="start with the HUD hidden")

    g = p.add_argument_group("run control")
    g.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds (default: run until 'q')")
    g.add_argument("--warmup", type=float, default=2.0,
                   help="seconds to discard before statistics start, letting "
                        "auto-exposure and timesync settle (default: %(default)s)")
    g.add_argument("--csv", type=Path, default=None, help="per-frame CSV log")
    g.add_argument("--json-out", type=Path, default=None, help="write the run summary here")
    g.add_argument("--save-dir", type=Path,
                   default=Path("c3_camera/captures"),
                   help="where snapshots go (default: %(default)s)")
    g.add_argument("--print-every", type=float, default=2.0,
                   help="seconds between console stat lines (default: %(default)s)")

    g = p.add_argument_group("recording")
    g.add_argument("--record", action="store_true",
                   help="start recording RGB-D to disk immediately (the 'v' key "
                        "toggles it at any time)")
    g.add_argument("--record-dir", type=Path, default=None,
                   help="recording directory (default: "
                        "c3_camera/recordings/<YYYYMMDD_HHMMSS>)")
    g.add_argument("--record-depth-format", default="png", choices=("png", "npy"),
                   help="depth on disk: 16-bit PNG (compressed, lossless) or raw "
                        ".npy (bigger, faster) (default: %(default)s)")
    g.add_argument("--record-queue", type=int, default=16,
                   help="frames buffered for the writer thread; larger rides out "
                        "disk hiccups at the cost of RAM (default: %(default)s)")
    g.add_argument("--record-block-ms", type=float, default=50.0,
                   help="how long a frame may wait for writer-queue space before "
                        "being dropped; bounded so a slow disk degrades into "
                        "dropping instead of throttling the camera "
                        "(default: %(default)s)")
    g.add_argument("--record-max-frames", type=int, default=None,
                   help="stop recording after N frames")
    g.add_argument("--strict", action="store_true",
                   help="abort if the device does not support the requested config")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve the config and print the bandwidth budget without "
                        "connecting — plan a setting without paying the 12 s boot "
                        "or taking the camera from whoever has it")
    PF.add_preflight_args(p)
    a = p.parse_args(argv)

    cfg = C.config_from_args(a)

    if a.dry_run:
        rc_dry = cfg.resolve()
        print("dry run — nothing connected\n")
        print(f"  {rc_dry.describe()}")
        print(f"  colour out : {rc_dry.color_out_size[0]}x{rc_dry.color_out_size[1]}")
        print(f"  depth out  : {rc_dry.depth_out_size[0]}x{rc_dry.depth_out_size[1]}")
        print(f"  mono       : {rc_dry.mono_size[0]}x{rc_dry.mono_size[1]} "
              f"@ {rc_dry.mono_fps:g} fps")
        print(f"  budget     : {rc_dry.budget_note()}")
        bw = rc_dry.bandwidth_mbps()
        if bw["total"] > C.POE_BUDGET_MBPS:
            headroom = C.POE_BUDGET_MBPS / bw["total"]
            print(f"\n  This asks for {bw['total']/C.POE_BUDGET_MBPS:.1f}x the measured "
                  f"link capacity, so expect roughly {cfg.fps * headroom:.1f} fps "
                  f"of the {cfg.fps:g} requested, with the rest dropped.")
            print("  Reduce it with --isp-scale, --color-encode mjpeg, or a lower --fps.")
        for w in rc_dry.warnings:
            print(f"  warning    : {w}")
        return 0
    min_mm, max_mm = a.min_mm, a.max_mm
    mode = a.view
    show_hud = not a.no_hud

    print("=" * 74)
    print("C3 direct DepthAI stream")
    print("=" * 74)

    # Readiness before the ~12 s PoE boot, so "the camera is owned" costs two
    # seconds and names its owner rather than three failed connection attempts.
    # No vehicle checks: this tool records no telemetry, so waiting on MAVLink
    # would be a delay that buys the operator nothing.
    rc_pf = PF.gate(a, title="c3 stream", vehicle=False,
                    out_dir=(a.record_dir or Path("c3_camera/recordings"))
                            if a.record else None,
                    display=not a.no_display)
    if rc_pf is not None:
        return rc_pf

    log = CsvLog(a.csv) if a.csv else None
    window = "C3 — RGB + depth"
    # --duration and --warmup must measure the streaming window, so the clock
    # starts after the device is up: connecting costs ~12 s on PoE and counting
    # it would silently eat most of a short run.
    t_connect = time.monotonic()
    t_start = t_connect
    reset_at = t_start
    warmed = a.warmup <= 0
    last_print = 0.0
    frames = 0
    rc = 0
    stall_warned = False
    rec: Recorder | None = None

    def start_recording(src: C3Source) -> Recorder:
        out = a.record_dir or (Path("c3_camera/recordings") /
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        r = Recorder(out, depth_format=a.record_depth_format,
                     queue_size=a.record_queue, block_ms=a.record_block_ms)
        r.write_meta(src.summary())
        colour_kb = 0.0
        cs = src.metrics.streams.get(STREAM_COLOR)
        if cs is not None and cs.mean_frame_kb:
            colour_kb = cs.mean_frame_kb
        rate = estimate_disk_mb_per_min(
            colour_kb, tuple(src.rc.depth_out_size), cfg.fps, a.record_depth_format)
        print(f"  RECORDING -> {out}")
        print(f"    depth as {a.record_depth_format} (uint16 mm, lossless); "
              f"colour {'JPEG passthrough (no re-encode)' if cfg.color_encode == 'mjpeg' else 'PNG'}")
        print(f"    expect roughly {rate:.0f} MB/min ({rate*60/1024:.1f} GB/hour) "
              f"at {cfg.fps:g} fps")
        return r

    try:
        # keep_raw lets the recorder write MJPEG colour without re-encoding it.
        with C3Source(cfg, strict=a.strict,
                      keep_raw=(cfg.color_encode == "mjpeg")) as src:
            t_start = time.monotonic()
            reset_at = t_start + max(0.0, a.warmup)
            print(f"  connected in {t_start - t_connect:.1f}s")
            print(f"  streaming: {', '.join(src.formats[k] + ':' + k for k in src.formats)}")
            if a.warmup > 0:
                print(f"  warming up {a.warmup:g}s before statistics start ...")
            if not a.no_display:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            while True:
                bundle = src.read(timeout_ms=2000)
                if bundle is None:
                    if a.duration and time.monotonic() - t_start > a.duration:
                        break
                    if not stall_warned:
                        stall_warned = True
                        print("  no frames for 2 s — connected but not delivering. "
                              "If this persists, lower --fps or --isp-scale; if it "
                              "only happens with --pair-mode timestamp, the two "
                              "streams are further apart than --pair-tolerance-ms.")
                    continue
                stall_warned = False

                src.metrics.mark_loop()
                frames += 1

                if not warmed and time.monotonic() >= reset_at:
                    # Discard the settling period so the reported numbers describe
                    # the steady state, not auto-exposure ramping up.
                    src.metrics = type(src.metrics)(cfg.streams)
                    warmed = True
                    print("  statistics reset after warm-up")

                dstats = (viz.depth_stats(bundle.depth.image)
                          if bundle.depth is not None else None)

                if a.record and rec is None:
                    rec = start_recording(src)
                if rec is not None:
                    rec.submit(bundle)
                    if a.record_max_frames and rec.stats.written >= a.record_max_frames:
                        print(f"  reached --record-max-frames "
                              f"({a.record_max_frames}); stopping recording")
                        rec.close()
                        print(f"  {json.dumps(rec.summary(), indent=2)}")
                        rec = None
                        a.record = False

                if log is not None and warmed:
                    log.write(src, bundle, t_start,
                              dstats["valid_pct"] if dstats else None)

                now = time.monotonic()
                if a.print_every and now - last_print >= a.print_every:
                    last_print = now
                    head = f"[{now - t_start:6.1f}s]"
                    for line in src.metrics.hud_lines():
                        print(f"  {head} {line}")
                        head = " " * len(head)

                if not a.no_display:
                    canvas = compose(bundle, mode, min_mm, max_mm, a.scale)
                    if canvas is not None:
                        if show_hud:
                            viz.draw_hud(canvas,
                                         hud_lines(src, bundle, mode, min_mm,
                                                   max_mm, dstats, rec))
                        cv2.imshow(window, canvas)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("s"):
                        print(f"  saved {save_snapshot(bundle, a.save_dir, min_mm, max_mm)}")
                    elif key == ord("c"):
                        path = save_pointcloud(bundle, a.save_dir)
                        if path:
                            print(f"  saved {path}")
                    elif key == ord("o"):
                        mode = VIEW_MODES[(VIEW_MODES.index(mode) + 1) % len(VIEW_MODES)]
                        print(f"  view: {mode}")
                    elif key == ord("h"):
                        show_hud = not show_hud
                    elif key == ord("p"):
                        print(json.dumps(src.metrics.summary(), indent=2))
                    elif key == ord("v"):
                        if rec is None:
                            rec = start_recording(src)
                            a.record = True
                        else:
                            rec.close()
                            print(f"  stopped recording: "
                                  f"{json.dumps(rec.summary(), indent=2)}")
                            rec = None
                            a.record = False
                    elif key == ord("r"):
                        src.metrics = type(src.metrics)(cfg.streams)
                        print("  statistics reset")
                    elif key == ord("["):
                        max_mm = max(min_mm + 200, max_mm - 500)
                        print(f"  depth range {min_mm:.0f}-{max_mm:.0f} mm")
                    elif key == ord("]"):
                        max_mm += 500
                        print(f"  depth range {min_mm:.0f}-{max_mm:.0f} mm")

                if a.duration and time.monotonic() - t_start > a.duration:
                    break

            summary = src.summary()

    except KeyboardInterrupt:
        print("\n  interrupted")
        summary = None
        rc = 130
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}")
        summary = None
        rc = 1
    finally:
        # Close the recorder first so the writer thread flushes its queue even
        # when we got here via Ctrl-C or an exception; a half-written recording
        # with a truncated index is worse than a short one.
        if rec is not None:
            rec.close()
            print("\n  recording:")
            print("  " + json.dumps(rec.summary(), indent=2).replace("\n", "\n  "))
            if rec.stats.dropped:
                print(f"  NOTE: {rec.stats.dropped} frames never reached disk "
                      f"({rec.stats.queue_drop_rate*100:.1f}%) — the disk could not "
                      f"keep up. Try --record-depth-format npy, a lower --fps, or "
                      f"a faster disk. Gaps are visible as jumps in frames.csv seq.")
        if log is not None:
            log.close()
            print(f"  wrote {log.path}")
        if not a.no_display:
            cv2.destroyAllWindows()

    if summary:
        print("\n" + "=" * 74)
        print("run summary")
        print("=" * 74)
        r = summary["resolved"]
        m = summary["metrics"]
        print(f"  color {r['color_out'][0]}x{r['color_out'][1]}  "
              f"depth {r['depth_out'][0]}x{r['depth_out'][1]}  |  "
              f"link {m.get('mbps_measured_total', 0):.0f} Mbit/s measured "
              f"(est. {r['bandwidth_mbps'].get('total', 0):.0f}) "
              f"= {m.get('mbps_measured_total', 0)/C.POE_BUDGET_MBPS*100:.0f}% of ceiling")
        for st in m["streams"]:
            if not st["frames"]:
                continue
            print(f"  {st['stream']:<6} {st['frames']:5d} frames  "
                  f"{st['fps_lifetime']:5.2f} fps  "
                  f"latency p50 {st['latency_p50_ms']:6.1f}  "
                  f"p95 {st['latency_p95_ms']:6.1f}  max {st['latency_max_ms']:6.1f} ms  "
                  f"dropped {st['dropped']} ({st['drop_rate']*100:.1f}%)  "
                  f"{st['mean_frame_kb']:6.1f} kB/frame  {st['mbps_measured']:5.1f} Mbit/s")
        if a.json_out:
            a.json_out.parent.mkdir(parents=True, exist_ok=True)
            a.json_out.write_text(json.dumps(summary, indent=2, default=str))
            print(f"  wrote {a.json_out}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
