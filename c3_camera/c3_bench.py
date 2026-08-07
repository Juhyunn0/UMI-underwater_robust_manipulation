#!/usr/bin/env python3
"""
c3_bench.py — sweep resolution/fps combinations and record what each one costs.

The point of this script is to answer the question the Luxonis latency table
poses ("150 ms at 4K/8fps, 530 ms at 4K/10fps") for *this* camera on *this*
network, instead of guessing. It reconnects for every combination — a DepthAI
pipeline is fixed at device boot, so each configuration is genuinely a new run —
and writes one CSV row per configuration.

    # the default ladder: raw vs device-side MJPEG at three frame rates
    python c3_camera/c3_bench.py --out bench.csv

    # find where the link saturates at full 1080p
    python c3_camera/c3_bench.py --color-res 1080p --isp-scale none \\
        --fps 5,10,15,20,30 --duration 12 --out bench_1080p.csv

    # how far down does the resolution ladder have to go for 30 fps?
    python c3_camera/c3_bench.py --isp-scale 1/4,1/3,1/2 --fps 30

    # depth is the larger stream once colour is encoded: where is *its* ceiling?
    python c3_camera/c3_bench.py --depth-size 320x180,480x270,640x360 \\
        --fps 15,20 --color-encode mjpeg --out bench_depth.csv

    # the encoder ladder. --streams color because uncompressed depth would
    # otherwise dominate the link and bury the difference being measured.
    python c3_camera/c3_bench.py --streams color --fps 20 \\
        --color-encode none,mjpeg,h264,h265 --mjpeg-quality 75,90 \\
        --video-bitrate-kbps 4000,8000,16000 --out bench_encode.csv

    # at a fixed bitrate, does a keyframe every 2 s beat one per second?
    python c3_camera/c3_bench.py --color-encode h265 --fps 15 \\
        --video-bitrate-kbps 8000 --video-keyframe-frequency 15,30

Axes a configuration cannot see are folded away before anything touches
hardware: bitrate and keyframe interval only vary for h264/h265, MJPEG quality
only for mjpeg, depth size only when depth is actually streamed. The encoder
ladder above therefore costs 9 runs (1 + 2 + 3 + 3), not 4x2x3 = 24, most of
which would have been byte-for-byte repeats.

One duplicate the fold cannot see: an explicit --depth-size that happens to
equal the derived one (640x360 at mono 400p with 16:9 colour) is a second run of
the same pipeline, because whether they coincide is only known after resolve().

Each row carries the measured fps and latency percentiles per stream plus the
drop rate, so a configuration that "reaches 30 fps" by discarding half its frames
is visible as such rather than looking like a win.

Expect roughly `--duration + --warmup + --settle + 14` seconds per combination:
connecting uploads the firmware and boots the device (~12 s measured), and a PoE
device resets when the owner disconnects and needs a few more seconds to return.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c3_camera import config as C
from c3_camera import preflight as PF
from c3_camera.config import STREAM_COLOR, STREAM_DEPTH, StreamConfig
from c3_camera.source import C3Source

FIELDS = (
    "timestamp", "ok", "error",
    "color_res", "isp_scale", "color_out", "encode",
    "mjpeg_quality", "video_bitrate_kbps", "video_keyframe_frequency",
    "mono_res", "depth_out",
    "requested_fps", "streams", "depth_align", "subpixel", "lr_check",
    "est_mbps", "measured_mbps",
    "color_fps", "color_lat_p50", "color_lat_p95", "color_lat_max",
    "color_frames", "color_dropped", "color_drop_rate",
    "color_kb_frame", "color_mbps",
    "depth_fps", "depth_lat_p50", "depth_lat_p95", "depth_lat_max",
    "depth_frames", "depth_dropped", "depth_drop_rate",
    "depth_kb_frame", "depth_mbps",
    "loop_hz", "duration_s",
)


def _csv_list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _parse_isp(text: str):
    if text.lower() in ("none", "off", "1", "1/1"):
        return None
    n, d = text.split("/", 1)
    return int(n), int(d)


def _parse_ints(text: str) -> list[int]:
    items = _csv_list(text)
    if not items:
        raise argparse.ArgumentTypeError(f"expected an integer list, got {text!r}")
    try:
        return [int(s) for s in items]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}") from e


def _parse_depth_sizes(text: str) -> list[tuple[int, int] | None]:
    """Comma list of WxH; 'derived' keeps the default mono-width sizing.

    The comma is the sweep separator here, so unlike config.py's --depth-size the
    "640,360" spelling of a single size is not accepted — it is indistinguishable
    from a request for two sizes.
    """
    out: list[tuple[int, int] | None] = []
    for tok in _csv_list(text):
        if tok.lower() in ("derived", "none", "auto"):
            out.append(None)
            continue
        if "x" not in tok:
            raise argparse.ArgumentTypeError(
                f"--depth-size wants WxH or 'derived', got {tok!r}")
        w, h = tok.split("x", 1)
        try:
            out.append((int(w), int(h)))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"--depth-size {tok!r} is not WxH") from e
    if not out:
        raise argparse.ArgumentTypeError(f"expected a size list, got {text!r}")
    return out


def _encode_label(encode: str, bitrate: int, keyframe: int, quality: int) -> str:
    """Encoder plus the one knob it reads.

    Without it a bitrate or quality ladder prints as several rows that look
    identical, which is the same failure the CSV columns exist to prevent.
    """
    if encode in ("h264", "h265"):
        kf = "" if not keyframe else f"/kf{keyframe}"
        return f"{encode}@{bitrate}k{kf}"
    if encode == "mjpeg":
        return f"mjpeg-q{quality}"
    return encode


def run_one(cfg: StreamConfig, duration: float, warmup: float,
            verbose: bool = True) -> dict:
    """Stream one configuration for `duration` seconds and return a CSV row."""
    row = {f: "" for f in FIELDS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "color_res": cfg.color_res,
        "isp_scale": "none" if cfg.isp_scale is None else f"{cfg.isp_scale[0]}/{cfg.isp_scale[1]}",
        "encode": cfg.color_encode,
        # Record only the knob the chosen encoder actually reads. An MJPEG row
        # carrying a bitrate — or an H.264 row carrying a JPEG quality — invites
        # exactly the misreading these columns were added to prevent.
        "mjpeg_quality": cfg.mjpeg_quality if cfg.color_encode == "mjpeg" else "",
        "video_bitrate_kbps": (cfg.video_bitrate_kbps
                               if cfg.color_encode in ("h264", "h265") else ""),
        "video_keyframe_frequency": (cfg.video_keyframe_frequency
                                     if cfg.color_encode in ("h264", "h265") else ""),
        "mono_res": cfg.mono_res,
        # The *requested* depth size, so a row that fails before resolve() still
        # says which rung of a --depth-size ladder it was; the resolved size
        # overwrites it below on every run that gets that far. Without this a
        # sweep whose combinations are rejected writes rows that are identical
        # except for the timestamp.
        "depth_out": ("derived" if cfg.depth_size is None
                      else f"{cfg.depth_size[0]}x{cfg.depth_size[1]}"),
        "requested_fps": cfg.fps,
        "streams": "+".join(cfg.streams),
        "depth_align": cfg.depth_align,
        "subpixel": int(cfg.subpixel),
        "lr_check": int(cfg.lr_check),
        "ok": 0,
    })

    t0 = time.monotonic()
    try:
        # resolve() inside the try: the sweep axes are free-form strings, so a
        # rejected combination must cost one failed row rather than abort a
        # multi-hour sweep that is holding the camera.
        rc_cfg = cfg.resolve()
        row["color_out"] = f"{rc_cfg.color_out_size[0]}x{rc_cfg.color_out_size[1]}"
        row["depth_out"] = f"{rc_cfg.depth_out_size[0]}x{rc_cfg.depth_out_size[1]}"
        row["est_mbps"] = round(rc_cfg.bandwidth_mbps()["total"], 1)

        encoded = cfg.color_encode in ("h264", "h265")
        with C3Source(cfg, verbose=verbose) as src:
            reset_done = warmup <= 0
            deadline = time.monotonic() + duration + max(0.0, warmup)
            reset_at = time.monotonic() + max(0.0, warmup)
            while time.monotonic() < deadline:
                b = src.read(timeout_ms=2000)
                if encoded:
                    # C3Source keeps every H.26x packet until someone drains it,
                    # because a P frame is worthless without its predecessors.
                    # The bench wants the byte counts, which metrics has already
                    # taken; keeping the bytes would grow ~50 MB per 30 Mbit/s run.
                    src.drain_encoded_rgb()
                if b is None:
                    continue
                src.metrics.mark_loop()
                if not reset_done and time.monotonic() >= reset_at:
                    src.metrics = type(src.metrics)(cfg.streams)
                    reset_done = True

            for tag, name in (("color", STREAM_COLOR), ("depth", STREAM_DEPTH)):
                st = src.metrics.streams.get(name)
                if st is None or not st.count:
                    continue
                s = st.summary()
                row[f"{tag}_fps"] = s["fps_lifetime"]
                row[f"{tag}_lat_p50"] = s["latency_p50_ms"]
                row[f"{tag}_lat_p95"] = s["latency_p95_ms"]
                row[f"{tag}_lat_max"] = s["latency_max_ms"]
                row[f"{tag}_frames"] = s["frames"]
                row[f"{tag}_dropped"] = s["dropped"]
                row[f"{tag}_drop_rate"] = s["drop_rate"]
                row[f"{tag}_kb_frame"] = s["mean_frame_kb"]
                row[f"{tag}_mbps"] = s["mbps_measured"]
            row["measured_mbps"] = round(src.metrics.mbps_total, 1)
            row["loop_hz"] = round(src.metrics.loop_hz, 2)
            row["ok"] = 1
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]

    row["duration_s"] = round(time.monotonic() - t0, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    C.add_connection_args(p)

    g = p.add_argument_group("sweep axes (comma-separated)")
    g.add_argument("--color-res", default="1080p",
                   help="e.g. 1080p,2024x1520,4k (default: %(default)s)")
    g.add_argument("--isp-scale", default="1/2",
                   help="e.g. 1/4,1/2,none (default: %(default)s)")
    g.add_argument("--fps", default="10,20,30", help="e.g. 5,10,15,30 (default: %(default)s)")
    g.add_argument("--mono-res", default="400p", help="e.g. 400p,720p (default: %(default)s)")
    g.add_argument("--color-encode", default="none,mjpeg",
                   help="none,mjpeg,h264,h265 (default: %(default)s — the "
                        "raw-vs-encoded comparison is the most informative "
                        "single sweep; h264/h265 need --video-bitrate-kbps to "
                        "say anything)")
    g.add_argument("--streams", default="color,depth",
                   help="stream set, or several sets separated by ';' "
                        "e.g. 'color;color,depth' (default: %(default)s)")
    g.add_argument("--depth-size", type=_parse_depth_sizes, default=[None],
                   metavar="WxH[,WxH...]",
                   help="depth output size(s), or 'derived' for the mono-width "
                        "default (default: derived). Worth sweeping: with an "
                        "encoded colour stream, uncompressed depth becomes the "
                        "larger stream by ~3.5x and sets the link ceiling")
    g.add_argument("--mjpeg-quality", type=_parse_ints, default=[90],
                   metavar="Q[,Q...]",
                   help="MJPEG quality 0-100, e.g. 75,90 (default: 90). MJPEG's "
                        "only bandwidth knob, so it is the fair counterpart to "
                        "--video-bitrate-kbps when comparing encoders")
    g.add_argument("--video-bitrate-kbps", type=_parse_ints,
                   default=[C.DEFAULT_VIDEO_BITRATE_KBPS], metavar="KBPS[,KBPS...]",
                   help=f"H.264/H.265 target bitrate(s), e.g. 4000,8000,16000 "
                        f"(default: {C.DEFAULT_VIDEO_BITRATE_KBPS}). Rate control "
                        f"is CBR, so this *is* the colour stream's link cost")
    g.add_argument("--video-keyframe-frequency", type=_parse_ints, default=[0],
                   metavar="N[,N...]",
                   help="H.264/H.265 keyframe interval in frames; 0 means one "
                        "per second (default: 0). Longer intervals buy bitrate "
                        "at the cost of a longer wait to start decoding")

    g = p.add_argument_group("fixed settings")
    g.add_argument("--depth-align", default="color",
                   choices=("color", "right", "left", "center", "none"))
    g.add_argument("--depth-preset", default="robotics", choices=sorted(C.DEPTH_PRESETS))
    g.add_argument("--subpixel", action="store_true")
    g.add_argument("--no-lr-check", dest="lr_check", action="store_false", default=True)

    g = p.add_argument_group("run control")
    g.add_argument("--duration", type=float, default=10.0,
                   help="measured seconds per configuration (default: %(default)s)")
    g.add_argument("--warmup", type=float, default=3.0,
                   help="discarded seconds per configuration (default: %(default)s)")
    g.add_argument("--settle", type=float, default=5.0,
                   help="pause between configurations, for the PoE device to reset "
                        "(default: %(default)s)")
    g.add_argument("--out", type=Path, default=Path("c3_camera/bench_results.csv"),
                   help="CSV output (default: %(default)s)")
    g.add_argument("--append", action="store_true",
                   help="append instead of overwriting; refused if the existing "
                        "file was written with a different column set")
    g.add_argument("--quiet", action="store_true", help="less per-run chatter")
    PF.add_preflight_args(p)
    a = p.parse_args(argv)

    stream_sets = [tuple(_csv_list(s)) for s in a.streams.split(";") if s.strip()]
    combos = []
    for (cres, isp, fps, mres, enc, streams,
         dsize, bitrate, keyframe, quality) in itertools.product(
            _csv_list(a.color_res),
            [_parse_isp(s) for s in _csv_list(a.isp_scale)],
            [float(f) for f in _csv_list(a.fps)],
            _csv_list(a.mono_res),
            _csv_list(a.color_encode),
            stream_sets,
            a.depth_size,
            a.video_bitrate_kbps,
            a.video_keyframe_frequency,
            a.mjpeg_quality):
        # Pin the axes this encoder cannot see, then drop the duplicates that
        # creates. none/mjpeg ignore bitrate and keyframe interval, h264/h265
        # ignore JPEG quality, so a 4-encoder x 3-bitrate ladder would otherwise
        # spend a third of its ~32 s runs re-measuring identical pipelines — and
        # the CSV could not tell those rows apart afterwards either.
        if enc not in ("h264", "h265"):
            bitrate = a.video_bitrate_kbps[0]
            keyframe = a.video_keyframe_frequency[0]
        if enc != "mjpeg":
            quality = a.mjpeg_quality[0]
        if STREAM_DEPTH not in streams:
            # Same rule for the depth ladder: with no depth stream the size
            # changes nothing on the wire, but resolve() still reports one, so
            # unfolded rows would differ in depth_out while measuring the very
            # same pipeline — a difference a reader would try to explain.
            dsize = a.depth_size[0]
        combos.append((cres, isp, fps, mres, enc, streams,
                       dsize, bitrate, keyframe, quality))
    combos = list(dict.fromkeys(combos))

    est = len(combos) * (a.duration + a.warmup + a.settle + 14)
    print("=" * 78)
    print(f"C3 sweep: {len(combos)} configurations, ~{est/60:.1f} min estimated")
    print("=" * 78)

    # A sweep is long and unattended, so it is the worst place to discover that
    # the camera was never free. Two seconds here, against a run that would
    # otherwise fail the same way on every configuration.
    rc_pf = PF.gate(a, title="c3 bench", vehicle=False, out_dir=a.out.parent)
    if rc_pf is not None:
        return rc_pf

    a.out.parent.mkdir(parents=True, exist_ok=True)
    appending = a.append and a.out.exists() and a.out.stat().st_size > 0
    if appending:
        # DictWriter emits values in FIELDS order under whatever header is
        # already in the file. A CSV written before a column was added would
        # therefore take every value after that column and file it one place to
        # the left — silently, and only ever caught by eye. Refuse instead.
        with a.out.open(newline="") as probe:
            existing = next(csv.reader(probe), [])
        if existing != list(FIELDS):
            print(f"error: {a.out} has {len(existing)} columns and this build "
                  f"writes {len(FIELDS)}; appending would misalign every row. "
                  f"Use a fresh --out for this stage.")
            return 2
    fh = a.out.open("a" if appending else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if not appending:
        writer.writeheader()

    rows = []
    try:
        for i, (cres, isp, fps, mres, enc, streams,
                dsize, bitrate, keyframe, quality) in enumerate(combos, 1):
            cfg = StreamConfig(
                ip=a.ip, mxid=a.mxid, streams=streams,
                depth_size=dsize,
                mjpeg_quality=quality,
                video_bitrate_kbps=bitrate,
                video_keyframe_frequency=keyframe,
                color_res=cres, isp_scale=isp, color_encode=enc,
                mono_res=mres, fps=fps,
                depth_align=a.depth_align, depth_preset=a.depth_preset,
                subpixel=a.subpixel, lr_check=a.lr_check,
            )
            label = (f"{cres} isp={'none' if isp is None else f'{isp[0]}/{isp[1]}'} "
                     f"{fps:g}fps mono={mres} "
                     f"depth={'derived' if dsize is None else f'{dsize[0]}x{dsize[1]}'} "
                     f"enc={_encode_label(enc, bitrate, keyframe, quality)} "
                     f"[{'+'.join(streams)}]")
            print(f"\n[{i}/{len(combos)}] {label}")

            row = run_one(cfg, a.duration, a.warmup, verbose=not a.quiet)
            rows.append(row)
            writer.writerow(row)
            fh.flush()

            if row["ok"]:
                print(f"    -> color {row['color_fps'] or '-'} fps "
                      f"p50 {row['color_lat_p50'] or '-'} ms | "
                      f"depth {row['depth_fps'] or '-'} fps "
                      f"p50 {row['depth_lat_p50'] or '-'} ms | "
                      f"drop {row['color_drop_rate'] or 0} / {row['depth_drop_rate'] or 0}")
            else:
                print(f"    -> FAILED: {row['error']}")

            if i < len(combos):
                time.sleep(a.settle)
    except KeyboardInterrupt:
        print("\ninterrupted — partial results kept")
    finally:
        fh.close()

    # ------------------------------------------------------------------ table
    print("\n" + "=" * 124)
    print(f"{'config':<52} {'req':>5} {'c.fps':>6} {'c.p50':>7} {'c.p95':>7} "
          f"{'d.fps':>6} {'d.p50':>7} {'drop%':>6} {'meas':>6} {'est':>6}")
    print(f"{'':<52} {'fps':>5} {'':>6} {'ms':>7} {'ms':>7} {'':>6} {'ms':>7} "
          f"{'':>6} {'Mbit/s':>6} {'Mbit/s':>6}")
    print("-" * 124)
    for r in rows:
        enc_label = _encode_label(r["encode"], r["video_bitrate_kbps"] or 0,
                                  r["video_keyframe_frequency"] or 0,
                                  r["mjpeg_quality"] or 0)
        cfg_label = (f"{r['color_res']}@{r['isp_scale']}->{r['color_out']} "
                     f"{enc_label} m{r['mono_res']} d{r['depth_out']}")
        if not r["ok"]:
            print(f"{cfg_label:<52} {r['requested_fps']:>5}  FAILED: {r['error'][:54]}")
            continue
        drop = r["color_drop_rate"] or 0
        print(f"{cfg_label:<52} {r['requested_fps']:>5} "
              f"{r['color_fps'] or 0:>6} {r['color_lat_p50'] or 0:>7} "
              f"{r['color_lat_p95'] or 0:>7} {r['depth_fps'] or 0:>6} "
              f"{r['depth_lat_p50'] or 0:>7} {float(drop)*100:>6.1f} "
              f"{r['measured_mbps'] or 0:>6} {r['est_mbps']:>6}")
    print("=" * 124)
    print(f"PoE link ceiling on this path: ~{C.POE_BUDGET_MBPS:.0f} Mbit/s. A row "
          f"whose 'meas' sits at the ceiling with a high drop% is link-limited, "
          f"not compute-limited.")
    print(f"wrote {a.out}")

    good = [r for r in rows if r["ok"] and r["color_fps"]]
    if good:
        best = min(good, key=lambda r: float(r["color_lat_p50"] or 1e9))
        fastest = max(good, key=lambda r: float(r["color_fps"] or 0))
        # Name the encoder and depth size too: on an encoder or depth ladder the
        # winner and the runner-up differ in nothing else.
        print()
        for tag, r in (("lowest latency", best), ("highest fps   ", fastest)):
            enc_label = _encode_label(r["encode"], r["video_bitrate_kbps"] or 0,
                                      r["video_keyframe_frequency"] or 0,
                                      r["mjpeg_quality"] or 0)
            print(f"{tag} : {r['color_res']}@{r['isp_scale']} {enc_label} "
                  f"d{r['depth_out']} {r['requested_fps']:g} fps -> "
                  f"{r['color_fps']} fps, p50 {r['color_lat_p50']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
