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
from c3_camera.config import STREAM_COLOR, STREAM_DEPTH, StreamConfig
from c3_camera.source import C3Source

FIELDS = (
    "timestamp", "ok", "error",
    "color_res", "isp_scale", "color_out", "encode", "mono_res", "depth_out",
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


def _parse_depth_size(text: str | None):
    if not text:
        return None
    sep = "x" if "x" in text else ","
    w, h = text.split(sep, 1)
    return int(w), int(h)


def run_one(cfg: StreamConfig, duration: float, warmup: float,
            verbose: bool = True) -> dict:
    """Stream one configuration for `duration` seconds and return a CSV row."""
    rc_cfg = cfg.resolve()
    row = {f: "" for f in FIELDS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "color_res": cfg.color_res,
        "isp_scale": "none" if cfg.isp_scale is None else f"{cfg.isp_scale[0]}/{cfg.isp_scale[1]}",
        "color_out": f"{rc_cfg.color_out_size[0]}x{rc_cfg.color_out_size[1]}",
        "encode": cfg.color_encode,
        "mono_res": cfg.mono_res,
        "depth_out": f"{rc_cfg.depth_out_size[0]}x{rc_cfg.depth_out_size[1]}",
        "requested_fps": cfg.fps,
        "streams": "+".join(cfg.streams),
        "depth_align": cfg.depth_align,
        "subpixel": int(cfg.subpixel),
        "lr_check": int(cfg.lr_check),
        "est_mbps": round(rc_cfg.bandwidth_mbps()["total"], 1),
        "ok": 0,
    })

    t0 = time.monotonic()
    try:
        with C3Source(cfg, verbose=verbose) as src:
            reset_done = warmup <= 0
            deadline = time.monotonic() + duration + max(0.0, warmup)
            reset_at = time.monotonic() + max(0.0, warmup)
            while time.monotonic() < deadline:
                b = src.read(timeout_ms=2000)
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
                   help="none,mjpeg (default: %(default)s — the raw-vs-encoded "
                        "comparison is the most informative single sweep)")
    g.add_argument("--streams", default="color,depth",
                   help="stream set, or several sets separated by ';' "
                        "e.g. 'color;color,depth' (default: %(default)s)")

    g = p.add_argument_group("fixed settings")
    g.add_argument("--depth-align", default="color",
                   choices=("color", "right", "left", "center", "none"))
    g.add_argument("--depth-preset", default="robotics", choices=sorted(C.DEPTH_PRESETS))
    g.add_argument("--depth-size", default=None, metavar="WxH",
                   help="override the depth output size. Worth sweeping: with MJPEG "
                        "colour, depth becomes the larger stream by ~3.5x")
    g.add_argument("--mjpeg-quality", type=int, default=90)
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
    g.add_argument("--append", action="store_true", help="append instead of overwriting")
    g.add_argument("--quiet", action="store_true", help="less per-run chatter")
    a = p.parse_args(argv)

    stream_sets = [tuple(_csv_list(s)) for s in a.streams.split(";") if s.strip()]
    combos = list(itertools.product(
        _csv_list(a.color_res),
        [_parse_isp(s) for s in _csv_list(a.isp_scale)],
        [float(f) for f in _csv_list(a.fps)],
        _csv_list(a.mono_res),
        _csv_list(a.color_encode),
        stream_sets,
    ))

    est = len(combos) * (a.duration + a.warmup + a.settle + 14)
    print("=" * 78)
    print(f"C3 sweep: {len(combos)} configurations, ~{est/60:.1f} min estimated")
    print("=" * 78)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = a.append and a.out.exists()
    fh = a.out.open("a" if new_file else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if not new_file:
        writer.writeheader()

    rows = []
    try:
        for i, (cres, isp, fps, mres, enc, streams) in enumerate(combos, 1):
            cfg = StreamConfig(
                ip=a.ip, mxid=a.mxid, streams=streams,
                depth_size=_parse_depth_size(a.depth_size),
                mjpeg_quality=a.mjpeg_quality,
                color_res=cres, isp_scale=isp, color_encode=enc,
                mono_res=mres, fps=fps,
                depth_align=a.depth_align, depth_preset=a.depth_preset,
                subpixel=a.subpixel, lr_check=a.lr_check,
            )
            label = (f"{cres} isp={'none' if isp is None else f'{isp[0]}/{isp[1]}'} "
                     f"{fps:g}fps mono={mres} enc={enc} [{'+'.join(streams)}]")
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
    print("\n" + "=" * 116)
    print(f"{'config':<42} {'req':>5} {'c.fps':>6} {'c.p50':>7} {'c.p95':>7} "
          f"{'d.fps':>6} {'d.p50':>7} {'drop%':>6} {'meas':>6} {'est':>6}")
    print(f"{'':<42} {'fps':>5} {'':>6} {'ms':>7} {'ms':>7} {'':>6} {'ms':>7} "
          f"{'':>6} {'Mbit/s':>6} {'Mbit/s':>6}")
    print("-" * 116)
    for r in rows:
        cfg_label = (f"{r['color_res']}@{r['isp_scale']}->{r['color_out']} "
                     f"{r['encode']} m{r['mono_res']}")
        if not r["ok"]:
            print(f"{cfg_label:<42} {r['requested_fps']:>5}  FAILED: {r['error'][:44]}")
            continue
        drop = r["color_drop_rate"] or 0
        print(f"{cfg_label:<42} {r['requested_fps']:>5} "
              f"{r['color_fps'] or 0:>6} {r['color_lat_p50'] or 0:>7} "
              f"{r['color_lat_p95'] or 0:>7} {r['depth_fps'] or 0:>6} "
              f"{r['depth_lat_p50'] or 0:>7} {float(drop)*100:>6.1f} "
              f"{r['measured_mbps'] or 0:>6} {r['est_mbps']:>6}")
    print("=" * 116)
    print(f"PoE link ceiling on this path: ~{C.POE_BUDGET_MBPS:.0f} Mbit/s. A row "
          f"whose 'meas' sits at the ceiling with a high drop% is link-limited, "
          f"not compute-limited.")
    print(f"wrote {a.out}")

    good = [r for r in rows if r["ok"] and r["color_fps"]]
    if good:
        best = min(good, key=lambda r: float(r["color_lat_p50"] or 1e9))
        fastest = max(good, key=lambda r: float(r["color_fps"] or 0))
        print(f"\nlowest latency : {best['color_res']}@{best['isp_scale']} "
              f"{best['requested_fps']:g} fps -> {best['color_fps']} fps, "
              f"p50 {best['color_lat_p50']} ms")
        print(f"highest fps    : {fastest['color_res']}@{fastest['isp_scale']} "
              f"{fastest['requested_fps']:g} fps -> {fastest['color_fps']} fps, "
              f"p50 {fastest['color_lat_p50']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
