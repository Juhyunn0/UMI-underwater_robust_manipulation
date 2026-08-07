#!/usr/bin/env python3
"""
c3_option_sweep.py — sweep the capture options, and keep a real SLAM dataset from
every one of them.

The four questions this answers are the ones you actually ask before a dive:

    how big should colour be?      --color-res / --isp-scale
    compress it, or not?           --color-encode (+ --mjpeg-quality / --video-bitrate-kbps)
    ship depth at all?             --depth on,off
    and at what resolution?        --depth-size match-color,derived,WxH

    V=~/.venvs/c3-depthai/bin/python

    # plan first — resolves every combination, prints wire + disk cost, no camera
    $V c3_camera/c3_option_sweep.py --dry-run

    # the default ladder: 2 colour sizes x raw/MJPEG x depth on/off x 2 depth grids
    $V c3_camera/c3_option_sweep.py

    # the user's question, spelled out: 4K at 2/3 with and without MJPEG,
    # against 1080p, depth on the colour grid
    $V c3_camera/c3_option_sweep.py \\
        --color-res 4k,1080p --isp-scale 2/3 --color-encode none,mjpeg \\
        --mjpeg-quality 90 --depth on --depth-size match-color --fps 20

    # how far can depth resolution be cut before the map suffers?
    $V c3_camera/c3_option_sweep.py --depth on \\
        --depth-size match-color,derived,320x180 --color-encode mjpeg

Relationship to the neighbouring scripts
----------------------------------------
`c3_bench.py` sweeps the same pipeline axes but keeps **no pixels** — it answers
"what fps and latency does this cost". `c3_encode_quality.py` sweeps the *colour
encoder* and keeps frames, but scores them against a median reference and writes
no dataset. This script is the third corner: it sweeps the axes above and writes,
for each cell, a **complete TUM RGB-D dataset in the same layout as
`c3_camera/datasets/dataset_*`** — rgb/ depth/ left/ right/, rgb.txt depth.txt
associations.txt frames.csv, calibration.json, an ORB-SLAM3 yaml, camera-IMU and
MAVLink CSVs, metadata.json/txt. So a setting is not judged only by its fps: you
can run SLAM on the recording it produced and judge it by the map.

That is the whole point of writing datasets rather than numbers. A configuration
that wins on Mbit/s and loses on tracking is invisible to a bench.

Reading the numbers honestly
---------------------------
* `*_mbps` is windowed over the last 120 frames — the tail of the capture.
  `*_mbps_run` is the whole-capture average from the lifetime byte count, and is
  the one to compare across cells.
* "Measured" throughput is only genuinely measured for a COMPRESSED stream. For
  `--color-encode none` the byte count is computed (`w*h*1.5` for NV12), and
  depth16 is always `w*h*2` — those are arithmetic, not measurement. This matters
  precisely on the compression axis: the raw rows are exact by construction while
  the MJPEG/H.26x rows carry real packet sizes.
* `check` is c3_dataset_check.py's verdict, and it is an RGB-D checker. Cells
  with `--depth off` are reported `skip-nodepth` rather than judged, and a cell
  whose depth is deliberately NOT on the colour grid is reported `*-offgrid`,
  because the resolution mismatch it flags is the setting under test.

What is deliberately NOT swept here
-----------------------------------
Axes that do not change what the recording *is*: pairing mode, queue depths,
median filter, confidence. Those belong on the command line as fixed settings
(they are, below) so that every cell of a sweep differs only in the axis under
test.

Cost
----
A DepthAI pipeline is fixed at device boot, so every cell is a separate
connection: budget `--warmup + --duration + --settle + ~14 s` of wall clock each,
plus disk. The header prints both estimates before anything connects, and
`--dry-run` prints them without taking the camera from whoever has it.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c3_camera import config as C
from c3_camera import device as D
from c3_camera import preflight as PF
from c3_camera.c3_collect import blueos_info, calibration_payload
from c3_camera.config import (STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT,
                              STREAM_RIGHT, StreamConfig)
from c3_camera.dataset import DEPTH_SCALE, DatasetWriter, estimate_rate
from c3_camera.imu import IMU_FIELDS
from c3_camera.mavlink_log import MavlinkLogger
from c3_camera.source import C3Source

FIELDS = (
    "n", "timestamp", "ok", "error", "dataset", "label",
    # --- the axes ---------------------------------------------------------
    "color_res", "isp_scale", "color_out", "encode",
    "mjpeg_quality", "video_bitrate_kbps",
    "depth", "depth_size_req", "depth_out", "depth_on_color_grid",
    "mono", "mono_res", "requested_fps", "streams",
    # --- link -------------------------------------------------------------
    "est_wire_mbps", "measured_mbps", "est_disk_mb_min",
    # --- per stream, as delivered ----------------------------------------
    "color_fps", "color_lat_p50", "color_lat_p95", "color_lat_max",
    "color_frames", "color_dropped", "color_drop_rate",
    "color_kb_frame", "color_mbps", "color_mbps_run",
    "depth_fps", "depth_lat_p50", "depth_lat_p95", "depth_lat_max",
    "depth_frames", "depth_dropped", "depth_drop_rate",
    "depth_kb_frame", "depth_mbps", "depth_mbps_run",
    "skew_ms_mean",
    # --- what landed on disk ---------------------------------------------
    "rec_seconds", "frames_written", "frames_dropped_writer", "writer_drop_rate",
    "images", "dataset_mb", "dataset_mb_per_min",
    "rgb_index", "depth_index", "assoc_lines", "rgb_on_disk", "depth_on_disk",
    "imu_samples", "imu_hz", "mav_msgs",
    "slam_config", "check", "loop_hz", "wall_s",
)


# =============================================================================
# sweep-axis parsing
# =============================================================================
def _csv_list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _choice_list(name: str, allowed):
    """A comma-list validator for the string axes.

    Without one, an empty or misspelled value reaches itertools.product as an
    empty factor, the product collapses to zero cells, and the sweep prints
    "0 configurations" and exits 0 — a silent no-op that looks like a run.
    """
    allowed = sorted(allowed)

    def parse(text: str) -> list[str]:
        items = list(dict.fromkeys(_csv_list(text)))
        if not items:
            raise argparse.ArgumentTypeError(f"{name} needs at least one value")
        bad = [s for s in items if s not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(
                f"{name}: unknown value(s) {bad}; choose from {allowed}")
        return items

    return parse


def _parse_isp_list(text: str) -> list[tuple[int, int] | None]:
    out: list[tuple[int, int] | None] = []
    for tok in _csv_list(text):
        if tok.lower() in ("none", "off", "1", "1/1"):
            out.append(None)
            continue
        if "/" not in tok:
            raise argparse.ArgumentTypeError(
                f"--isp-scale wants N/D or 'none', got {tok!r}")
        n, d = tok.split("/", 1)
        try:
            out.append((int(n), int(d)))
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--isp-scale {tok!r} is not N/D") from e
    if not out:
        raise argparse.ArgumentTypeError(f"expected an --isp-scale list, got {text!r}")
    return out


def _parse_int_list(text: str) -> list[int]:
    try:
        out = [int(s) for s in _csv_list(text)]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}") from e
    if not out:
        raise argparse.ArgumentTypeError(f"expected an integer list, got {text!r}")
    return out


def _parse_float_list(text: str) -> list[float]:
    try:
        out = [float(s) for s in _csv_list(text)]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}") from e
    if not out:
        raise argparse.ArgumentTypeError(f"expected a number list, got {text!r}")
    return out


def _parse_onoff(text: str) -> list[bool]:
    """`on`, `off`, or both — the two-valued axes (`--depth`, `--mono`)."""
    out: list[bool] = []
    for tok in _csv_list(text):
        low = tok.lower()
        if low in ("on", "yes", "true", "1"):
            out.append(True)
        elif low in ("off", "no", "false", "0"):
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"expected on/off (or a comma list of them), got {tok!r}")
    if not out:
        raise argparse.ArgumentTypeError(f"expected on/off, got {text!r}")
    return list(dict.fromkeys(out))


def _parse_depth_sizes(text: str) -> list[str]:
    """Kept as tokens, because two of the three are not sizes at all.

    `match-color` and `derived` are *policies* that only become pixel counts
    after resolve() has seen the colour size, so folding them into a (w, h) here
    would either duplicate config.py's sizing rule or lose the distinction
    between "the colour grid" and "whatever the colour grid happens to be today".
    """
    out: list[str] = []
    for tok in _csv_list(text):
        low = tok.lower()
        if low in ("match-color", "match", "color-grid"):
            out.append("match-color")
        elif low in ("derived", "auto", "default"):
            out.append("derived")
        elif "x" in low:
            w, h = low.split("x", 1)
            try:
                int(w), int(h)
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    f"--depth-size {tok!r} is not WxH") from e
            out.append(f"{int(w)}x{int(h)}")
        else:
            raise argparse.ArgumentTypeError(
                f"--depth-size wants match-color, derived, or WxH; got {tok!r}")
    if not out:
        raise argparse.ArgumentTypeError(f"expected a --depth-size list, got {text!r}")
    return list(dict.fromkeys(out))


# =============================================================================
# one cell of the sweep
# =============================================================================
class Combo:
    """One cell: the axes, the StreamConfig they build, and how to name it."""

    def __init__(self, a: argparse.Namespace, color_res, isp, encode, quality,
                 bitrate, depth_on, depth_tok, mono_on, mono_res, fps):
        self.color_res = color_res
        self.isp = isp
        self.encode = encode
        self.quality = quality
        self.bitrate = bitrate
        self.depth_on = depth_on
        self.depth_tok = depth_tok
        self.mono_on = mono_on
        self.mono_res = mono_res
        self.fps = fps

        streams = [STREAM_COLOR]
        if depth_on:
            streams.append(STREAM_DEPTH)
        if mono_on:
            streams += [STREAM_LEFT, STREAM_RIGHT]

        depth_size = None
        match_color = False
        if depth_on:
            if depth_tok == "match-color":
                match_color = True
            elif depth_tok != "derived":
                w, h = depth_tok.split("x", 1)
                depth_size = (int(w), int(h))

        self.cfg = StreamConfig(
            ip=a.ip, mxid=a.mxid,
            streams=tuple(streams),
            color_res=color_res, isp_scale=isp,
            color_encode=encode, mjpeg_quality=quality,
            video_bitrate_kbps=bitrate,
            video_keyframe_frequency=a.video_keyframe_frequency,
            color_wire=a.color_wire,
            mono_res=mono_res,
            mono_encode=a.mono_encode, mono_quality=a.mono_quality,
            mono_source="raw",
            depth_preset=a.depth_preset, depth_align=a.depth_align,
            depth_size=depth_size, depth_match_color=match_color,
            median=a.median, confidence=a.confidence,
            lr_check=a.lr_check, subpixel=a.subpixel, extended=a.extended,
            fps=fps,
            # A dataset wants colour and depth that were genuinely captured
            # together and can afford to wait up to a frame for it; "latest"
            # never waits, which is right for a live view and wrong here.
            pair_mode="timestamp",
            imu_enable=not a.no_camera_imu,
            imu_rate=a.imu_rate,
            # 10, not 1: batching costs latency nobody here is spending, and at
            # batch=1 the per-message overhead loses samples whenever the image
            # streams are busy. A dataset wants completeness.
            imu_batch_threshold=a.imu_batch,
            imu_max_batch_reports=max(20, a.imu_batch * 4),
            queue_size=a.queue_size,
        )

    # ------------------------------------------------------------- identity
    @property
    def isp_label(self) -> str:
        return "none" if self.isp is None else f"{self.isp[0]}/{self.isp[1]}"

    @property
    def encode_label(self) -> str:
        """The encoder plus the one knob it reads — never the ones it ignores.

        An MJPEG row carrying a bitrate, or an H.265 row carrying a JPEG quality,
        invites exactly the misreading the fold below exists to prevent.
        """
        if self.encode in ("h264", "h265"):
            return f"{self.encode}@{self.bitrate}k"
        if self.encode == "mjpeg":
            return f"mjpeg-q{self.quality}"
        return "raw"

    @property
    def depth_label(self) -> str:
        return f"depth-{self.depth_tok}" if self.depth_on else "nodepth"

    def label(self) -> str:
        mono = "+lr" if self.mono_on else ""
        return (f"{self.color_res}@{self.isp_label} {self.encode_label} "
                f"{self.depth_label} m{self.mono_res}{mono} {self.fps:g}fps")

    def slug(self, n: int) -> str:
        isp = "ispnone" if self.isp is None else f"isp{self.isp[0]}-{self.isp[1]}"
        enc = self.encode_label.replace("@", "").replace("/", "-")
        mono = "_lr" if self.mono_on else ""
        return (f"{n:02d}_{self.color_res}_{isp}_{enc}_"
                f"{self.depth_tok if self.depth_on else 'nodepth'}"
                f"{mono}_{self.fps:g}fps")

    def key(self) -> tuple:
        """Everything that shapes the pipeline EXCEPT how depth was sized.

        The depth axis is keyed separately by build_combos, on the pixel grid the
        request resolves to rather than the word used to ask for it.
        """
        return (self.color_res, self.isp, self.encode, self.quality, self.bitrate,
                self.depth_on, self.mono_on, self.mono_res, self.fps)


def build_combos(a: argparse.Namespace) -> list[Combo]:
    """Cartesian product, minus the cells that cannot differ from each other.

    An axis a configuration cannot see is pinned to its first value before the
    duplicates are dropped: `none`/`mjpeg` ignore the video bitrate, h264/h265
    ignore the JPEG quality, and with no depth stream the depth size changes
    nothing on the wire or on disk. Without the fold, a 4-encoder x 3-bitrate
    ladder would spend a third of its runs re-measuring byte-identical
    pipelines — and the CSV could not tell those rows apart afterwards either.
    """
    combos: list[Combo] = []
    seen: set[tuple] = set()
    for (cres, isp, enc, quality, bitrate, depth_on, dtok,
         mono_on, mres, fps) in itertools.product(
            a.color_res, a.isp_scale, a.color_encode,
            a.mjpeg_quality, a.video_bitrate_kbps, a.depth, a.depth_size,
            a.mono, a.mono_res, a.fps):
        if enc not in ("h264", "h265"):
            bitrate = a.video_bitrate_kbps[0]
        if enc != "mjpeg":
            quality = a.mjpeg_quality[0]
        if not depth_on:
            dtok = a.depth_size[0]
        c = Combo(a, cres, isp, enc, quality, bitrate, depth_on, dtok,
                  mono_on, mres, fps)
        # Deduplicate on the RESOLVED depth geometry, not on the policy token
        # that asked for it. "match-color" and "derived" name different requests
        # but land on the same pixel grid whenever the colour output happens to
        # be the mono width — recording that twice would spend a cell of camera
        # time on a byte-identical pipeline and produce two rows a reader would
        # then try to explain the difference between.
        try:
            key = c.key() + (c.cfg.resolve().depth_out_size if depth_on else None,)
        except ValueError:
            # An illegal combination keeps the token in its key so it survives
            # to be reported as REJECTED rather than silently folded away.
            key = c.key() + ("unresolved", dtok)
        if key in seen:
            continue
        seen.add(key)
        combos.append(c)
    return combos


# =============================================================================
# recording one cell
# =============================================================================
def _count_list(path: Path) -> int:
    """Data lines in a TUM index file — comments do not count as frames."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))


def _count_files(path: Path) -> int:
    """Images actually on disk, as opposed to images the index promises."""
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file())


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _extracted_png_gb(rc, frames: int) -> float:
    """Disk the H.26x -> PNG extraction will need, including its temp copy.

    estimate_rate bills an encoded cell at its bitstream size, which is the wire
    cost and roughly 1/30th of what extraction then writes: ffmpeg decodes every
    frame to a lossless PNG. Extraction also stages the whole set in a temp dir
    inside the dataset before renaming, so the peak is about twice the result.
    """
    w, h = rc.color_out_size
    return 2.0 * frames * w * h * 3 * 0.55 / 1e9


def _slam_config(writer: DatasetWriter, src: C3Source, combo: Combo,
                 rc, verbose: bool = True) -> tuple[str, str]:
    """Write the ORB-SLAM3 config the recorded streams can actually satisfy.

    The RGB-D yaml states that the colour intrinsics describe the depth image
    too. That is only true when depth was warped into the colour camera AND
    emitted on the colour grid, so it is written only then; any other cell gets
    the monocular config plus a recorded reason. Shipping an RGB-D yaml with a
    depth map of a different size is worse than shipping none: every depth
    lookup is then off by the resolution ratio, silently, in a file that claims
    to be ready to run.
    """
    intr = src.intrinsics.get(STREAM_COLOR)
    if intr is None:
        return "", "no colour intrinsics on the device"

    if combo.depth_on and combo.cfg.depth_align == "color" and \
            rc.depth_out_size == rc.color_out_size:
        try:
            # getBaselineDistance() is centimetres; ORB-SLAM3 wants millimetres.
            baseline_mm = src.device.readCalibration().getBaselineDistance() * 10.0
        except Exception:  # noqa: BLE001
            baseline_mm = None
        writer.write_orbslam3_yaml(intr, baseline_mm, combo.fps)
        return "orbslam3_rgbd.yaml", ""

    if combo.depth_on:
        reason = (f"depth {rc.depth_out_size[0]}x{rc.depth_out_size[1]} is not on "
                  f"the colour grid {rc.color_out_size[0]}x{rc.color_out_size[1]}"
                  f" (align={combo.cfg.depth_align}); the colour intrinsics do "
                  f"not describe it, so no RGB-D config is written. depth/ is "
                  f"still recorded and still valid at its own size.")
    else:
        reason = "no depth stream in this configuration"
    writer.write_orbslam3_mono_yaml(intr, combo.fps)
    if verbose:
        print(f"    SLAM config: monocular — {reason}")
    return "orbslam3_mono.yaml", reason


def _finalize_metadata(writer: DatasetWriter, src: C3Source, combo: Combo, rc,
                       mav, a: argparse.Namespace, elapsed: float,
                       slam_config: str, slam_reason: str, n: int,
                       total: int, mav_msgs: int = 0) -> None:
    """metadata.txt/json, with the sweep coordinates that name this cell.

    Recorded verbatim in `sweep`: without it a directory full of datasets is a
    set of recordings nobody can attribute to a setting six weeks later, which
    is the failure the whole script exists to avoid.
    """
    cs = src.metrics.streams.get(STREAM_COLOR)
    cfg = combo.cfg
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
        "mjpeg_quality": (cfg.mjpeg_quality if cfg.color_encode == "mjpeg"
                          else None),
        "mono_source": cfg.mono_source,
        "depth_align": cfg.depth_align,
        "depth_scale": DEPTH_SCALE,
        "streams": list(cfg.streams),
        "sweep": {
            "tool": "c3_option_sweep.py",
            "cell": f"{n}/{total}",
            "label": combo.label(),
            "axes": {
                "color_res": combo.color_res,
                "isp_scale": combo.isp_label,
                "color_encode": combo.encode,
                "mjpeg_quality": (combo.quality if combo.encode == "mjpeg"
                                  else None),
                "video_bitrate_kbps": (combo.bitrate if combo.encode in
                                       ("h264", "h265") else None),
                "depth": "on" if combo.depth_on else "off",
                "depth_size": combo.depth_tok if combo.depth_on else None,
                "mono": "on" if combo.mono_on else "off",
                "mono_res": combo.mono_res,
                "fps": combo.fps,
            },
            "slam_config": slam_config,
            "slam_config_note": slam_reason or "RGB-D: depth is on the colour grid",
            "warmup_s": a.warmup,
            "requested_duration_s": a.duration,
        },
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
            "extrinsics_note": (src.imu_extrinsics or {}).get("note", "not queried"),
            "samples": src.imu_stats.count,
        },
        "mavlink": ({**mav.summary(),
                     "records_this_dataset": mav_msgs,
                     "counts_note": "message_counts/achieved_rates_hz come from "
                                    "the one MavlinkLogger that serves the whole "
                                    "sweep and are therefore CUMULATIVE across "
                                    "cells; records_this_dataset is the count "
                                    "actually written into this directory"}
                    if mav is not None else {"transport": "none"}),
        "duration_s": round(elapsed, 1),
        "frames_written": writer.stats.written,
        "frames_dropped": writer.stats.dropped,
        "writer": writer.summary(),
        "camera_metrics": src.metrics.summary(),
    }
    if a.blueos_info:
        meta.update(blueos_info())
    writer.write_metadata(meta, notes=a.notes)


def _extract_encoded(combo: Combo, out_dir: Path) -> str:
    """Turn rgb.h264/rgb.h265 into the timestamped PNGs the TUM lists name.

    Without this an encoded cell writes an index full of `rgb/<t>.png` paths that
    do not exist, i.e. a dataset that looks complete and is not runnable. So it
    is on by default here, unlike in c3_collect.py where the operator may want
    the raw stream and nothing else.
    """
    from c3_camera.c3_extract_rgb_video import extract_rgb_video

    print(f"    extracting {combo.encode.upper()} RGB into timestamped PNGs ...")
    try:
        summary = extract_rgb_video(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"    RGB extraction FAILED: {type(e).__name__}: {e}")
        print(f"    the raw stream is intact; retry with: python "
              f"c3_camera/c3_extract_rgb_video.py {out_dir}")
        return f"extract failed: {type(e).__name__}"
    print(f"    extracted {summary['frames_decoded']} RGB frames")
    return ""


def run_one(combo: Combo, out_dir: Path, a: argparse.Namespace, mav,
            n: int, total: int) -> dict:
    """Connect, settle, record one dataset, and return its CSV row."""
    row = {f: "" for f in FIELDS}
    row.update({
        "n": n,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ok": 0,
        "dataset": out_dir.name,
        "label": combo.label(),
        "color_res": combo.color_res,
        "isp_scale": combo.isp_label,
        "encode": combo.encode,
        "mjpeg_quality": combo.quality if combo.encode == "mjpeg" else "",
        "video_bitrate_kbps": (combo.bitrate if combo.encode in ("h264", "h265")
                               else ""),
        "depth": "on" if combo.depth_on else "off",
        "depth_size_req": combo.depth_tok if combo.depth_on else "",
        "mono": "on" if combo.mono_on else "off",
        "mono_res": combo.mono_res,
        "requested_fps": combo.fps,
        "streams": "+".join(combo.cfg.streams),
    })

    t_wall = time.monotonic()
    writer: DatasetWriter | None = None
    src: C3Source | None = None
    rc = None
    skew_sum, skew_n = 0.0, 0
    mav_msgs = 0
    elapsed = 0.0
    t_rec: float | None = None
    recorded = False
    # Hoisted so the error path passes the real values rather than "", which
    # _finalize_metadata renders as "RGB-D was satisfied" — a claim no
    # half-finished cell has earned.
    slam_config, slam_reason = "", "not determined; the cell did not complete"
    try:
        # resolve() inside the try: the axes are free-form strings, so a rejected
        # combination must cost one failed row rather than abort a sweep that is
        # holding the camera.
        rc = combo.cfg.resolve()
        row["color_out"] = f"{rc.color_out_size[0]}x{rc.color_out_size[1]}"
        row["depth_out"] = (f"{rc.depth_out_size[0]}x{rc.depth_out_size[1]}"
                            if combo.depth_on else "")
        row["depth_on_color_grid"] = (
            int(combo.depth_on and combo.cfg.depth_align == "color"
                and rc.depth_out_size == rc.color_out_size))
        est = estimate_rate(rc.color_out_size, rc.depth_out_size, rc.mono_size,
                            combo.fps, combo.cfg.streams, combo.cfg.color_wire,
                            combo.cfg.color_encode, combo.cfg.video_bitrate_kbps)
        row["est_wire_mbps"] = round(est["wire_mbit_s"], 1)
        row["est_disk_mb_min"] = round(est["disk_mb_min"], 1)

        # A cell must never record into a directory that already holds a
        # recording. DatasetWriter re-opens the index files in "w" mode but
        # leaves rgb/ and depth/ alone, so a re-run into the same --out silently
        # merges two capture sessions: the indices describe only the new frames
        # while the old images stay on disk, orphaned but still counted in the
        # disk-cost column this sweep exists to measure.
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(
                f"{out_dir} already holds a recording. Use a fresh --out, or "
                f"move that directory aside; merging two sessions into one "
                f"dataset would leave orphaned images and a misleading size.")

        interframe = combo.encode in ("h264", "h265")
        # keep_raw lets MJPEG colour reach disk as the JPEG that came off the
        # wire, with no decode/re-encode round trip, and is how the encoded
        # formats reach the writer at all.
        with C3Source(combo.cfg, verbose=not a.quiet, strict=a.strict,
                      keep_raw=combo.encode != "none") as source:
            src = source
            t_start = time.monotonic()

            # ------------------------------------------------------ warm up
            # Discarded, and discarded BEFORE the writer exists, so the dataset
            # never contains the auto-exposure ramp. For an encoded stream the
            # queue is blocking, so it must still be drained here or the device
            # stalls behind a host that is not reading.
            deadline = t_start + max(0.0, a.warmup)
            while time.monotonic() < deadline:
                source.read(timeout_ms=2000)
                if interframe:
                    source.drain_encoded_rgb()
                source.drain_imu()
            source.metrics = type(source.metrics)(combo.cfg.streams)
            # The IMU counter has to be reset with it. ImuStats.count is a
            # lifetime total and drain_imu() feeds it during warm-up, so leaving
            # it alone made imu_samples include samples that were deliberately
            # thrown away — a per-cell IMU yield that no dataset contains.
            source.imu_stats = type(source.imu_stats)()
            if mav is not None:
                # Records buffered while the previous cell was closing belong to
                # the previous cell, not this one. Drop them rather than write
                # telemetry that predates the first frame.
                mav.drain()

            # ------------------------------------------------------- record
            writer = DatasetWriter(
                out_dir, mode="research",
                threads=a.writer_threads, queue_size=a.writer_queue,
                png_compression=a.png_compression, mp4_fps=combo.fps,
                write_mp4=not a.no_mp4 and not interframe,
                color_encode=combo.encode, verbose=not a.quiet)
            writer.write_calibration(calibration_payload(source))
            slam_config, slam_reason = _slam_config(writer, source, combo, rc,
                                                    verbose=not a.quiet)
            row["slam_config"] = slam_config

            t_rec = time.monotonic()
            rec_deadline = t_rec + a.duration
            last_print = 0.0
            next_disk_check = 0.0
            while time.monotonic() < rec_deadline:
                bundle = source.read(timeout_ms=2000)

                # Drained BEFORE the `bundle is None` test on purpose: in
                # pair_mode="timestamp" read() returns None whenever colour and
                # depth are further apart than the tolerance, but poll() has
                # already consumed those messages — so an early `continue` here
                # would silently discard encoded packets and IMU samples.
                encoded = source.drain_encoded_rgb()
                if encoded:
                    writer.encoded_rgb_frames(encoded)
                imu = source.drain_imu()
                if imu:
                    writer.imu_camera_rows(imu, IMU_FIELDS)
                if mav is not None:
                    recs = mav.drain()
                    if recs:
                        writer.mavlink_rows(recs)
                        mav_msgs += len(recs)

                if bundle is None:
                    continue
                source.metrics.mark_loop()
                writer.submit(bundle)
                if bundle.skew_ms == bundle.skew_ms:
                    skew_sum += bundle.skew_ms
                    skew_n += 1

                now = time.monotonic()
                # Wall-clock, not `written % 40`: stats.written is incremented by
                # three background writer threads, so an equality test sampled
                # from this loop skips multiples whenever completions arrive in
                # bursts — which is exactly what a thread pool does — and the
                # guard then never fires at all.
                if now >= next_disk_check:
                    next_disk_check = now + 2.0
                    if shutil.disk_usage(out_dir).free / 1e9 < a.min_free_gb:
                        print(f"    free disk below {a.min_free_gb:g} GB — "
                              f"ending this cell early")
                        break

                if a.print_every and now - last_print >= a.print_every:
                    last_print = now
                    print(f"    [{now - t_rec:5.1f}s] {writer.hud_line()}")

            elapsed = time.monotonic() - t_rec
            row["rec_seconds"] = round(elapsed, 1)

            # Read the counters while the source is still open, but do NOT do the
            # post-roll here: writer.close() flushes for up to 120 s and the
            # extraction shells out to ffmpeg for longer, and for an encoded
            # colour stream the device queue is BLOCKING — holding the device
            # open through minutes of host work stalls it behind a host that has
            # stopped reading. Everything after this point needs only out_dir.
            for tag, name in (("color", STREAM_COLOR), ("depth", STREAM_DEPTH)):
                st = source.metrics.streams.get(name)
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
                # summary()'s mbps_measured covers only the last 120 frames — the
                # tail of the run, which is the wrong span for a bandwidth
                # comparison across cells. total_bytes is lifetime and is not in
                # summary(), so the whole-capture average is computed here.
                row[f"{tag}_mbps_run"] = (round(st.total_bytes * 8 / elapsed / 1e6, 1)
                                          if elapsed > 0 else "")
            row["measured_mbps"] = round(source.metrics.mbps_total, 1)
            row["loop_hz"] = round(source.metrics.loop_hz, 2)
            row["skew_ms_mean"] = round(skew_sum / skew_n, 2) if skew_n else ""
            row["imu_samples"] = source.imu_stats.count
            row["imu_hz"] = round(source.imu_stats.hz, 1)
            # Counted here, not from mav.stats: one logger serves the whole sweep
            # and its counters are never reset, so mav.stats would report cell 10
            # as the running total of cells 1-10.
            row["mav_msgs"] = mav_msgs
            recorded = True

        # ------------------------------------------ post-roll, device released
        writer.close()
        _finalize_metadata(writer, src, combo, rc, mav, a, elapsed,
                           slam_config, slam_reason, n, total, mav_msgs)
        if interframe and not a.no_extract:
            # The extracted PNGs are ~30x the bitstream they come from, and
            # nothing has checked the disk since the recording ended.
            free_gb = shutil.disk_usage(out_dir).free / 1e9
            need_gb = _extracted_png_gb(rc, writer.stats.written)
            if free_gb - need_gb < a.min_free_gb:
                row["error"] = (f"skipped RGB extraction: needs ~{need_gb:.1f} GB "
                                f"and only {free_gb:.1f} GB is free")
                print(f"    {row['error']}; rgb.{combo.encode} is intact, "
                      f"extract it later with c3_extract_rgb_video.py")
            else:
                err = _extract_encoded(combo, out_dir)
                if err:
                    row["error"] = err

    except KeyboardInterrupt:
        row["error"] = "interrupted"
    except D.PipelineConfigError as e:
        row["error"] = f"PipelineConfigError: {e}".replace("\n", " ")[:300]
    except D.DeviceBusyError as e:
        row["error"] = f"DeviceBusyError: {e}".replace("\n", " ")[:300]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
    finally:
        # Close the writer even when we got here via Ctrl-C or an exception: a
        # half-written dataset with a truncated index is worse than a short one,
        # and close() is what flushes the reorder buffer into the TUM lists.
        # close() is idempotent, so the happy path having closed it already
        # costs nothing here — but it can still raise (ENOSPC on the final
        # flush), and an exception thrown from a finally would abort the whole
        # sweep from inside the one block whose job is to salvage a cell.
        if writer is not None:
            try:
                writer.close()
            except Exception as e:  # noqa: BLE001
                print(f"    writer close failed: {type(e).__name__}: {e}")
                row["error"] = row["error"] or f"close: {type(e).__name__}: {e}"
            # Gate on the artefact, not on writer._closed: close() sets that flag
            # as its FIRST statement and only then spends up to 120 s flushing,
            # so a Ctrl-C inside that window leaves _closed True with metadata
            # never written — and without metadata.json's depth_scale key,
            # c3_dataset_check rejects the whole directory.
            if src is not None and rc is not None and \
                    not (out_dir / "metadata.json").exists():
                try:
                    _finalize_metadata(
                        writer, src, combo, rc, mav, a,
                        # Never t_wall: that clock starts before the PoE boot, so
                        # it would bill ~14 s of firmware upload as capture time.
                        elapsed or (time.monotonic() - t_rec if t_rec else 0.0),
                        slam_config, slam_reason, n, total, mav_msgs)
                except Exception as e:  # noqa: BLE001
                    print(f"    could not write metadata: {type(e).__name__}: {e}")

    if writer is not None:
        s = writer.summary()
        row["frames_written"] = s["frames"]
        row["frames_dropped_writer"] = s["dropped"]
        row["writer_drop_rate"] = s["drop_rate"]
        row["images"] = s["images"]
        # Guarded by `writer is not None`, i.e. by this call having actually
        # opened the directory. Measuring out_dir unconditionally credited a cell
        # that never connected with whatever a previous run had left there.
        mb = _dir_bytes(out_dir) / 1e6
        row["dataset_mb"] = round(mb, 1)
        secs = float(row["rec_seconds"] or 0.0)
        row["dataset_mb_per_min"] = round(mb * 60 / secs, 1) if secs > 0 else ""
        # *_index counts INDEX LINES; *_on_disk counts files. They diverge
        # exactly when it matters — an encoded cell whose extraction failed has
        # a full index and an empty rgb/ — so both are recorded.
        row["rgb_index"] = _count_list(out_dir / "rgb.txt")
        row["depth_index"] = _count_list(out_dir / "depth.txt")
        row["assoc_lines"] = _count_list(out_dir / "associations.txt")
        row["rgb_on_disk"] = _count_files(out_dir / "rgb")
        row["depth_on_disk"] = _count_files(out_dir / "depth")
        if row["rgb_index"] and row["rgb_on_disk"] < row["rgb_index"]:
            row["error"] = row["error"] or (
                f"rgb.txt names {row['rgb_index']} images but rgb/ holds "
                f"{row['rgb_on_disk']}; SLAM cannot open this dataset")

    # A cell that recorded but then failed a post-roll step is NOT a success:
    # its results.csv verdict, the summary table and the final ranking all read
    # `ok`, and an un-runnable dataset marked ok=1 is the one outcome an
    # operator has no way to notice.
    row["ok"] = int(recorded and not row["error"])
    row["wall_s"] = round(time.monotonic() - t_wall, 1)
    return row


def check_dataset(out_dir: Path, combo: Combo, sample: int) -> str:
    """Run the existing verifier on what was just written. Never fatal.

    c3_dataset_check.py is an RGB-D checker by construction, and a sweep
    deliberately records cells that are not RGB-D. It still runs on all of them —
    calibration.json, metadata's depth_scale, the TUM index, timestamp
    monotonicity and the clock overlap are all depth-independent and all worth
    having — but its verdict is qualified by what the cell actually is:

    * `--depth off` — its two depth-specific hard failures (empty depth.txt,
      empty associations.txt) are exactly what this configuration is supposed to
      produce. They are subtracted, and the remaining problems decide a
      `pass-mono` / `problems-mono` verdict. Skipping the check entirely would
      have left monocular cells with no validation at all.
    * depth on a grid other than the colour grid — the resolution mismatch it
      flags IS the setting under test, so the verdict carries `-offgrid`.

    The report itself goes to <dataset>/check.txt rather than the console: it is
    ~40 lines per cell, and twelve of those would bury the sweep's own output.
    """
    import contextlib
    import io

    from c3_camera import c3_dataset_check as DC

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = DC.main([str(out_dir), "--sample", str(sample)])
    except SystemExit as e:          # argparse inside main()
        code = int(e.code or 0)
    except Exception as e:           # noqa: BLE001
        (out_dir / "check.txt").write_text(buf.getvalue())
        return f"error:{type(e).__name__}"
    report = buf.getvalue()
    (out_dir / "check.txt").write_text(report)

    fails = [l for l in report.splitlines() if l.strip().startswith("FAIL")]
    if not combo.depth_on:
        expected = ("depth.txt missing or empty",
                    "associations.txt missing or empty")
        fails = [l for l in fails if not any(e in l for e in expected)]
        return "pass-mono" if not fails else "problems-mono"
    verdict = "pass" if code == 0 else "problems"
    return verdict if _on_colour_grid(combo, out_dir) else f"{verdict}-offgrid"


def _on_colour_grid(combo: Combo, out_dir: Path) -> bool:
    """Whether this cell put depth on the colour grid.

    Read back from the metadata that was actually written rather than recomputed
    from the axes, so the verdict describes the recording rather than the
    intention. Falls back to the axes only if metadata is unreadable.
    """
    try:
        meta = json.loads((out_dir / "metadata.json").read_text())
        return (list(meta["resolved"]["depth_out"]) ==
                list(meta["resolved"]["color_out"])
                and meta.get("depth_align") == "color"
                and STREAM_DEPTH in meta.get("streams", []))
    except Exception:  # noqa: BLE001
        return (combo.depth_on and combo.depth_tok == "match-color"
                and combo.cfg.depth_align == "color")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    C.add_connection_args(p)

    g = p.add_argument_group("sweep axes (comma-separated; every combination is run)")
    g.add_argument("--color-res", type=_choice_list("--color-res",
                                                    C.COLOR_RESOLUTIONS),
                   default=["1080p"],
                   help=f"IMX378 sensor mode(s) from "
                        f"{sorted(C.COLOR_RESOLUTIONS)} (default: 1080p)")
    g.add_argument("--isp-scale", type=_parse_isp_list, default=[(1, 4), (1, 2)],
                   metavar="N/D[,N/D...]",
                   help="ISP downscale(s), or 'none'. This and --color-res "
                        "together are the colour-resolution axis "
                        "(default: 1/4,1/2 = 480x270 and 960x540 at 1080p)")
    g.add_argument("--color-encode",
                   type=_choice_list("--color-encode",
                                     ("none", "mjpeg", "h264", "h265")),
                   default=["none", "mjpeg"],
                   help="the compression axis: none,mjpeg,h264,h265 "
                        "(default: none,mjpeg — raw versus device MJPEG is the "
                        "single most informative comparison)")
    g.add_argument("--mjpeg-quality", type=_parse_int_list, default=[90],
                   metavar="Q[,Q...]", help="MJPEG quality 0-100 (default: 90)")
    g.add_argument("--video-bitrate-kbps", type=_parse_int_list,
                   default=[C.DEFAULT_VIDEO_BITRATE_KBPS], metavar="KBPS[,...]",
                   help=f"H.264/H.265 target bitrate(s) "
                        f"(default: {C.DEFAULT_VIDEO_BITRATE_KBPS})")
    g.add_argument("--depth", type=_parse_onoff, default=[True, False],
                   metavar="on|off[,on|off]",
                   help="ship the depth stream, or not (default: on,off)")
    g.add_argument("--depth-size", type=_parse_depth_sizes,
                   default=["match-color", "derived"],
                   metavar="match-color|derived|WxH[,...]",
                   help="the depth-resolution axis. 'match-color' puts depth on "
                        "exactly the colour grid, which is what RGB-D SLAM needs "
                        "(rgb[u,v] and depth[u,v] the same ray); 'derived' is the "
                        "cheaper mono-width default; or give explicit sizes "
                        "(default: match-color,derived). Ignored where "
                        "--depth off")
    g.add_argument("--mono", type=_parse_onoff, default=[False],
                   metavar="on|off[,on|off]",
                   help="also record the raw left/right pair (default: off). A "
                        "raw 640x400 pair is ~82 Mbit/s at 20 fps against a "
                        "~90 Mbit/s link, so turning it on changes what every "
                        "other axis can reach")
    g.add_argument("--mono-res", type=_choice_list("--mono-res",
                                                   C.MONO_RESOLUTIONS),
                   default=["400p"],
                   help=f"OV9282 sensor mode(s) from {sorted(C.MONO_RESOLUTIONS)} "
                        f"(default: 400p)")
    g.add_argument("--fps", type=_parse_float_list, default=[20.0],
                   metavar="N[,N...]", help="frame rate(s) (default: 20)")

    g = p.add_argument_group("fixed settings (identical in every cell)")
    g.add_argument("--depth-align", default="color",
                   choices=("color", "right", "left", "center", "none"),
                   help="(default: %(default)s — pixel-aligned RGB-D)")
    g.add_argument("--depth-preset", default="robotics",
                   choices=sorted(C.DEPTH_PRESETS))
    g.add_argument("--median", default="5x5", choices=sorted(C.MEDIAN_FILTERS))
    g.add_argument("--confidence", type=int, default=None, metavar="0-255")
    g.add_argument("--subpixel", action="store_true")
    g.add_argument("--extended", action="store_true")
    g.add_argument("--no-lr-check", dest="lr_check", action="store_false",
                   default=True)
    g.add_argument("--color-wire", default="nv12", choices=("nv12", "bgr"),
                   help="wire format when NOT encoding (default: %(default)s)")
    g.add_argument("--mono-encode", default="none", choices=("none", "mjpeg"),
                   help="encode left/right on the device (default: %(default)s). "
                        "Lossy: left/right PNGs are then re-encoded from JPEG")
    g.add_argument("--mono-quality", type=int, default=97,
                   help="mono MJPEG quality (default: %(default)s; below ~q95 "
                        "JPEG blocking starts costing stereo matches)")
    g.add_argument("--video-keyframe-frequency", type=int, default=0,
                   help="H.264/H.265 keyframe interval in frames; 0 = one per "
                        "second (default: %(default)s)")
    g.add_argument("--queue-size", type=int, default=2,
                   help="host output queue depth (default: %(default)s)")

    g = p.add_argument_group("run control")
    g.add_argument("--duration", type=float, default=20.0,
                   help="RECORDED seconds per cell (default: %(default)s)")
    g.add_argument("--warmup", type=float, default=3.0,
                   help="discarded seconds per cell, before the writer opens, so "
                        "the dataset does not contain the auto-exposure ramp "
                        "(default: %(default)s)")
    g.add_argument("--settle", type=float, default=5.0,
                   help="pause between cells, for the PoE device to reset "
                        "(default: %(default)s)")
    g.add_argument("--out", type=Path, default=None,
                   help="sweep directory (default: "
                        "c3_camera/sweeps/sweep_<YYYYMMDD_HHMMSS>)")
    g.add_argument("--append", action="store_true",
                   help="append to an existing results.csv in --out; refused if "
                        "its column set differs")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve every cell, print the wire/disk budget, and exit "
                        "without connecting to anything")
    g.add_argument("--yes", action="store_true",
                   help="do not ask before spending the estimated time and disk")
    g.add_argument("--strict", action="store_true",
                   help="fail a cell whose config the device does not support "
                        "instead of recording the downgraded version")
    g.add_argument("--min-free-gb", type=float, default=10.0,
                   help="end a cell early if free disk falls below this "
                        "(default: %(default)s)")
    g.add_argument("--no-check", dest="check", action="store_false", default=True,
                   help="skip c3_dataset_check.py on each finished dataset. It "
                        "normally runs on every depth-on cell and its full "
                        "report is written to <dataset>/check.txt")
    g.add_argument("--check-sample", type=int, default=4,
                   help="image pairs c3_dataset_check opens per dataset "
                        "(default: %(default)s)")
    g.add_argument("--no-extract", action="store_true",
                   help="leave H.264/H.265 colour as rgb.h26x instead of "
                        "decoding it into the PNGs the TUM lists name (which "
                        "leaves that cell's dataset un-runnable until you do)")
    g.add_argument("--no-mp4", action="store_true",
                   help="skip the rgb.mp4 preview in each dataset")
    g.add_argument("--notes", default="",
                   help="free-text session notes, copied into every metadata.txt")
    g.add_argument("--print-every", type=float, default=5.0,
                   help="seconds between progress lines while recording "
                        "(default: %(default)s)")
    g.add_argument("--quiet", action="store_true", help="less per-cell chatter")

    g = p.add_argument_group("writers")
    g.add_argument("--writer-threads", type=int, default=3)
    g.add_argument("--writer-queue", type=int, default=12)
    g.add_argument("--png-compression", type=int, default=1,
                   help="0-9, lossless at every level (default: %(default)s)")

    g = p.add_argument_group("camera IMU (BNO086, on the camera board)")
    g.add_argument("--imu-rate", type=int, default=200)
    g.add_argument("--imu-batch", type=int, default=10,
                   help="IMU reports per XLink message; 10 is what keeps samples "
                        "from being lost when the image streams are busy "
                        "(default: %(default)s)")
    g.add_argument("--no-camera-imu", action="store_true",
                   help="do not record the camera IMU at all")

    g = p.add_argument_group("MAVLink telemetry (one connection for the whole sweep)")
    g.add_argument("--mavlink-transport", default="udp",
                   choices=("udp", "rest", "none"),
                   help="(default: %(default)s; 'none' if no vehicle is on the net)")
    g.add_argument("--mavlink", default="udpin:0.0.0.0:14551",
                   help="pymavlink connection string (default: %(default)s)")
    g.add_argument("--mavlink-rest-url", default="http://192.168.2.2:6040")
    g.add_argument("--mavlink-wait", type=float, default=6.0)
    g.add_argument("--blueos-info", action="store_true",
                   help="query BlueOS for the ArduSub version per cell (adds a "
                        "few seconds each; off by default in a sweep)")

    PF.add_preflight_args(p)
    return p.parse_args(argv)


def _can_prompt() -> bool:
    """Whether asking a question on stdin will be answered rather than punish us.

    `isatty()` alone is not enough. `nohup ./c3_option_sweep.py &` leaves the
    controlling terminal on fd 0 while putting the process in a background
    process group, and a background read from the terminal raises SIGTTIN, whose
    default action STOPS the process — so an unattended sweep would hang before
    connecting, with no output explaining why.
    """
    try:
        return (sys.stdin.isatty() and
                os.getpgrp() == os.tcgetpgrp(sys.stdin.fileno()))
    except (OSError, AttributeError):
        return False


def _cell_cost(rc, c: Combo, a: argparse.Namespace) -> tuple[float, float]:
    """(Mbit/s on the wire, MB on disk) for one cell of the planned length.

    The wire figure comes from ResolvedConfig.bandwidth_mbps() rather than
    dataset.estimate_rate(): only the former models --mono-encode, and without
    it a feasible mono-MJPEG cell (23 Mbit/s at q90) is planned as raw GRAY8
    (82 Mbit/s) and flagged OVER budget.

    The disk figure corrects estimate_rate the other way. It bills an encoded
    colour stream at its bitstream size, which is right for the wire and roughly
    1/30th of what lands on disk once the mandatory PNG extraction has run.
    """
    est = estimate_rate(rc.color_out_size, rc.depth_out_size, rc.mono_size,
                        c.fps, c.cfg.streams, c.cfg.color_wire,
                        c.cfg.color_encode, c.cfg.video_bitrate_kbps)
    wire = rc.bandwidth_mbps()["total"]
    mb = est["disk_mb_min"] * a.duration / 60.0
    if c.encode in ("h264", "h265") and not a.no_extract:
        w, h = rc.color_out_size
        png_mb = w * h * 3 * 0.55 * c.fps * a.duration / 1e6
        bitstream_mb = c.bitrate * 1000 / 8 * a.duration / 1e6
        mb += png_mb - bitstream_mb
    return wire, mb


def print_plan(combos: list[Combo], a: argparse.Namespace) -> tuple[float, float]:
    """The table that makes an over-budget sweep visible before it is started."""
    print(f"{'#':>3} {'configuration':<48} {'color':>10} {'depth':>10} "
          f"{'wire':>7} {'%link':>6} {'disk':>8}")
    print(f"{'':>3} {'':<48} {'':>10} {'':>10} {'Mbit/s':>6} {'':>7} {'MB/cell':>8}")
    print("-" * 100)
    disk_total = 0.0
    over = 0
    # Every cell of a sweep shares the same sensor geometry, so a geometry
    # warning is the same sentence twelve times over. Print each distinct one
    # once, under the table, where it is still read but no longer buries the
    # numbers the table exists to show.
    warned: dict[str, int] = {}
    for i, c in enumerate(combos, 1):
        try:
            rc = c.cfg.resolve()
        except ValueError as e:
            print(f"{i:>3} {c.label()[:48]:<48} REJECTED: {e}")
            continue
        wire, mb = _cell_cost(rc, c, a)
        disk_total += mb
        frac = wire / C.POE_BUDGET_MBPS
        flag = "" if frac < 0.85 else ("  <- tight" if frac < 1.0 else "  <- OVER")
        if frac >= 1.0:
            over += 1
        color = f"{rc.color_out_size[0]}x{rc.color_out_size[1]}"
        depth = (f"{rc.depth_out_size[0]}x{rc.depth_out_size[1]}"
                 if c.depth_on else "-")
        print(f"{i:>3} {c.label()[:48]:<48} {color:>10} {depth:>10} "
              f"{wire:>7.0f} {frac*100:>5.0f}% {mb:>8.0f}{flag}")
        for w in rc.warnings:
            warned.setdefault(w, i)
    seconds = len(combos) * (a.duration + a.warmup + a.settle + 14)
    print("-" * 100)
    for w, first in sorted(warned.items(), key=lambda kv: kv[1]):
        print(f"  warning (first at #{first}): {w}")
    print(f"  {len(combos)} configurations, ~{seconds/60:.0f} min, "
          f"~{disk_total/1000:.1f} GB of datasets "
          f"(PoE ceiling on this path ~{C.POE_BUDGET_MBPS:.0f} Mbit/s)")
    if over:
        print(f"  {over} of {len(combos)} cells are estimated ABOVE that ceiling. "
              f"They still run — a link-limited cell delivers fewer fps than "
              f"requested, and that delivered number is the measurement. Drop "
              f"them with a narrower --isp-scale / --color-encode / --depth list "
              f"if you only want feasible settings.")
    return seconds, disk_total


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 132)
    print(f"{'#':>3} {'configuration':<44} {'c.fps':>6} {'c.p50':>7} {'d.fps':>6} "
          f"{'drop%':>6} {'meas':>6} {'frames':>7} {'MB':>7} {'rgb':>5} "
          f"{'depth':>6} {'assoc':>6} {'slam':>6} {'chk':>13}")
    print(f"{'':>3} {'':<44} {'':>6} {'ms':>7} {'':>6} {'':>6} {'Mbit/s':>6} "
          f"{'':>7} {'':>7} {'on disk':>5} {'on disk':>6} {'':>6} {'':>6} {'':>13}")
    print("-" * 132)
    for r in rows:
        if not r["frames_written"]:
            print(f"{r['n']:>3} {r['label'][:44]:<44} FAILED: {str(r['error'])[:60]}")
            continue
        slam = ("rgbd" if str(r["slam_config"]).endswith("rgbd.yaml")
                else ("mono" if r["slam_config"] else "-"))
        print(f"{r['n']:>3} {r['label'][:44]:<44} {r['color_fps'] or 0:>6} "
              f"{r['color_lat_p50'] or 0:>7} {r['depth_fps'] or '-':>6} "
              f"{float(r['color_drop_rate'] or 0)*100:>6.1f} "
              f"{r['measured_mbps'] or 0:>6} {r['frames_written'] or 0:>7} "
              f"{r['dataset_mb'] or 0:>7} {r['rgb_on_disk'] or 0:>5} "
              f"{r['depth_on_disk'] or 0:>6} {r['assoc_lines'] or 0:>6} "
              f"{slam:>6} {str(r['check'] or '-'):>13}")
        # A cell that recorded and then failed a post-roll step used to print as
        # a clean success with its error only in the CSV. Say it on the row.
        if not r["ok"]:
            print(f"{'':>3}   NOT USABLE: {str(r['error'])[:100]}")
    print("=" * 132)


def main(argv=None) -> int:
    a = parse_args(argv)
    combos = build_combos(a)

    print("=" * 96)
    print(f"C3 option sweep — {len(combos)} configurations, one SLAM dataset each")
    print("=" * 96)
    seconds, disk_mb = print_plan(combos, a)

    if a.dry_run:
        print("\ndry run — nothing connected, nothing written.")
        return 0

    out_root = a.out or (Path("c3_camera/sweeps") /
                         f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / 1e9
    print(f"  output : {out_root}   (free disk {free_gb:.1f} GB)")
    if disk_mb / 1000 > free_gb - a.min_free_gb:
        print(f"  WARNING: the estimate ({disk_mb/1000:.1f} GB) does not fit "
              f"inside the free space minus --min-free-gb; cells will end early.")
    # Before the prompt, not after: the operator is about to commit an hour and
    # tens of gigabytes, and "the camera is owned" is something they should be
    # able to read while deciding, rather than discover in cell 1 of 24.
    print()
    rc_pf = PF.gate(a, title="c3 option sweep", out_dir=out_root,
                    min_free_gb=a.min_free_gb)
    if rc_pf is not None:
        return rc_pf
    print()

    if not a.yes and _can_prompt():
        try:
            if input(f"  proceed? ~{seconds/60:.0f} min, "
                     f"~{disk_mb/1000:.1f} GB [y/N] ").strip().lower() != "y":
                print("  aborted.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n  aborted.")
            return 0

    # ------------------------------------------------------------------- CSV
    # Checked BEFORE anything connects: a refusal here must cost nothing, and a
    # sweep that discovers it an hour in has already taken the camera and
    # written datasets whose index row cannot be recorded.
    csv_path = out_root / "results.csv"
    appending = a.append and csv_path.exists() and csv_path.stat().st_size > 0
    if appending:
        # DictWriter emits values in FIELDS order under whatever header is
        # already in the file, so a CSV written before a column was added would
        # file every later value one place to the left — silently. Refuse.
        with csv_path.open(newline="") as probe:
            existing = next(csv.reader(probe), [])
        if existing != list(FIELDS):
            print(f"error: {csv_path} has {len(existing)} columns and this build "
                  f"writes {len(FIELDS)}; appending would misalign every row. "
                  f"Use a fresh --out.")
            return 2

    # ---------------------------------------------------------------- MAVLink
    # One connection for the whole sweep: the vehicle link has nothing to do with
    # the camera pipeline, and reconnecting it per cell would cost --mavlink-wait
    # seconds each for no gain. It is also not restartable — close() sets a stop
    # event that start() never clears, so a per-cell open/close would silently
    # produce telemetry-free datasets from the second cell onward.
    mav = None
    if a.mavlink_transport != "none":
        mav = MavlinkLogger(connection=a.mavlink, transport=a.mavlink_transport,
                            rest_url=a.mavlink_rest_url, set_rates=False)
        mav.start(wait_s=a.mavlink_wait)

    fh = csv_path.open("a" if appending else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if not appending:
        writer.writeheader()

    (out_root / "sweep_meta.json").write_text(json.dumps({
        "tool": "c3_option_sweep.py",
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "cells": [{"n": i, "dir": c.slug(i), "label": c.label(),
                   "config": c.cfg.to_dict()}
                  for i, c in enumerate(combos, 1)],
    }, indent=2, default=str))

    rows: list[dict] = []
    try:
        for i, combo in enumerate(combos, 1):
            out_dir = out_root / combo.slug(i)
            print(f"\n[{i}/{len(combos)}] {combo.label()}")
            print(f"    -> {out_dir}")
            # run_one returns its row on every path, interrupts included, rather
            # than carrying it out through an exception: a second Ctrl-C during
            # the writer flush would destroy an exception in flight and leave the
            # finished dataset with no results.csv entry at all.
            row = run_one(combo, out_dir, a, mav, i, len(combos))

            if row["frames_written"] and a.check:
                row["check"] = check_dataset(out_dir, combo, a.check_sample)
            rows.append(row)
            writer.writerow(row)
            fh.flush()

            if row["ok"]:
                print(f"    -> {row['frames_written']} frames, "
                      f"{row['dataset_mb']} MB, colour {row['color_fps']} fps "
                      f"p50 {row['color_lat_p50']} ms, link "
                      f"{row['measured_mbps']} Mbit/s, {row['slam_config']}"
                      f"{', check ' + str(row['check']) if row['check'] else ''}")
            elif row["frames_written"]:
                # Recorded, then something went wrong afterwards. Say both: the
                # frames are on disk, and the dataset is not usable as it stands.
                print(f"    -> recorded {row['frames_written']} frames but the "
                      f"cell is NOT usable: {row['error']}")
            else:
                print(f"    -> FAILED: {row['error']}")

            if str(row["error"]) == "interrupted":
                print("\n  interrupted — partial results kept")
                break

            if i < len(combos):
                time.sleep(a.settle)
    except KeyboardInterrupt:
        print("\n  interrupted — partial results kept")
    finally:
        fh.close()
        if mav is not None:
            mav.close()

    print_table(rows)
    print(f"wrote {csv_path}")
    print(f"datasets under {out_root}/ — each one is a TUM RGB-D directory in the "
          f"same layout as c3_camera/datasets/dataset_*")

    good = [r for r in rows if r["ok"] and r["color_fps"]]
    if good:
        print()
        # Ranked by what a dataset is actually judged on. Delivered fps first —
        # a cell that "reaches 20 fps" by discarding half its frames shows up
        # here as the lower number, not as a win.
        fastest = max(good, key=lambda r: float(r["color_fps"] or 0))
        lowlat = min(good, key=lambda r: float(r["color_lat_p50"] or 1e9))
        print(f"  highest delivered fps : [{fastest['n']}] {fastest['label']} "
              f"-> {fastest['color_fps']} fps, p50 {fastest['color_lat_p50']} ms")
        print(f"  lowest latency        : [{lowlat['n']}] {lowlat['label']} "
              f"-> {lowlat['color_fps']} fps, p50 {lowlat['color_lat_p50']} ms")
        rgbd = [r for r in good if str(r["slam_config"]).endswith("rgbd.yaml")]
        if rgbd:
            best = max(rgbd, key=lambda r: float(r["color_fps"] or 0))
            print(f"  best RGB-D-ready cell : [{best['n']}] {best['label']} "
                  f"-> {best['color_fps']} fps, {best['assoc_lines']} associated "
                  f"rgb/depth pairs")
        print("\n  Next: run SLAM on the cells you care about, then judge them by "
              "the map rather than by this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
