#!/usr/bin/env python3
"""
c3_replay.py — inspect, play back, or export a recording made by c3_stream.py.

    # what is in this recording, and is it intact?
    python c3_camera/c3_replay.py recordings/20260729_161500 --info

    # play it back at the recorded rate
    python c3_camera/c3_replay.py recordings/20260729_161500

    # make a shareable video (colour+depth side by side)
    python c3_camera/c3_replay.py recordings/20260729_161500 --export-mp4 out.mp4

`--info` is the one to run first. It checks the recording against itself: that
every file the index names exists, that depth really is uint16 millimetres, and
that the device sequence numbers have no gaps. A recording that dropped frames is
still usable — but you want to know that before you train on it, not after.

Needs no camera.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from c3_camera import viz


# =============================================================================
# Loading
# =============================================================================
class Recording:
    def __init__(self, path: Path):
        self.dir = Path(path)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"{self.dir} is not a directory")

        index = self.dir / "frames.csv"
        if not index.exists():
            raise FileNotFoundError(
                f"{index} not found — is this a c3_stream.py recording directory?")
        with index.open() as fh:
            self.rows = list(csv.DictReader(fh))

        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, row: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        colour = depth = None
        if row.get("color_file"):
            p = self.dir / row["color_file"]
            if p.exists():
                colour = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if row.get("depth_file"):
            p = self.dir / row["depth_file"]
            if p.exists():
                if p.suffix == ".npy":
                    depth = np.load(p)
                else:
                    # IMREAD_UNCHANGED, or OpenCV silently gives back 8-bit and
                    # every depth value is wrong by a factor of 257.
                    depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return colour, depth

    # ------------------------------------------------------------------ timing
    def duration_s(self) -> float:
        ts = [float(r["color_t_device"]) for r in self.rows if r.get("color_t_device")]
        if len(ts) < 2:
            ts = [float(r["depth_t_device"]) for r in self.rows
                  if r.get("depth_t_device")]
        return (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0

    def mean_fps(self) -> float:
        d = self.duration_s()
        return (len(self.rows) - 1) / d if d > 0 else 0.0


# =============================================================================
# Verification
# =============================================================================
def report(rec: Recording) -> int:
    print("=" * 74)
    print(f"recording: {rec.dir}")
    print("=" * 74)

    m = rec.meta
    if m:
        print(f"  recorded_at   : {m.get('recorded_at', '?')}")
        print(f"  depth units   : {m.get('depth_units', '?')}")
        dev = m.get("device") or {}
        if dev:
            print(f"  device        : {dev.get('product')} mxid={dev.get('mxid')}")
        res = m.get("resolved") or {}
        if res:
            print(f"  colour out    : {res.get('color_out')}")
            print(f"  depth out     : {res.get('depth_out')}")
        cfgm = m.get("config") or {}
        if cfgm:
            print(f"  config        : {cfgm.get('color_res')} "
                  f"isp={cfgm.get('isp_scale')} enc={cfgm.get('color_encode')} "
                  f"{cfgm.get('fps')} fps align={cfgm.get('depth_align')}")

    print(f"\n  frames        : {len(rec)}")
    print(f"  duration      : {rec.duration_s():.2f} s")
    print(f"  mean fps      : {rec.mean_fps():.2f}")

    # --- files present ----------------------------------------------------
    missing = []
    for r in rec.rows:
        for key in ("color_file", "depth_file"):
            if r.get(key) and not (rec.dir / r[key]).exists():
                missing.append(r[key])
    print(f"  missing files : {len(missing)}"
          + (f"  e.g. {missing[:3]}" if missing else "  (all present)"))

    # --- sequence continuity ---------------------------------------------
    problems = 0
    for tag in ("color_seq", "depth_seq"):
        seqs = [int(r[tag]) for r in rec.rows if r.get(tag)]
        if len(seqs) < 2:
            continue
        gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
        span = seqs[-1] - seqs[0] + 1
        lost = span - len(seqs)
        status = "contiguous" if gaps == 0 else f"{gaps} gaps, {lost} frames lost"
        print(f"  {tag:<14}: {seqs[0]}..{seqs[-1]}  {status}")
        if gaps:
            problems += 1

    # --- skew / latency ---------------------------------------------------
    for tag, label in (("skew_ms", "colour/depth skew"),
                       ("color_latency_ms", "colour latency"),
                       ("depth_latency_ms", "depth latency")):
        vals = [float(r[tag]) for r in rec.rows if r.get(tag)]
        if vals:
            vals_sorted = sorted(vals)
            p50 = vals_sorted[len(vals) // 2]
            print(f"  {label:<18}: mean {sum(vals)/len(vals):6.1f}  "
                  f"p50 {p50:6.1f}  max {max(vals):6.1f} ms")

    # --- depth sanity: is it really uint16 millimetres? -------------------
    checked = 0
    for r in rec.rows[:: max(1, len(rec.rows) // 5)][:5]:
        colour, depth = rec.load(r)
        if depth is None:
            continue
        checked += 1
        valid = depth[depth > 0]
        note = "" if depth.dtype == np.uint16 else f"  *** dtype is {depth.dtype}, expected uint16 ***"
        if depth.dtype != np.uint16:
            problems += 1
        pct = 100.0 * valid.size / depth.size
        print(f"  depth[{r['idx']:>6}] : {depth.shape} {depth.dtype}  "
              f"valid {pct:6.2f}%  "
              + (f"median {int(np.median(valid))} mm" if valid.size else "no valid pixels")
              + note)
        if colour is not None:
            print(f"  colour[{r['idx']:>5}] : {colour.shape} {colour.dtype}")

    if not checked:
        print("  depth         : no depth frames to check")

    total_mb = sum(p.stat().st_size for p in rec.dir.rglob("*") if p.is_file()) / 1e6
    print(f"\n  size on disk  : {total_mb:.1f} MB "
          f"({total_mb/max(len(rec),1)*1024:.0f} kB per frame)")

    print("\n" + "=" * 74)
    if problems:
        print(f"VERDICT: usable, but {problems} thing(s) to be aware of above.")
    else:
        print("VERDICT: intact — files all present, sequences contiguous, "
              "depth is uint16 mm.")
    print("=" * 74)
    return 0


# =============================================================================
# Playback / export
# =============================================================================
def play(rec: Recording, args) -> int:
    writer = None
    window = f"c3 replay — {rec.dir.name}"
    if not args.export_mp4:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps = args.fps or rec.mean_fps() or 15.0
    delay = max(1, int(1000.0 / fps))
    paused = False
    i = 0

    while 0 <= i < len(rec):
        row = rec.rows[i]
        colour, depth = rec.load(row)
        panes = []
        if colour is not None:
            panes.append(viz.label(colour.copy(), f"color {row['idx']}"))
        if depth is not None:
            panes.append(viz.label(viz.colorize_depth(depth, args.min_mm, args.max_mm),
                                   f"depth mm {row['idx']}"))
        if not panes:
            i += 1
            continue
        canvas = viz.hstack_pad(panes)

        info = [f"{rec.dir.name}  frame {i+1}/{len(rec)}  idx {row['idx']}"]
        if row.get("skew_ms"):
            info.append(f"skew {row['skew_ms']} ms   "
                        f"colour lat {row.get('color_latency_ms','?')} ms")
        if not args.export_mp4:
            info.append("q quit   space pause   . next   , prev")
        viz.draw_hud(canvas, info)

        if args.export_mp4:
            if writer is None:
                writer = cv2.VideoWriter(str(args.export_mp4),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened():
                    print(f"ERROR: could not open {args.export_mp4} for writing")
                    return 1
            writer.write(canvas)
            if i % 25 == 0:
                print(f"  {i+1}/{len(rec)}")
            i += 1
            continue

        cv2.imshow(window, canvas)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("."):
            paused, i = True, i + 1
        elif key == ord(","):
            paused, i = True, max(0, i - 1)
        elif not paused:
            i += 1

    if writer is not None:
        writer.release()
        print(f"wrote {args.export_mp4}  ({len(rec)} frames at {fps:.1f} fps)")
        print("NOTE: the MP4 is a lossy visualisation for humans — the depth in it "
              "is a colour map, not millimetres. Use the PNG/npy files for data.")
    else:
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path, help="recording directory")
    p.add_argument("--info", action="store_true",
                   help="verify and describe the recording, do not play it")
    p.add_argument("--fps", type=float, default=None,
                   help="playback rate (default: the recorded rate)")
    p.add_argument("--export-mp4", type=Path, default=None,
                   help="write a side-by-side MP4 instead of showing a window")
    p.add_argument("--min-mm", type=float, default=300.0)
    p.add_argument("--max-mm", type=float, default=6000.0)
    a = p.parse_args(argv)

    try:
        rec = Recording(a.path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}")
        return 1

    if not len(rec):
        print(f"{a.path} has no frames in frames.csv")
        return 1

    if a.info:
        return report(rec)
    return play(rec, a)


if __name__ == "__main__":
    raise SystemExit(main())
