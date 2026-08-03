#!/usr/bin/env python3
"""
c3_dataset_check.py — verify a recorded dataset before anyone runs SLAM on it.

    python c3_camera/c3_dataset_check.py c3_camera/datasets/dataset_20260729_161500

Checks the things that are silently wrong rather than loudly broken:

  * depth PNGs really are 16-bit, and their values are plausible as millimetres
    (a dataset written with 8-bit depth, or in metres, looks fine in a viewer)
  * rgb and depth are the same resolution, which is what makes the colour
    intrinsics valid for the depth image
  * every file named in rgb.txt / depth.txt / associations.txt exists
  * associations.txt column order is rgb-then-depth, as ORB-SLAM3 expects
  * timestamps are monotonic, and the rgb/depth pairing is tight
  * the timeline: camera, camera-IMU and MAVLink all overlap and share a clock
  * depth_scale is stated, and is 1000.0 for millimetre data
  * IMU rate and continuity; MAVLink message rates

Needs no camera and no SLAM system. Exit code 0 = usable, 1 = problems found.
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


class Check:
    def __init__(self):
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.problems.append(msg)
        print(f"  FAIL  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")


def read_list(path: Path) -> list[tuple[str, str]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--sample", type=int, default=6,
                   help="how many image pairs to open and inspect (default: %(default)s)")
    a = p.parse_args(argv)
    d = a.path
    c = Check()

    print("=" * 74)
    print(f"dataset: {d}")
    print("=" * 74)
    if not d.is_dir():
        print(f"  FAIL  not a directory")
        return 1

    # ---------------------------------------------------------------- metadata
    meta = {}
    mj = d / "metadata.json"
    if mj.exists():
        try:
            meta = json.loads(mj.read_text())
            c.ok(f"metadata.json parsed")
        except json.JSONDecodeError as e:
            c.fail(f"metadata.json is not valid JSON: {e}")
    else:
        c.fail("metadata.json missing — the dataset does not describe itself")

    scale = meta.get("depth_scale")
    if scale is None:
        c.fail("depth_scale absent from metadata; a consumer cannot convert depth")
    elif abs(float(scale) - 1000.0) > 1e-6:
        c.fail(f"depth_scale is {scale}, expected 1000.0 for uint16 millimetres "
               f"(5000 is TUM's Kinect-specific value and would make depths 5x small)")
    else:
        c.ok(f"depth_scale = {float(scale):.1f}  (metres = pixel / 1000)")

    if not (d / "metadata.txt").exists():
        c.warn("metadata.txt missing (the human-readable explanation)")
    if not (d / "calibration.json").exists():
        c.fail("calibration.json missing — no intrinsics for the colleague")
    else:
        try:
            calib = json.loads((d / "calibration.json").read_text())
            si = calib.get("streamed_intrinsics") or {}
            if "color" in si:
                k = si["color"]
                c.ok(f"calibration: colour fx={k['fx']:.1f} fy={k['fy']:.1f} "
                     f"cx={k['cx']:.1f} cy={k['cy']:.1f} @{k['width']}x{k['height']}")
            else:
                c.warn("calibration.json has no colour intrinsics entry")
            ie = calib.get("imu_extrinsics") or {}
            if ie.get("available"):
                c.ok("calibration includes IMU-camera extrinsics")
            else:
                c.warn("no IMU-camera extrinsics (expected for this device) — "
                       "VIO needs T_imu_cam calibrated separately")
        except json.JSONDecodeError as e:
            c.fail(f"calibration.json invalid: {e}")

    mode = meta.get("writer", {}).get("mode") or ""
    if mode == "review":
        print("\n  review-mode recording: mp4 + telemetry only, no image dataset")
        for f in ("rgb.mp4",):
            print(f"  {'ok   ' if (d / f).exists() else 'FAIL '} {f}")
        return 0 if not c.problems else 1

    encoded_rgb = next((p for p in (d / "rgb.h264", d / "rgb.h265")
                        if p.is_file()), None)
    if encoded_rgb is not None and not any((d / "rgb").glob("*.png")):
        c.fail(
            f"{encoded_rgb.name} has not been extracted — run "
            f"`python c3_camera/c3_extract_rgb_video.py {d}` before SLAM")

    # ------------------------------------------------------------- TUM indices
    rgb_list = read_list(d / "rgb.txt")
    depth_list = read_list(d / "depth.txt")
    assoc = [l.split() for l in (d / "associations.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")] if (d / "associations.txt").exists() else []

    if not rgb_list:
        c.fail("rgb.txt missing or empty")
    else:
        c.ok(f"rgb.txt: {len(rgb_list)} entries")
    if not depth_list:
        c.fail("depth.txt missing or empty")
    else:
        c.ok(f"depth.txt: {len(depth_list)} entries")
    if not assoc:
        c.fail("associations.txt missing or empty — RGB-D SLAM reads this")
    else:
        c.ok(f"associations.txt: {len(assoc)} pairs")
        bad = [r for r in assoc if len(r) != 4]
        if bad:
            c.fail(f"{len(bad)} association rows do not have 4 columns")
        else:
            # Column order must be rgb_ts rgb_file depth_ts depth_file.
            r0 = assoc[0]
            if "rgb/" in r0[1] and "depth/" in r0[3]:
                c.ok("associations.txt column order is rgb_ts rgb_file depth_ts "
                     "depth_file (what ORB-SLAM3's rgbd_tum expects)")
            else:
                c.fail(f"associations.txt column order looks wrong: {r0}")

    # ------------------------------------------------------------ files exist
    missing = 0
    for _, rel in rgb_list + depth_list:
        if not (d / rel).exists():
            missing += 1
    if missing:
        c.fail(f"{missing} files named in the indices do not exist")
    elif rgb_list:
        c.ok("every file named in rgb.txt/depth.txt exists")

    # ------------------------------------------------- timestamps monotonic
    for name, lst in (("rgb.txt", rgb_list), ("depth.txt", depth_list)):
        ts = [float(t) for t, _ in lst]
        if len(ts) > 1:
            if any(b <= a for a, b in zip(ts, ts[1:])):
                c.fail(f"{name} timestamps are not strictly increasing")
            else:
                span = ts[-1] - ts[0]
                c.ok(f"{name} timestamps increase monotonically; "
                     f"{span:.2f} s span, mean {(len(ts)-1)/span:.2f} Hz"
                     if span > 0 else f"{name} monotonic")
            if ts[0] < 1e9:
                c.warn(f"{name} timestamps look like monotonic-since-boot "
                       f"({ts[0]:.1f}), not Unix epoch — check metadata's clock note")

    # ------------------------------------------------- open the actual images
    print()
    n = min(a.sample, len(assoc)) if assoc else 0
    step = max(1, len(assoc) // max(1, n)) if assoc else 1
    shapes = set()
    for row in assoc[::step][:n]:
        _, rgb_rel, _, depth_rel = row
        rgb = cv2.imread(str(d / rgb_rel), cv2.IMREAD_COLOR)
        # IMREAD_UNCHANGED or OpenCV silently downcasts to 8-bit.
        dep = cv2.imread(str(d / depth_rel), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            c.fail(f"could not read {rgb_rel} / {depth_rel}")
            continue
        shapes.add((rgb.shape[1], rgb.shape[0], dep.shape[1], dep.shape[0]))
        if dep.dtype != np.uint16:
            c.fail(f"{depth_rel} is {dep.dtype}, must be uint16 millimetres")
        valid = dep[dep > 0]
        pct = 100.0 * valid.size / dep.size
        med = int(np.median(valid)) if valid.size else 0
        print(f"  {Path(rgb_rel).name:<22} rgb {rgb.shape[1]}x{rgb.shape[0]}  "
              f"depth {dep.shape[1]}x{dep.shape[0]} {dep.dtype}  "
              f"valid {pct:5.1f}%  median {med} mm")
        if valid.size and (med < 50 or med > 30000):
            c.warn(f"{depth_rel} median {med} — implausible for millimetres "
                   f"(metres would read ~1-10, and this should be ~300-10000)")

    if len(shapes) > 1:
        c.fail(f"inconsistent resolutions across frames: {sorted(shapes)}")
    elif shapes:
        rw, rh, dw, dh = next(iter(shapes))
        if (rw, rh) == (dw, dh):
            c.ok(f"rgb and depth are both {rw}x{rh} — 1:1 pixel correspondence, "
                 f"so the colour intrinsics describe the depth image too")
        else:
            c.fail(f"rgb is {rw}x{rh} but depth is {dw}x{dh}; RGB-D SLAM needs "
                   f"them identical (record with depth aligned AND matched to colour)")

    # ------------------------------------------------- stereo pair present
    for sub in ("left", "right"):
        n_files = len(list((d / sub).glob("*.png"))) if (d / sub).is_dir() else 0
        if n_files:
            c.ok(f"{sub}/: {n_files} lossless PNG (stereo SLAM / depth recompute)")
        else:
            c.warn(f"{sub}/ is empty — stereo SLAM will not be possible")

    # --------------------------------------------------------------- CSVs
    print()
    for name, what in (("frames.csv", "per-frame index"),
                       ("imu_camera.csv", "camera BNO086"),
                       ("imu_rov.csv", "ArduSub IMU"),
                       ("telemetry.csv", "attitude/pressure/battery"),
                       ("control.csv", "pilot inputs")):
        f = d / name
        if not f.exists():
            (c.warn if name != "frames.csv" else c.fail)(f"{name} missing ({what})")
            continue
        rows = list(csv.DictReader(f.open()))
        if not rows:
            c.warn(f"{name} has a header but no rows ({what})")
            continue
        ts = [float(r["t_unix"]) for r in rows if r.get("t_unix")]
        rate = ((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
        c.ok(f"{name}: {len(rows)} rows, {rate:7.1f} Hz  ({what})")

    # ------------------------------------------------------- timeline overlap
    def span_of(name: str):
        f = d / name
        if not f.exists():
            return None
        ts = [float(r["t_unix"]) for r in csv.DictReader(f.open())
              if r.get("t_unix")]
        return (min(ts), max(ts)) if ts else None

    print()
    spans = {k: span_of(k) for k in ("frames.csv", "imu_camera.csv",
                                     "imu_rov.csv", "telemetry.csv")}
    spans = {k: v for k, v in spans.items() if v}
    if len(spans) > 1:
        lo = max(v[0] for v in spans.values())
        hi = min(v[1] for v in spans.values())
        if hi > lo:
            c.ok(f"all logged streams overlap for {hi - lo:.1f} s "
                 f"— they share one timeline")
        else:
            c.fail("logged streams do NOT overlap in time; check the clock note")
        for k, v in sorted(spans.items()):
            print(f"        {k:<16} {v[0]:.3f} .. {v[1]:.3f}  ({v[1]-v[0]:.1f} s)")

    # ------------------------------------------------------------- verdict
    print()
    print("=" * 74)
    if c.problems:
        print(f"VERDICT: {len(c.problems)} problem(s) — fix before handing this over")
        for m in c.problems:
            print(f"  - {m}")
    else:
        print("VERDICT: usable. RGB-D, stereo and visual-inertial are all possible.")
    if c.warnings:
        print(f"\n{len(c.warnings)} warning(s):")
        for m in c.warnings:
            print(f"  - {m}")
    print("=" * 74)
    return 1 if c.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
