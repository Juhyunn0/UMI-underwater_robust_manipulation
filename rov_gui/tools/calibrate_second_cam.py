#!/usr/bin/env python3
"""calibrate_second_cam.py — intrinsics for a camera from a REC NAV run.

The ROV's default RGB camera has no factory calibration; ``hw_nav.yaml:
second_cam`` is a vendor-FOV focal length and NO distortion terms, and the
camera is wide-angle. That is why, on 2026-09-07, every multi-tag RGB frame
failed the reprojection gate while single tags passed: one tag fits any
pinhole, several tags across a distorted frame fit none.

The tag mat is a calibration target — 0.170 m squares at surveyed 3-D
positions (``config/tag_map_full.yaml``) — and a REC NAV run records every
detected quad of the fallback feed with the frame size it was seen at
(``frames_second.csv`` / ``detections_second.csv``). So this is exactly
``cv2.calibrateCamera``: known object points per frame, their pixels, and
the poses solved along the way. It is the checkerboard case with the board
lying still and the camera moving, which is fine as long as the views vary;
the printed standard deviations say whether they varied enough.

    python -m rov_gui.tools.calibrate_second_cam <run>/nav_HHMMSS
        [--feed second|main] [--map PATH] [--min-tags 4] [--max-frames 250]
        [--model 5|8] [--yaml]

``--feed main`` runs the same thing on the C3's own detections, whose
intrinsics are KNOWN (the EEPROM values ride frames.csv): a self-check that
the tool recovers a focal length it was not told.

Every number printed is a measurement of THIS run's detections; the run
path is the provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.control.tagnav import TagMap, tag_object_points  # noqa: E402


def _load(run: Path, feed: str):
    sfx = "" if feed == "main" else f"_{feed}"
    fr = {int(r["frame"]): r for r in
          csv.DictReader(open(run / f"frames{sfx}.csv", encoding="utf-8"))}
    dets = defaultdict(list)
    for r in csv.DictReader(open(run / f"detections{sfx}.csv", encoding="utf-8")):
        c = np.array([[float(r[f"x{i}"]), float(r[f"y{i}"])] for i in range(4)])
        dets[int(r["frame"])].append((int(r["tag_id"]), c))
    return fr, dets


def _map_path(run: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    for c in (run / "controller.json", run.parent / "controller.json"):
        if c.exists():
            p = json.load(open(c)).get("hardware", {}).get("tag_map")
            if p:
                return Path(p)
    return REPO / "config" / "tag_map_full.yaml"


def _views(fr, dets, tm, tag_size, min_tags, max_frames):
    """(object points, image points, frame ids) for every usable frame."""
    obj_tag = tag_object_points(tag_size).astype(np.float64)
    dup = tm.duplicate_ids
    views = []
    for f in sorted(dets):
        r = fr.get(f)
        if r is None:
            continue
        w, h = int(r["width"]), int(r["height"])
        obj, img = [], []
        for tid, c in dets[f]:
            if tid not in tm or tid in dup:
                continue                      # unmapped / ambiguous copy
            if (c[:, 0].min() < 0 or c[:, 1].min() < 0
                    or c[:, 0].max() > w - 1 or c[:, 1].max() > h - 1):
                continue                      # clipped: extrapolated corners
            R, t = tm.poses[tid]
            obj.append((R @ obj_tag.T).T + t)
            img.append(c)
        if len(obj) >= min_tags:
            views.append((np.concatenate(obj).astype(np.float32),
                          np.concatenate(img).astype(np.float32), f, (w, h)))
    if len(views) > max_frames:               # thin evenly across the run
        idx = np.linspace(0, len(views) - 1, max_frames).round().astype(int)
        views = [views[i] for i in sorted(set(idx))]
    return views


def _rms_with(views, K, dist):
    """Per-frame solvePnP with a GIVEN model -> (p50, p90) of per-frame RMS.
    Per-frame, not pooled: two garbage frames in 160 put the pooled figure
    at 45 px on a run whose frames sit at 2 px."""
    import cv2
    per = []
    for obj, img, _f, _wh in views:
        ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64),
                                      K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1)))))
    if not per:
        return float("nan"), float("nan")
    return float(np.median(per)), float(np.percentile(per, 90))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a REC NAV folder (…/nav_HHMMSS)")
    ap.add_argument("--feed", default="second", choices=("second", "main"))
    ap.add_argument("--map", default=None, help="tag map yaml (default: the "
                    "run's controller.json hardware.tag_map)")
    ap.add_argument("--min-tags", type=int, default=4,
                    help="unique, unclipped tags a frame needs (default 4)")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--model", default="5", choices=("5", "8"),
                    help="distortion terms: 5 = k1 k2 p1 p2 k3; 8 = rational "
                         "(+k4 k5 k6), for strong wide-angle distortion")
    ap.add_argument("--tag-size", type=float, default=0.170)
    ap.add_argument("--free-pp", action="store_true",
                    help="also fit the principal point (default: fixed at "
                         "the frame's own cx, cy). A PLANAR target seen at "
                         "similar tilts cannot pin cx/cy — on the C3 self-check "
                         "a free cy wandered 146 px at the same residual.")
    ap.add_argument("--free-aspect", action="store_true",
                    help="also fit fy independently of fx (default: square "
                         "pixels, fy = fx)")
    ap.add_argument("--yaml", action="store_true",
                    help="print a hw_nav.yaml second_cam snippet")
    a = ap.parse_args(argv)
    import cv2

    run = Path(a.run)
    fr, dets = _load(run, a.feed)
    mp = _map_path(run, a.map)
    tm = TagMap.load(mp)
    views = _views(fr, dets, tm, a.tag_size, a.min_tags, a.max_frames)
    if len(views) < 8:
        print(f"only {len(views)} usable frames (>= {a.min_tags} unique unclipped "
              f"tags each) — record with both cameras on the mat, moving the "
              f"vehicle through a few positions and headings", file=sys.stderr)
        return 2
    sizes = {v[3] for v in views}
    if len(sizes) != 1:
        print(f"frame size changed during the run: {sizes}", file=sys.stderr)
        return 2
    (w, h), = sizes
    r0 = fr[views[0][2]]
    K0 = np.array([[float(r0["fx"]), 0, float(r0["cx"])],
                   [0, float(r0["fy"]), float(r0["cy"])], [0, 0, 1.0]])
    n_pts = sum(len(v[1]) for v in views)
    print(f"run     {run}")
    print(f"feed    {a.feed}  {w}x{h}  {len(views)} frames  {n_pts} corners  "
          f"map {mp.name} ({len(tm)} ids)")
    b50, b90 = _rms_with(views, K0, None)
    print(f"before  fx {K0[0, 0]:.1f} fy {K0[1, 1]:.1f} cx {K0[0, 2]:.1f} "
          f"cy {K0[1, 2]:.1f}  dist none  ->  per-frame RMS p50 {b50:.2f} "
          f"p90 {b90:.2f} px")

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if a.model == "8":
        flags |= cv2.CALIB_RATIONAL_MODEL
    if not a.free_pp:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if not a.free_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    def fit(vs):
        objs = [v[0] for v in vs]
        imgs = [v[1].reshape(-1, 1, 2) for v in vs]
        return cv2.calibrateCameraExtended(objs, imgs, (w, h), K0.copy(), None,
                                           flags=flags)

    # ITERATED REJECTION. A recording is not a clean checkerboard session: a
    # quad the gripper half-covers, a tag at the very edge, a frame
    # mid-motion-blur all decode and all carry corners wrong by tens of
    # pixels, and with STRONG distortion calibrateCamera's own per-view
    # pose init lands some views in the flipped basin. Either kind drags the
    # whole fit — on the C3 self-check fx was off 47% until two frames went,
    # and on a synthetic 1280x720 k1=-0.28 set the first pass did not move
    # at all (RMS 600 px) while three passes of "drop views > 3x the median
    # residual, refit" reached fx -1%, k1/k2 to 0.005, RMS 0.5 px. The
    # criterion is RELATIVE to the current fit — an uncalibrated camera is
    # wrong everywhere, which is what the fit is for — so only views far
    # outside the pack go, and the count is printed every pass.
    for it in range(1, 5):
        rms, K, dist, rvecs, tvecs, sd_int, _sd_ext, per_view = fit(views)
        pv = np.asarray(per_view).ravel()
        keep = pv <= max(3.0 * float(np.median(pv)), 1.5)
        n_drop = int((~keep).sum())
        if n_drop == 0 or keep.sum() < 8:
            break
        views = [v for v, k in zip(views, keep) if k]
        print(f"pass {it}  RMS {rms:.2f} px; dropped {n_drop} frame(s) with "
              f"residual > 3x the median ({len(views)} kept) — refitting")
    dist = dist.ravel()
    sd = sd_int.ravel()
    hfov = 2 * math.degrees(math.atan(w / (2 * K[0, 0])))
    print(f"after   fx {K[0, 0]:.1f}±{sd[0]:.1f} fy {K[1, 1]:.1f}±{sd[1]:.1f} "
          f"cx {K[0, 2]:.1f}±{sd[2]:.1f} cy {K[1, 2]:.1f}±{sd[3]:.1f}  "
          f"->  RMS {rms:.2f} px   (HFOV {hfov:.1f} deg)")
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    print("dist    " + "  ".join(f"{n} {v:+.4f}±{s:.4f}"
                                 for n, v, s in zip(names, dist, sd[4:])))
    pv = np.asarray(per_view).ravel()
    print(f"per-frame RMS p50 {np.median(pv):.2f}  p90 {np.percentile(pv, 90):.2f}"
          f"  max {pv.max():.2f} px")
    # Did the views vary enough for a PLANAR target? What constrains K is
    # the camera's orientation relative to the plane, so: the optical axis
    # in the map frame, as incidence (angle from the floor normal) and
    # azimuth (heading about it). A fixed-tilt camera on a vehicle that only
    # translates gives one incidence and one azimuth — one view, however
    # many frames — and fx then trades freely against distortion.
    inc, azi = [], []
    for rv in rvecs:
        R_cm, _ = cv2.Rodrigues(rv)
        ax = R_cm.T[:, 2]                    # optical axis, map frame (+z down)
        inc.append(math.degrees(math.acos(max(-1.0, min(1.0, float(ax[2]))))))
        azi.append(math.degrees(math.atan2(float(ax[1]), float(ax[0]))))
    az = np.unwrap(np.radians(azi))
    print(f"view diversity: incidence {min(inc):.0f}-{max(inc):.0f} deg "
          f"(spread {np.ptp(inc):.1f}), azimuth spread {math.degrees(np.ptp(az)):.0f} deg "
          f"over {len(inc)} frames"
          + ("  <- small: fx trades against distortion, move/turn the vehicle more"
             if np.ptp(inc) < 8.0 and math.degrees(np.ptp(az)) < 30.0 else ""))
    if a.feed == "main":
        print(f"self-check vs the frame's own K: fx {100 * (K[0, 0] / K0[0, 0] - 1):+.2f}%"
              f"  cx {K[0, 2] - K0[0, 2]:+.1f} px  cy {K[1, 2] - K0[1, 2]:+.1f} px")
    if a.yaml:
        print("\n# hw_nav.yaml — paste under second_cam: (measured from "
              f"{run}, {len(views)} frames, RMS {rms:.2f} px)")
        print(f"  fx: {K[0, 0]:.2f}\n  fy: {K[1, 1]:.2f}\n  cx: {K[0, 2]:.2f}"
              f"\n  cy: {K[1, 2]:.2f}\n  width: {w}\n  height: {h}")
        print("  dist: [" + ", ".join(f"{v:.6f}" for v in dist) + "]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
