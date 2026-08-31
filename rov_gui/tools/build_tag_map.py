#!/usr/bin/env python3
"""build_tag_map.py — grow the surveyed tag map over a whole mat, from one pass.

    python -m rov_gui.tools.build_tag_map <run folder>/nav_<hhmmss> \
        --anchor config/tag_map.yaml -o config/tag_map_full.yaml

WHAT IT DOES, and why it cannot drift
-------------------------------------
The 2026-05-29 gantry survey pinned 47 tags to 5-20 mm (its own diagnostics:
median residual 0.227 px, anchor drift 0.0 mm). Those stay FROZEN and are the
only thing this tool trusts to start with.

Then, per frame:

    known tags in this frame  --solvePnP-->  camera pose (mm-level, locked map)
    camera pose + an unknown tag's corners  --IPPE-->  that tag's pose

Every new tag is solved against the LOCKED anchors, never against a chain of
other new tags in the same step. A tag is promoted into the anchor set only
after ``--min-obs`` independent observations agree to within ``--max-spread``,
and promotion happens between ROUNDS. So error does not accumulate along a
path the way a free SLAM graph's does — this is trilateration outward from a
fixed frame, not a pose graph, and it has no loop-closure failure mode.

That is deliberate: the operator's memory of a previous survey "warping" is a
real failure mode of graph SLAM under weak constraints, and the whole point of
this design is that the geometry that warps is not present.

DUPLICATED IDS
--------------
The mat reuses ids (config/hw_nav.yaml duplicate_ids). Those ids are pulled
OUT of the anchor set before the build — a duplicated id cannot be trusted to
localize a frame, because which copy it is, is the thing we do not know yet —
and are then rediscovered like any unknown tag. An id whose observations form
two well-separated clusters is written out as two instances and re-declared
under ``duplicate_ids`` in the output. Clusters closer together than
``--dup-sep`` are one tag, not two (that spread is noise).

If a rediscovered cluster lands on the surveyed pose (within ``--max-spread``)
the SURVEYED pose is kept rather than the bootstrap estimate, so the gantry
survey's numbers still win where it measured.

WHAT YOU MUST DO WHEN FLYING
----------------------------
Every frame that shows an UNKNOWN tag must also show KNOWN ones. Start over
the surveyed block and creep outward; never jump to a patch with no known
tags in view, or the chain breaks there and those tags simply never promote
(the report says which, and how many observations they got).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# =============================================================================
# io
# =============================================================================
def load_recording(run_dir: Path):
    """(frames, detections) from a REC NAV run. frames: idx -> dict."""
    frames = {}
    with open(run_dir / "frames.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frames[int(r["frame"])] = {
                "t": float(r["t_capture"]),
                "K": np.array([[float(r["fx"]), 0.0, float(r["cx"])],
                               [0.0, float(r["fy"]), float(r["cy"])],
                               [0.0, 0.0, 1.0]]),
                "dets": [],
            }
    with open(run_dir / "detections.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fr = frames.get(int(r["frame"]))
            if fr is None:
                continue
            fr["dets"].append((int(r["tag_id"]), np.array(
                [[float(r[f"x{i}"]), float(r[f"y{i}"])] for i in range(4)],
                dtype=np.float64)))
    return frames


def load_anchor_map(path: Path):
    """id -> [(R, t)] from a tag_map.yaml (a list so duplicates fit later)."""
    import yaml

    from ..control.tagnav import quat_wxyz_to_R

    raw = yaml.safe_load(open(path, encoding="utf-8"))
    out, anchor = {}, raw.get("anchor_tag_id")
    for tid, e in raw["tags"].items():
        out.setdefault(int(tid), []).append(
            (quat_wxyz_to_R(e["quaternion_wxyz"]),
             np.asarray(e["position_m"], float)))
    return out, (None if anchor is None else int(anchor))


def R_to_quat_wxyz(R: np.ndarray) -> list:
    """Rotation matrix -> (w, x, y, z), Shepperd's branchless-enough form."""
    m = np.asarray(R, float)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


# =============================================================================
# geometry
# =============================================================================
def camera_pose(dets, known, obj_tag, K, max_reproj_px, min_known=3):
    """solvePnP over the KNOWN tags in one frame -> (R_cam_map, t, rms, n).

    A known id with two instances is skipped here: which copy it is, is the
    very thing we cannot assume while building the map.
    """
    import cv2

    obj, img = [], []
    for tid, corners in dets:
        poses = known.get(tid)
        if not poses or len(poses) != 1:
            continue
        R_mt, t_mt = poses[0]
        obj.append((R_mt @ obj_tag.T).T + t_mt)
        img.append(corners)
    # More known tags than the localizer demands, on purpose: this pose is
    # about to DEFINE map geometry, and a two-tag fit on a flat floor is weak
    # enough to throw metre-scale outliers (seen in the real run).
    if len(obj) < min_known:
        return None
    O = np.concatenate(obj)
    I = np.concatenate(img)
    ok, rvec, tvec = cv2.solvePnP(O, I, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None
    proj, _ = cv2.projectPoints(O, rvec, tvec, K, None)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - I) ** 2).sum(1).mean()))
    # NOT `rms > max`: a NaN loses every comparison, so a degenerate solve
    # would sail through the gate and poison the map it is building.
    if not (rms <= max_reproj_px):
        return None
    R_cm, _ = cv2.Rodrigues(rvec)
    return R_cm, tvec.ravel(), rms, len(obj)


def tag_pose_from_frame(corners, R_cm, t_cm, obj_tag, K, max_reproj_px):
    """One tag's pose in the MAP frame, given the frame's camera pose.

    IPPE returns up to two solutions for a square; the one that matches the
    floor plane (tag +z along the map's +z, i.e. print-up) wins, and if both
    or neither do the observation is discarded rather than guessed.
    """
    import cv2

    try:
        n, rvecs, tvecs, _e = cv2.solvePnPGeneric(
            obj_tag, corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    best = None
    for i in range(n):
        # IPPE returns NaN on a degenerate square (grazing view, near-collinear
        # corners). Every `NaN > x` is False, so an unguarded NaN passes the
        # reprojection AND the face-up test and lands in the map as a tag whose
        # rotation poisons the average. Reject explicitly, first.
        if not (np.all(np.isfinite(rvecs[i])) and np.all(np.isfinite(tvecs[i]))):
            continue
        R_ct, _ = cv2.Rodrigues(rvecs[i])
        proj, _ = cv2.projectPoints(obj_tag, rvecs[i], tvecs[i], K, None)
        rms = float(np.sqrt(((proj.reshape(-1, 2) - corners) ** 2).sum(1).mean()))
        if not (rms <= max_reproj_px):
            continue
        # camera_T_tag -> map_T_tag
        R_mt = R_cm.T @ R_ct
        t_mt = R_cm.T @ (tvecs[i].ravel() - t_cm)
        # the mat is flat and printed face-up: tag +z must point along map +z
        if not (float(R_mt[2, 2]) >= 0.8):
            continue
        if best is None or rms < best[0]:
            best = (rms, R_mt, t_mt)
    return None if best is None else (best[1], best[2], best[0])


def find_instances(ts, min_obs, max_spread, dup_sep, max_inst=2):
    """Where do this id's observations actually pile up? -> [(idx, centre)].

    Mode-seeking, NOT "is the whole set tight": measured on the real pool run
    (sessions/nav_runs/20260813_215640) a tag's observations sit within about
    13 mm of their median while a few land METRES away, so any gate on the
    full extent rejects every tag (it rejected 89/89). Instead take the
    densest neighbourhood, keep what is within ``max_spread`` of its median,
    and drop those points before looking for a second pile — which is also
    how a duplicated id's two physical copies separate.
    """
    ts = np.asarray(ts, float)
    remaining = list(range(len(ts)))
    out = []
    while remaining and len(out) < max_inst:
        P = ts[remaining]
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        near = d <= max_spread
        seed = int(np.argmax(near.sum(1)))
        if int(near[seed].sum()) < min_obs:
            break
        med = np.median(P[near[seed]], axis=0)
        keep = [i for i in remaining
                if np.linalg.norm(ts[i] - med) <= max_spread]
        if len(keep) < min_obs:
            break
        out.append((keep, med))
        remaining = [i for i in remaining
                     if np.linalg.norm(ts[i] - med) > dup_sep]
    return out


def average_pose(Rs, ts):
    """Mean translation + the rotation closest to the mean (SVD projection).

    Belt and braces on the finiteness the callers already enforce: one NaN
    here would silently become a whole tag's orientation."""
    Rs = [R for R in Rs if np.all(np.isfinite(R))]
    ts = [t for t in ts if np.all(np.isfinite(t))]
    if not Rs or not ts:
        return None
    t = np.mean(ts, axis=0)
    M = np.mean(Rs, axis=0)
    U, _s, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, t


# =============================================================================
# the build
# =============================================================================
def build(frames, known, obj_tag, args, log=print):
    """Rounds of: solve camera poses from the known set, then solve unknowns."""
    obs = {}                      # id -> list of (R, t, rms)
    promoted_round = {}
    rnd = 0
    while True:
        rnd += 1
        n_cam = 0
        fresh = {}
        for fr in frames.values():
            cam = camera_pose(fr["dets"], known, obj_tag, fr["K"],
                              args.max_reproj_px, args.min_known)
            if cam is None:
                continue
            R_cm, t_cm, _rms, _n = cam
            n_cam += 1
            for tid, corners in fr["dets"]:
                if tid in known:
                    continue
                got = tag_pose_from_frame(corners, R_cm, t_cm, obj_tag,
                                          fr["K"], args.max_reproj_px)
                if got is not None:
                    fresh.setdefault(tid, []).append(got)
        for tid, lst in fresh.items():
            obs.setdefault(tid, []).extend(lst)

        # promote whatever is now well determined
        added = 0
        for tid, lst in sorted(obs.items()):
            if tid in known or len(lst) < args.min_obs:
                continue
            ts = np.array([t for _R, t, _e in lst])
            poses, inliers = [], []
            for keep, _med in find_instances(ts, args.min_obs,
                                             args.max_spread, args.dup_sep):
                avg = average_pose([lst[i][0] for i in keep],
                                   [ts[i] for i in keep])
                if avg is not None:
                    poses.append(avg)
                    inliers.append(len(keep))
            if not poses:
                continue
            known[tid] = poses
            promoted_round[tid] = (rnd, tuple(inliers), len(lst))
            added += 1
        log(f"  round {rnd}: {n_cam} frames localized, "
            f"{len(fresh)} unknown tags observed, {added} promoted "
            f"({len(known)} known)")
        if added == 0:
            break
    return known, obs, promoted_round


def refine(frames, known, frozen_ids, obj_tag, args, log=print):
    """Re-estimate every non-frozen tag with camera poses from the FULL map.

    The first build solves a new tag from frames where only a handful of tags
    were known yet, so its camera pose is the weakest link. Once the whole mat
    is in the map those same frames localize on ~20 tags instead of 3, and the
    tags re-solve much tighter. Anchors never move; this only sharpens what
    the build estimated.
    """
    for it in range(args.refine):
        obs, n_cam = {}, 0
        for fr in frames.values():
            cam = camera_pose(fr["dets"], known, obj_tag, fr["K"],
                              args.max_reproj_px, args.min_known)
            if cam is None:
                continue
            n_cam += 1
            R_cm, t_cm, _rms, _n = cam
            for tid, corners in fr["dets"]:
                if tid in frozen_ids:
                    continue
                got = tag_pose_from_frame(corners, R_cm, t_cm, obj_tag,
                                          fr["K"], args.max_reproj_px)
                if got is not None:
                    obs.setdefault(tid, []).append(got)
        moved = []
        for tid, lst in obs.items():
            ts = np.array([t for _R, t, _e in lst])
            poses = []
            for keep, _med in find_instances(ts, args.min_obs,
                                             args.max_spread, args.dup_sep):
                avg = average_pose([lst[i][0] for i in keep],
                                   [ts[i] for i in keep])
                if avg is not None:
                    poses.append(avg)
            if not poses or tid not in known or len(poses) != len(known[tid]):
                continue                     # never CHANGE an id's instance count
            for (R_new, t_new), (_R_old, t_old) in zip(poses, known[tid]):
                moved.append(float(np.linalg.norm(t_new - t_old)))
            known[tid] = poses
        if moved:
            log(f"  refine {it + 1}: {n_cam} frames localized on the full map, "
                f"{len(moved)} instances moved "
                f"p50 {np.percentile(moved, 50) * 1000:.1f} mm / "
                f"max {max(moved) * 1000:.1f} mm")
    return known


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path,
                    help="a nav_<hhmmss>/ folder inside a run folder "
                         "(or sessions/nav_runs/<stamp>/ for pre-2026-08-14 runs)")
    ap.add_argument("--anchor", type=Path, default=Path("config/tag_map.yaml"),
                    help="the surveyed map to grow from (stays frozen)")
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("config/tag_map_full.yaml"))
    ap.add_argument("--tag-size", type=float, default=0.170)
    ap.add_argument("--min-obs", type=int, default=20,
                    help="observations before a tag is promoted")
    ap.add_argument("--max-spread", type=float, default=0.03,
                    help="max spread of a tag's observations, metres")
    ap.add_argument("--dup-sep", type=float, default=0.10,
                    help="observation clusters further apart than this are "
                         "two PHYSICAL tags sharing one id")
    ap.add_argument("--max-reproj-px", type=float, default=3.0)
    ap.add_argument("--exclude", default="",
                    help="comma list of ids to leave OUT of the map entirely "
                         "(detected, but known not to belong to the mat)")
    ap.add_argument("--refine", type=int, default=3,
                    help="refinement passes over the completed map")
    ap.add_argument("--min-known", type=int, default=3,
                    help="known tags a frame needs before its camera pose is "
                         "trusted to define new geometry")
    ap.add_argument("--nav-config", type=Path,
                    default=Path("config/hw_nav.yaml"),
                    help="read duplicate_ids from here")
    ap.add_argument("--duplicate-ids", default=None,
                    help="comma list, overrides --nav-config")
    args = ap.parse_args(argv)

    from ..control.tagnav import tag_object_points

    obj_tag = tag_object_points(args.tag_size).astype(np.float64)
    frames = load_recording(args.run_dir)
    anchor_poses, anchor_id = load_anchor_map(args.anchor)
    n_anchor = len(anchor_poses)

    if args.duplicate_ids is not None:
        dup_in = {int(v) for v in args.duplicate_ids.split(",") if v.strip()}
    else:
        import yaml
        try:
            raw = yaml.safe_load(open(args.nav_config, encoding="utf-8")) or {}
            dup_in = {int(v) for v in (raw.get("duplicate_ids") or ())}
        except OSError:
            dup_in = set()

    # A duplicated id may not localize a frame, so it leaves the anchor set
    # and gets rediscovered from scratch — however many instances the data
    # actually shows.
    known = {t: p for t, p in anchor_poses.items() if t not in dup_in}
    pulled = sorted(set(anchor_poses) & dup_in)
    seen = {tid for fr in frames.values() for tid, _c in fr["dets"]}
    print(f"{args.run_dir}: {len(frames)} frames, "
          f"{sum(len(f['dets']) for f in frames.values())} detections, "
          f"{len(seen)} distinct ids")
    print(f"anchor map {args.anchor}: {n_anchor} tags, "
          f"{len(known)} frozen anchors")
    if dup_in:
        print(f"declared duplicate ids: {sorted(dup_in)}"
              + (f" ({pulled} pulled out of the anchor set for rediscovery)"
                 if pulled else ""))
    print(f"ids to solve in this recording: {len(seen - set(known))}")

    frozen_ids = set(known)
    drop = {int(v) for v in args.exclude.split(",") if v.strip()}
    if drop:
        for fr in frames.values():
            fr["dets"] = [d for d in fr["dets"] if d[0] not in drop]
        known = {t: p for t, p in known.items() if t not in drop}
        print(f"excluded ids (not part of the mat): {sorted(drop)}")
    known, obs, rounds = build(frames, known, obj_tag, args)
    known = refine(frames, known, frozen_ids, obj_tag, args)

    # Where a rediscovered cluster agrees with the survey, keep the SURVEYED
    # pose: the gantry measured it, this tool only estimated it.
    n_snap = 0
    for tid in pulled:
        if tid not in known:
            continue
        R_a, t_a = anchor_poses[tid][0]
        known[tid] = [((R_a, t_a) if np.linalg.norm(t - t_a) <= args.max_spread
                       else (R, t)) for R, t in known[tid]]
        n_snap += sum(1 for _R, t in known[tid] if np.allclose(t, t_a))
    if pulled:
        print(f"rediscovered {len(pulled)} duplicated id(s); {n_snap} "
              f"instance(s) matched the survey and kept its pose")

    new = {t: p for t, p in known.items() if t not in anchor_poses}
    dups = sorted(t for t, p in known.items() if len(p) > 1)
    failed = sorted(t for t in seen - set(known))
    n_rounds = max((r[0] for r in rounds.values()), default=0)
    print(f"\nPROMOTED {len(new)} new tags in {n_rounds} round(s) "
          f"-> {len(known)} ids, "
          f"{sum(len(v) for v in known.values())} physical tags")
    if dups:
        print(f"ids with TWO physical instances: {dups}")
    if failed:
        print(f"NOT promoted ({len(failed)}): "
              + ", ".join(f"{t}({len(obs.get(t, []))} obs)" for t in failed))
        print("  -> these were never seen together with enough known tags; "
              "fly over them again with the surveyed area in view")

    tags = {}
    for tid, poses in sorted(known.items()):
        for k, (R, t) in enumerate(poses):
            key = int(tid) if len(poses) == 1 else f"{int(tid)}#{k}"
            tags[key] = {"position_m": [float(v) for v in t],
                         "quaternion_wxyz": R_to_quat_wxyz(R)}
    doc = {
        "anchor_tag_id": anchor_id,
        "tags": tags,
        "duplicate_ids": dups,
        "metadata": {
            "source": f"{args.anchor} (frozen) grown over {args.run_dir}",
            "tool_version": "rov_gui.tools.build_tag_map 1.0",
            "n_anchor_tags": n_anchor,
            "n_new_tags": len(new),
            "n_frames": len(frames),
            "min_obs": args.min_obs,
            "min_known": args.min_known,
            "max_spread_m": args.max_spread,
            "dup_sep_m": args.dup_sep,
            "tag_size_m": args.tag_size,
            "not_promoted": failed,
            "excluded_ids": sorted(drop),
            "refine_passes": args.refine,
            "n_obs": {int(t): len(v) for t, v in sorted(obs.items())},
            "n_inliers": {int(t): list(r[1]) for t, r in sorted(rounds.items())},
        },
    }
    import yaml
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"\nwrote {args.out}")
    print("Anchor tags were copied through UNCHANGED — diff against the "
          "anchor map to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
