#!/usr/bin/env python3
"""extract_pose.py — offline 6-DoF pose labels for a handheld demonstration session.

Reads a recorded session (``sessions/demonstration_NNNN/``: ``left/<t>.png``
mono8 rectified-left frames + ``frames.csv`` + ``session.json``) and emits a
per-frame pose in the tag-map NED frame by running AprilTag PnP against a
surveyed tag map. The output is the training/replay label track for the
diffusion-policy pipeline:

    poses.npy    float64 (T, 8): [t_unix, x, y, z, qw, qx, qy, qz]
                 Row i corresponds to frames.csv row i — the SAME alignment
                 convention as gripper_width.npy (build_zarr.py checks
                 len == len(frames.csv rows) and zips them by index). A frame
                 with no accepted fix is all-NaN except t_unix.
    poses.json   schema "umi_handheld_poses/1": full provenance + run stats.

    python -m umi_handheld.extract_pose sessions/demonstration_0026

WHAT IT REUSES, AND WHY
-----------------------
Detection + PnP come from ``rov_gui/control/tagnav.py`` — the module the ROV
uses for its own fixes, and the one ``src/gantry_map_pose.py`` already replays
offline. Not a reimplementation: same corner ordering, same tag object points
(verbatim from src/tagslam_core.py:233), same map file format, same
reject-reason strings. ``tagnav`` and ``rov_gui.control.geometry`` import only
stdlib + numpy at module scope (verified 2026-08-30 by importing both in the
`robust` env and inspecting sys.modules: no PyQt/PySide/cv2 loaded), so this
tool stays Qt-free.

Intrinsics come from the session's OWN ``session.json`` (the
``camera_model_source`` block written at record time), not from whatever the
configs/*.yaml currently says — the frozen record beats a file that may have
been edited since. The source model is a rectified pinhole with zero
distortion, so ``dist=None`` and there is no undistort step
(umi_handheld/camera_model.py; configs/source_camera_air.yaml).

FRAME CONVENTIONS (see rov_gui/control/geometry.py's cheat-sheet)
-----------------------------------------------------------------
* map frame: tag +z INTO the printed face; floor tags print-up => map is
  NED-like, so ``R_ned_map`` is identity here (same as gantry_map_pose.py).
* ``--extrinsic camera`` (identity extrinsic): the pose of the camera optical
  frame itself, exactly what gantry_map_pose.py:167-181 does.
* ``--extrinsic c3`` (default): composes the measured C3 mount extrinsic
  (bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json, Onshape
  registration 2026-07-19) INSIDE TagNav.solve via its R_frd_cam/t_frd_cam
  parameters, so the output row is where the BlueROV BODY (FRD) origin would
  be if this camera were the C3 mounted on it. That is an ASSUMPTION about
  the handheld rig, recorded as such in poses.json.

TAG-MAP SAFETY
--------------
The default map is config/tag_map.yaml and must NEVER silently become
config/tag_map_full.yaml: the handheld gripper fingers carry tag36h11 ids 0
and 1 at 21.37 mm, while tag_map_full.yaml holds ids 0/1 as 170 mm FLOOR
tags — a finger detection solved against a floor pose is a silent ~8x scale
blowup. ``--exclude-ids 0,1`` (default) therefore drops those ids BEFORE the
solve regardless of which map is loaded, and a loaded map that still contains
a non-excluded 0/1 gets a loud warning.

SINGLE-TAG POLICY
-----------------
The handheld rig has no IMU, so the single-tag IPPE flip ambiguity has no
arbiter (TagNav's gravity gate needs an rp_hint this rig cannot provide).
Default is therefore ``--min-tags 2``; ``--allow-single-tag`` lowers the gate
to 1 and the affected rows are listed in poses.json (``single_tag_rows`` /
``ambiguous_rows``) so a training consumer can weight or drop them.

Zero fixes is a VALID outcome (the existing 25 demos likely contain no
environment tags): the tool writes an all-NaN track, warns loudly, and exits
0. Non-zero exit is reserved for hard errors (missing files, no readable
frames).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.camera_model import REPO, CameraModel            # noqa: E402
# Qt-free imports, verified (module docstring): geometry pulls json/math/
# dataclasses/pathlib/numpy only; tagnav is stdlib+numpy with lazy cv2.
from rov_gui.control.geometry import (                             # noqa: E402
    R_flu_cam_from_xyaxes, S_FLU_FRD,
)
from rov_gui.control.tagnav import (                               # noqa: E402
    Detection, TagDetector, TagMap, TagNav,
)

SCHEMA = "umi_handheld_poses/1"
COLUMNS = ("t_unix", "x", "y", "z", "qw", "qx", "qy", "qz")

# The measured C3 mount registration (Onshape export + registration 2026-07-19,
# process_c3_mesh.py; base_link FLU, origin = vehicle COM). Same file
# rov_gui/control/geometry.py:NavConfig.load reads for the live station.
C3_EXTRINSIC_JSON = "bluerov2_mujoco_marinegym/meshes/c3_payload_frames.json"

# The finger tags: tag36h11 ids 0 and 1 at 21.37 mm on the gripper jaws
# (umi_handheld/extract_gripper_width.py's markers). They are hand-relative,
# never world-fixed, and id-collide with tag_map_full.yaml's 170 mm floor
# tags — excluded from PnP by default for both reasons.
DEFAULT_EXCLUDE_IDS = (0, 1)


# =============================================================== small helpers
def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (body->parent) -> unit quaternion (w, x, y, z), w >= 0.

    Inverse of tagnav.quat_wxyz_to_R; Shepperd's branch selection keeps the
    conversion well-conditioned for every rotation, and the w >= 0 canonical
    sign makes the output track continuous to compare/serialize.
    """
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def load_c3_extrinsic(json_path=None):
    """(R_frd_cam, t_frd_cam, source_path) for the C3 mount.

    Same 4 lines of composition as geometry.NavConfig.R_t_frd_cam with
    cam_tilt_deg = 0 (the forward-level mount the registration measured):
    xyaxes -> R_flu_cam, then the FLU->FRD mirror S = diag(1,-1,-1) on both
    the rotation and the lever arm.
    """
    path = Path(json_path) if json_path else REPO / C3_EXTRINSIC_JSON
    d = json.loads(path.read_text(encoding="utf-8"))
    t_flu = np.asarray(d["cam_center_bl"], float)
    R_flu_cam = R_flu_cam_from_xyaxes(d["cam_xyaxes"])
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        rel = str(path)
    return S_FLU_FRD @ R_flu_cam, S_FLU_FRD @ t_flu, rel


def build_nav(tag_map: TagMap, tag_size_m: float, extrinsic: str = "c3",
              min_tags: int = 2, max_reproj_px: float = 3.0,
              extrinsic_json=None) -> tuple[TagNav, dict]:
    """A TagNav wired for offline handheld replay, plus its provenance block.

    ``extrinsic="camera"`` passes identity so NavSolution IS the camera pose
    (the gantry_map_pose.py pattern); ``"c3"`` passes the measured mount so
    the composition happens inside TagNav._solution — one extrinsic, one
    code path, never a post-hoc matrix multiply here.
    """
    if extrinsic == "camera":
        R_bc, t_bc = np.eye(3), np.zeros(3)
        ext = {"mode": "camera",
               "source": "identity — the pose of the camera optical frame "
                         "itself (src/gantry_map_pose.py:167-181 pattern)"}
    elif extrinsic == "c3":
        R_bc, t_bc, src = load_c3_extrinsic(extrinsic_json)
        ext = {"mode": "c3", "source": src,
               "assumption": ("the handheld camera is ASSUMED to sit at the "
                              "C3 mount pose; the output is the EQUIVALENT "
                              "vehicle-body (FRD) pose, not a measured one")}
    else:
        raise ValueError(f"extrinsic must be 'camera' or 'c3', got {extrinsic!r}")
    ext["R_frd_cam"] = np.asarray(R_bc, float).tolist()
    ext["t_frd_cam"] = np.asarray(t_bc, float).tolist()
    nav = TagNav(tag_map, float(tag_size_m),
                 R_frd_cam=R_bc, t_frd_cam=t_bc,
                 # The floor map is already NED-like (+z into the printed
                 # face, tags print-up): geometry.py uses identity for the
                 # floor geometry and so does gantry_map_pose.py.
                 R_ned_map=np.eye(3),
                 max_reproj_px=float(max_reproj_px),
                 min_tags=int(min_tags),
                 datum="map")               # absolute map coords, no re-zero
    return nav, ext


def _reject_key(reason: str) -> str:
    """Collapse a per-frame reject string to a countable bucket.

    tagnav's reasons embed frame-specific numbers ("reproj 5.1px > 3px
    (3 unique tags)"); replacing every number with '#' keeps the reason's
    shape while letting identical failure modes aggregate.
    """
    return re.sub(r"\d+(?:\.\d+)?", "#", reason) if reason else "unknown"


# ================================================================ core pipeline
def extract_poses(frames, detect_fn, nav: TagNav, K: np.ndarray, dist=None,
                  exclude_ids=frozenset()):
    """The per-frame pipeline, with the detector injected for testability.

    frames     iterable of (t_unix, payload). The payload is whatever
               ``detect_fn`` understands — an image for the CLI, a
               pre-built detection list for the tests.
    detect_fn  payload -> list[Detection], or None meaning "frame
               unreadable" (recorded as its own reject reason, never
               conflated with "no tags").
    nav        a TagNav from build_nav(); its min_tags / reproj gates and
               extrinsic composition all run inside nav.solve.
    K, dist    intrinsics matching the frames. dist=None for the rectified
               zero-distortion source model.

    Returns (poses, frame_stats): poses is float64 (T, 8) rows
    [t_unix, x, y, z, qw, qx, qy, qz], NaN except t_unix where no fix was
    accepted; frame_stats is one dict per row for the summary.
    """
    exclude_ids = frozenset(int(i) for i in exclude_ids)
    rows, stats = [], []
    for i, (t_unix, payload) in enumerate(frames):
        dets = detect_fn(payload)
        rec = {"row": i, "t_unix": float(t_unix), "fix": False,
               "n_detected": 0, "n_tags": 0, "tag_ids": (),
               "reproj_rms_px": None, "ambiguous": False, "reject": ""}
        if dets is None:
            sol, rec["reject"] = None, "frame missing/unreadable"
        else:
            rec["n_detected"] = len(dets)
            dets = [d for d in dets if int(d.tag_id) not in exclude_ids]
            # No rp_hint: the handheld rig has no gravity-referenced
            # attitude, so the single-tag IPPE flip has no arbiter — that is
            # why min_tags defaults to 2 upstream.
            sol = nav.solve(dets, K, dist)
            if sol is None:
                rec["reject"] = nav.last_reject or "solve returned None"
        if sol is None:
            rows.append([float(t_unix)] + [math.nan] * 7)
        else:
            q = R_to_quat_wxyz(sol.R_ned_body)
            p = np.asarray(sol.p_ned, float).ravel()
            rows.append([float(t_unix), p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
            rec.update(fix=True, n_tags=int(sol.n_tags),
                       tag_ids=tuple(int(x) for x in sol.tag_ids),
                       reproj_rms_px=float(sol.reproj_rms_px),
                       ambiguous=bool(sol.ambiguous))
        stats.append(rec)
    poses = (np.asarray(rows, dtype=np.float64) if rows
             else np.empty((0, 8), dtype=np.float64))
    return poses, stats


def summarize(frame_stats: list) -> dict:
    """Aggregate stats for poses.json — every number computed from the run."""
    n = len(frame_stats)
    fixes = [r for r in frame_stats if r["fix"]]
    rej = {}
    for r in frame_stats:
        if not r["fix"]:
            key = _reject_key(r["reject"])
            rej[key] = rej.get(key, 0) + 1
    reproj = np.asarray([r["reproj_rms_px"] for r in fixes], float)
    return {
        "n_frames": n,
        "n_fix": len(fixes),
        "fix_rate": (len(fixes) / n) if n else 0.0,
        "reproj_rms_px": ({"p50": float(np.percentile(reproj, 50)),
                           "p90": float(np.percentile(reproj, 90))}
                          if len(fixes) else None),
        "mean_tags_per_fix": (float(np.mean([r["n_tags"] for r in fixes]))
                              if fixes else None),
        "n_single_tag_fixes": sum(1 for r in fixes if r["n_tags"] == 1),
        "n_ambiguous_fixes": sum(1 for r in fixes if r["ambiguous"]),
        "reject_counts": dict(sorted(rej.items(), key=lambda kv: -kv[1])),
    }


# ================================================================== session IO
def camera_from_session(session: dict, session_json_path: Path) -> CameraModel:
    """A CameraModel rebuilt from the session's own camera_model_source block.

    session.json is the frozen record of the model that described the frames
    at record time; the configs/*.yaml it came from may have been edited
    since. Rebuilding a CameraModel (rather than re-loading the YAML) keeps
    the .K(width, height) rescaling logic while pinning the numbers to the
    recording.
    """
    blk = session.get("camera_model_source")
    if not isinstance(blk, dict):
        raise KeyError("session.json has no camera_model_source block")
    cfg = {
        "name": blk.get("name", "session"),
        "role": blk.get("role", "source"),
        "medium": blk.get("medium", "unknown"),
        "verified": bool(blk.get("verified", False)),
        "rectified": bool(blk.get("rectified", False)),
        "image_size": blk["image_size"],
        "intrinsics": {k: blk[k] for k in ("fx", "fy", "cx", "cy")},
        "distortion": blk.get("distortion", [0.0] * 8),
        "stereo": {"baseline_m": blk.get("baseline_m", 0.075)},
    }
    return CameraModel(cfg, session_json_path)


def read_frames_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows):
        if "t_unix" not in row or "left_file" not in row:
            raise ValueError(f"{path}: row {i} lacks t_unix/left_file")
    return rows


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ========================================================================== CLI
def _parse_ids(text: str) -> tuple:
    text = (text or "").strip()
    if not text or text.lower() == "none":
        return ()
    return tuple(sorted({int(v) for v in text.split(",") if v.strip() != ""}))


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m umi_handheld.extract_pose",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path,
                    help="sessions/demonstration_NNNN directory")
    ap.add_argument("--tag-map", default="config/tag_map.yaml",
                    help="surveyed tag map (default config/tag_map.yaml). Do "
                         "NOT point at tag_map_full.yaml casually: it holds "
                         "ids 0/1 as 170 mm floor tags while the gripper "
                         "fingers carry the same ids at 21.37 mm — a silent "
                         "~8x scale blowup unless --exclude-ids keeps them out.")
    ap.add_argument("--tag-size", type=float, default=0.170,
                    help="environment tag edge in metres (must match the tags "
                         "actually in frame; the surveyed floor map is 0.170)")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--min-tags", type=int, default=2,
                    help="minimum unique mapped tags per fix (default 2: the "
                         "handheld rig has no IMU to arbitrate the single-tag "
                         "IPPE ambiguity)")
    ap.add_argument("--allow-single-tag", action="store_true",
                    help="accept 1-tag fixes (rows are flagged in poses.json)")
    ap.add_argument("--exclude-ids", default="0,1",
                    help="comma-separated tag ids dropped before PnP "
                         "(default 0,1: the gripper finger tags)")
    ap.add_argument("--extrinsic", choices=("camera", "c3"), default="c3",
                    help="'camera' = pose of the camera optical frame; 'c3' "
                         "(default) = the equivalent vehicle BODY (FRD) pose, "
                         "composing the measured C3 mount extrinsic inside "
                         "TagNav")
    ap.add_argument("--max-reproj-px", type=float, default=3.0,
                    help="joint-PnP acceptance gate (tagnav default)")
    ap.add_argument("--quad-decimate", type=float, default=1.0,
                    help="detector quad decimation (1.0 = full resolution)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .npy path (default <session_dir>/poses.npy; "
                         "poses.json is written beside it)")
    return ap


def main(argv=None) -> int:
    a = make_parser().parse_args(argv)

    session_dir = a.session_dir
    if not session_dir.is_dir():
        print(f"ERROR: {session_dir} is not a directory", file=sys.stderr)
        return 2
    frames_csv = session_dir / "frames.csv"
    session_json = session_dir / "session.json"
    for need in (frames_csv, session_json):
        if not need.exists():
            print(f"ERROR: missing {need}", file=sys.stderr)
            return 2

    map_path = Path(a.tag_map)
    if not map_path.is_absolute() and not map_path.exists():
        map_path = REPO / a.tag_map          # repo-relative default works anywhere
    if not map_path.exists():
        print(f"ERROR: tag map not found: {a.tag_map}", file=sys.stderr)
        return 2

    exclude_ids = _parse_ids(a.exclude_ids)
    min_tags = 1 if a.allow_single_tag else max(1, int(a.min_tags))

    session = json.loads(session_json.read_text(encoding="utf-8"))
    try:
        cam = camera_from_session(session, session_json)
    except KeyError as e:
        print(f"ERROR: {session_json}: {e}", file=sys.stderr)
        return 2
    # Rectified pinhole with all-zero distortion -> dist=None, no undistort
    # (the source model contract; camera_model.py module docstring).
    dist = None if not np.any(cam.dist) else cam.dist

    rows = read_frames_csv(frames_csv)
    tag_map = TagMap.load(map_path)
    hot = sorted(set(exclude_ids) & set(tag_map.instances))
    if hot:
        print(f"note: excluded id(s) {hot} are present in {map_path.name}; "
              f"they will NOT be used for PnP")
    finger_in_map = sorted(set(DEFAULT_EXCLUDE_IDS) & set(tag_map.instances)
                           - set(exclude_ids))
    if finger_in_map:
        print(f"WARNING: map {map_path.name} contains id(s) {finger_in_map}, "
              f"which are ALSO the 21.37 mm gripper finger tags, and they are "
              f"NOT excluded — a finger detection solved against a 170 mm "
              f"floor pose is a ~8x scale error. Pass --exclude-ids 0,1.",
              file=sys.stderr)

    try:
        nav, ext = build_nav(tag_map, a.tag_size, a.extrinsic,
                             min_tags=min_tags, max_reproj_px=a.max_reproj_px)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot build the '{a.extrinsic}' extrinsic: {e}",
              file=sys.stderr)
        return 2
    detector = TagDetector(family=a.family, quad_decimate=float(a.quad_decimate))

    # Frame size from the first readable frame; K is rescaled to it the same
    # way every other consumer of this model does (CameraModel.K).
    import cv2
    first = None
    for row in rows:
        img = cv2.imread(str(session_dir / row["left_file"]),
                         cv2.IMREAD_GRAYSCALE)
        if img is not None:
            first = img
            break
    if rows and first is None:
        print(f"ERROR: none of the {len(rows)} left frames in {frames_csv} "
              f"could be read", file=sys.stderr)
        return 2
    frame_hw = first.shape[:2] if first is not None else (cam.height, cam.width)
    K = cam.K(frame_hw[1], frame_hw[0])

    def frames_iter():
        for row in rows:
            yield float(row["t_unix"]), session_dir / row["left_file"]

    n_total = len(rows)
    done = {"n": 0}

    def detect(path):
        done["n"] += 1
        if done["n"] % 50 == 0 or done["n"] == n_total:
            print(f"\r  {done['n']}/{n_total} frames", end="", file=sys.stderr)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if img.shape[:2] != frame_hw:
            # K was scaled for frame_hw; detecting on another size would
            # silently use wrong intrinsics — reject the frame instead.
            return None
        return detector.detect(img)

    print(f"session    : {session_dir}")
    print(f"frames     : {n_total}  ({frame_hw[1]}x{frame_hw[0]})")
    print(f"tag map    : {map_path}  anchor={tag_map.anchor_id} "
          f"tags={len(tag_map)}")
    print(f"tag size   : {a.tag_size:.4f} m   family {a.family}   "
          f"detector {detector.backend}")
    print(f"gates      : min_tags={min_tags}"
          f"{' (--allow-single-tag)' if a.allow_single_tag else ''}  "
          f"reproj<={a.max_reproj_px:g}px  exclude_ids={list(exclude_ids)}")
    print(f"extrinsic  : {a.extrinsic}  ({ext['source']})")

    poses, frame_stats = extract_poses(frames_iter(), detect, nav, K,
                                       dist=dist, exclude_ids=exclude_ids)
    if n_total:
        print(file=sys.stderr)
    stats = summarize(frame_stats)

    out_npy = a.out if a.out is not None else session_dir / "poses.npy"
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_npy.with_suffix(".json")
    np.save(out_npy, poses)

    meta = {
        "schema": SCHEMA,
        "pose_of": "camera" if a.extrinsic == "camera" else "body_frd",
        "frame": "map_ned",
        "columns": list(COLUMNS),
        "row_alignment": ("row i corresponds to frames.csv row i (same "
                          "convention as gripper_width.npy; no-fix rows are "
                          "NaN except t_unix)"),
        "quaternion": ("wxyz, w >= 0; R = quat_wxyz_to_R(q) maps body->map "
                       "(rov_gui/control/tagnav.py convention)"),
        "session": str(session_dir),
        "tag_map": {"path": str(map_path), "sha1": _sha1(map_path),
                    "anchor_tag_id": tag_map.anchor_id,
                    "n_tags": len(tag_map)},
        "tag_size_m": float(a.tag_size),
        "family": a.family,
        "detector_backend": detector.backend,
        "quad_decimate": float(a.quad_decimate),
        "excluded_ids": list(exclude_ids),
        "min_tags": min_tags,
        "allow_single_tag": bool(a.allow_single_tag),
        "max_reproj_px": float(a.max_reproj_px),
        "extrinsic": ext,
        "camera_model": {
            "provenance": ("session.json camera_model_source — the model "
                           "recorded with the frames, not the current "
                           "configs/*.yaml"),
            "frame_size_used": [int(frame_hw[1]), int(frame_hw[0])],
            "dist_used": None if dist is None else np.ravel(dist).tolist(),
            **session["camera_model_source"],
        },
        "assumptions": [
            "tag_size_m is a CLI input, not read from the image; it must "
            "match the physical environment tags in frame",
            "no refraction model: in-air rectified frames with the in-air "
            "source model (dist=None)",
        ] + ([ext["assumption"]] if "assumption" in ext else []),
        "stats": stats,
        "single_tag_rows": [r["row"] for r in frame_stats
                            if r["fix"] and r["n_tags"] == 1],
        "ambiguous_rows": [r["row"] for r in frame_stats
                           if r["fix"] and r["ambiguous"]],
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    p50 = (f"{stats['reproj_rms_px']['p50']:.2f} px"
           if stats["reproj_rms_px"] else "n/a")
    print(f"\nposes      : {out_npy}  shape {tuple(poses.shape)}")
    print(f"sidecar    : {out_json}")
    print(f"fixes      : {stats['n_fix']}/{stats['n_frames']}  "
          f"({100.0 * stats['fix_rate']:.1f}%)   reproj p50 {p50}")
    if stats["n_frames"] and stats["n_fix"] == 0:
        top = list(stats["reject_counts"].items())[:3]
        print("WARNING: 0 fixes — no mapped environment tag was accepted in "
              "any frame. This is a valid outcome for sessions recorded "
              "without environment tags in view; the pose track is all-NaN.")
        if top:
            print("  top reject reasons: "
                  + ";  ".join(f"{k} x{v}" for k, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
