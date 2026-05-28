#!/usr/bin/env python3
"""survey_tags.py — batch-optimize an existing recording into a reusable tag map.

This is a *post-processing* CLI: it does NO camera capture and NO gantry control.
You first produce a survey-grade recording with the panel (Recording tab + manual
jog), then run this tool to batch-optimize all tag observations into
``config/tag_map.yaml`` for future PnP-only localization.

    python -m src.survey_tags --input-dir data/YYYYMMDD/<ts>_recording \
                              --output config/tag_map.yaml \
                              [--anchor-tag-id 70] \
                              [--min-observations 10] \
                              [--max-iterations 200] \
                              [--use-frames]

Two observation sources:
  * CSV-only (default, fast): reconstruct camera_T_tag from the recorded
    camera_trajectory.csv poses + tag_poses.csv (the ``detected_tags`` column
    says which tags were active each frame). These observations are consistent
    with the recorded poses by construction, so the optimizer mainly confirms
    consistency and computes per-tag uncertainty from the observation counts.
  * --use-frames (slower): re-detect AprilTags in the saved frames and run
    solvePnP for an independent camera_T_tag per detection. NOTE: the recorder
    saves *undistorted, downscaled* JPEG frames, so the rectified intrinsics are
    scaled to the saved resolution and re-detection accuracy is bounded by it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gtsam  # noqa: E402
from gtsam import (  # noqa: E402
    BetweenFactorPose3,
    NonlinearFactorGraph,
    Point3,
    Pose3,
    PriorFactorPose3,
    Rot3,
    Values,
)
from gtsam.symbol_shorthand import L, X  # noqa: E402

from tagslam_core import (  # noqa: E402
    DEFAULT_TAG_SIZE_M,
    make_floor_prior_noise,
    make_pose_noise,
    parse_simple_yaml,
)

# ── constants (per spec) ──────────────────────────────────────────────────────
MIN_OBSERVATIONS_PER_TAG = 10
OPTIMIZER_MAX_ITERATIONS = 200
OPTIMIZER_RELATIVE_TOL = 1e-6
OPTIMIZER_ABSOLUTE_TOL = 1e-8
ANCHOR_PRIOR_SIGMA = 1e-6
TAG_BETWEEN_ROT_SIGMA_RAD = 0.08
TAG_BETWEEN_TRANS_SIGMA_M = 0.04
FLOOR_Z_SIGMA_M = 0.05  # only used when --floor-coplanar is set

TOOL_VERSION = "survey_tags.py 1.0"
EXIT_OK = 0
EXIT_USAGE = 2

_DEFAULT_CALIB = Path("config/fisheye_calibration.yaml")
_DEFAULT_CONFIG = Path("config/config.yaml")


# ── small helpers ─────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pose_from_xyz_rpy_deg(x, y, z, roll, pitch, yaw) -> Pose3:
    """Pose3 from a CSV row. Inverse of pose_translation()/pose_rpy() which write
    ``np.degrees(pose.rotation().rpy())`` — Rot3.RzRyRx round-trips rpy()."""
    rot = Rot3.RzRyRx(math.radians(float(roll)),
                      math.radians(float(pitch)),
                      math.radians(float(yaw)))
    return Pose3(rot, Point3(float(x), float(y), float(z)))


def _quat_wxyz(rot: Rot3) -> list[float]:
    """Quaternion [w, x, y, z] across GTSAM wrapper versions."""
    try:
        q = rot.toQuaternion()
        return [float(q.w()), float(q.x()), float(q.y()), float(q.z())]
    except Exception:
        q = np.asarray(rot.quaternion(), dtype=np.float64).reshape(-1)
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _parse_tag_ids(text: str) -> list[int]:
    out = []
    for tok in str(text or "").split():
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


# ── input loading ─────────────────────────────────────────────────────────────
def load_tag_poses(path: Path) -> dict[int, Pose3]:
    poses: dict[int, Pose3] = {}
    for r in _read_csv(path):
        try:
            tag_id = int(float(r["tag_id"]))
            poses[tag_id] = _pose_from_xyz_rpy_deg(
                r["x_m"], r["y_m"], r["z_m"], r["roll_deg"], r["pitch_deg"], r["yaw_deg"])
        except (KeyError, ValueError, TypeError):
            continue
    return poses


def load_camera_rows(path: Path) -> list[dict]:
    """Return [{frame, pose, tags, image_path, has_tag_update}] for rows that
    carry a usable camera pose."""
    rows = []
    for i, r in enumerate(_read_csv(path)):
        try:
            pose = _pose_from_xyz_rpy_deg(
                r["x_m"], r["y_m"], r["z_m"], r["roll_deg"], r["pitch_deg"], r["yaw_deg"])
        except (KeyError, ValueError, TypeError):
            continue
        try:
            frame = int(float(r.get("camera_index", i)))
        except (ValueError, TypeError):
            frame = i
        rows.append({
            "frame": frame,
            "pose": pose,
            "tags": _parse_tag_ids(r.get("detected_tags", "")),
            "image_path": str(r.get("image_path", "") or ""),
            "has_tag_update": str(r.get("has_tag_update", "")).strip().lower() in ("true", "1", "yes"),
        })
    return rows


def resolve_anchor(arg_anchor: int | None, tag_poses: dict[int, Pose3]) -> int:
    """Use --anchor-tag-id if given; else auto-detect the tag nearest the origin
    (the recording's world frame is pinned at the anchor)."""
    if arg_anchor is not None:
        return int(arg_anchor)
    best_id, best_d = None, float("inf")
    for tid, pose in tag_poses.items():
        d = float(np.linalg.norm(pose_translation_np(pose)))
        if d < best_d:
            best_id, best_d = tid, d
    if best_id is None:
        raise SystemExit("error: tag_poses.csv has no tags to anchor on")
    return int(best_id)


def pose_translation_np(pose: Pose3) -> np.ndarray:
    return np.asarray(pose.translation(), dtype=np.float64).reshape(3)


# ── observation builders ──────────────────────────────────────────────────────
def build_observations_csv(camera_rows, tag_poses):
    """Reconstruct (frame, tag_id, camera_T_tag) from recorded poses.

    Returns (observations, frame_init, tag_init, n_frames_used).
    """
    observations = []
    frame_init: dict[int, Pose3] = {}
    used_frames = set()
    for row in camera_rows:
        if not row["has_tag_update"]:
            continue
        wTc = row["pose"]
        n_here = 0
        for tag_id in row["tags"]:
            wTt = tag_poses.get(tag_id)
            if wTt is None:
                continue
            cTt = wTc.between(wTt)  # inverse(wTc) * wTt
            observations.append((row["frame"], tag_id, cTt))
            n_here += 1
        if n_here > 0:
            frame_init[row["frame"]] = wTc
            used_frames.add(row["frame"])
    return observations, frame_init, dict(tag_poses), len(used_frames)


def _detection_args(tag_size: float, tag_family: str, area_scale: float) -> argparse.Namespace:
    """Minimal Namespace with the fields make_detector / detect_observations read,
    using the live pipeline's default thresholds. ``area_scale`` scales the
    minimum-tag-area gate to the saved (downscaled) frame resolution so the same
    physical tags pass. Decoupled from the fisheye tool's gantry-coupled parser.
    """
    return argparse.Namespace(
        tag_family=str(tag_family),
        nthreads=2, quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25,
        max_tag_id=-1, max_hamming=0, min_decision_margin=30.0,
        min_tag_area_px=120.0 * max(0.05, float(area_scale)),
        water_correction_mode="none", water_scale=3.6,
        max_reprojection_error_px=5.0, max_off_nadir_deg=25.0,
        max_image_eccentricity=0.65, max_tag_tilt_deg=35.0,
        tag_size=float(tag_size),
    )


def build_observations_frames(camera_rows, tag_poses, calib, tag_size, tag_family):
    """Re-detect AprilTags in the saved frames and solvePnP each detection.

    Saved frames are the *undistorted* image downscaled to ~960 px wide, so we
    detect on them directly (no re-remap) with the rectified intrinsics scaled to
    each frame's resolution.

    Returns (observations, frame_init, tag_init, n_frames_processed).
    """
    import cv2
    from fisheye_gantry_tagslam import (
        build_fisheye_undistort_maps,
        rectified_camera_intrinsics,
    )
    from tagslam_core import detect_observations, make_detector, tag_object_points

    full_w, full_h = int(calib.image_size[0]), int(calib.image_size[1])
    # Rectified intrinsics for the *full* undistorted frame. Saved frames are this
    # undistorted image downscaled, so we scale K per loaded frame below.
    _m1, _m2, new_K = build_fisheye_undistort_maps(
        calib.K, calib.D, (full_w, full_h), 0.0)

    observations = []
    frame_init: dict[int, Pose3] = {}
    tag_init: dict[int, Pose3] = dict(tag_poses)
    n_processed = 0
    run_dir = input_dir_global

    detector = None
    object_points = None
    det_args = None

    for row in camera_rows:
        rel = row["image_path"]
        if not rel:
            continue
        img_path = (run_dir / rel) if not Path(rel).is_absolute() else Path(rel)
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_processed += 1
        h, w = img.shape[:2]
        sx, sy = w / float(full_w), h / float(full_h)
        if detector is None:  # first usable frame -> build detector with scaled gates
            det_args = _detection_args(tag_size, tag_family, sx * sy)
            detector = make_detector(det_args)
            object_points = tag_object_points(det_args.tag_size)
        K = new_K.astype(np.float64).copy()
        K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
        intrinsics = rectified_camera_intrinsics(K)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obs_list = detect_observations(gray, detector, intrinsics, object_points, det_args, None)
        wTc = row["pose"]
        frame_used = False
        for obs in obs_list:
            tag_id = int(obs.tag_id)
            observations.append((row["frame"], tag_id, obs.camera_T_tag))
            frame_used = True
            if tag_id not in tag_init:  # bootstrap a tag first seen here
                tag_init[tag_id] = wTc.compose(obs.camera_T_tag)
        if frame_used:
            frame_init[row["frame"]] = wTc
        if n_processed % 50 == 0:
            _log(f"[survey] re-detected {n_processed} frames…")
    return observations, frame_init, tag_init, n_processed


# ── factor graph + optimize ───────────────────────────────────────────────────
def reframe_to_anchor(anchor_pose: Pose3, frame_init, tag_init):
    """Rigidly transform all initial poses so the anchor starts at identity. Keeps
    the relative (between-factor) structure intact and conditions the solve well
    even when re-anchoring to a tag that was not the recording's origin."""
    T = anchor_pose.inverse()
    frame_init = {k: T.compose(v) for k, v in frame_init.items()}
    tag_init = {k: T.compose(v) for k, v in tag_init.items()}
    return frame_init, tag_init


def optimize(observations, frame_init, tag_init, anchor_id, min_obs,
             max_iters, floor_coplanar):
    """Build the batch graph, optimize, and return a results dict."""
    # qualify tags by observation count
    obs_count: dict[int, int] = {}
    for _frame, tag_id, _cTt in observations:
        obs_count[tag_id] = obs_count.get(tag_id, 0) + 1
    qualified = {t for t, n in obs_count.items() if n >= min_obs and t in tag_init}
    dropped = {t: n for t, n in obs_count.items() if t not in qualified}
    if anchor_id not in tag_init:
        raise SystemExit(f"error: anchor tag {anchor_id} not present in the recording")
    qualified.add(anchor_id)  # anchor always kept even if sparsely seen

    # qualify frames: keep frames with >=2 observations of qualifying tags
    per_frame: dict[int, int] = {}
    for frame, tag_id, _cTt in observations:
        if tag_id in qualified:
            per_frame[frame] = per_frame.get(frame, 0) + 1
    good_frames = {f for f, n in per_frame.items() if n >= 2 and f in frame_init}

    graph = NonlinearFactorGraph()
    init = Values()

    anchor_noise = make_pose_noise(ANCHOR_PRIOR_SIGMA, ANCHOR_PRIOR_SIGMA)
    tag_noise = make_pose_noise(TAG_BETWEEN_ROT_SIGMA_RAD, TAG_BETWEEN_TRANS_SIGMA_M)

    graph.add(PriorFactorPose3(L(anchor_id), Pose3(), anchor_noise))
    init.insert(L(anchor_id), Pose3())
    for tag_id in sorted(qualified):
        if tag_id == anchor_id:
            continue
        init.insert(L(tag_id), tag_init[tag_id])
    for frame in sorted(good_frames):
        init.insert(X(frame), frame_init[frame])

    if floor_coplanar:
        floor_noise = make_floor_prior_noise(FLOOR_Z_SIGMA_M, None)
        for tag_id in sorted(qualified):
            p = tag_init.get(tag_id, Pose3())
            t = pose_translation_np(p)
            floor_pose = Pose3(p.rotation(), Point3(float(t[0]), float(t[1]), 0.0))
            graph.add(PriorFactorPose3(L(tag_id), floor_pose, floor_noise))

    n_factors = 0
    for frame, tag_id, cTt in observations:
        if tag_id in qualified and frame in good_frames:
            graph.add(BetweenFactorPose3(X(frame), L(tag_id), cTt, tag_noise))
            n_factors += 1

    if n_factors == 0 or not good_frames:
        raise SystemExit("error: no qualifying observations to optimize "
                         "(need tags with >= min observations seen in frames with >= 2 tags)")

    initial_error = float(graph.error(init))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(max_iters))
    params.setRelativeErrorTol(OPTIMIZER_RELATIVE_TOL)
    params.setAbsoluteErrorTol(OPTIMIZER_ABSOLUTE_TOL)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init, params)

    prev = optimizer.error()
    iterations = 0
    converged = False
    while iterations < int(max_iters):
        optimizer.iterate()
        iterations += 1
        err = optimizer.error()
        if iterations == 1 or iterations % 10 == 0:
            _log(f"[survey] iter {iterations}: error {err:.4f}")
        if abs(prev - err) <= max(OPTIMIZER_ABSOLUTE_TOL,
                                  OPTIMIZER_RELATIVE_TOL * abs(prev)):
            converged = True
            break
        prev = err

    result = optimizer.values()
    final_error = float(graph.error(result))
    diverged = initial_error > 1e-6 and final_error > 2.0 * initial_error
    if diverged:
        converged = False
        _log(f"[survey] WARNING: optimizer diverged "
             f"(initial {initial_error:.3f} -> final {final_error:.3f}); saving anyway")

    # marginal covariance -> per-tag translational uncertainty (mm)
    uncertainty: dict[int, float] = {}
    try:
        marg = gtsam.Marginals(graph, result)
        for tag_id in sorted(qualified):
            cov = np.asarray(marg.marginalCovariance(L(tag_id)), dtype=np.float64)
            trans_cov = cov[3:6, 3:6]  # GTSAM Pose3 tangent: rot[0:3], trans[3:6]
            uncertainty[tag_id] = float(math.sqrt(max(0.0, np.trace(trans_cov))) * 1000.0)
    except Exception as exc:
        _log(f"[survey] WARNING: marginal covariance failed ({exc}); uncertainties unavailable")
        uncertainty = {t: float("nan") for t in qualified}

    tags_out = {}
    for tag_id in sorted(qualified):
        pose = result.atPose3(L(tag_id))
        tags_out[tag_id] = {
            "position_m": [round(float(v), 6) for v in pose_translation_np(pose)],
            "quaternion_wxyz": [round(v, 6) for v in _quat_wxyz(pose.rotation())],
            "n_observations": int(obs_count.get(tag_id, 0)),
            "uncertainty_mm": uncertainty.get(tag_id, float("nan")),
        }

    return {
        "tags": tags_out,
        "dropped": dropped,
        "obs_count": obs_count,
        "n_qualified": len(qualified),
        "n_dropped": len(dropped),
        "n_frames_in_graph": len(good_frames),
        "iterations": iterations,
        "initial_error": initial_error,
        "final_error": final_error,
        "converged": bool(converged),
    }


# ── outputs ───────────────────────────────────────────────────────────────────
def _yaml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "null"
        return f"{v:.6g}"
    if isinstance(v, (int,)):
        return str(v)
    return f'"{v}"'


def write_tag_map_yaml(path: Path, anchor_id: int, tags: dict, metadata: list[tuple]):
    lines = [f"anchor_tag_id: {anchor_id}", "", "tags:"]
    # anchor first, then ascending tag id
    order = [anchor_id] + [t for t in sorted(tags) if t != anchor_id]
    for tag_id in order:
        r = tags[tag_id]
        pos = r["position_m"]
        q = r["quaternion_wxyz"]
        unc = r["uncertainty_mm"]
        unc_str = "null" if not math.isfinite(unc) else f"{unc:.1f}"
        lines.append(f"  {tag_id}:")
        lines.append(f"    position_m: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
        lines.append(f"    quaternion_wxyz: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
        lines.append(f"    n_observations: {r['n_observations']}")
        lines.append(f"    uncertainty_mm: {unc_str}")
    lines.append("")
    lines.append("metadata:")
    for key, val in metadata:
        lines.append(f"  {key}: {_yaml_scalar(val)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(source, used_frames_redetection, n_frames_used, n_frames_total,
                 result, anchor_id, min_obs, output_path, layout_path):
    tags = result["tags"]
    obs_count = result["obs_count"]
    n_observed = len(set(obs_count))
    initial, final = result["initial_error"], result["final_error"]
    reduction = (1.0 - final / initial) * 100.0 if initial > 1e-9 else 0.0
    bar = "=" * 75
    print(bar)
    print("Tag Survey Report")
    print(bar)
    print(f"Source:            {source}")
    print(f"Frame redetection: {'YES' if used_frames_redetection else 'NO (using recorded observations)'}")
    print(f"Frames used:       {n_frames_used} / {n_frames_total}")
    print(f"Tags observed:     {n_observed}")
    print(f"Tags qualified:    {result['n_qualified']} (>= {min_obs} observations)")
    print(f"Anchor:            tag {anchor_id}")
    conv = f"converged in {result['iterations']} iterations" if result["converged"] \
        else f"did NOT converge ({result['iterations']} iterations)"
    print(f"Optimization:      {conv}")
    print(f"                   initial {initial:.1f} -> final {final:.1f} ({reduction:.1f}% reduction)")
    print()
    print("Per-tag results (sorted by uncertainty):")

    def _key(tid):
        u = tags[tid]["uncertainty_mm"]
        return (math.inf if not math.isfinite(u) else u)
    for tid in sorted(tags, key=_key):
        r = tags[tid]
        u = r["uncertainty_mm"]
        us = " nan" if not math.isfinite(u) else f"{u:5.1f}"
        star = "  ★ anchor" if tid == anchor_id else ""
        print(f"  Tag {tid:4d}:  obs={r['n_observations']:5d}  unc={us} mm{star}")
    for tid in sorted(result["dropped"]):
        n = result["dropped"][tid]
        print(f"  Tag {tid:4d}:  obs={n:5d}  ━ DROPPED (< {min_obs} obs)")
    print()
    if result["n_dropped"]:
        print(f"⚠ {result['n_dropped']} tag(s) dropped — drive the gantry through their region next time.")
    print()
    print(f"Output written to: {output_path}")
    if layout_path is not None:
        print(f"Layout plot:       {layout_path}")
    print(bar)


def plot_layout(path: Path, anchor_id: int, tags: dict, pool_cfg: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _log(f"[survey] layout plot skipped (matplotlib unavailable: {exc})")
        return None

    xs = [tags[t]["position_m"][0] for t in tags]
    ys = [tags[t]["position_m"][1] for t in tags]
    fig, ax = plt.subplots(figsize=(8, 6))

    # pool rectangle, centered on the tag centroid (placement is approximate)
    if xs and ys and pool_cfg:
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        L_m = float(pool_cfg.get("length_m", 4.877))
        W_m = float(pool_cfg.get("width_m", 1.8))
        if str(pool_cfg.get("pool_long_axis", "x")) == "x":
            half_x, half_y = L_m / 2.0, W_m / 2.0
        else:
            half_x, half_y = W_m / 2.0, L_m / 2.0
        ax.add_patch(plt.Rectangle((cx - half_x, cy - half_y), 2 * half_x, 2 * half_y,
                                   fill=False, edgecolor="#7a8794", linestyle="--", linewidth=1.0))

    def _color(u):
        if not math.isfinite(u):
            return "#9aa3ad"
        if u < 5.0:
            return "#2ca02c"
        if u <= 15.0:
            return "#e6b800"
        return "#d62728"

    for t in tags:
        x, y = tags[t]["position_m"][0], tags[t]["position_m"][1]
        u = tags[t]["uncertainty_mm"]
        if t == anchor_id:
            ax.scatter([x], [y], marker="*", s=320, c="#1f77b4", edgecolors="k",
                       zorder=5, label="anchor")
        else:
            ax.scatter([x], [y], marker="s", s=90, c=_color(u), edgecolors="k", zorder=4)
        ux = "n/a" if not math.isfinite(u) else f"{u:.1f}"
        ax.annotate(f"{t}\n{ux}mm", (x, y), textcoords="offset points",
                    xytext=(7, 6), fontsize=8)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Tag Map (top-down) — color = uncertainty (green<5, yellow 5-15, red>15 mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ── module-level handles (set in main, used by frame builder) ─────────────────
calib_path_global: Path = _DEFAULT_CALIB
input_dir_global: Path = Path(".")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="survey_tags",
        description="Batch-optimize a recording into config/tag_map.yaml (post-processing only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Recording folder (contains camera_trajectory.csv, tag_poses.csv).")
    p.add_argument("--output", type=Path, default=Path("config/tag_map.yaml"),
                   help="Output tag map YAML.")
    p.add_argument("--anchor-tag-id", type=int, default=None,
                   help="Anchor tag id (pinned at origin). Default: tag nearest origin.")
    p.add_argument("--min-observations", type=int, default=MIN_OBSERVATIONS_PER_TAG,
                   help="Minimum observations for a tag to be kept.")
    p.add_argument("--max-iterations", type=int, default=OPTIMIZER_MAX_ITERATIONS,
                   help="LevenbergMarquardt max iterations.")
    p.add_argument("--use-frames", action="store_true",
                   help="Re-detect AprilTags in saved frames (slower, independent).")
    p.add_argument("--calib-path", type=Path, default=_DEFAULT_CALIB,
                   help="Fisheye calibration YAML (intrinsics, for --use-frames).")
    p.add_argument("--config-path", type=Path, default=_DEFAULT_CONFIG,
                   help="config.yaml (pool outline for the layout plot).")
    p.add_argument("--tag-size", type=float, default=DEFAULT_TAG_SIZE_M,
                   help="AprilTag edge length (m), for --use-frames PnP.")
    p.add_argument("--tag-family", default="tag36h11",
                   help="AprilTag family, for --use-frames detection.")
    p.add_argument("--floor-coplanar", action="store_true",
                   help="Add a soft Z=0 co-planarity prior on every tag.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the tag_map_layout.png plot.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    global calib_path_global, input_dir_global
    args = parse_args(argv)

    input_dir = args.input_dir
    if not input_dir.is_dir():
        _log(f"error: input directory not found: {input_dir}")
        return EXIT_USAGE
    cam_csv = input_dir / "camera_trajectory.csv"
    tag_csv = input_dir / "tag_poses.csv"
    if not cam_csv.exists() or not tag_csv.exists():
        _log(f"error: {input_dir} is missing camera_trajectory.csv or tag_poses.csv")
        return EXIT_USAGE

    input_dir_global = input_dir
    calib_path_global = args.calib_path

    tag_poses = load_tag_poses(tag_csv)
    camera_rows = load_camera_rows(cam_csv)
    n_frames_total = len(camera_rows)
    if not tag_poses:
        _log("error: tag_poses.csv contained no tags")
        return EXIT_USAGE

    try:
        anchor_id = resolve_anchor(args.anchor_tag_id, tag_poses)
    except SystemExit as exc:
        _log(str(exc))
        return EXIT_USAGE
    if anchor_id not in tag_poses:
        _log(f"error: anchor tag {anchor_id} not present in tag_poses.csv")
        return EXIT_USAGE
    _log(f"[survey] anchor tag = {anchor_id}"
         + ("" if args.anchor_tag_id is not None else " (auto-detected nearest origin)"))

    # ── observations ──────────────────────────────────────────────────────────
    used_frames_redetection = False
    if args.use_frames:
        frames_dir = input_dir / "frames"
        if not frames_dir.is_dir():
            _log("[survey] WARNING: --use-frames set but frames/ missing — "
                 "falling back to CSV-only observations")
        else:
            try:
                calib = _load_calib(args.calib_path)
            except SystemExit as exc:
                _log(str(exc))
                return EXIT_USAGE
            _log("[survey] re-detecting AprilTags in saved frames…")
            observations, frame_init, tag_init, n_frames_used = build_observations_frames(
                camera_rows, tag_poses, calib, args.tag_size, args.tag_family)
            used_frames_redetection = True

    if not used_frames_redetection:
        observations, frame_init, tag_init, n_frames_used = build_observations_csv(
            camera_rows, tag_poses)

    if not observations:
        _log("error: no tag observations could be built from this recording")
        return EXIT_USAGE

    # ── reframe so anchor starts at identity, then optimize ───────────────────
    anchor_init = tag_init.get(anchor_id, Pose3())
    frame_init, tag_init = reframe_to_anchor(anchor_init, frame_init, tag_init)

    try:
        result = optimize(observations, frame_init, tag_init, anchor_id,
                          args.min_observations, args.max_iterations, args.floor_coplanar)
    except SystemExit as exc:
        _log(str(exc))
        return EXIT_USAGE

    # ── outputs ───────────────────────────────────────────────────────────────
    pool_cfg = {}
    try:
        if args.config_path.exists():
            pool_cfg = (parse_simple_yaml(args.config_path.read_text()) or {}).get("pool", {}) or {}
    except Exception:
        pool_cfg = {}

    layout_path = None
    if not args.no_plot:
        layout_path = args.output.with_name(args.output.stem + "_layout.png")
        layout_path = plot_layout(layout_path, anchor_id, result["tags"], pool_cfg)

    metadata = [
        ("source", str(input_dir)),
        ("used_frames_redetection", bool(used_frames_redetection)),
        ("n_frames_processed", int(n_frames_used)),
        ("n_tags_qualified", int(result["n_qualified"])),
        ("n_tags_dropped", int(result["n_dropped"])),
        ("min_observations", int(args.min_observations)),
        ("optimizer_iterations", int(result["iterations"])),
        ("initial_error", round(float(result["initial_error"]), 3)),
        ("final_error", round(float(result["final_error"]), 3)),
        ("converged", bool(result["converged"])),
        ("fisheye_calib_path", str(args.calib_path)),
        ("tool_version", TOOL_VERSION),
        ("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    ]
    write_tag_map_yaml(args.output, anchor_id, result["tags"], metadata)

    print_report(str(input_dir), used_frames_redetection, n_frames_used, n_frames_total,
                 result, anchor_id, args.min_observations, args.output, layout_path)
    return EXIT_OK


def _load_calib(path: Path):
    from fisheye_gantry_tagslam import load_fisheye_calibration
    if not path.exists():
        raise SystemExit(f"error: calibration file not found: {path}")
    return load_fisheye_calibration(path)


if __name__ == "__main__":
    raise SystemExit(main())
