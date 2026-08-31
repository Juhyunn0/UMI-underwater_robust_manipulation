#!/usr/bin/env python3
"""
test_gantry_map_pose.py — the gantry camera's position, in the ROV's frame.

    ~/miniforge3/envs/robust/bin/python src/tests/test_gantry_map_pose.py
    ~/miniforge3/envs/robust/bin/python -m pytest src/tests/test_gantry_map_pose.py -v

No Qt, no gantry, no camera. Two tiers:

  * SYNTHETIC (always runs) — place a virtual camera at a known map position,
    project the surveyed tags' corners into it with cv2.projectPoints, feed
    those corners back in as Detections, and require the pose to come back.
    This is what actually pins the frame convention: corner ordering, tag object
    points, the identity extrinsic, R_ned_map, and datum="map" all have to be
    right simultaneously or the recovered position is not the one we placed.

  * RECORDED (skips when data/ is absent — it is gitignored) — replay a real
    recording's frames and compare against its own camera_trajectory.csv. That
    file is in the run's per-run anchor frame (tag 67 for every 2026-05-28 run),
    so it is lifted into anchor-25 coordinates with world_T_tag67 from
    config/tag_map.yaml before comparing. Measured 2026-08-23 across 8 runs:
    p50 9-16 mm, p90 16-18 mm, 0 no-fix out of 640 sampled frames.

    That residual is expected and is NOT the readout's accuracy: this path is
    cv2 joint solvePnP over downscaled, annotated JPEGs, while the recording's
    path was GTSAM iSAM2 over full-resolution live frames. Two different
    estimators agreeing to ~1 cm on a map whose tags are pinned to 5-20 mm is
    the assembly being right, not the sensor being that good.
"""
from __future__ import annotations

import csv
import glob
import math
import os
from pathlib import Path
import sys

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
_REPO = _SRC.parent
for p in (str(_SRC), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gantry_map_pose as gmp                                    # noqa: E402
from rov_gui.control.tagnav import (                             # noqa: E402
    Detection, quat_wxyz_to_R, tag_object_points,
)

CALIB = _REPO / "config" / "fisheye_calibration.yaml"
TAG_MAP = _REPO / "config" / "tag_map.yaml"
RECORDING = _REPO / "data" / "20260528" / "20260528_215858_recording"

# The frozen survey, as the ROV also reads it (config/tag_map_full.yaml keeps
# these 47 frozen inside it and carries the same anchor).
EXPECTED_ANCHOR = 25
EXPECTED_N_TAGS = 47


# ============================================================== synthetic tier
def _pinhole_K(w=1280, h=720, f=1000.0) -> np.ndarray:
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def _project_map_tags(est, p_cam_map, R_map_cam, K, tag_ids):
    """Corners of `tag_ids` as a camera at (p_cam_map, R_map_cam) would see them.

    Mirrors TagNav._obj_img: map object points are (R_mt @ obj_tag.T).T + t_mt.
    """
    import cv2

    obj_tag = tag_object_points(gmp.DEFAULT_TAG_SIZE_M)
    R_cm = R_map_cam.T
    t_cm = -R_map_cam.T @ p_cam_map
    rvec, _ = cv2.Rodrigues(R_cm)
    dets = []
    for tid in tag_ids:
        R_mt, t_mt = est.tag_map.poses[tid]
        obj_map = (R_mt @ obj_tag.T).T + t_mt
        img, _ = cv2.projectPoints(obj_map.astype(np.float64), rvec,
                                   t_cm.astype(np.float64), K, None)
        dets.append(Detection(int(tid), img.reshape(4, 2).astype(np.float32), 99.0))
    return dets


def _down_looking_R() -> np.ndarray:
    """map_R_cam for a camera hanging above the floor, looking straight down.

    The map is NED-like (+z down), so "down" is map +z. Optical +z (forward)
    -> map +z, optical +x (image right) -> map +y, and optical +y closes the
    right-handed set at map -x.
    """
    return np.array([[0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])


def test_synthetic_roundtrip_recovers_the_placed_pose():
    """The whole frame convention, with no image involved."""
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    K = _pinhole_K()
    est._K_use = K                       # bypass rectification: corners are ideal
    est._maps = None
    est._frame_wh = (1280, 720)

    R_map_cam = _down_looking_R()
    # 0.85 m above the floor, near the middle of the surveyed patch; map +z is
    # DOWN, so a camera above the floor sits at NEGATIVE z.
    for p_true in (np.array([0.30, -0.50, -0.85]),
                   np.array([-0.10, 0.40, -0.90]),
                   np.array([0.00, 0.00, -0.80])):
        ids = [4, 5, 24, 25, 26]
        dets = _project_map_tags(est, p_true, R_map_cam, K, ids)
        sol = est.solve(dets)
        assert sol is not None, f"no fix at {p_true} ({est.last_reject})"
        err = np.linalg.norm(np.asarray(sol.p_ned).ravel() - p_true)
        assert err < 1e-6, f"placed {p_true}, recovered {sol.p_ned} (err {err:.2e} m)"
        assert sol.n_tags == len(ids)


def test_synthetic_single_tag_also_recovers():
    """min_tags=1: one tag must still resolve, and without an rp_hint the
    gravity/tilt gate must not fire (the gantry has no autopilot attitude).

    Looser tolerance than the multi-tag case on purpose: one tag takes the
    SOLVEPNP_IPPE_SQUARE branch rather than the joint LM one, and lands ~2 um
    off rather than ~1 nm. That is still exact for our purposes — the point of
    the bound is to catch a frame-convention error (which would be centimetres
    or a sign flip), not to grade the solver.
    """
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    K = _pinhole_K()
    est._K_use, est._maps, est._frame_wh = K, None, (1280, 720)

    p_true = np.array([0.05, -0.05, -0.85])
    dets = _project_map_tags(est, p_true, _down_looking_R(), K, [25])
    sol = est.solve(dets)
    assert sol is not None, f"single-tag no fix ({est.last_reject})"
    err = np.linalg.norm(np.asarray(sol.p_ned).ravel() - p_true)
    assert err < 1e-4, f"placed {p_true}, recovered {sol.p_ned} (err {err:.2e} m)"


def test_datum_is_map_so_positions_are_absolute():
    """datum='first_fix' would re-zero on the first solve and every reading
    after would be relative — useless for comparing against the ROV."""
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    assert est.nav.datum == "map"
    K = _pinhole_K()
    est._K_use, est._maps, est._frame_wh = K, None, (1280, 720)

    first = np.array([0.20, -0.30, -0.85])
    est.solve(_project_map_tags(est, first, _down_looking_R(), K, [4, 5, 25]))
    second = np.array([0.60, -0.30, -0.85])
    sol = est.solve(_project_map_tags(est, second, _down_looking_R(), K, [4, 5, 25]))
    assert sol is not None
    # Absolute, not relative to the first fix.
    assert np.linalg.norm(np.asarray(sol.p_ned).ravel() - second) < 1e-6


# ================================================================ config tier
def test_tag_map_is_the_one_the_rov_uses():
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    assert est.tag_map.anchor_id == EXPECTED_ANCHOR
    assert len(est.tag_map) == EXPECTED_N_TAGS


def test_tag_size_is_the_surveyed_one_not_the_stale_default():
    """tagslam_core.DEFAULT_TAG_SIZE_M is 0.085 — exactly half of the real
    0.170. Inheriting it would scale every distance by 2 and never say so."""
    assert abs(gmp.DEFAULT_TAG_SIZE_M - 0.170) < 1e-9
    assert abs(gmp.MapPoseEstimator(CALIB, TAG_MAP).tag_size_m - 0.170) < 1e-9


def test_wrong_resolution_is_refused_not_rescaled():
    """A raw feed at a size the intrinsics were not calibrated at makes K wrong
    by the ratio. Refusing beats a readout that is quietly 1.5x off."""
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    cw, ch = est.calib_wh
    est.configure((cw, ch))                       # the calibrated size is fine
    assert est.configured
    try:
        est.configure((1920, 1080))
    except gmp.CalibrationMismatch as e:
        assert f"{cw}x{ch}" in str(e)
    else:
        raise AssertionError("a 1920x1080 raw feed was accepted against a "
                             f"{cw}x{ch} calibration")


def test_rectified_intrinsics_scale_proportionally():
    """Recorded frames arrive already undistorted AND downscaled, so K must be
    scaled rather than remapped."""
    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    cw, ch = est.calib_wh
    est.configure((cw, ch), rectified=True)
    K_full = est._K_use.copy()
    est.configure((cw // 2, ch // 2), rectified=True)
    K_half = est._K_use

    for i, j in ((0, 0), (0, 2), (1, 1), (1, 2)):
        assert abs(K_half[i, j] - 0.5 * K_full[i, j]) < 1e-9, f"K[{i},{j}] did not halve"
    assert est._maps is None, "rectified frames must not be remapped again"


# ============================================================== recorded tier
def _have_recording() -> bool:
    return (RECORDING / "camera_trajectory.csv").exists() and \
           len(glob.glob(str(RECORDING / "frames" / "*.jpg"))) > 50


def test_recorded_frames_match_the_recorded_trajectory(n_frames: int = 40):
    """Replay real frames; compare against the run's own trajectory, lifted
    from its per-run anchor into anchor-25 coordinates."""
    if not _have_recording():
        print(f"    SKIP (no {RECORDING.relative_to(_REPO)} — data/ is gitignored)")
        return
    import cv2
    import yaml

    rows = list(csv.DictReader(open(RECORDING / "tag_poses.csv")))
    anchor = min(rows, key=lambda r: abs(float(r["x_m"])) + abs(float(r["y_m"]))
                 + abs(float(r["z_m"])))
    aid = int(anchor["tag_id"])
    entry = yaml.safe_load(open(TAG_MAP))["tags"][aid]
    R_a = quat_wxyz_to_R(entry["quaternion_wxyz"])
    p_a = np.asarray(entry["position_m"], float)

    traj = {i: np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
            for i, r in enumerate(csv.DictReader(
                open(RECORDING / "camera_trajectory.csv")))}
    frames = sorted(glob.glob(str(RECORDING / "frames" / "*.jpg")))

    est = gmp.MapPoseEstimator(CALIB, TAG_MAP)
    h, w = cv2.imread(frames[0]).shape[:2]
    est.configure((w, h), rectified=True)

    errs, miss = [], 0
    for f in frames[::max(1, len(frames) // n_frames)][:n_frames]:
        idx = int(os.path.basename(f).split("_")[0])
        if idx not in traj:
            continue
        sol = est.process(cv2.imread(f))
        if sol is None:
            miss += 1
            continue
        errs.append(np.linalg.norm(np.asarray(sol.p_ned).ravel()
                                   - (R_a @ traj[idx] + p_a)))
    errs = np.array(errs)
    assert len(errs) >= n_frames * 0.8, f"only {len(errs)} fixes, {miss} misses"
    p50 = float(np.percentile(errs, 50))
    p90 = float(np.percentile(errs, 90))
    print(f"    anchor={aid}  n={len(errs)} miss={miss}  "
          f"p50={p50 * 1000:.1f} mm  p90={p90 * 1000:.1f} mm")
    # Generous bounds: this compares two DIFFERENT estimators (see the module
    # docstring). Measured 2026-08-23 across 8 runs: p50 9-16, p90 16-18 mm.
    assert p50 < 0.050, f"p50 {p50 * 1000:.1f} mm — assembly likely wrong"
    assert p90 < 0.100, f"p90 {p90 * 1000:.1f} mm — assembly likely wrong"


def main() -> int:
    tests = [
        test_synthetic_roundtrip_recovers_the_placed_pose,
        test_synthetic_single_tag_also_recovers,
        test_datum_is_map_so_positions_are_absolute,
        test_tag_map_is_the_one_the_rov_uses,
        test_tag_size_is_the_surveyed_one_not_the_stale_default,
        test_wrong_resolution_is_refused_not_rescaled,
        test_rectified_intrinsics_scale_proportionally,
        test_recorded_frames_match_the_recorded_trajectory,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
