#!/usr/bin/env python3
"""test_extract_pose.py — the handheld pose extractor, with no image involved.

    ~/miniforge3/envs/robust/bin/python umi_handheld/tests/test_extract_pose.py
    ~/miniforge3/envs/robust/bin/python -m pytest umi_handheld/tests/test_extract_pose.py -q

Same pattern as src/tests/test_gantry_map_pose.py's synthetic tier: place
virtual tags in a map, project their corners through a known K with
cv2.projectPoints, feed the corners back in as Detections through the
extractor's injectable detect_fn, and require the placed pose to come back.
That pins corner ordering, tag object points, the extrinsic composition,
R_ned_map=I and datum="map" simultaneously — a frame-convention error shows
up as centimetres or a sign flip, not as solver noise.

Runs under pytest AND standalone (pytest is not installed in the `robust`
env, so main() at the bottom is the gantry-test precedent).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from umi_handheld import extract_pose as ep                        # noqa: E402
from rov_gui.control.tagnav import (                               # noqa: E402
    Detection, TagMap, quat_wxyz_to_R, tag_object_points,
)

TAG_SIZE = 0.170
K = np.array([[600.0, 0.0, 640.0],
              [0.0, 600.0, 400.0],
              [0.0, 0.0, 1.0]])


# ==================================================================== fixtures
def _floor_map() -> TagMap:
    """Five virtual tags flat on the floor, print-up (R = I => +z down: the
    map frame is NED-like, exactly like the surveyed pool map)."""
    at = {10: [0.0, 0.0, 0.0], 11: [0.6, 0.0, 0.0], 12: [0.0, 0.6, 0.0],
          13: [0.6, 0.6, 0.0], 14: [0.3, 0.3, 0.0]}
    return TagMap({tid: (np.eye(3), np.asarray(t, float))
                   for tid, t in at.items()},
                  anchor_id=10, source="synthetic-test")


def _down_R(yaw: float = 0.0) -> np.ndarray:
    """map_R_cam for a camera above the floor looking straight down, rotated
    ``yaw`` about the vertical. Same construction as
    src/tests/test_gantry_map_pose.py:_down_looking_R."""
    base = np.array([[0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ base


def _project(tag_map: TagMap, p_cam_map, R_map_cam, tag_ids,
             corrupt: dict | None = None) -> list[Detection]:
    """Corners as a camera at (p_cam_map, R_map_cam) would see them.

    Mirrors TagNav._obj_img: map object points are (R_mt @ obj.T).T + t_mt.
    ``corrupt`` optionally maps tag_id -> (p, R) of a DIFFERENT camera pose
    to project that tag from, producing inconsistent corners on purpose.
    """
    import cv2

    obj = tag_object_points(TAG_SIZE)
    dets = []
    for tid in tag_ids:
        p, R = (corrupt or {}).get(tid, (p_cam_map, R_map_cam))
        R_cm = R.T
        t_cm = -R.T @ np.asarray(p, float)
        rvec, _ = cv2.Rodrigues(R_cm)
        R_mt, t_mt = tag_map.poses[tid]
        obj_map = (R_mt @ obj.T).T + t_mt
        img, _ = cv2.projectPoints(obj_map.astype(np.float64), rvec,
                                   t_cm.astype(np.float64), K, None)
        dets.append(Detection(int(tid), img.reshape(4, 2).astype(np.float32),
                              99.0))
    return dets


def _smooth_path(n: int = 6):
    """(p_cam_map, R_map_cam) ground truth along a smooth arc, 0.9 m up."""
    out = []
    for i in range(n):
        s = i / max(n - 1, 1)
        p = np.array([0.30 + 0.25 * math.sin(s * 1.2),
                      0.30 + 0.20 * math.cos(s * 1.2) - 0.20,
                      -0.90 + 0.05 * math.sin(s * 2.0)])
        out.append((p, _down_R(yaw=0.3 * s)))
    return out


def _run(dets_per_frame, nav, exclude_ids=frozenset(), t0: float = 100.0):
    """Feed pre-built detection lists through the injectable pipeline."""
    frames = [(t0 + 0.1 * i, d) for i, d in enumerate(dets_per_frame)]
    return ep.extract_poses(frames, lambda payload: payload, nav, K,
                            dist=None, exclude_ids=exclude_ids)


# ====================================================================== tests
def test_synthetic_roundtrip_recovers_the_placed_path():
    """Multi-tag joint PnP round trip + NaN rows where detections are withheld."""
    m = _floor_map()
    nav, ext = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    assert ext["mode"] == "camera"

    truth = _smooth_path(6)
    ids = [10, 11, 12, 13, 14]
    dets = [_project(m, p, R, ids) for p, R in truth]
    withheld = {2, 4}
    for i in withheld:
        dets[i] = []                       # frame seen, no tags decoded

    poses, stats = _run(dets, nav)
    assert poses.shape == (6, 8) and poses.dtype == np.float64
    for i, (p_true, R_true) in enumerate(truth):
        assert abs(poses[i, 0] - (100.0 + 0.1 * i)) < 1e-9
        if i in withheld:
            assert np.all(np.isnan(poses[i, 1:])), f"row {i} should be NaN"
            assert stats[i]["reject"] == "no tags"
            continue
        err = np.linalg.norm(poses[i, 1:4] - p_true)
        assert err < 1e-6, f"row {i}: placed {p_true}, got {poses[i, 1:4]} " \
                           f"(err {err:.2e} m)"
        R_rec = quat_wxyz_to_R(poses[i, 4:8])
        assert np.max(np.abs(R_rec - R_true)) < 1e-6, f"row {i} attitude off"
        assert poses[i, 4] >= 0.0          # canonical w >= 0
        assert stats[i]["n_tags"] == len(ids)
        assert stats[i]["reproj_rms_px"] < 0.1


def test_c3_extrinsic_composes_to_the_body_pose():
    """--extrinsic c3 output == camera pose composed with the mount extrinsic,
    checked against direct matrix math (x_body = R_bc x_cam + t_bc =>
    map_T_body = map_T_cam o (body_T_cam)^-1)."""
    m = _floor_map()
    nav_c3, ext = ep.build_nav(m, TAG_SIZE, extrinsic="c3", min_tags=2)
    assert ext["mode"] == "c3"
    assert ext["source"].endswith("c3_payload_frames.json")
    R_bc = np.asarray(ext["R_frd_cam"])
    t_bc = np.asarray(ext["t_frd_cam"])
    # The independent load must agree with what build_nav wired in.
    R2, t2, _src = ep.load_c3_extrinsic()
    assert np.allclose(R_bc, R2) and np.allclose(t_bc, t2)

    truth = _smooth_path(4)
    ids = [10, 11, 13, 14]
    dets = [_project(m, p, R, ids) for p, R in truth]
    poses, stats = _run(dets, nav_c3)
    assert all(r["fix"] for r in stats)
    for i, (p_cam, R_map_cam) in enumerate(truth):
        R_body_exp = R_map_cam @ R_bc.T
        p_body_exp = p_cam - R_map_cam @ R_bc.T @ t_bc
        assert np.linalg.norm(poses[i, 1:4] - p_body_exp) < 1e-6, \
            f"row {i}: body position off"
        R_rec = quat_wxyz_to_R(poses[i, 4:8])
        assert np.max(np.abs(R_rec - R_body_exp)) < 1e-6, \
            f"row {i}: body attitude off"
    # The lever arm is real: body and camera tracks must differ by |t_bc|.
    nav_cam, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    poses_cam, _ = _run(dets, nav_cam)
    gap = np.linalg.norm(poses[:, 1:4] - poses_cam[:, 1:4], axis=1)
    assert np.allclose(gap, np.linalg.norm(t_bc), atol=1e-9)


def test_exclude_ids_drop_finger_tags_before_the_solve():
    """Detections of ids 0/1 (the gripper fingers) never reach PnP, even when
    the loaded map holds poses for those ids (the tag_map_full hazard)."""
    m = _floor_map()
    # Poison the map the way tag_map_full.yaml would: ids 0/1 exist as
    # (bogus, far-away) floor tags.
    m.instances[0] = [(np.eye(3), np.array([5.0, 5.0, 0.0]))]
    m.instances[1] = [(np.eye(3), np.array([-5.0, 4.0, 0.0]))]
    m.poses = {k: v[0] for k, v in m.instances.items()}

    nav, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    p_true, R_true = np.array([0.25, 0.35, -0.85]), _down_R(0.2)
    ids = [10, 11, 12, 0, 1]
    # Ids 0/1 get corners projected from a DIFFERENT camera pose: grossly
    # inconsistent with the map entries above if they ever join the solve.
    bad_pose = (np.array([1.5, -1.0, -0.5]), _down_R(1.0))
    dets = [_project(m, p_true, R_true, ids,
                     corrupt={0: bad_pose, 1: bad_pose})]

    poses, stats = _run(dets, nav, exclude_ids={0, 1})
    assert stats[0]["fix"], f"rejected: {stats[0]['reject']}"
    assert np.linalg.norm(poses[0, 1:4] - p_true) < 1e-6
    assert 0 not in stats[0]["tag_ids"] and 1 not in stats[0]["tag_ids"]

    # Load-bearing check: WITHOUT the exclusion the poisoned corners push the
    # joint solve past the reproj gate — no fix at all.
    nav2, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    poses2, stats2 = _run(dets, nav2, exclude_ids=frozenset())
    assert not stats2[0]["fix"]
    assert np.all(np.isnan(poses2[0, 1:]))


def test_min_tags_gate_and_allow_single_tag():
    """One tag in frame: NaN at min_tags=2; a flagged fix at min_tags=1."""
    m = _floor_map()
    p_true, R_true = np.array([0.32, 0.28, -0.80]), _down_R()
    dets = [_project(m, p_true, R_true, [14])]

    nav2, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    poses, stats = _run(dets, nav2)
    assert np.all(np.isnan(poses[0, 1:]))
    assert "unique tags" in stats[0]["reject"]   # "1/2 unique tags"

    nav1, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=1)
    poses1, stats1 = _run(dets, nav1)
    assert stats1[0]["fix"] and stats1[0]["n_tags"] == 1
    # IPPE branch: ~um-level, not the joint-LM nm (gantry test precedent).
    err = np.linalg.norm(poses1[0, 1:4] - p_true)
    assert err < 1e-4, f"single-tag err {err:.2e} m"
    s = ep.summarize(stats1)
    assert s["n_single_tag_fixes"] == 1


def test_unreadable_frame_gets_its_own_reject_reason():
    m = _floor_map()
    nav, _ = ep.build_nav(m, TAG_SIZE, extrinsic="camera", min_tags=2)
    frames = [(1.0, None)]                       # detect_fn returns None
    poses, stats = ep.extract_poses(frames, lambda payload: payload, nav, K)
    assert np.all(np.isnan(poses[0, 1:]))
    assert stats[0]["reject"] == "frame missing/unreadable"
    assert "frame missing/unreadable" in ep.summarize(stats)["reject_counts"]


def test_cli_outputs_shape_alignment_and_schema(tmp_path=None):
    """End-to-end main() on a fake session of blank frames: exit 0 with the
    0-fix warning path, poses.npy aligned to frames.csv, poses.json schema."""
    import cv2

    ctx = tempfile.TemporaryDirectory() if tmp_path is None else None
    root = Path(ctx.name) if ctx else tmp_path
    try:
        sess = root / "demonstration_9999"
        (sess / "left").mkdir(parents=True)
        stamps = ["1786060100.100000", "1786060100.200000", "1786060100.300000"]
        for st in stamps:
            cv2.imwrite(str(sess / "left" / f"{st}.png"),
                        np.zeros((64, 64), np.uint8))
        with (sess / "frames.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=("idx", "t_unix", "left_file"))
            w.writeheader()
            for i, st in enumerate(stamps):
                w.writerow({"idx": i + 1, "t_unix": st,
                            "left_file": f"left/{st}.png"})
        (sess / "session.json").write_text(json.dumps({
            "kind": "demonstration", "source": "synthetic-test",
            "camera_model_source": {
                "config": "synthetic-test", "name": "test_cam",
                "role": "source", "medium": "air", "verified": True,
                "rectified": True, "image_size": [64, 64],
                "fx": 60.0, "fy": 60.0, "cx": 32.0, "cy": 32.0,
                "distortion": [0.0] * 8, "baseline_m": 0.075}}))

        rc = ep.main([str(sess)])            # default map: config/tag_map.yaml
        assert rc == 0, "0 fixes must NOT be a hard error"

        poses = np.load(sess / "poses.npy")
        assert poses.shape == (3, 8) and poses.dtype == np.float64
        for i, st in enumerate(stamps):
            assert abs(poses[i, 0] - float(st)) < 1e-6, "t_unix misaligned"
            assert np.all(np.isnan(poses[i, 1:]))

        meta = json.loads((sess / "poses.json").read_text())
        assert meta["schema"] == "umi_handheld_poses/1"
        assert meta["pose_of"] == "body_frd"       # default --extrinsic c3
        assert meta["frame"] == "map_ned"
        assert meta["columns"] == ["t_unix", "x", "y", "z",
                                   "qw", "qx", "qy", "qz"]
        for key in ("tag_map", "tag_size_m", "family", "excluded_ids",
                    "min_tags", "extrinsic", "camera_model", "stats",
                    "assumptions", "single_tag_rows", "ambiguous_rows"):
            assert key in meta, f"poses.json missing {key!r}"
        assert len(meta["tag_map"]["sha1"]) == 40
        assert meta["excluded_ids"] == [0, 1]
        assert meta["stats"]["n_frames"] == 3
        assert meta["stats"]["n_fix"] == 0
        assert meta["stats"]["reproj_rms_px"] is None
        assert meta["stats"]["fix_rate"] == 0.0
        assert sum(meta["stats"]["reject_counts"].values()) == 3
        assert meta["extrinsic"]["mode"] == "c3"
        assert meta["camera_model"]["fx"] == 60.0
    finally:
        if ctx:
            ctx.cleanup()


def test_cli_missing_session_is_a_hard_error(tmp_path=None):
    ctx = tempfile.TemporaryDirectory() if tmp_path is None else None
    root = Path(ctx.name) if ctx else tmp_path
    try:
        assert ep.main([str(root / "no_such_session")]) != 0
        empty = root / "empty_session"
        empty.mkdir()
        assert ep.main([str(empty)]) != 0        # no frames.csv/session.json
    finally:
        if ctx:
            ctx.cleanup()


def test_quaternion_roundtrip_matches_tagnav_convention():
    rng = np.random.default_rng(7)
    for _ in range(50):
        v = rng.normal(size=4)
        q = v / np.linalg.norm(v)
        R = quat_wxyz_to_R(q)
        q2 = ep.R_to_quat_wxyz(R)
        assert np.max(np.abs(quat_wxyz_to_R(q2) - R)) < 1e-12
        assert q2[0] >= 0.0


# ============================================================ standalone runner
def main() -> int:
    tests = [
        test_synthetic_roundtrip_recovers_the_placed_path,
        test_c3_extrinsic_composes_to_the_body_pose,
        test_exclude_ids_drop_finger_tags_before_the_solve,
        test_min_tags_gate_and_allow_single_tag,
        test_unreadable_frame_gets_its_own_reject_reason,
        test_cli_outputs_shape_alignment_and_schema,
        test_cli_missing_session_is_a_hard_error,
        test_quaternion_roundtrip_matches_tagnav_convention,
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
