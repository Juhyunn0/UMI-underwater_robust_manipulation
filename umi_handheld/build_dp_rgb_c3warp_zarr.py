#!/usr/bin/env python
"""build_dp_rgb_c3warp_zarr.py -- RGB training store re-projected onto the C3 colour optics.

    python umi_handheld/build_dp_rgb_c3warp_zarr.py \\
        --labels "<DC>/slam/dataset_rgbd_peg_9_27_9_28.zarr.zip" \\
        --out    "<DC>/slam/dataset_rgb_c3warp_peg_9_27_9_28.zarr.zip" \\
        "<DC>/slam/peg_in_hole_9_27_26" "<DC>/slam/peg_in_hole_9_28_26"

Why: ``build_dataset.py`` writes ``camera0_rgb`` as a centre crop of the land OAK-D-W
CAM_A video, unrectified -- about 72 x 72 deg -- while the station feeds an RGB policy
the C3 colour frame's centre crop, about 38.6 x 38.6 deg underwater (KNOWN_ISSUES, RGB
checkpoint entry). The same object is ~1.8x larger on the ROV than in training. This
builder removes that geometric gap the way ``build_dp_depth_zarr.py`` removes it for
depth: every land frame is remapped into the C3 colour camera's projection, then given
exactly the crop the station applies, so train and deploy see the same field of view.

    rgb.mp4 frame (1280x720, BGR)                     land CAM_A, in air
      -> WarpStage remap (linear) onto C3 CAM_A 640x360  configs/target_camera_c3_cam_a_640x360.yaml
      -> centre 360x360 crop -> 224x224 INTER_AREA    = rov_gui ColorObsBuilder on the live frame
      -> BGR -> RGB

The warp treats both cameras as co-located (warp.py): it changes the projection, not the
viewpoint. What it cannot change is recorded, not hidden: colour/medium, the scene, and
that the narrower C3 view crops the handheld fingertips (~22 deg below the axis) out of
the frame while the ROV jaw (~11.5 deg) stays in it [유도].

What is checked before a byte is written, per episode (any failure aborts):

* the frame correspondence -- every decoded source frame, given build_dataset.py's own
  centre-crop + INTER_AREA 224 recipe, must equal the stored ``camera0_rgb`` row of the
  source ``dataset.zarr.zip`` BIT FOR BIT. That proves the warped row is made from the
  same video frame the label belongs to (and, again, that the source store is BGR);
* the labels -- every low-dim label and the episode lengths of the output come from the
  ``--labels`` store (the merged peg store the depth/RGB policies trained on) and must
  equal each source store's own labels bitwise, slice by slice.

Source optics: ``<dataset>/calibration.json`` socket 0 (CAM_A, 1280x800) taken to the
1280x720 the recorder requested with DepthAI's cover-then-crop rule -- the same rule as
``demonstration_processing/gripper.py:rgb_intrinsics`` (fx unchanged, cy - 40). Assumes
rgb.mp4 is the device's default CROP output without on-device undistortion [스펙: DepthAI
v3 requestOutput default resizeMode]; the store records that assumption.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from dp_zarr_io import (LABEL_KEYS, create_like, episode_bounds, open_zip,  # noqa: E402
                        prepare_out, write_episode_ends, zip_dir)
from umi_handheld.camera_model import CameraModel  # noqa: E402
from umi_handheld.warp import WarpStage  # noqa: E402

# Episodes run in forked processes; an OpenCV thread pool created in the parent before
# the fork can deadlock the children, and the parallelism lives in the pool anyway.
cv2.setNumThreads(1)

RGB_KEY = "camera0_rgb"
SRC_SIZE = (1280, 720)          # rgb.mp4 as recorded (oakd_record.py --rgb-size)
TGT_SIZE = (640, 360)           # the C3 colour stream the station feeds
DEFAULT_TARGET = REPO / "configs/target_camera_c3_cam_a_640x360.yaml"
# Depth-only blocks of the label store's .zattrs: they describe a key this store does
# not carry, so they are not copied (concat_dp_zarr.recipe_of would read them as ours).
DEPTH_ONLY_ATTRS = ("obs", "warp", "depth_source", "camera_model_source",
                    "camera_model_target", "obs_keys", "rgb", "merged_from", "tool")


def umi_underwater_repo(dataset: Path) -> Path:
    for c in [p / "UMI_Underwater" for p in (dataset, *dataset.parents)]:
        if (c / "umi_dataset.py").is_file():
            return c
    raise SystemExit(f"no UMI_Underwater checkout (umi_dataset.py) above {dataset}")


def square_224(img: np.ndarray, out: int) -> np.ndarray:
    """build_dataset.py:square(mode='center'): min-side centre crop, INTER_AREA."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (out, out),
                      interpolation=cv2.INTER_AREA)


def land_cam_a_model(dataset: Path) -> tuple[CameraModel, dict]:
    """CAM_A at 1280x720 from the dataset's own calibration.json (cover-then-crop)."""
    calib = json.loads((dataset / "calibration.json").read_text())
    entry = next(cam[1] for cam in calib["cameraData"] if cam[0] == 0)
    K = np.array(entry["intrinsicMatrix"], np.float64)
    W, H = float(entry["width"]), float(entry["height"])
    w, h = (float(v) for v in SRC_SIZE)
    s = max(w / W, h / H)
    K[:2] *= s
    K[0, 2] -= (W * s - w) / 2.0
    K[1, 2] -= (H * s - h) / 2.0
    d = np.zeros(14)
    c = np.asarray(entry["distortionCoeff"], np.float64).ravel()[:14]
    d[:len(c)] = c
    cfg = {
        "schema": "umi_camera_model/1", "name": "oakd_w_97_cam_a_1280x720",
        "role": "source", "medium": "air", "verified": True, "rectified": False,
        "image_size": list(SRC_SIZE), "model": "opencv_rational_14",
        "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                       "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "distortion": d.tolist(),
        "provenance": {"derived_from": f"{dataset / 'calibration.json'} cameraData socket 0 "
                                       f"({int(W)}x{int(H)}), cover-then-crop to "
                                       f"{SRC_SIZE[0]}x{SRC_SIZE[1]} (= gripper.py:rgb_intrinsics)",
                       "assumption": "rgb.mp4 = DepthAI default CROP output, not undistorted "
                                     "on device [스펙]"},
    }
    return CameraModel(cfg, dataset / "calibration.json"), {"native": [int(W), int(H)]}


# --- worker (fork: the maps and the decoder module are inherited) -------------------
_W: dict = {}


def _work(job):
    """Decode one episode, prove frame identity, return the warped rows."""
    from umi_dataset import read_video                       # noqa: E402
    ep_dir, video, lo, n, src_zip, s0 = job
    src = open_zip(src_zip)
    stored = np.asarray(src["data"][RGB_KEY][s0:s0 + n])
    out_res = stored.shape[1]
    y0 = (TGT_SIZE[1] - min(TGT_SIZE)) // 2
    x0 = (TGT_SIZE[0] - min(TGT_SIZE)) // 2
    side = min(TGT_SIZE)
    warped = np.empty((n, out_res, out_res, 3), np.uint8)
    k, worst = 0, 0
    for frame in read_video(video, start=lo, count=n):
        if frame.shape[1::-1] != SRC_SIZE:
            raise RuntimeError(f"{video}: frame {frame.shape[1::-1]} != {SRC_SIZE}")
        naive = square_224(frame, out_res)
        if not np.array_equal(naive, stored[k]):
            worst = max(worst, int(np.abs(naive.astype(np.int16) - stored[k]).max()))
            raise RuntimeError(f"{video}: frame {lo + k} does not reproduce stored row "
                               f"{s0 + k} (max |diff| {worst}) -- frame correspondence lost")
        g = cv2.remap(frame, _W["map_x"], _W["map_y"], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        g = cv2.resize(g[y0:y0 + side, x0:x0 + side], (out_res, out_res),
                       interpolation=cv2.INTER_AREA)
        warped[k] = g[..., ::-1]                              # BGR -> RGB
        k += 1
    if k != n:
        raise RuntimeError(f"{video}: decoded {k} of {n} frames")
    return ep_dir, warped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path,
                    help="dataset folders, in the label store's episode order")
    ap.add_argument("--labels", type=Path, required=True,
                    help="store whose labels/episodes the output copies (merged peg store)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="smoke test: only the first N episodes of the FIRST dataset; "
                         "writes no store, only checks + a QA image")
    ap.add_argument("--qa-png", type=Path, help="side-by-side naive|warped montage")
    ap.add_argument("--keep-dir", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()

    datasets = [d.expanduser().resolve() for d in a.datasets]
    sys.path.insert(0, str(umi_underwater_repo(datasets[0])))
    calib_md5 = {d.name: json.dumps(json.loads((d / "calibration.json").read_text()),
                                    sort_keys=True) for d in datasets}
    if len(set(calib_md5.values())) != 1:
        raise SystemExit("error: the datasets carry different calibration.json -- one "
                         "source model would be wrong for some of them")

    src_cam, src_extra = land_cam_a_model(datasets[0])
    tgt_cam = CameraModel.load(a.target_model)
    tgt_cam.require_verified("build_dp_rgb_c3warp_zarr")
    stage = WarpStage(src_cam, tgt_cam, TGT_SIZE, {"mono_interpolation": "linear"},
                      source_size=SRC_SIZE)
    side = min(TGT_SIZE)
    x0 = (TGT_SIZE[0] - side) // 2
    crop_cov = float(stage.inside[:, x0:x0 + side].mean())
    # Target self-consistency: its unproject (iterative) must invert its project.
    rays = tgt_cam.unproject(np.array([[0, 0], [639, 359], [320, 180], [140, 0]], float),
                             *TGT_SIZE)
    tgt_rt = float(np.abs(tgt_cam.project(rays, *TGT_SIZE) -
                          np.array([[0, 0], [639, 359], [320, 180], [140, 0]])).max())
    print(f"source {src_cam.name}: fx {src_cam.fx:.3f} cx {src_cam.cx:.3f} cy {src_cam.cy:.3f}")
    print(f"target {tgt_cam.name}: fx {tgt_cam.fx:.3f} cx {tgt_cam.cx:.3f} cy {tgt_cam.cy:.3f}")
    print(f"warp: coverage {100 * stage.coverage:.3f}% (centre {side}x{side} crop "
          f"{100 * crop_cov:.3f}%), behind {stage.behind}, source round-trip "
          f"{stage.roundtrip_px:.4f} px, target round-trip {tgt_rt:.2e} px")
    if crop_cov < 0.9999 or stage.behind:
        raise SystemExit("error: the C3 crop reaches outside the land frame")
    if tgt_rt > 0.05:
        raise SystemExit(f"error: target unproject does not invert project ({tgt_rt} px)")
    _W.update(map_x=stage.map_x, map_y=stage.map_y)

    labels = open_zip(a.labels)
    lab_starts, lab_ends = episode_bounds(labels)

    # --- plan: (dataset, plan entry) in label-store order ----------------------------
    jobs, ep_offset, row = [], 0, 0
    for d in datasets:
        src_zip = d / "dataset.zarr.zip"
        src = open_zip(src_zip)
        plan = list(dict(src.attrs)["dataset_plan"])
        s_starts, s_ends = episode_bounds(src)
        if len(plan) != len(s_ends):
            raise SystemExit(f"error: {src_zip}: plan {len(plan)} != {len(s_ends)} episodes")
        # Labels: this dataset's slice of the label store must equal its own labels.
        n_rows = int(s_ends[-1])
        for key in LABEL_KEYS:
            if key in src["data"] and key in labels["data"]:
                if not np.array_equal(src["data"][key][:],
                                      labels["data"][key][row:row + n_rows]):
                    raise SystemExit(f"error: {d.name}: data/{key} != label store rows "
                                     f"{row}..{row + n_rows}")
        for i, entry in enumerate(plan):
            ep = int(entry["episode"])
            act = np.load(d / "slam" / str(ep) / "actions.npz")
            lo_k, hi_k = (int(v) for v in act["kept_span"])
            frames = act["frame_index"][lo_k:hi_k].astype(np.int64)
            n = int(s_ends[i] - s_starts[i])
            if len(frames) != n or [lo_k, hi_k] != [int(v) for v in entry["kept_frame_range"]]:
                raise SystemExit(f"error: {d.name} ep {ep}: kept span / length mismatch")
            if not np.array_equal(frames, np.arange(frames[0], frames[0] + n)):
                raise SystemExit(f"error: {d.name} ep {ep}: frame_index not contiguous")
            g = ep_offset + i
            if int(lab_ends[g] - lab_starts[g]) != n:
                raise SystemExit(f"error: label store episode {g} length != {d.name} ep {ep}")
            jobs.append((f"{d.name}/{ep}", str(d / "videos" / str(ep) / "rgb.mp4"),
                         int(frames[0]), n, str(src_zip), int(s_starts[i])))
        ep_offset += len(plan)
        row += n_rows
        print(f"  {d.name}: {len(plan)} episodes, {n_rows} frames, labels == label store")
    if ep_offset != len(lab_ends) or row != int(lab_ends[-1]):
        raise SystemExit(f"error: datasets give {ep_offset} ep / {row} frames, label store "
                         f"has {len(lab_ends)} / {int(lab_ends[-1])}")

    smoke = a.limit_episodes > 0
    if smoke:
        jobs = jobs[:a.limit_episodes]
        print(f"SMOKE: {len(jobs)} episodes, no store written")

    # --- run ---------------------------------------------------------------------------
    if not smoke:
        tmp_dir = prepare_out(a.out)
        root = zarr.open_group(str(tmp_dir), mode="w")
        data, meta = root.create_group("data"), root.create_group("meta")
        ref = open_zip(jobs[0][4])["data"][RGB_KEY]
        dst_rgb = create_like(data, RGB_KEY, ref, image=True)
    qa_rows = []
    done_frames = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.jobs) as pool:
        for i, (name, warped) in enumerate(pool.imap(_work, jobs)):
            if not smoke:
                dst_rgb.append(warped)
            if len(qa_rows) < 6 and i % max(1, len(jobs) // 6) == 0:
                naive = np.asarray(open_zip(jobs[i][4])["data"][RGB_KEY][jobs[i][5]])[..., ::-1]
                qa_rows.append(np.concatenate([naive, warped[0]], 1))
            done_frames += len(warped)
            if i % 10 == 0 or i == len(jobs) - 1:
                print(f"  ep {i + 1:>3}/{len(jobs)} {name}: {done_frames} frames, "
                      f"{time.time() - t0:.0f} s")
    if a.qa_png and qa_rows:
        a.qa_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.qa_png), np.concatenate(qa_rows, 0)[..., ::-1])
        print(f"QA montage (left naive | right C3-warped) -> {a.qa_png}")
    if smoke:
        print(f"SMOKE OK: {done_frames} frames reproduced bit-exactly and warped")
        return 0

    for key in LABEL_KEYS:
        if key in labels["data"]:
            arr = create_like(data, key, labels["data"][key], image=False)
            arr.append(np.asarray(labels["data"][key][:]))
    write_episode_ends(meta, lab_ends)

    attrs = {k: v for k, v in dict(labels.attrs).items() if k not in DEPTH_ONLY_ATTRS}
    attrs["obs_keys"] = [RGB_KEY]
    attrs["rgb"] = {
        "key": RGB_KEY,
        "stored_channel_order": "RGB (warped from BGR decode, channels reversed)",
        "recipe": "rgb.mp4 1280x720 -> WarpStage(linear) onto C3 CAM_A 640x360 -> centre "
                  "360x360 -> 224 INTER_AREA -> RGB; deploy = ColorObsBuilder on the live "
                  "640x360 C3 colour frame (same crop)",
        "optics": "C3 CAM_A underwater (warped)",
        "frame_identity_proof": f"{done_frames} source frames reproduced the source "
                                f"stores' camera0_rgb rows bit-exactly before warping",
    }
    attrs["warp"] = dict(stage.info(), crop_coverage_fraction=crop_cov,
                         target_roundtrip_px=tgt_rt, applied=True,
                         target_grid=f"C3 CAM_A {TGT_SIZE[0]}x{TGT_SIZE[1]} colour stream")
    attrs["labels_from"] = {"path": str(a.labels),
                            "note": "labels/episode_ends copied bitwise; each source "
                                    "dataset's own labels asserted equal to its slice"}
    attrs["source_rgb_zarr"] = [str(d / "dataset.zarr.zip") for d in datasets]
    attrs["tool"] = "umi_handheld/build_dp_rgb_c3warp_zarr.py"
    attrs["n_frames"] = int(lab_ends[-1])
    attrs["n_episodes"] = int(len(lab_ends))
    root.attrs.update(json.loads(json.dumps(attrs, default=float)))
    zip_dir(tmp_dir, a.out, keep_dir=a.keep_dir)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
