#!/usr/bin/env python3
"""build_dp_depth_zarr.py — land UMI demos + FoundationStereo depth -> a depth-obs
dataset a Diffusion Policy can train on, already re-projected onto the C3's optics.

The chain this closes
---------------------
    oakd_record.py              raw lossless stereo pair + IMU + factory calibration
    replay_foundation_stereo.py FoundationStereo -> depth/<ep>/depth.zarr, float16 mm
                                on the RECTIFIED-LEFT grid (fx_rect from stereoRectify P1)
    THIS SCRIPT                 warp to C3 optics -> normalised inverse depth + validity
                                -> a UMI-schema zarr with the existing pose/gripper labels
    train.py                    diffusion_policy, task=umi_depth

Why the obs is depth and not colour
-----------------------------------
The policy is trained in air and flown in water. Colour does not survive that; geometry
does. Depth carries the medium difference in each camera's own calibration, which is
removed before this point -- so the same physical range lands on the same normalised
value on land and underwater.

Grid parity (the thing that has to be exactly right)
----------------------------------------------------
Both ends run the SAME FoundationStereo checkpoint on a HOST-rectified pair and turn
disparity into millimetres with that rectification's own focal length:

    land   umi_dataset.Rectifier(alpha=0.0) -> P1[0,0] = 249.5656 px @ 640x400
    water  c3_camera.host_depth.StereoRig   -> P1[0,0], see rov_gui/perception/fstereo.py

so neither end applies the `--depth-scale 0.64` stopgap -- that patch belongs to the C3's
on-device StereoDepth, and multiplying a rectification-corrected number by it would
correct a corrected number (rov_gui/README.md:753).

The warp source model is therefore the rectified-left pinhole, NOT the raw CAM_B model in
configs/source_camera_air.yaml -- that file describes alpha=-1.0 at 1280x800 (fx 566.376),
while the depth this script reads lives on alpha=0.0 at 640x400 (fx 249.566). Using the
wrong one reads the wrong pixels and fails silently. The model is rebuilt here from the
same Rectifier the depth came from, and the result is asserted against the depth store's
own `fx_rect_px` attribute.

The target is the C3's RAW CAM_B projection, because the rectified C3 model needs stereo
extrinsics the EEPROM dump does not carry (configs/target_camera_underwater.yaml
documents this). Every output records `target_grid` so a later dump can supersede it.

Deployment must reproduce, exactly, the recipe recorded in the output's `.zattrs['obs']`:
warp -> centre crop -> resize -> normalise with the same z_near/z_far -> same channel order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from umi_handheld.camera_model import CameraModel          # noqa: E402
from umi_handheld.build_zarr import normalise_depth        # noqa: E402
from umi_handheld.warp import WarpStage                    # noqa: E402

#: Labels copied through verbatim from the source dataset. These are what
#: diffusion_policy's UmiDataset consumes; `action` is synthesised by its sampler
#: from eef_pos + eef_rot_axis_angle + gripper_width, so it is not stored.
LABEL_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_gripper_closed",
    "robot0_demo_start_pose",
    "robot0_demo_end_pose",
    "timestamp",
)

#: Channel packing of the stored uint8 obs image. Three channels because the stock
#: timm/ResNet stems want three; ch2 repeats the depth rather than inventing a signal.
CHANNELS = ("inverse_depth", "validity_mask", "inverse_depth")


def umi_underwater_repo(dataset: Path) -> Path:
    """The UMI_Underwater checkout whose `Rectifier` the depth was built with.

    A dataset recorded in place keeps the checkout beside it, but a NESTED dataset
    (a collection root holding several dated datasets under it) keeps one checkout at
    the top for all of them -- so search the dataset first, then walk up. Walking up
    and not just falling back to the repo copy: the Rectifier has to be the same object
    `replay_foundation_stereo.py` used for THIS dataset's depth, and the checkout next
    to the data is the one that was used.

    `UMI_UNDERWATER_REPO` overrides the search, as `FOUNDATION_STEREO_REPO` does for
    replay_foundation_stereo.py.
    """
    env = os.environ.get("UMI_UNDERWATER_REPO")
    candidates = [Path(env).expanduser()] if env else []
    candidates += [p / "UMI_Underwater" for p in (dataset, *dataset.parents)]
    for c in candidates:
        if (c / "umi_dataset.py").is_file():
            return c
    raise FileNotFoundError(
        f"no UMI_Underwater checkout containing umi_dataset.py found in {dataset} or "
        f"any parent; set UMI_UNDERWATER_REPO to the checkout that produced this "
        f"dataset's depth")


def rectified_left_model(dataset: Path, size) -> tuple[CameraModel, dict]:
    """The pinhole the FoundationStereo depth actually lives on.

    Rebuilt from the same `Rectifier` (alpha=0.0) that `replay_foundation_stereo.py`
    used, rather than read from a config, so the two cannot drift apart.
    """
    sys.path.insert(0, str(umi_underwater_repo(dataset)))
    from umi_dataset import Rectifier                       # noqa: E402

    calibration = json.loads((dataset / "calibration.json").read_text())
    rect = Rectifier(calibration, size, alpha=0.0)
    P1 = np.asarray(rect.P1, dtype=np.float64)
    w, h = (int(v) for v in size)
    cfg = {
        "schema": "umi_camera_model/1",
        "name": "oakd_oak_d_w_rectified_left_alpha0",
        "role": "source",
        "medium": "air",
        "verified": True,
        "rectified": True,
        "image_size": [w, h],
        "model": "opencv_rational",
        "intrinsics": {"fx": float(P1[0, 0]), "fy": float(P1[1, 1]),
                       "cx": float(P1[0, 2]), "cy": float(P1[1, 2])},
        "distortion": [0.0] * 8,
        "stereo": {"baseline_m": float(rect.P2[0, 3] / -P1[0, 0])
                   if P1[0, 0] else 0.075317084},
        "provenance": {
            "derived_from": "umi_dataset.Rectifier(calibration.json, size, alpha=0.0)",
            "same_object_as": "replay_foundation_stereo.py's rectification",
            "device_mxid": "14442C10716DBCD600",
        },
    }
    return CameraModel(cfg, dataset / "calibration.json"), {"P1": P1.tolist()}


def episode_plan(zip_path: Path) -> tuple[list[dict], dict]:
    """(dataset_plan, root attrs) from the source dataset.zarr.zip.

    `dataset_plan[i]['episode']` is the SOURCE episode id, which is not i: episode 28 was
    removed after SLAM, so the plan runs 0..27, 29..75 over 75 rows. Joining on position
    instead of on this field silently shifts every label past 28 by one.
    """
    store = zarr.ZipStore(str(zip_path), mode="r")
    try:
        root = zarr.open_group(store, mode="r")
        attrs = dict(root.attrs)
        return list(attrs["dataset_plan"]), attrs
    finally:
        store.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path,
                    help="dataset directory from oakd_record.py (must contain depth/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .zarr.zip (default: <dataset>/dataset_depth.zarr.zip)")
    ap.add_argument("--target-model", type=Path,
                    default=REPO / "configs/target_camera_underwater.yaml")
    ap.add_argument("--warp-size", type=int, nargs=2, default=(640, 400), metavar=("W", "H"),
                    help="grid the warp lands on; keep the target's aspect (default 640 400)")
    ap.add_argument("--out-res", type=int, default=224,
                    help="square resolution the policy sees (default 224)")
    ap.add_argument("--z-near", type=float, default=0.20, metavar="M",
                    help="metres mapping to 1.0 in the normalised obs (default 0.20)")
    ap.add_argument("--z-far", type=float, default=3.00, metavar="M",
                    help="metres mapping to 0.0 (default 3.00)")
    ap.add_argument("--no-warp", action="store_true",
                    help="skip the C3 re-projection; writes land optics (ablation only)")
    ap.add_argument("--limit-episodes", type=int, default=0, metavar="N",
                    help="only the first N plan entries (smoke test)")
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()

    dataset = a.dataset.expanduser().resolve()
    src_zip = dataset / "dataset.zarr.zip"
    depth_root = dataset / "depth"
    out_path = (a.out or dataset / "dataset_depth.zarr.zip").expanduser().resolve()
    for p, what in ((src_zip, "source dataset"), (depth_root, "depth/ directory")):
        if not p.exists():
            print(f"error: {what} not found at {p}", file=sys.stderr)
            return 2

    plan, src_attrs = episode_plan(src_zip)
    if a.limit_episodes:
        plan = plan[:a.limit_episodes]
    print(f"{dataset}: {len(plan)} plan entries, "
          f"{sum(int(e['n_frames']) for e in plan)} frames")

    # --- the two optical models -------------------------------------------------
    warp_size = tuple(int(v) for v in a.warp_size)
    src_cam, src_extra = rectified_left_model(dataset, (640, 400))
    tgt_cam = CameraModel.load(a.target_model)
    tgt_cam.require_verified("build_dp_depth_zarr", allow=a.allow_unverified)
    print(f"  source optics {src_cam.name}: fx {src_cam.fx:.4f} cx {src_cam.cx:.4f} "
          f"cy {src_cam.cy:.4f} @ {src_cam.width}x{src_cam.height}")
    print(f"  target optics {tgt_cam.name}: fx {tgt_cam.fx:.2f} @ "
          f"{tgt_cam.width}x{tgt_cam.height}, medium {tgt_cam.medium}")

    stage = None
    warp_info = {"applied": False, "reason": "--no-warp"}
    if not a.no_warp:
        # nearest for depth: linear would average the 0 that means "invalid" into real
        # ranges and produce a plausible-looking distance nothing measured.
        stage = WarpStage(src_cam, tgt_cam, warp_size,
                          {"depth_interpolation": "nearest",
                           "mono_interpolation": "linear"},
                          source_size=(640, 400))
        warp_info = stage.info()
        warp_info["applied"] = True
        print(f"  warp {src_cam.width}x{src_cam.height} -> {warp_size}: "
              f"coverage {100 * stage.coverage:.2f}%, "
              f"behind-camera rays {stage.behind}, "
              f"round-trip {stage.roundtrip_px:.4f} px")

    # --- output store -----------------------------------------------------------
    R = int(a.out_res)
    tmp_dir = out_path.with_suffix("")           # <name>.zarr, zipped at the end
    if tmp_dir.exists():
        print(f"error: {tmp_dir} exists; remove it or pass a different --out",
              file=sys.stderr)
        return 2
    root = zarr.open_group(str(tmp_dir), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    img_comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.NOSHUFFLE)
    lo_comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    obs = data.create_dataset("camera0_depth", shape=(0, R, R, 3), chunks=(1, R, R, 3),
                              dtype="u1", compressor=img_comp)

    src_store = zarr.ZipStore(str(src_zip), mode="r")
    src = zarr.open_group(src_store, mode="r")
    label_arrs = {}
    for k in LABEL_KEYS:
        if k not in src["data"]:
            print(f"  note: source has no {k}; skipping", file=sys.stderr)
            continue
        ref = src["data"][k]
        label_arrs[k] = data.create_dataset(
            k, shape=(0,) + ref.shape[1:], chunks=(8192,) + ref.shape[1:],
            dtype=ref.dtype, compressor=lo_comp)

    src_ends = np.asarray(src["meta"]["episode_ends"][:])
    src_starts = np.concatenate([[0], src_ends[:-1]])

    ends, written, dropped = [], 0, []
    for i, entry in enumerate(plan):
        ep = int(entry["episode"])                     # SOURCE id, not the plan index
        lo, hi = (int(v) for v in entry["kept_frame_range"])
        store_p = depth_root / str(ep) / "depth.zarr"
        if not store_p.exists():
            dropped.append((ep, "no depth.zarr"))
            continue
        dz = zarr.open_group(str(store_p), mode="r")
        d_idx = np.asarray(dz["frame_index"][:])
        want = np.arange(lo, hi)
        sel = np.nonzero(np.isin(d_idx, want))[0]
        if sel.size != want.size:
            dropped.append((ep, f"depth has {sel.size} of {want.size} kept frames"))
            continue
        if i == 0:
            got_fx = float(dz.attrs.get("fx_rect_px", 0.0))
            if abs(got_fx - src_cam.fx) > 1e-3:
                print(f"error: depth store says fx_rect_px={got_fx}, but the source model "
                      f"rebuilt from Rectifier gives {src_cam.fx}. The warp would read the "
                      f"wrong pixels. Refusing.", file=sys.stderr)
                return 3
            print(f"  grid check OK: depth fx_rect_px {got_fx:.4f} == source model fx")

        depth_mm = np.asarray(dz["depth_mm"][:])[sel].astype(np.float32)
        frames = np.empty((depth_mm.shape[0], R, R, 3), dtype=np.uint8)
        for j, dm in enumerate(depth_mm):
            w = stage.apply_depth(dm) if stage is not None else dm
            s = min(w.shape[:2])
            y0, x0 = (w.shape[0] - s) // 2, (w.shape[1] - s) // 2
            w = w[y0:y0 + s, x0:x0 + s]
            # INTER_NEAREST again: same reason as the warp.
            w = cv2.resize(w, (R, R), interpolation=cv2.INTER_NEAREST)
            v = normalise_depth(w, a.z_near, a.z_far)
            valid = (w > 0).astype(np.uint8) * 255
            u8 = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
            frames[j] = np.stack([u8, valid, u8], axis=-1)

        obs.append(frames)
        s0, s1 = int(src_starts[i]), int(src_ends[i])
        for k, arr in label_arrs.items():
            arr.append(np.asarray(src["data"][k][s0:s1]))
        written += frames.shape[0]
        ends.append(written)
        print(f"  ep {ep:>2} (plan {i:>2}): {frames.shape[0]:>3} frames  -> {written}")

    src_store.close()
    meta.create_dataset("episode_ends", shape=(len(ends),), chunks=(max(len(ends), 1),),
                        dtype="i8", data=np.asarray(ends, dtype=np.int64),
                        compressor=lo_comp)

    root.attrs.update({
        "source_dataset": str(dataset),
        "source_zarr": str(src_zip),
        "n_episodes": len(ends),
        "n_frames": written,
        "fps": float(src_attrs.get("fps", 30.0)),
        "obs": {
            "key": "camera0_depth",
            "channels": list(CHANNELS),
            "dtype": "uint8, divide by 255 to recover [0,1]",
            "resolution": [R, R],
            "recipe": ["FoundationStereo depth (float16 mm, rectified-left grid)",
                       "warp to target optics (nearest)" if stage is not None
                       else "NO WARP (land optics)",
                       "centre crop to square", f"resize to {R}x{R} (nearest)",
                       "normalise_depth(z_near, z_far) -> [0,1], near = large",
                       "channel 1 = validity (255 where depth > 0)"],
            "z_near_m": a.z_near, "z_far_m": a.z_far,
            "invalid_convention": "channel 0 is 0.0 at invalid AND at z_far; "
                                  "channel 1 disambiguates",
        },
        "warp": warp_info,
        "camera_model_source": src_cam.provenance() | src_extra,
        "camera_model_target": tgt_cam.provenance(),
        "depth_source": {
            "tool": "UMI_Underwater/replay_foundation_stereo.py",
            "checkpoint": str(zarr.open_group(
                str(depth_root / str(int(plan[0]['episode'])) / 'depth.zarr'),
                mode='r').attrs.get('checkpoint', '')),
            "iters": 16, "scale": 1.0,
            "depth_scale_applied": 1.0,
            "depth_scale_note": "the 0.64 stopgap is for the C3's ON-DEVICE StereoDepth "
                                "only; the FoundationStereo path is already metric via "
                                "its own fx_rect (rov_gui/README.md:753)",
        },
        "labels": {"copied_from_source": list(label_arrs),
                   "action": "synthesised by diffusion_policy sampler.py:165-172 from "
                             "eef_pos + eef_rot_axis_angle + gripper_width"},
        "dataset_plan": plan,
    })

    if dropped:
        print(f"\n  DROPPED {len(dropped)} episodes:")
        for ep, why in dropped:
            print(f"    ep {ep}: {why}")

    print(f"\nzipping -> {out_path}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(tmp_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tmp_dir).as_posix())
    print(f"  {len(ends)} episodes, {written} frames, "
          f"{out_path.stat().st_size / 1e9:.2f} GB")
    print(f"  intermediate directory kept at {tmp_dir} (delete when happy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
