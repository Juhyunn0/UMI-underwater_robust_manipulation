#!/usr/bin/env python
"""build_dp_rgbd_zarr.py -- merge a dataset's RGB store and depth store into ONE
training store with both image keys, for the RGB+depth two-stream policy.

    python umi_handheld/build_dp_rgbd_zarr.py "/home/bdml/Desktop/data collection/slam/9_9_26"

reads   <dataset>/dataset.zarr.zip         data/camera0_rgb   (build_dataset.py, colour CAM_A)
        <dataset>/dataset_depth.zarr.zip   data/camera0_depth (build_dp_depth_zarr.py)
writes  <dataset>/dataset_rgbd.zarr.zip    data/camera0_rgb + data/camera0_depth + labels

Why a merge and not two paths: ``UmiDataset`` loads exactly one zip
(umi_dataset.py, ``zarr.ZipStore(dataset_path)``), so every obs key of a policy
has to live in the same store.

What is checked before a byte is written (any failure aborts):

* the two stores are frame-aligned -- ``meta/episode_ends`` and every low-dim
  label are bitwise identical (the depth builder copies them from the RGB store,
  so this is expected to hold; it is asserted, not assumed);
* the RGB store's channel order. ``build_dataset.py`` decodes ``rgb.mp4`` with
  ``format="bgr24"`` and never calls cvtColor, so ``camera0_rgb`` is BGR on disk
  (the 2026-09-01 audit found this; the 9_9_26 build inherits it). The merged
  store is written RGB, because the policy (and vanilla UMI) assume RGB. When
  ``<dataset>/videos/<ep>/rgb.mp4`` is available the order is VERIFIED, not
  assumed: the first kept frame of the first planned episode is decoded, given
  the builder's centre-crop + INTER_AREA 224 recipe, and must equal the stored
  frame exactly in BGR and differ in RGB. ``--no-video-check`` skips this
  (only for a dataset whose videos are gone).

The depth key, labels and ``.zattrs`` are copied from the depth store
unchanged; an ``rgb`` block records the RGB source, the flip and the
verification numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_zarr_io import (assert_same_labels, create_like, episode_bounds, open_zip,  # noqa: E402
                        prepare_out, write_episode_ends, zip_dir)

RGB_KEY, DEPTH_KEY = "camera0_rgb", "camera0_depth"


def verify_channel_order(dataset: Path, rgb: zarr.Group, depth_attrs: dict) -> dict:
    """Decode one source frame and prove the stored order is BGR. Returns the numbers."""
    import av      # the umi2 env has av + cv2; both are only needed for this check
    import cv2

    plan = depth_attrs.get("dataset_plan") or rgb.attrs.get("dataset_plan")
    if not plan:
        raise SystemExit("error: no dataset_plan in either store; cannot locate a source "
                         "frame (pass --no-video-check to skip the order proof)")
    ep = int(plan[0]["episode"])
    lo = int(plan[0]["kept_frame_range"][0])
    video = dataset / "videos" / str(ep) / "rgb.mp4"
    if not video.exists():
        raise SystemExit(f"error: {video} missing (pass --no-video-check to skip)")
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i == lo:
                bgr = fr.to_ndarray(format="bgr24")
                break
        else:
            raise SystemExit(f"error: {video} has fewer than {lo + 1} frames")
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    res = tuple(rgb["data"][RGB_KEY].shape[1:3])[::-1]          # (W, H)
    sq = cv2.resize(bgr[y0:y0 + side, x0:x0 + side], res, interpolation=cv2.INTER_AREA)
    stored = np.asarray(rgb["data"][RGB_KEY][0]).astype(np.int32)
    d_bgr = float(np.abs(stored - sq.astype(np.int32)).mean())
    d_rgb = float(np.abs(stored - sq[..., ::-1].astype(np.int32)).mean())
    print(f"  channel-order proof on {video.relative_to(dataset)} frame {lo}: "
          f"stored-vs-BGR {d_bgr:.4f}, stored-vs-RGB {d_rgb:.4f} (mean |diff|)")
    if not (d_bgr == 0.0 and d_rgb > 1.0):
        raise SystemExit("error: the stored camera0_rgb is NOT a bit-exact BGR copy of "
                         "the source video; refusing to guess the channel order")
    return {"video": str(video), "frame": lo, "mean_abs_diff_vs_bgr": d_bgr,
            "mean_abs_diff_vs_rgb": d_rgb}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dataset folder (holds dataset.zarr.zip etc.)")
    ap.add_argument("--rgb", type=Path, help="RGB store (default <dataset>/dataset.zarr.zip)")
    ap.add_argument("--depth", type=Path,
                    help="depth store (default <dataset>/dataset_depth.zarr.zip)")
    ap.add_argument("--out", type=Path, help="default <dataset>/dataset_rgbd.zarr.zip")
    ap.add_argument("--no-video-check", action="store_true",
                    help="skip decoding a source frame to prove the BGR order")
    ap.add_argument("--keep-dir", action="store_true",
                    help="keep the intermediate .zarr directory beside the zip")
    a = ap.parse_args(argv)

    dataset = a.dataset.expanduser().resolve()
    rgb_zip = a.rgb or dataset / "dataset.zarr.zip"
    depth_zip = a.depth or dataset / "dataset_depth.zarr.zip"
    out_zip = a.out or dataset / "dataset_rgbd.zarr.zip"
    for p in (rgb_zip, depth_zip):
        if not p.exists():
            raise SystemExit(f"error: {p} not found")
    tmp_dir = prepare_out(out_zip)

    rgb, depth = open_zip(rgb_zip), open_zip(depth_zip)
    print(f"{dataset}\n  rgb   {rgb_zip.name}: {rgb['data'][RGB_KEY].shape}\n"
          f"  depth {depth_zip.name}: {depth['data'][DEPTH_KEY].shape}")
    if rgb["data"][RGB_KEY].shape[0] != depth["data"][DEPTH_KEY].shape[0]:
        raise SystemExit("error: frame counts differ")
    label_keys = assert_same_labels(rgb, depth, "rgb vs depth")
    depth_attrs = dict(depth.attrs)

    proof = None
    if not a.no_video_check:
        proof = verify_channel_order(dataset, rgb, depth_attrs)
    else:
        print("  channel order NOT verified (--no-video-check); assuming BGR per "
              "build_dataset.py:146-150")

    # --- output store -----------------------------------------------------------
    root = zarr.open_group(str(tmp_dir), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    dst_rgb = create_like(data, RGB_KEY, rgb["data"][RGB_KEY], image=True)
    dst_depth = create_like(data, DEPTH_KEY, depth["data"][DEPTH_KEY], image=True)
    dst_lab = {k: create_like(data, k, depth["data"][k], image=False) for k in label_keys}

    starts, ends = episode_bounds(depth)
    for i, (s0, s1) in enumerate(zip(starts, ends)):
        # BGR -> RGB is the ONLY change to the pixels (a channel permutation).
        dst_rgb.append(np.ascontiguousarray(rgb["data"][RGB_KEY][s0:s1][..., ::-1]))
        dst_depth.append(np.asarray(depth["data"][DEPTH_KEY][s0:s1]))
        for k, arr in dst_lab.items():
            arr.append(np.asarray(depth["data"][k][s0:s1]))
        if i % 10 == 0 or i == len(ends) - 1:
            print(f"  ep {i:>3}/{len(ends)}: frames {s0}..{s1}")
    write_episode_ends(meta, ends)

    rgb_src_attrs = dict(rgb.attrs)
    attrs = dict(depth_attrs)
    attrs["obs_keys"] = [RGB_KEY, DEPTH_KEY]
    attrs["rgb"] = {
        "key": RGB_KEY,
        "source_zarr": str(rgb_zip),
        "source_recipe": rgb_src_attrs.get("rgb", rgb_src_attrs.get("config", {}).get("dataset")),
        "source_channel_order": "BGR (build_dataset.py decodes rgb.mp4 as bgr24, no cvtColor)",
        "stored_channel_order": "RGB (channels reversed at merge)",
        "channel_order_proof": proof or "not verified (--no-video-check)",
        "note": "colour CAM_A, unrectified, centre crop 224; NOT pixel-aligned with "
                "camera0_depth (rectified-left grid warped to C3 optics) -- two-stream only",
    }
    attrs["merged_from"] = {"rgb": str(rgb_zip), "depth": str(depth_zip),
                            "tool": "umi_handheld/build_dp_rgbd_zarr.py"}
    root.attrs.update(attrs)

    zip_dir(tmp_dir, out_zip, keep_dir=a.keep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
