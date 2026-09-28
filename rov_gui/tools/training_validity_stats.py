#!/usr/bin/env python3
"""training_validity_stats.py — the validity channel of the training depth obs,
tabulated (the number policy.min_obs_coverage is set against).

    ~/miniforge3/envs/umi2/bin/python rov_gui/tools/training_validity_stats.py \
        > rov_gui/tools/dp_policy_out/training_validity_<date>.txt

Reads data/camera0_depth channel 1 (255 where depth > 0) of the training zarr
zip and prints per-frame validity fraction statistics plus the corner blocks,
so the deployment coverage gate cites a file instead of a conversation.
Needs a real zarr (umi2 / fstereo envs); the station envs have none.
"""
import sys
import numpy as np
import zarr

path = sys.argv[1] if len(sys.argv) > 1 else "/home/bdml/Desktop/data collection/dataset_depth.zarr.zip"
store = zarr.ZipStore(path, mode="r")
root = zarr.open_group(store, mode="r")
arr = root["data"]["camera0_depth"]
n = arr.shape[0]
step = 2000
frac = np.empty(n, np.float64)
corner = []
for i in range(0, n, step):
    blk = np.asarray(arr[i:i + step])            # (b, 224, 224, 3) uint8
    v = blk[..., 1] > 0
    frac[i:i + blk.shape[0]] = v.mean(axis=(1, 2))
    c = np.stack([v[:, :16, :16], v[:, :16, -16:], v[:, -16:, :16], v[:, -16:, -16:]], 1)
    corner.append(c.mean(axis=(2, 3)))
corner = np.concatenate(corner)
q = lambda a, p: float(np.percentile(a, p))
print(f"source: {path}")
print(f"frames: {n}  obs {arr.shape[1:]}  channel 1 = validity (255 where depth > 0)")
print(f"validity fraction per frame: min {frac.min():.4f}  p1 {q(frac,1):.4f}  p10 {q(frac,10):.4f}  "
      f"median {q(frac,50):.4f}  mean {frac.mean():.4f}  max {frac.max():.4f}")
print(f"frames with validity < 0.985: {int((frac < 0.985).sum())} ({100*(frac<0.985).mean():.2f}%)")
print(f"frames with validity < 0.99 : {int((frac < 0.99).sum())} ({100*(frac<0.99).mean():.2f}%)")
print(f"corner 16x16 block validity (TL,TR,BL,BR) mean: {np.round(corner.mean(axis=0), 4).tolist()}  "
      f"min: {np.round(corner.min(axis=0), 4).tolist()}")
store.close()
