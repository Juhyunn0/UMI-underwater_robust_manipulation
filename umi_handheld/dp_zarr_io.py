"""dp_zarr_io.py -- shared pieces for the DP training-store builders.

The training stores (``dataset_depth.zarr.zip``, ``dataset_rgbd.zarr.zip``, the
episode-concatenated ones) all follow the layout ``UmiDataset`` reads
(``data/<key>`` arrays + ``meta/episode_ends``) and the codec/chunk conventions
of ``build_dp_depth_zarr.py``: images chunked one frame at a time under
blosc-zstd(5) NOSHUFFLE, low-dim arrays in 8192-row chunks with SHUFFLE, and a
ZIP_STORED archive so the zip is a plain container, not a second compressor.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

IMG_COMP = Blosc(cname="zstd", clevel=5, shuffle=Blosc.NOSHUFFLE)
LO_COMP = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

# Every low-dim label UmiDataset may read, in the order build_dp_depth_zarr writes them.
LABEL_KEYS = ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width",
              "robot0_gripper_closed", "robot0_demo_start_pose", "robot0_demo_end_pose",
              "timestamp")


def open_zip(path) -> zarr.Group:
    return zarr.open_group(zarr.ZipStore(str(path), mode="r"), mode="r")


def episode_bounds(group: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return starts, ends


def create_like(dst_data: zarr.Group, key: str, ref: zarr.Array, *, image: bool) -> zarr.Array:
    """An empty, appendable array with `ref`'s trailing shape/dtype and our codecs."""
    if image:
        chunks = (1,) + tuple(ref.shape[1:])
        comp = IMG_COMP
    else:
        chunks = (8192,) + tuple(ref.shape[1:])
        comp = LO_COMP
    return dst_data.create_dataset(key, shape=(0,) + tuple(ref.shape[1:]), chunks=chunks,
                                   dtype=ref.dtype, compressor=comp)


def write_episode_ends(dst_meta: zarr.Group, ends) -> None:
    ends = np.asarray(ends, dtype=np.int64)
    dst_meta.create_dataset("episode_ends", shape=(len(ends),), chunks=(max(len(ends), 1),),
                            dtype="i8", data=ends, compressor=LO_COMP)


def prepare_out(out_zip: Path) -> Path:
    """The intermediate directory (`<name>.zarr`) for `<name>.zarr.zip`; refuses to clobber."""
    out_zip = Path(out_zip)
    if out_zip.exists():
        raise SystemExit(f"error: {out_zip} exists; remove it or pass a different --out")
    tmp_dir = out_zip.with_suffix("")
    if tmp_dir.exists():
        raise SystemExit(f"error: {tmp_dir} exists; remove it or pass a different --out")
    return tmp_dir


def zip_dir(tmp_dir: Path, out_zip: Path, *, keep_dir: bool) -> None:
    print(f"\nzipping -> {out_zip}")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(Path(tmp_dir).rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tmp_dir).as_posix())
    # Reopen the archive the way UmiDataset will, so a truncated zip fails here.
    chk = open_zip(out_zip)
    n = int(chk["meta"]["episode_ends"][-1])
    for k in chk["data"].array_keys():
        assert chk["data"][k].shape[0] == n, (k, chk["data"][k].shape, n)
    print(f"  {len(chk['meta']['episode_ends'])} episodes, {n} frames, "
          f"{Path(out_zip).stat().st_size / 1e9:.2f} GB, reopened OK")
    if keep_dir:
        print(f"  intermediate directory kept at {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir)
        print(f"  intermediate directory {tmp_dir} removed (--keep-dir keeps it)")


def assert_same_labels(a: zarr.Group, b: zarr.Group, what: str) -> list[str]:
    """Bitwise equality of episode_ends and every shared low-dim label; returns the keys."""
    ea, eb = a["meta"]["episode_ends"][:], b["meta"]["episode_ends"][:]
    if not np.array_equal(ea, eb):
        raise SystemExit(f"error: {what}: meta/episode_ends differ "
                         f"({len(ea)} vs {len(eb)} episodes)")
    keys = [k for k in LABEL_KEYS if k in a["data"] and k in b["data"]]
    for k in keys:
        if not np.array_equal(a["data"][k][:], b["data"][k][:]):
            raise SystemExit(f"error: {what}: data/{k} differs -> the stores are not "
                             f"frame-aligned; refusing to merge")
    print(f"  {what}: episode_ends + {len(keys)} labels bitwise identical "
          f"({len(ea)} ep, {int(ea[-1])} frames)")
    return keys


def warn(msg: str) -> None:
    print(f"  warning: {msg}", file=sys.stderr)
