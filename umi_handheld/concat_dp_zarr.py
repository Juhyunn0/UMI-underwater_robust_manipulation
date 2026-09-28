#!/usr/bin/env python
"""concat_dp_zarr.py -- concatenate two or more DP training stores episode-wise.

    python umi_handheld/concat_dp_zarr.py --out <out>.zarr.zip <a>.zarr.zip <b>.zarr.zip ...

Every input must carry the same ``data/`` keys with the same trailing shapes and
dtypes, and -- for image keys -- the same obs recipe (``.zattrs['obs']``: channels,
z_near/z_far, resolution) and the same warp target optics; two stores whose
pixels mean different things must not be pooled, so a mismatch aborts. Episodes
are appended in the order given, ``meta/episode_ends`` is re-based, and the
output ``.zattrs`` keeps the first store's recipe blocks plus a ``concat_sources``
list saying which episodes came from where (``dataset_plan`` entries gain a
``source`` field, so the SOURCE episode id stays resolvable -- it is per store,
not global).

First use (2026-09-23): pooling the 2026-09-01 land can-picking set (75 ep,
"9_4_26") with the 2026-09-09 recollection (62 ep, "9_9_26") for the depth-only
policy; same rig, same calibration.json (md5 79a19079...), same builder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_zarr_io import (create_like, episode_bounds, open_zip, prepare_out,  # noqa: E402
                        write_episode_ends, zip_dir)

IMAGE_NDIM = 4      # (N, H, W, C)


def recipe_of(attrs: dict) -> dict:
    """The parts of .zattrs that define what a pixel means."""
    obs = dict(attrs.get("obs") or {})
    warp = dict(attrs.get("warp") or {})
    return {"obs": obs, "target_model": warp.get("target_model"),
            "source_model": (warp.get("source_model") or {}).get("name"),
            "rgb": {k: v for k, v in (attrs.get("rgb") or {}).items()
                    if k in ("stored_channel_order", "source_recipe")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="two or more .zarr.zip stores")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-dir", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="pool stores whose obs recipes differ (you are on your own)")
    a = ap.parse_args(argv)
    if len(a.inputs) < 2:
        ap.error("give at least two stores")
    for p in a.inputs:
        if not p.exists():
            raise SystemExit(f"error: {p} not found")
    tmp_dir = prepare_out(a.out)

    groups = [open_zip(p) for p in a.inputs]
    keys = sorted(groups[0]["data"].array_keys())
    ref_recipe = recipe_of(dict(groups[0].attrs))
    for p, g in zip(a.inputs, groups):
        if sorted(g["data"].array_keys()) != keys:
            raise SystemExit(f"error: {p}: keys {sorted(g['data'].array_keys())} != {keys}")
        for k in keys:
            x, y = groups[0]["data"][k], g["data"][k]
            if x.shape[1:] != y.shape[1:] or x.dtype != y.dtype:
                raise SystemExit(f"error: {p}: data/{k} {y.shape[1:]} {y.dtype} != "
                                 f"{x.shape[1:]} {x.dtype}")
        if recipe_of(dict(g.attrs)) != ref_recipe:
            msg = f"{p}: obs recipe / warp target differs from {a.inputs[0]}"
            if not a.force:
                raise SystemExit(f"error: {msg} (pass --force to pool anyway)")
            print(f"  warning: {msg}; pooling under --force", file=sys.stderr)
        _, ends = episode_bounds(g)
        print(f"  {p}: {len(ends)} ep, {int(ends[-1])} frames")

    root = zarr.open_group(str(tmp_dir), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    dst = {k: create_like(data, k, groups[0]["data"][k],
                          image=(groups[0]["data"][k].ndim == IMAGE_NDIM)) for k in keys}

    out_ends, plan, sources, written = [], [], [], 0
    for p, g in zip(a.inputs, groups):
        starts, ends = episode_bounds(g)
        g_attrs = dict(g.attrs)
        g_plan = list(g_attrs.get("dataset_plan") or [{} for _ in ends])
        for i, (s0, s1) in enumerate(zip(starts, ends)):
            for k in keys:
                dst[k].append(np.asarray(g["data"][k][s0:s1]))
            written += int(s1 - s0)
            out_ends.append(written)
            entry = dict(g_plan[i]) if i < len(g_plan) else {}
            entry["source"] = str(p)
            plan.append(entry)
        sources.append({"path": str(p), "n_episodes": int(len(ends)),
                        "n_frames": int(ends[-1]),
                        "source_dataset": g_attrs.get("source_dataset"),
                        "first_episode_index": len(out_ends) - len(ends)})
        print(f"  appended {p.name}: total {len(out_ends)} ep, {written} frames")
    write_episode_ends(meta, out_ends)

    attrs = dict(groups[0].attrs)
    attrs.update({"n_episodes": len(out_ends), "n_frames": written,
                  "dataset_plan": plan, "concat_sources": sources,
                  "source_dataset": [s["source_dataset"] for s in sources],
                  "source_zarr": [s["path"] for s in sources],
                  "tool": "umi_handheld/concat_dp_zarr.py"})
    root.attrs.update(attrs)
    zip_dir(tmp_dir, a.out, keep_dir=a.keep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
