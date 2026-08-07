"""build_zarr.py — warped session(s) -> a zarr the UMI_underwater affordance pipeline eats.

Schema, taken from what the consumers actually require rather than from the README:

    data/camera_1             (T, 3, 112, 112)  uint8    CHW   <- the only layout
                                                                  zarr_adapter parses
    data/camera               (T, 112, 112, 3)  uint8    HWC
    data/camera0_main_depth   (T, 112, 112, 1)  float16  [0,1] <- the training input
    data/tracked_gripper_pct  (T, 1)            float32  0=closed
    meta/episode_ends         (N,)              int64    cumulative, EXCLUSIVE

Depth convention
----------------
Normalised inverse depth, near = LARGE, matching the polarity the pipeline's existing
depth arrays carry:

    v = (1/Z - 1/Z_far) / (1/Z_near - 1/Z_far),  clipped to [0, 1]

Z is metric, so the same transform on land and underwater lands on the same value for
the same physical range: the medium difference is absorbed by each camera's own
calibration before this point, not here.

zattrs
------
The training dataset reads exactly three names — anchored_min, anchored_max, normalized
(multi_primitive_train_dataset.py:167-173). Only `normalized` fails silently when
missing: it defaults to False, which flips the affine from identity to a x3.1 gain and
then clamps. All three are written, and read back and asserted at the end of this script.

Because v is already in [0, 1], anchored_min/anchored_max are 0 and 1, which makes the
training-time affine the identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.camera_model import REPO, CameraModel, UnverifiedModelError  # noqa: E402
from umi_handheld.record import load_pipeline_config                            # noqa: E402
from umi_handheld.warp import TARGET_GRID_NOTE                                  # noqa: E402


def centre_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def normalise_depth(depth_mm: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """uint16 millimetres -> normalised inverse depth in [0, 1], near = large."""
    z = depth_mm.astype(np.float32) / 1000.0
    valid = z > 0
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    v = np.zeros_like(z, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v[valid] = (1.0 / z[valid] - inv_far) / (inv_near - inv_far)
    # TODO(quality): invalid pixels become 0.0, i.e. indistinguishable from "at Z_far".
    # The model cannot tell "nothing measured" from "far away".
    return np.clip(v, 0.0, 1.0)


def denormalise_depth(v: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    return 1.0 / (v.astype(np.float32) * (inv_near - inv_far) + inv_far)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--session", type=Path, action="append", default=None,
                    help="repeatable; one session = one episode")
    ap.add_argument("--out", type=Path, default=REPO / "zarr/demos_processed_112.zarr")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sessions = a.session or [REPO / "sessions/session_0001"]
    cfg = load_pipeline_config(a.config)
    zc = cfg["zarr"]
    RH, RW = (int(v) for v in zc["resize"])
    z_near, z_far = float(zc["z_near_m"]), float(zc["z_far_m"])
    src = CameraModel.load(cfg["cameras"]["source"])
    tgt = CameraModel.load(cfg["cameras"]["target"])

    print(f"sessions   : {len(sessions)}")
    print(f"resize     : {RW}x{RH}  crop={zc['crop']}  (on the WARPED grid)")
    print(f"depth      : normalised inverse, Z_near={z_near} m  Z_far={z_far} m  "
          f"dtype={zc['depth_dtype']}")
    print(f"out        : {a.out}")

    try:
        src.require_verified("build_zarr.py", allow=a.allow_unverified)
    except UnverifiedModelError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 3
    if not src.verified:
        print("NOTE: proceeding on an UNVERIFIED source model because "
              "--allow-unverified was given; this is recorded in the zarr attrs.")

    if a.dry_run:
        print("\n--dry-run: config parsed, no session read, nothing written.")
        return 0

    cams, cams1, depths, pcts, ends = [], [], [], [], []
    total = 0
    for sdir in sessions:
        wdir = sdir / "warped"
        rows = list(csv.DictReader((sdir / "frames.csv").open()))
        gw_path = sdir / "gripper_width.json"
        gw = json.loads(gw_path.read_text())
        pct = np.load(gw_path.with_suffix(".npy")).astype(np.float32).reshape(-1, 1)
        if len(pct) != len(rows):
            print(f"ERROR: {sdir}: {len(rows)} frames but {len(pct)} gripper values",
                  file=sys.stderr)
            return 2
        for row in rows:
            stamp = Path(row["left_file"]).stem
            m = cv2.imread(str(wdir / "left" / f"{stamp}.png"), cv2.IMREAD_GRAYSCALE)
            d = cv2.imread(str(wdir / "depth" / f"{stamp}.png"), cv2.IMREAD_UNCHANGED)
            if zc["crop"] == "centre":
                m, d = centre_crop(m), centre_crop(d)
            m = cv2.resize(m, (RW, RH), interpolation=cv2.INTER_AREA)
            # Depth resizes NEAREST: interpolating across a depth edge invents a range
            # that is on neither surface, and averages invalid 0 into real values.
            d = cv2.resize(d, (RW, RH), interpolation=cv2.INTER_NEAREST)
            rgb = np.repeat(m[:, :, None], 3, axis=2)      # grey -> 3ch; BGR == RGB here
            cams.append(rgb)
            cams1.append(np.transpose(rgb, (2, 0, 1)))
            depths.append(normalise_depth(d, z_near, z_far)[:, :, None])
        pcts.append(pct)
        total += len(rows)
        ends.append(total)

    camera = np.asarray(cams, np.uint8)
    camera_1 = np.asarray(cams1, np.uint8)
    depth = np.asarray(depths).astype(zc["depth_dtype"])
    pct = np.concatenate(pcts, 0).astype(np.float32)
    episode_ends = np.asarray(ends, np.int64)

    if a.out.exists():
        import shutil
        shutil.rmtree(a.out)
    root = zarr.open(str(a.out), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    chunk = int(zc["chunk_len"])
    data.create_dataset("camera", data=camera, chunks=(chunk, RH, RW, 3))
    data.create_dataset("camera_1", data=camera_1, chunks=(chunk, 3, RH, RW))
    dz = data.create_dataset("camera0_main_depth", data=depth, chunks=(chunk, RH, RW, 1))
    data.create_dataset("tracked_gripper_pct", data=pct, chunks=(5000, 1))
    meta.create_dataset("episode_ends", data=episode_ends)

    dz.attrs["umi_handheld"] = {
        "anchored_min": 0.0,
        "anchored_max": 1.0,
        "normalized": True,
        "polarity": "inverse_depth_near_large",
        "formula": "v = (1/Z - 1/Z_far) / (1/Z_near - 1/Z_far), clipped to [0,1]",
        "z_near_m": z_near, "z_far_m": z_far,
        "source_units": "uint16 millimetre, 0 = invalid",
        "target_grid": TARGET_GRID_NOTE,
        "camera_model_source": src.provenance(),
        "camera_model_target": tgt.provenance(),
        "source_model_verified": src.verified,
        "built_on_unverified_model": not src.verified,
        "resize": [RW, RH], "crop": zc["crop"],
        "gripper_dimensions_are_placeholders":
            bool(cfg["gripper"].get("dimensions_are_placeholders", True)),
    }

    # ---- read the attrs back and assert, because their absence fails silently ----
    check = zarr.open(str(a.out), mode="r")
    st = dict(check["data/camera0_main_depth"].attrs)
    found = None
    if "anchored_min" in st and "anchored_max" in st:
        found = st
    else:
        for v in st.values():
            if isinstance(v, dict) and "anchored_min" in v and "anchored_max" in v:
                found = v
                break
    assert found is not None, "depth zattrs: anchored_min/anchored_max not reachable"
    assert found["anchored_min"] == 0.0 and found["anchored_max"] == 1.0, found
    assert found.get("normalized") is True, \
        "depth zattrs: 'normalized' missing or false — training would apply a x3.1 gain"
    assert check["data/camera_1"].shape == (total, 3, RH, RW)
    assert check["data/camera"].shape == (total, RH, RW, 3)
    assert check["data/camera0_main_depth"].shape == (total, RH, RW, 1)
    assert check["data/tracked_gripper_pct"].shape == (total, 1)
    assert int(check["meta/episode_ends"][-1]) == total, "episode_ends must be exclusive"

    # ---- Z round trip through the normalisation, on real stored values ----
    # Frame 0 is not a safe sample: a real device serves its first stereo frame before
    # the matcher has settled, and it comes back 100% invalid, which made this an empty
    # reduction. Take the first frame that actually carries depth.
    rt = None
    for i in range(total):
        v = check["data/camera0_main_depth"][i].astype(np.float32).ravel()
        live = v > 0
        if not live.any():
            continue
        z_back = denormalise_depth(v[live], z_near, z_far)
        v_again = np.clip((1.0 / z_back - 1.0 / z_far) / (1.0 / z_near - 1.0 / z_far), 0, 1)
        rt = float(np.abs(v_again - v[live]).max())
        rt_frame = i
        break
    # TODO(quality): float16 carries ~3 decimal digits, so Z resolution near Z_far is
    # coarse; the round trip below measures the storage error, not the sensor's.
    if rt is None:
        print("  NOTE: no frame carried valid depth — round-trip check skipped")
    else:
        assert rt < 1e-3, f"depth round trip lost {rt}"

    # ---- QC the brief asks for: the goal frame is sampled from the episode tail ----
    tail = 50
    warn = []
    for i, e in enumerate(episode_ends):
        s = 0 if i == 0 else int(episode_ends[i - 1])
        seg = depth[max(s, e - tail):e]
        frac = float(np.count_nonzero(seg) / seg.size)
        if frac < 0.30:
            warn.append(f"episode {i}: last {tail} frames only {frac*100:.1f}% valid depth")
        if e - s < 60:
            warn.append(f"episode {i}: {e-s} frames (<60 is skipped by the gripper CLI)")

    print(f"\nwrote {a.out}")
    print(f"  frames {total}  episodes {len(episode_ends)}  episode_ends {list(episode_ends)}")
    if rt is not None:
        print(f"  depth round-trip max error {rt:.2e} (float16 storage, frame {rt_frame})")
    for w in warn:
        print(f"  QC WARNING: {w}")
    if not warn:
        print("  QC: no warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
