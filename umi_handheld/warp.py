"""warp.py — re-project a session from the bench camera's optics onto the C3's.

The affordance predictor is trained on land and deployed underwater. A flat port changes
the effective focal length, so the same scene fills a different part of the frame on each
camera: the bench OAK-D W sees 127 deg horizontally in air, the C3 sees 85.6 deg through
its port. Feeding the model land-shaped frames and then showing it water-shaped ones is a
domain gap that costs nothing to remove here.

Direction of the map
--------------------
cv2.remap wants, for every pixel of the OUTPUT, where to read in the INPUT. So the map is
built target-pixel -> source-pixel:

    target pixel -> (target model) ray -> (source model) -> source pixel

Both cameras are treated as co-located and co-oriented. Only the projection changes; this
is not a viewpoint change, so no extrinsics are involved and none are needed.

Why not initUndistortRectifyMap alone
-------------------------------------
That function's output camera is a pinhole defined by (R, newCameraMatrix) — it cannot
express a target that still has distortion, and the C3 model very much does. Using it
would silently deliver an undistorted pinhole grid instead of the C3's grid. The map is
therefore built directly from the two models, with cv2.undistortPoints doing the target
unprojection (the C3 model has non-zero p1/p2, so the radial-only inverse does not apply).
The round trip is checked at build time rather than assumed.

TODO(quality): the target is the RAW CAM_B model. The C3 actually serves depth on the
rectified-left grid, whose P1/R1 need stereo extrinsics that the EEPROM dump does not
carry. Every artifact records target_grid = "raw_CAM_B (rectified model unavailable)".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from umi_handheld.camera_model import REPO, CameraModel        # noqa: E402
from umi_handheld.record import load_pipeline_config           # noqa: E402

TARGET_GRID_NOTE = "raw_CAM_B (rectified model unavailable)"

INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
          "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


class WarpStage:
    """Source-optics -> target-optics resampler, maps built once and reused."""

    def __init__(self, src: CameraModel, tgt: CameraModel, out_size, wcfg: dict,
                 source_size=None):
        self.src, self.tgt = src, tgt
        self.W, self.H = (int(v) for v in out_size)
        # The session's images need not be at the model's native resolution — the device
        # streams 400p while the calibration is stored at 800p. Projecting into the
        # native size and then remapping a half-size image reads off the edge and returns
        # border zeros, which looks like a depth failure rather than a units bug.
        self.SW, self.SH = (int(v) for v in (source_size or (src.width, src.height)))
        self.mono_interp = INTERP[wcfg.get("mono_interpolation", "linear")]
        self.depth_interp = INTERP[wcfg.get("depth_interpolation", "nearest")]
        if self.depth_interp != cv2.INTER_NEAREST:
            print("WARNING: depth interpolation is not nearest; invalid (0) pixels will "
                  "be blended into real ranges", file=sys.stderr)
        self._build()

    def _build(self):
        u, v = np.meshgrid(np.arange(self.W, dtype=np.float64),
                           np.arange(self.H, dtype=np.float64))
        px = np.stack([u.ravel(), v.ravel()], 1)
        rays = self.tgt.unproject(px, self.W, self.H)          # target pixel -> ray
        # Behind-camera rays cannot be projected; the target FOV is narrower than the
        # source so this should be empty, but it is checked rather than assumed.
        self.behind = int(np.count_nonzero(rays[:, 2] <= 0))
        src_px = self.src.project(rays, self.SW, self.SH)
        self.map_x = src_px[:, 0].reshape(self.H, self.W).astype(np.float32)
        self.map_y = src_px[:, 1].reshape(self.H, self.W).astype(np.float32)

        inside = ((self.map_x >= 0) & (self.map_x <= self.SW - 1) &
                  (self.map_y >= 0) & (self.map_y <= self.SH - 1))
        self.coverage = float(inside.mean())
        self.inside = inside

        # Round trip: take the mapped source pixels back through the source model to
        # rays and re-project with the target model; we should land where we started.
        # TODO(quality): this measures cv2.undistortPoints' iterative inverse on a
        # 127 deg model, not the forward map (which is exact). Measured 0.92 px on the
        # synthetic pair — fine for punch-through, worth replacing with the monotone
        # radial inverse in camera_model before anything metric depends on it.
        keep = inside.ravel()
        back = self.src.unproject(src_px[keep], self.SW, self.SH)
        again = self.tgt.project(back, self.W, self.H)
        self.roundtrip_px = float(np.abs(again - px[keep]).max())

    def info(self) -> dict:
        return {
            "target_grid": TARGET_GRID_NOTE,
            "out_size": [self.W, self.H],
            "source_size": [self.SW, self.SH],
            "source_model": self.src.provenance(),
            "target_model": self.tgt.provenance(),
            "coverage_fraction": self.coverage,
            "rays_behind_camera": self.behind,
            "roundtrip_max_px": self.roundtrip_px,
            "mono_interpolation": [k for k, v in INTERP.items() if v == self.mono_interp][0],
            "depth_interpolation": [k for k, v in INTERP.items() if v == self.depth_interp][0],
        }

    def apply_mono(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map_x, self.map_y, self.mono_interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def apply_depth(self, depth: np.ndarray) -> np.ndarray:
        # TODO(quality): nearest keeps 0 (invalid) from being averaged into real ranges,
        # but it still lets a single invalid sample win a whole output pixel.
        return cv2.remap(depth, self.map_x, self.map_y, self.depth_interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--session", type=Path, default=REPO / "sessions/session_0001")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <session>/warped")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_pipeline_config(a.config)
    wcfg = cfg["warp"]
    src = CameraModel.load(cfg["cameras"]["source"])
    tgt = CameraModel.load(cfg["cameras"]["target"])
    rows = list(csv.DictReader((a.session / "frames.csv").open()))
    out = a.out or (a.session / "warped")

    print(f"source  : {src.rel}  {src.width}x{src.height}  HFOV-ish fx={src.fx:.1f}")
    print(f"target  : {tgt.rel}  {tgt.width}x{tgt.height}  fx={tgt.fx:.1f}  "
          f"rectified={tgt.rectified}")
    print(f"out size: {wcfg['target_size']}   target_grid = {TARGET_GRID_NOTE}")
    if a.dry_run:
        print("\n--dry-run: config parsed, no maps built, nothing written.")
        return 0

    probe = cv2.imread(str(a.session / rows[0]["left_file"]), cv2.IMREAD_GRAYSCALE)
    sh, sw = probe.shape
    if (sw, sh) != (src.width, src.height):
        print(f"note    : session images are {sw}x{sh}, model native is "
              f"{src.width}x{src.height} — intrinsics scaled to the session size")
    stage = WarpStage(src, tgt, wcfg["target_size"], wcfg, source_size=(sw, sh))

    # Depth need not share the mono grid: the decimation filter shrinks depth alone.
    # Its map has to be built at the depth image's own size or every sample lands in
    # the wrong place — the same class of bug as reusing a 1280-wide map for a 640-wide
    # stream, which reported 100% coverage while the depth collapsed to near zero.
    dprobe = cv2.imread(str(a.session / rows[0]["depth_file"]), cv2.IMREAD_UNCHANGED)
    dh, dw = dprobe.shape
    if (dw, dh) != (sw, sh):
        print(f"note    : depth is {dw}x{dh}, mono is {sw}x{sh} (decimated) — "
              f"building a second map at the depth size")
        depth_stage = WarpStage(src, tgt, wcfg["target_size"], wcfg, source_size=(dw, dh))
    else:
        depth_stage = stage

    info = stage.info()
    info["depth_source_size"] = [dw, dh]
    print(f"coverage: {info['coverage_fraction']*100:.2f}% of target pixels fall inside "
          f"the source image  (behind-camera rays: {info['rays_behind_camera']})")
    print(f"map round-trip max error: {info['roundtrip_max_px']:.4f} px")

    for sub in ("left", "depth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    valid = []
    for row in rows:
        stamp = Path(row["left_file"]).stem
        m = cv2.imread(str(a.session / row["left_file"]), cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(str(a.session / row["depth_file"]), cv2.IMREAD_UNCHANGED)
        wm, wd = stage.apply_mono(m), depth_stage.apply_depth(d)
        cv2.imwrite(str(out / "left" / f"{stamp}.png"), wm,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(out / "depth" / f"{stamp}.png"), wd,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        valid.append(float(np.count_nonzero(wd)) / wd.size)

    info["frames"] = len(rows)
    info["depth_valid_fraction"] = {"min": min(valid), "max": max(valid),
                                    "mean": float(np.mean(valid))}
    (out / "warp.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} warped frames to {out}  "
          f"depth valid {min(valid)*100:.1f}%-{max(valid)*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
