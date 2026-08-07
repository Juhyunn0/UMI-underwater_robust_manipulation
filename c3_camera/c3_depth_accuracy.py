#!/usr/bin/env python3
"""
c3_depth_accuracy.py — how wrong is the C3's depth, at a tape-measured distance?

One invocation = one rung. Aim the camera at a flat wall, type in the tape
reading, hold still while it captures, and the script appends one row to a CSV.
Run it again at the next distance, and at the next setting. `--analyze` reads the
whole accumulated CSV back and fits what the camera reports against what the tape
says — with no camera attached, so the analysis can be re-run and corrected
forever.

    # rung 1 at 300 mm with the package defaults
    ~/.venvs/c3-depthai/bin/python c3_camera/c3_depth_accuracy.py \\
        --truth-mm 300 --label default --datum "housing front face"

    # the same rung with extended disparity, the only way this camera reaches 15 cm
    ~/.venvs/c3-depthai/bin/python c3_camera/c3_depth_accuracy.py \\
        --truth-mm 300 --extended --label ext

    # ... then 185, 240, 280, 330, 450, 560, 660, 1000, 2000 mm, both arms ...

    # offline: slope + intercept per setting, and extended off vs on
    ~/.venvs/c3-depthai/bin/python c3_camera/c3_depth_accuracy.py --analyze

    # what the geometry predicts for a setting, without opening the camera
    ~/.venvs/c3-depthai/bin/python c3_camera/c3_depth_accuracy.py --dry-run \\
        --truth-mm 185 --extended

What gets compared against the tape
-----------------------------------
Not the centre pixel, and not the mean depth over the region. Both are polluted
by camera tilt: at 1 m the default ROI covers roughly 500 mm of wall, so five
degrees of yaw spreads the true depth across it by +-22 mm — an order of
magnitude more than the error being measured. Instead the ROI is deprojected to
3-D and a plane is fitted robustly; the reported number is rho, the perpendicular
distance from the optical centre to that plane. rho is the quantity a tape
measures, and to first order it does not care how the camera is aimed. Tilt is
reported separately so a badly aimed rung is visible rather than silently wrong.

The depth origin is unknown, so it is fitted rather than assumed
---------------------------------------------------------------
The tape is pulled from some mechanical face you can hold a square against
(--datum records which one). Depth is measured from the optical centre of
whichever camera depth is aligned to, which sits behind the front glass and the
pressure port by an amount nobody here has measured. Call that offset c:

    rho = a * tape + c

`--analyze` fits a and c by least squares over the rungs at or beyond
--fit-min-mm; near rungs are excluded because that is where any curvature and any
below-MinZ garbage lives, and both would drag the intercept. c then comes out as
a *result* — and as a consistency check, because an optical centre cannot depend
on the mono resolution: if c disagrees between settings by more than a few mm,
something other than the origin is wrong. `a` is the scale error, and it is the
number that becomes the refraction scale (~1.33 if uncorrected) the day this is
repeated in water.

Traps this script is built around
---------------------------------
* Stale frames. `C3Source.read()` hands back the newest frame of each stream, so
  a host loop faster than the depth rate sees the same depth image several times.
  Counted as if fresh, duplicates would drive the temporal-noise estimate toward
  zero and make the camera look perfectly stable. Only bundles carrying a new
  depth sequence number are accepted; the duplicates are counted into the CSV.
* MinZ is geometry, not a preference: MinZ = fx * baseline / max_disparity. At
  mono 400p (fx 380.14 px, baseline 75 mm) that is 300.1 mm, and 150.1 mm with
  --extended — and both are confirmed by the recorded depth PNGs, which floor at
  exactly 300 and 150 mm with nothing below (see FX_MONO_1280). A rung below
  MinZ is a negative control — it is *supposed* to
  fail. If it does not, something downstream (median filter, hole fill) is
  inventing values, and then every fill number in the whole run is suspect. The
  script checks for that explicitly rather than reporting the number.
* Extended disparity cannot fix the near-field edge band. Where the two mono
  frames stop overlapping there is no stereo information at all, and that loss is
  a band down one side of the frame roughly fx*b/Z pixels wide — 34% of the frame
  at 130 mm. So holes are reported per left/centre/right third, never as a single
  average: the average hides exactly the structure that identifies the cause.
* --depth-align color crops the stereo field of view from 80.4 to 63.9 degrees
  horizontally, which pushes the overlap band out of frame entirely beyond about
  338 mm. Band numbers taken under `--depth-align color` and `--depth-align none`
  are not comparable; the CSV records which was used.
* The `robotics` preset leaves the confidence threshold at 245 out of 255, i.e.
  almost fully permissive, so a good part of the fill can be garbage. This script
  pins --confidence explicitly and records it. Left unpinned, a conclusion like
  "extended disparity cost us fill" can be a preset artefact instead.
* Depth 0 means "no measurement", not 0 mm — and c3_collect.py's display floor is
  300 mm, which is precisely the range extended disparity unlocks. Here --min-mm
  defaults to 100 mm so 149-300 mm depths are not silently thrown away.
* The median filter fills holes and flatters the residual, so measurement runs
  default to --median off. Run `--median 5x5 --label med5` as its own arm if you
  want to price the filter instead of being fooled by it.
* Quantisation runs the other way from every gate above, and it is the trap that
  looks like success. dZ = Z^2/(fx*b) is 141 mm at 2 m on 400p, while a wall
  three degrees off square spreads the ROI over only ~50 mm — so every pixel
  lands in the SAME disparity bin, the plane residual comes back near zero, the
  inlier fraction is 1.00, and rho is the bin rather than the wall. The bias then
  carries +-dZ/2 that nothing downstream of the residual can see. A properly
  sampled grid gives resid ~ dZ/sqrt(12); below dZ/6 the rung is warned as
  quantisation-limited, and the far rungs of the ladder want 800p (half the step)
  or a few degrees of deliberate yaw.

Refusals
--------
Every gate that fires lands in the `warn` and `refused` columns, and a refused
row is marked ok=0 so `--analyze` leaves it out of the fit. The script would much
rather print "refused: ROI only 12% valid" than a confident bias computed from
12% of a region that may not even be pointed at the target. Exit code 0 means a
number worth standing behind, 1 means refused or a problem found.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from c3_camera import config as C
from c3_camera import preflight as PF
from c3_camera import viz
from c3_camera.config import (
    MONO_RESOLUTIONS,
    STREAM_COLOR,
    STREAM_DEPTH,
    STREAM_LEFT,
    StreamConfig,
)
from c3_camera.geometry import Intrinsics, depth_to_xyz

# =============================================================================
# Fixed hardware geometry
# =============================================================================
# Stereo baseline, from the factory calibration on this unit (7.5 cm).
BASELINE_MM = 75.0

# The block matcher's disparity search span. 95 is the DepthAI default; extended
# disparity runs a second pass at half resolution and doubles it. Note that
# `StereoDepth.initialConfig.getMaxDisparity()` does NOT reflect the extended
# flag on the host side — it returned 8170.0 for every configuration probed — so
# MinZ has to be computed from these constants. It has since been confirmed
# against real data — see FX_MONO_1280 below.
MAX_DISPARITY = 95
MAX_DISPARITY_EXTENDED = 190

# fx that the DEVICE's rectified stereo pair actually behaves like, at 1280x800.
#
# This is CAM_C's EEPROM value, and which of the two the pair inherits is settled,
# not assumed. CAM_B reads 757.12 and CAM_C 760.28 (a 0.42% difference), which put
# MinZ at 298.9 vs 300.1 mm at 400p. Every depth PNG this camera has ever written
# floors at exactly 300 mm, with ZERO pixels below it, and the --extended
# recordings floor at exactly 150 mm, also with zero below:
#
#   [측정] datasets/dataset_20260803_102816/depth/  floor 300 mm, 877,378 px at it
#          datasets/dataset_20260803_104235/depth/  floor 300 mm, 2,159,952 px at it
#          recordings/20260729_162216/depth/        floor 300 mm, 4.61% of all valid
#          datasets/dataset_20260803_162223/depth/  floor 150 mm (extended)
#          datasets/dataset_20260803_162025/depth/  floor 150 mm (extended)
#          (47,270 frames / 3.6e9 valid pixels scanned, 2026-08-05)
#
# A floor that prints as exactly 300 requires fx*B/95 in [300, 301), i.e. fx_rect
# in [380.0, 381.3) at 640 wide. CAM_C's 380.14 is inside; CAM_B's 378.56 would
# have printed 298 and never does. The 300:150 ratio being exactly 2:1 is the
# disparity span doubling and nothing else — there is no threshold filter and no
# disparityShift in this pipeline, and neither could produce that ratio anyway.
# host_depth.py's independent host-side rectification measured ~380.00, which
# agrees with 380.14 to 0.04%.
FX_MONO_1280 = 760.28
FX_MONO_1280_CAM_B = 757.12          # the other EEPROM value, kept for reference

# CAM_A (colour) fx at 3840x2160, same EEPROM. Needed because --depth-align color
# delivers depth on the colour grid, and the colour lens is much narrower than
# the stereo pair: 63.9 vs 80.4 degrees horizontally. That difference is what
# decides whether the near-field overlap band is visible at all.
FX_COLOR_3840 = 3080.35
COLOR_NATIVE_W = 3840

DEFAULT_CSV = Path(__file__).resolve().parent / "depth_accuracy" / "rungs.csv"


# =============================================================================
# Predicted geometry — what the optics say before any pixel is measured
# =============================================================================
def fx_mono_px(mono_res: str, fx_1280: float = FX_MONO_1280) -> float:
    """Rectified mono focal length at the streamed resolution.

    400p is 2x2 binning of the same sensor area, so fx halves while the field of
    view is untouched. 720p and 800p share a width of 1280 and therefore share
    fx: 720p is a vertical crop of 800p, which is why the two are indistinguish-
    able in every depth quantity and 720p is a control rather than a rung.
    """
    if mono_res not in MONO_RESOLUTIONS:
        raise ValueError(f"unknown mono resolution {mono_res!r}")
    width = MONO_RESOLUTIONS[mono_res][1]
    return fx_1280 * width / 1280.0


def min_z_mm(fx: float, extended: bool, baseline_mm: float = BASELINE_MM) -> float:
    """Closest distance the matcher can represent: fx * baseline / max_disparity."""
    d = MAX_DISPARITY_EXTENDED if extended else MAX_DISPARITY
    return fx * baseline_mm / d


def quantisation_mm(z_mm: float, fx: float, baseline_mm: float = BASELINE_MM) -> float:
    """Depth step for one disparity step: dZ = Z^2 / (fx * baseline).

    This is the real price of the mono-resolution axis. At 400p it is 3.2 mm at
    300 mm and 141 mm at 2 m; at 1280-wide it is exactly half of both.
    """
    if fx <= 0 or baseline_mm <= 0:
        return float("nan")
    return z_mm * z_mm / (fx * baseline_mm)


def overlap_band_px(z_mm: float, fx: float, baseline_mm: float = BASELINE_MM) -> float:
    """Width of the un-matchable band at one edge of the *rectified mono* frame.

    For parallel rectified cameras a point is only visible to both when it is at
    least b/Z inside the far edge in tangent units, which in pixels is exactly
    fx*b/Z. It is pure geometry: no confidence threshold, no extended disparity
    and no preset can recover it, because the second camera never saw that strip.
    Its *fraction* of the frame is resolution-independent, since fx/width is.
    """
    if z_mm <= 0:
        return float("nan")
    return fx * baseline_mm / z_mm


def tan_half_fov(width_px: float, fx: float) -> float:
    return 0.5 * width_px / fx


def overlap_band_frac(z_mm: float, tan_half_img: float, tan_half_stereo: float,
                      baseline_mm: float = BASELINE_MM) -> float:
    """Fraction of the *delivered* frame width lost to stereo overlap at range Z.

    A point at tangent angle t in the right camera is also seen by the left one
    only while t <= tan(theta_stereo) - b/Z. Expressed in the frame that is
    actually shipped — which after --depth-align color is the much narrower
    colour frame — the lost fraction is

        (tan(theta_img) - (tan(theta_stereo) - b/Z)) / (2 tan(theta_img))

    clamped to [0, 1]. When the delivered frame is the mono frame itself this
    reduces to fx*b/(Z*W), i.e. `overlap_band_px` over the width. The general
    form matters because aligning to colour crops the stereo field of view from
    80.4 to 63.9 degrees, which hides the band entirely beyond ~338 mm: measure
    the band under --depth-align color and the formula looks wrong.
    """
    if z_mm <= 0 or tan_half_img <= 0:
        return float("nan")
    lost = tan_half_img - (tan_half_stereo - baseline_mm / z_mm)
    return min(1.0, max(0.0, lost / (2.0 * tan_half_img)))


def band_vanish_mm(tan_half_img: float, tan_half_stereo: float,
                   baseline_mm: float = BASELINE_MM) -> float:
    """Range beyond which the overlap band is outside the delivered frame."""
    gap = tan_half_stereo - tan_half_img
    if gap <= 0:
        return float("inf")     # the delivered frame is as wide as the stereo pair
    return baseline_mm / gap


@dataclass
class Prediction:
    """Everything the optics fix before the camera is even switched on."""

    mono_res: str
    extended: bool
    fx_mono: float
    max_disparity: int
    min_z_mm: float
    quant_mm: float             # at the tape distance, INTEGER disparity only
    band_px: float              # in the rectified mono frame, at the tape distance
    band_frac: float            # as a fraction of the mono width
    mono_width: int
    # The same band as it appears in the frame actually delivered — after any
    # depth-align crop. This is what the per-third hole numbers can see.
    delivered_width: int = 0
    delivered_band_frac: float = float("nan")
    delivered_band_px: float = float("nan")
    delivered_vanish_mm: float = float("nan")
    # Subpixel interpolation subdivides every disparity step (3 fractional bits =
    # 1/8 on DepthAI 2.x, more on later firmware), so `quant_mm` becomes an upper
    # bound rather than the step. Carried here so the quantisation-limited warning
    # does not fire on every subpixel rung, where the grid it complains about has
    # been subdivided out of existence.
    subpixel: bool = False


def predict(mono_res: str, extended: bool, truth_mm: float,
            fx_1280: float = FX_MONO_1280,
            fx_delivered: float | None = None,
            width_delivered: int | None = None,
            subpixel: bool = False) -> Prediction:
    fx = fx_mono_px(mono_res, fx_1280)
    width = MONO_RESOLUTIONS[mono_res][1]
    band = overlap_band_px(truth_mm, fx) if truth_mm and truth_mm > 0 else float("nan")
    p = Prediction(
        mono_res=mono_res,
        extended=extended,
        fx_mono=fx,
        max_disparity=MAX_DISPARITY_EXTENDED if extended else MAX_DISPARITY,
        min_z_mm=min_z_mm(fx, extended),
        quant_mm=quantisation_mm(truth_mm, fx) if truth_mm else float("nan"),
        band_px=band,
        band_frac=band / width if band == band else float("nan"),
        mono_width=width,
        subpixel=subpixel,
    )
    if fx_delivered and width_delivered:
        t_img = tan_half_fov(width_delivered, fx_delivered)
        t_st = tan_half_fov(width, fx)
        p.delivered_width = int(width_delivered)
        p.delivered_vanish_mm = band_vanish_mm(t_img, t_st)
        if truth_mm and truth_mm > 0:
            p.delivered_band_frac = overlap_band_frac(truth_mm, t_img, t_st)
            p.delivered_band_px = p.delivered_band_frac * width_delivered
    return p


def load_focal_lengths(path: Path | None) -> tuple[float, float, str]:
    """(mono fx @1280, colour fx @3840, provenance) — from a recorded
    calibration.json if given, otherwise this camera's EEPROM values.

    CAM_C, not CAM_B: the observed depth floors say the rectified pair follows the
    right camera's focal length. See FX_MONO_1280. CAM_B is the fallback only
    because a calibration.json missing CAM_C is better served by 0.42% off than by
    an exception.
    """
    if path is None:
        return (FX_MONO_1280, FX_COLOR_3840,
                "built-in (this camera's EEPROM: CAM_C @1280x800, CAM_A @3840x2160)")
    data = json.loads(Path(path).read_text())
    intr = (data.get("device", {}).get("intrinsics", {}) or {})
    cam_b = intr.get("CAM_C") or intr.get("CAM_B")
    if not cam_b or "fx" not in cam_b:
        raise ValueError(f"{path} has no device/intrinsics/CAM_C/fx (nor CAM_B)")
    bw = (cam_b.get("default_size") or [1280, 800])[0]
    fx_mono = float(cam_b["fx"]) * 1280.0 / float(bw)
    cam_a = intr.get("CAM_A") or {}
    if "fx" in cam_a:
        aw = (cam_a.get("default_size") or [COLOR_NATIVE_W, 2160])[0]
        fx_color = float(cam_a["fx"]) * COLOR_NATIVE_W / float(aw)
    else:
        fx_color = FX_COLOR_3840
    return fx_mono, fx_color, str(path)


def delivered_optics(cfg, rc, fx_mono_1280: float,
                     fx_color_native: float) -> tuple[float, int]:
    """fx and width of the frame depth is actually delivered in.

    Aligning to colour re-renders depth through CAM_A's much narrower lens, so
    the frame the hole statistics are computed on is not the stereo frame. Any
    other alignment leaves depth on the rectified mono grid.
    """
    dw = rc.depth_out_size[0]
    if cfg.depth_align == "color":
        return fx_color_native * dw / COLOR_NATIVE_W, dw
    return fx_mono_px(cfg.mono_res, fx_mono_1280) * dw / rc.mono_size[0], dw


# =============================================================================
# Plane fitting — the measurement itself
# =============================================================================
@dataclass
class PlaneFit:
    """A plane n.X = rho fitted to a deprojected region, with its residuals."""

    ok: bool
    rho_mm: float = float("nan")
    tilt_deg: float = float("nan")
    resid_rms_mm: float = float("nan")   # expressed along the camera Z axis
    resid_mad_mm: float = float("nan")
    inlier_frac: float = float("nan")
    n_points: int = 0
    normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reason: str = ""


def robust_plane_fit(points_mm: np.ndarray, trim_sigma: float = 3.0,
                     iterations: int = 2, floor_mm: float = 1.0,
                     min_points: int = 50) -> PlaneFit:
    """Deterministic trimmed least-squares plane through an (N,3) point set.

    Trimmed rather than RANSAC on purpose: a measurement script has to give the
    same answer twice on the same data, and RANSAC's answer depends on its seed.
    The trim threshold is max(3 * 1.4826 * MAD, floor_mm); the floor exists
    because heavily quantised depth can put every point on one disparity level,
    making MAD exactly 0 and the threshold reject everything.

    The residual RMS is divided by |n_z| so it is reported along the camera's Z
    axis. Depth error is a Z-axis quantity; the perpendicular residual shrinks by
    cos(tilt), which would flatter a tilted rung.
    """
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        return PlaneFit(False, reason=f"expected (N,3) points, got {points_mm.shape}")
    n_total = points_mm.shape[0]
    if n_total < min_points:
        return PlaneFit(False, n_points=n_total,
                        reason=f"only {n_total} valid points (need {min_points})")

    pts = np.asarray(points_mm, dtype=np.float64)
    normal = np.array([0.0, 0.0, 1.0])
    centroid = pts.mean(axis=0)

    # Seed from the median DEPTH, not from a least-squares plane through
    # everything. A coherent block of outliers — the edge of a doorway, a table
    # in front of the wall — tilts the initial least-squares plane so far that
    # the trimming then converges onto the compromise instead of onto the wall.
    # The median depth of a region that is mostly wall is on the wall, and a
    # MAD threshold around it scales with the plane's own depth spread, so this
    # does not clip a legitimately tilted plane.
    z = pts[:, 2]
    z_med = float(np.median(z))
    z_mad = float(np.median(np.abs(z - z_med)))
    inl = np.abs(z - z_med) <= max(trim_sigma * 1.4826 * z_mad, floor_mm)
    if int(inl.sum()) < min_points:
        inl = np.ones(n_total, dtype=bool)

    for _ in range(iterations + 1):
        sel = pts[inl]
        if sel.shape[0] < min_points:
            return PlaneFit(False, n_points=n_total,
                            reason=f"trimming left only {sel.shape[0]} points")
        centroid = sel.mean(axis=0)
        # Smallest right singular vector of the centred points = plane normal.
        _, _, vt = np.linalg.svd(sel - centroid, full_matrices=False)
        normal = vt[-1]
        resid = (pts - centroid) @ normal
        med = float(np.median(resid[inl]))
        mad = float(np.median(np.abs(resid[inl] - med)))
        thr = max(trim_sigma * 1.4826 * mad, floor_mm)
        inl = np.abs(resid - med) <= thr

    sel = pts[inl]
    if sel.shape[0] < min_points:
        return PlaneFit(False, n_points=n_total,
                        reason=f"trimming left only {sel.shape[0]} points")

    rho = float(normal @ centroid)
    if rho < 0:                      # orient the normal away from the camera
        normal = -normal
        rho = -rho
    nz = abs(float(normal[2]))
    if nz < 1e-6:
        return PlaneFit(False, n_points=n_total,
                        reason="plane is edge-on to the camera (n_z ~ 0)")

    resid = (sel - centroid) @ normal
    rms = float(np.sqrt(np.mean(resid ** 2))) / nz
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826 / nz
    tilt = math.degrees(math.acos(min(1.0, nz)))

    return PlaneFit(
        ok=True, rho_mm=rho, tilt_deg=tilt, resid_rms_mm=rms, resid_mad_mm=mad,
        inlier_frac=float(inl.mean()), n_points=n_total,
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
    )


def roi_slice(height: int, width: int,
              frac: tuple[float, float]) -> tuple[int, int, int, int]:
    """Centred region of interest as (row0, row1, col0, col1)."""
    fw, fh = frac
    rw = max(2, int(round(width * fw)))
    rh = max(2, int(round(height * fh)))
    c0 = (width - rw) // 2
    r0 = (height - rh) // 2
    return r0, r0 + rh, c0, c0 + rw


def thirds_hole_fraction(fill_map: np.ndarray) -> dict:
    """Hole fraction in the left / centre / right thirds of the frame.

    Reported separately because the structural near-field loss is a band at one
    edge, not a uniform sprinkle: an overall fill of 70% means something quite
    different when the missing 30% is one solid column band than when it is
    speckle. `hole_asym` is left minus right, and its sign points at which
    camera's field of view ran out.
    """
    w = fill_map.shape[1]
    a, b = w // 3, 2 * (w // 3)
    left = 1.0 - float(fill_map[:, :a].mean())
    centre = 1.0 - float(fill_map[:, a:b].mean())
    right = 1.0 - float(fill_map[:, b:].mean())
    return {"hole_left": left, "hole_centre": centre, "hole_right": right,
            "hole_asym": left - right}


def column_fill_profile(fill_map: np.ndarray, bins: int = 24) -> list[float]:
    """Per-column fill, binned — the shape that distinguishes a band from noise."""
    cols = fill_map.mean(axis=0)
    edges = np.linspace(0, cols.size, bins + 1).astype(int)
    return [float(cols[a:b].mean()) if b > a else float("nan")
            for a, b in zip(edges[:-1], edges[1:])]


def measured_quantisation_mm(values: np.ndarray, min_levels: int = 8) -> float:
    """Modal gap between occupied depth levels — the quantisation as it really is.

    Settles what subpixel and extended disparity actually did to the depth grid
    without arguing about the firmware. Meaningless once depth has been resampled
    (aligning to colour interpolates), so read it together with depth_align.
    """
    v = np.unique(values[values > 0].astype(np.int64))
    if v.size < min_levels:
        return float("nan")
    gaps = np.diff(v)
    gaps = gaps[gaps > 0]
    # A starved or garbage ROI — a below-MinZ negative control, a texture-less
    # wall — can put every occupied level more than a metre from its neighbour.
    # There is no depth grid to report then, and an unguarded bincount over the
    # empty selection raises, which would take the whole captured rung down with
    # it *after* the hold is over.
    gaps = gaps[gaps < 1000]
    if gaps.size == 0:
        return float("nan")
    return float(np.argmax(np.bincount(gaps)))


# =============================================================================
# The per-rung measurement, as a pure function of a depth stack
# =============================================================================
def measure_stack(depth_stack: np.ndarray, intr: Intrinsics,
                  roi_frac: tuple[float, float] = (0.4, 0.4),
                  min_mm: float = 100.0, max_mm: float = 20000.0,
                  trim_sigma: float = 3.0, quant_hint_mm: float = 1.0,
                  bins: int = 24) -> dict:
    """Every depth statistic for one rung, from an (N,H,W) uint16 millimetre stack.

    Deliberately pure: capture writes the stack, this computes the numbers, and
    the test feeds it a synthetic plane with a known bias, noise and hole pattern.
    Anything that needs the camera stays out of here.

    Statistics are computed per frame and then reduced with a median across
    frames, never by averaging the frames into one map first. Averaging first
    would erase the temporal noise, which is the half of the error budget that
    integration can remove and therefore the half worth separating out.
    """
    stack = np.asarray(depth_stack)
    if stack.ndim != 3:
        raise ValueError(f"expected an (N,H,W) depth stack, got {stack.shape}")
    n_frames, h, w = stack.shape
    if n_frames == 0:
        raise ValueError("empty depth stack")

    r0, r1, c0, c1 = roi_slice(h, w, roi_frac)
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[r0:r1, c0:c1] = True

    intr_used = intr if (intr.width, intr.height) == (w, h) else intr.scaled_to(w, h)

    rho, tilt, rms, mad, inlier, fill_roi, patch_std, z_centre = ([] for _ in range(8))
    fill_all: list[float] = []
    fill_accum = np.zeros((h, w), dtype=np.float64)
    fit_reason = ""

    cr0, cr1 = h // 2 - 5, h // 2 + 6
    cc0, cc1 = w // 2 - 5, w // 2 + 6

    for i in range(n_frames):
        d = stack[i]
        valid = (d > 0) & (d >= min_mm) & (d <= max_mm)
        fill_accum += valid
        fill_all.append(float(valid.mean()))

        roi_valid = valid & roi_mask
        fill_roi.append(float(roi_valid.sum()) / float(roi_mask.sum()))

        # Zero everything outside the ROI and reuse the package deprojection, so
        # pixel coordinates stay absolute. Cropping the array first would restart
        # u,v at 0 and silently move the principal point.
        masked = np.where(roi_valid, d, 0).astype(np.uint16)
        xyz_mm = depth_to_xyz(masked, intr_used, 1, min_mm, max_mm).astype(np.float64)
        xyz_mm *= 1000.0

        fit = robust_plane_fit(xyz_mm, trim_sigma=trim_sigma,
                               floor_mm=max(1.0, quant_hint_mm))
        if fit.ok:
            rho.append(fit.rho_mm)
            tilt.append(fit.tilt_deg)
            rms.append(fit.resid_rms_mm)
            mad.append(fit.resid_mad_mm)
            inlier.append(fit.inlier_frac)
        elif not fit_reason:
            fit_reason = fit.reason

        # The naive alternative, kept so the difference is visible rather than
        # asserted: the raw spread of ROI depths still contains the tilt.
        roi_vals = d[roi_valid]
        patch_std.append(float(roi_vals.std()) if roi_vals.size > 8 else float("nan"))

        centre = d[cr0:cr1, cc0:cc1]
        centre = centre[(centre > 0) & (centre >= min_mm) & (centre <= max_mm)]
        z_centre.append(float(np.median(centre)) if centre.size else float("nan"))

    fill_map = fill_accum / n_frames

    # ---- temporal noise ---------------------------------------------------
    # Only pixels valid in EVERY frame take part, so a pixel that drops in and
    # out cannot contribute a step change that reads as noise. Common mode is
    # then removed, because a rig creeping 1 mm during the hold would otherwise
    # appear in every pixel's time series as if the sensor had produced it.
    roi_vals = stack[:, r0:r1, c0:c1].astype(np.float64)
    roi_ok = (roi_vals > 0) & (roi_vals >= min_mm) & (roi_vals <= max_mm)
    always = roi_ok.all(axis=0)
    temporal_rms = temporal_med = float("nan")
    if n_frames >= 3 and int(always.sum()) >= 20:
        series = roi_vals[:, always]
        common = np.median(series, axis=1)
        series = series - (common - np.median(common))[:, None]
        sd = series.std(axis=0, ddof=1)
        temporal_rms = float(np.sqrt(np.mean(sd ** 2)))
        temporal_med = float(np.median(sd))

    def med(xs: list[float]) -> float:
        arr = np.asarray([x for x in xs if x == x], dtype=np.float64)
        return float(np.median(arr)) if arr.size else float("nan")

    def iqr(xs: list[float]) -> float:
        arr = np.asarray([x for x in xs if x == x], dtype=np.float64)
        if arr.size < 4:
            return float("nan")
        return float(np.percentile(arr, 75) - np.percentile(arr, 25))

    resid_rms = med(rms)
    # Single-frame spatial residual carries fixed pattern + random noise;
    # per-pixel temporal spread carries the random part alone. Subtracting in
    # quadrature separates the error that averaging removes from the error that
    # survives any amount of it — a distinction SLAM cares about a great deal.
    fixed_rms = float("nan")
    if resid_rms == resid_rms and temporal_rms == temporal_rms:
        fixed_rms = math.sqrt(max(0.0, resid_rms ** 2 - temporal_rms ** 2))

    all_valid = stack[(stack > 0) & (stack >= min_mm) & (stack <= max_mm)]
    mid = stack[n_frames // 2][r0:r1, c0:c1]

    out = {
        "n_frames": n_frames,
        "depth_w": w, "depth_h": h,
        "roi_w_px": c1 - c0, "roi_h_px": r1 - r0,
        "n_fits": len(rho),
        "fit_reason": fit_reason,
        "fill_all": med(fill_all),
        "fill_roi": med(fill_roi),
        "fill_roi_inlier": (med(fill_roi) * med(inlier)) if inlier else float("nan"),
        "inlier_frac": med(inlier),
        "rho_mm": med(rho),
        "rho_iqr_mm": iqr(rho),
        "rho_range_mm": (float(max(rho) - min(rho)) if len(rho) > 1 else float("nan")),
        "tilt_deg": med(tilt),
        "resid_rms_mm": resid_rms,
        "resid_mad_mm": med(mad),
        "patch_std_mm": med(patch_std),
        "temporal_rms_mm": temporal_rms,
        "temporal_med_mm": temporal_med,
        "temporal_valid_frac": float(always.mean()),
        "fixed_rms_mm": fixed_rms,
        "z_center_mm": med(z_centre),
        "quant_dz_meas_mm": measured_quantisation_mm(mid),
        "col_fill_profile": column_fill_profile(fill_map, bins),
    }
    out.update(thirds_hole_fraction(fill_map))
    if all_valid.size:
        out.update({
            "z_min_mm": float(all_valid.min()),
            "z_p01_mm": float(np.percentile(all_valid, 1)),
            "z_p50_mm": float(np.percentile(all_valid, 50)),
            "z_p99_mm": float(np.percentile(all_valid, 99)),
        })
    else:
        out.update({"z_min_mm": float("nan"), "z_p01_mm": float("nan"),
                    "z_p50_mm": float("nan"), "z_p99_mm": float("nan")})
    return out


# =============================================================================
# Gates — the conditions under which no number is worth printing
# =============================================================================
@dataclass
class Gates:
    """Thresholds for refusing to answer. All in the CSV, so a run is auditable."""

    min_frames: int = 10
    min_fill_roi: float = 0.30
    min_inlier: float = 0.60
    max_resid_mm: float | None = None    # None -> derived per rung, see below
    max_tilt_deg: float = 15.0
    max_exposure_ms: float = 20.0
    max_motion: float = 6.0
    max_rho_iqr_mm: float = 5.0


def resid_limit_mm(truth_mm: float, quant_mm: float,
                   override: float | None = None) -> float:
    """How flat a real wall has to look before the fit is believable.

    Scaled with distance for two independent reasons: the disparity quantisation
    step grows as Z^2, and any residual tilt or wall texture projects to more
    millimetres further away. A fixed threshold would pass everything at 2 m and
    fail everything at 150 mm.

    One quantisation step is the term that usually binds. Quantisation alone
    contributes only dZ/sqrt(12) = 0.29 dZ of residual RMS, so allowing a full
    step leaves 3.4x of headroom for the wall's own flatness and for the fit,
    while still catching a region that is not one plane at all.
    """
    if override is not None:
        return override
    return max(8.0, (quant_mm if quant_mm == quant_mm else 0.0), 0.02 * truth_mm)


def apply_gates(m: dict, truth_mm: float, pred: Prediction, gates: Gates,
                exposure_ms: float = float("nan"),
                exposure_spread_ms: float = float("nan"),
                motion: float = float("nan")) -> tuple[int, list[str], list[str]]:
    """Return (ok, warnings, refusals) for one measured rung.

    A refusal means the row's bias and noise must not be believed and --analyze
    will skip it. A warning means the number stands but something about the setup
    deserves a look before it is quoted.
    """
    warns: list[str] = []
    fails: list[str] = []

    below_min_z = truth_mm < pred.min_z_mm

    if m["n_frames"] < gates.min_frames:
        fails.append(f"only {m['n_frames']} unique depth frames "
                     f"(need {gates.min_frames}) — the link or the fps is starving this")
    if m["n_fits"] == 0:
        fails.append(f"no frame produced a plane fit: {m['fit_reason'] or 'unknown'}")

    fill = m["fill_roi"]
    if fill != fill or fill < gates.min_fill_roi:
        msg = (f"ROI is only {100 * (fill if fill == fill else 0):.1f}% valid "
               f"(need {100 * gates.min_fill_roi:.0f}%)")
        if below_min_z:
            # Expected, not a defect: below MinZ there is nothing to match.
            warns.append(msg + f" — but {truth_mm:.0f} mm is below the predicted "
                               f"MinZ {pred.min_z_mm:.0f} mm, so this is the "
                               f"negative control behaving correctly")
            fails.append("below MinZ: no depth to measure (expected)")
        else:
            fails.append(msg)

    inl = m["inlier_frac"]
    if inl == inl and inl < gates.min_inlier:
        fails.append(f"only {100 * inl:.0f}% of ROI depths lie on the fitted plane "
                     f"— the target may not fill the ROI, or the depth is mostly "
                     f"outliers")

    limit = resid_limit_mm(truth_mm, pred.quant_mm, gates.max_resid_mm)
    rms = m["resid_rms_mm"]
    if rms == rms and rms > limit:
        fails.append(f"plane residual {rms:.1f} mm exceeds {limit:.1f} mm — the "
                     f"target is not flat, does not fill the ROI, or the depth is "
                     f"noise rather than surface")

    # The opposite failure, and the more dangerous one because it looks like an
    # excellent measurement. A disparity step of dZ = Z^2/(fx*b) grows fast: at
    # 2 m on 400p it is 141 mm, while a wall three degrees off square only spreads
    # the ROI over ~50 mm. Every pixel then lands in the SAME bin, so the plane
    # fit reports a residual of nearly zero, an inlier fraction of 1.0, and an rho
    # that is the bin rather than the wall — a bias carrying the full +-dZ/2 of
    # quantisation, which no gate downstream of the residual can see. A properly
    # dithered grid gives resid ~ dZ/sqrt(12) = 0.29 dZ, so a residual below dZ/6
    # means the grid is not being sampled at all.
    quant = pred.quant_mm
    if (not pred.subpixel and quant == quant and quant > 0
            and rms == rms and rms < quant / 6.0):
        warns.append(f"quantisation-limited: the ROI spreads only {rms:.2f} mm "
                     f"across a {quant:.0f} mm disparity step, so rho is a single-bin "
                     f"readout and this bias carries +-{quant / 2:.0f} mm the residual "
                     f"cannot show. Aim a few degrees off square, or measure this "
                     f"distance at 800p where the step is half as coarse, before "
                     f"quoting it")

    # The centre ROI has to sit clear of the near-field overlap band, or part of
    # the bias is being fitted to pixels no stereo pair ever saw. Compared in
    # delivered pixels on both sides: the mono-frame band is wider and mixing the
    # two units would cry wolf on every --depth-align color run.
    band_px = pred.delivered_band_px
    margin_px = (float(m.get("depth_w", 0)) - float(m.get("roi_w_px", 0))) / 2.0
    if band_px == band_px and margin_px > 0 and band_px > margin_px:
        warns.append(f"the predicted overlap band ({band_px:.0f} px of the delivered "
                     f"frame) is wider than the ROI margin ({margin_px:.0f} px), so "
                     f"part of the ROI is on pixels the stereo pair could not see — "
                     f"shrink --roi or move back")

    tilt = m["tilt_deg"]
    if tilt == tilt and tilt > gates.max_tilt_deg:
        warns.append(f"camera tilted {tilt:.1f} deg to the wall; rho is still the "
                     f"perpendicular distance, but re-aim before trusting the "
                     f"per-third hole numbers")

    if exposure_ms == exposure_ms and exposure_ms > gates.max_exposure_ms:
        note = (f"exposure {exposure_ms:.1f} ms (limit {gates.max_exposure_ms:.0f}) "
                f"— long enough that any rig motion smears the stereo pair")
        if motion == motion and motion > gates.max_motion / 2:
            fails.append(note + f", and the scene moved (madiff {motion:.1f})")
        else:
            warns.append(note + "; add light or pin the exposure")

    if (exposure_spread_ms == exposure_spread_ms and exposure_ms == exposure_ms
            and exposure_ms > 0 and exposure_spread_ms > 0.25 * exposure_ms):
        warns.append(f"exposure moved {exposure_spread_ms:.1f} ms during the hold "
                     f"— auto-exposure is hunting; different rungs are then "
                     f"compared at different exposures, not different geometry")

    if motion == motion and motion > gates.max_motion:
        fails.append(f"the rig or the scene moved during the hold "
                     f"(frame-to-frame madiff {motion:.1f} grey levels)")

    iqr = m["rho_iqr_mm"]
    if iqr == iqr and iqr > gates.max_rho_iqr_mm:
        warns.append(f"rho wandered {iqr:.1f} mm across the hold (IQR) — the rig "
                     f"is not still, or the matcher is bistable here")

    rho = m["rho_mm"]
    if rho == rho and truth_mm > 0 and abs(rho - truth_mm) > 0.30 * truth_mm:
        warns.append(f"reported {rho:.0f} mm against a tape of {truth_mm:.0f} mm "
                     f"— that is more than the datum offset can explain; check the "
                     f"tape, the datum, and that the ROI is on the target")

    if below_min_z and fill == fill and fill > 0.5:
        warns.append(f"{100 * fill:.0f}% of the ROI is 'valid' at {truth_mm:.0f} mm, "
                     f"below the geometric MinZ of {pred.min_z_mm:.0f} mm — "
                     f"something is inventing depth (median/hole fill), which makes "
                     f"every fill figure in this run suspect")

    return (0 if fails else 1), warns, fails


# =============================================================================
# Straight-line fit of reported against truth
# =============================================================================
@dataclass
class LineFit:
    """rho = slope * tape + intercept, with the uncertainty on both."""

    ok: bool
    n: int = 0
    slope: float = float("nan")
    slope_se: float = float("nan")
    intercept_mm: float = float("nan")
    intercept_se_mm: float = float("nan")
    r2: float = float("nan")
    resid_rms_mm: float = float("nan")
    reason: str = ""


def fit_reported_vs_truth(truth_mm, reported_mm) -> LineFit:
    """Ordinary least squares, with the standard errors that make it usable.

    The intercept is the whole point: it is the unknown offset between the datum
    the tape was pulled from and the optical centre depth is measured from. Its
    standard error decides whether an apparent 20 mm origin offset is real or is
    two noisy rungs pretending.
    """
    x = np.asarray(truth_mm, dtype=np.float64)
    y = np.asarray(reported_mm, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = x.size
    if n < 2:
        return LineFit(False, n=n, reason="need at least 2 usable rungs")
    if np.allclose(x, x[0]):
        return LineFit(False, n=n, reason="all rungs are at the same distance")

    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    sxy = float(((x - xbar) * (y - ybar)).sum())
    slope = sxy / sxx
    intercept = float(ybar - slope * xbar)
    pred = slope * x + intercept
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - ybar) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    resid_rms = math.sqrt(sse / n)

    if n > 2:
        s2 = sse / (n - 2)
        slope_se = math.sqrt(s2 / sxx)
        intercept_se = math.sqrt(s2 * (1.0 / n + xbar ** 2 / sxx))
    else:
        # Two points define a line exactly; there is no residual to estimate a
        # standard error from, and pretending otherwise would report 0.
        slope_se = intercept_se = float("nan")

    return LineFit(True, n=n, slope=slope, slope_se=slope_se,
                   intercept_mm=intercept, intercept_se_mm=intercept_se,
                   r2=r2, resid_rms_mm=resid_rms)


def scale_from_differences(truth_mm, reported_mm) -> float:
    """Scale estimated from adjacent-rung differences, where the origin cancels.

    d(rho)/d(tape) does not contain the datum offset at all, so this is an
    independent read on the scale that cannot be contaminated by a wrong guess
    about where depth is measured from.
    """
    order = np.argsort(np.asarray(truth_mm, dtype=np.float64))
    x = np.asarray(truth_mm, dtype=np.float64)[order]
    y = np.asarray(reported_mm, dtype=np.float64)[order]
    dx, dy = np.diff(x), np.diff(y)
    keep = np.abs(dx) > 1e-9
    if not keep.any():
        return float("nan")
    return float(np.median(dy[keep] / dx[keep]))


# =============================================================================
# CSV
# =============================================================================
FIELDS = (
    # identity
    "timestamp", "run_id", "label", "truth_mm", "datum", "notes",
    # configuration actually used
    "mono_res", "mono_w", "mono_h", "extended", "subpixel", "lr_check",
    "median", "confidence", "depth_preset", "depth_align",
    "depth_w", "depth_h", "color_out", "fps", "min_mm", "max_mm", "roi",
    # predicted geometry
    "fx_mono_px", "fx_depth_px", "baseline_mm", "max_disparity",
    "min_z_pred_mm", "quant_dz_pred_mm", "band_pred_px", "band_pred_frac",
    "band_delivered_frac", "band_vanish_mm",
    # capture health
    "n_frames", "n_dup", "duration_s", "depth_fps",
    "exposure_ms", "exposure_spread_ms", "motion_madiff",
    "texture_mean", "texture_p10", "saturated_frac",
    # coverage
    "fill_all", "fill_roi", "fill_roi_inlier", "inlier_frac",
    "hole_left", "hole_centre", "hole_right", "hole_asym",
    "z_min_mm", "z_p01_mm", "z_p50_mm", "z_p99_mm",
    # accuracy
    "rho_mm", "rho_iqr_mm", "rho_range_mm", "bias_mm", "bias_pct",
    "z_center_mm", "tilt_deg",
    # noise and flatness
    "resid_rms_mm", "resid_mad_mm", "patch_std_mm",
    "temporal_rms_mm", "temporal_med_mm", "temporal_valid_frac",
    "fixed_rms_mm", "quant_dz_meas_mm",
    # verdict
    "roi_w_px", "roi_h_px", "n_fits", "resid_limit_mm",
    "ok", "warn", "refused", "col_fill_profile",
)


def _fmt(value) -> str:
    """NaN becomes an empty cell: a missing measurement is not the number 'nan'."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value or math.isinf(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def append_row(path: Path, row: dict) -> None:
    """Append one rung, keeping an existing file's column order if it differs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(FIELDS)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as fh:
            found = next(csv.reader(fh), None)
        if found:
            header = found
            missing = [f for f in FIELDS if f not in header]
            if missing:
                print(f"  note: {path.name} predates these columns, they will not "
                      f"be stored: {', '.join(missing)}")
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _fmt(v) for k, v in row.items()})


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float:
    try:
        text = (row.get(key) or "").strip()
        return float(text) if text else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def setting_key(row: dict) -> str:
    """Everything that changes the stereo answer, in one comparable string.

    The preset and the delivered size belong here even though --confidence is
    pinned: a preset sets several matcher knobs besides the threshold, and the
    output size decides how much resampling stands between the disparity grid and
    the number being fitted. Two rungs that differ in either are not the same
    setting, and pooling them would make the datum-offset cross-check agree for
    the wrong reason.
    """
    return (f"mono={row.get('mono_res', '?')} ext={row.get('extended', '?')} "
            f"sub={row.get('subpixel', '?')} lr={row.get('lr_check', '?')} "
            f"med={row.get('median', '?')} conf={row.get('confidence', '?')} "
            f"align={row.get('depth_align', '?')} "
            f"preset={row.get('depth_preset', '?')} "
            f"out={row.get('depth_w', '?')}x{row.get('depth_h', '?')}")


def base_key(row: dict) -> str:
    """setting_key without the extended flag — what an off/on pair shares."""
    return (f"mono={row.get('mono_res', '?')} sub={row.get('subpixel', '?')} "
            f"lr={row.get('lr_check', '?')} med={row.get('median', '?')} "
            f"conf={row.get('confidence', '?')} align={row.get('depth_align', '?')} "
            f"preset={row.get('depth_preset', '?')} "
            f"out={row.get('depth_w', '?')}x{row.get('depth_h', '?')}")


# =============================================================================
# Reporting
# =============================================================================
def print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no rungs recorded yet)")
        return
    print(f"{'label':<12} {'tape':>6} {'rho':>7} {'bias':>7} {'resid':>6} "
          f"{'temp':>6} {'fill':>6} {'holes L/C/R':>17} {'tilt':>5} {'n':>3} ok")
    print(f"{'':<12} {'mm':>6} {'mm':>7} {'mm':>7} {'mm':>6} {'mm':>6} "
          f"{'ROI%':>6} {'%':>17} {'deg':>5}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x.get("label", ""), _f(x, "truth_mm"))):
        ok = (r.get("ok") or "0").strip() == "1"
        holes = "/".join(f"{100 * _f(r, k):4.0f}" if _f(r, k) == _f(r, k) else "   -"
                         for k in ("hole_left", "hole_centre", "hole_right"))
        def cell(key, w, p=1, mul=1.0):
            v = _f(r, key) * mul
            return f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}"
        print(f"{(r.get('label') or '')[:12]:<12} {cell('truth_mm', 6, 0)} "
              f"{cell('rho_mm', 7)} {cell('bias_mm', 7)} {cell('resid_rms_mm', 6)} "
              f"{cell('temporal_rms_mm', 6)} "
              f"{cell('fill_roi', 6, 1, 100.0)} {holes:>17} "
              f"{cell('tilt_deg', 5)} {(r.get('n_frames') or '-'):>3} "
              f"{'yes' if ok else 'NO'}")
        if not ok and (r.get("refused") or "").strip():
            print(f"{'':<12}   refused: {r['refused'][:110]}")


def print_prediction(pred: Prediction, truth_mm: float, source: str) -> None:
    print(f"  geometry (fx from {source}):")
    print(f"    mono {pred.mono_res}: fx = {pred.fx_mono:.2f} px, "
          f"baseline {BASELINE_MM:.0f} mm, max_disparity {pred.max_disparity}")
    print(f"    MinZ = fx*b/max_disparity = {pred.min_z_mm:.1f} mm"
          f"{'   <-- extended disparity' if pred.extended else ''}")
    if truth_mm and truth_mm > 0:
        sub_note = ("  (subpixel subdivides this: an upper bound, not the step)"
                    if pred.subpixel else "")
        print(f"    at {truth_mm:.0f} mm: one disparity step = "
              f"{pred.quant_mm:.1f} mm of depth{sub_note}")
        print(f"    at {truth_mm:.0f} mm: stereo overlap is lost in a band "
              f"{pred.band_px:.0f} px wide = {100 * pred.band_frac:.1f}% of the "
              f"{pred.mono_width}px mono frame")
        if pred.delivered_width:
            frac = pred.delivered_band_frac
            print(f"    at {truth_mm:.0f} mm: in the delivered "
                  f"{pred.delivered_width}px frame that band is "
                  f"{pred.delivered_band_px:.0f} px = {100 * frac:.1f}%"
                  f"{'  (cropped out of view)' if frac <= 0 else ''}")
            if math.isinf(pred.delivered_vanish_mm):
                print("    the delivered frame is as wide as the stereo pair, so "
                      "the whole band stays visible — this is the control "
                      "configuration for measuring it")
            else:
                print(f"    the band leaves the delivered frame entirely beyond "
                      f"{pred.delivered_vanish_mm:.0f} mm — beyond that, per-third "
                      f"hole numbers say nothing about stereo overlap, so measure "
                      f"the band with --depth-align none")
        if truth_mm < pred.min_z_mm:
            print(f"    NOTE: {truth_mm:.0f} mm is BELOW MinZ — this rung is a "
                  f"negative control and is supposed to come back empty")


# =============================================================================
# Capture (the only part that touches the camera)
# =============================================================================
@dataclass
class Capture:
    """Raw material from one hold: the depth stack plus the health evidence."""

    depth: list = field(default_factory=list)
    exposures: list = field(default_factory=list)
    motion: list = field(default_factory=list)
    duration_s: float = 0.0
    duplicates: int = 0
    last_color = None
    last_left = None
    depth_intrinsics: Intrinsics | None = None


def _mono_stats(left: np.ndarray | None) -> dict:
    """Texture and saturation of the left mono frame's centre.

    Holes have two very different causes that look identical in a fill
    percentage: the geometry could not match, or the wall had nothing to match.
    Local standard deviation separates them, which is why the left stream is on
    by default here even though depth alone would be cheaper.
    """
    if left is None or left.ndim != 2 or left.size == 0:
        return {"texture_mean": float("nan"), "texture_p10": float("nan"),
                "saturated_frac": float("nan")}
    h, w = left.shape
    r0, r1, c0, c1 = roi_slice(h, w, (0.4, 0.4))
    roi = left[r0:r1, c0:c1].astype(np.float32)
    mean = cv2.blur(roi, (5, 5))
    sq = cv2.blur(roi * roi, (5, 5))
    local_std = np.sqrt(np.maximum(0.0, sq - mean * mean))
    return {
        "texture_mean": float(local_std.mean()),
        "texture_p10": float(np.percentile(local_std, 10)),
        "saturated_frac": float((roi >= 250).mean()),
    }


def collect(src, a) -> Capture:
    """Hold still and keep only genuinely new depth frames.

    The duplicate check is the important line here. `read()` returns the newest
    frame of every stream whenever *any* stream is fresh, so at 5 fps depth and a
    fast host loop the same depth image comes back several times. Stacking those
    would report a temporal noise of zero and a rock-steady camera.
    """
    cap = Capture()
    cap.depth_intrinsics = src.intrinsics.get(STREAM_DEPTH)

    if a.settle > 0:
        print(f"  hold still — settling for {a.settle:.1f} s ...")
        t_end = time.monotonic() + a.settle
        while time.monotonic() < t_end:
            src.read(timeout_ms=500)

    last_depth_seq = None
    last_left = None
    warmed = 0
    t0 = time.monotonic()
    deadline = t0 + a.timeout

    while len(cap.depth) < a.frames and time.monotonic() < deadline:
        bundle = src.read(timeout_ms=2000)
        if bundle is None:
            continue
        depth = bundle.depth
        if depth is None or depth.image is None:
            continue
        if depth.seq == last_depth_seq:
            cap.duplicates += 1
            continue
        last_depth_seq = depth.seq

        if warmed < a.warmup_frames:
            warmed += 1
            continue

        cap.depth.append(depth.image.copy())

        left = bundle.left
        if left is not None and left.image is not None:
            cap.last_left = left.image
            if left.exposure_ms == left.exposure_ms:
                cap.exposures.append(left.exposure_ms)
            if last_left is not None and last_left.shape == left.image.shape:
                diff = cv2.absdiff(left.image[::4, ::4], last_left[::4, ::4])
                cap.motion.append(float(np.median(diff)))
            last_left = left.image.copy()
        elif bundle.color is not None and bundle.color.image is not None:
            if bundle.color.exposure_ms == bundle.color.exposure_ms:
                cap.exposures.append(bundle.color.exposure_ms)
        if bundle.color is not None and bundle.color.image is not None:
            cap.last_color = bundle.color.image

        if not a.quiet and len(cap.depth) % 10 == 0:
            print(f"    {len(cap.depth)}/{a.frames} frames")

        if a.display:
            _show(bundle, a, len(cap.depth))

    cap.duration_s = time.monotonic() - t0
    if a.display:
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
    return cap


def _show(bundle, a, n: int) -> None:
    """Aiming view: colour, depth, and the ROI the numbers come from."""
    panes = []
    if bundle.color is not None and bundle.color.image is not None:
        panes.append(viz.label(bundle.color.image.copy(), "colour"))
    if bundle.depth is not None and bundle.depth.image is not None:
        d = bundle.depth.image
        # Fixed millimetre range, never per-frame auto-scale: the whole point is
        # to see the same distance as the same colour between rungs.
        dc = viz.colorize_depth(d, a.min_mm, min(a.max_mm, a.truth_mm * 2 + 200))
        r0, r1, c0, c1 = roi_slice(d.shape[0], d.shape[1], a.roi)
        cv2.rectangle(dc, (c0, r0), (c1 - 1, r1 - 1), (255, 255, 255), 1)
        panes.append(viz.label(dc, "depth + ROI"))
    if not panes:
        return
    canvas = viz.hstack_pad(panes)
    viz.draw_hud(canvas, [f"tape {a.truth_mm:.0f} mm   frame {n}/{a.frames}",
                          "hold still"])
    try:
        cv2.imshow("c3 depth accuracy", canvas)
        cv2.waitKey(1)
    except Exception:  # noqa: BLE001 - headless is fine, just stop trying
        a.display = False


def build_config(a) -> StreamConfig:
    cfg = C.config_from_args(a)
    # Colour is only for aiming, so it stays small and encoded; left mono earns
    # its bandwidth by supplying exposure, texture and motion evidence.
    cfg.color_encode = "mjpeg"
    return cfg


def measure_rung(a) -> dict:
    """One rung end to end: connect, hold, compute, gate, return the CSV row."""
    from c3_camera.source import C3Source

    cfg = build_config(a)
    rc = cfg.resolve()
    fx_mono_1280, fx_color, _fx_source = load_focal_lengths(a.calibration)
    fx_del, w_del = delivered_optics(cfg, rc, fx_mono_1280, fx_color)
    pred = predict(cfg.mono_res, cfg.extended, a.truth_mm, fx_mono_1280,
                   fx_delivered=fx_del, width_delivered=w_del,
                   subpixel=cfg.subpixel)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {f: "" for f in FIELDS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "label": a.label,
        "truth_mm": a.truth_mm,
        "datum": a.datum,
        "notes": a.notes,
        "mono_res": cfg.mono_res,
        "mono_w": rc.mono_size[0], "mono_h": rc.mono_size[1],
        "extended": int(cfg.extended), "subpixel": int(cfg.subpixel),
        "lr_check": int(cfg.lr_check), "median": cfg.median,
        "confidence": "" if cfg.confidence is None else cfg.confidence,
        "depth_preset": cfg.depth_preset, "depth_align": cfg.depth_align,
        "depth_w": rc.depth_out_size[0], "depth_h": rc.depth_out_size[1],
        "color_out": f"{rc.color_out_size[0]}x{rc.color_out_size[1]}",
        "fps": cfg.fps, "min_mm": a.min_mm, "max_mm": a.max_mm,
        "roi": f"{a.roi[0]}x{a.roi[1]}",
        "fx_mono_px": round(pred.fx_mono, 2), "baseline_mm": BASELINE_MM,
        "max_disparity": pred.max_disparity,
        "min_z_pred_mm": round(pred.min_z_mm, 1),
        "quant_dz_pred_mm": round(pred.quant_mm, 2),
        "band_pred_px": round(pred.band_px, 1),
        "band_pred_frac": round(pred.band_frac, 4),
        "band_delivered_frac": round(pred.delivered_band_frac, 4),
        "band_vanish_mm": round(pred.delivered_vanish_mm, 1),
        "ok": 0,
    })

    # main() has already printed this block; repeating it here would put the same
    # twelve lines on screen twice for every rung.
    print()

    with C3Source(cfg, verbose=not a.quiet, retries=a.retries) as src:
        intr = src.intrinsics.get(STREAM_DEPTH)
        if intr is None:
            row["refused"] = ("no depth intrinsics from the device; deprojection "
                              "is impossible and no distance can be reported")
            print(f"  REFUSED: {row['refused']}")
            return row
        row["fx_depth_px"] = round(intr.fx, 2)
        cap = collect(src, a)

    if not cap.depth:
        row["refused"] = "no depth frames arrived"
        row["n_frames"] = 0
        row["duration_s"] = round(cap.duration_s, 1)
        print(f"  REFUSED: {row['refused']}")
        return row

    stack = np.stack(cap.depth)
    if a.save_npy:
        out = a.save_npy if a.save_npy.suffix == ".npy" else \
            a.save_npy / f"{run_id}_{a.label or 'rung'}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, stack)
        print(f"  raw depth stack -> {out}")

    m = measure_stack(stack, cap.depth_intrinsics, roi_frac=a.roi,
                      min_mm=a.min_mm, max_mm=a.max_mm,
                      quant_hint_mm=pred.quant_mm)

    exposure = float(np.mean(cap.exposures)) if cap.exposures else float("nan")
    spread = (float(max(cap.exposures) - min(cap.exposures))
              if len(cap.exposures) > 1 else float("nan"))
    motion = float(np.median(cap.motion)) if cap.motion else float("nan")

    gates = Gates(min_frames=a.min_frames, min_fill_roi=a.min_fill,
                  min_inlier=a.min_inlier, max_resid_mm=a.max_resid_mm,
                  max_tilt_deg=a.max_tilt, max_exposure_ms=a.max_exposure_ms,
                  max_motion=a.max_motion)
    ok, warns, fails = apply_gates(m, a.truth_mm, pred, gates,
                                   exposure, spread, motion)

    row.update({k: v for k, v in m.items() if k in FIELDS})
    row["col_fill_profile"] = ";".join(f"{v:.3f}" for v in m["col_fill_profile"])
    row.update(_mono_stats(cap.last_left))
    row.update({
        "n_dup": cap.duplicates,
        "duration_s": round(cap.duration_s, 1),
        "depth_fps": round(len(cap.depth) / cap.duration_s, 2) if cap.duration_s else "",
        "exposure_ms": exposure, "exposure_spread_ms": spread,
        "motion_madiff": motion,
        "bias_mm": m["rho_mm"] - a.truth_mm if m["rho_mm"] == m["rho_mm"] else float("nan"),
        "bias_pct": (100.0 * (m["rho_mm"] - a.truth_mm) / a.truth_mm
                     if m["rho_mm"] == m["rho_mm"] and a.truth_mm else float("nan")),
        "resid_limit_mm": round(resid_limit_mm(a.truth_mm, pred.quant_mm,
                                               a.max_resid_mm), 2),
        "ok": ok,
        "warn": " | ".join(warns),
        "refused": " | ".join(fails),
    })

    print()
    print(f"  frames {len(cap.depth)} unique ({cap.duplicates} duplicates skipped) "
          f"in {cap.duration_s:.1f} s")
    print(f"  fill   ROI {100 * m['fill_roi']:.1f}%  frame {100 * m['fill_all']:.1f}%"
          f"   holes L/C/R {100 * m['hole_left']:.0f}/"
          f"{100 * m['hole_centre']:.0f}/{100 * m['hole_right']:.0f}%")
    if ok:
        print(f"  rho    {m['rho_mm']:.1f} mm against a tape of {a.truth_mm:.0f} mm "
              f"-> bias {row['bias_mm']:+.1f} mm ({row['bias_pct']:+.2f}%)")
        print(f"  noise  plane residual {m['resid_rms_mm']:.2f} mm "
              f"(temporal {m['temporal_rms_mm']:.2f}, fixed {m['fixed_rms_mm']:.2f}) "
              f" vs naive ROI std {m['patch_std_mm']:.2f}")
        print(f"  tilt   {m['tilt_deg']:.1f} deg, inliers {100 * m['inlier_frac']:.0f}%")
    else:
        print("  REFUSED — no bias or noise figure will be quoted for this rung:")
        for f in fails:
            print(f"    - {f}")
    for w in warns:
        print(f"  warn: {w}")
    print(f"  the datum offset is NOT applied here; run --analyze to fit it "
          f"(bias above is measured from '{a.datum or 'unrecorded datum'}')")
    return row


# =============================================================================
# --analyze
# =============================================================================
def analyze(a) -> int:
    rows = read_rows(a.csv)
    if not rows:
        print(f"no rungs in {a.csv} — capture some first")
        return 1

    usable = [r for r in rows if (r.get("ok") or "0").strip() == "1"]
    print("=" * 100)
    print(f"depth accuracy: {len(rows)} rungs in {a.csv}, {len(usable)} usable")
    print("=" * 100)
    print_table(rows)

    refused = [r for r in rows if (r.get("ok") or "0").strip() != "1"]
    if refused:
        print(f"\n{len(refused)} rung(s) refused and excluded from every fit below:")
        for r in refused:
            print(f"  {(r.get('label') or '?'):<12} {_f(r, 'truth_mm'):>6.0f} mm  "
                  f"{(r.get('refused') or 'no reason recorded')[:80]}")

    # ---- reported vs truth, per setting ----------------------------------
    print()
    print("=" * 100)
    print(f"reported rho = a * tape + c   (rungs >= {a.fit_min_mm:.0f} mm only; "
          f"nearer rungs carry the curvature and the below-MinZ failures)")
    print("=" * 100)

    groups: dict[str, list[dict]] = {}
    for r in usable:
        groups.setdefault(setting_key(r), []).append(r)

    print(f"{'setting':<86} {'n':>2} {'a':>7} {'+-':>6} {'c mm':>8} {'+-':>6} "
          f"{'R2':>6} {'a_diff':>7}")
    print("-" * 136)
    fits: dict[str, LineFit] = {}
    for key in sorted(groups):
        sel = [r for r in groups[key] if _f(r, "truth_mm") >= a.fit_min_mm]
        t = [_f(r, "truth_mm") for r in sel]
        y = [_f(r, "rho_mm") for r in sel]
        fit = fit_reported_vs_truth(t, y)
        fits[key] = fit
        if not fit.ok:
            print(f"{key:<86} {len(sel):>2}  {fit.reason}")
            continue
        adiff = scale_from_differences(t, y)
        def s(v, w, p=3):
            return f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}"
        print(f"{key:<86} {fit.n:>2} {s(fit.slope, 7, 4)} {s(fit.slope_se, 6, 4)} "
              f"{s(fit.intercept_mm, 8, 2)} {s(fit.intercept_se_mm, 6, 2)} "
              f"{s(fit.r2, 6, 4)} {s(adiff, 7, 4)}")

    good = [f for f in fits.values() if f.ok and f.intercept_mm == f.intercept_mm]
    if len(good) >= 2:
        cs = np.array([f.intercept_mm for f in good])
        pooled = float(np.median(cs))
        spread = float(np.max(np.abs(cs - pooled)))
        print()
        print(f"  datum offset c: pooled {pooled:+.1f} mm, worst setting disagrees "
              f"by {spread:.1f} mm")
        print(f"  truth from the optical centre = tape + ({pooled:+.1f}) mm")
        if spread > a.c_tolerance_mm:
            print(f"  WARNING: the optical centre cannot depend on the stereo "
                  f"settings, so a {spread:.1f} mm spread (> {a.c_tolerance_mm:.0f}) "
                  f"means c is absorbing something else — most likely near rungs "
                  f"still in the fit, or the rig moving between settings")
        else:
            print(f"  the settings agree on c to within {a.c_tolerance_mm:.0f} mm, "
                  f"which is what makes it credible as a real origin offset")
    elif good:
        print(f"\n  only one setting has a fit: c = {good[0].intercept_mm:+.1f} mm, "
              f"with nothing to cross-check it against")
    else:
        print("\n  not enough usable rungs to fit the datum offset — it stays "
              "unknown, and every bias above is therefore relative to the tape "
              "datum, not to the optical centre")

    # ---- extended off vs on ----------------------------------------------
    print()
    print("=" * 100)
    print("extended disparity, off vs on, at matched distances")
    print("=" * 100)
    pairs: dict[tuple[str, float], dict[str, dict]] = {}
    for r in rows:
        ext = (r.get("extended") or "").strip()
        if ext not in ("0", "1"):
            continue
        pairs.setdefault((base_key(r), _f(r, "truth_mm")), {})[ext] = r

    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if not both:
        print("  no distance has been measured with extended both off and on yet")
    else:
        print(f"{'setting':<80} {'tape':>8} {'fill ROI %':>13} {'bias mm':>15} "
              f"{'resid mm':>15}")
        print(f"{'':<80} {'mm':>8} {'off':>6}{'on':>7} {'off':>7}{'on':>8} "
              f"{'off':>7}{'on':>8}")
        print("-" * 134)
        any_refused = False
        for (bkey, truth) in sorted(both, key=lambda k: (k[0], k[1])):
            off, on = both[(bkey, truth)]["0"], both[(bkey, truth)]["1"]
            ok_off = (off.get("ok") or "0").strip() == "1"
            ok_on = (on.get("ok") or "0").strip() == "1"
            any_refused = any_refused or not (ok_off and ok_on)

            def two(key, w, p=1):
                # A refused rung's accuracy and flatness stay unquoted here as
                # well. Its FILL is still evidence — a below-MinZ rung coming back
                # empty is exactly what this table is looking for — but its bias
                # and residual were refused for a reason, and printing them beside
                # a good rung invites the comparison the refusal ruled out.
                def one(row, ok, width):
                    v = _f(row, key) if ok else float("nan")
                    return f"{v:>{width}.{p}f}" if v == v else f"{'-':>{width}}"
                return one(off, ok_off, w) + one(on, ok_on, w + 1)
            mark = "" if (ok_off and ok_on) else " *"
            print(f"{bkey[:80]:<80} {truth:>6.0f}{mark:<2} "
                  f"{100 * _f(off, 'fill_roi'):>6.1f}{100 * _f(on, 'fill_roi'):>7.1f} "
                  f"{two('bias_mm', 7)} {two('resid_rms_mm', 7, 2)}")
        print()
        if any_refused:
            print("  * one side of this pair was refused: its fill is still shown "
                  "(that is the evidence), its bias and residual are not.")
        print("  read the fill columns against MinZ, not on their own: extended "
              "disparity halves MinZ (300 -> 150 mm at 400p, measured) but cannot touch "
              "the edge band, which is a field-of-view overlap loss. If fill rises "
              "while bias or resid worsens, it filled holes with wrong values.")

    # ---- where each setting actually starts working -----------------------
    print()
    print("=" * 100)
    print("smallest distance that produced a usable measurement, per setting")
    print("=" * 100)
    print(f"{'setting':<86} {'MinZ pred':>10} {'first ok':>9} {'z_min obs':>10}")
    print("-" * 119)
    # Every row, not just the usable ones: a rung refused *because* it is below
    # MinZ is precisely the evidence this table is looking for.
    all_groups: dict[str, list[dict]] = {}
    for r in rows:
        all_groups.setdefault(setting_key(r), []).append(r)
    for key in sorted(all_groups):
        sel = sorted(all_groups[key], key=lambda r: _f(r, "truth_mm"))
        first = next((_f(r, "truth_mm") for r in sel
                      if _f(r, "fill_roi") >= a.usable_fill), float("nan"))
        zmins = [_f(r, "z_min_mm") for r in sel if _f(r, "z_min_mm") == _f(r, "z_min_mm")]
        pred_minz = _f(sel[0], "min_z_pred_mm")
        print(f"{key:<86} {pred_minz:>10.1f} "
              f"{(f'{first:.0f}' if first == first else '-'):>9} "
              f"{(f'{min(zmins):.0f}' if zmins else '-'):>10}")
    print()
    print("  'first ok' well above MinZ means the limit is texture or confidence, "
          "not geometry. Valid depth *below* MinZ means something is inventing it.")
    print("=" * 100)
    return 0 if usable else 1


# =============================================================================
# CLI
# =============================================================================
def _parse_roi(text: str) -> tuple[float, float]:
    sep = "x" if "x" in text else ","
    if sep not in text:
        raise argparse.ArgumentTypeError(f"--roi wants WxH fractions, got {text!r}")
    w, h = (float(v) for v in text.split(sep, 1))
    if not (0.02 < w <= 1.0 and 0.02 < h <= 1.0):
        raise argparse.ArgumentTypeError("--roi fractions must be in (0.02, 1.0]")
    return w, h


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_connection_args(p)
    C.add_stream_args(p)

    g = p.add_argument_group("this rung")
    g.add_argument("--truth-mm", type=float, default=None,
                   help="tape-measured distance to the target, in millimetres. "
                        "Required unless --analyze.")
    g.add_argument("--label", default="",
                   help="name for this arm, e.g. 'default' / 'ext' / 'med5'. "
                        "Rows group by the recorded settings, not by this, so a "
                        "mislabelled row still lands in the right comparison.")
    g.add_argument("--datum", default="",
                   help="which mechanical face the tape was pulled from. The "
                        "offset between it and the optical centre is unknown and "
                        "gets FITTED by --analyze, so recording it matters.")
    g.add_argument("--notes", default="", help="free text stored with the row")

    g = p.add_argument_group("hold")
    g.add_argument("--frames", type=int, default=40,
                   help="unique depth frames to keep (default: %(default)s)")
    g.add_argument("--warmup-frames", type=int, default=10,
                   help="frames discarded first, while exposure and the matcher "
                        "settle (default: %(default)s)")
    g.add_argument("--settle", type=float, default=2.0,
                   help="seconds to stand still before capture starts "
                        "(default: %(default)s)")
    g.add_argument("--timeout", type=float, default=40.0,
                   help="give up on the hold after this long (default: %(default)s)")
    g.add_argument("--retries", type=int, default=3,
                   help="connection attempts (default: %(default)s)")

    g = p.add_argument_group("analysis region")
    g.add_argument("--roi", type=_parse_roi, default=(0.4, 0.4), metavar="WxH",
                   help="centre fraction used for bias/noise/flatness "
                        "(default: 0.4x0.4). Fill and hole statistics always use "
                        "the whole frame, because the near-field loss lives at "
                        "the edges and averaging it into the centre hides it.")
    g.add_argument("--min-mm", type=float, default=100.0,
                   help="smallest depth treated as a measurement (default: "
                        "%(default)s). NOT 300: that is c3_collect.py's display "
                        "floor and would erase exactly the range extended "
                        "disparity unlocks.")
    g.add_argument("--max-mm", type=float, default=20000.0,
                   help="largest depth treated as a measurement (default: %(default)s)")
    g.add_argument("--calibration", type=Path, default=None,
                   help="calibration.json to read CAM_B fx from, for the MinZ and "
                        "quantisation predictions (default: this camera's values)")

    g = p.add_argument_group("refusal thresholds")
    g.add_argument("--min-frames", type=int, default=10,
                   help="below this many unique frames, refuse (default: %(default)s)")
    g.add_argument("--min-fill", type=float, default=0.30,
                   help="minimum valid fraction of the ROI (default: %(default)s)")
    g.add_argument("--min-inlier", type=float, default=0.60,
                   help="minimum fraction of ROI depths lying on the fitted plane "
                        "(default: %(default)s)")
    g.add_argument("--max-resid-mm", type=float, default=None,
                   help="plane residual above which the fit is refused. Default is "
                        "derived per rung: max(8 mm, ONE disparity step, 2%% of the "
                        "distance), because both terms grow with range.")
    g.add_argument("--max-tilt", type=float, default=15.0,
                   help="tilt in degrees above which the aim is warned about "
                        "(default: %(default)s)")
    g.add_argument("--max-exposure-ms", type=float, default=20.0,
                   help="exposure above which motion blur is called out "
                        "(default: %(default)s)")
    g.add_argument("--max-motion", type=float, default=6.0,
                   help="frame-to-frame median abs difference in grey levels above "
                        "which the hold is refused as moving (default: %(default)s)")

    g = p.add_argument_group("output")
    g.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="accumulating one-row-per-rung CSV, appended to "
                        "(default: %(default)s)")
    g.add_argument("--save-npy", type=Path, default=None,
                   help="also write the raw (N,H,W) depth stack here, so the "
                        "analysis can be redone without recapturing")
    g.add_argument("--no-display", dest="display", action="store_false", default=True,
                   help="skip the aiming window (headless capture)")
    g.add_argument("--quiet", action="store_true", help="less chatter")

    g = p.add_argument_group("modes")
    g.add_argument("--analyze", action="store_true",
                   help="do not touch the camera: read the CSV and report the "
                        "fitted rho-vs-tape relation and the extended off/on "
                        "comparison")
    g.add_argument("--dry-run", action="store_true",
                   help="print the predicted geometry and the resolved pipeline "
                        "for this setting, then exit without connecting")
    g.add_argument("--fit-min-mm", type=float, default=450.0,
                   help="--analyze: smallest rung admitted to the line fit "
                        "(default: %(default)s)")
    g.add_argument("--usable-fill", type=float, default=0.5,
                   help="--analyze: ROI fill counted as 'this setting works here' "
                        "(default: %(default)s)")
    g.add_argument("--c-tolerance-mm", type=float, default=5.0,
                   help="--analyze: how far the fitted datum offset may disagree "
                        "between settings before it is called suspect "
                        "(default: %(default)s)")

    # Measurement discipline, overriding the package's live-view defaults:
    #   fps 5      - depth frames only need to be independent, not fast
    #   median off - the median filter fills holes and flatters the residual,
    #                which is the opposite of what a measurement wants
    #   confidence - pinned rather than inherited, because the robotics preset
    #                leaves it at 245/255 and an unrecorded threshold makes
    #                'extended cost us fill' unfalsifiable
    #   left       - carries the exposure, texture and motion evidence that the
    #                refusal gates are built on
    p.set_defaults(fps=5.0, median="off", confidence=245,
                   streams=(STREAM_COLOR, STREAM_DEPTH, STREAM_LEFT))
    PF.add_preflight_args(p)
    a = p.parse_args(argv)

    if not a.analyze and a.truth_mm is None:
        p.error("--truth-mm is required for a measurement (or use --analyze)")
    if a.truth_mm is not None and a.truth_mm <= 0:
        p.error("--truth-mm must be positive")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)

    if a.analyze:
        return analyze(a)

    cfg = build_config(a)
    rc = cfg.resolve()
    fx_mono_1280, fx_color, fx_source = load_focal_lengths(a.calibration)
    fx_del, w_del = delivered_optics(cfg, rc, fx_mono_1280, fx_color)
    pred = predict(cfg.mono_res, cfg.extended, a.truth_mm, fx_mono_1280,
                   fx_delivered=fx_del, width_delivered=w_del,
                   subpixel=cfg.subpixel)

    print("=" * 100)
    print(f"C3 depth accuracy — rung at {a.truth_mm:.0f} mm"
          f"{' [' + a.label + ']' if a.label else ''}")
    print("=" * 100)
    for w in rc.warnings:
        print(f"  warning: {w}")
    print(f"  pipeline: {rc.describe()}")
    print(f"  budget  : {rc.budget_note()}")
    if rc.bandwidth_mbps()["total"] > C.POE_BUDGET_MBPS:
        # Over the link ceiling the device drops frames, which costs unique
        # samples and can bias the temporal noise. Say what to do about it.
        print("  NOTE    : over the PoE ceiling — the device will drop frames. "
              "Drop 'left' from --streams (costs the exposure/texture gates), "
              "lower --fps, or set --depth-size to shrink the depth map.")
    print_prediction(pred, a.truth_mm, fx_source)

    # Deliberately NOT inside the --dry-run branch. A real capture needs the same
    # geometry, and above all the same band warning, *before* the hold rather than
    # only in a rehearsal — this is the version that arrives in time to re-aim.
    # apply_gates() repeats the band check afterwards so it also lands in the row.
    h, w = rc.depth_out_size[1], rc.depth_out_size[0]
    r0, r1, c0, c1 = roi_slice(h, w, a.roi)
    margin = (w - (c1 - c0)) / 2
    print(f"  ROI     : {c1 - c0}x{r1 - r0} px of {w}x{h}, "
          f"{margin:.0f} px clear of each edge")
    # Compare like with like: the ROI margin is in delivered pixels, so the band
    # has to be the delivered one too. The mono-frame band is bigger and comparing
    # it here would raise a false alarm on every --depth-align color run, where
    # most of it is cropped away before delivery.
    band_del = pred.delivered_band_px
    if band_del == band_del and band_del > margin:
        print(f"  WARNING : the predicted overlap band ({band_del:.0f} px in the "
              f"delivered frame) is wider than the ROI margin ({margin:.0f} px) "
              f"at this distance, so the bias would be measured partly on "
              f"missing data — shrink --roi")
    print(f"  refuse  : fill<{a.min_fill:.2f}  inliers<{a.min_inlier:.2f}  "
          f"resid>{resid_limit_mm(a.truth_mm, pred.quant_mm, a.max_resid_mm):.1f} mm  "
          f"motion>{a.max_motion:.1f}  exposure>{a.max_exposure_ms:.0f} ms")

    if a.dry_run:
        print("\n  --dry-run: no device was opened.")
        return 0

    # A rung is a physical setup — tape pulled, target aimed, someone holding
    # still. Finding out afterwards that the camera was never free wastes the
    # setup, not just the run.
    print()
    rc_pf = PF.gate(a, title="c3 depth-accuracy", vehicle=False,
                    out_dir=a.csv.parent, display=a.display)
    if rc_pf is not None:
        return rc_pf
    print()

    row = measure_rung(a)
    append_row(a.csv, row)
    print(f"\n  appended to {a.csv}")
    print()
    print_table(read_rows(a.csv))
    print(f"\n  run with --analyze once several rungs exist to fit the datum "
          f"offset and compare settings")
    return 0 if str(row.get("ok")) == "1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
