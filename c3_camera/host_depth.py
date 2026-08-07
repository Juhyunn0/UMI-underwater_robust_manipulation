#!/usr/bin/env python3
"""
host_depth.py — stereo matching on the topside computer instead of on the camera.

Why this exists
---------------
The C3 (OAK-D-W-POE) reaches the desk over one PoE link with a measured ceiling
near 91.5 Mbit/s. Measured on this project's own dataset
(``datasets/dataset_20260803_105221/metadata.json``):

    colour   6.7 Mbit/s   (14% of the traffic) — already MJPEG-encoded on device
    depth   41.5 Mbit/s   (86% of the traffic) — uint16, *cannot* be encoded
    total   48.2 Mbit/s

``depth`` is uncompressible because the device's VideoEncoder only takes 8-bit
formats. Mono *is* 8-bit. So shipping the raw stereo pair and matching up here
can cost less wire than shipping depth16 down — that is the entire thesis, and
this module is the host half of it.

What this module is NOT
-----------------------
It is not a replacement for the device depth path. That path is untouched and
must stay untouched: this file imports nothing from ``source.py`` /
``pipeline.py`` / ``config.py`` / ``c3_collect.py`` and is imported by none of
them. Turning host depth on is somebody else's one-line gate; with the gate off
this file is never even imported.

The output contract (copied from the writer, not re-invented)
-------------------------------------------------------------
``dataset.py:400-404`` refuses anything that is not ``uint16``::

    elif kind == "depth":
        if data.dtype != np.uint16:
            raise ValueError(f"depth must be uint16 millimetres, got {data.dtype}")

and ``dataset.py:58`` fixes ``DEPTH_SCALE = 1000.0``, i.e. ``metres = value/1000``.
So every public function here returns **uint16 millimetres with 0 meaning "no
measurement"** — byte-for-byte the same thing the device path hands the writer,
so ``DatasetWriter`` needs no change at all.

The refusals (each one is a bug this module would otherwise hide)
-----------------------------------------------------------------
1. **K must come from the rectified focal length.** ``K = fx_rect * baseline_mm``
   turns disparity into distance. On this camera the raw EEPROM values are
   CAM_B fx 757.12 and CAM_C fx 760.28 at 1280x800 (halving to 378.56 / 380.14
   at 640x400) while the rectified value is neither. Picking the wrong one is a
   0.4% systematic scale error — far too small to look broken and exactly large
   enough to poison a SLAM map. So there are exactly two honest sources of
   ``fx_rect`` here:

     * ``cv2.stereoRectify`` run by this module, where ``P1[0,0]`` *is* the
       focal length of the images being matched, by definition; or
     * a number the caller measured and passed in explicitly
       (``StereoRig.pre_rectified``).

   There is no third path, and in particular no fallback to ``K_left[0][0]``.
   Missing extrinsics raise ``CalibrationIncomplete``; they never degrade to a
   guess. (DepthAI 2.32 has no API that returns a rectified focal length —
   ``CalibrationHandler`` exposes only the 3x3 rectification *rotations* and
   ``CameraInfo.intrinsicMatrix`` is unrectified — which is why device-rectified
   input needs an explicitly measured number rather than a lookup.)

2. **Raw vs pre-rectified input is declared, never sniffed.** The two are the
   same shape, the same dtype and the same size; nothing in the pixels says
   which one you have. ``HostStereoMatcher`` takes ``mono_input`` as a required
   positional argument for that reason.

3. **d == 0 maps to 0, never to inf.** See ``depth_from_disparity`` — the most
   carefully reviewed line in this file, and the reason the division is written
   the way it is.

4. **Every depth resample is NEAREST.** Interpolating a depth map invents
   distances that were never measured: a bilinear tap that straddles a
   foreground/background edge returns the average of two surfaces, i.e. a
   distance where there is only empty space, and a tap that straddles the 0
   invalid sentinel returns a fraction of a real reading, i.e. a confident
   near-field ghost. This artifact is already visible in the *device* output —
   ``pipeline.py:239`` sets ``subpixel=False`` so the device computes integer
   disparity (at most 93 distinct values), yet a recorded 480x270 device frame
   carries 1800 distinct millimetre values, produced by decimation + spatial
   filtering + the output resize. Do not add a second copy of that artifact up
   here. Intensity resampling (rectification remap, downscale) is a different
   thing and stays bilinear/area — interpolating brightness is what
   rectification is *for*.

Running it
----------
    ~/.venvs/c3-depthai/bin/python c3_camera/host_depth.py --benchmark

No camera is opened, or can be: DepthAI is imported lazily and only inside
``StereoRig.from_calibration_handler``, which takes an already-constructed
``CalibrationHandler`` and never touches ``dai.Device``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# The output contract, taken from the writer that enforces it.
#
# Imported rather than retyped so the two can never drift apart. The fallback
# exists only so this module can be used for offline reprocessing on a machine
# where the depthai package (pulled in by c3_camera/__init__.py) is absent; the
# literal is the same 1000.0 that dataset.py:58 defines, and
# test_host_depth.test_depth_scale_matches_the_dataset_writer asserts the two
# agree whenever both are importable. A *silent* divergence is not possible.
# ---------------------------------------------------------------------------
try:
    from c3_camera.dataset import DEPTH_SCALE

    DEPTH_SCALE_SOURCE = "c3_camera.dataset.DEPTH_SCALE"
except Exception:  # noqa: BLE001 - depthai missing on an offline reprocessing box
    DEPTH_SCALE = 1000.0
    DEPTH_SCALE_SOURCE = "local literal (c3_camera.dataset unavailable)"

#: uint16 millimetres, so no depth beyond this is representable at all.
MAX_UINT16_MM = 65535

#: The invalid code. Not a distance. dataset.py's README_DATASET text spells it
#: out: "0 means NO MEASUREMENT (not zero distance) — mask it out".
INVALID_MM = 0

#: SGBM emits fixed-point disparity with 4 fractional bits (value = px * 16).
SGBM_FRACTIONAL_BITS = 4

MONO_INPUT_RAW = "raw"
MONO_INPUT_RECTIFIED = "rectified"
MONO_INPUTS = (MONO_INPUT_RAW, MONO_INPUT_RECTIFIED)


# =============================================================================
# Refusals
# =============================================================================
class HostDepthError(Exception):
    """Base for every refusal in this module."""


class CalibrationIncomplete(HostDepthError):
    """The calibration cannot produce a rectified focal length.

    Raised instead of falling back to the raw EEPROM fx. See refusal 1 in the
    module docstring: the difference is 0.4%, which is a silent scale bias.
    """


class UnsupportedCalibration(HostDepthError):
    """The calibration is well-formed but this module cannot honour it."""


# =============================================================================
# disparity -> millimetres
# =============================================================================
def depth_from_disparity(disparity_px: np.ndarray,
                         k_mm_px: float,
                         z_range_mm: tuple[float, float] | None = None
                         ) -> np.ndarray:
    """``Z = K / d`` in uint16 millimetres, 0 where there is no measurement.

    ``k_mm_px`` is ``fx_rect * baseline_mm`` (units mm*px), ``disparity_px`` is
    in pixels — *not* SGBM's fixed-point int16; divide that by 16 first, or use
    :func:`disparity_mm_lut` which does it for you.

    Pixels outside ``z_range_mm`` come back as 0. They are dropped rather than
    clamped, because clamping would report the range limit as if it had been
    measured there.
    """
    d = np.asarray(disparity_px, dtype=np.float32)
    z_mm = np.zeros(d.shape, dtype=np.float32)

    # ================== THE GUARD — read this before editing ==================
    # Z = K / d. Where d == 0 the two rays are parallel and never meet: there is
    # no distance to report. The right answer is the dataset's "no measurement"
    # code, which is 0 (dataset.py:58 and README_DATASET.md).
    #
    # Dividing first and repairing afterwards does NOT work, and the failure is
    # not theoretical. Measured in this venv (numpy 1.26.4, float32, K = 28500):
    #
    #     d px    = [ 0.0,  -0.0,   -1.0,   nan,   inf ]
    #     K / d   = [ inf,  -inf, -28500.0,  nan,   0.0 ]
    #     .astype(uint16) = [  0,     0,   37036,     0,     0 ]
    #                                      ^^^^^
    # A pixel the matcher explicitly flagged as UNMATCHED comes back as 37036 mm
    # — a confident 37 metre reading, inside range, indistinguishable from a real
    # one downstream. -1.0 px is not a hypothetical value: SGBM writes
    # (min_disparity - 1) * 16 into every pixel it could not match, i.e. the
    # fixed-point value -16, i.e. -1.0 px, for min_disparity = 0. The
    # +inf/-inf/nan casts happen to give 0 on this build, but float->integer
    # conversion of a non-finite value is undefined in C: numpy warns
    # ("invalid value encountered in cast") precisely because the result is the
    # compiler's business, not ours.
    #
    # So the division is never *performed* on those pixels. `where=` skips them
    # entirely and `out` keeps the zero it was initialised with. No inf, no NaN,
    # no dependence on how a cast rounds them, and no clean-up pass that a later
    # edit could reorder, vectorise away, or "optimise" into a np.where().
    #
    # d > 0 is also false for NaN, which is the behaviour we want: a NaN
    # disparity is not a measurement.
    np.divide(np.float32(k_mm_px), d, out=z_mm, where=(d > np.float32(0.0)))
    # ================================ END GUARD ===============================

    lo, hi = (0.0, float(MAX_UINT16_MM)) if z_range_mm is None else \
        (float(z_range_mm[0]), float(z_range_mm[1]))
    # uint16 millimetres cannot hold more than 65535 whatever the caller asks.
    hi = min(hi, float(MAX_UINT16_MM))

    out = np.zeros(d.shape, dtype=np.uint16)
    keep = (z_mm > 0.0) & (z_mm >= lo) & (z_mm <= hi)
    if keep.any():
        out[keep] = np.rint(z_mm[keep]).astype(np.uint16)
    return out


def disparity_mm_lut(k_mm_px: float,
                     z_range_mm: tuple[float, float] | None = None,
                     fractional_bits: int = SGBM_FRACTIONAL_BITS) -> np.ndarray:
    """Lookup table from SGBM's fixed-point int16 disparity straight to mm.

    64 KB, covers every non-negative value an int16 can hold, and — this is the
    point — is *built by* :func:`depth_from_disparity`, so the guard above is
    the only place in this module where the division happens. Entry 0 is
    therefore 0 mm for the same reason every other invalid pixel is.

    Measured elsewhere in this project: the LUT beats a float divide at
    1280x800 (0.93 vs 1.22 ms) and ties it at 640x400.
    """
    n = 1 << 15  # every non-negative int16
    d_px = np.arange(n, dtype=np.float32) / float(1 << fractional_bits)
    lut = depth_from_disparity(d_px, k_mm_px, z_range_mm)
    assert lut[0] == INVALID_MM, "LUT entry 0 must be the invalid code"
    return lut


def apply_mm_lut(disparity_fixed: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map an int16 fixed-point SGBM disparity image to uint16 mm via ``lut``."""
    if disparity_fixed.dtype != np.int16:
        raise TypeError(f"expected SGBM's int16 fixed-point disparity, "
                        f"got {disparity_fixed.dtype}")
    # Negative values are the matcher's own "no match" flag, not a far distance.
    # Folding them onto index 0 sends them to lut[0] == 0, the same invalid code
    # the guard produces for d == 0. Nothing else in the range needs clipping:
    # a non-negative int16 is always a valid index into a 32768-entry table.
    idx = np.maximum(disparity_fixed, np.int16(0))
    return lut[idx]


# =============================================================================
# NEAREST-only depth resampling
# =============================================================================
def resize_depth_nearest(depth_mm: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a uint16 mm depth map with NEAREST, and nothing but NEAREST.

    INTER_NEAREST is not a speed choice here, it is a correctness one. Every
    interpolating kernel mixes neighbouring samples, and on a depth map that
    means:

      * across a depth discontinuity it returns the average of a near surface
        and a far one — a distance at which nothing exists;
      * across the 0 invalid sentinel it returns some fraction of a real
        reading — a confident measurement in a place that was never measured,
        biased *towards the camera*, which is the dangerous direction for
        anything doing obstacle avoidance.

    The device output already shows this: it computes integer disparity
    (``pipeline.py:239`` sets subpixel off, so at most 93 distinct levels are
    possible) yet a recorded 480x270 frame carries 1800 distinct mm values —
    all of them manufactured by decimation, spatial filtering and the output
    resize. NEAREST cannot create a value that was not in the source, which is
    exactly the property ``test_nearest_resampling_invents_no_values`` pins.
    """
    if depth_mm.dtype != np.uint16:
        raise TypeError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")
    w, h = int(size[0]), int(size[1])
    if (depth_mm.shape[1], depth_mm.shape[0]) == (w, h):
        return depth_mm
    return cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# Calibration
# =============================================================================
@dataclass(frozen=True)
class ColorRig:
    """Everything needed to project rectified-left points into the colour frame.

    ``R``/``t_mm`` map a point from the RECTIFIED-LEFT frame to the colour
    frame: ``P_color = R @ P_rectleft + t_mm``, both in millimetres.
    """

    K: np.ndarray                 # 3x3, at `size`
    dist: np.ndarray              # OpenCV coefficients, may be empty
    size: tuple[int, int]         # (w, h)
    R: np.ndarray                 # 3x3
    t_mm: np.ndarray              # (3,)

    def describe(self) -> dict:
        return {
            "size": list(self.size),
            "fx": float(self.K[0, 0]), "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]), "cy": float(self.K[1, 2]),
            "distortion": [float(x) for x in np.asarray(self.dist).ravel()],
            "R_rectleft_to_color": self.R.tolist(),
            "t_rectleft_to_color_mm": [float(x) for x in self.t_mm],
        }


#: The only two provenances a rectified focal length is allowed to have.
FX_RECT_SOURCES = (
    "cv2.stereoRectify:P1[0,0]",   # computed here; P1 *defines* the images matched
    "explicit",                     # a number the caller measured and passed in
)


@dataclass(frozen=True)
class StereoRig:
    """Pure geometry: no OpenCV matcher state, no frames, no device.

    Split out from the matcher so every constructor can be exercised without a
    camera and so a saved dataset and a live session run identical code.
    """

    mono_size: tuple[int, int]          # (w, h) of the images that will be matched
    fx_rect: float
    fy_rect: float
    cx_rect: float
    cy_rect: float
    baseline_mm: float
    fx_rect_source: str
    baseline_source: str
    map_left: tuple[np.ndarray, np.ndarray] | None = None
    map_right: tuple[np.ndarray, np.ndarray] | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    alpha: float | None = None
    color: ColorRig | None = None
    provenance: dict = field(default_factory=dict)

    # -- invariants ---------------------------------------------------------
    def __post_init__(self) -> None:
        if self.fx_rect_source not in FX_RECT_SOURCES:
            # Belt and braces for refusal 1: even a future constructor cannot
            # smuggle a raw EEPROM fx in without naming a legal provenance.
            raise CalibrationIncomplete(
                f"fx_rect_source={self.fx_rect_source!r} is not one of "
                f"{FX_RECT_SOURCES}. The rectified focal length may only come "
                f"from cv2.stereoRectify's P1 or from an explicitly measured "
                f"value; the raw EEPROM fx differs by ~0.4% and that is a "
                f"silent metric scale bias.")
        if not (self.fx_rect > 0.0 and self.fy_rect > 0.0):
            raise CalibrationIncomplete(f"non-positive rectified focal length: "
                                        f"fx={self.fx_rect} fy={self.fy_rect}")
        # A baseline in the wrong sign or the wrong unit is the single most
        # likely silent error here: getCameraExtrinsics and getBaselineDistance
        # both return CENTIMETRES while depth is millimetres. 7.5 read as mm
        # instead of 75 makes everything ten times closer, uniformly.
        if not (10.0 <= self.baseline_mm <= 500.0):
            raise CalibrationIncomplete(
                f"baseline {self.baseline_mm:.4f} mm is outside the sane range "
                f"[10, 500] mm for this class of camera (this unit is ~75 mm). "
                f"A value near 7.5 means centimetres were used as millimetres; "
                f"a negative value means the extrinsics point right->left, so "
                f"pass them the other way round.")

    # -- derived ------------------------------------------------------------
    @property
    def k_mm_px(self) -> float:
        """``fx_rect * baseline_mm`` — the whole disparity->distance constant."""
        return float(self.fx_rect) * float(self.baseline_mm)

    @property
    def rectifies(self) -> bool:
        """True when this rig can turn raw mono into rectified mono."""
        return self.map_left is not None and self.map_right is not None

    def min_z_mm(self, num_disparities: int, min_disparity: int = 0) -> float:
        """Closest measurable distance: ``K / d_max``.

        ``d_max = min_disparity + num_disparities - 1`` because ``numDisparities``
        counts levels starting at ``min_disparity`` — nd=96 searches d in [0, 95],
        which is exactly the device's default max disparity of 95.
        """
        d_max = min_disparity + num_disparities - 1
        if d_max <= 0:
            raise ValueError("num_disparities must leave at least one positive "
                             "disparity level")
        return self.k_mm_px / float(d_max)

    def describe(self) -> dict:
        """Everything a dataset needs to be re-derivable later."""
        out: dict[str, Any] = {
            "mono_size": list(self.mono_size),
            "fx_rect": self.fx_rect, "fy_rect": self.fy_rect,
            "cx_rect": self.cx_rect, "cy_rect": self.cy_rect,
            "baseline_mm": self.baseline_mm,
            "k_mm_px": self.k_mm_px,
            "fx_rect_source": self.fx_rect_source,
            "baseline_source": self.baseline_source,
            "alpha": self.alpha,
            "rectifies_on_host": self.rectifies,
        }
        out.update(self.provenance)
        if self.color is not None:
            out["color"] = self.color.describe()
        return out

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_stereo_pair(cls, *,
                         mono_size: tuple[int, int],
                         K_left: np.ndarray, dist_left: np.ndarray,
                         K_right: np.ndarray, dist_right: np.ndarray,
                         R_left_to_right: np.ndarray,
                         T_left_to_right_mm: np.ndarray,
                         alpha: float = 0.0,
                         color: ColorRig | None = None,
                         build_maps: bool = True,
                         note: str = "") -> "StereoRig":
        """Run ``cv2.stereoRectify`` and take the rectified focal length from P1.

        ``R_left_to_right``/``T_left_to_right_mm`` follow OpenCV's convention,
        the same one DepthAI's ``getCameraExtrinsics(CAM_B, CAM_C)`` uses:
        ``P_right = R @ P_left + T``. For this camera T is about
        ``(-75, 0, 0)`` mm.

        Colour alignment is attached afterwards with :meth:`with_color`, which
        takes the raw ``getCameraExtrinsics(CAM_B, CAM_A)`` pair and composes R1
        in so the caller never has to.

        alpha is recorded, because it *is* the answer. Measured with this
        camera's real saved intrinsics (dataset_20260729_173152) at 640x400:

            alpha    fx_rect     K (mm*px)   MinZ@nd=96   MinZ@nd=192
            -1.0     379.416      28451.2     299.49 mm     148.96 mm
             0.0     364.814      27356.2     287.96 mm     143.23 mm
             1.0     326.027      24447.7     257.34 mm     128.00 mm

        That is a 16% spread in reported distance across the alpha range, so
        datasets built with different alpha are not comparable and the value
        goes into ``describe()``.

        It also settles an open question — the one ``TESTING.md`` used to record
        as unresolved: does the rectified pair inherit CAM_B's fx (757.12/2 =
        378.56) or CAM_C's (760.28/2 = 380.14)? **CAM_C.** Two independent
        measurements agree:

          * this session measured the device's rectified value as ~380.00 with
            MinZ 299 mm, which is 0.04% from CAM_C's 380.14 and 0.4% from
            CAM_B's 378.56;
          * every depth PNG the device has written floors at exactly 300 mm,
            and every ``--extended`` one at exactly 150 mm, with zero pixels
            below either (``c3_camera/datasets/*/depth/``,
            ``c3_camera/recordings/*/depth/`` — 47,270 frames, 2026-08-05).
            A floor printing as exactly 300 needs fx*B/95 in [300, 301), i.e.
            fx_rect in [380.0, 381.3). CAM_B's value would have printed 298.

        The alpha=-1 row above is 379.416 / 299.49 mm, which reproduces the
        host-side number. So the device's rectification behaves like OpenCV's
        default scaling, and the rectified focal length is still a *derived*
        quantity that no EEPROM entry contains — which is the whole reason this
        module refuses to guess it.

        The default here stays alpha=0 (crop to fully-valid pixels, no black
        border, every pixel a real measurement); switch to -1 if you need host
        depth to sit on the same scale as recorded device depth.
        """
        w, h = int(mono_size[0]), int(mono_size[1])
        K_l = np.asarray(K_left, dtype=np.float64).reshape(3, 3)
        K_r = np.asarray(K_right, dtype=np.float64).reshape(3, 3)
        D_l = np.asarray(dist_left, dtype=np.float64).ravel()
        D_r = np.asarray(dist_right, dtype=np.float64).ravel()
        R = np.asarray(R_left_to_right, dtype=np.float64).reshape(3, 3)
        T = np.asarray(T_left_to_right_mm, dtype=np.float64).ravel()

        for name, D in (("left", D_l), ("right", D_r)):
            if D.size not in (0, 4, 5, 8, 12, 14):
                raise UnsupportedCalibration(
                    f"{name} distortion has {D.size} coefficients; OpenCV takes "
                    f"4, 5, 8, 12 or 14")

        R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, (w, h), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(alpha))

        fx_rect = float(P1[0, 0])
        fy_rect = float(P1[1, 1])
        cx_rect = float(P1[0, 2])
        cy_rect = float(P1[1, 2])
        # For a horizontal rig OpenCV puts -fx*Tx into P2[0,3]; the baseline is
        # therefore self-consistent with the very rectification being used, which
        # is the only baseline that pairs correctly with this fx_rect.
        baseline_mm = float(-P2[0, 3] / P2[0, 0])

        maps_l = maps_r = None
        if build_maps:
            maps_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_16SC2)
            maps_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_16SC2)

        prov = {
            "fx_left_raw": float(K_l[0, 0]),
            "fx_right_raw": float(K_r[0, 0]),
            # Surfaced deliberately: this is the 0.4% that must never be
            # substituted for fx_rect. Recording it makes the gap auditable
            # instead of folklore.
            "fx_rect_vs_left_raw_pct":
                100.0 * (fx_rect - float(K_l[0, 0])) / float(K_l[0, 0]),
            "fx_rect_vs_right_raw_pct":
                100.0 * (fx_rect - float(K_r[0, 0])) / float(K_r[0, 0]),
            "T_left_to_right_mm": [float(x) for x in T],
            "R1": R1.tolist(), "P1": P1.tolist(), "P2": P2.tolist(),
        }
        if note:
            prov["note"] = note

        rig = cls(mono_size=(w, h), fx_rect=fx_rect, fy_rect=fy_rect,
                  cx_rect=cx_rect, cy_rect=cy_rect, baseline_mm=baseline_mm,
                  fx_rect_source="cv2.stereoRectify:P1[0,0]",
                  baseline_source="cv2.stereoRectify:-P2[0,3]/P2[0,0]",
                  map_left=maps_l, map_right=maps_r,
                  R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, alpha=float(alpha),
                  color=color, provenance=prov)
        return rig

    @classmethod
    def pre_rectified(cls, *,
                      mono_size: tuple[int, int],
                      fx_rect: float, fy_rect: float,
                      cx_rect: float, cy_rect: float,
                      baseline_mm: float,
                      color: ColorRig | None = None,
                      note: str = "") -> "StereoRig":
        """For images the DEVICE already rectified — every number supplied.

        There is no lookup for these. DepthAI 2.32 does not expose a rectified
        focal length anywhere: ``dai.StereoRectification`` carries only
        ``rectifiedRotationLeft``/``rectifiedRotationRight``,
        ``dai.CameraInfo.intrinsicMatrix`` is the *unrectified* K, and
        ``dai.ImgFrame`` carries no calibration at all. So if you take the
        device's rectified pair, you owe this module a measured fx_rect —
        measured, for instance, by fitting ``Z_device = K/d`` over a recording
        made with decimation and every filter off.

        Deriving these from the EEPROM instead is precisely the 0.4% error this
        module refuses to make, which is why there is no convenience overload
        that reads a calibration file and calls this.
        """
        return cls(mono_size=(int(mono_size[0]), int(mono_size[1])),
                   fx_rect=float(fx_rect), fy_rect=float(fy_rect),
                   cx_rect=float(cx_rect), cy_rect=float(cy_rect),
                   baseline_mm=float(baseline_mm),
                   fx_rect_source="explicit",
                   baseline_source="explicit",
                   color=color,
                   provenance={"note": note or "caller-supplied rectified intrinsics"})

    def with_color(self, *, K: np.ndarray, dist: np.ndarray,
                   size: tuple[int, int],
                   R_left_to_color: np.ndarray,
                   t_left_to_color_mm: np.ndarray) -> "StereoRig":
        """Attach colour alignment, composing R1 in so the caller passes the
        raw ``getCameraExtrinsics(CAM_B, CAM_A)`` pair.

        A point in the rectified-left frame is ``P_r = R1 @ P_left``, so
        ``P_left = R1.T @ P_r`` and
        ``P_color = R_left_to_color @ R1.T @ P_r + t``.
        """
        R_lc = np.asarray(R_left_to_color, dtype=np.float64).reshape(3, 3)
        t_lc = np.asarray(t_left_to_color_mm, dtype=np.float64).ravel()
        R1 = np.eye(3) if self.R1 is None else np.asarray(self.R1, dtype=np.float64)
        col = ColorRig(K=np.asarray(K, dtype=np.float64).reshape(3, 3),
                       dist=np.asarray(dist, dtype=np.float64).ravel(),
                       size=(int(size[0]), int(size[1])),
                       R=R_lc @ R1.T, t_mm=t_lc)
        return StereoRig(**{**self.__dict__, "color": col})

    # -- from a live handler ------------------------------------------------
    @classmethod
    def from_calibration_handler(cls, calib, mono_size: tuple[int, int], *,
                                 color_size: tuple[int, int] | None = None,
                                 alpha: float = 0.0,
                                 left_socket=None, right_socket=None,
                                 color_socket=None,
                                 use_spec_translation: bool = False
                                 ) -> "StereoRig":
        """Build from a live ``dai.CalibrationHandler``. Opens nothing.

        The handler is passed in already constructed (``device.readCalibration()``
        is the caller's business), so this function never sees a ``dai.Device``.
        ``depthai`` is imported here and nowhere else in the file, purely for the
        socket enum defaults.

        ``getCameraExtrinsics`` returns a 4x4 whose translation is in
        CENTIMETRES — the docstring says so — so it is multiplied by 10 here and
        the millimetre value is what gets stored. ``use_spec_translation=False``
        asks for the *calibrated* translation, which is what pairs with the
        calibrated rotation for rectification; the device's own disparity->depth
        uses the *spec* baseline instead (``setDisparityToDepthUseSpecTranslation``
        defaults to true), so both are recorded and any disagreement is visible
        in ``describe()``.
        """
        import depthai as dai  # local: nothing above this line needs depthai

        left_socket = dai.CameraBoardSocket.CAM_B if left_socket is None else left_socket
        right_socket = dai.CameraBoardSocket.CAM_C if right_socket is None else right_socket
        color_socket = dai.CameraBoardSocket.CAM_A if color_socket is None else color_socket

        w, h = int(mono_size[0]), int(mono_size[1])

        model_l = _distortion_model_name(calib, left_socket)
        model_r = _distortion_model_name(calib, right_socket)
        for name, model in (("left", model_l), ("right", model_r)):
            if model not in (None, "Perspective"):
                raise UnsupportedCalibration(
                    f"{name} camera uses the {model} distortion model; this "
                    f"module only implements OpenCV's Perspective (rational "
                    f"polynomial) path. cv2.fisheye.stereoRectify would be a "
                    f"different code path and is deliberately not guessed at.")

        K_l = np.array(calib.getCameraIntrinsics(left_socket, w, h), dtype=np.float64)
        K_r = np.array(calib.getCameraIntrinsics(right_socket, w, h), dtype=np.float64)
        D_l = np.array(calib.getDistortionCoefficients(left_socket), dtype=np.float64)
        D_r = np.array(calib.getDistortionCoefficients(right_socket), dtype=np.float64)

        try:
            ext = np.array(calib.getCameraExtrinsics(
                left_socket, right_socket, use_spec_translation), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise CalibrationIncomplete(
                f"getCameraExtrinsics({left_socket}, {right_socket}) failed: "
                f"{type(e).__name__}: {e}. Without the stereo extrinsics there "
                f"is no rectification and therefore no rectified focal length; "
                f"this module will not substitute the EEPROM fx.") from e
        R, T_mm = _split_extrinsics_cm(ext)

        rig = cls.from_stereo_pair(
            mono_size=(w, h), K_left=K_l, dist_left=D_l,
            K_right=K_r, dist_right=D_r,
            R_left_to_right=R, T_left_to_right_mm=T_mm, alpha=alpha,
            note="dai.CalibrationHandler")

        spec_baseline_mm = None
        try:
            spec_baseline_mm = float(calib.getBaselineDistance()) * 10.0
        except Exception:  # noqa: BLE001
            pass
        rig.provenance["baseline_spec_mm"] = spec_baseline_mm
        rig.provenance["extrinsics_use_spec_translation"] = bool(use_spec_translation)
        rig.provenance["distortion_model"] = {"left": model_l, "right": model_r}

        if color_size is not None:
            cw, ch = int(color_size[0]), int(color_size[1])
            K_c = np.array(calib.getCameraIntrinsics(color_socket, cw, ch),
                           dtype=np.float64)
            D_c = np.array(calib.getDistortionCoefficients(color_socket),
                           dtype=np.float64)
            try:
                ext_c = np.array(calib.getCameraExtrinsics(
                    left_socket, color_socket, use_spec_translation),
                    dtype=np.float64)
            except Exception as e:  # noqa: BLE001
                raise CalibrationIncomplete(
                    f"getCameraExtrinsics({left_socket}, {color_socket}) failed: "
                    f"{type(e).__name__}: {e}. Colour alignment needs it; ask "
                    f"for color_size=None to skip alignment instead.") from e
            R_lc, t_lc = _split_extrinsics_cm(ext_c)
            rig = rig.with_color(K=K_c, dist=D_c, size=(cw, ch),
                                 R_left_to_color=R_lc, t_left_to_color_mm=t_lc)
        return rig

    # -- from a saved dataset ----------------------------------------------
    @classmethod
    def from_calibration_json(cls, path, mono_size: tuple[int, int] | None = None, *,
                              color_size: tuple[int, int] | None = None,
                              alpha: float = 0.0,
                              stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None = None,
                              color_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
                              ) -> "StereoRig":
        """Rebuild the rig from a dataset's ``calibration.json``, for reprocessing.

        Intrinsics are read from ``streamed_intrinsics.left`` / ``.right`` when
        they are present (they are already at the streamed size and keep all 14
        distortion coefficients); otherwise from ``device.intrinsics.CAM_B`` /
        ``CAM_C``, rescaled — note that writer truncates distortion to 8
        coefficients (``device.py:399`` does ``dist[:8]``), which is lossless for
        this camera because coefficients 9-14 are all zero, and that is checked
        rather than assumed.

        Extrinsics are looked for, in order, at::

            device.extrinsics.left_to_right          {"rotation": 3x3, "t_mm"|"translation_mm"}
            device.extrinsics.CAM_B_to_CAM_C
            stereo_extrinsics                        (top level, same shape)

        A translation key **must** carry its unit in its name (``_mm`` or
        ``_cm``). An unlabelled ``translation`` is refused rather than assumed
        to be millimetres, because DepthAI's own API returns centimetres and a
        factor of ten here reads as a plausible scene rather than a broken one.

        Datasets recorded before the extrinsics were added carry none, so this
        raises ``CalibrationIncomplete`` for them. That is the correct outcome:
        they cannot be reprocessed into metric host depth, and pretending
        otherwise by reaching for the EEPROM fx is the 0.4% bias. Pass
        ``stereo_extrinsics=(R, T_mm)`` explicitly if you have them from
        elsewhere.
        """
        path = Path(path)
        doc = json.loads(path.read_text())

        intr_l, intr_r, intr_c = _intrinsics_from_doc(doc, mono_size, color_size)
        mono_size = (intr_l["width"], intr_l["height"])

        if stereo_extrinsics is None:
            stereo_extrinsics = _extrinsics_from_doc(
                doc, ("left_to_right", "CAM_B_to_CAM_C"), what="stereo (left->right)",
                source_hint=str(path))
        R, T_mm = (np.asarray(stereo_extrinsics[0], dtype=np.float64).reshape(3, 3),
                   np.asarray(stereo_extrinsics[1], dtype=np.float64).ravel())

        rig = cls.from_stereo_pair(
            mono_size=mono_size,
            K_left=_K_of(intr_l), dist_left=intr_l["distortion"],
            K_right=_K_of(intr_r), dist_right=intr_r["distortion"],
            R_left_to_right=R, T_left_to_right_mm=T_mm, alpha=alpha,
            note=f"calibration.json: {path}")
        rig.provenance["calibration_json"] = str(path)
        rig.provenance["intrinsics_source"] = {"left": intr_l["source"],
                                               "right": intr_r["source"]}

        if color_size is not None:
            if intr_c is None:
                raise CalibrationIncomplete(
                    "colour alignment requested but the file carries no colour "
                    "intrinsics (looked in streamed_intrinsics.color and "
                    "device.intrinsics.CAM_A)")
            if color_extrinsics is None:
                color_extrinsics = _extrinsics_from_doc(
                    doc, ("left_to_color", "CAM_B_to_CAM_A"),
                    what="colour (left->colour)", source_hint=str(path))
            rig = rig.with_color(K=_K_of(intr_c), dist=intr_c["distortion"],
                                 size=(intr_c["width"], intr_c["height"]),
                                 R_left_to_color=color_extrinsics[0],
                                 t_left_to_color_mm=color_extrinsics[1])
        return rig

    # -- for tests and benchmarks ------------------------------------------
    @classmethod
    def synthetic(cls, mono_size: tuple[int, int] = (640, 400), *,
                  fx_rect: float = 380.0, baseline_mm: float = 75.0,
                  color: ColorRig | None = None) -> "StereoRig":
        """An ideal rig: identity rectification, exact fx, no distortion.

        Nothing here comes off a camera, so ``fx_rect_source`` is "explicit" and
        the images are by definition already rectified — which makes this the
        rig for synthetic tests and for ``--benchmark``'s pre-rectified rows.
        fx scales with width so 1280x800 gets 760.0, matching how the sensor's
        400p mode is a 2x2 binning of the 800p mode.
        """
        w, h = int(mono_size[0]), int(mono_size[1])
        fx = float(fx_rect) * (w / 640.0)
        fy = fx
        return cls.pre_rectified(mono_size=(w, h), fx_rect=fx, fy_rect=fy,
                                 cx_rect=(w - 1) / 2.0, cy_rect=(h - 1) / 2.0,
                                 baseline_mm=float(baseline_mm), color=color,
                                 note="synthetic rig (no camera)")


# ---------------------------------------------------------------- json helpers
def _K_of(intr: dict) -> np.ndarray:
    return np.array([[intr["fx"], 0.0, intr["cx"]],
                     [0.0, intr["fy"], intr["cy"]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _scale_intr(intr: dict, width: int, height: int) -> dict:
    sx, sy = width / intr["width"], height / intr["height"]
    out = dict(intr)
    out.update(fx=intr["fx"] * sx, fy=intr["fy"] * sy,
               cx=intr["cx"] * sx, cy=intr["cy"] * sy,
               width=width, height=height)
    return out


def _intrinsics_from_doc(doc: dict, mono_size, color_size):
    """Pull left/right/colour intrinsics out of a calibration.json."""
    streamed = doc.get("streamed_intrinsics") or {}
    dev = ((doc.get("device") or {}).get("intrinsics")) or {}

    def pick(stream_key: str, socket_key: str, want) -> dict | None:
        if stream_key in streamed:
            s = streamed[stream_key]
            out = {"fx": float(s["fx"]), "fy": float(s["fy"]),
                   "cx": float(s["cx"]), "cy": float(s["cy"]),
                   "width": int(s["width"]), "height": int(s["height"]),
                   "distortion": np.asarray(s.get("distortion") or [], np.float64),
                   "source": f"streamed_intrinsics.{stream_key}"}
        elif socket_key in dev:
            d = dev[socket_key]
            if not isinstance(d, dict):
                return None
            dw, dh = d["default_size"]
            out = {"fx": float(d["fx"]), "fy": float(d["fy"]),
                   "cx": float(d["cx"]), "cy": float(d["cy"]),
                   "width": int(dw), "height": int(dh),
                   "distortion": np.asarray(d.get("distortion") or [], np.float64),
                   "source": f"device.intrinsics.{socket_key} (rescaled)"}
        else:
            return None
        if want is not None and (out["width"], out["height"]) != tuple(want):
            out = _scale_intr(out, int(want[0]), int(want[1]))
        return out

    intr_l = pick("left", "CAM_B", mono_size)
    intr_r = pick("right", "CAM_C", mono_size)
    if intr_l is None or intr_r is None:
        raise CalibrationIncomplete(
            "calibration.json carries no usable left/right intrinsics "
            "(looked in streamed_intrinsics.left/right and "
            "device.intrinsics.CAM_B/CAM_C)")
    if mono_size is None and (intr_l["width"], intr_l["height"]) != \
            (intr_r["width"], intr_r["height"]):
        raise CalibrationIncomplete(
            f"left and right intrinsics are stored at different sizes "
            f"({intr_l['width']}x{intr_l['height']} vs "
            f"{intr_r['width']}x{intr_r['height']}); pass mono_size explicitly")
    intr_c = pick("color", "CAM_A", color_size) if color_size is not None else None
    return intr_l, intr_r, intr_c


_TRANSLATION_KEYS_MM = ("t_mm", "translation_mm", "T_mm")
_TRANSLATION_KEYS_CM = ("t_cm", "translation_cm", "T_cm")
_ROTATION_KEYS = ("rotation", "R", "rotation_matrix")


def _extrinsics_from_doc(doc: dict, names: Sequence[str], *, what: str,
                         source_hint: str) -> tuple[np.ndarray, np.ndarray]:
    """Find one extrinsics block, insisting the translation names its unit."""
    containers = [(doc.get("device") or {}).get("extrinsics") or {},
                  doc.get("extrinsics") or {},
                  doc]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for name in names:
            blk = container.get(name)
            if not isinstance(blk, dict):
                continue
            R = None
            for rk in _ROTATION_KEYS:
                if rk in blk:
                    R = np.asarray(blk[rk], dtype=np.float64).reshape(3, 3)
                    break
            if R is None:
                raise CalibrationIncomplete(
                    f"{what} extrinsics block {name!r} has no rotation "
                    f"(looked for {_ROTATION_KEYS})")
            for tk in _TRANSLATION_KEYS_MM:
                if tk in blk:
                    return R, np.asarray(blk[tk], dtype=np.float64).ravel()
            for tk in _TRANSLATION_KEYS_CM:
                if tk in blk:
                    # DepthAI returns centimetres; depth is millimetres.
                    return R, np.asarray(blk[tk], dtype=np.float64).ravel() * 10.0
            raise CalibrationIncomplete(
                f"{what} extrinsics block {name!r} has a translation with no "
                f"unit in its key name. Store it as one of {_TRANSLATION_KEYS_MM} "
                f"or {_TRANSLATION_KEYS_CM}: DepthAI's getCameraExtrinsics "
                f"returns CENTIMETRES and this pipeline is millimetres, so an "
                f"unlabelled vector is a 10x scale bug waiting to happen.")
    raise CalibrationIncomplete(
        f"no {what} extrinsics in {source_hint}. Looked for {list(names)} under "
        f"device.extrinsics / extrinsics / the document root. Datasets recorded "
        f"before describe_calibration() started saving getCameraExtrinsics have "
        f"none, and they cannot be reprocessed into metric host depth: without "
        f"R and T there is no rectification, hence no rectified focal length, "
        f"and substituting the EEPROM fx is a ~0.4% silent scale bias. Pass "
        f"stereo_extrinsics=(R, T_mm) explicitly if you have them.")


def _split_extrinsics_cm(ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DepthAI's 4x4 extrinsics -> (R, T in MILLIMETRES).

    ``getCameraExtrinsics`` documents its translation as centimetres. Depth here
    is millimetres. The x10 lives in exactly this one function so there is one
    place to check when a distance looks ten times wrong.
    """
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        R = ext[:3, :3].copy()
        t_cm = ext[:3, 3].copy()
    elif ext.shape == (3, 4):
        R = ext[:3, :3].copy()
        t_cm = ext[:3, 3].copy()
    else:
        raise UnsupportedCalibration(f"expected a 4x4 or 3x4 extrinsics matrix, "
                                     f"got shape {ext.shape}")
    return R, t_cm * 10.0


def _distortion_model_name(calib, socket) -> str | None:
    try:
        return str(calib.getDistortionModel(socket).name)
    except Exception:  # noqa: BLE001
        return None


# =============================================================================
# Colour alignment
# =============================================================================
def warp_depth_to_color(depth_mm: np.ndarray, rig: StereoRig) -> np.ndarray:
    """Move a rectified-left depth map onto the colour grid.

    Forward scatter with a z-buffer and NEAREST placement, in that combination
    for measured reasons:

      * **Forward**, not inverse-warp, because an inverse warp needs a depth for
        every colour pixel before it has one — the chicken-and-egg that alignment
        exists to break.
      * **z-buffer**, because the projection is many-to-one: warping 1280x800
        depth onto a 480x270 colour grid collided on 70.9% of the projected
        points in a direct measurement, and letting the last write win instead of
        the nearest changed 17.11% of the grid. Without it a background pixel
        can overwrite the object in front of it.
      * **NEAREST** (``np.rint`` into an integer pixel index, never a weighted
        splat), for the invented-distance reason in ``resize_depth_nearest``.

    The colour lens distortion is applied. Skipping it changed 21.8% of the
    output grid in a direct check while costing about 2 ms — not an optimisation
    worth having.

    Note the shape constraint: the output grid must not be finer than the depth
    grid or the scatter goes to holes. 640x400 -> 480x270 fills 36.1% of the
    grid; 640x400 -> 960x540 fills 9.6%. A warning is emitted for the latter.
    """
    if rig.color is None:
        raise HostDepthError("this rig has no colour alignment; build it with "
                             "StereoRig.with_color(...) or from_calibration_json"
                             "(..., color_size=(w, h))")
    if depth_mm.dtype != np.uint16:
        raise TypeError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")

    col = rig.color
    cw, ch = col.size
    out = np.zeros((ch, cw), dtype=np.uint16)

    vs, us = np.nonzero(depth_mm)  # only measured pixels project anywhere
    if vs.size == 0:
        return out
    z = depth_mm[vs, us].astype(np.float64)

    # Deproject on the rectified-left grid.
    x = (us.astype(np.float64) - rig.cx_rect) * z / rig.fx_rect
    y = (vs.astype(np.float64) - rig.cy_rect) * z / rig.fy_rect
    pts = np.stack([x, y, z], axis=1)                 # (N, 3), millimetres

    # Rectified-left -> colour.
    pc = pts @ np.asarray(col.R, dtype=np.float64).T + \
        np.asarray(col.t_mm, dtype=np.float64).reshape(1, 3)
    zc = pc[:, 2]
    mm = depth_mm[vs, us]
    front = zc > 0.0
    if not front.all():          # the common case skips three array copies
        if not front.any():
            return out
        pc = pc[front]
        zc = pc[:, 2]
        mm = mm[front]

    u_c, v_c = _project_with_distortion(pc, col.K, col.dist)

    # NEAREST placement. rint, not floor: floor biases every point half a pixel
    # up-left, which shows up as a systematic shift when the grids are close in
    # size.
    ui = np.rint(u_c).astype(np.int32)
    vi = np.rint(v_c).astype(np.int32)
    inside = (ui >= 0) & (ui < cw) & (vi >= 0) & (vi < ch)
    if not inside.all():
        if not inside.any():
            return out
        ui, vi, zc, mm = ui[inside], vi[inside], zc[inside], mm[inside]

    # z-buffer by construction: write far surfaces first so near ones overwrite
    # them. One sort is cheaper and simpler to reason about than a
    # compare-and-swap loop, and it is deterministic.
    order = np.argsort(-zc, kind="stable")
    out[vi[order], ui[order]] = mm[order]
    return out


def _project_with_distortion(pts_cam: np.ndarray, K: np.ndarray,
                             dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points through K and the OpenCV distortion model.

    Written out rather than calling ``cv2.projectPoints`` because that call
    measured 7.6 ms for 50k points in this venv against well under 1 ms here,
    and this runs per frame. The rational + thin-prism terms are implemented; if
    the tilted-sensor terms (coefficients 13/14) are ever non-zero this falls
    back to ``cv2.projectPoints`` rather than quietly ignoring them. On this
    camera coefficients 9-14 are all exactly 0.0.
    """
    d = np.asarray(dist, dtype=np.float64).ravel()
    d14 = np.zeros(14, dtype=np.float64)
    d14[:min(d.size, 14)] = d[:14]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy = d14

    if taux != 0.0 or tauy != 0.0:
        uv, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3),
                                  np.zeros(3), np.zeros(3),
                                  np.asarray(K, np.float64), d)
        uv = uv.reshape(-1, 2)
        return uv[:, 0], uv[:, 1]

    xp = pts_cam[:, 0] / pts_cam[:, 2]
    yp = pts_cam[:, 1] / pts_cam[:, 2]
    r2 = xp * xp + yp * yp
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    xy2 = 2.0 * xp * yp
    xd = xp * radial + p1 * xy2 + p2 * (r2 + 2.0 * xp * xp) + s1 * r2 + s2 * r4
    yd = yp * radial + p1 * (r2 + 2.0 * yp * yp) + p2 * xy2 + s3 * r2 + s4 * r4
    K = np.asarray(K, dtype=np.float64)
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


# =============================================================================
# The matcher
# =============================================================================
@dataclass(frozen=True)
class MatcherParams:
    """SGBM knobs, with the measured trade of each one.

    Measured on 640x400 real frames from this camera (5-frame mean, baseline
    valid-pixel fraction 40.86%); time was 5.7-7.3 ms for *every* setting below
    except ``block_size`` and ``mode``, so these are quality choices, not speed
    ones:

      block_size 3 / 7 / 9        -> -2.71 / +2.91 / +5.50 pp valid
                                     (small keeps thin structure, large smears
                                      object boundaries)
      uniqueness_ratio 0 / 5 / 15 -> +8.66 / +4.89 / -4.75 pp
                                     (0 keeps every ambiguous match)
      speckle_window_size 0 / 200 -> +12.88 / -2.92 pp
                                     (the +12.88 pp is all speckle)
      p2 = 64*b*b                 -> +5.13 pp, at the cost of terracing slopes
      pre_filter_cap 15 / 63      -> +1.34 / -1.21 pp

    ``mode`` is fixed at SGBM_3WAY because it is the only mode with a
    ``parallel_for`` implementation: 3WAY goes 22.11 ms (1 thread) -> 6.94 ms
    (16 threads), while MODE_SGBM sits at ~26.9 ms no matter how many threads it
    is given, for +0.22 pp of validity.

    ``num_disparities`` does NOT need to be a multiple of 16 — that restriction
    is StereoBM's (``stereobm.cpp:1193``), and 95/190/191 all run fine in SGBM
    here. 96 and 192 are used because they give d_max 95 and 191, covering the
    device's 95 (default) and 190 (extended) exactly.

    The cost of ``num_disparities`` is not the arithmetic, it is the blind band:
    SGBM cannot match the leftmost ``num_disparities`` columns at all. At 640
    wide that is 15% of the image at nd=96 and 30% at nd=192 — which accounts
    for essentially the whole 40.86% -> 30.44% drop in validity when extending
    (restricted to x >= nd the two are 56.58% vs 53.09%). The device's depth has
    no such band, because alignment and the output resize hide it.
    """

    num_disparities: int = 96
    block_size: int = 5
    min_disparity: int = 0
    p1: int | None = None            # None -> 8 * block_size^2
    p2: int | None = None            # None -> 32 * block_size^2
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    mode: int = cv2.StereoSGBM_MODE_SGBM_3WAY

    def resolved_p1(self) -> int:
        return int(self.p1) if self.p1 is not None else 8 * self.block_size ** 2

    def resolved_p2(self) -> int:
        return int(self.p2) if self.p2 is not None else 32 * self.block_size ** 2

    def scaled(self, factor: int) -> "MatcherParams":
        """Same MinZ at 1/factor resolution: disparity scales with image width."""
        if factor <= 1:
            return self
        nd = max(1, int(round(self.num_disparities / factor)))
        return MatcherParams(**{**self.__dict__,
                                "num_disparities": nd,
                                "min_disparity": int(self.min_disparity // factor)})

    def create(self) -> "cv2.StereoSGBM":
        if self.min_disparity < 0:
            # A negative min_disparity would make legitimate matches negative,
            # and apply_mm_lut folds negatives onto the invalid code. Refusing is
            # better than silently deleting half the range.
            raise ValueError("min_disparity < 0 is not supported: negative "
                             "disparities collide with SGBM's own invalid flag")
        return cv2.StereoSGBM_create(
            minDisparity=int(self.min_disparity),
            numDisparities=int(self.num_disparities),
            blockSize=int(self.block_size),
            P1=self.resolved_p1(), P2=self.resolved_p2(),
            disp12MaxDiff=int(self.disp12_max_diff),
            uniquenessRatio=int(self.uniqueness_ratio),
            speckleWindowSize=int(self.speckle_window_size),
            speckleRange=int(self.speckle_range),
            preFilterCap=int(self.pre_filter_cap),
            mode=int(self.mode))

    def describe(self) -> dict:
        modes = {cv2.StereoSGBM_MODE_SGBM: "SGBM",
                 cv2.StereoSGBM_MODE_HH: "HH",
                 cv2.StereoSGBM_MODE_SGBM_3WAY: "SGBM_3WAY",
                 cv2.StereoSGBM_MODE_HH4: "HH4"}
        return {"num_disparities": self.num_disparities,
                "block_size": self.block_size,
                "min_disparity": self.min_disparity,
                "p1": self.resolved_p1(), "p2": self.resolved_p2(),
                "uniqueness_ratio": self.uniqueness_ratio,
                "speckle_window_size": self.speckle_window_size,
                "speckle_range": self.speckle_range,
                "disp12_max_diff": self.disp12_max_diff,
                "pre_filter_cap": self.pre_filter_cap,
                "mode": modes.get(int(self.mode), str(self.mode))}


class HostStereoMatcher:
    """(left, right) gray8 -> uint16 millimetre depth, on this computer.

    ``mono_input`` is positional and required. "raw" means the sensors' own
    frames and this object will rectify them; "rectified" means the device
    already did it and the remap is skipped. Nothing in the pixels distinguishes
    the two, so guessing would mean silently matching unrectified images — which
    does not fail, it just returns wrong distances along every epipolar line
    that is not horizontal.
    """

    def __init__(self, rig: StereoRig, mono_input: str,
                 params: MatcherParams | None = None, *,
                 z_range_mm: tuple[float, float] | None = None,
                 align_to_color: bool = False,
                 downscale: int = 1,
                 upscale_output: bool = True,
                 cv_threads: int | None = None):
        if mono_input not in MONO_INPUTS:
            raise ValueError(f"mono_input must be one of {MONO_INPUTS}, "
                             f"got {mono_input!r} — it is never inferred")
        if mono_input == MONO_INPUT_RAW and not rig.rectifies:
            raise HostDepthError(
                "mono_input='raw' needs rectification maps, and this rig has "
                "none. Build it with StereoRig.from_calibration_handler / "
                "from_calibration_json (which run cv2.stereoRectify), or pass "
                "already-rectified frames with mono_input='rectified'.")
        if mono_input == MONO_INPUT_RECTIFIED and rig.fx_rect_source != "explicit":
            # This rig's fx_rect describes the rectification *this module*
            # computed. Feeding it the device's rectified pair mixes two
            # different rectifications: the images come from one and the focal
            # length from the other, and the mismatch shows up only as a scale
            # error in the final map.
            raise HostDepthError(
                "mono_input='rectified' with a host-computed rig would pair the "
                "device's rectified images with this module's fx_rect. Use "
                "StereoRig.pre_rectified(...) with a measured fx_rect, or send "
                "raw mono and let this module rectify.")
        if downscale < 1:
            raise ValueError("downscale must be >= 1")

        self.rig = rig
        self.mono_input = mono_input
        self.params = params or MatcherParams()
        self.downscale = int(downscale)
        self.upscale_output = bool(upscale_output)
        self.align_to_color = bool(align_to_color)
        self.z_range_mm = z_range_mm

        if cv_threads is not None:
            # Process-global, deliberately opt-in. DatasetWriter runs its own
            # PNG-encoding pool (dataset.py:107-114), so pinning this low from
            # inside a library would quietly throttle recording. Set it once at
            # application start if you set it at all — and never save/restore it
            # around each call, which races between threads.
            cv2.setNumThreads(int(cv_threads))
        self.cv_threads = cv2.getNumThreads()

        self._params_eff = self.params.scaled(self.downscale)
        self._matcher = self._params_eff.create()

        w, h = rig.mono_size
        self._match_size = (max(1, w // self.downscale), max(1, h // self.downscale))
        # fx (and therefore K) scales with image width under downscaling.
        self._k_mm_px = rig.k_mm_px / float(self.downscale)
        self._lut = disparity_mm_lut(self._k_mm_px, z_range_mm)

        if align_to_color:
            if rig.color is None:
                raise HostDepthError("align_to_color=True needs a colour rig")
            cw, ch = rig.color.size
            mw, mh = self._match_size if not self.upscale_output else rig.mono_size
            if cw > mw or ch > mh:
                print(f"  host_depth: WARNING colour grid {cw}x{ch} is finer than "
                      f"the depth grid {mw}x{mh}; a forward scatter fills only "
                      f"~9.6% of a 960x540 grid from 640x400 depth. Align to a "
                      f"grid no larger than the depth grid.", file=sys.stderr)

    # -- geometry -----------------------------------------------------------
    @property
    def k_mm_px(self) -> float:
        """The disparity->distance constant actually in use (downscale folded in)."""
        return self._k_mm_px

    @property
    def min_z_mm(self) -> float:
        p = self._params_eff
        return self._k_mm_px / float(p.min_disparity + p.num_disparities - 1)

    @property
    def max_z_mm(self) -> float:
        """Furthest representable reading: the smallest non-zero disparity SGBM
        can emit is 1/16 px, so this is 16*K, capped by uint16 and by z_range."""
        far = self._k_mm_px * float(1 << SGBM_FRACTIONAL_BITS)
        cap = float(MAX_UINT16_MM) if self.z_range_mm is None \
            else min(float(self.z_range_mm[1]), float(MAX_UINT16_MM))
        return min(far, cap)

    # -- compute ------------------------------------------------------------
    def rectify(self, left: np.ndarray, right: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the rectification maps. No-op when the input is pre-rectified.

        INTER_LINEAR, deliberately: this resamples *intensity*, and smoothly
        interpolating brightness is exactly what rectification is for. The
        NEAREST-only rule in this module is about depth, where interpolation
        fabricates distances. Measured cost: 0.21-0.36 ms for the 640x400 pair.
        """
        if self.mono_input == MONO_INPUT_RECTIFIED:
            return left, right
        ml, mr = self.rig.map_left, self.rig.map_right
        return (cv2.remap(left, ml[0], ml[1], cv2.INTER_LINEAR),
                cv2.remap(right, mr[0], mr[1], cv2.INTER_LINEAR))

    def compute_disparity_fixed(self, left: np.ndarray, right: np.ndarray
                                ) -> np.ndarray:
        """SGBM's native int16 fixed-point disparity (px * 16) on the match grid."""
        _check_pair(left, right, self.rig.mono_size)
        l_r, r_r = self.rectify(left, right)
        if self.downscale > 1:
            # INTER_AREA, not NEAREST: this is an intensity image. Area averaging
            # is the correct decimation filter and it is what keeps the matcher's
            # texture usable at reduced size.
            l_r = cv2.resize(l_r, self._match_size, interpolation=cv2.INTER_AREA)
            r_r = cv2.resize(r_r, self._match_size, interpolation=cv2.INTER_AREA)
        return self._matcher.compute(l_r, r_r)

    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Disparity in pixels, float32. Invalid pixels are negative (SGBM's flag)."""
        return self.compute_disparity_fixed(left, right).astype(np.float32) / \
            float(1 << SGBM_FRACTIONAL_BITS)

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """The one that matters: uint16 millimetres, 0 = no measurement.

        Same dtype, same units and same invalid code as the device path, so a
        recorder can drop this straight into ``Bundle.frames['depth']`` and
        ``DatasetWriter`` needs no change (``dataset.py:400-404``).
        """
        disp16 = self.compute_disparity_fixed(left, right)
        depth = apply_mm_lut(disp16, self._lut)
        if self.downscale > 1 and self.upscale_output:
            depth = resize_depth_nearest(depth, self.rig.mono_size)
        if self.align_to_color:
            depth = warp_depth_to_color(depth, self.rig)
        assert depth.dtype == np.uint16  # the contract, checked on every path
        return depth

    __call__ = compute

    # -- metadata -----------------------------------------------------------
    def describe(self) -> dict:
        """Drop this into metadata.json so two datasets can never be silently mixed.

        Host depth and device depth differ in at least eight ways that make a
        naive comparison of "valid %" or "distinct values" give the wrong sign:
        the device decimates 2x then resizes, spatially filters and medians,
        emits *integer* disparity, thresholds on cost (confidence 245) rather
        than on a uniqueness ratio, has no left blind band, aligns in hardware,
        and rectifies with its own EEPROM mesh. Recording the parameters is what
        makes that recoverable later.
        """
        return {
            "depth_source": "host",
            "module": "c3_camera.host_depth",
            "depth_scale": DEPTH_SCALE,
            "depth_scale_source": DEPTH_SCALE_SOURCE,
            "mono_input": self.mono_input,
            "downscale": self.downscale,
            "upscale_output": self.upscale_output,
            "align_to_color": self.align_to_color,
            "match_size": list(self._match_size),
            "cv_threads": self.cv_threads,
            "opencv_version": cv2.__version__,
            "k_mm_px": self._k_mm_px,
            "min_z_mm": self.min_z_mm,
            "max_z_mm": self.max_z_mm,
            "z_range_mm": list(self.z_range_mm) if self.z_range_mm else None,
            "matcher": self._params_eff.describe(),
            "matcher_requested": self.params.describe(),
            "rig": self.rig.describe(),
        }


def _check_pair(left: np.ndarray, right: np.ndarray, size: tuple[int, int]) -> None:
    for name, img in (("left", left), ("right", right)):
        if img.ndim != 2:
            raise ValueError(f"{name} must be a single-channel gray image, got "
                             f"shape {img.shape}")
        if img.dtype != np.uint8:
            raise TypeError(f"{name} must be gray8, got {img.dtype}")
    if left.shape != right.shape:
        raise ValueError(f"left {left.shape} and right {right.shape} differ")
    if (left.shape[1], left.shape[0]) != tuple(size):
        raise ValueError(f"frames are {left.shape[1]}x{left.shape[0]} but the rig "
                         f"was built for {size[0]}x{size[1]}; rebuild the rig at "
                         f"the size you are actually streaming (intrinsics do not "
                         f"survive a resize by accident)")


# =============================================================================
# Worker-thread path
# =============================================================================
class AsyncHostStereo:
    """Frame-level thread pool, for when one frame at a time is not enough.

    Only needed above 640x400. Measured full-pipeline numbers on this machine:
    640x400/nd=96 runs 10.6 ms single-frame (89 fps aggregate, 11 ms latency),
    so a pool buys nothing there. 1280x800/nd=192 runs 11.5-13.4 fps
    single-frame and needs pool=4 to clear 20 fps — at 210 ms of per-frame
    latency, because pool depth multiplies latency. That is fine for offline
    collection and unusable for teleoperation, so the choice is deliberately the
    caller's.

    Each worker builds its own ``HostStereoMatcher``: ``cv2.StereoSGBM`` keeps
    internal scratch buffers and is not documented thread-safe, so sharing one
    across threads is a data race, not a saving.
    """

    def __init__(self, make_matcher: Callable[[], HostStereoMatcher],
                 workers: int = 2):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self._make = make_matcher
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="host_depth")
        self.workers = workers

    def _matcher(self) -> HostStereoMatcher:
        m = getattr(self._local, "matcher", None)
        if m is None:
            m = self._make()
            self._local.matcher = m
        return m

    def submit(self, left: np.ndarray, right: np.ndarray):
        """Returns a ``Future`` yielding uint16 mm depth."""
        return self._pool.submit(lambda: self._matcher().compute(left, right))

    def map(self, pairs: Iterable[tuple[np.ndarray, np.ndarray]]):
        """Submit everything, yield results in submission order."""
        futures = [self.submit(l, r) for l, r in pairs]
        for f in futures:
            yield f.result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "AsyncHostStereo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# =============================================================================
# Synthetic data — shared by --benchmark and the tests
# =============================================================================
def synthetic_pair(size: tuple[int, int], disparity_px: int, seed: int = 0,
                   blur: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A rectified pair of a fronto-parallel plane at a known integer disparity.

    Construction, and why this direction: with the LEFT image as reference,
    ``d = u_left - u_right``, so a feature at ``u_left`` sits at
    ``u_left - d`` in the right image — i.e. the right image is the left one
    shifted left by ``d``. Both are cut from one wider texture so there is no
    fabricated border content on either side.

    Slight blur, because a pure white-noise pair has no scale structure for the
    smoothness term to work with and SGBM's validity collapses on it; blurred
    noise recovers the true disparity to within 1/16 px, which is the point of a
    ground-truth fixture.
    """
    w, h = int(size[0]), int(size[1])
    d = int(disparity_px)
    if d < 0:
        raise ValueError("disparity_px must be >= 0")
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w + d), dtype=np.uint8)
    if blur:
        base = cv2.GaussianBlur(base, (3, 3), 0)
    left = np.ascontiguousarray(base[:, :w])
    right = np.ascontiguousarray(base[:, d:d + w])
    return left, right


#: Representative unrectified intrinsics for this camera at 640x400, from the
#: EEPROM values already saved in this repo's own dataset calibration files
#: (datasets/dataset_20260729_173152/calibration.json, streamed_intrinsics).
#: Used only to make --benchmark's remap cost realistic; no camera is read.
_BENCH_K_LEFT = np.array([[378.560547, 0.0, 324.433746],
                          [0.0, 378.753113, 190.638000],
                          [0.0, 0.0, 1.0]])
_BENCH_D_LEFT = np.array([-1.405320, -0.626550, -9e-06, -0.001516, 2.000602,
                          -1.174531, -1.184813, 2.438596, 0, 0, 0, 0, 0, 0])
_BENCH_K_RIGHT = np.array([[380.137756, 0.0, 324.441711],
                           [0.0, 380.079559, 193.807693],
                           [0.0, 0.0, 1.0]])
_BENCH_D_RIGHT = np.array([-1.385192, -0.562238, 0.000664, 0.002364, 1.280538,
                           -1.159886, -1.073795, 1.609050, 0, 0, 0, 0, 0, 0])
_BENCH_K_COLOR_480 = np.array([[385.043518, 0.0, 238.144424],
                               [0.0, 385.085297, 134.826645],
                               [0.0, 0.0, 1.0]])
_BENCH_D_COLOR = np.array([2.628258, 81.290871, 0.002348, -0.000616, 188.557495,
                           2.404471, 83.125740, 186.522308, 0, 0, 0, 0, 0, 0])


def bench_rig(mono_size: tuple[int, int], *, with_color: bool = True,
              color_size: tuple[int, int] = (480, 270),
              alpha: float = 0.0) -> StereoRig:
    """A realistic rig for timing: real distortion, real baseline, no camera."""
    w, h = mono_size
    sx, sy = w / 640.0, h / 400.0
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
    rig = StereoRig.from_stereo_pair(
        mono_size=(w, h),
        K_left=S @ _BENCH_K_LEFT, dist_left=_BENCH_D_LEFT,
        K_right=S @ _BENCH_K_RIGHT, dist_right=_BENCH_D_RIGHT,
        R_left_to_right=np.eye(3),
        T_left_to_right_mm=np.array([-75.0, 0.0, 0.0]),
        alpha=alpha,
        note="benchmark rig built from this repo's saved EEPROM values")
    if with_color:
        cw, ch = color_size
        cs = np.array([[cw / 480.0, 0, 0], [0, ch / 270.0, 0], [0, 0, 1.0]])
        rig = rig.with_color(K=cs @ _BENCH_K_COLOR_480, dist=_BENCH_D_COLOR,
                             size=(cw, ch),
                             R_left_to_color=np.eye(3),
                             t_left_to_color_mm=np.array([37.0, 0.0, 0.0]))
    return rig


# =============================================================================
# --benchmark
# =============================================================================
BENCH_CASES = (
    # (label, mono_size, num_disparities)
    ("400p  nd=96",  (640, 400), 96),
    ("400p  nd=192", (640, 400), 192),
    ("720p  nd=192", (1280, 720), 192),
    ("800p  nd=192", (1280, 800), 192),
    ("800p  nd=384", (1280, 800), 384),
)


def _time_stage(fn, frames, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn(0)
    out = []
    for i in range(frames):
        t0 = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def run_benchmark(frames: int = 12, threads: int | None = None,
                  workers: Sequence[int] = (1,), align_to_color: bool = True,
                  downscales: Sequence[int] = (1,),
                  cases: Sequence[tuple] = BENCH_CASES,
                  target_fps: float = 20.0) -> list[dict]:
    """Time the real pipeline on synthetic pairs. Reports what this machine does."""
    if threads is not None:
        cv2.setNumThreads(int(threads))
    print(f"host_depth --benchmark")
    print(f"  opencv {cv2.__version__}   numpy {np.__version__}   "
          f"cv2 threads {cv2.getNumThreads()}")
    print(f"  frames per case {frames}   target {target_fps:.0f} fps   "
          f"colour warp {'on' if align_to_color else 'off'}")
    print(f"  synthetic pairs, no camera; depth is uint16 mm, 0 = invalid")
    print(f"  rect/match/mm/warp are each timed IN ISOLATION and need not add up "
          f"to total.")
    print(f"  total is the end-to-end measurement of compute() and is the number "
          f"that matters:")
    print(f"  on a large-L3 CPU the colour warp evicts the matcher's working set, "
          f"so the")
    print(f"  whole costs measurably more than the sum of its parts. `sum` is "
          f"printed so the")
    print(f"  gap is visible instead of hidden.\n")

    hdr = (f"{'case':<14}{'ds':>3}{'pool':>5}{'rect':>7}{'match':>7}{'mm':>6}"
           f"{'warp':>7}{'sum':>7}{'total':>8}{'fps':>8}{'lat_ms':>8}"
           f"  {'MinZ':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict] = []
    for label, size, nd in cases:
        # Colour grid never finer than the depth grid — a forward scatter onto a
        # finer grid fills ~9.6% of it, which would be timing a hole pattern.
        color_size = (min(480, size[0]), min(270, size[1]))
        rig = bench_rig(size, with_color=align_to_color, color_size=color_size)
        pairs = [synthetic_pair(size, 24, seed=i) for i in range(max(4, frames))]
        for ds in downscales:
            params = MatcherParams(num_disparities=nd)
            mk = (lambda rig=rig, params=params, ds=ds:
                  HostStereoMatcher(rig, MONO_INPUT_RAW, params, downscale=ds,
                                    align_to_color=align_to_color))
            m = mk()

            # Per-stage breakdown, single frame.
            rect = _time_stage(lambda i: m.rectify(*pairs[i % len(pairs)]), frames)
            rectified = [m.rectify(*p) for p in pairs]
            if ds > 1:
                rectified = [(cv2.resize(a, m._match_size, interpolation=cv2.INTER_AREA),
                              cv2.resize(b, m._match_size, interpolation=cv2.INTER_AREA))
                             for a, b in rectified]
            match = _time_stage(
                lambda i: m._matcher.compute(*rectified[i % len(rectified)]), frames)
            disps = [m._matcher.compute(*r) for r in rectified]
            mmt = _time_stage(lambda i: apply_mm_lut(disps[i % len(disps)], m._lut),
                              frames)
            depths = [apply_mm_lut(d, m._lut) for d in disps]
            if ds > 1:
                depths = [resize_depth_nearest(d, size) for d in depths]
            if align_to_color:
                warp = _time_stage(
                    lambda i: warp_depth_to_color(depths[i % len(depths)], rig), frames)
            else:
                warp = [0.0]

            for pool in workers:
                if pool == 1:
                    lat = _time_stage(lambda i: m.compute(*pairs[i % len(pairs)]),
                                      frames)
                    total = float(np.median(lat))
                    fps = 1000.0 / total if total else float("inf")
                    lat_med = total
                else:
                    async_m = AsyncHostStereo(mk, workers=pool)
                    # Warm the per-thread matchers: the first compute() on a
                    # thread also builds that thread's SGBM object.
                    list(async_m.map(pairs[:pool]))
                    # Steady state, not a burst. A capture loop keeps a BOUNDED
                    # number of frames in flight; dumping every frame into the
                    # queue at t0 would report the whole queue drain as latency
                    # and make a deeper pool look arbitrarily worse. So: keep
                    # exactly `pool` frames outstanding, and measure each one
                    # submit-to-result — which is what an operator waits.
                    inflight: list[tuple[float, Any]] = []
                    lat = []
                    t0 = time.perf_counter()
                    for i in range(frames + pool):
                        if i < frames:
                            inflight.append((time.perf_counter(),
                                             async_m.submit(*pairs[i % len(pairs)])))
                        if len(inflight) > pool or i >= frames:
                            if not inflight:
                                break
                            ts, f = inflight.pop(0)
                            f.result()
                            lat.append((time.perf_counter() - ts) * 1000.0)
                    el = time.perf_counter() - t0
                    async_m.close()
                    fps = frames / el if el else float("inf")
                    total = 1000.0 / fps
                    lat_med = float(np.median(lat))

                verdict = "OK" if fps >= target_fps else "TOO SLOW"
                min_z = m.min_z_mm
                stage_sum = float(np.median(rect) + np.median(match) +
                                  np.median(mmt) + np.median(warp))
                print(f"{label:<14}{ds:>3}{pool:>5}"
                      f"{np.median(rect):>7.2f}{np.median(match):>7.2f}"
                      f"{np.median(mmt):>6.2f}{np.median(warp):>7.2f}"
                      f"{stage_sum:>7.2f}"
                      f"{total:>8.2f}{fps:>8.1f}{lat_med:>8.1f}"
                      f"  {min_z:>6.0f}mm  {verdict}")
                rows.append({
                    "stage_sum_ms": stage_sum,
                    "case": label, "mono_size": list(size), "num_disparities": nd,
                    "downscale": ds, "pool": pool,
                    "rect_ms": float(np.median(rect)),
                    "match_ms": float(np.median(match)),
                    "mm_ms": float(np.median(mmt)),
                    "warp_ms": float(np.median(warp)),
                    "total_ms": total, "fps": fps, "latency_ms": lat_med,
                    "min_z_mm": min_z, "k_mm_px": m.k_mm_px,
                    "meets_target": bool(fps >= target_fps),
                    "cv_threads": cv2.getNumThreads(),
                })
    print()
    ok = [r for r in rows if r["meets_target"]]
    if ok:
        def _say(tag: str, r: dict) -> None:
            print(f"  {tag}: {r['case'].strip()} downscale={r['downscale']} "
                  f"pool={r['pool']} -> {r['fps']:.1f} fps, "
                  f"{r['latency_ms']:.0f} ms latency, MinZ {r['min_z_mm']:.0f} mm")

        biggest = max(ok, key=lambda r: (r["mono_size"][0] * r["mono_size"][1],
                                         r["num_disparities"], -r["latency_ms"]))
        lowest_lat = min(ok, key=lambda r: r["latency_ms"])
        print(f"  configurations holding {target_fps:.0f} fps on this machine:")
        _say("highest resolution", biggest)
        _say("lowest latency    ", lowest_lat)
        if biggest is not lowest_lat and biggest["latency_ms"] > 100.0:
            print(f"  These are not the same choice. A frame pool buys throughput "
                  f"by running several\n  frames at once, so latency grows with "
                  f"pool depth: {biggest['latency_ms']:.0f} ms is fine for "
                  f"offline\n  collection and unusable for teleoperation. Pick "
                  f"against the loop you actually have.")
    else:
        print(f"  nothing here holds {target_fps:.0f} fps on this machine.")
    print("  NOTE fps is compute only. The wire is the other half: a 640x400 raw "
          "gray8 pair at 20 fps is 81.9 Mbit/s, which with colour lands at "
          "88.6 against a measured ~91.5 Mbit/s PoE ceiling.")
    return rows


# =============================================================================
# CLI
# =============================================================================
def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Host-side stereo matching for the C3. Opens no camera.")
    ap.add_argument("--benchmark", action="store_true",
                    help="time the real pipeline on synthetic pairs")
    ap.add_argument("--frames", type=int, default=12,
                    help="frames timed per case (default 12)")
    ap.add_argument("--threads", type=int, default=None,
                    help="cv2.setNumThreads value (default: leave OpenCV's)")
    ap.add_argument("--workers", default="1",
                    help="comma-separated frame-pool depths, e.g. 1,2,4")
    ap.add_argument("--downscale", default="1",
                    help="comma-separated downscale factors, e.g. 1,2")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated WxH:nd cases, e.g. 640x400:96,1280x800:192")
    ap.add_argument("--no-color-warp", action="store_true",
                    help="skip the colour-alignment stage")
    ap.add_argument("--target-fps", type=float, default=20.0)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the rows to this file")
    ap.add_argument("--describe", type=Path, default=None,
                    help="build a rig from a calibration.json and print it")
    args = ap.parse_args(argv)

    if args.describe is not None:
        try:
            rig = StereoRig.from_calibration_json(args.describe)
        except HostDepthError as e:
            print(f"REFUSED: {type(e).__name__}: {e}")
            return 1
        print(json.dumps(rig.describe(), indent=2, default=str))
        return 0

    if not args.benchmark:
        ap.print_help()
        return 0

    cases = BENCH_CASES
    if args.sizes:
        cases = []
        for tok in args.sizes.split(","):
            size_s, _, nd_s = tok.partition(":")
            size = _parse_size(size_s)
            nd = int(nd_s) if nd_s else 96
            cases.append((f"{size[0]}x{size[1]} nd={nd}", size, nd))

    rows = run_benchmark(
        frames=args.frames, threads=args.threads,
        workers=[int(x) for x in args.workers.split(",")],
        downscales=[int(x) for x in args.downscale.split(",")],
        align_to_color=not args.no_color_warp,
        cases=cases, target_fps=args.target_fps)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
