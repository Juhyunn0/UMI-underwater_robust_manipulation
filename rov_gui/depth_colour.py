#!/usr/bin/env python3
"""
depth_colour.py — THE depth colour rule, and the only implementation of it.

This module exists so the LIVE station and the OFFLINE comparison tool paint a
millimetre with the same colour. It was extracted verbatim from
``rov_gui/tools/depth_compare.py`` on 2026-09-06, when the FoundationStereo
panel adopted the rule (until then the panel was scene-adaptive: JET, a
per-frame 2/98 percentile range and a per-frame 64-knot histogram equalisation,
so the same distance was a different colour in every frame — and laying a
screenshot of it beside a land depth video measured the two renderers rather
than the two depths).

It deliberately depends on nothing but numpy and cv2, so importing it costs the
GUI process no zarr, no torch and no decoder machinery. ``depth_compare``
re-exports every name below, so both spellings keep working.

The rule, in one place:

    domain "obs"     palette position = (1/z - 1/z_far) / (1/z_near - 1/z_far)
                     which is EXACTLY umi_handheld.build_zarr.normalise_depth,
                     i.e. what the policy's channel 0 already holds
    domain "metric"  palette position = 1 - (z - z_near) / (z_far - z_near)
    endpoints        z_near 0.20 m, z_far 3.00 m, FIXED, never from the frame
    polarity         warm = NEAR in both domains
    colormap         TURBO by default
    invalid          BLACK, and distinguishable from z_far (which is also 0)
"""

from __future__ import annotations

import os

import numpy as np


def _cv2():
    """cv2, imported through the station's guard when there is one.

    ``rov_gui.qt.import_cv2`` restores QT_QPA_PLATFORM_PLUGIN_PATH afterwards;
    importing cv2 plainly inside a Qt process repoints it and the xcb platform
    plugin then fails to load. Offline callers (the fstereo env has no PyQt)
    fall through to the plain import.
    """
    try:
        from rov_gui.qt import import_cv2
        return import_cv2()
    except Exception:
        saved = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
        try:
            import cv2
        finally:
            if saved is None:
                os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            else:
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = saved
        return cv2


#: The obs endpoints. NOT a guess and NOT a tunable: this is what
#: ``umi_handheld/build_dp_depth_zarr.py`` froze into the training store and
#: what ``rov_gui/perception/policy_obs.py`` rebuilds live. Both are read back
#: from the artifacts at run time and asserted against these
#: [측정: /home/bdml/Desktop/data collection/dataset_depth.zarr.zip
#: .zattrs['obs'] -> z_near_m 0.2, z_far_m 3.0].
Z_NEAR_M = 0.20
Z_FAR_M = 3.00

#: Colormaps offered. TURBO is the default because it is perceptually ordered
#: and is already the convention in three of the four legacy renderers; JET is
#: kept only so a picture can be put beside an OLD station screenshot.
#: VIRIDIS is there for anyone who has to read the result in greyscale print.
CMAPS = {"turbo": "COLORMAP_TURBO", "jet": "COLORMAP_JET",
         "viridis": "COLORMAP_VIRIDIS", "magma": "COLORMAP_MAGMA",
         "inferno": "COLORMAP_INFERNO"}

#: What every caller uses unless it says otherwise. Changing this changes what
#: "the same colour rule" means on both sides at once, which is the only way it
#: may ever be changed.
DEFAULT_CMAP = "turbo"

#: Palette position of a distance, in each domain. 1.0 = NEAR in both, so the
#: two domains at least agree on which end is warm even though they disagree on
#: where the middle sits. This is the ONE formula; the colour bar ticks are
#: placed by inverting it, so the legend can never state a distance the picture
#: does not mean (the failure rov_gui/imaging.py:160-166 warns about).
def palette_pos(z_m: np.ndarray, domain: str,
                z_near: float = Z_NEAR_M, z_far: float = Z_FAR_M) -> np.ndarray:
    """Distance in METRES -> palette position in [0, 1], 1 = nearest.

    ``obs``    matches ``umi_handheld.build_zarr.normalise_depth`` exactly:
               (1/z - 1/z_far) / (1/z_near - 1/z_far).
    ``metric`` is the plain linear ramp 1 - (z - z_near)/(z_far - z_near).
    """
    z = np.asarray(z_m, dtype=np.float32)
    if domain == "obs":
        inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (1.0 / np.maximum(z, 1e-6) - inv_far) / (inv_near - inv_far)
        return np.clip(v, 0.0, 1.0)
    if domain == "metric":
        return np.clip(1.0 - (z - z_near) / max(z_far - z_near, 1e-6), 0.0, 1.0)
    raise ValueError(f"unknown domain {domain!r}")


def palette_pos_inverse(t: float, domain: str,
                        z_near: float = Z_NEAR_M,
                        z_far: float = Z_FAR_M) -> float:
    """Palette position -> metres. The colour bar's tick placer."""
    t = float(np.clip(t, 0.0, 1.0))
    if domain == "obs":
        inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
        return 1.0 / (t * (inv_near - inv_far) + inv_far)
    return z_near + (1.0 - t) * (z_far - z_near)


def _cv2():
    try:
        from rov_gui.qt import import_cv2
        return import_cv2()
    except Exception:
        import cv2
        return cv2


def colorize(t: np.ndarray, valid: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """Palette position (+ a validity mask) -> BGR. Invalid is BLACK.

    Black for "no measurement" and NOT for "far": at 3.0 m the obs value is 0
    too, so without the mask the two are the same pixel. The land store's own
    attrs call this out ("channel 0 is 0.0 at invalid AND at z_far; channel 1
    disambiguates"), and it is the single most misreadable thing about these
    pictures.
    """
    cv2 = _cv2()
    name = CMAPS.get(cmap)
    if name is None:
        raise ValueError(f"unknown cmap {cmap!r}; have {sorted(CMAPS)}")
    idx = np.clip(np.asarray(t, np.float32), 0.0, 1.0) * 255.0
    out = cv2.applyColorMap(idx.astype(np.uint8), getattr(cv2, name))
    out[~np.asarray(valid, bool)] = (0, 0, 0)
    return out


# =============================================================================
# the two sources, each reduced to (palette position, validity)
# =============================================================================

def obs_to_t_valid(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The stored/live obs tensor -> (palette position in the OBS domain, valid).

    ``obs`` is (H, W, 3) uint8 packed [inverse_depth, validity, inverse_depth].
    Channel 0 IS already the obs-domain palette position, scaled to 0..255 —
    that is what makes this domain apples-to-apples: nothing is recomputed, the
    picture is the tensor.
    """
    obs = np.asarray(obs)
    if obs.ndim != 3 or obs.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) obs, got {obs.shape}")
    t = obs[..., 0].astype(np.float32) / 255.0
    valid = obs[..., 1] > 0
    return t, valid


def obs_to_metres(obs: np.ndarray, z_near: float = Z_NEAR_M,
                  z_far: float = Z_FAR_M) -> np.ndarray:
    """The obs tensor -> METRES (NaN where invalid).

    The inverse of ``normalise_depth``. Quantisation is coarse near z_far and
    fine near z_near — one u8 step is ~2 mm at 0.25 m and ~200 mm at 2.8 m —
    which is a property of the representation the policy was trained on, not of
    this tool. Reported by ``stats`` so it is never mistaken for measurement
    noise.
    """
    t, valid = obs_to_t_valid(obs)
    inv_near, inv_far = 1.0 / z_near, 1.0 / z_far
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / (t * (inv_near - inv_far) + inv_far)
    z = z.astype(np.float32)
    z[~valid] = np.nan
    return z


def mm_to_t_valid(depth_mm: np.ndarray, domain: str,
                  z_near: float = Z_NEAR_M,
                  z_far: float = Z_FAR_M) -> tuple[np.ndarray, np.ndarray]:
    """uint16 millimetres (0 = no measurement) -> (palette position, valid)."""
    d = np.asarray(depth_mm)
    valid = d > 0
    z = d.astype(np.float32) / 1000.0
    t = palette_pos(np.where(valid, z, z_far), domain, z_near, z_far)
    return t, valid


def colour_rule_json(domain: str, cmap: str, z_near: float, z_far: float) -> dict:
    """The colour rule as data, written beside every output.

    A picture is only comparable if the rule that made it is recoverable, so
    this goes on disk with the same numbers that were burned into the frame.
    """
    return {
        "schema": "rov_gui/depth_compare/colour_rule/1",
        "domain": domain,
        "colormap": CMAPS[cmap],
        "warm_end": "near",
        "z_near_m": z_near,
        "z_far_m": z_far,
        "invalid": "black (validity channel 0 / depth == 0); NOT the same as z_far",
        "per_frame_adaptation": "none — no auto-range, no histogram equalisation",
        "palette_pos": ("(1/z - 1/z_far)/(1/z_near - 1/z_far)" if domain == "obs"
                        else "1 - (z - z_near)/(z_far - z_near)"),
        "obs_domain_note": (
            "the obs domain IS the stored tensor's channel 0 / 255 — nothing is "
            "recomputed, so land and water pixels of equal value mean equal "
            "distance by construction "
            "(umi_handheld/build_dp_depth_zarr.py vs rov_gui/perception/policy_obs.py)"),
        "written_by": "rov_gui/tools/depth_compare.py",
    }


# =============================================================================
# the live path: millimetres straight to a picture
# =============================================================================

def depth_mm_to_bgr(depth_mm: np.ndarray, domain: str = "obs",
                    cmap: str = "cmap_default",
                    z_near: float = Z_NEAR_M, z_far: float = Z_FAR_M
                    ) -> np.ndarray:
    """uint16/float millimetres (0 = no measurement) -> BGR, by THE rule.

    The station's depth panel and the land episode videos both go through this,
    so a colour on one screen means the same distance on the other. Nothing here
    looks at the frame's own content: no auto range, no equalisation, no
    percentile. That is the entire point — an adaptive picture cannot be
    compared with anything, including itself one frame later.
    """
    if cmap == "cmap_default":
        cmap = DEFAULT_CMAP
    d = np.asarray(depth_mm)
    if d.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got {d.shape}")
    t, valid = mm_to_t_valid(d, domain, z_near, z_far)
    return colorize(t, valid, cmap)


def rule_text(domain: str, cmap: str, z_near: float = Z_NEAR_M,
              z_far: float = Z_FAR_M) -> str:
    """The rule as a one-line label, for burning into a HUD or a frame."""
    return (f"{domain} | {cmap.upper()} | warm=NEAR | "
            f"{z_near:.2f}-{z_far:.2f} m | black = no measurement")
