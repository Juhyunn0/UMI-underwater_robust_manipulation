#!/usr/bin/env python3
"""
imaging.py — numpy frame -> QImage, and the depth colour map.

Everything in here runs on a *worker* thread, deliberately. Colour conversion,
rescaling and colourising are the expensive per-frame operations, and the GUI
thread's job is to blit an image that is already the right size and format.

The one rule that matters
-------------------------
``QImage(buffer, ...)`` does not copy — it wraps whoever's memory you handed it.
The numpy array behind a camera frame is reused for the next frame (DepthAI
recycles its pool) or freed when the local goes out of scope, and the result is
tearing, garbage, or a segfault in ``paintEvent`` — always some time later, and
never in the code that caused it. :func:`bgr_to_qimage` therefore always returns
a detached ``.copy()``. If you are tempted to save that copy, don't: it is one
memcpy of a frame you already spent milliseconds decoding.
"""

from __future__ import annotations

import numpy as np

from .depth_colour import (DEFAULT_CMAP, Z_FAR_M, Z_NEAR_M, depth_mm_to_bgr,
                           palette_pos)
from .qt import QImage, import_cv2

# Depth colourisation range in millimetres. Matches the convention in
# c3_camera/viz.py: a FIXED range, so a colour means the same distance in every
# frame, and 0 (no return) renders black so holes stay visibly holes.
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 6000.0

# Below this much shrinkage, resizing on the worker costs more latency than the
# bandwidth it saves. See scale_to_fit.
NO_RESIZE_ABOVE = 0.85


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """HxWx3 uint8 BGR (or HxW mono) -> a QImage that owns its pixels."""
    if bgr.ndim == 2:
        bgr = np.stack([bgr] * 3, axis=-1)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(bgr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_BGR888)
    return img.copy()          # detach: see the module docstring


def scale_to_fit(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Shrink to fit a panel, preserving aspect. Never enlarges.

    Enlarging here would push a bigger image across the thread boundary for no
    detail gain; the panel can upscale in ``drawImage`` for free. Shrinking is
    the case worth doing off the GUI thread — INTER_AREA on 960x540 is real
    work, and it is work the GUI thread must not be doing.
    """
    if target_w <= 0 or target_h <= 0:
        return bgr
    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    if scale >= NO_RESIZE_ABOVE:
        # Near-unity resizes are the worst deal in the whole pipeline: they
        # touch every pixel of a full-size output for a few percent of size,
        # and INTER_AREA is at its most expensive there because it integrates
        # fractional source areas. Measured on the real rig, a 960x540 -> 933x525
        # resize on the video thread cost the COLOUR feed ~25 ms of latency
        # (57.7 ms with it, 32.2 ms without). Let QPainter do the last few
        # percent while it is blitting anyway.
        return bgr
    cv2 = import_cv2()
    # INTER_AREA only earns its cost when shrinking a lot; below that the
    # aliasing it prevents is not visible and INTER_LINEAR is several times
    # cheaper.
    interp = cv2.INTER_AREA if scale < 0.5 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=interp)


def scale_depth(depth_mm: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Shrink a uint16 depth map BEFORE colourising it, nearest-neighbour.

    Two reasons, and the second one is not an optimisation:

    1. Colourising is per-pixel work on the video thread. Doing it at 640x400
       and then shrinking costs 4x what shrinking to a ~320x180 panel first
       does — measured as ~24 ms of added latency on the *colour* feed, because
       both share one worker thread.
    2. Nearest-neighbour is the only correct way to downscale depth. Averaging
       two pixels that straddle an edge produces a distance at which nothing
       exists — a floating halo around every object. Interpolating a colourised
       depth image is also wrong, but it at least fails visibly; interpolating
       the millimetres themselves fails invisibly.
    """
    if target_w <= 0 or target_h <= 0:
        return depth_mm
    h, w = depth_mm.shape[:2]
    scale = min(target_w / w, target_h / h)
    if scale >= 1.0:
        return depth_mm
    cv2 = import_cv2()
    return cv2.resize(depth_mm, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_NEAREST)


def depth_to_bgr(depth_mm: np.ndarray, min_mm: float = DEPTH_MIN_MM,
                 max_mm: float = DEPTH_MAX_MM) -> np.ndarray:
    """uint16 millimetre depth -> BGR, invalid (0) black, TURBO colour map.

    Deliberately a copy of ``c3_camera.viz.colorize_depth`` rather than an
    import of it. Importing anything from ``c3_camera`` runs that package's
    ``__init__``, which *requires depthai 2.x to be installed and refuses 3.x* —
    correct for the camera tools, fatal for a GUI that must also run on a
    laptop with no depthai at all. The convention (fixed range, TURBO, black
    holes) is kept identical on purpose, so the two views of the same depth map
    are comparable.
    """
    cv2 = import_cv2()
    if depth_mm.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got {depth_mm.shape}")
    invalid = depth_mm == 0
    span = max(1.0, float(max_mm - min_mm))
    norm = np.clip((depth_mm.astype(np.float32) - min_mm) / span, 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[invalid] = (0, 0, 0)
    return out


def depth_palette_knots(depth_mm: np.ndarray, steps: int = 64,
                        subsample: int = 30000, seed: int = 0):
    """Ascending millimetre quantiles that histogram-equalise the palette.

    Ported from the reference viewer
    (``~/Desktop/data collection/UMI_Underwater/oakd_viewer.py:139``), which is what
    makes its picture read as detailed where the station's read as flat. Every
    colour band then covers the same NUMBER OF PIXELS rather than the same
    number of millimetres, so a scene spanning 0.4 m to 28 m — one open doorway
    in the corner is enough — no longer spends nine tenths of the ramp on the
    far tenth of the pixels while the whole foreground collapses into one band.

    A linear range cannot fix that by moving its endpoints: the problem is the
    distribution between them, and this is the only knob that addresses it.

    Returns None when nothing is valid (the caller then falls back to a linear
    ramp). The tiny ascending nudge keeps the knots strictly increasing for
    ``np.interp``, which a flat wall filling the view would otherwise tie.
    """
    v = depth_mm[depth_mm > 0].astype(np.float32)
    if v.size == 0:
        return None
    if v.size > subsample:      # quantiles converge long before the full frame
        v = np.random.default_rng(seed).choice(v, subsample, replace=False)
    knots = np.quantile(v, np.linspace(0.0, 1.0, steps))
    return np.maximum.accumulate(knots) + np.arange(steps) * 1e-3


def depth_fraction(depth_mm, knots=None, min_mm: float = DEPTH_MIN_MM,
                   max_mm: float = DEPTH_MAX_MM, domain: str | None = None):
    """Depth -> palette position in [0, 1], 1 = NEAREST. The one formula.

    Colourisation and the colour bar must both go through this or the legend
    states distances the picture does not mean — the failure the legend's own
    docstring warns about. ``knots`` selects the equalised mapping; without
    them it is the plain linear ramp between ``min_mm`` and ``max_mm``.

    ``domain`` ("obs" or "metric") switches to the SHARED rule in
    :mod:`rov_gui.depth_colour` — the one :func:`depth_to_bgr_umi` painted with.
    It wins over ``knots``, which belong to the adaptive picture and cannot
    coexist with a fixed rule; passing both is a caller bug, so it is the fixed
    rule that survives rather than a silent blend of the two.
    """
    if domain:
        return palette_pos(np.asarray(depth_mm, dtype=np.float32) / 1000.0,
                           domain, min_mm / 1000.0, max_mm / 1000.0)
    z = np.asarray(depth_mm, dtype=np.float32)
    if knots is not None and len(knots) > 1:
        return np.interp(z, knots, np.linspace(1.0, 0.0, len(knots)))
    near = float(min_mm)
    far = max(float(max_mm), near + 1.0)
    return 1.0 - (np.clip(z, near, far) - near) / (far - near)


def depth_to_bgr_warm_near(depth_mm: np.ndarray, knots=None,
                           min_mm: float = DEPTH_MIN_MM,
                           max_mm: float = DEPTH_MAX_MM) -> np.ndarray:
    """uint16 mm -> BGR with WARM = NEAR, invalid black, JET.

    The reference viewer's convention, adopted for the learned-depth panel so
    the two could be compared frame by frame (2026-09-02). It is the opposite of
    :func:`depth_to_bgr`, which is TURBO with warm = FAR — deliberately left
    alone, because it is the device stereo panel's convention and a run uses
    exactly one depth instrument, so the two never share a screen.

    OPT-IN since 2026-09-06 (``--fstereo-palette adaptive``). The panel's
    default is :func:`depth_to_bgr_umi`, the shared fixed rule, because an
    adaptive picture cannot be compared with a land video — or with itself one
    frame earlier. This one is still the better read of a single flat scene,
    which is why it is kept rather than deleted.
    """
    cv2 = import_cv2()
    if depth_mm.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got {depth_mm.shape}")
    t = depth_fraction(depth_mm, knots, min_mm, max_mm)
    out = cv2.applyColorMap((np.clip(t, 0.0, 1.0) * 255).astype(np.uint8),
                            cv2.COLORMAP_JET)
    out[depth_mm == 0] = (0, 0, 0)
    return out


#: The SHARED rule's endpoints in millimetres, so the panel's stat can carry
#: them without importing metres into a file that speaks millimetres. They are
#: the training store's own endpoints (rov_gui/depth_colour.py), not a taste.
UMI_Z_NEAR_MM = Z_NEAR_M * 1000.0
UMI_Z_FAR_MM = Z_FAR_M * 1000.0


def depth_to_bgr_umi(depth_mm: np.ndarray, domain: str = "obs",
                     cmap: str | None = None) -> np.ndarray:
    """uint16 mm -> BGR by THE shared rule: TURBO, warm = NEAR, fixed 0.2-3.0 m.

    One line of delegation on purpose. The rule lives in
    :mod:`rov_gui.depth_colour` because the offline comparison tool and the land
    episode videos (``data_collection/make_depth_trajectory_video.py``) paint with
    the very same function, and a copy here would be a second implementation of
    the one thing that must not have two.

    Nothing about it looks at the frame: no auto range, no equalisation. That is
    what makes a station screenshot comparable with a land video, and with the
    same station a minute earlier — the property the adaptive palette
    (:func:`depth_to_bgr_warm_near`) trades away for contrast.
    """
    return depth_mm_to_bgr(depth_mm, domain, cmap or DEFAULT_CMAP)


def auto_depth_range(depth_mm: np.ndarray, lo_pct: float = 2.0,
                     hi_pct: float = 98.0,
                     fallback: tuple[float, float] = (DEPTH_MIN_MM,
                                                      DEPTH_MAX_MM),
                     min_span_mm: float = 100.0) -> tuple[float, float]:
    """Percentile depth range in MILLIMETRES over the valid pixels.

    The fixed 0.3-6 m range wastes the palette whenever the scene does not
    happen to fill it, and a manipulation scene never does: with everything
    inside ~1.5 m the whole picture lands in the bottom fifth of TURBO, which
    reads as "dark blue and black" — indistinguishable at a glance from the
    holes it is drawn next to. That is what made the station's depth panel
    look dirty beside the reference viewer
    (``~/Desktop/data collection/UMI_Underwater/oakd_viewer.py:194``), which has always
    auto-ranged; the maps themselves were comparable.

    Percentiles rather than min/max so a handful of stray pixels — one hot
    speckle at 8 m, one at the lens — cannot collapse the span again.

    Returns ``fallback`` when nothing is valid, so a dead frame keeps drawing
    on the same scale as a live one instead of inventing contrast from noise.
    """
    v = depth_mm[depth_mm > 0]
    if v.size == 0:
        return fallback
    lo, hi = (float(x) for x in np.percentile(v.astype(np.float32),
                                             [lo_pct, hi_pct]))
    if hi - lo < min_span_mm:            # near-flat scene; keep a usable span
        hi = lo + min_span_mm
    return lo, hi


def legend_labels(min_mm: float = DEPTH_MIN_MM,
                  max_mm: float = DEPTH_MAX_MM) -> tuple[str, str]:
    lo = f"{min_mm / 1000:.1f}"
    hi = f"{max_mm / 1000:.1f}" if max_mm < 10000 else f"{max_mm / 1000:.0f}"
    return (f"{lo} m", f"{hi} m")
