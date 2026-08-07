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


def legend_labels(min_mm: float = DEPTH_MIN_MM,
                  max_mm: float = DEPTH_MAX_MM) -> tuple[str, str]:
    return (f"{min_mm / 1000:.1f} m", f"{max_mm / 1000:.0f} m")
