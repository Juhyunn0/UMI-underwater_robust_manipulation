#!/usr/bin/env python3
"""
viz.py — depth colourisation and the on-screen HUD.

Display choices worth knowing about:

* Depth is colourised over an explicit millimetre range, not auto-scaled per
  frame. Auto-scaling makes a pretty picture that lies: the same colour means a
  different distance every frame, so you cannot judge stability by eye. Invalid
  pixels (0 mm) render black so holes stay visibly holes.
* TURBO rather than JET — perceptually monotonic, so ordering in the image
  matches ordering in depth.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def colorize_depth(depth_mm: np.ndarray, min_mm: float = 300.0,
                   max_mm: float = 6000.0,
                   colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """uint16 millimetre depth -> BGR image, invalid pixels black."""
    if depth_mm.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got {depth_mm.shape}")
    invalid = depth_mm == 0
    span = max(1.0, float(max_mm - min_mm))
    norm = np.clip((depth_mm.astype(np.float32) - min_mm) / span, 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    out[invalid] = (0, 0, 0)
    return out


def depth_stats(depth_mm: np.ndarray) -> dict:
    """Coverage and range of a depth map — the numbers that show whether the
    stereo matcher is actually working, as opposed to merely producing output."""
    valid = depth_mm[depth_mm > 0]
    total = depth_mm.size
    if valid.size == 0:
        return {"valid_pct": 0.0, "min_mm": 0, "median_mm": 0, "max_mm": 0}
    return {
        "valid_pct": 100.0 * valid.size / total,
        "min_mm": int(valid.min()),
        "median_mm": int(np.median(valid)),
        "max_mm": int(valid.max()),
    }


def draw_hud(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
             scale: float = 0.45, colour=(255, 255, 255),
             bg_alpha: float = 0.55) -> np.ndarray:
    """Monospace-ish text block over a translucent panel. Edits `img` in place."""
    if not lines:
        return img
    thickness = 1
    line_h = int(22 * max(scale / 0.45, 1.0))
    pad = 6
    widths = [cv2.getTextSize(t, FONT, scale, thickness)[0][0] for t in lines]
    w = max(widths) + 2 * pad
    h = line_h * len(lines) + 2 * pad
    x0, y0 = origin
    x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])

    if bg_alpha > 0:
        panel = img[y0:y1, x0:x1]
        if panel.size:
            dark = np.zeros_like(panel)
            img[y0:y1, x0:x1] = cv2.addWeighted(panel, 1 - bg_alpha, dark, bg_alpha, 0)

    y = y0 + pad + int(line_h * 0.75)
    for text in lines:
        cv2.putText(img, text, (x0 + pad, y), FONT, scale, (0, 0, 0), thickness + 2,
                    cv2.LINE_AA)
        cv2.putText(img, text, (x0 + pad, y), FONT, scale, colour, thickness,
                    cv2.LINE_AA)
        y += line_h
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Small caption in the bottom-left corner."""
    cv2.putText(img, text, (8, img.shape[0] - 8), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, img.shape[0] - 8), FONT, 0.5, (60, 255, 60), 1,
                cv2.LINE_AA)
    return img


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    """Scale to a common height, preserving aspect ratio."""
    if img.shape[0] == height:
        return img
    w = max(1, int(round(img.shape[1] * height / img.shape[0])))
    interp = cv2.INTER_AREA if img.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(img, (w, height), interpolation=interp)


def hstack_pad(images: list[np.ndarray], height: int | None = None) -> np.ndarray:
    """Side-by-side mosaic of differently sized frames."""
    images = [im for im in images if im is not None and im.size]
    if not images:
        return np.zeros((120, 320, 3), np.uint8)
    h = height or max(im.shape[0] for im in images)
    parts = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        parts.append(fit_height(im, h))
    return np.hstack(parts)


def overlay_depth_on_color(color_bgr: np.ndarray, depth_mm: np.ndarray,
                           alpha: float = 0.45, min_mm: float = 300.0,
                           max_mm: float = 6000.0) -> np.ndarray:
    """Blend colourised depth over colour — the quickest visual check that
    `--depth-align color` really registered the two."""
    dc = colorize_depth(depth_mm, min_mm, max_mm)
    if dc.shape[:2] != color_bgr.shape[:2]:
        dc = cv2.resize(dc, (color_bgr.shape[1], color_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(color_bgr, 1 - alpha, dc, alpha, 0)
