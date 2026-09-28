#!/usr/bin/env python3
"""
depth_panel.py — one depth frame as a labelled 640x480 picture.

Extracted from ``data_collection/make_depth_trajectory_video.py`` on 2026-09-06,
verbatim, when a SECOND offline video wanted the same panel
(``rov_gui/tools/render_run_scene.py``, the station-run scene). Sharing it is
the point: a land episode video and a water run video have to put the same
distance on the same colour AND in the same layout, or the eye compares the
two renderers instead of the two depths.

The colour rule itself is :mod:`rov_gui.depth_colour` — this module only draws
it: header text, the image at its native size where it fits, and the horizontal
colour bar whose ticks are placed by inverting the very function that coloured
the pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .depth_colour import colorize, palette_pos

PANEL_W, PANEL_H = 640, 480          # one panel; two of these plus a camera view
HEADER_H = 46                        # depth panel: text above the image
BAR_H = 34                           # depth panel: colour bar + tick labels below it
IMAGE_H = PANEL_H - HEADER_H - BAR_H  # == 400, exactly the fs depth's own height
BG = (24, 24, 24)


# ------------------------------------------------------------------------- drawing

def text(img, lines, org=(8, 18), scale=0.40, colour=(235, 235, 235), gap=14,
         outline=True):
    """Outlined text, readable over the depth image as well as over the flat header.

    The outline is four 1-px offset copies, NOT one thick black pass under a thin fill:
    OpenCV 5.0 advances glyphs further at thickness 3 than at thickness 1 (435 px vs 408 px
    for the same 79-character string at scale 0.40), so the usual thick-then-thin pair
    draws the outline progressively further right than the fill and leaves a legible ghost
    of the line's tail hanging off the end of every long label.
    """
    for k, line in enumerate(lines):
        x, y = org[0], org[1] + k * gap
        if outline:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv2.putText(img, line, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                    cv2.LINE_AA)


def colour_bar_h(width, domain, cmap, z_near, z_far, ticks_m=None):
    """The legend, horizontal, near on the LEFT.

    Tick positions come from `palette_pos` -- the same function that coloured the pixels --
    so a tick cannot claim a distance the picture does not mean. Only the layout differs
    from depth_compare.colour_bar; the formula does not.
    """
    bar_h = 13
    panel = np.zeros((BAR_H, width, 3), np.uint8)
    panel[:] = BG
    ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :]
    panel[0:bar_h] = colorize(np.repeat(ramp, bar_h, axis=0),
                              np.ones((bar_h, width), bool), cmap)
    if ticks_m is None:
        ticks_m = [z_near, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, z_far]
    last_x = -1e9
    for z in ticks_m:
        if not (z_near - 1e-9 <= z <= z_far + 1e-9):
            continue
        t = float(palette_pos(np.array([z], np.float32), domain, z_near, z_far)[0])
        x = int(round((1.0 - t) * (width - 1)))
        cv2.line(panel, (x, 0), (x, bar_h - 1), (255, 255, 255), 1)
        label = f"{z:.2f}"
        lw = 7 * len(label)
        if x - lw // 2 - last_x < 6:      # would collide with the previous label
            continue
        last_x = x + lw // 2
        text(panel, [label], org=(max(0, min(width - lw, x - lw // 2)), bar_h + 12),
             scale=0.34)
    # No "near"/"far" words: the header states warm=NEAR, and a word painted on the ramp
    # is illegible on whichever palette end happens to be light.
    return panel


def fit_into(img, w, h, interp=cv2.INTER_NEAREST):
    """Scale to fit (w, h) keeping aspect, centred on the background. Never crops."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = BG
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas


def rule_line(domain, cmap, z_near, z_far):
    """The colour rule, in words, burned into every frame -- the whole point of importing
    it from depth_compare rather than inventing one here."""
    return (f"{domain} | {cmap.upper()} | warm=NEAR | {z_near:.2f}-{z_far:.2f} m | "
            f"black = no measurement (not far)")


def depth_panel(t, valid, *, domain, cmap, z_near, z_far, lines, note=""):
    """One depth frame -> a 640x480 panel with its header and its legend."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    panel[:] = BG
    body = colorize(t, valid, cmap)
    panel[HEADER_H:HEADER_H + IMAGE_H] = fit_into(body, PANEL_W, IMAGE_H)
    panel[PANEL_H - BAR_H:] = colour_bar_h(PANEL_W, domain, cmap, z_near, z_far)
    text(panel, lines[:2], org=(8, 13), scale=0.38, gap=14, outline=False)
    text(panel, [rule_line(domain, cmap, z_near, z_far)], org=(8, 41), scale=0.36,
         colour=(170, 200, 255), outline=False)
    if note:
        text(panel, [note], org=(8, HEADER_H + IMAGE_H - 6), scale=0.40,
             colour=(120, 160, 255))
    return panel


def missing_panel(message, *, domain, cmap, z_near, z_far, lines):
    panel = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
    panel[:] = BG
    panel[PANEL_H - BAR_H:] = colour_bar_h(PANEL_W, domain, cmap, z_near, z_far)
    text(panel, lines[:2], org=(8, 13), scale=0.38, gap=14, outline=False)
    text(panel, [rule_line(domain, cmap, z_near, z_far)], org=(8, 41), scale=0.36,
         colour=(170, 200, 255), outline=False)
    text(panel, [message], org=(8, HEADER_H + IMAGE_H // 2), scale=0.5,
         colour=(120, 160, 255))
    return panel
