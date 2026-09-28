#!/usr/bin/env python3
"""make_depth_trajectory_html.py — trajectory_depth.mp4, in a page that answers "how far?".

    python data_collection/make_depth_trajectory_html.py "/home/bdml/Desktop/data collection" \
        --episodes 13

Writes `<dataset>/slam/<ep>/trajectory_depth.html`: the SAME three-panel picture as
`trajectory_depth.mp4` -- the mp4 itself, embedded, not a re-render -- plus one thing the
mp4 cannot do, which is tell you the distance under the cursor.

Why the mp4 is embedded rather than redrawn
-------------------------------------------
The point of `make_depth_trajectory_video.py` is that one colour rule paints every picture
of this depth; a second renderer here (a canvas, a JS colormap, a different letterbox) would
put the same millimetre on a different pixel and the page would quietly measure the two
renderers instead of the depth. So the page plays the reviewed mp4 and only ADDS a readout
on top of it. Everything about the layout the readout needs -- which third of the frame is
the depth panel, where the image sits inside it, what the header rows cost -- is imported
from `rov_gui/depth_panel.py`, the module the video's own layout came from.

Where the numbers come from
---------------------------
`trajectory_depth.colour_rule.json`, written beside the mp4 by the video tool, names the
depth source, the domain and the endpoints that video was built with; this tool reads them
back rather than re-deciding, so an HTML page cannot describe a ramp its video does not use.
The distances themselves are read out of that same artifact (`depth/<ep>/depth.zarr` for
`fs`), through `FsDepth.metres` / `ObsDepth.metres` -- the same frame -> row mapping the
video used, so the number quoted for a frame comes from the row that frame was painted from.

Size
----
The depth has to travel INSIDE the html: a browser opened on a `file://` page refuses to
`fetch` a sibling file, and an `<img>` loaded from one taints the canvas so its pixels
cannot be read back. So one PNG per frame goes in as a data URI -- 640x400 uint16 mm, split
into a high-byte and a low-byte plane stacked vertically because PNG's row filters do far
better on a smooth plane and a noisy plane kept apart than on the two interleaved. That is
~66 KB a frame at full resolution, i.e. ~12 MB for a 132-frame episode. `--depth-scale 2`
(2x2 block median) and `--quantise-pct` (log-spaced codes, e.g. 1.0 for +-0.5%) are the two
levers if that is too much; both are stated on the page so a reader knows what the readout's
precision actually is.

Run it in the env that produced the depth (numpy, cv2, zarr, av):

    ~/miniforge3/envs/fstereo/bin/python data_collection/make_depth_trajectory_html.py ...
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rov_gui.depth_panel import HEADER_H, IMAGE_H, PANEL_H, PANEL_W  # noqa: E402

from data_collection.make_depth_trajectory_video import (  # noqa: E402
    FsDepth, ObsDepth, episodes_on_disk, parse_episode_spec, probe)

HTML_NAME = "trajectory_depth.html"


# ------------------------------------------------------------------- depth -> tiles

def block_median(z: np.ndarray, k: int) -> np.ndarray:
    """k x k block median of the MEASURED values; NaN where a block has none.

    Median and not mean: a block straddling a depth edge has two populations in it, and
    the mean of them is a distance nothing in the scene is at.
    """
    if k == 1:
        return z
    h, w = z.shape
    H, W = h // k, w // k
    b = z[:H * k, :W * k].reshape(H, k, W, k).transpose(0, 2, 1, 3).reshape(H, W, k * k)
    with warnings.catch_warnings():                 # all-NaN block -> NaN, on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(b, axis=2)


def encode_codes(code: np.ndarray) -> str:
    """uint16 codes -> base64 PNG, high-byte plane above low-byte plane."""
    plane = np.vstack([(code >> 8).astype(np.uint8), (code & 0xFF).astype(np.uint8)])
    ok, buf = cv2.imencode(".png", plane, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Quantiser:
    """metres -> uint16 code, 0 = no measurement. Two schemes, both exactly invertible.

    ``mm``  code = round(z * 1000), i.e. the stored value, no loss at all.
    ``log`` code = round(log(z/z0)/log(base)) + 1, a constant RELATIVE step, which is the
            shape of stereo error itself -- and which compresses to about a quarter of the
            exact millimetres because the low byte stops being noise.
    """

    Z0_MM = 50.0

    def __init__(self, pct: float):
        self.pct = float(pct)
        self.log = self.pct > 0
        self.base = 1.0 + self.pct / 100.0 if self.log else 1.0

    def encode(self, z_m: np.ndarray) -> np.ndarray:
        mm = np.asarray(z_m, np.float64) * 1000.0
        good = np.isfinite(mm) & (mm > 0)
        if not self.log:
            code = np.where(good, np.round(mm), 0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.log(np.maximum(mm, self.Z0_MM) / self.Z0_MM) / math.log(self.base)
            code = np.where(good, np.round(c) + 1, 0)
        return np.clip(code, 0, 65535).astype(np.uint16)

    def js(self) -> dict:
        if not self.log:
            return {"mode": "mm", "unit_mm": 1.0,
                    "note": "exact stored millimetres, no quantisation"}
        return {"mode": "log", "z0_mm": self.Z0_MM, "base": self.base,
                "note": f"log-spaced codes, {self.pct:g}% steps "
                        f"(read-out error up to +-{self.pct / 2:g}%)"}


def fit_placement(iw: int, ih: int, w: int, h: int) -> dict:
    """Where `rov_gui.depth_panel.fit_into` put an iw x ih image inside a w x h box."""
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    return {"x0": (w - nw) // 2, "y0": (h - nh) // 2, "nw": nw, "nh": nh,
            "iw": iw, "ih": ih}


# -------------------------------------------------------------------------- the page

def render_html(meta: dict, frame_tile: list[int], tiles: list[str],
                video_b64: str | None, video_src: str) -> str:
    """The page, as one string. Built by concatenation, not by %-formatting the tiles in:
    the tile array is the bulk of a 12 MB file and copying it again is minutes of nothing."""
    head = TEMPLATE_HEAD.replace("__META__", json.dumps(meta, indent=2)) \
                        .replace("__FRAME_TILE__", json.dumps(frame_tile))
    if video_b64 is not None:
        source = f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
    else:
        source = f'<source src="{video_src}" type="video/mp4">'
    parts = [head.replace("__VIDEO_SOURCE__", source), "const TILES = [\n"]
    parts.extend(f'"{t}",\n' for t in tiles)
    parts.append("];\n")
    parts.append(TEMPLATE_TAIL)
    return "".join(parts)


TEMPLATE_HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trajectory_depth</title>
<style>
  :root { --bg:#181818; --fg:#e9e9e9; --dim:#8d8d8d; --accent:#aac8ff; --line:#333; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  main { max-width:1920px; margin:0 auto; padding:14px 16px 28px; }
  h1 { font-size:14px; font-weight:600; margin:0 0 2px; letter-spacing:.01em; }
  .sub { color:var(--dim); font-size:12px; margin:0 0 12px; }
  .sub b { color:var(--accent); font-weight:600; }
  .stage { position:relative; line-height:0; width:fit-content; margin:0 auto; }
  video { display:block; width:100%; height:auto; background:#000; }
  #overlay { position:absolute; inset:0; pointer-events:none; }
  #tip { position:absolute; pointer-events:none; z-index:3; display:none;
         background:rgba(12,12,12,.94); border:1px solid #555; border-radius:5px;
         padding:5px 8px; line-height:1.35; white-space:nowrap;
         box-shadow:0 3px 12px rgba(0,0,0,.55); }
  #tip .z { font-size:19px; font-weight:600; letter-spacing:-.01em; }
  #tip .z.bad { font-size:13px; font-weight:400; color:var(--dim); }
  #tip .d { font-size:11px; color:var(--dim); margin-top:1px; }
  #tip .w { font-size:11px; color:#ffc46b; margin-top:1px; }
  .bar { margin:10px auto 0; max-width:1920px; padding:8px 10px;
         border:1px solid var(--line); border-radius:5px; line-height:1.5; }
  .transport { display:flex; align-items:center; gap:10px; }
  #pp { width:32px; height:26px; flex:none; background:#262626; color:var(--fg);
        border:1px solid #454545; border-radius:4px; cursor:pointer; font:inherit;
        font-size:12px; line-height:1; }
  #pp:hover { background:#343434; }
  #seek { flex:1; min-width:120px; height:20px; accent-color:#8fb4ff; cursor:pointer; }
  .fields { display:flex; flex-wrap:wrap; gap:6px 20px; align-items:baseline;
            margin-top:8px; padding-top:8px; border-top:1px solid var(--line); }
  .fields span { color:var(--dim); }
  .fields span b { color:var(--fg); font-weight:600; }
  .hint { color:var(--dim); font-size:11.5px; margin:9px 2px 0; }
  kbd { border:1px solid #4a4a4a; border-bottom-width:2px; border-radius:3px;
        padding:0 4px; color:var(--fg); font-size:11px; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>

  <div class="stage" id="stage">
    <video id="v" preload="auto" playsinline>__VIDEO_SOURCE__</video>
    <svg id="overlay" preserveAspectRatio="none"></svg>
    <div id="tip"></div>
  </div>

  <div class="bar">
    <div class="transport">
      <button id="pp" title="play / pause">&#9654;</button>
      <input id="seek" type="range" min="0" step="1" value="0" title="frame">
    </div>
    <div class="fields">
      <span>frame <b id="r-frame">-</b></span>
      <span>time <b id="r-time">-</b></span>
      <span>cursor <b id="r-px">-</b></span>
      <span>distance <b id="r-z">move the cursor over the depth panel</b></span>
    </div>
  </div>
  <p class="hint">The middle panel is the depth. Hover it to read the distance at that
    pixel &mdash; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step one frame,
    <kbd>space</kbd> or a click on the picture play/pause. The transport sits under the
    video rather than on it, so nothing ever covers the colour bar.</p>
</main>

<script>
const META = __META__;
const FRAME_TILE = __FRAME_TILE__;
"""


TEMPLATE_TAIL = r"""
// ---------------------------------------------------------------- elements & labels
const video   = document.getElementById('v');
const stage   = document.getElementById('stage');
const overlay = document.getElementById('overlay');
const tip     = document.getElementById('tip');
const pp      = document.getElementById('pp');
const seek    = document.getElementById('seek');
const out = {frame: document.getElementById('r-frame'), time: document.getElementById('r-time'),
             px: document.getElementById('r-px'), z: document.getElementById('r-z')};
seek.max = META.frames - 1;

document.title = META.title;
document.getElementById('title').textContent = META.title;
document.getElementById('subtitle').innerHTML =
  META.subtitle + ' &middot; readout: <b>' + META.decode.note + '</b>';

// --------------------------------------------------------------------- depth tiles
// One PNG per frame: the high byte of the uint16 code on top, the low byte below.
const TW = META.tile.w, TH = META.tile.h, NPX = TW * TH;
const work = document.createElement('canvas');
work.width = TW; work.height = TH * 2;
const wctx = work.getContext('2d', {willReadFrequently: true});
const cache = new Map();                       // tile -> Uint16Array, or 'pending'
const MAX_CACHE = 24;

function requestTile(t) {
  if (t < 0) return null;
  const hit = cache.get(t);
  if (hit === 'pending') return null;
  if (hit) return hit;
  cache.set(t, 'pending');
  const img = new Image();
  img.onload = () => {
    wctx.clearRect(0, 0, TW, TH * 2);
    wctx.drawImage(img, 0, 0);
    const d = wctx.getImageData(0, 0, TW, TH * 2).data;
    const code = new Uint16Array(NPX);
    for (let i = 0; i < NPX; i++) code[i] = (d[i * 4] << 8) | d[(NPX + i) * 4];
    cache.set(t, code);
    while (cache.size > MAX_CACHE) {
      const k = cache.keys().next().value;      // Map keeps insertion order
      if (k === t) break;
      cache.delete(k);
    }
  };
  img.onerror = () => cache.set(t, new Uint16Array(NPX));
  img.src = 'data:image/png;base64,' + TILES[t];
  return null;
}

function codeToMetres(c) {
  if (c === 0) return NaN;                      // 0 is "no measurement", never a distance
  if (META.decode.mode === 'mm') return c * META.decode.unit_mm / 1000;
  return META.decode.z0_mm * Math.pow(META.decode.base, c - 1) / 1000;
}

// ------------------------------------------------------------------- frame tracking
// requestVideoFrameCallback reports the presentation time of the frame actually on
// screen; currentTime is only the fallback, because a readout one frame off the picture
// is worse than no readout.
let frameIdx = 0;
const clampFrame = i => Math.max(0, Math.min(META.frames - 1, i));
const timeToFrame = t => clampFrame(Math.floor(t * META.fps + 1e-3));

if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
  const onFrame = (_now, m) => {
    frameIdx = timeToFrame(m.mediaTime);
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
}

// ---------------------------------------------------------------------- hit testing
const P = META.panel, I = META.img, CELL = META.cell;
let mouse = null;                               // {cx, cy} in composite pixels

function compositeAt(ev) {
  const r = video.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return {cx: (ev.clientX - r.left) * (META.width / r.width),
          cy: (ev.clientY - r.top) * (META.height / r.height), rect: r};
}

function depthAt(cx, cy) {
  // composite pixel -> the depth image's own pixel, undoing depth_panel.fit_into
  const u = (cx - P.x - I.x0) * I.iw / I.nw;
  const v = (cy - P.y - I.y0) * I.ih / I.nh;
  if (u < 0 || u >= I.iw || v < 0 || v >= I.ih) return null;
  return {u, v,
          col: Math.min(TW - 1, Math.floor(u / CELL)),
          row: Math.min(TH - 1, Math.floor(v / CELL))};
}

const inPanel = (cx, cy) => cx >= P.x && cx < P.x + P.w && cy >= 0 && cy < META.height;

// ------------------------------------------------------------------------- painting
const SVG = 'http://www.w3.org/2000/svg';
function crosshair(x, y, w, h) {
  const g = [];
  const box = (sw, col) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${col}"` +
    ` stroke-width="${sw}"/>`;
  const arm = (sw, col) => {
    const cx = x + w / 2, cy = y + h / 2, gap = Math.max(w, h) / 2 + 2, len = 11;
    return `<path d="M${cx - gap - len} ${cy}H${cx - gap}M${cx + gap} ${cy}H${cx + gap + len}` +
           `M${cx} ${cy - gap - len}V${cy - gap}M${cx} ${cy + gap}V${cy + gap + len}"` +
           ` fill="none" stroke="${col}" stroke-width="${sw}"/>`;
  };
  g.push(box(3, 'rgba(0,0,0,.8)'), arm(3, 'rgba(0,0,0,.8)'), box(1, '#fff'), arm(1, '#fff'));
  return g.join('');
}

function clear() {
  overlay.innerHTML = '';
  tip.style.display = 'none';
  out.px.textContent = '-';
  out.z.textContent = 'move the cursor over the depth panel';
}

function paint() {
  const t = video.currentTime || 0;
  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) frameIdx = timeToFrame(t);
  requestTile(FRAME_TILE[clampFrame(frameIdx + 1)]);     // one ahead, for playback
  out.frame.textContent = frameIdx + ' / ' + (META.frames - 1);
  out.time.textContent = t.toFixed(2) + ' s';
  if (seek !== document.activeElement) seek.value = frameIdx;

  if (!mouse) { requestAnimationFrame(paint); return; }
  const r = video.getBoundingClientRect();
  const sx = r.width / META.width, sy = r.height / META.height;
  overlay.setAttribute('viewBox', `0 0 ${r.width} ${r.height}`);
  overlay.setAttribute('width', r.width);
  overlay.setAttribute('height', r.height);

  const hit = depthAt(mouse.cx, mouse.cy);
  if (!hit) {
    overlay.innerHTML = '';
    tip.style.display = 'none';
    out.px.textContent = '-';
    out.z.textContent = inPanel(mouse.cx, mouse.cy)
      ? 'header / colour bar (outside the depth image)'
      : (mouse.cx < P.x ? 'camera panel' : 'trajectory panel') + ' - no depth here';
    requestAnimationFrame(paint);
    return;
  }

  // the marker snaps to the cell that is being read, so the number and the pixel agree
  const x = (P.x + I.x0 + hit.col * CELL * I.nw / I.iw) * sx;
  const y = (P.y + I.y0 + hit.row * CELL * I.nh / I.ih) * sy;
  const w = Math.max(3, CELL * (I.nw / I.iw) * sx), h = Math.max(3, CELL * (I.nh / I.ih) * sy);
  overlay.innerHTML = crosshair(x, y, w, h);

  const tile = FRAME_TILE[frameIdx];
  const code = requestTile(tile);
  const pxLabel = `u ${Math.floor(hit.u)}, v ${Math.floor(hit.v)}`;
  out.px.textContent = pxLabel;

  let big, detail, warn = '';
  if (tile < 0) {
    big = '<div class="z bad">no depth for this frame</div>';
    detail = META.frame_note[frameIdx] || '';
    out.z.textContent = 'no depth for this frame';
  } else if (code === null) {
    big = '<div class="z bad">decoding&hellip;</div>';
    detail = pxLabel;
    out.z.textContent = 'decoding...';
  } else {
    const z = codeToMetres(code[hit.row * TW + hit.col]);
    if (!isFinite(z)) {
      big = '<div class="z bad">no measurement</div>';
      detail = pxLabel + ' &middot; black pixel (validity 0)';
      out.z.textContent = 'no measurement at this pixel';
    } else {
      big = `<div class="z">${z.toFixed(2)} m</div>`;
      detail = `${pxLabel} &middot; ${(z * 1000).toFixed(0)} mm &middot; frame ${frameIdx}`;
      out.z.textContent = `${z.toFixed(3)} m  (${(z * 1000).toFixed(0)} mm)`;
      if (z > META.z_far) warn = `beyond ${META.z_far.toFixed(2)} m - colour is clamped`;
      else if (z < META.z_near) warn = `nearer than ${META.z_near.toFixed(2)} m - colour is clamped`;
    }
  }
  tip.innerHTML = big + `<div class="d">${detail}</div>` +
                  (warn ? `<div class="w">${warn}</div>` : '');
  tip.style.display = 'block';

  // keep the tip inside the stage, and out from under the cursor
  const sr = stage.getBoundingClientRect();
  let tx = mouse.clientX - sr.left + 16, ty = mouse.clientY - sr.top + 16;
  if (tx + tip.offsetWidth > sr.width - 4) tx = mouse.clientX - sr.left - tip.offsetWidth - 14;
  if (ty + tip.offsetHeight > sr.height - 4) ty = mouse.clientY - sr.top - tip.offsetHeight - 14;
  tip.style.left = Math.max(4, tx) + 'px';
  tip.style.top = Math.max(4, ty) + 'px';

  requestAnimationFrame(paint);
}

// -------------------------------------------------------------------------- events
stage.addEventListener('mousemove', ev => {
  const c = compositeAt(ev);
  if (!c) return;
  mouse = {cx: c.cx, cy: c.cy, clientX: ev.clientX, clientY: ev.clientY};
});
stage.addEventListener('mouseleave', () => { mouse = null; clear(); });

function toggle() { video.paused ? video.play() : video.pause(); }
function goTo(i) {
  i = clampFrame(i);
  video.currentTime = (i + 0.5) / META.fps;      // mid-frame: never lands on the seam
  frameIdx = i;
}
pp.addEventListener('click', toggle);
stage.addEventListener('click', toggle);
video.addEventListener('play', () => { pp.innerHTML = '&#10074;&#10074;'; });
video.addEventListener('pause', () => { pp.innerHTML = '&#9654;'; });
seek.addEventListener('input', () => { video.pause(); goTo(+seek.value); });

document.addEventListener('keydown', ev => {
  if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') {
    ev.preventDefault();                        // the default is a 5-second seek
    video.pause();
    goTo(timeToFrame(video.currentTime) + (ev.key === 'ArrowRight' ? 1 : -1));
  } else if (ev.key === ' ' && ev.target !== seek) {
    ev.preventDefault();
    toggle();
  }
});

video.addEventListener('seeked', () => { frameIdx = timeToFrame(video.currentTime); });
requestTile(FRAME_TILE[0]);
requestAnimationFrame(paint);
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------ per episode

def build_episode(dataset: Path, slam_root: Path, ep: int, args) -> Path:
    ep_dir = slam_root / str(ep)
    mp4 = ep_dir / args.video_name
    rule_path = mp4.with_suffix(".colour_rule.json")
    if not mp4.exists():
        raise FileNotFoundError(f"no {args.video_name} (run data_collection/"
                                f"make_depth_trajectory_video.py first): {mp4}")
    if not rule_path.exists():
        raise FileNotFoundError(f"no colour rule beside the video: {rule_path}")
    rule = json.loads(rule_path.read_text())

    width, height, fps, _ = probe(mp4)
    if (width, height) != (3 * PANEL_W, PANEL_H):
        raise ValueError(f"{mp4.name} is {width}x{height}, not the "
                         f"{3 * PANEL_W}x{PANEL_H} three-panel layout this page "
                         f"knows how to read")
    n_frames = int(rule.get("frames") or 0)
    if n_frames <= 0:
        raise ValueError(f"{rule_path.name} does not say how many frames the video has")

    # the video's own rule, read back rather than re-decided
    domain = rule.get("domain", "obs")
    z_near = float(rule.get("z_near_m", 0.20))
    z_far = float(rule.get("z_far_m", 3.00))
    kind = rule.get("depth_source", "fs")
    if kind == "fs":
        src = FsDepth(dataset, ep, domain)
    else:
        src = ObsDepth(dataset, args.depth_store, domain).for_episode(ep)

    quant = Quantiser(args.quantise_pct)
    k = max(1, int(args.depth_scale))
    tiles: list[str] = []
    by_row: dict[int, int] = {}                 # stored row -> tile, so held frames share
    frame_tile = [-1] * n_frames
    frame_note = {}
    img = None
    for i in range(n_frames):
        z_m, row, note = src.metres(i)
        if z_m is None:
            frame_note[i] = note
            continue
        if img is None:
            ih, iw = z_m.shape
            img = fit_placement(iw, ih, PANEL_W, IMAGE_H)
        if row in by_row:
            frame_tile[i] = by_row[row]
        else:
            tiles.append(encode_codes(quant.encode(block_median(z_m, k))))
            by_row[row] = frame_tile[i] = len(tiles) - 1
        if note:
            frame_note[i] = note
        if args.progress and (i % 25 == 0 or i == n_frames - 1):
            print(f"    frame {i + 1}/{n_frames}", end="\r", flush=True)
    if args.progress:
        print(" " * 40, end="\r")
    if img is None:
        raise ValueError(f"ep {ep}: no depth for any of the {n_frames} video frames")

    meta = {
        "title": f"ep {ep} - camera | depth | trajectory",
        "subtitle": (f"{mp4.name} &middot; {kind} depth from "
                     f"{Path(rule.get('depth_artifact', '?')).name} &middot; "
                     f"{domain} | {rule.get('colormap', '?')} | warm=NEAR | "
                     f"{z_near:.2f}-{z_far:.2f} m | black = no measurement"),
        "episode": ep, "width": width, "height": height, "fps": round(fps),
        "frames": n_frames, "z_near": z_near, "z_far": z_far, "domain": domain,
        "panel": {"x": PANEL_W, "y": HEADER_H, "w": PANEL_W, "h": IMAGE_H},
        "img": img, "cell": k,
        "tile": {"w": img["iw"] // k, "h": img["ih"] // k},
        "decode": quant.js(), "frame_note": frame_note,
    }
    if k > 1:
        meta["decode"]["note"] += f"; {k}x{k} block median"

    out = ep_dir / args.out_name
    video_b64 = None if args.link_video else base64.b64encode(mp4.read_bytes()).decode("ascii")
    out.write_text(render_html(meta, frame_tile, tiles, video_b64, mp4.name))
    return out


# -------------------------------------------------------------------------------- cli

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="the UMI dataset root")
    p.add_argument("--slam-root", type=Path, default=None)
    p.add_argument("--episodes", default="all", help="all | 0,5,10-12")
    p.add_argument("--video-name", default="trajectory_depth.mp4",
                   help="the three-panel video to wrap (its .colour_rule.json must be "
                        "beside it -- that file is where the depth source and the ramp "
                        "are read from)")
    p.add_argument("--out-name", default=HTML_NAME)
    p.add_argument("--depth-store", type=Path, default=None,
                   help="depth_source obs: path to dataset_depth.zarr[.zip]")
    p.add_argument("--depth-scale", type=int, default=1,
                   help="store every k-th cell as the k x k block median (default 1, the "
                        "full depth grid). 2 quarters the page size and makes the readout "
                        "cell 2 depth pixels wide")
    p.add_argument("--quantise-pct", type=float, default=0.0,
                   help="0 (default) stores exact millimetres. A positive value stores "
                        "log-spaced codes with that RELATIVE step, e.g. 1.0 for +-0.5%% "
                        "and about a quarter of the size")
    p.add_argument("--link-video", action="store_true",
                   help="reference the mp4 by name instead of embedding it: a much "
                        "smaller file that only plays while it sits beside its mp4")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="do not print the per-frame counter (it encodes one PNG per "
                        "frame, so a long episode is a minute of silence otherwise)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    slam_root = args.slam_root or (args.dataset / "slam")
    if not slam_root.is_dir():
        print(f"no slam root: {slam_root}", file=sys.stderr)
        return 2
    episodes = parse_episode_spec(args.episodes, episodes_on_disk(slam_root))

    done = skipped = failed = 0
    for ep in episodes:
        out = slam_root / str(ep) / args.out_name
        if out.exists() and not args.overwrite:
            print(f"  ep {ep}: exists, skipping ({out})", flush=True)
            skipped += 1
            continue
        try:
            out = build_episode(args.dataset, slam_root, ep, args)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  ep {ep}: {exc}", flush=True)
            failed += 1
            continue
        done += 1
        print(f"  ep {ep}: {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"==> wrote {done} page(s), skipped {skipped}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
