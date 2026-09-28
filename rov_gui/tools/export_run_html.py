#!/usr/bin/env python3
"""
export_run_html.py — the same run as an INTERACTIVE page: drag the 3-D view.

    python -m rov_gui.tools.export_run_html \
        --run data/20260906/0906_191935_observe
    # -> <run>/scene_html/index.html   (open it in a browser)

    +-------------+-------------------+---------------------------+
    |   C3 RGB    | FoundationStereo  |  trajectory (drag to orbit)|
    +-------------+-------------------+---------------------------+
    [ play | speed | frame slider | tag ids | top-down | grid ]

Same four sources and the same frame algebra as ``render_run_scene.py`` (which
this imports rather than repeats — one loader, one datum inverse, one FLU->NED
flip). The difference is that the 3-D view is drawn by the BROWSER, so the
viewpoint is the reader's: left-drag orbits, right-drag (or shift-drag) pans,
the wheel zooms, R resets.

What is on the page, and what is deliberately not
-------------------------------------------------
* Every tag carries its ID, in both the 3-D view and the top-down inset.
* The vehicle is drawn at the CURRENT pose only — no path history, so the only
  line on the page is the policy's reference.
* The reference is the plan in force at that frame: its knots, its status, and
  a dropped line to the mat plane so its height reads as height.

No CDN, no framework, no build step: one HTML file with the data inlined
(``fetch`` of a sibling JSON is blocked under ``file://``, so inlining is what
makes the page work from a folder rather than from a server) and one JPEG per
exported frame beside it.

The RGB column
--------------
It lights up when the run has colour frames — ``--rgb-dir`` pointing at a
folder with the ``policy_obs`` layout (``index.csv`` with ``t_capture`` plus
images), which joins on the same clock as everything else. The 2026-09-06 run
has none: ``--record-depth`` persists the observation and the millimetre map,
and nothing in the station writes the colour stream to disk with per-frame
stamps (the ``ui_*.mp4`` screen capture and the ``c3_*_.mp4`` panel captures
carry only a start time and a nominal fps, which is not a join). The column
then says so rather than showing a plausible wrong frame.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rov_gui.depth_colour import Z_FAR_M, Z_NEAR_M, mm_to_t_valid  # noqa: E402
from rov_gui.depth_panel import (HEADER_H, IMAGE_H, PANEL_H,       # noqa: E402
                                 PANEL_W, depth_panel)
from rov_gui.tools.render_run_scene import (Run, _SHAPE,           # noqa: E402
                                            describe_siblings, tag_quads)

HTML = Path(__file__).with_name("_run_scene.html")


def _round(a, nd=4):
    return [round(float(v), nd) for v in np.asarray(a).ravel()]


def rov_model(run=None) -> dict:
    """The vehicle as JSON: body-FLU polygons from the tested geometry module.

    Sent once and transformed per frame in the browser, so the page cannot
    disagree with the station's own picture about what the vehicle is. The
    TCP dot is the RUN's own (meta ``rov_drawn_geometry.jaw_centre_m`` /
    ``policy.tcp.tcp_body_flu_m``) when a run is given — the jaw moved on
    2026-09-08 and an older run's plans were composed at the old point —
    and the live module's otherwise (``tcp_source`` says which).
    """
    tcp, src = (_SHAPE.tcp_body_for_meta(run.meta) if run is not None
                else (tuple(_SHAPE.tcp_body_m()), "live rov_shape"))
    parts = []
    for p in _SHAPE.parts():
        verts = [_round(v) for v in p["verts"]]
        faces = [{"i": [int(k) for k in idx], "n": _round(nrm)}
                 for idx, nrm in p["faces"]]
        parts.append({"kind": p["kind"], "name": p["name"],
                      "verts": verts, "faces": faces})
    return {"parts": parts, "tcp": _round(tcp), "tcp_source": src,
            "hull": [_SHAPE.HULL_L_M, _SHAPE.HULL_W_M]}


class RgbSource:
    """Colour frames with per-frame stamps, in the policy_obs layout."""

    def __init__(self, path: Path | None):
        self.dir, self.t, self.files = None, np.zeros(0), []
        if path is None:
            self.why = ("this run has no colour on disk: --record-depth writes "
                        "the observation and the millimetre map, and no station "
                        "writer stamps the colour stream per frame")
            return
        p = Path(path)
        idx = p / "index.csv"
        if not idx.exists():
            raise FileNotFoundError(
                f"{idx} not found. --rgb-dir wants the policy_obs layout: an "
                f"index.csv with a t_capture column and one image per row.")
        rows = list(csv.DictReader(open(idx)))
        # NOT obs_png: the observation is the depth tensor, and showing it in
        # the colour column would be a picture of the wrong thing that looks
        # like a picture of the right one.
        key = next((k for k in ("rgb_jpg", "rgb_png", "color_png", "image")
                    if rows and k in rows[0]), None)
        if key is None:
            raise ValueError(f"{idx}: no colour column "
                             f"(rgb_jpg / rgb_png / color_png / image)")
        rows = [r for r in rows if (r.get(key) or "").strip()]
        if not rows:
            raise ValueError(f"{idx}: the {key} column is empty on every row — "
                             f"this run recorded depth without colour "
                             f"(--no-record-depth-rgb, or a build before it)")
        self.dir = p
        self.t = np.array([float(r["t_capture"]) for r in rows])
        self.files = [p / r[key] for r in rows]
        self.why = ""

    def at(self, t_capture: float):
        if self.dir is None or not len(self.t):
            return None
        i = int(np.clip(np.searchsorted(self.t, t_capture), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t_capture) < abs(self.t[i] - t_capture):
            i -= 1
        return self.files[i], float(self.t[i] - t_capture)


#: How a frame is encoded. WebP because embedding doubles as the size budget:
#: a depth panel is 13.8 KiB as WebP q80 against 27.7 KiB as JPEG q80
#: [측정 2026-09-06, mean of seq 300/1200/2400 of this run], and base64 then
#: adds a third on top of whichever it is.
ENCODERS = {"webp": (".webp", "image/webp", "IMWRITE_WEBP_QUALITY"),
            "jpg": (".jpg", "image/jpeg", "IMWRITE_JPEG_QUALITY")}


def encode(img, fmt: str, quality: int) -> bytes:
    ext, _, par = ENCODERS[fmt]
    ok, buf = cv2.imencode(ext, img, [getattr(cv2, par), int(quality)])
    if not ok:
        raise RuntimeError(f"could not encode a frame as {fmt}")
    return buf.tobytes()


def encode_values(mm) -> bytes:
    """The frame's millimetres, EXACTLY, as a lossless picture the browser can
    read a pixel out of: high byte in R, low byte in G.

    The hover readout must not be read back out of the coloured panel. The
    panel is an 8-bit palette position, and in the ``obs`` domain that palette
    is deliberately crowded near the camera: t = (1/z - 1/z_far) /
    (1/z_near - 1/z_far) gives dz/dt = 4.667*z^2, so one of the 256 levels is
    0.16 m wide at 3 m [유도: depth_colour.palette_pos]. Add lossy WebP on top
    and a colour-derived number would be a measurement of the renderer, not of
    the depth. This layer is the recorded ``policy_obs/depth/*.png`` value
    itself -- lossless WebP round-trips it bit for bit.
    """
    im = np.zeros((*mm.shape[:2], 3), np.uint8)
    im[..., 2] = (mm >> 8) & 255          # BGR: R = high byte
    im[..., 1] = mm & 255                 #      G = low byte
    ok, buf = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, 101])
    if not ok:
        raise RuntimeError("could not encode the value layer")
    return buf.tobytes()


def panel_geometry(mm_shape) -> dict:
    """Where the depth image sits inside the 640x480 panel, so the page can
    turn a mouse position into a source pixel.

    Exported rather than hardcoded because ``depth_panel.fit_into`` centres and
    scales whatever it is given: the 2026-09 runs are 640x400 and land on the
    band 1:1, but a different grid would not, and a readout that assumed 1:1
    would name the wrong pixel without ever looking wrong.
    """
    ih, iw = int(mm_shape[0]), int(mm_shape[1])
    s = min(PANEL_W / iw, IMAGE_H / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    return {"pw": PANEL_W, "ph": PANEL_H, "hdr": HEADER_H,
            "iw": iw, "ih": ih, "nw": nw, "nh": nh,
            "x0": (PANEL_W - nw) // 2, "y0": HEADER_H + (IMAGE_H - nh) // 2}


def export(run: Run, out_dir: Path, *, every: int, limit: int | None,
           quality: int, domain: str, cmap: str, rgb: RgbSource,
           embed: bool = False, fmt: str = "jpg", out_name: str = "index.html",
           hover: bool = True) -> Path:
    """Write the page. ``embed`` inlines every frame as a data: URI, so the
    result is ONE file that shows the pictures too — at the cost of carrying
    them: see ENCODERS for what a frame costs and --every for how many there
    are."""
    frames_dir = out_dir / "frames"
    val_bytes = 0
    geom = None
    if not embed:
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, len(run.depth_rows), max(1, every)))
    if limit:
        idx = idx[:limit]

    fs = run.meta.get("fstereo", {})
    grid = (run.depth_meta.get("builder") or {}).get("grid") or {}
    fx = (grid.get("P1") or [[float("nan")]])[0][0]
    ck = Path(str(fs.get("ckpt", "?")))
    frames = []
    for n, i in enumerate(idx):
        row = run.depth_rows[i]
        t_cap = float(row["t_capture"])
        t_run = t_cap - run.t_offset
        j, dt_pose = run.pose_at(t_run)
        plan, age = run.plan_at(t_run)

        name = ""
        val = ""
        mm = None
        if row.get("depth_png"):
            mm = cv2.imread(str(run.dir / "policy_obs" / row["depth_png"]),
                            cv2.IMREAD_UNCHANGED)
        if mm is not None:
            t, valid = mm_to_t_valid(mm, domain, Z_NEAR_M, Z_FAR_M)
            panel = depth_panel(
                t, valid, domain=domain, cmap=cmap, z_near=Z_NEAR_M,
                z_far=Z_FAR_M,
                lines=[f"FoundationStereo depth   policy_obs/{row['depth_png']}"
                       f"   mm, 0 = invalid   seq {row['seq']}",
                       f"iters {fs.get('iters', '?')}  scale "
                       f"{fs.get('scale', '?')}  {ck.parent.name}/{ck.name}  "
                       f"fx_rect {fx:.2f} px"])
            blob = encode(panel, fmt, quality)
            if embed:
                name = (f"data:{ENCODERS[fmt][1]};base64,"
                        + base64.b64encode(blob).decode("ascii"))
            else:
                name = f"d{n:05d}{ENCODERS[fmt][0]}"
                (frames_dir / name).write_bytes(blob)
            if hover:
                if geom is None:
                    geom = panel_geometry(mm.shape)
                vblob = encode_values(mm)
                val_bytes += len(vblob)
                if embed:
                    val = ("data:image/webp;base64,"
                           + base64.b64encode(vblob).decode("ascii"))
                else:
                    val = f"v{n:05d}.webp"
                    (frames_dir / val).write_bytes(vblob)

        rgb_name = ""
        got = rgb.at(t_cap)
        if got is not None:
            src, drgb = got
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is not None:
                rblob = encode(img, fmt, quality)
                if embed:
                    rgb_name = (f"data:{ENCODERS[fmt][1]};base64,"
                                + base64.b64encode(rblob).decode("ascii"))
                else:
                    rgb_name = f"c{n:05d}{ENCODERS[fmt][0]}"
                    (frames_dir / rgb_name).write_bytes(rblob)

        d = mm[mm > 0] if mm is not None else np.zeros(0, np.uint16)
        frames.append({
            "t": round(t_run, 3), "seq": int(row["seq"]), "img": name,
            "rgb": rgb_name, "v": val,
            "dpng": (row.get("depth_png") or ""),
            "p": _round(run.p_map[j]), "yaw": round(float(run.yaw_map_all[j]), 5),
            "pitch": round(float(run.pitch_deg[j]), 2),
            "roll": round(float(run.roll_deg[j]), 2),
            "tags": int(run.n_tags[j]), "age": round(float(run.tag_age[j]), 3),
            "rms": round(float(run.pnp_rms[j]), 2), "zsrc": run.z_src[j],
            "dtp": round(float(dt_pose), 3),
            "plan": (plan["plan_id"] if plan else ""),
            "page": round(float(age), 2) if plan else 0.0,
            "valid": round(100.0 * float((mm > 0).mean()), 1),
            "near": (round(float(d.min()) / 1000.0, 2) if d.size else 0.0),
            "med": (round(float(np.median(d)) / 1000.0, 2) if d.size else 0.0),
        })
        if n % 200 == 0:
            print(f"  {n}/{len(idx)}  t {t_run:7.2f} s", flush=True)

    used = {f["plan"] for f in frames if f["plan"]}
    plans = {}
    for p in run.plans:
        if p["plan_id"] not in used:
            continue
        d = {"status": p["status"], "t": round(p["t_rel"], 3),
             "k": [_round(q) for q in p["knots_map"]],
             "tk": [round(v, 2) for v in p["t_knot"]],
             "why": (p.get("reason") or "")[:120]}
        # The policy's OWN 16 steps beside the 6 knots the controller got.
        # Recomposed by the run's own compose_plan and checked against the
        # recorded knots to 0.000 um — see Run._add_raw_horizon.
        if p.get("raw_map") is not None:
            d["r"] = [_round(q) for q in p["raw_map"]]
            d["nraw"] = int(p.get("n_raw", len(d["r"])))
            d["obsdt"] = round(float(p.get("obs_dt", 0.0)) * 1e3, 1)
        plans[p["plan_id"]] = d

    quads = tag_quads(run.tags, run.tag_size)
    tags = []
    for tid, w in quads:
        w = np.asarray(w, float)
        # The tag's outward normal, so a marker placed ON a tag stands OUT of
        # the mat rather than through it. Taken from the drawn corners (not a
        # second reading of the quaternion) and flipped to point up, which in
        # this NED map is -z.
        n = np.cross(w[1] - w[0], w[3] - w[0])
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        if n[2] > 0:
            n = -n
        tags.append({"id": tid, "q": [_round(c) for c in w],
                     "c": _round(w.mean(axis=0)), "n": _round(n)})

    allp = np.concatenate([np.asarray([q for _, q in quads]).reshape(-1, 3),
                           run.p_map])
    data = {
        "run": f"{run.dir.name}  ·  {run.meta_path.stem.replace('.meta','')}",
        "run_path": str(run.dir),
        "tagmap": {"file": run.tag_map_path.name, "sha1": run.map_sha1,
                   "matches": bool(run.map_matches),
                   "anchor": (int(run.tags.anchor_id)
                              if run.tags.anchor_id is not None else -1),
                   "size_m": run.tag_size, "n": len(quads)},
        "datum": {"p0": _round(run.p0),
                  "yaw0_deg": round(math.degrees(run.yaw0), 3)},
        "fstereo": {"iters": fs.get("iters"), "scale": fs.get("scale"),
                    "ckpt": f"{ck.parent.name}/{ck.name}",
                    "grid": (run.depth_meta.get("obs") or {}).get("grid_kind"),
                    "fx_rect": round(float(fx), 2)},
        "clock": {"offset": round(float(run.t_offset), 3),
                  "spread_ms": (round(float(run.offset_spread) * 1e3, 2)
                                if np.isfinite(run.offset_spread) else None)},
        "bounds": [_round([allp[:, 0].min(), allp[:, 0].max()]),
                   _round([allp[:, 1].min(), allp[:, 1].max()])],
        "rov": rov_model(run),
        "rgb_why": rgb.why,
        "has_depth": bool(run.has_depth),
        "depth_why": run.no_depth_why,
        "embedded": bool(embed),
        # The hover readout's two halves: where the picture sits in the panel,
        # and how a value-layer pixel decodes. Both live here so the page never
        # assumes a layout or a scale it was not told.
        "depth_geom": geom,
        "hover": {"on": bool(geom), "mm_per_lsb": 1,
                  "why": ("" if geom else
                          ("--no-hover: the page carries no value layer, so "
                           "the depth picture cannot be read back"
                           if not hover else
                           "this run recorded no depth frames"))},
        "fps": round(1.0 / max(1e-3, float(np.median(np.diff(run.t_depth)))
                               * max(1, every)), 3),
        "tags": tags,
        "plans": plans,
        "frames": frames,
    }

    if val_bytes:
        # Named separately because it is the price of the hover readout, and a
        # reader deciding between --every and --no-hover needs the two halves
        # apart rather than one total.
        print(f"hover    exact mm layer: {val_bytes / 1e6:.1f} MB over "
              f"{len(frames)} frames "
              f"({val_bytes / max(1, len(frames)) / 1e3:.0f} kB each"
              f"{', +33% inlined as base64' if embed else ', in frames/'})"
              f"  — --no-hover drops it")

    page = HTML.read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = out_dir / out_name
    out.write_text(page)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="output folder (default: <run>/scene_html)")
    p.add_argument("--rgb-dir", type=Path, default=None,
                   help="colour frames in the policy_obs layout (index.csv "
                        "with t_capture + one image per row). Not needed for a "
                        "run flown with --record-depth on a build from "
                        "2026-09-07: <run>/policy_obs is then found by itself.")
    p.add_argument("--tag-map", type=Path, default=None)
    p.add_argument("--allow-map-mismatch", action="store_true")
    p.add_argument("--engagement", default=None, metavar="NAME",
                   help="which engagement in the folder (fragment of the csv "
                        "name); the folder can hold several and their per-plan "
                        "logs are shared")
    p.add_argument("--every", type=int, default=2,
                   help="export every Nth recorded depth frame (default: "
                        "%(default)s). Each frame is a ~26 kB JPEG [측정 "
                        "2026-09-06, q80 on this run], so every-2 of a 2854 "
                        "frame run is about 37 MB.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--embed", action="store_true",
                   help="ONE FILE: inline every frame as a data: URI, so the "
                        "page shows the depth (and colour) pictures with no "
                        "frames/ folder beside it. A frame is 13.8 kB as WebP "
                        "q80 [측정 2026-09-06] and base64 adds a third, so the "
                        "page is roughly 18 kB x the frame count — ~14 MB at "
                        "--every 4, ~27 MB at --every 2, ~54 MB for all 2854. "
                        "Without this the pictures are JPEGs in frames/ and "
                        "the page is under a megabyte.")
    p.add_argument("--img-format", choices=("webp", "jpg"), default=None,
                   help="frame encoding (default: webp when --embed, jpg "
                        "otherwise). WebP is half the bytes at the same look.")
    p.add_argument("--out-name", default=None,
                   help="page filename (default: index.html, or "
                        "scene_embedded.html with --embed)")
    p.add_argument("--no-hover", action="store_true",
                   help="drop the per-frame value layer. With it (the default) "
                        "hovering the depth picture reads out the recorded "
                        "millimetre under the cursor; it costs a lossless WebP "
                        "of the mm map per frame — 48 kB [측정 2026-09-08, "
                        "0908_180453 640x400], so about 64 kB a frame inlined "
                        "by --embed, roughly 3.5x the page. Reading the colour "
                        "back instead is not offered: see encode_values.")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--domain", choices=("obs", "metric"), default="obs")
    p.add_argument("--cmap", default="turbo")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        run = Run(a.run, a.tag_map, strict_map=not a.allow_map_mismatch,
                  engagement=a.engagement)
    except (FileNotFoundError, ValueError) as e:
        print(f"{a.run}: {e}{describe_siblings(a.run)}", file=sys.stderr)
        return 2
    run.draw_tag_ids = True
    # --record-depth files the colour beside the obs since 2026-09-07, so the
    # colour column lights up with no flag when a run has it.
    rgb_dir = a.rgb_dir
    if rgb_dir is None:
        try:
            RgbSource(run.dir / "policy_obs")
            rgb_dir = run.dir / "policy_obs"
        except Exception:                                        # noqa: BLE001
            rgb_dir = None
    rgb = RgbSource(rgb_dir)
    fmt = a.img_format or ("webp" if a.embed else "jpg")
    out_name = a.out_name or ("scene_embedded.html" if a.embed else "index.html")
    out_dir = a.out or (run.dir if a.embed else run.dir / "scene_html")
    print(f"run      {run.dir}   engagement {run.meta_path.name} "
          f"(epoch {run.epoch})")
    print(f"tcp      {np.round(run.tcp_body, 4).tolist()}  <- {run.tcp_source}")
    print(f"tag map  {run.tag_map_path.name} sha1 {run.map_sha1}"
          f"{'  (matches)' if run.map_matches else '  != THE RUN'}")
    print(f"rgb      {'yes: ' + str(rgb.dir) if rgb.dir else 'none — ' + rgb.why}")
    out = export(run, out_dir, every=a.every, limit=a.limit,
                 quality=a.quality, domain=a.domain, cmap=a.cmap, rgb=rgb,
                 embed=a.embed, fmt=fmt, out_name=out_name,
                 hover=not a.no_hover)
    page_mb = out.stat().st_size / 1e6
    if a.embed:
        print(f"==> {out}   ONE FILE, {page_mb:.1f} MB "
              f"({fmt} q{a.quality}, every {a.every})")
        if page_mb > 60:
            print("    NOTE: that is a big page to open — raise --every "
                  "(each doubling halves it), lower --quality, or drop the "
                  "hover readout with --no-hover.")
    else:
        fr = list((out_dir / "frames").glob("*"))
        mb = sum(f.stat().st_size for f in fr) / 1e6
        print(f"==> {out}   ({len(fr)} frames, {mb:.0f} MB, page {page_mb:.1f} "
              f"MB). The page alone carries the 3-D scene; the pictures live "
              f"in frames/ — copy the folder, or use --embed for one file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
