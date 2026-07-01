#!/usr/bin/env python3
"""Generate the real-pool AprilTag floor for the MuJoCo sim (VISUAL ONLY).

Replicates the physical pool's tag36h11 floor (see ../config/config.yaml and
../config/tag_map.yaml) inside `bluerov2_mujoco_marinegym/` as a grid of textured
tiles laid on a seabed, plus a translucent water surface. Everything produced here
is VISUAL ONLY (contype=0 conaffinity=0) and never touches dynamics -- the hydro
callback (hydro.py) only acts on base_link, and MuJoCo's fluid model is OFF
(<option density=0 viscosity=0>).

Run (robust env only -- needs mujoco/cv2/pupil_apriltags):

    /home/bdml/miniforge3/envs/robust/bin/python gen_pool_apriltags.py --selftest
    /home/bdml/miniforge3/envs/robust/bin/python gen_pool_apriltags.py            # full build

Outputs (all under this directory):
    apriltags/tag36h11_<ID:05d>.png   one PNG per tag (round-trip verified)
    tag_floor.xml                     <mujocoinclude> fragment (assets + geoms)
    scene_bluerov_tags.xml            opt-in wrapper (bluerov.xml + tag_floor.xml)
    scene_bluerov_heavy_tags.xml      opt-in wrapper (bluerov_heavy.xml + tag_floor.xml)

Enable in the sim with `POOL_TAGS=1` (see rov_model.py).

Design/frame notes
------------------
* Real pool tag spec: family tag36h11, black-border edge = tags.tag_size_m = 0.170 m
  (== tag_object_points corners +/-0.085 in src/tagslam_core.py). With the mandatory
  1-module white quiet zone the full printed tile is 0.170 * 10/8 = 0.2125 m, so the
  box tile half-extent is 0.10625 m and the black square lands at exactly 0.170 m.
* Frame: tag_map.yaml is the anchor(tag 25) frame with +Z DOWN (config water.up_axis_world
  = [0,0,-1]); the sim is FLU with +Z UP. Map->sim tile: x_sim=x_map, y_sim=-y_map,
  z_sim=seabed_z (tags flat on the seabed). A MuJoCo box shows its texture on the +Z
  (top) face, so a face-up tile uses quat = identity plus the tag's in-plane yaw
  (NOT the physical-frame flip). The map yaw families are ~0 deg and ~180 deg; both are
  reproduced. The tiny survey tilt/z (~1 deg, ~+/-1 cm floor unevenness) is dropped so
  tiles stay perfectly flat.
* ID correctness: cv2.aruco DICT_APRILTAG_36h11 id<->pattern mapping is not guaranteed
  equal to canonical tag36h11, so every generated PNG is re-detected with
  pupil_apriltags(tag36h11) and asserted to decode to the intended id (self-correcting).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(REPO, "config")
APRILTAG_DIR = os.path.join(HERE, "apriltags")

TAG36H11_N_IDS = 587  # DICT_APRILTAG_36h11 holds ids 0..586

# ---- defaults (all overridable on the CLI) --------------------------------
DEF_PITCH_X = 0.2565   # survey median center-to-center along the pool WIDTH (X)
DEF_PITCH_Y = 0.2141   # survey median center-to-center along the pool LENGTH (Y)
DEF_MARGIN = 0.10      # keep tags this far from the pool walls
DEF_VIS_POOL_WIDTH = 2.6   # VISUAL sim pool width (X). The real pool is 1.8 m (config.yaml);
                           # widened so the scene isn't narrow. Grid just adds more columns at
                           # the same pitch/spec. Length (Y) stays at the real config value.
DEF_SEABED_Z = -0.5    # seabed (tag floor) height in the sim FLU frame
DEF_WATER_DEPTH = 2.0  # VISUAL water column depth -> surface at seabed_z + depth.
                       # (Purely cosmetic; the disturbance/wave PHYSICS depth is separate:
                       #  disturbance/waves.py h, disturbances.py z_surface -- left at 4 m.)
DEF_PX_PER_MODULE = 32 # tag36h11 code is 8 modules; PNG code side = 8*this
TILE_HALF_THICK = 0.002
TILE_LIFT = 0.003      # lift tile centre above the seabed plane (anti z-fighting)


# ---------------------------------------------------------------------------
# quaternion helpers (wxyz, Hamilton)
# ---------------------------------------------------------------------------
def yaw_of_quat_wxyz(q) -> float:
    """Yaw (rotation about the frame's +Z) of a wxyz quaternion, radians."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_about_z(yaw: float):
    """wxyz quaternion for a rotation of `yaw` radians about +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


# ---------------------------------------------------------------------------
# config / map loading
# ---------------------------------------------------------------------------
def load_config():
    with open(os.path.join(CONFIG_DIR, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, "tag_map.yaml")) as f:
        tm = yaml.safe_load(f)
    tag_size = float(cfg["tags"]["tag_size_m"])
    pool_w = float(cfg["pool"]["width_m"])    # X (short) axis
    pool_l = float(cfg["pool"]["length_m"])   # Y (long) axis
    anchor = int(tm.get("anchor_tag_id", -1))
    real = {}
    for tid, rec in (tm.get("tags") or {}).items():
        real[int(tid)] = {
            "pos": [float(v) for v in rec["position_m"]],
            "quat": [float(v) for v in rec["quaternion_wxyz"]],
        }
    return dict(tag_size=tag_size, pool_w=pool_w, pool_l=pool_l,
                anchor=anchor, real=real)


# ---------------------------------------------------------------------------
# tag PNG rendering + mandatory round-trip verification
# ---------------------------------------------------------------------------
def make_dictionary():
    import cv2
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def render_tag_png(dic, detector, tag_id: int, px_per_module: int, mirror: bool):
    """Render one tag36h11 PNG (black code + 1-module white quiet zone),
    then re-detect it with pupil_apriltags and assert it decodes to `tag_id`.
    Returns the uint8 grayscale image. Raises on any detection mismatch."""
    import cv2
    side = 8 * px_per_module                     # code is 8 modules (6 data + border)
    code = cv2.aruco.generateImageMarker(dic, tag_id, side)
    if mirror:
        code = cv2.flip(code, 1)
    m = px_per_module                            # 1-module white quiet zone
    img = cv2.copyMakeBorder(code, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    dets = detector.detect(img)
    ids = [d.tag_id for d in dets]
    if len(dets) != 1 or ids[0] != tag_id:
        raise RuntimeError(
            f"round-trip FAILED for tag {tag_id}: detected {ids} "
            f"(expected exactly [{tag_id}]). Try --mirror or the apriltag-imgs fallback."
        )
    return img


# ---------------------------------------------------------------------------
# tag layout: real survey tags + regular grid fill over the pool floor
# ---------------------------------------------------------------------------
def build_tag_list(cfg, layout, pitch_x, pitch_y, margin, pool_w=None, pool_l=None):
    """Return a list of {id, pos(map xyz), quat(map wxyz), src} covering the floor.

    layout=survey : only the surveyed tags.
    layout=grid   : a pure regular grid (new ids 0,1,2,... nominal yaw).
    layout=hybrid : surveyed tags at their real poses/ids + a regular grid filling
                    the rest of the pool floor with new ids (default).

    pool_w / pool_l override the covered footprint (default = config real-pool size).
    Widening pool_w just adds more grid columns at the SAME pitch/spec around the real
    surveyed tags, so the pattern stays consistent.
    """
    real = cfg["real"]
    if not real:
        raise RuntimeError("tag_map.yaml has no tags")
    pool_w = cfg["pool_w"] if pool_w is None else float(pool_w)
    pool_l = cfg["pool_l"] if pool_l is None else float(pool_l)
    rx = [v["pos"][0] for v in real.values()]
    ry = [v["pos"][1] for v in real.values()]
    cx = 0.5 * (min(rx) + max(rx))               # centre coverage on the survey bbox
    cy = 0.5 * (min(ry) + max(ry))
    hx = max(0.0, (pool_w - 2.0 * margin) / 2.0)
    hy = max(0.0, (pool_l - 2.0 * margin) / 2.0)

    tags = []
    if layout in ("survey", "hybrid"):
        for tid in sorted(real):
            tags.append(dict(id=tid, pos=real[tid]["pos"],
                             quat=real[tid]["quat"], src="survey"))

    if layout in ("grid", "hybrid"):
        used = set(real.keys()) if layout == "hybrid" else set()
        free = (i for i in range(TAG36H11_N_IDS) if i not in used)
        # grid nodes anchored on the anchor origin (0,0), covering the pool footprint
        i0, i1 = math.ceil((cx - hx) / pitch_x), math.floor((cx + hx) / pitch_x)
        j0, j1 = math.ceil((cy - hy) / pitch_y), math.floor((cy + hy) / pitch_y)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                x, y = i * pitch_x, j * pitch_y
                if layout == "hybrid" and _cell_occupied(x, y, real, pitch_x, pitch_y):
                    continue
                try:
                    nid = next(free)
                except StopIteration:
                    raise RuntimeError(
                        "grid needs more than 587 tag ids; increase pitch or shrink extent")
                tags.append(dict(id=nid, pos=[x, y, 0.0],
                                 quat=[1.0, 0.0, 0.0, 0.0], src="grid"))
    return tags, (cx, cy)


def _cell_occupied(x, y, real, pitch_x, pitch_y):
    for v in real.values():
        if abs(v["pos"][0] - x) < 0.5 * pitch_x and abs(v["pos"][1] - y) < 0.5 * pitch_y:
            return True
    return False


def flu_pose(entry, seabed_z):
    """Map (anchor-frame) tag pose -> sim FLU tile pose.
    Position: x kept, y negated, z on the seabed (+ a small anti-z-fight lift).
    Orientation: face UP on the box +Z face (identity base) carrying the tag's
    in-plane yaw. The map's +Z-down / sim +Z-up handedness flip reverses the yaw
    sense; that is exact for the ~0 and ~180 deg survey families used here."""
    x, y, _z = entry["pos"]
    pos = (x, -y, seabed_z + TILE_LIFT + TILE_HALF_THICK)
    quat = quat_about_z(-yaw_of_quat_wxyz(entry["quat"]))
    return pos, quat


# ---------------------------------------------------------------------------
# XML emission
# ---------------------------------------------------------------------------
def _fmt(vals):
    return " ".join(f"{v:.6f}" for v in vals)


def build_fragment_xml(cfg, tags, args, half_edge):
    """Return the <mujocoinclude> fragment string (assets + worldbody geoms)."""
    seabed_z = args.seabed_z
    surf_z = args.water_surface_z
    # sim-frame tile centres, to size the seabed/water planes
    poses = [flu_pose(t, seabed_z) for t in tags]
    xs = [p[0][0] for p in poses]
    ys = [p[0][1] for p in poses]
    cxs = 0.5 * (min(xs) + max(xs))
    cys = 0.5 * (min(ys) + max(ys))
    plane_hx = 0.5 * (max(xs) - min(xs)) + half_edge + 0.30
    plane_hy = 0.5 * (max(ys) - min(ys)) + half_edge + 0.30

    tex, mat, geo = [], [], []
    for t, (pos, quat) in zip(tags, poses):
        name = f"tag36h11_{t['id']:05d}"
        tex.append(f'    <texture name="{name}_tex" type="2d" file="apriltags/{name}.png"/>')
        mat.append(f'    <material name="{name}_mat" texture="{name}_tex" '
                   f'texrepeat="1 1" texuniform="false" specular="0.05" '
                   f'shininess="0.05" reflectance="0"/>')
        geo.append(f'    <geom name="{name}" type="box" '
                   f'size="{half_edge:.5f} {half_edge:.5f} {TILE_HALF_THICK:.4f}" '
                   f'pos="{_fmt(pos)}" quat="{_fmt(quat)}" '
                   f'material="{name}_mat" contype="0" conaffinity="0" group="1"/>')

    seabed = (f'    <geom name="pool_seabed" type="plane" '
              f'pos="{cxs:.5f} {cys:.5f} {seabed_z:.5f}" '
              f'size="{plane_hx:.5f} {plane_hy:.5f} 0.2" '
              f'material="pool_seabed_mat" contype="0" conaffinity="0" group="1"/>')

    # Water is a SINGLE translucent geom so the wavy top and the water column read as ONE
    # body (no seam between a separate "surface" sheet and a "body" box). VISUAL ONLY
    # (contype=0 conaffinity=0). Default (--water-body) fills the column: the animated
    # heightfield's underside skirt is extruded down to the seabed, so the same geom/material
    # is both the wavy surface and the submerged volume. --no-water-body -> a thin sheet only;
    # --no-water-anim -> a flat-topped solid box (still one geom). water_viz.py animates the
    # hfield top; it reads only pos_z + elev, so the tall skirt does not affect the animation.
    fill = getattr(args, "water_body", True)
    if getattr(args, "water_anim", True):
        elev = float(args.water_hf_elev)                 # hfield z half-range = max|eta| headroom
        pos_z = surf_z - 0.5 * elev                      # d=0.5 (mean) renders the top at surf_z
        base = (pos_z - seabed_z - 0.02) if fill else float(args.water_hf_base)
        water_asset = (f'    <hfield name="pool_water_hf" '
                       f'nrow="{int(args.water_hf_rows)}" ncol="{int(args.water_hf_cols)}" '
                       f'size="{plane_hx:.5f} {plane_hy:.5f} {elev:.5f} {max(base, 0.01):.5f}"/>')
        water = (f'    <geom name="pool_water_surface" type="hfield" hfield="pool_water_hf" '
                 f'pos="{cxs:.5f} {cys:.5f} {pos_z:.5f}" '
                 f'material="pool_water_mat" contype="0" conaffinity="0" group="1"/>')
    else:
        water_asset = ""
        cz = 0.5 * (surf_z + seabed_z) if fill else surf_z
        hz = 0.5 * (surf_z - seabed_z) if fill else 0.002
        water = (f'    <geom name="pool_water_surface" type="box" '
                 f'pos="{cxs:.5f} {cys:.5f} {cz:.5f}" '
                 f'size="{plane_hx:.5f} {plane_hy:.5f} {max(hz, 0.002):.5f}" '
                 f'material="pool_water_mat" contype="0" conaffinity="0" group="1"/>')

    water_alpha = float(getattr(args, "water_alpha", 0.18))
    nl = "\n"
    return f"""<mujocoinclude>
  <!-- GENERATED by gen_pool_apriltags.py -- do not edit by hand. VISUAL ONLY.
       {len(tags)} tag36h11 tiles + seabed + water surface ({'animated hfield' if getattr(args, 'water_anim', True) else 'flat box'}).
       seabed z={seabed_z}, water surface z={surf_z} (depth {surf_z - seabed_z:.3f} m).
       All geoms contype=0 conaffinity=0 group=1 (visible by default) -> zero effect on dynamics.
       The water hfield is animated at render time by water_viz.py (still dynamics-inert). -->
  <asset>
    <material name="pool_seabed_mat" rgba="0.50 0.50 0.53 1" specular="0.1" shininess="0.05" reflectance="0"/>
    <material name="pool_water_mat" rgba="0.13 0.33 0.53 {water_alpha:.3f}" specular="0.2" shininess="0.3" reflectance="0"/>
{water_asset}
{nl.join(tex)}
{nl.join(mat)}
  </asset>
  <worldbody>
{seabed}
{water}
{nl.join(geo)}
  </worldbody>
</mujocoinclude>
"""


WRAPPER_TMPL = """<mujoco model="{model}">
  <!-- GENERATED by gen_pool_apriltags.py. Opt-in pool scene = ROV + AprilTag floor.
       Selected by rov_model.py when POOL_TAGS=1. VISUAL-ONLY floor; dynamics unchanged. -->
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>
  <include file="{rov}"/>
  <include file="tag_floor.xml"/>
</mujoco>
"""


def write_wrappers():
    with open(os.path.join(HERE, "scene_bluerov_tags.xml"), "w") as f:
        f.write(WRAPPER_TMPL.format(model="bluerov2_tagpool", rov="bluerov.xml"))
    with open(os.path.join(HERE, "scene_bluerov_heavy_tags.xml"), "w") as f:
        f.write(WRAPPER_TMPL.format(model="bluerov2_heavy_tagpool", rov="bluerov_heavy.xml"))


# ---------------------------------------------------------------------------
# build + selftest
# ---------------------------------------------------------------------------
def generate_pngs(tags, px_per_module, mirror):
    detector = _make_detector()
    dic = make_dictionary()
    os.makedirs(APRILTAG_DIR, exist_ok=True)
    import cv2
    failures = []
    for t in tags:
        try:
            img = render_tag_png(dic, detector, t["id"], px_per_module, mirror)
        except RuntimeError as e:
            failures.append(str(e))
            continue
        cv2.imwrite(os.path.join(APRILTAG_DIR, f"tag36h11_{t['id']:05d}.png"), img)
    if failures:
        for msg in failures:
            print("  " + msg, file=sys.stderr)
        raise SystemExit(f"ABORT: {len(failures)} tag(s) failed round-trip verification.")


def _make_detector():
    import pupil_apriltags
    return pupil_apriltags.Detector(families="tag36h11", nthreads=4)


def run_selftest(cfg, args, half_edge):
    """Render a few tiles from straight above and confirm pupil_apriltags detects
    them with the correct ids -> proves the MuJoCo tiles are face-UP and readable."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco, cv2

    # pick the anchor + one ~180deg-yaw survey tag if available
    ids = [cfg["anchor"]]
    for tid, v in cfg["real"].items():
        if abs(yaw_of_quat_wxyz(v["quat"])) > 2.0 and tid not in ids:
            ids.append(tid); break
    ids = [i for i in ids if i in cfg["real"]][:2]
    subset = [dict(id=i, pos=cfg["real"][i]["pos"], quat=cfg["real"][i]["quat"],
                   src="survey") for i in ids]

    generate_pngs(subset, args.px_per_module, args.mirror)
    frag = build_fragment_xml(cfg, subset, args, half_edge)
    with open(os.path.join(HERE, "tag_floor.xml"), "w") as f:
        f.write(frag)

    # A separate top-down camera centred over each tile so every tile is validated,
    # not just the ones near the origin. Uses DEFAULT render options (group 1 must be
    # visible-by-default, exactly like the real viewer).
    poses = {t["id"]: flu_pose(t, args.seabed_z) for t in subset}
    cams = "\n".join(
        f'    <camera name="c{t["id"]}" pos="{poses[t["id"]][0][0]:.5f} '
        f'{poses[t["id"]][0][1]:.5f} {args.seabed_z + 1.0:.5f}" '
        f'xyaxes="1 0 0 0 1 0" fovy="28"/>'
        for t in subset)
    lights = "\n".join(
        f'    <light pos="{poses[t["id"]][0][0]:.5f} {poses[t["id"]][0][1]:.5f} '
        f'{args.seabed_z + 1.0:.5f}" dir="0 0 -1" directional="true"/>'
        for t in subset)
    scene = f"""<mujoco model="_selftest">
  <compiler angle="radian" autolimits="true"/>
  <visual><global offwidth="900" offheight="900"/>
    <headlight ambient="0.9 0.9 0.9" diffuse="0.4 0.4 0.4" specular="0 0 0"/></visual>
  <worldbody>
{lights}
{cams}
  </worldbody>
  <include file="tag_floor.xml"/>
</mujoco>"""
    path = os.path.join(HERE, "_selftest_scene.xml")
    with open(path, "w") as f:
        f.write(scene)
    try:
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 900, 900)
        detector = _make_detector()
        ok = True
        for t in subset:
            r.update_scene(d, camera=f"c{t['id']}")
            gray = cv2.cvtColor(r.render(), cv2.COLOR_RGB2GRAY)
            found = sorted(x.tag_id for x in detector.detect(gray))
            hit = t["id"] in found
            ok = ok and hit
            yaw = math.degrees(yaw_of_quat_wxyz(t["quat"]))
            print(f"[selftest] tile {t['id']:>4} (map yaw {yaw:+.0f} deg) "
                  f"-> detected {found}  {'OK' if hit else 'MISS'}")
        try:
            r.close()
        except Exception:
            pass
        print("[selftest] RESULT:", "PASS" if ok else "FAIL (tiles not readable from above)")
        return ok
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=["survey", "grid", "hybrid"], default="hybrid")
    ap.add_argument("--pitch-x", type=float, default=DEF_PITCH_X)
    ap.add_argument("--pitch-y", type=float, default=DEF_PITCH_Y)
    ap.add_argument("--margin", type=float, default=DEF_MARGIN)
    ap.add_argument("--pool-width", type=float, default=DEF_VIS_POOL_WIDTH,
                    help="VISUAL pool width (X) the tag grid fills; real pool is 1.8 m")
    ap.add_argument("--pool-length", type=float, default=None,
                    help="VISUAL pool length (Y); default = real config length")
    ap.add_argument("--seabed-z", type=float, default=DEF_SEABED_Z)
    ap.add_argument("--water-depth", type=float, default=DEF_WATER_DEPTH)
    ap.add_argument("--water-surface-z", type=float, default=None,
                    help="default = seabed_z + water_depth")
    # animated water surface (heightfield) — animated at render time by water_viz.py, VISUAL ONLY
    ap.add_argument("--no-water-anim", dest="water_anim", action="store_false",
                    help="emit a flat water box instead of the animated heightfield")
    ap.add_argument("--water-hf-rows", type=int, default=48, help="hfield rows (span pool X)")
    ap.add_argument("--water-hf-cols", type=int, default=96, help="hfield cols (span pool Y)")
    ap.add_argument("--water-hf-elev", type=float, default=0.6,
                    help="hfield z half-range (m); must be >= max wave |eta| to avoid clipping")
    ap.add_argument("--water-hf-base", type=float, default=0.02, help="hfield underside skirt (m)")
    # single translucent water geom filled down to the seabed (submerged look), VISUAL ONLY
    ap.add_argument("--no-water-body", dest="water_body", action="store_false",
                    help="don't fill the column: leave only a thin wavy sheet at the surface")
    ap.add_argument("--water-alpha", type=float, default=0.18,
                    help="opacity of the water (0=invisible, ~0.18 translucent tank)")
    ap.add_argument("--px-per-module", type=int, default=DEF_PX_PER_MODULE)
    ap.add_argument("--mirror", action="store_true",
                    help="horizontally mirror tag PNGs (only if selftest says tiles read mirrored)")
    ap.add_argument("--selftest", action="store_true",
                    help="render a couple of tiles from above and confirm detection, then exit")
    args = ap.parse_args()
    if args.water_surface_z is None:
        args.water_surface_z = args.seabed_z + args.water_depth

    cfg = load_config()
    half_edge = 0.5 * cfg["tag_size"] * 10.0 / 8.0   # 0.170 -> 0.10625
    pool_w = args.pool_width if args.pool_width is not None else cfg["pool_w"]
    pool_l = args.pool_length if args.pool_length is not None else cfg["pool_l"]
    print(f"tag_size(black)={cfg['tag_size']} m  full_tile={2*half_edge:.4f} m  "
          f"visual pool={pool_w}x{pool_l} m (real {cfg['pool_w']}x{cfg['pool_l']})  "
          f"anchor=tag {cfg['anchor']}")

    if args.selftest:
        ok = run_selftest(cfg, args, half_edge)
        raise SystemExit(0 if ok else 1)

    tags, _ = build_tag_list(cfg, args.layout, args.pitch_x, args.pitch_y, args.margin,
                             pool_w=pool_w, pool_l=pool_l)
    n_survey = sum(t["src"] == "survey" for t in tags)
    n_grid = len(tags) - n_survey
    print(f"layout={args.layout}: {len(tags)} tiles ({n_survey} survey + {n_grid} grid), "
          f"seabed z={args.seabed_z}, water surface z={args.water_surface_z}")

    print("rendering + round-trip verifying PNGs ...")
    generate_pngs(tags, args.px_per_module, args.mirror)
    print(f"  {len(tags)} PNGs verified ok -> {APRILTAG_DIR}")

    frag = build_fragment_xml(cfg, tags, args, half_edge)
    with open(os.path.join(HERE, "tag_floor.xml"), "w") as f:
        f.write(frag)
    write_wrappers()
    print("wrote tag_floor.xml + scene_bluerov_tags.xml + scene_bluerov_heavy_tags.xml")
    print("enable with:  POOL_TAGS=1 ROV_MODEL=heavy python teleop.py")


if __name__ == "__main__":
    main()
