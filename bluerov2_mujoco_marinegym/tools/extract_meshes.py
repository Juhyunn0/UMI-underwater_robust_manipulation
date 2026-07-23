#!/usr/bin/env python3
"""
Extract the BlueROV visual meshes from MarineGym's USD asset into ./meshes/.

Reads the real MarineGym meshes out of the (binary) USD crate, expresses each
in the correct link-local frame, welds duplicate vertices and decimates to a
MuJoCo-friendly poly count, then writes OBJ files referenced by bluerov.xml:

    meshes/bluerov_body.obj      (base_link visual, decimated)
    meshes/bluerov_thruster.obj  (one T200 thruster, instanced 6x in the MJCF)

    --colored  ->  meshes/rov_body_{cyan,white,black,silver}.obj
                   (BlueROVHeavy body split by GeomSubset into the BlueROV color palette;
                    used by bluerov_heavy.xml's colored skin. VISUAL ONLY — dynamics come
                    from the explicit <inertial>/collision box/thruster sites; verified Delta=0.)

Asset-prep only (NOT needed to load the model). Requires: usd-core, trimesh,
fast-simplification, numpy. CPU-only; nothing CUDA/MJX/JAX.

Source: external/MarineGym/.../usd/BlueROV/BlueROV.usd (gray body/thruster) and
        .../usd/BlueROVHeavy/Props/instanceable_meshes.usd (colored skin subsets).
"""
import argparse
import os
import re

import numpy as np
import trimesh
import fast_simplification
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.normpath(os.path.join(
    HERE, "..", "external", "MarineGym",
    "marinegym", "robots", "assets", "usd", "BlueROV", "BlueROV.usd"))
# Heavy asset, used for the COLORED skin: its body mesh carries per-face GeomSubsets whose
# material NAMES encode the CAD RGB (`material_<R>_<G>_<B>`), so we can split it by color.
USD_HEAVY = os.path.normpath(os.path.join(
    HERE, "..", "external", "MarineGym", "marinegym", "robots", "assets", "usd",
    "BlueROVHeavy", "Props", "instanceable_meshes.usd"))
OUT = os.path.join(HERE, "meshes")

# Canonical BlueROV palette the CAD colors are snapped to (rgba 0-1). The subset names give
# segmentation + a rough RGB; empty shaders + CAD artifacts (e.g. pure red) make the raw
# colors unreliable, so we classify into these four for a clean, paper-like look.
PALETTE = {"cyan":   (0.16, 0.55, 0.78, 1.0),    # buoyancy foam
           "white":  (0.90, 0.90, 0.88, 1.0),    # pressure tube / domes
           "black":  (0.10, 0.10, 0.11, 1.0),    # HDPE frame + thruster shrouds
           "silver": (0.60, 0.62, 0.65, 1.0)}    # hardware / everything else


def classify_color(r, g, b):
    """Map a CAD RGB (0-255) to one of the four palette buckets."""
    mx, mn = max(r, g, b), min(r, g, b)
    if b >= r and b > 110 and g > 70 and (b - r) > 20:
        return "cyan"
    if mn > 205:
        return "white"
    if mx < 95:
        return "black"
    return "silver"


def mat_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def mesh_in_frame(stage, xcache, mesh_path, frame_path):
    """Points/faces of `mesh_path` expressed in `frame_path`'s local frame.

    USD uses row-vector convention (world = local @ M), so a point is
    transformed as p_world = [x, y, z, 1] @ M, and into the target frame via
    multiplication by inv(frame_world).
    """
    mp = stage.GetPrimAtPath(mesh_path)
    fp = stage.GetPrimAtPath(frame_path)
    m = UsdGeom.Mesh(mp)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

    faces, o = [], 0
    for c in counts:                       # fan-triangulate any n-gons
        for k in range(1, c - 1):
            faces.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    faces = np.asarray(faces, dtype=np.int64)

    mesh_w = mat_np(xcache.GetLocalToWorldTransform(mp))
    frame_w = mat_np(xcache.GetLocalToWorldTransform(fp))
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    pl = (ph @ mesh_w) @ np.linalg.inv(frame_w)
    return pl[:, :3], faces


def prep(verts, faces, target, weld_digits=5):
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.merge_vertices(digits_vertex=weld_digits)   # close the triangle soup
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    if len(m.faces) > target:
        vv, ff = fast_simplification.simplify(
            m.vertices, m.faces, target_count=target, agg=5)
        m = trimesh.Trimesh(vertices=vv, faces=ff, process=True)
    return m


def extract_colored_body(total_faces=55000):
    """Split the BlueROVHeavy body mesh (frame + foam + tube + thruster shrouds/props) into
    the four palette parts, in base_link frame, decimated. Writes meshes/rov_body_<bucket>.obj
    used by bluerov_heavy.xml's colored skin. VISUAL ONLY (the model's dynamics come from the
    explicit <inertial>, collision box, and thruster sites; verified Delta=0)."""
    st = Usd.Stage.Open(USD_HEAVY)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    mp = st.GetPrimAtPath("/BlueROVHeavy/base_link/visuals/mesh_0")
    fp = st.GetPrimAtPath("/BlueROVHeavy/base_link")
    m = UsdGeom.Mesh(mp)
    pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
    tris, poly_of_tri, o = [], [], 0
    for p, c in enumerate(counts):                 # fan-triangulate, remember source polygon
        for k in range(1, c - 1):
            tris.append((idx[o], idx[o + k], idx[o + k + 1])); poly_of_tri.append(p)
        o += c
    tris = np.asarray(tris, np.int64); poly_of_tri = np.asarray(poly_of_tri, np.int64)

    mesh_w = mat_np(xc.GetLocalToWorldTransform(mp))
    frame_w = mat_np(xc.GetLocalToWorldTransform(fp))
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    pts_local = ((ph @ mesh_w) @ np.linalg.inv(frame_w))[:, :3]

    poly_bucket = np.full(len(counts), "silver", dtype=object)
    for s in [c for c in mp.GetChildren() if c.IsA(UsdGeom.Subset)]:
        b = UsdShade.MaterialBindingAPI(s).ComputeBoundMaterial()[0]
        mm = re.match(r"material_(\d+)_(\d+)_(\d+)", b.GetPrim().GetName() if b else "")
        bucket = classify_color(*[int(x) for x in mm.groups()]) if mm else "silver"
        pi = UsdGeom.Subset(s).GetIndicesAttr().Get()
        if pi is not None:
            poly_bucket[np.asarray(pi, np.int64)] = bucket
    tri_bucket = poly_bucket[poly_of_tri]

    for bucket in ("cyan", "white", "black", "silver"):
        sel = tris[tri_bucket == bucket]
        if len(sel) == 0:
            continue
        used = np.unique(sel); remap = {v: i for i, v in enumerate(used)}
        part = prep(pts_local[used], np.vectorize(remap.get)(sel),
                    max(200, int(total_faces * len(sel) / len(tris))))
        part.export(os.path.join(OUT, f"rov_body_{bucket}.obj"))
        print(f"rov_body_{bucket:6s}.obj : {len(part.vertices):6d} v  {len(part.faces):6d} f  "
              f"rgba={PALETTE[bucket]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-faces", type=int, default=40000)
    ap.add_argument("--thruster-faces", type=int, default=3000)
    ap.add_argument("--colored", action="store_true",
                    help="also extract the COLORED BlueROVHeavy skin (rov_body_{cyan,white,"
                         "black,silver}.obj) used by bluerov_heavy.xml. VISUAL ONLY.")
    ap.add_argument("--colored-faces", type=int, default=55000,
                    help="(--colored) total target face budget across the 4 color parts")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.colored:
        extract_colored_body(args.colored_faces)
        return

    stage = Usd.Stage.Open(USD)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    v, f = mesh_in_frame(stage, xcache, "/BlueROV/base_link/visuals/mesh_0",
                         "/BlueROV/base_link")
    body = prep(v, f, args.body_faces)
    body.export(os.path.join(OUT, "bluerov_body.obj"))
    print(f"bluerov_body.obj      : {len(body.vertices):6d} v  {len(body.faces):6d} f")

    v, f = mesh_in_frame(stage, xcache, "/BlueROV/rotor_0/visuals/mesh_0",
                         "/BlueROV/rotor_0")
    thr = prep(v, f, args.thruster_faces)
    thr.export(os.path.join(OUT, "bluerov_thruster.obj"))
    print(f"bluerov_thruster.obj  : {len(thr.vertices):6d} v  {len(thr.faces):6d} f")


if __name__ == "__main__":
    main()
