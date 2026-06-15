#!/usr/bin/env python3
"""
Extract the BlueROV visual meshes from MarineGym's USD asset into ./meshes/.

Reads the real MarineGym meshes out of the (binary) USD crate, expresses each
in the correct link-local frame, welds duplicate vertices and decimates to a
MuJoCo-friendly poly count, then writes OBJ files referenced by bluerov.xml:

    meshes/bluerov_body.obj      (base_link visual, decimated)
    meshes/bluerov_thruster.obj  (one T200 thruster, instanced 6x in the MJCF)

Asset-prep only (NOT needed to load the model). Requires: usd-core, trimesh,
fast-simplification, numpy. CPU-only; nothing CUDA/MJX/JAX.

Source: external/MarineGym/.../usd/BlueROV/BlueROV.usd  (+ instanceable_meshes.usd)
"""
import argparse
import os

import numpy as np
import trimesh
import fast_simplification
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
USD = os.path.normpath(os.path.join(
    HERE, "..", "external", "MarineGym",
    "marinegym", "robots", "assets", "usd", "BlueROV", "BlueROV.usd"))
OUT = os.path.join(HERE, "meshes")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-faces", type=int, default=40000)
    ap.add_argument("--thruster-faces", type=int, default=3000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
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
