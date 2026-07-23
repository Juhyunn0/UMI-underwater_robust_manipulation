#!/usr/bin/env python3
"""Process the MarineSitu C3 CAD meshes into MuJoCo-ready STLs for heavy_gripper.

Inputs : assets/CAD files/C3-BR-Camera.obj   (C3 housing; authored frame, metres)
         assets/CAD files/C3-BR-Mount.stl    (lab's C3-BR bracket, copied from the
                                              Onshape export onshape_export/assets/)
Outputs: meshes/c3_camera.stl, meshes/c3_mount.stl        (baked into the MOUNTED
         orientation, centred at their centroid)
         meshes/c3_payload_frames.json                    (placement numbers consumed
         by tools/gen_gripper_variant.py: centroids, camera lens-plane pose, optical axis)

MOUNTED POSE PROVENANCE (2026-07-19, replaces the earlier guessed front-top/45-down):
  The user's Onshape assembly (BROV2 Heavy + C3 on its bracket) was exported with
  onshape-to-robot (assets/CAD files/onshape_export/robot.xml); the vehicle geometry
  was registered to the sim base_link frame with the rotation constrained to a pure
  axis permutation (the CAD sits axis-aligned: bbox == vendor 575x254x457 mm):
      sim_x(fwd) = asm_Z,  sim_y(left) = asm_X,  sim_z(up) = asm_Y
  scale sim-mesh/CAD = 1.0233 (the MarineGym-derived skin is uniformly 2.3% large;
  payload placement below is TRUE METRIC, anchored at the COM), translation from a
  global voxel grid search + trimmed-ICP polish (residual 1.6 mm, seed spread <0.1 mm).
  Result: C3 sits FRONT-BOTTOM on the centreline, lens looking FORWARD, level
  (0.32 deg up-tilt kept verbatim from the CAD mates), stereo baseline horizontal.

  Authored C3 frame (verified earlier by rendering the 6 faces): optical(lens) = -Z,
  stereo baseline = X, mount panel = +Y.

MuJoCo still re-orients each mesh to principal-inertia axes on load (mesh_quat); the
generated MJCF cancels that with geom quat = conj(mesh_quat). Run in `robust`.
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAD = os.path.join(HERE, "..", "assets", "CAD files")
SRC_C3 = os.path.join(CAD, "C3-BR-Camera.obj")
SRC_MT = os.path.join(CAD, "C3-BR-Mount.stl")
DST_C3 = os.path.join(HERE, "meshes", "c3_camera.stl")
DST_MT = os.path.join(HERE, "meshes", "c3_mount.stl")
DST_META = os.path.join(HERE, "meshes", "c3_payload_frames.json")

# ---- frozen registration constants (see docstring) ------------------------------
# axis permutation asm -> base_link, and the assembly point that maps to the origin
R0 = np.array([[0.0, 0.0, 1.0],
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0]])
C_ASM = np.array([-0.16079, 0.22043, -0.11281])   # = R0^T @ (-t)/scale of the fit

# part poses in the assembly frame, copied from onshape_export/robot.xml geoms
P_C3_ASM = np.array([-0.155322, 0.0650249, 0.120326])
Q_C3_ASM = np.array([0.00279879, 0.999996, 0.0, -0.0])
P_MT_ASM = np.array([-0.155322, 0.0376153, 0.0372597])
Q_MT_ASM = np.array([0.884011, 0.467467, 0.0, -0.0])

Z_LENS_FACE = -0.00635    # authored-frame z of the lens face (front of the housing)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def bake(src, dst, R, label):
    """Load, clean, rotate into the mounted (base_link) orientation, centre at the
    centroid, export. Returns the centroid of the ROTATED mesh (= R @ centroid_local),
    for placement bookkeeping."""
    m = trimesh.load(src, force="mesh", process=True)
    m.update_faces(m.nondegenerate_faces())
    m.fix_normals()
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, "must be a proper rotation"
    T = np.eye(4)
    T[:3, :3] = R
    m.apply_transform(T)
    cen = np.asarray(m.centroid).copy()
    m.apply_translation(-cen)
    m.export(dst)
    print(f"wrote {os.path.relpath(dst, HERE)}: {len(m.faces)} tris, extent(m) "
          f"{np.round(m.extents, 4)}  ({label})")
    return cen


def build():
    R_c3 = R0 @ quat_to_R(Q_C3_ASM)
    R_mt = R0 @ quat_to_R(Q_MT_ASM)
    p_c3_bl = R0 @ (P_C3_ASM - C_ASM)     # part-frame origin in base_link (metric)
    p_mt_bl = R0 @ (P_MT_ASM - C_ASM)

    cen_c3_rot = bake(SRC_C3, DST_C3, R_c3, "C3: lens fwd level, baseline horizontal")
    cen_mt_rot = bake(SRC_MT, DST_MT, R_mt, "C3-BR mount bracket")

    c3_centroid_bl = p_c3_bl + cen_c3_rot
    mount_centroid_bl = p_mt_bl + cen_mt_rot
    optical = R_c3 @ np.array([0.0, 0.0, -1.0])
    baseline = R_c3 @ np.array([1.0, 0.0, 0.0])
    # stereo lens-plane centre: authored [0, 0, Z_LENS_FACE] (housing centre, lens face)
    cam_center_bl = p_c3_bl + R_c3 @ np.array([0.0, 0.0, Z_LENS_FACE])
    # MuJoCo camera frame (looks along -Z_cam, +Y_cam = image up):
    # image right when looking along +optical with up ~ +z_body  =>  X_cam = -baseline
    x_cam = -baseline
    y_cam = np.cross(-optical, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    meta = {
        "c3_centroid_bl": c3_centroid_bl.round(5).tolist(),
        "mount_centroid_bl": mount_centroid_bl.round(5).tolist(),
        "cam_center_bl": cam_center_bl.round(5).tolist(),
        "optical_bl": optical.round(5).tolist(),
        "baseline_dir_bl": baseline.round(5).tolist(),
        "cam_xyaxes": np.concatenate([x_cam, y_cam]).round(5).tolist(),
        "provenance": "onshape_export/robot.xml + registration 2026-07-19 "
                      "(tools/process_c3_mesh.py; metric, base_link origin = vehicle COM)",
    }
    with open(DST_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {os.path.relpath(DST_META, HERE)}:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    build()
