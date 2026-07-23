#!/usr/bin/env python3
"""Generate bluerov_heavy_gripper.xml from bluerov_heavy.xml (payload variant).

Reads the CURRENT heavy MJCF and patches it (string surgery on unique anchors, each
asserted to appear exactly once) so the variant can never silently drift from the
heavy baseline. Never edit the emitted file by hand — rerun this script.

THE BODY FRAME IS RE-ORIGINED AT THE COMPOSITE (payload-shifted) COM. The whole stack
— the dobmpc Fossen predictor, params.ZG_MASS=0, hydro's force application — assumes
body origin == COM (true for heavy by construction). Leaving the origin 3.3 cm off the
new COM puts an unmodeled m*r rotation-translation coupling (~0.45 kg*m vs Ixx
0.36 kg*m^2) into the plant, which destabilizes the NMPC closed loop (observed:
acados MINSTEP + divergence in <1 s). So every positioned element inherited from
heavy (thruster sites, collision box, skin meshes, base_origin marker) is shifted by
-COM_total, and the payload is placed in the same shifted frame. Thruster positions
RELATIVE TO THE COM are what allocation uses — B is unchanged vs heavy up to the
physical COM shift itself.

What gets patched (numbers from compute_payload_inertia.py):
  1. <inertial> -> baked composite (vehicle + gripper body + C3, jaws excluded),
     pos = baked COM in the new frame, fullinertia about that COM.
  2. Thruster sites / collision box / colored-skin meshes / base_origin: -COM_total.
  3. Payload visuals + two articulated jaw bodies (0.030 kg each, mirrored slide
     joints) + ONE <position> actuator "gripper" at ctrl index 8 (thruster code finds
     actuators BY NAME "thr<i>", so index 8 is invisible to it).
  4. Three C3 <camera> elements (stereo pair, baseline 7.5 cm + center) at the lens
     plane, looking FORWARD and level (C3 pose measured from the user's Onshape
     assembly 2026-07-19 — see tools/process_c3_mesh.py; placement numbers come from
     meshes/c3_payload_frames.json, cross-checked against CP.C3_POS).

Run in `robust`:  python tools/gen_gripper_variant.py
"""
import json
import os
import sys
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # package root
import compute_payload_inertia as CP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "bluerov_heavy.xml")
DST = os.path.join(HERE, "bluerov_heavy_gripper.xml")

JAW_TRAVEL = 0.031            # m per jaw -> 62 mm total opening (vendor spec)
JAW_SIZE = (0.030, 0.003, 0.012)   # half-extents: 60 mm long, 6 mm thick, 24 mm tall
JAW_Y = 0.008                 # closed-position half-gap of the jaw centres
STEREO_HALF_BASE = 0.0375     # C3 stereo baseline 7.5 cm


def _sub_once(text, pattern, repl, what):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise RuntimeError(f"anchor not found (exactly once) for: {what}")
    return new


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _verify_mesh_geoms(xml_path, expect):
    """Compile the emitted XML and check each payload mesh actually RENDERS at its
    intended pose. MuJoCo re-orients every mesh to principal axes internally but
    COMPOSES that reframe back into the compiled geom pos/quat — so a geom's XML
    pos/quat apply to the mesh AS AUTHORED, and pre-baked meshes need NO quat (this
    is why the quat-less rovc_* skins always rendered correctly). Two 'cancellation'
    quats shipped before this was understood (conj(mesh_quat), then mesh_quat —
    both wrong: the latter double-applies the reframe); this guard makes any such
    regression fail the build instead of shipping a mis-oriented payload."""
    import mujoco
    import trimesh
    m = mujoco.MjModel.from_xml_path(xml_path)
    for mesh_name, (want_pos, stl) in expect.items():
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        gi = next(g for g in range(m.ngeom) if m.geom_dataid[g] == mid)
        rendered = v.astype(float) @ _quat_to_R(m.geom_quat[gi]).T + m.geom_pos[gi]
        ref = trimesh.load(os.path.join(HERE, "meshes", stl), force="mesh", process=False)
        want = np.asarray(ref.vertices) + np.asarray(want_pos)
        if not (np.allclose(rendered.min(0), want.min(0), atol=1e-3)
                and np.allclose(rendered.max(0), want.max(0), atol=1e-3)):
            raise RuntimeError(
                f"{mesh_name}: rendered bbox {rendered.min(0)}..{rendered.max(0)} != "
                f"intended {want.min(0)}..{want.max(0)} — mesh orientation regression")
    print(f"  [verify] payload meshes render at their intended pose ({len(expect)} checked)")


def _fmt(v):
    return " ".join(f"{x:.5f}" for x in v)


def build():
    M, com_baked, I, _ = CP.compose()
    m_total, com, _I_total = CP.compose_total()     # com = TOTAL COM = new origin
    src = open(SRC).read()

    header = (
        "<!-- GENERATED by tools/gen_gripper_variant.py from bluerov_heavy.xml - DO NOT EDIT "
        "BY HAND (rerun the script).\n"
        "     BlueROV2 Heavy + Newton Subsea Gripper (jaws articulated) + MarineSitu C3 "
        "stereo camera.\n"
        f"     BODY FRAME RE-ORIGINED at the composite COM (shift {_fmt(com)} from the "
        "heavy frame) so origin==COM,\n"
        "     which the dobmpc predictor / params.ZG_MASS=0 / hydro all assume. Payload "
        "baked into <inertial>;\n"
        "     hydro coeffs stay the heavy set except volume/coBM "
        "(marinegym_assets/BlueROVHeavyGripper.yaml). Net buoyancy ~-5.7 N\n"
        "     (payload, no trim foam - SINKS by design). Gripper <position> actuator is "
        "LAST (ctrl index 8), invisible to the\n"
        "     name-based thruster code. See compute_payload_inertia.py / "
        "docs/CONTROL_METHODOLOGY.md. -->\n")

    out = _sub_once(src, r'<mujoco model="[^"]*">',
                    '<mujoco model="bluerov2_heavy_gripper">\n' + header,
                    "model name")

    # 1. composite inertial: pos = baked COM in the NEW (total-COM-origined) frame.
    # DIAGONAL on purpose: with fullinertia MuJoCo diagonalizes and SORTS the principal
    # axes, and for this payload (Iyy > Izz > Ixx) that PERMUTES the inertial frame vs
    # the body frame (body_iquat != identity). hydro.py measures nu via
    # mj_objectVelocity(mjOBJ_BODY, local=1), which reports in the INERTIAL frame, then
    # applies the drag wrench via xmat (body frame) — with a permuted iquat the drag
    # axes cross and PUMP energy (observed: torque-free kick -> |q| 60 rad/s in 1.5 s).
    # The dropped Ixz is no longer negligible since the C3 moved to its measured
    # front-BOTTOM mount (2026-07-19): Ixz = +0.064 kg*m^2 (16.8% of Ixx) — a real
    # roll-yaw product the plant won't have (tracked in KNOWN_ISSUES). Keeping the
    # inertial diagonal remains REQUIRED for hydro. compute_payload_inertia.py
    ip = com_baked - com
    di = f"{I[0,0]:.5f} {I[1,1]:.5f} {I[2,2]:.5f}"
    out = _sub_once(
        out, r'<inertial pos="0 0 0" mass="11\.5" diaginertia="[^"]*"/>',
        f'<inertial pos="{_fmt(ip)}" mass="{M:.4f}" diaginertia="{di}"/>\n'
        f'      <!-- ^ composite: heavy 11.5 + gripper body 0.464 + C3 1.700 kg (jaws '
        f'2x0.030 kg are child bodies below);\n'
        f'           frame origin = TOTAL COM (pos = baked-part COM offset); DIAGONAL '
        f'inertia so body_iquat stays identity\n'
        f'           (fullinertia would axis-permute the inertial frame and break '
        f'hydro\'s body-frame drag - see tools/gen_gripper_variant.py). -->',
        "heavy inertial")

    # 2a. shift the 8 thruster sites by -com (relative-to-COM positions preserved)
    def shift_site(match):
        name, x, y, z, rest = match.groups()
        p = np.array([float(x), float(y), float(z)]) - com
        return f'<site name="{name}" class="thruster" pos="{_fmt(p)}"{rest}'
    out, n = re.subn(
        r'<site name="(thruster_\d)" class="thruster" pos="([\-\d.e]+) ([\-\d.e]+) ([\-\d.e]+)"([^/]*/>)',
        shift_site, out)
    if n != 8:
        raise RuntimeError(f"expected 8 thruster sites, patched {n}")

    # 2b. collision box
    def shift_geom(match):
        x, y, z, rest = match.groups()
        p = np.array([float(x), float(y), float(z)]) - com
        return f'<geom class="collision" pos="{_fmt(p)}"{rest}'
    out, n = re.subn(
        r'<geom class="collision" pos="([\-\d.e]+) ([\-\d.e]+) ([\-\d.e]+)"([^/]*/>)',
        shift_geom, out)
    if n != 1:
        raise RuntimeError("collision box not patched")

    # 2c. colored-skin meshes (verts are in the OLD heavy frame -> offset by -com)
    out, n = re.subn(r'(<geom class="visual"\s+mesh="rovc_\w+"\s+material="rovc_\w+")\s*/>',
                     rf'\1 pos="{_fmt(-com)}"/>', out)
    if n != 4:
        raise RuntimeError(f"expected 4 skin geoms, patched {n}")

    # 2d. base_origin marker: keep marking the OLD geometric origin
    out = _sub_once(out, r'<site name="base_origin" class="ref" pos="0 0 0"',
                    f'<site name="base_origin" class="ref" pos="{_fmt(-com)}"',
                    "base_origin site")

    # 3-4. payload subtree (positions in the NEW frame), before the base_origin comment
    # C3/mount placement from the Onshape-derived meta (tools/process_c3_mesh.py); the
    # inertia-side literal CP.C3_POS must agree with it (drift guard).
    meta = json.load(open(os.path.join(HERE, "meshes", "c3_payload_frames.json")))
    if np.linalg.norm(np.array(meta["c3_centroid_bl"]) - CP.C3_POS) > 1e-3:
        raise RuntimeError("CP.C3_POS disagrees with meshes/c3_payload_frames.json — "
                           "rerun tools/process_c3_mesh.py and update compute_payload_inertia.py")
    g = CP.GRIP_POS - com
    c3 = CP.C3_POS - com
    mt = np.array(meta["mount_centroid_bl"]) - com
    cam = np.array(meta["cam_center_bl"]) - com
    cam_xy = " ".join(f"{v:.5f}" for v in meta["cam_xyaxes"])
    jaw = CP.JAW_POS - com
    payload = f"""
      <!-- ===== PAYLOAD: Newton Subsea Gripper (visual body + articulated jaws) ===== -->
      <geom class="visual" type="cylinder" size="{CP.GRIP_R} {CP.GRIP_L/2:.4f}"
            pos="{_fmt(g)}" zaxis="1 0 0" material="thruster"/>
      <body name="jaw_left" pos="{jaw[0]:.5f} {JAW_Y} {jaw[2]:.5f}">
        <inertial pos="0 0 0" mass="{CP.JAW_MASS}" diaginertia="4e-06 1e-05 1e-05"/>
        <joint name="jaw_left" type="slide" axis="0 1 0" range="0 {JAW_TRAVEL}"
               damping="2.0"/>
        <geom name="jaw_left" type="box" size="{JAW_SIZE[0]} {JAW_SIZE[1]} {JAW_SIZE[2]}"
              material="thruster" contype="1" conaffinity="1" group="2"/>
      </body>
      <body name="jaw_right" pos="{jaw[0]:.5f} -{JAW_Y} {jaw[2]:.5f}">
        <inertial pos="0 0 0" mass="{CP.JAW_MASS}" diaginertia="4e-06 1e-05 1e-05"/>
        <joint name="jaw_right" type="slide" axis="0 1 0" range="-{JAW_TRAVEL} 0"
               damping="2.0"/>
        <geom name="jaw_right" type="box" size="{JAW_SIZE[0]} {JAW_SIZE[1]} {JAW_SIZE[2]}"
              material="thruster" contype="1" conaffinity="1" group="2"/>
      </body>

      <!-- ===== PAYLOAD: MarineSitu C3 stereo camera + C3-BR bracket (real CAD meshes,
           FRONT-BOTTOM, lens FORWARD level — measured from the Onshape assembly
           2026-07-19, meshes/c3_payload_frames.json). The STLs are pre-baked into the
           mount orientation, so the geoms need NO quat: MuJoCo composes its internal
           principal-axis reframe back into the compiled geom pose, i.e. geom pos/quat
           apply to the mesh AS AUTHORED (same reason the quat-less rovc_* skins render
           correctly). The 3 <camera>s below sit at the LENS PLANE (xyaxes). ===== -->
      <geom class="visual" type="mesh" mesh="c3_mount" pos="{_fmt(mt)}"
            material="c3_mount_gray"/>
      <geom class="visual" type="mesh" mesh="c3_camera" pos="{_fmt(c3)}"
            material="c3_housing"/>
      <camera name="c3_center" pos="{cam[0]:.5f} {cam[1]:.5f} {cam[2]:.5f}"
              xyaxes="{cam_xy}" fovy="52.5"/>
      <camera name="c3_left" pos="{cam[0]:.5f} {cam[1] + STEREO_HALF_BASE:.5f} {cam[2]:.5f}"
              xyaxes="{cam_xy}" fovy="55.5"/>
      <camera name="c3_right" pos="{cam[0]:.5f} {cam[1] - STEREO_HALF_BASE:.5f} {cam[2]:.5f}"
              xyaxes="{cam_xy}" fovy="55.5"/>
"""
    out = _sub_once(out, r'(\n\s*<!-- reference site at the base_link origin -->)',
                    payload + r"\1", "payload subtree insert")

    # C3 housing + bracket materials (EXACT Onshape appearance rgba, from the export's
    # robot.xml materials: housing blue 0.231/0.380/0.706, bracket gray 0.753) + meshes
    out = _sub_once(out, r'(<material name="collision"[^/]*/>)',
                    r'\1' + '\n    <material name="c3_housing" '
                    'rgba="0.231373 0.380392 0.705882 1" specular="0.35" shininess="0.4"/>'
                    '\n    <material name="c3_mount_gray" rgba="0.752941 0.752941 0.752941 1" '
                    'specular="0.4" shininess="0.4"/>'
                    '\n    <mesh name="c3_camera" file="c3_camera.stl" inertia="shell"/>'
                    '\n    <mesh name="c3_mount" file="c3_mount.stl" inertia="shell"/>',
                    "c3 material + mesh")

    # gripper actuator (LAST, index 8) + jaw mirror equality
    out = _sub_once(
        out, r'(\n\s*</actuator>)',
        '\n    <!-- gripper: ONE position actuator (ctrl index 8) drives jaw_left; '
        'the equality below mirrors jaw_right = -jaw_left. 0 = closed, '
        f'{JAW_TRAVEL} = full 62 mm opening. -->\n'
        f'    <position name="gripper" joint="jaw_left" kp="100" '
        f'ctrlrange="0 {JAW_TRAVEL}" forcerange="-30 30"/>'
        r'\1'
        '\n\n  <equality>\n'
        '    <joint joint1="jaw_right" joint2="jaw_left" polycoef="0 -1 0 0 0"/>\n'
        '  </equality>',
        "gripper actuator + equality")

    with open(DST, "w") as f:
        f.write(out)
    _verify_mesh_geoms(DST, {"c3_camera": (c3, "c3_camera.stl"),
                             "c3_mount": (mt, "c3_mount.stl")})
    print(f"wrote {os.path.basename(DST)}: origin=TOTAL COM {_fmt(com)}, baked "
          f"{M:.4f} kg at {_fmt(com_baked - com)}, total {m_total:.4f} kg")


if __name__ == "__main__":
    build()
