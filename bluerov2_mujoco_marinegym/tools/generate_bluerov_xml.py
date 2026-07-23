#!/usr/bin/env python3
"""
Generate bluerov.xml (MJCF) from MarineGym's authoritative BlueROV.usd.

This reads the exact rigid-body mass/inertia and the 6 thruster (rotor_0..5)
mount transforms straight out of MarineGym's USD asset, then writes a clean
MuJoCo MJCF that references the meshes already extracted into ./meshes/.

Run from the bluerov2_mujoco_marinegym/ directory (or anywhere; paths are
resolved relative to this file). Requires usd-core + numpy (asset-prep only;
NOT needed to load the resulting model).

Provenance: external/MarineGym/marinegym/robots/assets/usd/BlueROV/BlueROV.usd
            whose own config.yaml documents the source as
            bluerov2_description/urdf/BlueROV.urdf.
"""
import os
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.normpath(os.path.join(
    HERE, "..", "external", "MarineGym",
    "marinegym", "robots", "assets", "usd", "BlueROV", "BlueROV.usd"))
OUT = os.path.join(HERE, "bluerov.xml")

stage = Usd.Stage.Open(USD)
xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

# ---- rigid-body mass / inertia (base_link) ----
base = stage.GetPrimAtPath("/BlueROV/base_link")
mapi = UsdPhysics.MassAPI(base)
mass = float(mapi.GetMassAttr().Get())
com = list(mapi.GetCenterOfMassAttr().Get())
diag = list(mapi.GetDiagonalInertiaAttr().Get())

# ---- 6 thruster mount transforms (pos + quat wxyz) in base_link frame ----
# base_link is at the world origin with identity orientation, so each rotor's
# local-to-world transform already expresses it in the base_link frame.
thrusters = []
for i in range(6):
    p = stage.GetPrimAtPath(f"/BlueROV/rotor_{i}")
    m = xcache.GetLocalToWorldTransform(p)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()      # USD quat: real + imaginary(x,y,z)
    im = q.GetImaginary()
    quat = np.array([q.GetReal(), im[0], im[1], im[2]], dtype=float)
    quat /= np.linalg.norm(quat)
    thrusters.append((np.array([t[0], t[1], t[2]]), quat))

def v3(a): return " ".join(f"{x:.6g}" for x in a)
def v4(a): return " ".join(f"{x:.6g}" for x in a)

# thruster geom (mesh prototype was extracted in rotor_0-local frame, so each
# instance is placed with that rotor's pos+quat) and a co-located named site
# whose local +X axis is the thrust direction (MarineGym applies thrust along
# the rotor body-frame X axis; reaction spin is about its Z axis).
thruster_blocks = []
for i, (pos, quat) in enumerate(thrusters):
    thruster_blocks.append(
f'''      <!-- thruster {i} -->
      <geom class="visual" mesh="bluerov_thruster" material="thruster" pos="{v3(pos)}" quat="{v4(quat)}"/>
      <site name="thruster_{i}" class="thruster" pos="{v3(pos)}" quat="{v4(quat)}"/>''')
thruster_xml = "\n".join(thruster_blocks)

# ---- Phase 2: one force actuator per thruster -------------------------------
# Each <general> actuator drives its thruster site with gear "1 0 0 0 0 0", i.e.
# a pure force along the site's local +X axis (= the thrust direction). ctrl is
# the thrust in NEWTONS; the throttle[-1,1] -> N mapping (T200 curve) lives in
# thrusters.py. ctrlrange is the T200 steady-state reverse/forward limit and is
# consistency-checked against thrusters.t200_thrust(-/+1) by tests/test_thrusters.py.
T200_MAX_FWD = 64.1319   # N at throttle +1  (MarineGym t200.py, clamp 3900 rpm)
T200_MAX_REV = -51.5507  # N at throttle -1
actuator_blocks = [
    f'    <general name="thr{i}" site="thruster_{i}" gear="1 0 0 0 0 0" '
    f'ctrlrange="{T200_MAX_REV:.4f} {T200_MAX_FWD:.4f}"/>'
    for i in range(6)
]
actuator_xml = "\n".join(actuator_blocks)

xml = f'''<!--
  BlueROV2 (heavy-config) rigid body imported from MarineGym's BlueROV asset.

  Provenance
  ----------
  Geometry, mass, inertia and the 6 thruster mount transforms are extracted
  verbatim from MarineGym's Isaac asset:
      external/MarineGym/.../usd/BlueROV/BlueROV.usd
  whose config.yaml documents its own source as
      bluerov2_description/urdf/BlueROV.urdf
  The visual meshes in ./meshes/ are the real MarineGym meshes (body decimated
  307785 -> 40000 faces; T200 thruster 6135 -> 2999 faces). Hydrodynamic
  coefficients + the T200 rotor config live in ./marinegym_assets/BlueROV.yaml
  and are used in the NEXT phase, not here.

  Frame
  -----
  MarineGym / URDF native frame: x forward, y left, z up (Z-up FLU).
  World is standard MuJoCo Z-up, gravity (0 0 -9.81), ON by default.

  Scope: rigid body (Phase 1) + thruster actuation (Phase 2) + hydrodynamics
  (Phase 3). Each thruster_i site's local +X axis is the thrust direction; one
  <general> force actuator ("thr0".."thr5") drives it. ctrl is thrust in NEWTONS
  (throttle[-1,1] -> N via the T200 curve in thrusters.py). MarineGym zeroes
  propeller reaction torque, so the actuators apply force only (no spin torque);
  body torque is r x F.
  Hydrodynamics (buoyancy/restoring/added-mass/drag) are applied at runtime by
  hydro.py via a passive-force callback (set_mjcb_passive); MuJoCo's built-in
  fluid stays off (density=viscosity=0). The Phase 1/2 tests zero gravity only
  to isolate thrust.
-->
<mujoco model="bluerov2_marinegym">
  <compiler angle="radian" meshdir="meshes" autolimits="true"/>

  <!-- density=0 viscosity=0: MuJoCo's built-in fluid model is OFF on purpose;
       all hydrodynamics are injected in the next phase. -->
  <option timestep="0.002" integrator="implicitfast" density="0" viscosity="0"/>

  <visual>
    <headlight ambient="0.6 0.6 0.62" diffuse="0.85 0.85 0.88" specular="0.25 0.25 0.25"/>
    <rgba haze="0.12 0.18 0.28 1"/>
    <scale framelength="0.25" framewidth="0.008"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <mesh name="bluerov_body"     file="bluerov_body.obj"/>
    <mesh name="bluerov_thruster" file="bluerov_thruster.obj"/>
    <material name="hull"      rgba="0.33 0.37 0.44 1" specular="0.35" shininess="0.3"/>
    <material name="thruster"  rgba="0.12 0.13 0.16 1" specular="0.4" shininess="0.4"/>
    <material name="collision" rgba="0.85 0.45 0.20 0.25"/>
  </asset>

  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="collision">
      <geom type="box" group="3" material="collision" contype="1" conaffinity="1"/>
    </default>
    <default class="thruster">
      <site type="cylinder" size="0.012 0.004" rgba="0.1 0.7 1 1" group="4"/>
    </default>
    <default class="ref">
      <site type="sphere" size="0.012" group="4"/>
    </default>
  </default>

  <worldbody>
    <light name="key"  pos="0.8 0.8 2.5"  dir="-0.3 -0.3 -1" directional="true"
           diffuse="0.55 0.55 0.6" specular="0.2 0.2 0.2"/>
    <light name="fill" pos="-0.8 -0.6 1.2" dir="0.4 0.3 -1" directional="true"
           diffuse="0.35 0.35 0.4" specular="0.1 0.1 0.1"/>
    <body name="base_link" pos="0 0 0">
      <freejoint name="free"/>
      <!-- mass/inertia copied verbatim from MarineGym USD (base_link). The
           explicit inertial element governs the dynamics; every geom below is
           visual/collision only and does not alter the body inertia. -->
      <inertial pos="{v3(com)}" mass="{mass:.6g}" diaginertia="{v3(diag)}"/>

      <!-- ===== body visual (real MarineGym mesh, decimated) ===== -->
      <geom class="visual" mesh="bluerov_body" material="hull"/>

      <!-- ===== collision proxy (MarineGym/Isaac box collider) ===== -->
      <geom class="collision" pos="0 0 -0.05" size="0.25 0.175 0.125"/>

      <!-- ===== 6 thrusters: real T200 mesh + named mount site ===== -->
{thruster_xml}

      <!-- reference site at the base_link origin -->
      <site name="base_origin" class="ref" pos="0 0 0" rgba="1 0.2 0.2 1"/>
    </body>
  </worldbody>

  <!-- ===== Phase 2: thruster force actuators (ctrl = thrust in N) ===== -->
  <actuator>
{actuator_xml}
  </actuator>
</mujoco>
'''

with open(OUT, "w") as f:
    f.write(xml)

print(f"wrote {OUT}")
print(f"mass={mass}  com={com}  diaginertia={diag}")
for i, (pos, quat) in enumerate(thrusters):
    print(f"thruster_{i}: pos={np.round(pos,5).tolist()}  quat(wxyz)={np.round(quat,5).tolist()}")
