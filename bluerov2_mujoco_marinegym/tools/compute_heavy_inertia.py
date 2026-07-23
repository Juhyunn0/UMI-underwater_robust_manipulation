#!/usr/bin/env python3
"""Derive a Heavy-specific inertia tensor from the BlueROV2 one (parallel-axis).

Why: the MarineGym/farol Heavy USD ships inertia [0.21,0.245,0.245], which is a
hand-tuned Gazebo-stability literal, not physical (see rov_model.py /
CONTROL_METHODOLOGY.md). Rather than reuse the BlueROV2 tensor verbatim as a proxy,
this computes a Heavy-specific estimate with a transparent, reproducible model.

Model:
  BlueROV2 Heavy = BlueROV2 with the VERTICAL thruster layout changed (the 4
  HORIZONTAL thrusters are at identical positions in both, so they cancel exactly in
  the BlueROV2->Heavy inertia *difference*). BlueROV2 has 2 near-centre verticals;
  Heavy has 4 corner verticals. Treating each thruster as a POINT MASS m_v at its
  site (its own ~0.0004 kg.m^2 spin inertia is negligible), the whole-vehicle inertia
  difference is exactly the parallel-axis contribution of the vertical layout change:

      I_heavy = I_bluerov2 + [ Sigma_heavy_verticals  m_v*(parallel-axis)
                             - Sigma_bluerov2_verticals m_v*(parallel-axis) ]

  point-mass parallel-axis (about the COM):
      dIxx = m*(y^2+z^2),  dIyy = m*(x^2+z^2),  dIzz = m*(x^2+y^2)

  This holds whether or not I_bluerov2 already "includes" its own thrusters, because
  the hull + 4 horizontals cancel in the difference.

m_v (point mass per vertical thruster) = 0.15 kg: MODEL-CONSISTENT with the mass
budget -- Heavy 11.5 kg - BlueROV2 11.2 kg = +0.3 kg over the (4-2)=2 extra vertical
thrusters => 0.15 kg each (so 2 removed + 4 added nets +0.3 kg, matching 11.2->11.5).
A real T200 is ~0.344 kg; the script prints that sensitivity too.

Positions are read straight from the two MJCFs (the exact sites the allocation uses),
so the result is reproducible and self-consistent. Run in `robust`.
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Standard BlueROV2 whole-vehicle inertia (bluerov.xml / MarineGym BlueROV USD).
I_BLUEROV2 = np.array([0.30375, 0.626, 0.5769])
M_THRUSTER = 0.15                 # kg per vertical thruster (model-consistent; see docstring)


def vertical_thruster_offsets(xml):
    """Positions (relative to the COM) of the VERTICAL thrusters in a model -- i.e.
    those whose thrust axis (site local +X) points along world z."""
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    com = np.array(d.subtree_com[bid])
    offs = []
    i = 0
    while True:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"thruster_{i}")
        if sid < 0:
            break
        axis = np.array(d.site_xmat[sid]).reshape(3, 3)[:, 0]   # thrust dir (local +X)
        if abs(axis[2]) > 0.9:                                  # vertical
            offs.append(np.array(d.site_xpos[sid]) - com)
        i += 1
    return np.array(offs)


def parallel_axis(offsets, m):
    """Sum of point-mass parallel-axis inertia [Ixx,Iyy,Izz] for masses m at offsets."""
    I = np.zeros(3)
    for x, y, z in offsets:
        I += m * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    return I


def heavy_inertia(m_v=M_THRUSTER):
    v_b2 = vertical_thruster_offsets("bluerov.xml")
    v_hv = vertical_thruster_offsets("bluerov_heavy.xml")
    delta = parallel_axis(v_hv, m_v) - parallel_axis(v_b2, m_v)
    return I_BLUEROV2 + delta, delta, v_b2, v_hv


if __name__ == "__main__":
    I_h, delta, v_b2, v_hv = heavy_inertia()
    print("BlueROV2 vertical thrusters (offset from COM):")
    for p in v_b2:
        print(f"   ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print("Heavy vertical thrusters (offset from COM):")
    for p in v_hv:
        print(f"   ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print(f"\nI_bluerov2        = [{I_BLUEROV2[0]:.5f}, {I_BLUEROV2[1]:.5f}, {I_BLUEROV2[2]:.5f}]")
    print(f"delta (m_v={M_THRUSTER}) = [{delta[0]:+.5f}, {delta[1]:+.5f}, {delta[2]:+.5f}]")
    print(f"I_heavy           = [{I_h[0]:.5f}, {I_h[1]:.5f}, {I_h[2]:.5f}]   <-- use this")
    # sensitivity to the thruster-mass assumption
    for mv in (0.10, 0.15, 0.20, 0.344):
        Ih, *_ = heavy_inertia(mv)
        print(f"   sensitivity m_v={mv:.3f} kg -> [{Ih[0]:.4f}, {Ih[1]:.4f}, {Ih[2]:.4f}]")
