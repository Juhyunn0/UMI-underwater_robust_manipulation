#!/usr/bin/env python3
"""Compose the heavy_gripper variant's rigid-body + buoyancy numbers (payload build-up).

BlueROV2 Heavy + Newton Subsea Gripper + MarineSitu C3 stereo camera, rigidly bolted.
Companion to tools/compute_heavy_inertia.py (same philosophy: transparent, reproducible,
parallel-axis composition instead of opaque hand-tuned literals).

Payload specs (verified 2026-07-12 against the vendor pages; see the memory note
heavy-gripper-variant / .claude/journal/research.md):
  * Newton Subsea Gripper (R2, w/ cable): 524 g in air, 267 g in water
      -> displaced volume (524-267)g / rho. Body ~cylinder D36 x 303 mm, axis +x,
      mounted on the bottom panel front, jaws forward.
      https://bluerobotics.com/store/thrusters/grippers/newton-gripper-asm-r2-rp/
  * MarineSitu C3 stereo camera: 1700 g in air, 430 g in water -> displaced 1270 g/rho.
      Box 95(L,optical) x 165(W) x 89(H) mm, mounted FRONT-BOTTOM on the centreline,
      lens FORWARD and level (0.32 deg up-tilt kept verbatim from the CAD). Position
      measured from the user's Onshape assembly via onshape-to-robot + registration
      (2026-07-19, see tools/process_c3_mesh.py docstring; replaces the earlier guessed
      front-top/45-down placement, which was wrong).
      https://bluerobotics.com/store/the-reef/marinesitu-c3-stereo-camera/

Composition model:
  * The gripper's two moving jaws (2 x JAW_MASS) live as articulated CHILD bodies in
    the MJCF (they must move to grasp), so they are EXCLUDED from the baked base_link
    inertial. Everything else (gripper body incl. cable share, camera) is rigidly
    baked into base_link's <inertial>: total mass, COM shift, and the parallel-axis
    inertia. Off-diagonals are REPORTED but the generated MJCF keeps a DIAGONAL
    inertial (see tools/gen_gripper_variant.py: fullinertia would axis-permute body_iquat
    and break hydro's body-frame drag); the C3's 8 mm lateral offset makes Ixy/Iyz
    nonzero but negligible.
  * Buoyancy: new displaced volume = heavy volume + payload displaced volumes; the
    volume-weighted CB is computed and coBM (= CB_z - COM_z, the only offset hydro.py
    models) reported. The CB_x - COM_x mismatch (hydro applies buoyancy directly above
    the COM) is reported as an unmodeled static pitch moment.

Run in `robust`:  python compute_payload_inertia.py
"""
import numpy as np

RHO = 997.0                      # kg/m^3 (fresh water; hydro.py RHO_FRESHWATER)
G = 9.81

# ---- vehicle (heavy) ----------------------------------------------------------
M_VEH = 11.5
I_VEH = np.diag([0.3291, 0.6347, 0.6109])       # about origin (= vehicle COM)
COM_VEH = np.zeros(3)
VOL_VEH = 0.0116499
COBM_VEH = 0.01                                  # CB is 1 cm above the vehicle COM

# ---- Newton gripper (rigid part = total minus the 2 moving jaws) ---------------
M_GRIP_TOTAL = 0.524             # kg in air, w/ cable (vendor)
M_GRIP_WATER = 0.267             # kg apparent in water (vendor)
JAW_MASS = 0.030                 # kg per moving jaw (2x, modeled as MJCF child bodies)
M_GRIP = M_GRIP_TOTAL - 2 * JAW_MASS
GRIP_POS = np.array([0.25, 0.0, -0.17])          # cylinder centre (bottom panel front)
GRIP_R, GRIP_L = 0.018, 0.303                    # radius, length (axis along +x)

# ---- MarineSitu C3 -------------------------------------------------------------
M_C3 = 1.700                     # kg in air (vendor)
M_C3_WATER = 0.430               # kg apparent in water (vendor)
# mesh-centroid position in the vehicle-COM frame, measured from the Onshape assembly
# (meshes/c3_payload_frames.json c3_centroid_bl; gen_gripper_variant asserts they match)
C3_POS = np.array([0.19869, 0.00800, -0.15583])  # front-bottom mount, centreline
C3_DIMS = np.array([0.095, 0.165, 0.089])        # local (x=optical depth, y=width, z=height)
C3_PITCH_DEG = 0.32                              # lens level, tiny CAD up-tilt (about +y)


def cylinder_inertia(m, r, L):
    """Solid cylinder about its COM, axis = local x."""
    Ix = 0.5 * m * r * r
    Iyz = m * (3 * r * r + L * L) / 12.0
    return np.diag([Ix, Iyz, Iyz])


def box_inertia(m, dims):
    a, b, c = dims
    return np.diag([m * (b * b + c * c) / 12.0,
                    m * (a * a + c * c) / 12.0,
                    m * (a * a + b * b) / 12.0])


def rot_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def parallel_axis(I_com, m, r):
    """Shift a COM inertia tensor to a point offset by -r (i.e. about a frame whose
    origin is at distance r from the part's COM)."""
    r = np.asarray(r, float)
    return I_com + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


JAW_POS = np.array([0.4165, 0.0, -0.17])   # jaw-pair centre at closed position (see
                                            # gen_gripper_variant.JAW_X); y cancels.


def compose():
    """Baked rigid part only (vehicle + gripper body + C3; jaws excluded)."""
    parts = [
        ("vehicle", M_VEH, COM_VEH, I_VEH),
        ("gripper", M_GRIP, GRIP_POS, cylinder_inertia(M_GRIP, GRIP_R, GRIP_L)),
        ("C3", M_C3, C3_POS, rot_y(C3_PITCH_DEG) @ box_inertia(M_C3, C3_DIMS)
                             @ rot_y(C3_PITCH_DEG).T),
    ]
    M = sum(p[1] for p in parts)
    com = sum(p[1] * p[2] for p in parts) / M
    I = np.zeros((3, 3))
    for _, m, pos, I_com in parts:
        I += parallel_axis(I_com, m, pos - com)          # about the COMPOSITE COM
    return M, com, I, parts


def compose_total():
    """Whole-vehicle composite INCLUDING the two jaws (closed, as point masses).

    Returns (m_total, com_total, I_total-about-com_total). com_total is the point the
    generated MJCF re-origins the body frame to — the whole stack (dobmpc fossen model,
    params.ZG_MASS=0, hydro force application) assumes body origin == COM, and leaving
    the origin 3.3 cm off the COM puts an unmodeled m*r rotation-translation coupling
    (~0.45 kg*m vs Ixx 0.36 kg*m^2) into the plant that destabilizes the NMPC."""
    M, com, I, _ = compose()
    m_j = 2 * JAW_MASS
    m_total = M + m_j
    com_total = (M * com + m_j * JAW_POS) / m_total
    I_total = parallel_axis(I, M, com - com_total)
    I_total += parallel_axis(np.zeros((3, 3)), m_j, JAW_POS - com_total)
    return m_total, com_total, I_total


def compose_c3():
    """Vehicle + C3 ONLY (no gripper) — the heavy_c3 variant, reflecting exactly what
    the lab's Onshape assembly contains (BROV2 Heavy + MarineSitu C3 on its bracket;
    the Newton gripper is not modeled in Onshape yet). No articulated bodies, so this
    composite IS the whole vehicle: its COM is the new body-frame origin (origin==COM,
    like heavy). Bracket mass is NOT included (visual-only, mass unknown — KNOWN_ISSUES)."""
    parts = [
        ("vehicle", M_VEH, COM_VEH, I_VEH),
        ("C3", M_C3, C3_POS, rot_y(C3_PITCH_DEG) @ box_inertia(M_C3, C3_DIMS)
                             @ rot_y(C3_PITCH_DEG).T),
    ]
    M = sum(p[1] for p in parts)
    com = sum(p[1] * p[2] for p in parts) / M
    I = np.zeros((3, 3))
    for _, m, pos, I_com in parts:
        I += parallel_axis(I_com, m, pos - com)          # about the COMPOSITE COM
    return M, com, I, parts


def buoyancy_c3():
    """Displaced volume + CB for vehicle + C3 only (no gripper)."""
    v_c3 = (M_C3 - M_C3_WATER) / RHO
    vol = VOL_VEH + v_c3
    cb = (VOL_VEH * np.array([0, 0, COBM_VEH]) + v_c3 * C3_POS) / vol
    return vol, cb, v_c3


def buoyancy():
    v_grip = (M_GRIP_TOTAL - M_GRIP_WATER) / RHO
    v_c3 = (M_C3 - M_C3_WATER) / RHO
    vol = VOL_VEH + v_grip + v_c3
    cb = (VOL_VEH * np.array([0, 0, COBM_VEH]) + v_grip * GRIP_POS + v_c3 * C3_POS) / vol
    return vol, cb, v_grip, v_c3


if __name__ == "__main__":
    M, com, I, parts = compose()
    vol, cb, v_grip, v_c3 = buoyancy()
    m_total, com_total, I_total = compose_total()
    weight = m_total * G
    buoy = RHO * G * vol
    print("parts (baked into base_link <inertial>):")
    for name, m, pos, _ in parts:
        print(f"  {name:8s} m={m:6.3f} kg  at ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")
    print(f"  jaws     m={2*JAW_MASS:6.3f} kg  (2 child bodies, NOT baked)")
    print(f"\nbaked mass      = {M:.4f} kg   (total incl. jaws = {m_total:.4f} kg)")
    print(f"baked-part COM  = ({com[0]:+.5f}, {com[1]:+.5f}, {com[2]:+.5f}) m")
    print(f"TOTAL COM       = ({com_total[0]:+.5f}, {com_total[1]:+.5f}, "
          f"{com_total[2]:+.5f}) m   <- new body-frame ORIGIN")
    print(f"inertia about baked COM (full):\n{np.array_str(I, precision=5)}")
    dT = np.diag(I_total)
    print(f"TOTAL inertia about TOTAL COM diag = [{dT[0]:.5f}, {dT[1]:.5f}, {dT[2]:.5f}]"
          f"   Ixz = {I_total[0,2]:+.5f}  (|Ixz|/Ixx = {abs(I_total[0,2])/dT[0]*100:.1f}%)")
    print(f"\ndisplaced volume = {vol:.7f} m^3   (grip {v_grip*1e3:.3f} L + C3 {v_c3*1e3:.3f} L)")
    print(f"CB               = ({cb[0]:+.5f}, {cb[1]:+.5f}, {cb[2]:+.5f}) m")
    print(f"coBM (CB_z-COM_z)= {cb[2]-com_total[2]:+.5f} m")
    print(f"net buoyancy     = {buoy - weight:+.2f} N  (B {buoy:.1f} - W {weight:.1f}; "
          f"negative = sinks, per the no-foam decision)")
    print(f"unmodeled static pitch moment |B*(CB_x-COM_x)| = "
          f"{abs(buoy*(cb[0]-com_total[0])):.2f} N*m  (hydro.py applies B above the COM)")
    print("\n--- rov_model.py registry numbers (about the TOTAL COM = new origin) ---")
    print(f'mass={m_total:.4f}, inertia=({dT[0]:.5f}, {dT[1]:.5f}, {dT[2]:.5f}), '
          f'volume={vol:.7f}')
    print("(the generated XML re-origins the body frame at the TOTAL COM; "
          "tools/gen_gripper_variant.py consumes compose()/compose_total() directly)")
