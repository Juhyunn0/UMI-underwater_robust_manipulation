"""BlueROV2 parameters for the DOB-MPC, **matched to the marinegym plant**.

These REPLACE the paper's params (bluerov2_mujoco_dobmpc/bluerov2mj/params.py) so
the EAOB/NMPC prediction model reproduces *this* simulator's dynamics, leaving
only the true current/wave/kick as the disturbance w. Values come from
`marinegym_assets/BlueROV.yaml` and `bluerov.xml` (read by hydro.py).

The internal model lives in NED/FRD (the copied fossen.py convention); the
FLU<->NED transform happens at the controller boundary (frames.py). The physics
is frame-agnostic once the state is converted, so only the *magnitudes* below
need to match marinegym -- with one sign subtlety on damping (see DL/DNL).
"""
import numpy as np

import rov_model as RM     # which BlueROV variant (env ROV_MODEL); see rov_model.py

GRAVITY = 9.81

# ---------------------------------------------------------------- model select
# bluerov2 (6 thr, rank-5, NU=4) vs heavy (8 thr, rank-6 fully actuated, NU=6).
MODEL = RM.MODEL
FULLY_ACTUATED = RM.FULLY_ACTUATED

# ---------------------------------------------------------------- rigid body
# mass (bluerov2 11.2 / heavy 11.5) from the MarineGym USD. Inertia: bluerov2
# [0.30375,0.626,0.5769]; heavy [0.3291,0.6347,0.6109] DERIVED from it by parallel-axis
# of the vertical-thruster layout change (compute_heavy_inertia.py) -- the farol Heavy
# USD's [0.21,0.245,0.245] is a Gazebo-stability hand-tune, not physical (see rov_model.py
# / CONTROL_METHODOLOGY.md 2026-06-18).
MASS = RM.MASS
IX, IY, IZ = RM.INERTIA

# CB is coBM = 0.01 m ABOVE the COM (BlueROV.yaml coBM). In the NED restoring
# g(eta) (fossen.restoring), ZG>0 gives a RIGHTING moment, matching marinegym's
# CB-above-COM stable trim. Magnitude = coBM. (Verified by the equilibrium-pitch
# test: a steady surge Fx trims at sin(theta)=0.0725*Fx/(ZG*WEIGHT) -> 6 N ~ 23 deg,
# reproducing controller.py's surge->pitch relation.)
ZG = 0.01
# marinegym's rigid body has its COM AT the body origin (bluerov.xml inertial
# pos="0 0 0"), so -- unlike Fossen's M_RB (Eq.10) which couples surge<->pitch via
# m*zg when the CG is offset from the CB -- the marinegym plant has NO such inertial
# coupling. We therefore zero the m*zg off-diagonal mass terms (ZG_MASS=0) while
# keeping the buoyancy restoring (ZG=0.01, from CB-above-COM). With the coupling in,
# the model trims ~11 deg at 6 N surge; without it, ~23 deg -- matching the plant.
ZG_MASS = 0.0

# volume from the variant's yaml (bluerov2 0.0113459, heavy 0.0116499 m^3), rho 997
# (fresh water). Net buoyancy ~ +1.1 N up. Added mass / damping are identical for both.
VOLUME = RM.VOLUME
RHO = 997.0
WEIGHT = MASS * GRAVITY                       # 109.872 N
BUOYANCY = RHO * GRAVITY * VOLUME             # ~110.97 N
NET_BUOYANCY = BUOYANCY - WEIGHT              # ~+1.10 N (B - W > 0)

# ------------------------------------------------------------- hydrodynamics
# All three from BlueROV.yaml hydro_coef (order [surge sway heave roll pitch yaw]).
# Added mass: positive SNAME magnitudes, used directly as M_A = diag(ADDED_MASS),
# exactly as hydro.py applies -M_A*nudot. (heave 14.57 > mass 11.2 -- fine.)
ADDED_MASS = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12])

# Damping SIGN CONVENTION (critical): marinegym stores damping POSITIVE and applies
# the dissipative force as -(lin*nu + quad*|nu|*nu) (hydro.py:144). The copied
# fossen.damping returns -(DL*nu + DNL*|nu|*nu); to reproduce marinegym's dissipation
# we set DL,DNL NEGATIVE = -(marinegym positive coeffs). (Get this backwards and the
# prediction model is anti-damped -> unstable. Tested: open-loop predicted nu_dot must
# decelerate a moving vehicle.)
_LINEAR_DAMPING = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07])      # YAML (positive)
_QUADRATIC_DAMPING = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55])  # YAML (positive)
DL = -_LINEAR_DAMPING
DNL = -_QUADRATIC_DAMPING

# --------------------------------------------------------------- propulsion
# The MPC input u is the BODY WRENCH it asks the allocation (thrusters.py) to realize.
#   * bluerov2 (rank-5):  NU=4, u=[X,Y,Z,N]; pitch/roll moments are NOT commandable
#     (the vectored-6 allocation projects them out), so they are absent from u.
#   * heavy (rank-6, fully actuated):  NU=6, u=[X,Y,Z,K,M,N]; roll K and pitch M are
#     now directly realizable by the 4 vertical thrusters.
# Surge is bounded for safety: on bluerov2 the surge->pitch coupling My=-0.0725*Fx
# tumbles past sin(theta)=1 at Fx~15 N, so 8 N (~32 deg trim) is safe; on heavy the
# vertical thrusters cancel that coupling, but we keep 8 N for a like-for-like compare.
# Sway/heave 30 N (= PID f_max); yaw 10 Nm; roll/pitch 8 Nm (heavy only).
if FULLY_ACTUATED:
    NU = 6
    U_MAX = np.array([8.0, 30.0, 30.0, 8.0, 8.0, 10.0])   # [X, Y, Z, K, M, N]
else:
    NU = 4
    U_MAX = np.array([8.0, 30.0, 30.0, 10.0])             # [X, Y, Z, N]
V_MAX = 1.5                                    # |linear velocity| bound [m/s]

# ------------------------------------------------------------------- timing
DT_CTRL = 0.05               # observer/MPC sample time [s]  (= 25 * DT_SIM, ZOH)
DT_SIM = 0.002               # marinegym physics step [s]

# ----------------------------------------------------------------------- MPC
# Solver backend: "acados" (SQP-RTI + HPIPM C-codegen, ~2-5 ms/step, the fast
# path) or "ipopt" (CasADi/Ipopt full-convergence, ~83 ms/step, the reference &
# fallback). make_nmpc() in mpc.py falls back to ipopt if acados is unavailable.
SOLVER = "acados"
MPC_N = 60                   # prediction horizon
# Stage weights over x=[x y z, phi theta psi, u v w, p q r].
#   * bluerov2 (rank-5): roll (phi) and pitch (theta) POSITIONS are zero-weighted --
#     both are uncommanded (tau has Mx=My=0), so penalizing them is futile; pitch
#     floats to its trim and the EAOB absorbs the steady surge->pitch coupling.
#   * heavy (fully actuated): roll AND pitch are now controllable, so they get a
#     real position weight and the MPC actively levels the vehicle.
if FULLY_ACTUATED:
    MPC_Q = np.array([300.0, 300.0, 150.0,   80.0, 80.0, 150.0,
                      10.0, 10.0, 10.0,      10.0, 10.0, 10.0])
    MPC_R = np.array([0.05, 0.05, 0.05, 0.01, 0.01, 0.005])   # [X,Y,Z,K,M,N]
else:
    MPC_Q = np.array([300.0, 300.0, 150.0,   0.0, 0.0, 150.0,
                      10.0, 10.0, 10.0,      10.0, 10.0, 10.0])
    MPC_R = np.array([0.05, 0.05, 0.05, 0.005])               # [X,Y,Z,N]
MPC_QN = MPC_Q.copy()

# ---------------------------------------------------------------------- EAOB
# Process/measurement noise = paper convention (bluerov2_dob.cpp): the wd=0 model.
EAOB_Q_POSE = DT_CTRL ** 4 / 4.0
EAOB_Q_VEL = DT_CTRL ** 2
EAOB_Q_DIST = DT_CTRL ** 2
EAOB_R = DT_CTRL ** 4 / 4.0
EAOB_P0 = 1.0

# Surge->pitch geometric coupling in the marinegym plant (thrusters 0.0725 m below
# COM): My ~= -SURGE_PITCH_COUPLING * Fx (FLU). Used by the equilibrium-pitch test and,
# under option (b) below, injected into the MPC/EAOB prediction model.
SURGE_PITCH_COUPLING = 0.0725

# --------------------------------------------------- option (b): pitch-aware MPC
# Diagnosis: pitch is the dominant orientation error (RMS 10-20 deg, max 62-67 deg)
# because the rank-5 surge->pitch coupling pitches the vehicle whenever the MPC raises
# surge to track position, and option (a) neither models that coupling as a function of
# the surge *decision* nor bounds pitch. Option (b) fixes both:
#   * model the coupling in the prediction:  tau_My = +SURGE_PITCH_COUPLING * u_surge
#     (NED sign; verified by the equilibrium-pitch gate in test_dobmpc). The EAOB is fed
#     the same tau so w[pitch] -> 0 (no double-count). The MPC now foresees that more
#     surge => more pitch.
#   * bound pitch with a tighter |theta| state constraint THETA_MAX, which implicitly
#     caps the surge the MPC will plan (sin(THETA_MAX)*zg*W / kappa ~= 5.9 N at 0.40 rad)
#     -- an *optimal* surge cap, the MPC-equivalent of the PID's surge limiter.
# Toggle PITCH_AWARE=False to recover option (a) (coupling off, |theta|<=1.2, no cap).
# HEAVY is fully actuated -> pitch is a *commanded* DOF, so option-(b) (a rank-5
# workaround) is OFF and pitch is bounded only by the |theta|<=1.2 singularity margin.
PITCH_AWARE = (not FULLY_ACTUATED)
THETA_MAX = 0.40             # |pitch| prediction bound [rad] (~23 deg) when PITCH_AWARE
