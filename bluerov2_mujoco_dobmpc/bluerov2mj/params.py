"""BlueROV2 parameters.

Values taken from the paper (Hu et al., JMSE 2024, Tables 2-5) and, where the
paper is silent, from the authors' reference implementation
(https://github.com/HKPolyU-UAV/bluerov2, bluerov2_dob.h).

Conventions (Fossen, exactly as in the paper):
  - Inertial frame (IRF): NED  (x north, y east, z DOWN)
  - Body frame (BRF):     FRD  (x fwd,   y stbd, z down), origin at the
    centre of buoyancy (CB)
  - eta = [x y z phi theta psi]   (IRF position + ZYX Euler angles)
  - nu  = [u v w p q r]           (BRF linear + angular velocity)
"""
import numpy as np

GRAVITY = 9.81

# ---------------------------------------------------------------- rigid body
MASS = 11.26                 # [kg]
IX, IY, IZ = 0.3, 0.63, 0.58  # [kg m^2]
ZG = 0.02                    # CG position below CB along body z (down) [m]

WEIGHT = MASS * GRAVITY               # W  [N]
NET_BUOYANCY = 0.661618               # B - W  [N]  (repo value)
BUOYANCY = WEIGHT + NET_BUOYANCY      # B  [N]

# ------------------------------------------------------------- hydrodynamics
# Added mass (positive SNAME magnitudes), M_A = diag(ADDED_MASS)
ADDED_MASS = np.array([1.7182, 0.0, 5.468, 0.0, 1.2481, 0.4006])

# Linear damping  D_L = -diag(DL)  -> force = DL * nu  (DL entries negative)
DL = np.array([-11.7391, -20.0, -31.8678, -25.0, -44.9085, -5.0])

# Quadratic damping (repo bluerov2_dob.h, not listed in the paper)
DNL = np.array([-18.18, -21.66, -36.99, -1.55, -1.55, -1.55])

# --------------------------------------------------------------- propulsion
# Propulsion matrix K (tau = K t), exact values from bluerov2_dob.cpp.
# Columns = thrusters 1..6, rows = [X Y Z K M N].
K_PROP = np.array([
    [0.7071067811847433,  0.7071067811847433, -0.7071067811919605, -0.7071067811919605,  0.0,     0.0],
    [0.7071067811883519, -0.7071067811883519,  0.7071067811811348, -0.7071067811811348,  0.0,     0.0],
    [0.0,                 0.0,                 0.0,                 0.0,                 1.0,     1.0],
    [0.051265241636155506, -0.05126524163615552, 0.05126524163563227, -0.05126524163563227, -0.1105, 0.1105],
    [-0.05126524163589389, -0.051265241635893896, 0.05126524163641713, 0.05126524163641713, -0.0025, -0.0025],
    [0.16652364696949604, -0.16652364696949604, -0.17500892834341342, 0.17500892834341342,  0.0,    0.0],
])

T200_MAX_THRUST = 35.0       # [N] per-thruster saturation (T200 @ ~16 V)

# Control input u = [X_u, Y_u, Z_u, N_u] bounds (paper: |u_i| <= fmax / Mmax)
U_MAX = np.array([60.0, 60.0, 60.0, 12.0])    # [N, N, N, Nm]
V_MAX = 1.5                                   # |linear velocity| bound [m/s]

# ------------------------------------------------------------------- timing
DT_CTRL = 0.05               # controller / observer sample time [s] (paper)
DT_SIM = 0.002               # MuJoCo physics step [s]

# ----------------------------------------------------------------------- MPC
MPC_N = 60                   # prediction horizon (paper Table 4: 60 x 0.05 s)
# Stage weights, paper Table 4: Q = [pos(6), vel(6)], R_u for [u1..u4].
MPC_Q = np.array([300.0, 300.0, 150.0, 10.0, 10.0, 150.0,
                  10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
MPC_R = np.array([0.05, 0.05, 0.05, 0.005])
# Units note: paper Table 4 lists R = [15 15 15 0.5], but in the reference
# implementation the control variable is thrust-normalised (u_scaled ~ F/106),
# so the *effective* penalty in physical Newtons is ~1.3e-3 F^2 - essentially
# negligible.  In Newton units R = 15 makes within-horizon drifting cheaper
# than cancelling a 10 N disturbance, and even DOBMPC stops rejecting it.
# With R too large, within-horizon drifting beats cancelling even for DOBMPC
# (channel-dependent: heave's strong damping makes drift cheapest).
# R = [.05 .05 .05 .005] (~40x the paper's effective penalty) keeps every
# channel in the "cancel" regime for DOBMPC while the baseline MPC retains
# a clear gain-limited steady-state offset - the paper's central comparison.
MPC_QN = MPC_Q.copy()        # terminal weight = stage weight (paper)

# ----------------------------------------------------------------------- PID
PID_KP = np.array([5.0, 5.0, 5.0, 7.0])      # surge sway heave yaw (Table 5)
PID_KI = np.array([0.05, 0.05, 0.05, 0.1])
PID_KD = np.array([1.2, 1.2, 1.2, 0.6])

# ---------------------------------------------------------------------- EAOB
# Process noise: diag([dt^4/4 x6 (pose), dt^2 x6 (vel), dt^2 x6 (dist)]),
# measurement noise R = I*(dt^4/4)  -- exactly bluerov2_dob.cpp.
EAOB_Q_POSE = DT_CTRL ** 4 / 4.0
EAOB_Q_VEL = DT_CTRL ** 2
EAOB_Q_DIST = DT_CTRL ** 2
EAOB_R = DT_CTRL ** 4 / 4.0
EAOB_P0 = 1.0                # initial covariance = I * P0

# ------------------------------------------------- default measurement noise
MEAS_NOISE = dict(pos=0.005, ang=0.002, lin_vel=0.005, ang_vel=0.002,
                  lin_acc=0.02, ang_acc=0.02)
