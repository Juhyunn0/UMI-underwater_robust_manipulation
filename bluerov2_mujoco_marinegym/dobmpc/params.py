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
# of the vertical-thruster layout change (tools/compute_heavy_inertia.py) -- the farol Heavy
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
# Surge bound -- PER VARIANT, because the reason for the old 8 N cap is rank-5 only:
#   * bluerov2 (rank-5): the surge->pitch coupling My=-0.0725*Fx tumbles past
#     sin(theta)=1 at Fx~15 N, so 8 N (~32 deg trim) is a REAL physical limit. Keep.
#   * heavy (rank-6): the 4 vertical thrusters cancel that coupling, so the cap has no
#     physical basis here. It used to sit at 8 N "for a like-for-like compare" -- which
#     was exactly backwards: the PID's f_max is 30 N on every axis, so the 8 N box
#     handed the NMPC a 3.75x surge handicap and the benchmark was NOT like-for-like.
#     Raised to 30 N on 2026-07-24 (= PID f_max). Measured effect (storm CW, 10 laps,
#     4 paired headings): the box was active on 60% of ticks (dobmpc 75%) at storm vs
#     7% at gentle, and releasing it moved PID-minus-MPC from -4.24 cm to +8.60 cm --
#     i.e. the whole "PID wins in strong waves" crossover was this constraint. Still
#     physically realizable: peak per-thruster demand 36.1 N (LOWER than the 36.9 N the
#     8 N box provoked) vs a T200 ctrlrange of -51.6..+64.1 N, sat_freq stays 0.
#     RECORDS BOUNDARY: mpc/dobmpc runs recorded BEFORE 2026-07-24 used 8 N -- run meta
#     now stamps controller.u_max, and results across that boundary must not be pooled.
# Sway/heave 30 N (= PID f_max); yaw 10 Nm; roll/pitch 8 Nm (heavy only).
if FULLY_ACTUATED:
    NU = 6
    U_MAX = np.array([30.0, 30.0, 30.0, 8.0, 8.0, 10.0])  # [X, Y, Z, K, M, N]
else:
    NU = 4
    U_MAX = np.array([8.0, 30.0, 30.0, 10.0])             # [X, Y, Z, N] (rank-5: real)
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
# Two tuning profiles (eaob.EAOB(profile=...)):
#   * "verify" -- the original paper-convention (bluerov2_dob.cpp) unit-blind
#     DT templates below. With noise-free sim measurements this is near-deadbeat
#     (w-error time constant 7-13 ms; measured 2026-07-23) -- keep it for wiring
#     checks only, it is NOT a realistic sensor assumption.
#   * "perf" (default) -- physics-derived covariances from the sensor sigmas
#     below; the SAME sigmas drive the sim-side measurement-noise injection
#     (dobmpc_controller meas_noise), so filter R == injected noise by
#     construction. Tuning knob: EAOB_TAU_DIST only (tau -> K -> q_dist chain
#     in eaob._perf_covariances; do NOT hand-pick q_dist).
EAOB_Q_POSE = DT_CTRL ** 4 / 4.0
EAOB_Q_VEL = DT_CTRL ** 2
EAOB_Q_DIST = DT_CTRL ** 2
EAOB_R = DT_CTRL ** 4 / 4.0
EAOB_P0 = 1.0

# Sensor sigmas (single source of truth for BOTH the sim-side injection and the
# perf-profile filter R). Values model a GOOD visual-inertial / SLAM + DVL + depth
# stack. Reduced 2026-07-24: the earlier x/y 5 cm SAT ABOVE the ~1.2 cm closed-loop
# tracking error, so with mpc_state_source="meas" the MPC chased its own pose jitter
# (9-15x command chatter, measured). The pose noise now sits WELL BELOW the tracking
# error (x/y 0.5 cm ~ 0.4x), which drops "meas" chatter to ~1.8x the truth-x0 level
# while tracking stays at the truth floor (verify_state_source, 2026-07-24).
# IMPORTANT: ACC/AACC/ALLOC (the disturbance pseudo-measurement channel -> R_tau,
# q_dist) are UNCHANGED, so w_hat quality / disturbance rejection is preserved; only
# the pose+vel channel that feeds the MPC x0 got tighter. angular accel is still a
# differentiated gyro (sqrt(2)/DT amplification).
EAOB_SIG_POS = np.array([0.005, 0.005, 0.003])         # m   (x,y vision/SLAM; z depth)
EAOB_SIG_ANG = np.deg2rad([0.3, 0.3, 0.5])             # rad (roll/pitch; yaw)
EAOB_SIG_LVEL = np.array([0.005, 0.005, 0.005])        # m/s (DVL / VIO)
EAOB_SIG_AVEL = np.array([0.002, 0.002, 0.002])        # rad/s (gyro)
EAOB_SIG_ACC = 0.05                                    # m/s^2 (IMU accelerometer)
EAOB_SIG_AACC = np.sqrt(2) * 0.003 / DT_CTRL           # rad/s^2 (differentiated gyro)
EAOB_SIG_ALLOC = np.array([0.3, 0.3, 0.3, 0.05, 0.05, 0.05])   # N, N*m (tau_cmd vs realized)
# The ONLY perf design knob. 0.5 s (the initial design point) failed the CDW
# consistency check (NIS 25.4 / NEES 77 -- the wave band outruns the w random
# walk); iterated 2026-07-23: 0.2 s passes CDW (NIS 17.1 / NEES 15.0), passes
# CD NIS (14.4; NEES ~9 = deliberate real-hardware conservatism), and the
# closed-loop square CDW radRMS is 3.51 cm vs the clean-deadbeat 3.09 cm.
# Slow-disturbance-only experiments may prefer 0.5-2.0 s (halves CD RMSE).
EAOB_TAU_DIST = 0.2                                    # s

# What state the NMPC consumes as x0 (applied whenever measurement noise is
# active, for BOTH mpc and dobmpc):
#   "meas"     = the state-estimator output the loop actually has -- in sim the
#                clean reading + sensor noise (the SAME sample the EAOB eats), on
#                hardware the SLAM / AI-model estimate. The DOB PLUG-IN default:
#                the EAOB hands the MPC only w_hat, never its filtered state, so
#                mpc and dobmpc share x0 and differ only by w_hat.
#   "estimate" = the EAOB-filtered eta_hat/nu_hat (offset-free-MPC style; dobmpc
#                only). The 2026-07-23 (2) default, now an explicit opt-in.
#   "truth"    = the clean plant state (regression / observer isolation; also the
#                no-noise path).
# Whatever the source, it feeds ONLY the MPC x0 + its yaw-reference anchor --
# never measurement generation, truth logging, or the EAOB's own inputs. Run
# meta records controller.mpc_state_source (absent key = truth).
MPC_STATE_SOURCE = "meas"

# What state the PID controller SEES (2026-07-24: apples-to-apples with the MPC).
#   "meas"  = the clean plant readout corrupted by the SAME Gaussian sensor model
#             the MPC x0 uses (EAOB_SIG_POS/ANG/LVEL/AVEL), sampled at the control
#             tick (DT_CTRL, 20 Hz) and HELD (ZOH) between ticks -- so the PID and
#             the (DOB-)MPC consume an identically-degraded 20 Hz state estimate and
#             differ ONLY by control law. Note this bundles the two coupled sensor
#             effects the MPC already has: Gaussian noise AND 20 Hz sampling lag.
#   "truth" = the clean 500 Hz plant readout (the pre-2026-07-24 behaviour; a
#             best-case PID baseline / regression path).
# Feeds ONLY the PID's own state reads -- truth logging (the recorded px,py and the
# radial-RMS metric) always stays on the plant, so it still measures TRUE tracking
# error. Run meta records controller.meas.state_source (absent key = truth).
PID_STATE_SOURCE = "meas"

# perf-profile Q floor (eta/nu random-walk intensities) and P0 blocks.
EAOB_Q_POSE_PERF = 1e-6
EAOB_Q_VEL_PERF = 4e-6
EAOB_P0_PERF = np.r_[np.full(6, 1e-2), np.full(6, 1e-2), np.full(3, 9.0), np.full(3, 9e-2)]

# chi-square innovation gate (perf profile only): chi2.ppf(0.999, 18). A frame
# whose NIS exceeds this keeps the prediction (reject the update) and is counted
# in eaob.n_gated. DEFAULT OFF: measured 2026-07-23 (square CDW seed-0 A/B), the
# w_dot=0 model is violated ROUTINELY by the wave band + corner maneuvers, so a
# 0.999-consistency gate rejects 25-76% of frames and blocks the very updates
# that track the waves (radRMS 4.5 -> 15.5 cm with the gate on). Keep it as an
# opt-in diagnostic; the anti-spike defense belongs to the per-axis W_HAT_CLIP
# (KNOWN_ISSUES, still open) or a harmonic-EAOB disturbance model.
EAOB_NIS_GATE = 42.31240
EAOB_GATE_ON = False

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
