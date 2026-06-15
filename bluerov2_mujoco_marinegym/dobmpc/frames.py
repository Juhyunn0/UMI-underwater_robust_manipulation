"""FLU <-> NED/FRD boundary adapter for the DOB-MPC.

marinegym is FLU (x-fwd, y-left, z-up); the copied EAOB/NMPC run in the paper's
NED inertial + FRD body. The two are related by a single 180-deg rotation about
the shared body-x axis:

    S = diag(1, -1, -1)            # FLU <-> FRD / NED-up <-> NED-down

det(S) = +1, so S is a *proper rotation* (not a reflection): every position,
linear/angular velocity, force and moment transforms by the same S, and a body
rotation matrix transforms by the similarity (conjugation) R_ned = S R_flu S.
S is an involution (S @ S = I), so the same functions invert themselves.

We never hand-flip Euler angles (the classic subtle-bug source): we conjugate the
rotation matrix and extract ZYX Euler consistent with fossen.rot_ib.
"""
import numpy as np

from . import fossen

S = np.diag([1.0, -1.0, -1.0])


def _euler_from_R(R):
    """ZYX Euler (phi, theta, psi) from a body->world rotation matrix, matching
    fossen.rot_ib = Rz(psi) Ry(theta) Rx(phi)."""
    theta = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])


# ----------------------------------------------------------- FLU -> NED (state)
def flu_to_ned_eta(p_flu, R_flu_bw):
    """(world FLU position, body->world FLU rotation) -> eta_ned = [x y z phi theta psi]."""
    p_ned = S @ np.asarray(p_flu, float)
    R_ned = S @ np.asarray(R_flu_bw, float) @ S
    return np.concatenate([p_ned, _euler_from_R(R_ned)])


def flu_to_ned_nu(nu_flu):
    """body-frame nu = [u v w p q r] (FLU) -> FRD. Same S on the linear and the
    angular half (S is a proper rotation)."""
    nu = np.asarray(nu_flu, float)
    return np.concatenate([S @ nu[:3], S @ nu[3:]])


# ----------------------------------------------------------- NED -> FLU (inverse)
def ned_to_flu_eta(eta_ned):
    """eta_ned -> (world FLU position, body->world FLU rotation matrix)."""
    eta = np.asarray(eta_ned, float)
    p_flu = S @ eta[:3]
    R_flu = S @ fossen.rot_ib(eta[3], eta[4], eta[5]) @ S
    return p_flu, R_flu


def ned_to_flu_nu(nu_ned):
    nu = np.asarray(nu_ned, float)
    return np.concatenate([S @ nu[:3], S @ nu[3:]])


# --------------------------------------------------------------- wrench out
def ned_wrench_to_flu(tau_ned):
    """NED body wrench [X Y Z K M N] -> FLU body wrench. With the model's K=M=0
    this is [X, -Y, -Z, 0, 0, -N]."""
    t = np.asarray(tau_ned, float)
    return np.concatenate([S @ t[:3], S @ t[3:]])


flu_wrench_to_ned = ned_wrench_to_flu      # involution


# --------------------------------------------------------------- disturbance out
def ned_w_world_to_flu(w_world_ned):
    """EAOB disturbance estimate in the NED inertial frame (eaob.w_world) -> FLU
    world, for comparison against the true current/wave (FLU world)."""
    w = np.asarray(w_world_ned, float)
    return np.concatenate([S @ w[:3], S @ w[3:]])
