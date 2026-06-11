"""Control allocation (paper Sec. 2.2.4).

The MPC outputs u = [X_u, Y_u, Z_u, N_u] (N, N, N, Nm).  Thrusts are obtained
with the minimum-norm allocation t = pinv(K_act) u over the four actuated
rows of the propulsion matrix K, then saturated at the T200 limit.  The
resulting *actual* body wrench tau = K t includes the small, unmodelled
roll/pitch coupling of the real thruster layout - exactly the kind of
parametric mismatch the EAOB is supposed to absorb.
"""
import numpy as np

from . import params as P

_ACT_ROWS = (0, 1, 2, 5)                       # X, Y, Z, N
_K_ACT = P.K_PROP[_ACT_ROWS, :]                # (4,6)
_ALLOC = np.linalg.pinv(_K_ACT)                # (6,4): u -> t


def thrusts_from_u(u):
    """u (4,) -> per-thruster forces t (6,), saturated."""
    t = _ALLOC @ np.asarray(u, dtype=float)
    return np.clip(t, -P.T200_MAX_THRUST, P.T200_MAX_THRUST)


def wrench_from_thrusts(t):
    """t (6,) -> body wrench tau (6,) about the body origin (Eq. 19)."""
    return P.K_PROP @ np.asarray(t, dtype=float)


def wrench_from_u(u):
    """Full pipeline: commanded u -> actual applied body wrench."""
    return wrench_from_thrusts(thrusts_from_u(u))


def u_to_tau_model(u):
    """Idealised mapping used inside the MPC / EAOB prediction model:
    tau = [u1, u2, u3, 0, 0, u4]."""
    return np.array([u[0], u[1], u[2], 0.0, 0.0, u[3]])
