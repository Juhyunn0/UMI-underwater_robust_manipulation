"""PID baseline (paper Sec. 5, Table 5).

Position errors are rotated into the body frame so the surge/sway/heave
channels map directly onto the control inputs u = [X_u, Y_u, Z_u, N_u].
Derivative action is taken on the measured body velocity (avoids set-point
kick); the integral term has a simple anti-windup clamp.
"""
import numpy as np

from .. import fossen
from .. import params as P


class PID:
    def __init__(self, kp=P.PID_KP, ki=P.PID_KI, kd=P.PID_KD,
                 dt=P.DT_CTRL, i_max=20.0, u_max=P.U_MAX):
        self.kp, self.ki, self.kd = (np.asarray(kp, float),
                                     np.asarray(ki, float),
                                     np.asarray(kd, float))
        self.dt, self.i_max, self.u_max = dt, i_max, np.asarray(u_max)
        self.reset()

    def reset(self):
        self.integ = np.zeros(4)

    def solve(self, x, xref):
        """x, xref: 12-dim [eta; nu] -> u (4,).

        Same call signature as NMPC.solve minus the disturbance/horizon
        arguments; xref may be the first column of the MPC reference.
        """
        eta, nu = x[:6], x[6:]
        R = fossen.rot_ib(*eta[3:6])
        e_b = R.T @ (xref[:3] - eta[:3])               # body-frame pos error
        e_psi = fossen.wrap_angle(xref[5] - eta[5])
        e = np.array([e_b[0], e_b[1], e_b[2], e_psi])

        self.integ = np.clip(self.integ + e * self.dt,
                             -self.i_max, self.i_max)
        vel_meas = np.array([nu[0], nu[1], nu[2], nu[5]])
        u = self.kp * e + self.ki * self.integ - self.kd * vel_meas
        return np.clip(u, -self.u_max, self.u_max)
