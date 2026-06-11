"""Extended Active Observer (EAOB), paper Sec. 3 / bluerov2_dob.cpp.

Augmented state  xa = [eta(6); nu(6); w(6)]  with internal disturbance model
w_dot = 0 (Eq. 24).  Predict with RK4 over the full Fossen model (Eq. 26);
update with measurements

    z = [eta_meas; nu_meas; tau_applied]                        (Eq. 27)

where the expected propulsion "measurement" is reconstructed from the
measured body acceleration (Eq. 28):

    h_tau(xa) = M a_meas + C_RB(nu)nu + D(nu)nu + g(eta) - w.

The -I dependence of h_tau on w is what renders the disturbance state
observable and gives the near-deadbeat estimation seen in the paper.
Jacobians F, H are evaluated numerically at each step (Eq. 32-33).
"""
import numpy as np
from scipy.linalg import expm

from . import fossen
from . import params as P


def _h(xa, a_meas):
    eta, nu, w = xa[:6], xa[6:12], xa[12:]
    # Note: unlike the reference code we include C_A here, keeping h()
    # consistent with the full-Fossen f() used for prediction (and with the
    # MPC model).  Omitting it makes the observer fold -C_A(nu)nu into w as
    # a phantom disturbance during sustained motion, which the MPC then
    # double-counts.
    tau_hat = (fossen.M_TOTAL @ a_meas + fossen.coriolis_rb(nu)
               + fossen.coriolis_added(nu)
               + fossen.damping(nu) + fossen.restoring(eta) - w)
    return np.concatenate([eta, nu, tau_hat])


def _num_jac(fun, x, *args, eps=1e-6):
    f0 = fun(x, *args)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        J[:, i] = (fun(xp, *args) - f0) / eps
    return J


class EAOB:
    def __init__(self, eta0, nu0=np.zeros(6), w0=np.zeros(6),
                 dt=P.DT_CTRL,
                 q_pose=P.EAOB_Q_POSE, q_vel=P.EAOB_Q_VEL,
                 q_dist=P.EAOB_Q_DIST, r=P.EAOB_R, p0=P.EAOB_P0):
        self.dt = dt
        self.x = np.concatenate([eta0, nu0, w0]).astype(float)
        self.P = np.eye(18) * p0
        self.Q = np.diag([q_pose] * 6 + [q_vel] * 6 + [q_dist] * 6)
        self.R = np.eye(18) * r

    def update(self, meas, tau_applied):
        """One predict-update cycle.

        meas:         dict with 'eta', 'nu', 'nudot' (from the plant)
        tau_applied:  body wrench actually commanded this step (6,)
        Returns (eta_hat, nu_hat, w_hat).
        """
        # ---- prediction (Eq. 36).  The roll/pitch subsystem is stiff
        # (time constants ~12-14 ms), so the mean is integrated with RK4
        # substeps and the covariance with the exact transition matrix
        # expm(F dt); a single 50 ms explicit step diverges.
        Fc = _num_jac(fossen.f_ekf, self.x, tau_applied)
        Phi = expm(self.dt * Fc)
        x_pred = self.x
        h = self.dt / 4
        for _ in range(4):
            x_pred = fossen.rk4(fossen.f_ekf, x_pred, h, tau_applied)
        P_pred = Phi @ self.P @ Phi.T + self.Q

        # ---- update (Eq. 37)
        z = np.concatenate([meas["eta"], meas["nu"], tau_applied])
        a_meas = meas["nudot"]
        y = z - _h(x_pred, a_meas)
        y[3:6] = fossen.wrap_angle(y[3:6])             # angle innovations
        H = _num_jac(_h, x_pred, a_meas)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        IKH = np.eye(18) - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T   # Joseph form

        return self.x[:6].copy(), self.x[6:12].copy(), self.x[12:].copy()

    def w_world(self, eta=None):
        """Estimated disturbance rotated to the inertial frame (for plots,
        as in the reference implementation)."""
        eta = self.x[:6] if eta is None else eta
        R = fossen.rot_ib(*eta[3:6])
        w = self.x[12:]
        return np.concatenate([R @ w[:3], R @ w[3:]])
