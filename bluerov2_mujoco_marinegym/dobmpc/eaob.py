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

Tuning profiles (2026-07-23): profile="verify" keeps the paper's unit-blind
DT-template covariances (near-deadbeat, w-error tau 7-13 ms -- only sound when
measurements are noise-free); profile="perf" (default) derives block-diagonal
R from the params.EAOB_SIG_* sensor sigmas and sets q_dist via the tau_dist ->
K -> q steady-state inversion (params.EAOB_TAU_DIST is the only knob). A
chi-square NIS gate is available but DEFAULT OFF (params.EAOB_GATE_ON) -- under
waves/maneuvers the w_dot=0 model is violated routinely, and gating blocks the
updates that track the disturbance (see params.py). See _perf_covariances().
"""
import numpy as np
from scipy.linalg import block_diag, expm

from . import fossen
from . import params as P


def _h(xa, a_meas):
    eta, nu, w = xa[:6], xa[6:12], xa[12:]
    # Note: unlike the reference code we include C_A here, keeping h()
    # consistent with the full-Fossen f() used for prediction (and with the
    # MPC model).  Omitting it makes the observer fold -C_A(nu)nu into w as
    # a phantom disturbance during sustained motion, which the MPC then
    # double-counts.
    tau_hat = (fossen.M_TOTAL @ a_meas + fossen.coriolis_rb(nu)            # fossen.M_TOTAL includes added mass by JJ 
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


def _perf_covariances():
    """Physics-derived Q, R, P0 for profile="perf" (params.EAOB_SIG_*).

    R is block-diagonal per channel. The tau pseudo-measurement noise is the
    accelerometer noise pushed through the mass matrix (h_tau = M a_meas + ...),
    so its covariance transforms as M Sigma_a M^T -- the translation/rotation
    scale is set by physics, not hand-tuning -- plus the allocation mismatch.
    q_dist comes from the scalar steady-state Kalman inversion for a random
    walk observed at gain 1 through noise R_tau:  tau_dist -> K -> q, so
    EAOB_TAU_DIST is the only knob (same time constant on all 6 axes)."""
    sig_a = np.concatenate([np.full(3, P.EAOB_SIG_ACC), np.full(3, P.EAOB_SIG_AACC)])
    R_eta = np.diag(np.concatenate([P.EAOB_SIG_POS, P.EAOB_SIG_ANG]) ** 2)
    R_nu = np.diag(np.concatenate([P.EAOB_SIG_LVEL, P.EAOB_SIG_AVEL]) ** 2)
    R_tau = (fossen.M_TOTAL @ np.diag(sig_a ** 2) @ fossen.M_TOTAL.T
             + np.diag(P.EAOB_SIG_ALLOC ** 2))
    R = block_diag(R_eta, R_nu, R_tau)
    K = 1.0 - np.exp(-P.DT_CTRL / P.EAOB_TAU_DIST)
    q_dist = np.diag(R_tau) * K ** 2 / (1.0 - K)
    Q = np.diag(np.r_[np.full(6, P.EAOB_Q_POSE_PERF),
                      np.full(6, P.EAOB_Q_VEL_PERF), q_dist])
    P0 = np.diag(P.EAOB_P0_PERF)
    return Q, R, P0


class EAOB:
    def __init__(self, eta0, nu0=np.zeros(6), w0=np.zeros(6),
                 dt=P.DT_CTRL, profile="perf",
                 q_pose=P.EAOB_Q_POSE, q_vel=P.EAOB_Q_VEL,
                 q_dist=P.EAOB_Q_DIST, r=P.EAOB_R, p0=P.EAOB_P0):
        """profile="perf" (default): sensor-sigma-derived covariances, tuned by
        params.EAOB_TAU_DIST alone; NIS gate only if params.EAOB_GATE_ON.
        profile="verify": the original near-deadbeat DT-template tuning (the
        q_*/r/p0 kwargs apply here only), no gate -- wiring checks with
        noise-free measurements."""
        assert profile in ("perf", "verify"), profile
        self.profile = profile
        self.dt = dt
        self.x = np.concatenate([eta0, nu0, w0]).astype(float)
        if profile == "perf":
            self.Q, self.R, self.P = _perf_covariances()
            self.gate = P.EAOB_NIS_GATE if P.EAOB_GATE_ON else None
        else:
            self.P = np.eye(18) * p0
            self.Q = np.diag([q_pose] * 6 + [q_vel] * 6 + [q_dist] * 6)
            self.R = np.eye(18) * r
            self.gate = None
        self.last_nis = 0.0
        self.last_zscore = np.zeros(18)      # per-channel y/sqrt(diag S) diagnostic
        self.n_upd = 0                       # update() calls
        self.n_gated = 0                     # frames rejected by the NIS gate

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
        self.n_upd += 1
        self.last_nis = float(y @ np.linalg.solve(S, y))
        self.last_zscore = y / np.sqrt(np.diag(S))     # per-channel diagnostic
        if self.gate is not None and self.last_nis > self.gate:
            # chi-square gate (perf profile): a frame this unlikely under the
            # filter's own statistics is an outlier -- keep the prediction.
            self.n_gated += 1
            self.x, self.P = x_pred, P_pred
            return self.x[:6].copy(), self.x[6:12].copy(), self.x[12:].copy()
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
