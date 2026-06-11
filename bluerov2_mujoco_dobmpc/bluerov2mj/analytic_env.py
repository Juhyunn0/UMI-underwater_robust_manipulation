"""Pure-NumPy RK4 integration of the full Fossen model (Eq. 22).

Same interface as BlueROV2MujocoEnv.  Used as ground truth to validate the
MuJoCo hydrodynamic-force injection, and as a lightweight plant for rapid
controller prototyping.
"""
import numpy as np

from . import allocation, fossen
from . import params as P


class BlueROV2AnalyticEnv:
    def __init__(self, dt_ctrl=P.DT_CTRL, meas_noise=None, seed=0,
                 n_sub=25):
        self.dt_ctrl = dt_ctrl
        self.n_sub = n_sub
        self.dt_sub = dt_ctrl / n_sub
        self.rng = np.random.default_rng(seed)
        self.noise = dict(P.MEAS_NOISE) if meas_noise is None else meas_noise
        self.reset()

    def reset(self, eta0=(0, 0, -20, 0, 0, 0), nu0=(0, 0, 0, 0, 0, 0)):
        self.x = np.concatenate([np.asarray(eta0, float),
                                 np.asarray(nu0, float)])
        self._nu_prev = self.x[6:].copy()
        self.t = 0.0
        return self.get_measurement()

    def step(self, u_cmd, w_world=np.zeros(6)):
        u_cmd = np.clip(np.asarray(u_cmd, float), -P.U_MAX, P.U_MAX)
        tau_b = allocation.wrench_from_u(u_cmd)
        for _ in range(self.n_sub):
            R = fossen.rot_ib(*self.x[3:6])
            w_body = np.concatenate([R.T @ w_world[:3], R.T @ w_world[3:]])
            self.x = fossen.rk4(fossen.f_state, self.x, self.dt_sub,
                                tau_b, w_body)
        self.t += self.dt_ctrl
        nu = self.x[6:]
        nudot = (nu - self._nu_prev) / self.dt_ctrl
        self._nu_prev = nu.copy()
        return self.get_measurement(nudot), self.x.copy()

    def get_true_state(self):
        return self.x.copy()

    def get_measurement(self, nudot=None):
        n = self.noise
        eta_m = self.x[:6] + np.concatenate([
            self.rng.normal(0, n["pos"], 3), self.rng.normal(0, n["ang"], 3)])
        nu_m = self.x[6:] + np.concatenate([
            self.rng.normal(0, n["lin_vel"], 3),
            self.rng.normal(0, n["ang_vel"], 3)])
        if nudot is None:
            nudot = np.zeros(6)
        nudot_m = nudot + np.concatenate([
            self.rng.normal(0, n["lin_acc"], 3),
            self.rng.normal(0, n["ang_acc"], 3)])
        return dict(eta=eta_m, nu=nu_m, nudot=nudot_m, t=self.t)
