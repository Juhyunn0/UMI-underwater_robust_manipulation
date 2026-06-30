#!/usr/bin/env python3
"""Ocean current: uniform mean + low-frequency drift (FLU, z up).

  v_c(t) = (V_bar_c + Vtilde(t)) * [cos theta_c, sin theta_c, 0] + [0, 0, v_z]

  * Mean         : constant horizontal speed V_bar_c at heading theta_c.
  * Drift Vtilde : a scalar 1st-order Gauss-Markov process modulating the horizontal
    speed, EXACT-discretised
        Vtilde[k+1] = a*Vtilde[k] + b*z_k,   z_k ~ N(0,1),  Vtilde[0] = 0
        a = exp(-dt/tau),  b = sigma_inf*sqrt(1 - exp(-2 dt/tau))
    so its stationary std -> sigma_inf and correlation time -> tau. (Horizontal
    only; there is NO vertical drift noise.)
  * Vertical v_z : a small CONSTANT drawn once ~ U[v_z_range] (not per-step white
    noise — dimension/units kept clean, the z channel is a slow bias).

Reproducibility: the whole Vtilde sequence over [0, T_sim] is precomputed from
`seed` at construction (like disturbances._gen_kicks), so current_velocity(t) is a
pure function of t and the sequence is identical across modes / controllers /
the hydro callback's twice-per-substep queries.
"""
import numpy as np


class CurrentField:
    def __init__(self, V_bar_c, theta_c, tau, sigma_inf, v_z_range,
                 dt, T_sim, seed=0):
        self.V_bar_c = float(V_bar_c)
        self.theta_c = float(theta_c)                    # radians
        self.tau = float(tau)
        self.sigma_inf = float(sigma_inf)
        self.dt = float(dt)
        self.T_sim = float(T_sim)
        self.seed = int(seed)
        self._dir = np.array([np.cos(self.theta_c), np.sin(self.theta_c), 0.0])

        rng = np.random.default_rng(seed)
        self.v_z = float(rng.uniform(*v_z_range))        # constant small vertical bias

        # exact-discretisation Gauss-Markov, precomputed over the horizon
        a = np.exp(-self.dt / self.tau) if self.tau > 0 else 0.0
        b = self.sigma_inf * np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * self.dt / self.tau))) \
            if self.tau > 0 else 0.0
        self.a, self.b = float(a), float(b)
        n = int(np.ceil(self.T_sim / self.dt)) + 2
        z = rng.standard_normal(n)
        v = np.empty(n)
        v[0] = 0.0
        for k in range(1, n):
            v[k] = a * v[k - 1] + b * z[k - 1]
        self._Vtilde = v
        self._n = n

    def _idx(self, t):
        return int(np.clip(round(float(t) / self.dt), 0, self._n - 1))

    def mean_velocity(self, t=0.0):
        """Mean current + constant vertical (drift EXCLUDED)."""
        return self.V_bar_c * self._dir + np.array([0.0, 0.0, self.v_z])

    def current_velocity(self, t):
        """Full current at time t: (V_bar_c + Vtilde(t))*dir + [0,0,v_z]."""
        Vt = self._Vtilde[self._idx(t)]
        return (self.V_bar_c + Vt) * self._dir + np.array([0.0, 0.0, self.v_z])

    def mean(self):
        """Mean horizontal current vector (for plot legends)."""
        return self.V_bar_c * self._dir

    def to_meta(self):
        return dict(
            V_bar_c=self.V_bar_c, theta_c_deg=float(np.degrees(self.theta_c)),
            tau=self.tau, sigma_inf=self.sigma_inf, v_z=self.v_z,
            a=self.a, b=self.b, dt=self.dt, T_sim=self.T_sim, seed=self.seed,
        )
