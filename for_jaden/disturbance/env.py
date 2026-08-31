#!/usr/bin/env python3
"""DisturbanceEnv — a drop-in disturbance field for hydro.py (FLU, z up).

Combines CurrentField (mean + Gauss-Markov drift) and DirectionalWaveField
(finite-depth irregular waves) and exposes EXACTLY the duck-typed interface
hydro.Hydrodynamics queries each substep:

    enabled                       (bool attribute)
    water_velocity(t, pos)  ->    (3,) world-FLU current(+drift)+wave velocity
    current_velocity()      ->    (3,) world-FLU current (diagnostics / recorder)
    wave_velocity(t, pos)   ->    (3,) world-FLU wave velocity (diagnostics)
    external_wrench(t, pos) ->    (F[3], T[3]) world-frame at COM

so `H.Hydrodynamics(model, disturbance=env).install()` works with ZERO hydro edits.

Physics split (no double-count, verified by advisors):
  * water_velocity feeds hydro's relative-velocity channel (nu_r = nu - R^T v_water)
    -> the hull D-matrix gives the wave+current DRAG, and M_A*d(nu_r)/dt gives the
    added-mass (C_a) part of the wave inertia, for free.
  * external_wrench injects ONLY the Froude-Krylov wave-inertia force
        F = rho * Vol * C_M * a_wave        (default C_M = 1, world FLU, no torque)
    i.e. the undisturbed-pressure term hydro does not model. NO separate Morison
    drag (would double-count the D-matrix). Legacy Poisson kicks are excluded.

5 modes (same seed -> wave phases + GM drift sequence are bit-identical across modes;
only the layer toggles differ, for a fair mode-by-mode comparison):
    NONE: still water                  (current off, drift off, waves off) -- baseline
    C   : mean current only            (drift off, waves off)
    CD  : mean current + drift         (waves off)
    CW  : mean current + waves         (drift off)
    CDW : mean current + drift + waves  (all on)
"""
import numpy as np

from .waves import DirectionalWaveField
from .current import CurrentField

MODES = ("NONE", "C", "CD", "CW", "CDW")


class DisturbanceEnv:
    def __init__(self, cfg, mode, seed, dt, T_sim):
        assert mode in MODES, f"mode {mode!r} not in {MODES}"
        self.cfg = cfg
        self.mode = mode
        self.seed = int(seed)
        self.dt = float(dt)
        self.T_sim = float(T_sim)
        self.use_current = mode in ("C", "CD", "CW", "CDW")   # NONE -> still water
        self.use_drift = mode in ("CD", "CDW")
        self.use_waves = mode in ("CW", "CDW")
        self.enabled = True

        # both layers always built from the SAME seed (so toggling a layer never
        # reshuffles the others); the mode just gates which contribute.
        self.current = CurrentField(
            V_bar_c=cfg.V_bar_c, theta_c=cfg.theta_c, tau=cfg.tau,
            sigma_inf=cfg.sigma_inf, v_z_range=cfg.v_z_range,
            dt=dt, T_sim=T_sim, seed=seed)
        self.waves = DirectionalWaveField(
            Hs=cfg.Hs, Tp=cfg.Tp, gamma=cfg.gamma, h=cfg.h, z_ROV=cfg.z_ROV,
            N_omega=cfg.N_omega, N_beta=cfg.N_beta,
            omega_min=cfg.omega_min, omega_max=cfg.omega_max,
            beta_bar=cfg.beta_bar, s=cfg.s, seed=seed)

        # Froude-Krylov inertia knobs
        self.rho = float(cfg.rho)
        self.vol = float(cfg.vol)
        self.C_M = float(cfg.C_M)            # default 1.0 (Froude-Krylov only)
        self.fk_mode = cfg.fk_mode           # "froude_krylov" | "morison_ca" | "off"
        if self.fk_mode == "morison_ca":
            # full Morison inertia per axis: C_M_axis = 1 + C_a, C_a = M_A/(rho*Vol).
            # (Only for the verification sweep — double-counts hydro's added mass.)
            M_A = np.asarray(cfg.added_mass_xyz, float)
            self._Cm_axis = 1.0 + M_A / (self.rho * self.vol)
        else:
            self._Cm_axis = np.array([self.C_M, self.C_M, self.C_M])

        self._t_last = 0.0                   # bridge for the no-arg current_velocity()

    # ----------------------------------------------------- hydro duck-type
    def water_velocity(self, t, pos):
        self._t_last = float(t)
        v = np.zeros(3)
        if self.use_current:
            v = (self.current.current_velocity(t) if self.use_drift
                 else self.current.mean_velocity(t))
        if self.use_waves:
            v = v + self.waves.velocity(t, pos)
        return v

    def current_velocity(self):
        """No-arg diagnostic (hydro viz + recorder): current at the last seen time."""
        if not self.use_current:
            return np.zeros(3)
        return (self.current.current_velocity(self._t_last) if self.use_drift
                else self.current.mean_velocity(self._t_last))

    def wave_velocity(self, t, pos):
        return self.waves.velocity(t, pos) if self.use_waves else np.zeros(3)

    def external_wrench(self, t, pos):
        """Froude-Krylov wave-inertia force (world FLU), no torque. Zero unless waves
        are active and fk_mode != 'off'."""
        if not (self.enabled and self.use_waves) or self.fk_mode == "off":
            return np.zeros(3), np.zeros(3)
        a_w = self.waves.acceleration(t, pos)
        F = self.rho * self.vol * self._Cm_axis * a_w
        return F, np.zeros(3)

    # ----------------------------------------------------- convenience
    def force(self, t, x_rov, v_rov):
        """Aggregate the disturbance contribution as (v_water, F_ext) in one call
        (thin wrapper over the duck-type members; does not alter the hydro path)."""
        return self.water_velocity(t, x_rov), self.external_wrench(t, x_rov)[0]

    def reset(self):
        self._t_last = 0.0
        self.waves._cache_key = None

    def summary(self):
        return (f"DisturbanceEnv mode={self.mode} seed={self.seed} "
                f"(current={self.use_current}, drift={self.use_drift}, "
                f"waves={self.use_waves}, fk={self.fk_mode}, "
                f"C_M={self._Cm_axis.round(2).tolist()})")

    def to_meta(self):
        return dict(
            schema_version=1, kind="finite_depth_env", mode=self.mode,
            seed=self.seed, enabled=bool(self.enabled), dt=self.dt, T_sim=self.T_sim,
            use_current=bool(self.use_current),
            use_drift=bool(self.use_drift), use_waves=bool(self.use_waves),
            current=self.current.to_meta(),
            waves=self.waves.to_meta(),
            froude_krylov=dict(rho=self.rho, vol=self.vol, fk_mode=self.fk_mode,
                               C_M=self.C_M, C_M_axis=self._Cm_axis.tolist()),
            note="kick_* CSV columns = wave Froude-Krylov inertia force (no Poisson kicks)",
        )
