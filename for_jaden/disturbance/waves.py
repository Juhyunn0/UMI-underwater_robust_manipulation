#!/usr/bin/env python3
"""Directional irregular waves with FINITE-DEPTH kinematics (FLU, z up).

Builds a 2-D (frequency x direction) wave component grid from a JONSWAP spectrum
S(omega) and a cos^{2s} directional spreading D(beta), then evaluates the FLU
world-frame particle VELOCITY and (analytic) ACCELERATION at any point and time.
The (i,j) grid is flattened to M = N_omega * N_beta one-dimensional components so
every per-substep evaluation is a handful of length-M numpy ops (no Python loop).

Finite-depth (the whole point vs the legacy deep-water disturbances.py):
  * dispersion        omega^2 = g k tanh(k h)          (Newton-solved for k)
  * horizontal profile Dh(z) = cosh(k (z+h)) / sinh(k h)
  * vertical profile   Dv(z) = sinh(k (z+h)) / sinh(k h)   (-> 0 at the seabed)

Frame / depth convention (FLU): the body MuJoCo z is `pos[2]` (~0 at the model
origin). The oceanographic vertical coordinate (up from the still surface, surface
at 0, seabed at -h) is  z_oc = z_ROV + pos[2]; the height above the seabed used by
the cosh/sinh profiles is  zb = z_oc + h = z_ROV + pos[2] + h, clamped to [0, h].

Validated by disturbance/test_waves.py (dispersion residual, Hs, spectrum/spread
normalisation, velocity<->acceleration finite-difference, seabed vanishing).
"""
import numpy as np

G = 9.81


# --------------------------------------------------------------------- spectrum
def jonswap_S(omega, omega_p, gamma):
    """Unnormalised JONSWAP spectral shape S(omega) (the leading alpha*g^2 constant
    is dropped — the field renormalises to the target Hs). sigma = 0.07 below the
    peak, 0.09 above. Vectorised over `omega` (>0)."""
    omega = np.asarray(omega, float)
    sig = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-((omega - omega_p) ** 2) / (2.0 * sig ** 2 * omega_p ** 2))
    return omega ** -5.0 * np.exp(-1.25 * (omega_p / omega) ** 4) * gamma ** r


def directional_spread(beta, beta_bar, s):
    """cos^{2s} directional spreading D(beta) = G(s) cos^{2s}((beta-beta_bar)/2),
    returned UN-normalised on the discrete `beta` grid (the caller renormalises so
    sum(D)*dBeta = 1, avoiding the closed-form G(s) constant). Zero outside the
    cos>0 support so it is safe over the full [beta_bar-pi, beta_bar+pi] span."""
    beta = np.asarray(beta, float)
    half = 0.5 * (beta - beta_bar)
    c = np.cos(half)
    return np.where(c > 0.0, c ** (2.0 * s), 0.0)


def solve_wavenumber(omega, h, g=G, itmax=100, tol=1e-12):
    """Finite-depth dispersion: solve omega^2 = g k tanh(k h) for k by Newton's
    method, vectorised over `omega`. Initial guess: deep-water k0 = omega^2/g when
    k0*h > pi, else shallow k0 = omega/sqrt(g h). Returns k (same shape as omega)."""
    omega = np.asarray(omega, float)
    w2 = omega ** 2
    k0_deep = w2 / g
    k0_shallow = omega / np.sqrt(g * h)
    k = np.where(k0_deep * h > np.pi, k0_deep, k0_shallow)
    k = np.maximum(k, 1e-9)
    for _ in range(itmax):
        kh = np.clip(k * h, 0.0, 30.0)          # tanh saturates; guard cosh overflow
        th = np.tanh(kh)
        f = g * k * th - w2
        # d/dk [g k tanh(k h)] = g tanh(kh) + g k h sech^2(kh)
        sech2 = 1.0 - th ** 2
        fp = g * th + g * k * h * sech2
        step = f / fp
        k = k - step
        k = np.maximum(k, 1e-9)
        if np.all(np.abs(step) < tol):
            break
    return k


# ------------------------------------------------------------------ wave field
class DirectionalWaveField:
    """Finite-depth directional irregular wave field; query velocity / acceleration
    / elevation at (t, pos). Reproducible via `seed`."""

    def __init__(self, Hs, Tp, gamma, h, z_ROV, N_omega, N_beta,
                 omega_min, omega_max, beta_bar, s, seed=0, g=G):
        self.Hs = float(Hs)
        self.Tp = float(Tp)
        self.gamma = float(gamma)
        self.h = float(h)
        self.z_ROV = float(z_ROV)
        self.beta_bar = float(beta_bar)        # radians
        self.s = float(s)
        self.seed = int(seed)
        self.g = float(g)

        rng = np.random.default_rng(seed)
        omega_p = 2.0 * np.pi / self.Tp

        # frequency grid (regular -> needed for the i,j tensor) and JONSWAP, renormalised
        # so that m0 = sum(S_i dOmega) = (Hs/4)^2 (the spectral-moment definition of Hs).
        omega = np.linspace(omega_min, omega_max, int(N_omega))
        dOmega = float(omega[1] - omega[0])
        S = jonswap_S(omega, omega_p, self.gamma)
        m0_target = (self.Hs / 4.0) ** 2
        S *= m0_target / (S.sum() * dOmega)
        self.omega_grid, self.dOmega, self.S = omega, dOmega, S

        # direction grid spanning the full cos^{2s} support, renormalised to unit area.
        beta = np.linspace(self.beta_bar - np.pi, self.beta_bar + np.pi, int(N_beta))
        dBeta = float(beta[1] - beta[0])
        D = directional_spread(beta, self.beta_bar, self.s)
        D /= (D.sum() * dBeta)
        self.beta_grid, self.dBeta, self.D = beta, dBeta, D

        # amplitudes a_ij = sqrt(2 S_i D_j dOmega dBeta); finite-depth wavenumbers k_i.
        a_ij = np.sqrt(2.0 * S[:, None] * D[None, :] * dOmega * dBeta)   # (N_omega, N_beta)
        k = solve_wavenumber(omega, self.h, g=self.g)                    # (N_omega,)
        eps_ij = rng.uniform(0.0, 2.0 * np.pi, (int(N_omega), int(N_beta)))

        # flatten (i,j) -> m for vectorised evaluation
        OM, BE = np.meshgrid(omega, beta, indexing="ij")                # (N_omega,N_beta)
        K, _ = np.meshgrid(k, beta, indexing="ij")
        self.a_m = a_ij.ravel()
        self.omega_m = OM.ravel()
        self.k_m = K.ravel()
        self.beta_m = BE.ravel()
        self.ex_m = np.cos(BE).ravel()
        self.ey_m = np.sin(BE).ravel()
        self.eps_m = eps_ij.ravel()
        self.M = self.a_m.size

        # precompute the depth-normaliser sinh(k h) (clamped for stability)
        self._sinh_kh = np.sinh(np.clip(self.k_m * self.h, 1e-6, 30.0))

        self._cache_key = None
        self._cache = None

    # ----------------------------------------------------------- internal kin
    def _depth_factors(self, z):
        """Horizontal/vertical depth profiles at body FLU z. zb = z_ROV + z + h is
        the height above the seabed, clamped to [0, h]."""
        zb = np.clip(self.z_ROV + float(z) + self.h, 0.0, self.h)
        kzb = np.clip(self.k_m * zb, 0.0, 30.0)
        Dh = np.cosh(kzb) / self._sinh_kh
        Dv = np.sinh(kzb) / self._sinh_kh
        return Dh, Dv

    def _kin(self, t, pos):
        """Shared kinematics at (t, pos): (amp_h, amp_v, cos_theta, sin_theta).
        1-deep memo so velocity()/acceleration()/elevation() at the same (t,pos)
        (the hydro callback queries all three per substep) evaluate the trig once."""
        key = (float(t), float(pos[0]), float(pos[1]), float(pos[2]))
        if self._cache_key == key:
            return self._cache
        theta = (self.k_m * (self.ex_m * pos[0] + self.ey_m * pos[1])
                 - self.omega_m * t + self.eps_m)
        Dh, Dv = self._depth_factors(pos[2])
        amp_h = self.omega_m * self.a_m * Dh
        amp_v = self.omega_m * self.a_m * Dv
        out = (amp_h, amp_v, np.cos(theta), np.sin(theta))
        self._cache_key, self._cache = key, out
        return out

    # --------------------------------------------------------------- queries
    def velocity(self, t, pos):
        """FLU world particle velocity (3,) — finite-depth linear (Airy) theory.
        u_x = sum amp_h ex cos(theta); u_z = sum amp_v sin(theta)."""
        amp_h, amp_v, c, s = self._kin(t, pos)
        ux = float(np.sum(amp_h * self.ex_m * c))
        uy = float(np.sum(amp_h * self.ey_m * c))
        uz = float(np.sum(amp_v * s))
        return np.array([ux, uy, uz])

    def acceleration(self, t, pos):
        """FLU world particle acceleration (3,) — analytic d/dt of velocity at a fixed
        point. d/dt cos(theta) = +omega sin(theta), d/dt sin(theta) = -omega cos(theta)."""
        amp_h, amp_v, c, s = self._kin(t, pos)
        ax = float(np.sum(amp_h * self.ex_m * self.omega_m * s))
        ay = float(np.sum(amp_h * self.ey_m * self.omega_m * s))
        az = float(np.sum(amp_v * (-self.omega_m) * c))
        return np.array([ax, ay, az])

    def elevation(self, t, pos):
        """Surface elevation eta(t, x, y) = sum a_m cos(theta) (for Hs/self-check)."""
        _, _, c, _ = self._kin(t, pos)
        return float(np.sum(self.a_m * c))

    # -------------------------------------------------------------- metadata
    def to_meta(self):
        return dict(
            kind="finite_depth_directional",
            Hs=self.Hs, Tp=self.Tp, gamma=self.gamma, h=self.h, z_ROV=self.z_ROV,
            beta_bar_deg=float(np.degrees(self.beta_bar)), s=self.s, seed=self.seed,
            N_omega=int(self.omega_grid.size), N_beta=int(self.beta_grid.size),
            omega_min=float(self.omega_grid[0]), omega_max=float(self.omega_grid[-1]),
            M=int(self.M),
        )
