"""Finite-depth environmental disturbance package for the BlueROV2 sim (FLU).

This is the **finite-depth successor** to the repo-root ``disturbances.py``. The
legacy ``DisturbanceField`` there is deep-water (k = omega^2/g, exp(-k*depth) decay)
and is kept for the legacy DP eval (``dobmpc/eval_dp.py``); it is NOT used here.

This package models the shallow / intermediate-water target site (h = 4 m,
Tp ~= 12 s, kh ~= 0.34) properly:

  * ``waves.py``   — directional irregular waves (JONSWAP x cos^{2s} spreading),
    finite-depth dispersion omega^2 = g k tanh(k h) and cosh/sinh depth profiles,
    exposing both the particle VELOCITY (drag/added-mass channel) and the analytic
    ACCELERATION (Froude-Krylov inertia force).
  * ``current.py`` — uniform mean current + low-frequency Gauss-Markov drift
    (exact discretisation) + a small constant vertical component.
  * ``env.py``     — ``DisturbanceEnv``: a drop-in for hydro's ``disturbance=``
    (duck-types ``enabled`` / ``water_velocity`` / ``current_velocity`` /
    ``wave_velocity`` / ``external_wrench``), combining the layers into 4 modes.

Physics decomposition (verified by the underwater-robotics & control-theory
advisors): the drag from current+waves enters through hydro's existing relative-
velocity channel (nu_r = nu - R^T v_water); the ONLY extra injected force is the
Froude-Krylov wave inertia rho*Vol*a_wave (C_M = 1). No separate Morison drag
(would double-count the hull D-matrix) and no full C_M (the added-mass C_a part is
already supplied by hydro's M_A*d(nu_r)/dt). Kicks are intentionally excluded.

Everything is FLU (z up). Only numpy required (matplotlib optional, for self-check).
"""
from .waves import DirectionalWaveField, jonswap_S, directional_spread, solve_wavenumber
from .current import CurrentField
from .env import DisturbanceEnv, MODES

__all__ = [
    "DirectionalWaveField", "jonswap_S", "directional_spread", "solve_wavenumber",
    "CurrentField", "DisturbanceEnv", "MODES",
]
