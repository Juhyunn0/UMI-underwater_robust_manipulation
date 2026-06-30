#!/usr/bin/env python3
"""Unit checks for disturbance/waves.py (assert-and-print).

Run:  python -m disturbance.test_waves   (or  python disturbance/test_waves.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disturbance.waves import (DirectionalWaveField, jonswap_S, directional_spread,
                               solve_wavenumber, G)

PASS, FAIL = 0, 0


def check(name, cond, info=""):
    global PASS, FAIL
    tag = "ok  " if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}  {info}")


def make(seed=0, Hs=0.75, Tp=12.0, h=4.0, z_ROV=-3.0, N_omega=60, N_beta=21):
    return DirectionalWaveField(Hs=Hs, Tp=Tp, gamma=5.0, h=h, z_ROV=z_ROV,
                                N_omega=N_omega, N_beta=N_beta, omega_min=0.2,
                                omega_max=1.6, beta_bar=0.0, s=30, seed=seed)


def test_dispersion():
    print("dispersion omega^2 = g k tanh(k h):")
    h = 4.0
    omega = np.linspace(0.2, 1.6, 40)
    k = solve_wavenumber(omega, h)
    res = np.abs(G * k * np.tanh(k * h) - omega ** 2).max()
    check("residual < 1e-8", res < 1e-8, f"max={res:.2e}")
    # deep limit: at large omega, k -> omega^2/g if k*h > pi
    kd = solve_wavenumber(np.array([4.0]), 100.0)[0]
    check("deep limit k≈w^2/g", abs(kd - 16.0 / G) / (16.0 / G) < 1e-3, f"k={kd:.4f}")
    # shallow limit: small omega in shallow water -> k ≈ omega/sqrt(g h)
    ks = solve_wavenumber(np.array([0.05]), 2.0)[0]
    check("shallow limit k≈w/sqrt(gh)", abs(ks - 0.05 / np.sqrt(G * 2.0)) < 5e-3,
          f"k={ks:.4f}")
    # the headline number for the target site
    kt = solve_wavenumber(np.array([2 * np.pi / 12.0]), 4.0)[0]
    check("target kh≈0.34", abs(kt * 4.0 - 0.339) < 0.02, f"kh={kt*4.0:.3f}")


def test_normalization():
    print("spectrum / spreading normalization:")
    f = make()
    m0 = (f.S.sum() * f.dOmega)
    check("sum(S)dOmega = (Hs/4)^2", abs(m0 - (f.Hs / 4) ** 2) < 1e-6 * (f.Hs / 4) ** 2 + 1e-9,
          f"m0={m0:.5f} target={(f.Hs/4)**2:.5f}")
    dsum = f.D.sum() * f.dBeta
    check("sum(D)dBeta = 1", abs(dsum - 1.0) < 1e-9, f"={dsum:.6f}")


def test_Hs():
    print("realized Hs (4*std of elevation):")
    f = make(seed=1)
    pos = np.array([0.0, 0.0, 0.0])
    ts = np.arange(0.0, 1200.0, 0.1)
    eta = np.array([f.elevation(t, pos) for t in ts])
    Hs_real = 4.0 * eta.std()
    check("4*std(eta) ≈ Hs (±12%)", abs(Hs_real - f.Hs) / f.Hs < 0.12,
          f"Hs_real={Hs_real:.3f} vs {f.Hs}")


def test_vel_acc():
    print("velocity <-> acceleration (analytic d/dt):")
    f = make(seed=2)
    pos = np.array([0.1, -0.05, 0.0])
    d = 1e-4
    err = 0.0
    for t in (3.3, 17.7, 42.1):
        fd = (f.velocity(t + d, pos) - f.velocity(t - d, pos)) / (2 * d)
        an = f.acceleration(t, pos)
        err = max(err, np.abs(fd - an).max())
    check("max |FD - analytic| < 1e-3", err < 1e-3, f"max={err:.2e}")


def test_depth_profile():
    print("finite-depth profile (vertical velocity -> 0 at seabed):")
    f = make(seed=3)                       # z_ROV=-3, h=4 -> seabed at body z=-1, surface at +3
    ts = np.arange(0.0, 200.0, 0.2)
    uz_bed = np.array([f.velocity(t, np.array([0, 0, -1.0]))[2] for t in ts])   # z_oc=-h
    uz_srf = np.array([f.velocity(t, np.array([0, 0, 3.0]))[2] for t in ts])    # z_oc=0
    check("|uz| at seabed ≈ 0", np.abs(uz_bed).max() < 1e-6,
          f"max={np.abs(uz_bed).max():.2e}")
    check("|uz| surface >> seabed", uz_srf.std() > 100 * (uz_bed.std() + 1e-12),
          f"srf_std={uz_srf.std():.3e} bed_std={uz_bed.std():.3e}")


def test_reproducible():
    print("seed reproducibility:")
    a = make(seed=7); b = make(seed=7); c = make(seed=8)
    check("same seed -> identical phases", np.array_equal(a.eps_m, b.eps_m))
    check("diff seed -> different phases", not np.array_equal(a.eps_m, c.eps_m))


if __name__ == "__main__":
    print("=== disturbance/test_waves ===")
    test_dispersion()
    test_normalization()
    test_Hs()
    test_vel_acc()
    test_depth_profile()
    test_reproducible()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
