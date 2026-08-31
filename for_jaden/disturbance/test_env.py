#!/usr/bin/env python3
"""Unit checks for disturbance/{current,env}.py (assert-and-print).

Run:  python -m disturbance.test_env   (or  python disturbance/test_env.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disturbance.env import DisturbanceEnv, MODES
from disturbance.current import CurrentField
from disturbance.config import load_config

PASS, FAIL = 0, 0


def check(name, cond, info=""):
    global PASS, FAIL
    tag = "ok  " if cond else "FAIL"
    PASS += cond
    FAIL += (not cond)
    print(f"  [{tag}] {name}  {info}")


def _cfg():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "base.yaml")
    return load_config(path)


def test_current_gm():
    print("Gauss-Markov drift (exact discretisation):")
    c = CurrentField(V_bar_c=0.2, theta_c=0.0, tau=300.0, sigma_inf=0.02,
                     v_z_range=(-0.03, 0.03), dt=0.002, T_sim=50.0, seed=0)
    check("a = exp(-dt/tau)", abs(c.a - np.exp(-0.002 / 300.0)) < 1e-12, f"a={c.a:.6f}")
    check("Vtilde[0] = 0", c._Vtilde[0] == 0.0)
    check("mean excludes drift", np.allclose(c.mean_velocity(), [0.2, 0.0, c.v_z]))
    v5 = c.current_velocity(5.0)
    check("vertical = const v_z", abs(v5[2] - c.v_z) < 1e-12, f"vz={c.v_z:.4f}")


def test_ducktype():
    print("hydro duck-type interface:")
    cfg = _cfg()
    env = DisturbanceEnv(cfg.dist, mode="CDW", seed=0, dt=0.002, T_sim=10.0)
    for name, arity in (("water_velocity", 2), ("wave_velocity", 2),
                        ("external_wrench", 2)):
        check(f"has {name}", callable(getattr(env, name, None)))
    check("has current_velocity()", callable(getattr(env, "current_velocity")))
    check("has enabled flag", isinstance(env.enabled, bool))
    pos = np.zeros(3)
    wv = env.water_velocity(1.0, pos)
    check("water_velocity -> (3,)", wv.shape == (3,))
    F, T = env.external_wrench(1.0, pos)
    check("external_wrench -> (3,),(3,)", F.shape == (3,) and T.shape == (3,))
    check("external_wrench torque = 0", np.allclose(T, 0))


def test_mode_gating():
    print("mode gating (layers on/off):")
    cfg = _cfg(); pos = np.zeros(3)
    envs = {m: DisturbanceEnv(cfg.dist, mode=m, seed=0, dt=0.002, T_sim=10.0) for m in MODES}
    # C: no waves -> wave_velocity 0, external_wrench 0
    check("C: wave_velocity = 0", np.allclose(envs["C"].wave_velocity(2.0, pos), 0))
    check("C: external_wrench = 0", np.allclose(envs["C"].external_wrench(2.0, pos)[0], 0))
    check("CW: wave_velocity != 0", not np.allclose(envs["CW"].wave_velocity(2.0, pos), 0))
    check("CW: FK force != 0", not np.allclose(envs["CW"].external_wrench(2.0, pos)[0], 0))
    # C vs CD differ only by drift; at t=0 drift=0 so they match, later they differ
    c0 = envs["C"].water_velocity(0.0, pos); cd0 = envs["CD"].water_velocity(0.0, pos)
    check("C==CD at t=0 (drift 0)", np.allclose(c0, cd0))


def test_reproducible_across_modes():
    print("same seed -> identical wave phases & GM across modes:")
    cfg = _cfg()
    a = DisturbanceEnv(cfg.dist, mode="CW", seed=5, dt=0.002, T_sim=10.0)
    b = DisturbanceEnv(cfg.dist, mode="CDW", seed=5, dt=0.002, T_sim=10.0)
    check("wave phases identical", np.array_equal(a.waves.eps_m, b.waves.eps_m))
    check("GM sequence identical", np.array_equal(a.current._Vtilde, b.current._Vtilde))


def test_fk_consistency():
    print("Froude-Krylov force = rho*vol*C_M*a_wave:")
    cfg = _cfg(); pos = np.zeros(3)
    env = DisturbanceEnv(cfg.dist, mode="CW", seed=0, dt=0.002, T_sim=10.0)
    t = 3.7
    F = env.external_wrench(t, pos)[0]
    a_w = env.waves.acceleration(t, pos)
    expect = env.rho * env.vol * env._Cm_axis * a_w
    check("FK matches rho*vol*C_M*a", np.allclose(F, expect), f"|F|={np.linalg.norm(F):.2f}N")
    check("default C_M = 1 (froude_krylov)", np.allclose(env._Cm_axis, 1.0),
          f"C_M={env._Cm_axis.tolist()}")


def test_water_velocity_sum():
    print("water_velocity = current + wave:")
    cfg = _cfg(); pos = np.array([0.1, 0.0, 0.0]); t = 4.2
    env = DisturbanceEnv(cfg.dist, mode="CW", seed=0, dt=0.002, T_sim=10.0)
    wv = env.water_velocity(t, pos)
    expect = env.current.mean_velocity(t) + env.waves.velocity(t, pos)
    check("CW water = mean_current + wave", np.allclose(wv, expect))


if __name__ == "__main__":
    print("=== disturbance/test_env ===")
    test_current_gm()
    test_ducktype()
    test_mode_gating()
    test_reproducible_across_modes()
    test_fk_consistency()
    test_water_velocity_sum()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
