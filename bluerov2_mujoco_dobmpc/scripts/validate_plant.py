"""Plant validation.

1. The CasADi prediction model used inside the MPC must equal the NumPy
   Fossen model to machine precision on random states.
2. The MuJoCo plant (rigid body + injected hydrodynamics) is integrated
   open-loop against the analytic RK4 Fossen plant under an identical
   excitation (thruster wrench + external disturbance) and the state
   divergence is reported.  This quantifies the only approximation in the
   MuJoCo port: the one-substep-lagged, low-pass-filtered added-mass force.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import casadi as ca
import numpy as np

from bluerov2mj import fossen
from bluerov2mj.analytic_env import BlueROV2AnalyticEnv
from bluerov2mj.controllers.mpc import _f_casadi
from bluerov2mj.mujoco_env import BlueROV2MujocoEnv
from bluerov2mj import disturbances


def check_casadi_model(n=200, seed=0):
    xs = ca.MX.sym("x", 12)
    us = ca.MX.sym("u", 4)
    ws = ca.MX.sym("w", 6)
    f = ca.Function("f", [xs, us, ws], [_f_casadi(xs, us, ws)])
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n):
        x = rng.uniform(-1, 1, 12)
        x[3:5] *= 0.4                      # keep away from theta = +-pi/2
        u = rng.uniform(-30, 30, 4)
        w = rng.uniform(-10, 10, 6)
        tau = np.array([u[0], u[1], u[2], 0, 0, u[3]])
        f_np = fossen.f_state(x, tau, w)
        f_ca = np.array(f(x, u, w)).ravel()
        err = max(err, np.abs(f_np - f_ca).max())
    print(f"[1] CasADi vs NumPy Fossen model: max |df| = {err:.3e}")
    assert err < 1e-9, "CasADi model mismatch"


def check_mujoco_vs_analytic(T=10.0, seed=2):
    noise0 = dict(pos=0, ang=0, lin_vel=0, ang_vel=0, lin_acc=0, ang_acc=0)
    envm = BlueROV2MujocoEnv(meas_noise=noise0)
    enva = BlueROV2AnalyticEnv(meas_noise=noise0)
    dist = disturbances.make("periodic", seed=seed)
    rng = np.random.default_rng(seed)

    n = int(T / envm.dt_ctrl)
    u = np.zeros(4)
    div = np.zeros((n, 12))
    for k in range(n):
        if k % 20 == 0:                    # piecewise-constant excitation
            u = rng.uniform(-1, 1, 4) * np.array([15, 15, 15, 3])
        w = dist(k * envm.dt_ctrl)
        _, xm = envm.step(u, w)
        _, xa = enva.step(u, w)
        div[k] = xm - xa
        div[k, 3:6] = fossen.wrap_angle(div[k, 3:6])

    pos = np.abs(div[:, :3]).max()
    ang = np.abs(div[:, 3:6]).max()
    vel = np.abs(div[:, 6:9]).max()
    print(f"[2] MuJoCo vs analytic Fossen over {T:.0f} s "
          f"(periodic disturbance + random thrust):")
    print(f"    max |pos err| = {pos * 100:.2f} cm,  "
          f"max |ang err| = {np.degrees(ang):.2f} deg,  "
          f"max |lin vel err| = {vel * 100:.2f} cm/s")
    return pos, ang, vel


if __name__ == "__main__":
    check_casadi_model()
    check_mujoco_vs_analytic()
    print("validation done")
