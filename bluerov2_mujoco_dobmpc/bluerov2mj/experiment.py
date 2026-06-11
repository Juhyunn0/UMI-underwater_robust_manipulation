"""Closed-loop experiment harness.

Wires plant -> EAOB -> controller exactly as in the paper (Fig. 3):

    measurement --> EAOB --> (eta_hat, nu_hat, w_hat) --> DOBMPC --> u
                     ^                                       |
                     +------ tau_applied = K t(u) <----------+

* DOBMPC : NMPC fed with the EAOB state estimate and disturbance parameter.
* MPC    : same NMPC, raw noisy measurements, w_hat = 0 (baseline).
* PID    : paper Table 5 gains, raw noisy measurements (baseline).
"""
import time

import numpy as np

from . import allocation, disturbances, fossen
from . import params as P
from .analytic_env import BlueROV2AnalyticEnv
from .controllers.mpc import NMPC
from .controllers.pid import PID
from .eaob import EAOB
from .mujoco_env import BlueROV2MujocoEnv


# ------------------------------------------------------------- references
def dp_reference(eta_d=(0.0, 0.0, -20.0, 0.0, 0.0, 0.0)):
    eta_d = np.asarray(eta_d, float)
    nu_d = np.zeros(6)
    return lambda t: (eta_d, nu_d)


def circle_reference(radius=2.0, period=12.5, z=-20.0):
    om = 2 * np.pi / period

    def ref(t):
        eta = np.array([radius * np.sin(om * t),
                        radius * (1.0 - np.cos(om * t)),
                        z, 0.0, 0.0, om * t])
        nu = np.array([radius * om, 0.0, 0.0, 0.0, 0.0, om])
        return eta, nu
    return ref


def lemniscate_reference(amp=2.0, period=25.0, z=-20.0):
    om = 2 * np.pi / period

    def ref(t):
        eta = np.array([amp * np.sin(om * t),
                        amp * np.sin(om * t) * np.cos(om * t),
                        z, 0.0, 0.0, 0.0])
        etad = np.array([amp * om * np.cos(om * t),
                         amp * om * np.cos(2 * om * t), 0.0])
        R = fossen.rot_ib(0.0, 0.0, 0.0)
        nu = np.concatenate([R.T @ etad, np.zeros(3)])
        return eta, nu
    return ref


# ----------------------------------------------------------------- runner
def make_env(plant="mujoco", seed=0, meas_noise=None):
    cls = BlueROV2MujocoEnv if plant == "mujoco" else BlueROV2AnalyticEnv
    return cls(seed=seed, meas_noise=meas_noise)


def run_closed_loop(controller, env, ref_fn, dist_fn, T,
                    mpc_obj=None, eaob_kwargs=None, verbose=True):
    """controller in {'pid', 'mpc', 'dobmpc'}.  Returns a log dict."""
    dt = env.dt_ctrl
    n = int(round(T / dt))
    eta0, _ = ref_fn(0.0)
    meas = env.reset(eta0=eta0)

    pid = PID() if controller == "pid" else None
    mpc = mpc_obj if controller in ("mpc", "dobmpc") else None
    if mpc is not None:
        mpc.reset()
    obs = EAOB(eta0=meas["eta"], nu0=meas["nu"],
               **(eaob_kwargs or {})) if controller == "dobmpc" else None

    log = dict(t=np.zeros(n), x=np.zeros((n, 12)), eta_ref=np.zeros((n, 6)),
               u=np.zeros((n, 4)), w_app=np.zeros((n, 6)),
               w_est=np.full((n, 6), np.nan), solve_ms=np.zeros(n))
    u = np.zeros(4)
    t_wall = time.time()
    for k in range(n):
        t = k * dt
        eta_r, nu_r = ref_fn(t)

        if obs is not None:
            eta_h, nu_h, w_h = obs.update(
                meas, allocation.wrench_from_u(u))
            x_ctrl = np.concatenate([eta_h, nu_h])
            w_mpc = w_h
            log["w_est"][k] = obs.w_world()
        else:
            x_ctrl = np.concatenate([meas["eta"], meas["nu"]])
            w_mpc = np.zeros(6)

        t0 = time.time()
        if controller == "pid":
            u = pid.solve(x_ctrl, np.concatenate([eta_r, nu_r]))
        else:
            refs = np.empty((12, mpc.N + 1))
            for j in range(mpc.N + 1):
                er, nr = ref_fn(t + j * dt)
                refs[:, j] = np.concatenate([er, nr])
            u = mpc.solve(x_ctrl, w_mpc, refs)
        log["solve_ms"][k] = (time.time() - t0) * 1e3

        w_now = dist_fn(t)
        meas, x_true = env.step(u, w_now)

        log["t"][k] = t + dt
        log["x"][k] = x_true
        log["eta_ref"][k] = ref_fn(t + dt)[0]
        log["u"][k] = u
        log["w_app"][k] = w_now
    if verbose:
        print(f"  {controller:7s}: {n} steps in {time.time()-t_wall:5.1f} s "
              f"(mean solve {np.nanmean(log['solve_ms']):.1f} ms)")
    return log


def rmse_table(logs):
    """logs: {name: log} -> {name: (ex, ey, ez, epsi)} RMSE."""
    out = {}
    for name, lg in logs.items():
        e = lg["x"][:, :6] - lg["eta_ref"]
        e[:, 5] = fossen.wrap_angle(e[:, 5])
        r = np.sqrt((e ** 2).mean(axis=0))
        out[name] = (r[0], r[1], r[2], r[5])
    return out


def print_rmse(rmse):
    hdr = f"{'controller':10s} {'x [m]':>9s} {'y [m]':>9s} {'z [m]':>9s} {'yaw [rad]':>10s}"
    lines = [hdr, "-" * len(hdr)]
    for name, r in rmse.items():
        lines.append(f"{name:10s} {r[0]:9.4f} {r[1]:9.4f} {r[2]:9.4f} {r[3]:10.4f}")
    txt = "\n".join(lines)
    print(txt)
    return txt
