#!/usr/bin/env python3
"""
BlueROV2 thruster actuation model (Phase 2) — MarineGym values, FLU frame.

Maps a 6-element thruster command to forces on the MarineGym-derived
`bluerov.xml`, using the standard BlueROV2 vectored-6 layout and the **T200
thrust curve taken verbatim from MarineGym** (`actuators/t200.py`, parameters in
`marinegym_assets/BlueROV.yaml`).

Command convention
------------------
Primary command = per-thruster normalized throttle `u in [-1, 1]^6`
(MarineGym's convention). `u > 0` pushes the vehicle along that thruster site's
local +X axis; `u < 0` reverses. The throttle -> thrust(N) map is the T200
steady-state curve `t200_thrust`. The MJCF actuators take thrust in NEWTONS, so
`set_thruster_commands` applies `data.ctrl = t200_thrust(u)`.

You may instead command a body wrench `[Fx,Fy,Fz,Mx,My,Mz]` (FLU body frame) and
allocate it with the pseudo-inverse of the allocation matrix `B` — see
`allocation_matrix` / `allocate`.

Signs / frame: native FLU — x forward, y left (PORT), z up. No NED here.

Scope: thrusters only. No hydrodynamics (buoyancy/drag/added mass) — next phase.
Only `mujoco` + `numpy` required.
"""
from __future__ import annotations

import numpy as np
import mujoco

# ----------------------------------------------------------------------------
# T200 steady-state thrust curve (MarineGym actuators/t200.py)
# ----------------------------------------------------------------------------
# throttle -> target rpm (a deadband of |throttle| <= 0.075 gives zero rpm,
# then the motor jumps to a minimum spin), rpm clamped to +/-3900;
# rpm -> thrust uses MarineGym's two asymmetric quadratic fits, output in kgf,
# then * 9.81 -> N. The leading `force_constants / 4.4e-7` factor is 1.0 for
# every BlueROV rotor (force_constants == 4.4e-7), so it is omitted here.
_RPM_MAX = 3900.0
_DEADBAND = 0.075
THRUSTER_DIRECTIONS = np.array([1., -1., 1., -1., 1., -1.])  # BlueROV.yaml; sets
# propeller reaction-torque sign in MarineGym but is multiplied by 0 there, so
# it has NO effect on force/torque in this model (kept only for documentation).


def _throttle_to_rpm(thr):
    thr = np.clip(np.asarray(thr, dtype=float), -1.0, 1.0)
    rpm = np.where(thr > _DEADBAND, 3.6599e3 * thr + 3.4521e2,
          np.where(thr < -_DEADBAND, 3.4944e3 * thr - 4.3350e2, 0.0))
    return np.clip(rpm, -_RPM_MAX, _RPM_MAX)


def _rpm_to_thrust(rpm):
    rpm = np.asarray(rpm, dtype=float)
    kgf = np.where(rpm > 0, 4.7368e-07 * rpm**2 - 1.9275e-04 * rpm + 8.4452e-02,
          np.where(rpm < 0, -3.8442e-07 * rpm**2 - 1.6186e-04 * rpm - 3.9139e-02,
                   0.0))
    return 9.81 * kgf


def t200_thrust(throttle):
    """Steady-state thrust [N] for normalized throttle in [-1, 1] (vectorized).

    t200_thrust(+1) = +64.13 N (max forward), t200_thrust(-1) = -51.55 N
    (max reverse) -> a forward/reverse asymmetry of ~1.24. Zero inside the
    +/-0.075 deadband.
    """
    return _rpm_to_thrust(_throttle_to_rpm(throttle))


# steady-state limits (used for actuator ctrlrange and sanity checks)
T200_MAX_FWD = float(t200_thrust(1.0))    # +64.1319 N
T200_MAX_REV = float(t200_thrust(-1.0))   # -51.5507 N


def t200_throttle_for_thrust(thrust_N, samples=4001):
    """Approximate inverse of the T200 curve: thrust [N] -> throttle [-1, 1].

    Numerically inverted on a dense grid (the curve is monotone outside the
    deadband). Thrusts below the deadband-jump minimum are not exactly
    achievable and map to the nearest feasible throttle. Convenience only —
    allocation works directly in force (N) space.
    """
    grid = np.linspace(-1.0, 1.0, samples)
    curve = t200_thrust(grid)
    thrust_N = np.atleast_1d(np.asarray(thrust_N, dtype=float))
    idx = np.argmin(np.abs(curve[None, :] - thrust_N[:, None]), axis=1)
    out = grid[idx]
    out[np.abs(thrust_N) < 1e-9] = 0.0
    return out if out.size > 1 else float(out[0])


class T200Dynamics:
    """Optional first-order T200 lag dynamics (MarineGym fidelity).

    Reproduces MarineGym's two lags: a per-call throttle lag (factor 0.43 toward
    the commanded throttle) and an rpm lag with time constant 0.01 s. Steady
    state equals `t200_thrust`. Use for realistic transients; the static curve
    is enough for the direction/allocation checks.
    """

    def __init__(self, n=6, tau_throttle=0.43, rpm_time_constant=0.01):
        self.throttle = np.zeros(n)
        self.rpm = np.zeros(n)
        self.tau_throttle = tau_throttle
        self.rpm_tc = rpm_time_constant

    def reset(self):
        self.throttle[:] = 0.0
        self.rpm[:] = 0.0

    def step(self, u_cmd, dt):
        u_cmd = np.clip(np.asarray(u_cmd, dtype=float), -1.0, 1.0)
        self.throttle += self.tau_throttle * (u_cmd - self.throttle)
        target_rpm = _throttle_to_rpm(self.throttle)
        alpha = np.exp(-dt / self.rpm_tc)
        self.rpm = np.clip(alpha * self.rpm + (1.0 - alpha) * target_rpm,
                           -_RPM_MAX, _RPM_MAX)
        return _rpm_to_thrust(self.rpm)


class ThrusterModel:
    """Realistic per-thruster actuator stage (OPT-IN; default path is ideal force).

    The controller still emits a desired per-thruster force `f_des` [N] from the
    allocation; on the real robot the low-level driver inverts the T200 curve to a
    throttle (at the *nominal* voltage), the ESC/motor realize it with a lag, and
    the actual thrust depends on the *actual* voltage. This stage reproduces that:

        f_des --T200 inverse(nominal V)--> throttle --(motor lag)--> T200 curve --> f
        f_real = voltage_scale * f                         (battery sag / wear / load)

    So it injects the **deadband** (small forces round-trip to 0 or the ~1.4 N
    minimum-spin jump), the **fwd/rev asymmetry + saturation** (the curve), the
    **motor lag** (T200Dynamics), and a **multiplicative thrust error** (voltage_
    scale) -- exactly the imperfections the ideal force path skips. The controller
    is NOT told about any of this, so realized != commanded == a robustness test."""

    def __init__(self, n=6, lag=True, voltage_scale=1.0):
        self.voltage_scale = float(voltage_scale)
        self.dyn = T200Dynamics(n=n) if lag else None

    def reset(self):
        if self.dyn is not None:
            self.dyn.reset()

    def realize(self, forces_des, dt):
        """Desired per-thruster force [N] -> actually-realized force [N]."""
        u = np.atleast_1d(t200_throttle_for_thrust(forces_des))   # driver inverse
        f = self.dyn.step(u, dt) if self.dyn is not None else t200_thrust(u)
        return self.voltage_scale * np.asarray(f, float)


# ----------------------------------------------------------------------------
# Geometry: thrust allocation matrix B  (wrench = B @ thrust_forces)
# ----------------------------------------------------------------------------
THRUSTER_SITES = [f"thruster_{i}" for i in range(6)]
ACTUATOR_NAMES = [f"thr{i}" for i in range(6)]


def allocation_matrix(model, data=None):
    """6x6 allocation B mapping the 6 thruster forces [N] to the body wrench
    [Fx,Fy,Fz,Mx,My,Mz] in the FLU body frame about the COM.

    Column i = [ d_i ; r_i x d_i ], with d_i the thruster's thrust axis (site
    local +X, in world == body frame at identity pose) and r_i its position
    relative to the body COM. Returns (B, site_names).
    """
    if data is None:
        data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    com = np.array(data.subtree_com[bid])
    B = np.zeros((6, 6))
    for i, sname in enumerate(THRUSTER_SITES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        pos = np.array(data.site_xpos[sid])
        axis = np.array(data.site_xmat[sid]).reshape(3, 3)[:, 0]  # local +X
        r = pos - com
        B[:3, i] = axis
        B[3:, i] = np.cross(r, axis)
    return B, list(THRUSTER_SITES)


def allocate(B, wrench_des):
    """Least-squares thruster forces [N] for a desired body wrench (FLU).

    Uses the pseudo-inverse, so unreachable wrench components (the vectored-6
    layout is rank-5: pitch My is not independently controllable) are projected
    out — you get the closest achievable wrench.
    """
    return np.linalg.pinv(B) @ np.asarray(wrench_des, dtype=float)


# ----------------------------------------------------------------------------
# Applying commands to a MuJoCo model (actuator ctrl = thrust in N)
# ----------------------------------------------------------------------------
def _ctrl_index(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ACTUATOR_NAMES]


def set_thruster_forces(model, data, forces_N):
    """Write thrust forces [N] directly to the 6 actuators (clipped to range)."""
    forces_N = np.asarray(forces_N, dtype=float)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    for k, ai in enumerate(_ctrl_index(model)):
        data.ctrl[ai] = float(np.clip(forces_N[k], lo[ai], hi[ai]))


def set_thruster_commands(model, data, throttles):
    """Set the 6 thrusters from normalized throttle u in [-1, 1] (T200 curve)."""
    set_thruster_forces(model, data, t200_thrust(throttles))


def step(model, data, throttles=None, forces_N=None, n=1):
    """Set a full 6-vector command and step the sim n times.

    Pass `throttles` (u in [-1,1]^6, mapped through the T200 curve) or
    `forces_N` (thrust in N applied directly). With neither, steps holding the
    current ctrl. Returns `data`.
    """
    if forces_N is not None:
        set_thruster_forces(model, data, forces_N)
    elif throttles is not None:
        set_thruster_commands(model, data, throttles)
    for _ in range(n):
        mujoco.mj_step(model, data)
    return data


def set_wrench_command(model, data, wrench_des, B=None, actuator=None):
    """Allocate a desired body wrench (FLU) to thruster forces and apply it.

    Returns the forces applied and the wrench actually realized (B @ forces),
    which differs from `wrench_des` by the unreachable (pitch) component.

    `actuator` (optional ThrusterModel): when given, the desired per-thruster
    forces are passed through the realistic T200 inverse/lag/voltage stage before
    being written -- so the returned `forces`/`realized` are what the plant truly
    got. Default None keeps the ideal force path (commanded == realized)."""
    if B is None:
        B, _ = allocation_matrix(model, data)
    forces = allocate(B, wrench_des)
    if actuator is not None:
        forces = actuator.realize(forces, float(model.opt.timestep))
    set_thruster_forces(model, data, forces)
    return forces, B @ forces
