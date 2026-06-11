"""MuJoCo plant for the BlueROV2.

MuJoCo natively integrates the rigid-body part of Fossen's model (M_RB with
the m*zg coupling, C_RB, gravity acting at the CG).  This wrapper injects,
at every physics substep and via ``xfrc_applied``:

  * buoyancy  B  acting upward at the CB (= body origin),
  * hydrodynamic damping        -(D_L + D_NL(nu)) nu,
  * added-mass Coriolis         -C_A(nu) nu,
  * added-mass inertial force   -M_A nu_dot   (one-substep-lagged, low-pass
    filtered acceleration - the same technique used by the Gazebo
    uuv_simulator plugin the paper was validated with),
  * the thruster wrench K t (held for one control period), and
  * the external disturbance wrench, given in the inertial frame at the CG
    (the MuJoCo equivalent of ROS ``ApplyBodyWrench``).

Measurements (eta, nu, nu_dot) are returned with configurable Gaussian
noise; psi is reported unwrapped so downstream consumers see a continuous
yaw angle.
"""
import os

import mujoco
import numpy as np

from . import allocation, fossen
from . import params as P

_XML = os.path.join(os.path.dirname(__file__), "bluerov2.xml")


class BlueROV2MujocoEnv:
    def __init__(self, dt_ctrl=P.DT_CTRL, meas_noise=None, seed=0,
                 acc_filter=0.3):
        self.model = mujoco.MjModel.from_xml_path(_XML)
        self.data = mujoco.MjData(self.model)
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                     "base_link")
        self.dt_ctrl = dt_ctrl
        self.dt_sim = self.model.opt.timestep
        self.n_sub = int(round(dt_ctrl / self.dt_sim))
        assert abs(self.n_sub * self.dt_sim - dt_ctrl) < 1e-9
        self.rng = np.random.default_rng(seed)
        self.noise = dict(P.MEAS_NOISE) if meas_noise is None else meas_noise
        self.acc_filter = acc_filter
        self.reset()

    # ------------------------------------------------------------------ api
    def reset(self, eta0=(0, 0, -20, 0, 0, 0), nu0=(0, 0, 0, 0, 0, 0)):
        mujoco.mj_resetData(self.model, self.data)
        eta0 = np.asarray(eta0, dtype=float)
        self.data.qpos[:3] = eta0[:3]
        self.data.qpos[3:7] = fossen.euler_to_quat(*eta0[3:])
        R = fossen.rot_ib(*eta0[3:])
        self.data.qvel[:3] = R @ np.asarray(nu0[:3], dtype=float)  # world lin
        self.data.qvel[3:6] = nu0[3:]                              # body ang
        mujoco.mj_forward(self.model, self.data)
        self._nu_prev_sub = self._nu_true()
        self._nudot_filt = np.zeros(6)
        self._nu_prev_ctrl = self._nu_true()
        self._psi_unwrapped = eta0[5]
        self.t = 0.0
        return self.get_measurement()

    def step(self, u_cmd, w_world=np.zeros(6)):
        """Advance one control period.

        u_cmd:   [X_u, Y_u, Z_u, N_u] commanded forces/moments.
        w_world: external disturbance wrench [F(3); L(3)] in the inertial
                 frame, applied at the CG (constant over the period).
        """
        u_cmd = np.clip(np.asarray(u_cmd, float), -P.U_MAX, P.U_MAX)
        tau_b = allocation.wrench_from_u(u_cmd)        # actual body wrench
        w_world = np.asarray(w_world, dtype=float)

        for _ in range(self.n_sub):
            self._apply_forces(tau_b, w_world)
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt_ctrl

        nu = self._nu_true()
        nudot_ctrl = (nu - self._nu_prev_ctrl) / self.dt_ctrl
        self._nu_prev_ctrl = nu
        return self.get_measurement(nudot_ctrl), self.get_true_state()

    # ------------------------------------------------------------- internals
    def _nu_true(self):
        """Body-frame velocity nu = [u v w p q r] at the body origin."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, self.bid, res, 1)
        return np.concatenate([res[3:6], res[0:3]])    # [lin; ang]

    def _eta_true(self):
        pos = self.data.xpos[self.bid].copy()
        eul = fossen.quat_to_euler(self.data.xquat[self.bid])
        # unwrap yaw for continuity
        dpsi = fossen.wrap_angle(eul[2] - self._psi_unwrapped)
        self._psi_unwrapped += dpsi
        eul[2] = self._psi_unwrapped
        return np.concatenate([pos, eul])

    def _apply_forces(self, tau_b, w_world):
        nu = self._nu_true()
        nudot_raw = (nu - self._nu_prev_sub) / self.dt_sim
        self._nu_prev_sub = nu
        a = self.acc_filter
        self._nudot_filt = a * nudot_raw + (1 - a) * self._nudot_filt

        # hydrodynamic wrench about the body origin, body frame
        R = self.data.xmat[self.bid].reshape(3, 3)     # body -> world(NED)
        f_buoy_b = R.T @ np.array([0.0, 0.0, -P.BUOYANCY])
        F_b = (tau_b[:3] + f_buoy_b
               - fossen.damping(nu)[:3]
               - fossen.coriolis_added(nu)[:3]
               - P.ADDED_MASS[:3] * self._nudot_filt[:3])
        L_b = (tau_b[3:]
               - fossen.damping(nu)[3:]
               - fossen.coriolis_added(nu)[3:]
               - P.ADDED_MASS[3:] * self._nudot_filt[3:])

        # convert (about body origin, body frame) -> (about CoM, world frame)
        F_w = R @ F_b
        L_w = R @ L_b + np.cross(self.data.xpos[self.bid]
                                 - self.data.xipos[self.bid], F_w)
        self.data.xfrc_applied[self.bid, :3] = F_w + w_world[:3]
        self.data.xfrc_applied[self.bid, 3:] = L_w + w_world[3:]

    # ----------------------------------------------------------- observation
    def get_true_state(self):
        return np.concatenate([self._eta_true_no_update(), self._nu_true()])

    def _eta_true_no_update(self):
        pos = self.data.xpos[self.bid].copy()
        eul = fossen.quat_to_euler(self.data.xquat[self.bid])
        eul[2] = self._psi_unwrapped + fossen.wrap_angle(
            eul[2] - self._psi_unwrapped)
        return np.concatenate([pos, eul])

    def get_measurement(self, nudot=None):
        eta = self._eta_true()
        nu = self._nu_true()
        n = self.noise
        eta_m = eta + np.concatenate([
            self.rng.normal(0, n["pos"], 3), self.rng.normal(0, n["ang"], 3)])
        nu_m = nu + np.concatenate([
            self.rng.normal(0, n["lin_vel"], 3),
            self.rng.normal(0, n["ang_vel"], 3)])
        if nudot is None:
            nudot = np.zeros(6)
        nudot_m = nudot + np.concatenate([
            self.rng.normal(0, n["lin_acc"], 3),
            self.rng.normal(0, n["ang_acc"], 3)])
        return dict(eta=eta_m, nu=nu_m, nudot=nudot_m, t=self.t)
