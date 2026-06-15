#!/usr/bin/env python3
"""DOB-MPC controller for the marinegym BlueROV2 sim (FLU).

Wraps the validated NED EAOB + NMPC (dobmpc/ subpackage) as a drop-in alternative
to PoseController: same set_target / compute / apply interface, same rank-5
thrusters.py allocation, so teleop.py and the mission classes drive it unchanged.

Per control tick (20 Hz, ZOH between ticks over 25 physics substeps):
  1. read FLU state (mj_objectVelocity local=1 for body-frame nu -- NOT qvel[:3])
  2. frames.flu_to_ned -> [eta; nu] NED;  a_meas = d(nu)/dt by finite difference
     over the CONTROL tick (never data.qacc -- marinegym applies added mass as an
     external force, so qacc would double-count it in the EAOB measurement model)
  3. EAOB.update(meas, tau_cmd_ned)  (dobmpc mode) -> w_hat ; plain MPC -> w_hat=0
  4. NMPC.solve(x_ned, w_hat, xref_ned) -> u = [X, Y, Z, N]
  5. tau_ned = [X,Y,Z,0,0,N] -> frames.ned_wrench_to_flu -> tau_FLU (ZOH)
  6. thrusters.set_wrench_command (rank-5 pinv projects out the uncommanded pitch)

Design notes (control-theory-advisor validated): pitch is left to float to its
physical trim (MPC_Q pitch-weight 0, the EAOB absorbs the steady surge->pitch
coupling into w); the disturbance model is w_dot=0, so the DC current is rejected
strongly while the JONSWAP wave band / kicks are only partially rejected.
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thrusters as T
from dobmpc import frames
from dobmpc import params as P
from dobmpc.fossen import rot_ib
from dobmpc.eaob import EAOB
from dobmpc.mpc import NMPC


def _Rz_flu(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class DOBMPCController:
    def __init__(self, model, hydro=None, mode="dobmpc", setpoint=(0.0, 0.0, 0.0),
                 yaw_ref=0.0, body="base_link", ctrl_hz=20.0, N=P.MPC_N):
        assert mode in ("dobmpc", "mpc"), mode
        self.model = model
        self.hydro = hydro                       # only for parity with PoseController
        self.mode = mode
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.B, _ = T.allocation_matrix(model)   # constant rank-5 body geometry

        self.dt_sim = float(model.opt.timestep)
        self.ctrl_dt = 1.0 / float(ctrl_hz)
        self.decim = max(1, round(self.ctrl_dt / self.dt_sim))   # substeps per tick
        self.ctrl_dt = self.decim * self.dt_sim                  # exact hold duration
        assert abs(self.ctrl_dt - P.DT_CTRL) < 1e-9, (
            f"control dt {self.ctrl_dt} != params DT_CTRL {P.DT_CTRL}; EAOB/MPC assume "
            f"DT_CTRL -- keep them equal")

        # references (world FLU)
        self.p_ref = np.asarray(setpoint, float)
        self.yaw_ref = float(yaw_ref)
        self.v_ref = np.zeros(3)                 # world-FLU reference velocity (DP: 0)

        self.nmpc = NMPC(N=N, dt=P.DT_CTRL)
        self.eaob = None                         # lazy-init at first tick (needs eta0)
        self._tau_flu = np.zeros(6)              # ZOH wrench held between ticks
        self._tau_ned_cmd = np.zeros(6)          # commanded NED wrench (fed to EAOB)
        self._nu_prev_ned = None
        self._k = 0                              # substep counter

        self.commanded = np.zeros(6)
        self.realized = np.zeros(6)
        self.w_hat = np.zeros(6)                 # latest EAOB disturbance (NED body)
        self.solve_ms = 0.0
        self.n_fail = 0

    # ----------------------------------------------------- interface parity
    def set_target(self, p_ref=None, yaw_ref=None, v_ref=None):
        if p_ref is not None:
            self.p_ref = np.asarray(p_ref, float)
        if yaw_ref is not None:
            self.yaw_ref = float(yaw_ref)
        if v_ref is not None:
            self.v_ref = np.asarray(v_ref, float)

    def reset(self):
        self.eaob = None
        self._tau_flu = np.zeros(6)
        self._tau_ned_cmd = np.zeros(6)
        self._nu_prev_ned = None
        self._k = 0
        self.nmpc.reset()

    # ------------------------------------------------------------- state I/O
    def _read_state(self, data):
        p = np.asarray(data.xpos[self.bid], float)            # world FLU position
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.bid, res, 1)             # 1 = local (body) frame
        nu_flu = np.concatenate([res[3:6], res[0:3]])         # [lin; ang], body frame
        return p, R, nu_flu

    def _xref_ned(self):
        """Constant-pose DP reference as (12, N+1). (v_ref handling for moving
        trajectories is a follow-up; DP holds nu_ref = 0.)"""
        eta_ref = frames.flu_to_ned_eta(self.p_ref, _Rz_flu(self.yaw_ref))
        x_ref = np.concatenate([eta_ref, np.zeros(6)])
        return np.tile(x_ref.reshape(12, 1), (1, self.nmpc.N + 1))

    # --------------------------------------------------------- control tick
    def _control_step(self, data):
        p, R, nu_flu = self._read_state(data)
        eta_ned = frames.flu_to_ned_eta(p, R)
        nu_ned = frames.flu_to_ned_nu(nu_flu)

        if self.eaob is None:                     # lazy init at the measured pose
            self.eaob = EAOB(eta0=eta_ned, nu0=nu_ned)
            self._nu_prev_ned = nu_ned.copy()

        a_meas = (nu_ned - self._nu_prev_ned) / self.ctrl_dt   # FD over the tick
        self._nu_prev_ned = nu_ned.copy()

        if self.mode == "dobmpc":
            meas = {"eta": eta_ned, "nu": nu_ned, "nudot": a_meas}
            _, _, self.w_hat = self.eaob.update(meas, self._tau_ned_cmd)
        else:                                     # plain MPC: no disturbance comp
            self.w_hat = np.zeros(6)

        x_ned = np.concatenate([eta_ned, nu_ned])
        u = self.nmpc.solve(x_ned, self.w_hat, self._xref_ned())
        self.n_fail = self.nmpc.n_fail
        tau_ned = np.array([u[0], u[1], u[2], 0.0, 0.0, u[3]])
        self._tau_ned_cmd = tau_ned
        self._tau_flu = frames.ned_wrench_to_flu(tau_ned)
        self.commanded = self._tau_flu.copy()

    # --------------------------------------------------------- public step
    def compute(self, data):
        """Return the body wrench currently held (parity with PoseController)."""
        return self._tau_flu.copy()

    def apply(self, model, data):
        """Run a control tick on the decimation boundary, hold (ZOH) otherwise,
        then allocate the held FLU wrench to thrusters. Returns (forces, realized)."""
        if self._k % self.decim == 0:
            self._control_step(data)
        self._k += 1
        forces, realized = T.set_wrench_command(model, data, self._tau_flu, self.B)
        self.realized = np.asarray(realized, float)
        return forces, self.realized

    # --------------------------------------------------- diagnostics
    def w_world_flu(self):
        """EAOB disturbance estimate in FLU world (for comparison vs the true
        current/wave, which live in FLU world). Zeros for plain MPC."""
        if self.eaob is None or self.mode != "dobmpc":
            return np.zeros(6)
        return frames.ned_w_world_to_flu(self.eaob.w_world())

    def status(self):
        tag = "DOB-MPC" if self.mode == "dobmpc" else "MPC"
        wn = np.linalg.norm(self.w_hat[:3])
        return f"{tag}  |w_hat|={wn:.1f}N  solve_fail={self.n_fail}"
