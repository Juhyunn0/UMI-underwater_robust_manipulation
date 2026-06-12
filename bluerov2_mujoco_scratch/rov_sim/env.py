"""The incremental BlueROV2 MuJoCo simulator.
점진적으로(phase 별로) 자라는 BlueROV2 MuJoCo 시뮬레이터.

EN: MuJoCo natively integrates the rigid body (M_RB with the m*z_G coupling,
    C_RB) and applies the weight W = m g at the CG.  Everything hydrodynamic
    is injected here through ``xfrc_applied``.  Each capability is gated by a
    flag so a phase can switch on exactly what it is testing:

       Phase 1  buoyancy           (always on)
       Phase 2  thrust  K t        (enable_thrust)
       Phase 3  -D(v)v, -C_A v, -M_A v_dot   (enable_hydro)
       Phase 5  measurement noise (meas_noise) + external disturbance (step arg)

KR: MuJoCo가 강체(M_RB, m*z_G 결합, C_RB)와 CG에서의 무게 W=m g 를 직접
    적분합니다. 유체 관련 힘은 모두 여기서 ``xfrc_applied`` 로 주입합니다.
    각 기능은 플래그로 켜고 끌 수 있어, 해당 phase 가 테스트하려는 항만
    켤 수 있습니다 (위 표 참고).

EN: Two MuJoCo conventions used below /  아래에서 쓰는 MuJoCo 규약 두 가지:
  * mj_objectVelocity(flg_local=1) returns [angular; linear] -> we reorder.
    (각속도; 선속도) 순서로 주므로 [선속도; 각속도] 로 재배열.
  * xfrc_applied = force at the CoM + torque about the CoM, world frame.
    A wrench about the body origin (CB) is moved to the CoM with
    L_com = R L_b + (x_cb - x_com) x F_w.
    xfrc_applied 는 'CoM에서의 힘 + CoM 기준 토크(월드 프레임)'. 물체 원점(CB)
    기준 wrench 는 위 식으로 CoM 기준으로 옮긴다.
"""
import os

import mujoco
import numpy as np

from .physics import allocation, fossen, params as P

_XML = os.path.join(os.path.dirname(__file__), "model.xml")


class ROVEnv:
    def __init__(self, buoyancy=None, enable_thrust=False, enable_hydro=False,
                 meas_noise=None, acc_filter=0.3, seed=0, dt_ctrl=P.DT_CTRL,
                 cg_z=None):
        self.model = mujoco.MjModel.from_xml_path(_XML)
        self.data = mujoco.MjData(self.model)
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                     "base_link")
        # EN: optionally move the CG along body z.  +z_G = CG below CB (stable);
        #     -z_G = CG above CB (unstable, capsizes).  Default keeps the MJCF.
        # KR: 선택적으로 CG 를 body z 로 이동. +z_G = CG가 CB 아래(안정);
        #     -z_G = CB 위(불안정, 뒤집힘). 기본값은 MJCF 그대로.
        if cg_z is not None:
            self.model.body_ipos[self.bid][2] = float(cg_z)
        self.dt_ctrl = dt_ctrl
        self.dt_sim = self.model.opt.timestep
        self.n_sub = int(round(dt_ctrl / self.dt_sim))     # substeps per period
        # EN: B = W gives neutral buoyancy; P.BUOYANCY is the real repo value.
        # KR: B = W 면 중립 부력; P.BUOYANCY 는 레포의 실제 값.
        self.B = P.WEIGHT if buoyancy is None else float(buoyancy)
        self.enable_thrust = enable_thrust
        self.enable_hydro = enable_hydro
        self.acc_filter = acc_filter                       # added-mass EMA alpha
        self.rng = np.random.default_rng(seed)
        # EN: meas_noise = None -> clean readouts; a dict -> Gaussian noise.
        # KR: meas_noise = None 이면 무잡음, dict 면 가우시안 잡음.
        self.noise = meas_noise
        self.reset()

    # ------------------------------------------------------------------ api
    def reset(self, eta0=(0, 0, -20, 0, 0, 0), nu0=(0, 0, 0, 0, 0, 0)):
        mujoco.mj_resetData(self.model, self.data)
        eta0 = np.asarray(eta0, float)
        self.data.qpos[:3] = eta0[:3]
        self.data.qpos[3:7] = fossen.euler_to_quat(*eta0[3:])
        R = fossen.rot_ib(*eta0[3:])
        self.data.qvel[:3] = R @ np.asarray(nu0[:3], float)   # world linear
        self.data.qvel[3:6] = nu0[3:]                         # body angular
        mujoco.mj_forward(self.model, self.data)
        self._nu_prev_sub = self._nu_true()                   # for added mass FD
        self._nudot_filt = np.zeros(6)
        self._nu_prev_ctrl = self._nu_true()                  # for nudot readout
        self._psi_unwrapped = eta0[5]
        self.t = 0.0
        return self.get_measurement()

    def step(self, u_cmd=np.zeros(4), w_world=np.zeros(6)):
        """Advance one control period.  / 한 제어주기만큼 전진.

        EN: u_cmd = [X_u, Y_u, Z_u, N_u] commanded force/moment;
            w_world = external disturbance wrench in the NED frame at the CG.
        KR: u_cmd = [X,Y,Z,N] 지령 힘/모멘트;
            w_world = NED 프레임 CG 기준 외란 wrench.
        """
        u_cmd = np.clip(np.asarray(u_cmd, float), -P.U_MAX, P.U_MAX)
        # EN: thrust -> actual body wrench K t (incl. small roll/pitch coupling)
        # KR: 추력 -> 실제 body wrench K t (작은 롤/피치 결합 포함)
        tau_b = allocation.wrench_from_u(u_cmd) if self.enable_thrust \
            else np.zeros(6)
        w_world = np.asarray(w_world, float)

        for _ in range(self.n_sub):
            self._apply_forces(tau_b, w_world)
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt_ctrl

        nu = self._nu_true()
        nudot_ctrl = (nu - self._nu_prev_ctrl) / self.dt_ctrl
        self._nu_prev_ctrl = nu
        return self.get_measurement(nudot_ctrl), self.get_true_state()

    # ----------------------------------------------------------- force model
    def _apply_forces(self, tau_b, w_world):
        nu = self._nu_true()
        # EN: lagged + low-pass body acceleration for the added-mass term
        # KR: 부가질량 항을 위한 1스텝 지연 + 저역통과 가속도
        nudot_raw = (nu - self._nu_prev_sub) / self.dt_sim
        self._nu_prev_sub = nu
        a = self.acc_filter
        self._nudot_filt = a * nudot_raw + (1 - a) * self._nudot_filt

        R = self.data.xmat[self.bid].reshape(3, 3)            # body -> world

        # --- Phase 1: buoyancy, world "up" (= -z in NED), expressed in body ---
        # --- Phase 1: 부력, 월드 위쪽(-z)을 body 프레임으로 ---
        F_b = R.T @ np.array([0.0, 0.0, -self.B])
        L_b = np.zeros(3)

        # --- Phase 2: thruster wrench / 추진기 wrench ---
        if self.enable_thrust:
            F_b = F_b + tau_b[:3]
            L_b = L_b + tau_b[3:]

        # --- Phase 3: hydrodynamics / 유체력 ---
        # EN: damping(nu) returns +D(nu)nu (dissipative); subtract it.
        #     -C_A(nu)nu added-mass Coriolis; -M_A nu_dot added-mass inertia.
        # KR: damping(nu) 는 +D(nu)nu(소산력)을 반환 -> 빼준다.
        #     -C_A(nu)nu 부가질량 코리올리; -M_A nu_dot 부가질량 관성력.
        if self.enable_hydro:
            d = fossen.damping(nu)
            ca = fossen.coriolis_added(nu)
            F_b = F_b - d[:3] - ca[:3] - P.ADDED_MASS[:3] * self._nudot_filt[:3]
            L_b = L_b - d[3:] - ca[3:] - P.ADDED_MASS[3:] * self._nudot_filt[3:]

        # EN: move the wrench from the body origin (CB) to the CoM, world frame
        # KR: wrench 를 body 원점(CB) -> CoM(월드 프레임)으로 변환
        F_w = R @ F_b
        L_w = R @ L_b + np.cross(self.data.xpos[self.bid]
                                 - self.data.xipos[self.bid], F_w)

        # --- Phase 5: external disturbance (already world frame, at CG) ---
        # --- Phase 5: 외란 (이미 월드 프레임, CG 작용) ---
        self.data.xfrc_applied[self.bid, :3] = F_w + w_world[:3]
        self.data.xfrc_applied[self.bid, 3:] = L_w + w_world[3:]

    # ----------------------------------------------------------- read-outs
    def _nu_true(self):
        """Body velocity nu = [u v w p q r]. / body 속도."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, self.bid, res, 1)
        return np.concatenate([res[3:6], res[0:3]])           # [lin; ang]

    def _eta_true(self, update=True):
        """eta with a continuously-unwrapped yaw. / 연속적으로 unwrap된 yaw 포함 eta."""
        pos = self.data.xpos[self.bid].copy()
        eul = fossen.quat_to_euler(self.data.xquat[self.bid])
        dpsi = fossen.wrap_angle(eul[2] - self._psi_unwrapped)
        if update:
            self._psi_unwrapped += dpsi
            eul[2] = self._psi_unwrapped
        else:
            eul[2] = self._psi_unwrapped + dpsi
        return np.concatenate([pos, eul])

    def get_true_state(self):
        """True [eta(6); nu(6)] (no noise). / 참값 상태(무잡음)."""
        return np.concatenate([self._eta_true(update=False), self._nu_true()])

    def get_measurement(self, nudot=None):
        """Sensor dict {eta, nu, nudot, t}, optionally noisy.
        센서 딕셔너리 {eta, nu, nudot, t}, 필요시 잡음 포함."""
        eta = self._eta_true(update=True)
        nu = self._nu_true()
        if nudot is None:
            nudot = np.zeros(6)
        if self.noise is None:
            return dict(eta=eta, nu=nu, nudot=nudot, t=self.t)
        n = self.noise
        eta = eta + np.concatenate([self.rng.normal(0, n["pos"], 3),
                                    self.rng.normal(0, n["ang"], 3)])
        nu = nu + np.concatenate([self.rng.normal(0, n["lin_vel"], 3),
                                  self.rng.normal(0, n["ang_vel"], 3)])
        nudot = nudot + np.concatenate([self.rng.normal(0, n["lin_acc"], 3),
                                        self.rng.normal(0, n["ang_acc"], 3)])
        return dict(eta=eta, nu=nu, nudot=nudot, t=self.t)
