"""Watch each phase live in the MuJoCo 3-D viewer.
각 단계(phase)를 MuJoCo 3D 뷰어로 직접 본다.

EN: Same physics the gates verify, now rendered in real time.  Pick a phase:
      1  buoyancy / static equilibrium  - tilted release rights itself (no drag,
                                          so it oscillates: the restoring couple)
      2  thrusters / directional motion - cycles surge / sway / heave / yaw
      3  hydrodynamics                  - a velocity kick decays to rest (drag)
      4  MPC                            - flies between set-points (green marker)
      5  EAOB + noise + disturbance     - holds station while a current (red
                                          arrow) pushes it (DOBMPC)
KR: 게이트가 검증한 바로 그 물리를 실시간으로 렌더링. 단계 선택:
      1 부력/정적평형  - 기울여 놓으면 복원 짝힘으로 흔들림(감쇠 없어 진동)
      2 추진기/방향이동 - 서지/스웨이/히브/요 순환
      3 유체력         - 초기 속도가 항력으로 감쇠해 정지
      4 MPC            - set-point(초록 마커) 사이를 날아 정위치
      5 EAOB+잡음+외란 - 해류(빨간 화살표)가 밀어도 정위치 유지(DOBMPC)

Run:  python view_phase.py --phase 5
      python view_phase.py --phase 1 --cg above   # CG above CB -> it capsizes
Controls / 조작:  drag=orbit, scroll=zoom, SPACE=pause, BACKSPACE=restart.
  Apply your OWN force / 직접 힘 주기: double-click the vehicle to select it,
  then Ctrl+right-drag = push (force), Ctrl+left-drag = twist (torque).
Note: NED world (+z down) so it may look upside-down until you orbit.
"""
import argparse
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import mujoco.viewer
import numpy as np

from rov_sim.env import ROVEnv
from rov_sim.physics import (EAOB, NMPC, allocation, disturbances,
                             params as P)

Z0 = -20.0                       # nominal depth / 기준 수심


# ----------------------------------------------------------------- drawing
def _add(scn, gtype, size, pos, rgba, mat=None):
    """Append one decorative geom (guarded). / 장식 지오메트리 1개 추가(가드)."""
    if scn.ngeom >= scn.maxgeom:
        return
    mat = np.eye(3).flatten() if mat is None else np.asarray(mat, float).flatten()
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(gtype),
                        np.asarray(size, float), np.asarray(pos, float),
                        mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _arrow(scn, p0, p1, rgba, width=0.03):
    """Draw an arrow p0 -> p1. / 화살표 p0 -> p1."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_ARROW), np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), width,
                         np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def draw(viewer, x_true, ref, w_app, trail, show_scene):
    scn = viewer.user_scn
    scn.ngeom = 0
    if show_scene:
        # sea bed below / water surface above (NED: below = larger z)
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (25, 25, 0.05),
             (0, 0, Z0 + 4.0), (0.55, 0.49, 0.38, 1.0))
        _add(scn, mujoco.mjtGeom.mjGEOM_BOX, (25, 25, 0.02),
             (0, 0, Z0 - 5.0), (0.10, 0.34, 0.52, 0.16))
    if ref is not None:                                   # set-point / hold goal
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.13, 0, 0), ref,
             (0.15, 0.95, 0.25, 0.55))
    if w_app is not None and np.linalg.norm(w_app[:3]) > 1e-6:   # disturbance
        p0 = x_true[:3]
        p1 = p0 + 0.06 * np.asarray(w_app[:3], float)     # scale N -> m
        _arrow(scn, p0, p1, (0.95, 0.2, 0.15, 0.9))
    for p in trail:
        _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), p,
             (0.30, 0.70, 1.0, 0.7))


# --------------------------------------------------------------- scenarios
class Scenario:
    """Base: owns the env, returns (u, w) each control step.
    베이스: env 를 소유하고 매 제어스텝마다 (u, w) 반환."""
    distance = 6.0

    def __init__(self):
        self.k = 0
        self.trail = deque(maxlen=150)
        self.ref = None
        self.w_app = None
        self.meas = None
        self._u = np.zeros(4)
        self.status = ""

    def _zero_velocity(self):
        """Stop the body without moving it. / 위치는 두고 속도만 0."""
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env._nu_prev_sub = self.env._nu_true()
        self.env._nudot_filt = np.zeros(6)
        self.env._nu_prev_ctrl = self.env._nu_true()


class P1(Scenario):
    name = "Phase 1 - buoyancy / static stability / 부력·정적안정성"
    SEG = 200                                            # re-perturb every 10 s
    KICK = 3                                             # impulse length (steps)

    def __init__(self, cg_z=None):
        super().__init__()
        self.cg_z = cg_z
        self.env = ROVEnv(buoyancy=None, cg_z=cg_z)       # neutral / 중립부력
        self.start()

    def start(self):
        # level at rest; a torque kick is applied repeatedly in control()
        # 수평 정지에서 시작; control()에서 토크 충격을 반복 인가
        self.meas = self.env.reset(eta0=(0, 0, Z0, 0, 0, 0))
        self.trail.clear()

    def control(self):
        # apply a roll+pitch torque kick, then release and watch it right itself
        # roll+pitch 토크 충격을 준 뒤 풀고, 스스로 복원하는지 본다
        if self.k % self.SEG < self.KICK:
            w = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 0.0])  # external torque kick
            self.w_app = w
            st = "torque kick / 토크 충격 인가"
        else:
            w = np.zeros(6)
            self.w_app = None
            if self.cg_z is not None and self.cg_z < 0:
                st = "released -> CG ABOVE CB -> CAPSIZES (unstable) / CB 위 CG -> 전복(불안정)"
            else:
                st = "released -> restoring couple rights it (undamped, rocks) / 복원 짝힘으로 복귀(감쇠없어 진동)"
        self.k += 1
        self.status = f"perturbation: {st}"
        return np.zeros(4), w


class P2(Scenario):
    name = "Phase 2 - thrusters / directional motion / 추진기·방향이동"
    distance = 6.5
    CMDS = [("surge +X", [22, 0, 0, 0]), ("sway +Y", [0, 22, 0, 0]),
            ("heave +Z", [0, 0, 22, 0]), ("yaw +N", [0, 0, 0, 7])]
    SEG = 45                                              # ~2.25 s per command

    def __init__(self, cg_z=None):
        super().__init__()
        self.env = ROVEnv(buoyancy=None, enable_thrust=True, enable_hydro=False,
                          cg_z=cg_z)
        self.start()

    def start(self):
        self.meas = self.env.reset(eta0=(0, 0, Z0, 0, 0, 0))
        self.trail.clear()

    def control(self):
        if self.k % self.SEG == 0:                        # new command segment
            self._zero_velocity()
            self.trail.clear()
        name, cmd = self.CMDS[(self.k // self.SEG) % len(self.CMDS)]
        self.k += 1
        self.status = (f"thrust {name}  (roll/pitch coupling is real BlueROV2 "
                       f"physics) / 추력 {name} (롤·피치 결합은 실제 물리)")
        return np.asarray(cmd, float), np.zeros(6)


class P3(Scenario):
    name = "Phase 3 - hydrodynamics / 유체력(항력)"
    SEG = 120                                             # re-kick every 6 s

    def __init__(self, cg_z=None):
        super().__init__()
        self.env = ROVEnv(buoyancy=None, enable_thrust=False, enable_hydro=True,
                          cg_z=cg_z)
        self.start()

    def start(self):
        # kick: surge + yaw spin -> drag bleeds the energy off to rest
        # 차주기: 서지 + 요 회전 -> 항력이 에너지를 빼 정지로
        self.meas = self.env.reset(eta0=(0, 0, Z0, 0, 0, 0),
                                   nu0=(1.6, 0.0, 0.0, 0.0, 0.0, 2.2))
        self.trail.clear()

    def control(self):
        if self.k > 0 and self.k % self.SEG == 0:
            self.start()
        self.k += 1
        spd = np.linalg.norm(self.env._nu_true()[:3])
        self.status = (f"velocity kick decaying under drag, |v|={spd:.2f} m/s "
                       f"/ 초기 속도가 항력으로 감쇠 중")
        return np.zeros(4), np.zeros(6)


class P4(Scenario):
    name = "Phase 4 - MPC set-point regulation / MPC 정위치"
    distance = 7.0
    SEG = 150                                             # new set-point / 7.5 s
    SP = [(1.2, 1.2, Z0, 0, 0, 0.4), (-1.2, 1.2, Z0 - 0.6, 0, 0, -0.4),
          (-1.2, -1.2, Z0 + 0.6, 0, 0, 0.0), (1.2, -1.2, Z0, 0, 0, 0.8)]

    def __init__(self, cg_z=None):
        super().__init__()
        self.env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=True,
                          enable_hydro=True, cg_z=cg_z)
        self.mpc = NMPC(N=40)
        self.start()

    def start(self):
        self.meas = self.env.reset(eta0=(0, 0, Z0, 0, 0, 0))
        self.trail.clear()

    def control(self):
        x = self.env.get_true_state()                     # true-state feedback
        eta_d = np.asarray(self.SP[(self.k // self.SEG) % len(self.SP)], float)
        refs = np.repeat(np.concatenate([eta_d, np.zeros(6)])[:, None],
                         self.mpc.N + 1, axis=1)
        u = self.mpc.solve(x, np.zeros(6), refs)
        self.ref = eta_d[:3]
        self.k += 1
        e = np.linalg.norm(x[:3] - eta_d[:3])
        self.status = (f"MPC -> set-point, |err|={e*100:.1f} cm "
                       f"/ MPC 정위치, 오차 {e*100:.1f} cm")
        return u, np.zeros(6)


class P5(Scenario):
    name = "Phase 5 - EAOB + noise + disturbance (DOBMPC) / 관측기·잡음·외란"
    HOLD = (0, 0, Z0, 0, 0, 0)

    def __init__(self, cg_z=None):
        super().__init__()
        self.env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=True,
                          enable_hydro=True, meas_noise=P.MEAS_NOISE, seed=1,
                          cg_z=cg_z)
        self.mpc = NMPC(N=40)
        self.dist = disturbances.ConstantCurrent(force=(10, 10, 10),
                                                 moment_z=5.0, t_on=3.0)
        self.start()

    def start(self):
        self.meas = self.env.reset(eta0=self.HOLD)
        self.obs = EAOB(eta0=self.meas["eta"], nu0=self.meas["nu"])
        self._u = np.zeros(4)
        self.trail.clear()

    def control(self):
        # observer predict/update with the last applied wrench, then solve
        # 직전 적용 wrench 로 관측기 예측·보정 후 MPC 풀기
        eta_h, nu_h, w_h = self.obs.update(self.meas,
                                           allocation.wrench_from_u(self._u))
        refs = np.repeat(np.concatenate([self.HOLD, np.zeros(6)])[:, None],
                         self.mpc.N + 1, axis=1)
        u = self.mpc.solve(np.concatenate([eta_h, nu_h]), w_h, refs)
        w = self.dist(self.env.t)
        self.ref = np.asarray(self.HOLD[:3], float)
        self.w_app = w
        self._u = u
        self.k += 1
        we = np.linalg.norm(self.obs.w_world()[:3])
        self.status = (f"DOBMPC hold | applied |F|={np.linalg.norm(w[:3]):4.1f} N, "
                       f"EAOB |F|={we:4.1f} N / 해류 버티는 중")
        return u, w


SCENARIOS = {1: P1, 2: P2, 3: P3, 4: P4, 5: P5}


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=5, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--cg", default="below", choices=["below", "above"],
                    help="CG below CB = stable; above = unstable (capsizes) "
                         "/ CG가 CB 아래=안정, 위=불안정(전복)")
    ap.add_argument("--no-scene", action="store_true",
                    help="hide sea-bed / surface decoration / 바닥·수면 끄기")
    args = ap.parse_args()

    cg_z = 0.02 if args.cg == "below" else -0.02
    scn = SCENARIOS[args.phase](cg_z=cg_z)
    env = scn.env
    # underwater ambiance (lighting only) / 수중 분위기(조명만)
    env.model.vis.headlight.ambient[:] = (0.22, 0.34, 0.44)
    env.model.vis.headlight.diffuse[:] = (0.34, 0.52, 0.62)

    state = {"paused": False, "restart": False}

    def key_callback(keycode):
        if keycode == 32:                                 # SPACE
            state["paused"] = not state["paused"]
        elif keycode in (259, 8):                         # BACKSPACE
            state["restart"] = True

    print(f"[view_phase] {scn.name}   (CG {args.cg} CB)")
    print("  apply your OWN force / 직접 힘 주기: double-click the vehicle, then")
    print("     Ctrl + right-drag = push (force) | Ctrl + left-drag = twist (torque)")
    print("  SPACE pause | BACKSPACE restart | drag orbit | scroll zoom\n")

    with mujoco.viewer.launch_passive(env.model, env.data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
        viewer.cam.trackbodyid = env.bid
        viewer.cam.distance = scn.distance
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 130.0

        last_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            if state["restart"]:
                scn.start()
                scn.k = 0
                state["restart"] = False
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                continue

            u, w = scn.control()
            # interactive mouse force: Ctrl+drag in the viewer applies a
            # perturbation; read it back and add it as an external wrench.
            # 마우스 힘: 뷰어에서 Ctrl+드래그한 perturbation 을 읽어 외란으로 더함
            mouse_w = np.zeros(6)
            pert = getattr(viewer, "perturb", None)
            if pert is not None:
                env.data.xfrc_applied[env.bid] = 0.0
                mujoco.mjv_applyPerturbForce(env.model, env.data, pert)
                mouse_w = env.data.xfrc_applied[env.bid].copy()
                env.data.xfrc_applied[env.bid] = 0.0
            scn.meas, x_true = env.step(u, w + mouse_w)
            scn.trail.append(x_true[:3].copy())
            draw(viewer, x_true, scn.ref, scn.w_app, scn.trail, not args.no_scene)
            viewer.sync()

            # pace each control step to real time / 각 제어스텝을 실시간으로
            dt_left = env.dt_ctrl - (time.time() - t0)
            if dt_left > 0:
                time.sleep(dt_left)

            now = time.time()
            if now - last_print >= 0.4:
                last_print = now
                sys.stdout.write("\r  " + scn.status + "    ")
                sys.stdout.flush()
    print("\n[view_phase] window closed. / 창 닫힘.")


if __name__ == "__main__":
    main()
