"""Phase 4 gate — MPC regulation with MuJoCo state feedback.
Phase 4 게이트 — MuJoCo 상태 피드백으로 MPC 정위치 제어.

EN: Close the loop  MuJoCo true state -> NMPC -> u -> MuJoCo.  Start offset
    from a set-point and check the MPC drives the error to ~0 (no disturbance,
    no observer, no noise yet).  The solver is the validated CasADi NMPC; to
    match the paper's real-time setup, the same model/cost port 1:1 to acados
    SQP-RTI (acados is not installed here, so we use Ipopt).
KR: 폐루프 MuJoCo 참값 상태 -> NMPC -> u -> MuJoCo. set-point 에서 떨어진
    곳에서 시작해 MPC 가 오차를 ~0 으로 모는지 확인(아직 외란·관측기·잡음
    없음). 솔버는 검증된 CasADi NMPC; 논문의 실시간 구성에 맞추려면 동일
    모델/비용을 acados SQP-RTI 로 1:1 이식 가능(여기엔 acados 미설치라 Ipopt 사용).

Run:  python phase4_mpc.py   (Ipopt ~70 ms/step, takes ~30 s)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv
from rov_sim.physics import NMPC, params as P


def main():
    start = np.array([0, 0, -20, 0, 0, 0.0])
    eta_d = np.array([1.0, -0.6, -20.5, 0, 0, 0.35])     # set-point / 목표
    T, dt = 15.0, P.DT_CTRL
    n = int(round(T / dt))

    # EN: plant uses the real net buoyancy, matching the MPC's Fossen model
    # KR: 플랜트는 실제 순부력 사용 -> MPC 의 Fossen 모델과 일치
    env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=True, enable_hydro=True)
    env.reset(eta0=start)
    mpc = NMPC(N=40)

    # EN: constant set-point reference over the horizon / 호라이즌 전체에 일정 목표
    ref_col = np.concatenate([eta_d, np.zeros(6)])
    refs = np.repeat(ref_col[:, None], mpc.N + 1, axis=1)

    x = env.get_true_state()
    errs = np.zeros((n, 4))
    t0 = time.time()
    for k in range(n):
        u = mpc.solve(x, np.zeros(6), refs)              # w_hat = 0 (no observer)
        _, x = env.step(u)
        errs[k, :3] = x[:3] - eta_d[:3]
        errs[k, 3] = (x[5] - eta_d[5] + np.pi) % (2 * np.pi) - np.pi

    init_off = np.linalg.norm(start[:3] - eta_d[:3])
    final_pos = np.linalg.norm(errs[-1, :3])
    final_yaw = abs(errs[-1, 3])
    # EN: steady-state RMSE over the last 3 s / 마지막 3초 정상상태 RMSE
    ss = errs[-int(3 / dt):]
    rmse = np.sqrt((ss ** 2).mean(axis=0))
    ok = final_pos < 0.06 and final_yaw < 0.06
    print(f"  start offset from set-point = {init_off:.2f} m, "
          f"{abs(start[5]-eta_d[5]):.2f} rad   (mean solve "
          f"{(time.time()-t0)/n*1e3:.0f} ms/step)\n")
    print(f"[P4] set-point regulation / 정위치 제어:")
    print(f"     final |pos err| = {final_pos*100:.2f} cm   "
          f"final |yaw err| = {final_yaw:.4f} rad")
    print(f"     last-3s RMSE  x={rmse[0]*100:.2f}  y={rmse[1]*100:.2f}  "
          f"z={rmse[2]*100:.2f} cm,  yaw={rmse[3]:.4f} rad   "
          f"-> {'PASS' if ok else 'FAIL'}")

    print(f"\nPhase 4 gate: {'PASS' if ok else 'FAIL'}"
          f"  (MPC feedback regulation / MPC 피드백 정위치)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
