"""Phase 5 gate — EAOB observer + measurement noise + disturbances (DOBMPC).
Phase 5 게이트 — EAOB 관측기 + 측정 잡음 + 외란 (DOBMPC).

EN: The full disturbance-observer MPC.  Readouts are corrupted with Gaussian
    noise; an external current (force + yaw moment) is switched on mid-run.
    The EAOB estimates the disturbance online and feeds it to the NMPC as a
    parameter, so station-keeping holds despite the current.  Two checks:
      A. the EAOB estimate w_hat tracks the applied wrench w,
      B. DOBMPC keeps the position error small under the disturbance.
KR: 외란관측 MPC 전체. 측정값에 가우시안 잡음을 섞고, 실행 도중 외부 해류(힘
    + 요 모멘트)를 켠다. EAOB 가 외란을 온라인 추정해 NMPC 에 파라미터로
    넘겨, 해류에도 정위치가 유지된다. 두 검증:
      A. EAOB 추정 w_hat 이 실제 외란 w 를 추종,
      B. DOBMPC 가 외란 하에서도 위치오차를 작게 유지.

Run:  python phase5_eaob_dob.py   (takes ~40 s)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv
from rov_sim.physics import (EAOB, NMPC, allocation, disturbances,
                             params as P)


def main():
    start = np.array([0, 0, -20, 0, 0, 0.0])
    T, dt = 14.0, P.DT_CTRL
    n = int(round(T / dt))
    # EN: constant current switched on at t = 2 s (10 N each axis + 5 Nm yaw)
    # KR: t=2 s 에 켜지는 일정 해류(각 축 10 N + 요 5 Nm)
    dist = disturbances.ConstantCurrent(force=(10, 10, 10), moment_z=5.0,
                                        t_on=2.0)

    env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=True, enable_hydro=True,
                 meas_noise=P.MEAS_NOISE, seed=1)
    meas = env.reset(eta0=start)
    mpc = NMPC(N=40)
    obs = EAOB(eta0=meas["eta"], nu0=meas["nu"])

    ref_col = np.concatenate([start, np.zeros(6)])
    refs = np.repeat(ref_col[:, None], mpc.N + 1, axis=1)

    u = np.zeros(4)
    log_t = np.zeros(n)
    err = np.zeros((n, 4))
    w_app = np.zeros((n, 6))
    w_est = np.zeros((n, 6))
    for k in range(n):
        t = env.t
        # EN: observer predict/update using last applied wrench / 이전 적용 wrench로 예측·보정
        eta_h, nu_h, w_h = obs.update(meas, allocation.wrench_from_u(u))
        u = mpc.solve(np.concatenate([eta_h, nu_h]), w_h, refs)
        w_now = dist(t)
        meas, x_true = env.step(u, w_now)
        log_t[k] = env.t
        err[k, :3] = x_true[:3] - start[:3]
        err[k, 3] = (x_true[5] - start[5] + np.pi) % (2 * np.pi) - np.pi
        w_app[k] = w_now
        w_est[k] = obs.w_world()

    # ---- A. disturbance tracking (after convergence, t > 6 s) ----
    # ---- A. 외란 추종(수렴 후, t>6 s) ----
    m = log_t > 6.0
    idx = [0, 1, 2, 5]                       # Fx Fy Fz Mz
    track_err = np.abs(w_est[m][:, idx] - w_app[m][:, idx]).mean(axis=0)
    okA = track_err[:3].max() < 2.0 and track_err[3] < 1.5
    print("[P5.A] EAOB disturbance estimate vs applied (t>6 s) / 외란 추정 vs 적용:")
    print(f"       applied   = [Fx {w_app[m][:,0].mean():5.1f}  "
          f"Fy {w_app[m][:,1].mean():5.1f}  Fz {w_app[m][:,2].mean():5.1f}  "
          f"Mz {w_app[m][:,5].mean():5.1f}]")
    print(f"       estimated = [Fx {w_est[m][:,0].mean():5.1f}  "
          f"Fy {w_est[m][:,1].mean():5.1f}  Fz {w_est[m][:,2].mean():5.1f}  "
          f"Mz {w_est[m][:,5].mean():5.1f}]")
    print(f"       mean |err| = [{track_err[0]:.2f} {track_err[1]:.2f} "
          f"{track_err[2]:.2f} N, {track_err[3]:.2f} Nm]  -> {'PASS' if okA else 'FAIL'}")

    # ---- B. DOBMPC station-keeping under the current ----
    # ---- B. 해류 하 DOBMPC 정위치 유지 ----
    ss = err[log_t > 6.0]
    rmse = np.sqrt((ss ** 2).mean(axis=0))
    okB = np.linalg.norm(rmse[:3]) < 0.05
    print("[P5.B] DOBMPC station-keeping under current / 해류 하 정위치:")
    print(f"       last-window RMSE  x={rmse[0]*100:.2f}  y={rmse[1]*100:.2f}  "
          f"z={rmse[2]*100:.2f} cm,  yaw={rmse[3]:.4f} rad  -> {'PASS' if okB else 'FAIL'}")

    ok = okA and okB
    print(f"\nPhase 5 gate: {'PASS' if ok else 'FAIL'}"
          f"  (EAOB + noise + disturbance / 관측기·잡음·외란 검증)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
