"""Phase 1 gate — buoyancy & static STABILITY under a force perturbation.
Phase 1 게이트 — 부력 & 교란력에 대한 정적 안정성.

EN: Instead of just checking the constant upward buoyancy, we PERTURB the
    vehicle with an external wrench and watch whether it comes back.

    The only self-righting effect in Phase 1 is the hydrostatic restoring
    couple: the CG sits z_G below the CB, so a tilt theta makes a moment
    -z_G W sin(theta) that pushes the attitude back to upright (metacentric
    stability).  There is NO damping yet (Phase 3) and NO control (Phase 4):

      A. roll / pitch  -> STABLE: a torque kick is restored; the attitude
         oscillates about upright (undamped -> it keeps oscillating; Phase 3's
         drag will make it settle).  Period ~ 2*pi/sqrt(z_G W / I).
      B. yaw           -> NEUTRAL: no hydrostatic restoring, so a yaw kick just
         keeps drifting.  (Translation is neutral the same way.)  This is why
         we will need damping + control later.

KR: 일정한 상향 부력만 확인하는 대신, 외부 wrench로 차량을 '교란'하고 다시
    돌아오는지 본다.

    Phase 1에서 유일한 자기복원 효과는 정수압 복원 짝힘이다: CG가 CB보다 z_G
    아래에 있어, 기울기 theta 에 대해 -z_G W sin(theta) 모멘트가 자세를 똑바로
    되돌린다(메타센터 안정성). 아직 감쇠(Phase 3)도 제어(Phase 4)도 없다:

      A. roll/pitch -> 안정: 토크 충격을 주면 복원되어 똑바로 위 주변에서 진동
         (감쇠가 없어 계속 진동; Phase 3 항력이 가라앉힘). 주기 ~ 2*pi/sqrt(z_G W / I).
      B. yaw        -> 중립: 정수압 복원이 없어 yaw 충격은 그대로 표류(병진도 동일).
         그래서 이후 감쇠+제어가 필요하다.

Run:  python phase1_static_equilibrium.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv
from rov_sim.physics import params as P


def zero_crossings(a):
    """Count sign changes (how many times it swings through zero).
    부호 변화 횟수(0을 몇 번 지나가는지)."""
    s = np.sign(a)
    s[s == 0] = 1
    return int(np.sum(np.diff(s) != 0))


def perturb_and_release(wrench, kick_steps=3, run_steps=240, buoyancy=None):
    """Apply an external wrench for kick_steps, then release and record eta.
    외부 wrench를 kick_steps 동안 가한 뒤 풀고 eta 기록."""
    env = ROVEnv(buoyancy=buoyancy)                 # neutral buoyancy, no drag
    env.reset(eta0=(0, 0, -20, 0, 0, 0))
    w = np.asarray(wrench, float)
    for _ in range(kick_steps):                     # apply the perturbation
        env.step(np.zeros(4), w)
    eta = np.zeros((run_steps, 6))
    for k in range(run_steps):                      # release (w = 0) and watch
        env.step(np.zeros(4))
        eta[k] = env.get_true_state()[:6]
    return eta, env.dt_ctrl


def main():
    zGW = P.ZG * P.WEIGHT
    print(f"  restoring stiffness z_G*W = {zGW:.3f} N*m/rad   "
          f"(CG is z_G={P.ZG} m below CB)\n")

    # ---- A. roll/pitch are STABLE (restoring couple) / roll·pitch 안정 ----
    eta, dt = perturb_and_release([0, 0, 0, 1.5, 1.5, 0.0])   # Mx, My torque kick
    roll, pitch = np.degrees(eta[:, 3]), np.degrees(eta[:, 4])
    nzr, nzp = zero_crossings(roll), zero_crossings(pitch)
    bounded = abs(roll).max() < 25 and abs(pitch).max() < 25
    okA = bounded and nzr >= 3 and nzp >= 3
    # measured vs predicted roll period / 측정 vs 예측 roll 주기
    T_pred = 2 * np.pi / np.sqrt(zGW / P.IX)
    cr = np.where(np.diff(np.sign(roll)) != 0)[0]
    T_meas = 2 * np.mean(np.diff(cr)) * dt if len(cr) >= 2 else float("nan")
    print("[P1.A] roll/pitch under a torque kick / roll·pitch 토크 충격:")
    print(f"       peak |roll|={abs(roll).max():.1f} deg, |pitch|={abs(pitch).max():.1f} deg "
          f"(bounded), swings through upright {nzr}/{nzp} times -> STABLE")
    print(f"       roll period: measured = {T_meas:.2f} s, "
          f"predicted 2pi/sqrt(z_G W/Ix) = {T_pred:.2f} s   "
          f"-> {'PASS' if okA else 'FAIL'}")

    # ---- B. yaw is NEUTRAL (no restoring) / yaw 중립(복원 없음) ----
    eta, _ = perturb_and_release([0, 0, 0, 0, 0, 1.0])        # Mz torque kick
    yaw = np.degrees(eta[:, 5])
    drifts = abs(yaw[-1]) > 60 and zero_crossings(yaw) <= 1   # keeps turning
    okB = drifts
    print("[P1.B] yaw under a torque kick / yaw 토크 충격:")
    print(f"       yaw drifts to {yaw[-1]:.0f} deg with {zero_crossings(yaw)} reversals "
          f"-> NEUTRAL (no restoring, needs control)   -> {'PASS' if okB else 'FAIL'}")

    ok = okA and okB
    print(f"\nPhase 1 gate: {'PASS' if ok else 'FAIL'}"
          f"  (attitude stable, yaw/translation neutral / 자세 안정, yaw·병진 중립)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
