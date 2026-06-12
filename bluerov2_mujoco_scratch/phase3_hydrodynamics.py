"""Phase 3 gate — drag D(v)v + added mass M_A.
Phase 3 게이트 — 항력 D(v)v + 부가질량 M_A.

A. EN: Terminal velocity.  Release positively-buoyant, no thrust: the body
      rises until net buoyancy balances drag, (B-W) = |Dl_w| w + |Dnl_w| w^2,
      whose positive root is hand-computable.
   KR: 종단속도. 양성부력으로 추력 없이 놓으면 순부력이 항력과 평형을 이룰
      때까지 상승: (B-W)=|Dl_w| w+|Dnl_w| w^2, 그 양의 근을 손으로 계산.

B. EN: Energy monotonicity.  Free decay from a mixed velocity: total
      mechanical energy E = 1/2 v.M.v - W z_cg + B z_cb must never increase
      (damping is dissipative; added mass must not inject energy).
   KR: 에너지 단조성. 혼합 속도에서 자유 감쇠: 총 역학에너지
      E = 1/2 v.M.v - W z_cg + B z_cb 는 절대 증가하면 안 됨(감쇠는 소산,
      부가질량은 에너지를 만들면 안 됨).

Run:  python phase3_hydrodynamics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv
from rov_sim.physics import fossen, params as P


def energy(env):
    """Total mechanical energy. / 총 역학에너지."""
    nu = env._nu_true()
    z_cg = env.data.xipos[env.bid][2]
    z_cb = env.data.xpos[env.bid][2]
    return 0.5 * nu @ (fossen.M_TOTAL @ nu) - P.WEIGHT * z_cg + P.BUOYANCY * z_cb


def main():
    # ---- A. terminal ascent velocity / 종단 상승속도 ----
    roots = np.roots([abs(P.DNL[2]), abs(P.DL[2]), -P.NET_BUOYANCY])
    w_hand = roots[roots > 0][0]
    env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=False, enable_hydro=True)
    env.reset()
    for _ in range(400):                     # 20 s >> time constant ~0.5 s
        env.step()
    w_meas = -env.get_true_state()[8]        # heave speed (NED up -> w<0)
    okA = abs(w_meas - w_hand) < 1e-3
    print("[P3.A] terminal ascent / 종단 상승속도:")
    print(f"       MuJoCo = {w_meas*100:.3f} cm/s   hand = {w_hand*100:.3f} cm/s"
          f"   -> {'PASS' if okA else 'FAIL'}")

    # ---- B. energy never increases / 에너지 비증가 ----
    env = ROVEnv(buoyancy=P.BUOYANCY, enable_thrust=False, enable_hydro=True)
    env.reset(nu0=(0.4, 0.3, 0.2, 0.3, 0.3, 0.4))
    E = [energy(env)]
    for _ in range(160):                     # 8 s
        env.step()
        E.append(energy(env))
    dE = np.diff(np.array(E))
    max_inc = max(dE.max(), 0.0)
    okB = max_inc < 1e-9
    print("[P3.B] free-decay energy / 자유감쇠 에너지:")
    print(f"       E0 = {E[0]:+.3f} J -> E(8s) = {E[-1]:+.3f} J   "
          f"dissipated = {E[0]-E[-1]:.3f} J   "
          f"max single-step INCREASE = {max_inc:.2e} J  -> {'PASS' if okB else 'FAIL'}")

    ok = okA and okB
    print(f"\nPhase 3 gate: {'PASS' if ok else 'FAIL'}"
          f"  (drag + added mass / 항력·부가질량 검증)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
