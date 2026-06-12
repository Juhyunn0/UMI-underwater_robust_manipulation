"""Phase 2 gate — thrusters & directional motion.
Phase 2 게이트 — 추진기 & 방향성 이동.

EN: With thrust ON, neutral buoyancy and NO drag, push one control axis at a
    time from rest and check the body accelerates in the *right* DOF with the
    *right* sign.  The control map is u = [X_u, Y_u, Z_u, N_u]:
        +X -> surge u(+),  +Y -> sway v(+),  +Z -> heave w(+, down),  +N -> yaw r(+)
    The thrust is the real K t wrench, and the horizontal thrusters sit below
    the CG, so +X/+Y also induce a real roll/pitch coupling through the
    thruster geometry and the m*z_G mass coupling.  That coupling is genuine
    BlueROV2 physics (paper Sec. 5.5), so the dominance test compares ONLY the
    4 directly-actuated DOFs {surge, sway, heave, yaw}; roll/pitch are merely
    reported.
KR: 추력 ON, 중립부력, 항력 OFF 상태에서 한 번에 한 축씩 정지에서 밀어보고,
    물체가 올바른 DOF·올바른 부호로 가속되는지 확인. 지령 u=[X,Y,Z,N]:
        +X->서지(+), +Y->스웨이(+), +Z->히브(+,아래), +N->요(+).
    추력은 실제 K t wrench 이고 수평 추진기가 CG 아래에 있어, +X/+Y 는 추진기
    기하와 m*z_G 질량결합을 통해 실제 roll/pitch 결합을 일으킨다. 이는 실제
    BlueROV2 물리(논문 5.5절)이므로, 우세도 검사는 직접 구동되는 4 DOF
    {서지·스웨이·히브·요} 끼리만 비교하고 roll/pitch 는 참고로만 표시한다.

Run:  python phase2_directional_motion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv

# EN: name, command [X,Y,Z,N], index of the expected dominant DOF (in nu)
# KR: 이름, 지령 [X,Y,Z,N], 기대되는 우세 DOF 인덱스(nu 기준)
CASES = [
    ("surge  +X", [15, 0, 0, 0], 0),
    ("sway   +Y", [0, 15, 0, 0], 1),
    ("heave  +Z", [0, 0, 15, 0], 2),
    ("yaw    +N", [0, 0, 0, 5], 5),
]


def push(cmd, n_steps=4):
    """Apply cmd from rest for n_steps control periods, return true [eta;nu].
    정지에서 cmd 를 n_steps 제어주기 적용 후 참값 [eta;nu] 반환."""
    env = ROVEnv(buoyancy=None, enable_thrust=True, enable_hydro=False)
    env.reset()
    for _ in range(n_steps):
        env.step(np.asarray(cmd, float))
    return env.get_true_state()


def main():
    print("  push each axis 0.2 s from rest (neutral buoyancy, no drag)\n"
          "  각 축을 정지에서 0.2초 밀기 (중립부력, 항력 없음)\n")
    labels = ["u", "v", "w", "p", "q", "r"]
    controlled = [0, 1, 2, 5]                # surge sway heave yaw / 직접 구동 DOF
    all_ok = True
    for name, cmd, k in CASES:
        nu = push(cmd)[6:]
        dom = nu[k]
        # EN: dominance is judged only among the 4 actuated DOFs; the
        #     roll/pitch coupling (p, q) is expected and only reported.
        # KR: 우세도는 구동되는 4 DOF 끼리만 판정; roll/pitch(p,q) 결합은
        #     예상된 것이라 참고로만 표시.
        others = max(abs(nu[j]) for j in controlled if j != k)
        coupling = max(abs(nu[3]), abs(nu[4]))
        ok = dom > 0 and dom > 5 * others
        all_ok &= ok
        nu_str = "  ".join(f"{labels[j]}={nu[j]:+.3f}" for j in range(6))
        print(f"[P2] {name}: {nu_str}")
        print(f"       dominant {labels[k]}={dom:+.3f}  vs  max|other actuated|="
              f"{others:.3f}   (roll/pitch coupling={coupling:.3f}, expected)"
              f"   -> {'PASS' if ok else 'FAIL'}")

    print(f"\nPhase 2 gate: {'PASS' if all_ok else 'FAIL'}"
          f"  (thrust direction / 추력 방향 검증)")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
