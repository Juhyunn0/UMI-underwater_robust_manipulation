"""Compare two CG placements — why the BlueROV2 is statically stable.
두 가지 CG 위치 비교 — BlueROV2 가 왜 정적으로 안정한가.

EN: A neutrally-buoyant body's only attitude stiffness is the couple between
    the weight (at the CG) and the buoyancy (at the CB).  Tilt it by theta:
      * CG BELOW CB (z_G > 0): the couple is RESTORING  -> stable (rights itself)
      * CG ABOVE CB (z_G < 0): the couple is CAPSIZING   -> unstable (flips over)
    Same torque kick, two CGs, watch the roll.
KR: 중립부력 물체의 유일한 자세 강성은 무게(CG)와 부력(CB) 사이의 짝힘이다.
    theta 만큼 기울이면:
      * CG가 CB 아래(z_G>0): 짝힘이 복원 -> 안정(스스로 일어섬)
      * CG가 CB 위(z_G<0):   짝힘이 전복 -> 불안정(뒤집힘)
    같은 토크 충격, 두 CG, roll 을 비교한다.

Run:  python compare_cg_stability.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rov_sim.env import ROVEnv


def roll_history(cg_z, kick=(0, 0, 0, 1.5, 0, 0), kick_steps=3, run_steps=160):
    """Torque-kick a neutrally-buoyant body with CG at cg_z, return roll(t).
    CG가 cg_z 인 중립부력 물체에 토크 충격을 주고 roll(t) 반환."""
    env = ROVEnv(buoyancy=None, cg_z=cg_z)
    env.reset(eta0=(0, 0, -20, 0, 0, 0))
    w = np.asarray(kick, float)
    for _ in range(kick_steps):
        env.step(np.zeros(4), w)
    roll = np.zeros(run_steps)
    for k in range(run_steps):
        env.step(np.zeros(4))
        roll[k] = np.degrees(env.get_true_state()[3])
    return roll


def main():
    print("  same roll torque kick, neutral buoyancy, no drag\n"
          "  같은 roll 토크 충격, 중립부력, 항력 없음\n")
    for cg_z, tag in [(+0.02, "CG BELOW CB (z_G=+0.02)"),
                      (-0.02, "CG ABOVE CB (z_G=-0.02)")]:
        roll = roll_history(cg_z)
        peak = np.abs(roll).max()
        # stable = stays bounded and swings back through upright;
        # unstable = grows past 90 deg (capsizes)
        capsized = peak > 90
        verdict = ("UNSTABLE - capsizes / 불안정 - 뒤집힘" if capsized
                   else "STABLE - rights itself / 안정 - 복원")
        print(f"  {tag}:  peak |roll| = {peak:6.1f} deg  ->  {verdict}")

    print("\n  => The CG sitting below the CB is exactly what makes the vehicle\n"
          "     self-right; flip it above and the same kick capsizes it.\n"
          "  => CG가 CB 아래에 있는 것이 차량을 스스로 일어서게 만든다;\n"
          "     위로 뒤집으면 같은 충격에 전복된다.")
    print("\n  watch it live / 라이브로 보기:")
    print("     python view_phase.py --phase 1 --cg below   # stable / 안정")
    print("     python view_phase.py --phase 1 --cg above   # capsizes / 전복")


if __name__ == "__main__":
    main()
