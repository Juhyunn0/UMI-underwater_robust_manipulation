"""Single source of truth for the dynamics — the *validated* code.
검증된 동역학 코드의 단일 진실 원천(single source of truth).

EN: Rather than re-derive M, C, D, g (and re-meet the sign / stiffness bugs
    the production package already solved), we import them from the sibling
    package ``bluerov2_mujoco_dobmpc/bluerov2mj``.  Everything MuJoCo-specific
    in this folder is written from scratch; only the math is reused.
KR: M, C, D, g 를 다시 유도(하다가 이미 해결된 부호·강성 버그를 다시 만들지)
    않기 위해, 옆 폴더 ``bluerov2_mujoco_dobmpc/bluerov2mj`` 의 검증된 구현을
    그대로 가져옵니다. 이 폴더의 MuJoCo 관련 코드만 새로 작성하고, 수식은
    재사용합니다.

Re-exported / 재노출:
  params        - constants (paper/repo) / 파라미터(논문·레포)
  fossen        - M, C_RB, C_A, D, g, J, quat<->euler / 포센 모델
  allocation    - u=[X,Y,Z,N] -> thrusts -> body wrench K t / 추력 배분
  NMPC          - CasADi nonlinear MPC / 비선형 MPC
  EAOB          - extended active observer (disturbance estimator) / 외란 관측기
  disturbances  - periodic / constant / mixed current generators / 외란 생성기
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOBMPC = os.path.normpath(
    os.path.join(_HERE, "..", "..", "bluerov2_mujoco_dobmpc"))
if _DOBMPC not in sys.path:
    sys.path.insert(0, _DOBMPC)

# EN: validated NumPy Fossen model + parameters
# KR: 검증된 NumPy 포센 모델 + 파라미터
from bluerov2mj import fossen                       # noqa: E402
from bluerov2mj import params                       # noqa: E402
# EN: validated control allocation, MPC, observer, disturbances
# KR: 검증된 추력 배분 / MPC / 관측기 / 외란
from bluerov2mj import allocation                   # noqa: E402
from bluerov2mj import disturbances                 # noqa: E402
from bluerov2mj.controllers.mpc import NMPC         # noqa: E402
from bluerov2mj.eaob import EAOB                     # noqa: E402

__all__ = ["fossen", "params", "allocation", "disturbances", "NMPC", "EAOB"]
