# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## 🐛 테스트 / 스크립트 함정

### w_hat ±50 클립 — 단일 스칼라가 N·N·m 겸용 + 발동 무음 + 상수 중복
- **발견**: 2026-07-22 (mpc_acados 리뷰 워크플로: control-theory ×2 + simulation-advisor,
  verifier 수치 검증 11/12 verified)
- **증상**: `np.clip(w_hat, ±50)`(mpc_acados.py:170, mpc.py:198)에 대해 —
  (1) 힘 채널 50 N은 이 환경 물리 상한(realistic X≈35 / Y≈37 / Z≈3 N)보다 위라 사실상
  안 물리지만, 스웨이 보수 스택(추력한계 속도로 파정 역류 과도) ≈85–100 N에서는 물리고
  이 영역이 정확히 acados blowup 레짐; (2) 토크 채널 50 N·m은 authority(8–10 N·m)의
  5–6배·물리 상한(Munk 미스매치 ≤10 N·m)의 5배 — 회전 유효관성 0.45–0.76 kg·m²라
  40 N·m대 추정 스파이크가 통과하면 66–111 rad/s² 예측 → slack/RTI blowup;
  (3) 발동 카운터/로그 없음 + 로그되는 wh0-2는 pre-clip·FLU-world라 body-frame 클립
  발동을 판별 불가(‖wh‖ 50–86.6 N 밴드는 모호, >86.6 N만 확정); (4) 상수가 두 파일에
  하드코드 중복.
- **임시 대응**: 없음(정상 미션에선 힘 클립이 거의 안 물림; 관측기 상태는 클립 안 되므로
  추정 자체는 오염 없음).
- **제대로 고치려면**: params.py에 per-axis `W_HAT_CLIP=[15,45,45,5,5,8]` 단일 정의 +
  발동 카운터를 n_fail 옆에 기록 + (주 방어선) EAOB innovation gating(Mahalanobis χ² gate).

### `tests/test_dobmpc.py` — rank-5(NU=4/option-b) 전제, bluerov2 제거로 실행 불가 (deferred)
- **발견**: 2026-07-06, **갱신 2026-07-21** (bluerov2 변종 제거)
- **증상**: `tests/test_dobmpc.py`의 trim/option-(b) 단정(6 N ≈ 23° pitch, `[Fx,0,0,0]` NU=4
  입력)은 제거된 rank-5 `bluerov2` 플랜트 기준. 문서화됐던 `ROV_MODEL=bluerov2` 우회는
  이제 ValueError(변종 없음). 기본 heavy에선 NU=6이라 option-(b) 단정이 어긋남.
- **임시 대응**: 없음 — dobmpc 정리를 미룬 상태라 이 테스트도 함께 보류.
- **제대로 고치려면**: **미룬 dobmpc NU=6-only 정리와 함께** 진행 — params.py의 NU=4/
  option-b 죽은 경로 제거 + test_dobmpc의 rank-5 trim 단정을 heavy(NU=6)용으로 재작성.

### dobmpc NU=4 / option-(b) 죽은 경로 (bluerov2 제거 후 도달 불가, deferred)
- **발견**: 2026-07-21 (bluerov2 변종 제거)
- **증상**: 모든 잔존 변종이 rank-6(NU=6)이라 `dobmpc/params.py`의 NU=4 분기,
  `mpc.py`/`mpc_acados.py`의 option-(b) surge→pitch 커플링 경로가 죽은 코드가 됨.
- **임시 대응**: 무해(도달 안 함) — 기능 영향 없음.
- **제대로 고치려면**: 사용자 지시로 **나중에** 일괄 단순화(NU=6 고정). test_dobmpc 재작성과
  함께.
- **2026-07-22 갱신**: **mpc.py는 정리 완료** — tau if/else 제거(`tau = u`), pitch bound의
  PITCH_AWARE 삼항 → 1.2 고정(PITCH_AWARE=False라 동작 동일), 낡은 주석/docstring 정정
  (git HEAD 대비 2000 랜덤 샘플 dynamics Δ=0 + IPOPT NMPC 스모크 검증).
  **잔여**: params.py의 NU=4 분기·PITCH_AWARE/THETA_MAX/SURGE_PITCH_COUPLING,
  mpc_acados.py:109의 option-(b) pitch bound 삼항(+docstring "u=[X,Y,Z,N] (4)"),
  dobmpc_controller.py:269-280의 rank-5 분기, test_dobmpc 재작성.

### `tools/gen_pool_apriltags.py --selftest` — `tag_floor.xml`을 덮어씀
- **발견**: 2026-07-06 (`--tag-mode plane` 개편 후에도 유효 — `run_selftest`가
  tools/gen_pool_apriltags.py:475에서 tag_floor.xml을 테스트 타일 2개짜리로 씀)
- **증상**: selftest 후 POOL_TAGS 씬이 타일 2개짜리 바닥으로 로드됨.
- **임시 대응**: selftest 후 `python tools/gen_pool_apriltags.py` full build 재실행
  (README §7에 경고 있음).
- **제대로 고치려면**: selftest는 별도 임시 파일에 쓰고 종료 시 삭제
  (기존 `_selftest_scene.xml`처럼).

### `tools/plot_wave_spreading.py` — config/base.yaml의 하드코딩 복사본
- **발견**: 2026-07-06
- **증상**: Hs/Tp/gamma/s/h/N_omega/N_beta가 스크립트 상수로 복사돼 있음(yaml을 읽지
  않음) → config를 바꾸면 슬라이드 figure가 실제 실험과 **조용히** 어긋남.
- **제대로 고치려면**: `disturbance.config.load_config`로 yaml을 직접 읽기.

### `tools/analyze_square3.py` / `tools/analyze_acados_vs_before.py` — 경로 하드코딩
- **발견**: 2026-07-06
- **증상**: `recordings/20260615/`의 특정 CSV 파일명이 하드코딩(`DIR`/`RUNS`/`PAIRS`
  상수) → 해당 recording이 없으면 crash, CLI 플래그 없음.
- **임시 대응**: 다른 run에 쓰려면 상단 상수 수정 (README §5에 명시).
- **제대로 고치려면**: `--dir` 인자화. 우선순위 낮음(일회성 분석 스크립트).

### `experiments/plot_trajectories.py` — docstring/코드 불일치
- **발견**: 2026-07-06
- **증상**: docstring은 error-vs-time 패널을 언급하지만 현재 코드에 없음.
  steady-state RMS(마지막 2랩)는 내부에서 계산만 하고 어디에도 표시하지 않음.
- **제대로 고치려면**: docstring 정리, 또는 패널/범례에 SS-RMS 추가.

### `verify/verify_acados.py` 게이트가 heavy_gripper에서 근소 초과 (0.2717 > 0.25 N)
- **발견**: 2026-07-12 (heavy_gripper 변종 검증 중)
- **증상**: acados RTI vs IPOPT worst-case |Δu| 게이트 0.25 N은 heavy 기준 캘리브레이션;
  heavy_gripper(13.7 kg)에서 0.2717 N — 30 N sway authority의 ~0.9%라 실효 동일 최적해,
  폐루프도 검증됨(DP hold 1.3 cm). heavy는 여전히 PASS.
- **제대로 고치려면**: 게이트를 변종별 스케일 또는 ‖u‖ 상대비로.

### 파랑 모드에서 acados 솔버 실패(n_fail)가 드물게 자세/깊이 blowup 유발
- **발견**: 2026-07-21 (`compare_20260720_221845` 분석 워크플로; n=200
  `compare_20260720_230025`로 규모 확정)
- **증상**: n=200 census — 3000 run 중 217 run에서 총 289회 실패, 전부 MPC 계열이고
  89–99%가 CW/CDW(run 실패율 **mpc 14.0% vs dobmpc 7.7%** — EAOB FF가 OCP를 오히려
  안정화). 실패 run에서 depth/pitch 결합 극단 excursion — mpc 최대 181.8 cm(pitch 최대
  79.5°), dobmpc 최대 134.7 cm(|pz| 97 cm). **dobmpc radial_max>40 cm는 23/23이 fail run**
  (클린 run 상한 37.8 cm) → worst-case 통계를 이 클래스가 지배. 트리거는 seed-0 공통
  파랑그룹의 lap-7/8 V3 턴 이벤트이고 실패는 증폭자(원인 아님, run-level 연관만 확인 가능).
- **임시 대응**: 분석 시 `n_fail>0` run의 radial_max는 별도 취급(RMS 집계는 강건:
  제외해도 평균 −3~−8%만 이동).
- **제대로 고치려면**: (1) 실패 **시각** 로깅(현재 run당 카운트만 있어 tick-level 인과
  확정 불가), (2) 실패 시 fallback 전략 점검(mpc_acados 실패 경로), (3) traj CSV에
  w_hat·solver-status 기록 추가.
- **2026-07-21 갱신**: reference preview 도입 후 실패율 급감(공유 50 heading 기준 dobmpc
  10–13→1–2런, mpc 16→7/12→4) 및 dobmpc >40 cm 꼬리 소멸 — 그러나 이슈 자체는 잔존
  (mpc CDW에 신규 210 cm blowup; 위 세 수정은 여전히 유효).

### hydro.py는 body_iquat=identity(대각 관성)를 암묵 전제
- **발견**: 2026-07-12 (heavy_gripper NMPC 발산 근본원인 추적으로 발견)
- **증상**: `mj_objectVelocity(mjOBJ_BODY, local=1)`은 **inertial(주축) 프레임** 기준인데
  hydro는 body 프레임으로 간주해 drag를 `xmat`으로 적용. `fullinertia`로 주축이
  정렬·순열되면(Iyy>Izz>Ixx 등) drag 축이 뒤엉켜 **에너지 주입 → 폭발**(torque-free
  kick 0.5 rad/s → 1.5 s 만에 |q|>60 rad/s로 재현).
- **임시 대응**: heavy_gripper 생성 XML이 diaginertia 강제 + `tests/test_heavy_gripper.py`가
  `body_iquat==identity` 회귀 가드. 기존 변종은 원래 대각이라 무증상.
- **제대로 고치려면**: hydro가 `mjOBJ_XBODY`(body 프레임)로 측정하거나 ximat로 변환 —
  물리 파일 수정이라 별도 검증(기존 변종 byte-identical 확인) 필요.
- **2026-07-19 갱신**: C3를 실측 위치(전방-하단)로 옮기면서 버려지는 Ixz가
  −0.0016(0.4%) → **heavy_gripper +0.064 kg·m² (Ixx의 16.8%) / heavy_c3 +0.046 (12.4%)**
  로 커짐. 실기체에 존재할 roll-yaw 곱관성이 플랜트에 없다는 뜻 — hydro를 body-frame으로
  고치기 전까지는 구조적으로 못 넣는다. 위 "제대로 고치려면"의 우선순위가 올라감.

## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

### heavy 회전 added mass = isotropic placeholder
- `[0.12, 0.12, 0.12]`는 임시값 — 문헌 근거 약함(von Benzon 30–100% 오차 보고,
  경쟁하는 0.40 세트 존재). 자체 system ID 전까지 HOLD.

### hydro는 MJX에서 안 돌아감 (`bluerov.xml` fixture로 확인)
- hydro가 CPU passive callback이라 MJX 미지원 — `verify/verify_gpu_mjx.py`의 bonus check가
  `bluerov.xml`(이제 검증 fixture) 로드로 non-gating 확인함. RL phase 전에 hydro의
  MJX 포팅 필요.

### C3-BR 마운트 브래킷 질량은 관성 합성에 미포함 (2026-07-19)
- heavy_gripper·heavy_c3의 브래킷(`meshes/c3_mount.stl`)은 **visual-only** — 재질/질량
  미상이라 `compute_payload_inertia.py` 합성에서 빠져 있음(카메라 1.7 kg 대비 수백 g 추정).
- 사용자에게 실물 브래킷 질량(또는 재질)을 받으면 C3처럼 합성에 추가할 것.

### Newton 그리퍼는 아직 Onshape에 없어 heavy_c3에서 제외 (2026-07-20)
- 사용자 요청: Onshape 어셈블리에 있는 것(차체 + C3)만 반영. 그리퍼는 CAD 추가 전까지
  `heavy_c3`에서 제외. `heavy_gripper` 변종은 그리퍼가 추가될 때를 위한 config로 유지되나,
  현재 그 GRIP_POS=[0.25,0,−0.17]는 여전히 **추정값**(Onshape 미검증)이다.
- 그리퍼가 Onshape에 추가되면: export 재실행 → 브래킷처럼 실측 위치로 GRIP_POS 갱신 →
  heavy_gripper 재생성.

### 방향 sweep이 seed-0 파랑 실현 하나를 공유 — worst-vertex 통계는 단일-실현 아티팩트
- 발견 2026-07-21 (`compare_20260720_230025` 코너 기하 분석): 모든 (current, wave) 헤딩쌍
  run이 **같은 seed-0 파랑 시계열**을 봄(실현 반복 주기 264.8 s ≈ run 길이 266.7 s) →
  특정 절대시각의 wave-group이 매 run 같은 lap/vertex를 때림(dobmpc 400 run 중 181개가
  t=200–210 s에 피크, worst vertex 66%가 V3). 방향 의존 결론은 **per-passage 상대각 통계**
  로만 뽑을 것; vertex별·시각별 주장은 multi-seed 재실행 전에는 출판 불가.
- 발견 2026-07-19 (C3 위치 정합 중): 스킨 bbox = 벤더 치수 × 1.0233 (세 축 균일).
- C3/페이로드 배치는 **실측 metric**(COM 앵커) 기준이라 동역학·카메라는 정확하지만,
  렌더에서 페이로드가 스킨 대비 ~3–5 mm 어긋나 보일 수 있음(코스메틱).

---
*마지막 갱신: 2026-07-21*
