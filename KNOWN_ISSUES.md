# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## 🐛 테스트 / 스크립트 함정

### `test_dobmpc.py` — 기본 `ROV_MODEL`(heavy)로는 FAIL
- **발견**: 2026-07-06 (README 커맨드 레퍼런스 작업 중 직접 실행으로 확인)
- **증상**: `python test_dobmpc.py`(기본 heavy) → trim 테스트 실패. 6 N ≈ 23° pitch-trim
  기준값이 bluerov2 파라미터(controller.py:43 유래)로 유도된 것이라 heavy에서는 어긋남.
- **임시 대응**: `ROV_MODEL=bluerov2 python test_dobmpc.py` (README §1에 이렇게 문서화)
- **제대로 고치려면**: test_observe.py / test_water_viz.py처럼 파일 상단에서
  `os.environ.setdefault("ROV_MODEL", "bluerov2")`, 또는 variant별 기대값 분기.

### 루트 스모크 테스트 3개 — `bluerov.xml` 하드코딩, heavy 커버리지 없음
- **발견**: 2026-07-06
- **증상**: `test_load.py:30`, `test_thrusters.py:30`, `test_controller.py:23`이
  rank-5 `bluerov.xml`을 직접 로드 — `ROV_MODEL`을 무시하므로 기본 variant인
  **heavy(8-thruster, rank-6)는 대응하는 스모크 테스트가 없음**.
- **임시 대응**: 없음 (heavy 검증은 verify_hydro / dobmpc 경로에 부분 의존)
- **제대로 고치려면**: `rov_model.XML_PATH` 기반으로 일반화하고 thruster 수 / rank /
  질량 기대값을 variant별 상수로 assert.

### `gen_pool_apriltags.py --selftest` — `tag_floor.xml`을 덮어씀
- **발견**: 2026-07-06 (`--tag-mode plane` 개편 후에도 유효 — `run_selftest`가
  gen_pool_apriltags.py:475에서 tag_floor.xml을 테스트 타일 2개짜리로 씀)
- **증상**: selftest 후 POOL_TAGS 씬이 타일 2개짜리 바닥으로 로드됨.
- **임시 대응**: selftest 후 `python gen_pool_apriltags.py` full build 재실행
  (README §7에 경고 있음).
- **제대로 고치려면**: selftest는 별도 임시 파일에 쓰고 종료 시 삭제
  (기존 `_selftest_scene.xml`처럼).

### `plot_wave_spreading.py` — config/base.yaml의 하드코딩 복사본
- **발견**: 2026-07-06
- **증상**: Hs/Tp/gamma/s/h/N_omega/N_beta가 스크립트 상수로 복사돼 있음(yaml을 읽지
  않음) → config를 바꾸면 슬라이드 figure가 실제 실험과 **조용히** 어긋남.
- **제대로 고치려면**: `disturbance.config.load_config`로 yaml을 직접 읽기.

### `analyze_square3.py` / `analyze_acados_vs_before.py` — 경로 하드코딩
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

### `verify_acados.py` 게이트가 heavy_gripper에서 근소 초과 (0.2717 > 0.25 N)
- **발견**: 2026-07-12 (heavy_gripper 변종 검증 중)
- **증상**: acados RTI vs IPOPT worst-case |Δu| 게이트 0.25 N은 heavy 기준 캘리브레이션;
  heavy_gripper(13.7 kg)에서 0.2717 N — 30 N sway authority의 ~0.9%라 실효 동일 최적해,
  폐루프도 검증됨(DP hold 1.3 cm). heavy/bluerov2는 여전히 PASS.
- **제대로 고치려면**: 게이트를 변종별 스케일 또는 ‖u‖ 상대비로.

### hydro.py는 body_iquat=identity(대각 관성)를 암묵 전제
- **발견**: 2026-07-12 (heavy_gripper NMPC 발산 근본원인 추적으로 발견)
- **증상**: `mj_objectVelocity(mjOBJ_BODY, local=1)`은 **inertial(주축) 프레임** 기준인데
  hydro는 body 프레임으로 간주해 drag를 `xmat`으로 적용. `fullinertia`로 주축이
  정렬·순열되면(Iyy>Izz>Ixx 등) drag 축이 뒤엉켜 **에너지 주입 → 폭발**(torque-free
  kick 0.5 rad/s → 1.5 s 만에 |q|>60 rad/s로 재현).
- **임시 대응**: heavy_gripper 생성 XML이 diaginertia 강제 + `test_heavy_gripper.py`가
  `body_iquat==identity` 회귀 가드. 기존 변종은 원래 대각이라 무증상.
- **제대로 고치려면**: hydro가 `mjOBJ_XBODY`(body 프레임)로 측정하거나 ximat로 변환 —
  물리 파일 수정이라 별도 검증(기존 변종 byte-identical 확인) 필요.
- **2026-07-19 갱신**: C3를 실측 위치(전방-하단)로 옮기면서 버려지는 Ixz가
  −0.0016(0.4%) → **heavy_gripper +0.064 kg·m² (Ixx의 16.8%) / heavy_c3 +0.046 (12.4%)**
  로 커짐. 실기체에 존재할 roll-yaw 곱관성이 플랜트에 없다는 뜻 — hydro를 body-frame으로
  고치기 전까지는 구조적으로 못 넣는다. 위 "제대로 고치려면"의 우선순위가 올라감.

## ⏳ 보류 중 (알고 있지만 아직 안 돌린 것)

### 새 PID gain으로 full compare 미재실행
- **상태**: 2026-07-03에 heavy PID를 pole-placed `GAINS_HEAVY`로 교체했지만
  **run_compare full matrix는 old gains 결과가 마지막**. 이전 PID 결과와 비교할 땐
  각 결과 폴더 `meta.json`의 `pid_gains`로 구분할 것.
- **할 일**: `python -m experiments.run_compare --config config/base.yaml` 재실행 후
  결과 figure/표 갱신.

## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

### heavy 회전 added mass = isotropic placeholder
- `[0.12, 0.12, 0.12]`는 임시값 — 문헌 근거 약함(von Benzon 30–100% 오차 보고,
  경쟁하는 0.40 세트 존재). 자체 system ID 전까지 HOLD.

### bluerov2 variant 스킨은 아직 회색
- 컬러 스킨(cyan/white/black/silver)은 heavy 전용; bluerov2는 gray body 그대로
  (future work).

### `bluerov.xml`은 MJX에서 안 돌아감
- hydro가 CPU passive callback이라 MJX 미지원 — `verify_gpu_mjx.py`의 bonus check가
  non-gating으로 확인함. RL phase 전에 hydro의 MJX 포팅 필요.

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

### sim 차체 스킨은 실물 대비 2.3% 큼 (MarineGym USD 유래)
- 발견 2026-07-19 (C3 위치 정합 중): 스킨 bbox = 벤더 치수 × 1.0233 (세 축 균일).
- C3/페이로드 배치는 **실측 metric**(COM 앵커) 기준이라 동역학·카메라는 정확하지만,
  렌더에서 페이로드가 스킨 대비 ~3–5 mm 어긋나 보일 수 있음(코스메틱).

---
*마지막 갱신: 2026-07-19*
