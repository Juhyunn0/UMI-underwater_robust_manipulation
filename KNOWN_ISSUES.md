# KNOWN_ISSUES — 아직 안 고친 것들

> Claude Code 세션 중 발견했지만 **아직 코드로 고치지 않은** 버그·함정·보류 사항의
> 살아있는 목록. 규칙: **고치면 그 항목을 삭제**한다 (고친 기록은 git 히스토리가 담당).
> 새로 발견하면 날짜와 함께 추가한다. 항목 형식: 증상 → 임시 대응 → 제대로 고치는 법.

## 🐛 테스트 / 스크립트 함정

### 📊 compare sweep 재실행 대기 — MPC surge 박스 8 → 30 N 변경 이후 (기록 경계)
- **발견/변경**: 2026-07-24 (wave-crossover 워크플로에서 벤치마크 불공정 발견 → 같은 날 수정)
- **증상**: `params.U_MAX[0]`을 8 → 30 N(= PID `f_max`)으로 고쳤으므로 **기존 recordings의 모든
  mpc/dobmpc 결과는 낡은 8 N 박스 하에서 측정된 것**. 특히 `compare_20260724_160210` 등에서
  관측된 "강파랑에서 PID가 MPC보다 낫다"는 **벤치마크 아티팩트이며 인용 금지**(ablation:
  storm PID−MPC −4.24 → +8.60 cm 부호 역전, crossover 소멸).
- **임시 대응**: 새 run meta는 `controller.u_max`를 기록하므로 신/구 기록은 구분 가능
  (키가 없으면 낡은 8 N). 경계를 넘어 결과를 합산하지 말 것.
- **제대로 고치는 법**: full sweep(`experiments/run_compare.py`, 6 sea state × 5 mode × 3 ctrl)
  재실행 후 이 항목 삭제. 재실행은 사용자 몫.

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
- **2026-07-23 갱신**: EAOB innovation gating을 **구현했으나 실측 후 기본 OFF로 결정**
  (`params.EAOB_GATE_ON=False`, 메커니즘·`n_gated` 카운터·meta 기록은 유지) — square
  seed-0 A/B(tau_dist=0.5 기준)에서 χ²(0.999,18) 게이트가 파랑/코너의 상시적 w_dot=0 위반
  프레임을 25–76% 기각해 파랑 추종 자체를 차단(CDW radRMS 4.5→15.5 cm, CD 1.35→1.43 cm);
  기각이 다시 혁신을 키우는 악순환으로 NIS 통계도 오염(137 vs gate-off 25). 최종 기본
  tau_dist=0.2에선 같은 시나리오에서 게이트가 아예 발동하지 않음(0/1067). 일관성 게이트는
  wave-band 모델 위반과 구조적으로 양립 불가 → 스파이크 방어는 여전히 per-axis
  `W_HAT_CLIP=[15,45,45,5,5,8]` 단일 정의 + 클립 발동 카운터 기록(mpc_acados.py:170·
  mpc.py의 ±50 하드코딩 중복은 그대로)이 담당해야 함. 근본 해결은 harmonic-EAOB.

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
  마지막-2랩 RMS(`rms_lastlaps`)는 내부에서 계산만 하고 어디에도 표시하지 않음.
  (2026-07-28에 *표시되는* RMS는 run_compare와 동일한 steady window로 통일됨 —
  이 항목은 남은 docstring 정리 + 미표시 `rms_lastlaps` 건에 한정.)
- **제대로 고치려면**: docstring 정리, 또는 패널/범례에 마지막-2랩 RMS 추가.

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

### 2026-07-23 base.yaml 파랑 강화 이후 CDW에서 EAOB perf 일관성 깨짐 (소스 무관)
- **발견**: 2026-07-23 (verify_state_source A/B 검증 중; 파랑 블록이 19:28에
  Hs 0.75→1.2 m, Tp 12→6 s, γ 5→2, s 30→10, ω_max 1.6→3.0으로 강화됨)
- **증상**: 새 해상 상태의 CDW에서 mpc_state_source와 무관하게 NIS 80(truth)/71(estimate)
  (DP), 44/42(square) — τ_dist=0.2로 검증했던 목표범위(14–24) 크게 이탈, radRMS도
  DP 1.5→9.6 cm / square 3.5→15.7 cm로 악화. w_dot=0 + τ_dist=0.2가 새 파랑 대역
  (ω_p≈1.05, 에너지 ~3 rad/s)을 못 쫓아가는 것; 구파랑 config로는 전 항목 PASS 재현.
- **임시 대응**: 없음(발산은 아님 — n_fail 0, 유한). `verify/verify_eaob.py`·
  `verify_state_source.py`가 현 config에서 exit 1로 신호.
- **2026-07-24(3) 갱신**: surge 박스 8→30 N 변경 A/B(같은 명령 `verify_eaob --no-plot`,
  CDW/T=60/seed 0)에서 **NIS 245.8→100.0, NEES 398.9→122.0으로 2.5–3.3× 개선**(여전히
  게이트 24 초과 = FAIL). 즉 이 일관성 붕괴의 일부는 **authority 부족으로 인한 큰 추종오차**
  였고(박스가 EAOB의 pseudo-measurement 채널까지 오염), 나머지는 원래 진단대로 τ_dist가
  새 파랑 대역을 못 쫓는 것. 재스윕은 30 N 박스 기준으로 할 것(옛 8 N 수치로 튜닝 금지).
- **제대로 고치려면**: 새 해상 상태 기준으로 EAOB_TAU_DIST 재스윕(0.05–0.1 예상) 또는
  harmonic-EAOB — 어느 쪽이든 실험 설계 결정이라 사용자 판단 필요.

## 📌 알려진 한계 (당장 고칠 계획 없음, 잊지 말 것)

### C3 BNO086 IMU: extrinsic 없음 + 가속도계 스케일 +20% → VIO 쓰려면 캘리브레이션 필수 (2026-07-29)
- **사실 1**: `getImuToCameraExtrinsics(CAM_A)` → `IMU calibration data is not available
  on device yet.` 공장 캘리브레이션은 카메라만 담고 있어 **T_imu_cam이 미지**다.
  (카메라 intrinsics·스테레오 extrinsics는 정상이니 RGB-D/스테레오는 영향 없음.)
- **사실 2**: 정지 상태에서 |a| = **11.8 m/s²** (중력 9.81 대비 **+20%**). raw/calibrated
  둘 다 동일, accuracy 플래그는 UNRELIABLE/MEDIUM. 즉 **IMU intrinsic(스케일/바이어스)이
  미보정**이다. 축 방향(카메라 광축 대비)도 벤더 미문서화·미확정.
- **영향**: visual-inertial SLAM(VIO)에서 스케일과 중력 정렬이 틀어진다. 더 나쁜 건
  **"알고리즘이 안 맞는 것처럼" 실패**해서 원인을 IMU로 의심하기 어렵다는 점.
  RGB-D SLAM / stereo SLAM만 쓸 거면 무관하다.
- **임시 대응**: `c3_collect.py`가 두 사실을 모든 데이터셋의 `metadata.txt`에 명시하고,
  샘플마다 accuracy 필드를 남긴다. `c3_dataset_check.py`도 extrinsic 부재를 경고한다.
  → 동료가 모르고 쓰는 일은 없다.
- **제대로 고치는 법**: (a) Kalibr류로 camera-IMU extrinsic + IMU intrinsic 동시
  캘리브레이션(체커보드, 공기 중 → 수중 재검), 또는 (b) extrinsic만 기계적 실측 +
  accel 스케일/바이어스 6-position 캘리브레이션. 완료 후 `calibration.json`에
  주입하는 경로를 만들고 이 항목 삭제.

### C3 직결(`c3_camera/`): 컬러/depth 촬영시각 skew 약 1 프레임 — 하드웨어 동기 여부 미확인 (2026-07-29)
- **사실**: `c3_stream.py` 기본(`--pair-mode latest`)에서 컬러(CAM_A)와 depth(CAM_B/C 스테레오)의
  `getTimestamp()` 차이가 **약 66 ms(15 fps에서 ≈1 프레임 간격)** 로 관측된다. 컬러 카메라와
  mono 쌍이 같은 파이프라인 안에서 **하드웨어 동기(FSYNC)로 묶여 있지 않기 때문**으로 보인다.
- **영향**: 카메라나 장면이 움직이는 동안 RGB-D 정합이 최대 한 프레임만큼 어긋난다. 정지 상태
  파악에는 무해하지만, **움직이는 매니퓰레이션 데이터 수집·학습에는 실제로 문제가 될 수 있다**
  (66 ms × 0.2 m/s = 1.3 cm 어긋남).
- **임시 대응**: `--pair-mode timestamp` (+`--pair-tolerance-ms`)를 쓰면 두 스트림의 촬영시각이
  허용범위 안에 들어올 때만 bundle을 내보낸다. 대가는 최대 한 프레임 지연.
  `Bundle.skew_ms`가 매 프레임 실제 skew를 보고하므로 HUD에서 감시 가능.
- **제대로 고치는 법**: OAK-D-W-POE가 CAM_A와 mono 쌍의 **FSYNC 하드웨어 트리거를 지원하는지
  미확인**(이 보드의 FFC/FSYNC 배선 여부에 달림). 지원하면 파이프라인에서 동기를 켠다.
  아니면 depthai `Sync` 노드(디바이스측 정렬, 지연 증가) 또는 호스트측 보간을 검토.
  판정 전에는 이 항목 삭제하지 말 것.

### square 참조가 코너에서 동역학적으로 실현 불가 → 지울 수 없는 ~2 cm 코너 오차 바닥 (2026-07-24)
- **사실**: `square_setpoint`의 위치 경로는 각진 사각형이라 코너에서 참조 속도가 한 샘플
  (0.05 s) 만에 90° 뒤집힌다(|Δv|=0.212 m/s → 요구 가속도 사실상 무한). 게다가 yaw 참조는
  60°/s로 슬루(90°에 1.5 s)라 위치 참조와 **서로 모순** — 코너를 실제로 돌 때 필요한 선회율
  126°/s(횡력 8 N)~474°/s(30 N)가 슬루 상한 60°/s를 크게 초과. 힘이 무한해도 기수가 경로를
  못 따라간다. (슬루 자체는 `slew_heading` docstring에 의도된 설계로 명시돼 있음.)
- **정량**(dobmpc, gentle, **NONE 모드 = 외란 0**, lap 2–10 folded): 코너 2.00/1.92/2.04/2.39 cm
  vs 직선 0.22–0.28 cm(≈10×). **surge 박스 8→30 N에서 소수점까지 불변**(코너 명령 surge는
  평균 0.9–1.4 N으로 박스 근처도 안 감, 발동 0.7%) → **authority가 아니라 참조 기하 문제**.
  MPC는 preview+2차 비용으로 코너를 미리 돌아 안쪽으로 자르는(corner-cutting) **최적 절충**을
  하는 것이지 고장이 아님. 파랑 하에선 박스도 일부 기여하나 코너·직선을 거의 같은 비율로
  줄여(33% vs 37%) 코너 특유 효과가 아님; 박스가 실제 개선하는 건 코너 **직후 회복**.
- **영향**: DOB-MPC는 직선이 0.6–1.3 cm로 거의 완벽해서 이 코너 바닥이 오차 예산을 지배하고
  trajectory_compare 그림에서 유독 도드라진다(절대값으로는 3사 중 최소: gentle CDW 코너
  PID 24.6 / MPC 13.2 / DOB 3.8 cm). 컨트롤러 튜닝으로는 제거 불가.
- **줄이려면**: 참조 설계를 고칠 것 — 코너 필렛(원호/스플라인, 반경을 달성 가능 횡력과 yaw
  슬루율에 정합) 또는 코너 감속 프로파일, 또는 yaw 슬루 상한 상향. 어느 쪽이든 벤치마크
  정의가 바뀌므로 기존 기록과 비교 불가 → 사용자 판단 필요.

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
