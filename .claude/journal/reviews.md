# P2 — Parallel-review syntheses

<!-- Populated when P2 (/review-change) lands. One block per review:
     ## YYYY-MM-DD <artifact>
     reviewers: <agents> · 합의: … · 이견: … · 리스크: … · 최소수정: … -->

## 2026-06-22 — Heavy 관성 텐서 [0.3291,0.6347,0.6109] (parallel-axis 유도, farol USD 대신)
reviewers: simulation-advisor · control-theory-advisor · underwater-robotics-advisor (병렬·독립)

- **합의(3/3):** (a) parallel-axis 유도법 타당, farol [0.21,0.245,0.245] 기각 정당(코너 질량 추가가 Iz를 낮출 수 없음 → 비물리·Gazebo hand-tune). (b) **in-sim non-issue** — MJCF plant `diaginertia` == predictor(params)라 closed-loop이 관성값에 불변; 논쟁은 sim-to-real에서만 유효. (c) 0.15kg/thruster는 예산 bookkeeping(실제 T200 ~0.344kg)이나 delta만 스케일 → 영향 작음(~0.03).
- **주요 발견(control+underwater 합의, sim도 지적):** dry inertia를 소수 3자리까지 다듬는 동안 **회전 added mass [0.12,0.12,0.12] (BlueROVHeavy.yaml)는 등방 placeholder로 방치** — 생략한 유체 r² 항이 계산한 ΔI와 동급 크기("두 동급 항 중 작은 쪽만 정밀화"). 권장 anisotropic ≈ Kp'0.07 / Mq'0.18 / Nr'0.22 (von Benzon&Fossen 2022 order로 주장됨 → P3에서 검증).
- **sim 지적:** 텐서 3중 복제(rov_model.py:54 · MJCF diaginertia · compute_heavy_inertia.py 출력)를 묶는 verify/test 없음; 스크립트가 baseline 하드코딩 → drift 위험.
- **control 지적:** Heavy는 roll/pitch commanded(NU=6) → 관성이 "don't care"→"~10-20% 중요"로 승격. EAOB는 quasi-static 오차는 흡수하나 ω̇-상관(기동 중) 오차는 못 함.
- **이견:** sim/control "in-sim 수정 불요(값 유지)" vs underwater "added mass 지금 고쳐라" → 화해: in-sim 깨진 것 없음, 단 added-mass가 진짜 fidelity gap이라 다음 투자는 dry inertia가 아니라 거기.
- **리스크:** 회전 added-mass 과소·등방(sim-to-real 자세 fidelity); 텐서 3중 복제 drift.
- **최소수정(영향순):** ① 민감도 스윕(plant 고정, predictor 관성 farol vs 유도 → pitch/roll·ŵ RMS 비교 → EAOB 흡수 정량화) ② BlueROVHeavy.yaml 회전 added mass anisotropic화 ③ test_heavy_inertia 3중 일치 게이트 + 스크립트 baseline import.
- **검증법:** roll/pitch step 후 ŵ의 ω̇-상관 펄스 관찰(관성오차 지문); yaw step→Iz 직접; bifilar(공기)+in-water decay 실측 bound.

- 2026-07-01 [workflow: review-observe-mode (4 dims × 2 skeptics)] Q: teleop `--observe` 자유표류 diff 적대적 리뷰(역학누출/루프·스레드/UX) → 5지적 중 1 확정: viser observe에서 조종버튼이 여전히 보임(비기능) → 숨김 처리. 반증 4: ①"hydro.reset()이 byte-identical 깨뜨림"=오판 — reset()은 필터상태만(계수/모델 불변, hydro.py:108-114), recenter는 의도적 상태리셋이라 byte-identical은 "recenter 미발동 시"만 주장. 오히려 reset 안 하면 표류속도가 필터에 남아 거대 added-mass 스파이크 발생 → 호출이 정답(viser Reset pose와 동일) ②③ _recenter_request 락 없음 = 무해(bool·멱등, 기본경로는 매프레임 재중심) ④ recenter CSV 속도 불연속 = 의도된 상태리셋. [memory: observe-free-drift-mode]

- 2026-07-03 [workflow: review-pid-poleplacement-apply (transcription+regression × verifier)] Q: pole-placement PID 적용 diff(GAINS_HEAVY/GAINS_BLUEROV2 분기 + r_ref yaw-rate FF) 적대적 리뷰 → 전사·게인값·r_cmd 계산 무결; 실회귀 1건 확정(실행 재현): test_controller.py·test_square_mission.py가 bluerov.xml(rank-5)을 하드코딩하는데 DEFAULT_GAINS 기본이 heavy가 되면서 GAINS_HEAVY 주입 → pitch 텀블(46.8°/58.7° > 45°) 테스트 실패 → 두 테스트에 `gains=C.GAINS_BLUEROV2` 명시 고정으로 수정, 기본 env 재실행 전부 PASS. [memory: pid-gains-provenance]

- 2026-07-07 [workflow: review-run-compare-extension (4 dims × adversarial verify)] Q: run_compare 원샷 배치 확장 diff(pairing:grid 방향그리드 + per-run runs/ CSV·meta + trajectory_compare 전방향 오버레이 + MPC 파랑) 적대적 리뷰 → 16 findings, 확정 6(중복 제거): ①_load_run_traj 무방비 genfromtxt — 부분/1행 CSV 하나가 그림 단계에서 main을 죽여 results.csv 통째 소실(재현됨) → try/except+atleast_1d+호출부 가드+부분파일 삭제 ②plot_trajectories._find가 새 _c/_w 파일명에서 --dir-deg/--seed 무시하고 sorted[-1]을 조용히 그림(mislabel, 재현됨) → seed·heading 패턴 추가+fallback WARN ③같은 scenario 두 블록이 runs/ 파일 경쟁 → 명시적 ValueError ④범례 n=sweep×seeds 과대표기 → n≤N 처리 ⑤단일모드 그림 full-run RMS 무표기 → 범례 title 명시 ⑥record_runs 디스크 무경고(dp 1000s×120런≈160MB) → 시나리오 길이 기반 추정 상시 출력. 반증 5(_deg_tag 충돌, wave_swept 일탈 등). 추가로 그리드 테스트가 기존 버그 발견: _prebuild_acados가 dobmpc 전용이라 mpc-only 병렬런이 tera 경합→IPOPT 폴백(20배 느림) → mpc/dobmpc 공용 프리빌드로 수정(107.6s→5.7s/run). 전부 수정 후 40런 e2e 재검증 PASS. [memory: finite-depth-disturbance-env]

- 2026-07-21 [P2: simulation-advisor + control-theory-advisor (병렬 독립)] Q: MPC reference-preview 구현 diff(set_reference_traj/_xref_ned_traj + make_square_ref + run_one/run_viewer 배선 + 테스트) 리뷰 → **wrong 0건 합의**(프레임 S-conjugation·−r 부호·stage/terminal 시간정렬·unwrap 앵커·샘플러=라이브루프 정렬 모두 양측 독립 검산 통과). 확정 최소수정 3건 반영: ①meta.json에 controller.ref_preview provenance(이전 기록과 비교불가 문제 — pid_gains 전례와 동일 원칙) ②샘플러 (K,3) 전치 반환의 silent scramble 방지 shape assert ③reset()이 _ref_traj 해제(스테일 샘플러 오동작 예방). 권고 반영: 샘플러↔라이브루프 동치 회귀 테스트 추가(test_square_ref_matches_live_loop, 1랩 <1e-9). 이론 리뷰 판정: mid-turn sway 속도참조 = position preview와 유일하게 운동학적 일관 선택(정답), sharp square 참조 유지가 벤치마크로 옳음(smoothing은 과제 변경), RTI warm-start는 오히려 개선 예상(vertex에서 전 stage 90° 동시회전 → tick당 1샘플 미끄러짐), QN=Q·terminal r(t0+N·dt) 정합. 이견 없음. 실험 재실행은 사용자 몫. [memory: mpc-reference-preview]
