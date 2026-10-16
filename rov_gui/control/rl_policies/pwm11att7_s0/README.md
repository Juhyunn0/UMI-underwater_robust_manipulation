# pwm11att7_s0 — RL_PWM_7D 정책(roll/pitch 기준 추종, 펄스 8개) (2026-10-15 설치, 수조 미검증)

- 출처: RL_controller `logs/rsl_rl/rov_posehold/2026-10-15_02-01-35_pwm11att7_s0/exported` (`model_1799.pt`, iteration 1799),
  레시피 `scripts/_pipeline_pwm11att7_seed.sh 0` = pwm11att6(전류 벌점 0.0025, att-fine 0.2, 바닥 종료) + dr 프리셋 `pwm11att7`
  (자세·자이로 관측 지연 0–0.15 s + 링크 스톨, k 2.0 N·m/rad 까지, 명령 지연 0–3 스텝 — 10-13 수조 트립 원인 대응
  [측정 RL_controller/logs/eval/pool_trips_1013_rootcause]) + `obs_goal_frame` + 학습 캡 0.5. 네 파일은 export 와 sha1 동일
  (설치 기록 SF/install/; SF = RL_controller/logs/eval/station_fix_20261015).
- **관측이 goal-frame 쿼터니언이다**(obs_spec `obs_goal_frame: true`, frame_layout `q_des_goalframe` / `q_est_goalframe`: q_des·q_est 를
  목표 yaw 의 반대로 돌린 뒤 w ≥ 0 정규화). 2026-10-15 스테이션 지원(`rl_policy.HwRl._frame` + `check_frame_layout`)이 있어야 하고,
  그 전 스테이션은 rl_pwm_7d 에서 거부하고 rl_pwm 에서는 world-frame 쿼터니언으로 잘못 날았을 것이다 [측정 SF/impl-goalframe/NOTES.md].
  스테이션 관측은 RL_controller `deploy/obs_builder.py` 와 120 프레임·24틱 이력(±180° 통과)에서 비트 동일 [측정 SF/verify-accept/REPORT.md,
  SF/install/out/accept_att7_installed.txt].
- 시뮬 게이트 **full PASS** [측정 RL_controller/logs/eval/PWM11ATT7_GATE.md]: |c_v| 0.03, |c_h| 0.02(최악 0.03 / 0.05 trim11),
  예측 전류 2.5 A(pwm11k0.2_s0 2.4), 포화 0.000, 수평 pitch −1.4°, roll 0.0°(최대 3.4°), 정상상태 각속도 p95 7.5°/s,
  사각형 1 m @0.10 m/s xy RMS **0.028 m**(pwm11k0.2_s0 0.039) / z 0.015 m, hwlike 사각형 xy 0.038 m, 외란 RMS 0.083 m.
- 지연·헤딩 [측정 RL_controller/logs/eval/PWM11ATT7_ROBUST.md, 트립 = |p|·|q| > 40°/s 3틱]: 자세 지연 0.25 s 에서 트립 **1회(1 env)**
  — pwm11k0.2_s0 는 12회(5 env); 0.15 s 와 poolA 세계 트립 0; 헤딩 180° 유지 포화 **0 %**, yaw 오차 RMS 1.7°(pwm11k0.2_s0 10.9°).
  10-13 같은 스톨 링크(link1013, 음성 대조)는 5회(4 env) — 그런 링크에서는 날지 않는다(AGE 인터록·교전 거부가 막는 대상).
- 자세 추종 [측정 RL_controller/logs/eval/ATT_COMPARE_PWM11ATT7.md, 시뮬 정수]: 수평 세계(cob_x 0.000) 비-level 프로그램 평균
  정상오차 pitch **3.6°** / roll **1.7°** — P 제어처럼 따라가며 정상오차가 남고, 기수-위 목표(pu10 / pu20)는 못 미친 채 유지
  (−5.9° / −7.5°); 최악 |ss pitch| 9.1°(cob_x −0.003). 안전 에뮬레이션 트립 0, 최대 기울기 15.6°. 같은 문서의 시드 승격 기준
  (B7, |ss pitch| ≤ 2° · |ss roll| ≤ 1.5° [예측])은 att7 시드 전부 미달 — roll/pitch 기준은 '대략 따라감'으로 읽을 것.
- 학습 캡 0.5(`pwm_cap_trained`) = `rl_pwm.pwm_cap` 0.5 → `allow_caps_below_trained` 없이 빌드. 학습 자세 범위 roll ±8.0° /
  pitch ±14.9°(0.14 / 0.26 rad), 목표 각속도 0.35 rad/s — 플랜의 roll/pitch 는 이 범위로 잘린다 [스펙 obs_spec attitude_reference].
- **수조 미검증.** 비행 전제: (1) AGE 인터록 armed — 교전 때 로그 "PER-THRUSTER INTERLOCKS ARMED (rl_pwm_7d): age 0.15 s/0.5 s |
  0.25 s/0.2 s …" 확인; (2) BlueOS 웹 UI 와 다른 GCS(QGC·Cockpit 탭) 닫기 — 파라미터 다운로드 → 배너 → 차량 쪽 큐 [측정
  pool_trips_1013_rootcause]; (3) 당일 수조 0단계에서 차량 시계 자세 나이(CSV `att_age_v_s`) p99 ≤ ~0.09 s [예측, 50 Hz 기준
  SF/impl-rates/NOTES.md]; (4) 추진기별 부호 프로브 아티팩트(현재 data/20261013/1013_181137_probe, `thruster_probe_max_age_days` 30) —
  없으면 교전 거부(설계). 순서: KNOWN_ISSUES 'pwm11att7_s0 를 설치' 항목(0단계 → 0c → 6-D pwm11k0.2_s0 수평 hold 30 s·60 s → rl_pwm_7d).
- 6-D 주의 [측정 SF/verify-accept/REPORT.md F1]: 이 export 는 rl_pwm(6-D, roll/pitch 기준 없음)에도 빌드되고 거기서는 수평(level q_des)으로
  난다 — 관측은 빌더와 일치하고 level 목표는 학습 분포 안(starts_level). rl_pwm 은 attitude_reference.trained 를 경고·거부하지 않으니
  rl_pwm.policy_dir 은 pwm11k0.2_s0 그대로 둔다(운용자 판단).
- 롤백: `config/hw_mpc.yaml` 의 `rl_pwm_7d.policy_dir: null`(LOW 꺼짐, 2026-10-10 상태). 이 폴더는 남겨 둬도 아무것도 읽지 않는다.
