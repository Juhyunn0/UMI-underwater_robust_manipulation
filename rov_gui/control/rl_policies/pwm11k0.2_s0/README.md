# pwm11k0.2_s0 — RL_PWM 후보 정책 (2026-10-10, 수조 미검증)

- 출처: RL_controller `logs/rsl_rl/rov_posehold/2026-10-10_*_pwm11k0.2_s0/exported` (iteration 1199), 레시피 `scripts/_pipeline_pwm11_seed.sh 0 0.2`
  = 보정된 유체정역학 세계(pwm11 프리셋: 기수-위 정지각 −3..+12°, 팔 0.05 m, k 3.4~10.8 N·m/rad) + null-space 벌점 k 0.2 + 학습 캡 0.5.
- 시뮬 게이트 전부 통과 [측정 RL_controller/logs/eval/PWM11_GATE.md]: |c_v| 0.01, |c_h| 0.03(최악 세계 0.03), 상쇄 추력 ≈ 0 N,
  예측 전류 2.4 A(pwm10_s1 11.1, att7pwm_s0 44), 수평 pitch −1.1°, hwlike 세계 기울기 1.7°(pwm10_s1 +16°), 정상상태 각속도 p95 6.9°/s,
  사각형 1 m @0.10 m/s xy RMS 0.039 m / z 0.021 m, 외란 RMS 0.089 m(traj10_s1 0.062: 권한 캡 0.5 의 대가 [유도]).
- 학습 캡 0.5 (`pwm_cap_trained`) → 스테이션은 `rl_pwm.pwm_cap` 과 trained 의 min 을 쓰고, trained 미만 설정은 거부.
- 교전 조건: 추진기별 부호 프로브 아티팩트(`rov_gui/tools/thruster_probe.py`, docs/thruster_probe.md) 필수. 없으면 RL_PWM 교전 거부(설계).
- 첫 수중 래더 [예측]: 60 s station hold ×3 → 1 m 사각형 2바퀴 @0.10 m/s → 0.25 m 스텝. 게이트: 전류 ≤ 7 A, |null_v_us|,|null_h_us| ≤ 40 µs,
  |pitch−rest| ≤ 5°, 각속도 p95 ≤ 20°/s, 인터록 트립 0회. 분석/근거: RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md, README §13.
