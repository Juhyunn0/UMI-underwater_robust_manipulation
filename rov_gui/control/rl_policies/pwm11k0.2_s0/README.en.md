# pwm11k0.2_s0 — RL_PWM candidate policy (2026-10-10, unverified in the pool)

> **English translation.** Source of truth: [`README.md`](README.md) (Korean). Snapshot of 2026-10-12, source blob `5f0dd9a580`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

- Source: RL_controller `logs/rsl_rl/rov_posehold/2026-10-10_*_pwm11k0.2_s0/exported` (iteration 1199), recipe `scripts/_pipeline_pwm11_seed.sh 0 0.2`
  = corrected hydrostatics world (pwm11 preset: nose-up rest angle −3..+12°, arm 0.05 m, k 3.4~10.8 N·m/rad) + null-space penalty k 0.2 + trained cap 0.5.
- All sim gates passed [측정 RL_controller/logs/eval/PWM11_GATE.md]: |c_v| 0.01, |c_h| 0.03 (worst world 0.03), cancelled thrust ≈ 0 N,
  predicted current 2.4 A (pwm10_s1 11.1, att7pwm_s0 44), level pitch −1.1°, hwlike world tilt 1.7° (pwm10_s1 +16°), steady-state angular rate p95 6.9°/s,
  1 m square @0.10 m/s xy RMS 0.039 m / z 0.021 m, disturbance RMS 0.089 m (traj10_s1 0.062: the cost of the authority cap 0.5 [유도]).
- Trained cap 0.5 (`pwm_cap_trained`) → the station uses the min of `rl_pwm.pwm_cap` and trained, and refuses a setting less than trained.
- Engagement condition: a per-thruster sign probe artifact (`rov_gui/tools/thruster_probe.py`, docs/thruster_probe.md) is required. Without it, RL_PWM engagement is refused (by design).
- First water ladder [예측]: 60 s station hold ×3 → 1 m square, 2 laps @0.10 m/s → 0.25 m step. Gates: current ≤ 7 A, |null_v_us|,|null_h_us| ≤ 40 µs,
  |pitch−rest| ≤ 5°, angular rate p95 ≤ 20°/s, 0 interlock trips. Analysis/evidence: RL_controller/logs/eval/pwm_cocontraction_analysis_20261010/REPORT.md, README §13.
