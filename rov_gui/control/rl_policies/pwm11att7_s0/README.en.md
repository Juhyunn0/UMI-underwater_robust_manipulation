# pwm11att7_s0 — RL_PWM_7D policy (roll/pitch reference tracking, 8 pulses) (installed 2026-10-15, unverified in the pool)

> **English translation.** Source of truth: [`README.md`](README.md) (Korean). Snapshot of 2026-10-15 (install day); later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor/export spec.

- Source: RL_controller `logs/rsl_rl/rov_posehold/2026-10-15_02-01-35_pwm11att7_s0/exported` (`model_1799.pt`, iteration 1799),
  recipe `scripts/_pipeline_pwm11att7_seed.sh 0` = pwm11att6 (current penalty 0.0025, att-fine 0.2, floor termination) + dr preset `pwm11att7`
  (attitude/gyro observation latency 0–0.15 s + link stalls, k down to 2.0 N·m/rad, command delay 0–3 steps — the answer to the 10-13 pool trips
  [측정 RL_controller/logs/eval/pool_trips_1013_rootcause]) + `obs_goal_frame` + trained cap 0.5. The four files have the export's sha1
  (install record SF/install/; SF = RL_controller/logs/eval/station_fix_20261015).
- **The observation uses goal-frame quaternions** (obs_spec `obs_goal_frame: true`, frame_layout `q_des_goalframe` / `q_est_goalframe`: q_des and q_est
  rotated by minus the goal yaw, then w ≥ 0). It needs the 2026-10-15 station support (`rl_policy.HwRl._frame` + `check_frame_layout`); the station
  before it refused it on rl_pwm_7d and would have flown it on rl_pwm with world-frame quaternions [측정 SF/impl-goalframe/NOTES.md].
  The station observation is bit-identical to RL_controller `deploy/obs_builder.py` over 120 frames and a 24-tick history crossing ±180° [측정 SF/verify-accept/REPORT.md,
  SF/install/out/accept_att7_installed.txt].
- Sim gate **full PASS** [측정 RL_controller/logs/eval/PWM11ATT7_GATE.md]: |c_v| 0.03, |c_h| 0.02 (worst 0.03 / 0.05 trim11),
  predicted current 2.5 A (pwm11k0.2_s0 2.4), saturation 0.000, level pitch −1.4°, roll 0.0° (max 3.4°), steady-state angular rate p95 7.5°/s,
  1 m square @0.10 m/s xy RMS **0.028 m** (pwm11k0.2_s0 0.039) / z 0.015 m, hwlike square xy 0.038 m, disturbance RMS 0.083 m.
- Latency / heading [측정 RL_controller/logs/eval/PWM11ATT7_ROBUST.md, trip = |p| or |q| > 40°/s for 3 ticks]: at 0.25 s attitude latency **1 trip (1 env)**
  — pwm11k0.2_s0 12 (5 envs); 0 trips at 0.15 s and in the poolA world; holding heading 180° saturation **0 %**, yaw error RMS 1.7° (pwm11k0.2_s0 10.9°).
  On a 10-13-like stalling link (link1013, negative control) 5 trips (4 envs) — such a link is not flown (the AGE interlock and engage refusal stop it).
- Attitude tracking [측정 RL_controller/logs/eval/ATT_COMPARE_PWM11ATT7.md, sim, still water]: level world (cob_x 0.000), mean over the non-level programs,
  steady error pitch **3.6°** / roll **1.7°** — it tracks like a P controller and keeps a steady error; nose-up goals (pu10 / pu20) are held short
  (−5.9° / −7.5°); worst |ss pitch| 9.1° (cob_x −0.003). Safety emulation 0 trips, max tilt 15.6°. The same document's seed promotion gate
  (B7, |ss pitch| ≤ 2° · |ss roll| ≤ 1.5° [예측]) is failed by every att7 seed — read the roll/pitch reference as "roughly followed".
- Trained cap 0.5 (`pwm_cap_trained`) = `rl_pwm.pwm_cap` 0.5 → builds without `allow_caps_below_trained`. Trained attitude envelope roll ±8.0° /
  pitch ±14.9° (0.14 / 0.26 rad), goal rate 0.35 rad/s — the plan's roll/pitch are clipped to it [스펙 obs_spec attitude_reference].
- **Unverified in the pool.** Preconditions: (1) the AGE interlock armed — at engage the log shows "PER-THRUSTER INTERLOCKS ARMED (rl_pwm_7d): age 0.15 s/0.5 s |
  0.25 s/0.2 s …"; (2) the BlueOS web UI and every other GCS (QGC, Cockpit tabs) closed — parameter download → banner → vehicle-side queue [측정
  pool_trips_1013_rootcause]; (3) vehicle-clock attitude age (CSV `att_age_v_s`) p99 at or below ~0.09 s at pool step 0 on the day [예측, at 50 Hz,
  SF/impl-rates/NOTES.md]; (4) the per-thruster sign probe artifact (currently data/20261013/1013_181137_probe, `thruster_probe_max_age_days` 30) —
  without it engage is refused (by design). Order: KNOWN_ISSUES entry 'pwm11att7_s0 를 설치' (step 0 → 0c → 6-D pwm11k0.2_s0 level hold 30 s, 60 s → rl_pwm_7d).
- 6-D note [측정 SF/verify-accept/REPORT.md F1]: this export also builds on rl_pwm (6-D, no roll/pitch reference) and flies level there — the observation matches
  the builder and level goals are in its training distribution (starts_level). rl_pwm neither warns about nor refuses attitude_reference.trained, so
  rl_pwm.policy_dir stays pwm11k0.2_s0 (operator's call).
- Rollback: `rl_pwm_7d.policy_dir: null` in `config/hw_mpc.yaml` (LOW off, the 2026-10-10 state). This folder can stay; nothing reads it then.
