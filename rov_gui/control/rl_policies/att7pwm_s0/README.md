# att7pwm_s0 — RL_PWM_7D policy (ENABLED 2026-10-03 by operator decision; failed the sim gate)

Per-thruster RL policy (`action_mode: pwm`, 8 throttles) trained WITH a roll/pitch reference in q_des — RL_controller
run `logs/rsl_rl/rov_posehold/2026-10-03_15-44-54_att7pwm_s0/` (model_1199), recipe `scripts/_pipeline_att7pwm_seed.sh`
(dr preset `traj10att`, **still water: no waves, no current** — `obs_spec.sea_state_training.p_mode` = NONE; the traj10
tether-like bias 0-4 N is still there), 2048 envs x 1200 it, 34:38 (`logs/train_att7pwm_s0.log`). Copied 2026-10-03.
Station LOW `rl_pwm_7d` (control/rl_policy.HwRlPwm7d); same Lua transport as pwm10_s1 (`rl_pwm_override.lua` is an
identical copy of the one already installed on the vehicle). motor_direction == pwm10_s1's, pwm_cap trained 1.0.

**Enabled 2026-10-03 at the operator's request** (`config/hw_mpc.yaml rl_pwm_7d.policy_dir:
rov_gui/control/rl_policies/att7pwm_s0`; it was shipped disabled first). Why it was held back — the still-water attitude evaluation
(`RL_controller/logs/eval/ATT_COMPARE_20261003.md`, `logs/eval/att_att7pwm_s0/`) says it tracks a tilted reference far
better than the level-trained pwm10_s1 (world cob 0: pitch RMS 1.46 vs 10.93 deg over 14 programs; pd20 steady error
+1.1 vs +27.3 deg) but it FAILS the [예측] promotion gate: worst steady pitch error 7.1 deg and roll 4.9 deg (in the
-11 deg trim world), a +-3 deg roll wobble even when holding level (e_roll RMS 2.8 deg vs pwm10_s1 0.6), vertical
|throttle| ~0.7 at level (pwm10_s1 0.3) and 24-38 % saturated ticks (pwm10_s1 0 %). s0 is the best of three seeds
(s1 57-83 %, s2 ~100 % saturated). Standard level evaluation: square pos RMS 0.041 m, sat 16 %
(`logs/eval/traj_att7pwm_s0_square/metrics.csv`) vs pwm10_s1 0.036 m, 1 %.

To disable it again: `rl_pwm_7d.policy_dir: null`. Before water: everything in `rov_gui/control/rl_policies/pwm10_s1/README.md` (propellers-off bench, per-thruster direction),
first with a 5-dim policy or a station hold (level reference), then the 7-dim checkpoint.
