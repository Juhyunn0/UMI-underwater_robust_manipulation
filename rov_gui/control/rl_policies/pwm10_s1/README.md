# pwm10_s1 — per-thruster RL policy for LOW mode `rl_pwm` (GUI: RL_PWM)

Shipped 2026-09-30. **Never run on the vehicle.**

| file | what | source |
|---|---|---|
| `obs_spec.json` | observation/action contract, `output.pwm_model` (motor_direction, T200 curve, B) | `RL_controller/logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s1/exported/` (sha1 6432eda5…) |
| `policy_weights.npz` | actor MLP, numpy | same folder (sha1 0dcde6f7…) |
| `rl_pwm_override.lua` | the VEHICLE-side script this transport needs | `RL_controller/deploy/ardusub/rl_pwm_override.lua` (sha1 03e21778…), untested [예측] |

Action: 8 normalized throttles, ArduSub motor order 1..8; pulse = 1500 + 400 x throttle [us]; the mixer is bypassed.
Expected `MOT_1..8_DIRECTION`: `[-1, -1, 1, 1, -1, 1, 1, -1]` — the station refuses to engage if the vehicle differs.

## Before the first run
1. Copy `rl_pwm_override.lua` to the Navigator's `scripts/` folder; set `SCR_ENABLE 1` (reboot), `SCR_HEAP_SIZE >= 100000`,
   `SCR_USER1 1`. Leave `SERVO1..8_FUNCTION` at 33..40.
2. Nothing may sit on the RCIN passthrough of RC 9..16 (lights / camera tilt / gripper outputs with
   `SERVOn_FUNCTION` 59..66): move it, or the station refuses. `RC9..16_OPTION` 0, `RC_OPTIONS` bit 1 clear,
   `RC_OVERRIDE_TIME` > 0, `SYSID_MYGCS` = the station's sysid (255).
3. PROPELLERS OFF on the bench: pick LOW `RL_PWM`, arm in MANUAL, engage. Expect the vehicle message
   `RLPWM: engaged`, pulses on `pwm1..8` within 1400..1600 us, and 1500 us on E-STOP / disengage / stick takeover.
4. Per-thruster sign probe (RL_controller/README.md §10.3) before any water.

5. While engaged on the bench, watch for `RLPWM: released` appearing mid-run: the station sends a frame every 50 ms and
   the script calls a 100 ms gap stale. If it flickers, the timing margin is too thin for water.

`rl_pwm.pwm_cap` in `config/hw_mpc.yaml` is 0.25 for the first water run; the policy was trained at 1.0. Motors 1..8
ride RC channels 9..16 in that order and that is not configurable. The gate only trusts what the vehicle said in the
last 30 s (re-read every 10 s while RL_PWM is selected).
Station side: `rov_gui/control/rl_pwm.py` (gate, wire frame), `rov_gui/control/rl_policy.py` `HwRlPwm`,
`rov_gui/backends/hardware.py` `MavlinkCommandSink.set_pwm`, tests `rov_gui/tests/test_rl_pwm.py`.
