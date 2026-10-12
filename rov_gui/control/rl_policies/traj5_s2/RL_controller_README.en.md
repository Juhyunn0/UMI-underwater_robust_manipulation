# RL_controller — reproducing "Learning to Swim": our ROV (BlueROV2 Heavy + C3 + Newton gripper), Vision + IMU only

> **English translation.** Source of truth: [`RL_controller_README.md`](RL_controller_README.md) (Korean). Snapshot of 2026-10-12, source blob `6556791ee4`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

This is a standalone package that re-implements the pipeline of the paper *Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven AUVs* (Cai, Chang, Girdhar, ICRA 2025,
[arXiv:2410.00120](https://arxiv.org/abs/2410.00120), code `warplab/isaac-auv-env`) — **Isaac Lab parallel simulation + RSL-RL PPO +
domain randomization → zero-shot transfer to hardware** — adapted to our vehicle and sensor configuration.
It does not import any code from outside this folder (the model, parameters, estimator and command path are all copied/rewritten into it).

Evidence-tag rules: `[측정]` ("measured") comes with the path of the measured artifact, `[스펙]` ("spec") datasheet/code constant, `[유도]` ("derived") calculation, `[예측]` ("predicted") unverified assumption.

## 1. What is the same as the paper and what is different

| Item | Paper (CUREE, WarpAUV) | This implementation (our ROV) |
|---|---|---|
| Simulator | Isaac Lab DirectRLEnv, 2048 envs, 120 Hz physics | Same (Isaac Sim 5.0 + Isaac Lab 2.2.0, `~/IsaacLab`, conda `env_isaaclab`) |
| Vehicle | 22.7 kg, 6 thrusters | BlueROV2 **Heavy** 8×T200 + C3 camera (top mount, tilted 43° down) + Newton gripper, **13.72 kg** (`configs/rov_params.yaml`, parallel-axis composite) |
| Dynamics | MuJoCo-style inertia-box drag + buoyancy (COB offset) + `C_t·Ω\|Ω\|` | Same drag model + nominal linear damping + **added mass M_A·ν̇ + C_A(ν)ν** (Fossen; 2026-09-21, `rov_rl/hydrodynamics.py`) + **MarineGym T200 curve** (voltage scale, deadband, 1st-order lag, command delay) |
| Action | Direct PWM to the 6 thrusters | **4-axis MANUAL_CONTROL** `[surge, sway(+우), heave(+상), yaw(+시계)]` ("[surge, sway(+right), heave(+up), yaw(+clockwise)]") → **ArduSub vectored-6DOF mixer** (`rov_rl/mixer.py`) → 8 throttle. Mixer, gains and motor signs verified against 64,512 rows of hardware logs [측정 `logs/mixer_fit_report.md`] |
| Sensors | DVL + IMU + AprilTag EKF (v_body observed directly) | **No DVL**: simulates tag PnP (≈15 Hz, ≈130 ms latency, noise) + ArduSub ATTITUDE/gyro, and builds v_body with **the same estimator** as the station's `state_assembler` (finite difference + LPF α=0.6, gyro/velocity bridging) (`rov_rl/sensors.py`, `rov_rl/estimator.py`) |
| Observation | 17-dim `[q_des, x_off_b, q, v_b, ω_b]` | 17 + goal velocity/yaw rate 4 (Section 8) + previous action 4 = 25, stacked as a **4-frame history** → 100-dim (latency compensation). Quaternion normalized to w≥0 (7.3). The 2026-09-16 runs are 84-dim |
| Reward | `0.2e^{-\|x\|²} + 0.5e^{-angle} + 0.2e^{-\|a\|²}` | Same (`rov_rl/rewards.py`, yaw-only option) |
| Control rate | 60 Hz training / 20 Hz deployment | **20 Hz** training = deployment (physics 1/120 s, decimation 6), episode 20 s (6 s for the 2026-09-16 runs) |
| DR | COB offset sphere, volume | `none/small/large` (paper counterparts) + `ours` (mass, inertia, drag, voltage, per-thruster gain, delay, sensors, impulse disturbance) `configs/dr_presets.yaml` |
| Goal attitude | Random unit quaternion | Uniform yaw (roll/pitch not controlled, being 4-axis) — `full_random_orientation` option |
| PPO | RSL-RL, 64×64 ELU, 24 steps, lr 5e-4 adaptive | Same (`rov_rl/agents/rsl_rl_ppo_cfg.py`) |
| Deployment | ROS/Jetson, direct-thruster | **numpy MLP** (`deploy/policy_numpy.py`) + `deploy/obs_builder.py` (numpy twin of the estimator, NED/FRD→z-up/FLU conversion) + `deploy/ardusub_bridge.py` (MANUAL_CONTROL 20 Hz, deadman, ARMED/MANUAL/JS_GAIN gates) |

Two values taken from the hardware logs and applied to the sim:
* **Mixer/gains** — `scripts/fit_mixer_from_logs.py`: matches the ArduSub theoretical factors exactly, effective pilot gain 0.49–0.50 (= `JS_GAIN_DEFAULT` 0.5), motor signs `[-1,-1,+1,+1,-1,+1,+1,-1]`.
* **Static pitch stiffness** — hardware pitch ≈ 4.0°/unit surge (181,097 rows). With the CAD-composite CB-COM arm of 7 mm, surge flips the vehicle over, so an effective arm of **0.15 m** is taken as nominal, with DR 0.5–1.5× [유도, `configs/rov_params.yaml` comment].

## 2. Installation / environment

* Training and evaluation: conda `env_isaaclab` (Isaac Sim 5.0 pip, Isaac Lab v2.2.0 editable, rsl-rl-lib 2.3.3, torch 2.7 cu128). Driver 570, `amd_iommu=off` — the install record is in `../umi_underwater_robust_control/ISAAC_AUV_SETUP_RUNBOOK.md`.
* Deployment and replay: conda `robust` (numpy, pymavlink; torch not needed).
* No install command — the scripts put the project root on `sys.path`. `export OMNI_KIT_ACCEPT_EULA=YES` is required.

```bash
conda activate env_isaaclab && cd ~/Desktop/RL_controller
python -m pytest -q tests                # 18 tests without the Isaac app (physics, mixer, sensors, estimator, deployment parity)
bash scripts/build_asset.sh              # URDF -> assets/rov/rov.usd (re-run after changing rov_params.yaml)
```

## 3. Training

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 600 --headless --dr small --run_name small
# DR presets: none | small | large | ours      logs: logs/rsl_rl/rov_posehold/<timestamp>_<run_name>/
```
On an RTX 5090, 2048 envs × 600 iter ≈ 3.5 min (≈0.35 s per iteration). It is faster than the paper's "10–20 min" because, apart from the history observation, the computation is small and there is no rendering.
Hydra overrides are supported: `env.rew_scale_actions=0.1 agent.policy.actor_hidden_dims=[128,128]` etc.
Practical (recommended) config: `--dr ours env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05 agent.algorithm.entropy_coef=0.005 --max_iterations 800` (7.2).

## 4. Evaluation (counterparts of the paper's figures/tables)

```bash
python scripts/plot_training.py logs/rsl_rl/rov_posehold/*_none logs/rsl_rl/rov_posehold/*_small ... -o logs/eval/training_curves.png   # Fig.3
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --tag rl_nominal                          # Table III (8-direction step)
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --shift "volume_l=-0.6,cob_x=0.02,drag=1.5,voltage=0.65" --tag rl_shift
python scripts/eval_sweep.py --controller pid --pid_derate 0.7 --headless --tag pid_nominal                       # PID baseline (derate 0.7 is the sim optimum)
python scripts/eval_disturbance.py --checkpoint <run>/model_599.pt --headless                                     # Fig.5–7 (impulse, RL vs PID)
python scripts/play.py --task ROV-PoseHold-v0 --num_envs 16 --dr small --checkpoint <run>/model_599.pt            # visualization (GUI; headless check with --headless --max_steps N)
python scripts/play_control.py --checkpoint <run>/model_599.pt --dr small                                         # steer the goal with the keyboard (counterpart of the paper's play_poshold, GUI)
```
Results: `logs/eval/<name>/{timeseries.csv, metrics.csv, *.png}`, summary table `python scripts/summarize_results.py` → `logs/eval/RESULTS.md`.
`--shift` is the paper's IV-B "shifted dynamics" protocol (shift amount = the bounds of the small DR): volume −0.6 L (−5.9 N), CB 2 cm forward, drag ×1.5, voltage 0.65.
(At −1.2 L or more, the heave authority (≈14 N) at gain 0.5 and voltage 0.6 is insufficient and both controllers hit the floor — not a test, but an authority limit.)
The PID gains are a port of the station's `rov_gui/control/pid.py` SIM_GAINS (`rov_rl/baselines/pid.py`).

## 5. Export and deployment

```bash
python scripts/export_policy.py --run logs/rsl_rl/rov_posehold/<run>        # -> <run>/exported/{policy_weights.npz, policy.onnx, obs_spec.json}
conda activate robust
python deploy/run_policy.py --export <run>/exported --replay ../umi_underwater_robust_control/data/2026.../mpc_*.csv --dry-run --log logs/eval/replay.csv
python deploy/run_policy.py --export <run>/exported --fix-udp 17000 --mav udpin:0.0.0.0:14552 --goal 0,0,-0.7 --goal-yaw 0 --axis-cap 0.5
```
* Observation-builder input = what the station already produces: `NavFix(p_ned, yaw_ned|R_ned_body, t_capture)` + ArduSub `ATTITUDE(roll,pitch,p,q,r)`.
  Just send each fix over UDP as one line of JSON (`{"t_capture":…, "p_ned":[…], "yaw_ned":…}`, monotonic seconds). The bridge parses ATTITUDE directly.
* Safety gates: ARMED + MANUAL mode + `JS_GAIN_DEFAULT` == training gain (0.5) + fresh ATTITUDE + tag fix within 0.5 s — otherwise NEUTRAL. deadman 500 ms. Start from `--axis-cap 0.5`.
* Frames: the policy was trained in z-up/FLU, and `obs_builder` converts NED/FRD with `C=diag(1,-1,-1)` (`tests/test_deploy_parity.py` verifies agreement with the torch pipeline to 1e-5).

## 6. Structure

```
configs/rov_params.yaml   vehicle properties, thrusters, mixer, sensors, pool (single source of truth)  configs/dr_presets.yaml  DR presets
configs/mixer_fitted.yaml fit results from hardware logs                        logs/mixer_fit_report.md fit report
rov_rl/params.py          parallel-axis composite, CB, thruster pose (composite-COM frame)  rov_rl/utils/math.py     quaternion utils (port of isaaclab.utils.math)
rov_rl/thrusters.py       T200 curve, delay, 1st-order lag                      rov_rl/mixer.py          ArduSub mixer, thrust signs, allocation B
rov_rl/hydrodynamics.py   the paper's drag, buoyancy (+Fossen)                  rov_rl/sensors.py        tag/IMU sim
rov_rl/estimator.py       port of state_assembler                               rov_rl/rewards.py        the paper's reward
rov_rl/env/rov_env*.py    Isaac Lab DirectRLEnv + cfg                           rov_rl/agents/           PPO cfg
rov_rl/baselines/pid.py   PID baseline                                          assets/rov/              URDF, meshes, USD (generated)
scripts/                  train/play/eval_*/plot_training/export/fit_mixer/build_asset
deploy/                   policy_numpy, obs_builder, ardusub_bridge, run_policy  tests/                   pytest
```

## 7. Results (2026-09-16, RTX 5090, 2048 envs) — full table in `logs/eval/RESULTS.md`

Training: 600 iter per preset ≈ 3.5 min [측정 `logs/train_*.log`], practical config 800 iter ≈ 5 min. Reward curves `logs/eval/training_curves_v2.png` (counterpart of the paper's Fig.3).
Evaluation: 8-direction step sweep (counterpart of the paper's Table III, MSE = whole window / steady state 2 s) and impulse-disturbance tests (counterpart of the paper's Fig.6–7, RL vs PID derate 0.7).
"Shifted dynamics" = the bounds of the small DR (volume −0.6 L, CB 2 cm forward, drag ×1.5, voltage 0.65). All numbers come from `logs/eval/<run>/metrics.csv` [측정].

### 7.1 Paper reproduction (the paper's reward as is, with observation-quaternion normalization applied = v2)

| run (DR) | ideal-environment pos MSE | ideal-environment ang MSE | shifted pos MSE | shifted ang MSE |
|---|---|---|---|---|
| none2 (no DR) | 0.231 / 0.238 | 0.026 / 0.003 | 0.255 / 0.274 | 0.049 / 0.019 |
| small2 | 0.071 / 0.041 | 0.028 / 0.005 | 0.086 / 0.061 | 0.049 / 0.018 |
| large2 | 0.065 / 0.037 | 0.028 / 0.004 | 0.076 / 0.049 | 0.048 / 0.020 |
| ours2 | 0.077 / 0.052 | 0.026 / 0.002 | 0.242 / 0.267 | 0.052 / 0.019 |
| Paper CUREE (12 directions) | 0.058 (no DR) / 0.056 (small) | 0.013 / 0.011 | 0.108 (no DR) / 0.081 (small) | 0.401 / 0.195 |

* Same trend as the paper: the more DR is applied, the better the position MSE under shifted dynamics (none 0.255 → small 0.086 → large 0.076; paper 0.108 → 0.081).
* With the paper's reward (position term σ = 1 m), steady-state precision stops at the 20–25 cm RMS level — the paper's own 0.056 MSE is also 24 cm RMS. The large position MSE of none2/ours2 is the result of the policy settling ~0.5 m away from the goal on this flat reward (not a physics or estimator problem; checked with `scripts/diag_policy_hover.py`).
* The angle MSE is larger than the paper's because, with 4 axes, roll/pitch are not controlled (the steady-state yaw error itself is 2–4°).

### 7.2 Practical config (recommended for hardware pool tests)

`--dr ours` + `env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05` (+ `agent.algorithm.entropy_coef=0.005`).

| run | ideal-environment pos MSE | ideal-environment ang MSE | shifted pos MSE | shifted ang MSE | disturbance pos RMS ideal/shifted [m] (PID) |
|---|---|---|---|---|---|
| **prac** (v2 observation, entropy 0) — **recommended for deployment** | 0.023 / 0.003 | **0.021 / 0.001** | 0.034 / 0.011 | **0.037 / 0.017** | 0.083 / 0.114 (PID 0.137 / 0.083) |
| prac2 (v2 observation + entropy 0.005) — **do not use: yaw chattering 79 %** | 0.020 / 0.003 | 0.032 / 0.011 | 0.020 / 0.002 | 0.047 / 0.025 | 0.066 / 0.057 |
| prac3 (prac2 + action-rate 0.3) — **do not use: chattering 66 %** | 0.021 / 0.004 | 0.038 / 0.019 | 0.025 / 0.006 | 0.055 / 0.034 | 0.082 / 0.086 |
| tuned3 (old observation + entropy) | 0.023 / 0.004 | 0.029 / 0.014 | 0.027 / 0.006 | 0.044 / 0.029 | 0.090 / 0.092 |
| PID derate 0.7 (baseline) | 0.029 / 0.003 | 0.023 / 0.002 | 0.042 / 0.015 | 0.042 / 0.018 | — |

* prac: steady-state position RMS ≈ 5.7 cm, yaw ≈ 2°, command saturation 0 %, mean |a| 0.08; disturbance rejection in the ideal environment is better than PID (same as the conclusion of the paper's Fig.6–7), while under shifted dynamics PID is slightly better.
* prac2 has the best metrics, but it is a bang-bang policy whose yaw command saturates at ±1 on 76 % of the steady-state steps, with the sign flipping every step (`logs/eval/RESULTS.md` saturation column). A failure mode you miss if you look only at the metrics — **before adopting a policy, always check the saturation-fraction and sign-flip columns**. In the v2 runs with entropy on, chattering remained even with the action-rate penalty raised to 0.3 (prac3) — with this observation structure (previous-action feedback) it is safer not to use the entropy bonus.
* Deployment checkpoint: `logs/rsl_rl/rov_posehold/*_prac/exported/`.

### 7.3 Two things found along the way (problems that do not show up in the training curves)
1. **Quaternion double representation**: sampling the goal yaw from [0, 2π) makes half of the quaternions have w<0, while the estimated quaternion always has w≥0, so the policy could not learn q and −q as the same rotation, and a +0.2–0.3 rad bias appeared near goal yaw 0 (confirmed by feeding synthetic observations through `deploy/policy_numpy.py`). Normalizing to w≥0 with `obs_quat_unique=True` → angle MSE 0.10 → 0.026.
2. **Exploration collapse vs thruster deadband**: with entropy 0 the PPO noise std shrinks to 0.02, so sampling happens only inside the T200 deadband (at gain 0.5, |axis| < 0.15 gives thrust 0) → there is no gradient for learning fine corrections. Keeping std ≈ 0.19 with `entropy_coef=0.005` (tuned3) → yaw RMS 0.24 → 0.06 rad. However, turning entropy on can produce bang-bang chattering (prac2), so the action-rate penalty has to be raised along with it.
3. **Limits of open-loop replay**: in `run_policy.py --replay` the recorded log does not respond to the policy's commands, so the feedback through the previous-action input locks up and the yaw command can saturate (there is no gyro column either). It is only for checking the pipeline — do not use it to judge performance; performance is judged with closed-loop sim evaluation.

## 8. Trajectory-tracking extension (2026-09-21)

A policy trained only on position hold lags behind a moving goal in proportion to its speed (no feedforward). So training, observation, evaluation and deployment were extended for trajectory tracking.
The controller structure (action, mixer, estimator, bridge) is unchanged.

| Item | Details |
|---|---|
| Goal motion | `rov_rl/goals.py`: per episode, stationary (30 %) / straight or curved (speed 0.02–0.25 m/s, turn rate ±0.6 rad/s, vertical ±0.05 m/s), heading-follow 50 %, reflects off the pool walls |
| Observation | 17 + **goal velocity (3, in the body frame) + goal yaw rate (1)** + previous action 4 = 25 × 4 frames = 100 (`obs_goal_velocity`; old checkpoints automatically get 84) |
| Episode | 6 s → **20 s** |
| Drag DR (`traj` preset) | drag ×0.7–3.0 + additional linear drag (30, 45, 10) N·s/m × 0–1.5 [유도: station speed law `speed = 0.692(\|axis\|−0.096)` → the drag on the real vehicle is 2–4× the model's] |
| Evaluation | `scripts/eval_trajectory.py`: square 1 m @0.15 m/s, circle r 0.5 @0.15 m/s, lawnmower — RL vs PID, cross-track p95, position RMS, yaw RMS, saturation fraction, `logs/eval/traj_<tag>/path.png` |
| Deployment | `deploy/run_policy.py --traj square:1.0,0.15 --laps 2` (defined in a frame referenced to the start pose, NED conversion automatic) — `deploy/trajectory.py` |

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 1000 --headless --dr traj --run_name traj \
    env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05
python scripts/eval_trajectory.py --checkpoint logs/rsl_rl/rov_posehold/*_traj/model_999.pt --traj square:1.0,0.15 --laps 2 --headless
```
To run it independently of the session: `setsid nohup bash scripts/_pipeline_traj.sh > logs/pipeline_traj_outer.log 2>&1 < /dev/null &`.

### 8.0 4th iteration (traj4, 2026-09-21 evening): addressing the depth problem
* **Per-axis separable reward** (`env.rew_separable=True`): horizontal position/velocity and depth position/velocity are rewarded separately (σ_xy 0.3 m, σ_z 0.15 m; velocity σ 0.15 / 0.10 m/s). With a single 3-D norm, the z gradient is diluted while horizontal error remains (`tests/test_hydro.py::test_separable_rewards_keep_z_gradient`).
* **Nominal linear damping** (`configs/rov_params.yaml hydro.linear_damping_nominal` = 20/30/15 N·s/m [유도 station speed law]): the paper's model has only 2nd-order drag, so low-speed damping is close to 0, which produces a heave limit cycle. It is now also included in the nominal model and in evaluation, and its DR is 0.5–1.5× (`traj4` preset).
* **Asymmetric actor-critic** (`env.critic_privileged=True`): the critic sees the policy observation (100) + the true state and DR parameters (31). The actor (deployed) is unchanged.
* **3 seeds** (`--seed 0/1/2`, 1200 iter) and selection by the worst metric with `scripts/select_best.py`.

### 8.0b 5th iteration (traj5): added mass (completing Fossen)
Until now, of the Fossen equations only the rigid-body, restoring-force and damping terms were present, and **the added mass M_A·ν̇ and C_A(ν)ν were missing**. For the BlueROV2 the added mass is 14.6 kg in heave and 12.7 kg in sway,
on the same scale as the vehicle mass (13.7), so the effective inertia differs by 2×. Implementation (`rov_rl/hydrodynamics.py AddedMass`): every substep the body acceleration is
estimated with a low-pass filter (α 0.3), and −M_A·ν̇ is applied as an external force and C_A(ν)ν in Fossen's diagonal form (the uuv_simulator / MarineGym approach; it is an explicit delay, but stable since α·M_A/m < 1,
`tests/test_hydro.py::test_added_mass_effective_inertia_and_coriolis`). Coefficients: MarineGym BlueROVHeavy (Wu 2018) [스펙], DR ×0.5–1.5 (`traj5` preset).
Since it is part of the nominal physics, it is also in the `--dr none` evaluation (results before this are values from a world without added mass). The result of re-evaluating traj4_s2 in the new physics is a sim-to-sim robustness indicator.

### 8.1 Results (square 1 m / circle r 0.5 m @0.15 m/s, heading-follow corner slew 60°/s, 2 laps; full table in `logs/eval/RESULTS.md`)

| Policy | Training config | square xy RMS / z RMS [m] | square along RMS | circle xy RMS / along RMS | circle yaw RMS [rad] | Saturation |
|---|---|---|---|---|---|---|
| prac (trained for position hold, no feedforward) | ours DR, stationary goal, 6 s | 0.139 / 0.034 | 0.133 | 0.152 / 0.143 | 0.132 | 4 % |
| traj | traj DR (heavy drag), moving goal, 20 s | 0.114 / 0.080 | 0.092 | 0.105 / 0.072 | 0.197 | 20 % |
| traj2 | traj2 DR, moving goal, velocity reward | **0.085 / 0.098** | **0.072** | **0.061 / 0.021** | **0.123** | 1 % |
| traj3 | traj2 + narrowed drag DR, constant-depth goal, rate 0.1 | 0.154 / 0.261 | 0.121 | 0.101 / 0.054 | 0.095 | 19 % |
| **traj4_s2** (8.0 config, seed 2) — **recommended for deployment** | separable reward + nominal linear damping + privileged critic | **0.059 / 0.032** | 0.052 | **0.058 / 0.035**, lag 0.040 | 0.069 | 3 % |
| traj4_s0 / traj4_s1 (seed spread) | 〃 | 0.063 / 0.032, 0.065 / 0.036 | 0.050, 0.049 | 0.066 / 0.028, 0.034 / 0.021 | 0.129, 0.063 | 5 %, 9 % |
| **traj5_s2** (8.0b, with added mass) — **final deployment recommendation** | traj4 + added mass M_A·ν̇ + C_A(ν)ν, DR ×0.5–1.5 | **0.053 / 0.020** | 0.047 | **0.057 / 0.021**, lag 0.044 | 0.043 | 2 % |
| traj5_s0 / traj5_s1 | 〃 | 0.054 / 0.023, 0.053 / 0.027 | 0.040, 0.042 | 0.062 / 0.015, 0.063 / 0.020 | 0.089, 0.059 | 17 % (corner yaw), 4 % |
| traj4_s2 re-evaluated in the added-mass world (sim-to-sim) | trained without added mass | 0.065 / 0.054 | 0.056 | 0.058 / 0.051 | 0.076 | 2 % |
| PID derate 0.7 | baseline (no feedforward) | 0.106 / 0.022 | 0.085 | 0.076 / 0.072 | 0.232 | 0 % |

* Effect of feedforward: along-track lag goes from prac 0.143 → traj2 **0.021 m** (circle). In horizontal tracking traj2 is better than PID (square xy 0.085 vs 0.106, circle 0.061 vs 0.076).
* Remaining problem: **depth** hold of the policies trained with moving goals (traj2 z RMS 0.098, ±0.1 m heave oscillation even when stationary, −0.3 m at corners; prac 0.034, PID 0.022). traj3, which narrowed the drag DR and fixed the depth goal, instead got worse (z RMS 0.26), so the cause is not the drag-DR width alone — the 20 s episodes + moving-goal training appear to loosen the heave loop; unresolved.
* Under shifted dynamics (drag ×1.5 etc.) RL is better than PID (traj square 0.094 vs 0.145 m) — the same direction as the station records showing that the drag on the real vehicle is larger than the model's.

**4th iteration (traj4) results — depth problem solved.** All 3 seeds trained with separable reward + nominal linear damping + privileged critic came down to a depth RMS in the 0.03 m range
(traj2 0.098), and horizontal tracking and disturbance rejection are also better than PID on every item. Disturbance-test pos RMS (3-D): 0.055–0.070 m (PID 0.08). The spread between seeds is small
(`scripts/select_best.py` worst-metric scores 0.91 / 1.08 / 1.79; prac 1.79, traj2 1.96).

**5th iteration (traj5, with added mass) results.** All 3 seeds are better than traj4 and the spread is small (worst-metric scores 0.85 / 0.91; s0 is at 3.35 because of corner-yaw saturation, but
saturation on straight segments is 1 % and sign flips 0.11/step — not chattering). Running traj4_s2, trained without added mass, in the added-mass world worsens depth RMS from 0.032 → 0.054 m
(sim-to-sim), so added mass is an important term for the transfer to hardware as well. Disturbance-test pos RMS (3-D) traj5: 0.050–0.060 m (PID 0.08).

**Final recommendation**: RL on all axes — **traj5_s2** (`logs/rsl_rl/rov_posehold/*_traj5_s2/exported/`; for position hold, trajectory tracking and disturbances alike). First hardware test: `--axis-cap 0.5`,
in the order stationary → square 1 m @0.10 m/s → 0.15 m/s. The observation spec (`obs_spec.json`) records whether feedforward is included, so `run_policy.py` matches it automatically.

## 9. Assumptions and limitations (checklist before hardware)
1. The tags are always visible (user assumption). On hardware, if the fix is lost for 0.5 s or more it goes NEUTRAL, so leaving the field of view = stop.
2. The effective restoring arm 0.15 m, net buoyancy +1 N, thruster delay 2 steps and voltage 0.72 are [유도]/[예측] — the DR ranges were set to cover them. First pool test with a `--dr ours` policy and `axis-cap 0.5`.
3. roll/pitch are not commanded (4 axes). Extending to 6 axes (MANUAL_CONTROL s,t) only requires opening the roll/pitch inputs in `mixer.py`.
4. The fix input of `run_policy.py` (UDP JSON) requires adding a small bridge on the station side that sends NavFix (or calling this script inside the station process).
