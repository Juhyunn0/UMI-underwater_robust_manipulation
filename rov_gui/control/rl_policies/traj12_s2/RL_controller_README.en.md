# RL_controller — "Learning to Swim" reproduction: our ROV (BlueROV2 Heavy + C3 + Newton gripper), Vision + IMU only

> **English translation.** Source of truth: [`RL_controller_README.md`](RL_controller_README.md) (Korean). Snapshot of 2026-10-12, source blob `6fb3ac2130`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

A standalone package that re-implements the pipeline of the paper *Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven AUVs* (Cai, Chang, Girdhar, ICRA 2025,
[arXiv:2410.00120](https://arxiv.org/abs/2410.00120), code `warplab/isaac-auv-env`) — **Isaac Lab parallel simulation + RSL-RL PPO +
domain randomization → zero-shot transfer to hardware** — to fit our vehicle and sensor configuration.
It imports no code from outside this folder (the model, parameters, estimator and command path are all copied/rewritten here).

Evidence-tag rules: `[측정]` ("measurement") comes with the path of a measured artifact, `[스펙]` ("spec") a datasheet/code constant, `[유도]` ("derived") a calculation, `[예측]` ("prediction") an unverified assumption.

## 1. What is the same as the paper and what is different

| Item | Paper (CUREE, WarpAUV) | This implementation (our ROV) |
|---|---|---|
| Simulator | Isaac Lab DirectRLEnv, 2048 envs, 120 Hz physics | Same (Isaac Sim 5.0 + Isaac Lab 2.2.0, `~/IsaacLab`, conda `env_isaaclab`) |
| Vehicle | 22.7 kg, 6 thrusters | BlueROV2 **Heavy** 8×T200 + C3 camera (upper mount, tilted 43° down) + Newton gripper, **13.72 kg** (`configs/rov_params.yaml`, parallel-axis composite) |
| Dynamics | MuJoCo-style inertia-box drag + buoyancy (COB offset) + `C_t·Ω\|Ω\|` | Same drag model + nominal linear damping + **added mass M_A·ν̇ + C_A(ν)ν** (Fossen; 2026-09-21, `rov_rl/hydrodynamics.py`) + **MarineGym T200 curve** (voltage scale · deadband · 1st-order lag · command delay) |
| Action | 6 thruster PWMs directly | **Two modes** (`env.action_mode`, §10, 2026-09-27): `axes` = **MANUAL_CONTROL 4 axes** `[surge, sway(+우), heave(+상), yaw(+시계)]` ("[surge, sway(+right), heave(+up), yaw(+clockwise)]") → **ArduSub vectored-6DOF mixer** (`rov_rl/mixer.py`) → 8 throttles (default; every checkpoint before 09-27); `pwm` = **8 thruster throttles directly** (mixer bypassed as in the paper, roll/pitch can be actuated). Mixer, gains and motor signs verified on 64,512 rows of hardware logs [측정 `logs/mixer_fit_report.md`] |
| Sensors | DVL + IMU + AprilTag EKF (v_body observed directly) | **No DVL**: simulates tag PnP (≈15 Hz, ≈130 ms latency, noise) + ArduSub ATTITUDE/gyro, and builds v_body with **the same estimator** as station's `state_assembler` (finite difference + LPF α=0.6, gyro/velocity bridging) (`rov_rl/sensors.py`, `rov_rl/estimator.py`) |
| Observation | 17-dim `[q_des, x_off_b, q, v_b, ω_b]` | 17 + goal velocity/yaw rate 4 (§8) + previous action 4 = 25, over a **4-frame history** → 100-dim (latency compensation). Quaternions are normalized to w≥0 (7.3). The 2026-09-16 runs are 84-dim |
| Reward | `0.2e^{-\|x\|²} + 0.5e^{-angle} + 0.2e^{-\|a\|²}` | Same (`rov_rl/rewards.py`, yaw-only option) |
| Control rate | 60 Hz training / 20 Hz deployment | **20 Hz** training = deployment (physics 1/120 s, decimation 6), episode 20 s (6 s for the 2026-09-16 runs) |
| DR | COB offset sphere, volume | `none/small/large` (counterparts of the paper's) + `ours` (mass · inertia · drag · voltage · per-thruster gain · delay · sensors · impulse disturbances) `configs/dr_presets.yaml` |
| Goal attitude | Arbitrary unit quaternion | Uniform yaw (roll/pitch level). The `goal_max_roll_pitch`/`full_random_orientation` cfg are **dead knobs that no code reads** (checked 2026-10-03). Roll/pitch references: §12 `att_*` (preset `traj10att`) |
| PPO | RSL-RL, 64×64 ELU, 24 steps, lr 5e-4 adaptive | Same (`rov_rl/agents/rsl_rl_ppo_cfg.py`) |
| Deployment | ROS/Jetson, thrusters directly | **numpy MLP** (`deploy/policy_numpy.py`) + `deploy/obs_builder.py` (numpy twin of the estimator, NED/FRD→z-up/FLU conversion) + `deploy/ardusub_bridge.py` (MANUAL_CONTROL 20 Hz, deadman, ARMED/MANUAL/JS_GAIN gates) |

Two values obtained from hardware logs and applied to the sim:
* **Mixer/gain** — `scripts/fit_mixer_from_logs.py`: matches the ArduSub theoretical factors exactly, effective pilot gain 0.49–0.50 (= `JS_GAIN_DEFAULT` 0.5), motor signs `[-1,-1,+1,+1,-1,+1,+1,-1]`.
* **Static pitch stiffness** — hardware pitch ≈ 4.0°/unit surge (181,097 rows). With the CAD-composite CB-COM arm of 7 mm it flips over under surge, so an effective arm of **0.15 m** is used as nominal, with DR 0.5–1.5× [유도, `configs/rov_params.yaml` comment].

## 2. Install / environment

* Training · evaluation: conda `env_isaaclab` (Isaac Sim 5.0 pip, Isaac Lab v2.2.0 editable, rsl-rl-lib 2.3.3, torch 2.7 cu128). Driver 570, `amd_iommu=off` — install record in `../umi_underwater_robust_control/ISAAC_AUV_SETUP_RUNBOOK.md`.
* Deployment · replay: conda `robust` (numpy, pymavlink; torch not needed).
* No install command — the scripts put the project root on `sys.path`. `export OMNI_KIT_ACCEPT_EULA=YES` is required.

```bash
conda activate env_isaaclab && cd ~/Desktop/RL_controller
python -m pytest -q tests                # 18 tests without the Isaac app (physics·mixer·sensors·estimator·deployment parity)
bash scripts/build_asset.sh              # URDF -> assets/rov/rov.usd (re-run after changing rov_params.yaml)
```

## 3. Training

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 600 --headless --dr small --run_name small
# DR presets: none | small | large | ours      logs: logs/rsl_rl/rov_posehold/<timestamp>_<run_name>/
```
On an RTX 5090, 2048 envs × 600 iter ≈ 3.5 min (≈0.35 s per iteration). It is faster than the paper's "10–20 min" because, besides the history observation, the compute is small and there is no rendering.
Hydra overrides are possible: `env.rew_scale_actions=0.1 agent.policy.actor_hidden_dims=[128,128]` etc.
Practical (recommended) settings: `--dr ours env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05 agent.algorithm.entropy_coef=0.005 --max_iterations 800` (7.2).

## 4. Evaluation (counterparts of the paper's figures/tables)

```bash
python scripts/plot_training.py logs/rsl_rl/rov_posehold/*_none logs/rsl_rl/rov_posehold/*_small ... -o logs/eval/training_curves.png   # Fig.3
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --tag rl_nominal                          # Table III (8-direction step)
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --shift "volume_l=-0.6,cob_x=0.02,drag=1.5,voltage=0.65" --tag rl_shift
python scripts/eval_sweep.py --controller pid --pid_derate 0.7 --headless --tag pid_nominal                       # PID baseline (derate 0.7 is the sim optimum)
python scripts/eval_disturbance.py --checkpoint <run>/model_599.pt --headless                                     # Fig.5–7 (impulse, RL vs PID)
python scripts/play.py --task ROV-PoseHold-v0 --num_envs 16 --dr small --checkpoint <run>/model_599.pt            # visualization (GUI; check without a display via --headless --max_steps N)
python scripts/play_control.py --checkpoint <run>/model_599.pt --dr small                                         # steer the goal with the keyboard (counterpart of the paper's play_poshold, GUI)
```
Results: `logs/eval/<name>/{timeseries.csv, metrics.csv, *.png}`, summary table `python scripts/summarize_results.py` → `logs/eval/RESULTS.md`.
`--shift` is the "shifted dynamics" protocol of the paper's IV-B (shift amount = the bounds of the small DR range): volume −0.6 L (−5.9 N), CB 2 cm forward, drag ×1.5, voltage 0.65.
(At −1.2 L or more, the heave authority (≈14 N) at gain 0.5 · voltage 0.6 is insufficient and both controllers hit the floor — an authority limit, not a test.)
The PID gains are a port of station's `rov_gui/control/pid.py` SIM_GAINS (`rov_rl/baselines/pid.py`).

## 5. Export and deployment

```bash
python scripts/export_policy.py --run logs/rsl_rl/rov_posehold/<run>        # -> <run>/exported/{policy_weights.npz, policy.onnx, obs_spec.json}
conda activate robust
python deploy/run_policy.py --export <run>/exported --replay ../umi_underwater_robust_control/data/2026.../mpc_*.csv --dry-run --log logs/eval/replay.csv
python deploy/run_policy.py --export <run>/exported --fix-udp 17000 --mav udpin:0.0.0.0:14552 --goal 0,0,-0.7 --goal-yaw 0 --axis-cap 0.5
```
* Observation-builder input = what station already produces: `NavFix(p_ned, yaw_ned|R_ned_body, t_capture)` + ArduSub `ATTITUDE(roll,pitch,p,q,r)`.
  Just send the fixes over UDP, one JSON line each (`{"t_capture":…, "p_ned":[…], "yaw_ned":…}`, monotonic seconds). The bridge parses ATTITUDE itself.
* Safety gates: ARMED + MANUAL mode + `JS_GAIN_DEFAULT` == training gain (0.5) + fresh ATTITUDE + tag fix within 0.5 s — otherwise NEUTRAL. deadman 500 ms. Start from `--axis-cap 0.5`.
* Frames: the policy was trained in z-up/FLU, and `obs_builder` converts NED/FRD with `C=diag(1,-1,-1)` (`tests/test_deploy_parity.py` verifies agreement with the torch pipeline to 1e-5).

### 5.1 Output interface = station MPC's wrench (2026-09-22)

The RL controller's official output is the same 6-DOF body wrench as the MPC's, **u = [X, Y, Z, K, M, N]** (NED/FRD body frame, N·N·m). The network still
outputs MANUAL_CONTROL 4 axes (−1..1), and a fixed linear head is attached on top (`deploy/wrench.py`; the same formulas as `axes_to_wrench` /
`wrench_to_axes` in station's `allocation.py` [code]): X = a0·60, Y = a1·60, Z = −a2·60 (NED Z positive down), K = M = 0, N = a3·20
[스펙 hw_mpc.yaml axis_gain; `configs/rov_params.yaml: interface`]. The MANUAL_CONTROL path does not actuate roll/pitch, so the MPC also discards
K and M — the effective actuated subspace of the two controllers is the same.
* `export_policy.py` → `policy_wrench.onnx` (input obs → output wrench (1,6)) and `obs_spec.json: output` (order · units · actuated · axis_gain).
  Parity: torch head vs numpy 1.8e-5, onnxruntime 1.0e-5 [측정].
* Added the column names `uX..uN`, the same as station's `mpc_*.csv`, to the `deploy/run_policy.py` log and the `eval_trajectory.py` timeseries. Station's `HwRl`
  returns this wrench to the worker from the start and records it in the same columns (0921_182926: heave −0.162 → uZ 9.73 N [측정]).
* `ArduSubBridge.send_wrench(u)` and `rov_rl/mixer.py: axes_to_wrench / wrench_to_axes` (torch) make it possible to drive by wrench in either sim or hardware.
* The paper output the 6 thruster commands directly (learning even the mixing, actuating roll/pitch). We run stock ArduSub MANUAL, which has no per-motor commands in flight, so
  we kept the 4 axes (user decision 2026-09-22) and only unified the interface as a wrench. **Since 2026-09-27 there is also a direct-thruster mode (`pwm`) (§10)** —
  sim · training · export · `deploy/` (RC passthrough transport) are in place, and the hardware-side transport (changing SERVOn_FUNCTION) remains a user decision.

## 6. Structure

```
configs/rov_params.yaml   vehicle properties·thrusters·mixer·sensors·pool (single source of truth)  configs/dr_presets.yaml  DR presets
configs/mixer_fitted.yaml hardware-log fit results                              logs/mixer_fit_report.md fit report
rov_rl/params.py          parallel-axis composite, CB, thruster pose (composite-COM frame)  rov_rl/utils/math.py  quaternion utils (port of isaaclab.utils.math)
rov_rl/thrusters.py       T200 curve·delay·1st-order lag                        rov_rl/mixer.py          ArduSub mixer·thrust signs·allocation B
rov_rl/hydrodynamics.py   the paper's drag·buoyancy (+Fossen)                   rov_rl/sensors.py        tag/IMU sim
rov_rl/estimator.py       port of state_assembler                               rov_rl/rewards.py        the paper's reward
rov_rl/env/rov_env*.py    Isaac Lab DirectRLEnv + cfg                           rov_rl/agents/           PPO cfg
rov_rl/baselines/pid.py   PID baseline                                          assets/rov/              URDF, meshes, USD (generated)
scripts/                  train/play/eval_*/plot_training/export/fit_mixer/build_asset
deploy/                   policy_numpy, obs_builder, ardusub_bridge, run_policy  tests/                   pytest
```

### 6.1 Payload visual meshes — C3 camera (upper mount) + Newton gripper (2026-09-21)

`assets/rov/rov.usd` includes, besides the hull (`rov_body_*.obj`), the two payloads the hardware carries, as visual meshes. The generator is
`scripts/bake_payload_meshes.py` (`robust` env: trimesh 4.12); afterwards regenerate the USD with `bash scripts/build_asset.sh`.
- Source: `umi_underwater_robust_control/assets/CAD files/onshape_export_20260909/` (onshape-to-robot 1.8.2, Onshape document "BlueROV2" /
  assembly tab "mujoco", 2026-09-09 11:42 −0700 snapshot). In `robot.xml`, the `<geom pos quat mesh>` entries are the part poses in the assembly frame
  (MuJoCo quat = w,x,y,z; every `<body>` is identity), and `assets/*.stl` are the part meshes (m). The masses are placeholders, so only the poses/meshes are used [스펙 export README].
- Registration (assembly → base_link "bl", FLU, origin = heavy COM): as fixed in `bluerov2_mujoco_marinegym/tools/process_c3_mesh.py`,
  `R0 = [[0,0,1],[1,0,0],[0,1,0]]`, `C_ASM = [-0.16079, 0.22043, -0.11281]`, `p_bl = R0 @ (p_asm − C_ASM)` [유도 — constants copied, no import].
- Verification (against `payload_frames_20260909.json`, asserted by the script): max bbox deviation over the 13 parts 0.05 mm (tolerance 2 mm);
  `c3_centroid_bl` / `gripper_housing_centroid_bl` / `gripper_jaw_centroid_bl` deviation 0.01 mm (the json's centroid = vertex mean);
  C3 optical axis = bl `[0.766, 0, −0.643]` → tilted 40.00° down (json `tilt_down_deg` 40); jaw x range `[0.3289, 0.4039]` m [유도 bake log].
- Meshes: `c3_payload.obj` = c3_br_june2025 + part_1 + part_3 (39,703 faces, URDF colour 0.23 0.38 0.71),
  `newton_gripper.obj` = housing · flange cap · piston · base · pusher · 2 jaws + 2 mounts + part_1__2 (65,836 faces, 0.15 0.15 0.15).
  Screws, O-rings, wlp cable penetrators and the `brov2_heavy_asm_*` hull parts are excluded. 150k faces or fewer, so no decimation.
- Placement: the bl frame = the heavy-COM frame of the hull mesh (`rov_params.yaml` uses `c3_centroid_bl` as-is as the heavy-frame pos), so
  the URDF `<visual>` origin is the same as the hull's: `−COM_total = [-0.02983, -0.00388, -0.01276]`. Consistency check: hull-mesh union bbox vs
  `reanalysis_20260909.json hull_bbox_bl` center deviation ≤ 0.6 mm, size ratio 1.023 (the MarineGym skin is 2.3 % larger than the CAD, `process_c3_mesh.py` comment), corners at most 6.9 mm [유도].
- The inertia is already composed in through the `payloads:` entries of `rov_params.yaml`, so the URDF `<inertial>` (13.724 kg) and the collision boxes are unchanged — visual only.
  Visual check `assets/rov/render_check.png` (front/right/top orthographic views: camera at the upper front tilted 40° down, gripper at the lower front +x, jaw tip x ≈ 0.40 m).
  Isaac Lab UrdfConverter rewrites `assets/rov/.asset_hash` (= MD5(conversion settings + URDF bytes)) on every build; no code reads it.

## 7. Results (2026-09-16, RTX 5090, 2048 envs) — full tables in `logs/eval/RESULTS.md`

Training: 600 iter per preset ≈ 3.5 min [측정 `logs/train_*.log`], practical settings 800 iter ≈ 5 min. Reward curves `logs/eval/training_curves_v2.png` (counterpart of the paper's Fig.3).
Evaluation: 8-direction step sweep (counterpart of the paper's Table III, MSE = full window / 2 s steady state) and impulse-disturbance tests (counterpart of the paper's Fig.6–7, RL vs PID derate 0.7).
"Shifted dynamics" = the small DR bounds (volume −0.6 L, CB 2 cm forward, drag ×1.5, voltage 0.65). All numbers are from `logs/eval/<run>/metrics.csv` [측정].

### 7.1 Paper reproduction (the paper's reward as is, observation-quaternion normalization applied = v2)

| run (DR) | ideal environment pos MSE | ideal environment ang MSE | shifted pos MSE | shifted ang MSE |
|---|---|---|---|---|
| none2 (no DR) | 0.231 / 0.238 | 0.026 / 0.003 | 0.255 / 0.274 | 0.049 / 0.019 |
| small2 | 0.071 / 0.041 | 0.028 / 0.005 | 0.086 / 0.061 | 0.049 / 0.018 |
| large2 | 0.065 / 0.037 | 0.028 / 0.004 | 0.076 / 0.049 | 0.048 / 0.020 |
| ours2 | 0.077 / 0.052 | 0.026 / 0.002 | 0.242 / 0.267 | 0.052 / 0.019 |
| Paper CUREE (12 directions) | 0.058 (no DR) / 0.056 (small) | 0.013 / 0.011 | 0.108 (no DR) / 0.081 (small) | 0.401 / 0.195 |

* Same trend as the paper: the more DR is applied, the better the position MSE under shifted dynamics (none 0.255 → small 0.086 → large 0.076; paper 0.108 → 0.081).
* With the paper's reward (position term σ = 1 m), steady-state precision stops at the 20–25 cm RMS level — the paper's own 0.056 MSE is also 24 cm RMS. The large position MSE of none2/ours2 is the result of the policy settling ~0.5 m away from the goal on this flat reward (not a physics/estimator problem, checked with `scripts/diag_policy_hover.py`).
* The angle MSE is larger than the paper's because, with 4 axes, roll/pitch are not controlled (the steady-state yaw error itself is 2–4°).

### 7.2 Practical settings (recommended for hardware pool tests)

`--dr ours` + `env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05` (+ `agent.algorithm.entropy_coef=0.005`).

| run | ideal environment pos MSE | ideal environment ang MSE | shifted pos MSE | shifted ang MSE | disturbance pos RMS ideal/shifted [m] (PID) |
|---|---|---|---|---|---|
| **prac** (v2 observation, entropy 0) — **recommended for deployment** | 0.023 / 0.003 | **0.021 / 0.001** | 0.034 / 0.011 | **0.037 / 0.017** | 0.083 / 0.114 (PID 0.137 / 0.083) |
| prac2 (v2 observation + entropy 0.005) — **do not use: yaw chattering 79 %** | 0.020 / 0.003 | 0.032 / 0.011 | 0.020 / 0.002 | 0.047 / 0.025 | 0.066 / 0.057 |
| prac3 (prac2 + action-rate 0.3) — **do not use: chattering 66 %** | 0.021 / 0.004 | 0.038 / 0.019 | 0.025 / 0.006 | 0.055 / 0.034 | 0.082 / 0.086 |
| tuned3 (old observation + entropy) | 0.023 / 0.004 | 0.029 / 0.014 | 0.027 / 0.006 | 0.044 / 0.029 | 0.090 / 0.092 |
| PID derate 0.7 (baseline) | 0.029 / 0.003 | 0.023 / 0.002 | 0.042 / 0.015 | 0.042 / 0.018 | — |

* prac: steady-state position RMS ≈ 5.7 cm, yaw ≈ 2°, command saturation 0 %, mean |a| 0.08; disturbance rejection in the ideal environment is better than PID (the same as the conclusion of the paper's Fig.6–7), while under shifted dynamics PID is slightly better.
* prac2 has the best metrics, but it is a bang-bang policy whose yaw command saturates at ±1 on 76 % of steady-state steps and flips sign every step (`logs/eval/RESULTS.md` saturation column). A failure mode you miss if you look only at the metrics — **before adopting a policy, always check the saturation-fraction and sign-flip columns**. In the v2 runs with entropy turned on, chattering remained even with the action-rate penalty raised to 0.3 (prac3) — with this observation structure (previous-action feedback), it is safer not to use the entropy bonus.
* Deployment checkpoint: `logs/rsl_rl/rov_posehold/*_prac/exported/`.

### 7.3 Two things found along the way (problems that do not show in the training curves)
1. **Quaternion double cover**: sampling the goal yaw in [0, 2π) makes half of the quaternions w<0, while the estimated quaternion is always w≥0, so the policy failed to learn q and −q as the same rotation, and a +0.2–0.3 rad bias appeared near goal yaw 0 (confirmed by feeding synthetic observations through `deploy/policy_numpy.py`). Normalizing to w≥0 with `obs_quat_unique=True` → angle MSE 0.10 → 0.026.
2. **Exploration collapse vs thruster deadband**: with entropy 0 the PPO noise std shrinks to 0.02, so sampling happens only inside the T200 deadband (at gain 0.5, |axis| < 0.15 gives thrust 0) → there is no gradient to learn fine corrections. `entropy_coef=0.005` keeps std ≈ 0.19 (tuned3) → yaw RMS 0.24 → 0.06 rad. However, turning entropy on can produce bang-bang chattering (prac2), so the action-rate penalty has to be raised along with it.
3. **Limits of open-loop replay**: with `run_policy.py --replay` the recorded log does not respond to the policy's commands, so the feedback through the previous-action input locks up and the yaw command can saturate (there is no gyro column either). It is only for checking the pipeline — do not use it to judge performance; performance is judged with closed-loop sim evaluation.

## 8. Trajectory-tracking extension (2026-09-21)

A policy that learned only position hold lags a moving goal in proportion to its speed (no feedforward). So training, observation, evaluation and deployment were extended for trajectory tracking.
The controller structure (action, mixer, estimator, bridge) is unchanged.

| Item | Content |
|---|---|
| Goal motion | `rov_rl/goals.py`: per episode stationary (30 %) / straight or curved (speed 0.02–0.25 m/s, turn rate ±0.6 rad/s, vertical ±0.05 m/s), heading-follow 50 %, reflected off the pool walls |
| Observation | 17 + **goal velocity (body frame, 3) + goal yaw rate (1)** + previous action 4 = 25 × 4 frames = 100 (`obs_goal_velocity`, old checkpoints automatically get 84) |
| Episode | 6 s → **20 s** |
| Drag DR (`traj` preset) | drag ×0.7–3.0 + added linear drag (30, 45, 10) N·s/m × 0–1.5 [유도: station speed law `speed = 0.692(\|axis\|−0.096)` → drag on the real vehicle is 2–4× the model's] |
| Evaluation | `scripts/eval_trajectory.py`: square 1 m @0.15 m/s, circle r 0.5 @0.15 m/s, lawnmower — RL vs PID, cross-track p95 · position RMS · yaw RMS · saturation fraction, `logs/eval/traj_<tag>/path.png` |
| Deployment | `deploy/run_policy.py --traj square:1.0,0.15 --laps 2` (defined in a frame anchored at the start pose, NED conversion automatic) — `deploy/trajectory.py` |

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 1000 --headless --dr traj --run_name traj \
    env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05
python scripts/eval_trajectory.py --checkpoint logs/rsl_rl/rov_posehold/*_traj/model_999.pt --traj square:1.0,0.15 --laps 2 --headless
```
To run it independently of the session: `setsid nohup bash scripts/_pipeline_traj.sh > logs/pipeline_traj_outer.log 2>&1 < /dev/null &`.

### 8.0 4th iteration (traj4, 2026-09-21 evening): addressing the depth problem
* **Per-axis separable reward** (`env.rew_separable=True`): horizontal position/velocity and depth position/velocity are rewarded separately (σ_xy 0.3 m, σ_z 0.15 m; velocity σ 0.15 / 0.10 m/s). With a single 3-D norm the z gradient is diluted while horizontal error remains (`tests/test_hydro.py::test_separable_rewards_keep_z_gradient`).
* **Nominal linear damping** (`configs/rov_params.yaml hydro.linear_damping_nominal` = 20/30/15 N·s/m [유도 station speed law]): the paper's model has only 2nd-order drag, so low-speed damping is close to 0 and it produces a heave limit cycle. It now also goes into the nominal model and evaluation, and its DR is 0.5–1.5× (`traj4` preset).
* **Asymmetric actor-critic** (`env.critic_privileged=True`): the critic sees the policy observation (100) + the true state and DR parameters (31). The actor (deployed) is unchanged.
* **3 seeds** (`--seed 0/1/2`, 1200 iter) and selection by the worst-case metric with `scripts/select_best.py`.

### 8.0b 5th iteration (traj5): added mass (completing Fossen)
Until now, of the Fossen equation only the rigid-body, restoring-force and damping terms were present, and **the added mass M_A·ν̇ and C_A(ν)ν were missing**. For the BlueROV2 the heave added mass is 14.6 kg and sway 12.7 kg,
the same class as the vehicle mass (13.7), so the effective inertia differs by 2×. Implementation (`rov_rl/hydrodynamics.py AddedMass`): every substep the body acceleration is
estimated with a low-pass filter (α 0.3), and −M_A·ν̇ is applied as an external force and C_A(ν)ν via the Fossen diagonal formula (the uuv_simulator / MarineGym approach; it is an explicit delay, but α·M_A/m < 1, so it is stable,
`tests/test_hydro.py::test_added_mass_effective_inertia_and_coriolis`). Coefficients MarineGym BlueROVHeavy (Wu 2018) [스펙], DR ×0.5–1.5 (`traj5` preset).
Since it is part of the nominal physics, it also enters `--dr none` evaluations (earlier results are values from a world without added mass). The result of re-evaluating traj4_s2 under the new physics is a sim-to-sim robustness metric.

### 8.1 Results (square 1 m / circle r 0.5 m @0.15 m/s, heading-follow corner slew 60°/s, 2 laps; full table `logs/eval/RESULTS.md`)

| Policy | Training setup | Square xy RMS / z RMS [m] | Square along RMS | Circle xy RMS / along RMS | Circle yaw RMS [rad] | Saturation |
|---|---|---|---|---|---|---|
| prac (position-hold training, no feedforward) | ours DR, stationary goal, 6 s | 0.139 / 0.034 | 0.133 | 0.152 / 0.143 | 0.132 | 4 % |
| traj | traj DR (heavy drag), moving goal, 20 s | 0.114 / 0.080 | 0.092 | 0.105 / 0.072 | 0.197 | 20 % |
| traj2 | traj2 DR, moving goal, velocity reward | **0.085 / 0.098** | **0.072** | **0.061 / 0.021** | **0.123** | 1 % |
| traj3 | traj2 + narrowed drag DR, constant-depth goal, rate 0.1 | 0.154 / 0.261 | 0.121 | 0.101 / 0.054 | 0.095 | 19 % |
| **traj4_s2** (8.0 setup, seed 2) — **recommended for deployment** | separable reward + nominal linear damping + privileged critic | **0.059 / 0.032** | 0.052 | **0.058 / 0.035**, lag 0.040 | 0.069 | 3 % |
| traj4_s0 / traj4_s1 (seed spread) | 〃 | 0.063 / 0.032, 0.065 / 0.036 | 0.050, 0.049 | 0.066 / 0.028, 0.034 / 0.021 | 0.129, 0.063 | 5 %, 9 % |
| **traj5_s2** (8.0b, with added mass) — **final recommendation for deployment** | traj4 + added mass M_A·ν̇ + C_A(ν)ν, DR ×0.5–1.5 | **0.053 / 0.020** | 0.047 | **0.057 / 0.021**, lag 0.044 | 0.043 | 2 % |
| traj5_s0 / traj5_s1 | 〃 | 0.054 / 0.023, 0.053 / 0.027 | 0.040, 0.042 | 0.062 / 0.015, 0.063 / 0.020 | 0.089, 0.059 | 17 % (corner yaw), 4 % |
| traj4_s2 re-evaluated in the added-mass world (sim-to-sim) | trained without added mass | 0.065 / 0.054 | 0.056 | 0.058 / 0.051 | 0.076 | 2 % |
| PID derate 0.7 | baseline (no feedforward) | 0.106 / 0.022 | 0.085 | 0.076 / 0.072 | 0.232 | 0 % |

* Effect of feedforward: the lag along the direction of travel (along-track) goes prac 0.143 → traj2 **0.021 m** (circle). In horizontal tracking traj2 is better than PID (square xy 0.085 vs 0.106, circle 0.061 vs 0.076).
* Remaining problem: **depth** hold of the policies trained with moving goals (traj2 z RMS 0.098, ±0.1 m heave oscillation even at rest, −0.3 m at the corners; prac 0.034, PID 0.022). traj3, with the drag DR narrowed and the depth goal fixed, got even worse (z RMS 0.26), so the cause is not the width of the drag DR alone — 20 s episodes + moving-goal training appear to loosen the heave loop; unresolved.
* Under shifted dynamics (drag ×1.5 etc.) RL is better than PID (traj square 0.094 vs 0.145 m) — the same direction as the station records showing that drag on the real vehicle is larger than in the model.

**Results of the 4th iteration (traj4) — depth problem solved.** All 3 seeds trained with separable reward + nominal linear damping + privileged critic came down to a depth RMS in the 0.03 m range
(traj2 0.098), and horizontal tracking and disturbance rejection are also better than PID on every item. Disturbance test pos RMS (3-D): 0.055–0.070 m (PID 0.08). The spread between seeds is small
(`scripts/select_best.py` worst-case-metric score 0.91 / 1.08 / 1.79; prac 1.79, traj2 1.96).

**Results of the 5th iteration (traj5, with added mass).** All 3 seeds are better than traj4 and the spread is small (worst-case-metric score 0.85 / 0.91; s0 is 3.35 because of corner yaw saturation, but
saturation on the straight segments is 1 % and sign flips are 0.11/step — not chattering). Running traj4_s2, trained without added mass, in the added-mass world worsens the depth RMS 0.032 → 0.054 m
(sim-to-sim), so added mass is an important term for the transfer to hardware as well. Disturbance test pos RMS (3-D) traj5: 0.050–0.060 m (PID 0.08).

**Final recommendation**: RL on all axes — **traj5_s2** (`logs/rsl_rl/rov_posehold/*_traj5_s2/exported/`; for position hold, trajectory tracking and disturbances alike). First hardware test with `--axis-cap 0.5`,
in the order stationary → square 1 m @0.10 m/s → 0.15 m/s. The observation spec (`obs_spec.json`) records whether feedforward is included, so `run_policy.py` matches it automatically.

### 8.2 After the first hardware test (2026-09-21 night): making the reference trajectory start at the vehicle's start point (goals.py v2, preset traj6 — **not trained yet**)

The first hardware test (`umi_underwater_robust_control/data/20260921/0921_182926`, LOW=rl, traj5_s2, axis_cap 0.5) completed 5 laps of the square 1 m @0.10 m/s
at xy RMS 0.042 m [측정], but in the approach segment it accelerated up to a maximum of 0.45 m/s [측정] and was too fast. The cause is the training start condition: in every episode the goal was
placed up to 1 m away from the vehicle, so the policy learned "full throttle if the error is 0.2 m or more" (static response 0.1 m → 0.52, 0.2 m → 1.0 [측정 exported traj5_s2]).
User decision: change the reference trajectory to a random curve that **starts from the vehicle's start position and heading**, and train on it.

**What changed (`rov_rl/goals.py` v2, implementation only, not trained yet).** At episode reset the whole 20 s reference path is sampled at once (referenced by step index; at 2048 envs
reset 1.6 ms, step 0.14 ms, buffer 10 MB [측정]).
* Start: position = vehicle spawn position, heading = vehicle yaw, speed 0 → reaches the cruise speed U[0.02, 0.25] m/s over a 1–3 s smoothstep ramp (±30 % slow modulation).
* Curvature: turn rate |ω| ≤ 0.6 rad/s = constant bias + 3 random sinusoids (0.02–0.15 Hz) → circles, arcs, S-curves. Vertical velocity |vz| ≤ 0.03 m/s in the same way.
* Heading: 50 % follow the direction of travel (station heading_follow), the rest an independent yaw drift |r| ≤ 0.4 rad/s. 30 % decelerate in the second half into station-keeping, 30 % are stationary goals (position hold).
* The pool box (±0.9 × ±2.0 m, altitude 0.4–0.8 m) is enforced by **wall steering**: while the heading is integrated in a causal loop at 0.25 s intervals, a term that turns inward within 0.6 m of a wall
  is mixed in (|ω| ≤ 0.6 rad/s maintained, no steering while at rest). The mirror folding implemented first (wall reflection as in v1) produced heading jumps with a
  median of 111° in 45 % of the 0.25 m/s paths (review finding), so it was discarded. Paths that touch a wall after steering: 5 % (all at cruise > 0.18 m/s, sliding along the wall via the position clamp; the yaw reference is
  continuous, ≤ 1.7° per step) [측정 N=4000]. Reset cost at 2048 envs 15 ms [측정].
* The yaw reference also starts from rest: both the turn rate and the yaw-rate are multiplied by the same envelope as the speed ramp, so the yaw-rate feedforward at t=0 = 0 (consistent with the hardware start condition),
  and on heading-follow paths the rotation stops when the position stops. The independent yaw drift can continue even during station-keeping (the same task as station's heading change while at rest).
* The evaluation scripts (episode 1e6 s) do not build the path (`eval_mode` → T=2; the 640 MB-per-env buffer/OOM problem the review caught). Long episodes outside training are
  capped at `horizon_max_s` 120 s.
* Feedforward = forward difference of the path (the amount the goal actually moves during the next step). The observation layout and `obs_spec` are unchanged.
* Reset (`rov_env.py`): if `moving_goal.start_at_robot` (traj6: true), the vehicle is placed at the path start point (options `init_offset_max`, `init_yaw_max`, default 0)
  and only roll/pitch ±0.1 rad and `init_vel_max` are perturbed. The paper presets (none/small/large/ours) and traj≤5 keep the former spawn ball with `start_at_robot: false`.
* Viewer (`scripts/play.py`, debug vis): draws the **reference path** instead of the goal point — green spheres for the segment ahead, grey spheres for the segment already passed (0.5 s spacing), orange spheres for the trail the vehicle actually travelled
  (last 5 s); the small goal cube and heading arrow are kept. Externally driven envs (evaluation scripts) do not draw the path.
* Reward option (off by default): if `rew_scale_speed_excess` > 0, a linear penalty on the amount by which the actual speed exceeds `rew_speed_limit` — a second remedy that makes the policy approach a distant goal at the capped speed
  (the 1st remedy, on the station side, is `rl.axis_cap 0.3`).

**Training/visualization commands** (for when the user turns them on):
```bash
# training (traj5 recipe as is, only the preset is traj6; keeps running after the session closes): 3 seeds sequentially + evaluation pipeline
setsid nohup bash scripts/_pipeline_traj6.sh > logs/pipeline_traj6_outer.log 2>&1 < /dev/null &
# watch it follow the path (workstation display; before training you can try having the traj5_s2 checkpoint follow the new paths)
DISPLAY=:1 /home/bdml/miniforge3/envs/env_isaaclab/bin/python -u scripts/play.py --task ROV-PoseHold-v0 --num_envs 64 --dr traj6 \
  --checkpoint logs/rsl_rl/rov_posehold/2026-09-21_16-53-50_traj5_s2/model_1199.pt
```
Verification: `tests/test_goals_traj.py` (path start point, continuity, box, stationary, yaw continuity, feedforward), 64-env headless play for 450 steps (markers included), 3-iteration training smoke test,
`eval_trajectory.py` 1 lap (traj5_s2: pos RMS 0.053 m — matches the 8.1 table) [측정 2026-09-21 passed]. The 3 confirmed findings of the adversarial review (4 perspectives × 2 refutations, 42 agents)
(evaluation buffer, yaw ramp, wall reflection) were applied. traj6 training with 3 seeds started 2026-09-21 20:05 (`logs/pipeline_traj6_outer.log`).

> **Evaluation checkpoint correction (2026-09-27 review)**: the checkpoint selection in `scripts/_pipeline_eval.sh` (`sort -t_ -k2 -n`) was keyed on the time field of the path and fell back to lexicographic
> order, so **all 1200 iter runs were evaluated at `model_950`** (the export is `model_1199`). The stored figures in the 8.3·8.4 tables and in `logs/eval/*traj{4..10}*` are the values at iteration 950
> (the `== checkpoint` line of each `logs/pipeline_*.log`). The script has been fixed and traj10_s0..2 were re-evaluated at 1199 (the old folders are kept as `*_it950`,
> and in the new folders `meta.json` records the ckpt path and iteration). The 4-axis vs pwm comparison in §10.5 is all at 1199.

### 8.3 traj6 → traj9 training results (2026-09-21 night, 2048 envs × 1200 iter, seeds in parallel; full table `logs/eval/RESULTS.md`)

| Policy | Setup | Square xy / z | Circle xy / along | Disturbance RMS | Saturation max | 0.5 m step peak speed / overshoot | Worst-case metric |
|---|---|---|---|---|---|---|---|
| **traj5_s2** (currently deployed) | spawn ball 1 m, moving goal v1 | 0.053 / 0.020 | 0.057 / 0.044 | 0.055 | 2 % | 0.48 m/s / 0.025 m | **0.85** |
| traj6_s3 (best tracking of 5 seeds) | path starts at the vehicle, offset 0 | **0.048 / 0.034** | **0.025 / 0.019** | 0.056 | 5 % | 0.58 / 0.113 | 1.22 |
| traj6_s1 | 〃 | 0.052 / 0.025 | 0.041 / 0.036 | 0.067 | 5 % | 0.54 / 0.120 | 1.02 |
| traj7_s0 (best of 3 seeds) | traj6 + offset 0.3 m + speed penalty 2.0 @0.25 m/s + turn rate 1.0 | 0.057 / 0.029 | 0.066 / 0.064 | 0.054 | 2 % | 0.53 / 0.064 | 0.89 |
| traj8_s0 (best of 3 seeds) | traj7 + offset 0.8 m + penalty 4.0 @0.25 m/s | 0.121 / 0.018 | 0.162 / 0.134 | 0.068 | 7 % | **0.32 / 0.000** | 1.68 |

* **traj6**: trajectory tracking improved a lot (circle xy 0.057 → 0.025 m, lag 0.044 → 0.019 m), but on the 0.5 m step it is faster than traj5 (0.58 m/s) with 4× the overshoot.
  Removing distant goals from training does not make the lunging go away; instead it yields a policy that **never learned how to handle distant goals at all** (static response 0.1 m → command 0.88).
* **traj7**: no change from the speed penalty alone — at offset 0.3 m the cap is never exceeded during training, so there are no states for the penalty to shape (penalty term mean −0.04 → 0).
* **traj8**: a distant start + a strong penalty shapes the step response as desired (peak 0.24–0.32 m/s, overshoot 0), but tracking collapses (along 0.13 m on the 0.15 m/s circle).
  The speed cap and tracking precision conflict directly within the reward.
* **traj9** (spawn 0.5 m, penalty 1.0 @0.30 m/s; cut off at 881 iter by a workstation reboot and resumed from `model_850`, `scripts/_resume_traj9_seed.sh`):
  the result is at the same level as traj5 — best seed traj9_s1 worst-case metric 1.08, 0.5 m step peak speed 0.48–0.51 m/s, overshoot 0.02–0.03 m, circle xy 0.046 m.
  Penalty 1.0 @0.30 did not change the step response (same reason as traj7: 0.30 m/s is almost never exceeded during training), and tracking is on par with traj5. **Conclusion: lowering the approach speed
  through the reward loses tracking (traj8), and applying it weakly has no effect (traj7·9). Handling it with the station cap (0.3) is the right way.** Steps under cap 0.3 [측정]:
  traj5_s2 0.16 m/s · overshoot 0.013 m, traj9_s2 0.17·0.017, traj6_s3 0.26·0.052.
* The current recommendation remains **traj5_s2 + station `rl.axis_cap 0.3`** (verified on hardware, smallest overshoot). If trajectory precision comes first and the approach speed can be handled with the station cap and leash,
  traj6_s3 is a candidate.

### 8.4 traj10 (2026-09-22): three changes following the hardware error diagnosis

Re-analysis of hardware run `0921_182926` [측정]: the along-track error is a **constant bias** of +0.034 m (standard deviation 0.016; cross-track −0.002 ± 0.013) → 80 % of the xy RMS 0.042.
Static pitch **−11°** (−10.6° at rest; runs before the 9/9 payload change were −3~−7°) → outside the sim training distribution (level ±6°).
1. **Narrow position reward terms** `rew_scale_pos_xy_fine 0.15 (σ 0.08 m)`, `rew_scale_pos_z_fine 0.15 (σ 0.04 m)`: the wide term (σ 0.3) has a slope of
   only 1 % at 3 cm, so the policy trades the last few cm for the action penalty; the narrow term makes it lose 13 % at 3 cm.
2. **Static CB trim** `dr.cob_trim_x_m −0.029` [유도 0.15 m·tan 11°] + the existing 1.5 cm disk (±6°): puts the hardware's nose-down 11° at the center of the training range.
   Restoring the trim with ballast on the real vehicle is the fundamental fix (19 % vertical leakage of thrust, camera tilted 51° down).
3. **Impulse disturbance removed, replaced with a tether-like bias force** (user decision): `push_prob_per_step 0`, `bias_force_n [0, 4]` N horizontal, direction random at reset + random walk
   (0.3 rad/√s) [예측]. The privileged critic sees this force (`critic_priv_bias`, width 31 → 33; for older checkpoints `_eval_common` restores 31).
   The impulses in the disturbance evaluation (`eval_disturbance`) are left as they are, so the results stay comparable.
4. Turn-rate range 1.0 rad/s (station corner slew).

**Results (3 seeds, 2026-09-22)** [측정]. Both in the standard world (level trim) and in a nose-down 11° world like the hardware (`--shift cob_x=-0.029`):

| World | Policy | Square xy / along RMS, along bias | Circle xy / along | z RMS | Disturbance RMS | Cap 0.3 step peak speed / overshoot / steady-state error | Saturation |
|---|---|---|---|---|---|---|---|
| Level | traj5_s2 | 0.053 / 0.047, +0.029 | 0.057 / 0.044 | 0.020 | 0.055 | 0.16 / 0.013 / 0.016 | 2 % |
| Level | **traj10_s1** | **0.035 / 0.027, +0.003** | **0.026 / 0.018** | 0.034 | 0.040 | 0.24 / 0.016 / 0.041 | 1 % |
| Level | traj10_s0 | 0.041 / 0.033, −0.016 | 0.019 / 0.014 | 0.036 | 0.039 | 0.24 / 0.010 / 0.032 | 4 % |
| −11° | traj5_s2 | 0.066 / 0.053, +0.030 | 0.064 / 0.046 | 0.025 | – | 0.20 / 0.012 / 0.024 | 2 % |
| −11° | **traj10_s1** | **0.047 / 0.030, −0.006** | **0.036 / 0.012** | 0.028 | – | 0.23 / 0.014 / 0.029 | 1 % |
| −11° | traj10_s0 | 0.049 / 0.037, −0.019 | 0.024 / 0.013 | 0.024 | – | 0.22 / 0.006 / 0.019 | 5 % |

* The along bias (80 % of the hardware error) vanished, +0.03 → 0, and in the trim world matching the hardware, horizontal tracking improved by 30–45 %. Even though impulse disturbances were removed from training,
  disturbance rejection is better (bias-force training). The cost: z RMS in the level world 0.020 → 0.034 (similar in the trim world, 0.025 vs 0.028).
* **Deployment (2026-09-22)**: station `rl.policy_dir` → `rl_policies/traj10_s1` (not yet on hardware; how to revert to traj5_s2 is in the hw_mpc.yaml comments).
  However, if the vehicle is brought back close to level with ballast, traj10 (trained on −11° ± 6°) is out of distribution, so at that point one must either go back to traj5_s2 or train a policy whose DR covers the trim range
  [−0.035, 0] m (the traj11 candidate).

> **Reference speed cap 0.25 → 0.12 m/s (2026-10-05, operator decision)**: `moving_goal.speed_range` of the presets `traj10` / `traj10sea` / `traj10att` and
> the `MovingGoalCfg` default were changed to `[0.02, 0.12]`. Runs before this date (traj10_s*, pwm10*, sea6*, att7pwm_s* etc.) were trained with 0.25
> (`moving_goal` in each run's `params/env.yaml` and in the export's `obs_spec.json`). The traj~traj9 presets are left at 0.25 for the record.
> Generator sample (4096 episodes × 20 s): tick mean 0.079 → 0.042, moving-episode mean 0.114 → 0.060, p90 0.224 → 0.110 m/s [유도: `rov_rl/goals.py` sampling].
> Three cautions:
> 1. A Hydra `env.moving_goal.*` override used together with `--dr` is **silently wiped**, because `apply_dr_preset` creates a new `MovingGoalCfg()`. The speed must be changed in the preset.
> 2. `_pipeline_eval.sh` evaluates at 0.15 m/s. For a 0.12 policy that is out of distribution.
> 3. This range sets only the speed of the moving reference. The speed at which the error is closed is set by the reward, not by the start distribution. The 0.5 m step peak speed is traj6 (start at the robot position) 0.57–0.66, traj10 (1 m ball) 0.50–0.56, and traj8 with a speed penalty 0.32 m/s. The small-error gain, conversely, grew the narrower the start was (static probe d 0.10 m surge, it1199: traj6 0.62–0.88 > traj7 0.55–0.71 > traj9 0.50–0.62 ≈ traj5 0.52 > traj8 0.47–0.49 > traj10 0.41–0.46, confounded with other changes) [측정: `logs/eval/start_dist_review_20261005/{step_metrics,static_probe}.out`].
> When retraining, a new run name must be chosen. Running an existing recipe script as is overwrites `logs/train_<run>.log`.

### 8.5 traj12 (2026-10-05, operator decision): start at the robot position + weak speed penalty, training only

The DR and the 6 reward keys are the same as traj10; what changed are the five things below.
- Preset `traj12`: `moving_goal.start_at_robot true`, `init_offset_max 0.05` m, `init_yaw_max 0.10` rad, `speed_range [0.02, 0.10]`, `init_vel_max 0.05` (traj10 has 0.2).
- hydra: `env.rew_scale_speed_excess=1.0 env.rew_speed_limit=0.12`.
`scripts/_pipeline_traj12_seed.sh` only does training and export and does not run the evaluation chain. Initially the 3 seeds were launched with `systemd-run --user --unit=rl-traj12-s{0,1,2}`.
Error-tail metrics were added to the separable reward logs of `rov_env.py`: `/err/frac_gt_0p10`, `/err/frac_gt_0p30`, `/err/pos_p95_m`, `/vel/frac_over_limit`. Only the logging grew; the behaviour is the same.
Background: even when starting on the path, the error grows during the episode. traj6 averaged 0.100–0.115 m late in training [측정: `logs/train_traj6_s{0,1,3}.log`]. That tail is being recorded for the first time now.

## 9. Assumptions and limitations (pre-hardware checklist)
1. The tags are always visible (user assumption). On hardware, if the fix drops out for 0.5 s or more it goes NEUTRAL, so leaving the field of view = stop.
2. The effective restoring arm 0.15 m, net buoyancy +1 N, thruster delay 2 steps and voltage 0.72 are [유도]/[예측] — the DR ranges were set to cover them. The first pool test is with a `--dr ours` policy and `axis-cap 0.5`.
3. roll/pitch are not commanded (4-axis). The 6-axis (MANUAL_CONTROL s,t) extension only requires opening the roll/pitch inputs of `mixer.py`.
4. The fix input of `run_policy.py` (UDP JSON) requires adding a small bridge on the station side that sends NavFix (or calling this script inside the station process).

## 10. Direct-thruster action mode `pwm` (2026-09-27) — a second version kept alongside the 4-axis policy

User request (2026-09-27): "Also make a policy that outputs PWM directly instead of one whose outputs are x,y,z,yaw, keep both, and train it too." There is one environment, and
`ROVEnvCfg.action_mode` changes only the output (`rov_rl/obs_spec.py`). The 4-axis policy, checkpoints and export do not change by a single byte
(`tests/test_pwm_mode.py::test_layout_two_modes`, the existing 33 tests still pass unchanged; no key in `params/env.yaml`/`obs_spec.json` = `axes`).

| | `axes` (default, everything before 09-27) | `pwm` (new) |
|---|---|---|
| Policy output | 4: `[surge, sway(+우), heave(+상), yaw(+시계)]` ("[surge, sway(+right), heave(+up), yaw(+clockwise)]") ∈ [−1,1] | **8: normalized throttle of motors 1..8** `a_i` ∈ [−1,1] (order = `configs/rov_params.yaml thrusters.positions` = ArduSub motor numbers) |
| To the thrusters | ArduSub vectored-6DOF mixer (pilot gain 0.5, group normalization) → 8 throttles | `throttle_i = pwm_throttle_max · a_i` (default 1.0 = the full 1100..1900 µs range), mixer **bypassed** |
| After that | Identical: command delay · 1st-order lag · per-thruster gain · voltage · T200 curve · thrust geometry `B` (`mixer.py` sigma signs) | Identical |
| Observation | 25 × 4 = 100 (prev_action 4) | **29 × 4 = 116** (prev_action 8); privileged critic 31/33 → 35/37 |
| Reward | The paper's formula as is (`rew_scale_actions` is an 8-dim norm) | Identical — the angle term is a full quaternion, so the pwm policy also actively holds roll/pitch level (impossible for 4-axis) |
| export | `obs_spec.json` `action_mode: axes`, wrench head = axes × axis_gain (K = M = 0) | `action_mode: pwm`, `output.pwm_model` (T200 curve · voltage · `B_flu` · sigma) + wrench head = nominal steady-state chain, all 6 axes actuated |
| Hardware transport | MANUAL_CONTROL (station `LOW=rl`, `deploy/ardusub_bridge.py`) | **`pwm_lua_override`**: RC_CHANNELS_OVERRIDE ch 9..16 → Navigator **Lua timeout override** (`deploy/ardusub/rl_pwm_override.lua`, the motors stay on the mixer) — `deploy/run_policy.py` only; station `HwRl` **refuses** pwm policies (prints the reason). RCIN passthrough was **discarded** in review (§10.3) |

### 10.1 Code paths
* `rov_rl/obs_spec.py`: `num_actions(mode)`, `frame_dim(goal_vel, n_act)`, `action_convention(mode, throttle_max)`.
* `rov_rl/env/rov_env.py`: `throttles_from_actions(a)` (the mode branch, the only branch point), `action_from_axes(a4)` (in the pwm env, converts the 4-axis commands of the PID baseline and scripts into
  "the throttle the mixer would have produced / pwm_throttle_max", so the same PID runs in both envs), `wrench_ned_estimate(a)` (nominal-chain wrench, for logging, FLU→FRD signs `[1,−1,−1,1,−1,−1]`).
* Evaluation scripts (`eval_sweep/eval_disturbance/eval_trajectory/play_control`): read the action width from the env (`env.n_act`), CSV columns `a0..a{n−1}`,
  `_eval_common.apply_run_obs_cfg` restores the run's `action_mode`/`pwm_throttle_max`. `select_best.py`/`summarize_results.py` generalize the `a*` columns
  (sign-flip metric: the yaw axis for 4-axis, the mean over the 8 channels for pwm).
* `deploy/pwm.py` (numpy, no torch): `PwmModel(obs_spec.output.pwm_model)` → throttle / pulse / wrench estimates, `PwmPolicy.step(obs) → u[6]`;
  besides the T200 curve, voltage, `B_flu` and sigma, `pwm_model` carries **`motor_direction`** (the `MOT_n_DIRECTION` assumed by the training mixer), `pwm_cap_trained` and `transport: pwm_lua_override`.
  **The cap has a single unit**: `env.pwm_cap` = `run_policy --pwm-cap` = the |throttle| cap (as a fraction of ±400 µs). The env stores `clamp(throttle)/pwm_throttle_max` as the action, so
  prev_action equals the bridge's feedback (`rc_override_to_action` = throttle/max), and the export's wrench head, `PwmModel` and `run_policy` (lowers it to the trained cap if it exceeds the trained cap) use the same cap.
  `deploy/ardusub_bridge.py`: `action_mode="pwm"`, `send_pwm(a)` (|throttle| ≤ `pwm_cap`, pulse = 1500 + 400·thr clamped to 1100..1900,
  channels other than ch 9..16 at 65535 = hold; 0/65534 are never sent — in Sub-4.5 they mean 'hold'/'release' respectively), every frame is accompanied by a neutral MANUAL_CONTROL (RC override does
  not refresh the pilot-input failsafe). Pre-engage gates (all read with PARAM_REQUEST_READ; **refuse** if even one mismatches): `SERVO1..8_FUNCTION = 33..40`
  (Motor1..8, mixer kept), `MOT_1..8_DIRECTION == motor_direction` (raw pulses skip the mixer's direction reversal, so a single parameter silently reverses a single thruster),
  `SCR_ENABLE 1`, `SCR_USER1 1`, `RC_OPTIONS` bit1 = 0, `RC_OVERRIDE_TIME > 0`, `RC9..16_OPTION 0`, `SYSID_MYGCS == 255`, script heartbeat
  `NAMED_VALUE_FLOAT "RLPWM"` within 2 s, MAVLink 2 (`MAVLINK20=1`; the v1 dialect has no ch 9..18). Conversely, the `axes` bridge refuses a vehicle whose outputs 1..8
  are RCIN passthrough. Neutral = all 8 channels at 1500 µs, deadman identical (`tests/test_pwm_mode.py::test_bridge_pwm_packing_gates_and_neutral`).
* `deploy/ardusub/rl_pwm_override.lua` (**vehicle side, not run [예측]**): receives RC_CHANNELS_OVERRIDE(70) via `mavlink:receive_chan` and forces ch 9..16
  onto the motor outputs with `SRV_Channels:set_output_pwm_chan_timeout(i, pwm, 100 ms)` — only when `SCR_USER1 == 1 ∧ armed ∧ MANUAL ∧ 250 ms 이내 프레임` ("SCR_USER1 == 1 ∧ armed ∧ MANUAL ∧ a frame within 250 ms").
  If the frames stop, the override expires after 1 neutral frame and the outputs return to the AP_Motors output (neutral mixer = 1500, 1500 when disarmed) (worst-case hold = STALE 100 + TIMEOUT 100 = 200 ms)
  → DISARM, E-stop and the leak, battery, pilot-input and GCS failsafes all stay alive. The script drops frames whose sysid is not `SYSID_MYGCS` or whose target is not `SYSID_THISMAV`
  (AP_Scripting hands messages to scripts **before** the GCS sysid filter). The packet offsets (2-byte checksum first, msgid 10..12, payload 13.., field order
  chan1..8 · target_system · target_component · chan9..18) were cross-checked by the 2026-09-27 review against pymavlink `mavlink_types.h` and ArduPilot `mavlink_msgs.lua` (the first draft had them wrong).
  Install: BlueOS file-browser `configs/ardupilot-manager/firmware/scripts/`, reboot after `SCR_ENABLE=1`, `SCR_HEAP_SIZE ≥ 100000`.
* `deploy/run_policy.py`: picks the path from `obs_spec.action_mode`, start at `--pwm-cap 0.25` (±100 µs), `--pwm-channels 9,..,16`; log columns `a_m1..8, pwm1..8, uX..uN`.
* station: `rov_gui/control/rl_policy.py` refuses a policy with `action_mode != axes` at construction time (`rov_gui/tests/test_control.py::test_rl_refuses_a_pwm_policy_with_the_reason`).

### 10.2 Training (traj10 recipe as is, only the mode is pwm; keeps running after the session closes)
```bash
for S in 0 1 2; do systemd-run --user --unit=rl-pwm10-s$S --collect -p WorkingDirectory=/home/bdml/Desktop/RL_controller \
    bash scripts/_pipeline_pwm10_seed.sh $S; done          # Linger=yes, so unaffected by logout / laptop shutdown
systemctl --user status rl-pwm10-s0                       # status;  journalctl --user -u rl-pwm10-s0
tail -3 logs/train_pwm10_s0.log                           # EXIT=0 when finished, then logs/pipeline_pwm10_s0.log (export + all evaluations)
python scripts/select_best.py traj10_s0 traj10_s1 traj10_s2 pwm10_s0 pwm10_s1 pwm10_s2   # 4-axis vs pwm worst-case metric comparison
```
`--dr traj10 env.action_mode=pwm env.rew_separable=True env.critic_privileged=True env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05
env.rew_scale_pos_xy_fine=0.15 env.rew_scale_pos_z_fine=0.15`, 2048 envs × 1200 iter, seeds 0/1/2 in parallel. Run name `pwm10_s<seed>`; the corresponding 4-axis run is `traj10_s<seed>`.

### 10.3 Pre-hardware checklist (pwm policy) — transport design decision (2026-09-27 review, evidence from the Sub-4.5 source)
* **RCIN passthrough (`SERVOn_FUNCTION = RCINk`) is discarded.** For a passthrough output, `SRV_Channel::output_ch()` writes the RC input through as is, so it does not look at ARM/DISARM;
  AP_Motors' disarm 1500 goes only to Motor functions; and since this ROV has no RC receiver, the last pulse remains **indefinitely** even after `RC_OVERRIDE_TIME` expires (if the bridge dies,
  the only way to stop it is cutting power). Sub 4.5 does not run the unassigned-motor arming check, so there is no warning either [스펙 ardupilot Sub-4.5: SRV_Channel_aux.cpp, RC_Channel.cpp,
  AP_Motors6DOF.cpp, AP_Arming_Sub.cpp — `.claude/journal/reviews.md` 2026-09-27]. `MAV_CMD_DO_SET_SERVO` (refuses Motor channels, no timeout) and
  `DO_MOTOR_TEST` (requires ARMED, 500 ms watchdog DISARM, no atomicity across the 8 motors) are also excluded from the candidates. The custom frame (`SUB_FRAME_CUSTOM`) is an empty case, so it needs a firmware build, and
  it is a 6-axis linear mixer anyway.
* **Adopted: `pwm_lua_override`** — the motors stay Motor1..8, and Lua on the Navigator forces them with `set_output_pwm_chan_timeout` for only 100 ms at a time. The gates run inside the vehicle.
  This path is still **unverified on the bench**. Order: (1) with the propellers removed, install the script and set `SCR_USER1=1`, (2) confirm the bridge dry-run passes the gates, (3) motor i only, 1560 µs for 1 s →
  confirm `SERVO_OUTPUT_RAW servo{i}_raw` follows → confirm it returns to 1500 within 200 ms when sending stops, (4) **measure the per-thruster signs**: in water, held on the tether, give +100 µs to one thruster at a time and
  check the physical direction of all 8 against `obs_spec.pwm_model.sigma` and `thrust_dirs_flu` from the sign of the gyro/ATTITUDE response — the 4 vertical thrusters are verified only as a sum on the single heave axis
  (`logs/mixer_fit_report.md`), so even if one is reversed the 4-axis policy cannot feel it, but for the pwm policy the leveling and yaw run away in the opposite direction.
* The first test is station-keeping only, at `--pwm-cap 0.25` (±100 µs, just above the ESC deadband of ±25 µs); the cap is fed back through prev_action (the same structure as the 4-axis `axis_cap`).
* Authority of the training default `pwm_throttle_max 1.0`: on a single-axis surge command it is **3.4×** that of 4-axis (JS_GAIN 0.5) (38 N vs 131 N; the T200 curve is 2nd-order in throttle [유도 mixer+thrusters]);
  however, 4-axis also reaches 1.0 per thruster on combined commands through group normalization. If set to 0.5, leveling the −11° trim (front 2 thrusters +0.75) leaves only 25 % headroom and depth saturates, so
  1.0 is right for training, and the comparison table is read with the saturation fraction, mean |a| and vertical-thruster mean |thr| (`/act/vert_mean_abs`) to separate the authority difference from control quality.
  `uX..uN` in the `eval_trajectory` CSV means different things per mode (4-axis = station pseudo-wrench 60 N/axis, pwm = physical estimate) — compare the two modes with the new columns `fX..fN` (a nominal
  physical estimate in both modes). In pwm mode the `pilot_gain_scale` DR has no effect (mixer bypassed), and voltage_scale ±17 % also serves that role.
* hydra checks types: `env.pwm_throttle_max=1` (int) is refused; use `1.0`.
* An `axis_cap`/leash like the 4-axis policy's exists only in station; for pwm, the caps, deadman and gates of `run_policy.py` are all there is.

### 10.4 Second recipe `pwm10t` — shaping for a policy that can tilt (3 review findings, `scripts/_pipeline_pwm10t_seed.sh`)
Alongside `pwm10` (the paper's reward as is), a variant with the three items the review flagged turned on is trained with 2 seeds. Everything else is the same, so the difference between the two runs is the shaping effect.
1. **Attitude reward yaw + tilt** (`env.ang_error_mode=yaw env.rew_scale_tilt=0.12 env.rew_tilt_sigma=0.10`): at the −11° trim the full-quaternion term's loss is 0.087/step
   against a leveling cost (4 vertical thrusters ≈ ±0.35 throttle) of 0.019/step, so the pwm policy definitely levels; but even at a 2° residual the angle loss of 0.017/step exceeds the fine-z loss
   of 0.009 for 1 cm of z, so leveling the last few degrees crowds out depth [유도]. A Gaussian tilt term (σ 0.1 rad) gives an incentive of 0.116 at 11° but is flat within ±3°.
2. **Tilt termination 60° · tag dropout 35°** (`env.terminate_tilt_rad=1.05 env.tag_dropout_tilt_rad=0.61`): cuts off the capsizing episodes of early exploration (std 1 × ±46 N vertical pair = 20 N·m roll),
   and by cutting the fix at the tilt where the C3's 40°-down mount loses the floor tags, also enforces leveling through observation availability.
3. **Exploration floor** (`--min_noise_std 0.05`, `scripts/train.py` clamps std after the PPO update, recorded in `params/min_noise_std.txt` → the export carries it as `obs_spec.json
   training.min_noise_std`, re-applied automatically on `--resume`): the final std of the 4-axis runs, 0.007 (traj10_s1) / 0.014 (traj5_s2)
   [측정 `logs/rsl_rl/rov_posehold/2026-09-22_04-15-41_traj10_s1/events.out.tfevents.*`, `…/2026-09-21_16-53-50_traj5_s2/events.out.tfevents.*`,
   `Policy/mean_noise_std` last iter], is inside the T200 deadband (0.075 in action units), so there is no gradient for fine correction, and on 8 channels the 1 N quantization
   causes temporal dither (= ESC reversals). At 0.05, 13 % of samples fall outside the deadband [유도 normal distribution σ 0.05, deadband 0.075: 2(1−Φ(1.5))]. Adoption metrics:
   the pwm sign flips in `summarize_results` (counting only samples outside the deadband) and the saturation fraction.
`/att/tilt_deg`, `/act/vert_mean_abs`, `/act/horiz_mean_abs`, `/act/in_deadband_frac` were added to the training logs (both recipes).

### 10.5 Results (2026-09-27, 2048 envs × 1200 iter, all at iteration 1199; table = `logs/eval/COMPARE_MODES_20260927.md`, a `meta.json` in every folder)

`python scripts/compare_modes.py traj10_s0 traj10_s1 traj10_s2 pwm10_s0 pwm10_s1 pwm10_s2 pwm10t_s0 pwm10t_s1` [측정 `logs/eval/{traj,sweep_rl,disturbance}_<run>_*`].
NOMINAL = level-trim world, TRIM = the hardware's −11° nose-down world (`--shift cob_x=-0.029`, the same protocol as 8.4). Units m / MSE, angles deg.

| run | mode | NOMINAL square xy / z / along | circle xy / along | step pos (steady) / ang | disturbance | saturation | \|a\| | vertical \|a\| | tilt° | TRIM square xy / z / along | circle xy / along | step pos / ang | pitch° | saturation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traj10_s0 | axes | 0.039 / 0.036 / 0.031 | 0.018 / 0.015 | 0.024(0.0016) / 0.049 | 0.060 | 4 % | 0.21 | – | 2.5 | 0.043 / 0.022 / 0.036 | 0.017 / 0.014 | 0.023 / 0.078 | +10.6 | 5 % |
| traj10_s1 | axes | 0.032 / 0.033 / 0.026 | 0.026 / 0.019 | 0.024(0.0017) / 0.038 | 0.062 | 0 % | 0.18 | – | 2.3 | 0.036 / 0.029 / 0.028 | 0.023 / 0.012 | 0.024 / 0.071 | +10.6 | 0 % |
| traj10_s2 | axes | 0.042 / 0.036 / 0.031 | 0.025 / 0.016 | 0.023(0.0016) / 0.033 | 0.062 | 10 % | 0.21 | – | 2.6 | 0.043 / 0.038 / 0.032 | 0.024 / 0.015 | 0.023 / 0.070 | +10.6 | 12 % |
| pwm10_s0 | pwm | 0.032 / 0.036 / 0.024 | 0.027 / 0.026 | 0.019(0.0009) / 0.036 | 0.059 | 2 % | 0.24 | 0.24 | 6.1 | 0.030 / 0.016 / 0.023 | 0.030 / 0.029 | 0.019 / 0.032 | +4.3 | 1 % |
| **pwm10_s1** | pwm | 0.031 / 0.018 / 0.021 | 0.018 / 0.015 | 0.019(0.0005) / 0.041 | 0.048 | 1 % | 0.30 | 0.29 | 7.7 | 0.038 / 0.020 / 0.027 | 0.026 / 0.024 | 0.019 / 0.024 | +0.8 | 2 % |
| pwm10_s2 | pwm | 0.035 / 0.027 / 0.019 | 0.022 / 0.018 | 0.018(0.0004) / 0.042 | 0.048 | **25 %** | 0.38 | 0.49 | 9.2 | 0.036 / 0.022 / 0.023 | 0.024 / 0.021 | 0.018 / 0.027 | +0.1 | 29 % |
| pwm10t_s0 | pwm(t) | 0.039 / **0.512** / 0.030 | 0.026 / 0.021 | 0.250 / 0.026 | 0.499 | **82 %** | 0.56 | 0.50 | 0.0 | 0.039 / 0.512 / 0.030 | 0.027 / 0.023 | 0.249 / 0.023 | −0.1 | 82 % |
| pwm10t_s1 | pwm(t) | 0.036 / 0.032 / 0.022 | 0.019 / 0.016 | 0.019(0.0011) / 0.030 | 0.051 | 13 % | 0.29 | 0.29 | 7.2 | 0.035 / 0.031 / 0.022 | 0.017 / 0.015 | 0.018 / 0.023 | +1.3 | 17 % |

How to read (authority and quality separated, as the review required):
* **Trajectory tracking is on par.** Square xy 0.031–0.035 (pwm10) vs 0.032–0.042 (4-axis), circle 0.018–0.027 vs 0.018–0.026, along 0.019–0.024 vs 0.026–0.031.
  Step response and disturbance rejection are slightly better with pwm (step pos MSE 0.018–0.019 vs 0.023–0.024, steady state 0.0004–0.0009 vs 0.0016; disturbance 0.048–0.059 vs 0.060–0.062).
  However, pwm's single-axis authority is 3.4× (§10.3), so part of this difference is authority — mean |a| is pwm 0.24–0.38 vs 4-axis 0.18–0.21.
* **What only pwm can do = level hold.** In the hardware trim world the 4-axis policy sits at +10.6° nose-down (not actuatable), while pwm10 rights it to +0.1~+4.3°
  → step angle MSE 0.070–0.078 → 0.024–0.032. Cost: a mean |throttle| of 0.24–0.29 on the 4 vertical thrusters (pwm10_s0/s1; ≈ 3–5 N/thruster via the T200 curve and V 0.72 [유도]), drawn constantly.
  In the level world the same policy **overcompensates** to 6–8° nose-up (negative FLU pitch) — like traj10, it is a policy fitted to a −11° ± 6° trim distribution, so
  if the ballast is brought back to level, retraining (trim range [−0.035, 0] DR, the traj11 candidate) is needed.
* **Two seed failure modes (columns you must check before hardware = saturation, vertical |a|).** (1) `pwm10_s2`: the 4 vertical thrusters' signs are `[+0.57, +0.36, +0.58, +0.33]` — multiplied by sigma they become
  FR↑·FL↓·RR↓·RL↑, burning ≈0.45 throttle in the **allocation null space** (wrench 0, the thrusts merely cancel) at 25–29 % saturation; the action penalty 0.05·exp(−|a|²) cannot prevent this
  → a per-thruster |a| penalty (L1) or a null-space projection penalty is the next remedy. (2) `pwm10t_s0`: fell into a **hull-on-the-floor optimum** during training (err/z 0.46 m, pressing down with 47 N,
  tilt 0.3°; `logs/train_pwm10t_s0.log`) — a local optimum where floor contact fills the tilt and yaw terms for free. It has the same shape as the hull-on-the-floor habit on hardware ([memory hull-on-floor]),
  so the sim also needs a floor-contact penalty / a stronger minimum-altitude bound. pwm10t_s1 is normal (best step angle 0.023–0.030, saturation 13–17 %).
* **pwm10t (yaw+tilt reward, std floor 0.05)** had 1 of its 2 seeds collapse, so its effect cannot be judged — redo with 3 or more seeds + a hull-on-the-floor penalty.
* **Candidate**: `pwm10_s1` (`logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s1/exported/`): z RMS 0.018–0.020, disturbance 0.048, trim-world pitch +0.8°, saturation 1–2 %.
  The 4-axis deployment stays traj10_s1. Station comparison of the two modes: with the `fX..fN` columns.
* Pre-hardware order (§10.3): Lua script bench (propellers removed) → per-thruster sign measurement → station-keeping at `--pwm-cap 0.25`. The station refuses pwm policies.

## 11. Wave/current disturbance training + 6-axis mode (2026-09-27 afternoon) — three versions trained on a mixed with/without-disturbance distribution

User request: using the wave/current formulas of `bluerov2_mujoco_marinegym/` as is, put the cases with and without disturbance into **one training distribution**,
train three versions, 4-axis / 6-axis (x,y,z,yaw,pitch,roll) / PWM, and then evaluate on 5 laps of a 1 m × 1 m square.

### 11.1 Disturbance port (`rov_rl/seastate.py` = a torch batched version of `bluerov2_mujoco_marinegym/disturbance/{waves,current,env}.py`)
* Same formulas: JONSWAP S(ω) × cos^{2s} D(β) component grid, a_ij = √(2 S D dω dβ), finite-depth dispersion ω² = g k tanh(kh) (Newton), depth profile
  cosh/sinh, velocity + analytic acceleration, Gauss-Markov drift (exact discretization), 5 modes NONE/C/CD/CW/CDW. Agrees with the numpy reference to 2e-4 at 4 (t, pos) points
  (`tests/test_seastate.py`).
* Same physics decomposition (MarineGym hydro.py): the relative velocity ν_r = ν − Rᵀ v_water enters the drag, linear damping and added-mass finite difference, and the only injected external force is
  Froude-Krylov ρ_fk·Vol·a_wave (ρ_fk 1025, C_M 1, no torque). The still-water env is bit-identical to before (the branch itself is skipped).
  The added-mass finite difference is **seeded with the relative velocity** at reset and in `set_sea_state` (the first version seeded it with the absolute velocity, giving an M_A·v_water kick at t=0 in every disturbance episode [유도 review]).
* Frame: floor z = 0, still-water surface z = h (DR 3.5–4.5 m) → a vehicle at 0.7 m altitude is ≈3.3 m below the surface (0.3 m deeper than the MarineGym site's h 4, z_ROV −3).
* The policy does not observe the water velocity (it cannot be seen on hardware either); with `env.critic_priv_sea=True` only the critic sees v_water_b + FK (6).
* Preset `traj10sea` = traj10 + `sea:` block: p_mode [0.30, 0.15, 0.15, 0.20, 0.20], V̄ 0.05–0.30 m/s, uniform direction, Hs 0.3–1.5 m, Tp 5–12 s, γ 1.5–5,
  s 5–30, uniform wave heading, h 3.5–4.5, 24×7 components (training cost). The disturbance field uses its own `torch.Generator`, so it does not touch the sensor-noise stream.
* The evaluation disturbance is explicit: `--sea "mode=CDW,preset=square|moderate,theta_c_deg=..,seed=.."` — `preset=square` = MarineGym scenario_square.yaml
  (current 0.20 m/s +x, Hs 0.75 / Tp 12 / γ 5 / s 30, h 4, **grid 60×21 0.2–1.6**), `moderate` = base.yaml moderate (Hs 1.0 / Tp 8, 68×21). Fixed envs share the phases and drift
  (RL and PID see the same sea); seed, preset and grid are recorded in meta.json. The evaluation scripts widen the termination box to (3,3)/(0.02,1.5) and record the reset count as `resets`.
* Sense of scale [측정 `logs/smoke/eval_traj10_sea_*.log`, 1 lap]: traj10_s1, trained without disturbance, goes from still water 0.045 m → C 0.126 → CDW **0.262 m** pos RMS (p95 0.50, saturation 14 %).
  A Tp 12 s wave in 4 m of water has an orbital velocity of ≈0.5 m/s [유도], so it is a hard condition.

### 11.2 6-axis mode `axes6` = [surge, sway, heave, yaw, pitch, roll]
* **Key fact (review, Sub-4.5 joystick.cpp)**: ArduSub takes the MANUAL_CONTROL extension fields s(pitch)/t(roll) and **adds them as raw µs** to the pitch/roll RC channels
  (`constrain(s + trim + 1500)`, no JS_GAIN). So the wire carries s = 400·a4, t = 400·a5 µs, and the mixer input = a4/a5 (no gain). The station's attitude axes are
  1000 µs/unit, so 1 policy unit = 0.4 station units (`HwRl` multiplies K/M by 0.4). The first version multiplied by gain 0.5, a plant off by 5×, and that run was discarded.
* Signs: +a4 → +M (nose-up, FRD), +a5 → +K (pinned by tests). `attitude_cap` (roll 0.5, pitch 0.75 in action units = station cap 0.2/0.3) is applied inside the env
  (same structure as pwm_cap); the export carries `attitude_cap_trained` and `attitude_wire`; the bridge's default caps are first-water (0.25, 0.375), sign +1 [assumption].
* obs 27×4 = 108, export WrenchHead 6×6 (roll_nm 13.2 / pitch_nm 7.2); station `HwRl` accepts it only when `engage.attitude_axes.enabled` (refuses otherwise).

### 11.3 Training (v2, restarted after applying the review)
```bash
for M in axes axes6 pwm; do for S in 0 1; do systemd-run --user --unit=rl-sea-$M-s$S --collect \
    -p WorkingDirectory=/home/bdml/Desktop/RL_controller bash scripts/_pipeline_sea_seed.sh $M $S; done; done
# run names sea4_s<S> / sea6_s<S> / seapwm_s<S>; --dr traj10sea env.critic_priv_sea=True + traj10 reward overrides; no exploration floor (same control condition as traj10)
bash scripts/_eval_square_sea_batch.sh sea4_s0 sea4_s1 sea6_s0 sea6_s1 seapwm_s0 seapwm_s1 traj10_s1 pwm10_s1   # square 5 laps × 6 cells
python scripts/plot_square_sea.py --runs ... --out logs/eval/square_sea_20260927                              # paths.png / bars.png / summary.md
```

### 11.4 Results — 1 m × 1 m square, 5 laps @0.15 m/s, 6 seas (2026-09-27, iteration 1199; `logs/eval/square_sea_20260927/{paths,bars}.png`, `summary.md`)

xy RMS [cm] (t ≥ 3 s), parentheses = fraction of steps with |a| ≥ 0.99 on any axis. Seas = MarineGym scenario_square (current 0.20 m/s +x; waves Hs 0.75 / Tp 12, 60×21),
CDWmod = base.yaml moderate (Hs 1.0 / Tp 8). 0 resets in every cell (boundary widened). PID is the baseline run in the same sea in each cell (derate 0.7).
[측정 `logs/eval/traj_<run>_sq5_<mode>/`]

| Policy | NONE | C | CD | CW | CDW | CDWmod |
|---|---|---|---|---|---|---|
| traj10_s1 (4-axis, still-water trained) | 3.4 (0 %) | 12.3 (0 %) | 11.9 (0 %) | 26.3 (15 %) | 25.8 (15 %) | 27.4 (19 %) |
| sea4_s0 (4-axis, disturbance-trained) | 5.2 (19 %) | 13.0 (20 %) | 12.7 (20 %) | 23.7 (27 %) | 23.2 (26 %) | 25.4 (31 %) |
| sea4_s1 (4-axis, disturbance-trained) | 5.3 (38 %) | 9.2 (37 %) | 9.0 (38 %) | 20.7 (45 %) | 20.5 (45 %) | 23.4 (46 %) |
| sea6_s0 (6-axis, disturbance-trained) | 3.6 (28 %) | 8.6 (28 %) | 8.3 (28 %) | 22.2 (37 %) | 21.6 (37 %) | 25.9 (39 %) |
| sea6_s1 (6-axis, disturbance-trained) | 4.4 (18 %) | 10.3 (17 %) | 10.3 (18 %) | 25.6 (33 %) | 24.9 (32 %) | 28.5 (35 %) |
| pwm10_s1 (PWM, still-water trained) | 3.3 (1 %) | 10.9 (3 %) | 10.6 (3 %) | 21.5 (8 %) | 21.3 (8 %) | 25.2 (11 %) |
| **seapwm_s0 (PWM, disturbance-trained)** | 3.8 (27 %) | **7.4 (29 %)** | **7.3 (27 %)** | **14.4 (34 %)** | **14.2 (34 %)** | **16.5 (40 %)** |
| seapwm_s1 (PWM, disturbance-trained) | 4.4 (38 %) | 6.3 (39 %) | 6.3 (39 %) | 14.6 (50 %) | 14.5 (50 %) | 17.4 (54 %) |
| PID derate 0.7 (baseline) | 22.0 | 27.9 | 27.4 | 44.6 | 43.4 | 47.0 |

How to read:
* **Waves dominate.** With current only (C/CD) every RL policy is at 9–13 cm; with waves (CW/CDW) they jump to 21–26 cm (PID 43–47 cm). A Tp 12 s wave in 4 m of water gives
  an orbital velocity of ≈0.5 m/s at the vehicle position [유도], which the authority of the 4-axis/6-axis policies (JS_GAIN 0.5, surge ≈38 N) cannot cancel — the 27–45 % saturation in the CW cells is the evidence.
* **Only PWM + disturbance training gets much better in waves**: 14.2–14.6 cm (p95 0.30 m) vs 21.3 cm for still-water-trained PWM and 20.5–25.6 cm for 4-axis/6-axis disturbance training. It is the result of combining the 3.4× authority (§10.3) with
  disturbance training, and the 34–50 % saturation means that authority is actually being used.
* **Current (C/CD)**: PWM disturbance training at 6.3–7.4 cm is best; 6-axis disturbance training 8.3–10.3; 4-axis disturbance training at 9.0–13.0 is not much different from still-water-trained 4-axis (11.9–12.3) —
  a 4-axis policy that does not observe the water velocity has a hard time noticing the current's DC offset (no integrator).
* **Drift (CD vs C, CDW vs CW)** makes no difference in either the 20 s training episodes or the 133 s evaluation (τ 300 s is nearly constant on this time scale).
* **Cost: disturbance training is aggressive even in still water.** Still-water cell saturation goes from traj10 0 % / pwm10 1 % → sea4 19–38 %, sea6 18–28 %, seapwm 27–38 %, and xy too goes 3.3–3.4 → 3.6–5.3 cm.
  Before hardware, retraining with an action penalty (per-thruster L1, rate) or with lowered pwm_cap/attitude_cap is needed.
* **6-axis (sea6) is better than 4-axis (sea4) but falls short of PWM**: current 8–10 vs 9–13 cm, waves 22–26 vs 21–24 cm. What pitch/roll gains is small (the mixer authority is unchanged),
  and it holds a tilt of 5.6–10 ° at all times.
* Seed spread: sea4's two seeds 3–4 cm, sea6 3–4 cm, seapwm 0.5 cm — PWM is the most consistent.


## 12. roll/pitch reference-tracking recipe `att7` (2026-10-03) — for station LOW `rl_7d`, **code only (training deferred)**

For the RL low-level controller to follow the roll/pitch of the 7-dim (`pos_rpy_width`) Diffusion Policy, the policy must have seen a **tilted q_des** during
training. Every policy before that (traj*, pwm10*, sea4/sea6/seapwm) had `q_des = quat_from_yaw(goal_yaw)` — it saw only level goals.
The axes6/pwm policies can **actuate** roll/pitch, but that is a "level out in the trim world" bias, not reference tracking
(mean pitch in level-world origin hold: sea6 −6.8°, pwm10_s1 −7.4°; trim world pwm10_s1 +0.7° — computed from `logs/eval/sweep_rl_*_{nominal,trim}/timeseries.csv`).
**RL_PWM, as is, cannot track the 7-dim policy's roll/pitch either.**

* **Goal generation** (`rov_rl/goals.py`, off by default): with `att_enabled`, a `p_att` (0.7) fraction of episodes get a smooth roll/pitch path
  (bias + harmonics, |roll| ≤ `att_roll_max` 0.17, |pitch| ≤ `att_pitch_max` 0.35 rad, Euler rate ≤ `att_rate_max` 0.35 rad/s,
  frequency `att_freq_range` 0.02–0.10 Hz, **starting from level** via the same smoothstep as the velocity, independent of position motion — station-keeping while tilted is the key point).
  The random numbers are drawn only **after** position/yaw have all been drawn, so (a) when off it is bit-identical to the existing recipe (new and old goals.py: tensors + RNG state identical in 12 combinations,
  `logs/smoke/att7_goals_byte_identity_20261003.txt`), (b) even when on, the position/yaw path for the same seed is the same tensor (`tests/test_att7_goals.py`).
  `set_external(roll=, pitch=)` gives level when they are omitted.
* **env**: `q_des = quat_from_euler_xyz(goal.roll, goal.pitch, goal.yaw)` (when off, `quat_from_yaw` as before), `set_goal(roll=, pitch=)`
  (allowed only in an att-training env). With att goals, `ang_error_mode` must be `full` and `rew_scale_tilt` must be 0 — otherwise ROVEnv
  **refuses at construction** (an absolute tilt term penalizes the commanded tilt). Logs `/att/ref_tilt_deg`, `/att/err_full_deg`. DR `cob_trim_x_range_m`
  (None = unused): draws the trim per episode so that closed-loop attitude control is learned instead of a single world's bias.
* **preset `traj10att`** (`configs/dr_presets.yaml`): the same DR and moving_goal as traj10 + att block + trim range [−0.035, 0.005] m
  (≈ −13 … +2° with the 0.15 m arm [유도]). **Pipeline** `scripts/_pipeline_att7_seed.sh <seed>`: axes6, `ang_error_mode=full`,
  `rew_scale_tilt=0`, `terminate_tilt_rad=1.05`, `tag_dropout_tilt_rad=0.61` (= station abort 35°), `--min_noise_std 0.05`, run name `att7_s<seed>`.
  **Not run** — to run: `for S in 0 1 2; do systemd-run --user --unit=rl-att7-s$S --collect -p WorkingDirectory=$PWD bash scripts/_pipeline_att7_seed.sh $S; done`.
* **export contract**: `obs_spec.json attitude_reference {trained, roll_max_rad, pitch_max_rad, rate_max_rad_s, p_att, freq_range_hz,
  starts_level, trim_x_range_m, q_des_convention, ned_mapping}`. If the key is missing or `trained: false`, it is **level-trained** — station `rl_7d` and
  `deploy/obs_builder.py set_goal(roll_ned=, pitch_ned=)` refuse rather than give a tilted goal to such an export (the layouts are the same, so
  this key is the only discriminator). Mapping roll_up = roll_ned, pitch_up = −pitch_ned, yaw_up = −yaw_ned.
* **Verification**: `tests/` 53/53 (new `test_att7_goals.py` 5), Isaac smoke 3 iterations (a code check, not training) — att7 axes6 training, the level recipe,
  the `ang_error_mode=yaw` refusal and export all passed (`logs/smoke/smoke_att7_20261003.log`; q_des actually tilted — ref tilt 3.7°).
* **Remaining**: the evaluation scripts (eval_sweep/eval_trajectory/summarize_results) have no roll/pitch reference steps or metrics; the training caps (0.5/0.75) and
  the station first-water caps (0.25/0.375 in policy units) differ — the station refuses `rl_7d` engage until the signs are proven; there is no pwm-mode att recipe and
  no q_des attitude feed for station `rl_pwm`. This repository is not git — pre-modification copy `_pre_att7_backup_20261003/`.

### 12.1 att7pwm — pwm mode of the same recipe, still water (2026-10-03, training and evaluation done)

`scripts/_pipeline_att7pwm_seed.sh <seed>`: traj10att (still water — no `sea:`, p_mode NONE; the tether-emulating external force of 0–4 N is the same as traj10) +
`env.action_mode=pwm`, ang_error_mode full, rew_scale_tilt 0, tumble 1.05 / tag dropout 0.61 rad, `--min_noise_std 0.05`. Seeds 0..2 were
run in parallel at 2026-10-03 15:44, 34:38–34:45 each [측정 `logs/train_att7pwm_s*.log`], and completed through the standard evaluation chain.

**Attitude-tracking evaluation** `scripts/eval_attitude.py` (+ `_eval_att.py` pure numpy, `_eval_att_batch.sh`, `compare_attitude.py`, `tests/test_eval_att.py`):
15 still-water programs (pitch ±10/±20°, roll ±5/±9.5°, combinations, triangle/sine, yaw and translation while tilted) × 3 trim worlds, with station interlock emulation.
Results [측정 `logs/eval/ATT_COMPARE_20261003.md`]: on tilted goals att7pwm is much better than the level-trained pwm10_s1 (world 0 pitch RMS 1.46 vs
10.93°) — but steady pitch error in the trim worlds up to 7–9°, roll wobble of ±3° even when level, vertical |throttle| ~0.7 (pwm10_s1 0.3), saturation 24–38 % (s0) /
57–83 % (s1) / ~100 % (s2): [예측] short of the promotion criteria. The best, s0, was copied to the station as a **disabled candidate** (`rov_gui/control/rl_policies/att7pwm_s0/`).
Candidates for the next attempt [예측]: a stronger per-thruster L1/action-magnitude penalty, a narrower trim range, more iterations.
