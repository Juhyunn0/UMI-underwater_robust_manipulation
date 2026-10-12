# RL_controller — "Learning to Swim" 재현: 우리 ROV(BlueROV2 Heavy + C3 + Newton gripper), Vision + IMU only

논문 *Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven AUVs* (Cai, Chang, Girdhar, ICRA 2025,
[arXiv:2410.00120](https://arxiv.org/abs/2410.00120), 코드 `warplab/isaac-auv-env`)의 파이프라인 — **Isaac Lab 병렬 시뮬레이션 + RSL-RL PPO +
domain randomization → zero-shot 실기 이식** — 을 우리 기체와 센서 구성에 맞게 다시 구현한 독립 패키지다.
이 폴더 밖의 코드는 import 하지 않는다 (모델·파라미터·추정기·명령 경로는 모두 여기로 복사/재작성).

근거 태그 규칙: `[측정]` 실측 산출물 경로 동반, `[스펙]` 데이터시트/코드 상수, `[유도]` 계산, `[예측]` 미검증 가정.

## 1. 논문과 무엇이 같고 무엇이 다른가

| 항목 | 논문 (CUREE, WarpAUV) | 이 구현 (우리 ROV) |
|---|---|---|
| 시뮬레이터 | Isaac Lab DirectRLEnv, 2048 envs, 120 Hz 물리 | 동일 (Isaac Sim 5.0 + Isaac Lab 2.2.0, `~/IsaacLab`, conda `env_isaaclab`) |
| 기체 | 22.7 kg, 추진기 6 | BlueROV2 **Heavy** 8×T200 + C3 카메라(상부 마운트 43° 하향) + Newton gripper, **13.72 kg** (`configs/rov_params.yaml`, 평행축 합성) |
| 동역학 | MuJoCo식 관성상자 항력 + 부력(COB 오프셋) + `C_t·Ω\|Ω\|` | 동일 항력 모델 + 공칭 선형 감쇠 + **부가질량 M_A·ν̇ + C_A(ν)ν** (Fossen; 2026-09-21, `rov_rl/hydrodynamics.py`) + **MarineGym T200 곡선**(전압 스케일·deadband·1차 지연·명령 지연) |
| 행동 | 추진기 6개 PWM 직접 | **두 모드** (`env.action_mode`, §10, 2026-09-27): `axes` = **MANUAL_CONTROL 4축** `[surge, sway(+우), heave(+상), yaw(+시계)]` → **ArduSub vectored-6DOF 믹서**(`rov_rl/mixer.py`) → 8 throttle (기본값, 09-27 이전 모든 체크포인트); `pwm` = **추진기 8개 throttle 직접**(논문과 같이 믹서 우회, roll/pitch 구동 가능). 믹서·게인·모터 부호는 실기 로그 64,512행으로 검증 [측정 `logs/mixer_fit_report.md`] |
| 센서 | DVL + IMU + AprilTag EKF (v_body 직접 관측) | **DVL 없음**: 태그 PnP(≈15 Hz, ≈130 ms 지연, 잡음) + ArduSub ATTITUDE/자이로를 시뮬레이션하고, station 의 `state_assembler` 와 **같은 추정기**(유한차분+LPF α=0.6, 자이로/속도 브리징)로 v_body 를 만든다 (`rov_rl/sensors.py`, `rov_rl/estimator.py`) |
| 관측 | 17차원 `[q_des, x_off_b, q, v_b, ω_b]` | 17 + 목표 속도/yaw rate 4 (8장) + 직전 행동 4 = 25 를 **4프레임 이력** → 100차원 (지연 보상). 사원수는 w≥0 으로 정규화 (7.3). 2026-09-16 런은 84차원 |
| 보상 | `0.2e^{-\|x\|²} + 0.5e^{-angle} + 0.2e^{-\|a\|²}` | 동일 (`rov_rl/rewards.py`, yaw-only 옵션) |
| 제어 주기 | 60 Hz 학습 / 20 Hz 배포 | **20 Hz** 학습=배포 (physics 1/120 s, decimation 6), 에피소드 20 s (2026-09-16 런은 6 s) |
| DR | COB 오프셋 구, 부피 | `none/small/large`(논문 대응) + `ours`(질량·관성·항력·전압·추진기별 gain·지연·센서·임펄스 외란) `configs/dr_presets.yaml` |
| 목표 자세 | 임의 단위 사원수 | yaw 균등(roll/pitch 수평). `goal_max_roll_pitch`/`full_random_orientation` cfg 는 **읽는 코드가 없는 죽은 노브**(2026-10-03 확인). roll/pitch 기준은 §12 `att_*`(preset `traj10att`) |
| PPO | RSL-RL, 64×64 ELU, 24 steps, lr 5e-4 adaptive | 동일 (`rov_rl/agents/rsl_rl_ppo_cfg.py`) |
| 배포 | ROS/Jetson, 추진기 직접 | **numpy MLP**(`deploy/policy_numpy.py`) + `deploy/obs_builder.py`(추정기 numpy 쌍둥이, NED/FRD→z-up/FLU 변환) + `deploy/ardusub_bridge.py`(MANUAL_CONTROL 20 Hz, deadman, ARMED/MANUAL/JS_GAIN 게이트) |

실기 로그에서 얻어 시뮬에 반영한 값 두 가지:
* **믹서/게인** — `scripts/fit_mixer_from_logs.py`: ArduSub 이론 팩터와 완전히 일치, 유효 pilot gain 0.49–0.50 (= `JS_GAIN_DEFAULT` 0.5), 모터 부호 `[-1,-1,+1,+1,-1,+1,+1,-1]`.
* **정적 pitch 강성** — 실기 pitch ≈ 4.0°/unit surge (181,097행). CAD 합성 CB-COM 팔 7 mm 로는 surge 에 뒤집히므로 유효 팔 **0.15 m** 를 공칭으로 두고 DR 0.5–1.5배 [유도, `configs/rov_params.yaml` 주석].

## 2. 설치 / 환경

* 학습·평가: conda `env_isaaclab` (Isaac Sim 5.0 pip, Isaac Lab v2.2.0 editable, rsl-rl-lib 2.3.3, torch 2.7 cu128). 드라이버 570, `amd_iommu=off` — 설치 기록은 `../umi_underwater_robust_control/ISAAC_AUV_SETUP_RUNBOOK.md`.
* 배포·리플레이: conda `robust` (numpy, pymavlink; torch 불필요).
* 설치 명령 없음 — 스크립트가 `sys.path` 에 프로젝트 루트를 넣는다. `export OMNI_KIT_ACCEPT_EULA=YES` 필요.

```bash
conda activate env_isaaclab && cd ~/Desktop/RL_controller
python -m pytest -q tests                # Isaac 앱 없이 18개 (물리·믹서·센서·추정기·배포 parity)
bash scripts/build_asset.sh              # URDF -> assets/rov/rov.usd (rov_params.yaml 을 바꾸면 다시)
```

## 3. 학습

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 600 --headless --dr small --run_name small
# DR 프리셋: none | small | large | ours      로그: logs/rsl_rl/rov_posehold/<시각>_<run_name>/
```
RTX 5090 기준 2048 envs × 600 iter ≈ 3.5 분 (이터당 ≈0.35 s). 논문의 "10–20분" 보다 빠른 것은 이력 관측 외에 연산이 작고 렌더가 없기 때문.
Hydra 덮어쓰기 가능: `env.rew_scale_actions=0.1 agent.policy.actor_hidden_dims=[128,128]` 등.
실용(권장) 설정: `--dr ours env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05 agent.algorithm.entropy_coef=0.005 --max_iterations 800` (7.2).

## 4. 평가 (논문 그림/표 대응)

```bash
python scripts/plot_training.py logs/rsl_rl/rov_posehold/*_none logs/rsl_rl/rov_posehold/*_small ... -o logs/eval/training_curves.png   # Fig.3
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --tag rl_nominal                          # Table III (8방향 step)
python scripts/eval_sweep.py --checkpoint <run>/model_599.pt --headless --shift "volume_l=-0.6,cob_x=0.02,drag=1.5,voltage=0.65" --tag rl_shift
python scripts/eval_sweep.py --controller pid --pid_derate 0.7 --headless --tag pid_nominal                       # PID 기준선 (derate 0.7 이 시뮬 최적)
python scripts/eval_disturbance.py --checkpoint <run>/model_599.pt --headless                                     # Fig.5–7 (임펄스, RL vs PID)
python scripts/play.py --task ROV-PoseHold-v0 --num_envs 16 --dr small --checkpoint <run>/model_599.pt            # 시각화 (GUI; --headless --max_steps N 으로 무화면 점검)
python scripts/play_control.py --checkpoint <run>/model_599.pt --dr small                                         # 키보드로 목표 조작 (논문 play_poshold 대응, GUI)
```
결과: `logs/eval/<name>/{timeseries.csv, metrics.csv, *.png}`, 요약표 `python scripts/summarize_results.py` → `logs/eval/RESULTS.md`.
`--shift` 는 논문 IV-B 의 "변형 동역학" 프로토콜(변형량 = small DR 의 경계): 부피 −0.6 L(−5.9 N), CB 전방 2 cm, 항력 ×1.5, 전압 0.65.
(−1.2 L 이상이면 gain 0.5·전압 0.6 에서 heave 권한(≈14 N)이 부족해 두 제어기 모두 바닥에 닿는다 — 시험이 아니라 권한 한계.)
PID 게인은 station 의 `rov_gui/control/pid.py` SIM_GAINS 이식(`rov_rl/baselines/pid.py`).

## 5. Export 와 배포

```bash
python scripts/export_policy.py --run logs/rsl_rl/rov_posehold/<run>        # -> <run>/exported/{policy_weights.npz, policy.onnx, obs_spec.json}
conda activate robust
python deploy/run_policy.py --export <run>/exported --replay ../umi_underwater_robust_control/data/2026.../mpc_*.csv --dry-run --log logs/eval/replay.csv
python deploy/run_policy.py --export <run>/exported --fix-udp 17000 --mav udpin:0.0.0.0:14552 --goal 0,0,-0.7 --goal-yaw 0 --axis-cap 0.5
```
* 관측 빌더 입력 = station 이 이미 만드는 것: `NavFix(p_ned, yaw_ned|R_ned_body, t_capture)` + ArduSub `ATTITUDE(roll,pitch,p,q,r)`.
  UDP 로 fix 를 JSON 한 줄씩 보내면 된다 (`{"t_capture":…, "p_ned":[…], "yaw_ned":…}`, monotonic 초). 브리지가 ATTITUDE 를 직접 파싱한다.
* 안전 게이트: ARMED + MANUAL 모드 + `JS_GAIN_DEFAULT` == 학습 게인(0.5) + ATTITUDE 신선 + 태그 fix 0.5 s 이내 — 아니면 NEUTRAL. deadman 500 ms. `--axis-cap 0.5` 부터 시작.
* 프레임: 정책은 z-up/FLU 로 학습됐고 `obs_builder` 가 NED/FRD 를 `C=diag(1,-1,-1)` 로 변환한다 (`tests/test_deploy_parity.py` 가 torch 파이프라인과 1e-5 일치 검증).

### 5.1 출력 인터페이스 = station MPC 의 wrench (2026-09-22)

RL 제어기의 공식 출력은 MPC 와 동일한 6-DOF 체 wrench **u = [X, Y, Z, K, M, N]** (NED/FRD 체 프레임, N·N·m) 이다. 네트워크는 여전히
MANUAL_CONTROL 4축(−1..1)을 내고, 그 위에 고정 선형 헤드가 붙는다 (`deploy/wrench.py`; station `allocation.py` 의 `axes_to_wrench` /
`wrench_to_axes` 와 같은 식 [코드]): X = a0·60, Y = a1·60, Z = −a2·60 (NED Z 는 아래 양), K = M = 0, N = a3·20
[스펙 hw_mpc.yaml axis_gain; `configs/rov_params.yaml: interface`]. roll/pitch 는 MANUAL_CONTROL 경로가 구동하지 않으므로 MPC 도 K, M 을
버린다 — 두 제어기의 실효 구동 부분공간은 같다.
* `export_policy.py` → `policy_wrench.onnx` (입력 obs → 출력 wrench (1,6)) 와 `obs_spec.json: output` (순서·단위·actuated·axis_gain).
  파리티: torch 헤드 vs numpy 1.8e-5, onnxruntime 1.0e-5 [측정].
* `deploy/run_policy.py` 로그와 `eval_trajectory.py` timeseries 에 station `mpc_*.csv` 와 같은 열 이름 `uX..uN` 추가. 스테이션의 `HwRl` 은
  처음부터 이 wrench 를 워커에 돌려주고 같은 열에 기록한다 (0921_182926: heave −0.162 → uZ 9.73 N [측정]).
* `ArduSubBridge.send_wrench(u)`, `rov_rl/mixer.py: axes_to_wrench / wrench_to_axes` (torch) 로 시뮬·실기 어느 쪽에서든 wrench 로 구동 가능.
* 논문은 추진기 6개 명령을 직접 냈다(믹싱까지 학습, roll/pitch 구동). 우리는 stock ArduSub MANUAL 이라 비행 중 개별 모터 명령이 없어
  4축을 유지하고(사용자 결정 2026-09-22), 인터페이스만 wrench 로 통일했다. **2026-09-27 부터는 추진기 직접 모드(`pwm`)도 있다(§10)** —
  시뮬·학습·export·`deploy/`(RC passthrough 전송)까지 갖췄고, 실기 쪽 전송(SERVOn_FUNCTION 변경)은 사용자 결정 사항으로 남아 있다.

## 6. 구조

```
configs/rov_params.yaml   기체 물성·추진기·믹서·센서·수조 (단일 진실원)      configs/dr_presets.yaml  DR 프리셋
configs/mixer_fitted.yaml 실기 로그 피팅 결과                                 logs/mixer_fit_report.md 피팅 리포트
rov_rl/params.py          평행축 합성, CB, 추진기 pose (composite-COM frame)  rov_rl/utils/math.py     사원수 유틸(isaaclab.utils.math 이식)
rov_rl/thrusters.py       T200 곡선·지연·1차 지연                              rov_rl/mixer.py          ArduSub 믹서·추력 부호·배분 B
rov_rl/hydrodynamics.py   논문 항력·부력 (+Fossen)                              rov_rl/sensors.py        태그/IMU 시뮬
rov_rl/estimator.py       state_assembler 이식                                  rov_rl/rewards.py        논문 보상
rov_rl/env/rov_env*.py    Isaac Lab DirectRLEnv + cfg                           rov_rl/agents/           PPO cfg
rov_rl/baselines/pid.py   PID 기준선                                            assets/rov/              URDF, 메쉬, USD(생성)
scripts/                  train/play/eval_*/plot_training/export/fit_mixer/build_asset
deploy/                   policy_numpy, obs_builder, ardusub_bridge, run_policy  tests/                   pytest
```

### 6.1 페이로드 시각 메쉬 — C3 카메라(상부 마운트) + Newton gripper (2026-09-21)

`assets/rov/rov.usd` 는 헐(`rov_body_*.obj`) 외에 실기가 싣고 있는 두 페이로드를 시각 메쉬로 포함한다. 생성기는
`scripts/bake_payload_meshes.py` (`robust` env: trimesh 4.12), 이후 `bash scripts/build_asset.sh` 로 USD 재생성.
- 출처: `umi_underwater_robust_control/assets/CAD files/onshape_export_20260909/` (onshape-to-robot 1.8.2, Onshape 문서 "BlueROV2" /
  어셈블리 탭 "mujoco", 2026-09-09 11:42 −0700 스냅샷). `robot.xml` 의 `<geom pos quat mesh>` 가 어셈블리 프레임 부품 pose
  (MuJoCo quat = w,x,y,z; 모든 `<body>` 는 identity), `assets/*.stl` 이 부품 메쉬(m). 질량은 placeholder 이므로 pose/메쉬만 사용 [스펙 export README].
- 등록 (어셈블리 → base_link "bl", FLU, 원점 = heavy COM): `bluerov2_mujoco_marinegym/tools/process_c3_mesh.py` 에 고정된
  `R0 = [[0,0,1],[1,0,0],[0,1,0]]`, `C_ASM = [-0.16079, 0.22043, -0.11281]`, `p_bl = R0 @ (p_asm − C_ASM)` [유도 — 상수 복사, import 없음].
- 검증 (`payload_frames_20260909.json` 대비, 스크립트가 assert): 13개 부품 bbox 최대 편차 0.05 mm (허용 2 mm);
  `c3_centroid_bl` / `gripper_housing_centroid_bl` / `gripper_jaw_centroid_bl` 편차 0.01 mm (json 의 centroid = vertex mean);
  C3 광축 = bl `[0.766, 0, −0.643]` → 40.00° 하향 (json `tilt_down_deg` 40); jaw x 범위 `[0.3289, 0.4039]` m [유도 bake 로그].
- 메쉬: `c3_payload.obj` = c3_br_june2025 + part_1 + part_3 (39,703 faces, URDF 색 0.23 0.38 0.71),
  `newton_gripper.obj` = 하우징·플랜지 캡·피스톤·베이스·푸셔·조 2 + 마운트 2 + part_1__2 (65,836 faces, 0.15 0.15 0.15).
  나사·오링·wlp 케이블 관통부·`brov2_heavy_asm_*` 헐 부품은 제외. 150k faces 이하라 decimation 없음.
- 배치: bl 프레임 = 헐 메쉬의 heavy-COM 프레임 (`rov_params.yaml` 이 `c3_centroid_bl` 을 그대로 heavy 프레임 pos 로 씀) 이므로
  URDF `<visual>` origin 은 헐과 동일한 `−COM_total = [-0.02983, -0.00388, -0.01276]`. 일관성 확인: 헐 메쉬 union bbox vs
  `reanalysis_20260909.json hull_bbox_bl` 중심 편차 ≤ 0.6 mm, 크기비 1.023 (MarineGym 스킨이 CAD 보다 2.3 % 큼, `process_c3_mesh.py` 주석), 모서리 최대 6.9 mm [유도].
- 관성은 이미 `rov_params.yaml` 의 `payloads:` 항목으로 합성돼 있어 URDF `<inertial>` (13.724 kg) 과 충돌 박스는 변경 없음 — 시각 전용.
  시각 검사 `assets/rov/render_check.png` (정면/우측/상면 정사영: 카메라가 전방 상부에 40° 하향, gripper 가 전방 하부 +x, 조 끝 x ≈ 0.40 m).
  Isaac Lab UrdfConverter 가 `assets/rov/.asset_hash` (= MD5(변환 설정 + URDF 바이트)) 를 빌드마다 다시 쓴다; 코드가 읽는 곳은 없다.

## 7. 결과 (2026-09-16, RTX 5090, 2048 envs) — 전체 표는 `logs/eval/RESULTS.md`

학습: 각 프리셋 600 iter ≈ 3.5 분 [측정 `logs/train_*.log`], 실용 설정 800 iter ≈ 5 분. 보상 곡선 `logs/eval/training_curves_v2.png` (논문 Fig.3 대응).
평가: 8방향 step sweep (논문 Table III 대응, MSE = 전체 구간 / 정상상태 2 s) 와 임펄스 외란 시험 (논문 Fig.6–7 대응, RL vs PID derate 0.7).
"변형 동역학" = small DR 경계 (부피 −0.6 L, CB 전방 2 cm, 항력 ×1.5, 전압 0.65). 모든 수치는 `logs/eval/<run>/metrics.csv` [측정].

### 7.1 논문 재현 (논문 보상 그대로, 관측 사원수 정규화 적용 = v2)

| run (DR) | 이상 환경 pos MSE | 이상 환경 ang MSE | 변형 pos MSE | 변형 ang MSE |
|---|---|---|---|---|
| none2 (DR 없음) | 0.231 / 0.238 | 0.026 / 0.003 | 0.255 / 0.274 | 0.049 / 0.019 |
| small2 | 0.071 / 0.041 | 0.028 / 0.005 | 0.086 / 0.061 | 0.049 / 0.018 |
| large2 | 0.065 / 0.037 | 0.028 / 0.004 | 0.076 / 0.049 | 0.048 / 0.020 |
| ours2 | 0.077 / 0.052 | 0.026 / 0.002 | 0.242 / 0.267 | 0.052 / 0.019 |
| 논문 CUREE (12방향) | 0.058 (no DR) / 0.056 (small) | 0.013 / 0.011 | 0.108 (no DR) / 0.081 (small) | 0.401 / 0.195 |

* 논문과 같은 경향: DR 을 줄수록 변형 동역학에서 위치 MSE 가 좋아진다 (none 0.255 → small 0.086 → large 0.076; 논문 0.108 → 0.081).
* 논문 보상(위치 항 σ = 1 m)은 정상상태 정밀도가 20–25 cm RMS 수준에서 멈춘다 — 논문 자신의 0.056 MSE 도 RMS 24 cm 이다. none2/ours2 의 큰 위치 MSE 는 이 평탄한 보상 위에서 정책이 목표에서 ~0.5 m 떨어진 곳에 정착한 결과(물리·추정기 문제 아님, `scripts/diag_policy_hover.py` 로 확인).
* 각도 MSE 가 논문보다 큰 것은 4축이라 roll/pitch 를 제어하지 않기 때문 (정상상태 yaw 오차 자체는 2–4°).

### 7.2 실용 설정 (실기 pool 시험용 권장)

`--dr ours` + `env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05` (+ `agent.algorithm.entropy_coef=0.005`).

| run | 이상 환경 pos MSE | 이상 환경 ang MSE | 변형 pos MSE | 변형 ang MSE | 외란 pos RMS 이상/변형 [m] (PID) |
|---|---|---|---|---|---|
| **prac** (v2 관측, entropy 0) — **배포 권장** | 0.023 / 0.003 | **0.021 / 0.001** | 0.034 / 0.011 | **0.037 / 0.017** | 0.083 / 0.114 (PID 0.137 / 0.083) |
| prac2 (v2 관측 + entropy 0.005) — **사용 금지: yaw 채터링 79 %** | 0.020 / 0.003 | 0.032 / 0.011 | 0.020 / 0.002 | 0.047 / 0.025 | 0.066 / 0.057 |
| prac3 (prac2 + action-rate 0.3) — **사용 금지: 채터링 66 %** | 0.021 / 0.004 | 0.038 / 0.019 | 0.025 / 0.006 | 0.055 / 0.034 | 0.082 / 0.086 |
| tuned3 (구 관측 + entropy) | 0.023 / 0.004 | 0.029 / 0.014 | 0.027 / 0.006 | 0.044 / 0.029 | 0.090 / 0.092 |
| PID derate 0.7 (기준선) | 0.029 / 0.003 | 0.023 / 0.002 | 0.042 / 0.015 | 0.042 / 0.018 | — |

* prac: 정상상태 위치 RMS ≈ 5.7 cm, yaw ≈ 2°, 명령 포화 0 %, 평균 |a| 0.08; 이상 환경 외란 억제는 PID 보다 좋고(논문 Fig.6–7 의 결론과 같음) 변형 동역학에서는 PID 가 약간 낫다.
* prac2 는 지표는 가장 좋지만 정상상태의 76 % 스텝에서 yaw 명령이 ±1 로 포화되고 매 스텝 부호가 뒤집히는 bang-bang 정책이다 (`logs/eval/RESULTS.md` 포화 열). 지표만 보면 놓치는 실패 모드 — **정책 채택 전 반드시 포화 비율·부호반전 열을 확인**할 것. entropy 를 켠 v2 런은 action-rate 페널티를 0.3 으로 올려도(prac3) 채터링이 남았다 — 이 관측 구조(직전 행동 되먹임)에서는 entropy 보너스를 쓰지 않는 것이 안전하다.
* 배포 체크포인트: `logs/rsl_rl/rov_posehold/*_prac/exported/`.

### 7.3 이 과정에서 찾은 두 가지 (학습 곡선에는 안 보이는 문제)
1. **사원수 이중 표현**: 목표 yaw 를 [0, 2π) 로 뽑으면 절반은 w<0 인 사원수가 되는데 추정 사원수는 항상 w≥0 이라, 정책이 q 와 −q 를 같은 회전으로 배우지 못해 목표 yaw 0 근처에서 +0.2–0.3 rad 편향이 생겼다 (`deploy/policy_numpy.py` 로 합성 관측을 넣어 확인). `obs_quat_unique=True` 로 w≥0 정규화 → 각도 MSE 0.10 → 0.026.
2. **탐색 붕괴 vs 추진기 데드밴드**: entropy 0 이면 PPO 노이즈 std 가 0.02 까지 줄어 T200 데드밴드(gain 0.5 에서 |axis| < 0.15 는 추력 0) 안에서만 샘플링 → 미세 보정을 배울 gradient 가 없다. `entropy_coef=0.005` 로 std ≈ 0.19 유지 (tuned3) → yaw RMS 0.24 → 0.06 rad. 단, entropy 를 켜면 bang-bang 채터링(prac2)이 나올 수 있으므로 action-rate 페널티를 같이 키워야 한다.
3. **open-loop 리플레이의 한계**: `run_policy.py --replay` 는 기록된 로그가 정책 명령에 반응하지 않으므로, 직전 행동 입력을 통한 되먹임이 잠겨 yaw 명령이 포화될 수 있다(자이로 열도 없음). 파이프라인 점검용일 뿐 성능 판단에는 쓰지 말 것; 성능은 closed-loop 시뮬 평가로 본다.

## 8. 궤적 추종 확장 (2026-09-21)

위치 유지만 배운 정책은 움직이는 목표에 속도에 비례해 뒤처진다(피드포워드 없음). 그래서 학습·관측·평가·배포를 궤적 추종용으로 확장했다.
제어기 구조(행동·믹서·추정기·브리지)는 그대로다.

| 항목 | 내용 |
|---|---|
| 목표 이동 | `rov_rl/goals.py`: 에피소드마다 정지(30 %) / 직선·곡선(속도 0.02–0.25 m/s, 회전율 ±0.6 rad/s, 수직 ±0.05 m/s), heading-follow 50 %, 수조 벽에서 반사 |
| 관측 | 17 + **목표 속도(몸통 기준 3) + 목표 yaw rate(1)** + 직전 행동 4 = 25 × 4프레임 = 100 (`obs_goal_velocity`, 구 체크포인트는 자동으로 84) |
| 에피소드 | 6 s → **20 s** |
| 항력 DR (`traj` 프리셋) | 항력 ×0.7–3.0 + 선형 항력 추가 (30, 45, 10) N·s/m × 0–1.5 [유도: station 속도법칙 `speed = 0.692(\|axis\|−0.096)` → 실기 항력이 모델의 2–4배] |
| 평가 | `scripts/eval_trajectory.py`: 사각형 1 m @0.15 m/s, 원 r 0.5 @0.15 m/s, lawnmower — RL vs PID, cross-track p95·위치 RMS·yaw RMS·포화 비율, `logs/eval/traj_<tag>/path.png` |
| 배포 | `deploy/run_policy.py --traj square:1.0,0.15 --laps 2` (시작 자세 기준 프레임에서 정의, NED 변환 자동) — `deploy/trajectory.py` |

```bash
python scripts/train.py --task ROV-PoseHold-v0 --num_envs 2048 --max_iterations 1000 --headless --dr traj --run_name traj \
    env.rew_pos_sigma=0.3 env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05
python scripts/eval_trajectory.py --checkpoint logs/rsl_rl/rov_posehold/*_traj/model_999.pt --traj square:1.0,0.15 --laps 2 --headless
```
세션과 무관하게 돌리려면 `setsid nohup bash scripts/_pipeline_traj.sh > logs/pipeline_traj_outer.log 2>&1 < /dev/null &`.

### 8.0 4차 반복 (traj4, 2026-09-21 저녁): 깊이 문제 대응
* **축별 분리 보상** (`env.rew_separable=True`): 수평 위치·속도와 깊이 위치·속도를 따로 보상 (σ_xy 0.3 m, σ_z 0.15 m; 속도 σ 0.15 / 0.10 m/s). 3-D 노름 하나로는 수평 오차가 남아 있는 동안 z 의 gradient 가 희석된다 (`tests/test_hydro.py::test_separable_rewards_keep_z_gradient`).
* **공칭 선형 감쇠** (`configs/rov_params.yaml hydro.linear_damping_nominal` = 20/30/15 N·s/m [유도 station 속도법칙]): 논문 모델은 2차 항력뿐이라 저속 감쇠가 0 에 가까워 heave 한계진동을 만든다. 이제 공칭·평가에도 들어가고 DR 은 0.5–1.5 배 (`traj4` 프리셋).
* **비대칭 actor-critic** (`env.critic_privileged=True`): critic 은 정책 관측(100) + 참 상태·DR 파라미터(31) 를 본다. actor(배포) 는 그대로.
* **시드 3개** (`--seed 0/1/2`, 1200 iter) 와 `scripts/select_best.py` 로 최악 지표 기준 선택.

### 8.0b 5차 반복 (traj5): 부가질량 (Fossen 완성)
지금까지 Fossen 방정식 중 강체·복원력·감쇠 항만 있었고 **부가질량 M_A·ν̇ 과 C_A(ν)ν 가 없었다**. BlueROV2 는 heave 부가질량 14.6 kg, sway 12.7 kg 으로
기체 질량(13.7)과 같은 급이라 유효 관성이 2배 차이난다. 구현(`rov_rl/hydrodynamics.py AddedMass`): 서브스텝마다 몸통 가속도를 저역통과(α 0.3)로
추정해 −M_A·ν̇ 을 외력으로, C_A(ν)ν 를 Fossen 대각식으로 가한다 (uuv_simulator / MarineGym 방식; 명시적 지연이지만 α·M_A/m < 1 이라 안정,
`tests/test_hydro.py::test_added_mass_effective_inertia_and_coriolis`). 계수 MarineGym BlueROVHeavy (Wu 2018) [스펙], DR ×0.5–1.5 (`traj5` 프리셋).
공칭 물리에 포함되므로 `--dr none` 평가에도 들어간다 (그 전 결과는 부가질량 없는 세계의 값). traj4_s2 를 새 물리에서 재평가한 결과는 sim-to-sim 강건성 지표.

### 8.1 결과 (사각형 1 m / 원 r 0.5 m @0.15 m/s, heading-follow 코너 슬루 60°/s, 2바퀴; 전체 표 `logs/eval/RESULTS.md`)

| 정책 | 학습 설정 | 사각형 xy RMS / z RMS [m] | 사각형 along RMS | 원 xy RMS / along RMS | 원 yaw RMS [rad] | 포화 |
|---|---|---|---|---|---|---|
| prac (위치 유지 학습, 피드포워드 없음) | ours DR, 정지 목표, 6 s | 0.139 / 0.034 | 0.133 | 0.152 / 0.143 | 0.132 | 4 % |
| traj | traj DR(무거운 항력), 이동 목표, 20 s | 0.114 / 0.080 | 0.092 | 0.105 / 0.072 | 0.197 | 20 % |
| traj2 | traj2 DR, 이동 목표, 속도 보상 | **0.085 / 0.098** | **0.072** | **0.061 / 0.021** | **0.123** | 1 % |
| traj3 | traj2 + 좁힌 항력 DR, 정깊이 목표, rate 0.1 | 0.154 / 0.261 | 0.121 | 0.101 / 0.054 | 0.095 | 19 % |
| **traj4_s2** (8.0 설정, 시드 2) — **배포 권장** | 분리 보상 + 공칭 선형 감쇠 + privileged critic | **0.059 / 0.032** | 0.052 | **0.058 / 0.035**, 뒤처짐 0.040 | 0.069 | 3 % |
| traj4_s0 / traj4_s1 (시드 편차) | 〃 | 0.063 / 0.032, 0.065 / 0.036 | 0.050, 0.049 | 0.066 / 0.028, 0.034 / 0.021 | 0.129, 0.063 | 5 %, 9 % |
| **traj5_s2** (8.0b, 부가질량 포함) — **최종 배포 권장** | traj4 + 부가질량 M_A·ν̇ + C_A(ν)ν, DR ×0.5–1.5 | **0.053 / 0.020** | 0.047 | **0.057 / 0.021**, 뒤처짐 0.044 | 0.043 | 2 % |
| traj5_s0 / traj5_s1 | 〃 | 0.054 / 0.023, 0.053 / 0.027 | 0.040, 0.042 | 0.062 / 0.015, 0.063 / 0.020 | 0.089, 0.059 | 17 %(코너 yaw), 4 % |
| traj4_s2 를 부가질량 세계에서 재평가 (sim-to-sim) | 부가질량 없이 학습 | 0.065 / 0.054 | 0.056 | 0.058 / 0.051 | 0.076 | 2 % |
| PID derate 0.7 | 기준선 (피드포워드 없음) | 0.106 / 0.022 | 0.085 | 0.076 / 0.072 | 0.232 | 0 % |

* 피드포워드의 효과: 진행 방향 뒤처짐(along-track)이 prac 0.143 → traj2 **0.021 m**(원). 수평 추종은 traj2 가 PID 보다 좋다 (사각형 xy 0.085 vs 0.106, 원 0.061 vs 0.076).
* 남은 문제: 이동 목표로 학습한 정책들의 **깊이** 유지 (traj2 z RMS 0.098, 정지 시에도 ±0.1 m heave 진동, 코너에서 −0.3 m; prac 0.034, PID 0.022). 항력 DR 을 좁히고 깊이 목표를 고정한 traj3 은 오히려 더 나빠져(z RMS 0.26) 원인이 항력 DR 폭만은 아니다 — 20 s 에피소드 + 이동 목표 학습이 heave 루프를 느슨하게 만드는 것으로 보이며, 미해결.
* 변형 동역학(항력 ×1.5 등)에서는 RL 이 PID 보다 낫다 (traj 사각형 0.094 vs 0.145 m) — 실기 항력이 모델보다 크다는 station 기록과 같은 방향.

**4차 반복(traj4) 결과 — 깊이 문제 해결.** 분리 보상 + 공칭 선형 감쇠 + privileged critic 으로 학습한 3개 시드 모두 깊이 RMS 0.03 m 대로 내려왔고
(traj2 0.098), 수평 추종·외란 억제도 전 항목에서 PID 보다 낫다. 외란 시험 pos RMS(3-D): 0.055–0.070 m (PID 0.08). 시드 간 편차는 작다
(`scripts/select_best.py` 최악지표 점수 0.91 / 1.08 / 1.79; prac 1.79, traj2 1.96).

**5차 반복(traj5, 부가질량 포함) 결과.** 3개 시드 모두 traj4 보다 좋고 편차가 작다 (최악지표 점수 0.85 / 0.91; s0 는 코너 yaw 포화로 3.35 이나
직선 구간 포화 1 %, 부호반전 0.11/스텝 — 채터링 아님). 부가질량 없이 학습한 traj4_s2 를 부가질량 세계에서 돌리면 깊이 RMS 가 0.032 → 0.054 m 로
나빠지므로(sim-to-sim), 부가질량은 실기 이식에서도 중요한 항이다. 외란 시험 pos RMS(3-D) traj5: 0.050–0.060 m (PID 0.08).

**최종 권장**: 전 축 RL — **traj5_s2** (`logs/rsl_rl/rov_posehold/*_traj5_s2/exported/`; 위치 유지·궤적 추종·외란 모두). 실기 첫 시험은 `--axis-cap 0.5`,
정지 → 사각형 1 m @0.10 m/s → 0.15 m/s 순서. 관측 스펙(`obs_spec.json`)에 피드포워드 포함 여부가 기록되어 있어 `run_policy.py` 가 자동으로 맞춘다.

### 8.2 실기 첫 시험 후 (2026-09-21 밤): 기준 궤적을 기체 시작점에서 출발시키기 (goals.py v2, 프리셋 traj6 — **아직 미학습**)

실기 첫 시험(`umi_underwater_robust_control/data/20260921/0921_182926`, LOW=rl, traj5_s2, axis_cap 0.5)은 사각형 1 m @0.10 m/s 5바퀴를
xy RMS 0.042 m [측정]로 완주했지만, 접근 구간에서 최대 0.45 m/s [측정]까지 가속해 너무 빨랐다. 원인은 학습 시작 조건: 매 에피소드 목표가
기체에서 최대 1 m 떨어진 곳에 찍혀 정책이 "오차 0.2 m 이상이면 풀스로틀"(정지 응답 0.1 m → 0.52, 0.2 m → 1.0 [측정 exported traj5_s2])을 배웠다.
사용자 결정: 기준 궤적을 **기체의 시작 위치·헤딩에서 출발**하는 랜덤 곡선으로 바꾸고, 그 위에서 학습한다.

**바뀐 것 (`rov_rl/goals.py` v2, 구현만, 학습은 아직).** 에피소드 리셋 때 20 s 기준 경로 전체를 한 번에 샘플한다 (스텝 인덱스로 참조; 2048 envs 기준
리셋 1.6 ms, 스텝 0.14 ms, 버퍼 10 MB [측정]).
* 시작: 위치 = 기체 스폰 위치, 헤딩 = 기체 yaw, 속도 0 → smoothstep 램프 1–3 s 로 순항 속도 U[0.02, 0.25] m/s 도달 (±30 % 느린 변조).
* 곡률: 회전율 |ω| ≤ 0.6 rad/s = 상수 편향 + 랜덤 사인 3개(0.02–0.15 Hz) → 원·호·S-커브. 수직 속도 |vz| ≤ 0.03 m/s 같은 방식.
* 헤딩: 50 % 는 진행 방향 추종(station heading_follow), 나머지는 독립 yaw 드리프트 |r| ≤ 0.4 rad/s. 30 % 는 후반에 감속해 정지 유지, 30 % 는 정지 목표(위치 유지).
* 수조 상자(±0.9 × ±2.0 m, 고도 0.4–0.8 m)는 **벽 조향**으로 지킨다: 헤딩을 0.25 s 간격 인과 루프로 적분하면서 벽 0.6 m 이내에서 안쪽으로 도는
  항을 섞는다 (|ω| ≤ 0.6 rad/s 유지, 정지 중엔 조향 없음). 처음 구현한 거울 접기(v1 과 같은 벽 반사)는 0.25 m/s 경로의 45 % 에서 헤딩 점프
  중앙값 111° 를 만들어(검토 지적) 버렸다. 조향 후 벽에 닿는 경로는 5 % (모두 순항 > 0.18 m/s, 위치 클램프로 벽을 따라 미끄러짐; yaw 참조는
  연속, 스텝당 ≤ 1.7°) [측정 N=4000]. 리셋 비용 2048 envs 15 ms [측정].
* yaw 참조도 정지에서 출발한다: 회전율·yaw-rate 모두 속도 램프와 같은 포락선을 곱해 t=0 의 yaw-rate 피드포워드 = 0 (실기 시작 조건과 일치),
  heading-follow 경로는 위치가 멈추면 회전도 멈춘다. 독립 yaw 드리프트는 정지 유지 중에도 계속될 수 있다 (station 의 정지 중 헤딩 변경과 같은 과제).
* 평가 스크립트(에피소드 1e6 s)에서는 경로를 만들지 않는다 (`eval_mode` → T=2; 검토가 잡은 env당 640 MB 버퍼/OOM 문제). 학습 외 긴 에피소드는
  `horizon_max_s` 120 s 로 캡.
* 피드포워드 = 경로의 전방 차분 (다음 스텝 동안 목표가 실제로 움직이는 양). 관측 레이아웃·`obs_spec` 은 그대로.
* 리셋(`rov_env.py`): `moving_goal.start_at_robot` (traj6: true) 이면 기체를 경로 시작점에 놓고(옵션 `init_offset_max`, `init_yaw_max`, 기본 0)
  roll/pitch ±0.1 rad 와 `init_vel_max` 만 흔든다. 논문 프리셋(none/small/large/ours)과 traj≤5 는 `start_at_robot: false` 로 종전 스폰 공을 유지.
* 뷰어(`scripts/play.py`, debug vis): 목표 점 대신 **기준 경로**를 그린다 — 앞으로 갈 구간 초록 구, 지난 구간 회색 구(0.5 s 간격), 기체가 실제 지나간 자취
  주황 구(최근 5 s), 작은 목표 큐브·헤딩 화살표는 유지. 외부 구동(평가 스크립트) env 는 경로를 그리지 않는다.
* 보상 옵션(기본 꺼짐): `rew_scale_speed_excess` > 0 이면 실속도가 `rew_speed_limit` 을 넘는 만큼 선형 벌점 — 정책이 먼 목표에 상한 속도로 접근하게
  하는 두 번째 처방(스테이션 쪽 1차 처방은 `rl.axis_cap 0.3`).

**학습/시각화 명령** (사용자가 켤 때):
```bash
# 학습 (traj5 레시피 그대로, 프리셋만 traj6; 세션이 닫혀도 계속): 3 시드 순차 + 평가 파이프라인
setsid nohup bash scripts/_pipeline_traj6.sh > logs/pipeline_traj6_outer.log 2>&1 < /dev/null &
# 경로를 따라가는 모습 보기 (워크스테이션 화면; 학습 전에는 traj5_s2 체크포인트로 새 경로를 따라가게 해 볼 수 있다)
DISPLAY=:1 /home/bdml/miniforge3/envs/env_isaaclab/bin/python -u scripts/play.py --task ROV-PoseHold-v0 --num_envs 64 --dr traj6 \
  --checkpoint logs/rsl_rl/rov_posehold/2026-09-21_16-53-50_traj5_s2/model_1199.pt
```
검증: `tests/test_goals_traj.py` (경로 시작점·연속성·상자·정지·yaw 연속·피드포워드), 64 envs 무화면 play 450 스텝(마커 포함), 3 iteration 학습 스모크,
`eval_trajectory.py` 1바퀴(traj5_s2: pos RMS 0.053 m — 8.1 표와 일치) [측정 2026-09-21 통과]. 적대적 검토(4 관점 × 2 반박, 42 에이전트) 의 확정 지적
3건(평가 버퍼, yaw 램프, 벽 반사)을 반영. 학습 traj6 3 시드는 2026-09-21 20:05 착수 (`logs/pipeline_traj6_outer.log`).

> **평가 체크포인트 정정 (2026-09-27 검토)**: `scripts/_pipeline_eval.sh` 의 체크포인트 선택(`sort -t_ -k2 -n`)이 경로의 시각 필드로 묶여 사전순으로
> 떨어져 **1200 iter 런은 전부 `model_950` 으로 평가**되었다(export 는 `model_1199`). 8.3·8.4 표와 `logs/eval/*traj{4..10}*` 의 저장 수치는 iteration 950
> 의 값이다(각 `logs/pipeline_*.log` 의 `== checkpoint` 줄). 스크립트는 고쳤고 traj10_s0..2 는 1199 로 재평가했다(구 폴더는 `*_it950` 으로 보존,
> 새 폴더에 `meta.json` 이 ckpt 경로·iteration 을 기록). §10.5 의 4축 vs pwm 비교는 모두 1199 다.

### 8.3 traj6 → traj9 학습 결과 (2026-09-21 밤, 2048 envs × 1200 iter, 시드 병렬; 전체 표 `logs/eval/RESULTS.md`)

| 정책 | 설정 | 사각형 xy / z | 원 xy / along | 외란 RMS | 포화 max | 0.5 m 스텝 최고속도 / 오버슈트 | 최악지표 |
|---|---|---|---|---|---|---|---|
| **traj5_s2** (배포 중) | 스폰 공 1 m, 이동 목표 v1 | 0.053 / 0.020 | 0.057 / 0.044 | 0.055 | 2 % | 0.48 m/s / 0.025 m | **0.85** |
| traj6_s3 (5 시드 중 최고 추종) | 경로가 기체에서 출발, 오프셋 0 | **0.048 / 0.034** | **0.025 / 0.019** | 0.056 | 5 % | 0.58 / 0.113 | 1.22 |
| traj6_s1 | 〃 | 0.052 / 0.025 | 0.041 / 0.036 | 0.067 | 5 % | 0.54 / 0.120 | 1.02 |
| traj7_s0 (3 시드 중 최고) | traj6 + 오프셋 0.3 m + 속도 벌점 2.0 @0.25 m/s + 회전율 1.0 | 0.057 / 0.029 | 0.066 / 0.064 | 0.054 | 2 % | 0.53 / 0.064 | 0.89 |
| traj8_s0 (3 시드 중 최고) | traj7 + 오프셋 0.8 m + 벌점 4.0 @0.25 m/s | 0.121 / 0.018 | 0.162 / 0.134 | 0.068 | 7 % | **0.32 / 0.000** | 1.68 |

* **traj6**: 궤적 추종은 크게 좋아졌지만(원 xy 0.057 → 0.025 m, 뒤처짐 0.044 → 0.019 m), 0.5 m 스텝에서는 traj5 보다 빠르고(0.58 m/s) 오버슈트가 4배다.
  먼 목표를 학습에서 없애면 돌진이 사라지는 게 아니라 **먼 목표를 다루는 법 자체를 못 배운** 정책이 나온다 (정지 응답 0.1 m → 명령 0.88).
* **traj7**: 속도 벌점만으로는 변화 없음 — 오프셋 0.3 m 에서는 학습 중 상한을 넘는 일이 없어 벌점이 형성할 상태가 없다 (벌점 항 평균 −0.04 → 0).
* **traj8**: 먼 시작 + 강한 벌점은 스텝 응답을 원하는 대로 만들지만(최고 0.24–0.32 m/s, 오버슈트 0) 추종이 무너진다 (0.15 m/s 원에서 along 0.13 m).
  속도 상한과 추종 정밀도는 보상 안에서 직접 상충한다.
* **traj9** (스폰 0.5 m, 벌점 1.0 @0.30 m/s; 워크스테이션 재부팅으로 881 iter 에서 끊겨 `model_850` 에서 재개, `scripts/_resume_traj9_seed.sh`):
  결과는 traj5 와 같은 수준 — 최고 시드 traj9_s1 최악지표 1.08, 0.5 m 스텝 최고 속도 0.48–0.51 m/s, 오버슈트 0.02–0.03 m, 원 xy 0.046 m.
  벌점 1.0 @0.30 은 스텝 응답을 바꾸지 못했고(traj7 과 같은 이유: 학습 중 0.30 m/s 를 거의 넘지 않음) 추종은 traj5 와 동급. **결론: 접근 속도는
  보상으로 낮추면 추종을 잃고(traj8), 약하게 주면 효과가 없다(traj7·9). 스테이션 캡(0.3)으로 다루는 것이 맞다.** 캡 0.3 아래 스텝 [측정]:
  traj5_s2 0.16 m/s·오버슈트 0.013 m, traj9_s2 0.17·0.017, traj6_s3 0.26·0.052.
* 현재 권장은 그대로 **traj5_s2 + 스테이션 `rl.axis_cap 0.3`** (실기 검증, 오버슈트 최소). 궤적 정밀도가 우선이고 접근 속도를 스테이션 캡·leash 로
  다룰 수 있다면 traj6_s3 이 후보.

### 8.4 traj10 (2026-09-22): 실기 오차 진단에 따른 세 가지 변경

실기 런 `0921_182926` 재분석 [측정]: 진행 방향 오차가 **상수 편향** +0.034 m(표준편차 0.016; 횡방향 −0.002 ± 0.013) → xy RMS 0.042 의 80 %.
정적 pitch **−11°** (정지 중 −10.6°; 9/9 페이로드 변경 전 런들은 −3~−7°) → 시뮬 학습 분포(수평 ±6°) 밖.
1. **좁은 위치 보상 항** `rew_scale_pos_xy_fine 0.15 (σ 0.08 m)`, `rew_scale_pos_z_fine 0.15 (σ 0.04 m)`: 넓은 항(σ 0.3)은 3 cm 에서 기울기가
   1 % 뿐이라 정책이 마지막 몇 cm 를 행동 벌점과 맞바꾼다; 좁은 항은 3 cm 에서 13 % 를 잃게 한다.
2. **정적 CB 트림** `dr.cob_trim_x_m −0.029` [유도 0.15 m·tan 11°] + 기존 1.5 cm 디스크(±6°): 실기 nose-down 11° 를 학습 범위 중앙에.
   실기 밸러스트로 트림을 되돌리는 것이 근본 해법(추력 19 % 수직 누설, 카메라 51° 하향).
3. **충격 외란 제거, 테더형 편향력으로 교체** (사용자 결정): `push_prob_per_step 0`, `bias_force_n [0, 4]` N 수평, 방향은 리셋 때 랜덤 + 랜덤워크
   (0.3 rad/√s) [예측]. privileged critic 이 이 힘을 본다(`critic_priv_bias`, 폭 31 → 33; 이전 체크포인트는 `_eval_common` 이 31 로 복원).
   외란 평가(`eval_disturbance`)의 충격은 그대로 두어 비교 가능.
4. 회전율 범위 1.0 rad/s (station 코너 슬루).

**결과 (3 시드, 2026-09-22)** [측정]. 표준 세계(수평 트림)와 실기와 같은 nose-down 11° 세계(`--shift cob_x=-0.029`) 모두:

| 세계 | 정책 | 사각형 xy / along RMS, along 편향 | 원 xy / along | z RMS | 외란 RMS | 캡 0.3 스텝 최고속도 / 오버슈트 / 정상오차 | 포화 |
|---|---|---|---|---|---|---|---|
| 수평 | traj5_s2 | 0.053 / 0.047, +0.029 | 0.057 / 0.044 | 0.020 | 0.055 | 0.16 / 0.013 / 0.016 | 2 % |
| 수평 | **traj10_s1** | **0.035 / 0.027, +0.003** | **0.026 / 0.018** | 0.034 | 0.040 | 0.24 / 0.016 / 0.041 | 1 % |
| 수평 | traj10_s0 | 0.041 / 0.033, −0.016 | 0.019 / 0.014 | 0.036 | 0.039 | 0.24 / 0.010 / 0.032 | 4 % |
| −11° | traj5_s2 | 0.066 / 0.053, +0.030 | 0.064 / 0.046 | 0.025 | – | 0.20 / 0.012 / 0.024 | 2 % |
| −11° | **traj10_s1** | **0.047 / 0.030, −0.006** | **0.036 / 0.012** | 0.028 | – | 0.23 / 0.014 / 0.029 | 1 % |
| −11° | traj10_s0 | 0.049 / 0.037, −0.019 | 0.024 / 0.013 | 0.024 | – | 0.22 / 0.006 / 0.019 | 5 % |

* along 편향(실기 오차의 80 %)이 +0.03 → 0 으로 사라졌고, 실기와 같은 트림 세계에서 수평 추종이 30–45 % 좋아졌다. 충격 외란을 학습에서 뺐는데도
  외란 억제는 더 낫다(편향력 학습). 대가: 수평 세계의 z RMS 0.020 → 0.034 (트림 세계에서는 0.025 vs 0.028 로 비슷).
* **배포 (2026-09-22)**: station `rl.policy_dir` → `rl_policies/traj10_s1` (미실기; traj5_s2 로 되돌리는 방법은 hw_mpc.yaml 주석).
  단, 실기를 밸러스트로 수평에 가깝게 되돌리면 traj10(−11° ± 6° 학습)은 분포 밖이므로 그때는 traj5_s2 로 돌아가거나 트림 범위
  [−0.035, 0] m 를 DR 로 덮은 정책(traj11 후보)을 학습해야 한다.

> **기준속도 상한 0.25 → 0.12 m/s (2026-10-05, 운영자 결정)**: 프리셋 `traj10` / `traj10sea` / `traj10att` 의 `moving_goal.speed_range` 와
> `MovingGoalCfg` 기본값을 `[0.02, 0.12]` 로 바꿨다. 이 날짜 이전의 런(traj10_s*, pwm10*, sea6*, att7pwm_s* 등)은 0.25 로 학습되었다
> (각 런의 `params/env.yaml`, export `obs_spec.json` 의 `moving_goal`). traj~traj9 프리셋은 기록용으로 0.25 그대로 둔다.
> 생성기 표본(4096 에피소드 × 20 s): 틱 평균 0.079 → 0.042, 이동 에피소드 평균 0.114 → 0.060, p90 0.224 → 0.110 m/s [유도: `rov_rl/goals.py` 샘플링].
> 주의 세 가지:
> 1. 하이드라 `env.moving_goal.*` 오버라이드는 `--dr` 와 함께 쓰면 `apply_dr_preset` 이 `MovingGoalCfg()` 를 새로 만들어 **조용히 지운다**. 속도는 프리셋에서 바꿔야 한다.
> 2. `_pipeline_eval.sh` 는 0.15 m/s 로 평가한다. 0.12 정책에는 분포 밖이다.
> 3. 이 범위는 움직이는 기준의 속도만 정한다. 오차를 메우는 속도는 시작 분포가 아니라 보상이 정한다. 0.5 m 스텝 최고속도는 traj6(로봇 위치 시작) 0.57–0.66, traj10(1 m 공) 0.50–0.56, 속도 벌점을 준 traj8 0.32 m/s다. 작은 오차 이득은 반대로 시작을 좁힐수록 커졌다(정적 프로브 d 0.10 m surge, it1199: traj6 0.62–0.88 > traj7 0.55–0.71 > traj9 0.50–0.62 ≈ traj5 0.52 > traj8 0.47–0.49 > traj10 0.41–0.46, 다른 변경과 교락) [측정: `logs/eval/start_dist_review_20261005/{step_metrics,static_probe}.out`].
> 재학습할 때는 run 이름을 새로 지어야 한다. 기존 레시피 스크립트를 그대로 돌리면 `logs/train_<run>.log` 를 덮어쓴다.

### 8.5 traj12 (2026-10-05, 운영자 결정): 로봇 위치에서 시작 + 약한 속도 벌점, 학습만

traj10 과 DR·보상 6키가 같고, 바뀐 것은 아래 다섯 가지다.
- 프리셋 `traj12`: `moving_goal.start_at_robot true`, `init_offset_max 0.05` m, `init_yaw_max 0.10` rad, `speed_range [0.02, 0.10]`, `init_vel_max 0.05` (traj10 은 0.2).
- hydra: `env.rew_scale_speed_excess=1.0 env.rew_speed_limit=0.12`.
`scripts/_pipeline_traj12_seed.sh` 는 학습과 export 만 하고 평가 체인은 돌리지 않는다. 처음에 3 시드를 `systemd-run --user --unit=rl-traj12-s{0,1,2}` 로 띄웠다.
`rov_env.py` 의 separable 보상 로그에 오차 꼬리 지표를 추가했다: `/err/frac_gt_0p10`, `/err/frac_gt_0p30`, `/err/pos_p95_m`, `/vel/frac_over_limit`. 로그만 늘렸고 동작은 같다.
배경: 경로 위에서 시작해도 에피소드 중 오차는 커진다. traj6 은 학습 말기 평균 0.100–0.115 m 였다 [측정: `logs/train_traj6_s{0,1,3}.log`]. 그 꼬리를 이번에 처음 기록한다.

## 9. 가정과 한계 (실기 전 확인 목록)
1. 태그는 항상 보인다(사용자 가정). 실기에서는 fix 0.5 s 이상 끊기면 NEUTRAL 이므로 시야 이탈 = 정지.
2. 유효 복원 팔 0.15 m, 순부력 +1 N, 추진기 지연 2 스텝, 전압 0.72 는 [유도]/[예측] — DR 범위가 이를 덮도록 잡았다. 첫 pool 시험은 `--dr ours` 정책과 `axis-cap 0.5` 로.
3. roll/pitch 는 명령하지 않는다(4축). 6축(MANUAL_CONTROL s,t) 확장은 `mixer.py` 의 roll/pitch 입력만 열면 된다.
4. `run_policy.py` 의 fix 입력(UDP JSON)은 station 쪽에서 NavFix 를 송신하는 작은 브리지를 추가해야 한다 (또는 이 스크립트를 station 프로세스 안에서 호출).

## 10. 추진기 직접 행동 모드 `pwm` (2026-09-27) — 4축 정책과 나란히 두는 두 번째 버전

사용자 요청(2026-09-27): "출력이 x,y,z,yaw 인 정책 대신 PWM 을 직접 내는 정책도 만들고, 둘 다 남기고, 학습까지." 환경은 하나이고
`ROVEnvCfg.action_mode` 가 출력만 바꾼다 (`rov_rl/obs_spec.py`). 4축 정책·체크포인트·export 는 바이트 하나 안 바뀐다
(`tests/test_pwm_mode.py::test_layout_two_modes`, 기존 33개 테스트 그대로 통과; `params/env.yaml`·`obs_spec.json` 에 키가 없으면 = `axes`).

| | `axes` (기본, 09-27 이전 전부) | `pwm` (신규) |
|---|---|---|
| 정책 출력 | 4: `[surge, sway(+우), heave(+상), yaw(+시계)]` ∈ [−1,1] | **8: 모터 1..8 의 정규화 throttle** `a_i` ∈ [−1,1] (순서 = `configs/rov_params.yaml thrusters.positions` = ArduSub 모터 번호) |
| 추진기까지 | ArduSub vectored-6DOF 믹서(pilot gain 0.5, 그룹 정규화) → 8 throttle | `throttle_i = pwm_throttle_max · a_i` (기본 1.0 = 1100..1900 µs 전 구간), 믹서 **우회** |
| 그 뒤 | 동일: 명령 지연·1차 지연·추진기별 gain·전압·T200 곡선·추력 기하 `B`(`mixer.py` sigma 부호) | 동일 |
| 관측 | 25 × 4 = 100 (prev_action 4) | **29 × 4 = 116** (prev_action 8); privileged critic 31/33 → 35/37 |
| 보상 | 논문 식 그대로 (`rew_scale_actions` 는 8차원 노름) | 동일 — 각도 항이 full quaternion 이라 pwm 정책은 roll/pitch 도 능동 수평 유지 (4축은 불가) |
| export | `obs_spec.json` `action_mode: axes`, wrench head = axes × axis_gain (K = M = 0) | `action_mode: pwm`, `output.pwm_model`(T200 곡선·전압·`B_flu`·sigma) + wrench head = 공칭 정상상태 체인, 6축 모두 actuated |
| 실기 전송 | MANUAL_CONTROL (station `LOW=rl`, `deploy/ardusub_bridge.py`) | **`pwm_lua_override`**: RC_CHANNELS_OVERRIDE ch 9..16 → Navigator **Lua timeout override**(`deploy/ardusub/rl_pwm_override.lua`, 모터는 믹서에 그대로) — `deploy/run_policy.py` 만; station `HwRl` 은 pwm 정책을 **거부**(이유 출력). RCIN passthrough 는 검토에서 **폐기**(§10.3) |

### 10.1 코드 경로
* `rov_rl/obs_spec.py`: `num_actions(mode)`, `frame_dim(goal_vel, n_act)`, `action_convention(mode, throttle_max)`.
* `rov_rl/env/rov_env.py`: `throttles_from_actions(a)` (모드 분기, 유일한 분기점), `action_from_axes(a4)` (PID 기준선·스크립트 4축 명령을 pwm env 에서
  "믹서가 냈을 throttle / pwm_throttle_max" 로 바꿔 같은 PID 가 두 env 를 탄다), `wrench_ned_estimate(a)` (공칭 체인 wrench, 로깅용, FLU→FRD 부호 `[1,−1,−1,1,−1,−1]`).
* 평가 스크립트(`eval_sweep/eval_disturbance/eval_trajectory/play_control`): 행동 폭을 env 에서 읽음(`env.n_act`), CSV 열 `a0..a{n−1}`,
  `_eval_common.apply_run_obs_cfg` 가 런의 `action_mode`/`pwm_throttle_max` 를 복원. `select_best.py`/`summarize_results.py` 는 `a*` 열을 일반화
  (부호반전 지표: 4축이면 yaw 축, pwm 이면 8 채널 평균).
* `deploy/pwm.py` (numpy, torch 없음): `PwmModel(obs_spec.output.pwm_model)` → throttle·펄스·wrench 추정, `PwmPolicy.step(obs) → u[6]`;
  `pwm_model` 에는 T200 곡선·전압·`B_flu`·sigma 외에 **`motor_direction`**(학습 믹서가 가정한 `MOT_n_DIRECTION`), `pwm_cap_trained`, `transport: pwm_lua_override` 가 실린다.
  **캡의 단위는 하나다**: `env.pwm_cap` = `run_policy --pwm-cap` = |throttle| 상한(±400 µs 의 비율). env 는 `clamp(throttle)/pwm_throttle_max` 를 행동으로 저장해
  prev_action 이 브리지의 되먹임(`rc_override_to_action` = throttle/max)과 같고, export 의 wrench head·`PwmModel`·`run_policy`(학습 캡 초과 시 학습 캡으로 낮춤) 가 같은 캡을 쓴다.
  `deploy/ardusub_bridge.py`: `action_mode="pwm"`, `send_pwm(a)` (|throttle| ≤ `pwm_cap`, 펄스 = 1500 + 400·thr 를 1100..1900 으로 클램프,
  ch 9..16 외 채널 65535 = 유지; 0/65534 는 절대 안 보냄 — Sub-4.5 에서 각각 '유지'/'해제'), 매 프레임 중립 MANUAL_CONTROL 동반(RC override 는
  pilot-input failsafe 를 갱신하지 않음). engage 전 게이트(모두 PARAM_REQUEST_READ 로 읽어 하나라도 어긋나면 **거부**): `SERVO1..8_FUNCTION = 33..40`
  (Motor1..8, 믹서 유지), `MOT_1..8_DIRECTION == motor_direction`(원시 펄스는 믹서의 방향 반전을 건너뛰므로 파라미터 하나가 추진기 하나를 조용히 반전시킨다),
  `SCR_ENABLE 1`, `SCR_USER1 1`, `RC_OPTIONS` bit1 = 0, `RC_OVERRIDE_TIME > 0`, `RC9..16_OPTION 0`, `SYSID_MYGCS == 255`, 스크립트 heartbeat
  `NAMED_VALUE_FLOAT "RLPWM"` 2 s 이내, MAVLink 2(`MAVLINK20=1`; v1 다이얼렉트에는 ch 9..18 이 없다). 반대로 `axes` 브리지는 출력 1..8 이
  RCIN passthrough 인 기체를 거부. 중립 = 8 채널 모두 1500 µs, deadman 동일 (`tests/test_pwm_mode.py::test_bridge_pwm_packing_gates_and_neutral`).
* `deploy/ardusub/rl_pwm_override.lua` (**기체 쪽, 미실행 [예측]**): RC_CHANNELS_OVERRIDE(70) 를 `mavlink:receive_chan` 으로 받아 ch 9..16 을
  `SRV_Channels:set_output_pwm_chan_timeout(i, pwm, 100 ms)` 로 모터 출력에 강제 — `SCR_USER1 == 1 ∧ armed ∧ MANUAL ∧ 250 ms 이내 프레임` 일 때만.
  프레임이 끊기면 중립 1 프레임 뒤 override 가 만료돼 AP_Motors 출력(중립 믹서 = 1500, disarm 시 1500)으로 복귀(최악 유지 = STALE 100 + TIMEOUT 100 = 200 ms)
  → DISARM·E-stop·leak·battery·pilot-input·GCS failsafe 가 전부 살아 있다. 스크립트는 `SYSID_MYGCS` 가 아닌 sysid·`SYSID_THISMAV` 가 아닌 target 의 프레임을 버린다
  (AP_Scripting 은 GCS sysid 필터 **앞에서** 스크립트에 메시지를 넘긴다). 패킷 오프셋(체크섬 2 바이트 선행, msgid 10..12, payload 13.., 필드 순서
  chan1..8·target_system·target_component·chan9..18)은 2026-09-27 검토가 pymavlink `mavlink_types.h`·ArduPilot `mavlink_msgs.lua` 로 대조했다(초안은 틀렸었다).
  설치: BlueOS file-browser `configs/ardupilot-manager/firmware/scripts/`, `SCR_ENABLE=1` 후 재부팅, `SCR_HEAP_SIZE ≥ 100000`.
* `deploy/run_policy.py`: `obs_spec.action_mode` 로 경로 선택, `--pwm-cap 0.25`(±100 µs) 부터, `--pwm-channels 9,..,16`; 로그 열 `a_m1..8, pwm1..8, uX..uN`.
* station: `rov_gui/control/rl_policy.py` 가 `action_mode != axes` 인 정책을 생성 시점에 거부(`rov_gui/tests/test_control.py::test_rl_refuses_a_pwm_policy_with_the_reason`).

### 10.2 학습 (traj10 레시피 그대로, 모드만 pwm; 세션이 닫혀도 계속)
```bash
for S in 0 1 2; do systemd-run --user --unit=rl-pwm10-s$S --collect -p WorkingDirectory=/home/bdml/Desktop/RL_controller \
    bash scripts/_pipeline_pwm10_seed.sh $S; done          # Linger=yes 이므로 로그아웃/노트북 종료와 무관
systemctl --user status rl-pwm10-s0                       # 상태;  journalctl --user -u rl-pwm10-s0
tail -3 logs/train_pwm10_s0.log                           # 끝나면 EXIT=0, 이어서 logs/pipeline_pwm10_s0.log (export + 평가 전부)
python scripts/select_best.py traj10_s0 traj10_s1 traj10_s2 pwm10_s0 pwm10_s1 pwm10_s2   # 4축 vs pwm 최악지표 비교
```
`--dr traj10 env.action_mode=pwm env.rew_separable=True env.critic_privileged=True env.rew_scale_actions=0.05 env.rew_scale_action_rate=0.05
env.rew_scale_pos_xy_fine=0.15 env.rew_scale_pos_z_fine=0.15`, 2048 envs × 1200 iter, 시드 0/1/2 병렬. 런 이름 `pwm10_s<seed>`; 4축 대응 런은 `traj10_s<seed>`.

### 10.3 실기 전 확인 목록 (pwm 정책) — 전송 설계 결정 (2026-09-27 검토, Sub-4.5 소스 근거)
* **RCIN passthrough(`SERVOn_FUNCTION = RCINk`) 는 폐기.** passthrough 출력은 `SRV_Channel::output_ch()` 가 RC 입력을 그대로 쓰므로 ARM/DISARM 을 보지 않고,
  AP_Motors 의 disarm 1500 은 Motor 함수에만 가며, 이 ROV 는 RC 수신기가 없어 `RC_OVERRIDE_TIME` 이 만료돼도 마지막 펄스가 **무기한** 남는다(브리지가 죽으면
  정지 수단은 전원 차단뿐). Sub 4.5 는 모터 미할당 arming 검사를 돌리지 않아 경고도 없다 [스펙 ardupilot Sub-4.5: SRV_Channel_aux.cpp, RC_Channel.cpp,
  AP_Motors6DOF.cpp, AP_Arming_Sub.cpp — `.claude/journal/reviews.md` 2026-09-27]. `MAV_CMD_DO_SET_SERVO`(Motor 채널 거부, 타임아웃 없음)와
  `DO_MOTOR_TEST`(ARMED 요구, 500 ms 워치독 DISARM, 8 모터 원자성 없음) 도 후보에서 제외. 커스텀 프레임(`SUB_FRAME_CUSTOM`)은 빈 케이스라 펌웨어 빌드가 필요하고
  어차피 6축 선형 믹서다.
* **채택: `pwm_lua_override`** — 모터는 Motor1..8 그대로, Navigator Lua 가 `set_output_pwm_chan_timeout` 으로 100 ms 씩만 강제. 게이트가 기체 안에서 돈다.
  이 경로는 아직 **벤치 미검증** 이다. 순서: (1) 프로펠러 분리 후 스크립트 설치·`SCR_USER1=1`, (2) 브리지 dry-run 게이트 통과 확인, (3) 모터 i 만 1560 µs 1 s →
  `SERVO_OUTPUT_RAW servo{i}_raw` 추종 확인 → 송신 중단 시 200 ms 내 1500 복귀 확인, (4) **추진기별 부호 실측**: 물속·테더 고정에서 한 기씩 +100 µs 를 주고
  자이로/ATTITUDE 응답 부호로 8기의 물리 방향을 `obs_spec.pwm_model.sigma`·`thrust_dirs_flu` 와 대조 — 수직 4기는 heave 한 축의 합으로만 검증돼 있어
  (`logs/mixer_fit_report.md`) 한 기가 반대여도 4축 정책은 못 느끼지만 pwm 정책은 수평화·yaw 가 반대로 폭주한다.
* 첫 시험은 `--pwm-cap 0.25`(±100 µs, ESC 데드밴드 ±25 µs 바로 위)에서 정지 유지만; 캡은 prev_action 으로 되먹임된다(4축의 `axis_cap` 과 같은 구조).
* 학습 기본 `pwm_throttle_max 1.0` 의 권위: 단일축 surge 명령에서 4축(JS_GAIN 0.5)의 **3.4배**(38 N vs 131 N, T200 곡선이 throttle 에 2차 [유도 mixer+thrusters]);
  단, 4축도 복합 명령에서는 그룹 정규화로 추진기당 1.0 까지 간다. 0.5 로 잡으면 −11° 트림 수평화(앞 2기 +0.75)에 headroom 25 % 만 남아 depth 가 포화하므로
  학습은 1.0 이 맞고, 비교표는 포화 비율·평균 |a|·수직 추진기 평균 |thr|(`/act/vert_mean_abs`) 로 권위 차이와 제어 품질을 분리해 읽는다.
  `eval_trajectory` CSV 의 `uX..uN` 은 모드마다 뜻이 다르다(4축 = station 의사-wrench 60 N/축, pwm = 물리 추정) — 두 모드 비교는 새 열 `fX..fN`(두 모드 모두
  공칭 물리 추정)으로. pwm 모드에서는 `pilot_gain_scale` DR 이 무효(믹서 우회)이며 voltage_scale ±17 % 가 그 역할을 겸한다.
* hydra 는 타입을 검사한다: `env.pwm_throttle_max=1` (int) 은 거부, `1.0` 으로.
* 4축 policy 와 같은 `axis_cap`/leash 는 station 에만 있다; pwm 은 `run_policy.py` 의 캡·deadman·게이트가 전부다.

### 10.4 두 번째 레시피 `pwm10t` — 기울 수 있는 정책을 위한 성형 (검토 지적 3건, `scripts/_pipeline_pwm10t_seed.sh`)
`pwm10`(논문 보상 그대로)과 나란히, 검토가 지적한 세 가지를 켠 변형을 시드 2개로 학습한다. 다른 것은 전부 같아 두 런의 차이가 성형 효과다.
1. **자세 보상 yaw + tilt** (`env.ang_error_mode=yaw env.rew_scale_tilt=0.12 env.rew_tilt_sigma=0.10`): full-quaternion 항은 −11° 트림에서 손실 0.087/step,
   수평화 비용(수직 4기 ≈ ±0.35 throttle) 0.019/step 이라 pwm 정책은 확실히 수평화하지만, 2° 잔차에서도 각도 손실 0.017/step 이 z 1 cm 의 fine-z 손실
   0.009 보다 커 마지막 몇 도의 수평화가 depth 를 밀어낸다 [유도]. Gaussian tilt 항(σ 0.1 rad)은 11° 에서 0.116 의 유인을 주되 ±3° 안에서는 평탄하다.
2. **tilt 종료 60°·태그 드롭아웃 35°** (`env.terminate_tilt_rad=1.05 env.tag_dropout_tilt_rad=0.61`): 초기 탐색(std 1 × ±46 N 수직 pair = 20 N·m roll)
   의 전복 에피소드를 자르고, C3 40° 하향 마운트가 바닥 태그를 잃는 기울기에서 fix 를 끊어 수평화를 관측 가용성으로도 강제한다.
3. **탐색 하한** (`--min_noise_std 0.05`, `scripts/train.py` 가 PPO update 뒤 std 를 클램프, `params/min_noise_std.txt` 기록 → export 가 `obs_spec.json
   training.min_noise_std` 로 실음, `--resume` 시 자동 재적용): 4축 런의 최종 std 0.007 (traj10_s1) / 0.014 (traj5_s2)
   [측정 `logs/rsl_rl/rov_posehold/2026-09-22_04-15-41_traj10_s1/events.out.tfevents.*`, `…/2026-09-21_16-53-50_traj5_s2/events.out.tfevents.*`,
   `Policy/mean_noise_std` 마지막 iter] 는 T200 데드밴드(행동 단위 0.075) 안이라 미세 보정 gradient 가 없고, 8 채널에서는 1 N 양자화가
   시간 디더(= ESC 역전)를 유도한다. 0.05 에서 데드밴드 밖 샘플 13 % [유도 정규분포 σ 0.05, 데드밴드 0.075: 2(1−Φ(1.5))]. 채택 지표:
   `summarize_results` 의 pwm 부호반전(데드밴드 밖 샘플만 셈)·포화 비율.
학습 로그에 `/att/tilt_deg`, `/act/vert_mean_abs`, `/act/horiz_mean_abs`, `/act/in_deadband_frac` 가 추가됐다(두 레시피 모두).

### 10.5 결과 (2026-09-27, 2048 envs × 1200 iter, 모두 iteration 1199; 표 = `logs/eval/COMPARE_MODES_20260927.md`, 폴더마다 `meta.json`)

`python scripts/compare_modes.py traj10_s0 traj10_s1 traj10_s2 pwm10_s0 pwm10_s1 pwm10_s2 pwm10t_s0 pwm10t_s1` [측정 `logs/eval/{traj,sweep_rl,disturbance}_<run>_*`].
NOMINAL = 수평 트림 세계, TRIM = 실기의 −11° nose-down 세계(`--shift cob_x=-0.029`, 8.4 와 같은 프로토콜). 단위 m / MSE, 각도 deg.

| run | mode | NOMINAL 사각 xy / z / along | 원 xy / along | 스텝 pos(정상) / ang | 외란 | 포화 | \|a\| | 수직 \|a\| | tilt° | TRIM 사각 xy / z / along | 원 xy / along | 스텝 pos / ang | pitch° | 포화 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traj10_s0 | axes | 0.039 / 0.036 / 0.031 | 0.018 / 0.015 | 0.024(0.0016) / 0.049 | 0.060 | 4 % | 0.21 | – | 2.5 | 0.043 / 0.022 / 0.036 | 0.017 / 0.014 | 0.023 / 0.078 | +10.6 | 5 % |
| traj10_s1 | axes | 0.032 / 0.033 / 0.026 | 0.026 / 0.019 | 0.024(0.0017) / 0.038 | 0.062 | 0 % | 0.18 | – | 2.3 | 0.036 / 0.029 / 0.028 | 0.023 / 0.012 | 0.024 / 0.071 | +10.6 | 0 % |
| traj10_s2 | axes | 0.042 / 0.036 / 0.031 | 0.025 / 0.016 | 0.023(0.0016) / 0.033 | 0.062 | 10 % | 0.21 | – | 2.6 | 0.043 / 0.038 / 0.032 | 0.024 / 0.015 | 0.023 / 0.070 | +10.6 | 12 % |
| pwm10_s0 | pwm | 0.032 / 0.036 / 0.024 | 0.027 / 0.026 | 0.019(0.0009) / 0.036 | 0.059 | 2 % | 0.24 | 0.24 | 6.1 | 0.030 / 0.016 / 0.023 | 0.030 / 0.029 | 0.019 / 0.032 | +4.3 | 1 % |
| **pwm10_s1** | pwm | 0.031 / 0.018 / 0.021 | 0.018 / 0.015 | 0.019(0.0005) / 0.041 | 0.048 | 1 % | 0.30 | 0.29 | 7.7 | 0.038 / 0.020 / 0.027 | 0.026 / 0.024 | 0.019 / 0.024 | +0.8 | 2 % |
| pwm10_s2 | pwm | 0.035 / 0.027 / 0.019 | 0.022 / 0.018 | 0.018(0.0004) / 0.042 | 0.048 | **25 %** | 0.38 | 0.49 | 9.2 | 0.036 / 0.022 / 0.023 | 0.024 / 0.021 | 0.018 / 0.027 | +0.1 | 29 % |
| pwm10t_s0 | pwm(t) | 0.039 / **0.512** / 0.030 | 0.026 / 0.021 | 0.250 / 0.026 | 0.499 | **82 %** | 0.56 | 0.50 | 0.0 | 0.039 / 0.512 / 0.030 | 0.027 / 0.023 | 0.249 / 0.023 | −0.1 | 82 % |
| pwm10t_s1 | pwm(t) | 0.036 / 0.032 / 0.022 | 0.019 / 0.016 | 0.019(0.0011) / 0.030 | 0.051 | 13 % | 0.29 | 0.29 | 7.2 | 0.035 / 0.031 / 0.022 | 0.017 / 0.015 | 0.018 / 0.023 | +1.3 | 17 % |

읽는 법 (검토가 요구한 대로 권위와 품질을 분리):
* **궤적 추종은 동급이다.** 사각형 xy 0.031–0.035 (pwm10) vs 0.032–0.042 (4축), 원 0.018–0.027 vs 0.018–0.026, along 0.019–0.024 vs 0.026–0.031.
  스텝 응답과 외란 억제는 pwm 이 조금 낫다(스텝 pos MSE 0.018–0.019 vs 0.023–0.024, 정상상태 0.0004–0.0009 vs 0.0016; 외란 0.048–0.059 vs 0.060–0.062).
  단 pwm 의 단일축 권위가 3.4배(§10.3)이므로 이 차이의 일부는 권위다 — 평균 |a| 는 pwm 0.24–0.38 vs 4축 0.18–0.21.
* **pwm 만 할 수 있는 것 = 수평 유지.** 실기 트림 세계에서 4축 정책은 +10.6° nose-down 으로 앉지만(구동 불가), pwm10 은 +0.1~+4.3° 까지 세운다
  → 스텝 각도 MSE 0.070–0.078 → 0.024–0.032. 비용: 수직 4기 평균 |throttle| 0.24–0.29 (pwm10_s0/s1; T200 곡선·V 0.72 로 ≈ 3–5 N/기 [유도]) 상시 소모.
  수평 세계에서는 같은 정책이 6–8° nose-up(FLU pitch 음수) 으로 **과보상**한다 — traj10 과 마찬가지로 −11° ± 6° 트림 분포에 맞춘 정책이라
  밸러스트를 수평으로 되돌리면 재학습(트림 범위 [−0.035, 0] DR, traj11 후보)이 필요하다.
* **시드 실패 모드 두 가지 (실기 전 반드시 볼 열 = 포화·수직 |a|).** (1) `pwm10_s2`: 수직 4기 부호가 `[+0.57, +0.36, +0.58, +0.33]` — sigma 를 곱하면
  FR↑·FL↓·RR↓·RL↑ 로 **배분 영공간**(wrench 0, 추력만 상쇄) 에 ≈0.45 throttle 을 태우며 포화 25–29 %; 행동 벌점 0.05·exp(−|a|²) 이 이를 못 막는다
  → 추진기별 |a| 벌점(L1) 또는 영공간 투영 벌점이 다음 처방. (2) `pwm10t_s0`: 학습 중 **바닥 착저 최적점**에 빠짐(err/z 0.46 m, 하방 47 N 을
  누르며 tilt 0.3°; `logs/train_pwm10t_s0.log`) — 바닥 접촉이 tilt 항·yaw 항을 공짜로 채워 주는 국소 최적. 실기의 착저 습관([memory hull-on-floor])과
  같은 모양이라 시뮬에도 바닥 접촉 페널티/고도 하한 강화가 필요하다. pwm10t_s1 은 정상(스텝 각도 최고 0.023–0.030, 포화 13–17 %).
* **pwm10t(yaw+tilt 보상, std 하한 0.05)** 는 2 시드 중 1 시드 붕괴라 효과를 판정할 수 없다 — 시드 3개 이상 + 착저 페널티로 다시.
* **후보**: `pwm10_s1` (`logs/rsl_rl/rov_posehold/2026-09-27_11-06-05_pwm10_s1/exported/`): z RMS 0.018–0.020, 외란 0.048, 트림 세계 pitch +0.8°, 포화 1–2 %.
  4축 배포본은 그대로 traj10_s1. 두 모드의 station 비교는 `fX..fN` 열로.
* 실기 전 순서(§10.3): Lua 스크립트 벤치(프로펠러 분리) → 추진기별 부호 실측 → `--pwm-cap 0.25` 정지 유지. station 은 pwm 정책을 거부한다.

## 11. 파도·조류 외란 학습 + 6축 모드 (2026-09-27 오후) — 세 버전을 외란 유/무 혼합 분포에서 학습

사용자 요청: `bluerov2_mujoco_marinegym/` 의 파도·조류 공식을 그대로 써서 외란이 있는 경우와 없는 경우를 **한 학습 분포에** 넣고,
4축 / 6축(x,y,z,yaw,pitch,roll) / PWM 세 버전을 학습한 뒤 1 m × 1 m 사각형 5바퀴로 평가.

### 11.1 외란 이식 (`rov_rl/seastate.py` = `bluerov2_mujoco_marinegym/disturbance/{waves,current,env}.py` 의 torch 배치판)
* 같은 공식: JONSWAP S(ω) × cos^{2s} D(β) 성분 격자, a_ij = √(2 S D dω dβ), 유한수심 분산 ω² = g k tanh(kh)(Newton), 깊이 프로파일
  cosh/sinh, 속도 + 해석적 가속도, Gauss-Markov 드리프트(정확 이산화), 5 모드 NONE/C/CD/CW/CDW. numpy 참조와 4 (t, pos) 점에서 2e-4 일치
  (`tests/test_seastate.py`).
* 같은 물리 분해(MarineGym hydro.py): 상대 속도 ν_r = ν − Rᵀ v_water 가 항력·선형 감쇠·부가질량 유한차분에 들어가고, 주입 외력은
  Froude-Krylov ρ_fk·Vol·a_wave 뿐(ρ_fk 1025, C_M 1, 토크 없음). 정지수 env 는 이전과 비트 동일(분기 자체를 건너뜀).
  부가질량 유한차분은 리셋·`set_sea_state` 에서 **상대 속도로 시드**한다(첫 판은 절대 속도로 시드해 외란 에피소드마다 t=0 에 M_A·v_water 킥 [유도 검토]).
* 프레임: 바닥 z = 0, 정지 수면 z = h(DR 3.5–4.5 m) → 고도 0.7 m 의 기체는 수면 아래 ≈3.3 m (MarineGym 사이트 h 4, z_ROV −3 보다 0.3 m 깊음).
* 정책은 물 속도를 관측하지 않는다(실기에서도 못 봄); `env.critic_priv_sea=True` 면 critic 만 v_water_b + FK (6) 를 본다.
* 프리셋 `traj10sea` = traj10 + `sea:` 블록: p_mode [0.30, 0.15, 0.15, 0.20, 0.20], V̄ 0.05–0.30 m/s, 방향 균등, Hs 0.3–1.5 m, Tp 5–12 s, γ 1.5–5,
  s 5–30, 파향 균등, h 3.5–4.5, 24×7 성분(학습 비용). 외란 필드는 자체 `torch.Generator` 를 써 센서 잡음 스트림을 건드리지 않는다.
* 평가 외란은 명시적: `--sea "mode=CDW,preset=square|moderate,theta_c_deg=..,seed=.."` — `preset=square` = MarineGym scenario_square.yaml
  (조류 0.20 m/s +x, Hs 0.75 / Tp 12 / γ 5 / s 30, h 4, **격자 60×21 0.2–1.6**), `moderate` = base.yaml moderate(Hs 1.0 / Tp 8, 68×21). 고정 env 는 위상·드리프트를
  공유(RL·PID 가 같은 바다), meta.json 에 seed·preset·격자 기록. 평가 스크립트는 종료 상자를 (3,3)/(0.02,1.5) 로 넓히고 리셋 횟수를 `resets` 로 기록한다.
* 규모 감각 [측정 `logs/smoke/eval_traj10_sea_*.log`, 1바퀴]: 외란 없이 학습한 traj10_s1 은 정지수 0.045 m → C 0.126 → CDW **0.262 m** pos RMS(p95 0.50, 포화 14 %).
  수심 4 m 의 Tp 12 s 파도는 궤도 속도 ≈0.5 m/s [유도] 라 어려운 조건이다.

### 11.2 6축 모드 `axes6` = [surge, sway, heave, yaw, pitch, roll]
* **핵심 사실(검토, Sub-4.5 joystick.cpp)**: ArduSub 는 MANUAL_CONTROL 확장 필드 s(pitch)/t(roll) 를 pitch/roll RC 채널에 **원시 µs 로 더한다**
  (`constrain(s + trim + 1500)`, JS_GAIN 없음). 그래서 와이어 s = 400·a4, t = 400·a5 µs 로 싣고 믹서 입력 = a4/a5 (gain 없음). 스테이션의 attitude 축은
  1000 µs/단위라 정책 1 단위 = 스테이션 0.4 단위(`HwRl` 이 K/M 에 0.4 를 곱함). 첫 판은 gain 0.5 를 곱해 5배 틀린 플랜트였고 그 런은 폐기했다.
* 부호: +a4 → +M(nose-up, FRD), +a5 → +K (테스트가 고정). `attitude_cap` (roll 0.5, pitch 0.75 행동 단위 = 스테이션 cap 0.2/0.3) 을 env 안에서
  적용(pwm_cap 과 같은 구조), export 에 `attitude_cap_trained`·`attitude_wire`, 브리지 기본 캡은 first-water (0.25, 0.375), 부호 +1 [가정].
* obs 27×4 = 108, export WrenchHead 6×6(roll_nm 13.2 / pitch_nm 7.2), station `HwRl` 은 `engage.attitude_axes.enabled` 일 때만 받는다(아니면 거부).

### 11.3 학습 (v2, 검토 반영 후 재시작)
```bash
for M in axes axes6 pwm; do for S in 0 1; do systemd-run --user --unit=rl-sea-$M-s$S --collect \
    -p WorkingDirectory=/home/bdml/Desktop/RL_controller bash scripts/_pipeline_sea_seed.sh $M $S; done; done
# 런 이름 sea4_s<S> / sea6_s<S> / seapwm_s<S>; --dr traj10sea env.critic_priv_sea=True + traj10 보상 오버라이드; 탐색 하한 없음(traj10 과 같은 대조)
bash scripts/_eval_square_sea_batch.sh sea4_s0 sea4_s1 sea6_s0 sea6_s1 seapwm_s0 seapwm_s1 traj10_s1 pwm10_s1   # 사각형 5바퀴 × 6 셀
python scripts/plot_square_sea.py --runs ... --out logs/eval/square_sea_20260927                              # paths.png / bars.png / summary.md
```

### 11.4 결과 — 1 m × 1 m 사각형 5바퀴 @0.15 m/s, 6개 바다 (2026-09-27, iteration 1199; `logs/eval/square_sea_20260927/{paths,bars}.png`, `summary.md`)

xy RMS [cm] (t ≥ 3 s), 괄호 = 어느 축이든 |a| ≥ 0.99 인 스텝 비율. 바다 = MarineGym scenario_square(조류 0.20 m/s +x; 파도 Hs 0.75 / Tp 12, 60×21),
CDWmod = base.yaml moderate(Hs 1.0 / Tp 8). 모든 셀 리셋 0 (경계 확장). PID 는 각 셀에서 같은 바다를 탄 기준선(derate 0.7).
[측정 `logs/eval/traj_<run>_sq5_<mode>/`]

| 정책 | NONE | C | CD | CW | CDW | CDWmod |
|---|---|---|---|---|---|---|
| traj10_s1 (4축, 정지수 학습) | 3.4 (0 %) | 12.3 (0 %) | 11.9 (0 %) | 26.3 (15 %) | 25.8 (15 %) | 27.4 (19 %) |
| sea4_s0 (4축, 외란 학습) | 5.2 (19 %) | 13.0 (20 %) | 12.7 (20 %) | 23.7 (27 %) | 23.2 (26 %) | 25.4 (31 %) |
| sea4_s1 (4축, 외란 학습) | 5.3 (38 %) | 9.2 (37 %) | 9.0 (38 %) | 20.7 (45 %) | 20.5 (45 %) | 23.4 (46 %) |
| sea6_s0 (6축, 외란 학습) | 3.6 (28 %) | 8.6 (28 %) | 8.3 (28 %) | 22.2 (37 %) | 21.6 (37 %) | 25.9 (39 %) |
| sea6_s1 (6축, 외란 학습) | 4.4 (18 %) | 10.3 (17 %) | 10.3 (18 %) | 25.6 (33 %) | 24.9 (32 %) | 28.5 (35 %) |
| pwm10_s1 (PWM, 정지수 학습) | 3.3 (1 %) | 10.9 (3 %) | 10.6 (3 %) | 21.5 (8 %) | 21.3 (8 %) | 25.2 (11 %) |
| **seapwm_s0 (PWM, 외란 학습)** | 3.8 (27 %) | **7.4 (29 %)** | **7.3 (27 %)** | **14.4 (34 %)** | **14.2 (34 %)** | **16.5 (40 %)** |
| seapwm_s1 (PWM, 외란 학습) | 4.4 (38 %) | 6.3 (39 %) | 6.3 (39 %) | 14.6 (50 %) | 14.5 (50 %) | 17.4 (54 %) |
| PID derate 0.7 (기준선) | 22.0 | 27.9 | 27.4 | 44.6 | 43.4 | 47.0 |

읽는 법:
* **파도가 지배적이다.** 조류만(C/CD)은 모든 RL 이 9–13 cm, 파도(CW/CDW)에서는 21–26 cm 로 뛴다(PID 43–47 cm). 수심 4 m 의 Tp 12 s 파도는 기체 위치에서
  ≈0.5 m/s 궤도 속도 [유도] 라 4축/6축 정책의 권위(JS_GAIN 0.5, 서지 ≈38 N)로는 상쇄가 안 된다 — CW 셀의 포화 27–45 % 가 그 증거다.
* **PWM + 외란 학습만 파도에서 크게 좋아진다**: 14.2–14.6 cm (p95 0.30 m) vs 정지수 학습 PWM 21.3 cm, 4축·6축 외란 학습 20.5–25.6 cm. 권위 3.4배(§10.3)와
  외란 학습이 결합된 결과이고, 포화 34–50 % 는 그 권위를 실제로 쓰고 있다는 뜻이다.
* **조류(C/CD)**: PWM 외란 학습 6.3–7.4 cm 가 최선; 6축 외란 학습 8.3–10.3; 4축 외란 학습은 9.0–13.0 으로 정지수 학습 4축(11.9–12.3) 과 큰 차이가 없다 —
  물 속도를 관측하지 않는 4축 정책은 조류의 DC 오프셋을 알아채기 어렵다(적분기 없음).
* **드리프트(CD vs C, CDW vs CW)** 는 20 s 학습 에피소드·133 s 평가 모두에서 차이가 없다(τ 300 s 는 이 시간 규모에서 거의 상수).
* **비용: 외란 학습은 정지수에서도 공격적이다.** 정지수 셀 포화가 traj10 0 % / pwm10 1 % → sea4 19–38 %, sea6 18–28 %, seapwm 27–38 %, xy 도 3.3–3.4 → 3.6–5.3 cm.
  실기 전에는 행동 벌점(추진기별 L1·rate) 또는 pwm_cap/attitude_cap 을 낮춘 재학습이 필요하다.
* **6축(sea6) 이 4축(sea4) 보다 낫지만 PWM 에는 못 미친다**: 조류 8–10 vs 9–13 cm, 파도 22–26 vs 21–24 cm. pitch/roll 로 얻는 것은 작고(믹서 권위는 그대로),
  tilt 5.6–10 ° 를 상시 유지한다.
* seed 편차: sea4 두 시드 3–4 cm, sea6 3–4 cm, seapwm 0.5 cm — PWM 이 가장 일관적.


## 12. roll/pitch 기준 추종 레시피 `att7` (2026-10-03) — 스테이션 LOW `rl_7d` 용, **코드만 (학습 보류)**

7-dim(`pos_rpy_width`) Diffusion Policy 의 roll/pitch 를 RL 하위제어기가 따라가게 하려면 정책이 **기울어진 q_des** 를 학습 때
본 적이 있어야 한다. 그 전까지의 모든 정책(traj*, pwm10*, sea4/sea6/seapwm)은 `q_des = quat_from_yaw(goal_yaw)` — 수평 목표만
봤다. axes6·pwm 정책이 roll/pitch 를 **구동**할 수는 있지만 그건 "트림 세계에서 수평 맞추기" 바이어스이지 기준 추종이 아니다
(수평 세계 원점 hold 평균 pitch: sea6 −6.8°, pwm10_s1 −7.4°; 트림 세계 pwm10_s1 +0.7° — `logs/eval/sweep_rl_*_{nominal,trim}/timeseries.csv` 에서 계산).
**RL_PWM 도 그대로는 7-dim 정책의 roll/pitch 를 추종하지 못한다.**

* **목표 생성** (`rov_rl/goals.py`, 기본 off): `att_enabled` 이면 에피소드 `p_att`(0.7) 비율에 매끈한 roll/pitch 경로
  (bias + 고조파, |roll| ≤ `att_roll_max` 0.17, |pitch| ≤ `att_pitch_max` 0.35 rad, Euler rate ≤ `att_rate_max` 0.35 rad/s,
  주파수 `att_freq_range` 0.02–0.10 Hz, 속도와 같은 smoothstep 으로 **수평에서 시작**, 위치 이동과 독립 — 기울어진 채 정지 유지가 핵심).
  난수는 위치/yaw 를 모두 뽑은 **뒤에만** 뽑으므로 (a) off 면 기존 레시피와 비트 동일(신·구 goals.py 12 조합 텐서+RNG 상태 동일,
  `logs/smoke/att7_goals_byte_identity_20261003.txt`), (b) on 이어도 같은 시드의 위치/yaw 경로는 같은 텐서다(`tests/test_att7_goals.py`).
  `set_external(roll=, pitch=)` 은 생략하면 수평.
* **env**: `q_des = quat_from_euler_xyz(goal.roll, goal.pitch, goal.yaw)` (off 면 `quat_from_yaw` 그대로), `set_goal(roll=, pitch=)`
  (att 학습 env 에서만 허용). att 목표면 `ang_error_mode` 는 `full` 이어야 하고 `rew_scale_tilt` 는 0 이어야 한다 — 아니면 ROVEnv 가
  **생성 시 거부**(절대 tilt 항은 명령된 기울기를 벌한다). 로그 `/att/ref_tilt_deg`, `/att/err_full_deg`. DR `cob_trim_x_range_m`
  (None = 미사용): 트림을 에피소드마다 뽑아 한 세계의 바이어스 대신 폐루프 자세 제어를 배우게 한다.
* **preset `traj10att`** (`configs/dr_presets.yaml`): traj10 과 같은 DR·moving_goal + att 블록 + 트림 범위 [−0.035, 0.005] m
  (0.15 m 팔로 ≈ −13 … +2° [유도]). **파이프라인** `scripts/_pipeline_att7_seed.sh <seed>`: axes6, `ang_error_mode=full`,
  `rew_scale_tilt=0`, `terminate_tilt_rad=1.05`, `tag_dropout_tilt_rad=0.61`(= 스테이션 abort 35°), `--min_noise_std 0.05`, 런 이름 `att7_s<seed>`.
  **미실행** — 실행은 `for S in 0 1 2; do systemd-run --user --unit=rl-att7-s$S --collect -p WorkingDirectory=$PWD bash scripts/_pipeline_att7_seed.sh $S; done`.
* **export 계약**: `obs_spec.json attitude_reference {trained, roll_max_rad, pitch_max_rad, rate_max_rad_s, p_att, freq_range_hz,
  starts_level, trim_x_range_m, q_des_convention, ned_mapping}`. 키가 없거나 `trained: false` 면 **수평 학습** — 스테이션 `rl_7d` 와
  `deploy/obs_builder.py set_goal(roll_ned=, pitch_ned=)` 는 그런 export 에 기울어진 목표를 주지 않고 거부한다(레이아웃이 같아서
  이 키가 유일한 구분자). 매핑 roll_up = roll_ned, pitch_up = −pitch_ned, yaw_up = −yaw_ned.
* **검증**: `tests/` 53/53(신규 `test_att7_goals.py` 5), Isaac 스모크 3 iteration(코드 확인, 학습 아님) — att7 axes6 학습·수평 레시피·
  `ang_error_mode=yaw` 거부·export 모두 통과(`logs/smoke/smoke_att7_20261003.log`; ref tilt 3.7° 로 q_des 가 실제로 기울어짐).
* **남은 것**: 평가 스크립트(eval_sweep/eval_trajectory/summarize_results)에 roll/pitch 기준 단계·지표가 없다; 학습 캡(0.5/0.75)과
  스테이션 first-water 캡(0.25/0.375 정책 단위)이 다르다 — 스테이션은 부호 증명 전엔 `rl_7d` engage 를 거부한다; pwm 모드 att 레시피와
  스테이션 `rl_pwm` 의 q_des 자세 공급은 없다. 이 저장소는 git 이 아니다 — 수정 전 사본 `_pre_att7_backup_20261003/`.

### 12.1 att7pwm — 같은 레시피의 pwm 모드, 정지수 (2026-10-03, 학습·평가 완료)

`scripts/_pipeline_att7pwm_seed.sh <seed>`: traj10att(정지수 — `sea:` 없음, p_mode NONE; 테더 모사 외력 0–4 N은 traj10과 같음) +
`env.action_mode=pwm`, ang_error_mode full, rew_scale_tilt 0, tumble 1.05 / tag dropout 0.61 rad, `--min_noise_std 0.05`. 시드 0..2를
2026-10-03 15:44에 병렬 실행, 각 34:38–34:45 [측정 `logs/train_att7pwm_s*.log`], 표준 평가 체인까지 완료.

**자세 추종 평가** `scripts/eval_attitude.py`(+ `_eval_att.py` 순수 numpy, `_eval_att_batch.sh`, `compare_attitude.py`, `tests/test_eval_att.py`):
정지수 15개 프로그램(pitch ±10/±20°, roll ±5/±9.5°, 조합, 삼각·사인, 기운 채 yaw·이동) × 트림 세계 3개, 스테이션 인터록 에뮬레이션.
결과 [측정 `logs/eval/ATT_COMPARE_20261003.md`]: 기울어진 목표에서 att7pwm이 수평 학습 pwm10_s1보다 크게 낫다(세계 0 pitch RMS 1.46 vs
10.93°) — 그러나 트림 세계 정상 pitch 오차 최대 7–9°, 수평에서도 roll ±3° 흔들림, 수직 |throttle| ~0.7(pwm10_s1 0.3), 포화 24–38 %(s0) /
57–83 %(s1) / ~100 %(s2)로 [예측] 승격 기준 미달. 최선 s0를 스테이션에 **꺼진 후보**로 복사(`rov_gui/control/rl_policies/att7pwm_s0/`).
다음 시도 후보 [예측]: 추진기별 L1/행동 크기 벌점 강화, 트림 범위 축소, iteration 증가.
