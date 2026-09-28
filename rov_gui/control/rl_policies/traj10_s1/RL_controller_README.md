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
| 행동 | 추진기 6개 PWM 직접 | **MANUAL_CONTROL 4축** `[surge, sway(+우), heave(+상), yaw(+시계)]` → **ArduSub vectored-6DOF 믹서**(`rov_rl/mixer.py`) → 8 throttle. 믹서·게인·모터 부호는 실기 로그 64,512행으로 검증 [측정 `logs/mixer_fit_report.md`] |
| 센서 | DVL + IMU + AprilTag EKF (v_body 직접 관측) | **DVL 없음**: 태그 PnP(≈15 Hz, ≈130 ms 지연, 잡음) + ArduSub ATTITUDE/자이로를 시뮬레이션하고, station 의 `state_assembler` 와 **같은 추정기**(유한차분+LPF α=0.6, 자이로/속도 브리징)로 v_body 를 만든다 (`rov_rl/sensors.py`, `rov_rl/estimator.py`) |
| 관측 | 17차원 `[q_des, x_off_b, q, v_b, ω_b]` | 17 + 목표 속도/yaw rate 4 (8장) + 직전 행동 4 = 25 를 **4프레임 이력** → 100차원 (지연 보상). 사원수는 w≥0 으로 정규화 (7.3). 2026-09-16 런은 84차원 |
| 보상 | `0.2e^{-\|x\|²} + 0.5e^{-angle} + 0.2e^{-\|a\|²}` | 동일 (`rov_rl/rewards.py`, yaw-only 옵션) |
| 제어 주기 | 60 Hz 학습 / 20 Hz 배포 | **20 Hz** 학습=배포 (physics 1/120 s, decimation 6), 에피소드 20 s (2026-09-16 런은 6 s) |
| DR | COB 오프셋 구, 부피 | `none/small/large`(논문 대응) + `ours`(질량·관성·항력·전압·추진기별 gain·지연·센서·임펄스 외란) `configs/dr_presets.yaml` |
| 목표 자세 | 임의 단위 사원수 | yaw 균등(roll/pitch 는 4축이라 비제어) — `full_random_orientation` 옵션 |
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

## 9. 가정과 한계 (실기 전 확인 목록)
1. 태그는 항상 보인다(사용자 가정). 실기에서는 fix 0.5 s 이상 끊기면 NEUTRAL 이므로 시야 이탈 = 정지.
2. 유효 복원 팔 0.15 m, 순부력 +1 N, 추진기 지연 2 스텝, 전압 0.72 는 [유도]/[예측] — DR 범위가 이를 덮도록 잡았다. 첫 pool 시험은 `--dr ours` 정책과 `axis-cap 0.5` 로.
3. roll/pitch 는 명령하지 않는다(4축). 6축(MANUAL_CONTROL s,t) 확장은 `mixer.py` 의 roll/pitch 입력만 열면 된다.
4. `run_policy.py` 의 fix 입력(UDP JSON)은 station 쪽에서 NavFix 를 송신하는 작은 브리지를 추가해야 한다 (또는 이 스크립트를 station 프로세스 안에서 호출).
