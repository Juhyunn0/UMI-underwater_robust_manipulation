# access GUI : `./c3 gui --source hw --allow-command --pose`

./c3 gui --source hw --allow-command --mpc --fstereo --policycd /home/bdml/Desktop/RL_controller && DISPLAY=:1 /home/bdml/miniforge3/envs/env_isaaclab/bin/python -u scripts/play.py --task ROV-PoseHold-v0 --num_envs 64 --checkpoint logs/rsl_rl/rov_posehold/2026-09-21_16-53-50_traj5_s2/model_1199.p

# → 패널: HIGH = Diffusion Policy, ckpt 칸의 "…"로 체크포인트 선택 (hw_mpc.yaml policy.ckpt는

# 기동 시드일 뿐; --policy-ckpt 플래그는 2026-09-11 제거), LOW = 추종자(PID / MPC / …) → START

# diffusion policy check image :

~/miniforge3/envs/robust/bin/python -m rov_gui.tools.plot_policy_map data/20260908/0908_170428_observe --plan-scale 0.25

# depth map html :

./c3 scene-html --run data/20260908/0908_180453_observe --embed --every 1

## 수중 DP만 검증 (저수준 제어기 없이, 조이스틱으로 직접 날며) — 패널의 LOW None

2026-09-11부터 이 런은 **같은 기동 명령**에서 패널로 고른다: HIGH = `Diffusion Policy`,
LOW = `None`(teleop) → START. `--policy-observe`는 LOW None을 미리 골라 두는 기동 별칭으로
계속 동작한다(아래 명령 그대로; `--mpc-mode none`과 같고, 다른 `--mpc-mode`와 같이 주면 기동

거부). LOW는 engagement 동안 고정이라 제어기로 돌아가려면 STOP TRAJ / DISENG 뒤에 LOW에서
PID/MPC/…를 고른다(REC CSV가 열려 있으면 거부).

```bash
./c3 gui --source hw --allow-command --mpc --fstereo --policy --policy-observe \
         --policy-weights model --record-depth
# 또는 --policy-observe 없이 기동한 뒤 패널에서 LOW = None
```

# train diffusion policy :

conda activate umi2
cd /home/bdml/Desktop/umi_underwater_robust_control/external/UMI_aquatic

python train.py
  --config-name=train_diffusion_transformer_umi_depth_workspace
  task=umi_depth_5d
  task.dataset.action_repr=pos_yaw_width
  dataloader.batch_size=32
  val_dataloader.batch_size=32
  training.num_epochs=200
  training.checkpoint_every=5
  checkpoint.topk.k=100
  logging.project=umi_land2water
  exp_name=can_grasp_depth_5d_v0

# for train, data should be at  :

/home/bdml/Desktop/data collection/dataset_depth.zarr.zip

# checkpoints :

/home/bdml/Desktop/umi_underwater_robust_control/data/          # 2026-09-14: 단일 날짜 루트
  20260907/0907_113747_train_umi_depth_5d_can_grasp_depth_5d_v0/
    checkpoints/
      epoch=0000..0195-train_loss=*.ckpt   ← 5 에폭마다, 각 704 MB (40개)
      latest.ckpt
      selected.ckpt  →  epoch=0195-train_loss=0.028.ckpt   ← 실기가 쓰는 것 (hw_mpc.yaml policy.ckpt의
                                                            기동 시드; 다른 에폭은 패널 ckpt 칸의 "…"로)
      selected.json                                        ← 왜 195를 골랐는지
    logs.json.txt, normalizer.pkl, .hydra/, heldout_*.log, train_curves_*.png
  train_logs/train_5d_20260907.log

**2026-09-06부터 뒤의 두 플래그를 붙이세요.**

`--policy-weights model`은 체크포인트 안의 비-EMA state dict를 띄웁니다. 기본값
`ema_model`은 held-out action MSE가 **1.8배 나쁩니다**(0.001535 vs 0.000842, pos RMSE
30.7 vs 20.5 mm) [측정 2026-09-06: 검증 에피소드 6/32/48/56, 219 윈도, DDIM 8]. EMA 평균
자체가 아니라 BatchNorm running statistics가 timm ImageNet 값 그대로 실려 온 게 원인이고,
그 buffer 108개만 갈아끼워도 격차의 92%가 회수됩니다. **다른 가중치로 돈 런끼리는 합산
금지**이고, 고른 값은 런 meta에 남습니다.

`--record-depth`는 정책이 실제로 소비한 depth를 `<run>/policy_obs/`에 무손실 PNG로
남깁니다. 이게 없으면 그 런은 depth를 **한 프레임도** 안 남기고, 원본 스테레오 쌍도
없어서 나중에 재계산조차 못 합니다. 육상 학습 관측과 같은 색 규칙으로 나란히 보려면:

```bash
python -m rov_gui.tools.depth_compare pair --run data/<날짜>/<시각>_observe \
        --episode 12 --out /tmp/pair            # --land-frame/--water-frame로 파지 시점 정렬
python -m rov_gui.tools.depth_compare stats --run data/<날짜>/<시각>_observe
```

## 육상 카메라로 직접 캡처 — `depth_capture_umi` (계기를 변수에서 뺀다)

육상 데모를 찍은 그 OAK-D-W(`14442C10716DBCD600`)가 기체에 달려 있으면, C3를 거치지 않고
**학습 파이프라인 그대로** 라이브 depth를 뜰 수 있습니다. 같은 센서, 같은 EEPROM 캘리브,
같은 rectifier 객체, 같은 FoundationStereo 설정(iters 16 / scale 1.0), 같은 warp입니다.
남는 차이는 장면·매질·마운트뿐이라 이게 가장 깨끗한 비교입니다.

```bash
python -m rov_gui.tools.depth_capture_umi --out /tmp/rovcap --frames 30 --medium air
python -m rov_gui.tools.depth_compare pair  --run /tmp/rovcap --episode 12 --out /tmp/pair
python -m rov_gui.tools.depth_compare stats --run /tmp/rovcap --episode 12
```

시작할 때 두 가지를 스스로 검사하고, 어긋나면 **거부**합니다. 라이브 EEPROM과
`calibration.json`의 K 비교(2026-09-06 실측 max|dK| **0.0000 px**), 그리고 rectifier가
내는 fx/fy/cx/cy를 학습 스토어의 `.zattrs['camera_model_source']`와 대조하는 것입니다.

`stats`의 **near-field row**를 꼭 보세요. 가장 가까운 8% 픽셀의 행 중심이고, 0이 화면 위,
1이 아래입니다. 육상 학습은 0.895(손가락이 아래에서 들어옴), 2026-09-06 벤치 캡처는
0.338이었습니다 — **겹치는 구간이 없습니다**. 색을 아무리 맞춰도 이건 안 고쳐집니다.

**`--medium`은 라벨이지 보정이 아닙니다.** 이 카메라의 캘리브는 in-air라, 물에 넣으면
depth가 굴절률만큼 길게 읽힙니다(C3가 공기 중에서 겪는 것의 반대 방향). 아무것도
자동으로 스케일하지 않습니다.

DP는 돌고 그 궤적은 trajectory 패널에 그려지는데, `ctrl.step()`은 아예 안 불리고
스테이션에서 기체로 가는 바이트가 **하나도 없습니다**. 조종은 내내 **조이스틱**이라
병 주위를 직접 날아다니며 여러 위치에서 망이 뭘 그리는지 볼 수 있습니다.
`--allow-command`는 여기선 **조종사가 날기 위해** 필요합니다(land-dry-run과 반대).

순서:

1. LOW 콤보 `None`(별칭으로 기동했으면 이미 골라져 있다) → ENGAGE → HIGH 콤보
   `Diffusion Policy` → START TRAJ. 칩이 `OBSERVE · DP NOT COMMANDING`으로 뜨면 열린
   루프입니다. None 아래서 START가 되는 HIGH는 Diffusion Policy와 Replay뿐이고(Station/Line/…
   은 이유와 함께 거부), None이면 맨 HOLD·replay도 `data/*/*_observe/`에 떨어집니다.
2. 조이스틱으로 직접 비행. 스틱을 움직여도 미션이 안 끊깁니다(제어 런에선 takeover로
   읽혀 disengage됩니다). 태그를 잠깐 놓쳐도 런이 안 끝나고 그 틱만 건너뜁니다.
3. 메인 플롯에 DP 플랜이 **굵은 빨간 점선**으로 뜹니다 — 굵은 것이 지금 것, 가는 것이
   바로 그 전. **딱 두 개**만 남고, **맨 위에** 그려서 선체에 가리지 않습니다. 궤적 꼬리는
   **3초**뿐입니다(전에는 90초라 플랜을 묻었습니다). 우하단 범례에 `DP` 스와치.

추종자를 전제하던 자동 정지 다섯 개(divergence·escalation latch·bridge-too-long·
workspace box·tag-loss disengage)가 **꺼집니다** — 조종사가 일부러 참조에서 멀어지는 것이
이 실험이라서. 턱은 강제 OFF, 런 시계는 `policy.observe_max_run_s`(1800 s).

### 나중에 다시 보기 — `./c3 run-replay`

```bash
./c3 run-replay data/<날짜>/<시각>_observe          # 창으로 재생
./c3 run-replay <RUN_DIR> --speed 2 --start 40 --3d
./c3 run-replay <RUN_DIR> --shots out/ --n-shots 6            # 창 없이 스틸만
```

기록된 런을 되감아 봅니다: **바닥 AprilTag 맵 + 로봇 위치 + DP 업데이트마다의 참조
궤적**. 스페이스=재생/정지, 화살표=한 틱씩, 드래그=이동, 휠=줌. `follow` 체크가 기본
켜짐이라 기체를 따라다니고, 시작 줌이 6배인 이유는 1초짜리 플랜이 **4 cm**라서입니다.

그리는 것은 **라이브와 같은 `TrajectoryView` 위젯**입니다 — CSV에서 `MpcStatus`와
`PolicyPlanViz`를 복원해 먹일 뿐, 좌표 변환을 자기가 하지 않습니다. 그래서 재생 그림이
틀리면 **그날 실제로 보던 그림도 같은 방식으로 틀렸던 것**입니다.

읽는 파일(전부 평범한 런이 이미 남기는 것): `mpc_*.csv`(20 Hz 기체 자세),
`mpc_*.meta.json`의 `hardware.datum_tag_frame`(이게 있어야 맵 프레임에 앉습니다),
`policy_plan.csv`(플랜 knot마다 한 줄, **거부된 것도 포함** — 이 런들에선 그쪽이 다수이고
더 흥미롭습니다), `nav_*/map.json`(REC NAV; 없으면 meta가 가리키는 태그맵으로 대체).

### 이 런에서 나온 숫자를 인용할 때

**폐루프 결과가 아닙니다.** 기체는 조종사가 간 곳으로 갔고 참조는 아무도 안 따랐습니다.
산출물은 `data/<날짜>/<시각>_observe/`에 따로 모이고, CSV `observe` 열이 1,
`uX..uN`/`w*`/`ax_*`/`e_along`/`e_cross`는 nan, `solver_status`는 -1,
`plans.jsonl`·`policy_plan.csv`는 `follower: observe`입니다. accept/reject 비율도
앵커를 손으로 끌고 다닌 결과라 **제어 런과 합산 금지**.

## 지상 DP 검증 (ROV를 손에 들고, ORB-SLAM3 맵에 DP 궤적)

```bash
./c3 gui --source hw --slam --mpc --fstereo --policy \
         --land-dry-run --mpc-config config/land_dp.yaml
```

`--allow-command`가 **없습니다**. 있으면 실행이 거부됩니다 (`__main__.check_slam`) —
사람이 들고 있는 기체에 스러스터 명령을 보내는 조합이라서. 그래서 이 런에서
스러스터로 가는 경로는 아예 만들어지지 않습니다 (`NullCommandSink`).

창이 세 개 뜹니다: rov_gui 스테이션, `ORB-SLAM3: Map Viewer`(Pangolin — 맵 포인트 +
DP 궤적), `ORB-SLAM3: Current Frame`.

순서:

1. 캔을 C3 시야에 두고, 방을 향해 든다. ORB-SLAM3는 한 프레임에서 ORB 키포인트
   500개를 넘겨야 맵을 만들기 시작하고, 못 넘기면 **아무 말도 안 합니다** — 빈 맵만
   보입니다. 질감 있는 1–5 m 장면을 보게 하고 조명을 켤 것. 스테이션 로그가
   `slam: tracking — first pose after …` 를 찍으면 시작된 것.
2. 스테이션에서 ENGAGE → HIGH 콤보 `Diffusion Policy`(LOW는 `land_dp.yaml mode: pid`가
   시드; 체크포인트는 ckpt 칸에서) → START TRAJ.
3. Pangolin 창을 본다. 청록 실선 = 컨트롤러에 도달한 플랜(accept/clip), 회색 점선 =
   스테이션이 거부한 플랜, 주황 = 안전 필터 이전의 네트워크 원본. 최신 12개가 남고
   오래된 것일수록 흐려진다. 최신 플랜에는 6 cm 축 삼각대가 붙는다.
4. 한 번의 리치가 끝나면 STOP TRAJ → START로 재무장한다. 학습 데모는 중앙값 6.7초,
   에피소드 시작 대비 회전이 최대 52.9°였다 [측정: "data collection"/dataset_depth.zarr
   75 ep / 15,746 frame]. 계속 켜둔 채 방을 한 바퀴 돌면 `rot_axis_angle_wrt_start`가
   학습 분포 밖으로 나가고, 그때부터의 예측은 정책의 잘못이 아니다.

### 이 런에서 나온 숫자를 인용할 때

`config/land_dp.yaml`은 `hw_mpc.yaml`에서 6개 값이 다르고, 그중 `obs_max_age_s`와
`reject_escalate`는 accept/clip/reject 분포를 직접 움직입니다. **풀장 런과 합산 금지.**
산출물은 `data/<날짜>/<시각>_landdry/` 에 모입니다 (`slam/orbslam.yaml`,
`slam/traj_frames.tum`, `plans.jsonl`, `policy_plan.csv`).

C3 공장 캘리브는 수중값이라 공기 중에서는 왜곡이 반경 방향으로 불균일합니다. 이
런에서 나온 **미터 단위 수치는 실측 스케일 없이 인용 불가**입니다 (`KNOWN_ISSUES.md`).

### 잘 안 될 때

| 증상                         | 원인                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------ |
| 맵이 계속 비어 있음          | 500-키포인트 게이트. 질감·조명·거리.`--slam-ini-fast 12 --slam-min-fast 5` |
| `NEW MAP (session N)` 로그 | 추적을 놓쳐 ORB-SLAM3가 새 맵을 만듦 → 원점이 이동. STOP 후 재무장            |
| Pangolin에 궤적이 안 그려짐  | ENGAGE 전이면 datum이 없어 좌표를 놓을 수 없음. ENGAGE 먼저                    |
| 회색 점선만 나옴             | 네트워크는 도는데 스테이션이 전부 거부 중.`plans.jsonl`의 status/reason      |

### 카메라 없이 전체 경로 확인

```bash
cd "$HOME/Desktop/data collection/UMI_Underwater/slam"
~/miniforge3/envs/umi2/bin/python oakx_replay.py "$HOME/Desktop/data collection" \
  --episode 0 \
  --driver "'$PWD/build/oakd_live_slam' $HOME/ORB_SLAM3/Vocabulary/ORBvoc.txt rect.yaml /tmp/ep0"
```

녹화 에피소드를 드라이버에 흘리고 합성 나선을 KNOT으로 주입합니다. 맵 창에 나선이
그려지면 프로토콜·오버레이·드로잉이 다 살아 있는 것.
