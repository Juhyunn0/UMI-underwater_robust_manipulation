#!/usr/bin/env bash
# DP ablation을 터미널과 분리해 순차 실행한다.
#
# 실행 (SSH/VS Code가 끊겨도, 노트북이 꺼져도 계속 돈다 — setsid + nohup + stdin 분리):
#   setsid nohup bash tools/run_ablations.sh [NAME ...] > <runner.log> 2>&1 < /dev/null &
#
# NAME 생략 시 기본 세트(E_baseline A_vitclip C_groupnorm)를 순서대로 실행한다.
#   예) bash tools/run_ablations.sh C_groupnorm      # C만
#   예) bash tools/run_ablations.sh D_dinov3b        # D (DINOv3 ViT-B/16) — A와 같은 레시피, 인코더만 교체
#       D는 umi2에 timm >= 1.0.20 필요(2026-09-14 부터 1.0.29); 첫 실행 시 HF에서 가중치 343 MB 다운로드.
#
# 2026-09-23 배치 — 이름은 <modality>_<encoder>_<data>:
#   rgbd_vitclip_9926        RGB+depth two-stream, ViT-B/16 CLIP,  9_9_26 (dataset_rgbd.zarr.zip)
#   rgbd_dinov3b_9926        RGB+depth two-stream, DINOv3 ViT-B/16, 9_9_26
#   depth_vitclip_9426_9926  depth only, ViT-B/16 CLIP,  9_4_26 + 9_9_26 (dataset_depth_9426_9926.zarr.zip)
#   depth_dinov3b_9426_9926  depth only, DINOv3 ViT-B/16, 9_4_26 + 9_9_26
#   데이터는 umi_handheld/build_dp_rgbd_zarr.py, umi_handheld/concat_dp_zarr.py 로 만든다.
#   best ckpt 항목 이름은 <task>_<exp>_<YYYYMMDD>_<HHMMSS> (best_ckpt.py, 날짜가 뒤).
#
# 2026-09-26 7-dim 변형 (pos_rpy_width: [dx,dy,dz,dyaw,droll,dpitch,width], task umi_depth_7d / umi_rgbd_7d):
#   depth_7d_E_baseline        depth only, BN-ResNet (E 레시피),   9_9_26
#   depth_7d_dinov3b           depth only, DINOv3 ViT-B/16 (D 레시피), 9_9_26
#   rgbd_7d_dinov3b            RGB+depth two-stream, DINOv3 ViT-B/16, 9_9_26 (batch 16 × accumulate 2)
#   depth_7d_dinov3b_9426_9926 depth only, DINOv3 ViT-B/16, 9_4_26 + 9_9_26
#   spec 7번째 필드가 action_repr 이다(기존 행은 pos_yaw_width 그대로 → 유효 action_repr 동일). `task.action_repr=` 로
#   넘긴다: task yaml 의 `dataset.action_repr` 와 `shape_meta.action.action_repr` 둘 다 `${task.action_repr}` 보간이라
#   한 번에 바뀐다(`task.dataset.action_repr` 만 덮으면 shape_meta 는 yaml 값으로 남아 shape [7] + 5열 라벨 불일치). 학습 폴더는
#   <stamp>_train_umi_depth_7d_<exp>, 폴더 안에 rp_label_stats.txt(|droll|/|dpitch| 백분위) 가 생긴다.
#   선택은 held-out 을 `model` 가중치로(ema_model 금지, EMA/BN 결함) — 5-dim 과 같은 규칙.
#
# 디스크: ckpt 하나가 1.2 GB(depth) / ~1.9 GB(rgbd)라 새 배치는 topk 2 + latest 만 남긴다.
#   run 시작 전 여유가 MIN_FREE_GB 미만이면 그 run 은 건너뛴다(조용히 꽉 채워 죽는 것보다 낫다).
# 재시도: train.py 가 0 이 아닌 코드로 끝났고 <run>/checkpoints/latest.ckpt 가 있으면 같은 폴더에서
#   training.resume=True 로 한 번 더 이어 돈다(2026-09-14 D run 의 수동 resume 을 자동화).
# 병렬: 기본은 train.py 가 이미 돌면 시작하지 않는다. 두 번째 러너를 일부러 띄울 때만 ALLOW_PARALLEL=1.
set -u

REPO="$HOME/Desktop/umi_underwater_robust_control"
# stdout of each run, beside the day's training runs (data/<YYYYMMDD>/<run>/ —
# the single dated root since 2026-09-14; was data/outputs/train_logs)
LOGDIR="$REPO/data/$(date +%Y%m%d)/train_logs"
PY="$HOME/miniforge3/envs/umi2/bin/python"
DC="/home/bdml/Desktop/data collection"
DS_9926_DEPTH="$DC/slam/9_9_26/dataset_depth.zarr.zip"
DS_9926_RGBD="$DC/slam/9_9_26/dataset_rgbd.zarr.zip"
DS_9426_9926_DEPTH="$DC/slam/dataset_depth_9426_9926.zarr.zip"
DS_PEG_9_27_9_28="$DC/slam/dataset_depth_peg_9_27_9_28.zarr.zip"
MIN_FREE_GB="${MIN_FREE_GB:-12}"

# NAME -> "config|task|dataset|exp_name|topk_k|extra hydra overrides (space-separated, may be empty)|action_repr"
# rgbd_dinov3b_9926: batch 32 는 RTX 5090 에서 OOM (2026-09-23 스모크: 24.7 GB 필요 + 다른 프로세스 6.2 GB)
#   → batch 16 × gradient_accumulate_every 2 = 유효 batch 32. optimizer/scheduler step 수는 동일하고
#   EMA 만 micro-batch 마다 갱신된다(배포는 `model` 가중치를 쓰므로 무관).
spec_for() {
  case "$1" in
    E_baseline)  echo "train_diffusion_transformer_umi_depth_workspace|umi_depth_5d|$DS_9926_DEPTH|depth5d_9926_E_baseline|10||pos_yaw_width" ;;
    A_vitclip)   echo "train_diffusion_transformer_umi_depth_vit_workspace|umi_depth_5d|$DS_9926_DEPTH|depth5d_9926_A_vitclip|10||pos_yaw_width" ;;
    C_groupnorm) echo "train_diffusion_transformer_umi_depth_gn_workspace|umi_depth_5d|$DS_9926_DEPTH|depth5d_9926_C_groupnorm|10||pos_yaw_width" ;;
    D_dinov3b)   echo "train_diffusion_transformer_umi_depth_dinov3b_workspace|umi_depth_5d|$DS_9926_DEPTH|depth5d_9926_D_dinov3b|10||pos_yaw_width" ;;
    rgbd_vitclip_9926)       echo "train_diffusion_transformer_umi_rgbd_vit_workspace|umi_rgbd_5d|$DS_9926_RGBD|vitclip_9926|2||pos_yaw_width" ;;
    rgbd_dinov3b_9926)       echo "train_diffusion_transformer_umi_rgbd_dinov3b_workspace|umi_rgbd_5d|$DS_9926_RGBD|dinov3b_9926|2|dataloader.batch_size=16 val_dataloader.batch_size=16 training.gradient_accumulate_every=2|pos_yaw_width" ;;
    depth_vitclip_9426_9926) echo "train_diffusion_transformer_umi_depth_vit_workspace|umi_depth_5d|$DS_9426_9926_DEPTH|vitclip_9426_9926|2||pos_yaw_width" ;;
    depth_dinov3b_9426_9926) echo "train_diffusion_transformer_umi_depth_dinov3b_workspace|umi_depth_5d|$DS_9426_9926_DEPTH|dinov3b_9426_9926|2||pos_yaw_width" ;;
    # 7-dim pos_rpy_width (2026-09-26)
    depth_7d_E_baseline)        echo "train_diffusion_transformer_umi_depth_workspace|umi_depth_7d|$DS_9926_DEPTH|depth7d_9926_E_baseline|2||pos_rpy_width" ;;
    depth_7d_dinov3b)           echo "train_diffusion_transformer_umi_depth_dinov3b_workspace|umi_depth_7d|$DS_9926_DEPTH|depth7d_9926_dinov3b|2||pos_rpy_width" ;;
    rgbd_7d_dinov3b)            echo "train_diffusion_transformer_umi_rgbd_dinov3b_workspace|umi_rgbd_7d|$DS_9926_RGBD|rgbd7d_9926_dinov3b|2|dataloader.batch_size=16 val_dataloader.batch_size=16 training.gradient_accumulate_every=2|pos_rpy_width" ;;
    depth_7d_dinov3b_9426_9926) echo "train_diffusion_transformer_umi_depth_dinov3b_workspace|umi_depth_7d|$DS_9426_9926_DEPTH|depth7d_9426_9926_dinov3b|2||pos_rpy_width" ;;
    # peg-in-hole 9/27 + 9/28 병합 (2026-09-28). config_A(ViT-B/16 CLIP) 레시피를 5d/7d 두 출력으로.
    #   5d = [dx,dy,dz,dyaw,width]   7d = [dx,dy,dz,dyaw,droll,dpitch,width]  — width 는 어느 쪽이든 마지막 컬럼.
    #   0924_081113_train_umi_depth_5d_vitclip_9426_9926 과 같은 overrides (batch 32, 200 epoch, topk 2).
    peg_5d_vitclip) echo "train_diffusion_transformer_umi_depth_vit_workspace|umi_depth_5d|$DS_PEG_9_27_9_28|peg5d_vitclip_9_27_9_28|2||pos_yaw_width" ;;
    peg_7d_vitclip) echo "train_diffusion_transformer_umi_depth_vit_workspace|umi_depth_7d|$DS_PEG_9_27_9_28|peg7d_vitclip_9_27_9_28|2||pos_rpy_width" ;;
    *)           echo "" ;;
  esac
}

free_gb() { df --output=avail -BG "$REPO/data" | tail -1 | tr -dc '0-9'; }

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(E_baseline A_vitclip C_groupnorm)

# 중복 실행 방지: 이미 학습이 돌고 있으면 시작하지 않는다.
RUNNING="$(ps -eo args --no-headers | grep -c "^[^ ]*python train\.py --config-name" || true)"
if [ "${RUNNING:-0}" -gt 0 ] && [ "${ALLOW_PARALLEL:-0}" != "1" ]; then
  echo "!! train.py가 이미 $RUNNING 개 실행 중 — 중복 방지를 위해 종료한다 (의도한 병렬이면 ALLOW_PARALLEL=1)"; exit 1
fi

cd "$REPO/external/UMI_aquatic" || { echo "cd 실패"; exit 1; }
mkdir -p "$LOGDIR"

for NAME in "${RUNS[@]}"; do
  SPEC="$(spec_for "$NAME")"
  if [ -z "$SPEC" ]; then echo "!! 알 수 없는 run: $NAME — 건너뜀"; continue; fi
  IFS='|' read -r CFG TASK DATASET EXP TOPK EXTRA REPR <<< "$SPEC"
  if [ -z "${REPR:-}" ]; then echo "!! $NAME: spec 에 action_repr 없음 — 건너뜀"; continue; fi
  if [ ! -f "$DATASET" ]; then echo "!! $NAME: 데이터셋 없음 $DATASET — 건너뜀"; continue; fi
  FREE="$(free_gb)"
  if [ "${FREE:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "!! $NAME: 디스크 여유 ${FREE} GB < ${MIN_FREE_GB} GB — 건너뜀 (오래된 run 의 topk ckpt 를 정리하고 다시)"; continue
  fi
  DAY="$(date +%Y%m%d)"; STAMP="$(date +%m%d_%H%M%S)"
  RUN_DIR="$REPO/data/$DAY/${STAMP}_train_${TASK}_${EXP}"
  LOG="$LOGDIR/${NAME}_${STAMP}.log"
  echo "=== $NAME 시작 $(date +%F_%T) (free ${FREE} GB) → $RUN_DIR ; log $LOG ==="
  ARGS=(train.py --config-name="$CFG"
      "task=$TASK" "task.dataset_path=$DATASET"
      "task.action_repr=$REPR"
      dataloader.batch_size=32 val_dataloader.batch_size=32
      training.num_epochs=200 training.checkpoint_every=5 "checkpoint.topk.k=$TOPK"
      logging.project=umi_land2water "exp_name=$EXP"
      "hydra.run.dir=$RUN_DIR" "multi_run.run_dir=$RUN_DIR")
  # shellcheck disable=SC2206  # EXTRA is a deliberate space-separated override list
  [ -n "$EXTRA" ] && ARGS+=($EXTRA)
  echo "    overrides: ${ARGS[*]:2}"
  "$PY" "${ARGS[@]}" > "$LOG" 2>&1
  RC=$?
  if [ "$RC" -ne 0 ] && [ -f "$RUN_DIR/checkpoints/latest.ckpt" ]; then
    echo "=== $NAME exit=$RC, latest.ckpt 있음 → 같은 폴더에서 resume 1회 $(date +%F_%T) ==="
    "$PY" "${ARGS[@]}" training.resume=True > "$LOG.resume" 2>&1
    RC=$?
  fi
  echo "=== $NAME 종료 exit=$RC $(date +%F_%T) ==="
done
echo "=== ALL DONE $(date +%F_%T) ==="
