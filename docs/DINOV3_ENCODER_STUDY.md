# The question of using DINOv3 as the Diffusion Policy encoder — research and measurement report (2026-09-13)

> **English translation.** Source of truth: [`DINOV3_ENCODER_STUDY.ko.md`](DINOV3_ENCODER_STUDY.ko.md) (Korean). Snapshot of 2026-10-12, source blob `6346e1d89c`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

Question: *"I want to try switching the DP encoder to DINOv3 — not the small one, but the most official (largest) version."*
No code was written. Close reading of the repo (3) + web research (4) + RTX 5090 measurements (2 kinds) + follow-up questions (4) +
per-claim adversarial verification (60 items: verified 33 / uncertain 14 / rejected 1 —
[`measurements/.../agent_results/verdicts.csv`](measurements/dinov3_encoder_study_20260913/agent_results/verdicts.csv)).
The measurement scripts ran in the session scratch (volatile), and the original CSV/JSON files were lost when the session ended.
The numbers below are the values the agents reported as structured output (the verifier read the original files at the time and cross-checked them),
and those reports were preserved in [`docs/measurements/dinov3_encoder_study_20260913/`](measurements/dinov3_encoder_study_20260913/)
— when citing, use that path. **If reproduction is needed, they have to be measured again** (the recipe is in each JSON).

---

## 0. Summary of conclusions

| Item | Conclusion |
|---|---|
| "Official/largest" DINOv3 | **ViT-7B/16** (6,716 M, 4096-d, the only from-scratch teacher). The rest — S/S+/B/L/H+ + ConvNeXt T/S/B/L — are all distilled from 7B. The largest distilled one is **ViT-H+/16** (840 M). [literature §1] |
| 7B on this machine | **Cannot be trained** (fp32 weights alone are 26.9 GB; the current recipe also keeps the EMA copy resident on the GPU). Only bf16 frozen inference is possible: 13.4 GB resident, 43 ms inference on 2 images at deployment, 1.75 s/step for a 128-image forward during training. [측정 §4] |
| H+ with the current recipe | Only up to batch **8** (28.0 GB, 0.7 GB headroom), 200 ep ≈ **46 h** (fp32) / 34 h (TF32) [유도]; checkpoints 7.3 GB × 21 = 153 GB > 87 GB of remaining disk. Unrealistic without bf16/grad-ckpt **code**. [측정 §4] |
| L with the current recipe | **OOM** at the actual batch 32 (64 images); batch 16 fp32 18.2 GB / 286 ms → 200 ep ≈ **16.5 h** (fp32) / 12.3 h (TF32) [유도]. Encoder alone with bf16+grad-ckpt: 8.0 GB at 128 images/step. [측정 §4] |
| Paper/benchmark view | The paper itself says "ViT-L comes close to the 7B teacher", and on dense tasks L ≈ H+ (ADE20k 54.9 vs 54.8, NYU 0.352 vs 0.352). For our encoder path, which uses the patch tokens wholesale, there is no evidence of an additional benefit from H+/7B. [literature §1] |
| Scope of code changes | The network graph is **config-only** (swap the timm name → the 1024→768 projection and the cond token count are automatic). The real blocker is **timm 0.9.7 in the training env `umi2`** (DINOv3 not registered) → upgrade to 1.0.29. The upgrade was confirmed **bit-for-bit harmless** for the existing CLIP/ResNet checkpoints. [측정 §3] |
| The observation is not RGB | The encoder input is `[inverse_depth, validity, inverse_depth]` u8/255, with no mean/std normalization. **The repo has no experiment** on how much of DINOv3's RGB prior survives this encoding, and in the literature (DeFM) the gain of DINOv3-L over DINOv2-L on depth input is only +0.45 pt. [§2, §5] |
| Verdict criterion | The failure mode is not held-out error but **land→water transfer of the yaw response**. In the new can 0913 closed loop, the current A (ViT-CLIP) has pooled r +0.116, slope +0.015 deg/deg (demonstrator ~+0.45) = effectively unresponsive. This is the baseline a DINOv3 run has to beat, and the tool (`policy_yaw_vs_can.py`) can be used as is. [측정 §6] |
| **Recommendation** | **Round 1: full fine-tuning of ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`)**, batch 16 fp32, or batch 32 with (code) bf16 autocast. 7B is possible only as "frozen features + head", and the literature signal that frozen giant models lose to fine-tuned mid-size models under depth and domain shift is consistent, so its experimental value is low. Before spending 16 h, the 3 preliminary checks in §7 are recommended. |

---

## 1. DINOv3 family facts (literature, verified)

Sources: Siméoni et al., arXiv 2508.10104 v1 (2025-08-13); github facebookresearch/dinov3; HF model card;
timm 1.0.29 source (`~/miniforge3/envs/rovgui-pose/.../timm/models/eva.py`).
[`agent_results/research_dinov3-facts.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_dinov3-facts.json),
[`followup_license-and-paper-facts.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_license-and-paper-facts.json).

| Model | Parameters | Width | Blocks | Notes |
|---|---|---|---|---|
| ViT-S/16 · S+/16 | 21 M · 29 M | 384 | 12 | distilled |
| ViT-B/16 | 86 M | 768 | 12 | distilled |
| **ViT-L/16** | 303 M | 1024 | 24 | distilled. Paper: "close to that of the original 7B teacher" |
| **ViT-H+/16** | 840 M | 1280 (SwiGLU) | 32 | largest distilled. Fig.16b "on par with the 8× larger ViT-7B" |
| **ViT-7B/16** | 6,716 M | 4096 (SwiGLU) | 40 | **the only teacher**, from scratch, 61,440 H100-h |
| ConvNeXt T/S/B/L | 29/50/89/198 M | – | – | distilled |

- Common: patch 16, **4 register tokens**, axial RoPE (no absolute pos-embed) → works without interpolation at any resolution that is a multiple of 16.
  224 input → 1 CLS + 4 reg + 196 patch = **201 tokens** (the current CLIP ViT-B has 197). Pretraining at 256, paper evaluation at 512.
- Data: LVD-1689M (k-means curation from a 17B Instagram pool) + a mix of public datasets; the SAT-493M satellite version exists only for L/7B and is unrelated to us.
- **Table 14 (frozen, L vs H+)**: ADE20k 54.9 vs 54.8, NYU depth 0.352 vs 0.352, DAVIS 79.9 vs 79.3, SPair 61.2 vs 56.3 — no H+ gain on dense tasks; where H+ leads is global classification (IN-R 88.1→90.0, ObjectNet 74.8→78.6).
  7B is in a separate table: ADE20k 55.9, NYU 0.309, ObjectNet 79.0 (the "91.1" in the initial research was a misreading — corrected in verification).
- First sentence of paper §6: "we keep DINOv3 frozen … finetuning is not necessary". Model card: "fine-tuning … last resort". → a recommendation **based on RGB natural images**.
- **License**: all models are under the "DINOv3 License" (Meta custom, commercial use allowed, no royalties). Obligations: on redistribution, the same agreement + a copy attached; **state the use of the DINO Materials in papers**; no reverse engineering; no military/ITAR use. The LICENSE.md (2025-08-19) published on HF·GitHub has **no** obligation to display "Built with DINOv3"; that appears only on the ai.meta.com page (2025-08-14).
- **HF gating**: `facebook/dinov3-*` is gated=manual (form + approval). The `timm/*dinov3*` mirror is **gated=False** — on this machine, without a token, `timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True)` **successfully downloaded 1,212 MB in 26.1 s** [측정: `agent_results/measure_rtx5090-bench.json` `pretrained_download_check`]. For reference, `~/.cache/huggingface/token` (2026-01-09) exists and `facebook/dinov3-vitb16` is already cached, so the account is already approved for the facebook gate too.
- Libraries: registered starting from timm ≥ 1.0.20 (2025-09-21) (including the `_qkvb` twins); transformers ≥ 4.56 has only `DINOv3ViTModel`, and `DINOv3ViTBackbone` only from 5.x.

## 2. What our encoder actually sees (close reading of the repo)

[`agent_results/understand_depth-obs.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_depth-obs.json)

- Stored observation: `(224,224,3)` u8 = `[inverse_depth, validity_mask, inverse_depth]`
  (`umi_handheld/build_dp_depth_zarr.py:82-84, 293-296`; identical at deployment in `rov_gui/perception/policy_obs.py:643-662`).
  ch0 = (1/Z − 1/3.0)/(1/0.2 − 1/3.0) clipped, near=1 / far·invalid=0.
- The dataset only applies `/255` (`umi_dataset.py:286`); the image normalizer is identity (`normalize_util.py:39-51`).
  The transforms are only RandomCrop(212)→Resize(224)→RandomRotation(±5°), **no ImageNet/CLIP mean-std** (`transformer_obs_encoder.py:171-178`).
  That is, the current CLIP is also already absorbing an "out-of-spec input" through fine-tuning.
- Training-set channel statistics [측정, 400-frame uniform sample, `dataset_depth.zarr.zip`]: ch0/ch2 mean 0.276 std 0.159; ch1 mean 1.0000 std 0.005 (effectively a constant 1; 99.94 % of frames have validity ≥ 0.985, `rov_gui/tools/dp_policy_out/training_validity_20260902.txt`). 4.3 % of valid pixels are at ≥3 m and clipped to 0.
- The DINOv3 timm cfg is ImageNet mean/std, 256 input. With fine-tuning, the patch-embed is linear, so a global affine shift can be absorbed [유도]; when **frozen**, it is a real distribution shift (ch0 x=0 pixels map to about +2.1 std, x=1 to −1.25 std).
- 224 is a value inherited from upstream UMI, with no independent evidence. Going to 256 takes a dataset rebuild (`--out-res 256`) + the task yaml + obs_res in `config/hw_mpc.yaml:691`/`backends/policy.py:143` — possible only with DINOv3 (CLIP-B is fixed at 224).

## 3. Code path: what is config-only and what is code

[`agent_results/understand_train-deploy-path.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_train-deploy-path.json),
[`followup_timm-upgrade-numerics.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_timm-upgrade-numerics.json),
[`followup_baseline-0913-yaw.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_baseline-0913-yaw.json) claim 3.

| Item | Status |
|---|---|
| Swapping `model_name` | **config-only.** `transformer_obs_encoder.py:91-96` `timm.create_model` → `startswith('vit')` branch (:139-146, all tokens) → if the width differs, `Linear(1024→768)` is created automatically (:191-200) → the policy sets the cond token count automatically via `output_shape()` (`cond_pos_emb` (1,411,768); the current A is (1,403,768)). Actual build and CUDA forward checked in rovgui-pose: output (1,410,768), 303.9 M. |
| Deployment loader | `dp_policy.py:672-681` rebuilds with `pretrained=False` from the model_name in the ckpt cfg, then does a strict load — **no backbone-name gate**. But if `frozen: True` is written in the yaml, the station load dies on an assert (:675-681 → `transformer_obs_encoder.py:99-100`). Freezing goes through `training.freeze_encoder`. |
| Register tokens | The 4 per frame go in as conditioning tokens as they are (not a blocker). Removing them takes an `aggregate_feature` change + a station pass-through (code). |
| Adding Normalize | Training: config; **deployment: mirroring code in `dp_policy.py:650-661`** needed + parity regeneration. |
| bf16 autocast / grad-ckpt | **Not present** in the workspace (`train_diffusion_transformer_timm_workspace.py:97` Accelerator has no mixed_precision; TF32 is also off). Using them takes code. |
| LoRA | peft/bitsandbytes not installed in either of the two envs. |
| **timm upgrade (umi2 0.9.7→1.0.29)** | The blocker, and safe: pip changes only timm (all dependencies unpinned, already satisfied); 173+40 module imports give identical results; with the existing A (ViT-CLIP ep195) and ResNet34 weights, forward/backward/BN buffers **max\|Δ\| = 0.0 (bit-identical)**. The only change: the seed-42 initialization stream of the ViT config (1.0.29 allocates pos_embed with `torch.empty`, so RNG consumption differs → 54/296 non-backbone tensors get different random initial values; statistically equivalent). |
| Disk | The `data/` volume has 87 GB free. ckpt (model+ema fp32) L ≈ 3.0 GB × 21 = 62 GB, H+ 7.3 × 21 = 153 GB, 7B 54 GB each → reducing `checkpoint.topk.k` is mandatory. |

## 4. RTX 5090 measurements (random-init, architecture cost)

### 4a. Encoder alone (`encoder_bench_rtx5090.csv`, 62 rows, 0 contaminated rows)

Conditions: rovgui-pose (torch 2.11+cu128, timm 1.0.29), 224², 3 warm-up + 10 timed, after confirming the GPU was idle.
"fp32" means TF32 allowed (the current workspace has TF32 off → in reality it is slower).

| Model | full-FT 128 images/step | grad-ckpt 128 images | frozen fwd 128 images (bf16) | inference 2 images (bf16 / fp32) |
|---|---|---|---|---|
| ViT-B CLIP (current) | 9.6 GB / 227 ms (fp32), 105 ms (bf16) | – | 0.6 GB / 31 ms | 1.6 / 1.9 ms |
| DINOv3-L | **OOM** (both fp32 and bf16) | 8.9 GB / 959 ms (fp32), 8.0 GB / 579 ms (bf16) | 1.9 GB / 141 ms | 6.4 / 5.9 ms (bf16 params 4.9) |
| DINOv3-H+ | **OOM** (OOM even with 32 images fp32) | 19.3 GB / 2,315 ms (fp32), 17.6 GB / 1,359 ms (bf16) | 6.7 GB / 328 ms | 11.9 / 12.8 ms (bf16 params 8.4) |
| DINOv3-7B | **OOM** (pure-bf16 32 images + ckpt) | – | 15.8 GB / 1,747 ms | 43.4 ms (bf16 params, 13.5 GB resident) |

### 4b. Whole policy graph, **actual training recipe** (`agent_results/followup_real-loop-vram-latency.json`)

Instantiated `train_diffusion_transformer_umi_depth_vit_workspace` + `umi_depth_5d` as is via hydra,
fp32 · no autocast · no grad-ckpt · EMA deepcopy resident on the GPU · AdamW · `EMAModel.step` included. The actual runs use batch **32** (64 images).

| Model | batch 32 | batch 24 | batch 16 | batch 8 | predict() DDIM 8, fp32 |
|---|---|---|---|---|---|
| ViT-B CLIP | 10.6 GB / 176.5 ms (harness ~6 % optimistic vs the real run's 97.3 s/epoch) | | | | 19.7 ms |
| DINOv3-L | **OOM** (failed at 29.0 GB) | 24.4 GB / 418 ms | 18.2 GB / 286 ms (TF32 213) | | **25.8 ms** (encoder 10.4 + denoiser 15.3) |
| DINOv3-H+ | **OOM** | | **OOM** | 28.0 GB / 403 ms (0.7 GB headroom) | 38.7 ms |

[유도] 200 ep at the same number of images/epoch: L@b16 ≈ 296 s/epoch → **16.5 h** (TF32 12.3 h); H+@b8 ≈ 835 s/epoch → **46 h** (TF32 34 h). The current ViT-B is ~5.4 h.
In fp32, SDPA cannot use flash (mem-efficient cutlass f32 + SIMT sgemm), while in bf16 flash is possible — bf16 autocast is a lever for both speed and memory.

## 5. What the literature says (only what passed verification)

[`agent_results/research_policy-encoders.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_policy-encoders.json),
[`research_depth-input.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_depth-input.json),
[`research_compute-recipes.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_compute-recipes.json).

- **Size scaling (UMI+DP, DINOv2)** — Lin et al. ICLR 2025 Table 2: full fine-tuning ViT-S 0.66 / B 0.81 / **L 0.90**; frozen 0.00; LoRA r8 0.72. But ~1,824 demos, effective batch 256 (8 GPUs), CLS pooling — a 25× difference from ours (62–75 episodes).
- **Freezing hurts (few demos)** — OpenVLA: frozen 47.0 % vs full FT 69.7 %, LoRA r32 68.2 %. Theia: naively fine-tuning DINOv2-L at lr 3e-4 loses badly to an 86 M model on 50 demos. The DINOv3-DP paper (2509.17684) covers only ViT-S at 76 px, and frozen ≪ fine-tuned.
- **Depth input** — DeFM (2601.18923): in a linear probe with depth fed as 3 channels, DINOv3-L 69.15 vs DINOv2-L 68.70 (**+0.45**), and for ViT-S DINOv3 is worse than DINOv2; frozen features collapse under a sensor-noise shift (0.65→0.21) but hold up with fine-tuning; "DINOv3 treats depth simply as another intensity channel". Tziafas (IROS 2023): depth-as-pseudo-RGB is −20 pt frozen, −13 pt after fine-tuning. Kabra (CVPR 2026): the depth features of frozen DINOv2 are almost unrelated to its RGB features, recovered by FT of the top 4 blocks.
- **Frozen 7B under domain shift** — chest X-ray (2510.07191): frozen 7B + MLP head trails fine-tuned ViT-B/ConvNeXt-B by 3.4–7.4 pp; LoRA is −1.3–5.1 pp relative to full FT.
- **OOD robustness of DINO-family ViTs** — Burns et al. CoRL 2024: of 15 frozen models, 6 of the top 7 on OOD are ViTs, and DINO ViTs beat manipulation-specific models; emergent segmentation is an OOD predictor. This is for **RGB appearance shift** and has never been reproduced on depth.
- DINOv3-L/H+/7B as policy encoders: **no paper compares them by size**. The H+ adoption case (Enfold) is 400–27,500 demos · 8–32 A100s.

## 6. This project's own evidence

[`agent_results/understand_prior-evidence.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_prior-evidence.json),
[`followup_baseline-0913-yaw.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_baseline-0913-yaw.json).

- E/A/C ablation (2026-09-10/11, 62 ep, batch 32, single seed 42): on held-out all three are similar at 17–21 mm · yaw corr 0.59–0.83; in water only A (ViT-CLIP) followed the can, in one run (173106). "The encoder is the bottleneck" is recorded as a **hypothesis based on one run**. ViT-B already overfits train/val 49× (train_action_mse 2e-5 vs val 9.8e-4).
- **New can 0913 closed-loop baseline [측정]**: A ep195 `model` weights, 8 folders, 609 plans (L/R 296/68): sign-agree 58 %, **r +0.116, slope +0.015 deg/deg** (demonstrator ~+0.45); 3 L/R-balanced runs (163921/170213/173702) pooled r +0.034, slope +0.004. → A is effectively unresponsive on the new can too. `policy_yaw_vs_can.py` runs with no code change given just the folder paths (but `mission_of()` reads only the first CSV, so for multi-engagement sessions the replay proprio is valid only for the first mission).
- The repo **has no pretrained-vs-scratch ablation on depth input**. The line in §4 of the plan document, "for depth input, CLIP pretraining gives almost no benefit" (`DP_TRAJECTORY_PLAN.ko.md:154-156`), has never been verified.

## 7. Recommendations and decisions

**Recommended experiment (minimal code)**
1. Install `timm==1.0.29` in `umi2` (confirmed safe).
2. `train_diffusion_transformer_umi_depth_vit_workspace.yaml`: `model_name: vit_large_patch16_dinov3.lvd1689m`, `pretrained: True`, `frozen: False`, reduce `checkpoint.topk.k` (≤ 8), `dataloader.batch_size: 16` (if staying in fp32). The rest is config-only.
3. The verdict is not held-out but **the `policy_yaw_vs_can` r/slope of L/R-balanced water runs on the new can**, compared with the 0913 A baseline. 2 or more seeds.

**Preliminary checks before spending 16 h (recommended)**
- (a) **RGB prior signal probe** [not run]: linear-probe knot-15 dyaw / a bearing proxy on the features of the frozen `vit_large_patch16_dinov3.lvd1689m` (already cached) vs a random-init of the same architecture — if the prior gives nothing on this depth encoding, it becomes "a bigger ViT on 62 episodes" and only the overfitting grows.
- (b) One run of a frozen/low-lr variant of the current CLIP ViT-B — separates the "pretrained-feature transfer" and "ViT architecture" effects (not in the repo).
- (c) Optional: bf16 autocast + a `set_grad_checkpointing(True)` workspace hook (code) — needed to run L at batch 32 and H+ at batch 16 or more.

**Open decisions (no evidence, a choice is needed)**
- Whether to keep the 4 register tokens in the conditioning (currently they stay automatically).
- Whether to add ImageNet Normalize — if added, the validity channel (constant 1) turns into a ≈+2.4 constant bias; repacking the channels is a dataset contract change.
- Keep 224 vs rebuild at 256.
- License: the obligation to state it in papers · the no-military-use clause · enclosing the agreement when redistributing trained ckpts — check at the project level.

**On 7B**: possible only as a frozen feature extractor (13.4 GB resident, +1.75 s per training step, +43 ms at deployment). Under depth and domain shift, frozen giant models fall behind fine-tuned mid-size models according to the literature, and that disadvantage is judged to be even larger given that this project's observation is not RGB. If one insists on testing the "most official" one, it is possible with a feature cache (201×4096 bf16 ≈ 1.6 MB per frame) + training a small head, but that is incompatible with our augmentation (RandomCrop/Rotation).

---

## 8. 2026-09-14 addition — variant D: DINOv3 ViT-B/16 (the proposer's intent was "base")

What the proposer wanted was not 7B but **ViT-B/16**. It is in the same size class as the current A (CLIP ViT-B/16) (86 M, 768-d, 12 blocks), so
A-vs-D is the cleanest A/B, isolating only the "pretraining prior (DINO vs CLIP)", and its training cost is also the same as A's.

Implementation (encoder swap only, no code change):
- Installed `timm==1.0.29` in `umi2` (pip replaced only timm; the bit-identity evidence is in §3).
- `external/UMI_aquatic/diffusion_policy/config/train_diffusion_transformer_umi_depth_dinov3b_workspace.yaml`
  = differs from the A yaml only in `model_name: vit_base_patch16_dinov3.lvd1689m` (the decisions are in the yaml comments).
- Runs via `tools/run_ablations.sh D_dinov3b` with the same overrides as A/E/C (batch 32, 200 ep, topk 10).

Verification [측정, 2026-09-14]:
- umi2 instantiate: 343 MB downloaded from the timm mirror in 10.1 s, pretrained load confirmed (qkv std 0.077), backbone `Eva` 85.6 M,
  `num_prefix_tokens` 5, projection `Identity()` (768=768), observation tokens (1, 410, 768), `cond_pos_emb` (1, 411, 768), policy 152.2 M.
- On the real dataset (9_9_26, batch 32), `training.debug=True` 2 epochs × 3 steps ran to completion, 3 ckpts saved, val computed
  (scratch `debug_run/`, volatile; the wandb offline run was deleted).
- Station loader (rovgui-pose, torch 2.11/timm 1.0.29): `DpPolicySession(weights="model")` strict-load OK,
  `describe()` model_name/n_obs_tokens 410, `predict()` (16, 5) **19.0–20.1 ms/call** (A: 19.7 ms).
  `encoder_batchnorm_kept: True` is a meaningless indicator for a ViT (same for A, cosmetic).

Remaining decisions (same as §7): no Normalize · 4 register tokens included · 224 kept — all left identical to A so that the A/B stays clean.
The verdict is the `policy_yaw_vs_can` of water runs on the new can (0913 A baseline r +0.116).

### 8a. 2026-09-15 — D training completed + held-out results [측정]

Run: `data/20260914/0914_120052_train_umi_depth_5d_depth5d_9926_D_dinov3b/` (batch 32, 200 ep; cut off by a SIGTERM at 15:36 and
resumed from ep110 `latest.ckpt` — so that a resume runs only the remaining epochs, the loop in `train_diffusion_transformer_timm_workspace.py`
was changed to `range(self.epoch, num_epochs)`; the Adam moments are not in the checkpoint, so there is a transient right after the resume). ep199 train_loss 0.0196.

held-out (62-ep split, episodes 5/40/47, 195 windows, stride 4, `model` weights, DDIM 8, center transform) —
`rov_gui/tools/dp_policy_offline.py replay --val-only`, logs `data/20260915/0915_131900_heldout_D_vs_A/{D_dinov3b,A_vitclip}_ep195.log`:

| ckpt ep195 | pos RMSE | yaw RMS (zero 4.00°) | yaw corr | knot-15 yaw corr | width RMSE |
|---|---|---|---|---|---|
| **D** DINOv3 ViT-B | 21.0 mm | 2.77° (0.69) | **+0.672** | +0.646 | 11.6 mm |
| **A** CLIP ViT-B | 21.1 mm | 2.81° (0.70) | +0.649 | +0.622 | 11.6 mm |
| E ResNet34-BN (2026-09-11 journal value, no artifact) | 17.7 mm | 2.03° | +0.83 | – | – |

A's figures reproduce the 2026-09-11 journal values (21.1 / 2.81 / +0.65) exactly — this log is A's first on-disk artifact.
**Conclusion: on held-out, D ≈ A (the difference is within seed noise, single seed).** Switching to the DINO prior did not change the land held-out.
As originally planned, the verdict has to come from the underwater yaw response (`policy_session_replay.py --ckpt` → `policy_yaw_vs_can`, A baseline r +0.116) — not run.
