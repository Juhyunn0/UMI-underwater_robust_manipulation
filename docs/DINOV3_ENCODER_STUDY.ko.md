# DINOv3를 Diffusion Policy 인코더로 쓰는 문제 — 조사·실측 보고 (2026-09-13)

질문: *"DP 인코더를 DINOv3로, small 말고 제일 정식(가장 큰) 버전으로 시도해 보고 싶다."*
코드는 쓰지 않았다. 리포 정독(3) + 웹 조사(4) + RTX 5090 실측(2종) + 후속 질문(4) +
주장별 적대 검증(60건: verified 33 / uncertain 14 / rejected 1 —
[`measurements/.../agent_results/verdicts.csv`](measurements/dinov3_encoder_study_20260913/agent_results/verdicts.csv)).
실측 스크립트는 세션 스크래치(휘발)에서 돌았고 원본 CSV/JSON은 세션 종료로 사라졌다.
아래 숫자는 에이전트가 구조화 출력으로 보고한 값(검증자가 당시 원본 파일을 읽고 대조)이며,
그 보고본을 [`docs/measurements/dinov3_encoder_study_20260913/`](measurements/dinov3_encoder_study_20260913/)에
보존했다 — 인용 시 그 경로를 쓴다. **재현이 필요하면 다시 재야 한다** (레시피는 각 JSON에 있다).

---

## 0. 결론 요약

| 항목 | 결론 |
|---|---|
| "정식/최대" DINOv3 | **ViT-7B/16** (6,716 M, 4096-d, 유일한 from-scratch teacher). 나머지 S/S+/B/L/H+ + ConvNeXt T/S/B/L은 전부 7B에서 증류. 증류 최대는 **ViT-H+/16** (840 M). [문헌 §1] |
| 7B를 이 장비에서 | **학습 불가** (fp32 가중치만 26.9 GB; 현재 레시피는 EMA 복제본까지 GPU 상주). bf16 동결 추론만 가능: 상주 13.4 GB, 배포 2장 추론 43 ms, 학습 시 128장 forward 1.75 s/step. [측정 §4] |
| H+를 현재 레시피로 | batch **8**까지만 (28.0 GB, 여유 0.7 GB), 200 ep ≈ **46 h**(fp32) / 34 h(TF32) [유도]; 체크포인트 7.3 GB × 21 = 153 GB > 남은 디스크 87 GB. bf16/grad-ckpt **코드** 없이는 비현실적. [측정 §4] |
| L을 현재 레시피로 | 실제 batch 32(64장)에서 **OOM**; batch 16 fp32 18.2 GB / 286 ms → 200 ep ≈ **16.5 h**(fp32) / 12.3 h(TF32) [유도]. 인코더만 bf16+grad-ckpt면 128장/step 8.0 GB. [측정 §4] |
| 논문/벤치 관점 | 논문 자체가 "ViT-L이 7B teacher에 근접"이라 말하고, 밀집(dense) 과제에서 L ≈ H+ (ADE20k 54.9 vs 54.8, NYU 0.352 vs 0.352). 패치 토큰을 통째로 쓰는 우리 인코더 경로엔 H+/7B의 추가 이득 근거가 없다. [문헌 §1] |
| 코드 변경 범위 | 네트워크 그래프는 **config-only** (timm 이름 교체 → 1024→768 projection·cond 토큰 수 자동). 진짜 블로커는 **학습 env `umi2`의 timm 0.9.7** (DINOv3 미등록) → 1.0.29 업그레이드. 업그레이드는 기존 CLIP/ResNet 체크포인트에 **비트 단위 무해** 확인. [측정 §3] |
| 관측이 RGB가 아님 | 인코더 입력은 `[inverse_depth, validity, inverse_depth]` u8/255, mean/std 정규화 없음. DINOv3의 RGB prior가 이 인코딩에 얼마나 남는지는 **리포에 실험이 없고**, 문헌(DeFM)은 depth 입력에서 DINOv3-L의 DINOv2-L 대비 이득이 +0.45 pt에 불과. [§2, §5] |
| 판정 기준 | 실패 모드는 held-out 오차가 아니라 **육상→수중 yaw 응답 전이**. 새 캔 0913 폐루프에서 현행 A(ViT-CLIP)는 pooled r +0.116, slope +0.015 deg/deg (시연자 ~+0.45) = 사실상 무반응. 이게 DINOv3 런이 이겨야 할 베이스라인이며 도구(`policy_yaw_vs_can.py`)는 그대로 쓸 수 있다. [측정 §6] |
| **권고** | **1차: ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`) 전체 파인튜닝**, batch 16 fp32 또는 (코드) bf16 autocast로 batch 32. 7B는 "동결 특징 + 헤드"로만 가능하며 depth·도메인 시프트에서 동결 거대 모델이 파인튜닝된 중형 모델에 진다는 문헌 신호가 일관돼 실험 가치가 낮다. 16 h를 쓰기 전에 §7의 선행 점검 3가지를 권한다. |

---

## 1. DINOv3 패밀리 사실 (문헌, 검증됨)

출처: Siméoni et al., arXiv 2508.10104 v1 (2025-08-13); github facebookresearch/dinov3; HF 모델 카드;
timm 1.0.29 소스(`~/miniforge3/envs/rovgui-pose/.../timm/models/eva.py`).
[`agent_results/research_dinov3-facts.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_dinov3-facts.json),
[`followup_license-and-paper-facts.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_license-and-paper-facts.json).

| 모델 | 파라미터 | 폭 | 블록 | 비고 |
|---|---|---|---|---|
| ViT-S/16 · S+/16 | 21 M · 29 M | 384 | 12 | 증류 |
| ViT-B/16 | 86 M | 768 | 12 | 증류 |
| **ViT-L/16** | 303 M | 1024 | 24 | 증류. 논문: "close to that of the original 7B teacher" |
| **ViT-H+/16** | 840 M | 1280 (SwiGLU) | 32 | 증류 최대. Fig.16b "on par with the 8× larger ViT-7B" |
| **ViT-7B/16** | 6,716 M | 4096 (SwiGLU) | 40 | **유일한 teacher**, from scratch, 61,440 H100-h |
| ConvNeXt T/S/B/L | 29/50/89/198 M | – | – | 증류 |

- 공통: patch 16, **레지스터 토큰 4개**, axial RoPE(절대 pos-embed 없음) → 16의 배수면 어떤 해상도든 보간 없이 동작.
  224 입력 → 1 CLS + 4 reg + 196 patch = **201 토큰** (현행 CLIP ViT-B는 197). 사전학습 256, 논문 평가 512.
- 데이터: LVD-1689M (Instagram 17B 풀에서 k-means 큐레이션) + 공개셋 혼합; SAT-493M 위성판은 L/7B만 있고 우리와 무관.
- **Table 14 (동결, L vs H+)**: ADE20k 54.9 vs 54.8, NYU depth 0.352 vs 0.352, DAVIS 79.9 vs 79.3, SPair 61.2 vs 56.3 — 밀집 과제에서 H+ 이득 없음; H+가 앞서는 건 전역 분류(IN-R 88.1→90.0, ObjectNet 74.8→78.6).
  7B는 별도 표: ADE20k 55.9, NYU 0.309, ObjectNet 79.0 (초기 조사의 "91.1"은 오독 — 검증에서 정정).
- 논문 §6 첫 문장: "we keep DINOv3 frozen … finetuning is not necessary". 모델 카드: "fine-tuning … last resort". → **RGB 자연영상 기준** 권고.
- **라이선스**: 전 모델 "DINOv3 License"(Meta 커스텀, 상업 사용 허용, 로열티 없음). 의무: 재배포 시 동일 계약+사본 첨부, **논문에 DINO Materials 사용 명기**, 리버스엔지니어링 금지, 군사/ITAR 용도 금지. HF·GitHub에 실린 LICENSE.md(2025-08-19)는 "Built with DINOv3" 표기 의무가 **없고**, ai.meta.com 페이지(2025-08-14)에만 있다.
- **HF 게이팅**: `facebook/dinov3-*`는 gated=manual(양식+승인). `timm/*dinov3*` 미러는 **gated=False** — 이 머신에서 `timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True)`가 토큰 없이 **26.1 s에 1,212 MB 다운로드 성공** [측정: `agent_results/measure_rtx5090-bench.json` `pretrained_download_check`]. 참고로 `~/.cache/huggingface/token`(2026-01-09)이 있고 `facebook/dinov3-vitb16`이 이미 캐시돼 있어 facebook 게이트도 이미 승인된 계정이다.
- 라이브러리: timm ≥ 1.0.20(2025-09-21)부터 등록(`_qkvb` 쌍둥이 포함); transformers ≥ 4.56은 `DINOv3ViTModel`만, `DINOv3ViTBackbone`은 5.x부터.

## 2. 우리 인코더가 실제로 보는 것 (리포 정독)

[`agent_results/understand_depth-obs.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_depth-obs.json)

- 저장 관측: `(224,224,3)` u8 = `[inverse_depth, validity_mask, inverse_depth]`
  (`umi_handheld/build_dp_depth_zarr.py:82-84, 293-296`; 배포 `rov_gui/perception/policy_obs.py:643-662` 동일).
  ch0 = (1/Z − 1/3.0)/(1/0.2 − 1/3.0) 클립, near=1 / far·invalid=0.
- 데이터셋에서 `/255`만 (`umi_dataset.py:286`), 이미지 normalizer는 항등 (`normalize_util.py:39-51`).
  변환은 RandomCrop(212)→Resize(224)→RandomRotation(±5°)뿐, **ImageNet/CLIP mean-std 없음** (`transformer_obs_encoder.py:171-178`).
  즉 현행 CLIP도 이미 "스펙 밖 입력"을 파인튜닝으로 흡수하고 있다.
- 학습셋 채널 통계 [측정, 400프레임 균등표본, `dataset_depth.zarr.zip`]: ch0/ch2 mean 0.276 std 0.159; ch1 mean 1.0000 std 0.005(사실상 상수 1; 99.94 % 프레임이 validity ≥ 0.985, `rov_gui/tools/dp_policy_out/training_validity_20260902.txt`). 유효 픽셀 4.3 %가 ≥3 m로 0에 클립.
- DINOv3 timm cfg는 ImageNet mean/std, 256 입력. 파인튜닝이면 patch-embed가 선형이라 전역 affine 시프트는 흡수 가능 [유도]; **동결**이면 실제 분포 시프트 (ch0 x=0 픽셀이 약 +2.1 std, x=1이 −1.25 std로 사상).
- 224는 상류 UMI에서 물려받은 값이고 독립 근거 없음. 256으로 가려면 데이터셋 재빌드(`--out-res 256`) + task yaml + `config/hw_mpc.yaml:691`/`backends/policy.py:143`의 obs_res — DINOv3만 가능(CLIP-B는 224 고정).

## 3. 코드 경로: 무엇이 config-only이고 무엇이 코드인가

[`agent_results/understand_train-deploy-path.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_train-deploy-path.json),
[`followup_timm-upgrade-numerics.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_timm-upgrade-numerics.json),
[`followup_baseline-0913-yaw.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_baseline-0913-yaw.json) claim 3.

| 항목 | 상태 |
|---|---|
| `model_name` 교체 | **config-only.** `transformer_obs_encoder.py:91-96` `timm.create_model` → `startswith('vit')` 분기(:139-146, 전 토큰) → 폭이 다르면 `Linear(1024→768)` 자동 생성(:191-200) → 정책이 `output_shape()`로 cond 토큰 수 자동(`cond_pos_emb` (1,411,768); 현 A는 (1,403,768)). rovgui-pose에서 실제 빌드·CUDA forward 확인: 출력 (1,410,768), 303.9 M. |
| 배포 로더 | `dp_policy.py:672-681`가 ckpt cfg의 model_name으로 `pretrained=False` 재구성 후 strict load — **백본 이름 게이트 없음**. 단 `frozen: True`를 yaml에 쓰면 스테이션 로드가 assert로 죽음(:675-681 → `transformer_obs_encoder.py:99-100`). 동결은 `training.freeze_encoder`로. |
| 레지스터 토큰 | 4개/프레임이 그대로 조건 토큰으로 들어감(블로커 아님). 떼려면 `aggregate_feature` 수정 + 스테이션 pass-through(코드). |
| Normalize 추가 | 학습은 config, **배포는 `dp_policy.py:650-661` 미러링 코드** 필요 + parity 재생성. |
| bf16 autocast / grad-ckpt | 워크스페이스에 **없음**(`train_diffusion_transformer_timm_workspace.py:97` Accelerator에 mixed_precision 없음; TF32도 꺼짐). 쓰려면 코드. |
| LoRA | peft/bitsandbytes 두 env 모두 미설치. |
| **timm 업그레이드 (umi2 0.9.7→1.0.29)** | 블로커이자 안전: pip는 timm 하나만 바꿈(의존성 전부 unpinned, 이미 충족); 173+40 모듈 import 결과 동일; 기존 A(ViT-CLIP ep195)·ResNet34 가중치로 forward/backward/BN 버퍼 **max\|Δ\| = 0.0 (비트 동일)**. 유일한 변화: ViT config의 seed-42 초기화 스트림(1.0.29는 pos_embed를 `torch.empty`로 잡아 RNG 소비가 달라짐 → 비백본 텐서 54/296이 다른 난수 초기값; 통계적으로 동등). |
| 디스크 | `data/` 볼륨 87 GB 여유. ckpt(model+ema fp32) L ≈ 3.0 GB × 21 = 62 GB, H+ 7.3 × 21 = 153 GB, 7B 54 GB/개 → `checkpoint.topk.k` 축소 필수. |

## 4. RTX 5090 실측 (random-init, 아키텍처 비용)

### 4a. 인코더 단독 (`encoder_bench_rtx5090.csv`, 62행, 오염 0행)

조건: rovgui-pose(torch 2.11+cu128, timm 1.0.29), 224², 3 warm-up + 10 timed, GPU 유휴 확인 후.
"fp32"는 TF32 허용(현 workspace는 TF32 꺼짐 → 실제는 더 느림).

| 모델 | full-FT 128장/step | grad-ckpt 128장 | 동결 fwd 128장 (bf16) | 추론 2장 (bf16 / fp32) |
|---|---|---|---|---|
| ViT-B CLIP (현행) | 9.6 GB / 227 ms (fp32), 105 ms (bf16) | – | 0.6 GB / 31 ms | 1.6 / 1.9 ms |
| DINOv3-L | **OOM** (fp32·bf16 모두) | 8.9 GB / 959 ms (fp32), 8.0 GB / 579 ms (bf16) | 1.9 GB / 141 ms | 6.4 / 5.9 ms (bf16 params 4.9) |
| DINOv3-H+ | **OOM** (32장 fp32도 OOM) | 19.3 GB / 2,315 ms (fp32), 17.6 GB / 1,359 ms (bf16) | 6.7 GB / 328 ms | 11.9 / 12.8 ms (bf16 params 8.4) |
| DINOv3-7B | **OOM** (pure-bf16 32장 + ckpt) | – | 15.8 GB / 1,747 ms | 43.4 ms (bf16 params, 상주 13.5 GB) |

### 4b. 정책 그래프 전체, **실제 학습 레시피** (`agent_results/followup_real-loop-vram-latency.json`)

hydra로 `train_diffusion_transformer_umi_depth_vit_workspace` + `umi_depth_5d`를 그대로 instantiate,
fp32·autocast 없음·grad-ckpt 없음·EMA deepcopy GPU 상주·AdamW·`EMAModel.step` 포함. 실제 런은 batch **32**(64장).

| 모델 | batch 32 | batch 24 | batch 16 | batch 8 | predict() DDIM 8, fp32 |
|---|---|---|---|---|---|
| ViT-B CLIP | 10.6 GB / 176.5 ms (실런 97.3 s/epoch 대비 하니스 ~6 % 낙관) | | | | 19.7 ms |
| DINOv3-L | **OOM** (29.0 GB에서 실패) | 24.4 GB / 418 ms | 18.2 GB / 286 ms (TF32 213) | | **25.8 ms** (인코더 10.4 + 디노이저 15.3) |
| DINOv3-H+ | **OOM** | | **OOM** | 28.0 GB / 403 ms (여유 0.7 GB) | 38.7 ms |

[유도] 같은 이미지 수/epoch 기준 200 ep: L@b16 ≈ 296 s/epoch → **16.5 h** (TF32 12.3 h); H+@b8 ≈ 835 s/epoch → **46 h** (TF32 34 h). 현 ViT-B는 ~5.4 h.
fp32에서는 SDPA가 flash를 못 쓰고(mem-efficient cutlass f32 + SIMT sgemm), bf16이면 flash 가능 — bf16 autocast가 속도·메모리 양쪽 레버.

## 5. 문헌이 말하는 것 (검증 통과분만)

[`agent_results/research_policy-encoders.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_policy-encoders.json),
[`research_depth-input.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_depth-input.json),
[`research_compute-recipes.json`](measurements/dinov3_encoder_study_20260913/agent_results/research_compute-recipes.json).

- **크기 스케일링(UMI+DP, DINOv2)** — Lin et al. ICLR 2025 Table 2: 전체 파인튜닝 ViT-S 0.66 / B 0.81 / **L 0.90**; 동결 0.00; LoRA r8 0.72. 단 ~1,824 데모, 유효 batch 256(8 GPU), CLS 풀링 — 우리(62–75 에피소드)와 25배 차이.
- **동결은 해롭다(소량 데모)** — OpenVLA: 동결 47.0 % vs 전체 FT 69.7 %, LoRA r32 68.2 %. Theia: DINOv2-L을 lr 3e-4로 순진하게 FT하면 50 데모에서 86 M 모델에 대패. DINOv3-DP 논문(2509.17684)은 ViT-S 76 px뿐이고 동결 ≪ 파인튜닝.
- **Depth 입력** — DeFM(2601.18923): depth를 3채널로 넣은 linear probe에서 DINOv3-L 69.15 vs DINOv2-L 68.70(**+0.45**), ViT-S는 DINOv3가 DINOv2보다 나쁨; 동결 특징은 센서 잡음 시프트에서 붕괴(0.65→0.21), 파인튜닝하면 유지; "DINOv3 treats depth simply as another intensity channel". Tziafas(IROS 2023): depth-as-pseudo-RGB는 동결 −20 pt, 파인튜닝 후 −13 pt. Kabra(CVPR 2026): 동결 DINOv2의 depth 특징은 RGB 특징과 거의 무관, 상위 4블록 FT로 회복.
- **도메인 시프트에서 동결 7B** — 흉부 X선(2510.07191): 동결 7B + MLP 헤드가 파인튜닝 ViT-B/ConvNeXt-B에 3.4–7.4 pp 뒤짐; LoRA는 전체 FT 대비 −1.3–5.1 pp.
- **DINO 계열 ViT의 OOD 강건성** — Burns et al. CoRL 2024: 동결 15개 중 OOD 상위 7개 중 6개가 ViT, DINO ViT가 조작 특화 모델을 이김; emergent segmentation이 OOD 예측자. **RGB 외형 시프트** 기준이며 depth에서 재현된 적 없음.
- DINOv3-L/H+/7B를 정책 인코더로 **크기 비교한 논문은 없다**. H+ 채용 사례(Enfold)는 400–27,500 데모·8–32 A100.

## 6. 이 프로젝트의 자체 근거

[`agent_results/understand_prior-evidence.json`](measurements/dinov3_encoder_study_20260913/agent_results/understand_prior-evidence.json),
[`followup_baseline-0913-yaw.json`](measurements/dinov3_encoder_study_20260913/agent_results/followup_baseline-0913-yaw.json).

- E/A/C 절제(2026-09-10/11, 62 ep, batch 32, seed 42 단일): held-out은 셋 다 17–21 mm·yaw corr 0.59–0.83로 비슷; 수중에선 A(ViT-CLIP)만 한 런(173106)에서 캔을 추종. "인코더가 병목"은 **한 런 근거의 가설**로 기록돼 있다. ViT-B는 이미 train/val 49× 과적합(train_action_mse 2e-5 vs val 9.8e-4).
- **새 캔 0913 폐루프 베이스라인 [측정]**: A ep195 `model` 가중치, 8 폴더 609 플랜(L/R 296/68): sign-agree 58 %, **r +0.116, slope +0.015 deg/deg** (시연자 ~+0.45); L/R 균형 3런(163921/170213/173702) pooled r +0.034, slope +0.004. → 새 캔에서도 A는 사실상 무반응. `policy_yaw_vs_can.py`는 폴더 경로만 주면 코드 변경 없이 돈다(단 `mission_of()`가 첫 CSV만 읽어 다중 engagement 세션의 리플레이 proprio는 첫 미션만 유효).
- 리포에 **사전학습 vs scratch 절제가 depth 입력에서 없다**. 계획서 §4의 "depth 입력엔 CLIP 사전학습 이득 거의 없음"(`DP_TRAJECTORY_PLAN.ko.md:154-156`)은 검증된 적이 없다.

## 7. 권고와 결정 사항

**권고 실험(코드 최소)**
1. `umi2`에 `timm==1.0.29` 설치(안전 확인됨).
2. `train_diffusion_transformer_umi_depth_vit_workspace.yaml`: `model_name: vit_large_patch16_dinov3.lvd1689m`, `pretrained: True`, `frozen: False`, `checkpoint.topk.k` 축소(≤ 8), `dataloader.batch_size: 16`(fp32 그대로면). 나머지 config-only.
3. 판정은 held-out이 아니라 **새 캔 L/R 균형 수중 런의 `policy_yaw_vs_can` r/slope**와 0913 A 베이스라인 비교. seed 2개 이상.

**16 h를 쓰기 전 선행 점검(권장)**
- (a) **RGB prior 신호 프로브** [미실행]: 동결 `vit_large_patch16_dinov3.lvd1689m`(이미 캐시됨) vs 같은 구조 random-init의 특징으로 knot-15 dyaw / 베어링 프록시를 linear probe — prior가 이 depth 인코딩에서 아무것도 안 주면 "더 큰 ViT를 62 에피소드에"가 되어 과적합만 커진다.
- (b) 현행 CLIP ViT-B의 동결/저lr 변형 한 런 — "사전학습 특징 전이"와 "ViT 구조" 효과 분리(리포에 없음).
- (c) 선택: bf16 autocast + `set_grad_checkpointing(True)` 워크스페이스 훅(코드) — L을 batch 32에서, H+를 batch 16 이상에서 돌리려면 필요.

**열린 결정(근거 없음, 선택 필요)**
- 레지스터 4토큰을 조건에 남길지(현재 자동으로 남음).
- ImageNet Normalize 추가 여부 — 추가하면 validity 채널(상수 1)이 ≈+2.4 상수 편향으로 바뀜; 채널 재포장은 데이터셋 계약 변경.
- 224 유지 vs 256 재빌드.
- 라이선스: 논문 명기 의무·군사용 금지 조항·학습 ckpt 재배포 시 계약 동봉 — 프로젝트 차원 확인.

**7B에 대해**: 동결 특징 추출기로만 가능(13.4 GB 상주, 학습 step당 +1.75 s, 배포 +43 ms). depth·도메인 시프트에서 동결 거대 모델은 문헌상 파인튜닝 중형 모델에 밀리며, 이 프로젝트의 관측이 RGB가 아니라는 점에서 그 불리함이 더 크다고 본다. "가장 정식"을 굳이 시험하려면 특징 캐시(프레임당 201×4096 bf16 ≈ 1.6 MB) + 작은 헤드 학습으로 가능하지만 우리 증강(RandomCrop/Rotation)과 양립하지 않는다.

---

## 8. 2026-09-14 추가 — 변형 D: DINOv3 ViT-B/16 (제안자 의도는 "base")

제안자가 원한 것은 7B가 아니라 **ViT-B/16**이었다. 현행 A(CLIP ViT-B/16)와 같은 체급(86 M, 768-d, 12 블록)이라
A-vs-D는 "사전학습 prior(DINO vs CLIP)"만 분리하는 가장 깨끗한 A/B이고, 학습 비용도 A와 같다.

구현(인코더만 교체, 코드 변경 없음):
- `umi2`에 `timm==1.0.29` 설치 (pip는 timm만 교체; §3의 비트 동일성 근거).
- `external/UMI_aquatic/diffusion_policy/config/train_diffusion_transformer_umi_depth_dinov3b_workspace.yaml`
  = A yaml에서 `model_name: vit_base_patch16_dinov3.lvd1689m`만 다름(결정 사항은 yaml 주석에).
- `tools/run_ablations.sh D_dinov3b` 로 A/E/C와 같은 오버라이드(batch 32, 200 ep, topk 10)로 실행.

검증 [측정, 2026-09-14]:
- umi2 instantiate: timm 미러에서 343 MB 다운로드 10.1 s, 사전학습 로드 확인(qkv std 0.077), backbone `Eva` 85.6 M,
  `num_prefix_tokens` 5, projection `Identity()`(768=768), 관측 토큰 (1, 410, 768), `cond_pos_emb` (1, 411, 768), 정책 152.2 M.
- 실제 데이터셋(9_9_26, batch 32)으로 `training.debug=True` 2 epoch × 3 step 완주, ckpt 3개 저장, val 계산됨
  (스크래치 `debug_run/`, 휘발; wandb offline 런은 삭제).
- 스테이션 로더(rovgui-pose, torch 2.11/timm 1.0.29): `DpPolicySession(weights="model")` strict-load OK,
  `describe()` model_name/n_obs_tokens 410, `predict()` (16, 5) **19.0–20.1 ms/call** (A: 19.7 ms).
  `encoder_batchnorm_kept: True`는 ViT에서 무의미한 표시(A도 동일, 코스메틱).

남은 결정(§7과 동일): Normalize 없음·레지스터 4토큰 포함·224 유지 — 전부 A와 동일하게 두어 A/B가 깨끗하도록 했다.
판정은 새 캔 수중 런의 `policy_yaw_vs_can` (0913 A 베이스라인 r +0.116).

### 8a. 2026-09-15 — D 학습 완료 + held-out 결과 [측정]

런: `data/20260914/0914_120052_train_umi_depth_5d_depth5d_9926_D_dinov3b/` (batch 32, 200 ep; 15:36 SIGTERM으로 끊겨
ep110 `latest.ckpt`에서 재개 — 재개 시 남은 epoch만 돌도록 `train_diffusion_transformer_timm_workspace.py`의
루프를 `range(self.epoch, num_epochs)`로 수정; Adam 모멘트는 체크포인트에 없어 재개 직후 과도 있음). ep199 train_loss 0.0196.

held-out (62-ep 분할 episodes 5/40/47, 195 windows, stride 4, `model` 가중치, DDIM 8, center transform) —
`rov_gui/tools/dp_policy_offline.py replay --val-only`, 로그 `data/20260915/0915_131900_heldout_D_vs_A/{D_dinov3b,A_vitclip}_ep195.log`:

| ckpt ep195 | pos RMSE | yaw RMS (zero 4.00°) | yaw corr | knot-15 yaw corr | width RMSE |
|---|---|---|---|---|---|
| **D** DINOv3 ViT-B | 21.0 mm | 2.77° (0.69) | **+0.672** | +0.646 | 11.6 mm |
| **A** CLIP ViT-B | 21.1 mm | 2.81° (0.70) | +0.649 | +0.622 | 11.6 mm |
| E ResNet34-BN (2026-09-11 저널값, 아티팩트 없음) | 17.7 mm | 2.03° | +0.83 | – | – |

A의 수치는 2026-09-11 저널값(21.1 / 2.81 / +0.65)을 그대로 재현 — 이번 로그가 A의 첫 디스크 아티팩트다.
**결론: held-out에서 D ≈ A (차이는 seed 잡음 범위, 단일 seed).** DINO prior로 바꿔도 육상 held-out은 달라지지 않았다.
판정은 원래 계획대로 수중 yaw 응답(`policy_session_replay.py --ckpt` → `policy_yaw_vs_can`, A 베이스라인 r +0.116)으로 해야 한다 — 미실행.
