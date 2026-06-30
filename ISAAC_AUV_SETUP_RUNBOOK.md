# Isaac-AUV-Env (Learning to Swim) — 설치/실행 Runbook

> 대상 코드: `bluerov2_issac_paper/` (= github.com/warplab/isaac-auv-env, "Learning to Swim",
> Cai/Chang/Girdhar, ICRA 2025, arXiv:2410.00120)
> 작성 기준 하드웨어: **RTX 5090 (Blackwell, sm_120) · Ubuntu 22.04.5 · driver 595.71.05**
>
> ⚠️ 이 문서는 **실행 순서 가이드**다. 아직 설치를 수행하지 않았다. 단계별로 승인 후 진행한다.
> 시스템 변경 명령(apt/pip/다운로드)은 사용자 승인 전까지 실행 금지.

---

## ✅ 최종 작동 구성 (2026-06-23 검증 — RTX 5090에서 학습 완주)

WarpAUV 정책 **2048 envs · 400 iter · ~19.7M steps GPU 학습 완료**(보상 0.78→~76, 에러 0). 4개 Blackwell 블로커를 순서대로 해결:

1. **드라이버 = `nvidia-driver-570-open` 570.211.01 (CUDA 12.8)** — ⭐핵심. 580/595(CUDA 13)는 PhysX GPU init 행. 570이 Isaac Sim 5.0(CUDA 12.8 빌드)와 매칭. (apt 파일충돌 시: `sudo apt --fix-broken install -o Dpkg::Options::="--force-overwrite"`)
2. **warp 1.8.1** — `…/isaacsim/extscache/omni.warp.core-1.7.1+lx64/warp/`를 warp-lang 1.8.1 **실복사본**으로 교체(심볼릭 링크 금지; 백업 `warp.1.7.1.bak`). cuDeviceGetUuid 버그 해결.
3. **`amd_iommu=off`** — GRUB에 추가(IOMMU P2P 검증 hang 우회).
4. **PhysX GPU init 행** → (1)의 드라이버 570으로 해결(IOMMU-off·EAGER로는 안 됨).

**학습 실행:**
```bash
cd ~/IsaacLab && source ~/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 2048 --headless
```
**학습 정책 재생/평가 (자기 학습분, logs/rsl_rl/warpauv_direct/<run> 자동 탐색):**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-WarpAUV-Direct-v1 --num_envs 32
```
> 진단 팁: GPU PhysX가 의심되면 `--device cpu`로 돌려보면 부팅·학습이 되므로 문제를 GPU PhysX로 격리 가능.

아래 §0~§12는 그 과정의 상세 기록(드라이버는 580으로 적힌 부분이 있으나 **최종 정답은 570**).

---

## 0. 핵심 결정 — 어떤 Isaac Sim 버전을 깔 것인가

> ✅ **확정 (2026-06-22): 경로 A — Isaac Sim 5.0 + Isaac Lab 2.2.0 (Python 3.11).**
> RTX 5090(Blackwell) 안정성 우선. 경로 B(4.5.0)는 논문 정확 재현용 폴백으로만 남김.


| 경로 | 스택 | RTX 5090 적합성 | 비고 |
|---|---|---|---|
| **A (권장)** | Isaac Sim **5.0** + Isaac Lab **2.2.0** | ✅ Blackwell 공식 지원 (torch 2.7.0+cu128, sm_120 커널 포함) | Python **3.11**. 논문이 명시한 "4.5.0"과 다름 → 4.5용 task 코드가 5.0에서 그대로 로드되는지 1차 확인 필요 |
| **B (논문 정확 재현)** | Isaac Sim **4.5.0** + Isaac Lab **2.2.0** | ⚠️ 번들 torch 2.5.1은 sm_90까지 → 그대로면 `no kernel image` 크래시. torch nightly cu128로 교체해야 텐서 연산만 통과. Kit/PhysX/Warp 레벨 sm_120 부재는 안 풀림(렌더 아티팩트·불안정 가능) | Python **3.10**. README가 검증한 정확 스택 |

> 두 경로의 명령은 **Isaac Sim 버전/다운로드 URL·Python 버전**만 다르고 나머지(Isaac Lab clone,
> symlink, task 등록, 학습/평가)는 동일하다. 아래는 **경로 A(5.0)** 기준이며, 경로 B 차이는
> 각 단계 끝의 `▸ 경로 B` 노트로 표기.

---

## 1. 사전 준비 — conda 빠져나오기 + apt 의존성

```bash
# robust 등 어떤 conda 환경에도 있지 않아야 함 (README 권고). 'robust'는 절대 재사용 안 함.
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null   # base까지 빠지려면 2번

# IsaacLab가 robomimic 빌드에 필요로 함
sudo apt update
sudo apt install -y cmake build-essential
```

---

## 2. Isaac Sim 5.0 바이너리 설치

```bash
# 설치 위치 제안: 홈 디렉토리 (디스크 625GB 여유로 충분)
mkdir -p ~/isaacsim && cd ~/isaacsim
# ── 다운로드: 아래 공식 페이지에서 Linux x86_64 zip을 받는다 (정확 파일명/URL은 다운로드 시 확인) ──
#   • Isaac Sim 다운로드/릴리스: https://github.com/isaac-sim/IsaacSim/releases  (5.0은 오픈소스/바이너리 동시 제공)
#   • 또는 NVIDIA 공식: https://docs.isaacsim.omniverse.nvidia.com/  (Download 섹션)
# 받은 zip 압축 해제:
#   unzip ~/Downloads/isaac-sim-<버전>-linux-x86_64.zip -d ~/isaacsim
# 설치 후 동작 확인 (GUI는 데스크톱 세션에서):
~/isaacsim/isaac-sim.sh        # 처음 실행 시 셰이더 캐시 컴파일로 수 분 소요 가능
```

> `▸ 경로 B (4.5.0)`: `https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html`
> 에서 4.5.0 Linux zip을 받아 `~/isaacsim`에 푼다. 나머지 동일.
>
> ⚠️ **확인 필요(추측 금지)**: 5.0 바이너리의 정확한 zip 파일명/URL은 위 릴리스 페이지에서 직접 확인.
> 5.0은 GitHub 오픈소스 빌드도 있으니, "바이너리 zip"이 맞는지(소스 빌드 아님) 받을 때 구분할 것.

---

## 3. Isaac Lab 2.2.0 설치 (clone → symlink → install)

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.2.0
cd ~/IsaacLab

# Isaac Sim 바이너리를 _isaac_sim 으로 soft-link
ln -s ~/isaacsim _isaac_sim
#   ※ 만약 zip이 ~/isaacsim/isaac-sim-5.0.x/ 같은 하위 폴더로 풀렸다면 그 폴더를 가리키게:
#      ln -s ~/isaacsim/isaac-sim-5.0.0 _isaac_sim   (kit/ 폴더가 보이는 레벨이 맞다)

# 번들 python pip 업그레이드 후 IsaacLab 설치
_isaac_sim/kit/python/bin/python3 -m pip install --upgrade pip
./isaaclab.sh --install            # rsl_rl만 원하면: ./isaaclab.sh --install rsl_rl
```

> `▸ 경로 B (4.5.0)`: `--branch v2.2.0` 그대로(README가 2.2.0을 검증). 단 canonical 짝은 2.1.0이므로,
> 만약 2.2.0이 4.5.0에서 import 에러를 내면 `git checkout v2.1.0` 폴백 고려.

---

## 4. RTX 5090(sm_120) PyTorch 점검 — **Blackwell 게이트**

```bash
# IsaacLab가 설치한 torch가 sm_120 커널을 갖는지 확인
_isaac_sim/kit/python/bin/python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("arch_list:", torch.cuda.get_arch_list())   # 여기에 'sm_120' 이 있어야 함
PY
```

- **경로 A (5.0)**: torch 2.7.0+cu128 → `arch_list`에 `sm_120` **있음**이 정상. 없으면 아래 nightly로 교체.
- **경로 B (4.5.0)**: 번들 torch 2.5.1 → `sm_120` **없음**(정상적 결함). 반드시 교체:

```bash
# (경로 B 전용, 또는 경로 A에서 sm_120이 없을 때) torch를 cu128 빌드로 교체
_isaac_sim/kit/python/bin/python3 -m pip install --upgrade --pre \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# 교체 후 위 점검 스크립트 재실행 → arch_list에 sm_120 확인
```

> ⚠️ 경로 B는 torch를 고쳐도 Kit/PhysX/Warp/NVRTC의 sm_120 부재가 남아 **viewport 렌더 아티팩트
> (fuzzy/voxel body, noisy shadow)** 나 일부 크래시가 보고됨. 학습은 `--headless`로 렌더를 끄면
> 우회 가능성이 있으나 **보장 안 됨**. 안정성을 원하면 경로 A.

---

## 5. 설치 검증

```bash
cd ~/IsaacLab
# 헤드리스(현재 셸은 tty라 이게 안전):
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# GUI 창 확인은 실제 데스크톱(gnome) 세션에서:
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

성공 시 시뮬레이터가 뜨고(헤드리스면 콘솔 로그만) 에러 없이 종료된다.

---

## 6. 이 task 패키지를 IsaacLab에 등록 (symlink)

> 핵심: gym entry_point가 `isaaclab_tasks.direct.isaac-auv-env:WarpAUVEnv` 이므로
> **링크 이름은 반드시 `isaac-auv-env`** 여야 한다 (현재 폴더명 `bluerov2_issac_paper` 아님,
> README docker 스텝의 오타 `isaac-warpauv-env`도 아님).

```bash
cd ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
ln -s /home/bdml/Desktop/umi_underwater_robust_control/bluerov2_issac_paper isaac-auv-env

# 등록 확인 (task id가 보여야 함)
cd ~/IsaacLab
./isaaclab.sh -p scripts/environments/list_envs.py 2>/dev/null | grep -i warpauv
# 또는 학습을 바로 1 iter 돌려 등록 여부 확인 (아래 7)
```

> play 스크립트가 추가로 쓰는 패키지(번들 env에 없을 수 있음): `pandas`, `opencv-python(cv2)`.
> play 단계에서 ImportError가 나면:
> `_isaac_sim/kit/python/bin/python3 -m pip install pandas opencv-python`

---

## 7. 학습 실행

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-WarpAUV-Direct-v1 \
    --num_envs 2048 \
    --headless
```

- 체크포인트 저장 위치: `logs/rsl_rl/warpauv_direct/<timestamp>/`
- README 기준: 2048 envs로 **~400 iter**에 수렴, mean reward ~95–100. 수렴 문제 시 action penalty↓.
- `experiment_name = "warpauv_direct"` (agents/rsl_rl_ppo_cfg.py), max_iterations=400, num_steps_per_env=24.

### num_envs 가이드 (VRAM 32GB)

| VRAM | 권장 --num_envs |
|---|---|
| 32 GB (내 PC) | **2048 기본**; 모델이 64×64 MLP·obs 17·act 6로 매우 가벼워 4096~8192도 시도 가능 |
| 8–12 GB | 1024 또는 512 |
| OOM 발생 | 절반씩 감소 (2048→1024→512). 디스플레이 없으면 항상 `--headless` |

> 참고: env cfg의 scene 기본은 `num_envs=4`(warpauv_env.py:67)지만 CLI `--num_envs`가 덮어쓴다.

---

## 8. 학습 정책 시각화 / 평가 (play / eval)

### 8-1. IsaacLab 표준 play (자기 학습 정책용)
```bash
# 내가 7단계로 학습한 정책을 재생 (logs/rsl_rl/warpauv_direct/<run> 레이아웃을 자동 탐색)
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-WarpAUV-Direct-v1 --num_envs 32
# (특정 run/checkpoint 지정: --load_run <run> --checkpoint <model.pt>)
```

### 8-2. 리포 제공 사전학습 가중치 재생 (position-hold, 제공된 .pt)
> 제공 weights는 `logs/...` 레이아웃이 아니라 **단일 파일**이라 표준 play.py가 자동으로 못 찾는다.
> 이 파일 전용 스크립트는 `play_poshold.py`이며 `--play_checkpoint`로 **직접 경로**를 받는다.
> cv2 키보드 teleop(w/a/s/d/r/f, i/j/k/l/o/u)이라 **GUI 창 필요(헤드리스 불가)** → 데스크톱 세션에서 실행.

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/custom_workflows/play_poshold.py \
    --task Isaac-WarpAUV-Direct-v1 \
    --play_checkpoint /home/bdml/Desktop/umi_underwater_robust_control/bluerov2_issac_paper/weights/2024-09-13_20-15-03_poshold_DR_2.pt
```

### 8-3. 정량 평가 / 논문 그림용 스크립트
- `custom_workflows/play_eval.py` — 12개 axis-direction sweep, des vs true 속도·MSE·reward를 `source/results/.../logs.csv`로 저장 (pandas 필요). `get_checkpoint_path(load_run, load_checkpoint)` 방식 → `logs/rsl_rl/warpauv_direct/<run>` 레이아웃 필요.
- `play_eval_for_publication_{pos,vel}.py`, `plot_metrics.py` — 논문 그림 재현용.

---

## 9. Migration 4.0 → 4.5 반영 점검 결과 (코드 직접 확인 — **이미 완료됨**)

| 항목 | 상태 | 근거 |
|---|---|---|
| `observation_space` / `action_space` / `state_space` 명시 | ✅ 반영됨 | `warpauv_env.py:70-72` — Box(17), Box(6, ±1), Box(17) |
| `articulation_enabled=False` (assets) | ✅ 반영됨 | `assets/warpauv.py:17-19` (RigidObjectCfg 안) |
| 종합 | **fully-applied** | 추가 작업 불필요 |

> 단, 5.0(경로 A)에서는 4.5→5.0 추가 마이그레이션이 필요할 수 있음 → 7단계 1-iter 스모크 테스트로
> import/API 에러를 먼저 확인. 에러 시 IsaacLab 5.0 migration 노트 참조.

---

## 10. 사용자 참고정보 대비 정정 사항 (실제 코드 기준)

1. **Task id** `Isaac-WarpAUV-Direct-v1` — ✅ 사용자 정보와 일치 (`__init__.py:20`).
2. **symlink 이름** = `isaac-auv-env` (하이픈) 고정. 폴더명 `bluerov2_issac_paper` 그대로 링크하면 안 됨.
3. **requirements.txt 없음** — repo에 의존성 파일 전무. deps는 IsaacLab 환경 + play용 `pandas`/`opencv-python`.
4. **차량은 WARPAUV** (22.7kg, 6-thruster fully-actuated) — 폴더명 'bluerov2'는 로컬 라벨일 뿐 BlueROV2가 아님.
5. **Isaac Lab 2.2.0의 공식 짝은 Isaac Sim 5.0** — 4.5는 backward-compat. README가 2.2.0+4.5.0을 검증했으니
   그 조합 자체는 유효하나, RTX 5090에선 5.0이 정도(正道).

---

## 11. 더 확인이 필요한(추측하지 않은) 항목

- [ ] Isaac Sim **5.0 바이너리 zip의 정확한 파일명/URL** (릴리스 페이지에서 직접 확인).
- [ ] 4.5용 task 코드가 **Isaac Sim 5.0에서 무수정 로드되는지** (7단계 1-iter 스모크로 검증).
- [ ] 경로 A에서 IsaacLab 2.2.0이 까는 torch가 실제 `sm_120` 포함인지 (4단계 점검).
- [ ] play_poshold.py의 cv2 창이 현재 GPU 렌더 스택에서 정상 동작하는지(경로 B면 아티팩트 위험).

---

## 12. ⛔ 실제 설치 결과 + 드라이버 블로커 (2026-06-22 진행 기록)

**완료된 것 (pip route, env `env_isaaclab` Py3.11):**
- ✅ `isaacsim==5.0.0.0` + `torch 2.7.0+cu128` (RTX 5090 `sm_120` matmul 검증됨)
- ✅ IsaacLab 2.2.0 clone, 확장 5개 + `rsl-rl-lib==2.3.3` (apt 우회: `setuptools<81` 제약으로 flatdict 빌드)
- ✅ task 심볼릭 링크 `direct/isaac-auv-env`, EULA 동의(`OMNI_KIT_ACCEPT_EULA=YES`)

**막힌 것:** Isaac Sim 기동 시 번들 `omni.warp.core 1.7.1`이
`Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'` (CUDA error 36) → **부팅 데드락**.
- 원인: **NVIDIA 드라이버 595.71.05**가 Isaac Sim Warp와 비호환 (알려진 버그: IsaacSim #537 "595 실패/580 동작", IsaacLab #3477, Warp #940). 소프트웨어 우회 없음.
- torch cu128(sm_120)은 595에서 정상 → 순수 RL 텐서 연산은 OK, **Isaac Sim 시뮬레이터만** 막힘.

### 12-1. 해결: NVIDIA 드라이버 595 → 580 다운그레이드 (sudo + 재부팅, 시스템 전역)

> 580도 RTX 5090(sm_120) + torch cu128 + MuJoCo-MJX(jax cuda12)를 지원하므로 기존 작업은 안전.
> ⚠️ 시스템 전역/재부팅이라 신중히. 아래는 **확인 → 설치 → 검증** 순서.

```bash
# (1) 현재 상태 & 사용 가능한 580 패키지 확인 (설치 전, 비파괴)
nvidia-smi --query-gpu=driver_version --format=csv,noheader
ubuntu-drivers list 2>/dev/null | grep -i 580 || apt-cache search '^nvidia-driver-580' 2>/dev/null
apt list --all-versions 'nvidia-driver-580*' 2>/dev/null | grep -i 580

# (2) 580 설치 — 이 PC는 현재 nvidia-driver-595-OPEN(open 커널모듈)이므로 같은 계열로:
sudo apt install nvidia-driver-580-open
#   ↳ 후보가 공식 580.159.03-0ubuntu0.22.04.1(jammy-updates/security)이면 이상적.
#     apt 플랜에서 nvidia-*-595* 가 제거되고 nvidia-*-580-open 이 설치되는지 확인 후 진행.
#     특정 버전 고정을 원하면: sudo apt install nvidia-driver-580-open=580.159.03-0ubuntu0.22.04.1

# (3) 재부팅
sudo reboot

# (4) 재부팅 후 검증
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # 580.xx 떠야 함
# env_isaaclab에서 torch가 여전히 sm_120 보는지
/home/bdml/miniforge3/envs/env_isaaclab/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

### 12-2. RTX 5090 기동을 막은 3개 블로커 (순서대로, 각각이 다음을 가림)

**① 드라이버 595 → 580** (12-1, 사용자 sudo+재부팅 완료). 595(R590)는 Blackwell에서 알려진 불량.

**② Warp 스왑 (완료).** 번들 `omni.warp.core 1.7.1`의 warp 1.7.1이 `cuDeviceGetUuid`/CUDA error 36 버그
(warp<1.8.1 × R580). **warp-lang 1.8.1**로 교체 — 픽스 + sm_120 커널 + `warp.types.array`(omni 1.7.1 글루 호환)
모두 만족하는 유일 버전. **심볼릭 링크 금지(Kit이 깨뜨림) → 실제 복사**. 적용한 명령:
```bash
SP=/home/bdml/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages
EXT=$SP/isaacsim/extscache/omni.warp.core-1.7.1+lx64
/home/bdml/miniforge3/envs/env_isaaclab/bin/pip install --no-deps warp-lang==1.8.1   # pip warp도 1.8.1로
mv "$EXT/warp" "$EXT/warp.1.7.1.bak"          # 원본 백업
cp -a "$SP/warp" "$EXT/warp"                    # 1.8.1 실제 복사
grep -m1 version "$EXT/warp/config.py"          # → 1.8.1 확인
```

**③ IOMMU CUDA-P2P 검증 hang (진짜 최종 원인).** Kit이 "Simulation App Startup Complete"까지 가서
`gpu.foundation.plugin: IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation`
에서 stall (IsaacLab #1764; 베어메탈 Linux는 IOMMU-on P2P 미지원, ~306–362s @0.2GB/s). 이 PC는 AMD +
`/proc/cmdline`에 iommu off 없음. **해결 = GRUB에 `amd_iommu=off` 추가:**
```bash
# /etc/default/grub 의 GRUB_CMDLINE_LINUX_DEFAULT 에 amd_iommu=off 추가
sudo sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 amd_iommu=off"/' /etc/default/grub
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub   # 확인
sudo update-grub
sudo reboot
# 재부팅 후: cat /proc/cmdline | grep amd_iommu=off  (적용 확인)
```
> 대안(재부팅 회피): IOMMU off 안 해도 **매 부팅마다 ~5–6분 P2P 검증을 기다리면** 통과한다(무한 행 아님).
> 단 부팅마다 6분 세금이라 반복 작업엔 비효율 → 학습을 길게 돌릴 거면 그냥 기다려도 되고, 편의를 위해선 IOMMU off 권장.

### 12-3. 위 3개 해결 후 — 학습까지 (Claude 자동 진행)

```bash
cd /home/bdml/IsaacLab && source /home/bdml/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
# (A) 기동 재검증
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# (B) WarpAUV 스모크 (4.5 코드가 5.0에서 로드되는지, 2 iter)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 16 --max_iterations 2 --headless
# (C) 본 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 2048 --headless
```

> 참고(비치명): 부팅 로그의 `errno=28 / No space left on device`(change watch)는 디스크가 아니라
> **inotify watch 한도(기본 65536) 초과** 경고다. 없애려면(선택, sudo):
> `sudo sysctl fs.inotify.max_user_watches=524288`. 학습엔 영향 없음.
