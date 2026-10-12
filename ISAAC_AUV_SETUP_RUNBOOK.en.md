# Isaac-AUV-Env (Learning to Swim) — Install/Run Runbook

> **English translation.** Source of truth: [`ISAAC_AUV_SETUP_RUNBOOK.md`](ISAAC_AUV_SETUP_RUNBOOK.md) (Korean). Snapshot of 2026-10-12, source blob `89872210e6`; later edits to the original are not reflected here.
>
> Provenance tags are kept as in the original: `[측정 …]` measured (artifact cited) · `[유도 …]` derived · `[예측 …]` predicted, not measured · `[스펙 …]` vendor spec · `실측` measured · `미실측` not measured · `산출물 없음` no artifact found · `예시/유도값` example or derived value.

> Target code: `bluerov2_issac_paper/` (= github.com/warplab/isaac-auv-env, "Learning to Swim",
> Cai/Chang/Girdhar, ICRA 2025, arXiv:2410.00120)
> Hardware this doc was written against: **RTX 5090 (Blackwell, sm_120) · Ubuntu 22.04.5 · driver 595.71.05**
>
> ⚠️ This document is an **execution-order guide**. The installation has not been performed yet. Proceed step by step, each step after approval.
> Do not run system-changing commands (apt/pip/download) until the user approves.

---

## ✅ Final working configuration (verified 2026-06-23 — training ran to completion on the RTX 5090)

WarpAUV policy: **2048 envs · 400 iter · ~19.7M steps of GPU training done** (reward 0.78→~76, errors 0). Resolved 4 Blackwell blockers in order:  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]

1. **Driver = `nvidia-driver-570-open` 570.211.01 (CUDA 12.8)** — ⭐the key. 580/595 (CUDA 13) hang at PhysX GPU init. 570 matches Isaac Sim 5.0 (a CUDA 12.8 build). (On an apt file conflict: `sudo apt --fix-broken install -o Dpkg::Options::="--force-overwrite"`)
2. **warp 1.8.1** — replaced `…/isaacsim/extscache/omni.warp.core-1.7.1+lx64/warp/` with a **real copy** of warp-lang 1.8.1 (do not symlink; backup `warp.1.7.1.bak`). Fixes the cuDeviceGetUuid bug.
3. **`amd_iommu=off`** — added to GRUB (works around the IOMMU P2P validation hang).
4. **PhysX GPU init hang** → fixed by the driver 570 from (1) (IOMMU-off·EAGER did not fix it).

**Run training:**
```bash
cd ~/IsaacLab && source ~/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 2048 --headless
```
**Replay/evaluate the trained policy (your own training runs; auto-discovers logs/rsl_rl/warpauv_direct/<run>):**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-WarpAUV-Direct-v1 --num_envs 32
```
> Diagnostic tip: if you suspect GPU PhysX, try running with `--device cpu` — it boots and trains, so you can isolate the problem to GPU PhysX.

§0~§12 below are the detailed record of that process (some parts give the driver as 580, but **the final answer is 570**).

---

## 0. Key decision — which Isaac Sim version to install

> ✅ **Decided (2026-06-22): Route A — Isaac Sim 5.0 + Isaac Lab 2.2.0 (Python 3.11).**
> Stability on the RTX 5090 (Blackwell) comes first. Route B (4.5.0) is kept only as a fallback for exact reproduction of the paper.


| Route | Stack | RTX 5090 suitability | Notes |
|---|---|---|---|
| **A (recommended)** | Isaac Sim **5.0** + Isaac Lab **2.2.0** | ✅ Official Blackwell support (torch 2.7.0+cu128, sm_120 kernels included) | Python **3.11**. Differs from the "4.5.0" the paper specifies → needs a 1st-pass check that the task code written for 4.5 loads as-is on 5.0 |
| **B (exact paper reproduction)** | Isaac Sim **4.5.0** + Isaac Lab **2.2.0** | ⚠️ The bundled torch 2.5.1 only goes up to sm_90 → as-is, it crashes with `no kernel image`. You must swap in torch nightly cu128, and even then only the tensor ops pass. The missing sm_120 at the Kit/PhysX/Warp level is not resolved (render artifacts / instability possible) | Python **3.10**. The exact stack the README verified |

> The commands for the two routes differ only in the **Isaac Sim version/download URL · Python version**; the rest (Isaac Lab clone,
> symlink, task registration, training/evaluation) is identical. The steps below follow **Route A (5.0)**; Route B differences are
> marked with a `▸ 경로 B` ("▸ Route B") note at the end of each step.

---

## 1. Preparation — leave conda + apt dependencies

```bash
# Must not be inside any conda env, robust included (README recommendation). Never reuse 'robust'.
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null   # 2 times, to get out of base as well

# IsaacLab needs these to build robomimic
sudo apt update
sudo apt install -y cmake build-essential
```

---

## 2. Install the Isaac Sim 5.0 binary

```bash
# Suggested install location: home directory (625GB of free disk space is enough)
mkdir -p ~/isaacsim && cd ~/isaacsim
# ── Download: get the Linux x86_64 zip from the official pages below (check the exact file name/URL when downloading) ──
#   • Isaac Sim downloads/releases: https://github.com/isaac-sim/IsaacSim/releases  (5.0 is offered as both open source and binary)
#   • or NVIDIA official: https://docs.isaacsim.omniverse.nvidia.com/  (Download section)
# Extract the downloaded zip:
#   unzip ~/Downloads/isaac-sim-<version>-linux-x86_64.zip -d ~/isaacsim
# After installing, check that it works (GUI from a desktop session):
~/isaacsim/isaac-sim.sh        # the first launch may take several minutes compiling the shader cache
```

> `▸ 경로 B (4.5.0)` ("▸ Route B (4.5.0)"): from `https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html`
> get the 4.5.0 Linux zip and extract it into `~/isaacsim`. The rest is the same.
>
> ⚠️ **Needs checking (no guessing)**: check the exact zip file name/URL of the 5.0 binary directly on the release page above.
> 5.0 also has a GitHub open-source build, so when downloading, make sure it is the "binary zip" (not a source build).

---

## 3. Install Isaac Lab 2.2.0 (clone → symlink → install)

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.2.0
cd ~/IsaacLab

# soft-link the Isaac Sim binary as _isaac_sim
ln -s ~/isaacsim _isaac_sim
#   ※ if the zip extracted into a subfolder such as ~/isaacsim/isaac-sim-5.0.x/, point the link at that folder:
#      ln -s ~/isaacsim/isaac-sim-5.0.0 _isaac_sim   (the right level is the one where the kit/ folder is visible)

# upgrade the bundled python's pip, then install IsaacLab
_isaac_sim/kit/python/bin/python3 -m pip install --upgrade pip
./isaaclab.sh --install            # if you only want rsl_rl: ./isaaclab.sh --install rsl_rl
```

> `▸ 경로 B (4.5.0)` ("▸ Route B (4.5.0)"): keep `--branch v2.2.0` (the README verified 2.2.0). But the canonical pairing is 2.1.0, so
> if 2.2.0 throws import errors on 4.5.0, consider falling back with `git checkout v2.1.0`.

---

## 4. RTX 5090 (sm_120) PyTorch check — **Blackwell gate**

```bash
# check that the torch IsaacLab installed has sm_120 kernels
_isaac_sim/kit/python/bin/python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("arch_list:", torch.cuda.get_arch_list())   # 'sm_120' must be in here
PY
```

- **Route A (5.0)**: torch 2.7.0+cu128 → `sm_120` being **present** in `arch_list` is normal. If it is absent, replace torch with the nightly below.
- **Route B (4.5.0)**: bundled torch 2.5.1 → `sm_120` **absent** (a normal defect). Must be replaced:

```bash
# (Route B only, or on Route A when sm_120 is missing) replace torch with the cu128 build
_isaac_sim/kit/python/bin/python3 -m pip install --upgrade --pre \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# after replacing, rerun the check script above → confirm sm_120 is in arch_list
```

> ⚠️ On Route B, even with torch fixed, the missing sm_120 in Kit/PhysX/Warp/NVRTC remains, and **viewport render artifacts
> (fuzzy/voxel body, noisy shadow)** or some crashes have been reported. For training, turning rendering off with `--headless`
> may work around this, but it is **not guaranteed**. If you want stability, go with Route A.

---

## 5. Verify the installation

```bash
cd ~/IsaacLab
# headless (the current shell is a tty, so this is the safe one):
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# check the GUI window from a real desktop (gnome) session:
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

On success, the simulator comes up (headless: console log only) and exits without errors.

---

## 6. Register this task package with IsaacLab (symlink)

> Key point: the gym entry_point is `isaaclab_tasks.direct.isaac-auv-env:WarpAUVEnv`, so
> **the link name must be `isaac-auv-env`** (not the current folder name `bluerov2_issac_paper`,
> and not `isaac-warpauv-env`, the typo in the README docker step, either).

```bash
cd ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/
ln -s /home/bdml/Desktop/umi_underwater_robust_control/bluerov2_issac_paper isaac-auv-env

# check the registration (the task id must show up)
cd ~/IsaacLab
./isaaclab.sh -p scripts/environments/list_envs.py 2>/dev/null | grep -i warpauv
# or run training for 1 iter right away to check whether it is registered (7 below)
```

> Packages the play scripts additionally use (may be missing from the bundled env): `pandas`, `opencv-python(cv2)`.
> If you get an ImportError at the play step:
> `_isaac_sim/kit/python/bin/python3 -m pip install pandas opencv-python`

---

## 7. Run training

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-WarpAUV-Direct-v1 \
    --num_envs 2048 \
    --headless
```

- Checkpoint save location: `logs/rsl_rl/warpauv_direct/<timestamp>/`
- Per the README: with 2048 envs it converges in **~400 iter**, mean reward ~95–100. On convergence problems, action penalty↓.  [UNVERIFIED: 산출물 없음 — docs/MEASUREMENT_AUDIT.md]
- `experiment_name = "warpauv_direct"` (agents/rsl_rl_ppo_cfg.py), max_iterations=400, num_steps_per_env=24.

### num_envs guide (VRAM 32GB)

| VRAM | Recommended --num_envs |
|---|---|
| 32 GB (my PC) | **2048 default**; the model is very light (64×64 MLP·obs 17·act 6), so 4096~8192 can also be tried |
| 8–12 GB | 1024 or 512 |
| On OOM | Halve it each time (2048→1024→512). With no display, always `--headless` |

> Note: the env cfg's scene default is `num_envs=4` (warpauv_env.py:67), but the CLI `--num_envs` overrides it.

---

## 8. Visualize / evaluate the trained policy (play / eval)

### 8-1. Standard IsaacLab play (for your own trained policy)
```bash
# replay the policy I trained in step 7 (auto-discovers the logs/rsl_rl/warpauv_direct/<run> layout)
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-WarpAUV-Direct-v1 --num_envs 32
# (to specify a particular run/checkpoint: --load_run <run> --checkpoint <model.pt>)
```

### 8-2. Replay the repo-provided pretrained weights (position-hold, the provided .pt)
> The provided weights are a **single file**, not the `logs/...` layout, so the standard play.py cannot find them automatically.
> The script dedicated to this file is `play_poshold.py`, which takes a **direct path** via `--play_checkpoint`.
> It uses cv2 keyboard teleop (w/a/s/d/r/f, i/j/k/l/o/u), so it **needs a GUI window (headless not possible)** → run it in a desktop session.

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/custom_workflows/play_poshold.py \
    --task Isaac-WarpAUV-Direct-v1 \
    --play_checkpoint /home/bdml/Desktop/umi_underwater_robust_control/bluerov2_issac_paper/weights/2024-09-13_20-15-03_poshold_DR_2.pt
```

### 8-3. Scripts for quantitative evaluation / the paper's figures
- `custom_workflows/play_eval.py` — a sweep of 12 axis-directions; saves des vs true velocity·MSE·reward to `source/results/.../logs.csv` (needs pandas). Uses the `get_checkpoint_path(load_run, load_checkpoint)` approach → needs the `logs/rsl_rl/warpauv_direct/<run>` layout.
- `play_eval_for_publication_{pos,vel}.py`, `plot_metrics.py` — for reproducing the paper's figures.

---

## 9. Results of checking that Migration 4.0 → 4.5 is applied (checked directly in the code — **already done**)

| Item | Status | Evidence |
|---|---|---|
| `observation_space` / `action_space` / `state_space` declared explicitly | ✅ Applied | `warpauv_env.py:70-72` — Box(17), Box(6, ±1), Box(17) |
| `articulation_enabled=False` (assets) | ✅ Applied | `assets/warpauv.py:17-19` (inside RigidObjectCfg) |
| Synthesis | **fully-applied** | No further work needed |

> However, on 5.0 (Route A) an additional 4.5→5.0 migration may be needed → use the step-7 1-iter smoke test
> to check for import/API errors first. On errors, see the IsaacLab 5.0 migration notes.

---

## 10. Corrections relative to the user's reference info (based on the actual code)

1. **Task id** `Isaac-WarpAUV-Direct-v1` — ✅ matches the user's info (`__init__.py:20`).
2. **symlink name** = `isaac-auv-env` (hyphens), fixed. Do not link it under the folder name `bluerov2_issac_paper` as is.
3. **No requirements.txt** — there is no dependency file in the repo at all. deps are the IsaacLab environment + `pandas`/`opencv-python` for play.
4. **The vehicle is WARPAUV** (22.7kg, 6-thruster fully-actuated) — the folder name 'bluerov2' is just a local label; it is not a BlueROV2.
5. **Isaac Lab 2.2.0's official pairing is Isaac Sim 5.0** — 4.5 is backward-compat. The README verified 2.2.0+4.5.0, so
   that combination itself is valid, but on the RTX 5090, 5.0 is the proper way.

---

## 11. Items that still need checking (not guessed)

- [ ] Isaac Sim **5.0 binary zip: exact file name/URL** (check directly on the release page).
- [ ] Whether the task code written for 4.5 **loads unmodified on Isaac Sim 5.0** (verify with the step-7 1-iter smoke test).
- [ ] Whether the torch that IsaacLab 2.2.0 installs on Route A actually includes `sm_120` (step-4 check).
- [ ] Whether play_poshold.py's cv2 window works correctly on the current GPU render stack (risk of artifacts on Route B).

---

## 12. ⛔ Actual installation results + driver blocker (2026-06-22 progress log)

**What is done (pip route, env `env_isaaclab` Py3.11):**
- ✅ `isaacsim==5.0.0.0` + `torch 2.7.0+cu128` (RTX 5090 `sm_120` matmul verified)
- ✅ IsaacLab 2.2.0 clone, 5 extensions + `rsl-rl-lib==2.3.3` (bypassing apt: built flatdict under a `setuptools<81` constraint)
- ✅ task symlink `direct/isaac-auv-env`, EULA accepted (`OMNI_KIT_ACCEPT_EULA=YES`)

**What is blocked:** on Isaac Sim startup, the bundled `omni.warp.core 1.7.1` hits
`Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'` (CUDA error 36) → **boot deadlock**.
- Cause: **NVIDIA driver 595.71.05** is incompatible with Isaac Sim Warp (known bug: IsaacSim #537 "595 fails/580 works", IsaacLab #3477, Warp #940). No way to work around it in software.
- torch cu128 (sm_120) works fine on 595 → pure RL tensor ops are OK; **only the Isaac Sim simulator** is blocked.

### 12-1. Fix: downgrade the NVIDIA driver 595 → 580 (sudo + reboot, system-wide)

> 580 also supports RTX 5090 (sm_120) + torch cu128 + MuJoCo-MJX (jax cuda12), so existing work is safe.
> ⚠️ It is system-wide and needs a reboot, so be careful. Below, the order is **check → install → verify**.

```bash
# (1) check the current state & the available 580 packages (before installing, non-destructive)
nvidia-smi --query-gpu=driver_version --format=csv,noheader
ubuntu-drivers list 2>/dev/null | grep -i 580 || apt-cache search '^nvidia-driver-580' 2>/dev/null
apt list --all-versions 'nvidia-driver-580*' 2>/dev/null | grep -i 580

# (2) install 580 — this PC currently has nvidia-driver-595-OPEN (open kernel module), so stay in the same family:
sudo apt install nvidia-driver-580-open
#   ↳ ideal if the candidate is the official 580.159.03-0ubuntu0.22.04.1 (jammy-updates/security).
#     proceed after checking that the apt plan removes nvidia-*-595* and installs nvidia-*-580-open.
#     to pin a specific version: sudo apt install nvidia-driver-580-open=580.159.03-0ubuntu0.22.04.1

# (3) reboot
sudo reboot

# (4) verify after the reboot
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # must show 580.xx
# whether torch in env_isaaclab still sees sm_120
/home/bdml/miniforge3/envs/env_isaaclab/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

### 12-2. The 3 blockers that prevented startup on the RTX 5090 (in order; each one hid the next)

**① Driver 595 → 580** (12-1; sudo+reboot done by the user). 595 (R590) is known to be faulty on Blackwell.

**② Warp swap (done).** The warp 1.7.1 in the bundled `omni.warp.core 1.7.1` has the `cuDeviceGetUuid`/CUDA error 36 bug
(warp<1.8.1 × R580). Replaced with **warp-lang 1.8.1** — the only version that satisfies all of: the fix + sm_120 kernels + `warp.types.array` (compatible with the omni 1.7.1 glue).
**Do not symlink (Kit breaks it) → make a real copy**. Commands applied:
```bash
SP=/home/bdml/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages
EXT=$SP/isaacsim/extscache/omni.warp.core-1.7.1+lx64
/home/bdml/miniforge3/envs/env_isaaclab/bin/pip install --no-deps warp-lang==1.8.1   # pip's warp to 1.8.1 as well
mv "$EXT/warp" "$EXT/warp.1.7.1.bak"          # back up the original
cp -a "$SP/warp" "$EXT/warp"                    # real copy of 1.8.1
grep -m1 version "$EXT/warp/config.py"          # → confirm 1.8.1
```

**③ IOMMU CUDA-P2P validation hang (the real final cause).** Kit gets as far as "Simulation App Startup Complete" and then
stalls at `gpu.foundation.plugin: IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation`
(IsaacLab #1764; bare-metal Linux does not support IOMMU-on P2P, ~306–362s @0.2GB/s). This PC is AMD +
has no iommu off in `/proc/cmdline`. **Fix = add `amd_iommu=off` to GRUB:**
```bash
# add amd_iommu=off to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub
sudo sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 amd_iommu=off"/' /etc/default/grub
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub   # check
sudo update-grub
sudo reboot
# after the reboot: cat /proc/cmdline | grep amd_iommu=off  (confirm it is applied)
```
> Alternative (avoids the reboot): even without turning IOMMU off, it gets through **if you wait out the ~5–6 min P2P validation on every boot** (not an infinite hang).  [UNVERIFIED: 예시/유도값 — docs/MEASUREMENT_AUDIT.md]
> But that is a 6-minute tax on every boot, inefficient for repeated work → if you are going to run training for a long time you can just wait; for convenience, IOMMU off is recommended.

### 12-3. After resolving the 3 above — through to training (Claude proceeds automatically)

```bash
cd /home/bdml/IsaacLab && source /home/bdml/miniforge3/etc/profile.d/conda.sh && conda activate env_isaaclab && export OMNI_KIT_ACCEPT_EULA=YES
# (A) re-verify startup
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
# (B) WarpAUV smoke (whether the 4.5 code loads on 5.0; 2 iter)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 16 --max_iterations 2 --headless
# (C) the main training run
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-WarpAUV-Direct-v1 --num_envs 2048 --headless
```

> Note (non-critical): the `errno=28 / No space left on device` (change watch) in the boot log is not about the disk; it is
> a warning that the **inotify watch limit (default 65536) was exceeded**. To get rid of it (optional, sudo):
> `sudo sysctl fs.inotify.max_user_watches=524288`. No effect on training.


<!-- MEASUREMENT AUDIT (2026-08-04): the following numbers appear inside code
     blocks above and could not be annotated inline without corrupting them. -->

> **[UNVERIFIED]** The figures on the line below were classified as unverified by the provenance audit — details in [docs/MEASUREMENT_AUDIT.md](docs/MEASUREMENT_AUDIT.en.md).
>
> - `ISAAC_AUV_SETUP_RUNBOOK.md:69` — 산출물 없음 — `# 설치 위치 제안: 홈 디렉토리 (디스크 625GB 여유로 충분)` ("# Suggested install location: home directory (625GB of free disk space is enough)")
