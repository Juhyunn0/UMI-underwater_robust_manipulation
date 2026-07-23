# 06 — Environment & portability

## Historical: macOS drafting (CPU, base MuJoCo)

> **Update (2026-06-14):** the project has transferred to the Linux + RTX 5090
> machine. The runtime now lives in the conda envs described under
> "Linux + RTX 5090 runtime (Phase 0 — DONE)" below. The macOS notes in this
> section are kept for history only.

We are drafting on a MacBook (no NVIDIA GPU). Everything here runs on CPU with
**base `mujoco`** and stays portable to the Linux/GPU runtime.

- **venv:** repo-root `.venv` (i.e. `UMI-underwater_robust_manipulation/.venv`),
  Python 3.13, `mujoco` 3.9.x. Activate: `source .venv/bin/activate` from the repo
  root.
- **Runtime deps (to load/run the model):** `mujoco` + `numpy` only. All of
  `bluerov.xml`, `thrusters.py`, `hydro.py`, `teleop.py`, `test_*.py` need just
  these.
- **Build-time-only deps (to regenerate the asset from the USD):** `usd-core`,
  `trimesh`, `fast-simplification`. Used by `tools/extract_meshes.py` /
  `tools/generate_bluerov_xml.py`. CPU-only, nothing CUDA. **Not** needed to run the sim.
- **Do NOT install** `mujoco-mjx`, `jax[cuda*]`, or anything CUDA on macOS — those
  are for the Linux machine (Phase 0). MJX has no CPU benefit here.
- **Do NOT run MarineGym** (it needs Isaac Sim). We only consume its **static
  files** from the `external/MarineGym/` git submodule (URDF/USD/meshes/yaml).

Run anything with plain `python <script>.py` from `bluerov2_mujoco_marinegym/`.

## ⚠ Gotcha: a space in the project path breaks `mjpython`

The absolute path contains spaces:
`/Users/.../Claude/Projects/UMI-underwater Robust manipulation/...`. macOS's
`mjpython` (required by MuJoCo's **passive** viewer `launch_passive`) does not
handle the spaced path, so `mjpython` is effectively unusable here.

Workarounds (in order of preference):
1. **Use the managed viewer** `mujoco.viewer.launch(...)` with plain `python` — it
   runs the GUI on the main thread and works fine. This is what `teleop.py` does
   (managed viewer + a keyboard thread); see [05_TELEOP.md](05_TELEOP.md).
   Also `python -m mujoco.viewer --mjcf=bluerov.xml` works for a static look.
2. If you *must* use `launch_passive`/`mjpython` (you currently don't), copy the
   folder to a **no-space** path (e.g. `/tmp/bluerov`), make a venv there, and run
   from it.

General shell hygiene: quote the path, and prefer absolute paths in scripts
(`os.path.dirname(__file__)`), which the code already does.

## Linux + RTX 5090 runtime (Phase 0 — DONE 2026-06-14)

The real runs happen on **Ubuntu 22.04 + NVIDIA RTX 5090 (Blackwell)**, driver
**595.71.05** (CUDA 13.2). The runtime is split across **two conda envs** because
the perception stack and GPU JAX impose contradictory numpy ranges:

| env | Python | numpy | role | key packages |
|---|---|---|---|---|
| **`robust`** | 3.14 | 1.26.4 (<2) | CPU sim + perception/SLAM + MPC | base `mujoco` 3.9, gtsam 4.2.1, pyzed, casadi, opencv |
| **`robust-mjx`** | 3.12 | 2.4.6 | GPU MJX rollouts + RL (Phases 7–8) | jax 0.10.1 + jaxlib (cuda12, bundled `nvidia-*-cu12` 12.9 wheels), mujoco-mjx 3.9 |

**Why two envs:** GPU `jax[cuda12]` hard-requires `numpy>=2`, but `robust`'s
**gtsam 4.2.1** is ABI-pinned to `numpy<2`. One env can't hold both, so GPU/MJX
lives in its own `robust-mjx`. Do **not** `pip install jax[cuda12]` / `mujoco-mjx`
into `robust` — it force-upgrades numpy and breaks gtsam.

How it was installed (already done):

```bash
conda create -n robust-mjx python=3.12 -y
# use the ABSOLUTE interpreter path (see gotcha below):
/home/bdml/miniforge3/envs/robust-mjx/bin/python -m pip install -U "jax[cuda12]"
/home/bdml/miniforge3/envs/robust-mjx/bin/python -m pip install -U mujoco mujoco-mjx
```

**Verified** by `bluerov2_mujoco_marinegym/verify/verify_gpu_mjx.py` (run with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`): `jax.default_backend()=='gpu'`,
`jax.devices()==[CudaDevice(id=0)]` = NVIDIA GeForce RTX 5090; a tiny MJX rollout
**and** the canonical `bluerov.xml` both step on GPU with finite states. (Benign
import warning `Failed to import warp/mujoco_warp` — mujoco-mjx's optional
NVIDIA-Warp backend is absent; we use the JAX backend.)

**Gotcha:** the VS Code terminal auto-activates `robust`, which shadows other envs,
so `conda run -n robust-mjx python …` can silently resolve to robust's 3.14
interpreter. For GPU work always invoke the **absolute** path
`/home/bdml/miniforge3/envs/robust-mjx/bin/python`.

**Still pending for MJX:** the CPU passive-callback hydro (`hydro.py`,
`set_mjcb_passive`) does **not** run under MJX, so MJX rollouts currently move the
rigid body + thrusters but not the Fossen hydrodynamics. Re-expressing the hydro
in JAX is the next GPU task (Phase 3-on-GPU); the model/physics are otherwise ready.

## Portability — what carries over unchanged vs not

| artifact | portable as-is? |
|---|---|
| `bluerov.xml` + `meshes/` + `marinegym_assets/` | **Yes** — load identically on Linux/MJX |
| `thrusters.py` (curve, allocation, command helpers) | Yes (pure numpy) |
| Coefficients + **FLU sign conventions** | Yes — identical on both |
| `hydro.py` (CPU passive callback) | **No** — `set_mjcb_passive` Python callbacks don't run under MJX; the **same Fossen equations + MarineGym coefficients must be re-expressed in JAX**. Keep this in mind for Phase 0/3-on-GPU. |
| Viewer/teleop | macOS-tuned (managed viewer); the viewer story differs on Linux but the model is the same |

When re-verifying on Linux, re-run the same `test_*.py` (they're CPU MuJoCo) to
confirm the MJCF/meshes behave identically, then port the hydro to JAX for MJX.
