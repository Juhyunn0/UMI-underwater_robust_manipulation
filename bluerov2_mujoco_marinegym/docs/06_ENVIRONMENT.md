# 06 — Environment & portability

## Current: macOS drafting (CPU, base MuJoCo)

We are drafting on a MacBook (no NVIDIA GPU). Everything here runs on CPU with
**base `mujoco`** and stays portable to the Linux/GPU runtime.

- **venv:** repo-root `.venv` (i.e. `UMI-underwater_robust_manipulation/.venv`),
  Python 3.13, `mujoco` 3.9.x. Activate: `source .venv/bin/activate` from the repo
  root.
- **Runtime deps (to load/run the model):** `mujoco` + `numpy` only. All of
  `bluerov.xml`, `thrusters.py`, `hydro.py`, `teleop.py`, `test_*.py` need just
  these.
- **Build-time-only deps (to regenerate the asset from the USD):** `usd-core`,
  `trimesh`, `fast-simplification`. Used by `extract_meshes.py` /
  `generate_bluerov_xml.py`. CPU-only, nothing CUDA. **Not** needed to run the sim.
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

## Planned: Linux + RTX 5090 runtime (Phase 0 — PENDING)

The real runs happen on **Ubuntu 22.04 + NVIDIA RTX 5090 (Blackwell)**:

- conda env **`robust`** (this env lives only on the Linux machine — do not try to
  use it on macOS).
- **JAX with CUDA 12.8 (cu128)** — required for Blackwell/RTX 5090.
- **`mujoco-mjx`** for GPU-accelerated, batched simulation (RL in Phase 8, fast MPC
  rollouts in Phase 7).

Phase 0 = standing up this env and verifying the model loads under MJX. It is the
only thing blocking GPU work; the model/physics are otherwise ready.

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
