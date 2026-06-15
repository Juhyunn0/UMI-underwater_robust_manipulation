# 00 — Overview (start here)

> Project memory for the BlueROV2 MuJoCo simulator. If you are a new session with
> no chat history: **read this file, then [01_DECISIONS.md](01_DECISIONS.md)**,
> before changing anything. The conventions and findings below prevent real bugs
> (NED sign flips, mixed parameters, commanding the underactuated pitch DOF).
>
> **Control methodology journal:** the *why-narrative* of the controller development
> (PID → MPC → DOB-MPC → …) lives in [CONTROL_METHODOLOGY.md](CONTROL_METHODOLOGY.md)
> (Korean: [CONTROL_METHODOLOGY.ko.md](CONTROL_METHODOLOGY.ko.md)). **Append a dated
> entry there on every major control change/decision** (keep both languages in sync).

## What this project is

A simulator for the **BlueROV2** underwater vehicle, built to be an **MPC testbed**
(specifically a disturbance-observer MPC / DOB-MPC), with **reinforcement learning**
planned later. The long-term research goal is energy-efficient, robust underwater
manipulation in dynamic currents.

- **Target runtime:** Ubuntu 22.04 + NVIDIA RTX 5090 (Blackwell). **Two** conda
  envs: **`robust`** (CPU — base `mujoco`, perception/SLAM, MPC; Python 3.14,
  numpy<2) and **`robust-mjx`** (GPU — MuJoCo **MJX** + JAX cuda12; Python 3.12,
  numpy 2). They are split because GPU JAX needs numpy≥2 but `robust`'s gtsam pins
  numpy<2. See [06_ENVIRONMENT.md](06_ENVIRONMENT.md).
- **Current status:** transferred to the Linux/RTX 5090 box; **Phase 0 done**
  (GPU/MJX verified, 2026-06-14). The build through MPC runs on CPU `mujoco` in
  `robust`; MJX/GPU (`robust-mjx`) is staged for the RL phase. The MJCF + meshes
  carry over unchanged.

## Engine: MuJoCo (MJX)

Chosen for RTX-5090/Blackwell support, Claude-Code-friendliness (single readable
MJCF + Python), built-in hydro/fluid options, MJPC for MPC, and GPU-parallel RL
scaling via MJX. Rationale and the rejected alternatives (Stonefish, Isaac Sim,
MarineGym-as-runtime) are in [01_DECISIONS.md](01_DECISIONS.md).

## Architecture & conventions (critical)

- **Frame: FLU** — x **forward**, y **left** (PORT), z **up**. Gravity (0,0,−9.81).
  The simulation is **entirely FLU**. Do **NOT** introduce NED anywhere in the
  sim. NED↔FLU conversion happens **only at the MPC boundary** (Phase 7), because
  the DOB-MPC math is written in NED. Mislabelling a frame here = silent sign-flip
  bugs.
- **Canonical model = the MarineGym-derived model** in this folder
  (`bluerov.xml` + `meshes/` + `marinegym_assets/`). There is a *separate*
  hand-built model elsewhere in the repo (`bluerov2_mujoco_dobmpc/`,
  `bluerov2_mujoco_scratch/`); **do NOT mix parameters between them.**
- **Vehicle is underactuated in pitch** (rank(allocation)=5). The controller must
  command surge/sway/heave/yaw/roll — **never pitch**. See
  [03_THRUSTERS.md](03_THRUSTERS.md).

## Where the code lives

All under `bluerov2_mujoco_marinegym/` (this folder):

| file | role |
|---|---|
| `bluerov.xml` | the MJCF (rigid body + 6 thruster sites + 6 force actuators) |
| `meshes/` | real MarineGym meshes (body + T200 thruster), extracted from USD |
| `marinegym_assets/` | MarineGym `BlueROV.yaml` (hydro+rotor coeffs) + `config.yaml` |
| `thrusters.py` | T200 curve, allocation matrix B / pinv(B), command helpers |
| `hydro.py` | Fossen buoyancy/restoring/added-mass/drag (passive-force callback); relative velocity for current |
| `disturbances.py` | current + waves + kicks + domain-randomization sampler |
| `teleop.py` | keyboard driving + **live force-arrow viz** (launch_passive/user_scn); `--managed` for the old no-arrow viewer; G toggles disturbances |
| `test_load.py`, `test_thrusters.py`, `test_hydro.py` | per-phase verification |
| `generate_bluerov_xml.py`, `extract_meshes.py` | regenerate the MJCF / meshes from the USD |
| `external/MarineGym/` (repo root) | MarineGym source asset (git submodule, files only) |

Run anything with plain `python <script>.py` from this folder (base `mujoco` +
`numpy`). Quick verify: `python test_load.py && python test_thrusters.py && python test_hydro.py`.

## Roadmap & status

| phase | scope | status | doc |
|---|---|---|---|
| **0** | GPU/runtime setup (Linux+5090, env `robust-mjx`, JAX cuda12, mujoco-mjx) | **DONE** ✓ (2026-06-14) | [06_ENVIRONMENT.md](06_ENVIRONMENT.md) |
| **1** | Rigid-body MJCF from MarineGym (mass/inertia/thruster sites, meshes) | **DONE** ✓ | [02_MODEL.md](02_MODEL.md) |
| **2** | Thruster actuation (T200 curve, 6 actuators, allocation matrix) | **DONE** ✓ | [03_THRUSTERS.md](03_THRUSTERS.md) |
| **—** | Keyboard teleop + live force-arrow visualization | **DONE** ✓ | [05_TELEOP.md](05_TELEOP.md) |
| **3** | Hydrodynamics (buoyancy/restoring + added mass + drag) | **DONE** ✓ | [04_HYDRO.md](04_HYDRO.md) |
| **4** | Disturbances (current + waves + kicks) + domain randomization | **DONE** ✓ | [07_DISTURBANCES.md](07_DISTURBANCES.md) |
| **5** | Sensors (IMU / depth / velocity measurements + noise) | **NEXT** | — |
| **6** | Environment completion (gym-style reset/obs/reward wrapper) | pending | — |
| **7** | MPC controller (DOB-MPC; NED↔FLU at this boundary only) | pending | — |
| **8** | RL training (MJX-parallel) | pending | — |

What's verified today: Phases 1–4 each have a passing `test_*.py` (load + stats +
stability; thruster directions + allocation; neutral buoyancy + self-righting +
terminal velocity; current/waves/kicks distinct + DR stable), re-verified on the
Linux/RTX 5090 box under `robust`. Phase 0 (GPU/MJX runtime) is now **done** —
the 5090 is detected and a tiny MJX rollout runs on GPU in `robust-mjx`.

## Going forward (doc maintenance pattern)

Each new phase **adds or updates its own doc here** and **updates the status table
above**. Keep docs consistent with the actual code/values — when you change a
parameter, update the doc in the same commit.

## Doc index

- [01_DECISIONS.md](01_DECISIONS.md) — locked decisions + rationale (read second).
- [02_MODEL.md](02_MODEL.md) — Phase 1 rigid body (provenance, values, frame).
- [03_THRUSTERS.md](03_THRUSTERS.md) — Phase 2 thrusters, allocation, **underactuation**.
- [04_HYDRO.md](04_HYDRO.md) — Phase 3 buoyancy/added-mass/drag.
- [05_TELEOP.md](05_TELEOP.md) — keyboard teleop tool.
- [06_ENVIRONMENT.md](06_ENVIRONMENT.md) — macOS drafting env, Linux/MJX runtime, portability.
- [07_DISTURBANCES.md](07_DISTURBANCES.md) — Phase 4 current/waves/kicks + domain randomization.
