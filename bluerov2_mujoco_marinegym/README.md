# BlueROV2 → MuJoCo (MarineGym import)

A **BlueROV2 underwater-vehicle simulator in MuJoCo**, imported from MarineGym's
own BlueROV asset, built as a **testbed for robust underwater control** — PID →
MPC → **disturbance-observer MPC (DOB-MPC)**, with reinforcement learning planned
later. The long-term research goal is energy-efficient, robust underwater
manipulation in dynamic currents.

The simulator models the full 6-DOF Fossen dynamics (buoyancy/restoring, added
mass, linear+quadratic drag), realistic **T200 thrusters** (with an optional
deadband/asymmetry/lag actuator model), and environmental **disturbances**
(ocean current + irregular JONSWAP waves + Poisson kicks), all in the **FLU**
frame. On top of it run three controllers compared on the same plant, seed, and
disturbance, plus a rigorous verification suite for the hydrodynamics and the
solver.

> **📖 The full project memory is in [`docs/`](docs/) — start at
> [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md)** (goal, engine choice, FLU
> convention, roadmap/status), then [docs/01_DECISIONS.md](docs/01_DECISIONS.md).
> The *why-narrative* of the controller development lives in
> [docs/CONTROL_METHODOLOGY.md](docs/CONTROL_METHODOLOGY.md)
> (Korean: [.ko.md](docs/CONTROL_METHODOLOGY.ko.md)). **This README is the
> run/how-to entry point and is kept up to date as the code grows.**

---

## What's in it

| capability | where | status |
|---|---|---|
| Rigid body + 6 thruster sites/actuators (MJCF) | `bluerov.xml`, `meshes/` | ✅ verified |
| T200 thruster curve, allocation `B` (rank 5), realistic actuator model | `thrusters.py` | ✅ |
| Fossen hydrodynamics: buoyancy/restoring + added mass + drag (passive callback) | `hydro.py` | ✅ verified |
| Disturbances: current + irregular waves + kicks + domain randomization | `disturbances.py` | ✅ verified |
| **Finite-depth** disturbance env: directional JONSWAP + current+drift + Froude-Krylov inertia (5 modes: NONE/C/CD/CW/CDW, NONE=still-water baseline) | `disturbance/` | ✅ verified |
| Keyboard teleop + live force-arrow viz + live dashboard | `teleop.py`, `monitor.py` | ✅ |
| Baseline **PD/PID** setpoint controller | `controller.py` | ✅ |
| **DOB-MPC** = Extended Active Observer (EAOB) + NMPC | `dobmpc_controller.py`, `dobmpc/` | ✅ |
| NMPC solved by **acados SQP-RTI** (~1 ms, default) with IPOPT fallback | `dobmpc/mpc_acados.py`, `dobmpc/mpc.py` | ✅ verified |
| Autonomous square-tracking mission + CSV recorder + run manifest (incl. kicks) | `mission.py`, `recorder.py` | ✅ |
| Experiments: station-keeping comparison, actuator-realism ablation | `dobmpc/eval_dp.py`, `ablation_thrusters.py` | ✅ |
| Experiment: 3 controllers × 4 disturbance modes × N seeds (DP + square), metrics + figures | `experiments/run_compare.py`, `config/*.yaml` | ✅ |
| **Live viewer**: watch ONE controller × mode run the square in real-time MuJoCo + save trajectory CSV + 1-lap mp4 | `experiments/run_viewer.py` | ✅ |
| Verification: hydro (smoke + precision), acados equivalence, run-meta | `verify_*.py` | ✅ |

Two model variants, selected by `ROV_MODEL` (see below): **heavy** (default,
8 thrusters, `rank = 6` — **fully actuated**, the NMPC commands the full 6-DOF
wrench incl. roll and pitch) and **bluerov2** (6 thrusters,
`rank(allocation) = 5` — under-actuated in pitch, command
surge/sway/heave/yaw/roll, **never pitch**).

---

## Environment

Two conda envs on the Ubuntu + RTX 5090 box (see
[docs/06_ENVIRONMENT.md](docs/06_ENVIRONMENT.md)):

- **`robust`** (CPU; Python 3.14, numpy<2) — everything in this folder: base
  `mujoco`, the controllers, acados + IPOPT. **Use this for all commands below.**
- **`robust-mjx`** (GPU; Python 3.12, numpy 2, JAX cuda12) — MuJoCo **MJX**,
  staged for the RL phase only.

```bash
conda activate robust
cd bluerov2_mujoco_marinegym
```

### Model variant: BlueROV2 vs BlueROV2 Heavy

Pick the vehicle with the **`ROV_MODEL`** env var (default **`heavy`**); a single
[rov_model.py](rov_model.py) registry keeps the plant and the controller in sync:

```bash
                   python teleop.py --square --ctrl dobmpc --disturb   # heavy (default)
ROV_MODEL=bluerov2 python teleop.py --square --ctrl dobmpc --disturb   # vectored-6 (rank-5)
                   python -m dobmpc.eval_dp --ctrls pid,mpc,dobmpc     # heavy, headless
```

| | bluerov2 | heavy |
|---|---|---|
| thrusters | 6 (rank 5) | 8 (rank 6, **fully actuated**) |
| mass / inertia | 11.2 kg / [0.30375, 0.626, 0.5769] | 11.5 kg / [0.3291, 0.6347, 0.6109]† |
| NMPC input | `u=[X,Y,Z,N]` (NU 4) | `u=[X,Y,Z,K,M,N]` (NU 6) |
| pitch | floats to trim (~12°) | actively leveled (~0.8°) |

Same T200 thrusters and hydro coefficients; only mass/volume/thruster layout
differ. **†** Heavy's inertia is *derived* from the bluerov2 tensor by adding the
parallel-axis term of the vertical-thruster layout change
([compute_heavy_inertia.py](compute_heavy_inertia.py)) — the farol Heavy USD's own
[0.21, 0.245, 0.245] is a hand-tuned Gazebo-stability literal, not physical. It's a
physically-motivated estimate (not a Heavy CAD measurement) but Heavy-specific and
≥ BlueROV2. See [docs/CONTROL_METHODOLOGY.md](docs/CONTROL_METHODOLOGY.md) (2026-06-18).

**Dependencies.** Running the *plant* needs only `mujoco` + `numpy`. The DOB-MPC
adds `casadi` (symbolic dynamics + IPOPT) and `acados_template` (the fast SQP-RTI
solver). Verification/analysis add `scipy` + `matplotlib`. acados is built at
`/home/bdml/acados` and **loads with no shell `LD_LIBRARY_PATH` export** (the
shared libs are pre-loaded via `ctypes RTLD_GLOBAL` in `dobmpc/_acados_env.py`).
If acados is unavailable it auto-falls back to the IPOPT NMPC.

### Pool AprilTag floor (visual, opt-in): `POOL_TAGS`

Set **`POOL_TAGS=1`** to load a **visual replica of the real test pool's floor** — a
dense grid of tag36h11 AprilTags (0.170 m black edge, the real spec from
[../config/config.yaml](../config/config.yaml) + [../config/tag_map.yaml](../config/tag_map.yaml)),
a seabed, and a single translucent water volume filled to the seabed with a wavy
animated surface (~2 m visual column; the tags read as submerged). It works with either
`ROV_MODEL`, and picks up automatically in every entry point that reads
`rov_model.XML_PATH` (teleop, `eval_dp`, `run_compare`, tests):

```bash
POOL_TAGS=1                    python teleop.py            # heavy + pool floor
POOL_TAGS=1 ROV_MODEL=bluerov2 python teleop.py            # bluerov2 + pool floor
```

The floor is built once by [gen_pool_apriltags.py](gen_pool_apriltags.py) (survey
tags at their real ids/poses + a grid fill; every PNG is round-trip verified with
`pupil_apriltags` so sim tag ids provably match the real family). It is **VISUAL
ONLY** — all added geoms are `contype=0 conaffinity=0` (group 1) and MuJoCo's fluid
model is off, so **dynamics are byte-for-byte identical** with `POOL_TAGS` set or
unset (verified: 3000-step rollout Δ=0, incl. `--disturb`). Regenerate / retune with:

```bash
python gen_pool_apriltags.py --selftest      # render a couple tiles + detect them
python gen_pool_apriltags.py                 # full build (default hybrid layout)
# knobs: --layout {survey,grid,hybrid} --pool-width (visual X, default 2.6 m vs real 1.8)
#        --pool-length --seabed-z --water-depth --pitch-x/--pitch-y ...
```

The **visual** water column (default **2 m**: seabed z=-0.5 → surface z=+1.5) is ONE
translucent geom — the animated heightfield's skirt is extruded down to the seabed, so
the wavy surface and the submerged column are a single unified body (no seam). It is fully
decoupled from the disturbance model's **physics** depth (`disturbances.py z_surface`=3,
`disturbance/waves.py h`=4) — cosmetic, never touches dynamics. Knobs: `--water-depth`,
`--water-alpha` (0.18), `--no-water-body` (thin sheet only), `--no-water-anim` (flat box).

#### Animated waves + current on the water surface

Under `POOL_TAGS=1` the water surface is a **heightfield** ([water_viz.py](water_viz.py))
that undulates like real waves and drifts with the current, reconstructed live from the
**same disturbance field the physics uses** (`eta(x,y,t)=Σ aᵢ cos(kᵢ·x − ωᵢt + φᵢ)`,
advected by the current so waves+current read as one surface). It animates in both the
live `teleop.py` viewer (`viewer.update_hfield`) and the `run_viewer.py` mp4
(`mjr_uploadHField`), and flattens when disturbances are off (`G`). Still **VISUAL ONLY**
— animating the hfield every step leaves dynamics byte-identical (verified Δ=0, both
variants, faithful + stylized).

```bash
POOL_TAGS=1 python teleop.py --disturb                    # waves undulate + drift (G toggles)
POOL_TAGS=1 WATER_WAVE_LAMBDA=0.9 python teleop.py --disturb   # exaggerated, clearly-sloshing
```

Real ocean wavelengths (6–76 m) dwarf the pool (~1.8×4.9 m), so the **default is
physically faithful** — a gentle heave/tilt, barely-rippled. `WATER_WAVE_LAMBDA=<m>`
shrinks the *visual* wavelength for dramatic ripples; `WATER_WAVE_AMP=<gain>` scales the
swing. Both are render-only (physics wavenumber/amplitude untouched). Grid/headroom knobs
live on the generator (`--water-hf-rows/-cols/-elev`, `--no-water-anim` for the old flat
box). Previews: [assets/screenshots/pool_waves_stylized.mp4](../assets/screenshots/pool_waves_stylized.mp4),
[pool_waves_faithful.mp4](../assets/screenshots/pool_waves_faithful.mp4).

---

## How to run

### 1. Smoke tests (fast, headless)

```bash
python test_load.py          # load, mass/inertia stats, zero-control stability
python test_thrusters.py     # thruster directions + allocation matrix (rank 5)
python test_hydro.py         # neutral buoyancy / self-righting / terminal velocity
python test_disturbances.py  # current / waves / kicks distinct + domain randomization
python test_controller.py    # PD/PID go-to-origin
python test_dobmpc.py        # EAOB + NMPC (acados) closed loop
python test_square_mission.py
# each test file is runnable directly (python test_<name>.py); `pytest -q` also
# works if pytest is installed (not in the base `robust` env).
```

### 2. Interactive teleop (needs a display)

```bash
python teleop.py                       # drive it, live force arrows, G toggles disturbances
python teleop.py --disturb             # start with current+waves+kicks ON
python teleop.py --ctrl dobmpc --goto-origin --disturb   # DOB-MPC holds the origin
python teleop.py --no-hydro            # thruster-only feel (gravity+hydro off)
```

Useful flags: `--ctrl {pd,pid,mpc,dobmpc}`, `--viser`/`--remote` (browser viewer),
`--managed` (old viewer, no arrows), `--no-monitor` (skip the dashboard). See
[docs/05_TELEOP.md](docs/05_TELEOP.md).

### 3. Autonomous square-tracking mission (needs a display; records to `recordings/`)

Approaches the origin, auto-starts recording, tracks a square for N laps:

```bash
python teleop.py --square --ctrl dobmpc --disturb
python teleop.py --square --ctrl pid  --disturb --laps 10 --square-size 1.0 --square-speed 0.15
```

The autonomous missions (`--square` / `--goto-origin`) use the **realistic T200
actuator by default** (deadband / fwd-rev asymmetry / motor lag / voltage), since
they exist to predict the real robot. Flags: `--ideal-thrusters` reverts to the
ideal force path (commanded == realized); `--thruster-voltage 0.72` sets the
battery thrust scale (default `0.72` = 4S nominal 14.8 V, datasheet-grounded — see
[docs/03_THRUSTERS.md](docs/03_THRUSTERS.md) and `analyze_t200_voltage.py`).

Each run writes `recordings/<timestamp>_square_<ctrl>.csv` **plus a sidecar
`<...>.meta.json`** capturing the full run manifest — controller config, actuator
config (`run.thrusters`), trajectory, and the exact disturbance schedule
**including every kick event**.

### 4. Experiments (headless, seed-controlled)

```bash
# Station-keeping (dynamic positioning) comparison: PID vs MPC vs DOB-MPC on the
# SAME plant / seed / disturbance / start offset
python -m dobmpc.eval_dp --ctrls pid,mpc,dobmpc --seed 0 --T 60

# Actuator-realism ablation: ideal force path vs realistic T200 (deadband /
# asymmetry / lag / voltage sag), PID/MPC/DOB-MPC averaged over 5 seeds
python ablation_thrusters.py

# Disturbance-mode comparison: 3 controllers × 4 finite-depth modes × N seeds.
# Finite-depth directional waves + current(+drift) + Froude-Krylov inertia, shared
# seed per (mode,seed) for a fair comparison. DP (rejection) + square (tracking).
python -m disturbance.test_waves && python -m disturbance.test_env   # 34 unit asserts
python -m experiments.run_compare --config config/base.yaml --smoke  # tiny pipeline check
python -m experiments.run_compare --config config/base.yaml          # full matrix (parallel)
# Runs are independent and run in parallel by default (min(cpu,16) procs); the acados
# solver is pre-built once, workers load it. `--jobs 1` forces serial; `--jobs N` caps it.
python -m experiments.run_compare --config config/base.yaml --jobs 8

# Live MuJoCo viewer for ONE (controller, mode) on the square — fix one seed + one
# current direction and WATCH it run in real time. Saves the full-run trajectory CSV
# and an mp4 of just ONE lap (the last, settled lap; full-run video would be huge).
# Same plant/hydro/DisturbanceEnv as run_compare, so the trajectory matches the batch.
python -m experiments.run_viewer --config config/base.yaml --ctrl dobmpc --mode CDW
python -m experiments.run_viewer --config config/base.yaml --ctrl pid --mode C --headless  # no window (CSV+mp4 only)
```

`eval_dp` prints a metrics table (radial RMS, DC bias, jitter, pitch, ŵ_x) and
saves a plot; `ablation_thrusters` writes `docs/figs/ablation_thrusters.png`;
`run_compare` writes `recordings/<date>/compare_<ts>/` with `results.csv` (mean±std
+ DRR), `results_raw.csv`, and `figures/` (per-mode time-histories, metric bars, and
a controller-independent disturbance self-check). Modes/params/seeds are all in the
YAML — no code edit needed. Config knobs: `inertia.fk_mode` (froude_krylov | morison_ca
| off), `experiment.{primary,secondary}`.

`run_viewer` runs exactly one `--ctrl {pid,mpc,dobmpc}` × `--mode {NONE,C,CD,CW,CDW}`
(call it once per combination) and accumulates outputs in
`recordings/<date>/square_view/`: `traj_*.csv` (full run, position + reference),
`lap_*.mp4` (one lap), and `meta_*.json`. Flags: `--seed`, `--dir-deg` (fixes the
single current heading), `--laps`, `--record-lap last|first|middle|<int>`,
`--heading follow|fixed` (face the travel direction; default follow),
`--yaw-rate <deg/s>` (corner-turn smoothing, default 60),
`--video-hz`, `--video-size WxH`, `--no-arrows`, `--no-video`, `--headless` (no
on-screen window; offscreen render only — use on a display-less host, or if the live
window and recorder clash, with `MUJOCO_GL=egl` auto-selected when there is no display).

### 5. Analysis of recordings

```bash
python analyze_square3.py            # off-path + time-ref error for the 3 square CSVs
python analyze_acados_vs_before.py   # acados vs pre-acados square comparison
```
These read fixed folders under `recordings/` — edit the `DIR` constant at the top
to point at your run.

### 6. Verification (V&V — run after touching dynamics or the solver)

```bash
python verify_hydro.py          # 32-check fast smoke (structural + behavioral)
python verify_hydro_precise.py  # 4-tier rigorous: order-of-accuracy, MMS,
                                #   skew/SPD identities, frame invariance, lag fidelity (slow)
python verify_acados.py         # acados NMPC == IPOPT NMPC (equivalence + timing)
python verify_meta.py           # the recorder sidecar manifest is complete
python analyze_t200_voltage.py  # datasheet provenance for the thruster voltage_scale (0.72)
```

---

## Controllers & solver

- **PID / PD** ([controller.py](controller.py)) — baseline world-frame setpoint
  + yaw reference, fully FLU.
- **DOB-MPC** ([dobmpc_controller.py](dobmpc_controller.py), [dobmpc/](dobmpc/)) —
  an **EAOB** (18-state EKF) estimates the disturbance wrench ŵ online; the
  **NMPC** predicts with the full Fossen model carrying ŵ as a horizon parameter
  and re-plans every 50 ms (N=60). `mode="mpc"` is the same NMPC with ŵ=0.
- **Solver:** the NMPC solves via **acados SQP-RTI + HPIPM** (~1 ms/tick,
  default), with **IPOPT** (CasADi) as the validated reference and as the runtime
  fallback if acados ever NaNs. Switch with `dobmpc/params.py::SOLVER`
  (`"acados"` / `"ipopt"`). See [docs/CONTROL_METHODOLOGY.md](docs/CONTROL_METHODOLOGY.md)
  and the memory note `acados-solver-toolchain`.

---

## Layout

```
bluerov2_mujoco_marinegym/
├── docs/                       # project memory — start at 00_OVERVIEW.md
│   ├── 00_OVERVIEW.md … 07_DISTURBANCES.md   # per-phase design docs
│   ├── CONTROL_METHODOLOGY.md / .ko.md        # controller why-journal (EN/KO)
│   └── HYDRO_VERIFICATION.md / .ko.md         # hydro V&V writeup (EN/KO)
├── rov_model.py                # variant registry (ROV_MODEL: bluerov2 | heavy) — single source of truth
├── bluerov.xml                 # bluerov2 MJCF (rigid body + 6 thruster sites + 6 actuators)
├── bluerov_heavy.xml           # heavy MJCF (8 thrusters, mass 11.5, fully actuated)
├── gen_pool_apriltags.py       # build the opt-in pool AprilTag floor (POOL_TAGS=1); VISUAL ONLY
├── water_viz.py                #   animated waves+current water surface (hfield); VISUAL ONLY
├── tag_floor.xml               #   generated <mujocoinclude>: seabed + tag36h11 grid + water hfield
├── scene_bluerov_tags.xml · scene_bluerov_heavy_tags.xml   # generated opt-in wrappers (ROV + tag_floor)
├── apriltags/                  # generated tag36h11 PNGs (one per tag id), round-trip verified
├── meshes/                     # real MarineGym meshes (body + T200), from the USD
├── marinegym_assets/           # MarineGym BlueROV.yaml / BlueROVHeavy.yaml (hydro/rotor coeffs) + config
│
├── thrusters.py                # T200 curve, allocation B/pinv(B), realistic ThrusterModel
├── hydro.py                    # Fossen buoyancy/restoring/added-mass/drag (passive callback)
├── disturbances.py             # current + waves + kicks + DR sampler
│
├── controller.py               # baseline PD/PID setpoint controller
├── dobmpc_controller.py        # DOB-MPC controller (wraps dobmpc/)
├── dobmpc/                      # EAOB + NMPC subpackage
│   ├── eaob.py                 #   Extended Active Observer (disturbance estimate ŵ)
│   ├── mpc.py                  #   CasADi/IPOPT NMPC (reference) + make_nmpc() factory
│   ├── mpc_acados.py           #   acados SQP-RTI port (default) + IPOPT fallback
│   ├── fossen.py / frames.py / params.py   # dynamics, NED↔FLU, plant-matched params
│   └── eval_dp.py              #   station-keeping PID/MPC/DOB-MPC comparison
│
├── teleop.py                   # keyboard teleop + force arrows + square mission driver
├── mission.py                  # autonomous square-trajectory phase machine
├── recorder.py                 # CSV logger + sidecar run manifest (.meta.json)
├── monitor.py                  # separate-process live dashboard
│
├── ablation_thrusters.py       # actuator-realism ablation experiment
├── analyze_square3.py · analyze_acados_vs_before.py   # recording analysis
├── analyze_t200_voltage.py     # datasheet provenance for thruster voltage_scale (0.72)
├── compute_heavy_inertia.py    # derive the Heavy inertia from bluerov2 (parallel-axis)
├── verify_hydro.py · verify_hydro_precise.py · verify_acados.py · verify_meta.py
├── test_*.py                   # per-component tests (pytest)
├── generate_bluerov_xml.py · extract_meshes.py        # regenerate MJCF/meshes from USD
└── recordings/                 # experiment CSVs + .meta.json sidecars
```

---

## Conventions you must not break (see docs for detail)

- **Frame is FLU** (x forward, y left, z up), gravity (0,0,−9.81). No NED in the
  sim; NED↔FLU conversion happens **only at the DOB-MPC boundary** (the math is
  written in NED). Mislabelling a frame = silent sign-flip bug.
- **The MarineGym-derived model is canonical** — don't mix parameters with the
  separate hand-built `bluerov2_mujoco_dobmpc/` model.
- **Pitch is underactuated on `bluerov2`** (`rank(allocation) = 5`) — command
  surge/sway/heave/yaw/roll, never pitch. (On `heavy` the 8-thruster allocation is
  rank 6 / fully actuated, so pitch IS commanded — keep the variants' assumptions
  straight via `rov_model.py`.)
- **Append to the methodology journal** on every major control change, and keep
  this README's run instructions current when entry points change.

## Doc index

- [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md) — goal, engine, architecture, roadmap/status (read first).
- [docs/01_DECISIONS.md](docs/01_DECISIONS.md) — locked decisions + rationale.
- [docs/02_MODEL.md](docs/02_MODEL.md) … [docs/07_DISTURBANCES.md](docs/07_DISTURBANCES.md) — per-phase design.
- [docs/CONTROL_METHODOLOGY.md](docs/CONTROL_METHODOLOGY.md) — controller development journal (PID → MPC → DOB-MPC → acados).
- [docs/HYDRO_VERIFICATION.md](docs/HYDRO_VERIFICATION.md) — hydrodynamics V&V writeup.
