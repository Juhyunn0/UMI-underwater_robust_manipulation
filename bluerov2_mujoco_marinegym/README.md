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
| Experiment: 3 controllers × 5 disturbance modes × N seeds × current/wave heading sweep (paired or full grid), metrics + per-run CSVs/meta + all figures in one run | `experiments/run_compare.py`, `config/*.yaml` | ✅ |
| **Live viewer**: watch ONE controller × mode run the square in real-time MuJoCo + save trajectory CSV + 1-lap mp4 | `experiments/run_viewer.py` | ✅ |
| Trajectory-overlay figure from `run_viewer` CSVs (single mode, or all-modes 2×3 grid) | `experiments/plot_trajectories.py` | ✅ |
| Slide figures: β̄ = mean-of-a-spread wave heading (sea snapshot + cos^2s lobe, s=30 vs s=2) → `assets/screenshots/waves/wave_*.png` | `plot_wave_spreading.py` | ✅ |
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

### Model variant: BlueROV2 vs BlueROV2 Heavy (vs Heavy+Gripper)

Pick the vehicle with the **`ROV_MODEL`** env var (default **`heavy`**); a single
[rov_model.py](rov_model.py) registry keeps the plant and the controller in sync:

```bash
                        python teleop.py --square --ctrl dobmpc --disturb   # heavy (default)
ROV_MODEL=bluerov2      python teleop.py --square --ctrl dobmpc --disturb   # vectored-6 (rank-5)
ROV_MODEL=heavy_gripper python teleop.py --observe                          # payload variant
                        python -m dobmpc.eval_dp --ctrls pid,mpc,dobmpc     # heavy, headless
```

| | bluerov2 | heavy | heavy_c3 | heavy_gripper |
|---|---|---|---|---|
| thrusters | 6 (rank 5) | 8 (rank 6, **fully actuated**) | 8 (rank 6) | 8 (rank 6) + gripper servo (ctrl 8) |
| mass / inertia | 11.2 kg / [0.30375, 0.626, 0.5769] | 11.5 kg / [0.3291, 0.6347, 0.6109]† | 13.2 kg / [0.37014, 0.73153, 0.67460]§ | 13.724 kg / [0.38154, 0.77780, 0.70954]‡ |
| net buoyancy | ~+1.1 N | ~+1.1 N | **−3.1 N (sinks)** | **−5.7 N (sinks)** |
| NMPC input | `u=[X,Y,Z,N]` (NU 4) | `u=[X,Y,Z,K,M,N]` (NU 6) | same as heavy | same as heavy |
| pitch | floats to trim (~12°) | actively leveled (~0.8°) | ~−1° (heavy gains, no rp-PD) | actively leveled (PID adds rp-PD) |

**§ heavy_c3 = heavy + MarineSitu C3 stereo camera on its C3-BR bracket** — EXACTLY the
lab's Onshape assembly (the Newton gripper is **not in Onshape yet**, so it is absent;
`heavy_gripper` is the future config that adds it). Same composition philosophy: vendor C3
mass 1.700 kg + parallel-axis via [compute_payload_inertia.py](compute_payload_inertia.py)
`compose_c3()`; MJCF GENERATED from `bluerov_heavy.xml` by
[gen_c3_variant.py](gen_c3_variant.py) (never hand-edit). Body frame re-origined at the
composite COM; inertia **diagonal** (`Ixz` +0.046 dropped, 12.4% of Ixx — KNOWN_ISSUES). C3
mount **measured from the Onshape assembly** (2026-07-19: front-bottom on the centreline,
lens forward and level; 3 cameras `c3_center/left/right` at the lens plane). Bracket is
visual-only (mass unknown). Uses `GAINS_HEAVY` (no active roll/pitch leveling → ~1° residual
pitch from the C3's static moment). Selected with `ROV_MODEL=heavy_c3`; `POOL_TAGS=1` →
`scene_bluerov_heavy_c3_tags.xml`. Regression: [test_heavy_c3.py](test_heavy_c3.py).

**‡ heavy_gripper = heavy + Newton Subsea Gripper + MarineSitu C3 stereo camera** (the
real payload this lab bolts on). Rigid-body numbers are COMPOSED, not hand-tuned:
vendor-verified masses (gripper 524 g air/267 g water; C3 1700 g/430 g) + parallel-axis
inertia via [compute_payload_inertia.py](compute_payload_inertia.py); the MJCF is
GENERATED from `bluerov_heavy.xml` by [gen_gripper_variant.py](gen_gripper_variant.py)
(never hand-edit it). Key properties:
- **Body frame re-origined at the composite COM** (origin==COM, like heavy) — the dobmpc
  predictor, `params.ZG_MASS=0`, and hydro all assume it. Inertia is emitted **diagonal**
  (`Ixz` +0.064 dropped, **16.8% of Ixx** since the C3 moved to its measured front-bottom
  mount): a `fullinertia` here gets axis-permuted by MuJoCo's principal-axis sort, which
  breaks hydro's body-frame drag (see KNOWN_ISSUES).
- **Articulated jaws**: two mirrored slide joints + ONE `position` actuator named
  `gripper` at **ctrl index 8** (`d.ctrl[8] = 0…0.031` = closed…62 mm open). All thruster
  code finds actuators by name (`thr0..7`), so the extra actuator is invisible to it.
- **3 onboard cameras** (`c3_center` 12 MP-equiv fovy 52.5°, `c3_left`/`c3_right` stereo
  pair, 7.5 cm baseline), mounted **front-bottom on the centreline, lens forward and
  level** — measured from the lab's Onshape assembly (2026-07-19, onshape-to-robot +
  mesh registration; the C3-BR bracket straddles the gripper tube with mm clearance).
  They see the gripper jaws dead ahead (render with
  `mujoco.Renderer(...).update_scene(d, camera="c3_center")`).
- **Negative buoyancy** (−5.7 N, no trim foam — deliberate): controllers hold depth with
  sustained upward thrust; PID gains re-derived (`GAINS_HEAVY_GRIPPER`, same pole
  placement at the payload masses + roll/pitch leveling PD `rp_kp/rp_kd`).
- Hydro added-mass/damping stay the heavy set (payload increments ≪ published-set spread;
  DNV build-up estimates documented in
  [marinegym_assets/BlueROVHeavyGripper.yaml](marinegym_assets/BlueROVHeavyGripper.yaml) —
  revisit with in-situ system ID). Verified: `python test_heavy_gripper.py`
  (composition/buoyancy/gripper/PID) + DOB-MPC DP hold 1.3 cm (still), 1.4 cm (current+waves).

The **heavy** visual is the real MarineGym BlueROVHeavy skin — the body mesh is split
by material into **cyan foam / white tube / black frame / silver hardware** (matching the
paper render), with the frame + thruster shrouds baked into that mesh set. It's **VISUAL
ONLY**: dynamics come from the explicit `<inertial>` (mass 11.5, the diaginertia below),
the collision box, and the 8 thruster sites/actuators — all unchanged, so the colored skin
is byte-identical in physics (verified 3000-step rollout Δ=0 vs the old gray body).
Regenerate the color parts with `python extract_meshes.py --colored`.

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

### Pool AprilTag floor (visual): `POOL_TAGS`

`POOL_TAGS` loads a **visual replica of the real test pool's floor** — tag36h11 AprilTags
(0.170 m black edge, the real spec from [../config/config.yaml](../config/config.yaml) +
[../config/tag_map.yaml](../config/tag_map.yaml)), a seabed, and a single translucent water
volume with a wavy animated surface. It works with either `ROV_MODEL` and picks up in every
entry point that reads `rov_model.XML_PATH`.

**Default = a moderately-enlarged floor (`--tag-mode plane`, `--floor-size` 8 m).** The tag
floor is ONE big plane carrying ONE repeating **tag-mosaic** texture (a 10×10 block of distinct
tags baked into a single PNG, tiled to the floor), and the wave surface is enlarged to match.
It loads **faster** than the old per-tile floor — **1 texture + 1 material + 1 plane geom**
instead of ~198 textured boxes (measured ~2.2× faster model parse; the GPU texture upload win
is larger). Each mosaic cell still renders at the real 0.170 m size (verified by top-down
`pupil_apriltags` detection). Tag ids repeat per block — fine for the current visual use (no
camera/detection path). The default 8 m keeps the tag floor noticeably bigger than the real
pool **while the wave undulation stays clearly visible**; larger `--floor-size` (e.g. 48 m)
reads as an endless pool but flattens the waves (amplitude shrinks relative to the huge
surface). `--tag-mode tiles` reproduces the old **unique-id** per-tile floor (bounded to 587
tags, one texture per tile) for a future onboard-camera/SLAM path.

**The interactive `teleop.py` viewer defaults `POOL_TAGS` ON** (that's the context you watch);
set **`POOL_TAGS=0`** for a bare scene. The headless experiment/test/verify entry points
(`eval_dp`, `run_compare`, `verify_*`, `test_*`) default it **OFF** — they keep the clean
plant baseline (the `POOL_TAGS on-vs-off ⇒ Δ=0` check needs an off default). Turn it on there
explicitly:

```bash
python teleop.py                            # heavy + enlarged pool floor (default ON in teleop)
POOL_TAGS=0 python teleop.py                # bare scene (no tags/water)
POOL_TAGS=1 python run_compare ...          # opt IN for an experiment (default off there)
```

The floor is built once by [gen_pool_apriltags.py](gen_pool_apriltags.py) (each tag PNG is
round-trip verified with `pupil_apriltags` so sim tag ids provably match the real family). It
is **VISUAL ONLY** — all added geoms are `contype=0 conaffinity=0` (group 1) and MuJoCo's fluid
model is off, so **dynamics are byte-for-byte identical** with `POOL_TAGS` set or unset
(verified: old-vs-new 3000-step rollout Δ=0, incl. `--disturb`). Regenerate / retune with:

```bash
python gen_pool_apriltags.py                 # default: 8 m plane floor (waves visible), sky=gradient
python gen_pool_apriltags.py --floor-size 24 # bigger floor (waves get gentler as it grows)
python gen_pool_apriltags.py --tag-mode tiles                     # old unique-id per-tile floor
python gen_pool_apriltags.py --selftest      # (tiles) render a couple tiles + detect them
# plane knobs: --floor-size (m, def 8) --mosaic-blocks (B, def 10) --mosaic-gap-modules --sky
# tiles knobs: --layout {survey,grid,hybrid} --pool-width/--pool-length --pitch-x/--pitch-y ...
```

**Pretty background (viewer only).** The POOL_TAGS wrapper bakes in a **skybox** so the
viewer isn't a black void — default `--sky gradient` (light blue → white), also `white`,
`blue`, `dark`, `none`. It's a skybox texture + a `<visual>` override (haze/ambient) placed
after the includes, so it's **pure rendering** — no dynamics, no plant-file edit — and only
the POOL_TAGS scene has it (training on the bare XML stays as-is). Preview:
[assets/screenshots/pool/pool_sky_gradient.png](../assets/screenshots/pool/pool_sky_gradient.png).

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
box). Previews: [assets/screenshots/pool/pool_waves_stylized.mp4](../assets/screenshots/pool/pool_waves_stylized.mp4),
[pool_waves_faithful.mp4](../assets/screenshots/pool/pool_waves_faithful.mp4).

---

## How to run

All commands run from `bluerov2_mujoco_marinegym/` in the **`robust`** conda env
(one exception: `verify_gpu_mjx.py` runs in `robust-mjx`, see §6). The env-var
knobs (`ROV_MODEL`, `POOL_TAGS`, …) are collected in [§8](#8-environment-variables) —
**`ROV_MODEL` is the only one that changes dynamics**; everything else is visual
or infrastructure.

### Command index

| command | what it does | § |
|---|---|---|
| `python teleop.py` (+ `--observe` / `--square` / `--goto-origin`) | interactive teleop, free-drift observation, autonomous missions | 2–3 |
| `python -m experiments.run_compare --config config/base.yaml` | headless batch: 3 controllers × 5 disturbance modes × seeds × heading grid → metrics CSVs + per-run CSVs/meta + trajectory_compare figures | 4 |
| `python -m experiments.run_viewer --config config/base.yaml --ctrl dobmpc --mode CDW` | watch ONE (controller, mode) square run live; trajectory CSV + 1-lap mp4 | 4 |
| `python -m experiments.plot_trajectories` | overlay figure of `run_viewer` trajectory CSVs | 4 |
| `python -m dobmpc.eval_dp` | station-keeping (DP) PID / MPC / DOB-MPC comparison | 4 |
| `python ablation_thrusters.py` | actuator-realism ablation (ideal / realistic / low-voltage T200) | 4 |
| `python test_<name>.py` (11 files) · `python -m disturbance.test_{waves,env}` | per-component smoke/unit tests | 1 |
| `python verify_hydro[_precise].py` · `verify_acados.py` · `verify_meta.py` · `verify_gpu_mjx.py` | V&V suite (hydro, solver equivalence, run manifest, GPU env) | 6 |
| `python analyze_square3.py` · `analyze_acados_vs_before.py` · `analyze_t200_voltage.py` | recording analysis + datasheet provenance | 5 |
| `python gen_pool_apriltags.py` · `extract_meshes.py` · `generate_bluerov_xml.py` · `compute_heavy_inertia.py` · `plot_wave_spreading.py` | asset & figure generators (occasional) | 7 |

### 1. Smoke / unit tests (fast, headless)

Every test file is directly runnable (`python test_<name>.py`); `pytest -q` also
works if pytest is installed (not in the base `robust` env). All are headless by
default; where a `--render` flag exists it opens a viewer on the same scenario
instead.

| command | what it checks |
|---|---|
| `python test_load.py` | model loads; mass ∈ [9, 12] kg; 6 thruster sites; zero-control stability, no NaN. Flags: `--seconds`, `--render out.png`, `--viewer` |
| `python test_thrusters.py` | ctrlrange = T200 curve limits; measured body wrench ≡ `B @ f`; surge→pitch / sway→roll coupling ratios; pitch underactuation (rank(B) = 5). Flags: `--T`, `--render` |
| `python test_hydro.py` | neutral-buoyancy hover; self-righting from 20° roll/pitch; drag-bounded terminal velocity + coast-to-stop; 60 s stability. Flag: `--render` |
| `python test_disturbances.py` | no-thrust drift converges to the current velocity (proves relative velocity is used); wave decay with depth; kick rate; distinct current/wave/kick signatures; bounded domain randomization. Flag: `--render` |
| `python test_controller.py` | PID/PD go-to-origin × {still water, 0.2 m/s current}: convergence, PID rejects the current, PD keeps a steady-state offset |
| `ROV_MODEL=bluerov2 python test_dobmpc.py` | FLU↔NED round-trips; Fossen predictor (6 N ≈ 23° trim — a bluerov2-derived ground truth, hence the env var); EAOB unbiasedness; CasADi MPC model ≡ NumPy fossen model (< 1e-9). Needs casadi; no MuJoCo |
| `python test_square_mission.py` | JONSWAP spectrum sanity (Hs recovery, irregularity, seed reproducibility) + a 2-lap auto-recorded square completes and returns |
| `python test_observe.py` | `--observe` contracts: drive keys are no-ops; recenter is state-only; the free-drift rollout is byte-identical with the observe gate installed |
| `python test_water_viz.py` | the animated water hfield visibly changes the render AND leaves a 2000-step rollout byte-identical (Δ = 0). Needs EGL offscreen GL |
| `python test_monitor_smoke.py` | pyqtgraph monitor builds/redraws headless; spawned-process handle round-trip; degenerate-time guards. Needs PyQt5 + pyqtgraph |
| `python test_viser_smoke.py` | `--viser` plumbing without a browser (starts a real viser server on port 8099 — the port must be free) |
| `python -m disturbance.test_waves` | 12 checks on the finite-depth directional wave field: dispersion residual + limits, JONSWAP + cos^2s normalization, realized Hs, seabed velocity limit, seed reproducibility |
| `python -m disturbance.test_env` | 21 checks on current/env: exact Gauss–Markov discretization, C/CD/CW/CDW mode gating, identical wave phases across modes per seed, FK force = ρ·vol·C_M·a_wave |

### 2. Interactive teleop (`teleop.py` — needs a display; `--viser` for browser/headless)

```bash
python teleop.py                       # drive it, live force arrows, G toggles disturbances
python teleop.py --disturb             # start with current+waves+kicks ON
python teleop.py --ctrl dobmpc --goto-origin --disturb   # DOB-MPC holds the origin
python teleop.py --no-hydro            # thruster-only feel (gravity+hydro off)
python teleop.py --observe             # DON'T pilot: release from rest, watch the flow carry it
python teleop.py --viser               # headless: browser UI over SSH/Tailscale, same controls as buttons
python teleop.py --selftest            # headless key-mapping self-check, then exit
```

**Keys** (focused viewer window; commands **latch** until changed): `W/S` surge ±x ·
`Q/E` sway ±y · `R/F` heave ±z · `A/D` yaw · `Z/C` roll · `X` stop (zero thrust) ·
`G` toggle disturbances · `V` toggle force/flow arrows · (observe only) `H` recenter,
`N` re-draw random headings. `--managed` reads keys from the focused *terminal*
instead (add `--pynput` for global capture) but cannot run the autonomous missions.

| flag (default) | meaning |
|---|---|
| `--ctrl {pd,pid,mpc,dobmpc}` (pid) | controller used by `--goto-origin` / `--square` |
| `--disturb` (off) | start with disturbances ON; `G` toggles at runtime |
| `--no-hydro` (off) | disable hydrodynamics AND gravity — thruster-only feel (excludes `--observe`) |
| `--waves {spectrum,classic}` (spectrum) | irregular JONSWAP spectrum vs the 3 classic sinusoids |
| `--sea HS,TP` (0.20,4.0) | sea state for `--waves spectrum`: significant wave height [m], peak period [s] |
| `--scale S` (1.0) | scale the latched command magnitudes (base: surge 8 N, sway 15 N, heave 20 N, yaw 6 N·m, roll 3 N·m) |
| `--viser` / `--remote` (off), `--port` (8080) | headless browser viewer: same controls as buttons/sliders, force arrows, monitor panels, manual CSV Record/Stop |
| `--managed` (off), `--pynput` (off) | old managed viewer (no arrows, keys from the terminal); `--pynput` = global key capture |
| `--no-arrows` (off) | start with the force/flow arrow overlay hidden (`V` re-enables) |
| `--plot` (off) | console sparklines of drag/wave/kick force magnitudes |
| `--monitor` (on) / `--no-monitor`, `--monitor-window` (30 s) | live dashboard: separate-process pyqtgraph window locally, browser panels under `--viser` |
| `--selftest` (off) | headless: presses each key programmatically, asserts the FLU DOF + sign |
| observe-mode flags | see below |
| mission flags | see §3 |

**Observe (free drift) — `--observe` / `--drift`.** Don't pilot the ROV at all: it is
released from rest with **thrust held at 0** and **disturbances forced ON**, so you can
watch the current + waves carry and rock it. Drive keys are disabled (only `G` toggles the
flow, `H` recenters now); it drifts freely **anywhere inside the water volume** and
**auto-recenters** back to the release pose only when it LEAVES the water (the
`pool_water_surface` extent — horizontal edges + the waterline; ~15 s per crossing at the
default sea state). `--recenter-radius R` swaps in a fixed drift distance instead;
`--no-recenter` lets it drift off entirely. Recenter only rewrites state (like the viewer's
*Reset pose*), so **dynamics/model are untouched** (verified: the free-drift rollout is
byte-identical with vs without the observe gate). The pool floor + animated water surface
are on by default in teleop (`POOL_TAGS=0` for a bare scene). Also works headless over
`--viser` (a **Recenter** button appears; drive buttons are hidden). Preview:
[assets/screenshots/pool/pool_observe_drift.mp4](../assets/screenshots/pool/pool_observe_drift.mp4).

**Random headings in observe.** `--observe` draws the **wave heading β** and **current
heading θ_c** *randomly* in [0,360°) each launch (the legacy-model equivalents of the
experiment config's `beta_bar_deg` / `theta_c_deg`), so you can watch how the combined
wave+current direction acts on the drifting ROV. Press **N** (or a viser button) to re-draw
new headings live; the status line shows `wavβ … curθ …`. Pin or seed them:

```bash
python teleop.py --observe                              # random β, θ_c each launch (N to re-draw)
python teleop.py --observe --dir-seed 7                 # reproducible random draw
python teleop.py --observe --wave-deg 0 --current-deg 90 --current-speed 0.3   # pin both
python teleop.py --observe --no-random-dirs             # fixed +x (both headings 0°)
```

The headings are a **disturbance scenario** setting (current/wave direction + speed), not a
plant change — they feed the existing `disturbances.py` field, which hydro reads live.

Observe-mode flags: `--recenter-radius R` (optional FIXED drift distance; default = recenter
at the water boundary, 3 m fallback on a bare scene), `--no-recenter` (drift freely; manual
`H` still works), `--wave-deg` / `--current-deg` (pin a heading), `--current-speed`
(0.20 m/s), `--dir-seed` (reproducible random draw), `--no-random-dirs` (fixed +x).
See [docs/05_TELEOP.md](docs/05_TELEOP.md).

### 3. Autonomous missions (needs a display or `--viser`; records to `recordings/`)

`--square` approaches the origin, auto-starts recording, tracks a square for N
laps, then auto-stops and saves; `--goto-origin` just flies to the origin:

```bash
python teleop.py --square --ctrl dobmpc --disturb
python teleop.py --square --ctrl pid  --disturb --laps 10 --square-size 1.0 --square-speed 0.15
python teleop.py --goto-origin --ctrl dobmpc --start 2,1.5,-1,45
```

Mission flags: `--laps` (10), `--square-size` (1.0 m), `--square-speed` (0.15 m/s),
`--start X,Y,Z,YAWDEG` (2,1.5,-1,45 — initial pose). The autonomous missions use
the **realistic T200 actuator by default** (deadband / fwd-rev asymmetry / motor
lag / voltage), since they exist to predict the real robot. `--ideal-thrusters`
reverts to the ideal force path (commanded == realized); `--thruster-voltage 0.72`
sets the battery thrust scale (default `0.72` = 4S nominal 14.8 V,
datasheet-grounded — see [docs/03_THRUSTERS.md](docs/03_THRUSTERS.md) and
`analyze_t200_voltage.py`).

Each run writes `recordings/<YYYYMMDD>/<timestamp>_square_<ctrl>.csv` **plus a
sidecar `<...>.meta.json`** capturing the full run manifest — controller + solver
config, actuator config (`run.thrusters`), trajectory, and the exact disturbance
schedule **including every kick event**. (Under `--viser`, the manual Record/Stop
buttons likewise write `<ts>_teleop.csv` or `<ts>_origin_<ctrl>.csv`.) `--managed`
cannot run missions (no thread-safe per-step control hook).

### 4. Experiments (headless, seed-controlled)

#### Disturbance-mode comparison matrix — `experiments.run_compare`

3 controllers × 5 finite-depth modes (NONE = still-water baseline, C = current,
CD = +drift, CW = current+waves, CDW = all) × seeds × current/wave headings,
with the disturbance realization shared per (mode, seed, heading) for a fair
comparison. Scenarios (DP rejection + square tracking) and every parameter live
in the YAML — no code edit needed. ONE invocation produces the whole result set:
metrics CSVs, per-run trajectory CSVs + meta, and every figure (including the
`trajectory_compare_*` overlays that previously required manual `run_viewer`
runs + `plot_trajectories`).

Heading sweep (`experiment.directions`). Default (shipped `base.yaml`):
**random sampling — `n_random: N` is the only knob**, drawing N (current, wave)
heading pairs (current from `direction_seed`, wave independently from
`wave_heading_seed` when `sweep_wave_heading: true`; drop that flag to keep the
wave heading fixed at `beta_bar_deg`). Alternative: `pairing: grid` runs EVERY
`headings_deg` (current) × `wave_headings_deg` (wave) combination — e.g.
`[0, 90, 180, 270]` × `[0, 90]` = 8 sweep points; random headings on the grid
via `n_random` / `n_random_wave` instead of the explicit lists. Explicit
`headings_deg` without `pairing` = legacy paired behaviour, unchanged.

```bash
python -m disturbance.test_waves && python -m disturbance.test_env   # 33 unit asserts first
python -m experiments.run_compare --config config/base.yaml --smoke  # tiny pipeline check
python -m experiments.run_compare --config config/base.yaml          # full matrix (parallel)
python -m experiments.run_compare --config config/base.yaml --jobs 8 --ctrls pid,dobmpc --seeds 0,1 --dirs 10
```

| flag (default) | meaning |
|---|---|
| `--config PATH` | experiment YAML (`site` / `waves` / `current` / `inertia` / `sim` / `experiment` blocks) |
| `--smoke` (off) | 5 s, pid-only, seed 0, single heading — end-to-end pipeline check |
| `--ctrls LIST` (config) | controller subset override, e.g. `pid,mpc` |
| `--seeds LIST` (config) | seed override applied to every scenario block |
| `--dirs N` (config) | override with N random current headings — legacy paired sweep, replaces any `directions` block incl. `pairing: grid` (keeps the config's `direction_seed`) |
| `--T SEC` (config) | DP duration override (square runs are laps-bounded instead) |
| `--jobs N` (min(cpu, 16)) | parallel worker processes; `1` = serial. acados is pre-built once in the parent and workers load it (`DOBMPC_ACADOS_BUILD=0`) |

Writes `recordings/<date>/compare_<ts>/`: `results.csv` (mean±std + DRR),
`results_raw.csv` (one row per run), `runs/` (per-run `traj_*.csv` +
`meta_*.json` in the run_viewer schema, filenames tagged
`_c<current°>_w<wave°>` — `plot_trajectories.py --dir <...>/runs` reads them
standalone), `figures/` (per-mode time histories, metric bars, direction
summary, `trajectory_compare_{NONE,C,CD,CW,CDW,ALLMODES}.png` with ALL sweep
headings overlaid per panel, and a controller-independent disturbance
`selfcheck/`), a `config.yaml` snapshot, and `meta.json` (the plant variant
`rov_model`, the exact PID gain set in effect, and run context — every result
folder is self-describing). `experiment.record_runs` (default **true**) gates
the per-run CSVs + trajectory figures — set `false` for huge sweeps (~0.4
MB/run). Config knobs: `inertia.fk_mode` (froude_krylov | morison_ca | off),
`experiment.{primary,secondary}`, `experiment.directions.*`.

#### Live single-run viewer — `experiments.run_viewer`

Watch exactly ONE `--ctrl {pid,mpc,dobmpc}` × `--mode {NONE,C,CD,CW,CDW}` run the
square in real-time MuJoCo (call it once per combination). Same
`build()`/`DisturbanceEnv` as `run_compare`, so the watched trajectory matches the
batch. Saves the full-run trajectory CSV and an mp4 of just ONE lap (the last,
settled lap by default — a full-run video would be huge).

```bash
python -m experiments.run_viewer --config config/base.yaml --ctrl dobmpc --mode CDW
python -m experiments.run_viewer --config config/base.yaml --ctrl pid --mode C --headless  # no window (CSV+mp4 only)
```

| flag (default) | meaning |
|---|---|
| `--ctrl`, `--mode` (required) | the single (controller, disturbance-mode) combination to run |
| `--seed` (0) | disturbance realization seed |
| `--dir-deg` (0.0) | fixed current heading [deg]; the waves' β̄ stays put |
| `--laps` / `--size` / `--speed` (config) | square geometry/speed overrides |
| `--record-lap` (last) | which lap goes to the mp4: `last` \| `first` \| `middle` \| lap index |
| `--heading {follow,fixed}` (follow) | face the travel direction vs keep yaw 0 |
| `--yaw-rate DEG/S` (60) | heading slew-rate limit for smooth corner turns |
| `--video-hz` (30), `--video-size` (720x480) | mp4 frame rate / resolution (needs opencv-python) |
| `--no-arrows`, `--no-video` (off) | skip the force-arrow overlays / skip the mp4 |
| `--headless` (off) | no on-screen window; offscreen render only (`MUJOCO_GL=egl` auto-selected when there is no display) |
| `--out DIR` (`recordings/<date>/square_view`) | output folder; repeated invocations accumulate here |

Outputs per invocation: `traj_square_<mode>_<ctrl>_seed<seed>_dir<deg>.csv` (full
run: position + reference + lap), `lap_square_<...>.mp4` (the one recorded lap),
`meta_square_<...>.json` (all parameters + disturbance snapshot).

#### Trajectory overlay figure — `experiments.plot_trajectories`

Publication-quality XY overlay of the PID / MPC / DOB-MPC paths from the
`run_viewer` CSVs, over the reference square, with per-controller radial RMS in
the legend. Pure post-processing (matplotlib only, no MuJoCo).

```bash
python -m experiments.plot_trajectories               # latest square_view folder, mode CDW
python -m experiments.plot_trajectories --all-modes   # 2×3 grid: one panel per mode NONE…CDW
```

Flags: `--dir` (square_view folder; default = latest under `recordings/`),
`--mode {NONE,C,CD,CW,CDW}` (CDW), `--seed` (0), `--dir-deg` (0), `--size` (1.0),
`--ctrls` (pid,mpc,dobmpc), `--all-modes`, `--out`. Writes
`<dir>/trajectory_compare_<mode|ALLMODES>.png` (300 dpi).

#### Station-keeping (DP) comparison — `dobmpc.eval_dp`

```bash
python -m dobmpc.eval_dp --ctrls pid,mpc,dobmpc --seed 0 --T 60
```

PID vs MPC vs DOB-MPC holding the origin on the SAME plant / seed / disturbance
(the legacy `disturbances.py` model: current + JONSWAP waves + kicks) from the
same `--start` offset (0.1,0.05,0). Prints the steady-window (t ≥ 10 s) metrics
table (radial RMS, DC bias, wave-residual std, pitch, EAOB ŵ_x) and saves a
6-panel comparison figure to `recordings/<date>/dp_compare_<ts>.png`.

#### Actuator-realism ablation — `ablation_thrusters.py`

```bash
python ablation_thrusters.py   # no flags: 3 ctrl × {ideal, realistic, realistic-LV} × 5 seeds
```

Re-runs the DP task under the ideal force path vs the realistic T200 model
(deadband / asymmetry / lag) vs realistic + 0.85× low-voltage, averaged over
seeds 0–4, to quantify the sim-to-real actuator gap per controller (deadband
limit-cycle jitter, voltage DC sag). Prints the degradation table and writes
`docs/figs/ablation_thrusters.png`. Takes several minutes (45 runs).

### 5. Analysis of recordings

| command | what it does |
|---|---|
| `python analyze_square3.py` | off-path + time-referenced error, orientation stats, and control effort for the 3 recorded square CSVs (PID/MPC/DOB-MPC) → comparison table + `recordings/20260615/square3_compare.png` |
| `python analyze_acados_vs_before.py` | before/after IPOPT→acados solver-swap deltas on the same square mission (PID as noise floor) → per-metric tables + `recordings/20260615/acados_vs_before.png` |
| `python analyze_t200_voltage.py` | parses the official T200 datasheet xlsx (stdlib-only) and re-derives the thruster `voltage_scale` for 14.8 V; MATCH-checks the live constant (0.72) |

The first two read **hard-coded folders/filenames** under `recordings/20260615/` —
edit the `DIR`/`RUNS`/`PAIRS` constants at the top to point at your run. All three
are headless (no flags).

### 6. Verification (V&V — run after touching dynamics or the solver)

| command | what it proves |
|---|---|
| `python verify_hydro.py` | 32-check first-principles smoke on hydro.py in still water: net buoyancy, terminal velocity / drag anisotropy, restoring pendulum, added mass, energy sanity, frame checks. Flag: `--no-plot`; figures → `docs/figs/hydro_T*.png`. Takes minutes |
| `python verify_hydro_precise.py` | rigorous 4-tier superset: structural Fossen identities (hard gate), order-of-accuracy convergence vs a DOP853 continuum reference, frame/Galilean invariance, added-mass-lag transfer-function fidelity. Flags: `--tier 1234`, `--ladder 2,1,0.5,0.25,0.125` [ms], `--no-plot`. Slow |
| `python verify_acados.py` | acados NMPC ≡ IPOPT NMPC: equivalence (worst max\|Δu\| < 0.25 N on interior states) + SQP-RTI timing vs the 50 ms @ 20 Hz budget. First run code-generates the solver into `dobmpc/_acados_gen/` |
| `python verify_meta.py` | the recorder sidecar manifest is complete: the disturbance snapshot round-trips (kicks/waves reproduced exactly), CSV header matches, solver + effective PID gains captured. Run from the package dir |
| `/home/bdml/miniforge3/envs/robust-mjx/bin/python verify_gpu_mjx.py` | Phase-0 GPU env check (**`robust-mjx` env, NOT `robust`**): JAX sees the CUDA GPU and a jitted MJX rollout actually runs on it; loading bluerov.xml under MJX is a non-gating bonus check (the CPU passive-callback hydro does not run under MJX) |

### 7. Asset & figure generators (occasional)

| command | what it (re)generates |
|---|---|
| `python gen_pool_apriltags.py` | the visual pool AprilTag floor + water scene: `apriltags/*.png` (round-trip verified), `tag_floor.xml`, the two `scene_*_tags.xml` wrappers. `--selftest` renders 2 tiles and re-detects them — **note: it overwrites `tag_floor.xml` with just those tiles, rerun the full build after**. All layout/water/sky knobs: see [Pool AprilTag floor](#pool-apriltag-floor-visual-pool_tags) above |
| `python extract_meshes.py` | gray `meshes/bluerov_body.obj` + `bluerov_thruster.obj` from the MarineGym USD. Flags: `--body-faces` (40000), `--thruster-faces` (3000) |
| `python extract_meshes.py --colored` | the 4-part Heavy color skin `meshes/rov_body_{cyan,white,black,silver}.obj` that `bluerov_heavy.xml` references (split by USD GeomSubset material). Flag: `--colored-faces` (55000 total budget) |
| `python generate_bluerov_xml.py` | **overwrites `bluerov.xml`** from the authoritative BlueROV.usd (mass/COM/inertia via UsdPhysics, thruster sites, force actuators). Hand edits to that file are lost on rerun. No flags |
| `python compute_heavy_inertia.py` | (stdout only) derives the Heavy rotational inertia from the bluerov2 tensor via the vertical-thruster parallel-axis term, + an m_v sensitivity sweep. No flags |
| `python plot_wave_spreading.py` | slide figures in `../assets/screenshots/`: `wave_beta_vectors.png` + `wave_spreading_s_compare.png` (β̄ is the MEAN of a cos^2s directional spread, not a single wave direction). Flags: `--beta-deg` (25), `--seed` (0), `--out-dir`, `--sea-lobe` (adds `wave_beta_spreading.png`). Wave params are hardcoded copies of `config/base.yaml` — keep them in sync |

The USD-reading generators (`extract_meshes.py`, `generate_bluerov_xml.py`) need
`usd-core` and the MarineGym checkout at `../external/MarineGym/`;
`gen_pool_apriltags.py` needs `pupil_apriltags` + `cv2` and reads
`../config/{config,tag_map}.yaml`.

### 8. Environment variables

| var | default | scope | effect |
|---|---|---|---|
| `ROV_MODEL` | `heavy` | **dynamics** | plant variant: `heavy` (8 thrusters, rank 6, NU=6) or `bluerov2` (6 thrusters, rank 5, NU=4). Read at import by `rov_model.py`. **The only env var that changes dynamics** |
| `POOL_TAGS` | `0` (teleop flips it to `1`) | visual | truthy loads the AprilTag pool wrapper scene (tags + seabed + animated water). Δ=0 verified; an explicit value always wins over teleop's default |
| `WATER_WAVE_LAMBDA` | unset (faithful) | visual | stylized visual wavelength [m] for the animated water surface (e.g. `0.9` = clearly sloshing); physics untouched |
| `WATER_WAVE_AMP` | `1.0` | visual | visual wave-height gain for the animated surface |
| `MUJOCO_GL` | glfw (auto-`egl` when headless) | rendering | GL backend: `glfw` / `egl` / `osmesa`; must be set before `import mujoco` |
| `DOBMPC_ACADOS_BUILD` | unset (build) | infra | `0` = load the pre-compiled acados solver from `dobmpc/_acados_gen/` instead of regenerating (run_compare sets this for its parallel workers) |
| `ACADOS_SOURCE_DIR` | `/home/bdml/acados` | infra | acados install location; its libs are ctypes-preloaded with RTLD_GLOBAL, so no shell `LD_LIBRARY_PATH` export is needed. A wrong path silently falls back to IPOPT |
| `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS` | `1` (set by run_compare) | perf | BLAS/OpenMP thread caps so `--jobs N` workers don't oversubscribe cores |
| `PYQTGRAPH_QT_LIB` / `QT_API` | `PyQt5` / `pyqt5` | UI | Qt binding for the monitor dashboard |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` (set by verify_gpu_mjx) | GPU mem | stops JAX preallocating ~75% of VRAM on the display-sharing GPU (robust-mjx only) |

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
