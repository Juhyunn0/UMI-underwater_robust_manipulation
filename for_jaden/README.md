# BlueROV2 MuJoCo Underwater Simulator (for Jaden)

A carve-out of `bluerov2_mujoco_marinegym/` containing only the **plant +
disturbances (waves/current) + PID baseline**. Every file is a verbatim copy of
the original; apart from this README, no new code was written.

On top of MuJoCo sit 6-DOF Fossen underwater dynamics (buoyancy/restoring forces,
added mass, linear + quadratic drag), T200 thrusters, and a finite-depth
directional JONSWAP wave field plus a current disturbance field.
**All frames are FLU (x-forward, y-left, z-up).**

---

## 1. Runtime environment

```bash
python -c "import mujoco, numpy, yaml"     # if this works, plant + disturbances run
```

| Purpose | Required packages |
|---|---|
| Plant + disturbances + PID (required) | `mujoco`, `numpy`, `pyyaml` |
| `verify/verify_hydro*.py` | + `scipy`, `matplotlib` |
| `teleop.py --monitor` (ON by default) | + `pyqtgraph`, `PyQt5` (optional: `qdarkstyle`) |
| `teleop.py --viser` (headless browser UI) | + `viser`, `trimesh` |
| `teleop.py --managed --pynput` | + `pynput` |
| `tests/test_water_viz.py` | EGL offscreen GL context |
| `verify/verify_gpu_mjx.py` | **separate env** — JAX (cuda) + MuJoCo MJX |

The verification below was run on this combination (2026-08-13):
Python 3.12.13 / mujoco 3.9.0 / numpy 1.26.4 / pyyaml 6.0.3 / scipy 1.17.1 /
matplotlib 3.10.9. numpy 2.x was not tested.

---

## 2. Environment variables — these two are all of them

| Variable | Default | Effect |
|---|---|---|
| `ROV_MODEL` | `heavy` | Selects the vehicle variant. **The only switch that changes dynamics.** |
| `POOL_TAGS` | `0` (but `1` in `teleop.py`) | AprilTag pool floor + water surface scene. **VISUAL ONLY** |

`ROV_MODEL` is resolved through a single registry in [rov_model.py](rov_model.py),
which is the single source of truth — so the plant (MJCF) and the controller
cannot end up assuming different vehicles.

| `ROV_MODEL` | MJCF | Mass | nu | Notes |
|---|---|---|---|---|
| `heavy` (default) | `bluerov_heavy.xml` | 11.5 kg | 8 | 8 thrusters, allocation rank 6 = **fully actuated** |
| `heavy_c3` | `bluerov_heavy_c3.xml` | 13.2 kg | 8 | + MarineSitu C3 stereo camera, negatively buoyant |
| `heavy_gripper` | `bluerov_heavy_gripper.xml` | 13.724 kg | 9 | + Newton gripper (ctrl index 8) + C3 |

The provenance of each mass/inertia/volume value is written in the comment next to
that entry in [rov_model.py](rov_model.py) (the heavy inertia is **derived** from
the BlueROV2 tensor via the parallel-axis theorem, not measured from CAD — see the
comment).

```bash
python tests/test_load.py                          # heavy (default)
ROV_MODEL=heavy_gripper python tests/test_load.py  # gripper variant
POOL_TAGS=1 python teleop.py                       # pool scene (teleop has it ON anyway)
```

Every geom added by the `POOL_TAGS` scene has `contype=0 conaffinity=0` and the
MuJoCo fluid model is off, so toggling it does not change the dynamics.

---

## 3. File map

### Plant
| File | Contents |
|---|---|
| [rov_model.py](rov_model.py) | Variant registry. Decides which MJCF/YAML/masses are used |
| [hydro.py](hydro.py) | Fossen hydrodynamics. Installed as a MuJoCo **passive callback** (`Hydrodynamics(model, disturbance).install()`) |
| [thrusters.py](thrusters.py) | T200 thrust curve + allocation matrix B. Two paths: ideal and realistic (deadband, forward/reverse asymmetry, motor lag, voltage) |
| `bluerov_heavy*.xml`, `scene_*_tags.xml`, `tag_floor.xml` | MJCF. `scene_*` are the `POOL_TAGS` wrappers |
| `bluerov.xml` | Legacy rank-5 BlueROV2. **Not a selectable `ROV_MODEL`** — kept only as a fixed fixture for the hydro/thruster verification scripts |
| `meshes/`, `apriltags/tag_mosaic.png` | Meshes/textures referenced by the MJCF |
| `marinegym_assets/*.yaml` | Fluid coefficients from MarineGym (added mass, linear/quadratic damping, volume, coBM) |

### Disturbances
| File | Contents |
|---|---|
| [disturbance/](disturbance/) | **Finite-depth** disturbance package. `waves.py` (directional JONSWAP + cos^2s spreading + dispersion relation), `current.py` (Gauss–Markov current + drift), `env.py` (mode gating + Froude-Krylov inertial force), `config.py` (YAML loader) |
| [disturbances.py](disturbances.py) | Legacy disturbance model (current + JONSWAP waves + Poisson kicks + domain randomization). This is the one `teleop.py` uses |
| [config/base.yaml](config/base.yaml) | Configuration for `disturbance/`: site/waves/current/inertia blocks |

The 5 modes in `disturbance/env.py`: `NONE` (still-water baseline) / `C` (current) /
`CD` (+ drift) / `CW` (current + waves) / `CDW` (everything). With the same seed the
wave phases are identical across modes, so mode-to-mode comparison is
apples-to-apples.

**Caution: `disturbance/` and `disturbances.py` are two different models.** The
former is the proper finite-depth model; the latter is the older model that
`teleop.py` is wired to.

### Controller (baseline)
| File | Contents |
|---|---|
| [controller.py](controller.py) | `PoseController` — PD/PID attitude and position control. Gains derived by pole placement (`GAINS_HEAVY`) |
| [dobmpc/params.py](dobmpc/params.py) | **Only the constants `controller.py` imports** were kept (sensor noise sigmas, `PID_STATE_SOURCE`, `DT_CTRL`) |

By default `controller.py` consumes **noisy 20 Hz state**
(`PID_STATE_SOURCE="meas"`). To use the clean 500 Hz true state, pass
`PoseController(..., meas_noise=False)`.

### Run / record
| File | Contents |
|---|---|
| [teleop.py](teleop.py) | Keyboard teleop + force-arrow overlay + autonomous missions (`--square`, `--goto-origin`) |
| [mission.py](mission.py) | Square-trajectory mission |
| [recorder.py](recorder.py) | CSV recording + run manifest (`.meta.json`) — stores enough to reproduce the disturbance schedule |
| [monitor.py](monitor.py) | pyqtgraph live dashboard (separate process) |
| [water_viz.py](water_viz.py) | Water-surface heightfield animation. **VISUAL ONLY** |
| [compute_payload_inertia.py](compute_payload_inertia.py) | Composed payload inertia for C3/gripper (referenced by the variant tests) |

### Tests / verification
12 files under `tests/`, 3 under `verify/`. All are standalone runnable
(`python tests/test_x.py`).

---

## 4. Trying it out

```bash
# Plant
python tests/test_load.py          # model load, mass, thruster sites, stability under gravity
python tests/test_thrusters.py     # T200 curve, allocation B, FLU signs, coupling
python tests/test_hydro.py         # neutral-buoyancy hover, self-righting, drag terminal velocity (a few minutes)

# Disturbances
python -m disturbance.test_waves   # 12 asserts on the finite-depth wave field
python -m disturbance.test_env     # 22 asserts on mode gating / Gauss-Markov / FK force
python tests/test_disturbances.py  # unthrusted drift converges to the current velocity, etc.

# PID
python tests/test_controller.py    # PD/PID return-to-origin × {still water, 0.2 m/s current}
python tests/test_square_mission.py

# Variants
ROV_MODEL=heavy_c3      python tests/test_heavy_c3.py
ROV_MODEL=heavy_gripper python tests/test_heavy_gripper.py

# Interactive (needs a display)
python teleop.py                          # pilot it. G toggles disturbances, V toggles force arrows
python teleop.py --square --ctrl pid --disturb
python teleop.py --selftest               # headless key/DOF self-check
python teleop.py --viser                  # headless: browser UI
```

`teleop.py --ctrl` still lists `mpc`/`dobmpc` as choices, but they **do not work in
this folder** (§5). Use only `pd`/`pid`.

---

## 5. What is **not** in this folder

Things that exist in the original and were deliberately left out.

| Left out | Why |
|---|---|
| The body of `dobmpc/` (`mpc.py`, `mpc_acados.py`, `eaob.py`, `fossen.py`, `frames.py`), `dobmpc_controller.py` | The MPC / DOB-MPC stack. It needs `casadi` plus a **from-source acados build**, which is a heavy setup burden. `dobmpc/params.py` and `__init__.py` were kept only because `controller.py` imports the sensor sigma constants from there (the docstring in `dobmpc/__init__.py` is verbatim from the original, so it mentions modules that are not here) |
| `experiments/` (`run_compare.py`, `run_viewer.py`, `plot_trajectories.py`, `wave_preview.py`) | All of them import `dobmpc_controller` at module top level, so they cannot run without the MPC stack |
| `tools/` | Asset generators (mesh extraction, MJCF generation, AprilTag floor generation) and recording analyzers. Their outputs are already included |
| `docs/`, `recordings/` | Design documents / experiment recordings (13 GB) |
| `verify/verify_{acados,eaob,state_source,meta}.py` | MPC / EAOB / recorder verification |
| `tests/test_dobmpc.py` | Needs the MPC stack |
| `meshes/c3_camera.obj`, `meshes/c3_payload_frames.json`, `marinegym_assets/*.xlsx`, `apriltags/tag36h11_*.png` | Not referenced by any included MJCF or code (the individual tag PNGs are only for `--tag-mode tiles`) |

If you end up needing the MPC side, copy all of `dobmpc/` + `dobmpc_controller.py` +
`experiments/` from the original folder. The file layout is identical to the
original.

 No code was modified during the copy.
