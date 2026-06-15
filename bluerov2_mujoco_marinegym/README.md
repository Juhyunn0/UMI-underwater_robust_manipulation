# BlueROV2 → MuJoCo (MarineGym import)

A BlueROV2 MuJoCo simulator imported **from MarineGym's own BlueROV asset**, built
as an MPC/RL testbed for robust underwater control.

> **📖 Full project memory is in [`docs/`](docs/) — start at
> [docs/00_OVERVIEW.md](docs/00_OVERVIEW.md).** New session with no chat history:
> read that first (goal, engine choice, FLU convention, roadmap/status), then
> [docs/01_DECISIONS.md](docs/01_DECISIONS.md). The docs hold the decisions,
> parameter values, key findings, and what's verified vs pending.

Status: Phase 1 (rigid body), Phase 2 (thrusters), Phase 3 (hydrodynamics),
Phase 4 (disturbances + domain randomization), and keyboard teleop are **done &
verified**. Next: Phase 5 (sensors). Phase 0 (Linux + RTX 5090 / MJX setup) is
pending. See the status table in the overview.

## Quick start (macOS, base MuJoCo, CPU)

```bash
# from the repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -U mujoco                       # base MuJoCo only — no MJX/JAX/CUDA

cd bluerov2_mujoco_marinegym
python test_load.py          # Phase 1: load, stats, zero-control stability
python test_thrusters.py     # Phase 2: thruster directions + allocation matrix
python test_hydro.py         # Phase 3: neutral buoyancy / self-righting / terminal velocity
python test_disturbances.py  # Phase 4: current / waves / kicks distinct + DR
python teleop.py             # drive it around with live force arrows (G toggles disturbances)
```

Teleop run commands per platform: **Ubuntu (target)** `python teleop.py` (force
arrows, plain python); **macOS preview** `mjpython teleop.py` from the no-space venv
`~/bluerov_venv`; **macOS quick check** `python teleop.py --managed` (no arrows). See
[docs/05_TELEOP.md](docs/05_TELEOP.md).

Running the model needs only `mujoco` + `numpy`. Regenerating the asset from the
USD additionally needs `usd-core`, `trimesh`, `fast-simplification` (build-time
only). The committed `bluerov.xml` + `meshes/` are self-contained and portable.

## Layout

```
bluerov2_mujoco_marinegym/
├── docs/                    # ← project memory (start at 00_OVERVIEW.md)
│   ├── 00_OVERVIEW.md       #   entry point: goal, engine, architecture, roadmap/status
│   ├── 01_DECISIONS.md      #   locked decisions + rationale (FLU, no param-mixing, ...)
│   ├── 02_MODEL.md          #   Phase 1 rigid body (provenance, values, frame)
│   ├── 03_THRUSTERS.md      #   Phase 2 thrusters, allocation, underactuation
│   ├── 04_HYDRO.md          #   Phase 3 buoyancy/added-mass/drag
│   ├── 05_TELEOP.md         #   keyboard teleop
│   ├── 06_ENVIRONMENT.md    #   macOS drafting env, Linux/MJX runtime, portability
│   └── 07_DISTURBANCES.md   #   Phase 4 current/waves/kicks + domain randomization
├── bluerov.xml              # the MJCF (rigid body + 6 thruster sites + 6 actuators)
├── meshes/                  # real MarineGym meshes (body + T200 thruster), from the USD
├── thrusters.py             # Phase 2: T200 curve, allocation B / pinv(B), command helpers
├── hydro.py                 # Phase 3: Fossen buoyancy/restoring/added-mass/drag callback
├── disturbances.py          # Phase 4: current + waves + kicks + DR sampler
├── teleop.py                # keyboard teleop + live force-arrow viz (launch_passive); --managed for old viewer
├── test_load.py · test_thrusters.py · test_hydro.py · test_disturbances.py  # per-phase verification
├── generate_bluerov_xml.py · extract_meshes.py        # regenerate MJCF / meshes from USD
├── marinegym_assets/        # MarineGym BlueROV.yaml (hydro+rotor coeffs) + config.yaml
└── preview.png              # headless render (verification)
```

## Conventions you must not break (see docs for detail)

- **Frame is FLU** (x fwd, y left, z up), gravity (0,0,−9.81). No NED in the sim;
  NED↔FLU only at the MPC boundary.
- **MarineGym-derived model is canonical** — don't mix parameters with the
  hand-built `bluerov2_mujoco_dobmpc/` model.
- **Pitch is underactuated** (rank(allocation)=5) — command surge/sway/heave/yaw/
  roll, never pitch.
