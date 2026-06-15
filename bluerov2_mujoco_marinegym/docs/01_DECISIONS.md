# 01 — Locked decisions & rationale

These are settled. Do not re-litigate or silently reverse them; if you must
change one, update this file and say why.

## D1 — The simulation is FLU; convert NED↔FLU only at the MPC boundary

- **Sim frame is FLU**: x forward, y **left/PORT**, z **up**; gravity (0,0,−9.81).
  Every model value, site, thruster axis, allocation matrix, and hydro force is
  FLU. **No NED in the sim.**
- The DOB-MPC literature (and this project's reference dobmpc implementation) is
  written in **NED** (x north, y east, z down). That conversion is confined to the
  **controller boundary (Phase 7)** — the MPC will convert its NED state/command
  to/from FLU there, and nowhere else.
- **Why:** keeping one frame in the sim eliminates an entire class of sign-flip
  bugs. MarineGym's own code does an internal FLU↔FRD flip for its Fossen terms;
  we avoid that because for **diagonal** added-mass/damping the flip cancels out
  (verified), so computing directly in FLU is identical and clearer.
- **Gotcha for a future session:** if you see code negating y/z or roll/pitch/yaw
  "to match the paper," that belongs only in the Phase-7 boundary, not in the sim.

## D2 — The MarineGym-derived model is canonical; do NOT mix parameters

- Canonical = `bluerov2_mujoco_marinegym/` (this folder): values come from
  **MarineGym's BlueROV asset** (`external/MarineGym/.../usd/BlueROV/`) and its
  `BlueROV.yaml`.
- There is a **separate, older hand-built model** in the same repo:
  `bluerov2_mujoco_dobmpc/` (a box-geom BlueROV2 in **NED**, with its own
  parameters and a full DOB-MPC stack) and `bluerov2_mujoco_scratch/`. These were
  reference/prototypes.
- **Do NOT copy parameters between the two.** They use different frames (NED vs
  FLU), different mass/inertia/coefficients, and different thruster geometry.
  Mixing them produces a physically inconsistent model that looks plausible but is
  wrong. When in doubt, every number in the canonical model traces to MarineGym.
- The dobmpc model is still useful as a **method reference** (e.g. the lagged
  added-mass injection technique) — borrow technique, never numbers.

## D3 — CB offset coBM = 0.01 m (center of buoyancy 0.01 m above COM, +z)

- From MarineGym's `BlueROV.yaml` (`coBM: 0.01`). The CB sits **0.01 m above the
  COM** along body +z (FLU). The restoring (self-righting) moment comes from this
  offset.
- It is deliberately **small** (weak metacentric restoring, max B·coBM ≈ 1.11 N·m).
  This matters: it is too weak to counter the surge→pitch coupling at high thrust
  (see [03_THRUSTERS.md](03_THRUSTERS.md) / [04_HYDRO.md](04_HYDRO.md)). Don't
  "fix" it by inventing a larger coBM — keep MarineGym's value.

## D4 — Standard 6-thruster BlueROV2 (not the 8-thruster Heavy)

- We use MarineGym's **`BlueROV`** asset = the standard vectored-**6** layout:
  4 horizontal thrusters at ±45° (surge/sway/yaw) + 2 vertical (heave/roll).
- MarineGym also ships **`BlueROVHeavy`** (8 thrusters) at
  `external/MarineGym/.../usd/BlueROVHeavy/`. We are **not** using it. If a future
  phase wants the Heavy variant, that's a deliberate new decision, not a default.

## D5 — Engine = MuJoCo (MJX on GPU)

**Why MuJoCo/MJX:**
- **RTX 5090 / Blackwell support** via JAX (cu128) — the target hardware.
- **Claude-Code-friendly**: the whole robot is one human-readable MJCF + Python;
  easy to inspect, diff, and regenerate. No opaque binary scene graph.
- **Built-in hydro/fluid options** and easy external-force injection
  (`xfrc_applied`, `set_mjcb_passive`) for Fossen-style hydrodynamics.
- **MJPC** gives a ready MPC/optimal-control playground for Phase 7.
- **MJX scales to massively-parallel RL** on GPU for Phase 8.

**Rejected alternatives:**
- **Stonefish** — purpose-built underwater sim, but C++/ROS-centric, harder to
  drive from Python/Claude Code, no GPU-parallel RL path, weaker for our MPC+RL
  testbed goal.
- **Isaac Sim / Isaac Lab** — heavy, USD/Omniverse runtime, GPU-vendor-locked
  workflow, slow iteration, overkill; MarineGym is built on it (see below).
- **MarineGym as a runtime** — it *is* Isaac-Sim-based; we only want its **static
  BlueROV asset + coefficients**, not its runtime. Decision: extract the files,
  rebuild natively in MuJoCo. (This is exactly what Phase 1 did.)

## D6 — Added-mass implementation: explicit lagged/filtered Fossen forces

Documented fully in [04_HYDRO.md](04_HYDRO.md). Short version: apply
`−M_A·ν̇` (and Fossen drag + added-mass Coriolis) as explicit forces via a passive
callback, with ν̇ one-substep-lagged + 0.3 low-pass filtered. Chosen over MuJoCo's
ellipsoid fluid model because explicit forces reproduce MarineGym's **diagonal
coefficients exactly**; chosen over folding added mass into the XML inertia to keep
`bluerov.xml` the pure rigid body. The lag+filter is what keeps it stable when
heave added mass (14.57) exceeds the body mass (11.2).
