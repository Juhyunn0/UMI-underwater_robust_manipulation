# 02 — Model (Phase 1: rigid body)

**Status: DONE ✓.** The rigid-body BlueROV2 MJCF, imported from MarineGym's asset.
Code: `bluerov.xml`, `meshes/`, regenerators `tools/generate_bluerov_xml.py` +
`tools/extract_meshes.py`. Verify: `python tests/test_load.py`.

## Provenance — this is the non-obvious part

MarineGym does **not** ship a URDF. Its BlueROV is a **binary Isaac USD crate**
(`external/MarineGym/marinegym/robots/assets/usd/BlueROV/BlueROV.usd` +
`Props/instanceable_meshes.usd`). So "convert the URDF" was impossible; instead:

- The authoritative **mass, inertia, COM, and the 6 thruster mount transforms**
  were read straight out of the USD with **`usd-core`** (`pxr`) and authored into
  `bluerov.xml` by `tools/generate_bluerov_xml.py`.
- The **visual meshes** were extracted from the USD by `tools/extract_meshes.py`
  (`UsdGeom.Mesh` points/faces → welded → decimated → OBJ): body
  `bluerov_body.obj` (307,785 → 40,000 faces) and one T200 `bluerov_thruster.obj`
  (instanced 6× in the MJCF). These are MarineGym's real meshes.
- MarineGym's `config.yaml` documents the USD's own source as
  `bluerov2_description/urdf/BlueROV.urdf` (a public BlueROV2 description) — not
  present locally and not needed, since the USD carries the authoritative numbers.
- `usd-core`, `trimesh`, `fast-simplification` are **build-time only** (regenerating
  the asset). The committed `bluerov.xml` + `meshes/` load with base `mujoco` +
  `numpy` alone and are portable.

To regenerate from the USD: `python tools/extract_meshes.py && python tools/generate_bluerov_xml.py`.

## Frame & world

- **FLU**: x forward, y left, z up. World is standard MuJoCo Z-up, gravity
  defaults to **(0,0,−9.81), ON**. (The Phase 1/2 tests zero gravity only to
  isolate effects; the canonical model has it on.)
- MuJoCo built-in fluid is **off** (`density=0 viscosity=0`); hydro is added
  externally in Phase 3.

## Key values (verified against the compiled model)

| quantity | value |
|---|---|
| total mass | **11.20 kg** (BlueROV2 ≈ 10–11 kg) |
| inertia (diagonal, Ixx,Iyy,Izz) | **0.30375, 0.626, 0.5769** kg·m² |
| COM (body frame) | **(0, 0, 0)** — COM is at the body origin |
| bodies / geoms / sites / meshes | 2 / 8 / 7 / 2 |
| DoF | 6 (one free joint `free`) |
| actuators (added in Phase 2) | 6 (`thr0..thr5`) |

> The table above is the **`bluerov2`** variant (`bluerov.xml`). The
> **`ROV_MODEL=heavy`** variant (`bluerov_heavy.xml`) is the same import pipeline
> with mass **11.5 kg** and **8** thruster sites/actuators (`thr0..thr7`) — see
> [03_THRUSTERS.md](03_THRUSTERS.md) and `rov_model.py`. Its inertia **[0.3291, 0.6347,
> 0.6109]** is *derived* from the bluerov2 tensor by adding the parallel-axis term of
> the vertical-thruster layout change (`tools/compute_heavy_inertia.py`): the farol Heavy USD's
> own [0.21, 0.245, 0.245] is a hand-tuned Gazebo-stability literal, not physical. See the
> 2026-06-18 entry in [CONTROL_METHODOLOGY.md](CONTROL_METHODOLOGY.md).
| collision | one box: center (0,0,−0.05), half-size (0.25,0.175,0.125) m |

The body inertia is set by an explicit `<inertial>`; every geom is
visual/collision-only and does **not** affect mass/inertia (verified: total model
mass == the inertial mass exactly).

## Thruster mount sites (FLU body frame)

Standard BlueROV2 **vectored-6**: 4 horizontal at ±45° + 2 vertical. Positions are
read from the USD; the site **local +X axis is the thrust direction** (used in
Phase 2). All 4 horizontal thrusters sit at **z = −0.0725 m (below the COM)** —
remember this, it causes the surge/sway couplings in [03_THRUSTERS.md](03_THRUSTERS.md).

| site | position (m) | role |
|---|---|---|
| `thruster_0` | (+0.1355, −0.100, −0.0725) | front-right, horizontal |
| `thruster_1` | (+0.1355, +0.100, −0.0725) | front-left, horizontal |
| `thruster_2` | (−0.1475, −0.100, −0.0725) | rear-right, horizontal |
| `thruster_3` | (−0.1475, +0.100, −0.0725) | rear-left, horizontal |
| `thruster_4` | (+0.0025, −0.1105, −0.005) | right vertical |
| `thruster_5` | (+0.0025, +0.1105, −0.005) | left vertical |

(Note the slight fore/aft asymmetry: front arm 0.1355 vs rear 0.1475 — this is
MarineGym's geometry, and it produces a tiny sway→yaw coupling.)

## CB / coBM

COM is at the origin; the **center of buoyancy is 0.01 m above it** (`coBM=0.01`,
body +z). The model file itself has no buoyancy — that is applied in Phase 3, which
uses this offset for the restoring moment. See [04_HYDRO.md](04_HYDRO.md).

## Verified (Phase 1)

`python tests/test_load.py`:
- Loads via `mujoco.MjModel.from_xml_path`, **zero compile warnings**.
- Mass 11.20 kg, inertia as above, 6 thruster sites, correct vectored layout.
- Zero-control free fall (gravity on, no buoyancy yet): Δz ≈ −19.6 m over 2 s,
  matching ½·g·t²; quaternion stays unit; **no NaN / no blow-up**.
- Headless render (`--render`) looks like a BlueROV2 (`preview.png`).

## Gotchas

- COM at the origin (not at the geometric center, not at the battery) is
  MarineGym's choice — it is **above** the horizontal thrusters, which is the
  root of the surge→pitch coupling. Don't "correct" it without re-checking the
  whole chain.
- Meshes are decimated; do not expect a watertight collision mesh — collision is a
  single box on purpose.
