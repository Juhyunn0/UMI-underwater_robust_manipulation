# 04 — Hydrodynamics (Phase 3: buoyancy, added mass, drag)

**Status: DONE ✓.** Code: `hydro.py`. Verify: `python tests/test_hydro.py`
(`--render` for the viewer). FLU, gravity **ON**, MarineGym coefficients only.

## Coefficients (all from `marinegym_assets/BlueROV.yaml`)

| term | value |
|---|---|
| volume V | 0.0113459 m³ |
| coBM (CB above COM, +z) | 0.01 m |
| added mass M_A | [5.5, 12.7, 14.57, 0.12, 0.12, 0.12] |
| linear damping D_L | [4.03, 6.22, 5.18, 0.07, 0.07, 0.07] |
| quadratic damping D_NL | [18.18, 21.66, 36.99, 1.55, 1.55, 1.55] |
| water density ρ | 997 kg/m³ (MarineGym `calculate_buoyancy`) |

Buoyancy **B = ρ·g·V = 110.97 N** vs weight **W = m·g = 109.87 N** → **net +1.10 N**
(slightly positive — realistic). `drag_coef: 0.3` in the yaml is unused by MarineGym
(the D_L/D_NL arrays are the drag), so unused here too.

## Implementation — explicit Fossen forces via a passive callback

Hydro is injected each substep by a MuJoCo **passive-force callback**
(`set_mjcb_passive`), so it runs inside plain `mj_step` **and** inside the managed
viewer (teleop) with no per-step Python loop. MuJoCo's own fluid model is left
**off** (`density=viscosity=0`).

**Why explicit Fossen forces, not MuJoCo's ellipsoid fluid model:** explicit forces
reproduce MarineGym's **diagonal coefficients exactly**; the ellipsoid model would
derive its own coefficients from geometry and not match. (See [01](01_DECISIONS.md) D6.)

Per substep, about the COM (= body origin here):
- **Buoyancy + restoring** — force `[0,0,B]` (world up) applied at the CB =
  `COM + coBM·ẑ_body`, via `mj_applyFT` (the offset point yields the restoring
  moment for free). Weight is MuJoCo's gravity at the COM. CB above COM ⇒ a tilt
  self-rights.
- **Drag** — `−(D_L + D_NL·|ν|)·ν` (Fossen D(ν)ν), body frame.
- **Added-mass Coriolis** — `−C_A(ν)ν` (Fossen, diagonal M_A).
- **Added-mass inertial** — `−M_A·ν̇`, with ν̇ a **one-substep-lagged, low-pass
  (α=0.3) filtered** finite difference of body-frame velocity (`mj_objectVelocity`).

ν = [v(3); ω(3)] is the body-frame velocity at the COM. Signs are first-principles
(drag opposes velocity, added mass opposes acceleration) and verified by behavior,
**not** copied from the dobmpc sign convention.

### Added-mass choice & the stability subtlety (important)

Both translational and rotational added mass are applied as the explicit
lagged/filtered `−M_A·ν̇` force — exactly MarineGym's method — rather than folding
the rotational part into the XML inertia (keeps `bluerov.xml` the pure rigid body).
The lag+filter is **essential for stability**: heave added mass (14.57) **exceeds**
the body mass (11.2), so an *unfiltered* explicit `−M_A·ν̇` diverges (gain >1). The
0.3 low-pass at dt = 2 ms keeps it stable — the same trick as uuv_simulator and this
repo's dobmpc plant. If you ever raise dt or drop the filter, re-check stability.

**Match to MarineGym:** coefficients are identical; only the integration host
differs (MuJoCo RB + lagged added mass vs Isaac). Results agree to the fidelity of
the added-mass lag.

## Verified (Phase 3) — `python tests/test_hydro.py`

1. **Neutral buoyancy** — no thrust, 10 s: steady **vz ≈ +0.115 m/s** (slow drift
   up from +1.1 N), not free-fall.
2. **Self-righting** — from 20°, no thrust: **pitch 20°→0.7°**, **roll 20°→0.2°**
   over 25 s (both damped). Confirms restoring works and **pitch is now passively
   stabilized for disturbances** (cf. the Phase-2 underactuation).
3. **Terminal velocity / drag** — gentle surge (Fx≈2.8 N): speed rises to a steady
   **≈0.32 m/s** at ~4° pitch (nearly level), bounded. Release → **horizontal
   speed → 0.005 m/s** (drag stops the surge); only the buoyancy drift remains.
4. **Straighter than Phase 2** — 3 s surge, hydro vs no-hydro: no-hydro 0.69 m/s &
   84° tilt (grows/coasts) vs hydro **0.29 m/s & 9°** (drag-bounded).
5. **Stability** — 60 s with thrust + tilt: finite, `|qvel|` ≈ 0.6, no NaN.

## ⚠ Finding — surge↔pitch coupling beats the weak restoring

The restoring moment max is only **B·coBM ≈ 1.11 N·m**, while surge makes a pitch
moment **Fx·0.0725** ([03_THRUSTERS.md](03_THRUSTERS.md)). So:
- Below ~Fx ≈ 5 N: stable, nearly-level glide (good for open-loop driving).
- Above the restoring limit: the surge pitch moment wins → the vehicle noses
  over/tumbles. Drag bounds the **rate**, not the **angle**.

This is MarineGym's geometry, **passive restoring alone cannot hold attitude during
vigorous surge** — a controller is needed. Teleop keeps the default surge gentle
because of this ([05_TELEOP.md](05_TELEOP.md)). Don't "fix" it by enlarging coBM.

## Portability

`bluerov.xml` + meshes stay portable. The hydro is runtime Python (a passive
callback), like the dobmpc plant. On Linux/MJX the same Fossen equations + MarineGym
coefficients must be re-expressed in JAX (the CPU callback doesn't run under MJX);
the **coefficients and FLU sign conventions carry over unchanged**.
