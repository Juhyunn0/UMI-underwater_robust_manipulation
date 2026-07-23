# 07 — Disturbances & domain randomization (Phase 4)

**Status: DONE ✓.** Code: `disturbances.py` (+ hydro.py / teleop.py hooks).
Verify: `python tests/test_disturbances.py` (`--render` for the viewer). FLU, on top of
Phase-3 hydro, gravity ON. MarineGym-derived model unchanged.

## The 3 layers (all FLU)

### 1. Uniform current
A constant FLU water-velocity vector `vc`. **It is not a force** — it's a water
velocity. Hydro (`hydro.py`) now uses the **relative velocity** `vr = v − v_water`
in drag, added-mass-Coriolis, and the added-mass finite difference. So an unpowered
vehicle is **carried by the flow** (drift → the current velocity) and holding
station needs thrust. Default `vc = (0.20, 0, 0)` m/s.

### 2. Waves
A few sinusoidal water-velocity components (deep-water circular orbital motion),
added to `v_water` so they enter `vr` naturally:

```
v_wave(t, depth) = Σ_i  U_i · e^(−k_i·depth) · [ dir_i·cos(ω_i t + φ_i) + ẑ·sin(ω_i t + φ_i) ]
ω_i = 2π/T_i ,   k_i = ω_i² / g   (deep-water dispersion)
depth = max(0, z_surface − z_body)
```

**Irregular (JONSWAP) waves — default in teleop.** The 3 fixed sinusoids above look
too regular. `jonswap_wave_specs(Hs, Tp, n=30, gamma=3.3, heading_deg, spread_s=4, seed)`
returns a realistic **irregular** wave field as the same `{U,T,heading_deg,phase_deg}`
component list (so `wave_velocity` / the equation are unchanged — only the components
differ). Method (validated, Fossen Ch.8 / DNV-RP-C205): sample N components from a
JONSWAP spectrum using **equal-energy bins with a random frequency per bin** (this kills
the artificial repeat period — the key to "looks random"), uniform random phases, and
`cos^(2s)` directional spreading (matters for yaw excitation). `U_i = ω_i·a_i` with
`a_i=(Hs/4)√(2/N)` so `4√(Σa_i²/2)=Hs`. Default sea **Hs=0.20 m, Tp=4.0 s**; teleop
`--waves spectrum` (default) / `--waves classic` (the 3 sinusoids) / `--sea "Hs,Tp"`.
Note: `k=ω²/g` is kept (deep-water), so for Tp≥~5 s swell the 3 m-site penetration is
an approximation; full `ω²=g·k·tanh(kd)` is a documented upgrade. `kick` is unchanged
(a 0.15 s impulsive gust, NOT a wave).

**Wave implementation choice:** via the water velocity (not a direct wrench).
Because `vr` also drives the added-mass finite difference, a time-varying
`v_water` excites **both** the wave drag **and** the wave added-mass (inertia)
force — a Morison-like model — with no extra term and full reuse of Phase-3 hydro.
The depth decay `e^(−k·depth)` ties penetration to period: longer-period swell
(small k) reaches deeper, short chop dies near the surface — exactly what matters
at the ~3 m target site.

### 3. Random kicks
Poisson-timed impulsive **force spikes** (turbulence/bumps): at exponential
inter-arrival times, a force of random direction (mostly horizontal) and magnitude
is applied at the COM for a short duration. Applied directly as a world-frame
external force (not via `v_water`). Events are precomputed deterministically from
the seed over a horizon.

## Where it plugs in

`Hydrodynamics(model, disturbance=field)`. Inside the passive callback hydro
queries `field.water_velocity(t, pos)` for `vr` and `field.external_wrench(t, pos)`
for the kick. `disturbance=None` → still water (Phase-3 behaviour, unchanged — the
Phase-3 test still passes). Per-layer flags `use_current / use_waves / use_kicks`
and a master `enabled` (toggled by teleop's **G** key).

## Parameter ranges (and why they're reasonable for ~3 m depth)

Depth convention: `z_surface = 3.0 m` default, so the body at the model origin
(z=0) sits at ~3 m depth.

| layer | default | DR range | rationale |
|---|---|---|---|
| current speed | 0.20 m/s | 0–0.4 m/s | typical coastal/pool currents |
| current vertical | 0 | ±0.03 m/s | small (currents ~horizontal) |
| wave components | 3 (T=7/3.5/2 s) | 1–3 comps | a mixed sea: swell + wind wave + chop |
| wave U (orbital) | 0.08–0.18 m/s | 0.05–0.25 m/s | small-amplitude surface orbital speed |
| wave T | — | 2–9 s | incl. long swell that reaches 3 m |
| kick rate | 0.2 /s | 0.1–0.5 /s | a bump every ~2–10 s |
| kick magnitude | 20–50 N | 8–60 N | ≈ a 0.2–0.4 m/s velocity jolt |
| kick duration | 0.15 s | 0.10–0.20 s | short impulse |

Depth-decay sanity at 3 m: swell T=7 s (k=0.082) keeps ~78%; chop T=2 s (k=1.0)
keeps ~5%. So at depth the sea is swell-dominated — physically correct.

## Domain randomization

`sample_config(seed)` draws a bounded config (all the above + a few **model**
params: `drag_scale` 0.7–1.3, `thruster_scale` 0.8–1.2, `buoyancy_trim` ±2 N).
`randomize(seed)` → `(DisturbanceField, model_params)` for a future episode reset;
`apply_model_params(...)` applies drag/buoyancy scaling to a `Hydrodynamics`
(pass the **base** coefficients so repeated calls don't compound) and returns the
thruster scale for the caller to apply at command time. This is the **knobs +
sampler**; full DR training comes in the RL phase (Phase 8).

## Verified (Phase 4) — `python tests/test_disturbances.py`

1. **Current**: at rest the first-step horizontal velocity is +x (so `vr`, not `v`,
   is used — with `v` an unpowered vehicle wouldn't move); after 40 s the
   horizontal velocity reaches the current (0.20 → 0.20 m/s).
2. **Waves**: field orbital speed decays with depth (0.281 → 0.143 m/s, 1→5 m);
   the vehicle's surge oscillation is much weaker deep (0.020 @1.5 m vs 0.003 @6 m,
   single T=3 s wave, neutral buoyancy to hold depth).
3. **Kicks**: ~the scheduled number of jolts are detected (18 scheduled, 14
   detected at rate 0.4/s over 40 s), each a sudden speed jump.
4. **Distinctness**: current = drift (mean 0.18, ~no osc), wave = oscillation
   (osc 0.057, ~no drift), kick = jolt (spike 0.31) — clearly separable.
5. **DR**: 6 seeds → varied current speeds, all ≤ 0.4, all finite & bounded.
6. **Combined**: all three for 60 s → finite, `|qvel|` bounded, no NaN.

## Teleop

`python teleop.py` attaches a `DisturbanceField` (off by default). **`G`** toggles
all disturbances on/off; `--disturb` starts them on. Drive into the current / feel
the wave bob / get knocked by kicks. (Controls otherwise unchanged — see
[05_TELEOP.md](05_TELEOP.md).)

## Gotchas

- **`set_mjcb_passive` is global and fires during `from_xml_path`'s internal
  forward.** If you compile a *new* model while a hydro callback is still
  installed, it crashes ("engine error: Python exception raised"). When rebuilding
  models in a loop, call `Hydrodynamics.uninstall()` before `from_xml_path`
  (see `test_disturbances.make`).
- Current/waves enter via `vr`; **kicks are forces**. Keep that distinction — a
  "current" expressed as a force would not give the correct drift-to-flow-velocity
  behaviour, and a "kick" expressed as a velocity would fight the drag oddly.
- FLU only. Water velocity is a world-frame FLU vector; hydro does the world→body
  rotation. No NED.
- On GPU/MJX later: like the hydro, this is a CPU Python callback — the same model
  (current/waves/kicks + depth decay) must be re-expressed in JAX.
