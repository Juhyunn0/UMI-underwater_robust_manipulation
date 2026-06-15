# 03 — Thrusters (Phase 2: actuation & allocation)

**Status: DONE ✓.** Code: `thrusters.py`, actuators in `bluerov.xml`. Verify:
`python test_thrusters.py`. Everything FLU; MarineGym coefficients only.

## T200 thrust model (from MarineGym `t200.py` + `BlueROV.yaml`)

Steady-state throttle→thrust curve, reproduced exactly in `thrusters.py`
(`t200_thrust(u)`):

| quantity | value |
|---|---|
| max forward (u=+1) | **+64.13 N** |
| max reverse (u=−1) | **−51.55 N** |
| forward/reverse asymmetry | **1.244** (forward stronger) |
| deadband | `|u| ≤ 0.075` → 0 N |
| per-rotor `force_constants/4.4e-7` | 1.0 (uniform) → thrust = 9.81·kgf(rpm) |

Notes: MarineGym multiplies propeller **reaction torque by 0**, so thrusters apply
**force only** (no spin torque) — all body torque is `r × F`. Its `directions` /
`moment_constants` are therefore unused here. A first-order throttle/rpm lag exists
in MarineGym (`T200Dynamics` reproduces it) but is not needed for the steady curve.
These max thrusts (~64/52 N) are a bit above a nominal 16 V T200 (~50 N); they are
MarineGym's fit, used as-is per "keep MarineGym values" ([01](01_DECISIONS.md) D2).

## Actuators & command convention

- The MJCF has **6 `<general>` force actuators** `thr0..thr5`, each a pure force
  along its thruster site's local **+X** (`gear="1 0 0 0 0 0"`).
- **`data.ctrl[i]` is the thrust in NEWTONS**, `ctrlrange = [−51.55, 64.13]`.
- Two command entry points (`thrusters.py`):
  1. **Per-thruster throttle** `u ∈ [−1,1]⁶` → `t200_thrust(u)` → `data.ctrl`
     (`set_thruster_commands`). `u>0` pushes along that thruster's +X.
  2. **Body wrench** `[Fx,Fy,Fz,Mx,My,Mz]` (FLU) → `pinv(B)` → forces → ctrl
     (`set_wrench_command`).
- `step(model, data, throttles=…|forces_N=…, n=…)` sets a command and steps.

## Allocation matrix B (wrench = B · thruster_forces, FLU, about COM)

```
rows [Fx,Fy,Fz,Mx,My,Mz], cols thruster_0..5
[[ 0.7071  0.7071 -0.7071 -0.7071  0.      0.    ]
 [ 0.7071 -0.7071  0.7071 -0.7071  0.      0.    ]
 [ 0.      0.      0.      0.      1.      1.    ]
 [ 0.0513 -0.0513  0.0513 -0.0513 -0.1105  0.1105]
 [-0.0513 -0.0513  0.0513  0.0513 -0.0025 -0.0025]
 [ 0.1665 -0.1665 -0.175   0.175   0.      0.    ]]
```

Built at runtime from the compiled model (`allocation_matrix()`): column *i* =
`[d_i ; r_i × d_i]`, with `d_i` the site +X axis and `r_i` the site position
relative to the COM. `allocate(B, wrench)` = `pinv(B) · wrench`.

## ⚠ KEY FINDING 1 — geometric couplings (thrusters below COM)

The 4 horizontal thrusters sit **z₀ = −0.0725 m below the COM**, so a horizontal
force makes a torque:
- **Surge → pitch:** `My = z₀·Fx = −0.0725·Fx`.
- **Sway → roll:**  `Mx = −z₀·Fy = +0.0725·Fy` (plus a small sway→yaw `Mz` from the
  fore/aft thruster x-asymmetry).

These are verified numerically (the test ties `My/Fx` and `Mx/Fy` to the measured
z₀). They are real geometry, not bugs. Note: when commanding via `pinv(B)`, the
**sway→roll** coupling is automatically cancelled (roll is controllable, so the
allocator uses the vertical thrusters), but **surge→pitch is not** (see below).

## ⚠ KEY FINDING 2 — UNDERACTUATED in pitch (rank(B) = 5)

`rank(B) = 5`, not 6. **Pitch (My) is not independently controllable.** It is
rigidly coupled to surge and heave: `My ≈ −0.0725·Fx − 0.0025·Fz`. Cause: the 4
horizontal thrusters share the same z-offset (so their pitch contribution is
proportional to Fx), and the 2 vertical thrusters share nearly the same x.

**Consequences for any controller / MPC (Phase 7):**
- Command **surge, sway, heave, yaw, roll** — **never pitch**. Asking for a pitch
  wrench yields a near-zero, heavily-coupled result.
- Surge inherently drags pitch with it. Phase 3's buoyant restoring only partly
  offsets this and only at low thrust (see [04_HYDRO.md](04_HYDRO.md)). Holding
  attitude during vigorous surge requires active control — this is a core reason
  the project needs the DOB-MPC.

## Verified (Phase 2)

`python test_thrusters.py` (gravity off, to isolate thrust):
- T200 curve limits match the actuator `ctrlrange`.
- Each commanded DOF gives the right **FLU** response: surge→+x, sway→+y (left),
  heave→+z (up), yaw→+Mz, roll→+Mx; measured wrench (from MuJoCo acceleration)
  equals the analytic `B·f`; structural-zero terms ≈ 0.
- The surge→pitch and sway→roll couplings equal the predicted `±z₀` ratios.
- `rank(B)=5` reported; a requested pure-pitch wrench is shown to be unrealizable.
- Stepped-motion (throttle→curve→ctrl→step) moves the vehicle in +x / +z.
