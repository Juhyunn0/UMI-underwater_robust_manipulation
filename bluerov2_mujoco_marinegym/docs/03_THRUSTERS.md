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
These max thrusts (+64.13 / −51.55 N = +6.54 / −5.26 kgf) are MarineGym's fit, used
as-is per "keep MarineGym values" ([01](01_DECISIONS.md) D2); they sit at the **top
of the published T200 voltage range (~20 V)** — see voltage grounding below.

## Realistic actuator model + voltage grounding

`ThrusterModel` (`thrusters.py`) is an **opt-in** realistic actuator stage:
`f_des --T200 inverse(nominal V)--> throttle --motor lag--> T200 curve --×voltage--> f_real`.
It injects the **deadband** (small forces round-trip to 0 or the ~1.4 N min-spin
jump), **fwd/rev asymmetry + saturation**, **motor lag** (`T200Dynamics`), and a
**multiplicative thrust loss** (`voltage_scale`). The controller is *not* told, so
`realized ≠ commanded` = a sim-to-real robustness test. It is **default-ON for the
autonomous teleop missions** (`--square` / `--goto-origin`); `--ideal-thrusters`
reverts to the ideal force path. (Manual keyboard teleop, `eval_dp`, and `ablation`
use their own explicit paths.)

**`voltage_scale` is grounded in the official datasheet** (Blue Robotics *T200
Public Performance Data 10–20 V*, Sep 2019; reproduce with `analyze_t200_voltage.py`):

| V | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|
| max fwd (kgf) | 2.93 | 3.71 | 4.53 | 5.25 | 6.02 | 6.72 |
| max rev (kgf) | −2.31 | −2.92 | −3.52 | −4.07 | −4.59 | −5.04 |

Our base curve (6.54 / 5.26 kgf) ⇒ `voltage_scale = 1.0` ≈ a **~20 V** thruster.
A real BlueROV2 runs a **4S Li-ion pack (nominal 14.8 V)**, where max thrust is
4.81 / 3.74 kgf ⇒ `14.8V/base` = 0.74 (fwd) / 0.71 (rev) ⇒ **`NOMINAL_VOLTAGE_SCALE
= 0.72`** (the teleop default; `--thruster-voltage` overrides — full-charge 16.8 V
≈ 0.83, near-empty 13 V ≈ 0.62). The curve is **not refitted** to 14.8 V (that would
move `T200_MAX` / `ctrlrange`); only the scalar is applied. Still the *static
(bollard)* curve — inflow-velocity (advance-ratio) dependence is out of scope.

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

> **This rank-5 finding is `bluerov2`-specific.** The **`ROV_MODEL=heavy`** variant
> ([bluerov_heavy.xml](../bluerov_heavy.xml)) adds 2 more vertical thrusters (4 at the
> corners `(±0.12, ±0.22, −0.005)`, all +Z), making **`rank(B) = 6` — fully actuated**:
> pitch (and roll) ARE directly controllable (verified: a pure pitch wrench realizes
> `My = 1.000`). The DOB-MPC then commands the full 6-DOF wrench `u=[X,Y,Z,K,M,N]`
> (NU 6) and actively levels the vehicle. Select via `rov_model.py` /
> `ROV_MODEL`. **Same T200 thrusters and ctrlrange** (the MarineGym Heavy yaml's
> `force_constants` 0.8e-7 would scale thrust to ~18 %, which is unphysical for the
> same T200, so we keep the validated 64 N curve). See
> [CONTROL_METHODOLOGY.md](CONTROL_METHODOLOGY.md) (2026-06-18).

## Verified (Phase 2)

`python test_thrusters.py` (gravity off, to isolate thrust):
- T200 curve limits match the actuator `ctrlrange`.
- Each commanded DOF gives the right **FLU** response: surge→+x, sway→+y (left),
  heave→+z (up), yaw→+Mz, roll→+Mx; measured wrench (from MuJoCo acceleration)
  equals the analytic `B·f`; structural-zero terms ≈ 0.
- The surge→pitch and sway→roll couplings equal the predicted `±z₀` ratios.
- `rank(B)=5` reported; a requested pure-pitch wrench is shown to be unrealizable.
- Stepped-motion (throttle→curve→ctrl→step) moves the vehicle in +x / +z.
