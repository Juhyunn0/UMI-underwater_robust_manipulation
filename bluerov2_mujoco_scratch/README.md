# BlueROV2 MuJoCo — from-scratch, incremental build

A pedagogical, **phase-by-phase** reconstruction of the BlueROV2 MuJoCo
simulator. Each phase adds *one* capability and ends with a **verification
gate** (a script that proves the new piece works against hand-computable
physics) before we move on.

The validated Fossen math (`M, C, D, g`, stiffness handling, sign
conventions, parameters) is **reused** from the sibling production package
`../bluerov2_mujoco_dobmpc/bluerov2mj` via `rov_sim/physics.py`, so we never
re-derive — and never re-introduce — the dynamics that package already got
right. Everything *MuJoCo-specific* (the MJCF, the force injection, the
controller/observer wiring) is built fresh here.

## Phases & gates  (모든 단계 통과 / all gates PASS)

| Phase | Adds | Verification gate | Status |
|------|------|-------------------|--------|
| **1** | BlueROV2 MJCF + buoyancy | **static stability**: a torque kick → roll/pitch self-right (period 2.33 s ≈ 2π/√(z_G·W/I)); yaw drifts (neutral, no restoring) | ✅ |
| **2** | Thrust wrench `K t` via `xfrc_applied` | **directional motion**: each `u` axis drives the right actuated DOF (roll/pitch coupling is real BlueROV2 physics) | ✅ |
| **3** | Hydrodynamics: drag `D(v)v`, `C_A`, added mass `M_A` | **terminal velocity** = 2.028 cm/s; **energy** never increases | ✅ |
| **4** | MPC (CasADi NMPC) with MuJoCo state feedback | **set-point regulation**: 1.27 m offset → 0.14 cm | ✅ |
| **5** | EAOB + measurement noise + external disturbance | **disturbance tracking** (±0.2 N) + DOBMPC holds station (~1–2 cm) under 10 N current | ✅ |

> Phase 4 uses **CasADi/Ipopt** (acados not installed here); the model + cost
> port 1:1 to **acados SQP-RTI** for the paper's real-time setup.
> Phase 4는 acados 미설치라 CasADi/Ipopt 사용 — 동일 모델/비용을 acados로 그대로 이식 가능.

## Layout

```
bluerov2_mujoco_scratch/
  rov_sim/
    physics.py     # re-exports validated params/fossen/allocation/NMPC/EAOB/disturbances
    model.xml      # the BlueROV2 MJCF (NED world, free body, explicit inertia, 6 thrusters)
    env.py         # the simulator: buoyancy + thrust + hydro + noise/disturbance (flag-gated)
  phase1_static_equilibrium.py   # Phase 1 gate
  phase2_directional_motion.py   # Phase 2 gate
  phase3_hydrodynamics.py        # Phase 3 gate
  phase4_mpc.py                  # Phase 4 gate  (~30 s)
  phase5_eaob_dob.py             # Phase 5 gate  (~40 s)
  view_phase.py                  # LIVE 3-D viewer for any phase (--phase N)
  compare_cg_stability.py        # CG below vs above CB: stable vs capsizes
```

## See it live / 눈으로 보기

```bash
python view_phase.py --phase 1   # a torque kick; roll/pitch self-right (restoring couple)
python view_phase.py --phase 2   # cycles surge / sway / heave / yaw
python view_phase.py --phase 3   # a velocity kick decays under drag
python view_phase.py --phase 4   # MPC flies between set-points (green marker)
python view_phase.py --phase 5   # DOBMPC holds station under a current (red arrow)
```

Tracking camera (vehicle stays centred), underwater lighting, green = goal,
red arrow = applied current, blue dots = trail.  `SPACE` pause, `BACKSPACE`
restart, drag = orbit.  NED world (+z down) so orbit if it looks upside-down.

**Push it yourself / 직접 힘 주기:** double-click the vehicle to select it, then
`Ctrl + right-drag` = apply a force, `Ctrl + left-drag` = apply a torque (the
standard MuJoCo perturbation, fed in as an external wrench).  Great on `--phase 4/5`
to shove the ROV and watch the controller fight back.

**Why it self-rights / 왜 스스로 일어서나** — the CG sits below the CB:

```bash
python compare_cg_stability.py            # numbers: stable vs capsize
python view_phase.py --phase 1 --cg below # CG below CB -> rights itself (stable)
python view_phase.py --phase 1 --cg above # CG above CB -> capsizes (unstable)
```

The simulator is one flag-gated `ROVEnv`; each phase script switches on only
the capability it tests (`enable_thrust`, `enable_hydro`, `meas_noise`, and the
disturbance passed to `step`), so you can test **step by step**.

## Run

```bash
conda activate robust
cd bluerov2_mujoco_scratch
python phase1_static_equilibrium.py     # buoyancy & static equilibrium
python phase2_directional_motion.py     # thrust directions
python phase3_hydrodynamics.py          # drag + added mass
python phase4_mpc.py                    # MPC regulation     (~30 s)
python phase5_eaob_dob.py               # EAOB + noise + disturbance  (~40 s)
```

Each script prints per-check `PASS/FAIL` and exits non-zero on failure, so they
double as a regression suite. All comments are **English + 한국어**.

## Conventions (identical to the production package)

- World = **NED** (x north, y east, **z down**), so MuJoCo `gravity = +9.81 z`.
- Body = **FRD** (x fwd, y stbd, z down), origin at the centre of buoyancy
  (CB); CG sits `z_G = 0.02 m` below it.
- `eta = [x y z φ θ ψ]` (ZYX Euler), `nu = [u v w p q r]` (body frame).
